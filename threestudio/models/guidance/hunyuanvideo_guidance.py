import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("hunyuanvideo-guidance")
class HunyuanVideoGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "HunyuanVideo-1.5/ckpts"
        model_type: str = "t2v"  # "t2v" (cfg-distill) or "i2v" (step-distill)
        resolution: str = "480p"  # 480p or 720p
        task: str = "i2v"  # t2v or i2v
        cfg_distilled: bool = False
        step_distilled: bool = True
        sparse_attn: bool = False
        guidance_scale: float = 6.0
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        weighting_strategy: str = "sds"  # sds | uniform
        half_precision_weights: bool = True
        negative_prompt: str = ""
        use_vision_encoder: bool = False

    cfg: Config

    def configure(self) -> None:
        # Align with HunyuanVideo infer_i2v.sh defaults
        env_defaults = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "FORCE_TORCH_ATTN": "0",
            "FORCE_FLASH2": "1",
        }
        for k, v in env_defaults.items():
            os.environ.setdefault(k, v)

        # Ensure hyvideo is importable
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "HunyuanVideo-1.5")
        )
        if repo_root not in sys.path:
            sys.path.append(repo_root)

        try:
            from hyvideo.pipelines.hunyuan_video_pipeline import (
                HunyuanVideo_1_5_Pipeline,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import HunyuanVideo pipeline. Please ensure HunyuanVideo-1.5 "
                "is available at repository root or installed."
            ) from e

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        threestudio.info("Loading HunyuanVideo pipeline ...")
        # Select transformer version based on model_type
        if self.cfg.model_type == "t2v":
            task = "t2v"
            cfg_distilled = True
            step_distilled = False
        elif self.cfg.model_type == "i2v":
            task = "i2v"
            cfg_distilled = False
            step_distilled = True
        else:
            raise ValueError(f"Unsupported model_type: {self.cfg.model_type}")

        transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
            self.cfg.resolution,
            task,
            cfg_distilled=cfg_distilled,
            step_distilled=step_distilled,
            sparse_attn=self.cfg.sparse_attn,
        )
        self.pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            self.cfg.pretrained_model_name_or_path,
            transformer_version=transformer_version,
            create_sr_pipeline=False,
            transformer_dtype=self.weights_dtype,
            device=self.device,
            transformer_init_device=self.device,
        )
        self.pipe.to(self.device)
        self.transformer = self.pipe.transformer.eval()
        self.vae = self.pipe.vae.eval()
        self.text_encoder = self.pipe.text_encoder.eval()
        self.vision_encoder = self.pipe.vision_encoder if self.cfg.use_vision_encoder else None
        if self.vision_encoder is not None:
            self.vision_encoder = self.vision_encoder.eval().to(self.device)
        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = len(self.scheduler.timesteps)
        self.set_min_max_steps()
        threestudio.info("Loaded HunyuanVideo pipeline.")

    def set_min_max_steps(self):
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

    @torch.no_grad()
    def encode_prompt(
        self, prompt: str, negative_prompt: Optional[str], batch_size: int
    ):
        prompt_embeds, prompt_embeds_2, prompt_mask = None, None, None
        negative_prompt_embeds, negative_prompt_embeds_2, negative_prompt_mask = (
            None,
            None,
            None,
        )
        # HunyuanVideo pipeline encodes prompts internally; we reuse it for CFG.
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.pipe.encode_prompt(
            prompt=[prompt] * batch_size,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[negative_prompt] * batch_size
            if negative_prompt is not None
            else None,
            data_type="video",
        )
        # Second encoder currently unused in open checkpoints
        prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = (
            None,
            None,
            None,
            None,
        )
        return (
            prompt_embeds,
            prompt_embeds_2,
            prompt_mask,
            prompt_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_2,
            negative_prompt_mask,
            negative_prompt_mask_2,
        )

    def _get_cond_latents(
        self,
        latents: Float[Tensor, "B C T H W"],
        task_type: str = "t2v",
        cond_latents: Optional[Float[Tensor, "B C 1 H W"]] = None,
    ):
        latent_target_length = latents.shape[2]
        multitask_mask = (
            self.pipe.get_task_mask(task_type, latent_target_length)
            .to(device=latents.device, dtype=latents.dtype)
            .expand(latents.shape[0], -1)
        )
        cond_latents = self.pipe._prepare_cond_latents(
            task_type, cond_latents, latents, multitask_mask
        ).to(dtype=latents.dtype)
        return cond_latents

    @torch.cuda.amp.autocast(enabled=False)
    def __call__(
        self,
        rgb: Float[Tensor, "B 3 T H W"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        if rgb.dim() != 5:
            raise ValueError(f"HunyuanVideo guidance expects video tensor (B,3,T,H,W), got {rgb.shape}")

        B, _, T, H, W = rgb.shape
        device = self.device

        rgb = rgb.to(device)
        rgb_norm = rgb.clamp(0.0, 1.0)
        # Encode with VAE
        with torch.no_grad():
            video_in = (rgb_norm * 2.0 - 1.0).to(self.weights_dtype)
            posterior = self.vae.encode(video_in)
            latents = posterior.latent_dist.sample() * self.vae.config.scaling_factor

        # Sample timestep
        # t_int = torch.randint(
        #     self.min_step, self.max_step + 1, (B,), device=device, dtype=torch.long
        # )
        t_int = torch.randint(
            self.min_step, self.max_step + 1, (B,), device=device, dtype=torch.long
        )
        # Scheduler buffers are on CPU; index there, then move to device.
        t_cpu = t_int.detach().cpu()
        t = self.scheduler.timesteps[t_cpu].to(device=device, dtype=self.weights_dtype)
        sigma = self.scheduler.sigmas[t_cpu].to(device=device, dtype=latents.dtype)
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)

        noise = torch.randn_like(latents)
        latents_noisy = latents + sigma * noise

        # Prepare condition latents/mask
        if self.cfg.model_type == "i2v":
            # Use first frame as condition image (i2v-style)
            ref_frame = rgb[:, :, 0, :, :]  # (B,3,H,W)
            with torch.no_grad():
                ref_in = (ref_frame.clamp(0.0, 1.0) * 2.0 - 1.0).to(self.weights_dtype)
                ref_in = ref_in.unsqueeze(2)  # (B,3,1,H,W)
                ref_posterior = self.vae.encode(ref_in)
                cond_latents_ref = ref_posterior.latent_dist.mode() * self.vae.config.scaling_factor
                cond_latents_ref = cond_latents_ref.to(dtype=latents.dtype, device=device)
            cond_latents = self._get_cond_latents(
                latents_noisy, task_type="i2v", cond_latents=cond_latents_ref
            )
            mask_type = "i2v"
        else:
            cond_latents = self._get_cond_latents(
                latents_noisy, task_type="t2v", cond_latents=None
            )
            mask_type = "t2v"
        latents_concat = torch.cat([latents_noisy, cond_latents], dim=1)

        # Encode prompts via Hunyuan text encoder
        prompt = getattr(prompt_utils, "prompt", None) or ""
        negative_prompt = self.cfg.negative_prompt
        (
            prompt_embeds,
            prompt_embeds_2,
            prompt_mask,
            prompt_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_2,
            negative_prompt_mask,
            negative_prompt_mask_2,
        ) = self.encode_prompt(prompt, negative_prompt, B)

        # set internal guidance scale flags for CFG / ByT5 prep
        self.pipe._guidance_scale = self.cfg.guidance_scale
        self.pipe._guidance_rescale = 0.0
        self.pipe._clip_skip = None
        do_cfg = self.pipe.do_classifier_free_guidance
        if do_cfg:
            latent_model_input = torch.cat([latents_concat] * 2, dim=0)
        else:
            latent_model_input = latents_concat

        t_expand = t.repeat(latent_model_input.shape[0])

        # prepare ByT5 embeddings if enabled in pipeline config
        extra_kwargs = {}
        try:
            extra_kwargs = self.pipe._prepare_byt5_embeddings(
                [prompt] * B, device=self.device
            )
        except Exception:
            extra_kwargs = {}

        cast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=cast_device, dtype=self.weights_dtype, enabled=True):
            output = self.transformer(
                latent_model_input,
                t_expand,
                prompt_embeds,
                prompt_embeds_2,
                prompt_mask,
                timestep_r=None,
                vision_states=None,
                mask_type=mask_type,
                guidance=None,
                return_dict=False,
                extra_kwargs=extra_kwargs,
            )
            noise_pred = output[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            w = sigma**2
        elif self.cfg.weighting_strategy == "uniform":
            w = 1.0
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / B

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out

    def destroy(self) -> None:
        cleanup()

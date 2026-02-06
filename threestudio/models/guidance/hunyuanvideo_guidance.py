import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import numpy as np
import cv2

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

# Global cache to avoid reloading large HunyuanVideo pipelines within the same process
_PIPE_CACHE = {}


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
        # debug options (parsing only; runtime uses __call__ args)
        debug_save_dir: Optional[str] = None
        debug_step: Optional[int] = None
        debug_panel: bool = True
        debug_pipe: bool = True
        debug_one_step: bool = True
        debug_num_steps: int = 12
        debug_video_length: Optional[int] = None
        debug_pipe_steps: int = 1

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

        cache_key = (
            self.cfg.pretrained_model_name_or_path,
            transformer_version,
            self.cfg.use_vision_encoder,
            str(self.device),
            self.weights_dtype,
        )
        if cache_key in _PIPE_CACHE:
            self.pipe, self.transformer, self.vae, self.text_encoder, self.vision_encoder, self.scheduler = _PIPE_CACHE[cache_key]
            threestudio.info("Reusing cached HunyuanVideo pipeline.")
        else:
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
            _PIPE_CACHE[cache_key] = (
                self.pipe,
                self.transformer,
                self.vae,
                self.text_encoder,
                self.vision_encoder,
                self.scheduler,
            )
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
        debug_save_dir: Optional[str] = None,
        debug_step: Optional[int] = None,
        debug_pipe: bool = True,
        debug_one_step: bool = True,
        debug_num_steps: int = 12,
        debug_video_length: Optional[int] = None,
        debug_panel: bool = True,
        debug_pipe_steps: int = 1,
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

        # Optional debug previews
        if debug_save_dir is not None and debug_step is not None:
            pipe_frames, sds_frames, pipe_latents, sds_x0_est = self._debug_save_videos(
                rgb=rgb,
                prompt=prompt,
                latents_noisy=latents_noisy,
                noise=noise,
                sigma=sigma,
                noise_pred=noise_pred,
                mask_type=mask_type,
                debug_save_dir=debug_save_dir,
                debug_step=debug_step,
                use_pipe=debug_pipe,
                use_one_step=debug_one_step,
                num_steps=debug_num_steps,
                video_length=debug_video_length or T,
                debug_pipe_steps=debug_pipe_steps,
            )
            if debug_panel:
                self._debug_save_panel(
                    rgb=rgb,
                    rgb_norm=rgb_norm,
                    video_in=video_in,
                    posterior=posterior,
                    latents=latents,
                    t_int=t_int,
                    t=t,
                    sigma=sigma,
                    noise=noise,
                    latents_noisy=latents_noisy,
                    cond_latents=cond_latents,
                    latents_concat=latents_concat,
                    noise_pred=noise_pred,
                    grad=grad,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                prompt=prompt,
                negative_prompt=self.cfg.negative_prompt,
                debug_save_dir=debug_save_dir,
                debug_step=debug_step,
                pipe_frames=pipe_frames,
                sds_frames=sds_frames,
                pipe_latents=pipe_latents,
                sds_x0_est=sds_x0_est,
            )

        return guidance_out

    @torch.no_grad()
    def _decode_latents(self, latents: Float[Tensor, "B C T H W"]) -> Float[Tensor, "B C T H W"]:
        """Decode latents with VAE (no SR, no tiling) into [0,1] videos."""
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)

        latents = latents.to(self.vae.dtype)
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor

        with torch.autocast(device_type="cuda", dtype=self.vae.dtype, enabled=True):
            self.vae.enable_tiling()
            video = self.vae.decode(latents, return_dict=False)[0]
            self.vae.disable_tiling()
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    @staticmethod
    def _save_video_tensor(video: Float[Tensor, "C T H W"], path: str, fps: int = 24) -> None:
        video_np = (video.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
        imageio.mimwrite(path, video_np, fps=fps)

    @staticmethod
    def _tile_channels(
        img_cthw: Float[Tensor, "C T H W"], target_hw: int = 128
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Returns:
          combined_img: concat [PCA(or RGB) , channel-grid] horizontally for first frame; frames in grid are concatenated horizontally.
          pca_info: text describing PCA mapping (if C>3)
        """
        C, T, H, W = img_cthw.shape

        # ---- channel grid over all frames ----
        imgs = []
        for t in range(T):
            slice_c = img_cthw[:, t]  # C,H,W
            if C == 3:
                arr = slice_c.permute(1, 2, 0).cpu().numpy()
                # Normalize to [0, 1] for better visualization (same as SD)
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max - arr_min > 1e-6:
                    arr = (arr - arr_min) / (arr_max - arr_min)
                arr = np.clip(arr, 0.0, 1.0)
            else:
                per = []
                side = int(np.ceil(np.sqrt(C)))
                for c in range(C):
                    ch = slice_c[c].unsqueeze(0)  #1,H,W
                    # Normalize each channel individually
                    ch_min, ch_max = ch.min(), ch.max()
                    if ch_max - ch_min > 1e-6:
                        ch = (ch - ch_min) / (ch_max - ch_min)
                    ch_img = ch.expand(3, -1, -1)
                    ch_img = F.interpolate(
                        ch_img.unsqueeze(0),
                        size=(target_hw, target_hw),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    per.append(ch_img)
                while len(per) < side * side:
                    per.append(torch.zeros_like(per[0]))
                rows = []
                for r in range(side):
                    row = torch.cat(per[r * side : (r + 1) * side], dim=2)
                    rows.append(row)
                grid = torch.cat(rows, dim=1)
                arr = grid.permute(1, 2, 0).cpu().numpy()
                arr = np.clip(arr, 0.0, 1.0)
            arr = cv2.resize(arr, (target_hw, target_hw), interpolation=cv2.INTER_AREA)
            imgs.append(arr)
        grid_img = np.concatenate(imgs, axis=1)  # H x (target_hw*T) x 3

        # ---- PCA (or RGB) for first frame ----
        if C == 3:
            pca_rgb = imgs[0]  # first frame already resized
            pca_info = None
        else:
            data = img_cthw[:, 0].permute(1, 2, 0).reshape(-1, C).detach().cpu().float()  # first frame
            data = data - data.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(data, full_matrices=False)
            comps = Vh[:3]  # (3,C)
            proj = data @ comps.T  # (N,3)
            proj -= proj.min(dim=0, keepdim=True)[0]
            denom = proj.max(dim=0, keepdim=True)[0].clamp(min=1e-6)
            proj = proj / denom
            pca_img = proj.reshape(H, W, 3)
            pca_img = cv2.resize(pca_img.numpy(), (target_hw, target_hw), interpolation=cv2.INTER_AREA)
            pca_rgb = (np.clip(pca_img, 0.0, 1.0) * 255).astype(np.uint8)
            # 记录每个 PC 的权重最大通道
            pc_top = []
            for i in range(3):
                weights = comps[i]
                topk = torch.topk(weights.abs(), k=min(3, C))
                pc_top.append(
                    "PC{} top: ".format(i + 1)
                    + ", ".join([f"ch{idx.item()}:{weights[idx].item():.3f}" for idx in topk.indices])
                )
            pca_info = " | ".join(pc_top)

        # ---- concat PCA/RGB and grid ----
        h = max(pca_rgb.shape[0], grid_img.shape[0])
        # pad heights if needed
        def pad_h(img, target_h):
            if img.shape[0] == target_h:
                return img
            pad = target_h - img.shape[0]
            return np.pad(img, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=255)

        pca_rgb = pad_h(pca_rgb, h)
        grid_img = pad_h(grid_img, h)
        combined = np.concatenate([pca_rgb, grid_img], axis=1)

        return combined, pca_info

    @staticmethod
    def _scalar_row(text: str, width: int = 400, height: int = 40) -> np.ndarray:
        width = int(width * 1.5)
        height = int(height * 2)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, text, (10, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        return canvas

    @torch.no_grad()
    def _debug_save_panel(
        self,
        rgb: Float[Tensor, "B 3 T H W"],
        rgb_norm: Float[Tensor, "B 3 T H W"],
        video_in: Float[Tensor, "B 3 T H W"],
        posterior,
        latents: Float[Tensor, "B 4 T H W"],
        t_int: Float[Tensor, "B"],
        t: Float[Tensor, "B"],
        sigma: Float[Tensor, ""],
        noise: Float[Tensor, "B 4 T H W"],
        latents_noisy: Float[Tensor, "B 4 T H W"],
        cond_latents: Float[Tensor, "B 4 T H W"],
        latents_concat: Float[Tensor, "B C T H W"],
        noise_pred: Float[Tensor, "B C T H W"],
        grad: Float[Tensor, "B C T H W"],
        prompt_embeds: Optional[Tensor],
        prompt_mask: Optional[Tensor],
        prompt: str,
        negative_prompt: str,
        debug_save_dir: str,
        debug_step: int,
        pipe_frames: Optional[Tensor],
        sds_frames: Optional[Tensor],
        pipe_latents: Optional[Tensor],
        sds_x0_est: Optional[Tensor],
    ) -> None:
        os.makedirs(debug_save_dir, exist_ok=True)
        rows = []
        shape_logs = []

        def add_tensor_row(label: str, tensor: Float[Tensor, "..."]):
            # assume B=1
            if tensor is None:
                shape_logs.append(f"{label}: None")
                return
            t_cpu = tensor.detach().float()
            # Squeeze leading singleton batch dims
            while t_cpu.ndim > 4 and t_cpu.shape[0] == 1:
                t_cpu = t_cpu.squeeze(0)
            if t_cpu.ndim == 4:  # C,T,H,W expected
                cthw = t_cpu
            elif t_cpu.ndim >= 5:
                cthw = t_cpu
                # try to squeeze trailing singleton dims until 4D
                while cthw.ndim > 4 and cthw.shape[-1] == 1:
                    cthw = cthw.squeeze(-1)
                while cthw.ndim > 4 and cthw.shape[1] == 1:
                    cthw = cthw.squeeze(1)
                if cthw.ndim != 4:
                    shape_logs.append(f"{label}: shape={tuple(t_cpu.shape)} (skipped, ndim={t_cpu.ndim})")
                    return
            else:
                shape_logs.append(f"{label}: shape={tuple(t_cpu.shape)} (skipped, ndim={t_cpu.ndim})")
                return
            shape_logs.append(f"{label}: shape={tuple(cthw.shape)}")
            row_img, pca_info = self._tile_channels(cthw)
            rows.append((label, row_img, f"shape={tuple(cthw.shape)}"))
            if pca_info:
                shape_logs.append(f"{label} PCA info: {pca_info}")

        # Helper visualization functions (defined early so they can be used in grad diagnostics)
        def _make_hist_image(arr: np.ndarray, bins: int = 30, width: int = 420, height: int = 160, title: str = "") -> np.ndarray:
            hist, bin_edges = np.histogram(arr, bins=bins)
            hist = hist.astype(np.float32)
            hist /= hist.max() + 1e-6
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            for i in range(bins):
                x0 = int(i * width / bins)
                x1 = int((i + 1) * width / bins)
                y1 = height - 10
                y0 = int((1.0 - hist[i]) * (height - 20)) + 10
                cv2.rectangle(img, (x0, y0), (x1, y1), (60, 120, 200), -1)
            if title:
                cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 1, cv2.LINE_AA)
            xmin, xmax = bin_edges[0], bin_edges[-1]
            cv2.putText(img, f"{xmin:.2e}", (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            txt = f"{xmax:.2e}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img, txt, (width - tw - 5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            return img

        def _make_heatmap_2d(arr_2d: np.ndarray, target: int = 128, label: str = "") -> np.ndarray:
            arr = arr_2d - arr_2d.min()
            arr = arr / (arr.max() + 1e-6)
            arr = cv2.resize(arr, (target, target), interpolation=cv2.INTER_AREA)
            cmap = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
            if label:
                cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            return img

        def _make_line_chart(values: list, width: int = 420, height: int = 160, title: str = "") -> np.ndarray:
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            if len(values) < 2:
                cv2.putText(img, "not enough points", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                return img
            vals = np.array(values, dtype=np.float32)
            vmax = vals.max() + 1e-8
            vmin = vals.min()
            span = vmax - vmin + 1e-8
            pts = []
            for i, v in enumerate(vals):
                x = int(i / (len(vals) - 1) * (width - 20)) + 10
                y = int((1.0 - (v - vmin) / span) * (height - 30)) + 15
                pts.append((x, y))
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (0, 100, 200), 2)
            cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
            cv2.putText(img, f"min={vmin:.2e}", (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            txt = f"max={vmax:.2e}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img, txt, (width - tw - 5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            return img

        # scalar/info rows
        rows.append(("num_train_timesteps=len(scheduler.timesteps)", self._scalar_row(f"{self.num_train_timesteps}"), ""))
        rows.append(("min_step", self._scalar_row(f"{self.min_step}"), ""))
        rows.append(("max_step", self._scalar_row(f"{self.max_step}"), ""))
        shape_logs.append(f"num_train_timesteps={self.num_train_timesteps}, min_step={self.min_step}, max_step={self.max_step}")
        sigma_val = sigma.detach().reshape(-1)[0].item()
        rows.append(("t_int = randint(min_step, max_step)", self._scalar_row(f"{t_int.cpu().tolist()}"), ""))
        rows.append(("t = scheduler.timesteps[t_int]", self._scalar_row(f"{t.cpu().tolist()}"), ""))
        rows.append(("sigma = scheduler.sigmas[t_int]", self._scalar_row(f"{sigma_val:.6f}"), ""))
        shape_logs.append(f"t_int={t_int.cpu().tolist()}, t={t.cpu().tolist()}, sigma={sigma_val:.6f}")
        rows.append(("prompt = text", self._scalar_row(f"prompt={prompt}", width=800), ""))
        rows.append(("negative_prompt = text", self._scalar_row(f"negative={negative_prompt}", width=800), ""))
        rows.append(("--- INPUT / ENCODE ---", self._scalar_row("RGB -> VAE latents"), ""))
        shape_logs.append(f"prompt len={len(prompt)}, negative len={len(negative_prompt)}")

        add_tensor_row("rgb", rgb)
        add_tensor_row("rgb_norm = rgb.clamp(0,1)", rgb_norm)
        add_tensor_row("video_in = (rgb_norm*2-1).to(weights_dtype)", video_in)
        add_tensor_row("posterior = vae.encode(video_in)", posterior.latent_dist.mean)
        add_tensor_row("latents = posterior.sample()*vae.config.scaling_factor", latents)
        add_tensor_row("noise = randn_like(latents)", noise)
        add_tensor_row("latents_noisy = latents + sigma*noise", latents_noisy)
        rows.append(("--- COND & CONCAT ---", self._scalar_row("Condition / concat latents"), ""))
        cond_desc = "cond_latents = (t2v: zeros+mask | i2v: encode rgb[0] + mask)"
        rows.append((cond_desc, self._scalar_row(cond_desc), ""))
        shape_logs.append(cond_desc)
        add_tensor_row("cond_latents = prepare_cond(latents_noisy, task)", cond_latents)
        add_tensor_row("latents_concat = concat(latents_noisy, cond_latents)", latents_concat)
        rows.append(("--- UNET / SDS ---", self._scalar_row("Transformer output and SDS grads"), ""))
        # Transformer inputs/outputs
        add_tensor_row("noise_pred = transformer(latents_concat, t, prompts)", noise_pred)

        # SDS weights / grads (scalar)
        w_val = float((sigma**2).detach().reshape(-1)[0].item())
        rows.append(("w = sigma**2", self._scalar_row(f"{w_val:.6f}"), ""))
        grad_sample = grad[0].detach()
        rows.append(("grad = w*(noise_pred-noise) (norm)", self._scalar_row(f"{grad_sample.norm().item():.4f}"), ""))
        add_tensor_row("grad = w*(noise_pred-noise)", grad)
        # Grad diagnostics immediately after grad tensor
        try:
            g = grad[0].detach().cpu().float()  # C,T,H,W
            g_abs = g.abs()
            g_map = g_abs.mean(dim=(0, 1)).numpy()  # H,W
            heat = _make_heatmap_2d(g_map, label="mean |grad| over C,T")
            rows.append(("grad_heatmap = mean(|grad| over C,T)", heat, f"shape={heat.shape}"))
            g_hist = _make_hist_image(g_abs.reshape(-1).numpy(), title="hist |grad|")
            rows.append(("grad_hist = hist(|grad|)", g_hist, f"shape={g_hist.shape}"))
            # grad norm history curve with moving average
            if not hasattr(self, "_grad_norm_hist"):
                self._grad_norm_hist = []
            if not hasattr(self, "_grad_norm_window"):
                self._grad_norm_window = getattr(self.cfg, "debug_video_interval", 50)
            window = self._grad_norm_window
            g_norm = float(g.norm().item())
            self._grad_norm_hist.append(g_norm)
            vals = np.array(self._grad_norm_hist, dtype=np.float32)
            smoothed = []
            for i in range(len(vals)):
                start = max(0, i - window + 1)
                smoothed.append(float(vals[start : i + 1].mean()))
            curve = _make_line_chart(smoothed, title=f"grad_norm MA(window={window})")
            rows.append(("grad_norm_curve (MA)", curve, f"points={len(smoothed)}"))
        except Exception as e:
            shape_logs.append(f"grad viz failed: {e}")

        rows.append(("--- SDS ONE-STEP PREVIEW ---", self._scalar_row("x0_est and 1-step decode"), ""))
        if sds_x0_est is not None:
            rows.append(("sds_x0_est = (latents_noisy - sigma*noise)/(1-sigma) stats", self._scalar_row(f"shape={tuple(sds_x0_est.shape)}, min={sds_x0_est.min().item():.4f}, max={sds_x0_est.max().item():.4f}, norm={sds_x0_est.norm().item():.4f}"), ""))
        if sds_x0_est is not None:
            add_tensor_row("x0_est = (latents_noisy - sigma*noise)/(1-sigma)", sds_x0_est)
        else:
            rows.append(("x0_est = (latents_noisy - sigma*noise)/(1-sigma)", self._scalar_row("x0_est None"), "value"))
        # Decode a single frame of x0_est for quick inspection
        try:
            if sds_x0_est is not None:
                # sds_x0_est is (B,C,T,H,W); ensure it's on the right device for decode
                x0_for_decode = sds_x0_est.to(self.device)
                x0_video = self._decode_latents(x0_for_decode)[0]  # (C,T,H,W)
                frame0 = x0_video[:, 0].clamp(0, 1)
                frame0 = torch.nn.functional.interpolate(frame0.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)[0]
                img_np = (frame0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rows.append(("x0_est_frame0 = decode(x0_est)[t=0]", img_np, f"shape={img_np.shape}"))
        except Exception as e:
            shape_logs.append(f"x0_est decode failed: {e}")

        if prompt_embeds is not None:
            rows.append(("prompt_embeds = text_encoder(prompt)", self._scalar_row(f"shape={tuple(prompt_embeds.shape)}, norm={prompt_embeds.norm().item():.3f}"), ""))
            shape_logs.append(f"prompt_embeds shape={tuple(prompt_embeds.shape)}, norm={prompt_embeds.norm().item():.3f}")
        if prompt_mask is not None:
            rows.append(("prompt_mask = attention_mask(prompt)", self._scalar_row(f"shape={tuple(prompt_mask.shape)}, sum={prompt_mask.sum().item():.3f}"), ""))
            shape_logs.append(f"prompt_mask shape={tuple(prompt_mask.shape)}, sum={prompt_mask.sum().item():.3f}")

        rows.append(("--- SDS 1-STEP VIDEO ---", self._scalar_row("Decode x0_est / latents / latents_noisy -> video (bucketed)"), ""))
        # assemble panel
        # helper to add video rows with optional bucket upsample
        def add_video_row(label: str, vid: Optional[Tensor]):
            if vid is None:
                return
            C, T, H, W = vid.shape
            tiles = []
            for t in range(T):
                frame = vid[:, t]
                if C == 3:
                    img = frame
                else:
                    img = frame[0:1].repeat(3, 1, 1)
                img = img.clamp(0, 1)
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
                )[0]
                tiles.append(img)
            grid = torch.cat(tiles, dim=2)  # concat along width
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rows.append((label, grid_np, f"shape={(C, T, H, W)}"))

        def decode_and_bucket(lat: Optional[Tensor]) -> Optional[Tensor]:
            if lat is None:
                return None
            # Ensure latent is on the correct device for VAE decode
            lat_device = lat.to(self.device)
            video = self._decode_latents(lat_device)[0]  # (C,T,H,W)
            # match bucket resolution used elsewhere (pipe/sds_frames)
            try:
                target_h, target_w = self.pipe.get_closest_resolution_given_original_size(
                    origin_size=(rgb.shape[-1], rgb.shape[-2]), target_size=self.cfg.resolution
                )
            except Exception:
                target_h, target_w = rgb.shape[-2], rgb.shape[-1]
            if video.shape[-2:] != (target_h, target_w):
                video_thcw = video.permute(1, 0, 2, 3)
                video_up = torch.nn.functional.interpolate(
                    video_thcw, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                video = video_up.permute(1, 0, 2, 3)
            return video

        # x0_est preview: prefer cached sds_frames (already bucketed), else decode and bucket
        x0_decoded = sds_frames if sds_frames is not None else decode_and_bucket(sds_x0_est if sds_x0_est is not None else None)
        add_video_row("sds1step video = decode(x0_est)", x0_decoded)
        add_video_row("sds1step video = decode(latents)", decode_and_bucket(latents))
        add_video_row("sds1step video = decode(latents_noisy)", decode_and_bucket(latents_noisy))

        rows.append(("--- PIPE PREVIEW (NO SDS) ---", self._scalar_row("Pipeline multi-step decode"), ""))
        add_tensor_row("pipe_latents = pipe(..., output_type='latent')", pipe_latents.unsqueeze(0) if pipe_latents is not None else None)
        add_video_row("pipe video = decode(pipe_latents)", pipe_frames)

        # Add [HY-XX] prefix to each row label based on appearance order
        rows_with_prefix = []
        for idx, (label, img, shape_text) in enumerate(rows):
            prefix = f"[HY-{idx:02d}] "
            rows_with_prefix.append((prefix + label, img, shape_text))
        rows = rows_with_prefix

        panel_imgs = []
        label_w = 1500  # expanded label column width for readability
        shape_w = 500  # column for shape info
        # Unified font settings (same as _scalar_row)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.2
        FONT_THICKNESS = 2
        MIN_ROW_HEIGHT = 40  # minimum height to fit text
        for label, img, shape_text in rows:
            if not shape_text:
                shape_text = "value"
            h, w = img.shape[:2]
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            # Ensure minimum height for text readability
            row_h = max(h, MIN_ROW_HEIGHT)
            canvas = np.ones((row_h, label_w + shape_w + w, 3), dtype=np.uint8) * 255
            # Center image vertically if row is taller than image
            y_offset = (row_h - h) // 2
            canvas[y_offset:y_offset+h, label_w + shape_w:, :] = (img * 255 if img.dtype != np.uint8 else img).astype(np.uint8)
            # Text at vertical center
            text_y = row_h // 2 + 8
            cv2.putText(canvas, label, (10, text_y), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)
            if shape_text:
                cv2.putText(canvas, shape_text, (label_w + 10, text_y), FONT, FONT_SCALE, (0, 128, 0), FONT_THICKNESS, cv2.LINE_AA)
            panel_imgs.append(canvas)

        if panel_imgs:
            max_w = max(im.shape[1] for im in panel_imgs)
            padded = []
            for im in panel_imgs:
                if im.shape[1] < max_w:
                    pad_w = max_w - im.shape[1]
                    im = np.pad(im, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
                padded.append(im)
            panel = np.concatenate(padded, axis=0)
        save_path = os.path.join(debug_save_dir, f"it{debug_step}-panel_2_hunyuan.png")
        imageio.imwrite(save_path, panel)
        # cache rows for potential cross-guidance alignment panel
        self.last_panel_rows = rows
        # log shapes to stdout
        for ln in shape_logs:
            threestudio.info(f"[panel_hunyuan] {ln}")

    @torch.no_grad()
    def _debug_save_videos(
        self,
        rgb: Float[Tensor, "B 3 T H W"],
        prompt: str,
        latents_noisy: Float[Tensor, "B C T H W"],
        noise: Float[Tensor, "B C T H W"],
        sigma: Float[Tensor, ""],
        noise_pred: Float[Tensor, "B C T H W"],
        mask_type: str,
        debug_save_dir: str,
        debug_step: int,
        use_pipe: bool,
        use_one_step: bool,
        num_steps: int,
        video_length: int,
        debug_pipe_steps: int = 1,
        **kwargs,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        os.makedirs(debug_save_dir, exist_ok=True)
        aspect = f"{rgb.shape[-1]}:{rgb.shape[-2]}"

        # helper to upsample preview to bucket resolution for comparability
        def _upsample_to_bucket(video_cthw: Float[Tensor, "C T H W"]) -> Float[Tensor, "C T Hb Wb"]:
            try:
                target_h, target_w = self.pipe.get_closest_resolution_given_original_size(
                    origin_size=(rgb.shape[-1], rgb.shape[-2]), target_size=self.cfg.resolution
                )
            except Exception:
                target_h, target_w = rgb.shape[-2], rgb.shape[-1]
            if target_h == video_cthw.shape[-2] and target_w == video_cthw.shape[-1]:
                return video_cthw
            video_thcw = video_cthw.permute(1, 0, 2, 3)  # T,C,H,W
            video_up = F.interpolate(
                video_thcw, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            return video_up.permute(1, 0, 2, 3)

        # Full pipeline generation (multi-step denoise)
        pipe_frames = None
        pipe_latents = None
        if use_pipe:
            ref_image = None
            if mask_type == "i2v":
                frame = rgb[0, :, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                ref_image = (frame * 255).astype(np.uint8)
            try:
                out = self.pipe(
                    prompt=prompt,
                    aspect_ratio=aspect,
                    num_inference_steps=debug_pipe_steps,
                    video_length=video_length,
                    negative_prompt=self.cfg.negative_prompt,
                    seed=0,
                    enable_sr=False,
                    prompt_rewrite=False,
                    output_type="latent",
                    reference_image=ref_image,
                )
                # out is latent tensor
                latents_pipe = out if isinstance(out, torch.Tensor) else out[0]
                if latents_pipe.dim() == 4:
                    latents_pipe = latents_pipe.unsqueeze(2)
                pipe_latents = latents_pipe[0].detach().cpu()
                video_pipe = self._decode_latents(latents_pipe)[0]  # (C,T,H,W)
                video_pipe = _upsample_to_bucket(video_pipe)
                pipe_path = os.path.join(debug_save_dir, f"it{debug_step}-pipe_hunyuan.mp4")
                # self._save_video_tensor(video_pipe, pipe_path)
                pipe_frames = video_pipe.cpu()
            except Exception as e:
                threestudio.warn(f"Debug pipe generation failed: {e}")

        # Single-step latent -> decode preview (SDS path)
        sds_frames = None
        sds_x0_est = None
        if use_one_step:
            try:
                # reconstruct clean latents with known noise and schedule
                sigma_val = sigma
                while sigma_val.ndim < latents_noisy.ndim:
                    sigma_val = sigma_val.unsqueeze(-1)
                denom = (1.0 - sigma_val).clamp(min=1e-4)
                x0_est = (latents_noisy - sigma_val * noise) / denom
                video_1step = self._decode_latents(x0_est)[0]  # (C,T,H,W)
                video_1step = _upsample_to_bucket(video_1step)
                one_step_path = os.path.join(debug_save_dir, f"it{debug_step}-sds1step_hunyuan.mp4")
                # self._save_video_tensor(video_1step, one_step_path)
                sds_frames = video_1step.cpu()
                sds_x0_est = x0_est.detach().cpu()
            except Exception as e:
                threestudio.warn(f"Debug one-step decode failed: {e}")

        # return frames/latents for panel use
        return pipe_frames, sds_frames, pipe_latents, sds_x0_est

    def destroy(self) -> None:
        cleanup()

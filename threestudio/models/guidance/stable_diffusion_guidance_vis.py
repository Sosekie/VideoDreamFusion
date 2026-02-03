from dataclasses import dataclass
from typing import Optional
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio

import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import (
    StableDiffusionGuidance,
)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.typing import *


def _tile_channels(img_cthw: Float[Tensor, "C T H W"], target_hw: int = 128):
    C, T, H, W = img_cthw.shape
    per_frame = []
    for t in range(T):
        sl = img_cthw[:, t]
        per = []
        side = int(np.ceil(np.sqrt(C)))
        for c in range(C):
            ch = sl[c].unsqueeze(0).expand(3, -1, -1)
            ch = F.interpolate(
                ch.unsqueeze(0), size=(target_hw, target_hw), mode="bilinear", align_corners=False
            )[0]
            per.append(ch)
        while len(per) < side * side:
            per.append(torch.zeros_like(per[0]))
        rows = []
        for r in range(side):
            rows.append(torch.cat(per[r * side:(r + 1) * side], dim=2))
        grid = torch.cat(rows, dim=1)
        per_frame.append(grid.permute(1, 2, 0).cpu().numpy())
    return np.concatenate(per_frame, axis=1)


@threestudio.register("stable-diffusion-guidance-vis")
class StableDiffusionGuidanceVis(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        debug_save_dir: Optional[str] = None
        debug_step: Optional[int] = None
        debug_panel: bool = True

    cfg: Config

    @torch.no_grad()
    def __call__(
        self,
        rgb: Float[Tensor, "..."],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        debug_save_dir: Optional[str] = None,
        debug_step: Optional[int] = None,
        debug_panel: bool = True,
        **kwargs,
    ):
        # accept (B,3,H,W) or (B,3,T,H,W); normalize to 5D for logging
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(2)

        out = super().__call__(
            rgb,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            rgb_as_latents=rgb_as_latents,
            guidance_eval=guidance_eval,
            **kwargs,
        )

        d_save_dir = debug_save_dir or self.cfg.debug_save_dir
        d_step = debug_step or self.cfg.debug_step
        if d_save_dir is not None and d_step is not None and debug_panel:
            with torch.no_grad():
                B, _, T, H, W = rgb.shape
                rgb_norm = rgb.clamp(0, 1)
                video_in = (rgb_norm * 2 - 1).to(self.weights_dtype)
                posterior = self.vae.encode(video_in)
                latents = posterior.latent_dist.sample() * self.vae.config.scaling_factor  # (B,4,T,H/8,W/8) if VAE supports video, else fallback
                if latents.dim() == 5:
                    latents = latents[:, :, 0]  # take first frame for SD
                latents = latents  # (B,4,h,w)
                t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=self.device, dtype=torch.long)
                t = self.scheduler.timesteps[t_int].to(device=self.device, dtype=latents.dtype)
                noise = torch.randn_like(latents)
                alphas = self.alphas[t_int]
                sigma = torch.sqrt(1 - alphas)
                latents_noisy = (alphas[:, None, None, None] ** 0.5) * latents + sigma[:, None, None, None] * noise
                encoder_hidden_states = prompt_utils.text_embeddings
                noise_pred = self.forward_unet(latents_noisy, t, encoder_hidden_states)

                # SDS components
                w = sigma**2 if self.cfg.weighting_strategy == "sds" else 1.0
                grad = w[:, None, None, None] * (noise_pred - noise)
                x0_est = (latents_noisy - sigma[:, None, None, None] * noise) / (1 - sigma[:, None, None, None]).clamp(min=1e-4)

                self._save_panel(
                    d_save_dir,
                    d_step,
                    rgb,
                    rgb_norm,
                    video_in,
                    posterior,
                    latents,
                    t_int,
                    t,
                    sigma,
                    noise,
                    latents_noisy,
                    noise_pred,
                    grad,
                    x0_est,
                    prompt_utils.prompt,
                    getattr(self.cfg, "negative_prompt", ""),
                )

        return out

    @torch.no_grad()
    def _save_panel(
        self,
        debug_save_dir: str,
        debug_step: int,
        rgb: Float[Tensor, "B 3 T H W"],
        rgb_norm: Float[Tensor, "B 3 T H W"],
        video_in: Float[Tensor, "B 3 T H W"],
        posterior,
        latents: Float[Tensor, "B 4 H W"],
        t_int: Float[Tensor, "B"],
        t: Float[Tensor, "B"],
        sigma: Float[Tensor, "B"],
        noise: Float[Tensor, "B 4 H W"],
        latents_noisy: Float[Tensor, "B 4 H W"],
        noise_pred: Float[Tensor, "B 4 H W"],
        grad: Float[Tensor, "B 4 H W"],
        x0_est: Float[Tensor, "B 4 H W"],
        prompt: str,
        negative_prompt: str,
    ):
        os.makedirs(debug_save_dir, exist_ok=True)
        rows = []
        shape_logs = []

        def add_tensor_row(label: str, tensor: Float[Tensor, "..."]):
            if tensor is None:
                shape_logs.append(f"{label}: None")
                return
            t_cpu = tensor[0].detach().float()
            if t_cpu.ndim == 4:  # C,H,W
                cthw = t_cpu.unsqueeze(1)  # C,1,H,W
            elif t_cpu.ndim == 3:
                cthw = t_cpu.unsqueeze(0)  # 1,C,H,W -> but expect C,T,H,W; treat as T=1 below
                cthw = cthw.permute(1, 0, 2, 3)
            else:
                shape_logs.append(f"{label}: shape={tuple(t_cpu.shape)} (skipped)")
                return
            shape_logs.append(f"{label}: shape={tuple(cthw.shape)}")
            row_img, pca_info = _tile_channels(cthw)
            rows.append((label, row_img))
            if pca_info:
                shape_logs.append(f"{label} PCA info: {pca_info}")

        # scalars
        rows.append(("num_train_timesteps", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{self.num_train_timesteps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("min_step", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{self.min_step}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("max_step", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{self.max_step}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("t_int", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{t_int.cpu().tolist()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("t", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{t.cpu().tolist()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("sigma", np.ones((60, 800, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{sigma.cpu().tolist()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("prompt", np.ones((60, 1200, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{prompt}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(("negative_prompt", np.ones((60, 1200, 3), dtype=np.uint8) * 255))
        cv2.putText(rows[-1][1], f"{negative_prompt}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        add_tensor_row("rgb", rgb)
        add_tensor_row("rgb_norm = rgb.clamp(0,1)", rgb_norm)
        add_tensor_row("video_in = (rgb_norm*2-1).to(weights_dtype)", video_in)
        add_tensor_row("posterior.mean (vae.encode)", posterior.latent_dist.mean if hasattr(posterior, "latent_dist") else None)
        add_tensor_row("posterior.std (vae.encode)", posterior.latent_dist.stddev if hasattr(posterior, "latent_dist") else None)
        add_tensor_row("latents=posterior.sample()*scaling_factor", latents)
        add_tensor_row("noise=randn_like(latents)", noise)
        add_tensor_row("latents_noisy = sqrt(alpha)*latents + sigma*noise", latents_noisy)
        add_tensor_row("noise_pred (transformer output)", noise_pred)
        add_tensor_row("grad = w*(noise_pred-noise)", grad)
        add_tensor_row("x0_est = (latents_noisy - sigma*noise)/(1-sigma)", x0_est)

        # stack
        max_w = max(img.shape[1] for _, img in rows)
        padded = []
        for label, img in rows:
            h, w, _ = img.shape
            if w < max_w:
                img = np.pad(img, ((0, 0), (0, max_w - w), (0, 0)), constant_values=255)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            padded.append(img)
        panel = np.concatenate(padded, axis=0)
        imageio.imwrite(os.path.join(debug_save_dir, f"it{debug_step}-panel_sd.png"), panel.astype(np.uint8))
        for ln in shape_logs:
            threestudio.info(f"[panel_sd] {ln}")

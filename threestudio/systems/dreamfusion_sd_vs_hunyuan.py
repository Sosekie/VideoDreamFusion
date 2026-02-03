import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.dreamfusion_hunyuanvideo import DreamFusionHunyuanVideo
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("dreamfusion-sd-vs-hunyuan-system")
class DreamFusionSDvsHunyuan(DreamFusionHunyuanVideo):
    @dataclass
    class Config(DreamFusionHunyuanVideo.Config):
        # SD guidance config block name
        sd_guidance_type: str = "stable-diffusion-guidance-vis"
        sd_guidance: dict = None

    cfg: Config

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # swap Hunyuan guidance to explicit attr, and add SD guidance
        self.hunyuan_guidance = self.guidance  # keep for logging
        self.sd_guidance = threestudio.find(self.cfg.sd_guidance_type)(self.cfg.sd_guidance)

    def training_step(self, batch, batch_idx):
        # prepare cameras and render (same as parent)
        multi_view_batch = self._prepare_arc_batch(batch)
        out = self(multi_view_batch)

        comp_rgb = out["comp_rgb"]
        if comp_rgb.dim() != 4 or comp_rgb.shape[-1] != 3:
            raise ValueError(f"Expected comp_rgb with shape [N, H, W, 3], got {comp_rgb.shape}")

        T = self.cfg.n_arc_views
        total_views, H, W, _ = comp_rgb.shape
        if total_views % T != 0:
            raise ValueError(f"Total rendered views {total_views} not divisible by n_arc_views {T}")
        B = total_views // T
        comp_rgb_video = comp_rgb.view(B, T, H, W, 3).permute(0, 4, 1, 2, 3).contiguous()
        out["comp_rgb"] = comp_rgb_video

        # debug panel image
        rows = []
        for b in range(B):
            cols = []
            for t in range(T):
                frame_chw = comp_rgb_video[b, :, t, :, :].detach().cpu().numpy()
                frame_hwc = (np.clip(frame_chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
                cols.append(frame_hwc)
            rows.append(np.concatenate(cols, axis=1))
        panel = np.concatenate(rows, axis=0)
        self.save_image(f"it{self.true_global_step}-panel.png", panel)

        comp_rgb_for_guidance = out["comp_rgb"]  # (B,3,T,H,W)
        prompt_utils = self.prompt_processor()

        # Hunyuan forward (no grad) for comparison / logging
        with torch.no_grad():
            _ = self.hunyuan_guidance(
                comp_rgb_for_guidance,
                prompt_utils,
                **multi_view_batch,
                rgb_as_latents=False,
                debug_save_dir=self.get_save_path("debug_videos") if self.cfg.debug_video_interval > 0 and self.true_global_step % self.cfg.debug_video_interval == 0 else None,
                debug_step=self.true_global_step,
                debug_panel=True,
            )

        # SD guidance (used for loss/backprop)
        sd_out = self.sd_guidance(
            comp_rgb_for_guidance[:, :, 0],  # SD expects image (B,3,H,W)
            prompt_utils,
            **multi_view_batch,
            rgb_as_latents=False,
            debug_save_dir=self.get_save_path("debug_videos") if self.cfg.debug_video_interval > 0 and self.true_global_step % self.cfg.debug_video_interval == 0 else None,
            debug_step=self.true_global_step,
            debug_panel=True,
        )

        loss = 0.0
        for name, value in sd_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/sd/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
            loss_orient = (out["weights"].detach() * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2).sum() / (
                out["opacity"] > 0
            ).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

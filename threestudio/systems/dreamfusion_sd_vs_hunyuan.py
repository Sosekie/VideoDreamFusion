import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.dreamfusion_hunyuanvideo import DreamFusionHunyuanVideo
from threestudio.utils.ops import (
    binary_cross_entropy,
    dot,
    get_ray_directions,
    get_rays,
    get_projection_matrix,
    get_mvp_matrix,
)
from threestudio.utils.typing import *


@threestudio.register("dreamfusion-sd-vs-hunyuan-system")
class DreamFusionSDvsHunyuan(DreamFusionHunyuanVideo):
    @dataclass
    class Config(DreamFusionHunyuanVideo.Config):
        # SD guidance config block name
        sd_guidance_type: str = "stable-diffusion-guidance-vis"
        sd_guidance: dict = None
        sd_prompt_processor_type: str = "stable-diffusion-prompt-processor"
        sd_prompt_processor: dict = None
        # Separate render resolutions: [width, height]
        sd_render_resolution: Tuple[int, int] = (64, 64)
        hy_render_resolution: Tuple[int, int] = (64, 64)
        # Gradient switches
        sd_guidance_requires_grad: bool = True
        hy_guidance_requires_grad: bool = False

    cfg: Config

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # swap Hunyuan guidance to explicit attr, and add SD guidance
        self.hunyuan_guidance = self.guidance  # keep for logging
        self.sd_guidance = threestudio.find(self.cfg.sd_guidance_type)(self.cfg.sd_guidance)
        self.sd_prompt_processor = threestudio.find(self.cfg.sd_prompt_processor_type)(self.cfg.sd_prompt_processor)

    def training_step(self, batch, batch_idx):
        # ===================================================================
        # Render once at HIGH resolution (Hunyuan), then DOWNSAMPLE for SD.
        # This ensures IDENTICAL colors for both (same lighting, same random seed).
        # ===================================================================

        # Sample arc trajectory cameras once (without fixing resolution)
        base_batch = self._prepare_arc_batch(batch)

        # Build Hunyuan batch with high resolution
        sd_w, sd_h = self.cfg.sd_render_resolution
        hy_w, hy_h = self.cfg.hy_render_resolution
        hy_batch = self._batch_with_resolution(base_batch, hy_h, hy_w)

        # Render ONCE at Hunyuan (high) resolution
        hy_out = self(hy_batch)
        comp_rgb_hy = hy_out["comp_rgb"]  # (N, H_hy, W_hy, 3) high-res

        if comp_rgb_hy.dim() != 4 or comp_rgb_hy.shape[-1] != 3:
            raise ValueError(f"Expected comp_rgb with shape [N, H, W, 3], got {comp_rgb_hy.shape}")

        # Downsample to SD resolution for identical colors
        # (N, H_hy, W_hy, 3) -> (N, 3, H_hy, W_hy) -> interpolate -> (N, 3, H_sd, W_sd) -> (N, H_sd, W_sd, 3)
        comp_rgb_hy_nchw = comp_rgb_hy.permute(0, 3, 1, 2)  # (N, 3, H_hy, W_hy)
        comp_rgb_sd_nchw = F.interpolate(comp_rgb_hy_nchw, size=(sd_h, sd_w), mode='bilinear', align_corners=False)
        comp_rgb_sd_full = comp_rgb_sd_nchw.permute(0, 2, 3, 1)  # (N, H_sd, W_sd, 3)

        # Reshape SD renders for SDS
        T = self.cfg.n_arc_views
        total_views, H_sd, W_sd, _ = comp_rgb_sd_full.shape
        if total_views % T != 0:
            raise ValueError(f"Total rendered views {total_views} not divisible by n_arc_views {T}")
        B = total_views // T
        comp_rgb_sd_video = comp_rgb_sd_full.view(B, T, H_sd, W_sd, 3).permute(0, 4, 1, 2, 3).contiguous()
        comp_rgb_sd = comp_rgb_sd_video.permute(0, 2, 3, 4, 1).reshape(total_views, H_sd, W_sd, 3)

        prompt_utils_hy = self.prompt_processor()
        prompt_utils_sd = self.sd_prompt_processor()

        # Also create sd_batch for passing camera params to guidance (rays not used since we downsampled)
        sd_batch = self._batch_with_resolution(base_batch, sd_h, sd_w)

        # Reshape Hunyuan renders for video guidance
        total_views_hy, H_hy, W_hy, _ = comp_rgb_hy.shape
        if total_views_hy % T != 0:
            raise ValueError(f"Total rendered views {total_views_hy} not divisible by n_arc_views {T}")
        B_hy = total_views_hy // T
        comp_rgb_hy_video = comp_rgb_hy.view(B_hy, T, H_hy, W_hy, 3).permute(0, 4, 1, 2, 3).contiguous()

        hy_guidance_call = lambda: self.hunyuan_guidance(
            comp_rgb_hy_video,
            prompt_utils_hy,
            **hy_batch,
            rgb_as_latents=False,
            debug_save_dir=self.get_save_path("debug_videos") if self.cfg.debug_video_interval > 0 and self.true_global_step % self.cfg.debug_video_interval == 0 else None,
            debug_step=self.true_global_step,
            debug_panel=True,
        )

        if self.cfg.hy_guidance_requires_grad:
            hy_guidance_out = hy_guidance_call()
        else:
            with torch.no_grad():
                hy_guidance_out = hy_guidance_call()

        # SD guidance uses the exact same renders (flattened BHWC)
        sd_call = lambda: self.sd_guidance(
            comp_rgb_sd,
            prompt_utils_sd,
            **sd_batch,
            rgb_as_latents=False,
            debug_save_dir=self.get_save_path("debug_videos") if self.cfg.debug_video_interval > 0 and self.true_global_step % self.cfg.debug_video_interval == 0 else None,
            debug_step=self.true_global_step,
            debug_panel=True,
        )
        if self.cfg.sd_guidance_requires_grad:
            sd_out = sd_call()
        else:
            with torch.no_grad():
                sd_out = sd_call()

        # If both guidances produced debug rows, build an aligned joint panel.
        save_dir = self.get_save_path("debug_videos") if self.cfg.debug_video_interval > 0 and self.true_global_step % self.cfg.debug_video_interval == 0 else None
        if save_dir is not None:
            sd_rows = getattr(self.sd_guidance, "last_panel_rows", None)
            hy_rows = getattr(self.hunyuan_guidance, "last_panel_rows", None)
            if sd_rows is not None and hy_rows is not None:
                joint_path = os.path.join(save_dir, f"it{self.true_global_step}-panel_3_sd_hunyuan.png")
                self._save_joint_panel(sd_rows, hy_rows, joint_path)

        loss = 0.0
        # SD losses (only add if grad enabled)
        if self.cfg.sd_guidance_requires_grad:
            for name, value in sd_out.items():
                if not (type(value) is torch.Tensor and value.numel() > 1):
                    self.log(f"train/sd/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        else:
            # Log SD metrics without adding to loss
            for name, value in sd_out.items():
                if not (type(value) is torch.Tensor and value.numel() > 1):
                    self.log(f"train/sd/{name}", value)
        # Hunyuan losses (only add if grad enabled)
        if self.cfg.hy_guidance_requires_grad:
            for name, value in hy_guidance_out.items():
                if not (type(value) is torch.Tensor and value.numel() > 1):
                    self.log(f"train/hy/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in hy_out:
                raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
            loss_orient = (
                hy_out["weights"].detach()
                * dot(hy_out["normal"], hy_out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (hy_out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (hy_out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = hy_out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def _batch_with_resolution(self, base_batch: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
        """
        Rebuild ray/direction tensors for a new spatial resolution while keeping
        camera extrinsics/intrinsics (fovy) identical to base_batch.
        """
        device = base_batch["rays_o"].device
        num_views = base_batch["rays_o"].shape[0]

        fovy = base_batch["fovy"]  # radians, shape [num_views]
        c2w = base_batch["c2w"]

        directions_unit_focal = get_ray_directions(H=height, W=width, focal=1.0).to(device)
        directions = directions_unit_focal[None, :, :, :].repeat(num_views, 1, 1, 1)
        focal_length = 0.5 * height / torch.tan(0.5 * fovy)
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

        proj_mtx = get_projection_matrix(fovy, float(width) / float(height), 0.01, 100.0).to(device)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        new_batch = dict(base_batch)
        new_batch.update(
            {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "mvp_mtx": mvp_mtx,
                "proj_mtx": proj_mtx,
                "height": height,
                "width": width,
                # keep other fields (camera_positions, light_positions, etc.)
            }
        )
        return new_batch

    @staticmethod
    def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w >= width:
            return img
        pad_w = width - w
        return np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=255)

    @staticmethod
    def _text_row(text: str, width: int, height: int = 40, color=(0, 0, 0)) -> np.ndarray:
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, text, (10, height // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
        return canvas

    def _save_joint_panel(self, sd_rows, hy_rows, save_path: str):
        """
        Build a joint panel with format:
        | Label | SD shape | SD image | HY shape | HY image |
        
        Each row from SD and HunyuanVideo is aligned by label.
        Images are NOT resized - original dimensions preserved.
        """
        import re
        
        # Consistent font settings (same as scalar_row in guidance files)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.2
        FONT_THICKNESS = 2
        TEXT_HEIGHT = 35  # line height for text

        # Helper to strip [SD-XX] or [HY-XX] prefix from label
        def strip_prefix(label: str) -> str:
            match = re.match(r'^\[(SD|HY)-\d+\]\s*', label)
            if match:
                return label[match.end():]
            return label

        # Build label -> (img, shape_text) maps using stripped labels
        # Keep original labels for display
        sd_labels_stripped = [strip_prefix(lbl) for lbl, _, _ in sd_rows]
        hy_labels_stripped = [strip_prefix(lbl) for lbl, _, _ in hy_rows]
        sd_map = {strip_prefix(label): (img, shape, label) for label, img, shape in sd_rows}
        hy_map = {strip_prefix(label): (img, shape, label) for label, img, shape in hy_rows}
        
        # Align Hunyuan x0_est label to SD label for joint panel
        hy_x0_label = "x0_est = (latents_noisy - sigma*noise)/(1-sigma)"
        sd_x0_label = "x0_est = (latents_noisy - sigma*noise_pred)/sqrt(alpha)"
        if hy_x0_label in hy_map and sd_x0_label not in hy_map:
            data = hy_map[hy_x0_label]
            insert_pos = hy_labels_stripped.index(hy_x0_label) + 1
            hy_labels_stripped.insert(insert_pos, sd_x0_label)
            hy_map[sd_x0_label] = data

        # Canonical execution order to align similar steps
        canonical_order = [
            "num_train_timesteps",
            "num_train_timesteps=len(scheduler.timesteps)",
            "min_step",
            "max_step",
            "t_int = randint(min_step, max_step)",
            "t = scheduler.timesteps[t_int]",
            "sigma = sqrt(1-alpha)",
            "sigma = scheduler.sigmas[t_int]",
            "prompt = text",
            "prompt",
            "negative_prompt = text",
            "negative_prompt",
            "--- INPUT / ENCODE ---",
            "rgb",
            "rgb_norm = rgb.clamp(0,1)",
            "video_in = (rgb_norm*2-1).to(weights_dtype)",
            "posterior.mean (vae.encode)",
            "posterior = vae.encode(video_in)",
            "latents = posterior.sample()*scaling_factor",
            "latents = posterior.sample()*vae.config.scaling_factor",
            "noise = randn_like(latents)",
            "latents_noisy = sqrt(alpha)*latents + sigma*noise",
            "latents_noisy = latents + sigma*noise",
            "--- COND & CONCAT ---",
            "cond_latents = (t2v: zeros+mask | i2v: encode rgb[0] + mask)",
            "cond_latents = prepare_cond(latents_noisy, task)",
            "latents_concat = concat(latents_noisy, cond_latents)",
            "--- UNET / SDS ---",
            "noise_pred (unet output)",
            "noise_pred = transformer(latents_concat, t, prompts)",
            "w = sigma**2",
            "grad = w*(noise_pred-noise) (norm)",
            "grad = w*(noise_pred-noise)",
            "grad_heatmap = mean(|grad| over C)",
            "grad_heatmap = mean(|grad| over C,T)",
            "grad_hist = hist(|grad|)",
            "grad_norm_curve (MA)",
            "x0_est = (latents_noisy - sigma*noise_pred)/sqrt(alpha)",
            "--- SDS ONE-STEP PREVIEW ---",
            "sds_x0_est = (latents_noisy - sigma*noise)/(1-sigma) stats",
            "x0_est = (latents_noisy - sigma*noise)/(1-sigma)",
            "x0_est_frame0 = decode(x0_est)[t=0]",
            "--- DECODER OUTPUTS ---",
            "latents_decoded = vae.decode(latents)",
            "latents_noisy_decoded = vae.decode(latents_noisy)",
            "--- SDS DENOISED OUTPUT ---",
            "x0_est_decoded = vae.decode(x0_est)",
            "--- SDS 1-STEP VIDEO ---",
            "sds1step video = decode(x0_est)",
            "sds1step video = decode(latents)",
            "sds1step video = decode(latents_noisy)",
            "--- PIPE PREVIEW (NO SDS) ---",
            "pipe_latents = pipe(..., output_type='latent')",
            "pipe video = decode(pipe_latents)",
            "prompt_embeds = text_encoder(prompt)",
            "prompt_mask = attention_mask(prompt)",
            "text_embeddings",
        ]
        canon_rank = {lbl: i for i, lbl in enumerate(canonical_order)}

        # Merge sd_labels_stripped and hy_labels_stripped preserving their own order,
        # preferring lower canonical rank when both available.
        i = j = 0
        label_order = []
        seen = set()
        while i < len(sd_labels_stripped) or j < len(hy_labels_stripped):
            lbl_sd = sd_labels_stripped[i] if i < len(sd_labels_stripped) else None
            lbl_hy = hy_labels_stripped[j] if j < len(hy_labels_stripped) else None
            if lbl_sd is not None and lbl_hy is not None and lbl_sd == lbl_hy:
                lbl = lbl_sd
                i += 1
                j += 1
            elif lbl_sd is not None and lbl_hy is not None:
                r_sd = canon_rank.get(lbl_sd, float("inf"))
                r_hy = canon_rank.get(lbl_hy, float("inf"))
                if r_sd <= r_hy:
                    lbl = lbl_sd
                    i += 1
                else:
                    lbl = lbl_hy
                    j += 1
            elif lbl_sd is not None:
                lbl = lbl_sd
                i += 1
            else:
                lbl = lbl_hy
                j += 1
            if lbl not in seen:
                label_order.append(lbl)
                seen.add(lbl)

        # Calculate max image widths for each side (use actual image widths, no resize)
        sd_img_w = 100  # minimum default
        hy_img_w = 100  # minimum default
        for _, img, _ in sd_rows:
            if img is not None:
                sd_img_w = max(sd_img_w, img.shape[1])
        for _, img, _ in hy_rows:
            if img is not None:
                hy_img_w = max(hy_img_w, img.shape[1])

        # Column widths - adaptive based on content (wider for larger font) - doubled
        label_w = 1400  # label column for joint panel row names
        shape_w = 560   # enough for shape text with large font
        
        total_w = label_w + shape_w + sd_img_w + shape_w + hy_img_w

        def ensure_uint8_rgb(img):
            """Ensure image is uint8 RGB with 3 channels."""
            if img is None:
                return None
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            return img

        def get_image_or_blank(data, target_w):
            """Get image from data, or create blank. NO resizing."""
            if data is None:
                return np.ones((30, target_w, 3), dtype=np.uint8) * 255
            img = ensure_uint8_rgb(data[0])
            if img is None:
                return np.ones((30, target_w, 3), dtype=np.uint8) * 255
            return img

        def make_text_cell(text, width, height, color=(0, 0, 0)):
            """Create a text cell with consistent font."""
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            # Truncate text if too long (larger font = fewer chars per pixel)
            max_chars = width // 14
            if len(text) > max_chars:
                text = text[:max_chars-3] + "..."
            y_pos = height // 2 + 8  # center vertically
            cv2.putText(canvas, text, (5, y_pos), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
            return canvas

        # Build header row
        header_h = 45  # taller for larger font
        header = np.ones((header_h, total_w, 3), dtype=np.uint8) * 230  # light gray
        y_text = header_h // 2 + 8  # center vertically
        cv2.putText(header, "Label", (10, y_text), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(header, "SD Shape", (label_w + 5, y_text), FONT, FONT_SCALE, (0, 128, 0), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(header, "SD Image", (label_w + shape_w + 5, y_text), FONT, FONT_SCALE, (0, 128, 0), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(header, "HY Shape", (label_w + shape_w + sd_img_w + 5, y_text), FONT, FONT_SCALE, (128, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(header, "HY Image", (label_w + shape_w * 2 + sd_img_w + 5, y_text), FONT, FONT_SCALE, (128, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

        blocks = [header]

        # Track indices for each prefix type
        jn_idx = 0  # Joint (both SD and HY have this row)
        sd_only_idx = 0  # SD-only rows
        hy_only_idx = 0  # HY-only rows
        
        for label in label_order:
            sd_data = sd_map.get(label)
            hy_data = hy_map.get(label)

            # Get images WITHOUT resizing
            sd_img = get_image_or_blank(sd_data, sd_img_w)
            hy_img = get_image_or_blank(hy_data, hy_img_w)
            
            # Get shape texts (use 'value' for scalar rows, 'N/A' only if that side is missing)
            if sd_data:
                sd_shape_text = sd_data[1] if sd_data[1] else "value"
            else:
                sd_shape_text = "N/A"
            if hy_data:
                hy_shape_text = hy_data[1] if hy_data[1] else "value"
            else:
                hy_shape_text = "N/A"

            # Determine row height (max of both images, minimum 45 for text)
            row_h = max(sd_img.shape[0], hy_img.shape[0], 45)

            # Create row
            row = np.ones((row_h, total_w, 3), dtype=np.uint8) * 255

            # Determine prefix based on whether both sides have this row
            if sd_data and hy_data:
                # Joint row - both SD and HY have this
                display_label = f"[JN-{jn_idx:02d}] {label}"
                jn_idx += 1
                label_color = (128, 0, 128)  # Purple for joint
            elif sd_data:
                # SD-only row
                display_label = f"[SD-{sd_only_idx:02d}] {label}"
                sd_only_idx += 1
                label_color = (0, 128, 0)  # Green for SD
            else:
                # HY-only row
                display_label = f"[HY-{hy_only_idx:02d}] {label}"
                hy_only_idx += 1
                label_color = (128, 0, 0)  # Red for HY

            # Label column
            label_cell = make_text_cell(display_label, label_w, row_h, label_color)
            row[:, :label_w, :] = label_cell

            # SD shape column
            sd_shape_cell = make_text_cell(sd_shape_text, shape_w, row_h, (0, 128, 0))
            row[:, label_w:label_w + shape_w, :] = sd_shape_cell

            # SD image column - pad to column width, preserve original size
            sd_img_padded = np.ones((row_h, sd_img_w, 3), dtype=np.uint8) * 255
            h_sd, w_sd = sd_img.shape[:2]
            sd_img_padded[:min(h_sd, row_h), :min(w_sd, sd_img_w), :] = sd_img[:min(h_sd, row_h), :min(w_sd, sd_img_w), :]
            row[:, label_w + shape_w:label_w + shape_w + sd_img_w, :] = sd_img_padded

            # HY shape column
            hy_shape_cell = make_text_cell(hy_shape_text, shape_w, row_h, (128, 0, 0))
            row[:, label_w + shape_w + sd_img_w:label_w + shape_w * 2 + sd_img_w, :] = hy_shape_cell

            # HY image column - pad to column width, preserve original size
            hy_img_padded = np.ones((row_h, hy_img_w, 3), dtype=np.uint8) * 255
            h_hy, w_hy = hy_img.shape[:2]
            hy_img_padded[:min(h_hy, row_h), :min(w_hy, hy_img_w), :] = hy_img[:min(h_hy, row_h), :min(w_hy, hy_img_w), :]
            row[:, label_w + shape_w * 2 + sd_img_w:, :] = hy_img_padded

            blocks.append(row)

            # Add thin separator line
            sep = np.ones((1, total_w, 3), dtype=np.uint8) * 200
            blocks.append(sep)

        if blocks:
            panel = np.concatenate(blocks, axis=0)
            # Ensure contiguous array for cv2
            panel = np.ascontiguousarray(panel)
            cv2.imwrite(save_path, panel[:, :, ::-1])  # RGB->BGR

from dataclasses import dataclass
from typing import Optional, Tuple
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


@threestudio.register("stable-diffusion-guidance-vis")
class StableDiffusionGuidanceVis(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        debug_save_dir: Optional[str] = None
        debug_step: Optional[int] = None
        debug_panel: bool = True

    cfg: Config

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
                # Normalize to [0, 1]
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max - arr_min > 1e-6:
                    arr = (arr - arr_min) / (arr_max - arr_min)
                arr = np.clip(arr, 0.0, 1.0)
            else:
                per = []
                side = int(np.ceil(np.sqrt(C)))
                for c in range(C):
                    ch = slice_c[c].unsqueeze(0)  # 1,H,W
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
            comps = Vh[:3]  # (3, C)
            proj = data @ comps.T  # (N, 3)
            proj -= proj.min(dim=0, keepdim=True)[0]
            denom = proj.max(dim=0, keepdim=True)[0].clamp(min=1e-6)
            proj = proj / denom
            pca_img = proj.reshape(H, W, 3)
            pca_img = cv2.resize(pca_img.numpy(), (target_hw, target_hw), interpolation=cv2.INTER_AREA)
            pca_rgb = np.clip(pca_img, 0.0, 1.0)
            # Record top channel contributions for each PC
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
            return np.pad(img, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=1.0)

        pca_rgb = pad_h(pca_rgb, h)
        grid_img = pad_h(grid_img, h)
        combined = np.concatenate([pca_rgb, grid_img], axis=1)
        
        # Convert to uint8
        combined = (combined * 255).clip(0, 255).astype(np.uint8)

        return combined, pca_info

    @staticmethod
    def _scalar_row(text: str, width: int = 400, height: int = 40) -> np.ndarray:
        width = int(width * 1.5)
        height = int(height * 2)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, text, (10, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        return canvas

    @torch.no_grad()
    def _decode_latents_to_image(
        self, latents: Float[Tensor, "B 4 H W"]
    ) -> Float[Tensor, "B 3 H W"]:
        """Decode latents back to RGB image using VAE decoder.
        
        Args:
            latents: Latent tensor of shape (B, 4, h, w)
            
        Returns:
            RGB image tensor of shape (B, 3, H, W) in range [0, 1]
        """
        # Undo the scaling factor
        latents_unscaled = latents / self.vae.config.scaling_factor
        # Decode with VAE
        decoded = self.vae.decode(latents_unscaled.to(self.weights_dtype)).sample
        # Convert from [-1, 1] to [0, 1]
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        return decoded.to(latents.dtype)

    # NOTE: Do NOT use @torch.no_grad() here! It would block gradient propagation for SDS loss!
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
        # accept (B,H,W,3) or (B,3,H,W) or (B,3,T,H,W); normalize to BCHW for parent
        original_rgb = rgb
        if rgb.dim() == 5:
            rgb_bchw = rgb[:, :, 0]  # (B,3,H,W) first frame
        elif rgb.dim() == 4 and rgb.shape[1] == 3:
            rgb_bchw = rgb  # already BCHW
        elif rgb.dim() == 4 and rgb.shape[-1] == 3:
            rgb_bchw = rgb.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported rgb shape {rgb.shape} for SD guidance vis")
        # for panel use 5D with T=1
        rgb_panel = rgb_bchw.unsqueeze(2)  # (B,3,1,H,W)
        # SD base class expects BHWC
        rgb = rgb_bchw.permute(0, 2, 3, 1).contiguous()

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
        d_step = debug_step if debug_step is not None else self.cfg.debug_step
        if d_save_dir is not None and d_step is not None and debug_panel:
            with torch.no_grad():
                B, _, T, H, W = rgb_panel.shape
                rgb_norm = rgb_panel.clamp(0, 1)
                # collapse T for VAE: BCHW
                rgb_bchw_local = rgb_norm[:, :, 0]
                video_in = (rgb_bchw_local * 2 - 1).to(self.weights_dtype)
                posterior = self.vae.encode(video_in)
                latents = posterior.latent_dist.sample() * self.vae.config.scaling_factor  # (B,4,h,w)
                t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=self.device, dtype=torch.long)
                # Scheduler buffers are on CPU; index there, then move to device
                t_cpu = t_int.detach().cpu()
                t = self.scheduler.timesteps[t_cpu].to(device=self.device, dtype=latents.dtype)
                noise = torch.randn_like(latents)
                alphas = self.alphas[t_cpu].to(device=self.device, dtype=latents.dtype)
                sigma = torch.sqrt(1 - alphas)
                latents_noisy = (alphas[:, None, None, None] ** 0.5) * latents + sigma[:, None, None, None] * noise
                encoder_hidden_states = prompt_utils.text_embeddings
                noise_pred = self.forward_unet(latents_noisy, t, encoder_hidden_states)

                # SDS components
                w = sigma**2 if self.cfg.weighting_strategy == "sds" else 1.0
                grad = w[:, None, None, None] * (noise_pred - noise)
                x0_est = (latents_noisy - sigma[:, None, None, None] * noise_pred) / (alphas[:, None, None, None] ** 0.5).clamp(min=1e-4)

                # ========== DECODER OUTPUTS ==========
                # Decode latents back to RGB for visualization
                latents_decoded = self._decode_latents_to_image(latents)  # (B,3,H,W)
                x0_est_decoded = self._decode_latents_to_image(x0_est)    # (B,3,H,W)
                latents_noisy_decoded = self._decode_latents_to_image(latents_noisy)  # (B,3,H,W)

                self._save_panel(
                    d_save_dir,
                    d_step,
                    rgb_panel,
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
                    encoder_hidden_states,
                    # Decoded images for visualization
                    latents_decoded,
                    x0_est_decoded,
                    latents_noisy_decoded,
                )

        return out

    @torch.no_grad()
    def _save_panel(
        self,
        debug_save_dir: str,
        debug_step: int,
        rgb: Float[Tensor, "B 3 T H W"],
        rgb_norm: Float[Tensor, "B 3 T H W"],
        video_in: Float[Tensor, "B 3 H W"],
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
        encoder_hidden_states: Optional[Tensor] = None,
        # Decoded images
        latents_decoded: Optional[Float[Tensor, "B 3 H W"]] = None,
        x0_est_decoded: Optional[Float[Tensor, "B 3 H W"]] = None,
        latents_noisy_decoded: Optional[Float[Tensor, "B 3 H W"]] = None,
    ):
        os.makedirs(debug_save_dir, exist_ok=True)
        rows = []
        shape_logs = []

        def _make_hist_image(arr: np.ndarray, bins: int = 30, width: int = 420, height: int = 160) -> np.ndarray:
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
            cv2.putText(img, "hist |grad|", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 1, cv2.LINE_AA)
            # min/max ticks
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
            # y-axis min/max
            cv2.putText(img, f"min={vmin:.2e}", (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            txt = f"max={vmax:.2e}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img, txt, (width - tw - 5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            return img

        def add_tensor_row(label: str, tensor: Float[Tensor, "..."]):
            # assume B=1
            if tensor is None:
                shape_logs.append(f"{label}: None")
                return
            t_cpu = tensor[0].detach().float()
            if t_cpu.ndim == 4:  # C,T,H,W expected
                cthw = t_cpu
            elif t_cpu.ndim == 3:  # C,H,W -> add T=1
                cthw = t_cpu.unsqueeze(1)  # C,1,H,W
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

        def add_rgb_image_row(label: str, tensor: Float[Tensor, "B 3 H W"], target_hw: int = 128):
            """Add a decoded RGB image as a simple image row (no channel grid)."""
            if tensor is None:
                shape_logs.append(f"{label}: None")
                return
            # Take first batch item: (3, H, W)
            img = tensor[0].detach().cpu().float()
            img = img.clamp(0, 1)
            # Convert to HWC numpy
            img_np = img.permute(1, 2, 0).numpy()
            # Resize to target size
            img_np = cv2.resize(img_np, (target_hw, target_hw), interpolation=cv2.INTER_AREA)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            shape_logs.append(f"{label}: shape={tuple(tensor[0].shape)}")
            rows.append((label, img_np, f"shape={tuple(tensor[0].shape)}"))

        # scalar/info rows
        rows.append(("num_train_timesteps", self._scalar_row(f"{self.num_train_timesteps}"), ""))
        rows.append(("min_step", self._scalar_row(f"{self.min_step}"), ""))
        rows.append(("max_step", self._scalar_row(f"{self.max_step}"), ""))
        shape_logs.append(f"num_train_timesteps={self.num_train_timesteps}, min_step={self.min_step}, max_step={self.max_step}")
        sigma_val = sigma.detach().reshape(-1)[0].item()
        rows.append(("t_int = randint(min_step, max_step)", self._scalar_row(f"{t_int.cpu().tolist()}"), ""))
        rows.append(("t = scheduler.timesteps[t_int]", self._scalar_row(f"{t.cpu().tolist()}"), ""))
        rows.append(("sigma = sqrt(1-alpha)", self._scalar_row(f"{sigma_val:.6f}"), ""))
        shape_logs.append(f"t_int={t_int.cpu().tolist()}, t={t.cpu().tolist()}, sigma={sigma_val:.6f}")
        rows.append(("prompt", self._scalar_row(f"prompt={prompt[:80]}...", width=800), ""))
        rows.append(("negative_prompt", self._scalar_row(f"negative={negative_prompt}", width=800), ""))
        shape_logs.append(f"prompt len={len(prompt)}, negative len={len(negative_prompt)}")

        add_tensor_row("rgb", rgb)
        add_tensor_row("rgb_norm = rgb.clamp(0,1)", rgb_norm)
        add_tensor_row("video_in = (rgb_norm*2-1).to(weights_dtype)", video_in.unsqueeze(2))  # add T dim
        add_tensor_row("posterior.mean (vae.encode)", posterior.latent_dist.mean if hasattr(posterior, "latent_dist") else None)
        add_tensor_row("latents = posterior.sample()*scaling_factor", latents)

        add_tensor_row("noise = randn_like(latents)", noise)
        add_tensor_row("latents_noisy = sqrt(alpha)*latents + sigma*noise", latents_noisy)

        add_tensor_row("noise_pred (unet output)", noise_pred)
        
        # SDS weights / grads (scalar)
        w_val = float((sigma**2).detach().reshape(-1)[0].item())
        rows.append(("w = sigma**2", self._scalar_row(f"{w_val:.6f}"), ""))
        grad_sample = grad[0].detach()
        rows.append(("grad = w*(noise_pred-noise) (norm)", self._scalar_row(f"{grad_sample.norm().item():.4f}"), ""))
        
        add_tensor_row("grad = w*(noise_pred-noise)", grad)
        add_tensor_row("x0_est = (latents_noisy - sigma*noise_pred)/sqrt(alpha)", x0_est)
        # Grad diagnostics placed after grad tensor
        try:
            g = grad[0].detach().cpu().float()
            if g.ndim >= 3:
                g_map = g.abs().mean(dim=0)
                # squeeze/aggregate until 2D
                while g_map.ndim > 2:
                    g_map = g_map.mean(dim=0)
                heat = _make_heatmap_2d(g_map.numpy(), label="mean |grad| over C")
                rows.append(("grad_heatmap = mean(|grad| over C)", heat, f"shape={heat.shape}"))
            g_hist = _make_hist_image(g.abs().reshape(-1).numpy())
            rows.append(("grad_hist = hist(|grad|)", g_hist, f"shape={g_hist.shape}"))
            # line chart of grad norm history (smoothed over window)
            if not hasattr(self, "_grad_norm_hist"):
                self._grad_norm_hist = []  # list of norms at each panel save
            if not hasattr(self, "_grad_norm_window"):
                self._grad_norm_window = getattr(self.cfg, "debug_video_interval", 50)
            window = self._grad_norm_window
            g_norm = float(g.norm().item())
            self._grad_norm_hist.append(g_norm)
            # build smoothed series using rolling mean over window (e.g., last debug_video_interval panels)
            vals = np.array(self._grad_norm_hist, dtype=np.float32)
            smoothed = []
            for i in range(len(vals)):
                start = max(0, i - window + 1)
                smoothed.append(float(vals[start : i + 1].mean()))
            curve = _make_line_chart(smoothed, title=f"grad_norm MA(window={window})")
            rows.append(("grad_norm_curve (MA)", curve, f"points={len(smoothed)}"))
        except Exception as e:
            shape_logs.append(f"grad viz failed: {e}")

        # ========== DECODER OUTPUTS (in computation order) ==========
        rows.append(("--- DECODER OUTPUTS ---", self._scalar_row("VAE decodes for sanity check"), ""))
        add_rgb_image_row("latents_decoded = vae.decode(latents)", latents_decoded)
        add_rgb_image_row("latents_noisy_decoded = vae.decode(latents_noisy)", latents_noisy_decoded)
        rows.append(("--- SDS DENOISED OUTPUT ---", self._scalar_row("This shows what SD 'thinks' the image should look like"), ""))
        add_rgb_image_row("x0_est_decoded = vae.decode(x0_est)", x0_est_decoded)

        if encoder_hidden_states is not None:
            rows.append(("text_embeddings", self._scalar_row(f"shape={tuple(encoder_hidden_states.shape)}, norm={encoder_hidden_states.norm().item():.3f}"), ""))
            shape_logs.append(f"text_embeddings shape={tuple(encoder_hidden_states.shape)}, norm={encoder_hidden_states.norm().item():.3f}")

        # Add [SD-XX] prefix to each row label based on appearance order
        rows_with_prefix = []
        for idx, (label, img, shape_text) in enumerate(rows):
            prefix = f"[SD-{idx:02d}] "
            rows_with_prefix.append((prefix + label, img, shape_text))
        rows = rows_with_prefix

        # assemble panel with separate label column (like hunyuanvideo)
        panel_imgs = []
        label_w = 1500  # label column width for readability
        shape_w = 500  # extra column for shape text
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
            canvas[y_offset:y_offset+h, label_w + shape_w:, :] = img.astype(np.uint8)
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
                padded.append(np.ascontiguousarray(im))
            panel = np.concatenate(padded, axis=0)
        else:
            panel = np.ones((100, 500, 3), dtype=np.uint8) * 255
            
        save_path = os.path.join(debug_save_dir, f"it{debug_step}-panel_1_sd.png")
        imageio.imwrite(save_path, panel.astype(np.uint8))
        # cache rows for potential cross-guidance alignment panel
        self.last_panel_rows = rows
        # log shapes to stdout
        for ln in shape_logs:
            threestudio.info(f"[panel_sd] {ln}")

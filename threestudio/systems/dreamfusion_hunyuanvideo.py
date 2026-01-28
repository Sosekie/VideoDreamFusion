import math
from dataclasses import dataclass, field

import numpy as np

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import (
    binary_cross_entropy,
    dot,
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@threestudio.register("dreamfusion-hunyuanvideo-system")
class DreamFusionHunyuanVideo(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        n_arc_views: int = 8
        arc_span_deg: float = 360.0
        arc_direction: str = "cw"  # "cw" or "ccw"
        fix_elevation: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self._arc_epoch: int = -1
        self._arc_start_azimuth_deg: Optional[float] = None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def _camera_cfg(self):
        if hasattr(self, "dataset") and hasattr(self.dataset, "cfg"):
            return self.dataset.cfg
        if (
            hasattr(self, "trainer")
            and getattr(self.trainer, "datamodule", None) is not None
            and hasattr(self.trainer.datamodule, "train_dataset")
            and hasattr(self.trainer.datamodule.train_dataset, "cfg")
        ):
            return self.trainer.datamodule.train_dataset.cfg
        return None

    def _prepare_arc_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = batch["rays_o"].device
        num_views = self.cfg.n_arc_views
        arc_span_deg = self.cfg.arc_span_deg
        signed_span = -abs(arc_span_deg) if self.cfg.arc_direction.lower() == "cw" else abs(arc_span_deg)

        new_epoch = False
        if self._arc_epoch != self.true_current_epoch or self._arc_start_azimuth_deg is None:
            self._arc_epoch = self.true_current_epoch
            self._arc_start_azimuth_deg = float(torch.rand(1).item() * 360.0)
            new_epoch = True

        cam_cfg = self._camera_cfg()
        cam_dist_range = (
            cam_cfg.camera_distance_range
            if cam_cfg is not None
            else (1.5, 2.0)
        )
        fovy_range = cam_cfg.fovy_range if cam_cfg is not None else (40.0, 70.0)
        light_dist_range = (
            cam_cfg.light_distance_range if cam_cfg is not None else (0.8, 1.5)
        )
        light_strategy = (
            cam_cfg.light_sample_strategy if cam_cfg is not None else "dreamfusion"
        )
        light_pos_perturb = (
            cam_cfg.light_position_perturb if cam_cfg is not None else 1.0
        )
        rays_d_normalize = cam_cfg.rays_d_normalize if cam_cfg is not None else True
        elev_range = (
            cam_cfg.elevation_range if cam_cfg is not None else (-10.0, 45.0)
        )

        height = int(batch.get("height", getattr(self.dataset, "height", 64)))
        width = int(batch.get("width", getattr(self.dataset, "width", 64)))

        directions_unit_focal = getattr(self.dataset, "directions_unit_focal", None)
        if (
            directions_unit_focal is None
            or directions_unit_focal.shape[0] != height
            or directions_unit_focal.shape[1] != width
        ):
            directions_unit_focal = get_ray_directions(
                H=height, W=width, focal=1.0
            )
        directions_unit_focal = directions_unit_focal.to(device)

        azimuth_deg = torch.linspace(
            self._arc_start_azimuth_deg,
            self._arc_start_azimuth_deg + signed_span,
            num_views,
            device=device,
        )
        if getattr(self.cfg, "fix_elevation", False):
            elevation_deg = torch.zeros(num_views, device=device)
        else:
            elevation_deg = (
                torch.zeros(num_views, device=device).uniform_(
                    float(elev_range[0]), float(elev_range[1])
                )
            )
        elevation = elevation_deg * math.pi / 180.0
        azimuth = azimuth_deg * math.pi / 180.0

        camera_distance = (
            torch.rand(1, device=device)
            * (cam_dist_range[1] - cam_dist_range[0])
            + cam_dist_range[0]
        )
        camera_distances = camera_distance.expand(num_views)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        center = torch.zeros_like(camera_positions)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device).expand(
            num_views, -1
        )

        fovy_deg = (
            torch.rand(1, device=device) * (fovy_range[1] - fovy_range[0])
            + fovy_range[0]
        ).expand(num_views)
        fovy = fovy_deg * math.pi / 180.0

        light_distances = (
            torch.rand(1, device=device)
            * (light_dist_range[1] - light_dist_range[0])
            + light_dist_range[0]
        ).expand(num_views)

        if light_strategy == "dreamfusion":
            light_direction = F.normalize(
                camera_positions
                + torch.randn_like(camera_positions) * light_pos_perturb,
                dim=-1,
            )
            light_positions = light_direction * light_distances[:, None]
        elif light_strategy == "magic3d":
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = torch.rand(num_views, device=device) * math.pi * 2 - math.pi
            light_elevation = (
                torch.rand(num_views, device=device) * math.pi / 3 + math.pi / 6
            )
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(f"Unknown light sample strategy: {light_strategy}")

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up_vec = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up_vec, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        focal_length = 0.5 * height / torch.tan(0.5 * fovy)
        directions = directions_unit_focal[None, :, :, :].repeat(num_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=rays_d_normalize
        )
        proj_mtx = get_projection_matrix(
            fovy, float(width) / float(height), 0.01, 100.0
        ).to(device)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        if new_epoch:
            threestudio.info(
                f"Arc views epoch {self._arc_epoch}: "
                f"azimuth_deg={azimuth_deg.detach().cpu().tolist()}, "
                f"elevation_deg={elevation_deg.detach().cpu().tolist()}, "
                f"camera_distance={camera_distance.item():.3f}"
            )

        return {
            "index": torch.arange(num_views, device=device),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": height,
            "width": width,
            "fovy": fovy,
            "proj_mtx": proj_mtx,
        }

    def training_step(self, batch, batch_idx):
        multi_view_batch = self._prepare_arc_batch(batch)
        out = self(multi_view_batch)

        comp_rgb = out["comp_rgb"]
        if comp_rgb.dim() != 4 or comp_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected comp_rgb with shape [N, H, W, 3], got {comp_rgb.shape}"
            )

        # Split batch dimension (B) and temporal sweep length (T = n_arc_views)
        # so downstream video guidance can consume (B, 3, T, H, W).
        T = self.cfg.n_arc_views
        total_views, H, W, _ = comp_rgb.shape
        if total_views % T != 0:
            raise ValueError(
                f"Total rendered views {total_views} not divisible by n_arc_views {T}"
            )
        B = total_views // T
        comp_rgb_video = (
            comp_rgb.view(B, T, H, W, 3).permute(0, 4, 1, 2, 3).contiguous()
        )
        out["comp_rgb"] = comp_rgb_video

        # Debug panel: save video batch as B rows x T cols mosaic for quick inspection.
        rows = []
        for b in range(B):
            cols = []
            for t in range(T):
                frame_chw = comp_rgb_video[b, :, t, :, :].detach().cpu().numpy()
                frame_hwc = (
                    np.clip(frame_chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0
                ).astype(np.uint8)
                cols.append(frame_hwc)
            rows.append(np.concatenate(cols, axis=1))
        panel = np.concatenate(rows, axis=0)
        self.save_image(f"it{self.true_global_step}-panel.png", panel)

        # If still using SD guidance, fall back to per-frame images (N, H, W, 3).
        if self.cfg.guidance_type == "stable-diffusion-guidance":
            comp_rgb_for_guidance = (
                comp_rgb_video.permute(0, 2, 3, 4, 1).reshape(total_views, H, W, 3)
            )
        else:
            comp_rgb_for_guidance = out["comp_rgb"]  # expected to be (B, 3, T, H, W)

        prompt_utils = self.prompt_processor()
        guidance_out = self.guidance(
            comp_rgb_for_guidance,
            prompt_utils,
            **multi_view_batch,
            rgb_as_latents=False,
        )

        loss = 0.0

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z-variance loss proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

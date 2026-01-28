from dataclasses import dataclass

import torch

import threestudio
from threestudio.models.prompt_processors.base import (
    PromptProcessor,
    PromptProcessorOutput,
    DirectionConfig,
    shift_azimuth_deg,
)
from threestudio.utils.typing import *


@threestudio.register("hunyuanvideo-prompt-processor")
class HunyuanVideoPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        view_dependent_prompting: bool = False

    cfg: Config

    def configure_text_encoder(self) -> None:
        # HunyuanVideo guidance encodes prompts internally; no external text encoder needed.
        self.text_embeddings = torch.empty(0, device=self.device)
        self.uncond_text_embeddings = torch.empty(0, device=self.device)
        self.text_embeddings_vd = torch.empty(0, device=self.device)
        self.uncond_text_embeddings_vd = torch.empty(0, device=self.device)

    def destroy_text_encoder(self) -> None:
        pass

    def configure(self) -> None:
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        self.negative_prompt = self.cfg.negative_prompt

        if self.cfg.view_dependent_prompting:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
            self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}
            self.prompts_vd = [
                self.cfg.get(f"prompt_{d.name}", None) or d.prompt(self.prompt)  # type: ignore
                for d in self.directions
            ]
            self.negative_prompts_vd = [
                d.negative_prompt(self.negative_prompt) for d in self.directions
            ]
        else:
            # no view-dependent prompting
            self.directions = []
            self.direction2idx = {}
            self.prompts_vd = [self.prompt]
            self.negative_prompts_vd = [self.negative_prompt]

        # minimal placeholder embeddings
        self.configure_text_encoder()

        if self.cfg.view_dependent_prompting:
            prompts_vd_display = " ".join(
                [f"[{d.name}]:[{p}]" for d, p in zip(self.directions, self.prompts_vd)]
            )
            threestudio.info(
                f"Using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}] "
                f"with view-dependent prompts {prompts_vd_display}"
            )
        else:
            threestudio.info(
                f"Using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}] (no view-dependent prompting)"
            )

    def __call__(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            prompt=self.prompt,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            prompts_vd=self.prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=False,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )

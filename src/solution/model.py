"""Reference solution model using HuggingFace diffusers.

Uses a pre-built UNet2DModel architecture, trained from scratch (random init).
This serves as a working baseline to verify the pipeline and as a benchmark to beat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from diffusers import UNet2DModel

if TYPE_CHECKING:
    from src.config import ModelConfig, SolutionConfig


class SolutionUNet(nn.Module):
    """Thin wrapper around HuggingFace UNet2DModel for the hackathon pipeline.

    Args:
        model_config (ModelConfig): Image spec (channels, size, timesteps).
        solution_config (SolutionConfig): Architecture hyperparameters.
    """

    def __init__(self, model_config: ModelConfig, solution_config: SolutionConfig) -> None:
        super().__init__()
        channels = solution_config.block_out_channels
        n_stages = len(channels)

        if solution_config.use_attention:
            down_types = ("DownBlock2D",) + ("AttnDownBlock2D",) * (n_stages - 1)
            up_types = ("AttnUpBlock2D",) * (n_stages - 1) + ("UpBlock2D",)
        else:
            down_types = ("DownBlock2D",) * n_stages
            up_types = ("UpBlock2D",) * n_stages

        self.unet = UNet2DModel(
            sample_size=model_config.image_size,
            in_channels=model_config.image_channels,
            out_channels=model_config.image_channels,
            layers_per_block=solution_config.layers_per_block,
            block_out_channels=channels,
            norm_num_groups=solution_config.norm_num_groups,
            down_block_types=down_types,
            up_block_types=up_types,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise in the noisy image.

        Args:
            x (torch.Tensor): Noisy images, shape (B, C, H, W).
            t (torch.Tensor): Timesteps, shape (B,), integer values in [0, num_timesteps).

        Returns:
            torch.Tensor: Predicted noise, same shape as x.
        """
        return self.unet(x, t, return_dict=False)[0]

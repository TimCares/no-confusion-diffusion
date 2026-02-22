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
    from src.config import ModelConfig


class SolutionUNet(nn.Module):
    """Thin wrapper around HuggingFace UNet2DModel for the hackathon pipeline.

    Args:
        config (ModelConfig): Model configuration with image_channels, image_size, num_timesteps.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.image_channels,
            out_channels=config.image_channels,
            layers_per_block=1,
            block_out_channels=(64, 128),
            norm_num_groups=32,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
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

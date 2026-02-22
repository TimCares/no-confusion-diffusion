"""Participant's diffusion model implementation.

This is YOUR file. Implement the DiffusionModel class below.

Your model receives noisy images and timesteps, and must predict the noise that was
added during the forward diffusion process.

Useful building blocks available from src.utils:
    - SinusoidalTimestepEmbedding: maps integer timesteps to dense vectors.

Tips for getting started:
    1. Start simple: a few Conv2d layers with a time embedding projection can already work.
    2. Add skip connections (U-Net style) to improve gradient flow.
    3. Use GroupNorm + SiLU as your go-to normalization and activation.
    4. Experiment with attention layers for further quality improvements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import ModelConfig


class DiffusionModel(nn.Module):
    """Your diffusion model.

    The model takes a noisy image and a timestep, and predicts the noise.

    Available config fields:
        - config.image_channels: number of input channels (1 for grayscale, 3 for RGB).
        - config.image_size: spatial resolution (height == width).
        - config.num_timesteps: total diffusion timesteps T.

    Args:
        config (ModelConfig): Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: Build your architecture here.
        raise NotImplementedError("Implement your diffusion model architecture!")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise in the noisy image.

        Args:
            x (torch.Tensor): Noisy images, shape (B, C, H, W).
            t (torch.Tensor): Timesteps, shape (B,), integer values in [0, num_timesteps).

        Returns:
            torch.Tensor: Predicted noise, must be the same shape as x.
        """
        # TODO: Implement the forward pass.
        raise NotImplementedError("Implement the forward pass!")

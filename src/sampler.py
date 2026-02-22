"""Participant's DDPM sampler implementation (Level 4 stretch goal).

This is YOUR file. Implement the reverse diffusion process below.

The sampler receives a trained denoiser model and iteratively denoises
from pure noise x_T ~ N(0, I) back to clean images x_0.

You have access to the diffusion schedule via self.diffusion, which provides:
    - diffusion.num_timesteps: total timesteps T
    - diffusion.betas: beta schedule, shape (T,)
    - diffusion.alphas: 1 - betas, shape (T,)
    - diffusion.alpha_bar: cumulative product of alphas, shape (T,)
    - diffusion.sqrt_alpha_bar: sqrt(alpha_bar), shape (T,)
    - diffusion.sqrt_one_minus_alpha_bar: sqrt(1 - alpha_bar), shape (T,)
    - diffusion.sqrt_recip_alpha: 1 / sqrt(alpha), shape (T,)
    - diffusion.posterior_variance: posterior variance for each step, shape (T,)
    - diffusion._extract(schedule, t, x_shape): index schedule at timesteps t
      and reshape for broadcasting over images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.diffusion import GaussianDiffusion


class DDPMSampler:
    """Your DDPM reverse process sampler.

    Args:
        diffusion (GaussianDiffusion): Diffusion process with pre-computed schedule.
        device (torch.device): Device to sample on.
    """

    def __init__(self, diffusion: GaussianDiffusion, device: torch.device) -> None:
        self.diffusion = diffusion
        self.device = device

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model (nn.Module): Denoiser model.
            x_t (torch.Tensor): Current noisy images, shape (B, C, H, W).
            t (torch.Tensor): Current timestep indices, shape (B,).

        Returns:
            torch.Tensor: Denoised images x_{t-1}, shape (B, C, H, W).
        """
        # TODO: Implement the reverse step.
        raise NotImplementedError("Implement the reverse diffusion step!")

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
        """Full DDPM sampling: generate images from pure noise.

        Args:
            model (nn.Module): Trained denoiser model.
            shape (tuple[int, ...]): Shape of samples to generate, e.g. (B, C, H, W).

        Returns:
            torch.Tensor: Generated images, shape (B, C, H, W), values in [-1, 1].
        """
        # TODO: Implement the full sampling loop.
        raise NotImplementedError("Implement the sampling loop!")

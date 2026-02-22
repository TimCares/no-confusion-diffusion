"""Reference DDPM sampler (reverse process) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from src.diffusion import GaussianDiffusion


class DDPMSampler:
    """DDPM reverse process sampler.

    Iteratively denoises from x_T ~ N(0, I) back to x_0 using the trained model.

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
        d = self.diffusion
        betas_t = d._extract(d.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = d._extract(d.sqrt_one_minus_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_t = d._extract(d.sqrt_recip_alpha, t, x_t.shape)

        predicted_noise = model(x_t, t)
        model_mean = sqrt_recip_alpha_t * (
            x_t - betas_t / sqrt_one_minus_alpha_bar_t * predicted_noise
        )

        if (t == 0).all():
            return model_mean

        posterior_var_t = d._extract(d.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
        """Full DDPM sampling: generate images from pure noise.

        Args:
            model (nn.Module): Trained denoiser model.
            shape (tuple[int, ...]): Shape of samples to generate, e.g. (B, C, H, W).

        Returns:
            torch.Tensor: Generated images, shape (B, C, H, W), values in [-1, 1].
        """
        model.eval()
        x = torch.randn(shape, device=self.device)

        for i in tqdm(reversed(range(self.diffusion.num_timesteps)), total=self.diffusion.num_timesteps, desc="Sampling"):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x.clamp(-1, 1)

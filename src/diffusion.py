"""DDPM diffusion process: noise schedule, forward process, and training loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import DiffusionConfig


class GaussianDiffusion:
    """Gaussian diffusion process for DDPM.

    Implements the linear beta schedule, closed-form forward process q(x_t | x_0),
    and the training loss computation. The reverse sampling process lives in the
    sampler classes (see ``src/solution/sampler.py`` and ``src/sampler.py``).

    Args:
        config (DiffusionConfig): Diffusion hyperparameters.
        device (torch.device): Device for pre-computed schedule tensors.
    """

    def __init__(self, config: DiffusionConfig, device: torch.device) -> None:
        self.num_timesteps: int = config.num_timesteps
        self.device: torch.device = device

        # Linear beta schedule
        self.betas: torch.Tensor = torch.linspace(
            config.beta_start, config.beta_end, self.num_timesteps, device=device
        )
        self.alphas: torch.Tensor = 1.0 - self.betas
        self.alpha_bar: torch.Tensor = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar: torch.Tensor = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar: torch.Tensor = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha: torch.Tensor = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance: torch.Tensor = (
            self.betas * (1.0 - torch.cat([torch.tensor([0.0], device=device), self.alpha_bar[:-1]]))
            / (1.0 - self.alpha_bar)
        )

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Index into a schedule tensor and reshape for broadcasting.

        Args:
            schedule (torch.Tensor): 1-D schedule of shape (T,).
            t (torch.Tensor): Timestep indices of shape (B,).
            x_shape (torch.Size): Shape of the image tensor for broadcast alignment.

        Returns:
            torch.Tensor: Values at timesteps t, shape (B, 1, 1, 1).
        """
        batch_size = t.shape[0]
        values = schedule.gather(0, t)
        return values.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from q(x_t | x_0).

        Args:
            x_0 (torch.Tensor): Clean images, shape (B, C, H, W).
            t (torch.Tensor): Timestep indices, shape (B,).
            noise (torch.Tensor | None): Pre-sampled noise. Sampled if None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (noisy images x_t, noise used).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape)

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def training_loss(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        """Compute the simplified DDPM training loss (MSE on predicted noise).

        Args:
            model (nn.Module): Denoiser model mapping (x_t, t) -> predicted noise.
            x_0 (torch.Tensor): Clean images, shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = model(x_t, t)
        return nn.functional.mse_loss(predicted_noise, noise)

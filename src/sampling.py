"""Sample generation using a trained diffusion model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

from src.utils import save_image_grid

if TYPE_CHECKING:
    from src.config import Config


class Sampler(Protocol):
    """Protocol that both solution and participant samplers satisfy."""

    def sample(self, model: nn.Module, shape: tuple[int, ...]) -> torch.Tensor: ...


def generate_samples(
    model: nn.Module,
    sampler: Sampler,
    config: Config,
    device: torch.device,
    num_samples: int = 64,
    filename: str = "samples.png",
) -> torch.Tensor:
    """Generate a grid of samples and save to disk.

    Args:
        model (nn.Module): Trained denoiser model.
        sampler (Sampler): Reverse-process sampler with a ``.sample()`` method.
        config (Config): Full configuration.
        device (torch.device): Device to sample on.
        num_samples (int): Number of images to generate. Defaults to 64.
        filename (str): Output filename within the output directory. Defaults to "samples.png".

    Returns:
        torch.Tensor: Generated images, shape (num_samples, C, H, W).
    """
    shape = (
        num_samples,
        config.model.image_channels,
        config.model.image_size,
        config.model.image_size,
    )

    print(f"\nGenerating {num_samples} samples...")
    samples = sampler.sample(model, shape)

    output_path = Path(config.training.output_dir) / filename
    save_image_grid(samples, output_path, nrow=8, title=f"Generated {config.data.dataset} samples")
    print(f"Saved sample grid to {output_path}")

    return samples

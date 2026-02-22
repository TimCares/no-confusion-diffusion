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
    output_dir: Path,
    *,
    num_samples: int = 64,
    filename: str = "samples.png",
    title: str | None = None,
) -> torch.Tensor:
    """Generate a grid of samples and save to disk.

    Args:
        model (nn.Module): Trained denoiser model.
        sampler (Sampler): Reverse-process sampler with a ``.sample()`` method.
        config (Config): Full configuration (used for image shape).
        output_dir (Path): Directory to save the sample grid into.
        num_samples (int): Number of images to generate. Defaults to 64.
        filename (str): Output filename within output_dir. Defaults to "samples.png".
        title (str | None): Grid title. Defaults to dataset name.

    Returns:
        torch.Tensor: Generated images, shape (num_samples, C, H, W).
    """
    shape = (
        num_samples,
        config.model.image_channels,
        config.model.image_size,
        config.model.image_size,
    )

    samples = sampler.sample(model, shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    grid_title = title or f"Generated {config.data.dataset} samples"
    save_image_grid(samples, output_path, nrow=8, title=grid_title)
    print(f"Saved sample grid to {output_path}")

    return samples

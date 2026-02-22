"""Shared utilities: sinusoidal timestep embedding and image grid visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timesteps to dense vectors using the sinusoidal encoding
    from "Attention Is All You Need", adapted for scalar timestep inputs.

    Args:
        dim (int): Embedding dimension (should be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for timesteps.

        Args:
            t (torch.Tensor): Integer timesteps, shape (B,).

        Returns:
            torch.Tensor: Embeddings, shape (B, dim).
        """
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -freq)
        embeddings = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    title: str | None = None,
) -> None:
    """Save a batch of images as a grid to disk.

    Args:
        images (torch.Tensor): Images in [-1, 1], shape (B, C, H, W).
        path (str | Path): Output file path.
        nrow (int): Number of images per row. Defaults to 8.
        title (str | None): Optional title above the grid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = (images.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
    images = images.cpu()

    n = min(images.shape[0], nrow * nrow)
    ncols = min(n, nrow)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = axes[i][j]
            ax.axis("off")
            if idx < n:
                img = images[idx]
                if img.shape[0] == 1:
                    ax.imshow(img.squeeze(0), cmap="gray")
                else:
                    ax.imshow(img.permute(1, 2, 0))

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to a torch.device, with auto-detection.

    Args:
        device_str (str): One of "auto", "cpu", "cuda", "mps".

    Returns:
        torch.device: Resolved device.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

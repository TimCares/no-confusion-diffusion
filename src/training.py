"""Training loop for the diffusion model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from src.config import Config
    from src.diffusion import GaussianDiffusion
    from src.sampling import Sampler


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run a single training or validation epoch.

    Args:
        model (nn.Module): The denoiser model.
        loader (DataLoader): DataLoader to iterate over.
        diffusion (GaussianDiffusion): Diffusion process for loss computation.
        optimizer (torch.optim.Optimizer | None): Optimizer. None for validation.
        device (torch.device): Device to move batches to.

    Returns:
        float: Average loss over the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_batches = 0

    desc = "Train" if is_train else "Val"
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch[0].to(device)
            loss = diffusion.training_loss(model, images)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    diffusion: GaussianDiffusion,
    sampler: Sampler,
    config: Config,
    device: torch.device,
    run_dir: Path,
) -> nn.Module:
    """Full training loop with per-epoch sampling, validation scoring, and checkpoints.

    Args:
        model (nn.Module): The denoiser model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        diffusion (GaussianDiffusion): Diffusion process.
        sampler (Sampler): Reverse-process sampler for per-epoch sample generation.
        config (Config): Full configuration.
        device (torch.device): Device to train on.
        run_dir (Path): Timestamped run directory for checkpoints and samples.

    Returns:
        nn.Module: The trained model (best checkpoint loaded).
    """
    from src.sampling import generate_samples

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"

    best_val_loss = float("inf")

    print(f"\nTraining for {config.training.epochs} epochs on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Run directory: {run_dir}")
    print("-" * 65)

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _run_epoch(model, train_loader, diffusion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, diffusion, None, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint)
            marker = " *"

        print(
            f"Epoch {epoch:>3}/{config.training.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Score: {val_loss:.6f}{marker}"
        )

        generate_samples(
            model, sampler, config, run_dir,
            num_samples=16,
            filename=f"samples_{epoch}.png",
            title=f"Epoch {epoch} | Val Loss: {val_loss:.4f}",
        )

    print("-" * 65)
    print(f"Best score: {best_val_loss:.6f}")

    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded best checkpoint from {best_checkpoint}")

    return model

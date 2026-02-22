"""Entry point for training and sampling diffusion models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.config import Config, load_config
from src.dataset import create_dataloaders
from src.diffusion import GaussianDiffusion
from src.sampling import generate_samples
from src.training import train
from src.utils import resolve_device


def _build_model(config: Config, *, use_solution: bool, device: torch.device) -> torch.nn.Module:
    """Instantiate and move the model to the target device.

    Args:
        config (Config): Full configuration.
        use_solution (bool): If True, use the reference SolutionUNet.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Model ready for training.
    """
    if use_solution:
        from src.solution.model import SolutionUNet
        model = SolutionUNet(config.model, config.solution)
    else:
        from src.model import DiffusionModel
        model = DiffusionModel(config.model)

    return model.to(device)


def _build_sampler(
    diffusion: GaussianDiffusion,
    device: torch.device,
    *,
    custom_sampler: bool,
) -> object:
    """Create the reverse-process sampler.

    Uses the solution sampler by default. Pass ``custom_sampler=True`` (Level 4)
    to use the participant's implementation from ``src/sampler.py``.
    """
    if custom_sampler:
        from src.sampler import DDPMSampler
        return DDPMSampler(diffusion, device)

    from src.solution.sampler import DDPMSampler as SolutionSampler
    return SolutionSampler(diffusion, device)


def main() -> None:
    """Parse arguments, train, and generate samples."""
    parser = argparse.ArgumentParser(description="No-Confusion Diffusion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--solution",
        action="store_true",
        help="Use the reference solution model instead of the participant model.",
    )
    parser.add_argument(
        "--custom-sampler",
        action="store_true",
        help="Use the participant sampler from src/sampler.py (Level 4).",
    )

    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]
    config: Config = load_config(args.config, overrides=overrides if overrides else None)

    device: torch.device = resolve_device(config.training.device)
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config.training.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(config, use_solution=args.solution, device=device)
    diffusion = GaussianDiffusion(config.diffusion, device=device)
    sampler = _build_sampler(diffusion, device, custom_sampler=args.custom_sampler)
    train_loader, val_loader = create_dataloaders(config.data)

    model = train(model, train_loader, val_loader, diffusion, sampler, config, device, run_dir)

    print("\nGenerating final samples with best checkpoint...")
    generate_samples(model, sampler, config, run_dir, filename="samples_final.png")


if __name__ == "__main__":
    main()

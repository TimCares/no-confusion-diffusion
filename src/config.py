"""Configuration models for the diffusion training pipeline."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Dataset and data loading configuration.

    Attributes:
        dataset (str): Dataset name, one of "fashion_mnist" or "cifar10".
        batch_size (int): Training batch size.
        num_workers (int): DataLoader worker processes.
        val_split (float): Fraction of training data used for validation.
    """

    dataset: str = "fashion_mnist"
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1


class DiffusionConfig(BaseModel):
    """Diffusion process hyperparameters.

    Attributes:
        num_timesteps (int): Total number of diffusion timesteps T.
        beta_start (float): Starting value of the linear beta schedule.
        beta_end (float): Ending value of the linear beta schedule.
    """

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class ModelConfig(BaseModel):
    """Model constructor configuration.

    These values are passed directly to the DiffusionModel constructor so the
    participant knows the input specification.

    Attributes:
        image_channels (int): Number of channels (1 for grayscale, 3 for RGB).
        image_size (int): Spatial resolution (height == width).
        num_timesteps (int): Must match DiffusionConfig.num_timesteps.
    """

    image_channels: int = 1
    image_size: int = 28
    num_timesteps: int = 1000


class TrainingConfig(BaseModel):
    """Training loop configuration.

    Attributes:
        epochs (int): Number of training epochs.
        learning_rate (float): Adam optimizer learning rate.
        device (str): Device string ("auto", "cpu", "cuda", "mps").
        checkpoint_dir (str): Directory for saving model checkpoints.
        output_dir (str): Directory for generated samples and results.
    """

    epochs: int = 50
    learning_rate: float = 1e-3
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"


class Config(BaseModel):
    """Top-level configuration combining all sub-configs.

    Attributes:
        data (DataConfig): Dataset configuration.
        diffusion (DiffusionConfig): Diffusion process configuration.
        model (ModelConfig): Model constructor configuration.
        training (TrainingConfig): Training loop configuration.
    """

    data: DataConfig = DataConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(config_path: str, overrides: list[str] | None = None) -> Config:
    """Load configuration from a YAML file with optional CLI overrides.

    Uses OmegaConf for YAML loading and dotlist merging, then validates
    the final dict through Pydantic.

    Args:
        config_path (str): Path to the YAML config file.
        overrides (list[str] | None): Dotlist overrides, e.g. ["training.epochs=100"].

    Returns:
        Config: Validated configuration object.
    """
    base: DictConfig = OmegaConf.create(Config().model_dump())
    file_cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]
    merged: DictConfig = OmegaConf.merge(base, file_cfg)

    if overrides:
        cli_cfg: DictConfig = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli_cfg)

    raw: dict = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    return Config(**raw)

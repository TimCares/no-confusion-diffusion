<p align="center">
  <img src="assets/gauss.jpg" alt="Portrait of Carl Friedrich Gauss" width="200" style="border-radius: 50%;">
</p>

<h1 align="center">No-Confusion Diffusion</h1>

<p align="center">
  A simple and minimal exercise repo to get familiar with basic Diffusion Models.<br>
  See the paper <a href="https://arxiv.org/pdf/2006.11239">here</a>.
</p>

<p align="center">
  <sub>Portrait: Von Gottlieb Biermann / Nach Christian Albrecht Jensen — Gauß-Gesellschaft Göttingen e.V. (Foto: A. Wittmann). Gemeinfrei, <a href="https://commons.wikimedia.org/w/index.php?curid=57629">Wikimedia Commons</a>.</sub>
</p>


## Getting Started
This repository is build using pytorch for development and Omegaconf for configuration management.
All config files can be found in [configs](configs/).
To keep compute and complexity of the task manageable, this repo only features two simple tasks/datasets: [fashion-mnist](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html) (see [fashion_mnist.yaml](configs/fashion_mnist.yaml)) and [cifar10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) (see [cifar10.yaml](configs/cifar10.yaml)).

However, others can easily be added.

## Prerequisites

[uv](https://docs.astral.sh/uv/) is used as the project and package manager, and I highly recommend using it to create the virtual environment and run the code:

Install uv using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # macOS and Linux
```

Windows? Please don't use Windows!

## Running the code
The main entry point of this project is [main.py](main.py).

You can run the whole code as is, with the solution already provided, to check if everything runs correctly:
```bash
uv run python main.py --config configs/fashion_mnist.yaml --solution
```
Note: Having a GPU is encouraged for faster training.

For a full run that should yield good results on a GPU (e.g. T4 on Colab):
```bash
uv run python main.py --config configs/fashion_mnist.yaml --solution training.epochs=100 data.batch_size=256
```

If you have a beefy GPU (e.g. RTX 5090):
```bash
uv run python main.py --config configs/fashion_mnist.yaml --solution training.epochs=200 data.batch_size=512 training.learning_rate=2e-3
```

You may override any default argument:
```bash
uv run python main.py --config configs/fashion_mnist.yaml training.epochs=100
```

## Your Task

Implement the `DiffusionModel` class in [src/model.py](src/model.py).

Your model receives:
- `x`: noisy images, shape `(B, C, H, W)`
- `t`: integer timesteps, shape `(B,)`, values in `[0, T)`

And returns the predicted noise, same shape as `x`.

Everything else (training, diffusion process, sampling) is already provided.
The solution model and sampler live in [src/solution/](src/solution/) -- feel free to look at the sampler to understand the reverse process, but the model is a HuggingFace wrapper and won't help you much architecturally.

Once your model is implemented, run:
```bash
uv run python main.py --config configs/fashion_mnist.yaml
```

Generated samples will be saved to `outputs/samples.png` after training.

## Tips

A `SinusoidalTimestepEmbedding` module is available in [src/utils.py](src/utils.py) for mapping integer timesteps to dense vectors. Import and use it freely.

Some architecture starter tips:
- Start with a simple ConvNet + time embedding projection, see if it trains at all
- Add skip connections (UNet-style) for better gradient flow
- `GroupNorm` + `SiLU` is a solid combo for normalization and activation
- Attention layers can further improve sample quality

## Challenges

| Level | Goal |
|-------|------|
| 1 | Implement a working model that produces recognizable Fashion-MNIST samples |
| 2 | Beat the solution model's validation loss on Fashion-MNIST |
| 3 | Get recognizable results on CIFAR-10 (`--config configs/cifar10.yaml`) |
| 4 (Stretch) | Implement the DDPM sampling loop yourself in [src/sampler.py](src/sampler.py), then run with `--custom-sampler` |
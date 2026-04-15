# Project Structure

This document describes the current repository layout and the purpose of each main directory.

```text
generative_models_lightining/
|-- callbacks/
|   |-- __init__.py
|   `-- plot_generated_data.py
|-- configs/
|   `-- config.yaml
|-- data_module/
|   `-- __init__.py
|-- docs/
|   |-- algorithms/
|   |   `-- diffusion.md
|   `-- tutorials/
|       |-- pre-commit.md
|       `-- project-structure.md
|-- generative_models_lightning/
|   |-- backbones/
|   |   |-- __init__.py
|   |   |-- autoencoder.py
|   |   |-- cond_unet.py
|   |   |-- unet.py
|   |   `-- vae.py
|   |-- diffusion/
|   |   |-- __init__.py
|   |   `-- gaussian_diffusion.py
|   |-- flow/
|   |   `-- __init__.py
|   |-- gan/
|   |   `-- __init__.py
|   |-- metrics/
|   |   |-- __init__.py
|   |   `-- losses.py
|   |-- utils/
|   |   `-- __init__.py
|   |-- vae/
|   |   `-- __init__.py
|   |-- base.py
|   `-- __init__.py
|-- .gitignore
|-- .pre-commit-config.yaml
|-- LICENSE
|-- README.md
`-- train.py
```

## Directory Notes

- `callbacks/`: PyTorch Lightning callbacks such as visualization and logging helpers.
- `configs/`: Training and experiment configuration files.
- `data_module/`: Dataset-specific `LightningDataModule` and dataset implementations.
- `docs/algorithms/`: Algorithm notes and design documents.
- `docs/tutorials/`: Development and usage guides for this repository.
- `generative_models_lightning/backbones/`: Reusable neural network building blocks.
- `generative_models_lightning/diffusion/`: Diffusion model implementations.
- `generative_models_lightning/flow/`: Flow-based model implementations.
- `generative_models_lightning/gan/`: GAN model implementations.
- `generative_models_lightning/losses/`: Shared loss functions and loss helpers.
- `generative_models_lightning/utils/`: Common utility functions.
- `generative_models_lightning/vae/`: VAE model implementations.
- `generative_models_lightning/base.py`: Shared base module for generative model training logic.
- `generative_models_lightning/__init__.py`: Public package exports.
- `train.py`: Entry point for training experiments.

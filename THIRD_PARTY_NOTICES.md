# Third-Party Notices

## NVLabs EDM

The EDM preconditioning, loss, and sampler in
`generative_models_lightning/diffusion/edm.py` are adapted from:

- Project: Elucidating the Design Space of Diffusion-Based Generative Models
- Repository: https://github.com/NVlabs/edm
- Reference commit: `008a4e5316c8e3bfe61a62f874bddba254295afb`
- Copyright: NVIDIA CORPORATION & AFFILIATES
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0

The adaptation removes NVLabs-specific persistence, `dnnlib`, distributed
training, and network construction code, and exposes the algorithm through
the local Lightning and conditional-generation interfaces.

## Meta Flow Matching

The low-level probability-path, scheduler, solver, and utility code in
`generative_models_lightning/flow/` is adapted from:

- Project: Flow Matching
- Repository: https://github.com/facebookresearch/flow_matching
- Reference paper: https://arxiv.org/abs/2412.06264
- Copyright: Meta Platforms, Inc. and affiliates
- License: CC BY-NC

These adaptations keep the project self-contained by replacing external
package imports with local intra-package imports and by exposing the code
through the repository's own Lightning modules.

## MeanFlow

The MeanFlow-specific components in `generative_models_lightning/flow/meanflow/`
are adapted from:

- Project: MeanFlow
- Repository: https://github.com/haidog-yaqub/MeanFlow
- Reference paper: https://arxiv.org/abs/2505.13447
- Copyright: Jiarui Hai and contributors
- License: MIT

The adaptation decomposes the upstream implementation into a small local
sub-package and integrates it with the repository's shared flow-module
interfaces and conditioning conventions.

## PyTorch Examples DCGAN

The DCGAN architecture and weight initialization in
`generative_models_lightning/gan/dcgan.py` and
`generative_models_lightning/gan/networks.py` are adapted from:

- Project: PyTorch Examples (`dcgan`)
- Repository: https://github.com/pytorch/examples
- Reference commit: `acc295dc7b90714f1bf47f06004fc19a7fe235c4`
- Copyright: PyTorch contributors
- License: BSD 3-Clause

The adaptation ports the example into a reusable Lightning module, adds
32x32 support for CIFAR-10, and aligns generation with the repository's
shared module interfaces.

## WGAN-GP

The WGAN-GP generator/critic structure and gradient-penalty logic in
`generative_models_lightning/gan/wgan_gp.py` and
`generative_models_lightning/gan/networks.py` are adapted from:

- Project: `wgan-gp`
- Repository: https://github.com/caogang/wgan-gp
- Reference commit: `ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064`
- Copyright: Ishaan Gulrajani
- License: MIT

The adaptation removes the original standalone training loop, ports the model
to Lightning manual optimization, and exposes the gradient-penalty logic as a
local reusable helper.

## pix2pix and CycleGAN

The translation generators, PatchGAN discriminator, GAN loss helpers, and
replay-buffer logic in `generative_models_lightning/gan/` and the paired /
unpaired folder handling in `data_module/image_translation.py` are adapted
from:

- Project: `pytorch-CycleGAN-and-pix2pix`
- Repository: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- Reference commit: `2a7afba2895d52556dd5dfe07e8555ef657ced6f`
- Copyright: Jun-Yan Zhu, Taesung Park, Phillip Isola, and contributors
- License: BSD-style / BSD 3-Clause compatible notices (see upstream LICENSE)

The adaptation extracts only the core network and loss components, replaces the
original options / script framework with Hydra + Lightning modules, and
normalizes training and inference around the repository's shared batch
interfaces.

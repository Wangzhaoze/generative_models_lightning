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

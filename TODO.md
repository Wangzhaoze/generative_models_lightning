# Roadmap

## 1. Latent Diffusion

- Define an autoencoder adapter with `encode`, `decode`, latent scaling, and
  frozen/trainable checkpoint policies.
- Run Gaussian diffusion and EDM in the same latent space without coupling
  either process to a particular VAE implementation.
- Add image-space versus latent-space reconstruction checks and a conditional
  training/inference example configured through the shared `train.py`.

## 2. Flow Matching

- Adapt the minimal Meta Flow Matching path, loss, scheduler, and ODE solver
  primitives behind a Lightning module.
- Reuse the CIFAR-10 condition adapter and backbone contract so Gaussian, EDM,
  and Flow Matching comparisons do not change the data representation.
- Add Flow Matching as an experiment config for the shared `train.py`; do not
  introduce a flow-specific training entry point.
- Keep MeanFlow and domain-specific variants experimental until the base flow
  path is covered by numerical and end-to-end tests.

## Later Infrastructure

- Add package metadata, dependency extras, CI, and a finalized project license.
- Unify cross-paradigm sampling configuration only after EDM, latent
  diffusion, and flow APIs are stable.
- Add FID/KID evaluation, benchmark presets, and full convergence reports.

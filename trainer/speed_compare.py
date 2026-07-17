#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified inference-speed benchmark for DDPM, Flow Matching, and Mean Flow.

This file is intentionally self-contained so we can iterate on benchmarking
without touching the training scripts. The current design focuses on:

1. A single argparse entrypoint.
2. A small model registry so new methods can be added in one place.
3. Benchmarking either:
   - randomly initialized models ("scratch"), useful for pure runtime checks, or
   - trained Lightning checkpoints, useful for deployment-style comparisons.

Important notes:
- If you only want to compare raw inference speed, a trained checkpoint is not
  strictly necessary. Runtime is mostly determined by architecture and the
  number of sampling steps.
- If you want a fair "quality at a given speed" comparison, all methods should
  be trained and evaluated under matched settings.
- The checkpoint builders below reconstruct the default module structures from
  the existing training scripts in this repository. If a future experiment uses
  a different backbone layout, extend the corresponding builder or add a new
  one to MODEL_REGISTRY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from generative_models_lightning.backbones.cond_unet import UNetModel
from generative_models_lightning.diffusion.process import (
    DiffusionLossType,
    DiffusionMeanType,
    DiffusionVarType,
    GaussianDiffusion,
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from generative_models_lightning.flow_model.meanflow.models import MFUNet
from trainer.train_ddpm import DDPMModel
from trainer.train_flow import FlowMatchingModel
from trainer.train_mean_flow import MeanFlowModel


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[argparse.Namespace], torch.nn.Module]
    ckpt_arg: str
    default_sample_steps: int


@dataclass
class BenchmarkResult:
    model: str
    device: str
    batch_size: int
    sample_steps: int
    init_mode: str
    checkpoint: str | None
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_samples_per_sec: float
    output_shape: list[int]
    peak_memory_mb: float | None


def build_ddpm(args: argparse.Namespace) -> DDPMModel:
    betas = get_named_beta_schedule(args.ddpm_beta_schedule, args.ddpm_num_timesteps)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=DiffusionMeanType.EPSILON,
        model_var_type=DiffusionVarType.LEARNED_RANGE,
        loss_type=DiffusionLossType.RESCALED_MSE,
    )
    denoiser = UNetModel(
        image_size=(args.height, args.width),
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.in_channels * 2,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_classes=args.cond_channels,
        dropout=args.dropout,
        channel_mult=args.channel_mult,
    )
    return DDPMModel(
        diffusion_process=diffusion,
        denoiser=denoiser,
        in_channels=args.in_channels,
        image_shape=(args.height, args.width),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_flow(args: argparse.Namespace) -> FlowMatchingModel:
    unet = UNetModel(
        image_size=(args.height, args.width),
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.in_channels * 2,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_classes=args.cond_channels,
        dropout=args.dropout,
        channel_mult=args.channel_mult,
    )
    return FlowMatchingModel(
        model=unet,
        in_channels=args.in_channels,
        image_shape=(args.height, args.width),
        drop_rate=args.flow_drop_rate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.flow_warmup_steps,
        lognorm_mean=args.flow_lognorm_mean,
        lognorm_std=args.flow_lognorm_std,
    )


def build_mean_flow(args: argparse.Namespace) -> MeanFlowModel:
    model_cond_channels = args.mean_flow_reduced_cond_channels + args.mean_flow_extra_time_cond_channels
    unet = UNetModel(
        image_size=(args.height, args.width),
        in_channels=args.in_channels,
        model_channels=args.mean_flow_model_channels,
        out_channels=args.in_channels,
        num_res_blocks=args.mean_flow_num_res_blocks,
        attention_resolutions=args.mean_flow_attention_resolutions,
        num_classes=model_cond_channels,
        dropout=args.dropout,
        channel_mult=args.mean_flow_channel_mult,
        use_checkpoint=args.mean_flow_use_grad_checkpointing,
        use_fp16=args.mean_flow_use_fp16 and args.device.startswith("cuda"),
    )
    backbone = MFUNet(
        unet,
        in_channels=args.in_channels,
        cond_in_channels=args.cond_channels,
        cond_proj_channels=args.mean_flow_reduced_cond_channels,
    )
    return MeanFlowModel(
        model=backbone,
        in_channels=args.in_channels,
        image_shape=(args.height, args.width),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.mean_flow_warmup_steps,
        flow_ratio=args.mean_flow_ratio,
        time_dist=["lognorm", args.mean_flow_time_dist_mu, args.mean_flow_time_dist_sigma],
        cfg_ratio=args.mean_flow_cfg_ratio,
        cfg_scale=args.mean_flow_cfg_scale,
        cfg_uncond=args.mean_flow_cfg_uncond,
        jvp_api=args.mean_flow_jvp_api,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ddpm": ModelSpec(
        name="ddpm",
        builder=build_ddpm,
        ckpt_arg="ddpm_ckpt",
        default_sample_steps=50,
    ),
    "flow": ModelSpec(
        name="flow",
        builder=build_flow,
        ckpt_arg="flow_ckpt",
        default_sample_steps=50,
    ),
    "mean_flow": ModelSpec(
        name="mean_flow",
        builder=build_mean_flow,
        ckpt_arg="mean_flow_ckpt",
        default_sample_steps=10,
    ),
}


def resolve_model_names(mode: str) -> list[str]:
    if mode == "all":
        return list(MODEL_REGISTRY.keys())
    return [mode]


def resolve_sample_steps(args: argparse.Namespace, model_name: str) -> int:
    specific = {
        "ddpm": args.ddpm_sample_steps,
        "flow": args.flow_sample_steps,
        "mean_flow": args.mean_flow_sample_steps,
    }[model_name]
    if specific is not None:
        steps = specific
    elif args.sample_steps is not None:
        steps = args.sample_steps
    else:
        steps = MODEL_REGISTRY[model_name].default_sample_steps

    if model_name == "ddpm" and steps == 1:
        print("[warn] ddpm sample_steps=1 is not supported by the preview sampler; using 2 instead.")
        steps = 2

    return steps


def load_checkpoint_if_needed(
    module: torch.nn.Module,
    checkpoint_path: str | None,
    strict: bool,
) -> tuple[str, str | None]:
    if not checkpoint_path:
        return "scratch", None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

    incompatible = module.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f"[warn] missing keys while loading {checkpoint_path}: {missing[:8]}")
        if unexpected:
            print(f"[warn] unexpected keys while loading {checkpoint_path}: {unexpected[:8]}")
    return "checkpoint", checkpoint_path


def prepare_condition(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.cond_file is None:
        cond = torch.randn(
            args.batch_size,
            args.cond_channels,
            args.height,
            args.width,
            device=device,
            dtype=torch.float32,
        )
        return cond

    cond_path = Path(args.cond_file)
    if not cond_path.exists():
        raise FileNotFoundError(f"Condition file not found: {cond_path}")

    if cond_path.suffix == ".npy":
        cond_np = np.load(cond_path)
        cond = torch.from_numpy(cond_np)
    else:
        loaded = torch.load(cond_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            if "cond" in loaded:
                cond = loaded["cond"]
            elif "state_dict" in loaded:
                raise ValueError(
                    f"{cond_path} looks like a checkpoint, not a condition tensor. "
                    "Pass a tensor-like .pt/.pth/.npy file instead."
                )
            else:
                first_tensor = next((v for v in loaded.values() if torch.is_tensor(v)), None)
                if first_tensor is None:
                    raise ValueError(f"Could not find tensor data inside {cond_path}")
                cond = first_tensor
        else:
            cond = loaded

    if not torch.is_tensor(cond):
        cond = torch.as_tensor(cond)

    cond = cond.float()
    if cond.ndim == 3:
        cond = cond.unsqueeze(0)
    if cond.ndim != 4:
        raise ValueError(f"Condition tensor must be [B,C,H,W] or [C,H,W], got shape {tuple(cond.shape)}")

    if cond.shape[1] != args.cond_channels:
        raise ValueError(
            f"Condition channels mismatch: expected {args.cond_channels}, got {cond.shape[1]}"
        )
    if cond.shape[2] != args.height or cond.shape[3] != args.width:
        raise ValueError(
            f"Condition spatial size mismatch: expected {(args.height, args.width)}, got {(cond.shape[2], cond.shape[3])}"
        )

    if cond.shape[0] == 1 and args.batch_size > 1:
        cond = cond.repeat(args.batch_size, 1, 1, 1)
    elif cond.shape[0] != args.batch_size:
        raise ValueError(
            f"Condition batch mismatch: expected batch_size={args.batch_size}, got {cond.shape[0]}"
        )

    return cond.to(device=device, dtype=torch.float32)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_generate(
    module: torch.nn.Module,
    cond: torch.Tensor,
    batch_size: int,
    sample_steps: int,
    warmup_runs: int,
    benchmark_runs: int,
    device: torch.device,
) -> tuple[list[float], torch.Tensor, float | None]:
    output = None

    for _ in range(warmup_runs):
        output = module.generate(cond=cond, batch_size=batch_size, sample_steps=sample_steps)
        synchronize_if_needed(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for _ in range(benchmark_runs):
        synchronize_if_needed(device)
        start = time.perf_counter()
        output = module.generate(cond=cond, batch_size=batch_size, sample_steps=sample_steps)
        synchronize_if_needed(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    if output is None:
        raise RuntimeError("Benchmark did not produce an output tensor.")

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return times_ms, output, peak_memory_mb


def summarize_result(
    model_name: str,
    device: torch.device,
    batch_size: int,
    sample_steps: int,
    init_mode: str,
    checkpoint: str | None,
    times_ms: list[float],
    output: torch.Tensor,
    peak_memory_mb: float | None,
) -> BenchmarkResult:
    times = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(times.mean())
    throughput = float(batch_size / (mean_ms / 1000.0))
    return BenchmarkResult(
        model=model_name,
        device=str(device),
        batch_size=batch_size,
        sample_steps=sample_steps,
        init_mode=init_mode,
        checkpoint=checkpoint,
        mean_ms=mean_ms,
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        throughput_samples_per_sec=throughput,
        output_shape=list(output.shape),
        peak_memory_mb=peak_memory_mb,
    )


def print_result_table(results: list[BenchmarkResult]) -> None:
    if not results:
        return

    header = (
        f"{'model':<12} {'steps':>7} {'mean_ms':>12} {'std_ms':>12} "
        f"{'throughput':>14} {'peak_mem_mb':>14} {'init':>12}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        mem_text = f"{item.peak_memory_mb:.1f}" if item.peak_memory_mb is not None else "n/a"
        print(
            f"{item.model:<12} {item.sample_steps:>7d} {item.mean_ms:>12.3f} "
            f"{item.std_ms:>12.3f} {item.throughput_samples_per_sec:>14.3f} "
            f"{mem_text:>14} {item.init_mode:>12}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare inference speed across DDPM / Flow / Mean Flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["ddpm", "flow", "mean_flow", "all"],
        default="all",
        help="Which model family to benchmark.",
    )
    parser.add_argument("--device", type=str, default=_default_device(), help="Torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scratch init and random cond.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=60)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--cond-channels", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Global sample step override for all models.",
    )
    parser.add_argument("--ddpm-sample-steps", type=int, default=None)
    parser.add_argument("--flow-sample-steps", type=int, default=None)
    parser.add_argument("--mean-flow-sample-steps", type=int, default=None)
    parser.add_argument("--cond-file", type=str, default=None, help="Optional .npy/.pt condition tensor file.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON results.")
    parser.add_argument("--strict-load", action="store_true", help="Strictly enforce checkpoint key matching.")

    # Shared backbone defaults.
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=_parse_csv_ints, default=(2, 4))
    parser.add_argument("--channel-mult", type=_parse_csv_ints, default=(1, 2, 4))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # DDPM-specific settings.
    parser.add_argument("--ddpm-ckpt", type=str, default=None)
    parser.add_argument("--ddpm-num-timesteps", type=int, default=1000)
    parser.add_argument("--ddpm-beta-schedule", type=str, default="cosine")

    # Flow-specific settings.
    parser.add_argument("--flow-ckpt", type=str, default=None)
    parser.add_argument("--flow-drop-rate", type=float, default=0.1)
    parser.add_argument("--flow-warmup-steps", type=int, default=2000)
    parser.add_argument("--flow-lognorm-mean", type=float, default=0.0)
    parser.add_argument("--flow-lognorm-std", type=float, default=1.0)

    # Mean Flow-specific settings.
    parser.add_argument("--mean-flow-ckpt", type=str, default=None)
    parser.add_argument("--mean-flow-model-channels", type=int, default=64)
    parser.add_argument("--mean-flow-num-res-blocks", type=int, default=1)
    parser.add_argument("--mean-flow-attention-resolutions", type=_parse_csv_ints, default=())
    parser.add_argument("--mean-flow-channel-mult", type=_parse_csv_ints, default=(1, 2, 4))
    parser.add_argument("--mean-flow-warmup-steps", type=int, default=2000)
    parser.add_argument("--mean-flow-ratio", type=float, default=0.50)
    parser.add_argument("--mean-flow-time-dist-mu", type=float, default=-0.4)
    parser.add_argument("--mean-flow-time-dist-sigma", type=float, default=1.0)
    parser.add_argument("--mean-flow-cfg-ratio", type=float, default=0.0)
    parser.add_argument("--mean-flow-cfg-scale", type=float, default=0.0)
    parser.add_argument("--mean-flow-disable-cfg-scale", action="store_true")
    parser.add_argument("--mean-flow-cfg-uncond", type=str, default="v")
    parser.add_argument("--mean-flow-reduced-cond-channels", type=int, default=14)
    parser.add_argument("--mean-flow-extra-time-cond-channels", type=int, default=2)
    parser.add_argument("--mean-flow-jvp-api", type=str, choices=["funtorch", "autograd"], default="funtorch")
    parser.add_argument("--mean-flow-use-grad-checkpointing", action="store_true")
    parser.add_argument("--mean-flow-use-fp16", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mean_flow_disable_cfg_scale:
        args.mean_flow_cfg_scale = None
    elif args.mean_flow_cfg_scale == 0.0:
        # Keep a convenient CLI default that matches the training script.
        args.mean_flow_cfg_scale = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    cond = prepare_condition(args, device)

    results: list[BenchmarkResult] = []
    for model_name in resolve_model_names(args.mode):
        spec = MODEL_REGISTRY[model_name]
        print(f"\n[info] building model={model_name}")
        module = spec.builder(args)
        init_mode, checkpoint = load_checkpoint_if_needed(
            module=module,
            checkpoint_path=getattr(args, spec.ckpt_arg),
            strict=args.strict_load,
        )
        module.to(device)
        module.eval()

        sample_steps = resolve_sample_steps(args, model_name)
        print(
            f"[info] benchmark model={model_name} init={init_mode} "
            f"steps={sample_steps} device={device} batch={args.batch_size}"
        )

        times_ms, output, peak_memory_mb = benchmark_generate(
            module=module,
            cond=cond,
            batch_size=args.batch_size,
            sample_steps=sample_steps,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            device=device,
        )
        result = summarize_result(
            model_name=model_name,
            device=device,
            batch_size=args.batch_size,
            sample_steps=sample_steps,
            init_mode=init_mode,
            checkpoint=checkpoint,
            times_ms=times_ms,
            output=output,
            peak_memory_mb=peak_memory_mb,
        )
        results.append(result)

        print(
            f"[done] {model_name}: mean={result.mean_ms:.3f} ms, "
            f"throughput={result.throughput_samples_per_sec:.3f} samples/s, "
            f"output_shape={tuple(result.output_shape)}"
        )

        del module
        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print_result_table(results)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2),
            encoding="utf-8",
        )
        print(f"\n[info] wrote JSON results to {json_path}")


if __name__ == "__main__":
    main()

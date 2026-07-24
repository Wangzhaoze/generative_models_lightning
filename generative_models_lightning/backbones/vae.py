#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-07
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/generative_models_lightning
# @File    : generative_models_lightning/backbones/vae.py
# @IDE     : vscode

"""Locally editable KL-VAE compatible with Diffusers 0.37 ``AutoencoderKL``.

The core architecture is ported from Hugging Face Diffusers (Apache-2.0):
``Encoder``/``Decoder``, ``ResnetBlock2D``, encoder/decoder sampling blocks,
the VAE mid-block attention, diagonal Gaussian posterior, and quantization
convolutions.  Hub/model mixins, PEFT, offload, tiling, slicing, and pluggable
attention processors are intentionally excluded because they are infrastructure
rather than part of the trainable KL-VAE architecture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .autoencoder import AutoEncoderBackbone


def get_activation(name: str) -> nn.Module:
    """Subset of Diffusers activations used by AutoencoderKL blocks."""
    normalized = name.lower()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "mish":
        return nn.Mish()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation {name!r}")


class DiagonalGaussianDistribution:
    """Diffusers-compatible diagonal Gaussian latent distribution."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def kl(
        self,
        other: "DiagonalGaussianDistribution | None" = None,
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
        if other is None:
            return 0.5 * torch.sum(
                self.mean.square() + self.var - 1.0 - self.logvar,
                dim=(1, 2, 3),
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).square() / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=(1, 2, 3),
        )

    def nll(self, sample: torch.Tensor, dims=(1, 2, 3)) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0],
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=sample.device, dtype=sample.dtype)
        )
        return 0.5 * torch.sum(
            log_two_pi + self.logvar + (sample - self.mean).square() / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class Downsample2D(nn.Module):
    """Convolutional downsampling used by Diffusers DownEncoderBlock2D."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name
        if use_conv:
            self.conv = nn.Conv2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=bias,
            )
        else:
            if channels != self.out_channels:
                raise ValueError("Average-pool downsampling cannot change channels")
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(
                f"Downsample2D expected {self.channels} channels, "
                f"got {hidden_states.shape[1]}"
            )
        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1))
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    """Nearest-neighbour upsampling followed by an optional convolution."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name
        self.conv = (
            nn.Conv2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
            if use_conv
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(
                f"Upsample2D expected {self.channels} channels, "
                f"got {hidden_states.shape[1]}"
            )
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        hidden_states = F.interpolate(
            hidden_states,
            size=output_size,
            scale_factor=None if output_size is not None else 2.0,
            mode="nearest",
        )
        if self.conv is not None:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    """Core Diffusers ResnetBlock2D used by the VAE."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int | None = 512,
        groups: int = 32,
        groups_out: int | None = None,
        pre_norm: bool = True,
        eps: float = 1.0e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: bool | None = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if time_embedding_norm not in {"default", "scale_shift", "group"}:
            raise ValueError(
                "The local AutoencoderKL supports default/scale_shift/group "
                "ResNet normalization only"
            )
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        groups_out = groups if groups_out is None else groups_out
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if temb_channels is None:
            self.time_emb_proj = None
        else:
            projected_channels = (
                2 * out_channels
                if time_embedding_norm == "scale_shift"
                else out_channels
            )
            self.time_emb_proj = nn.Linear(temb_channels, projected_channels)
        self.norm2 = nn.GroupNorm(groups_out, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, 3, padding=1)
        self.nonlinearity = get_activation(non_linearity)

        if use_in_shortcut is None:
            use_in_shortcut = in_channels != conv_2d_out_channels
        self.use_in_shortcut = use_in_shortcut
        self.conv_shortcut = (
            nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                bias=conv_shortcut_bias,
            )
            if use_in_shortcut
            else None
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.conv1(self.nonlinearity(self.norm1(input_tensor)))
        if self.time_emb_proj is not None:
            if temb is None:
                raise ValueError("temb is required when temb_channels is configured")
            projected_temb = temb if self.skip_time_act else self.nonlinearity(temb)
            projected_temb = self.time_emb_proj(projected_temb)[:, :, None, None]
        else:
            projected_temb = None

        if self.time_embedding_norm == "default":
            if projected_temb is not None:
                hidden_states = hidden_states + projected_temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if projected_temb is None:
                raise ValueError("scale_shift normalization requires temb")
            time_scale, time_shift = torch.chunk(projected_temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.conv2(
            self.dropout(self.nonlinearity(hidden_states))
        )
        if self.conv_shortcut is not None:
            if self.training:
                input_tensor = input_tensor.contiguous()
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


# Backward-compatible project export; implementation is now ResnetBlock2D.
ResnetBlock = ResnetBlock2D


class AttentionBlock(nn.Module):
    """VAE mid-block attention with Diffusers parameter naming."""

    def __init__(
        self,
        channels: int,
        norm_num_groups: int,
        eps: float = 1.0e-6,
        rescale_output_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.heads = 1
        self.dim_head = channels
        self.rescale_output_factor = rescale_output_factor
        self.group_norm = nn.GroupNorm(
            norm_num_groups,
            channels,
            eps=eps,
            affine=True,
        )
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del temb
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape(batch, channels, -1).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        head_dim = channels // self.heads
        query = query.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, channels)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return (hidden_states + residual) / self.rescale_output_factor


class UNetMidBlock2D(nn.Module):
    """Diffusers VAE bottleneck: ResNet -> attention -> ResNet."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int | None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1.0e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int | None = 1,
        output_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_head_dim not in {None, in_channels}:
            raise ValueError(
                "The vendored VAE attention supports the AutoencoderKL "
                "single-head setting only"
            )
        self.add_attention = add_attention
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            ]
        )
        self.attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(
                AttentionBlock(
                    in_channels,
                    resnet_groups,
                    eps=resnet_eps,
                    rescale_output_factor=output_scale_factor,
                )
                if add_attention
                else None
            )
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                hidden_states = attention(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1.0e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=block_in,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.downsamplers = (
            nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
            if add_downsample
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1.0e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=block_in,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.upsamplers = (
            nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                    )
                ]
            )
            if add_upsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    """Diffusers AutoencoderKL encoder core."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Sequence[str] = ("DownEncoderBlock2D",),
        block_out_channels: Sequence[int] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        if len(down_block_types) != len(block_out_channels):
            raise ValueError(
                "down_block_types and block_out_channels must have equal length"
            )
        if any(name != "DownEncoderBlock2D" for name in down_block_types):
            raise ValueError("Only Diffusers DownEncoderBlock2D is vendored")
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for index, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[index]
            self.down_blocks.append(
                DownEncoderBlock2D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    add_downsample=index < len(block_out_channels) - 1,
                    resnet_eps=1.0e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                )
            )
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1.0e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )
        self.conv_norm_out = nn.GroupNorm(
            norm_num_groups,
            block_out_channels[-1],
            eps=1.0e-6,
        )
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )
        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        return self.conv_out(self.conv_act(self.conv_norm_out(sample)))


class Decoder(nn.Module):
    """Diffusers AutoencoderKL decoder core."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Sequence[str] = ("UpDecoderBlock2D",),
        block_out_channels: Sequence[int] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        if norm_type != "group":
            raise ValueError("SpatialNorm is not part of the vendored AutoencoderKL core")
        if len(up_block_types) != len(block_out_channels):
            raise ValueError(
                "up_block_types and block_out_channels must have equal length"
            )
        if any(name != "UpDecoderBlock2D" for name in up_block_types):
            raise ValueError("Only Diffusers UpDecoderBlock2D is vendored")
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], 3, padding=1)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1.0e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for index, _ in enumerate(up_block_types):
            previous_output_channel = output_channel
            output_channel = reversed_channels[index]
            self.up_blocks.append(
                UpDecoderBlock2D(
                    num_layers=layers_per_block + 1,
                    in_channels=previous_output_channel,
                    out_channels=output_channel,
                    add_upsample=index < len(block_out_channels) - 1,
                    resnet_eps=1.0e-6,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    temb_channels=None,
                    resnet_time_scale_shift=norm_type,
                )
            )
        self.conv_norm_out = nn.GroupNorm(
            norm_num_groups,
            block_out_channels[0],
            eps=1.0e-6,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, latent_embeds)
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        return self.conv_out(self.conv_act(self.conv_norm_out(sample)))


class AutoencoderKL(AutoEncoderBackbone):
    """Trainable core of Hugging Face Diffusers ``AutoencoderKL``."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Sequence[str] = ("DownEncoderBlock2D",),
        up_block_types: Sequence[str] = ("UpDecoderBlock2D",),
        block_out_channels: Sequence[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int | Sequence[int] = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=None if latents_mean is None else tuple(latents_mean),
            latents_std=None if latents_std is None else tuple(latents_std),
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=tuple(down_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        latents = self.encoder(x)
        if self.quant_conv is not None:
            latents = self.quant_conv(latents)
        return DiagonalGaussianDistribution(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        posterior = self.encode(sample)
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        return self.decode(latents)


__all__ = [
    "AttentionBlock",
    "AutoencoderKL",
    "Decoder",
    "DiagonalGaussianDistribution",
    "DownEncoderBlock2D",
    "Downsample2D",
    "Encoder",
    "ResnetBlock",
    "ResnetBlock2D",
    "UNetMidBlock2D",
    "UpDecoderBlock2D",
    "Upsample2D",
]

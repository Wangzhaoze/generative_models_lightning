"""GAN building blocks adapted from classic reference implementations."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import init


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(norm_type: str = "instance"):
    """Return a normalization layer factory."""
    if norm_type == "batch":
        return functools.partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True,
        )
    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )
    if norm_type == "none":
        return lambda _: Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_gan_weights(
    module: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> None:
    """Initialize module weights using CycleGAN/pix2pix defaults."""

    def init_func(layer: nn.Module) -> None:
        class_name = layer.__class__.__name__
        if hasattr(layer, "weight") and (
            "Conv" in class_name or "Linear" in class_name
        ):
            if init_type == "normal":
                init.normal_(layer.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(layer.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(layer.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(layer.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if getattr(layer, "bias", None) is not None:
                init.constant_(layer.bias.data, 0.0)
        elif class_name == "BatchNorm2d":
            init.normal_(layer.weight.data, 1.0, init_gain)
            init.constant_(layer.bias.data, 0.0)

    module.apply(init_func)


def init_dcgan_weights(module: nn.Module) -> None:
    """Classic DCGAN initialization from the PyTorch reference example."""

    def init_func(layer: nn.Module) -> None:
        class_name = layer.__class__.__name__
        if "Conv" in class_name:
            init.normal_(layer.weight.data, 0.0, 0.02)
        elif "BatchNorm" in class_name:
            init.normal_(layer.weight.data, 1.0, 0.02)
            init.zeros_(layer.bias.data)

    module.apply(init_func)


class GANLoss(nn.Module):
    """GAN objectives shared by pix2pix and CycleGAN."""

    def __init__(
        self,
        gan_mode: str,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def get_target_tensor(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        label = self.real_label if target_is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        if self.gan_mode in {"lsgan", "vanilla"}:
            target = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target)
        if target_is_real:
            return -prediction.mean()
        return prediction.mean()


def calculate_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    *,
    lambda_gp: float = 10.0,
    constant: float = 1.0,
    interpolation: str = "mixed",
) -> torch.Tensor:
    """Calculate the WGAN-GP gradient penalty."""
    if lambda_gp <= 0.0:
        return torch.zeros((), device=real_data.device, dtype=real_data.dtype)

    if interpolation == "real":
        interpolates = real_data
    elif interpolation == "fake":
        interpolates = fake_data
    elif interpolation == "mixed":
        alpha = torch.rand(
            real_data.shape[0],
            1,
            device=real_data.device,
            dtype=real_data.dtype,
        )
        alpha = alpha.expand(real_data.shape[0], real_data[0].numel()).contiguous()
        alpha = alpha.view_as(real_data)
        interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    else:
        raise NotImplementedError(f"{interpolation} interpolation is not implemented")

    interpolates.requires_grad_(True)
    scores = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(real_data.shape[0], -1)
    return ((gradients.norm(2, dim=1) - constant) ** 2).mean() * lambda_gp


class DCGANGenerator(nn.Module):
    """DCGAN generator adapted from the PyTorch example."""

    def __init__(
        self,
        latent_channels: int = 100,
        ngf: int = 64,
        out_channels: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size == 64:
            layers: list[nn.Module] = [
                nn.ConvTranspose2d(latent_channels, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        elif image_size == 32:
            layers = [
                nn.ConvTranspose2d(latent_channels, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        else:
            raise ValueError("DCGANGenerator supports image_size 32 or 64 only")
        self.main = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.main(latents)


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator adapted from the PyTorch example."""

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size == 64:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ]
        elif image_size == 32:
            layers = [
                nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ]
        else:
            raise ValueError("DCGANDiscriminator supports image_size 32 or 64 only")
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


class WGANGPGenerator(nn.Module):
    """CIFAR-sized generator adapted from the reference WGAN-GP implementation."""

    def __init__(
        self,
        latent_dim: int = 128,
        dim: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * dim),
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(dim, out_channels, 2, stride=2)
        self.activation = nn.Tanh()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        output = self.preprocess(latents)
        output = output.view(-1, output.shape[1] // 16, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        return self.activation(output)


class WGANGPCritic(nn.Module):
    """CIFAR-sized critic adapted from the reference WGAN-GP implementation."""

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 64,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(4 * 4 * 4 * dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.main(x)
        output = output.view(output.shape[0], -1)
        return self.linear(output).view(-1)


class ResnetGenerator(nn.Module):
    """ResNet generator adapted from the CycleGAN reference implementation."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for index in range(n_downsampling):
            mult = 2**index
            model.extend(
                [
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            )

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model.append(
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )

        for index in range(n_downsampling):
            mult = 2 ** (n_downsampling - index)
            model.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            )
        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim=dim,
            padding_type=padding_type,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_bias=use_bias,
        )

    def _build_conv_block(
        self,
        *,
        dim: int,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        padding = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        layers.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True),
            ]
        )
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        padding = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        layers.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
            ]
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class UnetGenerator(nn.Module):
    """U-Net generator adapted from the pix2pix reference implementation."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=block,
            norm_layer=norm_layer,
        )
        block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=block,
            norm_layer=norm_layer,
        )
        block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=block,
            norm_layer=norm_layer,
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        *,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            model: list[nn.Module] = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator adapted from the pix2pix/CycleGAN reference."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer=nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for layer_index in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_index, 8)
            sequence.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            )
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding),
            ]
        )
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def make_resnet_generator(
    *,
    input_nc: int,
    output_nc: int,
    ngf: int = 64,
    norm: str = "instance",
    use_dropout: bool = False,
    n_blocks: int = 9,
) -> ResnetGenerator:
    return ResnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        norm_layer=get_norm_layer(norm),
        use_dropout=use_dropout,
        n_blocks=n_blocks,
    )


def make_unet_generator(
    *,
    input_nc: int,
    output_nc: int,
    ngf: int = 64,
    norm: str = "batch",
    use_dropout: bool = False,
    netG: str = "unet_256",
) -> UnetGenerator:
    if netG == "unet_128":
        num_downs = 7
    elif netG == "unet_256":
        num_downs = 8
    else:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized")
    return UnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=num_downs,
        ngf=ngf,
        norm_layer=get_norm_layer(norm),
        use_dropout=use_dropout,
    )


def make_patch_discriminator(
    *,
    input_nc: int,
    ndf: int = 64,
    n_layers: int = 3,
    norm: str = "instance",
) -> NLayerDiscriminator:
    return NLayerDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_layer=get_norm_layer(norm),
    )

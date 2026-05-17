# Vendored from https://github.com/haidog-yaqub/MeanFlow
# Commit: 18b67b25a3af86d199005023bb190024d28233e7
# Original license: MIT. See ../LICENSE.

import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from generative_models_lightning.flow.meanflow import MeanFlow
from generative_models_lightning.flow.meanflow.models import MFDiT


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


if __name__ == "__main__":
    n_epoch = 20
    batch_size = 64
    image_size = 256
    train_path = "/home/jerry/Projects/Dataset/imagenet/train"

    os.makedirs("images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")

    transform = T.Compose(
        [
            T.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )

    trainset = ImageFolder(train_path, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(accelerator.device)
    latent_factor = 0.18215

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=384,
        depth=8,
        num_heads=6,
        num_classes=1000,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(
        channels=4,
        image_size=32,
        num_classes=1000,
        normalizer=["mean_std", 0.0, 1 / latent_factor],
        flow_ratio=0.50,
        time_dist=["lognorm", -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond="u",
    )

    model, vae, optimizer, train_dataloader = accelerator.prepare(
        model,
        vae,
        optimizer,
        train_dataloader,
    )

    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 1000
    sample_step = 1000

    for epoch in range(n_epoch):
        model.train()
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        for x, c in tqdm(train_dataloader):
            x = x.to(accelerator.device)
            c = c.to(accelerator.device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample()

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process and global_step % log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                batch_info = f"Global Step: {global_step}"
                loss_info = (
                    f"Loss: {losses / log_step:.6f}    "
                    f"MSE_Loss: {mse_losses / log_step:.6f}"
                )
                lr = optimizer.param_groups[0]["lr"]
                lr_info = f"Learning Rate: {lr:.6f}"

                log_message = f"{current_time}\n{batch_info}    {loss_info}    {lr_info}\n"

                with open("log.txt", mode="a", encoding="utf-8") as handle:
                    handle.write(log_message)

                losses = 0.0
                mse_losses = 0.0

            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, "module") else model
                    z = meanflow.sample_each_class(
                        model_module,
                        n_per_class=1,
                        classes=[157, 281, 1, 404, 805],
                    )
                    with torch.no_grad():
                        z = vae.decode(z).sample
                    z = z * 0.5 + 0.5
                    log_img = make_grid(z, nrow=10)
                    img_save_path = f"images/step_{global_step}.png"
                    save_image(log_img, img_save_path)
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, "module") else model
        ckpt_path = f"checkpoints/step_{global_step}.pt"
        accelerator.save(model_module.state_dict(), ckpt_path)

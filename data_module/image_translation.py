"""Image-to-image translation data modules for paired and unpaired folders."""

from __future__ import annotations

import random
from pathlib import Path

import lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class _FakePairedDataset(Dataset):
    def __init__(self, size: int, image_shape: tuple[int, int, int], seed: int) -> None:
        self.size = int(size)
        self.image_shape = image_shape
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        source = torch.rand(self.image_shape, generator=generator) * 2.0 - 1.0
        target = source.flip(-1)
        return {"source": source, "target": target}


class _FakeUnpairedDataset(Dataset):
    def __init__(self, size: int, image_shape: tuple[int, int, int], seed: int) -> None:
        self.size = int(size)
        self.image_shape = image_shape
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator_source = torch.Generator().manual_seed(self.seed + index)
        generator_target = torch.Generator().manual_seed(self.seed + 10_000 + index)
        source = torch.rand(self.image_shape, generator=generator_source) * 2.0 - 1.0
        target = torch.rand(self.image_shape, generator=generator_target) * 2.0 - 1.0
        return {"source": source, "target": target}


class PairedImageDataset(Dataset):
    """Read facades-style concatenated image pairs from a split directory."""

    def __init__(
        self,
        split_dir: Path,
        *,
        image_size: int,
    ) -> None:
        self.files = _list_images(split_dir)
        self.transform = _default_transform(image_size)
        if not self.files:
            raise FileNotFoundError(f"No image files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.files[index]).convert("RGB")
        width, height = image.size
        midpoint = width // 2
        source = image.crop((0, 0, midpoint, height))
        target = image.crop((midpoint, 0, width, height))
        return {
            "source": self.transform(source),
            "target": self.transform(target),
        }


class UnpairedImageDataset(Dataset):
    """Read horse2zebra-style unpaired folders (trainA/trainB/testA/testB)."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        image_size: int,
        random_target: bool,
    ) -> None:
        self.source_files = _list_images(source_dir)
        self.target_files = _list_images(target_dir)
        self.transform = _default_transform(image_size)
        self.random_target = bool(random_target)
        if not self.source_files:
            raise FileNotFoundError(f"No source images found in {source_dir}")
        if not self.target_files:
            raise FileNotFoundError(f"No target images found in {target_dir}")

    def __len__(self) -> int:
        return max(len(self.source_files), len(self.target_files))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_path = self.source_files[index % len(self.source_files)]
        if self.random_target:
            target_index = random.randint(0, len(self.target_files) - 1)
        else:
            target_index = index % len(self.target_files)
        target_path = self.target_files[target_index]
        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        return {
            "source": self.transform(source),
            "target": self.transform(target),
        }


class PairedImageFolderDataModule(pl.LightningDataModule):
    """LightningDataModule for facades-style paired translation datasets."""

    def __init__(
        self,
        data_dir: str,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        image_size: int = 256,
        pin_memory: bool = True,
        use_fake_data: bool = False,
        fake_train_size: int = 16,
        fake_val_size: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = int(image_size)
        self.pin_memory = bool(pin_memory)
        self.use_fake_data = bool(use_fake_data)
        self.fake_train_size = int(fake_train_size)
        self.fake_val_size = int(fake_val_size)
        self.seed = int(seed)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        image_shape = (3, self.image_size, self.image_size)
        if self.use_fake_data:
            self.train_dataset = _FakePairedDataset(
                self.fake_train_size,
                image_shape,
                self.seed,
            )
            self.val_dataset = _FakePairedDataset(
                self.fake_val_size,
                image_shape,
                self.seed + 1_000,
            )
            self.test_dataset = _FakePairedDataset(
                self.fake_val_size,
                image_shape,
                self.seed + 2_000,
            )
            return

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / ("val" if (self.data_dir / "val").exists() else "test")
        test_dir = self.data_dir / ("test" if (self.data_dir / "test").exists() else "val")
        self.train_dataset = PairedImageDataset(train_dir, image_size=self.image_size)
        self.val_dataset = PairedImageDataset(val_dir, image_size=self.image_size)
        self.test_dataset = PairedImageDataset(test_dir, image_size=self.image_size)

    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)


class UnpairedImageFolderDataModule(pl.LightningDataModule):
    """LightningDataModule for horse2zebra-style unpaired translation datasets."""

    def __init__(
        self,
        data_dir: str,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        image_size: int = 256,
        pin_memory: bool = True,
        use_fake_data: bool = False,
        fake_train_size: int = 16,
        fake_val_size: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = int(image_size)
        self.pin_memory = bool(pin_memory)
        self.use_fake_data = bool(use_fake_data)
        self.fake_train_size = int(fake_train_size)
        self.fake_val_size = int(fake_val_size)
        self.seed = int(seed)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        image_shape = (3, self.image_size, self.image_size)
        if self.use_fake_data:
            self.train_dataset = _FakeUnpairedDataset(
                self.fake_train_size,
                image_shape,
                self.seed,
            )
            self.val_dataset = _FakeUnpairedDataset(
                self.fake_val_size,
                image_shape,
                self.seed + 1_000,
            )
            self.test_dataset = _FakeUnpairedDataset(
                self.fake_val_size,
                image_shape,
                self.seed + 2_000,
            )
            return

        train_source = self.data_dir / "trainA"
        train_target = self.data_dir / "trainB"
        val_source = self.data_dir / ("testA" if (self.data_dir / "testA").exists() else "trainA")
        val_target = self.data_dir / ("testB" if (self.data_dir / "testB").exists() else "trainB")
        self.train_dataset = UnpairedImageDataset(
            train_source,
            train_target,
            image_size=self.image_size,
            random_target=True,
        )
        self.val_dataset = UnpairedImageDataset(
            val_source,
            val_target,
            image_size=self.image_size,
            random_target=False,
        )
        self.test_dataset = UnpairedImageDataset(
            val_source,
            val_target,
            image_size=self.image_size,
            random_target=False,
        )

    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class PreprocessedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.preprocess = v2.Compose(
            [
                v2.Grayscale(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        processed_samples = []
        for path, target in self.samples:
            sample = self.loader(path)
            processed_sample = self.preprocess(sample)
            processed_samples.append((processed_sample, target))

        self.samples = processed_samples

    def __getitem__(self, index):
        sample, target = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


augmentations = v2.Compose(
    [
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=48, scale=(0.9, 1.1), antialias=True),
        v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)


def make_dls(train_ds, valid_ds, batch_size=64, num_workers=2):
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )  # prefetch_factor=8?
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    dls = SimpleNamespace(**{"train": train_dl, "valid": valid_dl})
    return dls

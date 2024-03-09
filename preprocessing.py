import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PreprocessedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.preprocess = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
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

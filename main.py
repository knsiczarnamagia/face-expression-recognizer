# %%
# Imports and vars
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
labels = [label.name for label in list(Path("./dataset/train").iterdir())]
labels_dict = {label: i for (i, label) in enumerate(labels)}

# %%
# Create dataset


class CustomImageDataset(Dataset):
    def __init__(self, path, labels_dict):
        self.transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )
        self.data = []
        for dir_ in Path(path).iterdir():
            self.data.extend(
                [
                    (self.transform(Image.open(im_path)), labels_dict[dir_.name])
                    for im_path in tqdm(
                        dir_.iterdir(),
                        desc=f"Opening and preprocessing images in {dir_.name} dir",
                    )
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


train_ds = CustomImageDataset("./dataset/train", labels_dict)
test_ds = CustomImageDataset("./dataset/test", labels_dict)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=True)

# %%
# Create model


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(labels)),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)


# %%
# Train

model = DummyModel()
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.NLLLoss()

epochs = 100
losses = []
pbar = tqdm(range(epochs))

model.train()
for epoch in pbar:
    for i, (image, label) in enumerate(train_dl):
        image, label = image.to(device), label.to(device)
        optim.zero_grad()
        preds = model(image)
        loss = loss_fn(preds, label)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if i % 100 == 0 and i > 0:
            pbar.set_description(f"mean loss: {np.mean(losses[:-100]).round(3)}")

plt.plot(range(len(losses)), losses)

# %%
# Eval


def acc(model, dl):
    with torch.no_grad():
        model.eval()
        preds = []
        labels = []
        for image, label in tqdm(dl, total=len(dl)):
            pred = model(image.to(device))
            preds.append(pred.to("cpu"))
            labels.append(label)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        return (F.softmax(preds, dim=1).argmax(dim=1) == labels).float().mean().item()


print("test accuracy: ", acc(model, test_dl))
print("train accuracy: ", acc(model, train_dl))

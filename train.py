import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import DummyModel
from preprocessing import PreprocessedImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])

train_ds = PreprocessedImageFolder("./dataset/train", transform)
test_ds = PreprocessedImageFolder("./dataset/test", transform)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=True)

model = DummyModel(n_classes=len(train_ds.classes))
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.NLLLoss()

epochs = 10
pbar = tqdm(range(epochs))

train_avg_loss = []
test_avg_loss = []

model.train()
for epoch in pbar:
    train_losses = []
    for i, (image, label) in enumerate(train_dl):
        image, label = image.to(device), label.to(device)
        optim.zero_grad()
        preds = model(image)
        loss = loss_fn(preds, label)
        loss.backward()
        optim.step()
        train_losses.append(loss.item())

    test_losses = []
    for i, (image, label) in enumerate(test_dl):
        image, label = image.to(device), label.to(device)
        preds = model(image)
        loss = loss_fn(preds, label)
        test_losses.append(loss.item())

    train_avg_loss.append(np.mean(train_losses).round(3))
    test_avg_loss.append(np.mean(test_losses).round(3))
    pbar.set_description(
        f"train loss: {train_avg_loss[-1]} | test loss: {test_avg_loss[-1]}"
    )


plt.plot(range(len(train_avg_loss)), train_avg_loss, label="train")
plt.plot(range(len(test_avg_loss)), test_avg_loss, label="test")
plt.legend()
plt.show()


def calc_preds(model, dl):
    with torch.no_grad():
        model.eval()
        preds = []
        labels = []
        for image, label in tqdm(dl, total=len(dl)):
            pred = model(image.to(device))
            preds.append(pred.to("cpu"))
            labels.append(label)

        preds = torch.cat(preds, dim=0)
        preds = F.softmax(preds, dim=1).argmax(dim=1)
        labels = torch.cat(labels, dim=0)
        return labels, preds


print(classification_report(*calc_preds(model, test_dl)))

torch.save(model.state_dict(), "./model.pth")

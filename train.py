import torch
import torch.nn.functional as F
from torch import nn

from model import DummyModel
from preprocessing import PreprocessedImageFolder, augmentations, make_dls
from trainer import ActivationStatsCB, AugmentCB, DeviceCB, ProgressCB, Trainer, WandBCB

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = PreprocessedImageFolder("./dataset/train", None)
valid_ds = PreprocessedImageFolder("./dataset/test", None)

dls = make_dls(train_ds, valid_ds, batch_size=32, num_workers=2)

# model = ResNet18(in_channels=1, num_classes=len(train_ds.classes))
model = DummyModel(n_classes=len(train_ds.classes))


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


model.apply(init_weights)


# lr_find = LRFinderCB(min_lr=1e-4, max_lr=0.1, max_mult=3)
act_stats = ActivationStatsCB(lambda x: isinstance(x, nn.Linear))
progress = ProgressCB(in_notebook=False)
wandb_cb = WandBCB(proj_name="test", model_path="./model.pth")
augment = AugmentCB(device=device, transform=augmentations)

trainer = Trainer(
    model,
    dls,
    F.nll_loss,
    torch.optim.SGD,
    lr=5e-3,
    cbs=[DeviceCB(device), augment, progress, act_stats, wandb_cb],
)  # lr_find
trainer.fit(2, True, True)

# TODO: saving plots to wandb
progress.plot_losses(save=True)
act_stats.plot_stats(save=True)
act_stats.color_dim(save=True)
act_stats.dead_chart(save=True)

# TODO: Metrics
acc = []
with torch.inference_mode():
    for images, labels in dls.valid:
        acc.append(
            (
                F.softmax(trainer.predict(images.to(device)), dim=1).argmax(1)
                == labels.to(device)
            ).float()
        )

    print(f"Accuracy: {torch.hstack(acc).mean().item():.4f}")

# torch.save(trainer.model.state_dict(), "./model.pth") # done by WandBCB

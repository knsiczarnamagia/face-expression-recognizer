import torch
import torch.nn.functional as F
from torch import nn

from model import ResNet18
from preprocessing import PreprocessedImageFolder, augmentations, make_dls
from trainer import (
    LRFinderCB,
    ActivationStatsCB,
    AugmentCB,
    DeviceCB,
    MultiClassAccuracyCB,
    ProgressCB,
    Trainer,
    WandBCB,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = PreprocessedImageFolder("./dataset/train", None)
valid_ds = PreprocessedImageFolder("./dataset/test", None)
dls = make_dls(train_ds, valid_ds, batch_size=32, num_workers=2)

model = ResNet18(in_channels=1, n_classes=len(train_ds.classes))


# lr_find = LRFinderCB(min_lr=1e-4, max_lr=0.1, max_mult=3)
# act_stats = ActivationStatsCB(mod_filter=lambda x: isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear), with_wandb=True) # for debugging purposes
progress = ProgressCB(in_notebook=False)
wandb_cb = WandBCB(proj_name="test", model_path="./model.pth")
augment = AugmentCB(device=device, transform=augmentations)
acc_cb = MultiClassAccuracyCB(with_wandb=True)

trainer = Trainer(
    model,
    dls,
    F.cross_entropy,
    torch.optim.SGD,
    lr=1e-4,
    cbs=[DeviceCB(device), augment, progress, wandb_cb, acc_cb],
)  # act_stats, lr_find
trainer.fit(5, True, True)

# TODO: saving plots to wandb
progress.plot_losses(save=True)
# act_stats.plot_stats(save=True)
# act_stats.color_dim(save=True)
# act_stats.dead_chart(save=True)

# torch.save(trainer.model.state_dict(), "./model.pth") # done by WandBCB

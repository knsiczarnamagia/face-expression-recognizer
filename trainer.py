import math
import os
from functools import partial
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import wandb


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class Callback:
    order = 0


class with_cbs:
    """Decorator that wraps function and calls certain callbacks before/after that function."""

    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


def run_cbs(cbs, method_nm, trainer=None):
    for cb in sorted(cbs, key=attrgetter("order")):  # sort callbacks by 'order'
        method = getattr(
            cb, method_nm, None
        )  # get method from callback e.g. `before_batch`
        if method is not None:
            method(trainer)  # if callback has such method then call it


class Trainer:
    """Trainer with callbacks"""

    def __init__(
        self,
        model,
        dls=(0,),
        loss_func=F.mse_loss,
        opt_func=torch.optim.SGD,
        lr=0.1,
        cbs=[],
        n_inp=1,
    ):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.lr = lr
        self.cbs = cbs
        self.n_inp = n_inp

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for epoch in range(self.n_epochs):
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        self.n_epochs = n_epochs
        if lr is not None:
            self.lr = lr
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self._fit(train, valid)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    def predict(self, x=None):
        if x is not None:
            return self.model(x)
        self.preds = self.model(*self.batch[: self.n_inp])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, *self.batch[self.n_inp :])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

    @property
    def training(self):
        return self.model.training


class ProgressCB(Callback):
    """Adds progress bar to Trainer"""

    def __init__(self, in_notebook=False):
        super().__init__()
        self.train_loss = []
        self.valid_loss = []
        self.in_notebook = in_notebook

    def before_fit(self, trainer):
        if self.in_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.pbar = tqdm(total=trainer.n_epochs)

    def after_epoch(self, trainer):
        if trainer.training:
            self.pbar.update(1)

    def after_loss(self, trainer):
        if trainer.training:
            self.train_loss.append(trainer.loss.item())
            tmp_train_loss = (
                np.mean(self.train_loss[-10:]) if len(self.train_loss) > 10 else 0
            )
            tmp_valid_loss = (
                np.mean(self.valid_loss[-len(trainer.dls.valid) :])
                if len(self.valid_loss) > 0
                else 0
            )
            self.pbar.set_description(
                f"train loss: {tmp_train_loss:.3f} | valid loss: {tmp_valid_loss:.3f}"
            )
        else:
            self.valid_loss.append(trainer.loss.item())

    def after_fit(self, trainer):
        self.pbar.close()

    def plot_losses(self, save=True):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.train_loss)
        ax[0].set_title("train loss")
        ax[1].plot(self.valid_loss)
        ax[1].set_title("valid loss")
        if save:
            if not os.path.exists("./plots"):
                os.makedirs("./plots")
            plt.savefig("./plots/losses.png")
        else:
            plt.show()


class DeviceCB(Callback):
    """Moves model and batches to device"""

    def __init__(self, device="cpu"):
        self.device = device

    def before_fit(self, trainer):
        if hasattr(trainer.model, "to"):
            trainer.model.to(self.device)

    def before_batch(self, trainer):
        trainer.batch = tuple(t.to(self.device) for t in trainer.batch)


class Hook:
    """Registers PyTorch forward hook with provided function"""

    def __init__(self, name, mod, f):
        self.hook = mod.register_forward_hook(partial(f, self, name))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(list):
    """List of hooks"""

    def __init__(self, mods, f):
        super().__init__([Hook(n, m, f) for n, m in mods])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


class HooksCB(Callback):
    """Appends hooks with some `hookfunc` to selected layers filtered by `mod_filter`."""

    def __init__(self, hookfunc, mod_filter=lambda x: True):
        super().__init__()
        self.hookfunc = hookfunc
        self.mod_filter = mod_filter

    def before_fit(self, trainer):
        mods = [
            (name, mod)
            for name, mod in trainer.model.named_modules()
            if self.mod_filter(mod)
        ]
        self.hooks = Hooks(mods, partial(self._hookfunc, trainer.training))

    def _hookfunc(self, training, *args, **kwargs):
        if training:
            self.hookfunc(*args, **kwargs)

    def after_fit(self, trainer):
        self.hooks.remove()

    def __iter__(self):
        return iter(self.hooks)

    def __len__(self):
        return len(self.hooks)


def append_stats(hook, name, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = {"mean": [], "std": [], "abs": []}
    acts = outp.detach().cpu()
    hook.stats["mean"].append(acts.mean().item())
    hook.stats["std"].append(acts.std().item())
    hook.stats["abs"].append(acts.abs().histc(40, 0, 10).tolist())
    wandb.log(
        {
            f"{name}/mean": acts.mean().item(),
            f"{name}/std": acts.std().item(),
            f"{name}/abs": wandb.Histogram(acts.abs().histc(40, 0, 10).tolist()),
        },
        commit=False,
    )


def get_grid(n, figsize):
    return plt.subplots(round(n / 2), 2, figsize=figsize)


class WandBCB(Callback):
    """Inits and logs to W&B. Every `wandb.log()` outside this callback should have property `commit=False` because this callback gathers all logs in given batch."""

    order = math.inf  # make sure that this callback will be called last

    def __init__(self, proj_name, model_path):
        self.proj_name = proj_name
        self.model_path = model_path

    def before_fit(self, trainer):
        wandb.init(
            project=self.proj_name,
            config={"lr": trainer.lr, "n_epochs": trainer.n_epochs},
        )
        wandb.watch(trainer.model, log="all")

    def after_loss(self, trainer):
        if trainer.training:
            wandb.log({"loss/train": trainer.loss.item()}, commit=False)
        else:
            wandb.log({"loss/valid": trainer.loss.item()}, commit=False)

    def after_batch(self, trainer):
        wandb.log({}, commit=True)

    def after_fit(self, trainer):
        torch.save(trainer.model.state_dict(), self.model_path)
        wandb.save(self.model_path)
        wandb.finish()


class ActivationStatsCB(HooksCB):
    def __init__(self, mod_filter=lambda x: x):
        super().__init__(append_stats, mod_filter)

    def plot_stats(self, save=True):  # plot output means & std devs of each module
        fig, axes = get_grid(2, figsize=(20, 10))
        for h in self.hooks:
            for i, name in enumerate(["mean", "std dev"]):
                axes[i].plot(h.stats[i])
                axes[i].set_title(name)
        plt.legend(range(len(self.hooks)))
        if save:
            if not os.path.exists("./plots"):
                os.makedirs("./plots")
            plt.savefig("./plots/mean_std_stats.png")
        else:
            plt.show()

    # plot "color dim" that shows abs values of outputs through training time (should be normally distributed - uniform gradient)
    def color_dim(self, save=True):
        fig, axes = get_grid(len(self.hooks), figsize=(20, 10))
        for ax, h in zip(axes.flatten(), self.hooks):
            ax.set_ylim(0, 40)
            ax.imshow(self.get_hist(h), aspect="auto")
        if save:
            if not os.path.exists("./plots"):
                os.makedirs("./plots")
            plt.savefig("./plots/color_dim.png")
        else:
            plt.show()

    def dead_chart(self, save=True):  # plot % of dead neurons
        fig, axes = get_grid(len(self.hooks), figsize=(20, 10))
        for ax, h in zip(axes.flatten(), self.hooks):
            ax.plot(self.get_min(h))
            ax.set_ylim(0, 1)
        if save:
            if not os.path.exists("./plots"):
                os.makedirs("./plots")
            plt.savefig("./plots/dead_neurons_perc.png")
        else:
            plt.show()

    # ratio of dead neurons (activations near 0)
    def get_min(self, h):
        h1 = torch.stack(h.stats[2]).t().float()
        return h1[0] / h1.sum(0)

    def get_hist(self, h):
        return torch.stack(h.stats[2]).t().float().log1p()


class LRFinderCB(Callback):
    """Suggests an approx. good LR for a model. Usually you should choose value where loss is still decreasing (steepest slope), not the lowest value."""

    def __init__(self, min_lr=1e-6, max_lr=1, max_mult=3, num_iter=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_mult = max_mult
        self.num_iter = num_iter
        self.lr_factor = (max_lr / min_lr) ** (1 / num_iter)

    def before_fit(self, trainer):
        self.lrs, self.losses = [], []
        self.min = math.inf
        self.i = 0
        trainer.opt.param_groups[0]["lr"] = self.min_lr

    def before_batch(self, trainer):
        trainer.opt.param_groups[0]["lr"] *= self.lr_factor

    def after_batch(self, trainer):
        if not trainer.training:
            raise CancelEpochException()
        self.lrs.append(trainer.opt.param_groups[0]["lr"])
        loss = trainer.loss.to("cpu").item()
        self.losses.append(loss)
        if loss < self.min:
            self.min = loss
        self.i += 1
        if (
            math.isnan(loss)
            or (loss > self.min * self.max_mult)
            or (self.i > self.num_iter)
        ):
            raise CancelFitException()

    def plot_lrs(self, log=True):
        plt.plot(self.lrs, self.losses)
        plt.title("LR finder")
        if log:
            plt.xscale("log")
        self.best_lr = self.lrs[self.losses.index(self.min)]
        plt.plot(
            self.best_lr, self.min, "ro"
        )  # TODO: find not lowest but steepest slope value


class AugmentCB(Callback):
    """Computes augmentation transformations on device (e.g. GPU) for faster training."""

    def __init__(self, device="cpu", transform=None):
        super().__init__()
        self.device = device
        self.transform = transform

    def before_batch(self, trainer):
        trainer.batch = tuple(
            [
                *[self.transform(t) for t in trainer.batch[: trainer.n_inp]],
                *trainer.batch[trainer.n_inp :],
            ]
        )


class MultiClassAccuracyCB(Callback):
    def __init__(self):
        self.all_acc = {"train": [], "valid": []}

    def before_epoch(self, trainer):
        self.acc = []

    def after_predict(self, trainer):
        self.acc = []
        with torch.inference_mode():
            self.acc.append(
                (
                    F.softmax(trainer.preds, dim=1).argmax(1)
                    == trainer.batch[trainer.n_inp :][0]
                ).float()
            )

    def after_epoch(self, trainer):
        final_acc = torch.hstack(self.acc).mean().item()
        if trainer.training:
            wandb.log({"accuracy/train": final_acc}, commit=False)
            self.all_acc["train"].append(final_acc)
        else:
            wandb.log({"accuracy/valid": final_acc}, commit=False)
            self.all_acc["valid"].append(final_acc)
        self.acc = []

    def plot_acc(self):
        fig, axes = get_grid(2, (20, 10))
        axes[0].plot(self.all_acc["train"])
        axes[0].set_title("train acc")
        axes[1].plot(self.all_acc["valid"])
        axes[1].set_title("valid acc")

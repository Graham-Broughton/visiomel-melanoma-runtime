import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from box import Box

from .dataset import get_loaders
from .loss import get_criterion
from .metric import Accuracy, Score
from .model import get_model, get_opt
from .utils import to_gpu


def parse_args(CFG):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=None)

    args = parser.parse_args()

    if args.fold is not None:
        CFG.fold = args.fold
        if CFG.resume:
            CFG.resume = f"{CFG.resume}/{args.fold}/last.pth"

    return CFG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(CFG):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if CFG.deterministic:
        set_seed(CFG.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    CFG.distributed = False
    if "WORLD_SIZE" in os.environ:
        CFG.distributed = int(os.environ["WORLD_SIZE"]) > 1

    CFG.gpu = 0
    CFG.world_size = 1
    if CFG.distributed:
        CFG.gpu = CFG.local_rank
        torch.cuda.set_device(CFG.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        CFG.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def epoch_step(
    loader, desc, model, criterion, metrics, scaler, opt=None, batch_accum=1
):
    is_train = opt is not None
    if is_train:
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_loss = n = 0
    loc_accum = 1

    for x, y in loader:
        x = to_gpu(x, CFG.gpu)
        y = to_gpu(y, CFG.gpu)
        # x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y) / batch_accum

        if is_train:
            scaler.scale(loss).backward()

            if loc_accum == batch_accum:
                scaler.step(opt)
                scaler.update()
                for p in model.parameters():
                    p.grad = None
                # opt.zero_grad()

                loc_accum = 1
            else:
                loc_accum += 1

            logits = logits.detach()

        bs = len(x)
        loc_loss += loss.item() * bs * batch_accum
        n += bs

        for metric in metrics.values():
            metric.update(logits, y)

        torch.cuda.synchronize()

        if CFG.local_rank == 0:
            postfix = {"loss": f"{loc_loss / n:.3f}"}
            postfix.update(
                {k: f"{metric.evaluate():.3f}" for k, metric in metrics.items()}
            )
            if is_train:
                postfix.update({"lr": f'{next(iter(opt.param_groups))["lr"]:.3}'})
            pbar.set_postfix(**postfix)
            pbar.update()

    if is_train and loc_accum != batch_accum:
        scaler.step(opt)
        scaler.update()
        for p in model.parameters():
            p.grad = None
        # opt.zero_grad()

    pbar.close()

    return loc_loss / n


def plot_hist(history, path):
    history_len = len(history)
    n_rows = history_len // 2 + 1
    n_cols = 2
    plt.figure(figsize=(12, 4 * n_rows))
    for i, (m, vs) in enumerate(history.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        for k, v in vs.items():
            if "loss" in m:
                ep = np.argmin(v)
            else:
                ep = np.argmax(v)
            plt.title(f"{v[ep]:.4} on {ep}")
            plt.plot(v, label=f"{k} {v[-1]:.4}")

        plt.xlabel("#epoch")
        plt.ylabel(f"{m}")
        plt.legend()
        plt.grid(ls="--")

    plt.tight_layout()
    plt.savefig(path / "evolution.png")
    plt.close()


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size(0)
    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= CFG.world_size

    return rt


def main():
    global CFG

    CFG = parse_args(CFG)
    print(CFG)

    init_dist(CFG)

    (train_loader, train_sampler), dev_loader = get_loaders(CFG)

    model = get_model(CFG)
    # model = model.to(memory_format=torch.channels_last)
    if CFG.sync_bn:
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model.cuda()

    criterion = get_criterion(CFG).cuda()

    opt = get_opt(CFG, model, criterion)

    scaler = torch.cuda.amp.GradScaler()

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if CFG.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    best_score = 0
    metrics = {"score": Score(), "acc": Accuracy()}

    history = {k: {k_: [] for k_ in ["train", "dev"]} for k in ["loss"]}
    history.update({k: {v: [] for v in ["train", "dev"]} for k in metrics})

    work_dir = Path(CFG.work_dir) / f"{CFG.fold}"
    if CFG.local_rank == 0 and not work_dir.exists():
        work_dir.mkdir(parents=True)

    # Optionally load model from a checkpoint
    if CFG.load:

        def _load():
            path_to_load = Path(CFG.load).expanduser()
            if path_to_load.is_file():
                print(f"=> loading model '{path_to_load}'")
                checkpoint = torch.load(
                    path_to_load,
                    map_location=lambda storage, loc: storage.cuda(CFG.gpu),
                )
                model.load_state_dict(checkpoint["state_dict"])
                print(f"=> loaded model '{path_to_load}'")
            else:
                print(f"=> no model found at '{path_to_load}'")

        _load()

    scheduler = None
    if CFG.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=CFG.opt.T_max, eta_min=max(CFG.opt.lr * 1e-2, 1e-6)
        )

    # Optionally resume from a checkpoint
    if CFG.resume:
        # Use a local scope to avoid dangling references
        def _resume():
            nonlocal history, best_score
            path_to_resume = Path(CFG.resume).expanduser()
            if path_to_resume.is_file():
                print(f"=> loading resume checkpoint '{path_to_resume}'")
                checkpoint = torch.load(
                    path_to_resume,
                    map_location=lambda storage, loc: storage.cuda(CFG.gpu),
                )
                CFG.start_epoch = checkpoint["epoch"] + 1
                history = checkpoint["history"]
                best_score = max(history["score"]["dev"])
                model.load_state_dict(checkpoint["state_dict"])
                opt.load_state_dict(checkpoint["opt_state_dict"])
                scheduler.load_state_dict(checkpoint["sched_state_dict"])
                scaler.load_state_dict(checkpoint["scaler"])
                print(
                    f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
                )
            else:
                print(f"=> no checkpoint found at '{path_to_resume}'")

        _resume()

    def saver(path):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "history": history,
                "state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict(),
                "CFG": CFG,
            },
            path,
        )

    for epoch in range(CFG.start_epoch, CFG.epochs + 1):

        if CFG.dist.distributed:
            train_sampler.set_epoch(epoch)

        for metric in metrics.values():
            metric.clean()

        loss = epoch_step(
            train_loader,
            f"[ Training {epoch}/{CFG.epochs}.. ]",
            model=model,
            criterion=criterion,
            metrics=metrics,
            scaler=scaler,
            opt=opt,
            batch_accum=CFG.batch_accum,
        )
        history["loss"]["train"].append(loss)
        for k, metric in metrics.items():
            history[k]["train"].append(metric.evaluate())

        if not CFG.ft:
            with torch.no_grad():
                for metric in metrics.values():
                    metric.clean()
                loss = epoch_step(
                    dev_loader,
                    f"[ Validating {epoch}/{CFG.epochs}.. ]",
                    model=model,
                    criterion=criterion,
                    metrics=metrics,
                    scaler=scaler,
                    opt=None,
                )
                history["loss"]["dev"].append(loss)
                for k, metric in metrics.items():
                    history[k]["dev"].append(metric.evaluate())
        else:
            history["loss"]["dev"].append(loss)
            for k, metric in metrics.items():
                history[k]["dev"].append(metric.evaluate())

        if scheduler is not None:
            scheduler.step()

        if CFG.dist.local_rank == 0:
            if history["score"]["dev"][-1] > best_score:
                best_score = history["score"]["dev"][-1]
                saver(work_dir / "best.pth")

            saver(work_dir / "last.pth")
            plot_hist(history, work_dir)

    return 0


if __name__ == "__main__":
    exit(main())
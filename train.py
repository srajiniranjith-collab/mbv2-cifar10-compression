"""Baseline training of MobileNetV2 on CIFAR-10 (FP32) + the CIFAR-10 data pipeline.

Data pipeline (get_loaders / get_calib_loader) lives here so both train.py and eval.py can share it;
CIFAR-10 is downloaded into ./data/.

Train transforms : RandomCrop(32, padding=4) -> RandomHorizontalFlip -> ToTensor -> Normalize
                   (+ optional RandomErasing / Cutout-style regularisation)
Test transforms  : ToTensor -> Normalize
Normalisation    : per-channel mean/std of the CIFAR-10 training set.

python train.py --epochs 200 --seed 42
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import evaluate, set_seed, confusion_matrix
from models.mobilenetv2 import MobileNetV2

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def train_transform(erasing: float = 0.0):
    t = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)]
    if erasing > 0:
        t.append(T.RandomErasing(p=erasing, scale=(0.02, 0.2)))
    return T.Compose(t)


def test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def get_loaders(data_dir="./data", batch_size=128, workers=4, erasing=0.0):
    tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform(erasing))
    te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform())
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(tr, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(te, 512, shuffle=False, **kw))


def get_calib_loader(data_dir="./data", n_samples=1024, batch_size=128, seed=0):
    """Un-augmented training images used ONLY for activation-range calibration."""
    ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transform())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
    return DataLoader(Subset(ds, idx), batch_size, shuffle=False)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5, help="linear warm-up epochs")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--erasing", type=float, default=0.25, help="RandomErasing prob (0=off)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.workers, args.erasing)

    model = MobileNetV2(10, args.width_mult, args.dropout).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    # no weight decay on BN / bias parameters
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    steps = args.epochs * len(train_loader)
    warm = args.warmup * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    for ep in range(args.epochs):
        model.train()
        t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        te_acc, te_loss = evaluate(model, test_loader, dev)
        hist["train_loss"].append(run_loss / n); hist["train_acc"].append(100 * correct / n)
        hist["test_loss"].append(te_loss); hist["test_acc"].append(te_acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        print(f"ep {ep + 1:3d}/{args.epochs} loss {run_loss / n:.4f} train {100 * correct / n:.2f} "
              f"test {te_acc:.2f} ({time.time() - t0:.0f}s)", flush=True)

    # Report the FINAL-epoch model (no checkpoint selection on the test set).
    acc, loss, preds, tgts = evaluate(model, test_loader, dev, return_preds=True)
    cm = confusion_matrix(preds, tgts)
    per_class = {c: 100 * cm[i, i].item() / cm[i].sum().item() for i, c in enumerate(CLASSES)}
    print(f"FINAL test top-1 = {acc:.2f}%"); print(json.dumps(per_class, indent=1))

    torch.save({"model": model.state_dict(), "args": vars(args), "test_acc": acc},
               os.path.join(args.out_dir, "mbv2_cifar10.pt"))
    json.dump({"final_test_acc": acc, "history": hist, "per_class_acc": per_class,
               "confusion_matrix": cm.tolist()}, open(os.path.join(args.out_dir, "train_log.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("Top-1 accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "curves.png"), dpi=150)
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(10), CLASSES, rotation=90); plt.yticks(range(10), CLASSES)
    plt.title("Confusion matrix (test)"); plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion.png"), dpi=150)


if __name__ == "__main__":
    main()

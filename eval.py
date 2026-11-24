"""Evaluation + compression experiments for MobileNetV2 / CIFAR-10.

Modes (all share the same flags):
  python eval.py --fp32                                              # uncompressed baseline accuracy
  python eval.py --weight_quant_bits 8 --activation_quant_bits 8     # ONE compression run
  python eval.py --weight_quant_bits 4 --dw_quant_bits 6 --activation_quant_bits 6 --wandb
  python eval.py --sweep --wandb                                     # 16-config sweep -> W&B parallel coordinates
  python eval.py --selftest                                          # offline unit checks (random data, no dataset)

Reusable helpers (set_seed, evaluate, confusion_matrix, load_fp_model) are imported by train.py, so this
module must NOT import train.py at import time (data loaders are imported lazily inside functions).
"""
import argparse
import copy
import csv
import json
import os
import random

import numpy as np
import torch

from models.mobilenetv2 import MobileNetV2
from compression.quantize import (build_quantized_model, qat_finetune, finalize, export_packed,
                                  load_packed_into, fold_bn, pack_codes, unpack_codes)
from compression.memory import (size_report, activation_report, overhead_table, packed_resident_bytes,
                                vram_allocated_bytes, MB)


# --------------------------------------------------------------------------------------
# Generic evaluation utilities
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    preds_all, tgts_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += torch.nn.functional.cross_entropy(out, y, reduction="sum").item()
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total += y.numel()
        if return_preds:
            preds_all.append(p.cpu())
            tgts_all.append(y.cpu())
    acc, loss = 100.0 * correct / total, loss_sum / total
    if return_preds:
        return acc, loss, torch.cat(preds_all), torch.cat(tgts_all)
    return acc, loss


def confusion_matrix(preds, tgts, n=10):
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(tgts.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def load_fp_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    m = MobileNetV2(num_classes=10, width_mult=a["width_mult"], dropout=a["dropout"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def get_parser():
    ap = argparse.ArgumentParser()
    # mode
    ap.add_argument("--fp32", action="store_true", help="only evaluate the uncompressed FP32 model")
    ap.add_argument("--sweep", action="store_true", help="run the 16-config compression sweep")
    ap.add_argument("--selftest", action="store_true", help="offline unit checks (no dataset / checkpoint)")
    # data / checkpoint
    ap.add_argument("--ckpt", default="checkpoints/mbv2_cifar10.pt")
    ap.add_argument("--data_dir", default="./data")
    # compression settings
    ap.add_argument("--weight_quant_bits", type=int, default=8, help="bits of pointwise (1x1) convs")
    ap.add_argument("--dw_quant_bits", type=int, default=None, help="depthwise convs (default: = weight bits)")
    ap.add_argument("--first_quant_bits", type=int, default=8, help="stem conv")
    ap.add_argument("--fc_quant_bits", type=int, default=8, help="classifier")
    ap.add_argument("--activation_quant_bits", type=int, default=8, help="post-ReLU6 activations")
    ap.add_argument("--act_bits_linear", type=int, default=None,
                    help="signed tensors (projection outputs, residual sums); default = activation bits")
    ap.add_argument("--act_percentile", type=float, default=99.99, help="calibration clipping percentile (100=min/max)")
    ap.add_argument("--calib_samples", type=int, default=1024)
    ap.add_argument("--clip_grid", type=int, default=25, help="# range-shrink candidates for weight MSE search (1=off)")
    ap.add_argument("--bit_overrides", type=str, default="{}", help='JSON, e.g. \'{"features.5.conv.1": 8}\'')
    ap.add_argument("--qat_epochs", type=int, default=0)
    ap.add_argument("--qat_lr", type=float, default=1e-3)
    # misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="mbv2-cifar10-compression")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_dir", default="results")
    return ap


# --------------------------------------------------------------------------------------
# One compression experiment
# --------------------------------------------------------------------------------------
def run_experiment(args):
    from train import get_loaders, get_calib_loader  # lazy: train.py imports this module

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dw = args.dw_quant_bits or args.weight_quant_bits
    abl = args.act_bits_linear or args.activation_quant_bits
    name = args.run_name or f"w{args.weight_quant_bits}_dw{dw}_a{args.activation_quant_bits}_q{args.qat_epochs}"

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=name, config=vars(args), reinit=True)

    fp = load_fp_model(args.ckpt, dev)
    train_loader, test_loader = get_loaders(args.data_dir, 128, args.workers)
    calib = get_calib_loader(args.data_dir, args.calib_samples)
    fp_acc, _ = evaluate(fp, test_loader, dev)

    cfg = dict(pw_bits=args.weight_quant_bits, dw_bits=dw, first_bits=args.first_quant_bits,
               fc_bits=args.fc_quant_bits, act_bits=args.activation_quant_bits, act_bits_linear=abl,
               act_pct=args.act_percentile, clip_grid=args.clip_grid, overrides=json.loads(args.bit_overrides))
    qm = build_quantized_model(fp, cfg, calib, dev)
    ptq_acc, _ = evaluate(qm, test_loader, dev)
    if args.qat_epochs > 0:
        qat_finetune(qm, train_loader, dev, args.qat_epochs, args.qat_lr)
    finalize(qm)

    # Real compressed representation: bit-packed integer codes + FP16 metadata. Load it back and evaluate.
    packed = export_packed(qm)
    rt_err = load_packed_into(qm, packed, dev)
    q_acc, _ = evaluate(qm, test_loader, dev)

    sizes = size_report(fp, packed)
    resident = packed_resident_bytes(packed)
    assert resident == sizes["quant_model_bytes"], "analytical size != bytes actually resident"
    acts = activation_report(qm, dev)
    total_fp = sizes["fp32_model_bytes"] + acts["act_fp32_total_bytes"]
    total_q = sizes["quant_model_bytes"] + acts["act_quant_total_bytes"]
    res = dict(run=name, baseline_acc=fp_acc, ptq_acc_before_qat=ptq_acc, quantized_acc=q_acc,
               packed_roundtrip_max_err=rt_err, vram_allocated_bytes=vram_allocated_bytes(dev),
               **sizes, **acts,
               compression_ratio=sizes["model_ratio"],  # model_size_mb already comes from **sizes
               weight_quant_bits=args.weight_quant_bits, dw_quant_bits=dw,
               activation_quant_bits=args.activation_quant_bits,
               weights_plus_acts_ratio=total_fp / total_q)

    print(f"\n=== {name} ===")
    print(f"FP32 acc {fp_acc:.2f} | PTQ acc {ptq_acc:.2f} | final quantized acc {q_acc:.2f}  (packed round-trip err {rt_err:.2e})")
    print(f"Model: {sizes['fp32_model_mb']:.3f} MB -> {sizes['model_size_mb']:.3f} MB  (x{sizes['model_ratio']:.2f})")
    print(overhead_table(sizes) + f"\n  avg weight bits {sizes['avg_weight_bits']:.2f}")
    print(f"Weights ratio: x{sizes['weight_ratio_with_meta']:.2f} (with metadata), x{sizes['weight_ratio_codes_only']:.2f} (codes only)")
    print(f"Activations ratio: total x{acts['act_ratio_total']:.2f}, peak x{acts['act_ratio_peak']:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir, f"{name}.json"), "w"), indent=1)
    if run is not None:
        import wandb
        wandb.log({k: v for k, v in res.items() if isinstance(v, (int, float))})
        run.finish()
    return res


# --------------------------------------------------------------------------------------
# Sweep (>= 8 configs required by the assignment; 16 here)
# --------------------------------------------------------------------------------------
# (pointwise bits, depthwise bits, activation bits)
CONFIGS = [
    (8, 8, 8), (8, 8, 6), (6, 6, 8), (6, 8, 6), (4, 8, 8), (4, 6, 8), (4, 6, 6), (4, 4, 6),
    (4, 8, 4), (3, 6, 8), (3, 6, 6), (3, 8, 6), (3, 4, 6), (2, 8, 8), (2, 8, 6), (2, 6, 6),
]


def run_sweep(args):
    rows = []
    for pw, dw, a in CONFIGS:
        cur = copy.copy(args)          # sweep values override the per-run bit flags
        cur.weight_quant_bits, cur.dw_quant_bits, cur.activation_quant_bits = pw, dw, a
        cur.act_bits_linear, cur.run_name = None, None
        name = f"w{pw}_dw{pw if dw is None else dw}_a{a}_q{cur.qat_epochs}"
        done = os.path.join(args.out_dir, f"{name}.json")
        if os.path.exists(done):  # resume after a Colab disconnect
            print(f"skipping {name} (already done)")
            rows.append(json.load(open(done)))
            continue
        rows.append(run_experiment(cur))
    os.makedirs(args.out_dir, exist_ok=True)
    keys = ["run", "weight_quant_bits", "dw_quant_bits", "activation_quant_bits", "quantized_acc", "compression_ratio",
            "model_size_mb", "weight_ratio_with_meta", "act_ratio_total", "act_ratio_peak", "weights_plus_acts_ratio"]
    path = os.path.join(args.out_dir, "sweep_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------
# Offline self-test (random data): python eval.py --selftest
# --------------------------------------------------------------------------------------
def selftest():
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    for b in range(2, 9):  # 1. bit packing round trip for every width
        c = torch.randint(0, 2 ** b, (1237,), dtype=torch.uint8)
        p = pack_codes(c, b)
        assert p.numel() == (1237 * b + 7) // 8
        assert torch.equal(unpack_codes(p, b, 1237), c), b
    print("pack/unpack OK")

    m = MobileNetV2().eval()  # 2. BN folding preserves outputs
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.1)
    x = torch.randn(4, 3, 32, 32)
    ref = m(x)
    f = fold_bn(copy.deepcopy(m)).eval()
    print("fold max diff", (f(x) - ref).abs().max().item())
    assert torch.allclose(f(x), ref, atol=1e-3)

    calib = DataLoader(TensorDataset(torch.randn(64, 3, 32, 32), torch.zeros(64)), batch_size=32)
    cfg = dict(pw_bits=4, dw_bits=6, first_bits=8, fc_bits=8, act_bits=6, act_bits_linear=6,  # 3. full pipeline
               act_pct=99.99, clip_grid=10, overrides={})
    dev = torch.device("cpu")
    q = build_quantized_model(m, cfg, calib, dev)
    finalize(q)
    packed = export_packed(q)
    err = load_packed_into(q, packed, dev)
    print("packed round-trip err", err)
    assert err < 1e-4
    s, a = size_report(m, packed), activation_report(q, dev)
    assert packed_resident_bytes(packed) == s["quant_model_bytes"]
    print(overhead_table(s))
    print({k: round(v, 3) for k, v in s.items() if "ratio" in k or "mb" in k or "bits" in k})
    print({k: round(v, 3) for k, v in a.items() if "ratio" in k})
    assert s["model_ratio"] > 4
    print("ALL OK")


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.selftest:
        selftest()
    elif args.sweep:
        run_sweep(args)
    elif args.fp32:
        from train import get_loaders, CLASSES
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(args.seed)
        _, te = get_loaders(args.data_dir, workers=2)
        acc, loss, p, t = evaluate(load_fp_model(args.ckpt, dev), te, dev, return_preds=True)
        print(f"FP32 test top-1: {acc:.2f}%  loss: {loss:.4f}")
        cm = confusion_matrix(p, t)
        for i, c in enumerate(CLASSES):
            print(f"{c:12s} {100 * cm[i, i].item() / cm[i].sum().item():.1f}%")
    else:
        run_experiment(args)

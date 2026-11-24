# MobileNetV2 on CIFAR-10 + custom weight/activation compression (CS6886W Assignment 2)

**Author:** Rajalakshmi S (CS25M528) | **Repo:** https://github.com/srajiniranjith-collab/mbv2-cifar10-compression | **W&B:** https://wandb.ai/cs25m528-iit-madras/mbv2-cifar10-compression

**Report:** `report/CS25M528_Assignment2_Report.pdf`

**Headline:** FP32 baseline 95.11% top-1 (8.662 MB) -> **0.663 MB (x13.06) at 93.85%** with 2-bit pointwise / 6-bit depthwise
weights, 8-bit stem + classifier, 6-bit activations and 20 epochs of QAT.

All compression code (BN folding, quantizers, bit-packing, size accounting) is written from scratch in
`compression/`; no compression library/API is used. Sizes are **in-memory** sizes (actual tensor bytes), not file sizes.

## Repository layout
```
.
├── data/                    # CIFAR-10 is downloaded here automatically (git-ignored)
├── checkpoints/             # trained FP32 baseline: mbv2_cifar10.pt (+ curves.png, train_log.json)
├── results/                 # per-run JSON + sweep_summary.csv for every reported run
├── figures/                 # training curves, confusion matrix, W&B parallel-coordinates screenshot
├── report/                  # CS25M528_Assignment2_Report.pdf
├── models/
│   ├── __init__.py
│   └── mobilenetv2.py       # MobileNet-v2 adapted to 32x32 inputs (stem stride 1, stage-2 stride 1)
├── compression/
│   ├── __init__.py
│   ├── quantize.py          # BN folding, per-channel MSE-clipped weight quant, activation quant, QAT (STE), bit packing
│   └── memory.py            # in-memory size / RAM-VRAM accounting, overhead tracking, activation measurement
├── train.py                 # FP32 baseline training (+ CIFAR-10 data pipeline, curves, per-class acc, confusion matrix)
├── eval.py                  # evaluation, single compression run, 16-config sweep + W&B logging, offline self-test
├── requirements.txt         # pinned library versions (torch, torchvision, wandb, numpy, matplotlib)
└── README.md
```
`checkpoints/` and `results/` are committed so the reported numbers can be checked without retraining:
`python eval.py --fp32 --ckpt checkpoints/mbv2_cifar10.pt` reproduces the 95.11% baseline, and the commands below re-run any
compression config from that checkpoint. Only `data/` (CIFAR-10 download), `wandb/` and `__pycache__/` are git-ignored.

| File | Role |
|---|---|
| `models/mobilenetv2.py` | Model definition. Every conv block has an `aq` slot (Identity in FP32, replaced by `ActQuant` when compressed). |
| `compression/quantize.py` | The compression method (what is quantized and how). |
| `compression/memory.py` | The accounting (how big it is, incl. all overheads). |
| `train.py` | Training only. Also hosts the data loaders, which `eval.py` imports lazily. |
| `eval.py` | Everything after training: FP32 eval, one compression run, sweep, W&B logging, self-test. |

## Environment
```
python 3.13 (the version used for all reported results)
pip install -r requirements.txt
```
Versions used to produce the reported results (see `requirements.txt`): torch 2.14.1, torchvision 0.29.1,
numpy 2.4.6, matplotlib 3.10.0, wandb 0.28.1. Only these five packages are imported by the code. No quantization/compression
package (e.g. `torchao`, `torchtune`) is used anywhere, even if it happens to be installed in the environment.
If you need a specific CUDA build, install torch/torchvision from the matching PyTorch wheel index first, then the rest.

## Seeds
Everything uses `--seed 42` by default (`random`, `numpy`, `torch`, CUDA). The calibration subset uses a fixed generator seed 0.
GPU kernels are not bit-exact deterministic, so expect +-0.2% accuracy variation across machines.

## Reproduce
```bash
pip install -r requirements.txt
python eval.py --selftest                                    # sanity check (no dataset / checkpoint needed)

# optional: retrain the baseline (skip if using the committed checkpoint)
python train.py --epochs 200 --seed 42                       # baseline -> checkpoints/mbv2_cifar10.pt, curves.png, train_log.json
python eval.py --fp32 --ckpt checkpoints/mbv2_cifar10.pt     # uncompressed FP32 test accuracy

# single compression runs (assignment-style commands)
python eval.py --weight_quant_bits 8 --activation_quant_bits 8
python eval.py --weight_quant_bits 4 --dw_quant_bits 6 --activation_quant_bits 6
python eval.py --weight_quant_bits 3 --dw_quant_bits 6 --activation_quant_bits 6 --qat_epochs 5

# best model (x13.06, 93.85%)
python eval.py --weight_quant_bits 2 --dw_quant_bits 6 --activation_quant_bits 6 --qat_epochs 20 --wandb

# sweep with Weights & Biases logging (16 PTQ runs)
wandb login
python eval.py --sweep --wandb                                       # PTQ sweep
python eval.py --sweep --wandb --qat_epochs 5 --out_dir results_qat  # optional QAT sweep
```
The five QAT runs reported below were launched individually (3/6/6 q5, 3/4/6 q5, 3/6/4 q5, 2/8/8 q10, 2/6/6 q20) plus an extra
4/6/4 q5 run, using the single-run command above with the matching `--qat_epochs`.
Stem and classifier stay at 8 bits in every run (`--first_quant_bits`, `--fc_quant_bits` defaults).

Run names include the QAT epoch count (`..._q5`), so PTQ and QAT runs stay distinct in W&B; use a separate `--out_dir`
for QAT results. `<out_dir>/sweep_summary.csv` is overwritten by each sweep with the same `--out_dir`.

In the W&B UI: Add panel -> **Parallel coordinates** -> columns `activation_quant_bits`, `weight_quant_bits`,
`compression_ratio`, `model_size_mb`, `quantized_acc`.

## Method summary
* **BN folding** removes all BN parameters; the folded conv carries an FP16 bias.
* **Weights**: per-output-channel asymmetric uniform quantization, MSE-optimal range clipping, FP16 `lo`/`scale` per channel.
  Bits are configurable for pointwise / depthwise / stem / classifier (+ per-layer JSON overrides via `--bit_overrides`).
* **Real packing**: integer codes are bit-packed into `uint8` tensors (LSB-first, any width 2-8). Sizes come from actual
  tensor bytes (`eval.py` asserts analytical size == resident bytes); the model is reloaded from the packed buffers before
  the final accuracy measurement.
* **Activations**: per-tensor asymmetric uniform quantization after each ReLU6, projection output, residual sum and the
  pooled vector; ranges calibrated with the 99.99 percentile on 1024 un-augmented training images.
  Activation size is measured by tracing one forward pass (see `activation_report` in `compression/memory.py`).

## Storage overheads (all included in reported sizes)
| Overhead | Cost | Where tracked |
|---|---|---|
| Per-channel `lo` + `scale` (FP16) | 4 B / output channel | `metadata_bytes` |
| Folded-BN bias (FP16) | 2 B / output channel | `bias_bytes` |
| Bit-packing padding | < 1 B / layer | `packed_code_bytes - ideal_code_bytes` |
| Activation ranges `lo`,`hi` (FP32) | 8 B / quantizer | `act_metadata_bytes` |

`python eval.py ...` prints this breakdown for every run; each run's JSON in `results/` holds all raw numbers.

## Results
FP32 baseline = 8.662 MB (conv/linear weights + BN params + BN running stats; 1 MB = 2^20 B). All sizes are in-memory.

| Item | Value |
|---|---|
| FP32 baseline test top-1 | **95.11%** (200 epochs, seed 42, final epoch, no checkpoint selection on test) |
| Best model compression ratio (in-memory) | **x13.06** (8.662 MB -> 0.663 MB), 93.85% acc, `w2_dw6_a6_q20` |
| Best weight compression ratio | **x13.32** incl. metadata (x14.86 codes only), avg 2.15 bits/weight |
| Best activation compression ratio (traced forward pass) | **x8.00** at 4 bits (93.55%, `w3_dw6_a4_q5`); x5.33 at 6 bits for the best model |
| Final model size after compression | **0.663 MB (679.10 KB)** |
| W&B parallel-coordinates report | https://wandb.ai/cs25m528-iit-madras/mbv2-cifar10-compression |

### Post-training quantization sweep (pw / dw / act bits, no fine-tuning)
| pw/dw/act | Acc (%) | Size (MB) | Model x | Weight x | Act x |
|---|---|---|---|---|---|
| 8/8/8 | 95.12 | 2.198 | 3.94 | 3.88 | 4.00 |
| 8/8/6 | 94.92 | 2.198 | 3.94 | 3.88 | 5.33 |
| 6/6/8 | 95.17 | 1.676 | 5.17 | 5.11 | 4.00 |
| 6/8/6 | 94.91 | 1.692 | 5.12 | 5.06 | 5.33 |
| 4/8/8 | 94.95 | 1.185 | 7.31 | 7.29 | 4.00 |
| 4/6/8 | 94.86 | 1.170 | 7.41 | 7.39 | 4.00 |
| 4/6/6 | 94.74 | 1.170 | 7.41 | 7.39 | 5.33 |
| 4/4/6 | 94.60 | 1.154 | 7.50 | 7.49 | 5.33 |
| 4/8/4 | 90.12 | 1.185 | 7.31 | 7.29 | 8.00 |
| 3/6/8 | 92.53 | 0.916 | 9.45 | 9.51 | 4.00 |
| 3/6/6 | 92.32 | 0.916 | 9.45 | 9.51 | 5.33 |
| 3/8/6 | 92.55 | 0.932 | 9.30 | 9.34 | 5.33 |
| 3/4/6 | 91.65 | 0.901 | 9.61 | 9.67 | 5.33 |
| 2/8/8 | 25.55 | 0.678 | 12.77 | 13.01 | 4.00 |
| 2/8/6 | 25.44 | 0.678 | 12.77 | 13.01 | 5.33 |
| 2/6/6 | 25.52 | 0.663 | 13.06 | 13.32 | 5.33 |

### With QAT fine-tuning
| pw/dw/act | QAT epochs | Acc before | Acc after | Size (MB) | Model x | Weight x | Act x |
|---|---|---|---|---|---|---|---|
| 3/6/6 | 5 | 92.26 | 94.83 | 0.916 | 9.45 | 9.51 | 5.33 |
| 3/4/6 | 5 | 91.77 | 94.58 | 0.901 | 9.61 | 9.67 | 5.33 |
| 3/6/4 | 5 | 84.85 | 93.55 | 0.916 | 9.45 | 9.51 | 8.00 |
| 4/6/4 | 5 | 89.72 | 93.56 | 1.170 | 7.41 | 7.39 | 8.00 |
| 2/8/8 | 10 | 25.58 | 93.62 | 0.678 | 12.77 | 13.01 | 4.00 |
| **2/6/6** | **20** | 25.58 | **93.85** | **0.663** | **13.06** | **13.32** | 5.33 |

Takeaways: 4-bit weights are almost free under PTQ (<= 0.5 pt loss at x7.3-7.5); 3-bit loses ~2.6-2.8 pt and 2-bit collapses
without QAT; QAT recovers 2-bit to 93.85%. 4-bit activations cost ~4.8 pt under PTQ but are recoverable with QAT. Depthwise bits
barely change size, so keep them at 6. Accuracy-first choice: 3/6/6 + QAT 5 (94.83%, x9.45).

### Storage breakdown of the reported configurations
| Config (pw/dw/act), QAT | Codes (KB) | Scale+offset (KB) | Bias (KB) | Total (KB) | Size (MB) | Metadata + bias share |
|---|---|---|---|---|---|---|
| FP32 baseline | - | - | - | 8869 | 8.662 | - |
| 8/8/8, PTQ | 2150.94 | 66.66 | 33.33 | 2250.93 | 2.198 | 4.5% |
| 4/6/6, PTQ | 1097.82 | 66.66 | 33.33 | 1197.82 | 1.170 | 8.4% |
| 3/6/6, QAT 5 | 838.46 | 66.66 | 33.33 | 938.46 | 0.916 | 10.7% |
| 2/6/6, QAT 20 | 579.10 | 66.66 | 33.33 | 679.10 | 0.663 | 14.7% |

### Activation memory (one 32x32 sample, batch 1, 63 quantized tensors)
| Act bits | Total (MB) | Total x | Peak (MB) | Peak x |
|---|---|---|---|---|
| FP32 | 6.530 | 1.00 | 1.125 | 1.00 |
| 8 | 1.633 | 4.00 | 0.281 | 4.00 |
| 6 | 1.225 | 5.33 | 0.211 | 5.33 |
| 4 | 0.817 | 8.00 | 0.141 | 8.00 |

Peak = largest in+out working set of consecutive layers (ignores the residual skip tensor, so it is a lower bound). Both scale
linearly with batch size.

### Notes and limitations
* Configurations were compared on test accuracy (no separate validation split), so the best numbers are slightly optimistic.
* Single seed per configuration; differences of ~0.1-0.2 pt are within noise.
* Weights are held as bit-packed integers and decoded for evaluation (round-trip error 0 in every run); activation quantization
  is simulated (fake-quant), so activation sizes are what b-bit storage would occupy, not measured allocations.
* Ratios are against the trained FP32 model, which includes BN parameters that are folded away in the compressed model.

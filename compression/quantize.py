"""compression/quantize.py -- custom weight & activation quantization for MobileNetV2 (written from scratch,
no compression library/API). Size / RAM accounting lives in compression/memory.py.

Pipeline
--------
1. Fold BatchNorm into the preceding conv (W' = W*gamma/sigma, b' = beta - mu*gamma/sigma).
   BN disappears from memory entirely; the folded conv gets a bias.
2. Weights: per-output-channel *asymmetric uniform* quantization, w ~= lo_c + sc_c * q,  q in [0, 2^b - 1].
   The range [lo, hi] of each channel is shrunk by a grid search that minimises the MSE
   (clipping outliers matters a lot at 2-4 bits).  lo and sc are stored in FP16 (32 bits/channel metadata).
   Bit-widths are configurable per layer type (pointwise / depthwise / first conv / classifier) + overrides.
3. Integer codes are REALLY bit-packed into uint8 tensors (any width 2..8), so the in-memory size is
   measured from actual tensor bytes (numel * element_size), not estimated from file size.
4. Activations: per-tensor asymmetric uniform fake-quantization after every ReLU6 / linear-projection /
   residual-add / global pool, with ranges calibrated (percentile clipping) on un-augmented training images.
5. Optional QAT fine-tuning with a straight-through estimator (STE).
"""
import copy
import json
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobilenetv2 import ConvBNAct, InvertedResidual


# --------------------------------------------------------------------------------------
# 1. BN folding
# --------------------------------------------------------------------------------------
@torch.no_grad()
def fold_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, ConvBNAct) and not m.folded:
            bn = m.bn
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            m.conv.weight.mul_(scale.view(-1, 1, 1, 1))
            m.conv.bias = nn.Parameter(bn.bias - bn.running_mean * scale)
            m.bn = nn.Identity()
            m.folded = True
    return model


# --------------------------------------------------------------------------------------
# 2. Weight quantization
# --------------------------------------------------------------------------------------
def _round_ste(x):
    return (x.round() - x).detach() + x


@torch.no_grad()
def weight_qparams(w: torch.Tensor, bits: int, n_grid: int = 25):
    """Per-output-channel (lo, scale) minimising MSE over a grid of range-shrink factors."""
    C = w.shape[0]
    flat = w.reshape(C, -1).float()
    wmin, wmax = flat.min(1).values, flat.max(1).values
    levels = 2 ** bits - 1
    best_err = torch.full((C,), float("inf"), device=w.device)
    best_lo, best_sc = wmin.clone(), ((wmax - wmin) / levels).clamp_min(1e-6)
    ratios = torch.linspace(1.0, 0.3, n_grid).tolist() if n_grid > 1 else [1.0]
    for r in ratios:
        lo, hi = wmin * r, wmax * r
        sc = ((hi - lo) / levels).clamp_min(1e-6)
        q = ((flat - lo[:, None]) / sc[:, None]).round().clamp(0, levels)
        err = ((q * sc[:, None] + lo[:, None] - flat) ** 2).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_lo = torch.where(better, lo, best_lo)
        best_sc = torch.where(better, sc, best_sc)
    # metadata is stored in FP16, so round now to keep simulation == deployed model
    return best_lo.half().float(), best_sc.half().float().clamp_min(1e-6)


class QLayer(nn.Module):
    """Wraps a Conv2d/Linear; forward uses fake-quantized weights (STE for QAT)."""

    def __init__(self, layer: nn.Module, bits: int, n_grid: int = 25):
        super().__init__()
        self.layer, self.bits, self.n_grid = layer, bits, n_grid
        C = layer.weight.shape[0]
        self.register_buffer("lo", torch.zeros(C, device=layer.weight.device))
        self.register_buffer("sc", torch.ones(C, device=layer.weight.device))
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        lo, sc = weight_qparams(self.layer.weight, self.bits, self.n_grid)
        self.lo.copy_(lo)
        self.sc.copy_(sc)

    def _shape(self):
        return (-1,) + (1,) * (self.layer.weight.dim() - 1)

    def qweight(self):
        w, s = self.layer.weight, self._shape()
        lo, sc = self.lo.view(s), self.sc.view(s)
        q = _round_ste(((w - lo) / sc).clamp(0, 2 ** self.bits - 1))
        return q * sc + lo

    @torch.no_grad()
    def codes(self):
        s = self._shape()
        q = ((self.layer.weight - self.lo.view(s)) / self.sc.view(s)).round().clamp(0, 2 ** self.bits - 1)
        return q.to(torch.uint8)

    def forward(self, x):
        w, l = self.qweight(), self.layer
        if isinstance(l, nn.Conv2d):
            return F.conv2d(x, w, l.bias, l.stride, l.padding, l.dilation, l.groups)
        return F.linear(x, w, l.bias)


# --------------------------------------------------------------------------------------
# 3. Activation quantization
# --------------------------------------------------------------------------------------
def _range_stat(x: torch.Tensor, pct: float):
    flat = x.detach().flatten().float()
    if flat.numel() > 2_000_000:
        flat = flat[torch.randint(0, flat.numel(), (2_000_000,), device=flat.device)]
    if pct >= 100:
        return flat.min(), flat.max()
    q = torch.tensor([(100 - pct) / 100, pct / 100], device=flat.device)
    r = torch.quantile(flat, q)
    return r[0], r[1]


class ActQuant(nn.Module):
    """Per-tensor asymmetric uniform activation quantizer. modes: off | calib | quant."""
    trace = None  # when a list, forward() appends (name, per-sample numel, bits)

    def __init__(self, name: str, bits: int, pct: float = 99.99):
        super().__init__()
        self.name, self.bits, self.pct, self.mode = name, bits, pct, "off"
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(6.0))
        self._los: List[torch.Tensor] = []
        self._his: List[torch.Tensor] = []

    def forward(self, x):
        if ActQuant.trace is not None:
            ActQuant.trace.append((self.name, x[0].numel(), self.bits))
        if self.mode == "calib":
            lo, hi = _range_stat(x, self.pct)
            self._los.append(lo)
            self._his.append(hi)
            return x
        if self.mode == "quant":
            sc = (self.hi - self.lo) / (2 ** self.bits - 1)
            q = _round_ste((x.clamp(self.lo, self.hi) - self.lo) / sc)
            return q * sc + self.lo
        return x

    @torch.no_grad()
    def finalize_calib(self):
        lo = torch.stack(self._los).mean().clamp(max=0.0)
        hi = torch.stack(self._his).mean()
        hi = torch.maximum(hi, lo + 1e-4)
        self.lo.copy_(lo)
        self.hi.copy_(hi)
        self._los, self._his, self.mode = [], [], "quant"


# --------------------------------------------------------------------------------------
# 4. Build the quantized model
# --------------------------------------------------------------------------------------
def build_quantized_model(fp_model, cfg: Dict, calib_loader, device):
    """cfg keys: pw_bits, dw_bits, first_bits, fc_bits, act_bits, act_bits_linear, act_pct,
    clip_grid, overrides (dict layer-name -> bits)."""
    m = copy.deepcopy(fp_model).to(device).eval()
    fold_bn(m)
    ov = cfg.get("overrides", {}) or {}
    for name, mod in list(m.named_modules()):
        if isinstance(mod, ConvBNAct):
            if name in ov:
                bits = ov[name]
            elif name == "features.0":
                bits = cfg["first_bits"]
            elif mod.conv.groups > 1:
                bits = cfg["dw_bits"]
            else:
                bits = cfg["pw_bits"]
            mod.conv = QLayer(mod.conv, bits, cfg["clip_grid"])
            ab = cfg["act_bits"] if mod.has_act else cfg["act_bits_linear"]
            mod.aq = ActQuant(name, ab, cfg["act_pct"])
        elif isinstance(mod, InvertedResidual) and mod.use_res:
            mod.out_aq = ActQuant(name + ".res_out", cfg["act_bits_linear"], cfg["act_pct"])
    m.fc = QLayer(m.fc, ov.get("fc", cfg["fc_bits"]), cfg["clip_grid"])
    m.pool_aq = ActQuant("pool", cfg["act_bits"], cfg["act_pct"])

    # calibrate activation ranges with the weights already quantized
    aqs = [a for a in m.modules() if isinstance(a, ActQuant)]
    for a in aqs:
        a.mode = "calib"
    with torch.no_grad():
        for x, _ in calib_loader:
            m(x.to(device))
    for a in aqs:
        a.finalize_calib()
    return m.to(device)  # quantizers were created after the model was moved: put their buffers on `device`


def qat_finetune(qm, train_loader, device, epochs, lr=1e-3, log=print):
    """Short STE fine-tuning. BN is already folded, so only dropout is active in train mode."""
    qlayers = [q for q in qm.modules() if isinstance(q, QLayer)]
    params = [p for p in qm.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)
    total = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    for ep in range(epochs):
        for q in qlayers:
            q.refresh()  # re-fit ranges to the current float weights
        qm.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(qm(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        log(f"  QAT epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    qm.eval()


@torch.no_grad()
def finalize(qm):
    """Freeze: refit ranges, make biases FP16-representable (that is how they are stored)."""
    for q in qm.modules():
        if isinstance(q, QLayer):
            q.refresh()
            if q.layer.bias is not None:
                q.layer.bias.data = q.layer.bias.data.half().float()


# --------------------------------------------------------------------------------------
# 5. Real bit-packing + packed export
# --------------------------------------------------------------------------------------
def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes (each < 2^bits) LSB-first into a dense uint8 tensor of ceil(n*bits/8) bytes."""
    c = codes.flatten().to(torch.int32)
    shifts = torch.arange(bits, device=c.device, dtype=torch.int32)
    b = ((c.unsqueeze(1) >> shifts) & 1).flatten()
    pad = (-b.numel()) % 8
    if pad:
        b = torch.cat([b, b.new_zeros(pad)])
    w = (1 << torch.arange(8, device=c.device, dtype=torch.int32))
    return (b.view(-1, 8) * w).sum(1).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    p = packed.to(torch.int32).unsqueeze(1)
    bm = ((p >> torch.arange(8, device=p.device, dtype=torch.int32)) & 1).flatten()[: n * bits].view(n, bits)
    return (bm * (1 << torch.arange(bits, device=p.device, dtype=torch.int32))).sum(1).to(torch.uint8)


@torch.no_grad()
def export_packed(qm) -> Dict[str, Dict]:
    out = {}
    for name, q in qm.named_modules():
        if isinstance(q, QLayer):
            codes = q.codes()
            b = q.layer.bias
            out[name] = dict(bits=q.bits, shape=tuple(codes.shape),
                             packed=pack_codes(codes, q.bits).cpu(),
                             lo=q.lo.half().cpu(), sc=q.sc.half().cpu(),
                             bias=None if b is None else b.half().cpu())
    return out


@torch.no_grad()
def dequantize(entry, device):
    n = 1
    for d in entry["shape"]:
        n *= d
    codes = unpack_codes(entry["packed"].to(device), entry["bits"], n).view(entry["shape"]).float()
    s = (-1,) + (1,) * (len(entry["shape"]) - 1)
    return codes * entry["sc"].to(device).float().view(s) + entry["lo"].to(device).float().view(s)


@torch.no_grad()
def load_packed_into(qm, packed, device):
    """Overwrite float weights with weights dequantized from the packed buffers (proves the round trip)."""
    max_err = 0.0
    for name, q in qm.named_modules():
        if isinstance(q, QLayer):
            w = dequantize(packed[name], device)
            max_err = max(max_err, (w - q.qweight()).abs().max().item())
            q.layer.weight.data.copy_(w)
            if packed[name]["bias"] is not None:
                q.layer.bias.data.copy_(packed[name]["bias"].to(device).float())
    return max_err

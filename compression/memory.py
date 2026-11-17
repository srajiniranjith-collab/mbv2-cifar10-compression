"""compression/memory.py -- in-memory (RAM/VRAM) size accounting and overhead tracking.

All sizes are computed from the ACTUAL tensors held in memory (numel * element_size), never from file size.
MB = 2^20 bytes.  Overheads that are tracked and INCLUDED in every compressed-size figure:
  * per-output-channel FP16 `lo` and `scale`  (4 bytes / channel)   -> metadata_bytes
  * FP16 folded-BN bias                       (2 bytes / channel)   -> bias_bytes
  * bit-packing padding (< 1 byte / layer)    -> packed_code_bytes - ideal_code_bytes
  * activation quantizer ranges lo/hi (FP32, 8 bytes / quantizer)   -> act_metadata_bytes
"""
import torch
import torch.nn as nn

from compression.quantize import ActQuant

# --------------------------------------------------------------------------------------
MB = 1024 * 1024


def fp32_size_report(fp_model):
    sd = fp_model.state_dict()
    total = sum(t.numel() * t.element_size() for t in sd.values() if t.dtype.is_floating_point)
    conv_lin_w = sum(m.weight.numel() * 4 for m in fp_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    return dict(fp32_model_bytes=total, fp32_convlinear_weight_bytes=conv_lin_w)


def size_report(fp_model, packed):
    fp = fp32_size_report(fp_model)
    codes = sum(e["packed"].numel() * e["packed"].element_size() for e in packed.values())
    meta = sum((e["lo"].numel() * e["lo"].element_size() + e["sc"].numel() * e["sc"].element_size())
               for e in packed.values())
    bias = sum(e["bias"].numel() * e["bias"].element_size() for e in packed.values() if e["bias"] is not None)
    ideal = sum(e["bits"] * int(torch.tensor(e["shape"]).prod()) for e in packed.values()) / 8
    qmodel = codes + meta + bias
    return dict(**fp, packed_code_bytes=codes, metadata_bytes=meta, bias_bytes=bias, ideal_code_bytes=ideal,
                quant_model_bytes=qmodel,
                model_ratio=fp["fp32_model_bytes"] / qmodel,
                weight_ratio_with_meta=fp["fp32_convlinear_weight_bytes"] / (codes + meta),
                weight_ratio_codes_only=fp["fp32_convlinear_weight_bytes"] / codes,
                model_size_mb=qmodel / MB, fp32_model_mb=fp["fp32_model_bytes"] / MB,
                avg_weight_bits=8 * codes / (fp["fp32_convlinear_weight_bytes"] / 4))


@torch.no_grad()
def activation_report(qm, device, hw=32):
    """Activations are measured with forward tracing: one forward pass on a single (1,3,hw,hw) sample,
    recording the per-sample element count of every tensor that passes through an activation quantizer
    (post-ReLU6 outputs, linear-projection outputs, residual sums, pooled vector).
      total = sum over all those tensors            (footprint if every inter-layer tensor were kept)
      peak  = max over consecutive pairs (in+out)   (working set of the most expensive layer; the
                                                     residual skip tensor is not added -> lower bound)
    FP32 baseline uses the same tensors at 32 bits. Per-quantizer metadata = lo,hi in FP32 (8 B each).
    Results scale linearly with batch size."""
    qm.eval()
    ActQuant.trace = []
    qm(torch.randn(1, 3, hw, hw, device=device))
    tr, ActQuant.trace = ActQuant.trace, None
    n_q = len(tr)
    fp_total = sum(n * 4 for _, n, _ in tr)
    q_total = sum(n * b / 8 for _, n, b in tr) + 8 * n_q
    fp_peak = max((tr[i - 1][1] if i else 0) * 4 + tr[i][1] * 4 for i in range(n_q))
    q_peak = max((tr[i - 1][1] * tr[i - 1][2] / 8 if i else 0) + tr[i][1] * tr[i][2] / 8 for i in range(n_q))
    return dict(act_fp32_total_bytes=fp_total, act_quant_total_bytes=q_total, act_ratio_total=fp_total / q_total,
                act_fp32_peak_bytes=fp_peak, act_quant_peak_bytes=q_peak, act_ratio_peak=fp_peak / q_peak,
                act_metadata_bytes=8 * n_q, n_act_tensors=n_q)


# --------------------------------------------------------------------------------------
# Live-memory helpers (sanity check that the analytical numbers match what is really resident)
# --------------------------------------------------------------------------------------
def packed_resident_bytes(packed) -> int:
    """Bytes actually held by the exported compressed buffers (codes + FP16 lo/scale + FP16 bias)."""
    tot = 0
    for e in packed.values():
        for k in ("packed", "lo", "sc", "bias"):
            t = e[k]
            if t is not None:
                tot += t.numel() * t.element_size()
    return tot


def vram_allocated_bytes(device) -> int:
    """Current CUDA allocator usage (0 on CPU)."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)
    return 0


def overhead_table(sizes: dict) -> str:
    """Human-readable breakdown of the compressed model size, including all storage overheads."""
    tot = sizes["quant_model_bytes"]
    rows = [("packed weight codes", sizes["packed_code_bytes"]),
            ("per-channel lo/scale metadata (FP16)", sizes["metadata_bytes"]),
            ("folded-BN bias (FP16)", sizes["bias_bytes"])]
    lines = [f"  {n:40s} {b / 1024:9.2f} KB  {100 * b / tot:5.1f}%" for n, b in rows]
    pad = sizes["packed_code_bytes"] - sizes["ideal_code_bytes"]
    lines.append(f"  {'(bit-packing padding, inside codes)':40s} {pad:9.1f} B")
    lines.append(f"  {'TOTAL':40s} {tot / 1024:9.2f} KB  ({tot / MB:.3f} MB)")
    return "\n".join(lines)

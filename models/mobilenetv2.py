"""MobileNetV2 adapted to CIFAR-10 (32x32 inputs).

Architecture follows torchvision's mobilenetv2.py, with two changes for small images:
  * the stem conv uses stride 1 (instead of 2)
  * the second stage (24 channels) uses stride 1 (instead of 2)
so the feature map goes 32 -> 32 -> 16 -> 8 -> 8 -> 4 instead of collapsing too early.

Every conv is a ConvBNAct block. Each block carries an `aq` slot (activation-quantizer hook)
that is an Identity in the FP32 model and replaced by quant.ActQuant during compression.
"""
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    """conv -> BN -> (ReLU6) -> [activation quantizer slot]."""

    def __init__(self, inp: int, oup: int, k: int = 3, stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, stride, (k - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.has_act = act
        self.act = nn.ReLU6(inplace=False) if act else nn.Identity()
        self.aq = nn.Identity()  # replaced by ActQuant in the compressed model
        self.folded = False

    def forward(self, x: Tensor) -> Tensor:
        return self.aq(self.act(self.bn(self.conv(x))))


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(round(inp * expand_ratio))
        self.use_res = stride == 1 and inp == oup
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden, k=1))                                   # pw expand
        layers.append(ConvBNAct(hidden, hidden, k=3, stride=stride, groups=hidden))     # depthwise
        layers.append(ConvBNAct(hidden, oup, k=1, act=False))                            # pw-linear project
        self.conv = nn.Sequential(*layers)
        self.out_aq = nn.Identity()  # quantizes the residual sum (only used if use_res)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.use_res:
            return self.out_aq(x + out)
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2,
                 round_nearest: int = 8, cifar: bool = True):
        super().__init__()
        s2 = 1 if cifar else 2
        s_stem = 1 if cifar else 2
        # t (expand), c (channels), n (repeats), s (stride of first block)
        setting = [[1, 16, 1, 1], [6, 24, 2, s2], [6, 32, 3, 2], [6, 64, 4, 2],
                   [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        inp = make_divisible(32 * width_mult, round_nearest)
        last = make_divisible(1280 * max(1.0, width_mult), round_nearest)
        feats: List[nn.Module] = [ConvBNAct(3, inp, k=3, stride=s_stem)]
        for t, c, n, s in setting:
            oup = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                feats.append(InvertedResidual(inp, oup, s if i == 0 else 1, t))
                inp = oup
        feats.append(ConvBNAct(inp, last, k=1))
        self.features = nn.Sequential(*feats)
        self.pool_aq = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last, num_classes)

        for m in self.modules():  # same init as torchvision
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.pool_aq(x)
        return self.fc(self.dropout(x))

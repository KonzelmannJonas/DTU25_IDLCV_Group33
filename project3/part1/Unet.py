
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 then DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscale by 2, concatenate skip, then DoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_ch = in_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            conv_in_ch = (in_ch // 2) + skip_ch

        self.conv = DoubleConv(conv_in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    """1x1 conv head that returns LOGITS (no sigmoid)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# UNet

class UNet(nn.Module):
    """
    Standard U-Net (logits out). Recommended input H,W divisible by 2**depth.
    Args:
        in_ch:      input channels (3 for RGB)
        out_ch:     output channels (1 for binary seg)
        base_ch:    base channel width (64 classic; 32 is lighter)
        depth:      # of downs (classic U-Net uses 4)
        bilinear:   use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base_ch: int = 64, depth: int = 4, bilinear: bool = True):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"
        self.depth = depth
        self.bilinear = bilinear

        # Encoder
        chs = [base_ch * (2 ** i) for i in range(depth)]  # e.g. [64,128,256,512]
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(chs[i], chs[i + 1]))

        # Bottleneck
        bottleneck_ch = chs[-1] * 2
        self.bottleneck = DoubleConv(chs[-1], bottleneck_ch)

        # Decoder
        self.ups = nn.ModuleList()
        up_in = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = chs[i]
            out_ch_i = chs[i]  # mirror
            self.ups.append(Up(in_ch=up_in, skip_ch=skip_ch, out_ch=out_ch_i, bilinear=bilinear))
            up_in = out_ch_i

        self.outc = OutConv(chs[0], out_ch)

    def forward(self, x):
        # Encoder saves skips
        skips = []
        x1 = self.inc(x)       # level 0
        skips.append(x1)
        x = x1
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # bottleneck
        x = self.bottleneck(skips[-1])

        # Decoder
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 1)]
            x = up(x, skip)


        logits = self.outc(x)
        return logits  # logits — apply sigmoid only for metrics/visualization

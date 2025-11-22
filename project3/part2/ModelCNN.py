import torch
import torch.nn as nn
import torch.nn.functional as F


# --- small building blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --- tiny UNet-like model ---
class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, base_ch)         # 3 -> 32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)     # 32 -> 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)   # 64 -> 128 (bottleneck)

        # decoder
        self.up2 = UpBlock(in_ch=base_ch*4, skip_ch=base_ch*2, out_ch=base_ch*2)  # 128 -> 64
        self.up1 = UpBlock(in_ch=base_ch*2, skip_ch=base_ch,   out_ch=base_ch)    # 64 -> 32

        # head (no sigmoid here)
        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # encode
        s1 = self.enc1(x)            # [B,32,H,W]
        x  = self.pool1(s1)          # [B,32,H/2,W/2]
        s2 = self.enc2(x)            # [B,64,H/2,W/2]
        x  = self.pool2(s2)          # [B,64,H/4,W/4]
        x  = self.enc3(x)            # [B,128,H/4,W/4]

        # decode
        x = self.up2(x, s2)          # [B,64,H/2,W/2]
        x = self.up1(x, s1)          # [B,32,H,W]
        logits = self.head(x)        # [B,1,H,W]
        return logits

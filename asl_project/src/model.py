import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = ConvBNSiLU(in_ch, in_ch, k=3, s=s, groups=in_ch)
        self.pw = ConvBNSiLU(in_ch, out_ch, k=1, s=1)
    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparable(ch, ch),
            DepthwiseSeparable(ch, ch),
        )
    def forward(self, x):
        return x + self.block(x)


class ASLNet(nn.Module):
    def __init__(self, num_classes=29, width=0.5):
        super().__init__()
        c = lambda x: int(x * width)
        self.stem = nn.Sequential(
            ConvBNSiLU(3, c(32), k=3, s=2),
            ConvBNSiLU(c(32), c(64), k=3, s=2),
        )
        self.stage1 = nn.Sequential(*[ResidualBlock(c(64)) for _ in range(2)])
        self.down2 = ConvBNSiLU(c(64), c(128), k=3, s=2)
        self.stage2 = nn.Sequential(*[ResidualBlock(c(128)) for _ in range(2)])
        self.down3 = ConvBNSiLU(c(128), c(256), k=3, s=2)
        self.stage3 = nn.Sequential(*[ResidualBlock(c(256)) for _ in range(2)])
        self.down4 = ConvBNSiLU(c(256), c(512), k=3, s=2)
        self.stage4 = nn.Sequential(*[ResidualBlock(c(512)) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(c(512), num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.down2(x))
        x = self.stage3(self.down3(x))
        x = self.stage4(self.down4(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation 블록 """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=[2, 3])  # Global Average Pooling
        y = F.silu(self.fc1(y))  # Swish 활성화
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y  # 채널별 중요도 조절


class GhostModule(nn.Module):
    """ Ghost Module - MobileNetV3 스타일 (경량화된 CNN) """
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super(GhostModule, self).__init__()
        self.primary_conv = nn.Conv2d(in_channels, out_channels // ratio, kernel_size, padding=1, bias=False)
        self.cheap_conv = nn.Conv2d(out_channels // ratio, out_channels // ratio, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        primary = F.silu(self.primary_conv(x))  # Swish 활성화
        cheap = F.silu(self.cheap_conv(primary))
        out = torch.cat([primary, cheap], dim=1)
        return self.bn(out)


class BottleneckBlock(nn.Module):
    """ ResNet 스타일의 Bottleneck 블록 """
    def __init__(self, in_channels, out_channels, expansion=4):
        super(BottleneckBlock, self).__init__()
        hidden_dim = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)  # Squeeze-and-Excitation 블록

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.silu(self.bn1(self.conv1(x)))  # Swish 활성화
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.shortcut:
            identity = self.shortcut(identity)

        return F.silu(out + identity)


class Net(nn.Module):
    """ 32x32 Grayscale(흑백) 이미지를 위한 복잡한 CNN 모델 """
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),  # Grayscale 지원 (채널=1)
            nn.BatchNorm2d(64),
            nn.SiLU()  # Swish 활성화 함수
        )

        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 128),
            GhostModule(128, 128)
        )

        self.layer2 = nn.Sequential(
            BottleneckBlock(128, 256),
            GhostModule(256, 256)
        )

        self.layer3 = nn.Sequential(
            BottleneckBlock(256, 512),
            GhostModule(512, 512),
            SEBlock(512)  # 추가적인 SE 블록
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.fc(x)
        return x

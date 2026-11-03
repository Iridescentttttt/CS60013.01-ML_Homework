"""卷积神经网络：MNIST 与可选的 ImageNet mini 共用同一套模块。

四种结构：
- TinyCNN：一层卷积，作为更弱的对照。
- LeNet：经典手写数字网络，作为 MNIST 基线。
- SmallCNN：带批归一化与全局平均池化的卷积网络，两种数据集都能用。
- MiniResNet：带残差连接的较深网络，面向 ImageNet mini 这类彩色自然图像。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """只有一层卷积的小网络，用来作为更弱的对照。

    4 个 3x3 卷积核，ReLU，2x2 最大池化，再全局平均池化。
    分类层只看见 4 个数，没有批归一化、Dropout 和残差。
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(torch.flatten(x, 1))


class LeNet(nn.Module):
    """LeNet-5 的现代写法。输入固定为 1x28x28，输出 10 类。

    与 1998 年原版的差别只有激活函数：这里用 ReLU，训练更稳定。
    空间尺寸：28 -> conv5 -> 24 -> pool -> 12 -> conv5 -> 8 -> pool -> 4。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ConvBlock(nn.Module):
    """两层 3x3 卷积 + BN + ReLU，最后 2x2 最大池化，通道翻倍、空间减半。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    """面向中小分辨率图像的卷积分类网络。

    三组卷积块逐步把通道扩到 4 * width，再用全局平均池化压成向量。
    因此同一套权重结构可以吃 28x28 的 MNIST，也可以吃 84x84 的 ImageNet mini，
    只需要改输入通道数和类别数。Dropout 放在分类头，减轻过拟合。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        width: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, width),
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    """两层 3x3 卷积的残差块。stride=2 时在第一条支路上下采样。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class MiniResNet(nn.Module):
    """给 ImageNet mini 用的轻量残差网络。

    茎部用 3x3、stride=1 的卷积，避免 7x7/stride=2 把 84x84 的小图压得太狠。
    四个阶段通道为 32-64-128-256，每阶段两个残差块，后三个阶段第一步下采样。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(32, 32, n_blocks=2, stride=1)
        self.layer2 = self._make_stage(32, 64, n_blocks=2, stride=2)
        self.layer3 = self._make_stage(64, 128, n_blocks=2, stride=2)
        self.layer4 = self._make_stage(128, 256, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "tiny_cnn":
        return TinyCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "lenet":
        if in_channels != 1:
            raise ValueError("LeNet 只用于单通道 MNIST，彩色数据请用 tiny_cnn、small_cnn 或 mini_resnet。")
        return LeNet(num_classes=num_classes)
    if name == "small_cnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "mini_resnet":
        return MiniResNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"未知模型 {name}，可选：tiny_cnn, lenet, small_cnn, mini_resnet")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

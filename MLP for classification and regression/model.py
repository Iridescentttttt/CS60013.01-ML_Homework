"""多层感知机：MNIST 分类和波士顿房价回归共用同一个模块。

最后一层不加激活。分类时它输出 logits，回归时它输出一个房价标量。
图像输入会在前向里拉成向量，因此分类网络不使用空间结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """全连接网络。hidden_dims 里的每一层后面接 ReLU，可选批归一化和 Dropout。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


# 分类：无隐藏层的线性分类器作基线；浅层网络一个隐藏层；深层网络加宽并做批归一化。
CLASSIFIER_PRESETS: dict[str, dict] = {
    "linear": {"hidden_dims": (), "dropout": 0.0, "batch_norm": False},
    "mlp_shallow": {"hidden_dims": (256,), "dropout": 0.2, "batch_norm": False},
    "mlp_deep": {"hidden_dims": (512, 256, 128), "dropout": 0.3, "batch_norm": True},
}

# 回归样本只有几百条。tiny 只有 4 个隐藏单元，用来和 64-32、128-64-32 对比容量。
REGRESSOR_PRESETS: dict[str, dict] = {
    "mlp_tiny": {"hidden_dims": (4,), "dropout": 0.0, "batch_norm": False},
    "mlp_small": {"hidden_dims": (64, 32), "dropout": 0.1, "batch_norm": False},
    "mlp_wide": {"hidden_dims": (128, 64, 32), "dropout": 0.2, "batch_norm": False},
}


def build_model(task: str, name: str, in_dim: int, out_dim: int) -> MLP:
    presets = CLASSIFIER_PRESETS if task == "mnist" else REGRESSOR_PRESETS
    if name not in presets:
        choices = ", ".join(presets)
        raise ValueError(f"任务 {task} 没有模型 {name}，可选：{choices}")
    cfg = presets[name]
    return MLP(
        in_dim=in_dim,
        hidden_dims=cfg["hidden_dims"],
        out_dim=out_dim,
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

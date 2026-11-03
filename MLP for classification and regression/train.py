"""训练并评估多层感知机。

两个任务：
1. MNIST 手写数字分类。数据用 torchvision 自动下载，已有缓存则直接读。
2. 波士顿房价回归。本地没有 CSV 时，从 StatLib 下载 Harrison & Rubinfeld (1978)
   的原始表。这和 Kaggle「Boston House Prices」以及旧版 sklearn 的 load_boston
   是同一份 506 x 13 的数据。新版 sklearn 已移除 load_boston。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import REGRESSOR_PRESETS, build_model, count_parameters


MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
BOSTON_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
BOSTON_URL = "http://lib.stat.cmu.edu/datasets/boston"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


def mnist_loaders(data_root: Path, batch_size: int, val_ratio: float, seed: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    full_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    n = len(full_train)
    n_val = int(n * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, [str(c) for c in test_set.classes]


def parse_statlib_boston(text: str) -> np.ndarray:
    """StatLib 原文把每条样本拆成两行，前一行 11 个数，后一行 3 个数。"""
    values: list[float] = []
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        if not started:
            if stripped.startswith("0.00632"):
                started = True
            else:
                continue
        if not stripped:
            continue
        values.extend(float(token) for token in stripped.split())
    data = np.asarray(values, dtype=np.float64)
    if data.size % 14 != 0:
        raise ValueError(f"波士顿数据列数不是 14 的倍数，读到 {data.size} 个数。")
    data = data.reshape(-1, 14)
    if data.shape[0] != 506:
        raise ValueError(f"波士顿数据应有 506 条，实际 {data.shape[0]} 条。")
    return data


def download_statlib_boston() -> str:
    """StatLib 原文。urllib 走不通时改用 curl。"""
    try:
        with urllib.request.urlopen(BOSTON_URL, timeout=30) as response:
            return response.read().decode("latin-1")
    except Exception as first_error:
        try:
            result = subprocess.run(
                ["curl", "-fsSL", "--max-time", "30", BOSTON_URL],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as curl_error:
            raise RuntimeError(
                "下载波士顿房价数据失败。请将 CSV 放到 data/boston_house_prices.csv，"
                "列为 CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV。"
            ) from curl_error
        if not result.stdout:
            raise RuntimeError("StatLib 返回空文件。") from first_error
        return result.stdout.decode("latin-1")


def ensure_boston_csv(path: Path) -> np.ndarray:
    if path.exists() and path.stat().st_size > 0:
        loaded = np.loadtxt(path, delimiter=",", skiprows=1)
        if loaded.shape == (506, 14):
            return loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    data = parse_statlib_boston(download_statlib_boston())
    header = ",".join(BOSTON_COLUMNS)
    body = "\n".join(",".join(f"{value:.10g}" for value in row) for row in data)
    path.write_text(header + "\n" + body + "\n")
    return data


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_cls_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_classes(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def run_reg_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    pred_parts, true_parts = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features).squeeze(-1)
            loss = criterion(preds, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            total += features.size(0)
            pred_parts.append(preds.detach().cpu().numpy())
            true_parts.append(targets.detach().cpu().numpy())
    return total_loss / total, np.concatenate(true_parts), np.concatenate(pred_parts)


def save_cls_curves(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    labels = [name.split(" - ")[0] for name in class_names]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reg_curves(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_rmse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("MSE loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_rmse"], label="train")
    axes[1].plot(epochs, history["val_rmse"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE (1000 USD)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residual = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_true, y_pred, s=18, alpha=0.75)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set_xlabel("True MEDV")
    axes[0].set_ylabel("Predicted MEDV")
    axes[0].set_title("Test set")
    axes[1].hist(residual, bins=20, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Prediction - true")
    axes[1].set_ylabel("count")
    axes[1].set_title("Test residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_mnist(args, device: torch.device) -> None:
    train_loader, val_loader, test_loader, classes = mnist_loaders(
        Path(args.data_root), args.batch_size, args.val_ratio, args.seed
    )
    model = build_model("mnist", args.model, in_dim=28 * 28, out_dim=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output) / f"mnist_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  model={args.model}  params={count_parameters(model):,}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_cls_epoch(
            model, tqdm(train_loader, desc=f"train {epoch}"), criterion, device, optimizer
        )
        val_loss, val_acc = run_cls_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_true, y_pred = predict_classes(model, test_loader, device)
    test_acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    save_cls_curves(history, out_dir / "curves.png")
    save_confusion(cm, classes, out_dir / "confusion_matrix.png")
    summary = {
        "task": "mnist",
        "model": args.model,
        "params": count_parameters(model),
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "epochs": args.epochs,
        "history": history,
        "classification_report": report,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "classification_report.txt").write_text(report)
    print(report)
    print(f"test_acc={test_acc:.4f}  best_val_acc={best_acc:.4f}")
    print(f"结果已写入 {out_dir}")


def train_boston(args, device: torch.device) -> None:
    table = ensure_boston_csv(Path(args.data_root) / "boston_house_prices.csv")
    features = table[:, :-1]
    target = table[:, -1]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features, target, test_size=0.2, random_state=args.seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, random_state=args.seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ArrayDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = build_model("boston", args.model, in_dim=features.shape[1], out_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 3, 1)
    )

    out_dir = Path(args.output) / f"boston_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}  model={args.model}  params={count_parameters(model):,}  "
        f"train={len(y_train)} val={len(y_val)} test={len(y_test)}"
    )

    history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        run_reg_epoch(model, train_loader, criterion, device, optimizer)
        # 曲线上的训练指标关掉 Dropout，否则训练损失会被随机置零抬高。
        train_loss, y_true_tr, y_pred_tr = run_reg_epoch(model, train_loader, criterion, device)
        val_loss, y_true_va, y_pred_va = run_reg_epoch(model, val_loader, criterion, device)
        train_rmse = float(np.sqrt(mean_squared_error(y_true_tr, y_pred_tr)))
        val_rmse = float(np.sqrt(mean_squared_error(y_true_va, y_pred_va)))
        scheduler.step(val_rmse)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f}  "
            f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}"
        )
        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"验证集 RMSE 连续 {args.patience} 轮没有下降，在 epoch {epoch} 停止。")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, y_true, y_pred = run_reg_epoch(model, test_loader, criterion, device)
    test_metrics = regression_metrics(y_true, y_pred)

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    linear_test = regression_metrics(y_test, linear.predict(x_test))
    linear_coef = {name: float(value) for name, value in zip(BOSTON_COLUMNS[:-1], linear.coef_)}

    save_reg_curves(history, out_dir / "curves.png")
    save_pred_plot(y_true, y_pred, out_dir / "pred_vs_true.png")
    summary = {
        "task": "boston",
        "model": args.model,
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "test": test_metrics,
        "linear_regression_test": linear_test,
        "linear_coefficients_standardized": linear_coef,
        "split": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "epochs_ran": len(history["train_loss"]),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"test_rmse={test_metrics['rmse']:.4f}  test_mae={test_metrics['mae']:.4f}  "
        f"test_r2={test_metrics['r2']:.4f}  best_val_rmse={best_rmse:.4f} @ epoch {best_epoch}"
    )
    print(
        f"linear_test_rmse={linear_test['rmse']:.4f}  "
        f"linear_test_mae={linear_test['mae']:.4f}  linear_test_r2={linear_test['r2']:.4f}"
    )
    print(f"结果已写入 {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="MLP classification on MNIST and regression on Boston housing")
    parser.add_argument("--task", choices=["mnist", "boston"], required=True)
    parser.add_argument("--model", default=None, help="mnist: linear / mlp_shallow / mlp_deep；boston: mlp_tiny / mlp_small / mlp_wide")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="MNIST 训练集中划给验证集的比例")
    parser.add_argument("--patience", type=int, default=40, help="波士顿回归的早停耐心")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps，默认自动选择")
    args = parser.parse_args()
    if args.model is None:
        args.model = "mlp_deep" if args.task == "mnist" else "mlp_small"
    if args.task == "boston" and args.model not in REGRESSOR_PRESETS:
        raise SystemExit(f"波士顿回归可选模型：{', '.join(REGRESSOR_PRESETS)}")
    if args.epochs is None:
        args.epochs = 8 if args.task == "mnist" else 300
    if args.batch_size is None:
        args.batch_size = 128 if args.task == "mnist" else 32
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    if args.task == "mnist":
        train_mnist(args, device)
    else:
        train_boston(args, device)

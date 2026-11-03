"""训练并评估卷积分类网络。

MNIST 可以直接跑，数据会自动下载。ImageNet mini 是可选项：
把数据放到 --data-root 后，用 --dataset imagenet_mini 启动。

目录支持两种常见摆法：
1. ImageFolder：<root>/{train,val}/<类别名>/*.jpg
2. mini-ImageNet CSV：<root>/images/*.jpg，以及 train.csv / val.csv / test.csv
   CSV 至少包含 filename 和 label 两列。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import build_model, count_parameters


MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class PathImageDataset(Dataset):
    """按 (路径, 标签) 列表读图，用来把互斥类别重新做成普通分类划分。"""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def images_by_class(split_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not split_dir.is_dir():
        return groups
    suffixes = {".jpg", ".jpeg", ".png"}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        files = [path for path in class_dir.iterdir() if path.suffix.lower() in suffixes]
        if files:
            groups[class_dir.name] = files
    return groups


class CsvImageDataset(Dataset):
    """读取 mini-ImageNet 那种「图片目录 + CSV 标签」的数据。"""

    def __init__(self, image_dir: Path, csv_path: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{csv_path} 是空的")
        fields = {name.lower(): name for name in rows[0].keys()}
        file_key = fields.get("filename") or fields.get("file") or fields.get("image")
        label_key = fields.get("label") or fields.get("class") or fields.get("category")
        if file_key is None or label_key is None:
            raise ValueError(f"{csv_path} 需要 filename 和 label 两列，实际列名：{list(rows[0])}")
        labels = sorted({row[label_key] for row in rows})
        self.class_to_idx = {name: i for i, name in enumerate(labels)}
        self.classes = labels
        self.samples = [(row[file_key], self.class_to_idx[row[label_key]]) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, label = self.samples[index]
        path = self.image_dir / name
        if not path.exists():
            path = self.image_dir / Path(name).name
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def mnist_loaders(data_root: Path, batch_size: int, val_ratio: float, seed: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    full_train = datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.MNIST(data_root, train=True, download=True, transform=eval_tf)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=eval_tf)

    n = len(full_train)
    n_val = int(n * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, test_set.classes, 1


def stratified_classification_split(
    groups: dict[str, list[Path]], seed: int
) -> tuple[list[str], list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """每个类别按 8:1:1 分成训练、验证、测试。类别按名字排序，标签稳定。"""
    classes = sorted(groups)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []
    for name in classes:
        files = groups[name][:]
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, int(round(n * 0.1)))
        n_val = max(1, int(round(n * 0.1)))
        if n_test + n_val >= n:
            n_test = max(1, n // 10)
            n_val = max(1, n // 10)
        label = class_to_idx[name]
        test_samples.extend((path, label) for path in files[:n_test])
        val_samples.extend((path, label) for path in files[n_test : n_test + n_val])
        train_samples.extend((path, label) for path in files[n_test + n_val :])
    return classes, train_samples, val_samples, test_samples


def imagenet_mini_loaders(data_root: Path, batch_size: int, image_size: int, seed: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dir = data_root / "train"
    val_dir = data_root / "val" if (data_root / "val").exists() else data_root / "valid"
    test_dir = data_root / "test"
    if train_dir.is_dir() and val_dir.is_dir():
        grouped = images_by_class(train_dir)
        for extra in (val_dir, test_dir):
            for name, files in images_by_class(extra).items():
                grouped.setdefault(name, []).extend(files)
        train_names = set(images_by_class(train_dir))
        val_names = set(images_by_class(val_dir))
        disjoint = train_names.isdisjoint(val_names)
        if disjoint:
            classes, train_samples, val_samples, test_samples = stratified_classification_split(grouped, seed)
            print(
                "官方 train/val/test 的类别互不重叠（64/16/20，少样本划分）。"
                f"已合并为 {len(classes)} 类普通分类，每类按 8:1:1 重新划分："
                f"train {len(train_samples)}，val {len(val_samples)}，test {len(test_samples)}。"
            )
            train_set = PathImageDataset(train_samples, transform=train_tf)
            val_set = PathImageDataset(val_samples, transform=eval_tf)
            test_set = PathImageDataset(test_samples, transform=eval_tf)
        else:
            train_set = datasets.ImageFolder(train_dir, transform=train_tf)
            val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
            test_set = datasets.ImageFolder(test_dir, transform=eval_tf) if test_dir.is_dir() else val_set
            classes = train_set.classes
    else:
        image_dir = data_root / "images" if (data_root / "images").is_dir() else data_root
        train_csv = data_root / "train.csv"
        val_csv = data_root / "val.csv"
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(
                f"在 {data_root} 下没有找到 ImageNet mini。\n"
                "请提供以下任一结构：\n"
                "  1) train/ 与 val/ 子目录，每个类别一个文件夹；\n"
                "  2) images/ 加上 train.csv、val.csv（可选 test.csv）。"
            )
        train_set = CsvImageDataset(image_dir, train_csv, transform=train_tf)
        val_set = CsvImageDataset(image_dir, val_csv, transform=eval_tf)
        test_csv = data_root / "test.csv"
        test_set = CsvImageDataset(image_dir, test_csv, transform=eval_tf) if test_csv.exists() else val_set
        classes = train_set.classes

    loader_kwargs = {"num_workers": 4, "persistent_workers": True}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, classes, 3


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
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
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def save_curves(history: dict, path: Path) -> None:
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
    fig_w = max(6, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    labels = class_names if len(class_names) <= 20 else [str(i) for i in range(len(class_names))]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    data_root = Path(args.data_root)
    out_dir = Path(args.output) / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mnist":
        train_loader, val_loader, test_loader, classes, in_channels = mnist_loaders(
            data_root, args.batch_size, args.val_ratio, args.seed
        )
    elif args.dataset == "imagenet_mini":
        train_loader, val_loader, test_loader, classes, in_channels = imagenet_mini_loaders(
            data_root, args.batch_size, args.image_size, args.seed
        )
    else:
        raise ValueError(args.dataset)

    model = build_model(args.model, in_channels, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"device={device}  model={args.model}  params={count_parameters(model):,}")
    print(f"classes={len(classes)}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, tqdm(train_loader, desc=f"train {epoch}"), criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
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
    y_true, y_pred = predict_all(model, test_loader, device)
    test_acc = float((y_true == y_pred).mean())
    names = [str(c) for c in classes]
    report = classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    save_curves(history, out_dir / "curves.png")
    if len(classes) <= 30:
        save_confusion(cm, names, out_dir / "confusion_matrix.png")

    summary = {
        "dataset": args.dataset,
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


def smoke_imagenet(device: torch.device) -> None:
    """不下载数据，用随机张量确认 ImageNet mini 网络能前向和反传。"""
    model = build_model("mini_resnet", in_channels=3, num_classes=100).to(device)
    images = torch.randn(4, 3, 84, 84, device=device)
    labels = torch.randint(0, 100, (4,), device=device)
    logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    print(f"mini_resnet smoke ok  logits={tuple(logits.shape)}  loss={loss.item():.4f}  params={count_parameters(model):,}")


def parse_args():
    parser = argparse.ArgumentParser(description="CNN classification on MNIST / ImageNet mini")
    parser.add_argument("--dataset", choices=["mnist", "imagenet_mini"], default="mnist")
    parser.add_argument("--model", choices=["tiny_cnn", "lenet", "small_cnn", "mini_resnet"], default="small_cnn")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=84, help="ImageNet mini 的输入边长")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps，默认自动选择")
    parser.add_argument("--smoke-imagenet", action="store_true", help="只用随机张量检查 mini_resnet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_imagenet:
        set_seed(args.seed)
        smoke_imagenet(torch.device(args.device) if args.device else get_device())
    else:
        train(args)

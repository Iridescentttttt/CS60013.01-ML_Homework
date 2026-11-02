"""训练并评估支持向量机。

两个任务：
1. MNIST 手写数字分类。线性 SVM 用原始像素；非线性 SVM 用 RBF 核。
   另外在同一组主成分上训练线性 SVM，用来把降维和核函数分开。
2. 波士顿房价回归。线性 SVR 和 RBF 核 SVR。数据与多层感知机实验同一份 CSV。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
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
from sklearn.svm import SVC, LinearSVC, SVR
from torchvision import datasets


LINEAR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
PCA_DIMS = (40, 80, 120)
PCA_CURVE_DIM = 200
RBF_C_GRID = (0.1, 1.0, 10.0, 100.0)
# 相对 sklearn 的 gamma="scale"。10 倍会让核过尖，支持向量接近全体样本，
# 单次拟合要一两分钟，验证准确率掉到 0.4 附近，所以网格停在 3 倍。
RBF_GAMMA_MULT = (0.1, 1.0, 3.0)
SVR_C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SVR_EPSILON_GRID = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
SVR_GAMMA_MULT = (0.1, 0.3, 1.0, 3.0, 10.0)

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
DIGIT_NAMES = [str(i) for i in range(10)]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False))


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def environment() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def timed_fit(clf, features: np.ndarray, target: np.ndarray) -> float:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(features, target)
    elapsed = time.perf_counter() - started
    if caught:
        print("  liblinear 在 max_iter 内没有收敛", flush=True)
    return elapsed


def save_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_weight_images(weights: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    for digit, axis in enumerate(axes.ravel()):
        image = weights[digit].reshape(28, 28)
        limit = float(np.max(np.abs(image)))
        if limit == 0:
            limit = 1.0
        axis.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(str(digit))
        axis.axis("off")
    fig.suptitle("One-vs-rest weights in pixel space")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_c_curve(trials: list[dict], path: Path, title: str) -> None:
    ordered = sorted(trials, key=lambda item: item["C"])
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = [item["C"] for item in ordered]
    ax.plot(cs, [item["train_acc"] for item in ordered], marker="o", label="train")
    ax.plot(cs, [item["val_acc"] for item in ordered], marker="o", label="val")
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pca_linear_curve(trials: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for dim in PCA_DIMS:
        chosen = sorted((item for item in trials if item["pca_dim"] == dim), key=lambda item: item["C"])
        ax.plot(
            [item["C"] for item in chosen],
            [item["val_acc"] for item in chosen],
            marker="o",
            label=f"{dim} components",
        )
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Linear SVM on PCA features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pca_curve(cumulative: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    components = np.arange(1, len(cumulative) + 1)
    ax.plot(components, cumulative)
    for level in (0.80, 0.90, 0.95):
        ax.axhline(level, color="gray", linewidth=0.8)
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA on training pixels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rbf_heatmaps(trials: list[dict], best: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(PCA_DIMS), figsize=(4.2 * len(PCA_DIMS), 3.8), squeeze=False)
    scores = [item["val_acc"] for item in trials]
    low, high = min(scores), max(scores)
    if high - low < 1e-6:
        low, high = high - 0.01, high
    for axis, dim in zip(axes[0], PCA_DIMS):
        grid = np.full((len(RBF_GAMMA_MULT), len(RBF_C_GRID)), np.nan)
        for item in trials:
            if item["pca_dim"] != dim:
                continue
            row = RBF_GAMMA_MULT.index(item["gamma_mult"])
            col = RBF_C_GRID.index(item["C"])
            grid[row, col] = item["val_acc"]
        image = axis.imshow(grid, vmin=low, vmax=high, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(RBF_C_GRID)))
        axis.set_xticklabels([str(c) for c in RBF_C_GRID])
        axis.set_yticks(range(len(RBF_GAMMA_MULT)))
        axis.set_yticklabels([f"{mult:g}x" for mult in RBF_GAMMA_MULT])
        axis.set_xlabel("C")
        axis.set_ylabel("gamma / scale")
        axis.set_title(f"{dim} components")
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                axis.text(col, row, f"{grid[row, col]:.3f}", ha="center", va="center", color="white", fontsize=8)
        if best["pca_dim"] == dim:
            row = RBF_GAMMA_MULT.index(best["gamma_mult"])
            col = RBF_C_GRID.index(best["C"])
            axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))
        fig.colorbar(image, ax=axis, fraction=0.046)
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


def save_svr_curve(trials: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    epsilons = sorted({item["epsilon"] for item in trials})
    gammas = sorted({item.get("gamma_mult") for item in trials})
    # 线性 SVR 没有 gamma。RBF 只画验证集上最优 gamma 的那一组，避免线叠在一起。
    if gammas != [None]:
        best_gamma = min(trials, key=lambda item: item["val_rmse"])["gamma_mult"]
        trials = [item for item in trials if item["gamma_mult"] == best_gamma]
        title = f"{title}, gamma={best_gamma:g}x scale"
    for epsilon in epsilons:
        chosen = sorted((item for item in trials if item["epsilon"] == epsilon), key=lambda item: item["C"])
        if not chosen:
            continue
        ax.plot(
            [item["C"] for item in chosen],
            [item["val_rmse"] for item in chosen],
            marker="o",
            label=f"epsilon={epsilon:g}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("validation RMSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, limit: int = 5) -> list[dict]:
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), DIGIT_NAMES[i], DIGIT_NAMES[j]))
    pairs.sort(reverse=True)
    return [{"count": count, "true": true, "pred": pred} for count, true, pred in pairs[:limit]]


def cls_key(trial: dict) -> tuple:
    return (trial["val_acc"], -trial["C"], -trial.get("pca_dim", 0), -trial.get("gamma", 0.0))


def reg_key(trial: dict) -> tuple:
    return (-trial["val_rmse"], -trial["C"], -trial.get("gamma", 0.0), trial["epsilon"])


def load_mnist(data_root: Path):
    train_set = datasets.MNIST(data_root, train=True, download=True)
    test_set = datasets.MNIST(data_root, train=False, download=True)

    def pack(dataset):
        pixels = dataset.data.numpy().reshape(len(dataset), -1).astype(np.float64) / 255.0
        labels = dataset.targets.numpy().astype(np.int64)
        return pixels, labels

    return pack(train_set), pack(test_set)


def split_train_val(n: int, val_ratio: float, seed: int):
    n_val = int(n * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).cpu().numpy()
    return perm[n_val:], perm[:n_val]


def prepare_mnist(args):
    (x_all, y_all), (x_test, y_test) = load_mnist(Path(args.data_root))
    train_idx, val_idx = split_train_val(len(y_all), args.val_ratio, args.seed)
    pixel_std = x_all[train_idx].std(axis=0)
    return {
        "x_train": x_all[train_idx],
        "y_train": y_all[train_idx],
        "x_val": x_all[val_idx],
        "y_val": y_all[val_idx],
        "x_test": x_test,
        "y_test": y_test,
        "n_constant_pixels": int(np.sum(pixel_std == 0)),
        "n_low_variance_pixels": int(np.sum(pixel_std < 0.05)),
    }


def fit_pca(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, seed: int):
    pca = PCA(n_components=PCA_CURVE_DIM, svd_solver="full", random_state=seed)
    z_train = pca.fit_transform(x_train)
    z_val = pca.transform(x_val)
    z_test = pca.transform(x_test)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return pca, z_train, z_val, z_test, cumulative


def pca_summary(cumulative: np.ndarray) -> dict:
    def components_for(level: float) -> int:
        return int(np.searchsorted(cumulative, level) + 1)

    return {
        "components_for_variance": {
            "0.80": components_for(0.80),
            "0.90": components_for(0.90),
            "0.95": components_for(0.95),
        },
        "variance_at_dims": {str(dim): float(cumulative[dim - 1]) for dim in PCA_DIMS},
    }


def slice_pca(features: np.ndarray, dim: int) -> np.ndarray:
    return np.ascontiguousarray(features[:, :dim])


def gamma_scale(features: np.ndarray) -> float:
    variance = float(features.var())
    if variance == 0:
        return 1.0
    return 1.0 / (features.shape[1] * variance)


def pixel_weights(clf, pca: PCA | None, pca_dim: int | None) -> np.ndarray:
    coef = np.asarray(clf.coef_, dtype=np.float64)
    if pca is None:
        return coef
    return coef @ pca.components_[:pca_dim]


def evaluate_classifier(clf, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    pred = clf.predict(features)
    return float((pred == labels).mean()), pred


def write_classifier_outputs(
    args,
    name: str,
    clf,
    trial: dict,
    trials: list[dict],
    x_train,
    y_train,
    x_test,
    y_test,
    pca: PCA | None = None,
    pca_dim: int | None = None,
    extra: dict | None = None,
) -> None:
    out_dir = Path(args.output) / f"mnist_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_acc, _ = evaluate_classifier(clf, x_train, y_train)
    test_acc, y_pred = evaluate_classifier(clf, x_test, y_test)
    report = classification_report(y_test, y_pred, target_names=DIGIT_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    save_confusion(cm, DIGIT_NAMES, out_dir / "confusion_matrix.png")
    if hasattr(clf, "coef_"):
        save_weight_images(pixel_weights(clf, pca, pca_dim), out_dir / "weights.png")

    n_support = None
    n_support_per_class = None
    if hasattr(clf, "n_support_"):
        n_support_per_class = [int(v) for v in clf.n_support_]
        n_support = int(np.sum(clf.n_support_))

    summary = {
        "task": "mnist",
        "model": name,
        "best": trial,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_support": n_support,
        "n_support_per_class": n_support_per_class,
        "split": {
            "train": int(len(y_train)),
            "val": int(trial["n_val"]),
            "test": int(len(y_test)),
        },
        "top_confusions": top_confusions(cm),
        "confusion_matrix": cm,
        "trials": sorted(trials, key=cls_key, reverse=True),
        "classification_report": report,
        "environment": environment(),
        "elapsed_sec": trial.get("search_sec"),
    }
    if extra:
        summary.update(extra)
    dump_json(out_dir / "metrics.json", summary)
    (out_dir / "classification_report.txt").write_text(report)
    dump(
        {"model": clf, "pca": pca, "pca_dim": pca_dim, "params": to_builtin(trial)},
        out_dir / "model.joblib",
        compress=1,
    )
    print(report)
    print(
        f"test_acc={test_acc:.4f}  best_val_acc={trial['val_acc']:.4f}  train_acc={train_acc:.4f}",
        flush=True,
    )
    print(f"结果已写入 {out_dir}", flush=True)


def run_linear(args, data: dict) -> None:
    print("训练线性 SVM（原始像素，LinearSVC，one-vs-rest）", flush=True)
    trials = []
    best_clf = None
    best = None
    started = time.perf_counter()
    for c_value in LINEAR_C_GRID:
        clf = LinearSVC(
            C=c_value,
            loss="squared_hinge",
            dual=False,
            max_iter=5000,
            tol=1e-4,
            random_state=args.seed,
        )
        elapsed = timed_fit(clf, data["x_train"], data["y_train"])
        train_acc, _ = evaluate_classifier(clf, data["x_train"], data["y_train"])
        val_acc, _ = evaluate_classifier(clf, data["x_val"], data["y_val"])
        trial = {
            "C": c_value,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "seconds": elapsed,
            "n_val": int(len(data["y_val"])),
            "max_n_iter": int(np.max(clf.n_iter_)),
        }
        trials.append(trial)
        print(
            f"linear C={c_value:<6g}  {elapsed:6.1f}s  train={train_acc:.4f} val={val_acc:.4f}",
            flush=True,
        )
        if best is None or cls_key(trial) > cls_key(best):
            best = trial
            best_clf = clf
    best = dict(best)
    best["search_sec"] = time.perf_counter() - started
    out_dir = Path(args.output) / "mnist_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_c_curve(trials, out_dir / "c_curve.png", "Linear SVM")
    write_classifier_outputs(
        args,
        "linear",
        best_clf,
        best,
        trials,
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
        extra={
            "n_constant_pixels": data["n_constant_pixels"],
            "n_low_variance_pixels": data["n_low_variance_pixels"],
            "n_parameters": int(best_clf.coef_.size + best_clf.intercept_.size),
        },
    )


def run_linear_pca(args, data: dict) -> None:
    print("训练线性 SVM（PCA 特征，用来和 RBF 对照）", flush=True)
    pca, z_train, z_val, z_test, cumulative = fit_pca(
        data["x_train"], data["x_val"], data["x_test"], args.seed
    )
    trials = []
    best_clf = None
    best = None
    started = time.perf_counter()
    for dim in PCA_DIMS:
        x_train = slice_pca(z_train, dim)
        x_val = slice_pca(z_val, dim)
        for c_value in LINEAR_C_GRID:
            clf = LinearSVC(
                C=c_value,
                loss="squared_hinge",
                dual=False,
                max_iter=5000,
                tol=1e-4,
                random_state=args.seed,
            )
            elapsed = timed_fit(clf, x_train, data["y_train"])
            train_acc, _ = evaluate_classifier(clf, x_train, data["y_train"])
            val_acc, _ = evaluate_classifier(clf, x_val, data["y_val"])
            trial = {
                "pca_dim": dim,
                "C": c_value,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "seconds": elapsed,
                "n_val": int(len(data["y_val"])),
                "explained_variance": float(cumulative[dim - 1]),
            }
            trials.append(trial)
            print(
                f"linear_pca dim={dim:<3d} C={c_value:<6g}  {elapsed:6.1f}s  "
                f"train={train_acc:.4f} val={val_acc:.4f}",
                flush=True,
            )
            if best is None or cls_key(trial) > cls_key(best):
                best = trial
                best_clf = clf
    best = dict(best)
    best["search_sec"] = time.perf_counter() - started
    dim = int(best["pca_dim"])
    out_dir = Path(args.output) / "mnist_linear_pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pca_curve(cumulative, out_dir / "pca_variance.png")
    save_pca_linear_curve(trials, out_dir / "c_curve.png")
    write_classifier_outputs(
        args,
        "linear_pca",
        best_clf,
        best,
        trials,
        slice_pca(z_train, dim),
        data["y_train"],
        slice_pca(z_test, dim),
        data["y_test"],
        pca=pca,
        pca_dim=dim,
        extra={
            "pca": pca_summary(cumulative),
            "n_parameters": int(best_clf.coef_.size + best_clf.intercept_.size),
        },
    )


def run_rbf(args, data: dict) -> None:
    print("训练 RBF SVM（SVC，one-vs-one）", flush=True)
    pca, z_train, z_val, z_test, cumulative = fit_pca(
        data["x_train"], data["x_val"], data["x_test"], args.seed
    )
    trials = []
    best_clf = None
    best = None
    started = time.perf_counter()
    total = len(PCA_DIMS) * len(RBF_C_GRID) * len(RBF_GAMMA_MULT)
    done = 0
    print(f"RBF 超参共 {total} 组", flush=True)
    for dim in PCA_DIMS:
        x_train = slice_pca(z_train, dim)
        x_val = slice_pca(z_val, dim)
        base_gamma = gamma_scale(x_train)
        for c_value in RBF_C_GRID:
            for mult in RBF_GAMMA_MULT:
                done += 1
                gamma = base_gamma * mult
                clf = SVC(C=c_value, kernel="rbf", gamma=gamma, cache_size=1024)
                elapsed = timed_fit(clf, x_train, data["y_train"])
                val_acc, _ = evaluate_classifier(clf, x_val, data["y_val"])
                trial = {
                    "pca_dim": dim,
                    "C": c_value,
                    "gamma": gamma,
                    "gamma_mult": mult,
                    "gamma_scale": base_gamma,
                    "val_acc": val_acc,
                    "n_support": int(np.sum(clf.n_support_)),
                    "seconds": elapsed,
                    "n_val": int(len(data["y_val"])),
                    "explained_variance": float(cumulative[dim - 1]),
                }
                trials.append(trial)
                print(
                    f"[{done:02d}/{total}] rbf dim={dim:<3d} C={c_value:<6g} "
                    f"gamma={gamma:.4g} ({mult:g}x)  "
                    f"{elapsed:6.1f}s  val={val_acc:.4f} sv={trial['n_support']}",
                    flush=True,
                )
                if best is None or cls_key(trial) > cls_key(best):
                    best = trial
                    best_clf = clf
    best = dict(best)
    best["search_sec"] = time.perf_counter() - started
    dim = int(best["pca_dim"])
    out_dir = Path(args.output) / "mnist_rbf"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pca_curve(cumulative, out_dir / "pca_variance.png")
    save_rbf_heatmaps(trials, best, out_dir / "val_heatmap.png")
    write_classifier_outputs(
        args,
        "rbf",
        best_clf,
        best,
        trials,
        slice_pca(z_train, dim),
        data["y_train"],
        slice_pca(z_test, dim),
        data["y_test"],
        pca=pca,
        pca_dim=dim,
        extra={"pca": pca_summary(cumulative)},
    )


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
    table = np.asarray(values, dtype=np.float64)
    if table.size % 14 != 0:
        raise ValueError(f"波士顿数据列数不是 14 的倍数，读到 {table.size} 个数。")
    table = table.reshape(-1, 14)
    if table.shape[0] != 506:
        raise ValueError(f"波士顿数据应有 506 条，实际 {table.shape[0]} 条。")
    return table


def download_statlib_boston() -> str:
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
    table = parse_statlib_boston(download_statlib_boston())
    header = ",".join(BOSTON_COLUMNS)
    body = "\n".join(",".join(f"{value:.10g}" for value in row) for row in table)
    path.write_text(header + "\n" + body + "\n")
    return table


def load_boston(args):
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
    return {
        "x_train": scaler.fit_transform(x_train),
        "x_val": scaler.transform(x_val),
        "x_test": scaler.transform(x_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "target": target,
        "feature_names": BOSTON_COLUMNS[:-1],
    }


def linear_baseline(data: dict) -> tuple[dict[str, float], dict[str, float]]:
    model = LinearRegression()
    model.fit(data["x_train"], data["y_train"])
    metrics = regression_metrics(data["y_test"], model.predict(data["x_test"]))
    coefficients = {
        name: float(value) for name, value in zip(data["feature_names"], model.coef_)
    }
    return metrics, coefficients


def residual_notes(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    order = np.argsort(-np.abs(error))[:5]
    capped = np.isclose(y_true, 50.0)
    notes = {
        "largest_abs_errors": [
            {"true": float(y_true[i]), "pred": float(y_pred[i]), "error": float(error[i])}
            for i in order
        ]
    }
    if np.any(capped):
        notes["capped_at_50"] = {
            "count": int(capped.sum()),
            "mae": float(np.mean(np.abs(error[capped]))),
            "mean_pred": float(np.mean(y_pred[capped])),
        }
    else:
        notes["capped_at_50"] = {"count": 0}
    return notes


def write_regressor_outputs(args, name: str, clf, trial: dict, trials: list[dict], data: dict, extra: dict) -> None:
    out_dir = Path(args.output) / f"boston_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_train = clf.predict(data["x_train"])
    pred_val = clf.predict(data["x_val"])
    pred_test = clf.predict(data["x_test"])
    test_metrics = regression_metrics(data["y_test"], pred_test)
    linear_test, linear_coef = linear_baseline(data)
    save_pred_plot(data["y_test"], pred_test, out_dir / "pred_vs_true.png")
    save_svr_curve(trials, out_dir / "val_rmse.png", name)
    coef = None
    if hasattr(clf, "coef_"):
        coef = {
            feature: float(value)
            for feature, value in zip(data["feature_names"], np.ravel(clf.coef_))
        }
    n_support = int(clf.support_.size) if hasattr(clf, "support_") else None
    summary = {
        "task": "boston",
        "model": name,
        "best": trial,
        "train": regression_metrics(data["y_train"], pred_train),
        "val": regression_metrics(data["y_val"], pred_val),
        "test": test_metrics,
        "linear_regression_test": linear_test,
        "linear_regression_coefficients_standardized": linear_coef,
        "svr_coefficients_standardized": coef,
        "n_support": n_support,
        "split": {
            "train": int(len(data["y_train"])),
            "val": int(len(data["y_val"])),
            "test": int(len(data["y_test"])),
        },
        "target_mean": float(data["target"].mean()),
        "target_std": float(data["target"].std()),
        "residual_notes": residual_notes(data["y_test"], pred_test),
        "trials": sorted(trials, key=lambda item: item["val_rmse"]),
        "environment": environment(),
    }
    summary.update(extra)
    dump_json(out_dir / "metrics.json", summary)
    dump({"model": clf, "params": to_builtin(trial)}, out_dir / "model.joblib", compress=1)
    print(
        f"test_rmse={test_metrics['rmse']:.4f}  test_mae={test_metrics['mae']:.4f}  "
        f"test_r2={test_metrics['r2']:.4f}  best_val_rmse={trial['val_rmse']:.4f}",
        flush=True,
    )
    print(
        f"linear_test_rmse={linear_test['rmse']:.4f}  "
        f"linear_test_mae={linear_test['mae']:.4f}  linear_test_r2={linear_test['r2']:.4f}",
        flush=True,
    )
    print(f"结果已写入 {out_dir}", flush=True)


def run_linear_svr(args, data: dict) -> None:
    print("训练线性 SVR（SVR，线性核）", flush=True)
    trials = []
    best_clf = None
    best = None
    started = time.perf_counter()
    for c_value in SVR_C_GRID:
        for epsilon in SVR_EPSILON_GRID:
            clf = SVR(kernel="linear", C=c_value, epsilon=epsilon, cache_size=512)
            elapsed = timed_fit(clf, data["x_train"], data["y_train"])
            val_rmse = regression_metrics(data["y_val"], clf.predict(data["x_val"]))["rmse"]
            trial = {
                "C": c_value,
                "epsilon": epsilon,
                "val_rmse": val_rmse,
                "seconds": elapsed,
                "n_support": int(clf.support_.size),
            }
            trials.append(trial)
            if best is None or reg_key(trial) > reg_key(best):
                print(
                    f"  linear_svr C={c_value:<6g} epsilon={epsilon:<3g}  val_rmse={val_rmse:.4f}",
                    flush=True,
                )
                best = trial
                best_clf = clf
    best = dict(best)
    best["search_sec"] = time.perf_counter() - started
    write_regressor_outputs(args, "linear_svr", best_clf, best, trials, data, {})


def run_rbf_svr(args, data: dict) -> None:
    print("训练 RBF SVR", flush=True)
    base_gamma = gamma_scale(data["x_train"])
    trials = []
    best_clf = None
    best = None
    started = time.perf_counter()
    for c_value in SVR_C_GRID:
        for epsilon in SVR_EPSILON_GRID:
            for mult in SVR_GAMMA_MULT:
                gamma = base_gamma * mult
                clf = SVR(kernel="rbf", C=c_value, epsilon=epsilon, gamma=gamma, cache_size=512)
                timed_fit(clf, data["x_train"], data["y_train"])
                val_rmse = regression_metrics(data["y_val"], clf.predict(data["x_val"]))["rmse"]
                trial = {
                    "C": c_value,
                    "epsilon": epsilon,
                    "gamma": gamma,
                    "gamma_mult": mult,
                    "gamma_scale": base_gamma,
                    "val_rmse": val_rmse,
                    "n_support": int(clf.support_.size),
                }
                trials.append(trial)
                if best is None or reg_key(trial) > reg_key(best):
                    print(
                        f"  rbf_svr C={c_value:<6g} epsilon={epsilon:<3g} gamma={gamma:.4g} ({mult:g}x)  "
                        f"val_rmse={val_rmse:.4f} sv={trial['n_support']}",
                        flush=True,
                    )
                    best = trial
                    best_clf = clf
    best = dict(best)
    best["search_sec"] = time.perf_counter() - started
    write_regressor_outputs(args, "rbf_svr", best_clf, best, trials, data, {})


def parse_args():
    parser = argparse.ArgumentParser(description="SVM classification on MNIST and SVR on Boston housing")
    parser.add_argument("--task", choices=["mnist", "boston", "all"], required=True)
    parser.add_argument(
        "--model",
        default="all",
        help="mnist: linear / linear_pca / rbf；boston: linear_svr / rbf_svr；all 跑当前任务的全部模型",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="MNIST 训练集中划给验证集的比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mnist_models = ["linear", "linear_pca", "rbf"]
    boston_models = ["linear_svr", "rbf_svr"]
    if args.task == "mnist":
        allowed = mnist_models
    elif args.task == "boston":
        allowed = boston_models
    else:
        allowed = boston_models + mnist_models
    if args.model != "all" and args.model not in allowed:
        raise SystemExit(f"任务 {args.task} 可选模型：all, {', '.join(allowed)}")
    if args.model == "all":
        args.models = allowed
    else:
        args.models = [args.model]
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(environment(), flush=True)
    runners = {
        "linear": ("mnist", run_linear),
        "linear_pca": ("mnist", run_linear_pca),
        "rbf": ("mnist", run_rbf),
        "linear_svr": ("boston", run_linear_svr),
        "rbf_svr": ("boston", run_rbf_svr),
    }
    mnist_data = None
    boston_data = None
    if any(name in ("linear", "linear_pca", "rbf") for name in args.models):
        mnist_data = prepare_mnist(args)
        print(
            f"MNIST train={len(mnist_data['y_train'])} val={len(mnist_data['y_val'])} "
            f"test={len(mnist_data['y_test'])}  "
            f"constant_pixels={mnist_data['n_constant_pixels']} "
            f"low_variance_pixels={mnist_data['n_low_variance_pixels']}",
            flush=True,
        )
    if any(name in ("linear_svr", "rbf_svr") for name in args.models):
        boston_data = load_boston(args)
        print(
            f"Boston train={len(boston_data['y_train'])} val={len(boston_data['y_val'])} "
            f"test={len(boston_data['y_test'])}",
            flush=True,
        )
    for name in args.models:
        kind, runner = runners[name]
        data = mnist_data if kind == "mnist" else boston_data
        runner(args, data)


if __name__ == "__main__":
    main()

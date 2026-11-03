"""用岭回归和 Lasso 预测波士顿房价。

数据与 Kaggle「Boston House Prices」、旧版 sklearn load_boston 相同，共 506 条。
当前 sklearn 已移除 load_boston，因此直接读原始表。

求解器是自己写的：岭回归用正规方程，Lasso 用坐标下降。
sklearn 的 Ridge / Lasso 只用来核对系数，不参与最终预测。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
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
]
TARGET_NAME = "MEDV"
BOSTON_COLUMNS = FEATURE_NAMES + [TARGET_NAME]
BOSTON_URL = "http://lib.stat.cmu.edu/datasets/boston"

# 与 sklearn 的目标函数一致，所以两个 alpha 不在同一尺度上：
# 岭回归惩罚的是残差平方和，Lasso 惩罚的是残差平方和除以 2n。
RIDGE_ALPHAS = np.logspace(-2, 4, 49)
LASSO_ALPHAS = np.logspace(-3, 1, 49)
LASSO_MAX_ITER = 20000
LASSO_TOL = 1e-8
SPARSE_RMSE_TOL = 0.05


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
    }


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
                "下载波士顿房价数据失败。请将 CSV 放到 data/boston_house_prices.csv。"
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


def predict(features: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return features @ coef + intercept


def soft_threshold(value: float, threshold: float) -> float:
    magnitude = abs(value) - threshold
    if magnitude <= 0.0:
        return 0.0
    if value > 0.0:
        return magnitude
    return -magnitude


def fit_ols(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """普通最小二乘。截距不放进设计矩阵，先中心化再求最小二乘。"""
    x_mean = features.mean(axis=0)
    y_mean = float(target.mean())
    centered = features - x_mean
    coef, *_ = np.linalg.lstsq(centered, target - y_mean, rcond=None)
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """岭回归闭式解。惩罚只加在系数上，截距不惩罚。

    目标是 ||y - Xw - b||^2 + alpha ||w||^2，与 sklearn.linear_model.Ridge 相同。
    """
    x_mean = features.mean(axis=0)
    y_mean = float(target.mean())
    centered = features - x_mean
    width = centered.shape[1]
    gram = centered.T @ centered
    gram.flat[:: width + 1] += alpha
    coef = np.linalg.solve(gram, centered.T @ (target - y_mean))
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def fit_lasso(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    coef_init: np.ndarray | None = None,
    max_iter: int = LASSO_MAX_ITER,
    tol: float = LASSO_TOL,
) -> tuple[np.ndarray, float, int, bool]:
    """Lasso 的循环坐标下降。

    目标是 (1 / (2n)) ||y - Xw - b||^2 + alpha ||w||_1，与 sklearn Lasso 相同。
    特征先按训练集中心化，因此最优截距就是目标均值，循环里只更新系数。
    """
    n_samples, width = features.shape
    x_mean = features.mean(axis=0)
    y_mean = float(target.mean())
    centered = features - x_mean
    centered_target = target - y_mean
    column_scale = np.sum(centered * centered, axis=0) / n_samples

    if coef_init is None:
        coef = np.zeros(width, dtype=np.float64)
    else:
        coef = np.asarray(coef_init, dtype=np.float64).copy()

    converged = False
    used_iter = max_iter
    for iteration in range(1, max_iter + 1):
        max_delta = 0.0
        residual = centered_target - centered @ coef
        for index in range(width):
            scale = column_scale[index]
            if scale <= 1e-18:
                if coef[index] != 0.0:
                    residual += centered[:, index] * coef[index]
                    max_delta = max(max_delta, abs(coef[index]))
                    coef[index] = 0.0
                continue
            rho = float(np.dot(centered[:, index], residual) / n_samples + scale * coef[index])
            updated = soft_threshold(rho, alpha) / scale
            delta = updated - coef[index]
            if delta != 0.0:
                residual -= centered[:, index] * delta
                coef[index] = updated
                max_delta = max(max_delta, abs(delta))
        if max_delta < tol:
            converged = True
            used_iter = iteration
            break
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept, used_iter, converged


def coefficient_gap(
    coef: np.ndarray,
    intercept: float,
    reference_coef: np.ndarray,
    reference_intercept: float,
) -> float:
    coef_gap = float(np.max(np.abs(coef - reference_coef)))
    intercept_gap = abs(intercept - float(reference_intercept))
    return max(coef_gap, intercept_gap)


def named_coefficients(coef: np.ndarray, threshold: float = 1e-8) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, coef)
        if abs(value) > threshold
    }


def nonzero_count(coef: np.ndarray, threshold: float = 1e-8) -> int:
    return int(np.sum(np.abs(coef) > threshold))


def evaluate_model(
    coef: np.ndarray,
    intercept: float,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float]]:
    return {
        name: regression_metrics(target, predict(features, coef, intercept))
        for name, (features, target) in splits.items()
    }


def residual_notes(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    order = np.argsort(-np.abs(error))[:5]
    capped = y_true >= 49.999
    notes = {
        "largest_abs_errors": [
            {
                "true": float(y_true[index]),
                "pred": float(y_pred[index]),
                "error": float(error[index]),
            }
            for index in order
        ]
    }
    if np.any(capped):
        notes["capped_at_50"] = {
            "count": int(capped.sum()),
            "mae": float(np.mean(np.abs(error[capped]))),
            "mean_pred": float(np.mean(y_pred[capped])),
        }
    else:
        notes["capped_at_50"] = {"count": 0, "mae": None, "mean_pred": None}
    notes["n"] = int(y_true.shape[0])
    notes["within_5"] = int(np.sum(np.abs(error) <= 5))
    notes["pred_min"] = float(y_pred.min())
    notes["pred_max"] = float(y_pred.max())
    return notes


def select_by_validation(records: list[dict]) -> dict:
    """验证 RMSE 最小；差距小于 1e-6 时取更大的 alpha。"""
    best_rmse = min(record["val_rmse"] for record in records)
    candidates = [record for record in records if record["val_rmse"] <= best_rmse + 1e-6]
    return max(candidates, key=lambda record: record["alpha"])


def select_sparse(records: list[dict], best_rmse: float, tolerance: float) -> dict | None:
    """在验证 RMSE 不比最优差过 tolerance 的模型里，取非零系数最少的。"""
    eligible = [record for record in records if record["val_rmse"] <= best_rmse + tolerance]
    if not eligible:
        return None
    return min(eligible, key=lambda record: (record["n_nonzero"], -record["alpha"]))


def dataset_summary(table: np.ndarray) -> dict:
    features = table[:, :-1]
    target = table[:, -1]
    correlation = np.corrcoef(table, rowvar=False)
    feature_rows = []
    for index, name in enumerate(FEATURE_NAMES):
        column = features[:, index]
        feature_rows.append(
            {
                "name": name,
                "mean": float(column.mean()),
                "std": float(column.std()),
                "min": float(column.min()),
                "max": float(column.max()),
                "corr_medv": float(correlation[index, -1]),
            }
        )
    pairs = []
    width = len(FEATURE_NAMES)
    for left in range(width):
        for right in range(left + 1, width):
            value = float(correlation[left, right])
            if abs(value) >= 0.7:
                pairs.append(
                    {
                        "left": FEATURE_NAMES[left],
                        "right": FEATURE_NAMES[right],
                        "corr": value,
                    }
                )
    pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
    return {
        "n_samples": int(table.shape[0]),
        "target": {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "median": float(np.median(target)),
            "min": float(target.min()),
            "max": float(target.max()),
            "capped_at_50": int(np.sum(target >= 49.999)),
        },
        "features": feature_rows,
        "high_correlations": pairs,
        "correlation": correlation,
        "condition_number": float(np.linalg.cond(np.corrcoef(features, rowvar=False))),
    }


def ridge_path(features: np.ndarray, target: np.ndarray, alphas: np.ndarray) -> list[dict]:
    records = []
    for alpha in alphas:
        coef, intercept = fit_ridge(features, target, float(alpha))
        records.append(
            {
                "alpha": float(alpha),
                "coef": coef,
                "intercept": intercept,
                "n_nonzero": nonzero_count(coef),
            }
        )
    return records


def lasso_path(features: np.ndarray, target: np.ndarray, alphas: np.ndarray) -> list[dict]:
    """从大 alpha 暖启动到小 alpha，零系数会逐个回来。"""
    order = np.argsort(alphas)[::-1]
    coef = np.zeros(features.shape[1], dtype=np.float64)
    slotted: list[dict | None] = [None] * len(alphas)
    for index in order:
        coef, intercept, used_iter, converged = fit_lasso(
            features, target, float(alphas[index]), coef_init=coef
        )
        slotted[index] = {
            "alpha": float(alphas[index]),
            "coef": coef.copy(),
            "intercept": intercept,
            "n_nonzero": nonzero_count(coef),
            "n_iter": used_iter,
            "converged": converged,
        }
    return [record for record in slotted if record is not None]


def attach_split_scores(
    records: list[dict],
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for record in records:
        scores = evaluate_model(record["coef"], record["intercept"], splits)
        record["train"] = scores["train"]
        record["val"] = scores["val"]
        record["train_rmse"] = scores["train"]["rmse"]
        record["val_rmse"] = scores["val"]["rmse"]


def sklearn_reference_gap(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    kind: str,
    coef: np.ndarray,
    intercept: float,
) -> float:
    if kind == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif kind == "lasso":
        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=LASSO_MAX_ITER,
            tol=LASSO_TOL,
        )
    else:
        model = LinearRegression(fit_intercept=True)
    model.fit(features, target)
    return coefficient_gap(coef, intercept, model.coef_, model.intercept_)


def describe_solution(record: dict, splits, test_features, test_target) -> dict:
    scores = evaluate_model(record["coef"], record["intercept"], splits)
    test_pred = predict(test_features, record["coef"], record["intercept"])
    scores["test"] = regression_metrics(test_target, test_pred)
    kept = named_coefficients(record["coef"])
    dropped = [name for name in FEATURE_NAMES if name not in kept]
    return {
        "alpha": record.get("alpha"),
        "intercept": record["intercept"],
        "n_nonzero": nonzero_count(record["coef"]),
        "coefficients": {name: float(value) for name, value in zip(FEATURE_NAMES, record["coef"])},
        "dropped": dropped,
        "train": scores["train"],
        "val": scores["val"],
        "test": scores["test"],
        "n_iter": record.get("n_iter"),
        "converged": record.get("converged"),
        "residuals": residual_notes(test_target, test_pred),
    }


def save_correlation(correlation: np.ndarray, path: Path) -> None:
    labels = FEATURE_NAMES + [TARGET_NAME]
    fig, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    for row in range(correlation.shape[0]):
        for col in range(correlation.shape[1]):
            axis.text(col, row, f"{correlation[row, col]:.2f}", ha="center", va="center", fontsize=6.5)
    axis.set_title("Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_alpha_curves(ridge_records: list[dict], lasso_records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = (
        (axes[0], ridge_records, "Ridge"),
        (axes[1], lasso_records, "Lasso"),
    )
    for axis, records, title in panels:
        alphas = [record["alpha"] for record in records]
        axis.plot(alphas, [record["train_rmse"] for record in records], label="Train RMSE")
        axis.plot(alphas, [record["val_rmse"] for record in records], label="Val RMSE")
        chosen = select_by_validation(records)
        axis.axvline(chosen["alpha"], color="black", linewidth=1, linestyle="--", label="Selected")
        axis.set_xscale("log")
        axis.set_xlabel("alpha")
        axis.set_ylabel("RMSE")
        axis.set_title(title)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_coefficient_paths(ridge_records: list[dict], lasso_records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(FEATURE_NAMES)))
    panels = ((axes[0], ridge_records, "Ridge"), (axes[1], lasso_records, "Lasso"))
    for axis, records, title in panels:
        alphas = np.array([record["alpha"] for record in records])
        coefs = np.vstack([record["coef"] for record in records])
        for index, name in enumerate(FEATURE_NAMES):
            axis.plot(alphas, coefs[:, index], color=colors[index], label=name, linewidth=1.4)
        chosen = select_by_validation(records)
        axis.axvline(chosen["alpha"], color="black", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("alpha")
        axis.set_title(title)
        axis.axhline(0, color="gray", linewidth=0.6)
    axes[0].set_ylabel("Standardized coefficient")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_coefficient_bars(solutions: list[tuple[str, np.ndarray]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    positions = np.arange(len(FEATURE_NAMES))
    height = 0.8 / len(solutions)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    for index, (name, coef) in enumerate(solutions):
        offset = (index - (len(solutions) - 1) / 2) * height
        axis.barh(positions + offset, coef, height=height, label=name, color=colors[index % len(colors)])
    axis.set_yticks(positions)
    axis.set_yticklabels(FEATURE_NAMES)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Standardized coefficient")
    axis.legend(frameon=False)
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pred_residual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residual = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    axes[0].scatter(y_true, y_pred, s=18, alpha=0.75, color="steelblue")
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


def save_predictions(panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, y_true, y_pred) in zip(axes, panels):
        axis.scatter(y_true, y_pred, s=16, alpha=0.75, color="steelblue")
        low = float(min(y_true.min(), y_pred.min()))
        high = float(max(y_true.max(), y_pred.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        metrics = regression_metrics(y_true, y_pred)
        axis.set_title(f"{title}\nRMSE {metrics['rmse']:.3f}")
        axis.set_xlabel("True MEDV")
        axis.set_ylabel("Predicted MEDV")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slim_path(records: list[dict]) -> list[dict]:
    slim = []
    for record in records:
        item = {
            "alpha": record["alpha"],
            "train_rmse": record["train_rmse"],
            "val_rmse": record["val_rmse"],
            "n_nonzero": record["n_nonzero"],
            "coefficients": {
                name: float(value) for name, value in zip(FEATURE_NAMES, record["coef"])
            },
        }
        if "n_iter" in record:
            item["n_iter"] = record["n_iter"]
            item["converged"] = record["converged"]
        slim.append(item)
    return slim


def run(args: argparse.Namespace) -> None:
    table = ensure_boston_csv(Path(args.data_root) / "boston_house_prices.csv")
    summary = dataset_summary(table)
    features = table[:, :-1]
    target = table[:, -1]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=args.seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=args.val_size, random_state=args.seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    split_xy = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
    }

    ols_coef, ols_intercept = fit_ols(x_train, y_train)
    ols_gap = sklearn_reference_gap(x_train, y_train, 0.0, "ols", ols_coef, ols_intercept)
    ols_record = {
        "alpha": 0.0,
        "coef": ols_coef,
        "intercept": ols_intercept,
        "n_nonzero": nonzero_count(ols_coef),
    }

    ridge_records = ridge_path(x_train, y_train, RIDGE_ALPHAS)
    lasso_records = lasso_path(x_train, y_train, LASSO_ALPHAS)
    attach_split_scores(ridge_records, split_xy)
    attach_split_scores(lasso_records, split_xy)
    attach_split_scores([ols_record], split_xy)

    ridge_best = select_by_validation(ridge_records)
    lasso_best = select_by_validation(lasso_records)
    lasso_sparse = select_sparse(lasso_records, lasso_best["val_rmse"], SPARSE_RMSE_TOL)

    ridge_gap = sklearn_reference_gap(
        x_train, y_train, ridge_best["alpha"], "ridge", ridge_best["coef"], ridge_best["intercept"]
    )
    lasso_gap = sklearn_reference_gap(
        x_train, y_train, lasso_best["alpha"], "lasso", lasso_best["coef"], lasso_best["intercept"]
    )
    if lasso_sparse is not None:
        sparse_gap = sklearn_reference_gap(
            x_train,
            y_train,
            lasso_sparse["alpha"],
            "lasso",
            lasso_sparse["coef"],
            lasso_sparse["intercept"],
        )
    else:
        sparse_gap = None

    failed = [record for record in lasso_records if not record["converged"]]
    if failed:
        print(f"有 {len(failed)} 个 Lasso alpha 在 {LASSO_MAX_ITER} 次迭代内未收敛。")

    null_scores = {
        name: regression_metrics(values, np.full_like(values, y_train.mean()))
        for name, (_, values) in {**split_xy, "test": (x_test, y_test)}.items()
    }
    solutions = {
        "ols": describe_solution(ols_record, split_xy, x_test, y_test),
        "ridge": describe_solution(ridge_best, split_xy, x_test, y_test),
        "lasso": describe_solution(lasso_best, split_xy, x_test, y_test),
    }
    if lasso_sparse is not None and lasso_sparse["alpha"] != lasso_best["alpha"]:
        solutions["lasso_sparse"] = describe_solution(lasso_sparse, split_xy, x_test, y_test)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_correlation(summary["correlation"], out_dir / "correlation.png")
    save_alpha_curves(ridge_records, lasso_records, out_dir / "alpha_curves.png")
    save_coefficient_paths(ridge_records, lasso_records, out_dir / "coefficient_paths.png")

    bar_solutions = [
        ("OLS", ols_record["coef"]),
        (f"Ridge α={ridge_best['alpha']:.4g}", ridge_best["coef"]),
        (f"Lasso α={lasso_best['alpha']:.4g}", lasso_best["coef"]),
    ]
    pred_panels = [
        ("ols", y_test, predict(x_test, ols_record["coef"], ols_record["intercept"])),
        ("ridge", y_test, predict(x_test, ridge_best["coef"], ridge_best["intercept"])),
        ("lasso", y_test, predict(x_test, lasso_best["coef"], lasso_best["intercept"])),
    ]
    if "lasso_sparse" in solutions:
        bar_solutions.append((f"Lasso sparse α={lasso_sparse['alpha']:.4g}", lasso_sparse["coef"]))
        pred_panels.append(
            (
                "lasso_sparse",
                y_test,
                predict(x_test, lasso_sparse["coef"], lasso_sparse["intercept"]),
            )
        )
    display_names = {
        "ols": "OLS",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "lasso_sparse": "Lasso sparse",
    }
    save_coefficient_bars(bar_solutions, out_dir / "coefficients.png")
    save_predictions(
        [(display_names[title], y_true, y_pred) for title, y_true, y_pred in pred_panels],
        out_dir / "pred_vs_true.png",
    )
    for title, y_true, y_pred in pred_panels:
        save_pred_residual(y_true, y_pred, out_dir / f"{title}_pred.png")

    payload = {
        "task": "boston_ridge_lasso",
        "split": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
        },
        "standardize": "StandardScaler fit on train only",
        "target": summary["target"],
        "features": summary["features"],
        "high_correlations": summary["high_correlations"],
        "condition_number": summary["condition_number"],
        "null_model": null_scores,
        "sklearn_coefficient_gap": {
            "ols": ols_gap,
            "ridge": ridge_gap,
            "lasso": lasso_gap,
            "lasso_sparse": sparse_gap,
        },
        "solutions": solutions,
        "ridge_path": slim_path(ridge_records),
        "lasso_path": slim_path(lasso_records),
        "lasso_unconverged": len(failed),
        "environment": environment(),
    }
    dump_json(out_dir / "metrics.json", payload)

    print(
        f"split train/val/test = {len(y_train)}/{len(y_val)}/{len(y_test)}  "
        f"cond={summary['condition_number']:.2f}"
    )
    print(
        f"sklearn gap  ols={ols_gap:.3e}  ridge={ridge_gap:.3e}  lasso={lasso_gap:.3e}"
    )
    for name, solution in solutions.items():
        test = solution["test"]
        print(
            f"{name:12s}  alpha={solution['alpha']}  nonzero={solution['n_nonzero']:2d}  "
            f"val_rmse={solution['val']['rmse']:.4f}  "
            f"test_rmse={test['rmse']:.4f}  test_mae={test['mae']:.4f}  test_r2={test['r2']:.4f}"
        )
        if solution["dropped"]:
            print(f"  dropped: {', '.join(solution['dropped'])}")
    print(f"结果已写入 {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge and Lasso regression on Boston housing")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2, help="从训练余量中划给验证集的比例")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

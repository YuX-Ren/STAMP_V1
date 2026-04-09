from __future__ import annotations

import csv
import json
import random
import zlib
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from .common import canonicalize_smiles, iterable_with_progress
from .representations import UnsupportedRepresentation, build_representation


def _safe_float(v: Sequence[float], fn, default: float = float("nan")) -> float:
    if not v:
        return default
    return float(fn(v))


def _regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
        }

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))

    sst = float(np.sum(np.square(yt - float(np.mean(yt)))))
    if sst <= 0:
        r2 = float("nan")
    else:
        sse = float(np.sum(np.square(err)))
        r2 = 1.0 - (sse / sst)

    if len(yt) < 2:
        pearson_r = float("nan")
    else:
        corr = np.corrcoef(yt, yp)
        pearson_r = float(corr[0, 1]) if corr.shape == (2, 2) else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def _average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true or not y_score or len(y_true) != len(y_score):
        return float("nan")
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    n_pos = sum(1 for _, y in pairs if int(y) == 1)
    if n_pos <= 0:
        return float("nan")

    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for _, y in pairs:
        if int(y) == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / max(1, tp + fp)
        if int(y) == 1:
            ap += precision * max(0.0, recall - prev_recall)
            prev_recall = recall
    return float(ap)


def _roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true or not y_score or len(y_true) != len(y_score):
        return float("nan")
    ys = [int(y) for y in y_true]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        v = y_score[order[i]]
        while j + 1 < len(order) and y_score[order[j + 1]] == v:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = 0.0
    for idx, y in enumerate(ys):
        if y == 1:
            sum_pos_ranks += ranks[idx]
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _classification_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    if not y_true or not y_score or len(y_true) != len(y_score):
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
        }
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)

    prob = 1.0 / (1.0 + np.exp(-ys))
    yhat = (prob >= 0.5).astype(int)

    acc = float(np.mean(yhat == yt))
    tp = int(np.sum((yhat == 1) & (yt == 1)))
    fp = int(np.sum((yhat == 1) & (yt == 0)))
    fn = int(np.sum((yhat == 0) & (yt == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    roc_auc = _roc_auc(list(yt), list(prob))
    pr_auc = _average_precision(list(yt), list(prob))
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": acc,
        "f1": float(f1),
    }


def _token_hash(token: str, n_features: int) -> int:
    return int(zlib.crc32(token.encode("utf-8")) % n_features)


def _featurize_smiles(rep, smiles: str, n_features: int) -> Optional[np.ndarray]:
    try:
        text = rep.encode(smiles)
        toks = rep.tokenize(text)
    except Exception:
        return None
    if not toks:
        return None

    x = np.zeros(n_features, dtype=np.float64)
    for tok in toks:
        x[_token_hash(tok, n_features)] += 1.0

    denom = float(np.sum(x))
    if denom > 0:
        x /= denom
    return x


def _load_property_rows(
    input_path: str | Path,
    smiles_column: str,
    target_column: str,
    task_type: Literal["regression", "classification"],
    max_mols: Optional[int],
    deduplicate: bool,
    show_progress: bool,
) -> list[Tuple[str, float]]:
    path = Path(input_path)
    rows: list[Tuple[str, float]] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        iterator = iterable_with_progress(reader, show_progress=show_progress, desc="Load property rows")
        for row in iterator:
            smi_raw = (row.get(smiles_column) or "").strip()
            y_raw = (row.get(target_column) or "").strip()
            if not smi_raw or not y_raw:
                continue
            can = canonicalize_smiles(smi_raw, require_single_component=True)
            if can is None:
                continue
            if deduplicate and can in seen:
                continue
            try:
                y = float(y_raw)
            except Exception:
                continue
            if not np.isfinite(y):
                continue
            if task_type == "classification":
                if y in (0.0, 1.0):
                    pass
                elif y == -1.0:
                    # MoleculeNet commonly uses -1 as missing labels in multi-task tables.
                    continue
                else:
                    continue

            if deduplicate:
                seen.add(can)
            rows.append((can, y))
            if max_mols is not None and len(rows) >= max_mols:
                break
    return rows


def _fit_hashed_ridge(
    train_rows: Sequence[Tuple[str, float]],
    rep,
    n_features: int,
    ridge_alpha: float,
) -> Tuple[np.ndarray, int]:
    d = n_features + 1  # +1 bias term
    xtx = np.zeros((d, d), dtype=np.float64)
    xty = np.zeros(d, dtype=np.float64)
    n_encoded = 0

    for smi, y in train_rows:
        x = _featurize_smiles(rep, smi, n_features=n_features)
        if x is None:
            continue
        xb = np.empty(d, dtype=np.float64)
        xb[:n_features] = x
        xb[-1] = 1.0
        xtx += np.outer(xb, xb)
        xty += xb * float(y)
        n_encoded += 1

    if n_encoded <= 0:
        raise RuntimeError("No encodable training rows for this representation.")

    reg = np.eye(d, dtype=np.float64) * max(0.0, float(ridge_alpha))
    reg[-1, -1] = 0.0  # do not regularize bias
    a = xtx + reg
    try:
        w = np.linalg.solve(a, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ xty
    return w, n_encoded


def _predict_rows(
    rows: Sequence[Tuple[str, float]],
    rep,
    w: np.ndarray,
    n_features: int,
) -> Tuple[list[float], list[float], int]:
    y_true: list[float] = []
    y_pred: list[float] = []
    n_encoded = 0
    w_main = w[:n_features]
    b = float(w[-1])

    for smi, y in rows:
        x = _featurize_smiles(rep, smi, n_features=n_features)
        if x is None:
            continue
        pred = float(np.dot(w_main, x) + b)
        y_true.append(float(y))
        y_pred.append(pred)
        n_encoded += 1
    return y_true, y_pred, n_encoded


def run_property_prediction_benchmark(
    input_path: str | Path,
    output_dir: str | Path,
    target_column: str,
    representations: Sequence[str],
    task_type: Literal["regression", "classification"] = "regression",
    smiles_column: str = "smiles",
    test_size: float = 0.2,
    seed: int = 42,
    max_mols: Optional[int] = None,
    n_features: int = 2048,
    ridge_alpha: float = 1e-3,
    deduplicate: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"`test_size` must be in (0, 1), got {test_size}.")
    if n_features <= 0:
        raise ValueError(f"`n_features` must be > 0, got {n_features}.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_property_rows(
        input_path=input_path,
        smiles_column=smiles_column,
        target_column=target_column,
        task_type=task_type,
        max_mols=max_mols,
        deduplicate=deduplicate,
        show_progress=show_progress,
    )
    if len(rows) < 2:
        raise ValueError(
            f"Need at least 2 valid rows after filtering, got {len(rows)} from {input_path}."
        )

    idx = list(range(len(rows)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_test = max(1, int(round(len(rows) * float(test_size))))
    n_test = min(n_test, len(rows) - 1)
    test_idx = set(idx[:n_test])
    train_rows = [rows[i] for i in idx if i not in test_idx]
    test_rows = [rows[i] for i in idx if i in test_idx]

    results: list[dict[str, Any]] = []
    iterator = iterable_with_progress(
        list(representations),
        show_progress=show_progress,
        total=len(representations),
        desc="Property prediction",
    )
    for rep_name in iterator:
        try:
            rep = build_representation(rep_name)
            if isinstance(rep, UnsupportedRepresentation):
                raise RuntimeError("Representation placeholder not implemented.")

            w, n_train_encoded = _fit_hashed_ridge(
                train_rows=train_rows,
                rep=rep,
                n_features=n_features,
                ridge_alpha=ridge_alpha,
            )
            y_true_train, y_pred_train, n_train_pred = _predict_rows(
                rows=train_rows,
                rep=rep,
                w=w,
                n_features=n_features,
            )
            y_true_test, y_pred_test, n_test_pred = _predict_rows(
                rows=test_rows,
                rep=rep,
                w=w,
                n_features=n_features,
            )

            row = {
                "repr": rep_name,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_train_encoded": n_train_encoded,
                "n_train_pred": n_train_pred,
                "n_test_pred": n_test_pred,
                "train_metrics": (
                    _regression_metrics(y_true_train, y_pred_train)
                    if task_type == "regression"
                    else _classification_metrics([int(y) for y in y_true_train], y_pred_train)
                ),
                "test_metrics": (
                    _regression_metrics(y_true_test, y_pred_test)
                    if task_type == "regression"
                    else _classification_metrics([int(y) for y in y_true_test], y_pred_test)
                ),
            }
            if task_type == "regression":
                row["mae"] = float(row["test_metrics"]["mae"])
                row["rmse"] = float(row["test_metrics"]["rmse"])
                row["r2"] = float(row["test_metrics"]["r2"])
                row["pearson_r"] = float(row["test_metrics"]["pearson_r"])
            else:
                row["roc_auc"] = float(row["test_metrics"]["roc_auc"])
                row["pr_auc"] = float(row["test_metrics"]["pr_auc"])
                row["accuracy"] = float(row["test_metrics"]["accuracy"])
                row["f1"] = float(row["test_metrics"]["f1"])
            results.append(row)
        except Exception as exc:
            results.append(
                {
                    "repr": rep_name,
                    "n_train": len(train_rows),
                    "n_test": len(test_rows),
                    "n_train_encoded": 0,
                    "n_train_pred": 0,
                    "n_test_pred": 0,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "r2": float("nan"),
                    "pearson_r": float("nan"),
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "accuracy": float("nan"),
                    "f1": float("nan"),
                    "train_metrics": {},
                    "test_metrics": {},
                    "error": str(exc),
                }
            )

    summary: Dict[str, Any] = {
        "input_path": str(input_path),
        "output_dir": str(out),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "task_type": task_type,
        "test_size": float(test_size),
        "seed": int(seed),
        "max_mols": max_mols,
        "n_features": int(n_features),
        "ridge_alpha": float(ridge_alpha),
        "deduplicate": bool(deduplicate),
        "n_total_rows": len(rows),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "representations": list(representations),
        "results": results,
        "best_by_rmse": None,
        "best_by_roc_auc": None,
    }

    if task_type == "regression":
        ok_rows = [r for r in results if "error" not in r and np.isfinite(float(r.get("rmse", float("nan"))))]
        if ok_rows:
            best = min(ok_rows, key=lambda r: float(r["rmse"]))
            summary["best_by_rmse"] = {
                "repr": best["repr"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "r2": float(best["r2"]),
                "pearson_r": float(best["pearson_r"]),
            }
    else:
        ok_rows = [r for r in results if "error" not in r and np.isfinite(float(r.get("roc_auc", float("nan"))))]
        if ok_rows:
            best = max(ok_rows, key=lambda r: float(r["roc_auc"]))
            summary["best_by_roc_auc"] = {
                "repr": best["repr"],
                "roc_auc": float(best["roc_auc"]),
                "pr_auc": float(best["pr_auc"]),
                "accuracy": float(best["accuracy"]),
                "f1": float(best["f1"]),
            }

    (out / "property_prediction_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    field_order = [
        "repr",
        "n_train",
        "n_test",
        "n_train_encoded",
        "n_train_pred",
        "n_test_pred",
        "mae",
        "rmse",
        "r2",
        "pearson_r",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "error",
    ]
    with (out / "property_prediction_metrics_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in field_order})

    return summary

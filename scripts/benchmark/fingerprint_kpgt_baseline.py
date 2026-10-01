#!/usr/bin/env python3
"""Fingerprint-only baseline on KPGT MoleculeNet splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.benchmark.finetune_cams_decoder_pp import (
    KPGT_TASK_ALIASES,
    KPGT_TASK_TYPES,
    MULTITASK_SETS,
    _compute_metric,
    _load_kpgt_split_task,
)


def _fingerprint(smiles: str, *, fp_type: str, n_bits: int, radius: int) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.float32)
    if fp_type == "rdk":
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=n_bits)
    elif fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        raise ValueError(f"Unsupported fp_type={fp_type}")
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _rows_for_task(d: dict, task: str, is_class: bool) -> Tuple[list, list, list, int, bool]:
    task_key = KPGT_TASK_ALIASES.get(task, task)
    is_multi = is_class and task_key in MULTITASK_SETS and "train_multi" in d
    if is_multi:
        rows_tr = d["train_multi"]
        rows_va = d["valid_multi"]
        rows_te = d["test_multi"]
        n_tasks = max((len(r[1]) for r in rows_tr), default=1)
    else:
        rows_tr = d["train"]
        rows_va = d["valid"]
        rows_te = d["test"]
        n_tasks = 1
    return rows_tr, rows_va, rows_te, n_tasks, is_multi


def _featurize_rows(rows: Sequence, n_tasks: int, is_multi: bool, *, fp_type: str, n_bits: int, radius: int):
    fps: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    failed = 0
    for item in rows:
        fp = _fingerprint(item[0], fp_type=fp_type, n_bits=n_bits, radius=radius)
        if fp is None:
            failed += 1
            continue
        y = np.full((n_tasks,), np.nan, dtype=np.float32)
        if is_multi:
            for j, v in enumerate(item[1][:n_tasks]):
                y[j] = float(v)
        else:
            y[0] = float(item[1])
        fps.append(fp)
        labels.append(y)
    if not fps:
        return np.empty((0, n_bits), dtype=np.float32), np.empty((0, n_tasks), dtype=np.float32), failed
    return np.stack(fps).astype(np.float32), np.stack(labels).astype(np.float32), failed


class _FpDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class FingerprintMLP(nn.Module):
    def __init__(self, n_bits: int, n_tasks: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _eval(model: nn.Module, loader: DataLoader, device: torch.device, is_class: bool, y_std: float):
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if is_class:
                preds.append(torch.sigmoid(logits).float().cpu().numpy())
            else:
                preds.append(logits.float().cpu().numpy())
            labels.append(y.numpy())
    return _compute_metric(np.concatenate(preds, 0), np.concatenate(labels, 0), is_class, y_std=y_std)


def _train_task(
    task: str,
    data: dict,
    *,
    split_name: str,
    fp_type: str,
    n_bits: int,
    radius: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    num_workers: int,
    device: torch.device,
) -> Dict:
    task_key = KPGT_TASK_ALIASES.get(task, task)
    is_class = KPGT_TASK_TYPES[task_key] == "classification"
    rows_tr, rows_va, rows_te, n_tasks, is_multi = _rows_for_task(data, task, is_class)
    x_tr, y_tr, fail_tr = _featurize_rows(rows_tr, n_tasks, is_multi, fp_type=fp_type, n_bits=n_bits, radius=radius)
    x_va, y_va, fail_va = _featurize_rows(rows_va, n_tasks, is_multi, fp_type=fp_type, n_bits=n_bits, radius=radius)
    x_te, y_te, fail_te = _featurize_rows(rows_te, n_tasks, is_multi, fp_type=fp_type, n_bits=n_bits, radius=radius)

    print(
        f"[{task}] split={split_name} fp={fp_type} type={'classification' if is_class else 'regression'} "
        f"multi={is_multi} n_tasks={n_tasks} train={len(x_tr)} valid={len(x_va)} test={len(x_te)} "
        f"failed={fail_tr}/{fail_va}/{fail_te}",
        flush=True,
    )

    pin_memory = device.type == "cuda"
    ld_tr = DataLoader(_FpDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    ld_va = DataLoader(_FpDataset(x_va, y_va), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    ld_te = DataLoader(_FpDataset(x_te, y_te), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = FingerprintMLP(n_bits, n_tasks, hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if is_class:
            mask = torch.isfinite(y) & ((y == 0.0) | (y == 1.0))
            if not mask.any():
                return logits.sum() * 0.0
            target = torch.where(mask, y, torch.zeros_like(y))
            per = bce(logits, target)
            return (per * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return F.mse_loss(logits, y)

    best_valid = -float("inf") if is_class else float("inf")
    best_test = float("nan")
    best_epoch = -1
    best_per = None
    y_std = float(data.get("y_std", 1.0))
    patience_left = patience
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in ld_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        valid_m, _valid_per = _eval(model, ld_va, device, is_class, y_std)
        test_m, test_per = _eval(model, ld_te, device, is_class, y_std)
        better = valid_m > best_valid if is_class else valid_m < best_valid
        if better:
            best_valid = valid_m
            best_test = test_m
            best_epoch = ep
            best_per = test_per
            patience_left = patience
        else:
            patience_left -= 1
        metric = "auc" if is_class else "rmse"
        print(
            f"[{task}] ep{ep:02d} train_loss={running / max(n_batches, 1):.4f} "
            f"valid_{metric}={valid_m:.4f} test_{metric}={test_m:.4f} "
            f"best={best_test:.4f}@ep{best_epoch}",
            flush=True,
        )
        if patience_left <= 0:
            print(f"[{task}] early stop at ep{ep}", flush=True)
            break

    return {
        "task": task,
        "split": split_name,
        "fp_type": fp_type,
        "fp_bits": n_bits,
        "radius": radius,
        "is_class": is_class,
        "is_multi": is_multi,
        "n_tasks": n_tasks,
        "n_train": int(x_tr.shape[0]),
        "n_valid": int(x_va.shape[0]),
        "n_test": int(x_te.shape[0]),
        "failed": {"train": fail_tr, "valid": fail_va, "test": fail_te},
        "best_valid": best_valid,
        "best_test": best_test,
        "best_epoch": best_epoch,
        "per_task_test": best_per,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/kpgt/datasets")
    ap.add_argument("--tasks", nargs="+", default=["toxcast"])
    ap.add_argument("--splits", nargs="+", default=["scaffold-0"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fp-type", choices=["rdk", "morgan"], default="rdk")
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--morgan-radius", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    results: Dict[str, Dict] = {}
    for split in args.splits:
        split_results: Dict[str, Dict] = {}
        for task in args.tasks:
            data = _load_kpgt_split_task(task, args.data_dir, split)
            split_results[task] = _train_task(
                task,
                data,
                split_name=split,
                fp_type=args.fp_type,
                n_bits=args.fp_bits,
                radius=args.morgan_radius,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                device=device,
            )
        results[split] = split_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "hyperparams": vars(args) | {"output": str(args.output), "data_dir": str(args.data_dir)},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

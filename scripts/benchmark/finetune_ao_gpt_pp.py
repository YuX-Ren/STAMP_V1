"""Full fine-tune of an AO-GPT-PP checkpoint on MoleculeNet.

Adds a mean-pooled head (classification or regression, single- or multi-task)
on top of the pretrained TinyDecoderLM backbone and fine-tunes end-to-end per
task, reporting ROC-AUC (classification) or raw-scale RMSE (regression).

Usage:
    python scripts/benchmark/finetune_ao_gpt_pp.py \
        --checkpoint-dir results/ao_gpt_pp/cm31k_ring_small_v1 \
        --tasks bbbp bace tox21 clintox sider esol freesolv lipophilicity \
        --output results/ao_gpt_pp/cm31k_ring_small_v1/finetune.json \
        --epochs 30 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from rdkit import RDLogger

from stamp.benchmark.lm import TokenVocab, load_trained_model
from stamp.benchmark.roberta_property import load_moleculenet_task


MULTITASK_SETS = {"tox21", "clintox", "sider", "muv"}


class _MolPropertyDataset(Dataset):
    def __init__(self, ids_list: List[List[int]], labels: np.ndarray) -> None:
        self.ids = ids_list
        self.labels = labels  # shape [N, n_tasks], NaN = missing

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.labels[idx]


def _collate(pad_id: int):
    def _fn(batch):
        ids_list = [b[0] for b in batch]
        labels = np.stack([b[1] for b in batch], axis=0)
        m = max(len(s) for s in ids_list)
        arr = np.full((len(batch), m), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), m), dtype=np.float32)
        for i, s in enumerate(ids_list):
            arr[i, : len(s)] = s
            mask[i, : len(s)] = 1.0
        return (
            torch.from_numpy(arr),
            torch.from_numpy(mask),
            torch.from_numpy(labels.astype(np.float32)),
        )
    return _fn


class AOBackboneWithHead(nn.Module):
    """Wrap pretrained TinyDecoderLM with a mean-pool + MLP head."""

    def __init__(self, backbone, d_model: int, n_tasks: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_tasks),
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Capture last-block hidden states via a hook on the fly.
        captured: List[torch.Tensor] = []

        def hook(_m, _i, output):
            captured.append(output)

        h = self.backbone.blocks[-1].register_forward_hook(hook)
        try:
            _ = self.backbone(input_ids, force_causal=True)
        finally:
            h.remove()
        feat = captured[-1][:, 1:, :]  # drop prepended None prefix
        m = attn_mask.unsqueeze(-1)
        pooled = (feat * m).sum(1) / m.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)


def _tokenize_rows(
    rows, rep, vocab: TokenVocab, max_seq_len: int
) -> Tuple[List[List[int]], List[int]]:
    """Returns (ids_list, keep_indices)."""
    ids_list: List[List[int]] = []
    keep: List[int] = []
    for i, item in enumerate(rows):
        smi = item[0]
        try:
            text = rep.encode(smi)
            toks = rep.tokenize(text)
            ids = [vocab.bos_id] + vocab.encode_tokens(toks)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            if len(ids) < 2:
                continue
            ids_list.append(ids)
            keep.append(i)
        except Exception:
            continue
    return ids_list, keep


def _prepare_task_data(d: dict, task: str, tasks_info: dict, rep, vocab: TokenVocab, max_seq_len: int) -> dict:
    """Return tokenized train/valid/test + labels + task meta."""
    is_class = tasks_info[task] == "classification"
    is_multi = is_class and task in MULTITASK_SETS and "train_multi" in d

    if is_multi:
        rows_tr = d["train_multi"]
        rows_va = d["valid_multi"]
        rows_te = d["test_multi"]
        n_tasks = max((len(r[1]) for r in rows_tr), default=1)

        def _labels(rows, keep):
            arr = np.full((len(keep), n_tasks), np.nan, dtype=np.float32)
            for out_i, in_i in enumerate(keep):
                y_list = rows[in_i][1]
                for j, v in enumerate(y_list[:n_tasks]):
                    arr[out_i, j] = float(v)
            return arr
    else:
        rows_tr = d["train"]
        rows_va = d["valid"]
        rows_te = d["test"]
        n_tasks = 1

        def _labels(rows, keep):
            arr = np.empty((len(keep), 1), dtype=np.float32)
            for out_i, in_i in enumerate(keep):
                arr[out_i, 0] = float(rows[in_i][1])
            return arr

    ids_tr, k_tr = _tokenize_rows(rows_tr, rep, vocab, max_seq_len)
    ids_va, k_va = _tokenize_rows(rows_va, rep, vocab, max_seq_len)
    ids_te, k_te = _tokenize_rows(rows_te, rep, vocab, max_seq_len)
    Y_tr = _labels(rows_tr, k_tr)
    Y_va = _labels(rows_va, k_va)
    Y_te = _labels(rows_te, k_te)
    return {
        "is_class": is_class,
        "is_multi": is_multi,
        "n_tasks": n_tasks,
        "ids_tr": ids_tr, "Y_tr": Y_tr,
        "ids_va": ids_va, "Y_va": Y_va,
        "ids_te": ids_te, "Y_te": Y_te,
        "y_mean": float(d.get("y_mean", 0.0)),
        "y_std": float(d.get("y_std", 1.0)),
    }


def _compute_metric(preds: np.ndarray, labels: np.ndarray, is_class: bool, y_std: float = 1.0) -> Tuple[float, Optional[List[float]]]:
    if is_class:
        from sklearn.metrics import roc_auc_score
        n_tasks = labels.shape[1]
        per = []
        good = []
        for j in range(n_tasks):
            y = labels[:, j]
            p = preds[:, j]
            mask = np.isfinite(y) & ((y == 0.0) | (y == 1.0))
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
                per.append(float("nan"))
                continue
            try:
                auc = float(roc_auc_score(y[mask], p[mask]))
            except Exception:
                auc = float("nan")
            per.append(auc)
            if not np.isnan(auc):
                good.append(auc)
        return (float(np.mean(good)) if good else float("nan"), per)
    else:
        # Raw-scale RMSE (un-normalize z-scored predictions).
        rmse_norm = float(np.sqrt(np.mean((preds[:, 0] - labels[:, 0]) ** 2)))
        return rmse_norm * y_std, None


def _eval_loop(model, loader, device, is_class: bool, y_std: float = 1.0):
    model.eval()
    all_p = []
    all_y = []
    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(x, mask)
            if is_class:
                p = torch.sigmoid(logits).float().cpu().numpy()
            else:
                p = logits.float().cpu().numpy()
            all_p.append(p)
            all_y.append(y.numpy())
    return _compute_metric(np.concatenate(all_p, 0), np.concatenate(all_y, 0), is_class, y_std=y_std)


def _train_one_task(
    task: str,
    data: dict,
    checkpoint_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    max_seq_len: int,
    dropout: float,
    patience: int,
) -> Dict:
    # Re-load a fresh backbone per task (clean weights).
    backbone, vocab, _rep, _dev = load_trained_model(str(checkpoint_dir), device=str(device))
    d_model = backbone.cfg.d_model
    model = AOBackboneWithHead(backbone, d_model, data["n_tasks"], dropout=dropout).to(device)

    collate = _collate(vocab.pad_id)
    ds_tr = _MolPropertyDataset(data["ids_tr"], data["Y_tr"])
    ds_va = _MolPropertyDataset(data["ids_va"], data["Y_va"])
    ds_te = _MolPropertyDataset(data["ids_te"], data["Y_te"])
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    def _loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if data["is_class"]:
            # Mask NaN labels (missing in multi-task) before BCE.
            mask = torch.isfinite(y) & ((y == 0.0) | (y == 1.0))
            if not mask.any():
                return logits.sum() * 0.0
            per = bce(logits, y.clamp(0.0, 1.0))
            return (per * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            return F.mse_loss(logits, y)

    best_valid = -float("inf") if data["is_class"] else float("inf")
    best_test_m = float("nan")
    best_epoch = -1
    best_per = None
    patience_left = patience

    y_std = data["y_std"]
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_b = 0
        for x, mask, y in ld_tr:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = _loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_b += 1
        vm, vper = _eval_loop(model, ld_va, device, data["is_class"], y_std=y_std)
        tm, tper = _eval_loop(model, ld_te, device, data["is_class"], y_std=y_std)

        better = (vm > best_valid) if data["is_class"] else (vm < best_valid)
        if better:
            best_valid = vm
            best_test_m = tm
            best_epoch = ep
            best_per = tper
            patience_left = patience
        else:
            patience_left -= 1

        metric_name = "auc" if data["is_class"] else "rmse"
        print(
            f"[{task}] ep{ep:02d} train_loss={running/max(n_b,1):.4f} valid_{metric_name}={vm:.4f} test_{metric_name}={tm:.4f} best={best_test_m:.4f}@ep{best_epoch}",
            flush=True,
        )
        if patience_left <= 0:
            print(f"[{task}] early stop at ep{ep}", flush=True)
            break

    return {
        "task": task,
        "is_class": data["is_class"],
        "is_multi": data["is_multi"],
        "n_tasks": data["n_tasks"],
        "n_train": len(data["ids_tr"]),
        "n_valid": len(data["ids_va"]),
        "n_test": len(data["ids_te"]),
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_test": best_test_m,
        "per_task_test": best_per,
        "y_std": y_std,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["bbbp", "bace", "tox21", "clintox", "sider", "esol", "freesolv", "lipophilicity"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    tasks_info = {
        "bbbp": "classification",
        "bace": "classification",
        "hiv": "classification",
        "tox21": "classification",
        "clintox": "classification",
        "sider": "classification",
        "muv": "classification",
        "esol": "regression",
        "freesolv": "regression",
        "lipophilicity": "regression",
    }

    # Build a throwaway tokenizer once to preprocess rows (all tasks share it).
    backbone0, vocab0, rep0, _ = load_trained_model(str(args.checkpoint_dir), device=str(device))
    # Force ring-aware encode to match pretraining.
    if hasattr(rep0, "codec") and hasattr(rep0.codec, "motif_tokenizer") and rep0.codec.motif_tokenizer is not None:
        tok = rep0.codec.motif_tokenizer
        tok.allow_ring_single_edges = True
        tok.blessed_rings = set(tok.vocab.keys())
    max_seq_len = args.max_seq_len or backbone0.cfg.max_seq_len
    print(f"[setup] d_model={backbone0.cfg.d_model} max_seq_len={max_seq_len}", flush=True)
    del backbone0  # free; each task reloads its own copy

    results = {}
    for task in args.tasks:
        if task not in tasks_info:
            print(f"[skip] unknown task {task}", flush=True)
            continue
        t0 = time.time()
        try:
            d = load_moleculenet_task(task, data_dir=str(ROOT / "data"))
        except Exception as e:
            print(f"[skip] {task}: load failed ({e})", flush=True)
            continue
        data = _prepare_task_data(d, task, tasks_info, rep0, vocab0, max_seq_len)
        print(f"[{task}] type={'classification' if data['is_class'] else 'regression'} "
              f"multi={data['is_multi']} n_tasks={data['n_tasks']} "
              f"train={len(data['ids_tr'])} valid={len(data['ids_va'])} test={len(data['ids_te'])}", flush=True)
        res = _train_one_task(
            task, data, args.checkpoint_dir, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            max_seq_len=max_seq_len, dropout=args.dropout, patience=args.patience,
        )
        res["elapsed_sec"] = time.time() - t0
        results[task] = res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint_dir),
        "hyperparams": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "dropout": args.dropout, "patience": args.patience},
        "tasks": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fine-tune ChemBERTa-2 PubChem10M on Tox21/SR-p53.

This is a narrow baseline for checking the ChemBERTa-2 paper's SR-p53 number
under the same local DeepChem scaffold split used by the STAMP RoBERTa runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from rdkit import RDLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true or len(y_true) != len(y_score):
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
        val = y_score[order[i]]
        while j + 1 < len(order) and y_score[order[j + 1]] == val:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i] for i, y in enumerate(ys) if y == 1)
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _load_tox21_srp53(data_dir: str | Path, split_seed: int) -> Dict[str, Any]:
    import deepchem as dc

    tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer=dc.feat.DummyFeaturizer(),
        splitter="scaffold",
        data_dir=str(data_dir),
        transformers=[],
    )
    target = "SR-p53"
    col_idx = list(tasks).index(target)

    def extract(ds) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        for i in range(len(ds)):
            w = getattr(ds, "w", None)
            if w is not None and float(w[i][col_idx]) <= 0.0:
                continue
            y = float(ds.y[i][col_idx])
            if y not in (0.0, 1.0) or not math.isfinite(y):
                continue
            can = canonicalize_smiles(str(ds.ids[i]), strip_salts=True)
            if can is None:
                continue
            rows.append((can, y))
        return rows

    train, valid, test = datasets
    return {
        "train": extract(train),
        "valid": extract(valid),
        "test": extract(test),
        "tasks": list(tasks),
        "target": target,
        "splitter": "scaffold",
        "split_seed": split_seed,
    }


class _ClsDataset(Dataset):
    def __init__(self, input_ids: Sequence[List[int]], attn: Sequence[List[int]], labels: Sequence[float]) -> None:
        self.samples = [(x, a, float(y)) for x, a, y in zip(input_ids, attn, labels) if len(x) >= 2]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _collate_cls(pad_id: int):
    def collate(batch):
        max_len = max(len(x) for x, _, _ in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        a_out = torch.zeros((bsz, max_len), dtype=torch.long)
        y_out = torch.zeros(bsz, dtype=torch.float)
        for i, (x, a, y) in enumerate(batch):
            x_out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            a_out[i, : len(a)] = torch.tensor(a, dtype=torch.long)
            y_out[i] = y
        return x_out, a_out, y_out

    return collate


class ChemBERTaClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = base_model
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out(x).squeeze(-1)


def _encode_rows(tokenizer, rows: List[Tuple[str, float]], max_len: int):
    smiles = [s for s, _ in rows]
    labels = [y for _, y in rows]
    enc = tokenizer(
        smiles,
        max_length=max_len,
        padding=False,
        truncation=True,
        return_attention_mask=True,
    )
    return enc["input_ids"], enc["attention_mask"], labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="seyonec/PubChem10M_SMILES_BPE_60k")
    ap.add_argument("--data-dir", default="/scratch/yuxuan.ren/data/moleculenet")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    _set_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    from transformers import AutoModel, AutoTokenizer

    data = _load_tox21_srp53(args.data_dir, args.split_seed)
    print(
        f"[chemberta2] rows train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])}",
        flush=True,
    )
    for split in ("train", "valid", "test"):
        ys = [y for _, y in data[split]]
        print(f"[chemberta2] {split} positives={int(sum(ys))}/{len(ys)} ({sum(ys)/max(len(ys),1):.4f})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=bool(args.local_files_only))
    base = AutoModel.from_pretrained(args.model_name, local_files_only=bool(args.local_files_only))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    train_ids, train_attn, train_y = _encode_rows(tok, data["train"], args.max_len)
    valid_ids, valid_attn, valid_y = _encode_rows(tok, data["valid"], args.max_len)
    test_ids, test_attn, test_y = _encode_rows(tok, data["test"], args.max_len)
    trunc = {
        "train": int(sum(len(x) >= args.max_len for x in train_ids)),
        "valid": int(sum(len(x) >= args.max_len for x in valid_ids)),
        "test": int(sum(len(x) >= args.max_len for x in test_ids)),
    }
    print(f"[chemberta2] tokenizer={tok.__class__.__name__} vocab={tok.vocab_size} trunc={trunc}", flush=True)

    collate = _collate_cls(pad_id)
    common = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "num_workers": max(0, args.num_workers),
        "pin_memory": dev.type == "cuda",
    }
    if args.num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 4
    dl_train = DataLoader(_ClsDataset(train_ids, train_attn, train_y), shuffle=True, **common)
    dl_valid = DataLoader(_ClsDataset(valid_ids, valid_attn, valid_y), shuffle=False, **common)
    dl_test = DataLoader(_ClsDataset(test_ids, test_attn, test_y), shuffle=False, **common)

    model = ChemBERTaClassifier(base, base.config.hidden_size, args.dropout).to(dev)
    opt_kwargs: Dict[str, Any] = {"lr": args.lr, "weight_decay": args.weight_decay}
    if dev.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except TypeError:
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    amp_enabled = dev.type == "cuda" and args.amp != "none"
    if args.amp == "bf16" or (args.amp == "auto" and dev.type == "cuda"):
        amp_dtype = torch.bfloat16
        scaler_enabled = False
    elif args.amp == "fp16":
        amp_dtype = torch.float16
        scaler_enabled = True
    else:
        amp_dtype = torch.float32
        scaler_enabled = False
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    loss_fn = nn.BCEWithLogitsLoss()

    def evaluate(loader):
        model.eval()
        y_true: List[float] = []
        y_pred: List[float] = []
        losses: List[float] = []
        with torch.no_grad():
            for x, a, y in loader:
                x = x.to(dev, non_blocking=True)
                a = a.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
                with ctx:
                    logits = model(x, a)
                    loss = loss_fn(logits, y)
                losses.append(float(loss.item()))
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return float(np.mean(losses)), _roc_auc([int(y) for y in y_true], y_pred), y_true, y_pred

    best_auc = -1.0
    best_epoch = 0
    best_state = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x, a, y in dl_train:
            x = x.to(dev, non_blocking=True)
            a = a.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with ctx:
                logits = model(x, a)
                loss = loss_fn(logits, y)
            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            train_losses.append(float(loss.item()))
        valid_loss, valid_auc, _, _ = evaluate(dl_valid)
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_roc_auc": valid_auc})
        print(
            f"[chemberta2] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_auc={valid_auc:.4f}",
            flush=True,
        )
        if not math.isnan(valid_auc) and valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)
    _, test_auc, test_y_true, test_y_pred = evaluate(dl_test)
    test_acc = float(np.mean([(p >= 0.5) == bool(int(y)) for y, p in zip(test_y_true, test_y_pred)]))
    print(f"[chemberta2] Test ROC-AUC={test_auc:.4f} Accuracy={test_acc:.4f} best_epoch={best_epoch}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "tox21_srp53",
        "model_name": args.model_name,
        "target": data["target"],
        "splitter": data["splitter"],
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
        "test_roc_auc": test_auc,
        "test_accuracy": test_acc,
        "best_valid_roc_auc": best_auc,
        "best_epoch": best_epoch,
        "n_train": len(train_y),
        "n_valid": len(valid_y),
        "n_test": len(test_y),
        "history": history,
        "args": vars(args),
    }
    (out / "finetune_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state": best_state, "args": vars(args)}, out / "finetuned_model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fine-tune a pretrained STAMP-MolFormer checkpoint on MoleculeNet tasks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.lm import TokenVocab, _resolve_amp_mode, _resolve_torch_device, _set_seed  # noqa: E402
from stamp.benchmark.molformer_stamp import (  # noqa: E402
    Lamb,
    MolFormerEncoder,
    MolFormerSTAMPConfig,
    build_stamp_molformer_vocab,
    ids_from_stamp_tokens,
    load_molformer_checkpoint,
    smiles_vocab_tokens_from_stamp_vocabs,
)
from stamp.benchmark.multiscale_stamp import MultiScaleSTAMPEncoder  # noqa: E402
from stamp.benchmark.representations import STAMPRepresentation  # noqa: E402

RDLogger.DisableLog("rdApp.*")


TASKS: dict[str, dict[str, Any]] = {
    "bbbp": {"type": "classification", "file": "BBBP.csv", "smiles": "smiles", "targets": ["p_np"]},
    "hiv": {"type": "classification", "file": "HIV.csv", "smiles": "smiles", "targets": ["HIV_active"]},
    "bace": {"type": "classification", "file": "bace.csv", "smiles": "mol", "targets": ["Class"], "split": "Model"},
    "clintox": {"type": "classification", "file": "clintox.csv.gz", "smiles": "smiles", "targets": ["FDA_APPROVED", "CT_TOX"]},
    "tox21": {
        "type": "classification",
        "file": "tox21.csv.gz",
        "smiles": "smiles",
        "targets": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    },
    "sider": {
        "type": "classification",
        "file": "sider.csv.gz",
        "smiles": "smiles",
        "targets": None,
    },
    "esol": {
        "type": "regression",
        "file": "delaney-processed.csv",
        "smiles": "smiles",
        "targets": ["measured log solubility in mols per litre"],
    },
    "freesolv": {"type": "regression", "file": "SAMPL.csv", "smiles": "smiles", "targets": ["expt"]},
    "lipophilicity": {"type": "regression", "file": "Lipophilicity.csv", "smiles": "smiles", "targets": ["exp"]},
}

PAPER_TARGETS: dict[str, list[str]] = {
    "tox21": ["SR-p53"],
    "clintox": ["CT_TOX"],
    "sider": ["Hepatobiliary disorders"],
}


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _find_task_file(root: Path, name: str) -> Path:
    candidates = [root / name, root / name.lower(), root / "moleculenet" / name, root / "moleculenet" / name.lower()]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {name} under {root}")


def _float_or_nan(value: Any) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return float("nan")
    try:
        out = float(text)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # Use the largest component for the split key only; training still keeps
    # salts/counter-ions in the encoded input.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _scaffold_split(
    rows: Sequence[tuple[str, list[float], str | None]],
) -> tuple[list[tuple[str, list[float]]], list[tuple[str, list[float]]], list[tuple[str, list[float]]]]:
    buckets: dict[str, list[tuple[str, list[float]]]] = {}
    for smi, y, _ in rows:
        buckets.setdefault(_scaffold_key(smi), []).append((smi, y))
    groups = sorted(buckets.values(), key=lambda g: (-len(g), g[0][0]))
    n = sum(len(g) for g in groups)
    train_cut = int(0.8 * n)
    valid_cut = int(0.9 * n)
    train: list[tuple[str, list[float]]] = []
    valid: list[tuple[str, list[float]]] = []
    test: list[tuple[str, list[float]]] = []
    for group in groups:
        if len(train) + len(group) <= train_cut:
            train.extend(group)
        elif len(train) + len(valid) + len(group) <= valid_cut:
            valid.extend(group)
        else:
            test.extend(group)
    if not valid and test:
        valid, test = test[:1], test[1:]
    if not test and valid:
        test, valid = valid[:1], valid[1:]
    return train, valid, test


def load_task_rows(
    task: str,
    data_dir: Path,
    seed: int,
    split_mode: str = "random",
    target_mode: str = "all",
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    info = dict(TASKS[task])
    if target_mode == "paper" and task in PAPER_TARGETS:
        info["targets"] = list(PAPER_TARGETS[task])
    path = _find_task_file(data_dir, info["file"])
    rows: list[tuple[str, list[float], str | None]] = []
    with _open_csv(path) as f:
        reader = csv.DictReader(f)
        targets = info["targets"]
        if targets is None:
            targets = [c for c in (reader.fieldnames or []) if c != info["smiles"]]
            info["targets"] = targets
        for row in reader:
            smi = (row.get(info["smiles"]) or "").strip()
            can = canonicalize_smiles(smi, strip_salts=False)
            if can is None:
                continue
            y = [_float_or_nan(row.get(t)) for t in targets]
            if info["type"] == "classification":
                y = [v if v in (0.0, 1.0) else float("nan") for v in y]
                if not any(np.isfinite(v) for v in y):
                    continue
            else:
                if not np.isfinite(y[0]):
                    continue
            split = str(row.get(info.get("split", ""), "")).strip().lower() if info.get("split") else None
            rows.append((can, y, split))

    if split_mode == "predefined" and any(s in {"train", "valid", "validation", "test"} for _, _, s in rows if s):
        train = [(s, y) for s, y, sp in rows if sp == "train"]
        valid = [(s, y) for s, y, sp in rows if sp in {"valid", "validation"}]
        test = [(s, y) for s, y, sp in rows if sp == "test"]
        if train and valid and test:
            return {
                "task": task,
                "task_type": info["type"],
                "targets": list(info["targets"]),
                "train": train,
                "valid": valid,
                "test": test,
                "source": "local_csv_predefined_split_keep_salts",
            }

    if split_mode == "scaffold":
        train, valid, test = _scaffold_split(rows)
        return {
            "task": task,
            "task_type": info["type"],
            "targets": list(info["targets"]),
            "train": train,
            "valid": valid,
            "test": test,
            "source": "local_csv_murcko_scaffold_keep_salts",
        }

    rnd = random.Random(seed)
    simple = [(s, y) for s, y, _ in rows]
    rnd.shuffle(simple)
    n = len(simple)
    n_test = max(1, int(0.1 * n))
    n_valid = max(1, int(0.1 * n))
    test = simple[:n_test]
    valid = simple[n_test : n_test + n_valid]
    train = simple[n_test + n_valid :]
    return {
        "task": task,
        "task_type": info["type"],
        "targets": list(info["targets"]),
        "train": train,
        "valid": valid,
        "test": test,
        "source": "local_csv_random_keep_salts",
    }


class StampEncoderForPP:
    def __init__(
        self,
        *,
        stamp_vocab: str | None,
        multiscale_vocabs: Sequence[str],
        multiscale_scale_names: Sequence[str],
        force_ring_aware_encode: bool = False,
        strict_vocab: bool = True,
    ) -> None:
        self.multiscale = bool(multiscale_vocabs)
        self.force_ring_aware_encode = bool(force_ring_aware_encode)
        self.strict_vocab = bool(strict_vocab)
        self.vocab_paths = list(multiscale_vocabs) if self.multiscale else ([stamp_vocab] if stamp_vocab else [])
        if self.multiscale:
            names = list(multiscale_scale_names) if multiscale_scale_names else [f"s{i}" for i in range(len(multiscale_vocabs))]
            self.rep = MultiScaleSTAMPEncoder(
                multiscale_vocabs,
                scale_names=names,
                force_ring_aware_encode=self.force_ring_aware_encode,
                strict_vocab=self.strict_vocab,
            )
        else:
            if not stamp_vocab:
                raise ValueError("--stamp-vocab is required when --multiscale-vocabs is not set.")
            self.rep = STAMPRepresentation(
                vocab_path=stamp_vocab,
                force_ring_aware_encode=self.force_ring_aware_encode,
                strict_vocab=self.strict_vocab,
            )

    def tokens(self, smiles: str) -> list[str]:
        scale_texts = self._encode_scale_texts(smiles)
        toks: list[str] = []
        for i, text in enumerate(scale_texts):
            if i > 0:
                toks.append("[CONCAT]")
            toks.extend(str(text).split())
        return toks

    def _encode_scale_texts(self, smiles: str) -> list[str]:
        try:
            return self._encode_single_component(smiles)
        except Exception:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or len(Chem.GetMolFrags(mol)) <= 1:
                raise
            comps = [
                Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
                for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if frag is not None and frag.GetNumAtoms() > 0
            ]
            if not comps:
                raise
            encoded = [self._encode_single_component(comp) for comp in comps]
            n_scales = len(encoded[0])
            merged: list[str] = []
            for scale_idx in range(n_scales):
                scale_tokens: list[str] = []
                for comp_idx, comp_texts in enumerate(encoded):
                    if comp_idx > 0:
                        scale_tokens.append("[.]")
                    scale_tokens.extend(comp_texts[scale_idx].split())
                merged.append(" ".join(scale_tokens))
            return merged

    def _encode_single_component(self, smiles: str) -> list[str]:
        if self.multiscale:
            texts = self.rep.encode_scales(smiles)
            return [str(text) for text in texts]
        return [str(self.rep.encode(smiles))]

    def smiles_vocab_tokens(self) -> list[str]:
        return smiles_vocab_tokens_from_stamp_vocabs([p for p in self.vocab_paths if p])


def _extend_encoder_vocab(
    encoder: MolFormerEncoder,
    vocab: TokenVocab,
    extra_tokens: Sequence[str],
) -> tuple[TokenVocab, int]:
    added = [tok for tok in extra_tokens if tok and tok not in vocab.stoi]
    if not added:
        return vocab, 0
    old_n = len(vocab.itos)
    new_itos = list(vocab.itos) + added
    new_vocab = TokenVocab(stoi={tok: i for i, tok in enumerate(new_itos)}, itos=new_itos)

    old_emb = encoder.tok_emb
    new_emb = nn.Embedding(len(new_itos), old_emb.embedding_dim, padding_idx=encoder.pad_id)
    encoder._init_weights(new_emb)
    with torch.no_grad():
        new_emb.weight[:old_n].copy_(old_emb.weight)
    encoder.tok_emb = new_emb
    encoder.cfg.vocab_size = len(new_itos)
    return new_vocab, len(added)


def _encode_rows(
    rows: Sequence[tuple[str, list[float]]],
    encoder: StampEncoderForPP,
    vocab: TokenVocab,
    max_seq_len: int,
    fallback_oov_to_smiles: bool,
) -> tuple[list[list[int]], np.ndarray]:
    ids: list[list[int]] = []
    labels: list[list[float]] = []
    for smi, y in rows:
        try:
            toks = encoder.tokens(smi)
            seq = ids_from_stamp_tokens(
                toks,
                vocab,
                max_seq_len=max_seq_len,
                fallback_oov_to_smiles=fallback_oov_to_smiles,
            )
        except Exception:
            continue
        if len(seq) < 3:
            continue
        ids.append(seq)
        labels.append([float(v) for v in y])
    y_arr = np.asarray(labels, dtype=np.float32) if labels else np.empty((0, 1), dtype=np.float32)
    return ids, y_arr


class PPDataset(Dataset):
    def __init__(self, ids: Sequence[list[int]], labels: np.ndarray) -> None:
        self.ids = list(ids)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[list[int], np.ndarray]:
        return self.ids[idx], self.labels[idx]


def _collate(pad_id: int):
    def fn(batch: Sequence[tuple[list[int], np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x, _ in batch)
        rem = max_len % 8
        if rem:
            max_len += 8 - rem
        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        y = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return x, torch.from_numpy(y)

    return fn


class MolFormerPropertyModel(nn.Module):
    def __init__(self, encoder: MolFormerEncoder, n_tasks: int, dropout: float, pooling: str) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        d = encoder.cfg.d_model
        # IBM MolFormer finetuning uses two GELU MLP layers with residual skips.
        self.fc1 = nn.Linear(d, d)
        self.dropout1 = nn.Dropout(dropout)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(d, d)
        self.dropout2 = nn.Dropout(dropout)
        self.act2 = nn.GELU()
        self.final = nn.Linear(d, n_tasks)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids)
        if self.pooling == "cls":
            pooled = h[:, 0, :]
        else:
            mask = input_ids.ne(self.encoder.pad_id).unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        x = self.act1(self.dropout1(self.fc1(pooled)))
        x = x + pooled
        z = self.act2(self.dropout2(self.fc2(x)))
        return self.final(z + x)


def _metric(pred: np.ndarray, labels: np.ndarray, task_type: str, label_mean: float, label_std: float) -> tuple[float, list[float] | None]:
    if task_type == "classification":
        from sklearn.metrics import roc_auc_score

        per: list[float] = []
        good: list[float] = []
        for j in range(labels.shape[1]):
            y = labels[:, j]
            p = pred[:, j]
            mask = np.isfinite(y) & np.isfinite(p) & ((y == 0.0) | (y == 1.0))
            if int(mask.sum()) < 5 or len(np.unique(y[mask])) < 2:
                per.append(float("nan"))
                continue
            auc = float(roc_auc_score(y[mask], p[mask]))
            per.append(auc)
            if np.isfinite(auc):
                good.append(auc)
        return (float(np.mean(good)) if good else float("nan")), per
    pred_raw = pred[:, 0] * label_std + label_mean
    y_raw = labels[:, 0] * label_std + label_mean
    return float(np.sqrt(np.mean((pred_raw - y_raw) ** 2))), None


def _eval(model, loader, dev, task_type: str, label_mean: float, label_std: float, use_amp: bool, amp_dtype) -> tuple[float, list[float] | None]:
    model.eval()
    ps: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=dev.type == "cuda")
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else torch.no_grad()
            if not use_amp:
                from contextlib import nullcontext

                ctx = nullcontext()
            with ctx:
                logits = model(x)
            pred = torch.sigmoid(logits).float().cpu().numpy() if task_type == "classification" else logits.float().cpu().numpy()
            ps.append(pred)
            ys.append(y.numpy())
    return _metric(np.concatenate(ps, axis=0), np.concatenate(ys, axis=0), task_type, label_mean, label_std)


def _train_task(
    *,
    task: str,
    checkpoint: Path,
    data: dict[str, Any],
    stamp_encoder: StampEncoderForPP,
    output_dir: Path,
    max_seq_len: int,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    dropout: float,
    patience: int,
    pooling: str,
    amp: str,
    freeze_encoder: bool,
    fallback_oov_to_smiles: bool,
    optimizer_name: str,
    weight_decay: float,
) -> dict[str, Any]:
    dev = _resolve_torch_device(device)
    encoder, _, vocab, payload = load_molformer_checkpoint(checkpoint, map_location="cpu", load_mlm_head=False)
    added_smiles_vocab = 0
    if fallback_oov_to_smiles:
        vocab, added_smiles_vocab = _extend_encoder_vocab(encoder, vocab, stamp_encoder.smiles_vocab_tokens())
    encoder.to(dev)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    ids_tr, y_tr = _encode_rows(data["train"], stamp_encoder, vocab, max_seq_len, fallback_oov_to_smiles)
    ids_va, y_va = _encode_rows(data["valid"], stamp_encoder, vocab, max_seq_len, fallback_oov_to_smiles)
    ids_te, y_te = _encode_rows(data["test"], stamp_encoder, vocab, max_seq_len, fallback_oov_to_smiles)
    if not ids_tr or not ids_va or not ids_te:
        raise ValueError(f"{task}: empty encoded split train={len(ids_tr)} valid={len(ids_va)} test={len(ids_te)}")

    task_type = data["task_type"]
    label_mean = 0.0
    label_std = 1.0
    if task_type == "regression":
        finite = np.isfinite(y_tr[:, 0])
        label_mean = float(np.mean(y_tr[finite, 0]))
        label_std = float(np.std(y_tr[finite, 0]))
        if label_std < 1e-8:
            label_std = 1.0
        for arr in (y_tr, y_va, y_te):
            arr[:, 0] = (arr[:, 0] - label_mean) / label_std

    n_tasks = int(y_tr.shape[1])
    model = MolFormerPropertyModel(encoder, n_tasks=n_tasks, dropout=dropout, pooling=pooling).to(dev)
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(amp, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_scaler)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "lamb":
        opt = Lamb(opt_params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(opt_params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    collate = _collate(vocab.pad_id)
    pin = dev.type == "cuda"
    dl_tr = DataLoader(PPDataset(ids_tr, y_tr), batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=pin)
    dl_va = DataLoader(PPDataset(ids_va, y_va), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=pin)
    dl_te = DataLoader(PPDataset(ids_te, y_te), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=pin)

    best_valid = -float("inf") if task_type == "classification" else float("inf")
    best_state = None
    best_epoch = 0
    best_test = float("nan")
    best_per = None
    history: list[dict[str, Any]] = []
    patience_left = patience
    bce = nn.BCEWithLogitsLoss(reduction="none")
    t0 = time.time()

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if task_type == "classification":
            mask = torch.isfinite(y) & ((y == 0.0) | (y == 1.0))
            if not bool(mask.any()):
                return logits.sum() * 0.0
            target = torch.where(mask, y, torch.zeros_like(y)).clamp(0.0, 1.0)
            per = bce(logits, target)
            return (per * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        mask = torch.isfinite(y)
        return F.mse_loss(logits[mask], y[mask])

    for ep in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=pin)
            y = y.to(dev, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else torch.no_grad()
            if not use_amp:
                from contextlib import nullcontext

                ctx = nullcontext()
            with ctx:
                logits = model(x)
                loss = loss_fn(logits, y)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.item()))

        vm, _ = _eval(model, dl_va, dev, task_type, label_mean, label_std, use_amp, amp_dtype)
        tm, tper = _eval(model, dl_te, dev, task_type, label_mean, label_std, use_amp, amp_dtype)
        improved = (vm > best_valid) if task_type == "classification" else (vm < best_valid)
        if improved:
            best_valid = vm
            best_epoch = ep
            best_test = tm
            best_per = tper
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        metric = "auc" if task_type == "classification" else "rmse"
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            f"valid_{metric}": vm,
            f"test_{metric}": tm,
            "best_test": best_test,
            "best_epoch": best_epoch,
        }
        history.append(row)
        print(
            f"[{task}] ep={ep:03d} loss={row['train_loss']:.4f} "
            f"valid_{metric}={vm:.4f} test_{metric}={tm:.4f} best={best_test:.4f}@{best_epoch}",
            flush=True,
        )
        if patience_left <= 0:
            break

    task_out = output_dir / task
    task_out.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "task": task,
                "task_type": task_type,
                "targets": data["targets"],
                "checkpoint": str(checkpoint),
                "pretrain_global_step": int(payload.get("global_step", 0)),
                "label_mean": label_mean,
                "label_std": label_std,
                "max_seq_len": max_seq_len,
                "pooling": pooling,
                "fallback_oov_to_smiles": bool(fallback_oov_to_smiles),
                "added_smiles_vocab": int(added_smiles_vocab),
                "vocab": vocab.to_dict(),
            },
            task_out / "finetuned_model.pt",
        )
    result = {
        "task": task,
        "task_type": task_type,
        "targets": data["targets"],
        "n_tasks": n_tasks,
        "n_train": len(ids_tr),
        "n_valid": len(ids_va),
        "n_test": len(ids_te),
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_test": best_test,
        "per_task_test": best_per,
        "history": history,
        "label_mean": label_mean,
        "label_std": label_std,
        "elapsed_sec": time.time() - t0,
        "amp": amp_label,
        "fallback_oov_to_smiles": bool(fallback_oov_to_smiles),
        "added_smiles_vocab": int(added_smiles_vocab),
    }
    (task_out / "finetune_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_csv_arg(text: str | None) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("/scratch/yuxuan.ren/data/moleculenet"))
    p.add_argument("--tasks", default="bbbp,hiv,bace,clintox,tox21,sider,esol,freesolv,lipophilicity")
    p.add_argument("--stamp-vocab", default="")
    p.add_argument("--multiscale-vocabs", default="")
    p.add_argument("--multiscale-scale-names", default="")
    p.add_argument("--force-ring-aware-encode", action="store_true", help="Override vocab header and emit [Ring] closures.")
    p.add_argument(
        "--no-strict-vocab",
        action="store_false",
        dest="strict_vocab",
        help="Allow irreducible fragments outside the motif vocab instead of dropping those molecules.",
    )
    p.add_argument(
        "--fallback-oov-to-smiles",
        action="store_true",
        help="Map non-vocab STAMP motif tokens to regex SMILES tokens instead of <unk>; missing SMILES tokens are appended to the property head encoder.",
    )
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--pooling", choices=["avg", "cls"], default="avg")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--split-mode", choices=["scaffold", "random", "predefined"], default="scaffold")
    p.add_argument("--target-mode", choices=["all", "paper"], default="all")
    p.add_argument("--optimizer", choices=["lamb", "adamw"], default="lamb")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _set_seed(args.seed)
    multiscale_vocabs = _parse_csv_arg(args.multiscale_vocabs)
    scale_names = _parse_csv_arg(args.multiscale_scale_names)
    stamp_encoder = StampEncoderForPP(
        stamp_vocab=args.stamp_vocab or (multiscale_vocabs[0] if multiscale_vocabs else None),
        multiscale_vocabs=multiscale_vocabs,
        multiscale_scale_names=scale_names,
        force_ring_aware_encode=bool(args.force_ring_aware_encode),
        strict_vocab=bool(args.strict_vocab),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for task in _parse_csv_arg(args.tasks):
        t0 = time.time()
        try:
            data = load_task_rows(
                task,
                args.data_dir,
                seed=args.seed,
                split_mode=args.split_mode,
                target_mode=args.target_mode,
            )
            print(
                f"[{task}] type={data['task_type']} targets={len(data['targets'])} "
                f"rows train={len(data['train'])} valid={len(data['valid'])} test={len(data['test'])} "
                f"source={data.get('source', 'unknown')}",
                flush=True,
            )
            res = _train_task(
                task=task,
                checkpoint=args.checkpoint,
                data=data,
                stamp_encoder=stamp_encoder,
                output_dir=args.output_dir,
                max_seq_len=args.max_seq_len,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                dropout=args.dropout,
                patience=args.patience,
                pooling=args.pooling,
                amp=args.amp,
                freeze_encoder=bool(args.freeze_encoder),
                fallback_oov_to_smiles=bool(args.fallback_oov_to_smiles),
                optimizer_name=args.optimizer,
                weight_decay=float(args.weight_decay),
            )
            res["total_elapsed_sec"] = time.time() - t0
            res["source"] = data.get("source", "unknown")
            results[task] = res
        except Exception as exc:
            print(f"[{task}] failed: {exc}", flush=True)
            results[task] = {"error": str(exc)}
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "tasks": results,
        "hyperparams": {
            "max_seq_len": args.max_seq_len,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "patience": args.patience,
            "pooling": args.pooling,
            "freeze_encoder": args.freeze_encoder,
            "split_mode": args.split_mode,
            "target_mode": args.target_mode,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
        },
    }
    (args.output_dir / "molformer_stamp_finetune_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {args.output_dir / 'molformer_stamp_finetune_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

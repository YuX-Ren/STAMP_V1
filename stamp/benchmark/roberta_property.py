"""RoBERTa-based MLM pre-training and MoleculeNet fine-tuning for molecular representations.

Pipeline:
  1. Pre-train a RoBERTa encoder with masked-language-modelling on a corpus of
     SMILES (converted to each representation).
  2. Fine-tune the encoder on MoleculeNet tasks:
     - Classification (BBBP, HIV, Tox21, BACE, ClinTox, SIDER, MUV) — BCE loss, ROC-AUC metric
     - Regression (ESOL, FreeSolv, Lipophilicity) — MSE loss, RMSE/R² metrics

Architecture follows the reference paper:
  - 6 hidden layers, hidden_size 768, intermediate_size 1536, 12 attention heads
  - 15% token masking, AdamW, 20 pre-training epochs
  - Fine-tune with 80/10/10 split
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .common import canonicalize_smiles, iterable_with_progress
from .lm import (
    TokenVocab,
    _encode_sequences_with_rep,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
    build_dynamic_lm_collate_fn,
)
from .representations import (
    BaseRepresentation,
    RepresentationError,
    UnsupportedRepresentation,
    build_representation,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RoBERTaConfig:
    """Hyper-parameters for the RoBERTa encoder."""
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 512
    mask_prob: float = 0.15
    # training
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    pretrain_epochs: int = 20
    finetune_epochs: int = 20
    finetune_lr: float = 2e-5
    finetune_batch_size: int = 32
    seed: int = 42
    num_data_workers: int = 0
    prefetch_factor: int = 2
    amp: str = "auto"
    torch_compile: bool = False
    tf32: bool = True


# ---------------------------------------------------------------------------
# Masked-LM dataset
# ---------------------------------------------------------------------------

class MaskedLMDataset(Dataset):
    """Creates masked-LM samples: 15% of tokens are selected; of those 80% → [MASK],
    10% → random token, 10% → unchanged."""

    def __init__(
        self,
        sequences: Sequence[List[int]],
        vocab: TokenVocab,
        mask_prob: float = 0.15,
        max_seq_len: int = 512,
        seed: int = 42,
    ) -> None:
        self.samples: List[List[int]] = []
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.mask_id = vocab.stoi.get("<mask>", vocab.unk_id)
        self.rng = random.Random(seed)
        # vocab tokens eligible for random replacement (exclude specials)
        self._n_vocab = len(vocab.itos)

        for seq in sequences:
            if len(seq) < 3:  # at least bos + one token + eos
                continue
            self.samples.append(seq[:max_seq_len])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        seq = list(self.samples[idx])
        labels = [-100] * len(seq)

        for i in range(1, len(seq) - 1):  # skip bos/eos
            if self.rng.random() < self.mask_prob:
                labels[i] = seq[i]
                r = self.rng.random()
                if r < 0.8:
                    seq[i] = self.mask_id
                elif r < 0.9:
                    seq[i] = self.rng.randint(4, self._n_vocab - 1)  # random token (skip specials 0-3)
                # else: keep original

        return seq, labels


def build_mlm_collate_fn(pad_id: int):
    def _collate(batch: Sequence[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x, _ in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            n = len(x)
            x_out[i, :n] = torch.tensor(x, dtype=torch.long)
            y_out[i, :n] = torch.tensor(y, dtype=torch.long)
        return x_out, y_out
    return _collate


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[List[int]],
        labels: Sequence[float],
        max_seq_len: int = 512,
        pad_id: int = 0,
    ) -> None:
        self.samples: List[Tuple[List[int], float]] = []
        for seq, label in zip(sequences, labels):
            if len(seq) < 2:
                continue
            self.samples.append((seq[:max_seq_len], float(label)))
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], float]:
        return self.samples[idx]


def build_cls_collate_fn(pad_id: int):
    def _collate(batch: Sequence[Tuple[List[int], float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x, _ in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.zeros(bsz, dtype=torch.float)
        for i, (x, y) in enumerate(batch):
            n = len(x)
            x_out[i, :n] = torch.tensor(x, dtype=torch.long)
            y_out[i] = y
        return x_out, y_out
    return _collate


# ---------------------------------------------------------------------------
# RoBERTa model
# ---------------------------------------------------------------------------

class MolRoBERTa(nn.Module):
    """RoBERTa-style encoder for molecular representations."""

    def __init__(self, vocab_size: int, cfg: RoBERTaConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_ln = nn.LayerNorm(cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_cls: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pad_mask = (input_ids == self.pad_id)  # True where padded

        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_ln(x)
        x = self.emb_drop(x)

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)

        if return_cls:
            return x[:, 0, :]  # BOS token as [CLS]
        return x


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense(x))
        x = self.ln(x)
        return self.proj(x)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        x = self.dropout(cls_repr)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out(x)


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        x = self.dropout(cls_repr)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# Vocab builder (adds <mask> token)
# ---------------------------------------------------------------------------

def build_roberta_vocab(
    tokenized_sequences: Sequence[Sequence[str]],
    min_freq: int = 1,
) -> TokenVocab:
    """Build TokenVocab with an extra <mask> token for MLM."""
    from collections import Counter

    cnt = Counter()
    for seq in tokenized_sequences:
        cnt.update(seq)

    specials = [TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK, "<mask>"]
    vocab_list = list(specials)
    special_set = set(specials)
    for tok, freq in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and tok not in special_set:
            vocab_list.append(tok)

    stoi = {tok: i for i, tok in enumerate(vocab_list)}
    return TokenVocab(stoi=stoi, itos=vocab_list)


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain_roberta(
    train_smiles: Sequence[str],
    valid_smiles: Sequence[str],
    representation: str,
    output_dir: str | Path,
    cfg: Optional[RoBERTaConfig] = None,
    min_token_freq: int = 1,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
) -> Dict[str, Any]:
    """Pre-train a RoBERTa MLM on molecular representations."""
    rep = build_representation(representation)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot pre-train RoBERTa for unsupported representation: {representation}"
        )

    cfg = cfg or RoBERTaConfig()
    _set_seed(cfg.seed)
    dev = _resolve_torch_device(device)

    print(f"[pretrain-roberta] representation={representation}", flush=True)
    print(f"[pretrain-roberta] encoding train set...", flush=True)
    train_tok = _encode_sequences_with_rep(
        train_smiles, rep,
        show_progress=show_progress,
        desc=f"Encode train[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )
    print(f"[pretrain-roberta] encoding valid set...", flush=True)
    valid_tok = _encode_sequences_with_rep(
        valid_smiles, rep,
        show_progress=show_progress,
        desc=f"Encode valid[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )
    if not train_tok:
        raise ValueError("No train samples encoded successfully.")

    vocab = build_roberta_vocab(train_tok, min_freq=min_token_freq)
    mask_id = vocab.stoi["<mask>"]

    # Add BOS/EOS wrapping
    train_ids = [[vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id] for toks in train_tok]
    valid_ids = [[vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id] for toks in valid_tok]

    ds_train = MaskedLMDataset(train_ids, vocab, mask_prob=cfg.mask_prob, max_seq_len=cfg.max_seq_len, seed=cfg.seed)
    ds_valid = MaskedLMDataset(valid_ids, vocab, mask_prob=cfg.mask_prob, max_seq_len=cfg.max_seq_len, seed=cfg.seed + 1)

    if len(ds_train) == 0:
        raise ValueError("Training dataset is empty after tokenization.")

    # AMP
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    collate_fn = build_mlm_collate_fn(vocab.pad_id)
    pin = dev.type == "cuda"
    dl_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collate_fn,
        "pin_memory": pin,
        "num_workers": max(0, cfg.num_data_workers),
    }
    if cfg.num_data_workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, cfg.prefetch_factor)
        dl_kwargs["persistent_workers"] = True

    loader_train = DataLoader(ds_train, **dl_kwargs)
    dl_valid_kwargs = dict(dl_kwargs)
    dl_valid_kwargs["shuffle"] = False
    loader_valid = DataLoader(ds_valid, **dl_valid_kwargs)

    # Model
    model = MolRoBERTa(vocab_size=len(vocab.itos), cfg=cfg, pad_id=vocab.pad_id)
    mlm_head = MLMHead(cfg.d_model, len(vocab.itos))

    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            mlm_head = torch.compile(mlm_head)
            print("[pretrain-roberta] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[pretrain-roberta] torch.compile fallback: {exc}", flush=True)

    model.to(dev)
    mlm_head.to(dev)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in mlm_head.parameters())
    print(f"[pretrain-roberta] device={dev}, params={n_params} (~{n_params/1e6:.2f}M)", flush=True)
    print(f"[pretrain-roberta] amp={amp_label}, vocab_size={len(vocab.itos)}", flush=True)
    print(f"[pretrain-roberta] train_seqs={len(ds_train)}, valid_seqs={len(ds_valid)}", flush=True)

    all_params = list(model.parameters()) + list(mlm_head.parameters())
    adamw_kwargs: Dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(all_params, **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(all_params, **adamw_kwargs)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    non_blocking = dev.type == "cuda"

    # LR scheduler: linear warmup (10% of steps) + cosine decay
    total_steps = len(loader_train) * cfg.pretrain_epochs
    warmup_steps = max(1, total_steps // 10)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    print(f"[pretrain-roberta] total_steps={total_steps}, warmup_steps={warmup_steps}", flush=True)

    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        mlm_head.train()
        train_losses: List[float] = []

        train_iter = iterable_with_progress(
            loader_train,
            show_progress=show_progress,
            total=len(loader_train),
            desc=f"Pretrain[{representation}] ep{epoch}/{cfg.pretrain_epochs}",
        )
        for x, y in train_iter:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp else nullcontext()
            )
            with amp_ctx:
                h = model(x)
                logits = mlm_head(h)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        model.eval()
        mlm_head.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp else nullcontext()
                )
                with amp_ctx:
                    h = model(x)
                    logits = mlm_head(h)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                valid_losses.append(float(loss.item()))

        t_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        v_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
        history.append({
            "epoch": epoch,
            "train_loss": t_loss,
            "valid_loss": v_loss,
            "train_ppl": float(math.exp(t_loss)) if not math.isnan(t_loss) else float("nan"),
            "valid_ppl": float(math.exp(v_loss)) if not math.isnan(v_loss) else float("nan"),
        })
        print(
            f"[pretrain-roberta] epoch {epoch}/{cfg.pretrain_epochs}  "
            f"train_loss={t_loss:.4f}  valid_loss={v_loss:.4f}  "
            f"train_ppl={history[-1]['train_ppl']:.2f}  valid_ppl={history[-1]['valid_ppl']:.2f}",
            flush=True,
        )

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    head_to_save = mlm_head._orig_mod if hasattr(mlm_head, "_orig_mod") else mlm_head

    torch.save({
        "encoder_state": encoder_to_save.state_dict(),
        "mlm_head_state": head_to_save.state_dict(),
        "cfg": {k: v for k, v in vars(cfg).items()},
        "representation": representation,
    }, out_dir / "pretrained_roberta.pt")

    (out_dir / "vocab.json").write_text(
        json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pretrain_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pretrain_meta.json").write_text(
        json.dumps({
            "representation": representation,
            "train_size": len(train_smiles),
            "valid_size": len(valid_smiles),
            "n_train_encoded": len(train_tok),
            "n_valid_encoded": len(valid_tok),
            "vocab_size": len(vocab.itos),
            "n_params": n_params,
            "cfg": {k: v for k, v in vars(cfg).items()},
            "device": str(dev),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "history": history,
        "vocab_size": len(vocab.itos),
        "n_params": n_params,
        "device": str(dev),
    }


# ---------------------------------------------------------------------------
# Load pre-trained model
# ---------------------------------------------------------------------------

def load_pretrained_roberta(
    model_dir: str | Path,
    device: Optional[str] = None,
) -> Tuple[MolRoBERTa, TokenVocab, BaseRepresentation, RoBERTaConfig, torch.device]:
    d = Path(model_dir)
    ckpt = torch.load(d / "pretrained_roberta.pt", map_location="cpu", weights_only=False)
    cfg = RoBERTaConfig(**ckpt["cfg"])
    rep_name = ckpt["representation"]

    vocab = TokenVocab.from_dict(json.loads((d / "vocab.json").read_text(encoding="utf-8")))
    model = MolRoBERTa(vocab_size=len(vocab.itos), cfg=cfg, pad_id=vocab.pad_id)
    state = ckpt["encoder_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)

    dev = _resolve_torch_device(device)
    model.to(dev)
    model.eval()

    rep = build_representation(rep_name)
    return model, vocab, rep, cfg, dev


# ---------------------------------------------------------------------------
# MoleculeNet dataset loading
# ---------------------------------------------------------------------------

_MOLECULENET_TASKS: Dict[str, Dict[str, Any]] = {
    # --- Classification tasks ---
    "bbbp": {
        "dataset_name": "BBBP",
        "target_column": "p_np",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "hiv": {
        "dataset_name": "HIV",
        "target_column": "HIV_active",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "tox21": {
        "dataset_name": "Tox21",
        "target_column": "SR-p53",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "bace": {
        "dataset_name": "BACE",
        "target_column": "Class",
        "smiles_column": "mol",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "clintox": {
        "dataset_name": "ClinTox",
        "target_column": "CT_TOX",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "sider": {
        "dataset_name": "SIDER",
        "target_column": "Hepatobiliary disorders",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    "muv": {
        "dataset_name": "MUV",
        "target_column": "MUV-466",
        "smiles_column": "smiles",
        "task_type": "classification",
        "n_classes": 2,
        "metric": "roc_auc",
    },
    # --- Regression tasks ---
    "esol": {
        "dataset_name": "ESOL",
        "target_column": "measured log solubility in mols per litre",
        "smiles_column": "smiles",
        "task_type": "regression",
        "metric": "rmse",
    },
    "freesolv": {
        "dataset_name": "FreeSolv",
        "target_column": "expt",
        "smiles_column": "smiles",
        "task_type": "regression",
        "metric": "rmse",
    },
    "lipophilicity": {
        "dataset_name": "Lipophilicity",
        "target_column": "exp",
        "smiles_column": "smiles",
        "task_type": "regression",
        "metric": "rmse",
    },
}


def load_moleculenet_task(
    task_name: str,
    data_dir: Optional[str | Path] = None,
    split_seed: int = 42,
) -> Dict[str, Any]:
    """Load a MoleculeNet task (classification or regression) and return 80/10/10 splits.

    Tries DeepChem first; falls back to a minimal CSV loader if DeepChem is
    unavailable.
    """
    task_key = task_name.strip().lower()
    if task_key not in _MOLECULENET_TASKS:
        raise ValueError(f"Unknown MoleculeNet task: {task_name}. Available: {list(_MOLECULENET_TASKS.keys())}")

    task_info = _MOLECULENET_TASKS[task_key]
    data_root = str(data_dir) if data_dir else "."

    try:
        return _load_via_deepchem(task_key, task_info, data_root, split_seed)
    except Exception:
        pass

    # Fallback: look for a local CSV
    return _load_from_local_csv(task_key, task_info, data_root, split_seed)


def _load_via_deepchem(
    task_key: str,
    task_info: Dict[str, Any],
    data_root: str,
    split_seed: int,
) -> Dict[str, Any]:
    import deepchem as dc

    loaders = {
        "bbbp": dc.molnet.load_bbbp,
        "hiv": dc.molnet.load_hiv,
        "tox21": dc.molnet.load_tox21,
        "bace": dc.molnet.load_bace_classification,
        "clintox": dc.molnet.load_clintox,
        "sider": dc.molnet.load_sider,
        "muv": dc.molnet.load_muv,
        "esol": dc.molnet.load_delaney,
        "freesolv": dc.molnet.load_sampl,
        "lipophilicity": dc.molnet.load_lipo,
    }
    loader = loaders[task_key]

    # scaffold split is standard for MoleculeNet; fallback to random
    try:
        tasks_list, datasets, transformers = loader(
            featurizer=dc.feat.DummyFeaturizer(),
            splitter="scaffold",
            data_dir=data_root,
        )
    except Exception:
        tasks_list, datasets, transformers = loader(
            featurizer=dc.feat.DummyFeaturizer(),
            splitter="random",
            data_dir=data_root,
        )

    train_ds, valid_ds, test_ds = datasets

    is_classification = task_info["task_type"] == "classification"

    def _extract(ds, target_col: str) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        for i in range(len(ds)):
            smi = str(ds.ids[i])
            can = canonicalize_smiles(smi, require_single_component=True)
            if can is None:
                continue
            y = ds.y[i]
            if hasattr(y, '__len__'):
                # Multi-task: find the right column index
                if target_col in tasks_list:
                    col_idx = tasks_list.index(target_col)
                else:
                    col_idx = 0
                val = float(y[col_idx])
            else:
                val = float(y)
            if not np.isfinite(val):
                continue
            if is_classification and val not in (0.0, 1.0):
                continue
            rows.append((can, val))
        return rows

    target = task_info["target_column"]
    return {
        "task": task_key,
        "train": _extract(train_ds, target),
        "valid": _extract(valid_ds, target),
        "test": _extract(test_ds, target),
        "task_info": task_info,
        "source": "deepchem",
    }


def _load_from_local_csv(
    task_key: str,
    task_info: Dict[str, Any],
    data_root: str,
    split_seed: int,
) -> Dict[str, Any]:
    """Fallback: load from a local CSV and do 80/10/10 random split."""
    root = Path(data_root)
    candidates = [
        root / f"{task_key}.csv",
        root / f"{task_info['dataset_name']}.csv",
        root / "moleculenet" / f"{task_key}.csv",
        root / "moleculenet" / f"{task_info['dataset_name']}.csv",
        root / "data" / f"{task_key}.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError(
            f"No local CSV found for {task_key}. Install `deepchem` or provide "
            f"a CSV at one of: {[str(c) for c in candidates]}"
        )

    rows: List[Tuple[str, float]] = []
    smi_col = task_info["smiles_column"]
    target_col = task_info["target_column"]
    is_classification = task_info["task_type"] == "classification"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get(smi_col) or "").strip()
            y_raw = (row.get(target_col) or "").strip()
            if not smi or not y_raw:
                continue
            can = canonicalize_smiles(smi, require_single_component=True)
            if can is None:
                continue
            try:
                y = float(y_raw)
            except ValueError:
                continue
            if not np.isfinite(y):
                continue
            if is_classification and y not in (0.0, 1.0):
                continue
            rows.append((can, y))

    rnd = random.Random(split_seed)
    rnd.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * 0.1))
    n_valid = max(1, int(n * 0.1))
    test = rows[:n_test]
    valid = rows[n_test:n_test + n_valid]
    train = rows[n_test + n_valid:]

    return {
        "task": task_key,
        "train": train,
        "valid": valid,
        "test": test,
        "task_info": task_info,
        "source": "local_csv",
    }


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

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

    sum_pos_ranks = sum(ranks[idx] for idx, y in enumerate(ys) if y == 1)
    return float((sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    mse = float(np.mean([(t - p) ** 2 for t, p in zip(y_true, y_pred)]))
    return float(math.sqrt(mse))


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    yt = np.array(y_true)
    yp = np.array(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def finetune_roberta(
    pretrained_dir: str | Path,
    task_name: str,
    output_dir: str | Path,
    cfg: Optional[RoBERTaConfig] = None,
    data_dir: Optional[str | Path] = None,
    split_seed: int = 42,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
) -> Dict[str, Any]:
    """Fine-tune a pre-trained RoBERTa encoder on a MoleculeNet task (classification or regression)."""
    encoder, vocab, rep, pretrained_cfg, dev = load_pretrained_roberta(pretrained_dir, device=device)
    cfg = cfg or pretrained_cfg

    _set_seed(cfg.seed)

    # Override device if needed
    dev = _resolve_torch_device(device)
    encoder.to(dev)

    # Load task data
    print(f"[finetune-roberta] Loading task: {task_name}", flush=True)
    task_data = load_moleculenet_task(task_name, data_dir=data_dir, split_seed=split_seed)
    task_info = task_data["task_info"]
    is_regression = task_info["task_type"] == "regression"
    train_rows = task_data["train"]
    valid_rows = task_data["valid"]
    test_rows = task_data["test"]
    print(
        f"[finetune-roberta] train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}",
        flush=True,
    )

    # Encode SMILES → representation → token ids
    def _encode_rows(rows: List[Tuple[str, float]]) -> Tuple[List[List[int]], List[float]]:
        smiles_list = [s for s, _ in rows]
        labels = [y for _, y in rows]

        tok_seqs = _encode_sequences_with_rep(
            smiles_list, rep,
            show_progress=False,
            desc="encode",
            num_workers=encode_workers,
            chunksize=encode_chunksize,
        )

        # Some SMILES may fail encoding; align labels
        encoded_ids: List[List[int]] = []
        encoded_labels: List[float] = []
        encode_idx = 0
        for i, smi in enumerate(smiles_list):
            if encode_idx < len(tok_seqs):
                # Check if this SMILES was successfully encoded
                # _encode_sequences_with_rep skips failures, so we try to encode
                # each one to check
                try:
                    text = rep.encode(smi)
                    toks = rep.tokenize(text)
                    if toks:
                        ids = [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
                        encoded_ids.append(ids)
                        encoded_labels.append(labels[i])
                except Exception:
                    pass

        return encoded_ids, encoded_labels

    print(f"[finetune-roberta] Encoding datasets for {rep.name}...", flush=True)
    train_ids, train_labels = _encode_rows(train_rows)
    valid_ids, valid_labels = _encode_rows(valid_rows)
    test_ids, test_labels = _encode_rows(test_rows)
    print(
        f"[finetune-roberta] encoded: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}",
        flush=True,
    )

    if not train_ids:
        raise ValueError("No training samples encoded successfully.")

    # For regression: z-score normalize labels using train set statistics
    label_mean = 0.0
    label_std = 1.0
    if is_regression:
        label_mean = float(np.mean(train_labels))
        label_std = float(np.std(train_labels))
        if label_std < 1e-8:
            label_std = 1.0
        train_labels = [(y - label_mean) / label_std for y in train_labels]
        valid_labels = [(y - label_mean) / label_std for y in valid_labels]
        test_labels = [(y - label_mean) / label_std for y in test_labels]
        print(f"[finetune-roberta] label normalization: mean={label_mean:.4f}, std={label_std:.4f}", flush=True)

    ds_train = ClassificationDataset(train_ids, train_labels, max_seq_len=cfg.max_seq_len, pad_id=vocab.pad_id)
    ds_valid = ClassificationDataset(valid_ids, valid_labels, max_seq_len=cfg.max_seq_len, pad_id=vocab.pad_id)
    ds_test = ClassificationDataset(test_ids, test_labels, max_seq_len=cfg.max_seq_len, pad_id=vocab.pad_id)

    # AMP
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    collate_fn = build_cls_collate_fn(vocab.pad_id)
    pin = dev.type == "cuda"
    dl_kwargs: Dict[str, Any] = {
        "batch_size": cfg.finetune_batch_size,
        "collate_fn": collate_fn,
        "pin_memory": pin,
        "num_workers": max(0, cfg.num_data_workers),
    }
    if cfg.num_data_workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, cfg.prefetch_factor)
        dl_kwargs["persistent_workers"] = True

    loader_train = DataLoader(ds_train, shuffle=True, drop_last=False, **dl_kwargs)
    loader_valid = DataLoader(ds_valid, shuffle=False, drop_last=False, **dl_kwargs)
    loader_test = DataLoader(ds_test, shuffle=False, drop_last=False, **dl_kwargs)

    # Task head
    if is_regression:
        head = RegressionHead(cfg.d_model, dropout=cfg.dropout)
    else:
        head = ClassificationHead(cfg.d_model, n_classes=1, dropout=cfg.dropout)
    head.to(dev)

    # Fine-tune: encoder + head
    encoder.train()
    all_params = list(encoder.parameters()) + list(head.parameters())
    ft_kwargs: Dict[str, Any] = {"lr": cfg.finetune_lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            ft_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(all_params, **ft_kwargs)
    except TypeError:
        ft_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(all_params, **ft_kwargs)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    non_blocking = dev.type == "cuda"

    if is_regression:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # For classification: track best valid AUC (higher is better)
    # For regression: track best valid RMSE (lower is better)
    best_valid_metric = -1.0 if not is_regression else float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.finetune_epochs + 1):
        encoder.train()
        head.train()
        train_losses: List[float] = []

        train_iter = iterable_with_progress(
            loader_train,
            show_progress=show_progress,
            total=len(loader_train),
            desc=f"Finetune[{task_name}] ep{epoch}/{cfg.finetune_epochs}",
        )
        for x, y in train_iter:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp else nullcontext()
            )
            with amp_ctx:
                cls_repr = encoder(x, return_cls=True)
                preds = head(cls_repr).squeeze(-1)
                loss = loss_fn(preds, y)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
            train_losses.append(float(loss.item()))

        # Validate
        encoder.eval()
        head.eval()
        valid_y_true: List[float] = []
        valid_y_pred: List[float] = []
        valid_losses: List[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp else nullcontext()
                )
                with amp_ctx:
                    cls_repr = encoder(x, return_cls=True)
                    preds = head(cls_repr).squeeze(-1)
                    loss = loss_fn(preds, y)
                valid_losses.append(float(loss.item()))
                if is_regression:
                    valid_y_pred.extend(preds.cpu().tolist())
                else:
                    valid_y_pred.extend(torch.sigmoid(preds).cpu().tolist())
                valid_y_true.extend(y.cpu().tolist())

        t_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        v_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")

        if is_regression:
            # Un-normalize for metrics
            v_true_orig = [y * label_std + label_mean for y in valid_y_true]
            v_pred_orig = [y * label_std + label_mean for y in valid_y_pred]
            v_rmse = _rmse(v_true_orig, v_pred_orig)
            v_r2 = _r2(v_true_orig, v_pred_orig)
            history.append({
                "epoch": epoch,
                "train_loss": t_loss,
                "valid_loss": v_loss,
                "valid_rmse": v_rmse,
                "valid_r2": v_r2,
            })
            print(
                f"[finetune-roberta] epoch {epoch}/{cfg.finetune_epochs}  "
                f"train_loss={t_loss:.4f}  valid_loss={v_loss:.4f}  "
                f"valid_rmse={v_rmse:.4f}  valid_r2={v_r2:.4f}",
                flush=True,
            )
            improved = not math.isnan(v_rmse) and v_rmse < best_valid_metric
        else:
            v_auc = _roc_auc([int(y) for y in valid_y_true], valid_y_pred)
            history.append({
                "epoch": epoch,
                "train_loss": t_loss,
                "valid_loss": v_loss,
                "valid_roc_auc": v_auc,
            })
            print(
                f"[finetune-roberta] epoch {epoch}/{cfg.finetune_epochs}  "
                f"train_loss={t_loss:.4f}  valid_loss={v_loss:.4f}  valid_auc={v_auc:.4f}",
                flush=True,
            )
            improved = not math.isnan(v_auc) and v_auc > best_valid_metric

        if improved:
            best_valid_metric = v_rmse if is_regression else v_auc
            best_epoch = epoch
            enc_to_save = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder
            head_to_save = head._orig_mod if hasattr(head, "_orig_mod") else head
            best_state = {
                "encoder_state": {k: v.cpu().clone() for k, v in enc_to_save.state_dict().items()},
                "head_state": {k: v.cpu().clone() for k, v in head_to_save.state_dict().items()},
            }

    # Restore best and evaluate on test set
    if best_state is not None:
        enc_clean = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder
        head_clean = head._orig_mod if hasattr(head, "_orig_mod") else head
        enc_clean.load_state_dict(best_state["encoder_state"])
        head_clean.load_state_dict(best_state["head_state"])
        encoder.to(dev)
        head.to(dev)

    encoder.eval()
    head.eval()
    test_y_true: List[float] = []
    test_y_pred: List[float] = []
    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp else nullcontext()
            )
            with amp_ctx:
                cls_repr = encoder(x, return_cls=True)
                preds = head(cls_repr).squeeze(-1)
            if is_regression:
                test_y_pred.extend(preds.cpu().tolist())
            else:
                test_y_pred.extend(torch.sigmoid(preds).cpu().tolist())
            test_y_true.extend(y.cpu().tolist())

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if is_regression:
        # Un-normalize to original scale for metrics
        test_y_true = [y * label_std + label_mean for y in test_y_true]
        test_y_pred = [y * label_std + label_mean for y in test_y_pred]
        test_rmse_val = _rmse(test_y_true, test_y_pred)
        test_r2_val = _r2(test_y_true, test_y_pred)
        print(
            f"[finetune-roberta] Test RMSE={test_rmse_val:.4f}  R²={test_r2_val:.4f}  (best_epoch={best_epoch})",
            flush=True,
        )
        results = {
            "task": task_name,
            "task_type": "regression",
            "representation": rep.name,
            "pretrained_dir": str(pretrained_dir),
            "test_rmse": test_rmse_val,
            "test_r2": test_r2_val,
            "best_valid_rmse": best_valid_metric,
            "best_epoch": best_epoch,
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "n_test": len(test_ids),
            "history": history,
            "device": str(dev),
        }
    else:
        test_auc = _roc_auc([int(y) for y in test_y_true], test_y_pred)
        test_preds_cls = [1 if p >= 0.5 else 0 for p in test_y_pred]
        test_acc = float(np.mean([1 if p == int(t) else 0 for p, t in zip(test_preds_cls, test_y_true)])) if test_y_true else float("nan")
        print(
            f"[finetune-roberta] Test ROC-AUC={test_auc:.4f}  Accuracy={test_acc:.4f}  (best_epoch={best_epoch})",
            flush=True,
        )
        results = {
            "task": task_name,
            "task_type": "classification",
            "representation": rep.name,
            "pretrained_dir": str(pretrained_dir),
            "test_roc_auc": test_auc,
            "test_accuracy": test_acc,
            "best_valid_roc_auc": best_valid_metric,
            "best_epoch": best_epoch,
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "n_test": len(test_ids),
            "history": history,
            "device": str(dev),
        }

    if best_state is not None:
        torch.save({
            "encoder_state": best_state["encoder_state"],
            "head_state": best_state["head_state"],
            "cfg": {k: v for k, v in vars(cfg).items()},
            "representation": rep.name,
            "task": task_name,
            "task_type": task_info["task_type"],
            "label_mean": label_mean,
            "label_std": label_std,
        }, out / "finetuned_model.pt")

    (out / "finetune_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return results


# ---------------------------------------------------------------------------
# Full pipeline: pretrain + finetune on all tasks
# ---------------------------------------------------------------------------

def run_roberta_property_benchmark(
    train_smiles: Sequence[str],
    valid_smiles: Sequence[str],
    representations: Sequence[str],
    output_dir: str | Path,
    tasks: Sequence[str] = ("bbbp", "hiv", "tox21", "bace", "clintox", "sider", "muv", "esol", "freesolv", "lipophilicity"),
    cfg: Optional[RoBERTaConfig] = None,
    data_dir: Optional[str | Path] = None,
    split_seed: int = 42,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
) -> Dict[str, Any]:
    """Run the full RoBERTa pre-train + fine-tune benchmark across representations and tasks."""
    cfg = cfg or RoBERTaConfig()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "representations": list(representations),
        "tasks": list(tasks),
        "results": {},
    }

    for rep_name in representations:
        print(f"\n{'='*60}", flush=True)
        print(f"[benchmark] Pre-training RoBERTa for: {rep_name}", flush=True)
        print(f"{'='*60}", flush=True)

        pretrain_dir = out_root / "pretrained" / rep_name
        try:
            pretrain_result = pretrain_roberta(
                train_smiles=train_smiles,
                valid_smiles=valid_smiles,
                representation=rep_name,
                output_dir=pretrain_dir,
                cfg=cfg,
                show_progress=show_progress,
                device=device,
                encode_workers=encode_workers,
                encode_chunksize=encode_chunksize,
            )
        except Exception as exc:
            print(f"[benchmark] Pre-training failed for {rep_name}: {exc}", flush=True)
            all_results["results"][rep_name] = {"error": str(exc)}
            continue

        rep_results: Dict[str, Any] = {"pretrain": pretrain_result, "finetune": {}}

        for task_name in tasks:
            print(f"\n[benchmark] Fine-tuning {rep_name} on {task_name}", flush=True)
            ft_dir = out_root / "finetune" / rep_name / task_name
            try:
                ft_result = finetune_roberta(
                    pretrained_dir=pretrain_dir,
                    task_name=task_name,
                    output_dir=ft_dir,
                    cfg=cfg,
                    data_dir=data_dir,
                    split_seed=split_seed,
                    show_progress=show_progress,
                    device=device,
                    encode_workers=encode_workers,
                    encode_chunksize=encode_chunksize,
                )
                rep_results["finetune"][task_name] = ft_result
            except Exception as exc:
                print(f"[benchmark] Fine-tune failed for {rep_name}/{task_name}: {exc}", flush=True)
                rep_results["finetune"][task_name] = {"error": str(exc)}

        all_results["results"][rep_name] = rep_results

    # Summary table
    summary_rows: List[Dict[str, Any]] = []
    for rep_name in representations:
        rep_res = all_results["results"].get(rep_name, {})
        if "error" in rep_res:
            continue
        ft = rep_res.get("finetune", {})
        row: Dict[str, Any] = {"representation": rep_name}
        for task_name in tasks:
            task_res = ft.get(task_name, {})
            task_type = _MOLECULENET_TASKS.get(task_name, {}).get("task_type", "classification")
            if task_type == "regression":
                row[f"{task_name}_rmse"] = task_res.get("test_rmse", float("nan"))
                row[f"{task_name}_r2"] = task_res.get("test_r2", float("nan"))
            else:
                row[f"{task_name}_roc_auc"] = task_res.get("test_roc_auc", float("nan"))
                row[f"{task_name}_accuracy"] = task_res.get("test_accuracy", float("nan"))
        summary_rows.append(row)

    all_results["summary"] = summary_rows

    (out_root / "roberta_benchmark_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV summary
    if summary_rows:
        fields = list(summary_rows[0].keys())
        with (out_root / "roberta_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)

    return all_results

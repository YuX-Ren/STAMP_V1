#!/usr/bin/env python3
"""CamS-aligned property-prediction fine-tune for decoder STAMP checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from stamp.benchmark.hybrid_vocab import smiles_char_tokens
from stamp.benchmark.lm import LMConfig, LlamaDecoderLM, TinyDecoderLM, TokenVocab
from stamp.benchmark.multiscale_stamp import (
    MultiScaleSTAMPEncoder,
    parse_csv_arg,
    resolve_multiscale_vocab_paths,
)
from stamp.benchmark.representations import build_representation
from stamp.benchmark.roberta_property import load_moleculenet_task


MULTITASK_SETS = {"tox21", "toxcast", "clintox", "sider", "muv", "estrogen", "metstab"}

KPGT_TASK_ALIASES = {
    "lipophilicity": "lipo",
}

KPGT_TASK_TYPES = {
    "bace": "classification",
    "bbbp": "classification",
    "clintox": "classification",
    "estrogen": "classification",
    "metstab": "classification",
    "sider": "classification",
    "tox21": "classification",
    "toxcast": "classification",
    "esol": "regression",
    "freesolv": "regression",
    "lipo": "regression",
}


def _resolve_kpgt_root(data_dir: Path) -> Path:
    if (data_dir / "bbbp" / "bbbp.csv").exists():
        return data_dir
    if (data_dir / "datasets" / "bbbp" / "bbbp.csv").exists():
        return data_dir / "datasets"
    raise FileNotFoundError(f"Could not find KPGT datasets under {data_dir}")


def _load_kpgt_split_task(task_name: str, data_dir: Path, split_name: str) -> Dict:
    task_key = KPGT_TASK_ALIASES.get(task_name, task_name)
    root = _resolve_kpgt_root(data_dir)
    task_dir = root / task_key
    csv_path = task_dir / f"{task_key}.csv"
    split_path = task_dir / "splits" / f"{split_name}.npy"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    import pandas as pd

    df = pd.read_csv(csv_path)
    split_idxs = np.load(split_path, allow_pickle=True)
    if len(split_idxs) != 3:
        raise ValueError(f"Expected [train, val, test] indices in {split_path}, got {len(split_idxs)} entries")
    label_cols = [c for c in df.columns if c != "smiles"]
    if not label_cols:
        raise ValueError(f"No label columns found in {csv_path}")
    task_type = KPGT_TASK_TYPES.get(task_key)
    if task_type is None:
        raise ValueError(f"Unknown KPGT task type for {task_key}")

    def _scalar_rows(indices: Sequence[int]) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        target_col = label_cols[0]
        for idx in indices:
            r = df.iloc[int(idx)]
            smi = str(r["smiles"])
            val = float(r[target_col])
            if not np.isfinite(val):
                continue
            if task_type == "classification" and val not in (0.0, 1.0):
                continue
            rows.append((smi, val))
        return rows

    def _multi_rows(indices: Sequence[int]) -> List[Tuple[str, List[float]]]:
        rows: List[Tuple[str, List[float]]] = []
        for idx in indices:
            r = df.iloc[int(idx)]
            smi = str(r["smiles"])
            vals = [float(r[c]) if np.isfinite(float(r[c])) else float("nan") for c in label_cols]
            rows.append((smi, vals))
        return rows

    train_idx, valid_idx, test_idx = split_idxs
    out = {
        "task": task_name,
        "kpgt_task": task_key,
        "split_name": split_name,
        "source": "kpgt_figshare",
        "tasks_list": label_cols,
        "task_info": {"task_type": task_type, "target_column": label_cols[0]},
        "train": _scalar_rows(train_idx),
        "valid": _scalar_rows(valid_idx),
        "test": _scalar_rows(test_idx),
        "y_mean": 0.0,
        "y_std": 1.0,
    }
    if task_type == "classification" and len(label_cols) > 1:
        out["train_multi"] = _multi_rows(train_idx)
        out["valid_multi"] = _multi_rows(valid_idx)
        out["test_multi"] = _multi_rows(test_idx)
    return out


def _strip_orig_mod(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _load_vocab_for_checkpoint(model_dir: Path) -> TokenVocab:
    vocab_path = model_dir / "vocab.json"
    if vocab_path.exists():
        return TokenVocab.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    cache_path = model_dir / "_scan_cache_cams_decoder_v4.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        itos = list(cache["vocab_itos"])
        return TokenVocab(stoi={tok: i for i, tok in enumerate(itos)}, itos=itos)
    raise FileNotFoundError(f"Missing vocab.json and scan cache under {model_dir}")


def _resolve_checkpoint_paths(checkpoint_dir: Path, checkpoint_file: Optional[Path]) -> tuple[Path, Path]:
    if checkpoint_file is not None:
        ckpt_file = checkpoint_file
        if not ckpt_file.exists():
            raise FileNotFoundError(ckpt_file)
        model_dir = ckpt_file.parent.parent if ckpt_file.parent.name == "checkpoints" else ckpt_file.parent
        return model_dir, ckpt_file
    model_dir = checkpoint_dir
    ckpt_file = model_dir / "model.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)
    return model_dir, ckpt_file


def _load_decoder_checkpoint_bundle(checkpoint_dir: Path, checkpoint_file: Optional[Path]) -> dict:
    model_dir, ckpt_file = _resolve_checkpoint_paths(checkpoint_dir, checkpoint_file)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ckpt["cfg"])
    cfg.model_type = str(ckpt.get("model_type", cfg.model_type))
    return {
        "model_dir": model_dir,
        "ckpt_file": ckpt_file,
        "ckpt": ckpt,
        "cfg": cfg,
        "vocab": _load_vocab_for_checkpoint(model_dir),
        "backbone_name": str(ckpt.get("backbone", "tiny")),
        "tie_embeddings": bool(ckpt.get("tie_embeddings", True)),
        "model_state": _strip_orig_mod(ckpt["model_state"]),
    }


def _instantiate_decoder_from_bundle(bundle: dict, *, device: torch.device) -> nn.Module:
    cfg = bundle["cfg"]
    vocab = bundle["vocab"]
    if bundle["backbone_name"] == "llama":
        backbone: nn.Module = LlamaDecoderLM(
            vocab_size=len(vocab.itos),
            cfg=cfg,
            tie_embeddings=bool(bundle["tie_embeddings"]),
        )
    else:
        backbone = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    backbone.load_state_dict(bundle["model_state"], strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone


def _load_decoder_checkpoint(
    checkpoint_dir: Path,
    checkpoint_file: Optional[Path],
    *,
    device: torch.device,
) -> tuple[nn.Module, TokenVocab, object, dict, Path]:
    bundle = _load_decoder_checkpoint_bundle(checkpoint_dir, checkpoint_file)
    ckpt = bundle["ckpt"]
    vocab = bundle["vocab"]
    model_dir = bundle["model_dir"]
    backbone = _instantiate_decoder_from_bundle(bundle, device=device)

    stamp_vocab_file = ckpt.get("stamp_vocab")
    stamp_vocab_path = None
    if stamp_vocab_file is not None:
        candidate = model_dir / str(stamp_vocab_file)
        if candidate.exists():
            stamp_vocab_path = str(candidate)
    rep = build_representation(str(ckpt.get("representation", "stamp")), stamp_vocab=stamp_vocab_path)
    return backbone, vocab, rep, ckpt, model_dir


def _encode_tokens_with_smiles_fallback(vocab: TokenVocab, toks: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for tok in toks:
        tok_id = vocab.stoi.get(tok)
        if tok_id is not None:
            ids.append(tok_id)
            continue
        try:
            fallback = smiles_char_tokens(str(tok))
        except ValueError:
            fallback = []
        if fallback and all(piece in vocab.stoi for piece in fallback):
            ids.extend(vocab.stoi[piece] for piece in fallback)
        else:
            ids.append(vocab.unk_id)
    return ids


def _rdkit_topological_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _split_smiles_components(smiles: str) -> List[str]:
    return [part.strip() for part in str(smiles).split(".") if part.strip()]


def _encode_multiscale_tokens_preserve_components(rep: MultiScaleSTAMPEncoder, smiles: str) -> tuple[List[str], bool]:
    components = _split_smiles_components(smiles)
    if len(components) <= 1:
        return MultiScaleSTAMPEncoder.tokenize(" [CONCAT] ".join(rep.encode_scales(smiles))), False

    per_scale: List[List[str]] = []
    for scale_rep in rep.representations:
        scale_toks: List[str] = []
        for comp_i, comp in enumerate(components):
            if comp_i > 0:
                scale_toks.append("[.]")
            scale_toks.extend(MultiScaleSTAMPEncoder.tokenize(scale_rep.encode(comp)))
        per_scale.append(scale_toks)

    toks: List[str] = []
    for scale_i, scale_toks in enumerate(per_scale):
        if scale_i > 0:
            toks.append("[CONCAT]")
        toks.extend(scale_toks)
    return toks, True


def _encode_row_tokens(rep, smiles: str) -> tuple[List[str], str]:
    try:
        if isinstance(rep, MultiScaleSTAMPEncoder):
            toks, used_components = _encode_multiscale_tokens_preserve_components(rep, smiles)
        else:
            text = rep.encode(smiles)
            toks = rep.tokenize(text) if hasattr(rep, "tokenize") else MultiScaleSTAMPEncoder.tokenize(text)
            used_components = False
        return toks, ("components" if used_components else "stamp")
    except Exception:
        return smiles_char_tokens(str(smiles)), "smiles_fallback"


def _truncate_ids_keep_eos(ids: List[int], max_seq_len: int, eos_id: int) -> List[int]:
    if len(ids) <= max_seq_len:
        return ids
    if max_seq_len <= 1:
        return ids[:max_seq_len]
    return ids[: max_seq_len - 1] + [eos_id]


class _MolPropertyDataset(Dataset):
    def __init__(self, ids_list: List[List[int]], fps: np.ndarray, labels: np.ndarray) -> None:
        self.ids = ids_list
        self.fps = fps
        self.labels = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.fps[idx], self.labels[idx]


def _collate(pad_id: int):
    def _fn(batch):
        ids_list = [b[0] for b in batch]
        fps = np.stack([b[1] for b in batch], axis=0)
        labels = np.stack([b[2] for b in batch], axis=0)
        m = max(len(s) for s in ids_list)
        arr = np.full((len(batch), m), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), m), dtype=np.float32)
        for i, seq in enumerate(ids_list):
            arr[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1.0
        return (
            torch.from_numpy(arr),
            torch.from_numpy(mask),
            torch.from_numpy(fps.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
        )
    return _fn


class DecoderBackboneForPP(nn.Module):
    def __init__(
        self,
        backbone,
        d_model: int,
        n_tasks: int,
        *,
        dropout: float = 0.1,
        fp_bits: int = 2048,
        fp_mode: str = "fp_token",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fp_mode = str(fp_mode)
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_bits, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if self.fp_mode in {"none", "fp_token"}:
            head_in = d_model
        elif self.fp_mode in {"late_fusion", "fp_token_late"}:
            head_in = 2 * d_model
        else:
            raise ValueError(f"Unsupported fp_mode={self.fp_mode}")
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_tasks),
        )

    def _pool_sequence(self, hidden: torch.Tensor, attn_mask: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        feat = hidden[:, offset + 1 :, :]
        mask = attn_mask[:, 1:].unsqueeze(-1)
        return (feat * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def _causal_token_mask(attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = attn_mask.shape
        device = attn_mask.device
        base = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        return base.unsqueeze(0).unsqueeze(1) & attn_mask.bool()[:, None, None, :]

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        embedding_perturb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "forward_embeddings") and hasattr(self.backbone, "tok_emb"):
            embeds = self.backbone.tok_emb(input_ids)
            if embedding_perturb is not None:
                embeds = embeds + embedding_perturb
            return self.backbone.forward_embeddings(
                embeds,
                attn_mask=self._causal_token_mask(attn_mask),
            )
        if embedding_perturb is not None:
            raise ValueError("embedding perturbation requires backbone.forward_embeddings/tok_emb support.")
        if hasattr(self.backbone, "forward_hidden"):
            return self.backbone.forward_hidden(input_ids)
        captured: List[torch.Tensor] = []

        def hook(_m, _i, output):
            captured.append(output)

        h = self.backbone.blocks[-1].register_forward_hook(hook)
        try:
            _ = self.backbone(input_ids, force_causal=True)
        finally:
            h.remove()
        return captured[-1]

    @staticmethod
    def _global_token_mask(attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = attn_mask.shape
        total = seq_len + 1
        device = attn_mask.device
        base = torch.ones((total, total), dtype=torch.bool, device=device).tril()
        base[0, :] = True
        base[:, 0] = True
        key_mask = torch.cat(
            [torch.ones((bsz, 1), dtype=torch.bool, device=device), attn_mask.bool()],
            dim=1,
        )
        return base.unsqueeze(0).unsqueeze(1) & key_mask[:, None, None, :]

    def _forward_with_fp_token(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        fps: torch.Tensor,
        embedding_perturb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not hasattr(self.backbone, "forward_embeddings") or not hasattr(self.backbone, "tok_emb"):
            raise ValueError("fp_token mode requires a backbone with forward_embeddings/tok_emb support.")
        tok_emb = self.backbone.tok_emb(input_ids)
        fp_emb = self.fp_proj(fps).unsqueeze(1)
        embeds = torch.cat([fp_emb, tok_emb], dim=1)
        if embedding_perturb is not None:
            embeds = embeds + embedding_perturb
        hidden = self.backbone.forward_embeddings(
            embeds,
            attn_mask=self._global_token_mask(attn_mask),
        )
        global_feat = self.dropout(hidden[:, 0, :])
        if self.fp_mode == "fp_token":
            return self.head(global_feat)
        pooled = self.dropout(self._pool_sequence(hidden, attn_mask, offset=1))
        return self.head(torch.cat([pooled, global_feat], dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        embedding_perturb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fp_mode in {"fp_token", "fp_token_late"}:
            if fps is None:
                raise ValueError("fps is required when fp_mode uses fingerprint tokens")
            return self._forward_with_fp_token(input_ids, attn_mask, fps, embedding_perturb=embedding_perturb)

        pooled = self.dropout(self._pool_sequence(self._forward_hidden(input_ids, attn_mask, embedding_perturb), attn_mask))
        if self.fp_mode == "none":
            fused = pooled
        else:
            if fps is None:
                raise ValueError("fps is required when fp_mode != none")
            fp_feat = self.fp_proj(fps)
            fused = torch.cat([pooled, fp_feat], dim=-1)
        return self.head(fused)


def _tokenize_rows(
    rows,
    rep,
    vocab: TokenVocab,
    max_seq_len: int,
    fp_bits: int,
) -> Tuple[List[List[int]], np.ndarray, List[int], Dict[str, int]]:
    ids_list: List[List[int]] = []
    fps_list: List[np.ndarray] = []
    keep: List[int] = []
    stats = {"raw": 0, "encoded": 0, "dropped": 0, "components": 0, "smiles_fallback": 0, "unk_tokens": 0}
    for i, item in enumerate(rows):
        stats["raw"] += 1
        smi = item[0]
        try:
            toks, mode = _encode_row_tokens(rep, smi)
            ids = [vocab.bos_id] + _encode_tokens_with_smiles_fallback(vocab, toks) + [vocab.eos_id]
            if len(ids) > max_seq_len:
                ids = _truncate_ids_keep_eos(ids, max_seq_len, vocab.eos_id)
            if len(ids) < 2:
                stats["dropped"] += 1
                continue
            stats["encoded"] += 1
            stats["unk_tokens"] += sum(1 for token_id in ids if token_id == vocab.unk_id)
            if mode in stats:
                stats[mode] += 1
            ids_list.append(ids)
            fps_list.append(_rdkit_topological_fp(smi, n_bits=fp_bits))
            keep.append(i)
        except Exception:
            stats["dropped"] += 1
            continue
    if not fps_list:
        fps_arr = np.empty((0, fp_bits), dtype=np.float32)
    else:
        fps_arr = np.stack(fps_list, axis=0).astype(np.float32)
    return ids_list, fps_arr, keep, stats


def _prepare_task_data(d: dict, task: str, tasks_info: dict, rep, vocab: TokenVocab, max_seq_len: int, fp_bits: int) -> dict:
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

    ids_tr, fps_tr, k_tr, stats_tr = _tokenize_rows(rows_tr, rep, vocab, max_seq_len, fp_bits)
    ids_va, fps_va, k_va, stats_va = _tokenize_rows(rows_va, rep, vocab, max_seq_len, fp_bits)
    ids_te, fps_te, k_te, stats_te = _tokenize_rows(rows_te, rep, vocab, max_seq_len, fp_bits)
    Y_tr = _labels(rows_tr, k_tr)
    Y_va = _labels(rows_va, k_va)
    Y_te = _labels(rows_te, k_te)
    return {
        "is_class": is_class,
        "is_multi": is_multi,
        "n_tasks": n_tasks,
        "ids_tr": ids_tr,
        "ids_va": ids_va,
        "ids_te": ids_te,
        "fps_tr": fps_tr,
        "fps_va": fps_va,
        "fps_te": fps_te,
        "Y_tr": Y_tr,
        "Y_va": Y_va,
        "Y_te": Y_te,
        "y_mean": float(d.get("y_mean", 0.0)),
        "y_std": float(d.get("y_std", 1.0)),
        "encode_stats": {"train": stats_tr, "valid": stats_va, "test": stats_te},
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
    rmse_norm = float(np.sqrt(np.mean((preds[:, 0] - labels[:, 0]) ** 2)))
    return rmse_norm * y_std, None


def _eval_loop(model, loader, device, is_class: bool, y_std: float = 1.0, fp_mode: str = "late_fusion"):
    model.eval()
    all_p = []
    all_y = []
    with torch.no_grad():
        for x, mask, fps, y in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fps = fps.to(device, non_blocking=True)
            logits = model(x, mask, None if fp_mode == "none" else fps)
            if is_class:
                p = torch.sigmoid(logits).float().cpu().numpy()
            else:
                p = logits.float().cpu().numpy()
            all_p.append(p)
            all_y.append(y.numpy())
    return _compute_metric(np.concatenate(all_p, 0), np.concatenate(all_y, 0), is_class, y_std=y_std)


def _scheduled_lr(
    step: int,
    *,
    base_lr: float,
    total_updates: int,
    warmup_updates: int,
    scheduler: str,
    end_lr: float,
) -> float:
    if scheduler == "constant":
        if warmup_updates > 0 and step <= warmup_updates:
            return base_lr * float(step) / float(max(1, warmup_updates))
        return base_lr
    if scheduler != "polynomial":
        raise ValueError(f"Unsupported lr_scheduler={scheduler}")
    if warmup_updates > 0 and step <= warmup_updates:
        return base_lr * float(step) / float(max(1, warmup_updates))
    if step >= total_updates:
        return end_lr
    pct_remaining = 1.0 - float(step - warmup_updates) / float(max(1, total_updates - warmup_updates))
    return (base_lr - end_lr) * pct_remaining + end_lr


def _reset_module_parameters(module: nn.Module, init_fn) -> None:
    if init_fn is not None:
        module.apply(init_fn)

    def _reset_norm(m: nn.Module) -> None:
        if isinstance(m, nn.LayerNorm):
            m.reset_parameters()
        elif m.__class__.__name__ == "_RMSNorm" and hasattr(m, "weight"):
            with torch.no_grad():
                m.weight.fill_(1.0)

    module.apply(_reset_norm)


def _apply_reinit(model: DecoderBackboneForPP, n_layers: int) -> int:
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        return 0
    n = max(0, min(int(n_layers), len(blocks)))
    init_fn = getattr(model.backbone, "_init_weights", None)
    for block in list(blocks)[-n:]:
        _reset_module_parameters(block, init_fn)
    return n


def _llrd_parameter_groups(model: DecoderBackboneForPP, base_lr: float, decay_coef: float) -> list[dict]:
    blocks = getattr(model.backbone, "blocks", None)
    n_layers = len(blocks) if blocks is not None else 0
    groups: list[dict] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = float(base_lr)
        if name.startswith("backbone.blocks.") and n_layers > 0:
            parts = name.split(".")
            layer_idx = int(parts[2])
            lr = float(base_lr) * (float(decay_coef) ** (n_layers - layer_idx))
        elif name.startswith("backbone."):
            lr = float(base_lr) * (float(decay_coef) ** (n_layers + 1))
        groups.append({"params": [param], "lr": lr})
    return groups


def _l2sp_reference(model: DecoderBackboneForPP) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith("backbone.") and param.requires_grad
    }


def _l2sp_penalty(model: DecoderBackboneForPP, ref: dict[str, torch.Tensor]) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for name, param in model.named_parameters():
        ref_param = ref.get(name)
        if ref_param is None:
            continue
        term = 0.5 * torch.sum((param - ref_param.to(device=param.device, dtype=param.dtype)) ** 2)
        total = term if total is None else total + term
    if total is None:
        return next(model.parameters()).sum() * 0.0
    return total


def _embedding_perturb_shape(model: DecoderBackboneForPP, input_ids: torch.Tensor) -> tuple[int, int, int]:
    extra = 1 if model.fp_mode in {"fp_token", "fp_token_late"} else 0
    return (input_ids.shape[0], input_ids.shape[1] + extra, model.backbone.cfg.d_model)


def _train_one_task(
    task: str,
    data: dict,
    checkpoint_dir: Path,
    checkpoint_file: Optional[Path],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
    patience: int,
    fp_bits: int,
    fp_mode: str,
    warmup_ratio: float,
    warmup_steps: int,
    lr_scheduler: str,
    end_lr: float,
    drop_last_train: bool,
    grad_clip_norm: float,
    loss_normalization: str,
    finetune_strategy: str,
    llrd_decay_coef: float,
    l2sp_weight: float,
    flag_m: int,
    flag_step_size: float,
    reinit_num_layers: int,
    checkpoint_bundle: Optional[dict] = None,
) -> Dict:
    if checkpoint_bundle is None:
        backbone, vocab, _rep, _ckpt, _model_dir = _load_decoder_checkpoint(
            checkpoint_dir,
            checkpoint_file,
            device=device,
        )
    else:
        backbone = _instantiate_decoder_from_bundle(checkpoint_bundle, device=device)
        vocab = checkpoint_bundle["vocab"]
    d_model = backbone.cfg.d_model
    model = DecoderBackboneForPP(
        backbone,
        d_model,
        data["n_tasks"],
        dropout=dropout,
        fp_bits=fp_bits,
        fp_mode=fp_mode,
    ).to(device)
    reinit_applied = 0
    if finetune_strategy == "reinit":
        reinit_applied = _apply_reinit(model, reinit_num_layers)
    l2sp_ref = _l2sp_reference(model) if finetune_strategy == "l2sp" else None

    collate = _collate(vocab.pad_id)
    ds_tr = _MolPropertyDataset(data["ids_tr"], data["fps_tr"], data["Y_tr"])
    ds_va = _MolPropertyDataset(data["ids_va"], data["fps_va"], data["Y_va"])
    ds_te = _MolPropertyDataset(data["ids_te"], data["fps_te"], data["Y_te"])
    ld_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        drop_last=drop_last_train,
    )
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1, pin_memory=True)

    if finetune_strategy == "llrd":
        opt = torch.optim.AdamW(
            _llrd_parameter_groups(model, lr, llrd_decay_coef),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_updates = max(1, len(ld_tr) * max(1, epochs))
    n_warmup = int(warmup_steps) if int(warmup_steps) > 0 else int(total_updates * float(warmup_ratio))
    global_update = 0

    def _loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if data["is_class"]:
            mask = torch.isfinite(y) & ((y == 0.0) | (y == 1.0))
            if not mask.any():
                return logits.sum() * 0.0
            target = torch.where(mask, y, torch.zeros_like(y))
            per = bce(logits, target)
            masked = per * mask.float()
            if loss_normalization == "kpgt_mean":
                return masked.mean()
            return masked.sum() / mask.float().sum().clamp(min=1)
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
        for x, mask, fps, y in ld_tr:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fps = fps.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if finetune_strategy == "flag":
                m = max(1, int(flag_m))
                perturb = torch.empty(
                    _embedding_perturb_shape(model, x),
                    dtype=torch.float32,
                    device=device,
                ).uniform_(-float(flag_step_size), float(flag_step_size))
                perturb.requires_grad_()
                logits = model(x, mask, None if fp_mode == "none" else fps, embedding_perturb=perturb)
                loss = _loss_fn(logits, y) / float(m)
                for _ in range(m - 1):
                    loss.backward()
                    if perturb.grad is None:
                        raise RuntimeError("FLAG perturbation did not receive gradients.")
                    perturb = (
                        perturb.detach() + float(flag_step_size) * torch.sign(perturb.grad.detach())
                    ).detach()
                    perturb.requires_grad_()
                    logits = model(x, mask, None if fp_mode == "none" else fps, embedding_perturb=perturb)
                    loss = _loss_fn(logits, y) / float(m)
                loss.backward()
            else:
                logits = model(x, mask, None if fp_mode == "none" else fps)
                loss = _loss_fn(logits, y)
                if finetune_strategy == "l2sp" and l2sp_ref is not None:
                    loss = loss + float(l2sp_weight) * _l2sp_penalty(model, l2sp_ref)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            global_update += 1
            lr_now = _scheduled_lr(
                global_update,
                base_lr=lr,
                total_updates=total_updates,
                warmup_updates=n_warmup,
                scheduler=lr_scheduler,
                end_lr=end_lr,
            )
            for group in opt.param_groups:
                group["lr"] = lr_now
            opt.step()
            running += float(loss.item())
            n_b += 1

        vm, vper = _eval_loop(model, ld_va, device, data["is_class"], y_std=y_std, fp_mode=fp_mode)
        tm, tper = _eval_loop(model, ld_te, device, data["is_class"], y_std=y_std, fp_mode=fp_mode)
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
            f"[{task}] ep{ep:02d} train_loss={running/max(n_b,1):.4f} "
            f"valid_{metric_name}={vm:.4f} test_{metric_name}={tm:.4f} "
            f"best={best_test_m:.4f}@ep{best_epoch}",
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
        "fp_mode": fp_mode,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "lr_scheduler": lr_scheduler,
        "end_lr": float(end_lr),
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": int(n_warmup),
        "drop_last_train": bool(drop_last_train),
        "grad_clip_norm": float(grad_clip_norm),
        "loss_normalization": loss_normalization,
        "finetune_strategy": finetune_strategy,
        "llrd_decay_coef": float(llrd_decay_coef),
        "l2sp_weight": float(l2sp_weight),
        "flag_m": int(flag_m),
        "flag_step_size": float(flag_step_size),
        "reinit_num_layers": int(reinit_num_layers),
        "reinit_applied": int(reinit_applied),
        "encode_stats": data["encode_stats"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-file", type=Path, default=None)
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["bbbp", "bace", "tox21", "clintox", "sider", "esol", "freesolv", "lipophilicity"],
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--split-source", choices=["deepchem", "kpgt"], default="deepchem")
    ap.add_argument("--split-name", default="scaffold-0", help="KPGT split name, e.g. scaffold-0.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--fp-mode", choices=["none", "late_fusion", "fp_token", "fp_token_late"], default="fp_token")
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--lr-scheduler", choices=["constant", "polynomial"], default="constant")
    ap.add_argument("--end-lr", type=float, default=1e-9)
    ap.add_argument("--drop-last-train", action="store_true")
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--loss-normalization", choices=["valid_mean", "kpgt_mean"], default="valid_mean")
    ap.add_argument("--finetune-strategy", choices=["normal", "l2sp", "llrd", "flag", "reinit"], default="normal")
    ap.add_argument("--llrd-decay-coef", type=float, default=0.8)
    ap.add_argument("--l2sp-weight", type=float, default=1e-4)
    ap.add_argument("--flag-m", type=int, default=3)
    ap.add_argument("--flag-step-size", type=float, default=1e-3)
    ap.add_argument("--reinit-num-layers", type=int, default=3)
    ap.add_argument("--multiscale-vocabs", default="", help="Optional comma-separated override for multiscale vocab paths.")
    ap.add_argument("--multiscale-scale-names", default="", help="Optional comma-separated scale names matching --multiscale-vocabs.")
    ap.add_argument("--force-ring-aware-encode", action="store_true", help="Override vocab header and emit [Ring] closures.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    tasks_info = {
        "bbbp": "classification",
        "bace": "classification",
        "hiv": "classification",
        "tox21": "classification",
        "clintox": "classification",
        "sider": "classification",
        "muv": "classification",
        "toxcast": "classification",
        "estrogen": "classification",
        "metstab": "classification",
        "esol": "regression",
        "freesolv": "regression",
        "lipophilicity": "regression",
        "lipo": "regression",
    }

    backbone0, vocab0, rep0, ckpt0, model_dir = _load_decoder_checkpoint(
        args.checkpoint_dir,
        args.checkpoint_file,
        device=device,
    )
    override_multiscale_vocab_paths = parse_csv_arg(args.multiscale_vocabs)
    override_multiscale_scale_names = parse_csv_arg(args.multiscale_scale_names)
    if override_multiscale_vocab_paths:
        if override_multiscale_scale_names and len(override_multiscale_scale_names) != len(override_multiscale_vocab_paths):
            raise ValueError("--multiscale-scale-names must match --multiscale-vocabs length.")
        if not override_multiscale_scale_names:
            override_multiscale_scale_names = [f"s{i}" for i in range(len(override_multiscale_vocab_paths))]
        rep_for_pp = MultiScaleSTAMPEncoder(
            override_multiscale_vocab_paths,
            scale_names=override_multiscale_scale_names,
            force_ring_aware_encode=bool(args.force_ring_aware_encode),
            strict_vocab=False,
        )
        mode_desc = f"multiscale_override[{','.join(override_multiscale_scale_names)}]"
    else:
        ckpt = ckpt0
        ms_vocab_files = list(ckpt.get("multiscale_vocab_files", []) or [])
        ms_scale_names = list(ckpt.get("multiscale_scale_names", []) or [])
        if ms_vocab_files:
            if not ms_scale_names:
                ms_scale_names = [f"s{i}" for i in range(len(ms_vocab_files))]
            rep_for_pp = MultiScaleSTAMPEncoder(
                resolve_multiscale_vocab_paths(model_dir, ms_vocab_files),
                scale_names=ms_scale_names,
                force_ring_aware_encode=bool(args.force_ring_aware_encode),
                strict_vocab=False,
            )
            mode_desc = f"multiscale_ckpt[{','.join(ms_scale_names)}]"
        else:
            if bool(args.force_ring_aware_encode) and hasattr(rep0, "codec") and hasattr(rep0.codec, "motif_tokenizer") and rep0.codec.motif_tokenizer is not None:
                tok = rep0.codec.motif_tokenizer
                tok.allow_ring_single_edges = True
                tok.blessed_rings = set(tok.vocab.keys())
            rep_for_pp = rep0
            mode_desc = "single_scale"
    max_seq_len = args.max_seq_len or backbone0.cfg.max_seq_len
    print(
        f"[setup] model_type={backbone0.cfg.model_type} d_model={backbone0.cfg.d_model} "
        f"max_seq_len={max_seq_len} fp_mode={args.fp_mode} input_mode={mode_desc}",
        flush=True,
    )
    del backbone0

    results = {}
    for task in args.tasks:
        if task not in tasks_info:
            print(f"[skip] unknown task {task}", flush=True)
            continue
        t0 = time.time()
        try:
            if args.split_source == "kpgt":
                d = _load_kpgt_split_task(task, args.data_dir, args.split_name)
            else:
                d = load_moleculenet_task(task, data_dir=str(args.data_dir))
        except Exception as e:
            print(f"[skip] {task}: load failed ({e})", flush=True)
            continue
        data = _prepare_task_data(d, task, tasks_info, rep_for_pp, vocab0, max_seq_len, args.fp_bits)
        print(
            f"[{task}] type={'classification' if data['is_class'] else 'regression'} "
            f"multi={data['is_multi']} n_tasks={data['n_tasks']} "
            f"train={len(data['ids_tr'])} valid={len(data['ids_va'])} test={len(data['ids_te'])} "
            f"encode_stats={data['encode_stats']} "
            f"source={d.get('source')} split={d.get('split_name', 'deepchem')}",
            flush=True,
        )
        res = _train_one_task(
            task,
            data,
            args.checkpoint_dir,
            args.checkpoint_file,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            dropout=args.dropout,
            patience=args.patience,
            fp_bits=args.fp_bits,
            fp_mode=args.fp_mode,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            lr_scheduler=args.lr_scheduler,
            end_lr=args.end_lr,
            drop_last_train=args.drop_last_train,
            grad_clip_norm=args.grad_clip_norm,
            loss_normalization=args.loss_normalization,
            finetune_strategy=args.finetune_strategy,
            llrd_decay_coef=args.llrd_decay_coef,
            l2sp_weight=args.l2sp_weight,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            reinit_num_layers=args.reinit_num_layers,
        )
        res["elapsed_sec"] = time.time() - t0
        res["split_source"] = args.split_source
        res["split_name"] = args.split_name
        results[task] = res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint_dir),
        "checkpoint_file": str(args.checkpoint_file) if args.checkpoint_file is not None else None,
        "data_dir": str(args.data_dir),
        "split_source": args.split_source,
        "split_name": args.split_name,
        "hyperparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "patience": args.patience,
            "fp_bits": args.fp_bits,
            "fp_mode": args.fp_mode,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler": args.lr_scheduler,
            "end_lr": args.end_lr,
            "drop_last_train": args.drop_last_train,
            "grad_clip_norm": args.grad_clip_norm,
            "loss_normalization": args.loss_normalization,
            "finetune_strategy": args.finetune_strategy,
            "llrd_decay_coef": args.llrd_decay_coef,
            "l2sp_weight": args.l2sp_weight,
            "flag_m": args.flag_m,
            "flag_step_size": args.flag_step_size,
            "reinit_num_layers": args.reinit_num_layers,
        },
        "tasks": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import os
import json
import math
import random
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .common import iterable_with_progress
from .representations import BaseRepresentation, RepresentationError, UnsupportedRepresentation, build_representation

_WORKER_REP: Optional[BaseRepresentation] = None


@dataclass
class LMConfig:
    model_type: str = "decoder"  # decoder | diffusion
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 256
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    seed: int = 42
    num_data_workers: int = 0
    prefetch_factor: int = 2
    persistent_data_workers: bool = True
    amp: str = "auto"  # auto | bf16 | fp16 | none
    torch_compile: bool = False
    tf32: bool = True
    diffusion_steps: int = 128
    diffusion_time_epsilon: float = 1e-3
    diffusion_scheduler: str = "linear"  # linear | cosine
    diffusion_loss_weight: str = "scheduler"  # scheduler | uniform
    diffusion_remasking: str = "low_confidence"  # low_confidence | random
    diffusion_stochastic_transfer: bool = False
    diffusion_block_size: int = 0  # <=0 means full generation window


class TokenVocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    MASK = "<mask>"

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self.pad_id = stoi[self.PAD]
        self.bos_id = stoi[self.BOS]
        self.eos_id = stoi[self.EOS]
        self.unk_id = stoi[self.UNK]
        self.mask_id = stoi.get(self.MASK, self.unk_id)

    @classmethod
    def build(
        cls,
        tokenized_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        add_mask_token: bool = False,
    ) -> "TokenVocab":
        cnt = Counter()
        for seq in tokenized_sequences:
            cnt.update(seq)

        vocab = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        if add_mask_token:
            vocab.append(cls.MASK)
        specials = set(vocab)
        for tok, freq in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and tok not in specials:
                vocab.append(tok)

        stoi = {tok: i for i, tok in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                out.append(self.UNK)
            else:
                out.append(self.itos[i])
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenVocab":
        itos = list(data["itos"])
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class NextTokenDataset(Dataset):
    def __init__(self, sequences: Sequence[List[int]], max_seq_len: int, pad_id: int) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.pad_id = pad_id

        for seq in sequences:
            if len(seq) < 2:
                continue
            ids = seq[: max_seq_len + 1]
            x = ids[:-1]
            y = ids[1:]
            if not x:
                continue
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


def build_dynamic_lm_collate_fn(pad_id: int, pad_to_multiple_of: Optional[int] = None):
    def _collate(batch: Sequence[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x, _ in batch)
        if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
            rem = max_len % pad_to_multiple_of
            if rem != 0:
                max_len += pad_to_multiple_of - rem

        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            n = len(x)
            x_out[i, :n] = torch.tensor(x, dtype=torch.long)
            y_out[i, :n] = torch.tensor(y, dtype=torch.long)
        return x_out, y_out

    return _collate


class DenoiseDataset(Dataset):
    def __init__(self, sequences: Sequence[List[int]], max_seq_len: int) -> None:
        self.samples: List[List[int]] = []
        for seq in sequences:
            ids = seq[:max_seq_len]
            if len(ids) < 2:
                continue
            self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.samples[idx]


def build_dynamic_denoise_collate_fn(
    pad_id: int,
    pad_to_multiple_of: Optional[int] = None,
):
    def _collate(batch: Sequence[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        if pad_to_multiple_of is not None and pad_to_multiple_of > 1:
            rem = max_len % pad_to_multiple_of
            if rem != 0:
                max_len += pad_to_multiple_of - rem

        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, ids in enumerate(batch):
            n = len(ids)
            x_out[i, :n] = torch.tensor(ids, dtype=torch.long)
            y_out[i, :n] = torch.tensor(ids, dtype=torch.long)
            y_out[i, 0] = -100  # keep BOS fixed
        return x_out, y_out

    return _collate


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seqlen} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        attn_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=attn_mask)
        x = self.ln_f(x)
        return self.head(x)


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    if t.dim() != 1:
        raise ValueError(f"Expected 1D timesteps, got shape={tuple(t.shape)}")
    half = dim // 2
    if half == 0:
        return t.unsqueeze(-1)
    device = t.device
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TinyDiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seqlen} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        if timesteps is None:
            timesteps = torch.zeros(bsz, device=input_ids.device, dtype=torch.float32)
        if timesteps.shape != (bsz,):
            raise ValueError(f"timesteps shape mismatch: expected {(bsz,)}, got {tuple(timesteps.shape)}")

        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        tok = self.tok_emb(input_ids) + self.pos_emb(pos)
        t_emb = self.time_mlp(_timestep_embedding(timesteps, self.cfg.d_model)).unsqueeze(1)
        x = tok + t_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def _alpha_from_t(t: torch.Tensor, scheduler: str) -> torch.Tensor:
    sched = (scheduler or "linear").strip().lower()
    if sched == "linear":
        return 1.0 - t
    if sched == "cosine":
        return 1.0 - torch.cos((math.pi / 2.0) * (1.0 - t))
    raise ValueError(f"Unsupported diffusion scheduler: {scheduler}")


def _alpha_weight_from_t(t: torch.Tensor, scheduler: str) -> torch.Tensor:
    sched = (scheduler or "linear").strip().lower()
    if sched == "linear":
        alpha = _alpha_from_t(t, sched)
        return torch.ones_like(t) / (1.0 - alpha + 1e-6)
    if sched == "cosine":
        z = (math.pi / 2.0) * (1.0 - t)
        alpha_deriv = -(math.pi / 2.0) * torch.sin(z)
        alpha = _alpha_from_t(t, sched)
        return -alpha_deriv / (1.0 - alpha + 1e-6)
    raise ValueError(f"Unsupported diffusion scheduler: {scheduler}")


def _reverse_mask_prob(s: float, t: float, scheduler: str) -> float:
    s_t = torch.tensor([s], dtype=torch.float32)
    t_t = torch.tensor([t], dtype=torch.float32)
    a_s = _alpha_from_t(s_t, scheduler)[0]
    a_t = _alpha_from_t(t_t, scheduler)[0]
    out = (1.0 - a_s) / (1.0 - a_t + 1e-8)
    return float(max(0.0, min(1.0, out.item())))


def _get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: str,
    stochastic: bool = False,
) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True).to(dtype=torch.long)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long)

    for i in range(mask_num.size(0)):
        for t_step, s_step, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s = float(s_step) / float(steps)
            t = float(t_step) / float(steps)
            reverse_transfer_prob = 1.0 - _reverse_mask_prob(s=s, t=t, scheduler=scheduler)

            if not stochastic:
                x = float(mask_num[i, 0].item()) * reverse_transfer_prob
                k = int(round(x))
            else:
                n = int(mask_num[i, 0].item())
                if n <= 0:
                    k = 0
                else:
                    sample = torch.distributions.Binomial(total_count=n, probs=reverse_transfer_prob).sample()
                    k = int(sample.item())
            k = min(k, int(mask_num[i, 0].item()))
            num_transfer_tokens[i, j] = k
            mask_num[i, 0] -= k
            if int(mask_num[i, 0].item()) == 0:
                break

    rows: List[torch.Tensor] = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, int(nonzero.numel()))
    if max_len == 0:
        return torch.zeros((num_transfer_tokens.size(0), 0), dtype=torch.long, device=mask_index.device)

    padded_rows = []
    for r in rows:
        if int(r.numel()) < max_len:
            pad = torch.zeros(max_len - int(r.numel()), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad], dim=0)
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


LMModel = TinyDecoderLM | TinyDiffusionLM


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_torch_device(device: Optional[str] = None) -> torch.device:
    req = (device or "auto").strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Please check your GPU environment or pass --device cpu."
            )
        return torch.device(device)
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device spec: {device}")


def _resolve_amp_mode(amp: str, dev: torch.device) -> Tuple[bool, Optional[torch.dtype], bool, str]:
    mode = (amp or "auto").strip().lower()
    if dev.type != "cuda" or mode in {"none", "off", "false", "0"}:
        return False, None, False, "none"
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False, "bf16"
        return True, torch.float16, True, "fp16"
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            return True, torch.float16, True, "fp16(auto_fallback)"
        return True, torch.bfloat16, False, "bf16"
    if mode in {"fp16", "float16", "half"}:
        return True, torch.float16, True, "fp16"
    raise ValueError(f"Unsupported amp mode: {amp}")


def _encode_sequences_with_rep(
    smiles_list: Sequence[str],
    rep: BaseRepresentation,
    show_progress: bool = False,
    desc: str = "Encode",
    num_workers: int = 1,
    chunksize: int = 128,
) -> List[List[str]]:
    tokenized: List[List[str]] = []
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    if num_workers == 1:
        iterator = iterable_with_progress(
            smiles_list,
            show_progress=show_progress,
            total=len(smiles_list),
            desc=desc,
        )
        for smiles in iterator:
            toks = _encode_one_smiles_local(rep, smiles)
            if toks:
                tokenized.append(toks)
        return tokenized

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_encode_worker,
        initargs=(rep.name,),
    ) as pool:
        mapped = pool.map(
            _encode_one_smiles_worker,
            smiles_list,
            chunksize=max(1, chunksize),
        )
        mapped = iterable_with_progress(
            mapped,
            show_progress=show_progress,
            total=len(smiles_list),
            desc=desc,
        )
        for toks in mapped:
            if toks:
                tokenized.append(toks)
    return tokenized


def _init_encode_worker(representation: str) -> None:
    global _WORKER_REP
    _WORKER_REP = build_representation(representation)


def _encode_one_smiles_local(rep: BaseRepresentation, smiles: str) -> Optional[List[str]]:
    try:
        text = rep.encode(smiles)
        toks = rep.tokenize(text)
        if toks:
            return toks
    except Exception:
        return None
    return None


def _encode_one_smiles_worker(smiles: str) -> Optional[List[str]]:
    rep = _WORKER_REP
    if rep is None:
        raise RuntimeError("Encode worker was not initialized.")
    return _encode_one_smiles_local(rep, smiles)


def _ids_with_special_tokens(tokenized: Sequence[Sequence[str]], vocab: TokenVocab) -> List[List[int]]:
    return [[vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id] for toks in tokenized]


def _normalize_model_type(model_type: str) -> str:
    mt = (model_type or "decoder").strip().lower()
    if mt not in {"decoder", "diffusion"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected one of: decoder, diffusion")
    return mt


def _compute_diffusion_loss(
    model: TinyDiffusionLM,
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    mask_id: int,
) -> torch.Tensor:
    bsz, seqlen = clean_ids.shape
    maskable_mask = labels != -100
    t = cfg.diffusion_time_epsilon + (1.0 - cfg.diffusion_time_epsilon) * torch.rand(
        bsz, device=clean_ids.device
    )
    p_mask = 1.0 - _alpha_from_t(t, scheduler=cfg.diffusion_scheduler)
    masked_mask = (torch.rand((bsz, seqlen), device=clean_ids.device) < p_mask.unsqueeze(1)) & maskable_mask

    if not bool(masked_mask.any().item()):
        any_maskable = maskable_mask.any(dim=1)
        if bool(any_maskable.any().item()):
            row_idx = torch.where(any_maskable)[0]
            first_pos = torch.argmax(maskable_mask[row_idx].to(dtype=torch.float32), dim=1)
            masked_mask[row_idx, first_pos] = True

    noised_ids = torch.where(masked_mask, torch.full_like(clean_ids, mask_id), clean_ids)
    logits = model(noised_ids, timesteps=t)
    token_nll = F.cross_entropy(
        logits.transpose(1, 2),
        clean_ids,
        reduction="none",
    )

    weight_mode = (cfg.diffusion_loss_weight or "scheduler").strip().lower()
    if weight_mode == "scheduler":
        loss_weights = _alpha_weight_from_t(t, scheduler=cfg.diffusion_scheduler).unsqueeze(1).expand_as(token_nll)
    elif weight_mode == "uniform":
        loss_weights = torch.ones_like(token_nll)
    else:
        raise ValueError(
            f"Unsupported diffusion_loss_weight: {cfg.diffusion_loss_weight}. Expected scheduler|uniform."
        )

    token_nll = token_nll * masked_mask.to(dtype=token_nll.dtype) * loss_weights
    denom = masked_mask.sum().clamp_min(1)
    return token_nll.sum() / denom


def train_language_model(
    train_smiles: Sequence[str],
    valid_smiles: Sequence[str],
    representation: str,
    output_dir: str | Path,
    cfg: Optional[LMConfig] = None,
    min_token_freq: int = 1,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
) -> Dict[str, Any]:
    rep = build_representation(representation)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot train LM for unsupported placeholder representation: {representation}"
        )

    cfg = cfg or LMConfig()
    model_type = _normalize_model_type(cfg.model_type)
    cfg.model_type = model_type
    if not (0.0 < float(cfg.diffusion_time_epsilon) < 1.0):
        raise ValueError("diffusion_time_epsilon must be in (0, 1).")
    if int(cfg.diffusion_steps) <= 0:
        raise ValueError("diffusion_steps must be >= 1.")

    _set_seed(cfg.seed)
    dev = _resolve_torch_device(device)

    if encode_workers <= 0:
        encode_workers = max(1, os.cpu_count() or 1)
    print(
        f"[train-lm] tokenize_workers={encode_workers}, tokenize_chunksize={max(1, encode_chunksize)}",
        flush=True,
    )
    train_tok = _encode_sequences_with_rep(
        train_smiles,
        rep,
        show_progress=show_progress,
        desc=f"Tokenize train[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )
    valid_tok = _encode_sequences_with_rep(
        valid_smiles,
        rep,
        show_progress=show_progress,
        desc=f"Tokenize valid[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )
    if not train_tok:
        raise ValueError("No train samples encoded successfully.")

    vocab = TokenVocab.build(
        train_tok,
        min_freq=min_token_freq,
        add_mask_token=(model_type == "diffusion"),
    )
    train_ids = _ids_with_special_tokens(train_tok, vocab)
    valid_ids = _ids_with_special_tokens(valid_tok, vocab)
    if model_type == "decoder":
        ds_train: Dataset[Any] = NextTokenDataset(train_ids, cfg.max_seq_len, vocab.pad_id)
        ds_valid: Dataset[Any] = NextTokenDataset(valid_ids, cfg.max_seq_len, vocab.pad_id)
    else:
        ds_train = DenoiseDataset(train_ids, cfg.max_seq_len)
        ds_valid = DenoiseDataset(valid_ids, cfg.max_seq_len)

    if len(ds_train) == 0:
        raise ValueError("Training dataset is empty after tokenization/padding.")

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    if model_type == "decoder":
        collate_fn = build_dynamic_lm_collate_fn(
            vocab.pad_id,
            pad_to_multiple_of=(8 if dev.type == "cuda" else None),
        )
    else:
        collate_fn = build_dynamic_denoise_collate_fn(
            vocab.pad_id,
            pad_to_multiple_of=(8 if dev.type == "cuda" else None),
        )
    pin_memory = dev.type == "cuda"
    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    valid_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    if cfg.num_data_workers > 0:
        pf = max(2, int(cfg.prefetch_factor))
        pw = bool(cfg.persistent_data_workers)
        train_loader_kwargs["prefetch_factor"] = pf
        train_loader_kwargs["persistent_workers"] = pw
        valid_loader_kwargs["prefetch_factor"] = pf
        valid_loader_kwargs["persistent_workers"] = pw

    loader_train = DataLoader(ds_train, **train_loader_kwargs)
    loader_valid = DataLoader(ds_valid, **valid_loader_kwargs)

    if model_type == "decoder":
        model: LMModel = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        model = TinyDiffusionLM(vocab_size=len(vocab.itos), cfg=cfg)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
            print("[train-lm] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[train-lm] torch.compile unavailable/fallback: {exc}", flush=True)
    model.to(dev)
    n_model_params = int(sum(p.numel() for p in model.parameters()))
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
        print(f"[train-lm] device={dev} ({gpu_name})", flush=True)
    else:
        print(f"[train-lm] device={dev}", flush=True)
    print(f"[train-lm] n_model_params={n_model_params} (~{n_model_params/1e6:.2f}M)", flush=True)
    print(
        f"[train-lm] amp={amp_label}, tf32={bool(cfg.tf32)}, "
        f"num_data_workers={max(0, int(cfg.num_data_workers))}, prefetch_factor={max(2, int(cfg.prefetch_factor))}",
        flush=True,
    )
    if model_type == "decoder":
        train_effective_max_seq_len = max((len(x) for x, _ in ds_train.samples), default=0)  # type: ignore[attr-defined]
        valid_effective_max_seq_len = max((len(x) for x, _ in ds_valid.samples), default=0)  # type: ignore[attr-defined]
    else:
        train_effective_max_seq_len = max((len(x) for x in ds_train.samples), default=0)  # type: ignore[attr-defined]
        valid_effective_max_seq_len = max((len(x) for x in ds_valid.samples), default=0)  # type: ignore[attr-defined]
    print(
        f"[train-lm] dynamic_padding=batch_max (cfg_max_seq_len={cfg.max_seq_len}, "
        f"train_effective_max_seq_len={train_effective_max_seq_len}, "
        f"valid_effective_max_seq_len={valid_effective_max_seq_len})",
        flush=True,
    )
    print(f"[train-lm] model_type={model_type}", flush=True)
    if model_type == "diffusion":
        print(
            "[train-lm] diffusion_cfg="
            f"steps={int(cfg.diffusion_steps)}, scheduler={cfg.diffusion_scheduler}, "
            f"time_epsilon={float(cfg.diffusion_time_epsilon):.4g}, "
            f"loss_weight={cfg.diffusion_loss_weight}",
            flush=True,
        )

    adamw_kwargs: Dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover
    non_blocking = dev.type == "cuda"

    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: List[float] = []

        train_iter = iterable_with_progress(
            loader_train,
            show_progress=show_progress,
            total=len(loader_train),
            desc=f"Train[{representation}] ep{epoch}",
        )
        for x, y in train_iter:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                if model_type == "decoder":
                    logits = model(x)  # type: ignore[operator]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                else:
                    loss = _compute_diffusion_loss(
                        model,  # type: ignore[arg-type]
                        clean_ids=x,
                        labels=y,
                        cfg=cfg,
                        mask_id=vocab.mask_id,
                    )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp
                    else nullcontext()
                )
                with amp_ctx:
                    if model_type == "decoder":
                        logits = model(x)  # type: ignore[operator]
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                    else:
                        loss = _compute_diffusion_loss(
                            model,  # type: ignore[arg-type]
                            clean_ids=x,
                            labels=y,
                            cfg=cfg,
                            mask_id=vocab.mask_id,
                        )
                valid_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_ppl": float(math.exp(train_loss)) if not math.isnan(train_loss) else float("nan"),
                "valid_ppl": float(math.exp(valid_loss)) if not math.isnan(valid_loss) else float("nan"),
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "cfg": vars(cfg),
            "representation": representation,
            "model_type": model_type,
        },
        out_dir / "model.pt",
    )
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "representation": representation,
                "train_size": len(train_smiles),
                "valid_size": len(valid_smiles),
                "n_train_encoded": len(train_tok),
                "n_valid_encoded": len(valid_tok),
                "vocab_size": len(vocab.itos),
                "model_type": model_type,
                "cfg": vars(cfg),
                "device": str(dev),
                "n_model_params": n_model_params,
                "padding_mode": "dynamic_batch_max",
                "train_effective_max_seq_len": train_effective_max_seq_len,
                "valid_effective_max_seq_len": valid_effective_max_seq_len,
                "tokenize_workers": encode_workers,
                "tokenize_chunksize": max(1, encode_chunksize),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "history": history,
        "vocab_size": len(vocab.itos),
        "device": str(dev),
        "n_model_params": n_model_params,
    }


def load_trained_model(
    model_dir: str | Path,
    device: Optional[str] = None,
) -> Tuple[LMModel, TokenVocab, BaseRepresentation, torch.device]:
    model_path = Path(model_dir) / "model.pt"
    vocab_path = Path(model_dir) / "vocab.json"

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = LMConfig(**ckpt["cfg"])
    model_type = _normalize_model_type(str(ckpt.get("model_type", cfg.model_type)))
    cfg.model_type = model_type
    rep_name = ckpt["representation"]

    vocab = TokenVocab.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    if model_type == "decoder":
        model: LMModel = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        model = TinyDiffusionLM(vocab_size=len(vocab.itos), cfg=cfg)
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)

    dev = _resolve_torch_device(device)
    model.to(dev)
    model.eval()

    rep = build_representation(rep_name)
    return model, vocab, rep, dev


def _sample_next_tokens(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, idx = torch.topk(logits, k=k, dim=-1)
        probs = torch.softmax(v, dim=-1)
        rel = torch.multinomial(probs, num_samples=1)
        token = idx.gather(1, rel).squeeze(1)
    else:
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return token


def _apply_top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k=k, dim=-1)
    cutoff = v[..., -1:].expand_as(logits)
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise.clamp_min(1e-12))) ** temperature
    return logits.exp() / gumbel_noise


def _trim_special_tail(row: List[int], vocab: TokenVocab) -> List[int]:
    cut = len(row)
    for token_id in (vocab.eos_id, vocab.pad_id):
        if token_id in row:
            cut = min(cut, row.index(token_id))
    return row[:cut]


def _sample_model_sequences_decoder(
    model: TinyDecoderLM,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> List[List[str]]:
    dev = device or next(model.parameters()).device
    prompt_ids = vocab.encode_tokens(list(prompt_tokens or []))
    prefix = [vocab.bos_id] + prompt_ids

    if n_samples <= 0:
        return []

    ids = torch.tensor(prefix, dtype=torch.long, device=dev).unsqueeze(0).repeat(n_samples, 1)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=dev)

    for _step in range(max_new_tokens):
        ctx = ids[:, -model.cfg.max_seq_len :]
        with torch.no_grad():
            logits = model(ctx)[:, -1, :]
        for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id):
            if 0 <= int(suppressed_id) < logits.size(-1):
                logits[:, int(suppressed_id)] = float("-inf")
        next_ids = _sample_next_tokens(logits, temperature=temperature, top_k=top_k)
        eos_fill = torch.full_like(next_ids, vocab.eos_id)
        next_ids = torch.where(finished, eos_fill, next_ids)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == vocab.eos_id)
        if bool(torch.all(finished).item()):
            break

    out: List[List[str]] = []
    all_ids = ids[:, 1:].detach().cpu().tolist()
    for row in all_ids:
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def _sample_model_sequences_diffusion(
    model: TinyDiffusionLM,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_block_size: Optional[int] = None,
    diffusion_remasking: Optional[str] = None,
    diffusion_stochastic_transfer: Optional[bool] = None,
) -> List[List[str]]:
    if n_samples <= 0:
        return []
    if vocab.mask_id == vocab.unk_id and TokenVocab.MASK not in vocab.stoi:
        raise ValueError(
            "Diffusion sampling requires a vocab with <mask> token. Please retrain with --model-type diffusion."
        )

    dev = device or next(model.parameters()).device
    prompt_ids = vocab.encode_tokens(list(prompt_tokens or []))
    prefix = [vocab.bos_id] + prompt_ids
    prefix = prefix[: model.cfg.max_seq_len]
    if len(prefix) >= model.cfg.max_seq_len:
        max_new_tokens = 0
    else:
        max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - len(prefix))

    total_len = len(prefix) + max_new_tokens
    if total_len <= 0:
        return [[] for _ in range(n_samples)]

    x = torch.full((n_samples, total_len), vocab.eos_id, dtype=torch.long, device=dev)
    x[:, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=dev).unsqueeze(0).repeat(n_samples, 1)
    if max_new_tokens > 0:
        x[:, len(prefix) :] = vocab.mask_id

    steps = max(1, int(diffusion_steps if diffusion_steps is not None else model.cfg.diffusion_steps))
    block_size = diffusion_block_size if diffusion_block_size is not None else model.cfg.diffusion_block_size
    if block_size is None or int(block_size) <= 0:
        block_size = max(1, max_new_tokens)
    block_size = max(1, int(block_size))
    remasking = (diffusion_remasking or model.cfg.diffusion_remasking or "low_confidence").strip().lower()
    stochastic_transfer = bool(
        model.cfg.diffusion_stochastic_transfer
        if diffusion_stochastic_transfer is None
        else diffusion_stochastic_transfer
    )

    if max_new_tokens > 0:
        num_blocks = int(math.ceil(max_new_tokens / block_size))
        steps_per_block = int(math.ceil(steps / num_blocks))

        for b in range(num_blocks):
            start = len(prefix) + b * block_size
            end = min(start + block_size, total_len)
            block_mask_index = x[:, start:end] == vocab.mask_id
            num_transfer_tokens = _get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=model.cfg.diffusion_scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = int(num_transfer_tokens.size(1))
            for i in range(effective_steps):
                mask_index = x == vocab.mask_id
                if not bool(mask_index.any().item()):
                    break
                time_ratio = max(float(model.cfg.diffusion_time_epsilon), float(effective_steps - i) / float(effective_steps))
                t = torch.full((n_samples,), time_ratio, device=dev, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(x, timesteps=t)
                for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id):
                    if 0 <= int(suppressed_id) < logits.size(-1):
                        logits[:, :, suppressed_id] = float("-inf")
                logits = _apply_top_k_logits(logits, top_k=top_k)
                logits_for_decode = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_for_decode, dim=-1).to(dtype=torch.long)

                if remasking == "low_confidence":
                    p = torch.softmax(logits, dim=-1)
                    x0_conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_conf = torch.rand((n_samples, total_len), device=dev)
                else:
                    raise ValueError(
                        f"Unsupported diffusion_remasking: {remasking}. Expected low_confidence|random."
                    )

                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_conf, torch.full_like(x0_conf, float("-inf")))
                conf[:, :start] = float("-inf")
                conf[:, end:] = float("-inf")

                transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                for row_idx in range(n_samples):
                    k = int(num_transfer_tokens[row_idx, i].item())
                    if k <= 0:
                        continue
                    finite_count = int(torch.isfinite(conf[row_idx]).sum().item())
                    if finite_count <= 0:
                        continue
                    k = min(k, finite_count)
                    _, select_idx = torch.topk(conf[row_idx], k=k)
                    transfer_index[row_idx, select_idx] = True

                x[transfer_index] = x0[transfer_index]

    x = torch.where(x == vocab.mask_id, torch.full_like(x, vocab.eos_id), x)
    out: List[List[str]] = []
    all_ids = x[:, 1:].detach().cpu().tolist()
    for row in all_ids:
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def sample_model_sequences(
    model: LMModel,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_block_size: Optional[int] = None,
    diffusion_remasking: Optional[str] = None,
    diffusion_stochastic_transfer: Optional[bool] = None,
) -> List[List[str]]:
    model_type = _normalize_model_type(getattr(model.cfg, "model_type", "decoder"))
    if model_type == "decoder":
        return _sample_model_sequences_decoder(
            model=model,  # type: ignore[arg-type]
            vocab=vocab,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt_tokens,
            device=device,
        )
    return _sample_model_sequences_diffusion(
        model=model,  # type: ignore[arg-type]
        vocab=vocab,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        prompt_tokens=prompt_tokens,
        device=device,
        diffusion_steps=diffusion_steps,
        diffusion_block_size=diffusion_block_size,
        diffusion_remasking=diffusion_remasking,
        diffusion_stochastic_transfer=diffusion_stochastic_transfer,
    )

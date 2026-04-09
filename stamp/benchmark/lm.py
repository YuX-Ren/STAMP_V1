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


class TokenVocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self.pad_id = stoi[self.PAD]
        self.bos_id = stoi[self.BOS]
        self.eos_id = stoi[self.EOS]
        self.unk_id = stoi[self.UNK]

    @classmethod
    def build(cls, tokenized_sequences: Iterable[Sequence[str]], min_freq: int = 1) -> "TokenVocab":
        cnt = Counter()
        for seq in tokenized_sequences:
            cnt.update(seq)

        vocab = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        specials = {cls.PAD, cls.BOS, cls.EOS, cls.UNK}
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

    vocab = TokenVocab.build(train_tok, min_freq=min_token_freq)
    train_ids = _ids_with_special_tokens(train_tok, vocab)
    valid_ids = _ids_with_special_tokens(valid_tok, vocab)
    ds_train = NextTokenDataset(train_ids, cfg.max_seq_len, vocab.pad_id)
    ds_valid = NextTokenDataset(valid_ids, cfg.max_seq_len, vocab.pad_id)

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

    collate_fn = build_dynamic_lm_collate_fn(
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

    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
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
    train_effective_max_seq_len = max((len(x) for x, _ in ds_train.samples), default=0)
    valid_effective_max_seq_len = max((len(x) for x, _ in ds_valid.samples), default=0)
    print(
        f"[train-lm] dynamic_padding=batch_max (cfg_max_seq_len={cfg.max_seq_len}, "
        f"train_effective_max_seq_len={train_effective_max_seq_len}, "
        f"valid_effective_max_seq_len={valid_effective_max_seq_len})",
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
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

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
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
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


def load_trained_model(model_dir: str | Path, device: Optional[str] = None) -> Tuple[TinyDecoderLM, TokenVocab, BaseRepresentation, torch.device]:
    model_path = Path(model_dir) / "model.pt"
    vocab_path = Path(model_dir) / "vocab.json"

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = LMConfig(**ckpt["cfg"])
    rep_name = ckpt["representation"]

    vocab = TokenVocab.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
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


def sample_model_sequences(
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
        if vocab.eos_id in row:
            row = row[: row.index(vocab.eos_id)]
        out.append(vocab.decode_ids(row))
    return out

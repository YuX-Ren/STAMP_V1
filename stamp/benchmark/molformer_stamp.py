from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .hybrid_vocab import smiles_char_tokens
from .lm import TokenVocab


@dataclass
class MolFormerSTAMPConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 768
    dropout: float = 0.2
    max_seq_len: int = 1024
    num_random_features: int = 32
    vocab_size: int = 0
    pad_id: int = 0
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MolFormerSTAMPConfig":
        fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.shape[-1]
    if dim % 2 != 0:
        return q, k
    seq_len = q.shape[-2]
    pos = torch.arange(seq_len, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=q.dtype) / dim))
    freqs = torch.einsum("t,d->td", pos, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1).unsqueeze(0).unsqueeze(0)
    cos = emb.cos()
    sin = emb.sin()
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LinearRotarySelfAttention(nn.Module):
    """MolFormer-style linear attention with rotary embeddings.

    IBM MolFormer uses fast-transformers FAVOR+/linear attention with rotary
    embeddings. This module keeps the same asymptotic behavior and RoPE signal,
    but avoids the unmaintained Apex/fast-transformers runtime dependency.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.eps = float(eps)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rotary(q, k)
        q = self._feature_map(q)
        k = self._feature_map(k)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, :, None].to(dtype=k.dtype)
            k = k * key_mask
            v = v * key_mask

        kv = torch.einsum("bhtd,bhte->bhde", k, v)
        k_sum = k.sum(dim=2)
        denom = torch.einsum("bhtd,bhd->bht", q, k_sum).clamp_min(self.eps)
        out = torch.einsum("bhtd,bhde,bht->bhte", q, kv, denom.reciprocal())
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.dropout(self.proj(out))


class MolFormerBlock(nn.Module):
    def __init__(self, cfg: MolFormerSTAMPConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = LinearRotarySelfAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ff(self.ln_2(x))
        return x


class MolFormerEncoder(nn.Module):
    def __init__(self, cfg: MolFormerSTAMPConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("cfg.vocab_size must be set.")
        self.cfg = cfg
        self.pad_id = int(cfg.pad_id)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=self.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MolFormerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, *, return_cls: bool = False) -> torch.Tensor:
        attention_mask = input_ids.ne(self.pad_id)
        x = self.drop(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        if return_cls:
            return x[:, 0, :]
        return x


class MolFormerMLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense(x))
        x = self.ln(x)
        return self.proj(x)


def read_stamp_vocab_tokens(path: str | Path) -> list[str]:
    out: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        first = f.readline()
        try:
            json.loads(first)
        except Exception:
            if first.strip():
                out.append(first.split("\t")[0].strip())
        for line in f:
            if line.strip():
                out.append(line.split("\t")[0].strip())
    return [tok for tok in out if tok]


def smiles_vocab_tokens_from_stamp_vocabs(vocab_paths: Sequence[str | Path]) -> list[str]:
    tokens = {"(", ")", "=", "#", "-", "+", "/", "\\", "@", ".", ":", "$", "*"}
    tokens.update(str(i) for i in range(10))
    for path in vocab_paths:
        for tok in read_stamp_vocab_tokens(path):
            if TokenVocab._is_stamp_structural(tok) or tok in {"[BOS]", "[EOS]", "[CONCAT]"}:
                continue
            try:
                tokens.update(smiles_char_tokens(tok))
            except ValueError:
                # Keep vocab construction robust for unusual motif strings;
                # those tokens can still be represented by their original id.
                continue
    return sorted(tokens)


def default_stamp_structural_tokens(max_index: int = 256) -> list[str]:
    toks = ["[END]", "[BACK]", "[Ring]", "[CONCAT]", "[BOS]", "[EOS]", "[.]"]
    toks += ["[S_R]", "[S_S]", "[S_E]", "[S_Z]"]
    toks += [f"[J_{i}]" for i in range(max_index)]
    toks += [f"[B_{i}]" for i in range(max_index)]
    return toks


def build_stamp_molformer_vocab(
    vocab_paths: Sequence[str | Path],
    *,
    extra_tokens: Sequence[str] = (),
    include_smiles_tokens: bool = False,
    structural_max_index: int = 256,
) -> TokenVocab:
    vocab = [TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK, TokenVocab.MASK]
    added = set(vocab)
    if include_smiles_tokens:
        extra_tokens = list(extra_tokens) + smiles_vocab_tokens_from_stamp_vocabs(vocab_paths)
    for tok in list(extra_tokens) + default_stamp_structural_tokens(structural_max_index):
        if tok not in added:
            vocab.append(tok)
            added.add(tok)
    for path in vocab_paths:
        for tok in read_stamp_vocab_tokens(path):
            if tok not in added:
                vocab.append(tok)
                added.add(tok)
    return TokenVocab(stoi={tok: i for i, tok in enumerate(vocab)}, itos=vocab)


def strip_literal_wrappers(tokens: Sequence[str]) -> list[str]:
    toks = [str(t) for t in tokens if str(t)]
    if toks and toks[0] == "[BOS]":
        toks = toks[1:]
    if toks and toks[-1] == "[EOS]":
        toks = toks[:-1]
    return toks


def split_concat_views(tokens: Sequence[str]) -> list[list[str]]:
    views: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok == "[CONCAT]":
            if current:
                views.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        views.append(current)
    return views


def expand_stamp_views(tokens: Sequence[str], mode: str) -> list[list[str]]:
    toks = list(tokens)
    if mode == "concat_only":
        return [toks]
    if mode == "single_views":
        views = split_concat_views(toks)
        return views if views else [toks]
    if mode == "cams_multiview":
        views = split_concat_views(toks)
        if len(views) <= 1:
            return [toks]
        return views + [toks]
    raise ValueError(f"Unsupported view mode: {mode}")


def encode_stamp_inner_tokens(
    tokens: Sequence[str],
    vocab: TokenVocab,
    *,
    fallback_oov_to_smiles: bool = False,
    motif_smiles_fallback_prob: float = 0.0,
    rng: random.Random | None = None,
) -> list[int]:
    ids: list[int] = []
    rnd = rng if rng is not None else random
    for tok in strip_literal_wrappers(tokens):
        if tok in vocab.stoi:
            if (
                motif_smiles_fallback_prob > 0.0
                and not TokenVocab._is_stamp_structural(tok)
                and tok not in {TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK, TokenVocab.MASK}
                and rnd.random() < motif_smiles_fallback_prob
            ):
                try:
                    ids.extend(vocab.stoi.get(t, vocab.unk_id) for t in smiles_char_tokens(tok))
                    continue
                except ValueError:
                    pass
            ids.append(vocab.stoi[tok])
            continue
        if fallback_oov_to_smiles:
            try:
                ids.extend(vocab.stoi.get(t, vocab.unk_id) for t in smiles_char_tokens(tok))
                continue
            except ValueError:
                pass
        ids.append(vocab.unk_id)
    return ids


def ids_from_stamp_tokens(
    tokens: Sequence[str],
    vocab: TokenVocab,
    max_seq_len: int,
    *,
    fallback_oov_to_smiles: bool = False,
    motif_smiles_fallback_prob: float = 0.0,
    rng: random.Random | None = None,
) -> list[int]:
    inner = strip_literal_wrappers(tokens)
    ids = [vocab.bos_id] + encode_stamp_inner_tokens(
        inner,
        vocab,
        fallback_oov_to_smiles=fallback_oov_to_smiles,
        motif_smiles_fallback_prob=motif_smiles_fallback_prob,
        rng=rng,
    ) + [vocab.eos_id]
    return ids[:max_seq_len]


def token_ids_are_structural(vocab: TokenVocab) -> set[int]:
    return {i for i, tok in enumerate(vocab.itos) if TokenVocab._is_stamp_structural(tok) or tok in {"[BOS]", "[EOS]"}}


def save_molformer_checkpoint(
    path: str | Path,
    *,
    encoder: nn.Module,
    mlm_head: nn.Module | None,
    cfg: MolFormerSTAMPConfig,
    vocab: TokenVocab,
    global_step: int,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    enc = encoder.module if hasattr(encoder, "module") else encoder
    head = mlm_head.module if (mlm_head is not None and hasattr(mlm_head, "module")) else mlm_head
    payload: dict[str, Any] = {
        "model_state": enc.state_dict(),
        "mlm_state": head.state_dict() if head is not None else None,
        "config": cfg.to_dict(),
        "vocab": vocab.to_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_molformer_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    load_mlm_head: bool = False,
) -> tuple[MolFormerEncoder, MolFormerMLMHead | None, TokenVocab, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    vocab = TokenVocab.from_dict(payload["vocab"])
    cfg = MolFormerSTAMPConfig.from_dict(payload["config"])
    cfg.vocab_size = len(vocab.itos)
    cfg.pad_id = vocab.pad_id
    encoder = MolFormerEncoder(cfg)
    encoder.load_state_dict(payload["model_state"], strict=True)
    mlm_head = None
    if load_mlm_head and payload.get("mlm_state") is not None:
        mlm_head = MolFormerMLMHead(cfg.d_model, len(vocab.itos))
        mlm_head.load_state_dict(payload["mlm_state"], strict=True)
    return encoder, mlm_head, vocab, payload


class Lamb(torch.optim.Optimizer):
    """Minimal LAMB optimizer for MolFormer-style large-batch MLM."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients.")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt().add_(eps))
                if wd:
                    update = update.add(p, alpha=wd)
                w_norm = torch.linalg.vector_norm(p)
                u_norm = torch.linalg.vector_norm(update)
                trust_ratio = torch.where(
                    (w_norm > 0) & (u_norm > 0),
                    w_norm / u_norm,
                    torch.ones_like(w_norm),
                )
                p.add_(update, alpha=-lr * float(trust_ratio.clamp(max=10.0)))
        return loss

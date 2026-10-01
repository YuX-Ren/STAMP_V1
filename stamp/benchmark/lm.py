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
    diffusion_tokens_per_step: int = 0
    diffusion_chemical_role_masking: bool = False
    diffusion_structure_constraints: bool = False
    # When False, skip AdaLN modulation and the none_emb/target_pos_emb machinery
    # used by AO-GPT. Standard masked-diffusion (MDM/MDLM) sets this to False.
    use_adaln: bool = True
    # ao_gpt_pp (property-prediction) type-aware masking. Only consulted when
    # model_type == "ao_gpt_pp"; the plain "ao_gpt" path ignores these.
    ao_pp_frag_mask_rate: float = 0.20
    ao_pp_topo_mask_rate: float = 0.05
    # Per-batch permutation policy mixture for ao_gpt_pp:
    #   p_uniform: run existing _sample_ao_orders (random AO)
    #   p_type_aware: run type-aware biased ordering
    #   p_identity: keep identity order (L->R GPT semantics)
    ao_pp_mix_uniform: float = 0.50
    ao_pp_mix_type_aware: float = 0.40
    ao_pp_mix_identity: float = 0.10
    # AO-GPT training: fraction of batches that use a RANDOM AO permutation
    # (vs identity). Default 0.9 = mostly random orders (the prior recipe).
    # Set to 0 to train purely as AR (matches AR inference, recommended for
    # downstream tasks where generation is left-to-right).
    ao_random_ratio: float = 0.9
    # Tie input token embedding with output head weight. Standard for modern
    # LMs; saves vocab*d_model params (e.g. 51K*1024 = 52M for d=1024).
    tied_embeddings: bool = False


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

    # Regex for STAMP structural tokens that should always be in-vocab.
    _STRUCT_RE = None

    @classmethod
    def _is_stamp_structural(cls, tok: str) -> bool:
        if cls._STRUCT_RE is None:
            import re
            cls._STRUCT_RE = re.compile(
                r"^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring)\]$"
            )
        return bool(cls._STRUCT_RE.match(tok)) or tok in {"[END]", "[BACK]"}

    @classmethod
    def build(
        cls,
        tokenized_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        add_mask_token: bool = False,
        seed_tokens: Optional[Sequence[str]] = None,
        closed_vocab: bool = False,
    ) -> "TokenVocab":
        cnt: Counter = Counter()
        for seq in tokenized_sequences:
            cnt.update(seq)

        vocab = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        if add_mask_token:
            vocab.append(cls.MASK)

        if seed_tokens is not None:
            added = set(vocab)
            for tok in seed_tokens:
                if tok not in added:
                    vocab.append(tok)
                    added.add(tok)

        added = set(vocab)
        for tok, freq in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and tok not in added:
                if closed_vocab and not cls._is_stamp_structural(tok):
                    continue
                vocab.append(tok)
                added.add(tok)

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


_ADALN_COND_DIM = 128


# ============================================================================
# Llama-style AR GPT — clean baseline for left-to-right AR training.
# No AO-GPT machinery (target_pos_emb, AdaLN, shuffle, none_emb). Uses RoPE,
# RMSNorm, SwiGLU FFN, tied embeddings, no biases. Use cfg.model_type='llama'.
# ============================================================================


def _precompute_rope_freqs(d_head: int, max_seq_len: int, base: float = 10000.0,
                          dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (max_seq_len, d_head)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    # duplicate to full d_head (interleaved chunks for [x1; x2] rotation)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D). Apply RoPE rotation."""
    T = x.size(-2)
    cos_t = cos[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin_t = sin[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos_t + rotated * sin_t


class _LlamaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = _RMSNorm(self.d_head)
        self.k_norm = _RMSNorm(self.d_head)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(y))


class _LlamaSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class _LlamaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.attn = _LlamaAttention(d_model, n_heads, dropout)
        self.ffn_norm = _RMSNorm(d_model)
        self.ffn = _LlamaSwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaDecoderLM(nn.Module):
    """Clean Llama-style AR GPT: RoPE + RMSNorm + SwiGLU + tied embeddings,
    no biases. Standard left-to-right causal attention. Use this for AR
    training/inference; far simpler than the AO-GPT TinyDecoderLM and
    eliminates the train↔inference paradigm mismatch (random AO orders during
    training vs identity AR at inference)."""

    def __init__(self, vocab_size: int, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            _LlamaBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = _RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Tied embeddings (Llama / GPT-2 default)
        self.head.weight = self.tok_emb.weight
        # RoPE freqs
        cos, sin = _precompute_rope_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith(("out.weight", "down_proj.weight")):
                nn.init.trunc_normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers), a=-0.06, b=0.06)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """AR forward. Ignores `orders` kwarg if passed (Llama is pure AR)."""
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")
        x = self.drop(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.head(x)


def _compute_ar_loss(model, clean_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE for AR LM. labels has -100 at positions to ignore
    (PAD, BOS already shifted out by the data pipeline)."""
    logits = model(clean_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _RMSNorm(nn.Module):
    def __init__(self, ndim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, 1e-5)


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.q_norm = _RMSNorm(d_model // n_heads)
        self.k_norm = _RMSNorm(d_model // n_heads)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if attn_mask is not None:
            # explicit boolean mask overrides is_causal
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _AOBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_adaln: bool = True) -> None:
        super().__init__()
        self.use_adaln = use_adaln
        self.ln_1 = _RMSNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads, dropout)
        self.ln_2 = _RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True), nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True), nn.Dropout(dropout),
        )
        if use_adaln:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(_ADALN_COND_DIM, 6 * d_model, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        is_causal: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_adaln:
            s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(cond).chunk(6, dim=-1)
            x = x + g_a * self.attn(_modulate(self.ln_1(x), s_a, sc_a), is_causal=is_causal, attn_mask=attn_mask)
            x = x + g_m * self.mlp(_modulate(self.ln_2(x), s_m, sc_m))
        else:
            x = x + self.attn(self.ln_1(x), is_causal=is_causal, attn_mask=attn_mask)
            x = x + self.mlp(self.ln_2(x))
        return x


class _AOFinalLayer(nn.Module):
    def __init__(self, d_model: int, use_adaln: bool = True) -> None:
        super().__init__()
        self.use_adaln = use_adaln
        self.ln_f = _RMSNorm(d_model)
        if use_adaln:
            self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(_ADALN_COND_DIM, 2 * d_model, bias=True))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_adaln:
            shift, scale = self.adaLN_final(cond).chunk(2, dim=-1)
            return _modulate(self.ln_f(x), shift, scale)
        return self.ln_f(x)


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: LMConfig, bidirectional: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.bidirectional = bidirectional
        self.use_adaln = bool(getattr(cfg, "use_adaln", True))
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        pos_len = cfg.max_seq_len + 1 if self.use_adaln else cfg.max_seq_len
        self.pos_emb = nn.Embedding(pos_len, cfg.d_model)
        if self.use_adaln:
            self.target_pos_emb = nn.Embedding(cfg.max_seq_len, _ADALN_COND_DIM)
            self.none_emb = nn.Embedding(1, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            _AOBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, use_adaln=self.use_adaln)
            for _ in range(cfg.n_layers)
        ])
        self.final_layer = _AOFinalLayer(cfg.d_model, use_adaln=self.use_adaln)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=True)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.trunc_normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers), a=-0.06, b=0.06)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def _shuffle(self, x: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = x.shape[:2]
        batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand(bsz, seqlen)
        return x[batch_idx, orders]

    def forward(self, input_ids: torch.Tensor, force_causal: bool = False, orders: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        device = input_ids.device
        bsz, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")

        use_causal = bool(force_causal or not self.bidirectional)

        if not self.use_adaln:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.tok_emb(input_ids)
            pos_emb = self.pos_emb(pos).unsqueeze(0).expand(bsz, -1, -1)
            x = self.drop(tok_emb + pos_emb)
            for block in self.blocks:
                x = block(x, None, is_causal=use_causal)
            x = self.final_layer(x, None)
            return self.head(x)

        pos = torch.arange(0, t + 1, dtype=torch.long, device=device)

        if orders is None:
            orders = torch.arange(t, device=device).unsqueeze(0).expand(bsz, t)

        idx = self._shuffle(input_ids, orders)
        tok_emb = self.tok_emb(idx)
        none_emb = self.none_emb(torch.zeros(bsz, 1, dtype=torch.long, device=device))
        tok_emb = torch.cat([none_emb, tok_emb], dim=1)

        pos_emb = self.pos_emb(pos).unsqueeze(0).expand(bsz, -1, -1)
        pos_emb_prefix = pos_emb[:, :1]
        pos_emb_postfix = self._shuffle(pos_emb[:, 1:], orders)

        tpe = self.target_pos_emb(pos[:t]).unsqueeze(0).expand(bsz, -1, -1)
        tpe_shuffled = self._shuffle(tpe, orders)
        tpe_zeros = torch.zeros(bsz, 1, _ADALN_COND_DIM, device=device)
        target_cond = torch.cat([tpe_zeros, tpe_shuffled], dim=1)

        x = self.drop(tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1))

        for block in self.blocks:
            x = block(x, target_cond, is_causal=use_causal)

        x = self.final_layer(x, target_cond)
        logits = self.head(x)
        return logits[:, 1:, :]


def _block_diff_mask(n: int, device: torch.device) -> torch.Tensor:
    """Block_diff attention mask for Eso-LM sequential branch (block_size=1).

    Operates on a concatenated [zt, x0] sequence of length 2n. Each zt query
    attends to (i) itself (block_diagonal), (ii) x0 positions strictly earlier
    in sort order (offset_block_causal). Each x0 query attends causally to
    itself and earlier x0 positions (block_causal). x0 positions never attend
    to zt positions. Returns a bool tensor of shape [2n, 2n].
    """
    total = 2 * n
    q_idx = torch.arange(total, device=device).unsqueeze(1)  # [2n, 1]
    kv_idx = torch.arange(total, device=device).unsqueeze(0)  # [1, 2n]
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q, q_idx - n, q_idx)
    block_kv = torch.where(x0_flag_kv, kv_idx - n, kv_idx)
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
    return (block_diagonal | offset_block_causal | block_causal)


class TinyEsoDiT(nn.Module):
    """Eso-LM backbone (arXiv 2506.01928).

    Sequential branch (used when some positions are kept clean, alpha_0 < 1):
      input = concat([zt_sorted, x0_sorted]) of length 2n
      attention = block_diff_mask (block_size=1)
        - zt[q] sees self + x0[0..q-1]
        - x0[q] sees causally over x0 half only
      output = logits at first n positions (zt positions)

    Diffusion branch (alpha_0 = 1, pure MDM):
      input = zt_sorted of length n
      attention = bidirectional
      output = logits [B, n, V]

    Positional embeddings are learned and gathered by sort_idx so that each
    sorted position carries its original-layout positional signature. The
    zt and x0 halves share the same per-position pos embs.
    """

    def __init__(self, vocab_size: int, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            _AOBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, use_adaln=False)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = _RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=True)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.trunc_normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers), a=-0.06, b=0.06)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(
        self,
        zt: torch.Tensor,
        x0: Optional[torch.Tensor],
        sort_idx: torch.Tensor,
        mode: str = "sequential",
    ) -> torch.Tensor:
        device = zt.device
        B, n = zt.shape
        if n > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {n} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos_ids = sort_idx  # [B, n]; each position carries its original index's pos_emb
        pos_embs = self.pos_emb(pos_ids)  # [B, n, d]

        if mode == "diffusion":
            tok = self.tok_emb(zt) + pos_embs
            x = self.drop(tok)
            for block in self.blocks:
                # bidirectional: pass a 1 mask (all-True = bidirectional) or skip via is_causal=False
                x = block(x, cond=None, is_causal=False, attn_mask=None)
            x = self.final_norm(x)
            return self.head(x)

        # sequential mode: concat [zt, x0], run block_diff_mask attention
        if x0 is None:
            raise ValueError("sequential mode requires x0")
        tok_zt = self.tok_emb(zt) + pos_embs
        tok_x0 = self.tok_emb(x0) + pos_embs
        x = self.drop(torch.cat([tok_zt, tok_x0], dim=1))  # [B, 2n, d]

        attn_mask = _block_diff_mask(n, device=device)  # [2n, 2n] bool
        # Expand for SDPA: broadcasting handles batch+heads
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 2n, 2n]

        for block in self.blocks:
            x = block(x, cond=None, is_causal=False, attn_mask=attn_mask)
        x = self.final_norm(x)
        logits = self.head(x)
        return logits[:, :n, :]  # first half = zt positions


def _compute_sort_idx(
    xt: torch.Tensor,
    mask_id: int,
    shuffle_unmasked: bool = False,
) -> torch.Tensor:
    """Compute sort index so unmasked tokens come first (shuffled or sequential)
    and masked tokens come last in sequential order within each row (per the
    official Eso-LM sort_indices algorithm)."""
    B, n = xt.shape
    device = xt.device
    masked = (xt == mask_id)
    if shuffle_unmasked:
        offsets = torch.rand((B, n), device=device) * 0.9
    else:
        offsets = torch.linspace(0.0, 0.9, n, device=device).unsqueeze(0).expand(B, n).clone()
    # Masks get a monotone offset in [0, 1] so they sort after the unmasked block
    # (which has offset < 0.9). Within each row, earlier masked indices get
    # smaller offsets so they end up sequentially at the tail.
    total_masked = int(masked.sum().item())
    if total_masked > 0:
        linspace = torch.linspace(0.0, 1.0, total_masked, device=device)
        offsets_flat = offsets.reshape(-1).clone()
        masked_flat = masked.reshape(-1)
        offsets_flat[masked_flat] = linspace
        offsets = offsets_flat.view(B, n)
    keys = masked.to(torch.float32) + offsets
    return keys.argsort(dim=-1).to(torch.long)


def _compute_eso_loss_v2(
    model: TinyEsoDiT,
    x0: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    mask_id: int,
    alpha_0: float = 0.0,
    shuffle_unmasked: bool = False,
) -> torch.Tensor:
    """Full Eso-LM sequential-branch training loss (mirrors the official
    `_reconstruction_loss`).

    Uses fixed alpha_0 as the mask rate boundary (NOT a sampled t). With
    prob (1 - alpha_0), each position is masked in zt. Sort so unmasked
    tokens come first (shuffled or sequential) and masked tokens come last
    in sequential order. Forward through TinyEsoDiT with block_diff_mask
    attention over [zt, x0]. Zero-mask trick suppresses mask_id in the head
    output. Loss is raw NLL (no scheduler weight) at masked positions only.
    """
    B, n = x0.shape
    device = x0.device

    maskable = labels != -100

    # At alpha_0 mask rate: each maskable position is masked independently
    # with probability (1 - alpha_0). alpha_0 = 0 -> all masked (pure AR),
    # alpha_0 = 0.5 -> half masked, etc.
    p_mask = float(1.0 - alpha_0)
    mask_draw = (torch.rand((B, n), device=device) < p_mask) & maskable

    # Force at least one mask per row that has maskable positions
    any_maskable = maskable.any(dim=1)
    any_masked = mask_draw.any(dim=1)
    need_fix = any_maskable & (~any_masked)
    if bool(need_fix.any().item()):
        row_idx = torch.where(need_fix)[0]
        first_pos = torch.argmax(maskable[row_idx].to(torch.float32), dim=1)
        mask_draw[row_idx, first_pos] = True

    xt = torch.where(mask_draw, torch.full_like(x0, mask_id), x0)

    sort_idx = _compute_sort_idx(xt, mask_id, shuffle_unmasked=shuffle_unmasked)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n)
    zt = xt[batch_idx, sort_idx]
    x0_sorted = x0[batch_idx, sort_idx]

    logits = model(zt, x0=x0_sorted, sort_idx=sort_idx, mode="sequential")
    # Zero-mask trick
    neg_inf = torch.finfo(logits.dtype).min / 2
    logits = logits.clone()
    logits[..., mask_id] = neg_inf
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, x0_sorted.unsqueeze(-1)).squeeze(-1)  # [B, n]

    loss_mask = (zt == mask_id).to(nll.dtype)
    denom = loss_mask.sum().clamp_min(1.0)
    return (nll * loss_mask).sum() / denom


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


def _get_num_transfer_tokens_by_budget(
    mask_index: torch.Tensor,
    tokens_per_step: int,
) -> torch.Tensor:
    budget = max(1, int(tokens_per_step))
    mask_num = mask_index.sum(dim=1).to(dtype=torch.long)
    rows: List[torch.Tensor] = []
    max_len = 0
    for i in range(mask_num.size(0)):
        n = int(mask_num[i].item())
        if n <= 0:
            row = torch.zeros(0, dtype=torch.long, device=mask_index.device)
        else:
            chunks: List[int] = []
            while n > 0:
                k = min(budget, n)
                chunks.append(k)
                n -= k
            row = torch.tensor(chunks, dtype=torch.long, device=mask_index.device)
        rows.append(row)
        max_len = max(max_len, int(row.numel()))
    if max_len == 0:
        return torch.zeros((mask_num.size(0), 0), dtype=torch.long, device=mask_index.device)
    padded_rows = []
    for r in rows:
        if int(r.numel()) < max_len:
            pad = torch.zeros(max_len - int(r.numel()), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad], dim=0)
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


LMModel = TinyDecoderLM


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
        initargs=(
            rep.name,
            getattr(rep, "_vocab_path", None),
            getattr(rep, "_force_ring_aware_encode", False),
        ),
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


def _init_encode_worker(
    representation: str,
    stamp_vocab: Optional[str] = None,
    stamp_force_ring_aware_encode: bool = False,
) -> None:
    global _WORKER_REP
    _WORKER_REP = build_representation(
        representation,
        stamp_vocab=stamp_vocab,
        stamp_force_ring_aware_encode=stamp_force_ring_aware_encode,
    )


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
    if mt not in {"decoder", "diffusion", "ao_gpt", "ao_gpt_pp", "eso", "eso_v2", "llama"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected one of: decoder, diffusion, ao_gpt, ao_gpt_pp, eso, eso_v2, llama")
    return mt


def _compute_ao_gpt_loss(
    model: TinyDecoderLM,
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    random_ratio: float = 0.9,
) -> torch.Tensor:
    """AO-GPT loss: any-order autoregressive with random permutations.

    The model's forward() handles shuffling internally and prepends [None].
    Returns shifted logits (length t) predicting shuffled targets.
    """
    bsz, seqlen = clean_ids.shape
    device = clean_ids.device

    orders = _sample_ao_orders(labels, random_ratio=random_ratio)

    logits = model(clean_ids, orders=orders)

    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)
    targets = labels[batch_idx, orders]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss


def _sample_ao_orders(labels: torch.Tensor, random_ratio: float = 0.9) -> torch.Tensor:
    """Sample AO-GPT permutations over valid positions only.

    First valid stays first (BOS context), last valid stays last (EOS aligned
    with left-to-right decoding), middle valids are uniformly shuffled, and
    ignored (-100) positions are appended in their original order.

    Fully vectorized argsort-based impl: O(1) Python, no per-row launches.
    """
    bsz, seqlen = labels.shape
    device = labels.device
    valid_mask = labels != -100                                   # (bsz, seq)
    n_valid = valid_mask.sum(dim=-1)                              # (bsz,)

    # Per-row: permute vs identity (matches original random.random() per row,
    # and the "<=2 valid" fallback to identity).
    gate = (torch.rand(bsz, device=device) < float(random_ratio)) & (n_valid > 2)

    # Base keys: uniform [0,1) for valid, [3 + i/seqlen) for invalid — invalids
    # share the tail and preserve original-index order via the monotone offset.
    rand_keys = torch.rand(bsz, seqlen, device=device)
    pad_keys = 3.0 + torch.arange(seqlen, device=device, dtype=rand_keys.dtype) / float(max(1, seqlen))
    keys = torch.where(valid_mask, rand_keys, pad_keys.unsqueeze(0).expand(bsz, -1))

    # Pin first valid to -1.0 (sorts first) and last valid to 2.0 (sorts after
    # the random middle keys but before invalid tail).
    first_idx = valid_mask.to(torch.int32).argmax(dim=-1)         # (bsz,)
    last_idx = (seqlen - 1) - valid_mask.flip(-1).to(torch.int32).argmax(dim=-1)
    rows = torch.arange(bsz, device=device)
    keys[rows, first_idx] = -1.0
    keys[rows, last_idx] = 2.0

    # Rows that don't permute: collapse to identity order via ascending keys.
    identity_keys = torch.arange(seqlen, device=device, dtype=keys.dtype).unsqueeze(0).expand(bsz, -1)
    keys = torch.where(gate.unsqueeze(-1), keys, identity_keys)

    return keys.argsort(dim=-1).to(torch.long)


def _sample_ao_orders_type_aware(
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    topology_id_tensor: torch.Tensor,
    frag_mask_rate: float = 0.20,
    topo_mask_rate: float = 0.05,
) -> torch.Tensor:
    """Type-aware AO permutation for property-prediction pretraining.

    Per position (valid, non-BOS, non-EOS):
      - If the token is a STAMP topology token, sample mask with prob
        ``topo_mask_rate``; else (fragment) use ``frag_mask_rate``.
      - Masked positions are placed at the END of the permutation (the
        model predicts them last, conditioned on full context).
      - Unmasked positions are shuffled among the front ("visible context").

    BOS stays first, EOS stays last, pad/ignored positions are appended
    unchanged — same contract as ``_sample_ao_orders``.
    """
    bsz, seqlen = labels.shape
    device = labels.device
    # Boolean mask: True = token is a topology token.
    is_topo = torch.zeros_like(clean_ids, dtype=torch.bool)
    if topology_id_tensor.numel() > 0:
        is_topo = torch.isin(clean_ids, topology_id_tensor)

    orders = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, seqlen).clone()

    for b in range(bsz):
        valid = torch.nonzero(labels[b] != -100, as_tuple=False).squeeze(1)
        invalid = torch.nonzero(labels[b] == -100, as_tuple=False).squeeze(1)
        if int(valid.numel()) <= 2:
            full_order = torch.cat([valid, invalid], dim=0)
            orders[b] = full_order
            continue

        middle = valid[1:-1]
        # Per-position is-topology flag for the middle.
        is_t_middle = is_topo[b, middle]
        # Bernoulli per position with type-specific rate.
        rates = torch.where(is_t_middle, torch.full_like(is_t_middle, 0, dtype=torch.float32) + topo_mask_rate,
                            torch.full_like(is_t_middle, 0, dtype=torch.float32) + frag_mask_rate)
        mask_decision = torch.rand(middle.shape, device=device) < rates

        visible = middle[~mask_decision]
        masked = middle[mask_decision]
        # Shuffle within each group for diversity.
        if visible.numel() > 0:
            perm = torch.randperm(visible.numel(), device=device)
            visible = visible[perm]
        if masked.numel() > 0:
            perm = torch.randperm(masked.numel(), device=device)
            masked = masked[perm]

        valid_order = torch.cat([valid[:1], visible, masked, valid[-1:]], dim=0)
        full_order = torch.cat([valid_order, invalid], dim=0)
        orders[b] = full_order
    return orders


def _compute_ao_gpt_pp_loss(
    model: TinyDecoderLM,
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    topology_id_tensor: torch.Tensor,
) -> torch.Tensor:
    """AO-GPT loss for property-prediction pretraining.

    Each batch samples ONE of three permutation policies according to the
    mixture in ``cfg``:
      - uniform  : existing random AO (same as ``_compute_ao_gpt_loss``)
      - type_aware: fragment/topology-biased masking, masked tokens go last
      - identity : left-to-right GPT order

    This preserves generation capability (majority uniform+identity) while
    routing a portion of training toward fragment in-fill / property prediction.
    """
    bsz, seqlen = clean_ids.shape
    device = clean_ids.device

    r = random.random()
    total = max(
        1e-9,
        cfg.ao_pp_mix_uniform + cfg.ao_pp_mix_type_aware + cfg.ao_pp_mix_identity,
    )
    p_uni = cfg.ao_pp_mix_uniform / total
    p_ta = p_uni + cfg.ao_pp_mix_type_aware / total

    if r < p_uni:
        orders = _sample_ao_orders(labels, random_ratio=0.9)
    elif r < p_ta:
        orders = _sample_ao_orders_type_aware(
            clean_ids,
            labels,
            topology_id_tensor,
            frag_mask_rate=cfg.ao_pp_frag_mask_rate,
            topo_mask_rate=cfg.ao_pp_topo_mask_rate,
        )
    else:
        # Identity order: same as random_ratio=0 fallback of uniform sampler.
        orders = _sample_ao_orders(labels, random_ratio=0.0)

    logits = model(clean_ids, orders=orders)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)
    targets = labels[batch_idx, orders]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss


def _compute_diffusion_loss(
    model: TinyDecoderLM,
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    mask_id: int,
    structural_ids: Optional[set] = None,
) -> torch.Tensor:
    """Masked diffusion loss with chemical-role-aware noise scheduling.

    When *structural_ids* is provided, each position gets a per-role
    noise exponent: structural tokens (exp=1.5) are masked last,
    motif tokens (exp=0.5) are masked first.  This turns the continuous
    diffusion into a chemistry-aware curriculum.
    """
    bsz, seqlen = clean_ids.shape
    maskable_mask = labels != -100
    t = cfg.diffusion_time_epsilon + (1.0 - cfg.diffusion_time_epsilon) * torch.rand(
        bsz, device=clean_ids.device
    )

    if structural_ids:
        is_struct = torch.zeros((bsz, seqlen), device=clean_ids.device, dtype=torch.bool)
        for sid in structural_ids:
            is_struct |= (clean_ids == sid)
        role_exp = torch.where(is_struct, 1.5, 0.5).to(dtype=torch.float32, device=clean_ids.device)
        p_mask = t.unsqueeze(1).expand(bsz, seqlen).pow(role_exp)
        masked_mask = (torch.rand((bsz, seqlen), device=clean_ids.device) < p_mask) & maskable_mask
    else:
        p_mask = 1.0 - _alpha_from_t(t, scheduler=cfg.diffusion_scheduler)
        masked_mask = (torch.rand((bsz, seqlen), device=clean_ids.device) < p_mask.unsqueeze(1)) & maskable_mask

    if not bool(masked_mask.any().item()):
        any_maskable = maskable_mask.any(dim=1)
        if bool(any_maskable.any().item()):
            row_idx = torch.where(any_maskable)[0]
            first_pos = torch.argmax(maskable_mask[row_idx].to(dtype=torch.float32), dim=1)
            masked_mask[row_idx, first_pos] = True

    noised_ids = torch.where(masked_mask, torch.full_like(clean_ids, mask_id), clean_ids)
    logits = model(noised_ids)
    token_nll = F.cross_entropy(logits.transpose(1, 2), clean_ids, reduction="none")

    if structural_ids:
        role_exp_t = torch.where(is_struct, 1.5, 0.5).to(dtype=torch.float32, device=clean_ids.device)
        alpha_vals = 1.0 - t.unsqueeze(1).expand(bsz, seqlen).pow(role_exp_t)
        loss_weights = 1.0 / (1.0 - alpha_vals + 1e-5)
    else:
        weight_mode = (cfg.diffusion_loss_weight or "scheduler").strip().lower()
        if weight_mode == "scheduler":
            loss_weights = _alpha_weight_from_t(t, scheduler=cfg.diffusion_scheduler).unsqueeze(1).expand_as(token_nll)
        elif weight_mode == "uniform":
            loss_weights = torch.ones_like(token_nll)
        else:
            raise ValueError(f"Unsupported diffusion_loss_weight: {cfg.diffusion_loss_weight}")

    token_nll = token_nll * masked_mask.to(dtype=token_nll.dtype) * loss_weights
    return token_nll.sum() / masked_mask.sum().clamp_min(1)


def _compute_eso_loss(
    model: TinyDecoderLM,
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: LMConfig,
    mask_id: int,
    random_ratio: float = 0.9,
    ar_frac: float = 0.5,
) -> torch.Tensor:
    """Eso-LM loss mixing AR (alpha_0=0) and MDM (alpha_0=1) per batch.

    With probability ``ar_frac``, runs the AO-GPT objective (next-token CE on
    clean tokens along a random order -- equivalent to Eso-LM at alpha_0=0).
    Otherwise runs the MDM-under-shuffled-causal objective: sample a random
    time t, mask some positions, predict masked tokens from causal predecessors
    in the shuffled order, weighted by ``alpha'_t / (1 - alpha_t)``.

    The AR branch is equivalent to AO-GPT on the same data; mixing both
    regimes lets the shared TinyDecoderLM handle alpha_0 continuously at
    inference time.
    """
    bsz, seqlen = clean_ids.shape
    device = clean_ids.device

    do_ar = (ar_frac >= 1.0) or (ar_frac > 0.0 and bool(torch.rand(1, device=device).item() < ar_frac))
    orders = _sample_ao_orders(labels, random_ratio=random_ratio)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)

    if do_ar:
        logits = model(clean_ids, orders=orders)
        targets_shuf = labels[batch_idx, orders]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets_shuf.reshape(-1),
            ignore_index=-100,
        )

    # MDM branch
    maskable_mask = labels != -100
    t = cfg.diffusion_time_epsilon + (1.0 - cfg.diffusion_time_epsilon) * torch.rand(
        bsz, device=device
    )
    p_mask = 1.0 - _alpha_from_t(t, scheduler=cfg.diffusion_scheduler)
    masked_mask = (
        torch.rand((bsz, seqlen), device=device) < p_mask.unsqueeze(1)
    ) & maskable_mask

    if not bool(masked_mask.any().item()):
        any_maskable = maskable_mask.any(dim=1)
        if bool(any_maskable.any().item()):
            row_idx = torch.where(any_maskable)[0]
            first_pos = torch.argmax(
                maskable_mask[row_idx].to(dtype=torch.float32), dim=1
            )
            masked_mask[row_idx, first_pos] = True

    noised_ids = torch.where(masked_mask, torch.full_like(clean_ids, mask_id), clean_ids)
    logits = model(noised_ids, orders=orders)
    targets_shuf = clean_ids[batch_idx, orders]
    masked_shuf = masked_mask[batch_idx, orders]
    token_nll = F.cross_entropy(logits.transpose(1, 2), targets_shuf, reduction="none")

    weight_mode = (cfg.diffusion_loss_weight or "scheduler").strip().lower()
    if weight_mode == "scheduler":
        loss_weights = _alpha_weight_from_t(t, scheduler=cfg.diffusion_scheduler).unsqueeze(1).expand_as(token_nll)
    elif weight_mode == "uniform":
        loss_weights = torch.ones_like(token_nll)
    else:
        raise ValueError(f"Unsupported diffusion_loss_weight: {cfg.diffusion_loss_weight}")

    token_nll = token_nll * masked_shuf.to(dtype=token_nll.dtype) * loss_weights
    return token_nll.sum() / masked_shuf.sum().clamp_min(1)


def train_language_model(
    train_smiles: Sequence[str],
    valid_smiles: Sequence[str],
    representation: str,
    output_dir: str | Path,
    vocab_fit_smiles: Optional[Sequence[str]] = None,
    cfg: Optional[LMConfig] = None,
    min_token_freq: int = 1,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
    stamp_vocab: Optional[str] = None,
    stamp_force_ring_aware_encode: bool = False,
    init_from: Optional[str] = None,
) -> Dict[str, Any]:
    rep = build_representation(
        representation,
        stamp_vocab=stamp_vocab,
        stamp_force_ring_aware_encode=stamp_force_ring_aware_encode,
    )
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

    vocab_fit_source = list(vocab_fit_smiles) if vocab_fit_smiles is not None else list(train_smiles)
    if not vocab_fit_source:
        raise ValueError("No vocab-fit samples provided.")
    if vocab_fit_smiles is not None:
        vocab_fit_tok = _encode_sequences_with_rep(
            vocab_fit_source,
            rep,
            show_progress=show_progress,
            desc=f"Tokenize vocab-fit[{representation}]",
            num_workers=encode_workers,
            chunksize=encode_chunksize,
        )
    else:
        vocab_fit_tok = train_tok
    if not vocab_fit_tok:
        raise ValueError("No vocab-fit samples encoded successfully.")

    train_token_lengths = [len(x) for x in train_tok]
    valid_token_lengths = [len(x) for x in valid_tok]
    train_token_length_stats = _summarize_token_lengths(train_token_lengths)
    valid_token_length_stats = _summarize_token_lengths(valid_token_lengths)
    train_token_length_hist = _token_length_histogram(train_token_lengths)
    valid_token_length_hist = _token_length_histogram(valid_token_lengths)

    motif_seed = None
    if hasattr(rep, "codec") and getattr(rep.codec, "motif_tokenizer", None) is not None:
        motif_seed = list(rep.codec.motif_tokenizer.idx2smi)

    vocab = TokenVocab.build(
        vocab_fit_tok,
        min_freq=min_token_freq,
        add_mask_token=(model_type == "diffusion"),
        seed_tokens=motif_seed,
    )

    train_ids = _ids_with_special_tokens(train_tok, vocab)
    valid_ids = _ids_with_special_tokens(valid_tok, vocab)
    if model_type in ("decoder", "ao_gpt", "ao_gpt_pp"):
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

    if model_type in ("decoder", "ao_gpt", "ao_gpt_pp"):
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

    if model_type in ("decoder", "ao_gpt", "ao_gpt_pp"):
        model: LMModel = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        # Standard MDLM denoiser: bidirectional attention, no AdaLN/target-pos
        # conditioning, no none_emb prefix. Force cfg.use_adaln=False so the
        # arch matches standard MDM transformers.
        cfg.use_adaln = False
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=True)

    if init_from is not None:
        init_path = Path(init_from) / "model.pt"
        if init_path.exists():
            src = torch.load(init_path, map_location="cpu")["model_state"]
            src = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in src.items()}
            tgt = model.state_dict()
            loaded = 0
            for k in tgt:
                if k in src:
                    sv, tv = src[k], tgt[k]
                    if sv.shape == tv.shape:
                        tgt[k] = sv
                        loaded += 1
                    elif sv.dim() == tv.dim() and all(s <= t for s, t in zip(sv.shape, tv.shape)):
                        slices = tuple(slice(0, s) for s in sv.shape)
                        tgt[k][slices] = sv
                        loaded += 1
            model.load_state_dict(tgt)
            print(f"[train-lm] init_from {init_from}: loaded {loaded}/{len(tgt)} params", flush=True)

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
    if model_type in ("decoder", "ao_gpt", "ao_gpt_pp"):
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
            f"loss_weight={cfg.diffusion_loss_weight}, "
            f"tokens_per_step={int(getattr(cfg, 'diffusion_tokens_per_step', 0))}",
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

    # Pre-compute structural token IDs for role-aware masking (diffusion only).
    _structural_ids: set = set()
    if model_type == "diffusion" and bool(getattr(cfg, "diffusion_chemical_role_masking", False)):
        for tok_str, tok_id in vocab.stoi.items():
            if TokenVocab._is_stamp_structural(tok_str):
                _structural_ids.add(tok_id)
        if _structural_ids:
            print(f"[train-lm] chemical-role masking: {len(_structural_ids)} structural token IDs (exp=1.5), rest motif (exp=0.5)", flush=True)

    # Pre-compute topology token ID tensor for ao_gpt_pp type-aware masking.
    _topology_id_tensor = torch.empty(0, dtype=torch.long, device=dev)
    if model_type == "ao_gpt_pp":
        topo_ids = [
            tok_id for tok_str, tok_id in vocab.stoi.items()
            if TokenVocab._is_stamp_structural(tok_str)
        ]
        _topology_id_tensor = torch.tensor(topo_ids, dtype=torch.long, device=dev)
        print(
            f"[train-lm] ao_gpt_pp: {len(topo_ids)} topology token IDs; "
            f"frag_mask={cfg.ao_pp_frag_mask_rate} topo_mask={cfg.ao_pp_topo_mask_rate}; "
            f"mix uni/ta/id = {cfg.ao_pp_mix_uniform}/{cfg.ao_pp_mix_type_aware}/{cfg.ao_pp_mix_identity}",
            flush=True,
        )

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
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                elif model_type == "ao_gpt":
                    loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=cfg.ao_random_ratio)
                elif model_type == "ao_gpt_pp":
                    loss = _compute_ao_gpt_pp_loss(model, x, y, cfg, _topology_id_tensor)
                else:
                    loss = _compute_diffusion_loss(
                        model,  # type: ignore[arg-type]
                        clean_ids=x,
                        labels=y,
                        cfg=cfg,
                        mask_id=vocab.mask_id,
                        structural_ids=_structural_ids if _structural_ids else None,
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
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                    elif model_type == "ao_gpt":
                        loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=cfg.ao_random_ratio)
                    elif model_type == "ao_gpt_pp":
                        loss = _compute_ao_gpt_pp_loss(model, x, y, cfg, _topology_id_tensor)
                    else:
                        loss = _compute_diffusion_loss(
                            model,  # type: ignore[arg-type]
                            clean_ids=x,
                            labels=y,
                            cfg=cfg,
                            mask_id=vocab.mask_id,
                            structural_ids=_structural_ids if _structural_ids else None,
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
    stamp_vocab_rel = None
    if stamp_vocab is not None:
        import shutil
        dest = out_dir / "stamp_vocab.txt"
        shutil.copy2(stamp_vocab, dest)
        stamp_vocab_rel = "stamp_vocab.txt"
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "cfg": vars(cfg),
            "representation": representation,
            "model_type": model_type,
            "stamp_vocab": stamp_vocab_rel,
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
                "train_token_length_stats": train_token_length_stats,
                "valid_token_length_stats": valid_token_length_stats,
                "train_token_length_histogram": train_token_length_hist,
                "valid_token_length_histogram": valid_token_length_hist,
                "vocab_fit_size": len(vocab_fit_source),
                "n_vocab_fit_encoded": len(vocab_fit_tok),
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
    if model_type == "llama":
        model: LMModel = LlamaDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    elif model_type in ("decoder", "ao_gpt", "ao_gpt_pp"):
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        # Standard MDLM denoiser: bidirectional attention, no AdaLN/target-pos
        # conditioning, no none_emb prefix. Force cfg.use_adaln=False so the
        # arch matches standard MDM transformers.
        cfg.use_adaln = False
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=True)
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

    stamp_vocab_file = ckpt.get("stamp_vocab")
    stamp_vocab_path = None
    if stamp_vocab_file is not None:
        candidate = Path(model_dir) / stamp_vocab_file
        if candidate.exists():
            stamp_vocab_path = str(candidate)
    rep = build_representation(rep_name, stamp_vocab=stamp_vocab_path)
    return model, vocab, rep, dev


def _sample_next_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    typical_p: float = 0.0,
) -> torch.Tensor:
    """Temperature + top-k + top-p (nucleus) + typical sampling.

    Order of filters: top-k first, then top-p, then typical.
    - top_p: keep smallest prefix (descending prob) with cumulative prob >= top_p.
    - typical_p: locally typical sampling (Meister et al. 2023). Sort tokens by
      |I(w) - H(p)| ascending (most typical first), keep smallest prefix with
      cumulative prob >= typical_p. Pass 0 to disable.
    """
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        logits = _apply_top_k_logits(logits, top_k=k)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative - sorted_probs >= top_p
        remove_sorted[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    if 0.0 < typical_p < 1.0:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Entropy per row; mask NaN from -inf*0 contributions.
        entropy = -(probs * log_probs).nansum(dim=-1, keepdim=True)
        # Signed deviation |I(w) - H(p)| = |-log p(w) - H(p)|
        shifted = ((-log_probs) - entropy).abs()
        sorted_shifted, sorted_idx = torch.sort(shifted, dim=-1)  # ascending
        sorted_probs = probs.gather(-1, sorted_idx)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative - sorted_probs >= typical_p
        remove_sorted[..., 0] = False  # always keep most-typical token
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

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


def _stamp_token_id_groups(vocab: TokenVocab) -> Dict[str, List[int]]:
    joint_ids: List[int] = []
    branch_ids: List[int] = []
    stereo_ids: List[int] = []
    ring_ids: List[int] = []
    motif_ids: List[int] = []
    special_ids = {vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.unk_id}
    for tok, tok_id in vocab.stoi.items():
        if tok_id in special_ids:
            continue
        if tok.startswith("[J_"):
            joint_ids.append(int(tok_id))
        elif tok.startswith("[B_"):
            branch_ids.append(int(tok_id))
        elif tok.startswith("[S_"):
            stereo_ids.append(int(tok_id))
        elif tok == "[Ring]":
            ring_ids.append(int(tok_id))
        else:
            motif_ids.append(int(tok_id))
    return {
        "joint_ids": joint_ids,
        "branch_ids": branch_ids,
        "stereo_ids": stereo_ids,
        "ring_ids": ring_ids,
        "motif_ids": motif_ids,
    }


def _apply_stamp_structure_constraints(
    logits: torch.Tensor,
    current_ids: torch.Tensor,
    vocab: TokenVocab,
) -> torch.Tensor:
    groups = _stamp_token_id_groups(vocab)
    joint_ids = groups["joint_ids"]
    branch_ids = groups["branch_ids"]
    stereo_ids = groups["stereo_ids"]
    ring_ids = groups["ring_ids"]
    motif_ids = groups["motif_ids"]
    if not (joint_ids or branch_ids or stereo_ids or ring_ids):
        return logits

    prev_ids = torch.full_like(current_ids, vocab.bos_id)
    prev_ids[:, 1:] = current_ids[:, :-1]

    allowed_joint_prev = torch.zeros_like(current_ids, dtype=torch.bool)
    allowed_stereo_prev = torch.zeros_like(current_ids, dtype=torch.bool)
    allowed_ring_prev = torch.zeros_like(current_ids, dtype=torch.bool)
    allowed_branch_prev = torch.zeros_like(current_ids, dtype=torch.bool)

    for tok_id in motif_ids:
        allowed_joint_prev |= (prev_ids == int(tok_id))
    for tok_id in joint_ids:
        allowed_joint_prev |= (prev_ids == int(tok_id))
        allowed_stereo_prev |= (prev_ids == int(tok_id))
        allowed_ring_prev |= (prev_ids == int(tok_id))
        allowed_branch_prev |= (prev_ids == int(tok_id))
    for tok_id in stereo_ids:
        allowed_joint_prev |= (prev_ids == int(tok_id))
        allowed_ring_prev |= (prev_ids == int(tok_id))
        allowed_branch_prev |= (prev_ids == int(tok_id))
    for tok_id in ring_ids:
        allowed_joint_prev |= (prev_ids == int(tok_id))
        allowed_branch_prev |= (prev_ids == int(tok_id))

    if joint_ids:
        logits[:, :, joint_ids] = logits[:, :, joint_ids].masked_fill((~allowed_joint_prev).unsqueeze(-1), float("-inf"))
    if stereo_ids:
        logits[:, :, stereo_ids] = logits[:, :, stereo_ids].masked_fill((~allowed_stereo_prev).unsqueeze(-1), float("-inf"))
    if ring_ids:
        logits[:, :, ring_ids] = logits[:, :, ring_ids].masked_fill((~allowed_ring_prev).unsqueeze(-1), float("-inf"))
    if branch_ids:
        invalid_branch = ~allowed_branch_prev
        invalid_branch[:, -1] = True
        logits[:, :, branch_ids] = logits[:, :, branch_ids].masked_fill(invalid_branch.unsqueeze(-1), float("-inf"))
    return logits


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64) / float(temperature)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(noise.clamp_(1e-12, 1.0 - 1e-12)))
    return logits + gumbel


def _trim_special_tail(row: List[int], vocab: TokenVocab) -> List[int]:
    cut = len(row)
    for token_id in (vocab.eos_id, vocab.pad_id):
        if token_id in row:
            cut = min(cut, row.index(token_id))
    return row[:cut]


def _length_percentile(lengths: Sequence[int], q: float) -> int:
    if not lengths:
        return 0
    qq = min(1.0, max(0.0, float(q)))
    ordered = sorted(int(x) for x in lengths)
    idx = int(math.ceil((len(ordered) - 1) * qq))
    idx = min(len(ordered) - 1, max(0, idx))
    return int(ordered[idx])


def _summarize_token_lengths(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    vals = [int(x) for x in lengths]
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": float(sum(vals) / len(vals)),
        "p50": _length_percentile(vals, 0.50),
        "p90": _length_percentile(vals, 0.90),
        "p95": _length_percentile(vals, 0.95),
        "p99": _length_percentile(vals, 0.99),
        "max": max(vals),
    }


def _token_length_histogram(lengths: Sequence[int]) -> Dict[str, int]:
    hist = Counter(int(x) for x in lengths)
    return {str(k): int(v) for k, v in sorted(hist.items())}


def _sample_model_sequences_decoder(
    model: TinyDecoderLM,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    typical_p: float = 0.0,
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

    use_ao_orders = hasattr(model, "target_pos_emb") and _normalize_model_type(
        getattr(model.cfg, "model_type", "decoder")
    ) == "ao_gpt"

    for _step in range(max_new_tokens):
        ctx = ids[:, -model.cfg.max_seq_len :]
        with torch.no_grad():
            if use_ao_orders:
                seq_len = ctx.size(1)
                ao_orders = torch.arange(seq_len, device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                logits = model(ctx, orders=ao_orders)[:, -1, :]
            else:
                logits = model(ctx)[:, -1, :]
        for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id):
            if 0 <= int(suppressed_id) < logits.size(-1):
                logits[:, int(suppressed_id)] = float("-inf")
        next_ids = _sample_next_tokens(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p,
        )
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


def _sample_model_sequences_ao_canvas_confidence(
    model: TinyDecoderLM,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    placeholder_token: str = "unk",
) -> List[List[str]]:
    if n_samples <= 0:
        return []
    if prompt_tokens:
        raise ValueError("ao_canvas_confidence currently does not support prompt_tokens.")

    dev = device or next(model.parameters()).device
    total_len = min(int(model.cfg.max_seq_len), max(2, int(max_new_tokens) + 1))  # includes BOS slot
    placeholder_id = vocab.unk_id if str(placeholder_token).strip().lower() != "pad" else vocab.pad_id

    x = torch.full((n_samples, total_len), int(placeholder_id), dtype=torch.long, device=dev)
    x[:, 0] = int(vocab.bos_id)
    filled = torch.zeros((n_samples, total_len), dtype=torch.bool, device=dev)
    filled[:, 0] = True
    finished = torch.zeros(n_samples, dtype=torch.bool, device=dev)
    candidate_chunk_size = 256

    for _step in range(total_len - 1):
        active_rows = torch.nonzero(~finished, as_tuple=False).squeeze(1)
        if int(active_rows.numel()) == 0:
            break

        expanded_inputs = []
        expanded_orders = []
        expanded_batch_idx = []
        expanded_candidate_pos = []
        expanded_pred_idx = []

        for b in active_rows.tolist():
            observed = torch.nonzero(filled[b], as_tuple=False).squeeze(1)
            candidates = torch.nonzero(~filled[b], as_tuple=False).squeeze(1)
            if int(candidates.numel()) == 0:
                finished[b] = True
                continue
            for pos in candidates.tolist():
                pos_t = torch.tensor([int(pos)], dtype=torch.long, device=dev)
                remaining = candidates[candidates != int(pos)]
                order = torch.cat([observed, pos_t, remaining], dim=0)
                expanded_inputs.append(x[b])
                expanded_orders.append(order)
                expanded_batch_idx.append(int(b))
                expanded_candidate_pos.append(int(pos))
                expanded_pred_idx.append(int(observed.numel()))

        if not expanded_inputs:
            break

        expanded_batch_idx_t = torch.tensor(expanded_batch_idx, dtype=torch.long, device=dev)
        best_conf = torch.full((n_samples,), float("-inf"), device=dev)
        best_row = torch.full((n_samples,), -1, dtype=torch.long, device=dev)
        non_bos_filled = filled.sum(dim=1) - 1

        for start in range(0, len(expanded_inputs), candidate_chunk_size):
            end = min(start + candidate_chunk_size, len(expanded_inputs))
            x_exp = torch.stack(expanded_inputs[start:end], dim=0)
            orders_exp = torch.stack(expanded_orders[start:end], dim=0)
            pred_idx_t = torch.tensor(expanded_pred_idx[start:end], dtype=torch.long, device=dev)
            row_idx_t = torch.arange(end - start, device=dev, dtype=torch.long)
            chunk_batch_idx = expanded_batch_idx_t[start:end]

            with torch.no_grad():
                logits = model(x_exp, orders=orders_exp)
            logits_next = logits[row_idx_t, pred_idx_t, :]
            for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id):
                if 0 <= int(suppressed_id) < logits_next.size(-1):
                    logits_next[:, int(suppressed_id)] = float("-inf")
            if 0 <= int(vocab.eos_id) < logits_next.size(-1):
                eos_forbidden = non_bos_filled[chunk_batch_idx] <= 0
                logits_next[eos_forbidden, int(vocab.eos_id)] = float("-inf")
            logits_next = _apply_top_k_logits(logits_next, top_k=top_k)
            probs = torch.softmax(logits_next, dim=-1)
            conf = probs.max(dim=-1).values

            for i in range(end - start):
                global_i = start + i
                b = int(expanded_batch_idx[global_i])
                if float(conf[i].item()) > float(best_conf[b].item()):
                    best_conf[b] = conf[i]
                    best_row[b] = int(global_i)

        chosen_rows = best_row[active_rows]
        valid_choice = chosen_rows >= 0
        if not bool(valid_choice.any().item()):
            break
        chosen_rows = chosen_rows[valid_choice]
        chosen_batch = active_rows[valid_choice]
        chosen_inputs = torch.stack([expanded_inputs[int(i)] for i in chosen_rows.tolist()], dim=0)
        chosen_orders = torch.stack([expanded_orders[int(i)] for i in chosen_rows.tolist()], dim=0)
        chosen_pred_idx = torch.tensor([expanded_pred_idx[int(i)] for i in chosen_rows.tolist()], dtype=torch.long, device=dev)
        chosen_row_idx = torch.arange(chosen_inputs.size(0), device=dev, dtype=torch.long)
        with torch.no_grad():
            chosen_logits = model(chosen_inputs, orders=chosen_orders)[chosen_row_idx, chosen_pred_idx, :]
        for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id):
            if 0 <= int(suppressed_id) < chosen_logits.size(-1):
                chosen_logits[:, int(suppressed_id)] = float("-inf")
        if 0 <= int(vocab.eos_id) < chosen_logits.size(-1):
            eos_forbidden = non_bos_filled[chosen_batch] <= 0
            chosen_logits[eos_forbidden, int(vocab.eos_id)] = float("-inf")
        chosen_logits = _apply_top_k_logits(chosen_logits, top_k=top_k)
        sampled = torch.argmax(_add_gumbel_noise(chosen_logits, temperature=temperature), dim=-1).to(dtype=torch.long)
        chosen_pos = torch.tensor([expanded_candidate_pos[int(i)] for i in chosen_rows.tolist()], dtype=torch.long, device=dev)

        for idx in range(int(chosen_batch.numel())):
            b = int(chosen_batch[idx].item())
            pos = int(chosen_pos[idx].item())
            tok = int(sampled[idx].item())
            x[b, pos] = tok
            filled[b, pos] = True
            if tok == int(vocab.eos_id):
                if pos + 1 < total_len:
                    x[b, pos + 1 :] = int(vocab.pad_id)
                    filled[b, pos + 1 :] = True
                finished[b] = True

    out: List[List[str]] = []
    all_ids = x[:, 1:].detach().cpu().tolist()
    for row in all_ids:
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def _sample_model_sequences_diffusion(
    model: TinyDecoderLM,
    vocab: TokenVocab,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_tokens_per_step: Optional[int] = None,
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

    x = torch.full((n_samples, total_len), vocab.pad_id, dtype=torch.long, device=dev)
    x[:, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=dev).unsqueeze(0).repeat(n_samples, 1)
    if max_new_tokens > 0:
        x[:, len(prefix) :] = vocab.mask_id

    steps = max(1, int(diffusion_steps if diffusion_steps is not None else model.cfg.diffusion_steps))
    tokens_per_step = int(
        diffusion_tokens_per_step
        if diffusion_tokens_per_step is not None
        else getattr(model.cfg, "diffusion_tokens_per_step", 0)
    )
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
            if tokens_per_step > 0:
                num_transfer_tokens = _get_num_transfer_tokens_by_budget(
                    mask_index=block_mask_index,
                    tokens_per_step=tokens_per_step,
                )
            else:
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
                with torch.no_grad():
                    logits = model(x)
                # Allow EOS (model must learn to terminate sequences). Still
                # suppress PAD/BOS/MASK/UNK which are never valid outputs.
                for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
                    if 0 <= int(suppressed_id) < logits.size(-1):
                        logits[:, :, suppressed_id] = float("-inf")
                if bool(getattr(model.cfg, "diffusion_structure_constraints", False)):
                    logits = _apply_stamp_structure_constraints(logits, current_ids=x, vocab=vocab)
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

                ks = num_transfer_tokens[:, i].clamp(min=0)
                max_k = int(ks.max().item())
                if max_k > 0:
                    _, topk_idx = torch.topk(conf, k=max_k, dim=-1)
                    arange_k = torch.arange(max_k, device=conf.device).unsqueeze(0)
                    transfer_index = arange_k < ks.unsqueeze(1)
                    scatter_mask = torch.zeros_like(mask_index, dtype=torch.bool)
                    scatter_mask.scatter_(1, topk_idx, transfer_index)
                    transfer_index = scatter_mask
                else:
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)

                x[transfer_index] = x0[transfer_index]

    x = torch.where(x == vocab.mask_id, torch.full_like(x, vocab.pad_id), x)
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
    top_p: float = 0.0,
    typical_p: float = 0.0,
    prompt_tokens: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    diffusion_steps: Optional[int] = None,
    diffusion_tokens_per_step: Optional[int] = None,
    diffusion_block_size: Optional[int] = None,
    diffusion_remasking: Optional[str] = None,
    diffusion_stochastic_transfer: Optional[bool] = None,
    ao_decode_mode: str = "ar",
    ao_canvas_placeholder: str = "unk",
) -> List[List[str]]:
    model_type = _normalize_model_type(getattr(model.cfg, "model_type", "decoder"))
    if model_type in ("decoder", "ao_gpt", "ao_gpt_pp", "llama"):
        if model_type in ("ao_gpt", "ao_gpt_pp") and str(ao_decode_mode).strip().lower() == "ao_canvas_confidence":
            return _sample_model_sequences_ao_canvas_confidence(
                model=model,  # type: ignore[arg-type]
                vocab=vocab,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                prompt_tokens=prompt_tokens,
                device=device,
                placeholder_token=ao_canvas_placeholder,
            )
        return _sample_model_sequences_decoder(
            model=model,  # type: ignore[arg-type]
            vocab=vocab,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
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
        diffusion_tokens_per_step=diffusion_tokens_per_step,
        diffusion_block_size=diffusion_block_size,
        diffusion_remasking=diffusion_remasking,
        diffusion_stochastic_transfer=diffusion_stochastic_transfer,
    )

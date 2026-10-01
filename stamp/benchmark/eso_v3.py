"""ESO-LM v3: STAMP-side port of official EsoLMDiT (third_party/Eso-LMs/models/dit.py).

Includes all official architectural tricks:
  - flex_attention with create_block_mask (fused, score_mod-based)
  - Rotary positional embeddings (RoPE)
  - AdaLN with sigma conditioning (TimestepEmbedder + adaLN_modulation per block)
  - EmbeddingLayer with zero-mask trick at output (forbid mask_id in head)
  - block_diff_mask for sequential branch [zt, x0] of length 2n
  - bidirectional attention for diffusion branch (length n)

Differences from official:
  - No huggingface_hub / wandb / lightning
  - No KV cache (only training path)
  - Uses STAMP LMConfig + TokenVocab/HybridVocab interface
  - dynamic_shape=False (fixed seq len from cfg.max_seq_len)
"""
from __future__ import annotations
import math
from functools import partial, lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# ============================================================
# Rotary positional embeddings
# ============================================================
class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # identity transform on v
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin):
    """x: (b, s, h, d). cos, sin: (b, s, 1, d) or broadcastable."""
    ro_dim = cos.shape[-1]
    x_ro = x[..., :ro_dim]
    x_pass = x[..., ro_dim:]
    out_ro = x_ro * cos + rotate_half(x_ro) * sin
    return torch.cat([out_ro, x_pass], dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    """qkv: [b, s, 3, h, d]. cos, sin: [b, s, 3, 1, d]."""
    cos, sin = rotary_cos_sin
    q, k, v = qkv.chunk(3, dim=2)
    cos_q = cos[:, :, 0, :, :]
    sin_q = sin[:, :, 0, :, :]
    cos_k = cos[:, :, 1, :, :]
    sin_k = sin[:, :, 1, :, :]
    q = apply_rotary_emb_torch(q.squeeze(dim=2), cos_q, sin_q)
    k = apply_rotary_emb_torch(k.squeeze(dim=2), cos_k, sin_k)
    v = v.squeeze(dim=2)
    return q, k, v


# ============================================================
# AdaLN-related: LayerNorm + modulate + TimestepEmbedder
# ============================================================
class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def modulate_fused(x, shift, scale):
    return x * (1 + scale) + shift


def bias_dropout_add_scale(x, bias, scale, residual, prob, training):
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def bias_dropout_add_scale_fused_train(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


def bias_dropout_add_scale_fused_inference(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


class TimestepEmbedder(nn.Module):
    """sigma → conditioning vector c via sinusoidal embedding + 2-layer MLP."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period: int = 10_000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ============================================================
# Embedding layer (with optional zero-mask trick at output)
# ============================================================
class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


# ============================================================
# DiT Block with flex_attention
# ============================================================
@lru_cache(maxsize=8)
def _block_diff_block_mask(n: int, device_str: str):
    """Cached BlockMask for [zt, x0] sequential branch (length 2n)."""
    def mask_mod(b, h, q_idx, kv_idx):
        x0_q = q_idx >= n
        x0_kv = kv_idx >= n
        bq = torch.where(x0_q, q_idx - n, q_idx)
        bkv = torch.where(x0_kv, kv_idx - n, kv_idx)
        block_diag = (bq == bkv) & (x0_q == x0_kv)
        offset_block_causal = (bq > bkv) & x0_kv & (~x0_q)
        block_causal = (bq >= bkv) & x0_kv & x0_q
        return block_diag | offset_block_causal | block_causal
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=2 * n, KV_LEN=2 * n, device=device_str)


@lru_cache(maxsize=8)
def _bidirectional_block_mask(n: int, device_str: str):
    def mask_mod(b, h, q_idx, kv_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=device_str)


# Cached compiled flex_attention function
_FLEX_COMPILED = None
def _get_flex_compiled():
    """Compiled flex_attention. Uses dynamic=False because flex_attention +
    create_block_mask have shape-coupled internals that don't compose with
    dynamic shapes (despite official's dynamic_shape flag, it has issues).
    Caller must pad inputs to fixed seq_len to avoid recompilation."""
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    return _FLEX_COMPILED


def flex_attention_mh(q, k, v, block_mask):
    """q, k, v: [b, s, h, d]. Returns [b, s, h*d]."""
    q = q.transpose(1, 2).contiguous()  # [b, h, s, d]
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    out = _get_flex_compiled()(q, k, v, block_mask=block_mask)
    out = out.transpose(1, 2).contiguous()  # [b, s, h, d]
    b, s, h, d = out.shape
    return out.reshape(b, s, h * d)


class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, adaLN: bool, cond_dim: Optional[int] = None,
                  mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.adaLN = adaLN

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        if adaLN:
            assert cond_dim is not None
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _bdas(self, x, scale, residual):
        return bias_dropout_add_scale(x, None, scale, residual, self.dropout, self.training)

    def forward(self, x, rotary_cos_sin, c=None, block_mask=None):
        x_skip = x
        x = self.norm1(x)
        if self.adaLN:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)  # [b, s, 3*dim]
        b, s, _ = qkv.shape
        qkv = qkv.reshape(b, s, 3, self.n_heads, self.head_dim).contiguous()
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        if block_mask is not None:
            x = flex_attention_mh(q, k, v, block_mask)
        else:
            # Bidirectional fallback (use bidirectional block_mask)
            n = q.shape[1]
            bm = _bidirectional_block_mask(n, str(q.device))
            x = flex_attention_mh(q, k, v, bm)

        if self.adaLN:
            x = self._bdas(self.attn_out(x), gate_msa, x_skip)
            x = self._bdas(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp, x)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = self._bdas(self.attn_out(x), scale, x_skip)
            x = self._bdas(self.mlp(self.norm2(x)), scale, x)
        return x


# ============================================================
# Main model: TinyEsoDiTV3
# ============================================================
class TinyEsoDiTV3(nn.Module):
    """STAMP TinyEsoDiT v3: full official-style architecture port.

    Modes:
      - "sequential": [zt, x0] of length 2n, block_diff_mask attention
      - "diffusion": zt of length n, bidirectional attention (for future MDM training)
    """
    def __init__(self, vocab_size: int, cfg, mask_index: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.mask_index = int(mask_index)
        self.adaLN = True
        self.cond_dim = cond_dim if cond_dim is not None else 128

        self.vocab_embed = EmbeddingLayer(cfg.d_model, vocab_size)
        # Sigma conditioning
        self.sigma_embedder = TimestepEmbedder(self.cond_dim)
        # Rotary
        self.rotary_dim = cfg.d_model // cfg.n_heads
        self.rotary_emb = Rotary(self.rotary_dim)

        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=cfg.d_model,
                n_heads=cfg.n_heads,
                adaLN=self.adaLN,
                cond_dim=self.cond_dim,
                mlp_ratio=cfg.d_ff // cfg.d_model,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=True)
        self.neg_inf = -1e6

    def _sort_rotary_cos_sin(self, rotary_cos_sin, sort_idx):
        """Gather rotary by sort_idx so each sorted position carries its
        original-layout pos signature."""
        cos, sin = rotary_cos_sin
        bs = sort_idx.shape[0]
        cos = cos.expand(bs, -1, -1, -1, -1)
        sin = sin.expand(bs, -1, -1, -1, -1)
        cos = torch.gather(cos, dim=1, index=sort_idx[:, :, None, None, None].expand(-1, -1, 3, -1, self.rotary_dim)).contiguous()
        sin = torch.gather(sin, dim=1, index=sort_idx[:, :, None, None, None].expand(-1, -1, 3, -1, self.rotary_dim)).contiguous()
        return cos, sin

    def forward(self, zt, x0=None, sort_idx=None, sigma=None, mode: str = "sequential"):
        device = zt.device
        B, n = zt.shape

        # sigma defaults to zeros (no conditioning, e.g. alpha_0=0 case)
        if sigma is None:
            sigma = torch.zeros(B, device=device, dtype=torch.float32)
        c = self.sigma_embedder(sigma)  # [B, cond_dim]

        if mode == "diffusion":
            x = self.vocab_embed(zt)
            # Rotary for length n (no sort needed for bidirectional)
            rotary_cos_sin = self.rotary_emb(x)
            if sort_idx is not None:
                rotary_cos_sin = self._sort_rotary_cos_sin(rotary_cos_sin, sort_idx)
            else:
                # Just expand
                cos, sin = rotary_cos_sin
                rotary_cos_sin = (cos.expand(B, -1, -1, -1, -1).contiguous(),
                                    sin.expand(B, -1, -1, -1, -1).contiguous())
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c=c, block_mask=None)
            x = self.final_norm(x)
            logits = self.head(x)
            # zero-mask trick
            logits = logits.clone()
            logits[..., self.mask_index] = self.neg_inf
            return logits

        # sequential: concat [zt, x0]
        if x0 is None or sort_idx is None:
            raise ValueError("sequential mode requires x0 and sort_idx")
        tok_zt = self.vocab_embed(zt)
        tok_x0 = self.vocab_embed(x0)
        x = torch.cat([tok_zt, tok_x0], dim=1)  # [B, 2n, d]

        # Rotary: built for length 2n (or compute on x and tile) — official sorts then duplicates
        # Approach: compute base rotary for length n, gather by sort_idx, then duplicate for x0 half
        # Build a dummy of length n to compute rotary
        dummy = torch.empty(1, n, 1, 1, device=device, dtype=x.dtype)
        rotary_cos_sin_n = self.rotary_emb(dummy)  # cos, sin of shape [1, n, 3, 1, rotary_dim]
        rotary_cos_sin_n = self._sort_rotary_cos_sin(rotary_cos_sin_n, sort_idx)
        # Duplicate for the x0 half (positions n..2n-1 use the same per-position rotary as zt half)
        cos_n, sin_n = rotary_cos_sin_n  # [B, n, 3, 1, d]
        cos_2n = torch.cat([cos_n, cos_n], dim=1)  # [B, 2n, 3, 1, d]
        sin_2n = torch.cat([sin_n, sin_n], dim=1)
        rotary_2n = (cos_2n, sin_2n)

        block_mask = _block_diff_block_mask(n, str(device))
        for block in self.blocks:
            x = block(x, rotary_2n, c=c, block_mask=block_mask)
        x = self.final_norm(x)
        logits = self.head(x)
        # Zero-mask trick on output for the zt-half (first n positions)
        logits = logits.clone()
        logits[..., self.mask_index] = self.neg_inf
        return logits[:, :n, :]  # only return zt-half


# ============================================================
# Loss functions for v3 (sequential branch — alpha_0 fixed mask rate)
# ============================================================
def compute_eso_v3_seq_loss(
    model: TinyEsoDiTV3,
    x0: torch.Tensor,
    labels: torch.Tensor,
    cfg,
    mask_id: int,
    alpha_0: float = 0.0,
    shuffle_unmasked: bool = False,
) -> torch.Tensor:
    """Sequential-branch reconstruction loss. Aligned with official _reconstruction_loss."""
    B, n = x0.shape
    device = x0.device
    maskable = labels != -100
    p_mask = float(1.0 - alpha_0)
    mask_draw = (torch.rand((B, n), device=device) < p_mask) & maskable

    # Force at least one mask per row
    any_maskable = maskable.any(dim=1)
    any_masked = mask_draw.any(dim=1)
    need_fix = any_maskable & (~any_masked)
    if bool(need_fix.any().item()):
        row_idx = torch.where(need_fix)[0]
        first_pos = torch.argmax(maskable[row_idx].to(torch.float32), dim=1)
        mask_draw[row_idx, first_pos] = True

    xt = torch.where(mask_draw, torch.full_like(x0, mask_id), x0)

    sort_idx = _compute_sort_idx_official_style(xt, mask_id, shuffle_unmasked=shuffle_unmasked)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n)
    zt = xt[batch_idx, sort_idx]
    x0_sorted = x0[batch_idx, sort_idx]

    # No sigma at alpha_0=0 reconstruction (use 0)
    sigma = torch.zeros(B, device=device, dtype=torch.float32)

    logits = model(zt, x0=x0_sorted, sort_idx=sort_idx, sigma=sigma, mode="sequential")
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, x0_sorted.unsqueeze(-1)).squeeze(-1)

    loss_mask = (zt == mask_id).to(nll.dtype)
    denom = loss_mask.sum().clamp_min(1.0)
    return (nll * loss_mask).sum() / denom


def _compute_sort_idx_official_style(xt, mask_id, shuffle_unmasked=False):
    """Match official _sort_indices(z0, shuffle, keep_masks_unshuffled=True)."""
    B, n = xt.shape
    device = xt.device
    masked = (xt == mask_id)
    if shuffle_unmasked:
        offsets = torch.rand((B, n), device=device) * 0.9
    else:
        offsets = torch.linspace(0.0, 0.9, n, device=device).unsqueeze(0).expand(B, n).clone()
    total_masked = int(masked.sum().item())
    if total_masked > 0:
        linspace = torch.linspace(0.0, 1.0, total_masked, device=device)
        offsets_flat = offsets.reshape(-1).clone()
        masked_flat = masked.reshape(-1)
        offsets_flat[masked_flat] = linspace
        offsets = offsets_flat.view(B, n)
    keys = masked.to(torch.float32) + offsets
    return keys.argsort(dim=-1).to(torch.long)

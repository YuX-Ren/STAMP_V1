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
    model_type: str = "decoder"  # decoder | ao_gpt | ao_gpt_pp
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
    # When False, skip AdaLN modulation and the none_emb/target_pos_emb machinery
    # used by AO-GPT.
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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LMConfig":
        """Read supported fields from archived checkpoint configurations."""
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


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


_ADALN_COND_DIM = 128


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
    if mt not in {"decoder", "ao_gpt", "ao_gpt_pp"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected one of: decoder, ao_gpt, ao_gpt_pp")
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
        add_mask_token=False,
        seed_tokens=motif_seed,
    )

    train_ids = _ids_with_special_tokens(train_tok, vocab)
    valid_ids = _ids_with_special_tokens(valid_tok, vocab)
    ds_train: Dataset[Any] = NextTokenDataset(train_ids, cfg.max_seq_len, vocab.pad_id)
    ds_valid: Dataset[Any] = NextTokenDataset(valid_ids, cfg.max_seq_len, vocab.pad_id)

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

    model: LMModel = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)

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
    train_effective_max_seq_len = max((len(x) for x, _ in ds_train.samples), default=0)  # type: ignore[attr-defined]
    valid_effective_max_seq_len = max((len(x) for x, _ in ds_valid.samples), default=0)  # type: ignore[attr-defined]
    print(
        f"[train-lm] dynamic_padding=batch_max (cfg_max_seq_len={cfg.max_seq_len}, "
        f"train_effective_max_seq_len={train_effective_max_seq_len}, "
        f"valid_effective_max_seq_len={valid_effective_max_seq_len})",
        flush=True,
    )
    print(f"[train-lm] model_type={model_type}", flush=True)

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
    cfg = LMConfig.from_dict(ckpt["cfg"])
    model_type = _normalize_model_type(str(ckpt.get("model_type", cfg.model_type)))
    cfg.model_type = model_type
    rep_name = ckpt["representation"]

    vocab = TokenVocab.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()
    }
    if "blocks.layers.0.self_attn.in_proj_weight" in state_dict:
        from .controlled_decoder import ControlledDecoderLM

        cfg.model_type = "decoder"
        model = ControlledDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
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
    ao_decode_mode: str = "ar",
    ao_canvas_placeholder: str = "unk",
) -> List[List[str]]:
    model_type = _normalize_model_type(getattr(model.cfg, "model_type", "decoder"))
    if model_type in ("ao_gpt", "ao_gpt_pp") and str(ao_decode_mode).strip().lower() == "ao_canvas_confidence":
        return _sample_model_sequences_ao_canvas_confidence(
            model=model,
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
        model=model,
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

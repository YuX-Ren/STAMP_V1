"""Original four-layer causal decoder used in the controlled comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class ControlledDecoderLM(nn.Module):
    """Preserve the architecture and parameter names of the original run."""

    def __init__(self, vocab_size: int, cfg: Any) -> None:
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
            raise ValueError(f"Sequence length {seqlen} exceeds {self.cfg.max_seq_len}")
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = torch.triu(
            torch.ones(seqlen, seqlen, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.head(self.ln_f(self.blocks(x, mask=mask)))


def load_controlled_model(model_dir: str | Path, device: str = "cpu"):
    """Load the original weights and token vocabulary using PyTorch alone.

    Returns ``(model, itos)``. Molecular decoding requires the corresponding
    representation adapter; this helper returns token logits.
    """
    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    itos = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))["itos"]
    cfg = SimpleNamespace(**checkpoint["cfg"])
    cfg.model_type = "decoder"
    model = ControlledDecoderLM(len(itos), cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), itos

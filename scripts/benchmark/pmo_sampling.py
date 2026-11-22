"""Checkpoint-aware AO-GPT loading and motif-span sampling for PMO."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch

from stamp.benchmark.hybrid_vocab import HybridVocab, is_stamp_structural
from stamp.benchmark.lm import LMConfig, TinyDecoderLM, TokenVocab, load_trained_model
from stamp.benchmark.mcts_unified.edit_policy import AOEditPolicy


def validate_sampling(settings: dict) -> None:
    required = {"temperature", "top_k", "motif_strict_vocab"}
    if not required.issubset(settings) or settings.keys() - required - {"top_p"}:
        raise ValueError("Generator requires temperature, top_k, motif_strict_vocab; top_p is optional")
    temperature = settings["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Generator temperature must be positive and finite")
    if type(settings["top_k"]) is not int or settings["top_k"] < 0:
        raise ValueError("Generator top_k must be a nonnegative integer (0 disables truncation)")
    top_p = settings.get("top_p", 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("Generator top_p must be in (0, 1]")
    if type(settings["motif_strict_vocab"]) is not bool:
        raise ValueError("motif_strict_vocab must be a boolean")


def load_pmo_model(model_dir: Path, device: str):
    """Preserve the legacy loader; strictly load hybrid checkpoint weights."""
    raw_vocab = json.loads((model_dir / "vocab.json").read_text())
    hybrid = "atomic_motifs" in raw_vocab
    if not hybrid:
        model, vocab, _, dev = load_trained_model(str(model_dir), device=device)
    else:
        checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False, mmap=True)
        raw_config = dict(checkpoint["cfg"])
        if raw_config.get("tied_embeddings", False):
            raise ValueError("A tied-embedding checkpoint requires its matching model architecture")
        if checkpoint.get("model_type", raw_config.get("model_type")) != "ao_gpt":
            raise ValueError("Hybrid PMO sampling requires an AO-GPT checkpoint")
        config = LMConfig.from_dict(raw_config)
        vocab = HybridVocab.from_dict(raw_vocab)
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=config)
        state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model_state"].items()}
        model.load_state_dict(state, strict=True)
        dev = torch.device(device)
        model.to(dev).eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    meta = json.loads((model_dir / "train_meta.json").read_text())
    if count != meta["n_model_params"]:
        raise ValueError(f"Checkpoint parameter count mismatch: {count} != {meta['n_model_params']}")
    return model, vocab, dev, {
        "model_type": "ao_gpt", "n_model_params": count,
        "vocab_type": "hybrid" if hybrid else "token", "vocab_size": len(vocab.itos),
        "d_model": model.cfg.d_model, "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads, "max_seq_len": model.cfg.max_seq_len,
        "weights_updated_during_pmo": False,
    }


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                  suppressed_ids: Sequence[int]) -> int:
    logits = logits.float().clone() / temperature
    logits[list(suppressed_ids)] = -torch.inf
    if top_k > 0:
        values, ids = torch.topk(logits, min(top_k, logits.numel()))
    else:
        values, ids = logits, torch.arange(logits.numel(), device=logits.device)
    if top_p < 1:
        values, order = torch.sort(values, descending=True)
        ids = ids[order]
        cumulative = torch.softmax(values, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(values, dim=-1) >= top_p
        values = values.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(values, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("No finite probability remains after token filtering")
    return int(ids[torch.multinomial(probabilities, 1)].item())


class HybridAOEditPolicy(AOEditPolicy):
    """Expand hybrid motifs before AO infill, retaining the original motif boundaries.

    The selected action is still a motif subtree. Character pieces are the trained
    model's representation of that subtree, not a separate atom-edit operator.
    Unknown input tokens fail explicitly instead of silently substituting <unk>.
    """
    def __init__(self, model, vocab, *, device, temperature=0.7, top_k=0, top_p=0.95):
        super().__init__(model, vocab, device=device, temperature=temperature, top_k=top_k)
        self.top_p = top_p
        self.stats = {"resample_calls": 0, "hybrid_expanded_motifs": 0,
                      "unknown_context_rejections": 0, "context_length_rejections": 0,
                      "sampled_special_rejections": 0}

    def encode_with_offsets(self, tokens):
        ids, offsets = [self.vocab.bos_id], [1]
        for token in tokens:
            piece = self.vocab.encode_stream([token], training=False)
            if self.vocab.unk_id in piece:
                self.stats["unknown_context_rejections"] += 1
                raise ValueError(f"Hybrid vocabulary cannot encode token {token!r}")
            self.stats["hybrid_expanded_motifs"] += len(piece) > 1
            ids.extend(piece)
            offsets.append(len(ids))
        if len(ids) > self._max_seq_len:
            self.stats["context_length_rejections"] += 1
            raise ValueError("Hybrid context exceeds checkpoint max_seq_len; context is not truncated")
        return ids, offsets

    @staticmethod
    def collapse_pieces(pieces):
        output, motif = [], []
        for piece in pieces:
            if is_stamp_structural(piece):
                if motif:
                    output.append("".join(motif))
                    motif = []
                output.append(piece)
            else:
                motif.append(piece)
        if motif:
            output.append("".join(motif))
        return output

    @torch.no_grad()
    def score_span(self, tokens, span_start, span_end):
        if not 0 <= span_start <= span_end <= len(tokens):
            raise ValueError("Span is outside the original motif-token sequence")
        if span_start == span_end:
            return 0.0
        ids, offsets = self.encode_with_offsets(tokens)
        start, end = offsets[span_start], offsets[span_end]
        orders, observed = self._build_ao_order(len(ids), start, end)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        order_tensor = torch.tensor([orders], dtype=torch.long, device=self.device)
        logits = self.model(x, orders=order_tensor)[0, observed:observed + end - start]
        targets = x[0, start:end]
        return float(torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).sum())

    @torch.no_grad()
    def resample_span(self, tokens, span_start, span_end, *, temperature=None, top_k=None, top_p=None):
        if not 0 <= span_start <= span_end <= len(tokens):
            raise ValueError("Span is outside the original motif-token sequence")
        if span_start == span_end:
            return list(tokens)
        self.stats["resample_calls"] += 1
        ids, offsets = self.encode_with_offsets(tokens)
        start, end = offsets[span_start], offsets[span_end]
        orders, observed = self._build_ao_order(len(ids), start, end)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        order_tensor = torch.tensor([orders], dtype=torch.long, device=self.device)
        for index in range(end - start):
            logits = self.model(x, orders=order_tensor)[0, observed + index]
            chosen = sample_logits(logits, self.temperature if temperature is None else temperature,
                                   self.top_k if top_k is None else top_k,
                                   self.top_p if top_p is None else top_p, self._suppressed_ids + [self.vocab.unk_id])
            if chosen == self.vocab.eos_id:
                self.stats["sampled_special_rejections"] += 1
                return list(tokens)
            x[0, start + index] = chosen
        pieces = [self.vocab.itos[int(i)] for i in x[0, start:end].tolist()]
        return list(tokens[:span_start]) + self.collapse_pieces(pieces) + list(tokens[span_end:])


def make_edit_policy(model, vocab, device, settings):
    validate_sampling(settings)
    if isinstance(vocab, HybridVocab):
        return HybridAOEditPolicy(model, vocab, device=device, temperature=settings["temperature"],
                                  top_k=settings["top_k"], top_p=settings.get("top_p", 1.0))
    if settings.get("top_p", 1.0) != 1.0:
        raise ValueError("Nucleus PMO sampling is implemented for hybrid checkpoints")
    return AOEditPolicy(model, vocab, device=device, temperature=settings["temperature"], top_k=settings["top_k"])

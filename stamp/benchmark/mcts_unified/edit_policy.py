"""AO-GPT-backed edit policy: score and resample a subtree span using
**bidirectional context**.

Two operations are exposed:

- ``score_span``: given a full canonical STAMP sequence and a subtree span,
  compute ``log P(span_tokens | left_context, right_context)`` — i.e. with
  *both* sides observed — under the AO-GPT.  Used to define the edit-action
  prior ∝ 1 / P (low-confidence remasking).

- ``resample_span``: re-sample tokens at the span positions, **conditioned
  on the full surrounding sequence** (left + right) by exploiting
  ``TinyDecoderLM``'s ``orders`` argument.

How AO ordering enables bidirectional infill
--------------------------------------------
``TinyDecoderLM`` is causal *over an arbitrary permutation* of token
positions, supplied via ``orders``.  Concretely if we set::

    orders = [observed_positions..., masked_positions...]

then in the shuffled sequence the masked positions come last, so each
masked position's prediction attends (causally) to **all** observed
positions — left *and* right context simultaneously.  Per-position AdaLN
conditioning on the original target position keeps the model aware of
where each token actually lives.

Scoring is one-shot: we keep the original (correct) span tokens at their
positions, run a single forward pass with the AO order, and sum the
log-probs at the masked indices.  Resampling is sequential through the
masked positions so each new token sees all previously sampled siblings.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..lm import LMModel, TokenVocab, _normalize_model_type


class AOEditPolicy:
    """Wraps an AO-GPT checkpoint to score and resample STAMP subtree spans
    with full bidirectional context."""

    def __init__(
        self,
        model: LMModel,
        vocab: TokenVocab,
        device: Optional[torch.device] = None,
        *,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.device = device or next(model.parameters()).device
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self._suppressed_ids = [
            i for i in (vocab.pad_id, vocab.bos_id, vocab.mask_id)
            if 0 <= int(i) < len(vocab.itos)
        ]
        self._max_seq_len = int(model.cfg.max_seq_len)

    # ------------------------------------------------------------------
    # AO-order machinery
    # ------------------------------------------------------------------
    def _build_ao_order(
        self,
        total_len: int,
        span_start_in_ids: int,
        span_end_in_ids: int,
    ) -> Tuple[List[int], int]:
        """Return ``(orders, n_observed)``.

        ``orders`` is a permutation of ``[0, total_len-1]`` placing observed
        positions before masked.  Position 0 (BOS) is always at the head.
        ``n_observed`` is the number of observed positions (incl. BOS).
        """
        observed = [
            p for p in range(total_len)
            if not (span_start_in_ids <= p < span_end_in_ids)
        ]
        masked = list(range(span_start_in_ids, span_end_in_ids))
        orders = observed + masked
        return orders, len(observed)

    def _truncate_to_max_seq(
        self,
        ids: List[int],
        span_start_in_ids: int,
        span_end_in_ids: int,
    ) -> Tuple[List[int], int, int]:
        """Drop tokens from the *left* (after BOS) until ``ids`` fits in
        ``max_seq_len``; return ``(new_ids, new_span_start, new_span_end)``.
        """
        if len(ids) <= self._max_seq_len:
            return ids, span_start_in_ids, span_end_in_ids
        drop = len(ids) - self._max_seq_len
        # Never drop BOS (index 0) or any of the masked span.
        max_drop_before_span = max(0, span_start_in_ids - 1)
        drop = min(drop, max_drop_before_span)
        if drop <= 0:
            # Cannot fit; truncate from the right (after span) as a last resort.
            extra = len(ids) - self._max_seq_len
            ids = ids[: self._max_seq_len]
            return ids, span_start_in_ids, min(span_end_in_ids, len(ids))
        new_ids = [ids[0]] + ids[1 + drop:]
        return new_ids, span_start_in_ids - drop, span_end_in_ids - drop

    # ------------------------------------------------------------------
    # Scoring (single forward pass, bidirectional)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_span(
        self,
        tokens: Sequence[str],
        span_start: int,
        span_end: int,
    ) -> float:
        """Return ``log P(span_tokens | left_context, right_context)`` under
        the AO-GPT.

        Implementation:
          1. ``input_ids = [BOS] + encoded(tokens)`` (already in original
             order; correct values everywhere including the span — they are
             the labels we are scoring).
          2. ``orders = [observed_positions_in_ids..., masked_positions_in_ids...]``.
          3. Single forward pass.  ``logits[i]`` predicts the i-th shuffled
             token.  For each masked position M_k at shuffled index
             ``n_obs + k``, accumulate ``log_softmax(logits[n_obs+k])[ids[M_k]]``.

        A more negative result ⇒ the model finds this span unlikely given
        both surrounding sides ⇒ higher edit-prior (1/P).
        """
        if span_end <= span_start:
            return 0.0
        ids = [self.vocab.bos_id] + self.vocab.encode_tokens(list(tokens))
        # Convert flat-token indices to input_ids indices (shifted by +1 for BOS).
        s_ids, e_ids = span_start + 1, span_end + 1
        ids, s_ids, e_ids = self._truncate_to_max_seq(ids, s_ids, e_ids)
        if e_ids <= s_ids:
            return 0.0
        T = len(ids)

        orders, n_obs = self._build_ao_order(T, s_ids, e_ids)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        orders_t = torch.tensor([orders], dtype=torch.long, device=self.device)

        logits = self.model(x, orders=orders_t)[0]  # [T, V]
        # Each masked position contributes log P(actual_token | ctx).
        logp_sum = 0.0
        for k in range(e_ids - s_ids):
            shuffled_idx = n_obs + k
            target_pos = orders[shuffled_idx]  # original position in ids
            target_id = ids[target_pos]
            logit = logits[shuffled_idx]
            log_probs = F.log_softmax(logit.float(), dim=-1)
            logp_sum += float(log_probs[int(target_id)].item())
        return logp_sum

    # ------------------------------------------------------------------
    # Resampling (sequential over masked positions, bidirectional context)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def resample_span(
        self,
        tokens: Sequence[str],
        span_start: int,
        span_end: int,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Re-sample ``tokens[span_start:span_end]`` conditioned on the full
        surrounding sequence (left + right) via AO ordering.

        Sampling order over the masked positions is left-to-right in the
        original sequence, so each new token sees all observed context plus
        all already-sampled earlier siblings.

        The output preserves the original span length — re-sample never
        changes ``len(tokens)``.  This is a deliberate simplification matched
        to STAMP's tree structure: a given subtree slot accepts at most one
        motif-group; the LM may pick a different motif but the action stays
        local to the subtree slot.
        """
        if span_end <= span_start:
            return list(tokens)
        T_temp = temperature if temperature is not None else self.temperature
        K = top_k if top_k is not None else self.top_k

        ids = [self.vocab.bos_id] + self.vocab.encode_tokens(list(tokens))
        s_ids, e_ids = span_start + 1, span_end + 1
        ids, s_ids, e_ids = self._truncate_to_max_seq(ids, s_ids, e_ids)
        if e_ids <= s_ids:
            return list(tokens)
        T = len(ids)
        n_mask = e_ids - s_ids

        orders, n_obs = self._build_ao_order(T, s_ids, e_ids)
        orders_t = torch.tensor([orders], dtype=torch.long, device=self.device)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Iteratively sample masked positions left-to-right.  After each
        # sample, write the token back into ``x`` so subsequent masked
        # positions condition on it.
        for k in range(n_mask):
            shuffled_idx = n_obs + k
            logits = self.model(x, orders=orders_t)[0, shuffled_idx, :]
            for sup in self._suppressed_ids:
                logits[int(sup)] = float("-inf")
            # Allow EOS — treat it as a filler if it appears.
            logits = logits / max(T_temp, 1e-6)
            if K > 0:
                k_eff = min(K, logits.size(-1))
                v, idx = torch.topk(logits, k=k_eff, dim=-1)
                probs = torch.softmax(v, dim=-1)
                rel = torch.multinomial(probs, num_samples=1)
                tok_id = int(idx[rel].item())
            else:
                probs = torch.softmax(logits, dim=-1)
                tok_id = int(torch.multinomial(probs, num_samples=1).item())
            target_pos = orders[shuffled_idx]
            x[0, target_pos] = int(tok_id)

        # Decode the sampled span and splice into the full token list.
        new_span_ids = [int(x[0, p].item()) for p in range(s_ids, e_ids)]
        new_span_tokens = self.vocab.decode_ids(new_span_ids)
        # Note: span length is preserved by construction.
        new_full_tokens = list(tokens[:span_start]) + new_span_tokens + list(tokens[span_end:])
        return new_full_tokens

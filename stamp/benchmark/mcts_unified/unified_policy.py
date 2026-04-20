"""UnifiedPolicyModel: dispatches between growth (AR) and edit (AO + transplant)
actions based on whether the current state is terminal.

Edit zone has **two action sources** that compete for budget:

1. **Diffusion edits** — AO-GPT bidirectional resample of a subtree span.
   Encoded with the ``EDIT_SENTINEL``.  Signal = per-token geometric mean
   probability under AO-GPT.

2. **Transplants** — splice a port-compatible donor subtree from the
   :class:`SubtreePool` (harvested from past high-scoring molecules).
   Encoded with the ``TRANSPLANT_SENTINEL``.  Signal = donor's oracle score.

Per the design lock-in: signals are normalized by per-edit-location softmax
(τ=0.5), then mass is split equally across edit locations.  PUCT does
the rest.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...core.stamp_codec import STAMPEncoder
from ..lm import LMModel, TokenVocab
from .edit_policy import AOEditPolicy
from .port_signature import heavy_atom_count, port_signature_for_span
from .subtree import (
    ParsedMolecule,
    SubtreeSpan,
    edit_candidate_spans,
    parse_terminal_sequence,
    splice_span,
)
from .subtree_pool import SubtreePool
from .transplant import (
    TRANSPLANT_SENTINEL,
    encode_transplant_action,
    splice_donor,
)

# Ensure ``mcts_protype`` is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_protype import Action, PolicyModel, STAMPState  # noqa: E402
from stamp.benchmark.mcts_guacamol import ARPolicyModel  # noqa: E402


EDIT_SENTINEL = "<EDIT>"


def _span_tag(start: int, end: int) -> str:
    return f"<span>{start}:{end}"


def _context_hash(tokens: Sequence[str], start: int, end: int) -> str:
    """Stable short hash of the (prefix, suffix) around a span — used as part
    of the action key so the same span in different contexts is distinct."""
    prefix = " ".join(tokens[:start])
    suffix = " ".join(tokens[end:])
    h = hashlib.md5(f"{prefix}|{suffix}".encode("utf-8")).hexdigest()[:10]
    return h


def encode_edit_action(
    pre_tokens: Sequence[str],
    span_start: int,
    span_end: int,
    new_flat_tokens: Sequence[str],
    prior: float,
) -> Action:
    """Pack an edit as an ``Action`` with a sentinel-prefixed tokens tuple.

    ``new_flat_tokens`` is the *full post-edit* flat token list (post-splice).
    """
    tokens: Tuple[str, ...] = (
        EDIT_SENTINEL,
        _span_tag(span_start, span_end),
        _context_hash(pre_tokens, span_start, span_end),
        *new_flat_tokens,
    )
    return Action(tokens=tokens, prior=float(prior))


def is_edit_action(action: Action) -> bool:
    return bool(action.tokens) and action.tokens[0] == EDIT_SENTINEL


def decode_edit_action(action: Action) -> Tuple[int, int, List[str]]:
    """Return ``(span_start, span_end, full_new_tokens)`` for an edit action."""
    assert is_edit_action(action)
    span_tag = action.tokens[1]
    # "<span>{start}:{end}"
    body = span_tag[len("<span>"):]
    s_str, e_str = body.split(":")
    return int(s_str), int(e_str), list(action.tokens[3:])


@dataclass
class _CandidateAction:
    """Internal: a tentative edit candidate before B' prior normalization."""
    span: SubtreeSpan
    new_flat_tokens: List[str]
    canon_smiles: str
    source: str                # "diffusion" or "transplant"
    signal: float              # per-token prob (diffusion) or donor_score (transplant)
    donor_score: float = 0.0
    donor_smiles: str = ""


class UnifiedPolicyModel(PolicyModel):
    """Wraps (growth AR LM, edit AO LM, subtree pool).  Dispatches per state."""

    def __init__(
        self,
        growth: ARPolicyModel,
        edit: AOEditPolicy,
        encoder: STAMPEncoder,
        *,
        pool: Optional[SubtreePool] = None,
        memory=None,
        edit_max_span_len: Optional[int] = 8,
        edit_temperature: float = 1.0,
        edit_top_k: int = 50,
        diffusion_samples_per_span: int = 2,
        transplants_per_span: int = 5,
        size_delta: int = 4,
        prior_softmax_temperature: float = 0.5,
        edit_prior_eps: float = 1e-3,
        edit_fallback_uniform: bool = True,
    ) -> None:
        self.growth = growth
        self.edit = edit
        self.encoder = encoder
        self.pool = pool if pool is not None else SubtreePool()
        self.memory = memory  # MoleculeMemory; if set, skip already-scored candidates
        self.edit_max_span_len = edit_max_span_len
        self.edit_temperature = float(edit_temperature)
        self.edit_top_k = int(edit_top_k)
        self.diffusion_samples_per_span = int(diffusion_samples_per_span)
        self.transplants_per_span = int(transplants_per_span)
        self.size_delta = int(size_delta)
        self.prior_softmax_temperature = float(prior_softmax_temperature)
        self.edit_prior_eps = float(edit_prior_eps)
        self.edit_fallback_uniform = bool(edit_fallback_uniform)

    # ------------------------------------------------------------------
    # Terminal-state detection: reuse the decoder's port-stack heuristic.
    # ------------------------------------------------------------------
    @staticmethod
    def is_canonical_terminal(tokens: Sequence[str]) -> bool:
        # Any state that successfully parses to a ParsedMolecule is terminal.
        return "[END]" in tokens or _has_end_shape(tokens)

    # ------------------------------------------------------------------
    # Top-k actions dispatcher
    # ------------------------------------------------------------------
    def top_k_actions(self, prefix_tokens: List[str], k: int) -> List[Action]:
        # Try to parse as a terminal molecule — if it parses, propose edit
        # actions; otherwise fall back to growth proposals.
        parsed = parse_terminal_sequence(prefix_tokens, self.encoder)
        if parsed is not None and parsed.n_groups >= 2:
            edit_actions = self._propose_edit_actions(parsed, k=k)
            if edit_actions:
                return edit_actions
        return self.growth.top_k_actions(prefix_tokens, k)

    # ------------------------------------------------------------------
    # Rollout: only used from non-terminal states in practice.
    # ------------------------------------------------------------------
    def rollout(self, prefix_tokens: List[str], max_steps: int) -> List[str]:
        parsed = parse_terminal_sequence(prefix_tokens, self.encoder)
        if parsed is not None:
            # From a terminal state the "rollout" is itself — growth-policy
            # rollout would just append to an already-complete molecule.
            return list(parsed.flat_tokens)
        return self.growth.rollout(prefix_tokens, max_steps)

    # ------------------------------------------------------------------
    # Edit proposal: mix transplant + diffusion candidates with B' priors.
    # ------------------------------------------------------------------
    def _propose_edit_actions(
        self,
        parsed: ParsedMolecule,
        k: int,
    ) -> List[Action]:
        from rdkit import Chem

        spans = edit_candidate_spans(
            parsed,
            include_root=False,
            max_span_len=self.edit_max_span_len,
        )
        if not spans:
            return []

        seen_smiles = {parsed.smiles}
        candidates_by_loc: Dict[int, List[_CandidateAction]] = defaultdict(list)
        parent_mol = Chem.MolFromSmiles(parsed.smiles)

        # ---- Transplant candidates: query pool by signature ----
        for s in spans:
            sig = port_signature_for_span(parsed, s, parent_mol)
            if sig is None:
                continue
            target_size = heavy_atom_count(parsed, s)
            donors = self.pool.query(
                sig, target_size,
                size_delta=self.size_delta,
                top_k=self.transplants_per_span,
            )
            for donor in donors:
                new_flat = splice_donor(parsed.flat_tokens, s, donor.tokens)
                if list(new_flat) == list(parsed.flat_tokens):
                    continue
                cand_smi = self._safe_canonical_smiles(new_flat)
                if not cand_smi or cand_smi in seen_smiles:
                    continue
                # Skip candidates already scored by oracle (we have their reward).
                if self.memory is not None:
                    _c, cached = self.memory.lookup(cand_smi)
                    if cached is not None:
                        continue
                seen_smiles.add(cand_smi)
                candidates_by_loc[s.group_idx].append(_CandidateAction(
                    span=s,
                    new_flat_tokens=new_flat,
                    canon_smiles=cand_smi,
                    source="transplant",
                    signal=float(donor.score),  # donor's oracle score is in [0, 1]
                    donor_score=float(donor.score),
                    donor_smiles=donor.source_smiles,
                ))
                donor.n_uses += 1

        # ---- Diffusion candidates: AO-GPT bidirectional resample ----
        for s in spans:
            for _ in range(self.diffusion_samples_per_span):
                new_flat = self.edit.resample_span(
                    parsed.flat_tokens,
                    s.token_start,
                    s.token_end,
                    temperature=self.edit_temperature,
                    top_k=self.edit_top_k,
                )
                if list(new_flat) == list(parsed.flat_tokens):
                    continue
                cand_smi = self._safe_canonical_smiles(new_flat)
                if not cand_smi or cand_smi in seen_smiles:
                    continue
                # Skip candidates already scored by oracle (we have their reward).
                if self.memory is not None:
                    _c, cached = self.memory.lookup(cand_smi)
                    if cached is not None:
                        continue
                seen_smiles.add(cand_smi)
                # Per-token geometric mean prob, in [0, 1]
                lp = self.edit.score_span(
                    parsed.flat_tokens, s.token_start, s.token_end
                )
                span_len = max(1, s.span_len)
                signal = float(math.exp(lp / span_len))
                candidates_by_loc[s.group_idx].append(_CandidateAction(
                    span=s,
                    new_flat_tokens=new_flat,
                    canon_smiles=cand_smi,
                    source="diffusion",
                    signal=signal,
                ))

        if not candidates_by_loc:
            return []

        # ---- B' prior normalization: per-loc softmax + equal mass across locs
        actions: List[Action] = []
        n_loc = max(1, len(candidates_by_loc))
        tau = max(1e-6, self.prior_softmax_temperature)
        for loc_actions in candidates_by_loc.values():
            signals = np.array([c.signal for c in loc_actions], dtype=np.float64)
            # numerically-stable softmax in [0, 1]-bounded signal space
            scaled = signals / tau
            scaled -= scaled.max()
            weights = np.exp(scaled)
            probs = weights / max(weights.sum(), 1e-12)
            for c, p in zip(loc_actions, probs):
                prior = max(self.edit_prior_eps, float(p) / n_loc)
                if c.source == "transplant":
                    actions.append(encode_transplant_action(
                        pre_tokens=parsed.flat_tokens,
                        span_start=c.span.token_start,
                        span_end=c.span.token_end,
                        full_new_tokens=c.new_flat_tokens,
                        prior=prior,
                        donor_score=c.donor_score,
                        donor_smiles=c.donor_smiles,
                    ))
                else:
                    actions.append(encode_edit_action(
                        pre_tokens=parsed.flat_tokens,
                        span_start=c.span.token_start,
                        span_end=c.span.token_end,
                        new_flat_tokens=c.new_flat_tokens,
                        prior=prior,
                    ))

        # Dedup by action key, sort by prior descending, truncate to k.
        seen = set()
        dedup: List[Action] = []
        for a in sorted(actions, key=lambda a: -a.prior):
            if a.key in seen:
                continue
            seen.add(a.key)
            dedup.append(a)
            if len(dedup) >= k:
                break
        return dedup

    # ------------------------------------------------------------------
    def _safe_canonical_smiles(self, flat_tokens: Sequence[str]) -> str:
        from rdkit import Chem
        try:
            mol = self.encoder.decode_stamp_to_mol(" ".join(list(flat_tokens) + ["[END]"]))
        except Exception:
            return ""
        if mol is None or mol.GetNumAtoms() == 0:
            return ""
        if len(Chem.GetMolFrags(mol)) != 1:
            return ""
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return ""


def _has_end_shape(tokens: Sequence[str]) -> bool:
    """Cheap heuristic: is this token list a complete molecule in terms of
    balanced joints/motifs?  Mirrors the approximation in
    ``mcts_guacamol.count_open_ports``.
    """
    motifs = 0
    joints = 0
    for t in tokens:
        if t.startswith("[J_"):
            joints += 1
        elif t and not t.startswith("["):
            motifs += 1
    if motifs == 0:
        return False
    return joints - (motifs - 1) <= 0

"""Transplant action: splice a donor subtree (from the SubtreePool) into a
target hole in the current molecule.

Canonical-form contract
-----------------------
The leading ``[B_*]`` token of a span is *parent-context* metadata (it
declares how the parent's branch is bookkept), not subtree content.  When
we transplant we therefore:

- **Strip** the leading ``[B_*]`` from the *donor* before storing it in the
  pool — donors carry only their substantive content.
- **Re-prepend** the *hole*'s original leading ``[B_*]`` (if any) when
  splicing the donor back in — the parent's branch declaration is
  preserved exactly.

This keeps the decoder's stack-machine state-transitions identical to the
original (modulo the donor's own internal ``[B_*]`` for *its* descendants).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .port_signature import (
    PortSignature,
    heavy_atom_count,
    port_signature_for_span,
)
from .subtree import ParsedMolecule, SubtreeSpan, splice_span


TRANSPLANT_SENTINEL = "<TRANSPLANT>"


# ---------------------------------------------------------------------------
# Strip / restore parent's [B_*] marker
# ---------------------------------------------------------------------------

def split_leading_branch(
    tokens: Sequence[str],
) -> Tuple[Tuple[str, ...], Optional[str]]:
    """Return ``(body_tokens, leading_branch_token_or_None)``."""
    if tokens and tokens[0].startswith("[B_"):
        return tuple(tokens[1:]), tokens[0]
    return tuple(tokens), None


def reattach_leading_branch(
    body_tokens: Sequence[str],
    leading_branch: Optional[str],
) -> List[str]:
    if leading_branch is None:
        return list(body_tokens)
    return [leading_branch] + list(body_tokens)


# ---------------------------------------------------------------------------
# Donor extraction (used during harvest)
# ---------------------------------------------------------------------------

def extract_donor_records(
    parsed: ParsedMolecule,
    *,
    include_root: bool = False,
    max_span_len: Optional[int] = 12,
):
    """Yield ``(PortSignature, body_tokens, atom_count)`` for every
    transplantable subtree in ``parsed``.

    ``body_tokens`` excludes the parent's leading ``[B_*]`` marker so the
    donor is context-free.  Use :func:`splice_donor` to re-attach it on
    transplant.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(parsed.smiles)
    for span in parsed.subtree_spans:
        if span.is_root and not include_root:
            continue
        if max_span_len is not None and span.span_len > max_span_len:
            continue
        sig = port_signature_for_span(parsed, span, mol)
        if sig is None:
            continue
        full_span_tokens = parsed.flat_tokens[span.token_start : span.token_end]
        body, _leading = split_leading_branch(full_span_tokens)
        if not body:
            continue
        atom_count = heavy_atom_count(parsed, span)
        yield sig, body, atom_count


# ---------------------------------------------------------------------------
# Splice donor into a hole in the current molecule's flat sequence
# ---------------------------------------------------------------------------

def splice_donor(
    parent_flat_tokens: Sequence[str],
    span: SubtreeSpan,
    donor_body_tokens: Sequence[str],
) -> List[str]:
    """Return the new full flat-token sequence after replacing the hole's
    body with ``donor_body_tokens`` (preserving the hole's leading ``[B_*]``).
    """
    hole_full = parent_flat_tokens[span.token_start : span.token_end]
    _hole_body, leading = split_leading_branch(hole_full)
    new_span = reattach_leading_branch(donor_body_tokens, leading)
    return splice_span(parent_flat_tokens, span.token_start, span.token_end, new_span)


# ---------------------------------------------------------------------------
# Action encoding (parallel to mcts_unified.unified_policy.encode_edit_action)
# ---------------------------------------------------------------------------

def encode_transplant_action(
    pre_tokens: Sequence[str],
    span_start: int,
    span_end: int,
    full_new_tokens: Sequence[str],
    prior: float,
    donor_score: float,
    donor_smiles: str = "",
):
    """Pack a transplant as an ``Action`` with the TRANSPLANT sentinel."""
    import hashlib
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from mcts_protype import Action  # type: ignore

    span_tag = f"<span>{span_start}:{span_end}"
    ctx = " ".join(list(pre_tokens[:span_start])) + "|" + " ".join(list(pre_tokens[span_end:]))
    ctx_hash = hashlib.md5(ctx.encode("utf-8")).hexdigest()[:10]
    # Donor identity hash makes each donor a distinct action even when two
    # donors splice to the same final tokens (rare but possible).
    donor_hash = hashlib.md5(" ".join(full_new_tokens).encode("utf-8")).hexdigest()[:8]
    tokens: Tuple[str, ...] = (
        TRANSPLANT_SENTINEL,
        span_tag,
        ctx_hash,
        donor_hash,
        f"<score>{donor_score:.4f}",
        *full_new_tokens,
    )
    return Action(tokens=tokens, prior=float(prior))


def is_transplant_action(action) -> bool:
    return bool(action.tokens) and action.tokens[0] == TRANSPLANT_SENTINEL


def decode_transplant_action(action) -> Tuple[int, int, List[str], float]:
    """Return ``(span_start, span_end, full_new_tokens, donor_score)``."""
    assert is_transplant_action(action)
    span_tag = action.tokens[1]
    s_str, e_str = span_tag[len("<span>"):].split(":")
    score_tag = action.tokens[4]
    score = float(score_tag[len("<score>"):])
    full_new = list(action.tokens[5:])
    return int(s_str), int(e_str), full_new, score

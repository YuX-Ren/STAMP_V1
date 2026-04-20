"""STAMP subtree parsing + flat token-span extraction.

Given a complete (terminal) STAMP token sequence, we re-encode it through
``STAMPEncoder.encode_mol_to_groups`` to obtain the canonical group list,
then compute each group's subtree span in the flat token sequence.

The canonical tokens (``flat_tokens``) become the authoritative state after
entering the edit zone — subsequent edit actions splice in-place on
``flat_tokens``, not on whatever token path the growth MCTS used to reach
this molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

from ...core.stamp_codec import STAMPEncoder, STAMPGroup


@dataclass(frozen=True)
class SubtreeSpan:
    """A candidate edit span in a canonical STAMP sequence.

    Attributes
    ----------
    group_idx         : index of the root group of this subtree in the
                        canonical ``groups`` list.
    group_count       : number of groups this subtree covers (= the group's
                        ``subtree_group_count``).
    token_start       : start of the flat token span (inclusive).
    token_end         : end of the flat token span (exclusive).
    is_root           : whether this subtree is the whole molecule.
    depth             : rough depth from the root group (0 = root).
    """

    group_idx: int
    group_count: int
    token_start: int
    token_end: int
    is_root: bool
    depth: int

    @property
    def span_len(self) -> int:
        return self.token_end - self.token_start


@dataclass
class ParsedMolecule:
    """Result of re-parsing a terminal STAMP sequence into groups + spans."""

    smiles: str                     # canonical SMILES
    groups: List[STAMPGroup]        # canonical group list from the encoder
    flat_tokens: List[str]          # canonical flat sequence
    subtree_spans: List[SubtreeSpan]  # one per group

    @property
    def n_groups(self) -> int:
        return len(self.groups)


def _compute_group_depths(groups: Sequence[STAMPGroup]) -> List[int]:
    """Depth of each group from the root (0 = root)."""
    # Build frag_id -> group_idx mapping (frag_id is unique per group)
    id_to_idx = {g.frag_id: i for i, g in enumerate(groups)}
    depths: List[int] = [0] * len(groups)
    for i, g in enumerate(groups):
        if g.parent_frag_id is None:
            depths[i] = 0
            continue
        parent_idx = id_to_idx.get(g.parent_frag_id)
        if parent_idx is None or parent_idx >= i:
            # Safety fallback: depth 1
            depths[i] = 1
        else:
            depths[i] = depths[parent_idx] + 1
    return depths


def parse_terminal_sequence(
    tokens: Sequence[str],
    encoder: STAMPEncoder,
) -> Optional[ParsedMolecule]:
    """Re-parse a terminal STAMP token sequence into canonical groups + spans.

    Returns ``None`` if the sequence does not decode to a valid connected
    molecule.  Uses the codec's R4 fallbacks to recover from minor issues.
    """
    cleaned = [t for t in tokens if t and not t.startswith("<")]
    if not cleaned:
        return None
    if "[END]" not in cleaned:
        cleaned = cleaned + ["[END]"]
    try:
        mol = encoder.decode_stamp_to_mol(" ".join(cleaned))
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

    # Re-encode via the codec to get canonical groups + subtree metadata.
    try:
        groups = encoder.encode_mol_to_groups(mol)
    except Exception:
        return None
    if not groups:
        return None

    flat: List[str] = []
    group_token_starts: List[int] = []
    for g in groups:
        group_token_starts.append(len(flat))
        flat.extend(g.tokens)
    group_token_starts.append(len(flat))  # sentinel

    depths = _compute_group_depths(groups)
    spans: List[SubtreeSpan] = []
    for i, g in enumerate(groups):
        subtree_count = max(1, int(g.subtree_group_count))
        start = group_token_starts[i]
        # End at the end of the last group in the subtree.  Encoder emits
        # subtree groups contiguously (DFS order), so the span is
        # [start, group_token_starts[i + subtree_count]).
        end_idx = min(i + subtree_count, len(groups))
        end = group_token_starts[end_idx]
        spans.append(
            SubtreeSpan(
                group_idx=i,
                group_count=subtree_count,
                token_start=start,
                token_end=end,
                is_root=(i == 0),
                depth=depths[i],
            )
        )

    return ParsedMolecule(
        smiles=smiles,
        groups=groups,
        flat_tokens=flat,
        subtree_spans=spans,
    )


def splice_span(
    flat_tokens: Sequence[str],
    span_start: int,
    span_end: int,
    new_tokens: Sequence[str],
) -> List[str]:
    """Return ``flat_tokens`` with ``flat_tokens[span_start:span_end]`` replaced."""
    out: List[str] = list(flat_tokens[:span_start])
    out.extend(new_tokens)
    out.extend(flat_tokens[span_end:])
    return out


def edit_candidate_spans(
    parsed: ParsedMolecule,
    *,
    include_root: bool = False,
    max_span_len: Optional[int] = None,
) -> List[SubtreeSpan]:
    """Enumerate candidate edit spans.

    We exclude the whole-molecule root by default (editing the root is
    equivalent to resampling everything from scratch — growth zone already
    handles that via rollout).  Toggle ``include_root=True`` to allow it.
    """
    out: List[SubtreeSpan] = []
    for s in parsed.subtree_spans:
        if s.is_root and not include_root:
            continue
        if max_span_len is not None and s.span_len > max_span_len:
            continue
        out.append(s)
    return out

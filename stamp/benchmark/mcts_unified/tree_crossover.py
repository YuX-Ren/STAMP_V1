"""STAMP-native tree crossover."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from rdkit import Chem

from ...core.stamp_codec import STAMPEncoder
from .port_signature import heavy_atom_count, port_signature_for_span
from .subtree import ParsedMolecule, SubtreeSpan, edit_candidate_spans, parse_terminal_sequence
from .transplant import splice_donor, split_leading_branch


def subtree_topo_hash(parsed: ParsedMolecule, span: SubtreeSpan) -> Tuple:
    """Canonical hash of subtree's tree-shape.

    Two subtrees with equal ``subtree_topo_hash`` are topologically isomorphic
    (same children count at every internal node, recursively).  Tuples are
    sorted at each level so the hash is order-invariant.

    This is the topology-aware replacement for the ``port_signature + size_delta``
    filter in ``tree_crossover``: instead of matching only the root motif's
    declared join count, we require the *entire* subtree shape to align.

    Examples (informal):
      leaf                        -> ()
      root + 1 leaf child         -> ((),)
      root + 2 leaf children      -> ((), ())
      root + (inner with 2 leaves) -> (((), ()),)
    """
    root_idx = span.group_idx
    n = span.group_count
    if n <= 0:
        return ()
    groups = parsed.groups[root_idx : root_idx + n]
    # Build parent_frag_id -> [child frag_ids] map within this subtree only.
    in_subtree = {g.frag_id for g in groups}
    children = {fid: [] for fid in in_subtree}
    for g in groups[1:]:
        if g.parent_frag_id in in_subtree:
            children[g.parent_frag_id].append(g.frag_id)

    def _rec(fid: int) -> Tuple:
        ch = children.get(fid, [])
        return tuple(sorted(_rec(c) for c in ch))

    return _rec(groups[0].frag_id)


def tree_crossover(
    parent_a_smi: str,
    parent_b_smi: str,
    encoder: STAMPEncoder,
    *,
    size_delta: int = 4,
    max_tries: int = 3,
    match_mode: str = "port_sig",
    parse_cache=None,
    n_swaps: int = 1,
) -> Optional[str]:
    # parse_cache: optional object with a .parsed(smi) -> ParsedMolecule method
    # (the optimizer's _STAMPCache). Population parents are reused 100+ times, so
    # caching encode+parse here removes the dominant cost (~24ms x ~25 retries/call).
    try:
        if parse_cache is not None:
            parsed_a = parse_cache.parsed(parent_a_smi)
            parsed_b = parse_cache.parsed(parent_b_smi)
        else:
            parsed_a = parse_terminal_sequence(encoder.encode_smiles(parent_a_smi).split(), encoder)
            parsed_b = parse_terminal_sequence(encoder.encode_smiles(parent_b_smi).split(), encoder)
    except Exception:
        return None
    if parsed_a is None or parsed_b is None:
        return None
    # mol_a/mol_b only needed for port_signature (port_sig mode).
    mol_a = mol_b = None
    if match_mode == "port_sig":
        mol_a = Chem.MolFromSmiles(parsed_a.smiles)
        mol_b = Chem.MolFromSmiles(parsed_b.smiles)
        if mol_a is None or mol_b is None:
            return None
    spans_a = edit_candidate_spans(parsed_a, include_root=False)
    spans_b = edit_candidate_spans(parsed_b, include_root=False)
    if not spans_a or not spans_b:
        return None
    # Precompute matching keys once per parent if topo_hash mode (cheap).
    if match_mode == "topo_hash":
        keys_b = {id(s): subtree_topo_hash(parsed_b, s) for s in spans_b}
        keys_a = {id(s): subtree_topo_hash(parsed_a, s) for s in spans_a}
        size_a_map = {id(s): heavy_atom_count(parsed_a, s) for s in spans_a}
    def _find_holes(span_b):
        """Matching holes in A for a given donor span_b under the active mode."""
        if match_mode == "random":
            return list(spans_a)
        if match_mode == "topo_hash":
            key_b = keys_b[id(span_b)]
            size_b = heavy_atom_count(parsed_b, span_b)
            return [
                s_a for s_a in spans_a
                if keys_a[id(s_a)] == key_b and abs(size_a_map[id(s_a)] - size_b) <= size_delta
            ]
        sig_b = port_signature_for_span(parsed_b, span_b, mol_b)
        if sig_b is None:
            return []
        size_b = heavy_atom_count(parsed_b, span_b)
        out = []
        for s_a in spans_a:
            sig_a = port_signature_for_span(parsed_a, s_a, mol_a)
            if sig_a is None or sig_a != sig_b:
                continue
            if abs(heavy_atom_count(parsed_a, s_a) - size_b) > size_delta:
                continue
            out.append(s_a)
        return out

    def _disjoint(hole, chosen):
        """True if hole's token range doesn't overlap/nest with any already chosen."""
        for h, _ in chosen:
            if not (hole.token_end <= h.token_start or hole.token_start >= h.token_end):
                return False
        return True

    k = max(1, int(n_swaps))
    for _ in range(max_tries):
        # Collect up to k disjoint (hole_a, donor_body) pairs.
        chosen = []  # list of (hole_span, donor_body_tokens)
        for span_b in random.sample(spans_b, len(spans_b)):
            holes = _find_holes(span_b)
            random.shuffle(holes)
            hole = next((h for h in holes if _disjoint(h, chosen)), None)
            if hole is None:
                continue
            donor_body, _ = split_leading_branch(
                parsed_b.flat_tokens[span_b.token_start : span_b.token_end]
            )
            if not donor_body:
                continue
            chosen.append((hole, donor_body))
            if len(chosen) >= k:
                break
        if not chosen:
            continue
        # Splice right-to-left so earlier token indices stay valid.
        new_flat = list(parsed_a.flat_tokens)
        for hole, donor_body in sorted(chosen, key=lambda x: -x[0].token_start):
            new_flat = splice_donor(new_flat, hole, donor_body)
        cleaned = [t for t in new_flat if t and not t.startswith("<")]
        if "[END]" not in cleaned:
            cleaned = cleaned + ["[END]"]
        try:
            child_mol = encoder.decode_stamp_to_mol(" ".join(cleaned))
        except Exception:
            continue
        if child_mol is None or child_mol.GetNumAtoms() == 0:
            continue
        if len(Chem.GetMolFrags(child_mol)) != 1:
            continue
        try:
            return Chem.MolToSmiles(child_mol, canonical=True, isomericSmiles=True)
        except Exception:
            continue
    return None

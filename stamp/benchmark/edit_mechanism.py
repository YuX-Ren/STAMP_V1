"""Mechanistic edit-locality analysis for STAMP and complete languages.

The routines in this module are deliberately descriptive rather than a
claim of a unique chemical edit path.  They provide reproducible summaries
for questions such as:

- which STAMP edit axis changes across paired molecules;
- whether a language localizes a paired edit into few token spans;
- whether a STAMP token change falls inside one contiguous subtree span.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..core.stamp_codec import STAMPEncoder, STAMPGroup
from .common import canonicalize_smiles
from .representations import BaseRepresentation, build_representation

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StampNode:
    group_idx: int
    frag_id: int
    motif: str
    tokens: Tuple[str, ...]
    parent_idx: Optional[int]
    child_indices: Tuple[int, ...]
    subtree_group_count: int
    depth: int
    token_start: int
    token_end: int
    leading_branch: Optional[str]
    parent_joint: Optional[str]
    joint_signature: Tuple[str, ...]


@dataclass(frozen=True)
class StampTree:
    smiles: str
    stamp: str
    groups: Tuple[STAMPGroup, ...]
    nodes: Tuple[StampNode, ...]
    flat_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    a_to_b: Mapping[int, int]
    b_to_a: Mapping[int, int]
    substitution_a_to_b: Mapping[int, int]
    substitution_b_to_a: Mapping[int, int]
    motif_insertions: int
    motif_deletions: int
    motif_substitutions: int


def _is_joint(token: str) -> bool:
    return token.startswith("[J_") and token.endswith("]")


def _is_branch(token: str) -> bool:
    return token.startswith("[B_") and token.endswith("]")


def _parse_branch_count(token: str) -> Optional[int]:
    if not _is_branch(token):
        return None
    try:
        return int(token[3:-1])
    except ValueError:
        return None


def _motif_pos(group: STAMPGroup) -> int:
    for i, token in enumerate(group.tokens):
        if token == group.motif:
            return i
    return 0


def _parent_joint(group: STAMPGroup) -> Optional[str]:
    if group.parent_frag_id is None:
        return None
    start = _motif_pos(group) + 1
    for token in group.tokens[start:]:
        if _is_joint(token):
            return token
    return None


def _joint_signature(group: STAMPGroup) -> Tuple[str, ...]:
    return tuple(
        token
        for token in group.tokens
        if _is_joint(token) or token.startswith("[S_") or token == "[Ring]"
    )


def _leading_branch(group: STAMPGroup) -> Optional[str]:
    if group.tokens and _is_branch(group.tokens[0]):
        return group.tokens[0]
    return None


def _compute_depths(groups: Sequence[STAMPGroup]) -> List[int]:
    id_to_idx = {int(group.frag_id): i for i, group in enumerate(groups)}
    depths = [0] * len(groups)
    for i, group in enumerate(groups):
        parent_id = group.parent_frag_id
        if parent_id is None:
            depths[i] = 0
            continue
        parent_idx = id_to_idx.get(int(parent_id))
        if parent_idx is None or parent_idx >= i:
            depths[i] = 1
        else:
            depths[i] = depths[parent_idx] + 1
    return depths


def build_stamp_tree(smiles: str, encoder: STAMPEncoder) -> StampTree:
    """Encode a molecule into STAMP groups plus tree/span metadata."""

    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.MolFromSmiles(canonical)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Disconnected molecules are not supported by group-level STAMP analysis.")

    groups = tuple(encoder.encode_mol_to_groups(mol))
    flat_tokens: List[str] = []
    starts: List[int] = []
    for group in groups:
        starts.append(len(flat_tokens))
        flat_tokens.extend(group.tokens)
    starts.append(len(flat_tokens))

    id_to_idx = {int(group.frag_id): i for i, group in enumerate(groups)}
    children: Dict[int, List[int]] = defaultdict(list)
    parent_idx_by_group: List[Optional[int]] = []
    for i, group in enumerate(groups):
        parent_id = group.parent_frag_id
        if parent_id is None:
            parent_idx_by_group.append(None)
            continue
        parent_idx = id_to_idx.get(int(parent_id))
        parent_idx_by_group.append(parent_idx)
        if parent_idx is not None:
            children[parent_idx].append(i)

    depths = _compute_depths(groups)
    nodes: List[StampNode] = []
    for i, group in enumerate(groups):
        subtree_count = max(1, int(group.subtree_group_count))
        token_end_idx = min(i + subtree_count, len(groups))
        nodes.append(
            StampNode(
                group_idx=i,
                frag_id=int(group.frag_id),
                motif=str(group.motif),
                tokens=tuple(group.tokens),
                parent_idx=parent_idx_by_group[i],
                child_indices=tuple(children.get(i, [])),
                subtree_group_count=subtree_count,
                depth=int(depths[i]),
                token_start=int(starts[i]),
                token_end=int(starts[token_end_idx]),
                leading_branch=_leading_branch(group),
                parent_joint=_parent_joint(group),
                joint_signature=_joint_signature(group),
            )
        )

    return StampTree(
        smiles=canonical,
        stamp=" ".join(flat_tokens),
        groups=groups,
        nodes=tuple(nodes),
        flat_tokens=tuple(flat_tokens),
    )


def _align_equal_motifs_in_replace_blocks(
    motifs_a: Sequence[str],
    motifs_b: Sequence[str],
    opcodes: Sequence[Tuple[str, int, int, int, int]],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], int, int, int]:
    a_to_b: Dict[int, int] = {}
    b_to_a: Dict[int, int] = {}
    substitution_a_to_b: Dict[int, int] = {}
    substitution_b_to_a: Dict[int, int] = {}
    insertions = 0
    deletions = 0
    substitutions = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for off in range(i2 - i1):
                ai = i1 + off
                bi = j1 + off
                a_to_b[ai] = bi
                b_to_a[bi] = ai
            continue
        if tag == "insert":
            insertions += j2 - j1
            continue
        if tag == "delete":
            deletions += i2 - i1
            continue

        # Within a replace block, retain exact motif matches greedily in
        # sequence order.  This keeps repeated motifs deterministic without
        # introducing a heavyweight graph edit solver.
        unused_b_by_motif: Dict[str, List[int]] = defaultdict(list)
        for bj in range(j1, j2):
            unused_b_by_motif[motifs_b[bj]].append(bj)
        unmatched_a: List[int] = []
        matched_b: set[int] = set()
        for ai in range(i1, i2):
            candidates = unused_b_by_motif.get(motifs_a[ai], [])
            bj = None
            while candidates:
                candidate = candidates.pop(0)
                if candidate not in matched_b:
                    bj = candidate
                    break
            if bj is None:
                unmatched_a.append(ai)
            else:
                a_to_b[ai] = bj
                b_to_a[bj] = ai
                matched_b.add(bj)
        unmatched_b = [bj for bj in range(j1, j2) if bj not in matched_b]
        n_sub = min(len(unmatched_a), len(unmatched_b))
        substitutions += n_sub
        for ai, bj in zip(unmatched_a[:n_sub], unmatched_b[:n_sub]):
            substitution_a_to_b[ai] = bj
            substitution_b_to_a[bj] = ai
        deletions += max(0, len(unmatched_a) - len(unmatched_b))
        insertions += max(0, len(unmatched_b) - len(unmatched_a))

    return (
        a_to_b,
        b_to_a,
        substitution_a_to_b,
        substitution_b_to_a,
        insertions,
        deletions,
        substitutions,
    )


def align_stamp_groups(tree_a: StampTree, tree_b: StampTree) -> AlignmentResult:
    motifs_a = [node.motif for node in tree_a.nodes]
    motifs_b = [node.motif for node in tree_b.nodes]
    matcher = SequenceMatcher(a=motifs_a, b=motifs_b, autojunk=False)
    (
        a_to_b,
        b_to_a,
        substitution_a_to_b,
        substitution_b_to_a,
        insertions,
        deletions,
        substitutions,
    ) = _align_equal_motifs_in_replace_blocks(
        motifs_a,
        motifs_b,
        matcher.get_opcodes(),
    )
    return AlignmentResult(
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        substitution_a_to_b=substitution_a_to_b,
        substitution_b_to_a=substitution_b_to_a,
        motif_insertions=int(insertions),
        motif_deletions=int(deletions),
        motif_substitutions=int(substitutions),
    )


def _unmatched_subtree_roots(nodes: Sequence[StampNode], unmatched: set[int]) -> int:
    roots = 0
    for idx in unmatched:
        parent = nodes[idx].parent_idx
        if parent is None or parent not in unmatched:
            roots += 1
    return roots


def _smallest_covering_subtree(
    nodes: Sequence[StampNode],
    changed_min: int,
    changed_max: int,
    *,
    require_nonroot: bool = False,
) -> Optional[StampNode]:
    if changed_min < 0 or changed_max < 0:
        return None
    candidates = [
        node
        for node in nodes
        if node.token_start <= changed_min
        and changed_max <= node.token_end
        and (not require_nonroot or node.parent_idx is not None)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda node: (
            node.token_end - node.token_start,
            node.subtree_group_count,
            -node.depth,
        ),
    )


def _typed_patch_kind(motif_delta: int, joint_delta: int, branch_delta: int) -> str:
    axes: List[str] = []
    if motif_delta:
        axes.append("motif")
    if joint_delta:
        axes.append("joint")
    if branch_delta:
        axes.append("branch")
    if not axes:
        return "no_change"
    if axes == ["motif"]:
        return "motif_patch"
    if axes == ["joint"]:
        return "joint_patch"
    if axes == ["branch"]:
        return "branch_topology_patch"
    if "branch" in axes:
        return "subtree_patch"
    return "motif_joint_patch"


def decompose_stamp_pair(
    smiles_a: str,
    smiles_b: str,
    *,
    encoder: Optional[STAMPEncoder] = None,
    edit_class: str = "all",
    pair_id: str = "",
) -> Dict[str, Any]:
    """Return pair-level STAMP motif/joint/branch edit descriptors."""

    enc = encoder or STAMPEncoder()
    tree_a = build_stamp_tree(smiles_a, enc)
    tree_b = build_stamp_tree(smiles_b, enc)
    alignment = align_stamp_groups(tree_a, tree_b)

    matched_pairs = sorted(alignment.a_to_b.items())
    paired_a_to_b = dict(alignment.a_to_b)
    paired_a_to_b.update(alignment.substitution_a_to_b)
    paired_b_to_a = dict(alignment.b_to_a)
    paired_b_to_a.update(alignment.substitution_b_to_a)
    joint_signature_changes = 0
    parent_joint_changes = 0
    branch_parent_changes = 0
    branch_degree_changes = 0
    branch_token_changes = 0
    subtree_size_changes = 0

    for ai, bi in matched_pairs:
        node_a = tree_a.nodes[ai]
        node_b = tree_b.nodes[bi]
        if node_a.joint_signature != node_b.joint_signature:
            joint_signature_changes += 1
        if node_a.parent_joint != node_b.parent_joint:
            parent_joint_changes += 1
        if node_a.leading_branch != node_b.leading_branch:
            branch_token_changes += 1
        if node_a.subtree_group_count != node_b.subtree_group_count:
            subtree_size_changes += 1

        parent_a = node_a.parent_idx
        parent_b = node_b.parent_idx
        mapped_parent_a = paired_a_to_b.get(parent_a) if parent_a is not None else None
        if mapped_parent_a != parent_b:
            branch_parent_changes += 1

        if len(node_a.child_indices) != len(node_b.child_indices):
            branch_degree_changes += 1

    for ai, bi in sorted(alignment.substitution_a_to_b.items()):
        node_a = tree_a.nodes[ai]
        node_b = tree_b.nodes[bi]
        parent_a = node_a.parent_idx
        parent_b = node_b.parent_idx
        mapped_parent_a = paired_a_to_b.get(parent_a) if parent_a is not None else None
        if mapped_parent_a != parent_b:
            branch_parent_changes += 1
        if len(node_a.child_indices) != len(node_b.child_indices):
            branch_degree_changes += 1
        if node_a.leading_branch != node_b.leading_branch:
            branch_token_changes += 1
        if node_a.subtree_group_count != node_b.subtree_group_count:
            subtree_size_changes += 1

    unmatched_a = set(range(len(tree_a.nodes))) - set(paired_a_to_b.keys())
    unmatched_b = set(range(len(tree_b.nodes))) - set(paired_b_to_a.keys())
    deleted_subtree_roots = _unmatched_subtree_roots(tree_a.nodes, unmatched_a)
    inserted_subtree_roots = _unmatched_subtree_roots(tree_b.nodes, unmatched_b)

    motif_delta = (
        alignment.motif_insertions
        + alignment.motif_deletions
        + alignment.motif_substitutions
    )
    joint_delta = joint_signature_changes
    branch_delta = (
        branch_parent_changes
        + branch_degree_changes
        + branch_token_changes
        + inserted_subtree_roots
        + deleted_subtree_roots
    )

    token_metrics = token_edit_metrics(tree_a.flat_tokens, tree_b.flat_tokens)
    subtree_a = changed_region_within_one_subtree(
        tree_a.nodes,
        token_metrics["changed_min_a"],
        token_metrics["changed_max_a"],
    )
    subtree_b = changed_region_within_one_subtree(
        tree_b.nodes,
        token_metrics["changed_min_b"],
        token_metrics["changed_max_b"],
    )
    patch_node_a = _smallest_covering_subtree(
        tree_a.nodes,
        token_metrics["changed_min_a"],
        token_metrics["changed_max_a"],
        require_nonroot=True,
    )
    patch_node_b = _smallest_covering_subtree(
        tree_b.nodes,
        token_metrics["changed_min_b"],
        token_metrics["changed_max_b"],
        require_nonroot=True,
    )
    typed_patch_kind = _typed_patch_kind(int(motif_delta), int(joint_delta), int(branch_delta))
    source_patch_port_signature = (
        _port_signature_for_node(tree_a, patch_node_a) if patch_node_a is not None else None
    )
    target_patch_port_signature = (
        _port_signature_for_node(tree_b, patch_node_b) if patch_node_b is not None else None
    )
    executable_as_one_typed_patch = bool(
        typed_patch_kind != "no_change"
        and (
            token_metrics["one_local_span"]
            or patch_node_a is not None
            or patch_node_b is not None
        )
    )
    transferable_patch = bool(
        typed_patch_kind != "no_change"
        and (source_patch_port_signature is not None or target_patch_port_signature is not None)
    )
    source_patch_span = (
        patch_node_a.token_end - patch_node_a.token_start if patch_node_a is not None else None
    )
    target_patch_span = (
        patch_node_b.token_end - patch_node_b.token_start if patch_node_b is not None else None
    )
    executable_patch_span = min(
        [span for span in (source_patch_span, target_patch_span) if span is not None],
        default=None,
    )
    patch_axes = "+".join(
        axis
        for axis, value in (
            ("motif", motif_delta),
            ("joint", joint_delta),
            ("branch", branch_delta),
        )
        if value
    ) or "none"

    return {
        "pair_id": pair_id,
        "edit_class": edit_class or "all",
        "smiles_a": tree_a.smiles,
        "smiles_b": tree_b.smiles,
        "stamp_a": tree_a.stamp,
        "stamp_b": tree_b.stamp,
        "n_groups_a": len(tree_a.nodes),
        "n_groups_b": len(tree_b.nodes),
        "n_matched_groups": len(matched_pairs),
        "n_substituted_group_pairs": len(alignment.substitution_a_to_b),
        "motif_insertions": int(alignment.motif_insertions),
        "motif_deletions": int(alignment.motif_deletions),
        "motif_substitutions": int(alignment.motif_substitutions),
        "delta_motif": int(motif_delta),
        "joint_signature_changes": int(joint_signature_changes),
        "parent_joint_changes": int(parent_joint_changes),
        "delta_joint": int(joint_delta),
        "branch_parent_changes": int(branch_parent_changes),
        "branch_degree_changes": int(branch_degree_changes),
        "branch_token_changes": int(branch_token_changes),
        "subtree_size_changes": int(subtree_size_changes),
        "inserted_subtree_roots": int(inserted_subtree_roots),
        "deleted_subtree_roots": int(deleted_subtree_roots),
        "delta_branch": int(branch_delta),
        "delta_stamp_total": int(motif_delta + joint_delta + branch_delta),
        "stamp_token_edit_distance": token_metrics["edit_distance"],
        "stamp_normalized_token_edit_distance": token_metrics["normalized_edit_distance"],
        "stamp_edit_script_length": token_metrics["edit_script_length"],
        "stamp_changed_token_fraction": token_metrics["changed_token_fraction"],
        "stamp_changed_span_count": token_metrics["changed_span_count"],
        "stamp_disconnected_edit_regions": token_metrics["disconnected_edit_regions"],
        "stamp_largest_changed_span": token_metrics["largest_changed_span"],
        "stamp_minimal_contiguous_edit_span": token_metrics["minimal_contiguous_edit_span"],
        "stamp_one_local_span": token_metrics["one_local_span"],
        "stamp_changed_within_one_subtree_a": subtree_a,
        "stamp_changed_within_one_subtree_b": subtree_b,
        "stamp_subtree_contiguous_edit": bool(subtree_a and subtree_b),
        "stamp_patch_axes": patch_axes,
        "stamp_typed_patch_kind": typed_patch_kind,
        "stamp_executable_as_one_typed_patch": executable_as_one_typed_patch,
        "stamp_transferable_to_compatible_scaffold": transferable_patch,
        "stamp_source_patch_span": source_patch_span,
        "stamp_target_patch_span": target_patch_span,
        "stamp_executable_patch_span": executable_patch_span,
        "stamp_source_patch_port_signature": str(source_patch_port_signature)
        if source_patch_port_signature is not None
        else "",
        "stamp_target_patch_port_signature": str(target_patch_port_signature)
        if target_patch_port_signature is not None
        else "",
    }


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return int(previous[-1])


def token_edit_metrics(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> Dict[str, Any]:
    matcher = SequenceMatcher(a=list(tokens_a), b=list(tokens_b), autojunk=False)
    edit_distance = levenshtein_distance(tokens_a, tokens_b)
    denom = max(1, max(len(tokens_a), len(tokens_b)))

    changed_ranges_a: List[Tuple[int, int]] = []
    changed_ranges_b: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            changed_ranges_a.append((i1, i2))
        if j2 > j1:
            changed_ranges_b.append((j1, j2))

    span_count_a = len(changed_ranges_a)
    span_count_b = len(changed_ranges_b)
    largest_a = max((end - start for start, end in changed_ranges_a), default=0)
    largest_b = max((end - start for start, end in changed_ranges_b), default=0)
    changed_min_a = min((start for start, _end in changed_ranges_a), default=-1)
    changed_max_a = max((end for _start, end in changed_ranges_a), default=-1)
    changed_min_b = min((start for start, _end in changed_ranges_b), default=-1)
    changed_max_b = max((end for _start, end in changed_ranges_b), default=-1)
    minimal_span_a = changed_max_a - changed_min_a if changed_min_a >= 0 else 0
    minimal_span_b = changed_max_b - changed_min_b if changed_min_b >= 0 else 0

    return {
        "edit_distance": int(edit_distance),
        "normalized_edit_distance": float(edit_distance / denom),
        "edit_script_length": int(edit_distance),
        "changed_token_fraction": float(edit_distance / denom),
        "changed_span_count_a": int(span_count_a),
        "changed_span_count_b": int(span_count_b),
        "changed_span_count": int(max(span_count_a, span_count_b)),
        "disconnected_edit_regions": int(max(span_count_a, span_count_b)),
        "largest_changed_span_a": int(largest_a),
        "largest_changed_span_b": int(largest_b),
        "largest_changed_span": int(max(largest_a, largest_b)),
        "minimal_contiguous_edit_span_a": int(minimal_span_a),
        "minimal_contiguous_edit_span_b": int(minimal_span_b),
        "minimal_contiguous_edit_span": int(max(minimal_span_a, minimal_span_b)),
        "one_local_span": bool(max(span_count_a, span_count_b) <= 1),
        "changed_min_a": int(changed_min_a),
        "changed_max_a": int(changed_max_a),
        "changed_min_b": int(changed_min_b),
        "changed_max_b": int(changed_max_b),
    }


def changed_region_within_one_subtree(
    nodes: Sequence[StampNode],
    changed_min: int,
    changed_max: int,
) -> bool:
    if changed_min < 0 or changed_max < 0:
        return True
    for node in nodes:
        if node.token_start <= changed_min and changed_max <= node.token_end:
            return True
    return False


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def summarize_numeric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "all") or "all")].append(row)

    summary: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        out: Dict[str, Any] = {group_key: key, "n_pairs": len(group_rows)}
        for metric in metric_keys:
            vals: List[float] = []
            for row in group_rows:
                value = row.get(metric)
                if isinstance(value, bool):
                    vals.append(float(value))
                elif isinstance(value, (int, float)) and not math.isnan(float(value)):
                    vals.append(float(value))
            out[f"mean_{metric}"] = _mean(vals)
        summary.append(out)
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def analyze_cross_language_pair(
    smiles_a: str,
    smiles_b: str,
    *,
    representation: BaseRepresentation,
    pair_id: str = "",
    edit_class: str = "all",
    stamp_encoder: Optional[STAMPEncoder] = None,
) -> Dict[str, Any]:
    text_a = representation.encode(smiles_a)
    text_b = representation.encode(smiles_b)
    tokens_a = representation.tokenize(text_a)
    tokens_b = representation.tokenize(text_b)
    metrics = token_edit_metrics(tokens_a, tokens_b)
    row: Dict[str, Any] = {
        "pair_id": pair_id,
        "edit_class": edit_class or "all",
        "representation": representation.name,
        "smiles_a": canonicalize_smiles(smiles_a) or smiles_a,
        "smiles_b": canonicalize_smiles(smiles_b) or smiles_b,
        "n_tokens_a": len(tokens_a),
        "n_tokens_b": len(tokens_b),
        "edit_distance": metrics["edit_distance"],
        "edit_script_length": metrics["edit_script_length"],
        "normalized_token_edit_distance": metrics["normalized_edit_distance"],
        "changed_token_fraction": metrics["changed_token_fraction"],
        "changed_span_count": metrics["changed_span_count"],
        "disconnected_edit_regions": metrics["disconnected_edit_regions"],
        "changed_span_count_a": metrics["changed_span_count_a"],
        "changed_span_count_b": metrics["changed_span_count_b"],
        "largest_changed_span": metrics["largest_changed_span"],
        "largest_changed_span_a": metrics["largest_changed_span_a"],
        "largest_changed_span_b": metrics["largest_changed_span_b"],
        "minimal_contiguous_edit_span": metrics["minimal_contiguous_edit_span"],
        "minimal_contiguous_edit_span_a": metrics["minimal_contiguous_edit_span_a"],
        "minimal_contiguous_edit_span_b": metrics["minimal_contiguous_edit_span_b"],
        "one_local_span": metrics["one_local_span"],
    }
    if representation.name == "stamp":
        enc = stamp_encoder or getattr(representation, "codec", None) or STAMPEncoder()
        try:
            tree_a = build_stamp_tree(smiles_a, enc)
            tree_b = build_stamp_tree(smiles_b, enc)
            row["changed_within_one_subtree_a"] = changed_region_within_one_subtree(
                tree_a.nodes,
                metrics["changed_min_a"],
                metrics["changed_max_a"],
            )
            row["changed_within_one_subtree_b"] = changed_region_within_one_subtree(
                tree_b.nodes,
                metrics["changed_min_b"],
                metrics["changed_max_b"],
            )
            row["subtree_contiguous_edit"] = bool(
                row["changed_within_one_subtree_a"] and row["changed_within_one_subtree_b"]
            )
        except Exception as exc:
            row["stamp_subtree_error"] = str(exc)
            row["subtree_contiguous_edit"] = False
    return row


def build_representations(
    names: Sequence[str],
    *,
    stamp_vocab: Optional[str] = None,
    stamp_force_ring_aware_encode: bool = False,
    strict: bool = False,
) -> Tuple[List[BaseRepresentation], List[Dict[str, str]]]:
    reps: List[BaseRepresentation] = []
    errors: List[Dict[str, str]] = []
    for name in names:
        clean = name.strip()
        if not clean:
            continue
        try:
            reps.append(
                build_representation(
                    clean,
                    stamp_vocab=stamp_vocab,
                    stamp_force_ring_aware_encode=stamp_force_ring_aware_encode,
                )
            )
        except Exception as exc:
            if strict:
                raise
            errors.append({"representation": clean, "error": str(exc)})
    return reps, errors


@dataclass(frozen=True)
class DonorRecord:
    signature: Any
    body_tokens: Tuple[str, ...]
    group_count: int
    atom_count: int
    source_smiles: str


def _split_leading_branch(tokens: Sequence[str]) -> Tuple[Tuple[str, ...], Optional[str]]:
    if tokens and _is_branch(tokens[0]):
        return tuple(tokens[1:]), tokens[0]
    return tuple(tokens), None


def _heavy_atom_count_for_groups(groups: Sequence[STAMPGroup]) -> int:
    return int(sum(len(group.atom_indices) for group in groups))


def _safe_mol_from_tokens(
    tokens: Sequence[str],
    encoder: STAMPEncoder,
) -> Tuple[Optional[str], bool, Optional[int]]:
    text = " ".join(tokens)
    fallback_count: Optional[int] = None
    try:
        mol, report = encoder.decode_stamp_to_mol(text, return_report=True)
        fallback_count = int(report.total_fallbacks())
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            return None, False, fallback_count
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None, False, fallback_count
    strict_ok = False
    try:
        strict_mol = encoder.decode_stamp_to_mol(text, strict=True)
        strict_ok = strict_mol is not None and strict_mol.GetNumAtoms() > 0
    except Exception:
        strict_ok = False
    return smiles, strict_ok, fallback_count


def _stamp_tree_from_tokens(tokens: Sequence[str], encoder: STAMPEncoder) -> Optional[StampTree]:
    smiles, _strict_ok, _fallbacks = _safe_mol_from_tokens(tokens, encoder)
    if not smiles:
        return None
    try:
        return build_stamp_tree(smiles, encoder)
    except Exception:
        return None


def _port_signature_for_node(tree: StampTree, node: StampNode) -> Optional[Tuple[str, int]]:
    if node.parent_idx is None:
        return None
    n_open = sum(1 for token in node.tokens if _is_joint(token))
    return ("single", int(n_open))


def harvest_stamp_donors(
    trees: Sequence[StampTree],
    *,
    max_span_len: Optional[int] = None,
) -> Dict[Tuple[str, int], List[DonorRecord]]:
    """Collect transplantable subtree bodies keyed by a simple port signature."""

    donors: Dict[Tuple[str, int], List[DonorRecord]] = defaultdict(list)
    for tree in trees:
        for node in tree.nodes:
            if node.parent_idx is None:
                continue
            if max_span_len is not None and (node.token_end - node.token_start) > max_span_len:
                continue
            signature = _port_signature_for_node(tree, node)
            if signature is None:
                continue
            span_tokens = tree.flat_tokens[node.token_start : node.token_end]
            body, _leading = _split_leading_branch(span_tokens)
            if not body:
                continue
            groups = tree.groups[node.group_idx : node.group_idx + node.subtree_group_count]
            donors[signature].append(
                DonorRecord(
                    signature=signature,
                    body_tokens=body,
                    group_count=int(node.subtree_group_count),
                    atom_count=_heavy_atom_count_for_groups(groups),
                    source_smiles=tree.smiles,
                )
            )
    return donors


def splice_stamp_subtree(
    tree: StampTree,
    target: StampNode,
    donor: DonorRecord,
) -> List[str]:
    """Splice donor body and adjust ancestor branch counts by group delta."""

    original = list(tree.flat_tokens)
    target_span = original[target.token_start : target.token_end]
    _body, leading = _split_leading_branch(target_span)
    if leading is not None:
        new_span = [f"[B_{int(donor.group_count)}]"] + list(donor.body_tokens)
    else:
        new_span = list(donor.body_tokens)

    delta_groups = int(donor.group_count) - int(target.subtree_group_count)
    out = original[: target.token_start] + new_span + original[target.token_end :]
    if delta_groups:
        for node in tree.nodes:
            if node.group_idx == target.group_idx:
                continue
            inside = node.group_idx < target.group_idx < node.group_idx + node.subtree_group_count
            if not inside:
                continue
            if node.token_start >= target.token_start:
                continue
            old_token = original[node.token_start]
            if not _is_branch(old_token):
                continue
            new_count = max(1, int(node.subtree_group_count) + delta_groups)
            out[node.token_start] = f"[B_{new_count}]"
    return out


def _random_nonempty_span(length: int, rng: random.Random, max_len: Optional[int] = None) -> Tuple[int, int]:
    if length <= 0:
        return 0, 0
    start = rng.randrange(length)
    limit = length - start
    if max_len is not None:
        limit = min(limit, max(1, int(max_len)))
    span_len = rng.randint(1, max(1, limit))
    return start, start + span_len


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    fp_a = _morgan_fp(smiles_a)
    fp_b = _morgan_fp(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def murcko_retained(parent_smiles: str, child_smiles: str) -> Optional[bool]:
    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return bool(child.HasSubstructMatch(scaffold))


def summarize_operator_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("operator", "unknown"))].append(row)
    summary: List[Dict[str, Any]] = []
    for op in sorted(grouped):
        op_rows = grouped[op]
        valid_rows = [row for row in op_rows if bool(row.get("valid"))]
        unique_smiles = {str(row.get("offspring_smiles")) for row in valid_rows if row.get("offspring_smiles")}
        out = {
            "operator": op,
            "n_trials": len(op_rows),
            "valid_rate": _mean([float(bool(row.get("valid"))) for row in op_rows]),
            "strict_valid_rate": _mean([float(bool(row.get("strict_valid"))) for row in op_rows]),
            "fallback_free_rate": _mean(
                [
                    float(bool(row.get("valid")) and int(row.get("fallback_count") or 0) == 0)
                    for row in op_rows
                ]
            ),
            "unique_valid_rate": float(len(unique_smiles) / max(1, len(valid_rows))),
            "duplicate_valid_rate": 1.0 - float(len(unique_smiles) / max(1, len(valid_rows))),
            "mean_edit_distance": _mean(
                [
                    float(row["edit_distance"])
                    for row in op_rows
                    if isinstance(row.get("edit_distance"), (int, float))
                ]
            ),
            "mean_parent_tanimoto": _mean(
                [
                    float(row["parent_tanimoto"])
                    for row in op_rows
                    if isinstance(row.get("parent_tanimoto"), (int, float))
                ]
            ),
            "core_retention_rate": _mean(
                [
                    float(bool(row["core_retained"]))
                    for row in op_rows
                    if isinstance(row.get("core_retained"), bool)
                ]
            ),
        }
        summary.append(out)
    return summary


def sample_operator_trials(
    trees: Sequence[StampTree],
    *,
    encoder: STAMPEncoder,
    n_trials: int,
    seed: int = 0,
    max_span_len: Optional[int] = None,
    size_delta: int = 4,
    operators: Sequence[str] = ("stamp-subtree", "stamp-random-span", "smiles-substring"),
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    donors_by_sig = harvest_stamp_donors(trees, max_span_len=max_span_len)
    rows: List[Dict[str, Any]] = []
    if not trees:
        return rows
    safe_representation: Optional[BaseRepresentation] = None

    for operator in operators:
        if operator == "safe-fragment-splice" and safe_representation is None:
            safe_representation = build_representation("safe")
        for trial in range(int(n_trials)):
            target_tree = rng.choice(list(trees))
            donor_tree = rng.choice(list(trees))
            row: Dict[str, Any] = {
                "operator": operator,
                "trial": trial,
                "target_smiles": target_tree.smiles,
                "donor_smiles": donor_tree.smiles,
                "valid": False,
                "strict_valid": False,
                "fallback_count": None,
                "offspring_smiles": "",
                "edit_distance": None,
                "parent_tanimoto": None,
                "core_retained": None,
            }

            if operator == "stamp-subtree":
                candidates = [
                    node
                    for node in target_tree.nodes
                    if node.parent_idx is not None
                    and (max_span_len is None or (node.token_end - node.token_start) <= max_span_len)
                ]
                rng.shuffle(candidates)
                offspring_tokens: Optional[List[str]] = None
                target_node: Optional[StampNode] = None
                donor_record: Optional[DonorRecord] = None
                for node in candidates:
                    sig = _port_signature_for_node(target_tree, node)
                    if sig is None:
                        continue
                    groups = target_tree.groups[node.group_idx : node.group_idx + node.subtree_group_count]
                    target_atoms = _heavy_atom_count_for_groups(groups)
                    donor_candidates = [
                        donor
                        for donor in donors_by_sig.get(sig, [])
                        if abs(int(donor.atom_count) - target_atoms) <= int(size_delta)
                        and tuple(donor.body_tokens)
                        != tuple(target_tree.flat_tokens[node.token_start : node.token_end])
                    ]
                    if not donor_candidates:
                        continue
                    donor_record = rng.choice(donor_candidates)
                    target_node = node
                    offspring_tokens = splice_stamp_subtree(target_tree, node, donor_record)
                    break
                if offspring_tokens is None or target_node is None or donor_record is None:
                    row["error"] = "no_compatible_subtree"
                    rows.append(row)
                    continue
                smiles, strict_ok, fallbacks = _safe_mol_from_tokens(offspring_tokens, encoder)
                row.update(
                    {
                        "target_group_idx": target_node.group_idx,
                        "target_span_len": target_node.token_end - target_node.token_start,
                        "donor_span_len": len(donor_record.body_tokens),
                        "target_group_count": target_node.subtree_group_count,
                        "donor_group_count": donor_record.group_count,
                        "group_count_delta": donor_record.group_count - target_node.subtree_group_count,
                        "fallback_count": fallbacks,
                        "strict_valid": strict_ok,
                    }
                )
                if smiles:
                    row["valid"] = True
                    row["offspring_smiles"] = smiles
                    row["edit_distance"] = levenshtein_distance(target_tree.flat_tokens, offspring_tokens)

            elif operator == "stamp-random-span":
                s1, e1 = _random_nonempty_span(len(target_tree.flat_tokens), rng, max_span_len)
                s2, e2 = _random_nonempty_span(len(donor_tree.flat_tokens), rng, max_span_len)
                offspring_tokens = (
                    list(target_tree.flat_tokens[:s1])
                    + list(donor_tree.flat_tokens[s2:e2])
                    + list(target_tree.flat_tokens[e1:])
                )
                smiles, strict_ok, fallbacks = _safe_mol_from_tokens(offspring_tokens, encoder)
                row.update(
                    {
                        "target_span_len": e1 - s1,
                        "donor_span_len": e2 - s2,
                        "fallback_count": fallbacks,
                        "strict_valid": strict_ok,
                    }
                )
                if smiles:
                    row["valid"] = True
                    row["offspring_smiles"] = smiles
                    row["edit_distance"] = levenshtein_distance(target_tree.flat_tokens, offspring_tokens)

            elif operator == "smiles-substring":
                s1, e1 = _random_nonempty_span(len(target_tree.smiles), rng, max_span_len)
                s2, e2 = _random_nonempty_span(len(donor_tree.smiles), rng, max_span_len)
                candidate = target_tree.smiles[:s1] + donor_tree.smiles[s2:e2] + target_tree.smiles[e1:]
                canonical = canonicalize_smiles(candidate)
                row.update(
                    {
                        "target_span_len": e1 - s1,
                        "donor_span_len": e2 - s2,
                        "fallback_count": 0 if canonical else None,
                        "strict_valid": canonical is not None,
                    }
                )
                if canonical:
                    row["valid"] = True
                    row["offspring_smiles"] = canonical
                    row["edit_distance"] = levenshtein_distance(list(target_tree.smiles), list(candidate))
            elif operator == "safe-fragment-splice":
                assert safe_representation is not None
                try:
                    target_fragments = [
                        fragment
                        for fragment in safe_representation.encode(target_tree.smiles).split(".")
                        if fragment
                    ]
                    donor_fragments = [
                        fragment
                        for fragment in safe_representation.encode(donor_tree.smiles).split(".")
                        if fragment
                    ]
                    if not target_fragments or not donor_fragments:
                        raise ValueError("empty_safe_fragment_sequence")
                    s1, e1 = _random_nonempty_span(len(target_fragments), rng, max_span_len)
                    s2, e2 = _random_nonempty_span(len(donor_fragments), rng, max_span_len)
                    offspring_fragments = (
                        target_fragments[:s1] + donor_fragments[s2:e2] + target_fragments[e1:]
                    )
                    candidate = ".".join(offspring_fragments)
                    decoded = safe_representation.decode(candidate)
                    row.update(
                        {
                            "target_span_len": e1 - s1,
                            "donor_span_len": e2 - s2,
                            "fallback_count": 0,
                            "strict_valid": True,
                        }
                    )
                    row["valid"] = True
                    row["offspring_smiles"] = decoded
                    row["edit_distance"] = levenshtein_distance(target_fragments, offspring_fragments)
                except Exception as exc:
                    row["error"] = str(exc)
            else:
                row["error"] = f"unknown_operator:{operator}"

            if row.get("offspring_smiles"):
                row["parent_tanimoto"] = tanimoto(target_tree.smiles, str(row["offspring_smiles"]))
                row["core_retained"] = murcko_retained(target_tree.smiles, str(row["offspring_smiles"]))
            rows.append(row)
    return rows


def load_pair_rows(
    path: Path,
    *,
    smiles_a_col: str,
    smiles_b_col: str,
    edit_class_col: str = "edit_class",
    pair_id_col: str = "pair_id",
    max_pairs: int = 0,
) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader):
            smiles_a = (row.get(smiles_a_col) or "").strip()
            smiles_b = (row.get(smiles_b_col) or "").strip()
            if not smiles_a or not smiles_b:
                continue
            rows.append(
                {
                    "pair_id": (row.get(pair_id_col) or str(idx)).strip(),
                    "edit_class": (row.get(edit_class_col) or "all").strip() or "all",
                    "smiles_a": smiles_a,
                    "smiles_b": smiles_b,
                }
            )
            if max_pairs > 0 and len(rows) >= max_pairs:
                break
    return rows


def load_smiles_rows(
    path: Path,
    *,
    smiles_col: str = "smiles",
    max_mols: int = 0,
) -> List[str]:
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return []
    smiles: List[str] = []
    if "," in text[0] or "\t" in text[0]:
        sample = "\n".join(text[:20])
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, dialect=dialect)
            for row in reader:
                value = (row.get(smiles_col) or "").strip()
                if value:
                    smiles.append(value)
                if max_mols > 0 and len(smiles) >= max_mols:
                    break
    else:
        for line in text:
            value = line.strip().split()[0] if line.strip() else ""
            if value and not value.lower().startswith("smiles"):
                smiles.append(value)
            if max_mols > 0 and len(smiles) >= max_mols:
                break
    return smiles

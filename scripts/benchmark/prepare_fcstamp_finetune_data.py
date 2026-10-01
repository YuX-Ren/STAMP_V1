#!/usr/bin/env python3
"""Sample large-scale FC-STAMP finetuning rows from STAMP parquet files.

By default this builds in-place fill-in-the-middle rows:

    [TASK] left [INFILL] right [GEN] middle [EOS]

The generated middle is spliced back into the fixed context before decoding.
``--construction completion`` remains available for debugging only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASK_TOKEN = {
    "linker_design": "[TASK_LINKER]",
    "motif_extension": "[TASK_EXTEND]",
    "scaffold_decoration": "[TASK_DECORATE]",
    "superstructure_generation": "[TASK_DECORATE]",
}

SINGLE_INFILL = {
    "linker_design": "[INFILL_PATH]",
    "motif_extension": "[INFILL_TAIL]",
    "scaffold_decoration": "[INFILL_SUBTREE]",
    "superstructure_generation": "[INFILL_SUBTREE]",
}

FRAG_SEP = "[FRAG_SEP]"

FIELD_NAMES = [
    "source_smiles",
    "source_stamp",
    "task",
    "task_token",
    "query_smiles",
    "query_stamp",
    "split_random",
    "all_pass",
    "stamp_smiles",
    "prompt_stamp",
    "target_stamp",
    "prompt_len",
    "target_len",
    "label_start",
    "loss_mask_spec",
    "full_stamp",
    "masked_stamp",
    "left_stamp",
    "middle_stamp",
    "right_stamp",
    "n_groups",
    "n_keep_groups",
    "n_mask_groups",
    "n_holes",
    "hole_spans",
    "keep_group_indices",
    "mask_group_indices",
    "construction_mode",
    "decode_mode",
]

SCHEMA = pa.schema(
    [
        pa.field("source_smiles", pa.string()),
        pa.field("source_stamp", pa.string()),
        pa.field("task", pa.string()),
        pa.field("task_token", pa.string()),
        pa.field("query_smiles", pa.string()),
        pa.field("query_stamp", pa.string()),
        pa.field("split_random", pa.string()),
        pa.field("all_pass", pa.bool_()),
        pa.field("stamp_smiles", pa.string()),
        pa.field("prompt_stamp", pa.string()),
        pa.field("target_stamp", pa.string()),
        pa.field("prompt_len", pa.int32()),
        pa.field("target_len", pa.int32()),
        pa.field("label_start", pa.int32()),
        pa.field("loss_mask_spec", pa.string()),
        pa.field("full_stamp", pa.string()),
        pa.field("masked_stamp", pa.string()),
        pa.field("left_stamp", pa.string()),
        pa.field("middle_stamp", pa.string()),
        pa.field("right_stamp", pa.string()),
        pa.field("n_groups", pa.int32()),
        pa.field("n_keep_groups", pa.int32()),
        pa.field("n_mask_groups", pa.int32()),
        pa.field("n_holes", pa.int32()),
        pa.field("hole_spans", pa.string()),
        pa.field("keep_group_indices", pa.string()),
        pa.field("mask_group_indices", pa.string()),
        pa.field("construction_mode", pa.string()),
        pa.field("decode_mode", pa.string()),
    ]
)


def _resolve_paths(specs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        expanded = sorted(glob.glob(str(spec)))
        if not expanded and os.path.exists(str(spec)):
            expanded = [str(spec)]
        for path in expanded:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                out.append(resolved)
                seen.add(resolved)
    return out


def _detok(tokens: Sequence[str]) -> str:
    return " ".join(str(tok) for tok in tokens if str(tok))


def _is_stereo_token(tok: str) -> bool:
    return str(tok).startswith("[S_") and str(tok).endswith("]")


def _strip_stereo_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tok) for tok in tokens if not _is_stereo_token(str(tok))]


def _stamp_fragment_groups(tokens: Sequence[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    pending_prefix: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "[END]":
            break
        if tok.startswith("[B_"):
            pending_prefix.append(tok)
            i += 1
            continue
        if tok.startswith("[J_") or tok.startswith("[S_") or tok in {"[BACK]", "[Ring]"}:
            if groups:
                groups[-1].append(tok)
            else:
                pending_prefix.append(tok)
            i += 1
            continue
        group = list(pending_prefix)
        pending_prefix.clear()
        group.append(tok)
        i += 1
        while i < n and (tokens[i].startswith("[J_") or tokens[i].startswith("[S_") or tokens[i] == "[Ring]"):
            group.append(tokens[i])
            i += 1
        groups.append(group)
    if pending_prefix and groups:
        groups[-1].extend(pending_prefix)
    return groups


def _flatten_groups(groups: Sequence[Sequence[str]], indices: Optional[Sequence[int]] = None) -> list[str]:
    selected = range(len(groups)) if indices is None else indices
    out: list[str] = []
    for idx in selected:
        out.extend(groups[int(idx)])
    return out


def _condition_tokens_from_groups(
    groups: Sequence[Sequence[str]],
    keep_indices: Sequence[int],
) -> list[str]:
    out: list[str] = []
    for run in _contiguous_runs(keep_indices):
        toks = _flatten_groups(groups, run)
        if not toks:
            continue
        if out:
            out.append(FRAG_SEP)
        out.extend(toks)
    return out


def _branch_group_count(group: Sequence[str]) -> Optional[int]:
    if not group:
        return None
    tok = str(group[0])
    match = re.match(r"^\[B_(\d+)\]$", tok)
    if not match:
        return None
    try:
        return max(1, int(match.group(1)))
    except ValueError:
        return None


def _is_branch_token(tok: str) -> bool:
    return str(tok).startswith("[B_") and str(tok).endswith("]")


def _contains_branch_token(tokens: Sequence[str]) -> bool:
    return any(_is_branch_token(str(tok)) for tok in tokens)


def _joint_specs(group: Sequence[str]) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    i = 0
    vals = [str(tok) for tok in group]
    while i < len(vals):
        if not vals[i].startswith("[J_"):
            i += 1
            continue
        start = i
        i += 1
        if i < len(vals) and vals[i].startswith("[S_"):
            i += 1
        if i < len(vals) and vals[i] == "[Ring]":
            i += 1
        specs.append((start, i))
    return specs


def _move_joint_specs_to_end(group: Sequence[str], spec_indices: Sequence[int]) -> list[str]:
    vals = [str(tok) for tok in group]
    specs = _joint_specs(vals)
    selected = [int(idx) for idx in spec_indices]
    if not selected:
        return vals
    if any(idx < 0 or idx >= len(specs) for idx in selected):
        return []

    selected_set = set(selected)
    skip_positions: set[int] = set()
    moved: list[str] = []
    for spec_idx in selected:
        start, end = specs[spec_idx]
        skip_positions.update(range(start, end))
        moved.extend(vals[start:end])
    return [tok for pos, tok in enumerate(vals) if pos not in skip_positions] + moved


def _parse_stamp_tree(groups: Sequence[Sequence[str]]) -> Optional[dict[int, dict[str, Any]]]:
    n = len(groups)
    if n == 0 or _is_branch_token(groups[0][0] if groups[0] else ""):
        return None
    nodes: dict[int, dict[str, Any]] = {}

    def parse_node(
        start: int,
        end: int,
        *,
        parent: Optional[int],
        edge_kind: str,
        edge_order: int,
    ) -> bool:
        if start < 0 or start >= end or end > n:
            return False
        node = {
            "start": int(start),
            "end": int(end),
            "parent": None if parent is None else int(parent),
            "edge_kind": str(edge_kind),
            "edge_order": int(edge_order),
            "children": [],
        }
        nodes[int(start)] = node
        cursor = int(start) + 1
        branch_order = 0
        while cursor < end:
            branch_n = _branch_group_count(groups[cursor])
            if branch_n is None:
                break
            child_end = cursor + int(branch_n)
            if child_end > end:
                return False
            node["children"].append(int(cursor))
            if not parse_node(
                cursor,
                child_end,
                parent=int(start),
                edge_kind="branch",
                edge_order=int(branch_order),
            ):
                return False
            cursor = child_end
            branch_order += 1
        if cursor < end:
            node["children"].append(int(cursor))
            if not parse_node(
                cursor,
                end,
                parent=int(start),
                edge_kind="main",
                edge_order=int(branch_order),
            ):
                return False
        return True

    if not parse_node(0, n, parent=None, edge_kind="root", edge_order=0):
        return None
    return nodes


def _node_middle_tokens(groups: Sequence[Sequence[str]], run: Sequence[int]) -> list[str]:
    tokens = _flatten_groups(groups, run)
    if tokens and _is_branch_token(tokens[0]):
        return tokens[1:]
    return tokens


def _tree_decorate_candidates(
    groups: Sequence[Sequence[str]],
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    min_keep_fraction: float,
    max_keep_fraction: float,
) -> list[list[int]]:
    tree = _parse_stamp_tree(groups)
    if tree is None:
        return []
    n = len(groups)
    out: list[list[int]] = []
    for start, node in sorted(tree.items()):
        if int(start) == 0:
            continue
        end = int(node["end"])
        k = end - int(start)
        if k < int(min_mask_groups) or k > int(max_mask_groups):
            continue
        keep_fraction = (n - k) / max(1, n)
        if keep_fraction < float(min_keep_fraction) or keep_fraction > float(max_keep_fraction):
            continue
        run = list(range(int(start), end))
        middle = _node_middle_tokens(groups, run)
        if not middle or _contains_branch_token(middle):
            continue
        out.append(run)
    return out


def _runs_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return bool({int(x) for x in a} & {int(x) for x in b})


def _motif_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tok) for tok in tokens if not (str(tok).startswith("[") and str(tok).endswith("]"))]


def _group_motif_signature(group: Sequence[str]) -> str:
    motifs = _motif_tokens(group)
    return _detok(motifs) if motifs else _detok(group)


def _linker_boundary_signature(
    groups: Sequence[Sequence[str]],
    run: Sequence[int],
    *,
    mode: str = "motif",
) -> str:
    if not run:
        return ""
    start = int(run[0])
    end = int(run[-1]) + 1
    if start <= 0 or end >= len(groups):
        return ""
    if mode == "group":
        left = _detok(groups[start - 1])
        right = _detok(groups[end])
    elif mode == "motif":
        left = _group_motif_signature(groups[start - 1])
        right = _group_motif_signature(groups[end])
    else:
        raise ValueError(f"Unsupported linker boundary signature mode: {mode}")
    return f"{left} {FRAG_SEP} {right}"


def _parse_count_spec(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not str(spec).strip():
        return out
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            key, val = item.split(":", 1)
        elif "=" in item:
            key, val = item.split("=", 1)
        else:
            raise ValueError(f"Invalid count spec item: {item}")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid count spec key: {item}")
        out[key] = int(val)
    return out


def _parse_int_count_spec(spec: str) -> dict[int, int]:
    return {int(key): int(val) for key, val in _parse_count_spec(spec).items()}


def _contiguous_runs(indices: Sequence[int]) -> list[list[int]]:
    vals = sorted({int(x) for x in indices})
    if not vals:
        return []
    runs: list[list[int]] = []
    cur = [vals[0]]
    for value in vals[1:]:
        if value == cur[-1] + 1:
            cur.append(value)
        else:
            runs.append(cur)
            cur = [value]
    runs.append(cur)
    return runs


def _middle_tokens_for_task(
    groups: Sequence[Sequence[str]],
    run: Sequence[int],
    *,
    task: str,
) -> tuple[list[str], list[str]]:
    """Return fixed tokens before the infill and the generated middle tokens."""
    if not run:
        return [], []
    first_group = list(groups[int(run[0])])
    fixed_before: list[str] = []
    if task == "scaffold_decoration" and first_group and _is_branch_token(first_group[0]):
        fixed_before = [str(first_group[0])]
        middle = [str(tok) for tok in first_group[1:]]
        for idx in list(run)[1:]:
            middle.extend(str(tok) for tok in groups[int(idx)])
        return fixed_before, middle
    return fixed_before, _flatten_groups(groups, run)


def _build_prompt_target(
    groups: Sequence[Sequence[str]],
    runs: Sequence[Sequence[int]],
    *,
    task: str,
) -> dict[str, Any]:
    run_by_start = {int(run[0]): (hole_idx, list(run)) for hole_idx, run in enumerate(runs)}
    n_holes = len(runs)
    prompt_body: list[str] = []
    holes: list[dict[str, Any]] = []
    i = 0
    while i < len(groups):
        item = run_by_start.get(i)
        if item is None:
            prompt_body.extend(groups[i])
            i += 1
            continue
        hole_idx, run = item
        infill = SINGLE_INFILL[task] if n_holes == 1 else f"[INFILL_{hole_idx}]"
        fixed_before, middle = _middle_tokens_for_task(groups, run, task=task)
        prompt_body.extend(fixed_before)
        prompt_body.append(infill)
        holes.append(
            {
                "hole_index": int(hole_idx),
                "infill_token": infill,
                "start_group": int(run[0]),
                "end_group": int(run[-1] + 1),
                "n_groups": int(len(run)),
                "middle_stamp": _detok(middle),
                "middle_len": int(len(middle)),
            }
        )
        i = int(run[-1]) + 1

    prompt_tokens = [TASK_TOKEN[task]] + prompt_body + ["[GEN]"]
    if n_holes == 1:
        _, middle = _middle_tokens_for_task(groups, runs[0], task=task)
        target_tokens = (middle if middle else ["[EMPTY]"]) + ["[EOS]"]
    else:
        target_tokens = []
        for hole in holes:
            target_tokens.append(f"[FILL_{hole['hole_index']}]")
            middle = str(hole["middle_stamp"]).split()
            target_tokens.extend(middle if middle else ["[EMPTY]"])
            target_tokens.append("[/FILL]")
        target_tokens.append("[EOS]")
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_body_tokens": prompt_body,
        "target_tokens": target_tokens,
        "holes": holes,
    }


def _sample_linker_runs(
    groups: Sequence[Sequence[str]],
    rng: random.Random,
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    min_target_tokens: int = 0,
    length_strategy: str = "candidate",
) -> Optional[list[list[int]]]:
    candidates_by_k = _linker_candidates_by_k(
        groups,
        min_mask_groups=min_mask_groups,
        max_mask_groups=max_mask_groups,
        min_target_tokens=min_target_tokens,
    )
    if not candidates_by_k:
        return None
    if length_strategy == "uniform_length":
        k = rng.choice(sorted(candidates_by_k))
        return [rng.choice(candidates_by_k[k])]
    candidates = [run for vals in candidates_by_k.values() for run in vals]
    return [rng.choice(candidates)] if candidates else None


def _linker_candidates_by_k(
    groups: Sequence[Sequence[str]],
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    min_target_tokens: int = 0,
) -> dict[int, list[list[int]]]:
    n = len(groups)
    if n < min_mask_groups + 2:
        return {}
    candidates_by_k: dict[int, list[list[int]]] = {}
    max_k = min(max_mask_groups, n - 2)
    for k in range(min_mask_groups, max_k + 1):
        for start in range(1, n - k):
            run = list(range(start, start + k))
            middle = _flatten_groups(groups, run)
            if _contains_branch_token(middle):
                continue
            if min_target_tokens > 0 and len(middle) < int(min_target_tokens):
                continue
            candidates_by_k.setdefault(k, []).append(run)
    return candidates_by_k


def _sample_extend_runs(
    n: int,
    rng: random.Random,
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    length_strategy: str = "candidate",
) -> Optional[list[list[int]]]:
    if n < min_mask_groups + 1:
        return None
    max_k = min(max_mask_groups, n - 1)
    if length_strategy == "uniform_length":
        k = rng.choice(list(range(min_mask_groups, max_k + 1)))
    else:
        k = rng.randint(min_mask_groups, max_k)
    return [list(range(n - k, n))]


def _sample_decorate_runs(
    groups: Sequence[Sequence[str]],
    rng: random.Random,
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    min_keep_fraction: float,
    max_keep_fraction: float,
    preferred_holes: Optional[Sequence[int]] = None,
    allow_linker_fallback: bool = True,
) -> Optional[list[list[int]]]:
    candidates = _tree_decorate_candidates(
        groups,
        min_mask_groups=min_mask_groups,
        max_mask_groups=max_mask_groups,
        min_keep_fraction=min_keep_fraction,
        max_keep_fraction=max_keep_fraction,
    )
    if candidates:
        max_holes = max(1, min(8, len(candidates)))
        possible = list(range(1, max_holes + 1))
        preferred = [int(h) for h in (preferred_holes or []) if 1 <= int(h) <= max_holes]
        hole_counts = preferred or possible
        rng.shuffle(hole_counts)
        for n_holes in hole_counts:
            shuffled = [list(run) for run in candidates]
            rng.shuffle(shuffled)
            selected: list[list[int]] = []
            used: set[int] = set()
            for run in shuffled:
                run_set = {int(idx) for idx in run}
                if run_set & used:
                    continue
                if any(_runs_overlap(run, prev) for prev in selected):
                    continue
                selected.append(run)
                used.update(run_set)
                if len(selected) >= int(n_holes):
                    return sorted(selected, key=lambda run: run[0])
    if not allow_linker_fallback:
        return None
    return _sample_linker_runs(groups, rng, min_mask_groups=min_mask_groups, max_mask_groups=max_mask_groups)


def _runs_mask_root(runs: Sequence[Sequence[int]]) -> bool:
    return any(0 in {int(idx) for idx in run} for run in runs)


def _build_scaffold_prompt_target(
    groups: Sequence[Sequence[str]],
    runs: Sequence[Sequence[int]],
    *,
    task: str,
) -> Optional[dict[str, Any]]:
    tree = _parse_stamp_tree(groups)
    if tree is None:
        return None

    sorted_runs = sorted([list(run) for run in runs], key=lambda run: int(run[0]) if run else -1)
    selected_by_start: dict[int, dict[str, Any]] = {}
    external_by_parent: dict[int, list[dict[str, Any]]] = {}
    holes: list[dict[str, Any]] = []
    n_holes = len(sorted_runs)

    for hole_idx, run in enumerate(sorted_runs):
        if not run:
            return None
        start = int(run[0])
        node = tree.get(start)
        if node is None or node.get("parent") is None:
            return None
        expected = list(range(start, int(node["end"])))
        if [int(x) for x in run] != expected:
            return None
        middle = _node_middle_tokens(groups, run)
        if not middle or _contains_branch_token(middle):
            return None

        parent = int(node["parent"])
        input_specs = 0 if parent == 0 else 1
        spec_idx = int(input_specs + int(node["edge_order"]))
        if spec_idx >= len(_joint_specs(groups[parent])):
            return None

        infill = SINGLE_INFILL[task] if n_holes == 1 else f"[INFILL_{hole_idx}]"
        hole = {
            "hole_index": int(hole_idx),
            "infill_token": infill,
            "start_group": int(start),
            "end_group": int(node["end"]),
            "n_groups": int(len(run)),
            "middle_stamp": _detok(middle),
            "middle_len": int(len(middle)),
            "attach_group": int(parent),
            "attach_joint_spec": int(spec_idx),
        }
        holes.append(hole)
        selected_by_start[start] = hole
        external_by_parent.setdefault(parent, []).append(
            {
                "spec_idx": int(spec_idx),
                "hole_index": int(hole_idx),
                "infill": infill,
            }
        )

    def fixed_size(start: int) -> int:
        if start in selected_by_start:
            return 0
        node = tree[int(start)]
        return 1 + sum(fixed_size(int(child)) for child in node["children"])

    prompt_body: list[str] = []

    def emit(start: int) -> bool:
        if start in selected_by_start:
            return True
        node = tree[int(start)]
        external = sorted(
            external_by_parent.get(int(start), []),
            key=lambda item: (int(item["spec_idx"]), int(item["hole_index"])),
        )
        group = _move_joint_specs_to_end(groups[int(start)], [int(item["spec_idx"]) for item in external])
        if not group:
            return False
        if group and _is_branch_token(group[0]):
            group = [f"[B_{fixed_size(int(start))}]", *group[1:]]
        prompt_body.extend(group)
        for child in node["children"]:
            if not emit(int(child)):
                return False
        for item in external:
            prompt_body.append(str(item["infill"]))
        return True

    if not emit(0):
        return None

    prompt_tokens = [TASK_TOKEN[task]] + prompt_body + ["[GEN]"]
    if n_holes == 1:
        middle = str(holes[0]["middle_stamp"]).split()
        target_tokens = (middle if middle else ["[EMPTY]"]) + ["[EOS]"]
    else:
        target_tokens = []
        for hole in holes:
            target_tokens.append(f"[FILL_{hole['hole_index']}]")
            middle = str(hole["middle_stamp"]).split()
            target_tokens.extend(middle if middle else ["[EMPTY]"])
            target_tokens.append("[/FILL]")
        target_tokens.append("[EOS]")

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_body_tokens": prompt_body,
        "target_tokens": target_tokens,
        "holes": holes,
    }


def _sample_runs_for_task(
    groups: Sequence[Sequence[str]],
    task: str,
    rng: random.Random,
    *,
    min_mask_groups: int,
    max_mask_groups: int,
    min_keep_fraction: float,
    max_keep_fraction: float,
    linker_min_target_tokens: int = 0,
    length_strategy: str = "candidate",
    preferred_scaffold_holes: Optional[Sequence[int]] = None,
    scaffold_allow_linker_fallback: bool = True,
) -> Optional[list[list[int]]]:
    n = len(groups)
    if task == "linker_design":
        return _sample_linker_runs(
            groups,
            rng,
            min_mask_groups=min_mask_groups,
            max_mask_groups=max_mask_groups,
            min_target_tokens=linker_min_target_tokens,
            length_strategy=length_strategy,
        )
    if task == "motif_extension":
        return _sample_extend_runs(
            n,
            rng,
            min_mask_groups=min_mask_groups,
            max_mask_groups=max_mask_groups,
            length_strategy=length_strategy,
        )
    if task == "scaffold_decoration":
        return _sample_decorate_runs(
            groups,
            rng,
            min_mask_groups=min_mask_groups,
            max_mask_groups=max_mask_groups,
            min_keep_fraction=min_keep_fraction,
            max_keep_fraction=max_keep_fraction,
            preferred_holes=preferred_scaffold_holes,
            allow_linker_fallback=scaffold_allow_linker_fallback,
        )
    if task == "superstructure_generation":
        return _sample_decorate_runs(
            groups,
            rng,
            min_mask_groups=min_mask_groups,
            max_mask_groups=max_mask_groups,
            min_keep_fraction=min_keep_fraction,
            max_keep_fraction=max_keep_fraction,
            preferred_holes=preferred_scaffold_holes,
            allow_linker_fallback=scaffold_allow_linker_fallback,
        )
    raise ValueError(f"Unsupported task: {task}")


def _build_row(
    *,
    source_smiles: str,
    source_stamp: str,
    groups: Sequence[Sequence[str]],
    task: str,
    runs: Sequence[Sequence[int]],
    split: str,
    all_pass: bool,
    max_total_tokens: int,
) -> Optional[dict[str, Any]]:
    mask_indices = sorted({idx for run in runs for idx in run})
    if not mask_indices:
        return None
    if _runs_mask_root(runs):
        return None
    keep_indices = [idx for idx in range(len(groups)) if idx not in set(mask_indices)]
    if not keep_indices:
        return None
    if task == "linker_design" and _contains_branch_token(_flatten_groups(groups, mask_indices)):
        return None
    if task in {"scaffold_decoration", "superstructure_generation"}:
        built = _build_scaffold_prompt_target(groups, runs, task=task)
        if built is None:
            return None
    else:
        built = _build_prompt_target(groups, runs, task=task)
    prompt_tokens = built["prompt_tokens"]
    target_tokens = built["target_tokens"]
    train_tokens = list(prompt_tokens) + list(target_tokens)
    if max_total_tokens > 0 and len(train_tokens) > max_total_tokens:
        return None

    single = built["holes"][0] if len(built["holes"]) == 1 else None
    left_tokens: list[str] = []
    middle_tokens: list[str] = []
    right_tokens: list[str] = []
    if single is not None:
        start = int(single["start_group"])
        end = int(single["end_group"])
        left_tokens = _flatten_groups(groups, range(0, start))
        if task in {"scaffold_decoration", "superstructure_generation"}:
            middle_tokens = _node_middle_tokens(groups, range(start, end))
        else:
            _, middle_tokens = _middle_tokens_for_task(groups, range(start, end), task=task)
        right_tokens = _flatten_groups(groups, range(end, len(groups)))

    query_stamp = _detok(_flatten_groups(groups, keep_indices))
    return {
        "source_smiles": str(source_smiles or ""),
        "source_stamp": str(source_stamp),
        "task": task,
        "task_token": TASK_TOKEN[task],
        "query_smiles": "",
        "query_stamp": query_stamp,
        "split_random": split,
        "all_pass": bool(all_pass),
        "stamp_smiles": _detok(train_tokens),
        "prompt_stamp": _detok(prompt_tokens),
        "target_stamp": _detok(target_tokens),
        "prompt_len": int(len(prompt_tokens)),
        "target_len": int(len(target_tokens)),
        "label_start": int(len(prompt_tokens)),
        "loss_mask_spec": f"prompt0:{len(prompt_tokens)},target1:{len(target_tokens)}",
        "full_stamp": _detok(_flatten_groups(groups)),
        "masked_stamp": _detok(built["prompt_body_tokens"]),
        "left_stamp": _detok(left_tokens),
        "middle_stamp": _detok(middle_tokens),
        "right_stamp": _detok(right_tokens),
        "n_groups": int(len(groups)),
        "n_keep_groups": int(len(keep_indices)),
        "n_mask_groups": int(len(mask_indices)),
        "n_holes": int(len(built["holes"])),
        "hole_spans": json.dumps(built["holes"], ensure_ascii=False),
        "keep_group_indices": json.dumps(keep_indices),
        "mask_group_indices": json.dumps(mask_indices),
        "construction_mode": "splice",
        "decode_mode": "splice",
    }


def _build_completion_row(
    *,
    source_smiles: str,
    source_stamp: str,
    groups: Sequence[Sequence[str]],
    task: str,
    runs: Sequence[Sequence[int]],
    split: str,
    all_pass: bool,
    max_total_tokens: int,
) -> Optional[dict[str, Any]]:
    mask_indices = sorted({idx for run in runs for idx in run})
    keep_indices = [idx for idx in range(len(groups)) if idx not in set(mask_indices)]
    if not keep_indices:
        return None

    query_tokens = _condition_tokens_from_groups(groups, keep_indices)
    if not query_tokens:
        return None
    full_tokens = _flatten_groups(groups)
    prompt_tokens = [TASK_TOKEN[task]] + query_tokens + ["[GEN]"]
    target_tokens = list(full_tokens) + ["[EOS]"]
    train_tokens = list(prompt_tokens) + list(target_tokens)
    if max_total_tokens > 0 and len(train_tokens) > max_total_tokens:
        return None

    query_stamp = _detok(query_tokens)
    return {
        "source_smiles": str(source_smiles or ""),
        "source_stamp": str(source_stamp),
        "task": task,
        "task_token": TASK_TOKEN[task],
        "query_smiles": "",
        "query_stamp": query_stamp,
        "split_random": split,
        "all_pass": bool(all_pass),
        "stamp_smiles": _detok(train_tokens),
        "prompt_stamp": _detok(prompt_tokens),
        "target_stamp": _detok(target_tokens),
        "prompt_len": int(len(prompt_tokens)),
        "target_len": int(len(target_tokens)),
        "label_start": int(len(prompt_tokens)),
        "loss_mask_spec": f"prompt0:{len(prompt_tokens)},target1:{len(target_tokens)}",
        "full_stamp": _detok(full_tokens),
        "masked_stamp": query_stamp,
        "left_stamp": query_stamp,
        "middle_stamp": _detok(full_tokens),
        "right_stamp": "",
        "n_groups": int(len(groups)),
        "n_keep_groups": int(len(keep_indices)),
        "n_mask_groups": int(0),
        "n_holes": int(0),
        "hole_spans": "[]",
        "keep_group_indices": json.dumps(keep_indices),
        "mask_group_indices": "[]",
        "construction_mode": "completion",
        "decode_mode": "generated_full",
    }


def _choose_split(
    rng: random.Random,
    *,
    valid_fraction: float,
    existing_split: Any,
    use_existing_split: bool,
) -> str:
    if use_existing_split and existing_split is not None and str(existing_split).strip():
        raw = str(existing_split).strip()
        if raw in {"valid", "validation"}:
            return "val"
        return raw
    if valid_fraction > 0.0 and rng.random() < valid_fraction:
        return "val"
    return "train"


def _length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(int(x) for x in values)

    def pct(q: float) -> int:
        idx = int(math.ceil((len(vals) - 1) * q))
        return vals[min(len(vals) - 1, max(0, idx))]

    return {
        "count": len(vals),
        "min": vals[0],
        "mean": sum(vals) / len(vals),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": vals[-1],
    }


def _task_quotas(tasks: Sequence[str], total: int) -> dict[str, int]:
    vals = list(tasks)
    if not vals:
        return {}
    base = int(total) // len(vals)
    rem = int(total) % len(vals)
    return {task: base + (1 if idx < rem else 0) for idx, task in enumerate(vals)}


def _top_counter(counter: Counter[Any], n: int = 20) -> list[dict[str, Any]]:
    return [{"key": str(key), "count": int(value)} for key, value in counter.most_common(n)]


def _prescan_linker_boundary_alternatives(
    *,
    paths: Sequence[str],
    text_column: str,
    batch_size: int,
    min_mask_groups: int,
    max_mask_groups: int,
    min_target_tokens: int,
    min_alternatives: int,
    boundary_signature: str,
    max_rows: int,
) -> tuple[set[str], dict[str, Any]]:
    """Return boundary signatures that have at least N distinct linker middles."""
    eligible: set[str] = set()
    pending: dict[str, set[str]] = {}
    scanned_rows = 0
    candidate_spans = 0
    skipped: Counter[str] = Counter()
    t0 = time.time()

    for path in paths:
        if max_rows > 0 and scanned_rows >= max_rows:
            break
        pf = pq.ParquetFile(path)
        names = set(pf.schema.names)
        text_col = text_column if text_column in names else ("stamp" if "stamp" in names else "")
        if not text_col:
            skipped["missing_text_column"] += 1
            continue
        for batch in pf.iter_batches(batch_size=int(batch_size), columns=[text_col]):
            stamps = batch.column(0).to_pylist()
            for stamp in stamps:
                if max_rows > 0 and scanned_rows >= max_rows:
                    break
                scanned_rows += 1
                if stamp is None:
                    skipped["empty_stamp"] += 1
                    continue
                tokens = str(stamp).strip().split()
                if not tokens:
                    skipped["empty_stamp"] += 1
                    continue
                groups = _stamp_fragment_groups(tokens)
                if len(groups) < 2:
                    skipped["too_few_groups"] += 1
                    continue
                candidates_by_k = _linker_candidates_by_k(
                    groups,
                    min_mask_groups=min_mask_groups,
                    max_mask_groups=max_mask_groups,
                    min_target_tokens=min_target_tokens,
                )
                for runs in candidates_by_k.values():
                    for run in runs:
                        boundary = _linker_boundary_signature(groups, run, mode=boundary_signature)
                        if not boundary or boundary in eligible:
                            continue
                        middle = _detok(_flatten_groups(groups, run))
                        values = pending.setdefault(boundary, set())
                        values.add(middle)
                        candidate_spans += 1
                        if len(values) >= int(min_alternatives):
                            eligible.add(boundary)
                            pending.pop(boundary, None)

    summary = {
        "enabled": True,
        "min_alternatives": int(min_alternatives),
        "boundary_signature": str(boundary_signature),
        "max_rows": int(max_rows),
        "scanned_rows": int(scanned_rows),
        "candidate_spans": int(candidate_spans),
        "eligible_boundaries": int(len(eligible)),
        "pending_boundaries": int(len(pending)),
        "skipped": dict(skipped),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    return eligible, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample FC-STAMP finetune data from STAMP parquet shards.")
    parser.add_argument("--parquet", nargs="+", required=True, help="Input parquet paths/globs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-column", default="stamp_smiles", help="STAMP text column; fallback to 'stamp' if absent.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--split-column", default="split_random")
    parser.add_argument("--all-pass-column", default="all_pass")
    parser.add_argument("--use-existing-split", action="store_true")
    parser.add_argument("--valid-fraction", type=float, default=0.02)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_TOKEN), choices=list(TASK_TOKEN))
    parser.add_argument("--balance-tasks", action="store_true", help="Write an equal quota for each requested task.")
    parser.add_argument(
        "--task-quotas",
        default="",
        help="Explicit output quotas, e.g. linker_design:250000,motif_extension:250000,scaffold_decoration:250000,superstructure_generation:250000.",
    )
    parser.add_argument(
        "--scaffold-hole-quotas",
        default="",
        help="Optional scaffold_decoration quotas by hole count, e.g. 1:80000,2:85000,3:85000.",
    )
    parser.add_argument(
        "--construction",
        choices=("completion", "splice"),
        default="splice",
        help="splice = in-place FIM; completion = debugging full-generation rows.",
    )
    parser.add_argument("--tasks-per-mol", type=int, default=1)
    parser.add_argument("--variants-per-task", type=int, default=1)
    parser.add_argument("--max-output-rows", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--write-batch-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-files", action="store_true", default=True)
    parser.add_argument("--no-shuffle-files", dest="shuffle_files", action="store_false")
    parser.add_argument("--min-mask-groups", type=int, default=1)
    parser.add_argument("--max-mask-groups", type=int, default=4)
    parser.add_argument("--linker-min-target-tokens", type=int, default=0)
    parser.add_argument(
        "--linker-middle-frequency-cap",
        type=int,
        default=0,
        help="Optional max rows per exact linker middle_stamp. 0 disables the cap.",
    )
    parser.add_argument(
        "--linker-target-motif-tuple-cap",
        type=int,
        default=0,
        help="Optional max rows per linker target motif tuple. 0 disables the cap.",
    )
    parser.add_argument(
        "--linker-target-motif-token-cap",
        type=int,
        default=0,
        help="Optional max occurrences per linker target motif token. 0 disables the cap.",
    )
    parser.add_argument(
        "--linker-boundary-frequency-cap",
        type=int,
        default=0,
        help="Optional max rows per linker A/B boundary signature. 0 disables the cap.",
    )
    parser.add_argument(
        "--linker-min-boundary-alternatives",
        type=int,
        default=1,
        help="Require linker A/B boundary signatures to have at least this many distinct candidate middles.",
    )
    parser.add_argument(
        "--linker-boundary-signature",
        choices=("motif", "group"),
        default="motif",
        help="Boundary key used for linker alternative collection and boundary frequency caps.",
    )
    parser.add_argument(
        "--linker-boundary-prescan-rows",
        type=int,
        default=0,
        help="Max input rows for linker boundary alternative prescan. 0 scans all matched inputs.",
    )
    parser.add_argument("--length-strategy", choices=("candidate", "uniform_length"), default="candidate")
    parser.add_argument("--scaffold-linker-fallback", dest="scaffold_linker_fallback", action="store_true")
    parser.add_argument("--no-scaffold-linker-fallback", dest="scaffold_linker_fallback", action="store_false")
    parser.set_defaults(scaffold_linker_fallback=False)
    parser.add_argument("--min-keep-fraction", type=float, default=0.4)
    parser.add_argument("--max-keep-fraction", type=float, default=1.0)
    parser.add_argument("--max-total-tokens", type=int, default=256)
    parser.add_argument(
        "--strip-stereo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop [S_*] stereo tokens when constructing FC-STAMP rows.",
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    paths = _resolve_paths(args.parquet)
    if not paths:
        raise FileNotFoundError(f"No parquet paths matched: {args.parquet}")
    if args.shuffle_files:
        rng.shuffle(paths)
    task_quota = _parse_count_spec(args.task_quotas)
    if not task_quota and args.balance_tasks:
        task_quota = _task_quotas(args.tasks, int(args.max_output_rows))
    scaffold_hole_quota = _parse_int_count_spec(args.scaffold_hole_quotas)
    if scaffold_hole_quota:
        task_quota["scaffold_decoration"] = sum(scaffold_hole_quota.values())
    task_quota = {task: int(quota) for task, quota in task_quota.items() if task in TASK_TOKEN and int(quota) > 0}

    eligible_linker_boundaries: Optional[set[str]] = None
    linker_boundary_prescan_summary: dict[str, Any] = {"enabled": False}
    if "linker_design" in set(args.tasks) and int(args.linker_min_boundary_alternatives) > 1:
        eligible_linker_boundaries, linker_boundary_prescan_summary = _prescan_linker_boundary_alternatives(
            paths=paths,
            text_column=str(args.text_column),
            batch_size=int(args.batch_size),
            min_mask_groups=int(args.min_mask_groups),
            max_mask_groups=int(args.max_mask_groups),
            min_target_tokens=int(args.linker_min_target_tokens),
            min_alternatives=int(args.linker_min_boundary_alternatives),
            boundary_signature=str(args.linker_boundary_signature),
            max_rows=int(args.linker_boundary_prescan_rows),
        )
        print(json.dumps({"linker_boundary_prescan": linker_boundary_prescan_summary}, ensure_ascii=False, indent=2))

    def done() -> bool:
        if written >= int(args.max_output_rows):
            return True
        if task_quota:
            tasks_done = all(n_by_task[task] >= quota for task, quota in task_quota.items())
            holes_done = all(n_by_task_holes[("scaffold_decoration", holes)] >= quota for holes, quota in scaffold_hole_quota.items())
            return bool(tasks_done and holes_done)
        return False

    def task_done(task: str) -> bool:
        quota = task_quota.get(task)
        return quota is not None and n_by_task[task] >= int(quota)

    def needed_scaffold_holes() -> list[int]:
        if not scaffold_hole_quota:
            return []
        return [
            holes
            for holes, quota in sorted(scaffold_hole_quota.items())
            if n_by_task_holes[("scaffold_decoration", holes)] < int(quota)
        ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    pending: list[dict[str, Any]] = []
    written = 0
    scanned_rows = 0
    skipped: Counter[str] = Counter()
    n_by_task: Counter[str] = Counter()
    n_by_task_holes: Counter[tuple[str, int]] = Counter()
    n_by_split: Counter[str] = Counter()
    n_by_holes: Counter[int] = Counter()
    linker_middle_counts: Counter[str] = Counter()
    linker_target_motif_tuple_counts: Counter[str] = Counter()
    linker_target_motif_token_counts: Counter[str] = Counter()
    linker_boundary_counts: Counter[str] = Counter()
    total_lengths: list[int] = []
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal writer, pending
        if not pending:
            return
        table = pa.Table.from_pylist(pending, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), SCHEMA, compression="snappy")
        writer.write_table(table)
        pending = []

    try:
        for path in paths:
            if done():
                break
            pf = pq.ParquetFile(path)
            names = set(pf.schema.names)
            text_col = args.text_column if args.text_column in names else ("stamp" if "stamp" in names else "")
            if not text_col:
                skipped["missing_text_column"] += 1
                continue
            columns = [text_col]
            has_smiles = bool(args.smiles_column) and args.smiles_column in names
            has_split = bool(args.split_column) and args.split_column in names
            has_all_pass = bool(args.all_pass_column) and args.all_pass_column in names
            if has_smiles:
                columns.append(args.smiles_column)
            if has_split:
                columns.append(args.split_column)
            if has_all_pass:
                columns.append(args.all_pass_column)

            for batch in pf.iter_batches(batch_size=int(args.batch_size), columns=columns):
                cols = {name: batch.column(i).to_pylist() for i, name in enumerate(columns)}
                n = len(cols[text_col])
                for idx in range(n):
                    if done():
                        break
                    scanned_rows += 1
                    stamp = cols[text_col][idx]
                    if stamp is None:
                        skipped["empty_stamp"] += 1
                        continue
                    stamp_text = str(stamp).strip()
                    if not stamp_text:
                        skipped["empty_stamp"] += 1
                        continue
                    tokens = stamp_text.split()
                    if bool(args.strip_stereo):
                        tokens = _strip_stereo_tokens(tokens)
                        stamp_text = _detok(tokens)
                    groups = _stamp_fragment_groups(tokens)
                    if len(groups) < 2:
                        skipped["too_few_groups"] += 1
                        continue
                    source_smiles = cols[args.smiles_column][idx] if has_smiles else ""
                    split = _choose_split(
                        rng,
                        valid_fraction=float(args.valid_fraction),
                        existing_split=(cols[args.split_column][idx] if has_split else None),
                        use_existing_split=bool(args.use_existing_split),
                    )
                    all_pass = bool(cols[args.all_pass_column][idx]) if has_all_pass else True
                    tasks = list(args.tasks)
                    if task_quota:
                        tasks = [task for task in tasks if not task_done(task)]
                        if not tasks:
                            break
                    rng.shuffle(tasks)
                    built_for_row = 0
                    for task in tasks:
                        if built_for_row >= max(1, int(args.tasks_per_mol)):
                            break
                        if task_done(task):
                            continue
                        for _variant in range(max(1, int(args.variants_per_task))):
                            if built_for_row >= max(1, int(args.tasks_per_mol)) or task_done(task):
                                break
                            preferred_holes = needed_scaffold_holes() if task == "scaffold_decoration" else []
                            if task == "scaffold_decoration" and scaffold_hole_quota and not preferred_holes:
                                break
                            runs = _sample_runs_for_task(
                                groups,
                                task,
                                rng,
                                min_mask_groups=int(args.min_mask_groups),
                                max_mask_groups=int(args.max_mask_groups),
                                min_keep_fraction=float(args.min_keep_fraction),
                                max_keep_fraction=float(args.max_keep_fraction),
                                linker_min_target_tokens=int(args.linker_min_target_tokens),
                                length_strategy=str(args.length_strategy),
                                preferred_scaffold_holes=preferred_holes,
                                scaffold_allow_linker_fallback=bool(args.scaffold_linker_fallback),
                            )
                            if not runs:
                                skipped[f"{task}:no_span"] += 1
                                continue
                            if args.construction == "completion":
                                row = _build_completion_row(
                                    source_smiles=source_smiles,
                                    source_stamp=stamp_text,
                                    groups=groups,
                                    task=task,
                                    runs=runs,
                                    split=split,
                                    all_pass=all_pass,
                                    max_total_tokens=int(args.max_total_tokens),
                                )
                            else:
                                row = _build_row(
                                    source_smiles=source_smiles,
                                    source_stamp=stamp_text,
                                    groups=groups,
                                    task=task,
                                    runs=runs,
                                    split=split,
                                    all_pass=all_pass,
                                    max_total_tokens=int(args.max_total_tokens),
                            )
                            if row is None:
                                skipped[f"{task}:filtered"] += 1
                                continue
                            linker_boundary = ""
                            linker_target_motif_tuple = ""
                            linker_target_motif_counts: Counter[str] = Counter()
                            if task == "linker_design":
                                linker_boundary = _linker_boundary_signature(
                                    groups,
                                    runs[0],
                                    mode=str(args.linker_boundary_signature),
                                )
                                if not linker_boundary:
                                    skipped[f"{task}:missing_boundary"] += 1
                                    continue
                                if eligible_linker_boundaries is not None and linker_boundary not in eligible_linker_boundaries:
                                    skipped[f"{task}:boundary_alternatives_lt_{int(args.linker_min_boundary_alternatives)}"] += 1
                                    continue
                                middle_stamp = str(row["middle_stamp"])
                                target_motifs = _motif_tokens(middle_stamp.split())
                                linker_target_motif_tuple = _detok(target_motifs)
                                linker_target_motif_counts = Counter(target_motifs)
                                if (
                                    int(args.linker_middle_frequency_cap) > 0
                                    and linker_middle_counts[middle_stamp] >= int(args.linker_middle_frequency_cap)
                                ):
                                    skipped[f"{task}:middle_frequency_cap"] += 1
                                    continue
                                if (
                                    int(args.linker_target_motif_tuple_cap) > 0
                                    and linker_target_motif_tuple_counts[linker_target_motif_tuple]
                                    >= int(args.linker_target_motif_tuple_cap)
                                ):
                                    skipped[f"{task}:motif_tuple_frequency_cap"] += 1
                                    continue
                                if int(args.linker_target_motif_token_cap) > 0 and any(
                                    linker_target_motif_token_counts[motif] + count > int(args.linker_target_motif_token_cap)
                                    for motif, count in linker_target_motif_counts.items()
                                ):
                                    skipped[f"{task}:motif_token_frequency_cap"] += 1
                                    continue
                                if (
                                    int(args.linker_boundary_frequency_cap) > 0
                                    and linker_boundary_counts[linker_boundary] >= int(args.linker_boundary_frequency_cap)
                                ):
                                    skipped[f"{task}:boundary_frequency_cap"] += 1
                                    continue
                            row_holes = int(row["n_holes"])
                            if task == "scaffold_decoration" and scaffold_hole_quota:
                                if row_holes not in scaffold_hole_quota:
                                    skipped[f"{task}:hole_not_requested:{row_holes}"] += 1
                                    continue
                                if n_by_task_holes[(task, row_holes)] >= int(scaffold_hole_quota[row_holes]):
                                    skipped[f"{task}:hole_quota_done:{row_holes}"] += 1
                                    continue
                            pending.append(row)
                            written += 1
                            built_for_row += 1
                            n_by_task[task] += 1
                            n_by_task_holes[(task, row_holes)] += 1
                            n_by_split[split] += 1
                            n_by_holes[row_holes] += 1
                            if task == "linker_design":
                                linker_middle_counts[str(row["middle_stamp"])] += 1
                                linker_target_motif_tuple_counts[linker_target_motif_tuple] += 1
                                linker_target_motif_token_counts.update(linker_target_motif_counts)
                                linker_boundary_counts[linker_boundary] += 1
                            total_lengths.append(len(str(row["stamp_smiles"]).split()))
                            prompt_lengths.append(int(row["prompt_len"]))
                            target_lengths.append(int(row["target_len"]))
                            if len(pending) >= int(args.write_batch_size):
                                flush()
                            if done():
                                break
                        if done():
                            break
                    if built_for_row == 0:
                        skipped["row_no_task"] += 1
                if done():
                    break
    finally:
        flush()
        if writer is not None:
            writer.close()

    summary = {
        "output_path": str(out_path),
        "input_path_count": len(paths),
        "input_paths_head": paths[:10],
        "scanned_rows": int(scanned_rows),
        "output_rows": int(written),
        "tasks": list(args.tasks),
        "balance_tasks": bool(args.balance_tasks),
        "task_quotas": task_quota,
        "scaffold_hole_quotas": {str(k): int(v) for k, v in sorted(scaffold_hole_quota.items())},
        "construction": str(args.construction),
        "tasks_per_mol": int(args.tasks_per_mol),
        "variants_per_task": int(args.variants_per_task),
        "length_strategy": str(args.length_strategy),
        "strip_stereo": bool(args.strip_stereo),
        "linker_min_target_tokens": int(args.linker_min_target_tokens),
        "linker_middle_frequency_cap": int(args.linker_middle_frequency_cap),
        "linker_target_motif_tuple_cap": int(args.linker_target_motif_tuple_cap),
        "linker_target_motif_token_cap": int(args.linker_target_motif_token_cap),
        "linker_boundary_frequency_cap": int(args.linker_boundary_frequency_cap),
        "linker_min_boundary_alternatives": int(args.linker_min_boundary_alternatives),
        "linker_boundary_signature": str(args.linker_boundary_signature),
        "linker_boundary_prescan": linker_boundary_prescan_summary,
        "scaffold_linker_fallback": bool(args.scaffold_linker_fallback),
        "n_by_task": dict(n_by_task),
        "n_by_task_holes": {f"{task}:{holes}": int(v) for (task, holes), v in sorted(n_by_task_holes.items())},
        "n_by_split": dict(n_by_split),
        "n_by_holes": {str(k): int(v) for k, v in sorted(n_by_holes.items())},
        "linker_unique_middle": int(len(linker_middle_counts)),
        "linker_unique_target_motif_tuple": int(len(linker_target_motif_tuple_counts)),
        "linker_unique_target_motif_tokens": int(len(linker_target_motif_token_counts)),
        "linker_unique_boundary": int(len(linker_boundary_counts)),
        "linker_top_middle": _top_counter(linker_middle_counts),
        "linker_top_target_motif_tuple": _top_counter(linker_target_motif_tuple_counts),
        "linker_top_target_motif_tokens": _top_counter(linker_target_motif_token_counts),
        "linker_top_boundary": _top_counter(linker_boundary_counts),
        "n_by_decode_mode": {"generated_full": int(written)} if args.construction == "completion" else {"splice": int(written)},
        "skipped": dict(skipped),
        "total_token_lengths": _length_summary(total_lengths),
        "prompt_token_lengths": _length_summary(prompt_lengths),
        "target_token_lengths": _length_summary(target_lengths),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    summary_path = out_path.with_name(out_path.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

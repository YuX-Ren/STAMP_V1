#!/usr/bin/env python3
"""Run a GenMol-aligned STAMP attaching baseline on PMO/TDC tasks.

The baseline keeps the same high-level loop as GenMol's attaching ablation:

    oracle-pretested unit bank
      -> sample two attachable units
      -> deterministic STAMP concatenation
      -> decode / size-filter / oracle
      -> decompose scored child back into attachable units
      -> keep top population

This script deliberately does not use AO-GPT rollout or planner actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.mcts_guacamol import (  # noqa: E402
    STAMPDecoderImpl,
    count_open_ports,
    parse_next_group,
)
from stamp.benchmark.mcts_unified.pmo_oracle import (  # noqa: E402
    PMO_TASKS,
    PMOOracle,
    auc_top_k_from_log,
)


@dataclass
class AttachUnit:
    tokens: Tuple[str, ...]
    score: float
    source: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clean_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if not tok or tok.startswith("<"):
            continue
        out.append(tok)
        if tok == "[END]":
            break
    return out


def split_motif_groups(tokens: Sequence[str]) -> List[Tuple[int, int, Tuple[str, ...]]]:
    cleaned = clean_tokens(tokens)
    groups: List[Tuple[int, int, Tuple[str, ...]]] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "[END]":
            break
        end = parse_next_group(cleaned, i)
        if end is None or end <= i:
            i += 1
            continue
        groups.append((i, end, tuple(cleaned[i:end])))
        i = end
    return groups


def decode_tokens(decoder: STAMPDecoderImpl, tokens: Sequence[str]) -> Optional[str]:
    cleaned = clean_tokens(tokens)
    if not cleaned:
        return None
    if "[END]" not in cleaned:
        cleaned = cleaned + ["[END]"]
    return decoder.decode_to_smiles(cleaned)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumAtoms())


def task_atom_window(task: str) -> Tuple[int, int]:
    if task in {"gsk3b", "jnk3"}:
        return (30, 80)
    if (
        "similarity" in task
        or "rediscovery" in task
        or task.startswith("median")
        or task.startswith("isomers_")
        or task == "qed"
    ):
        return (10, 30)
    return (20, 40)


def is_attachable(tokens: Sequence[str]) -> bool:
    return count_open_ports(tokens) == 1


def iter_attachable_subtrees(
    tokens: Sequence[str],
    *,
    max_subtree_groups: int,
) -> Iterable[Tuple[str, ...]]:
    groups = split_motif_groups(tokens)
    cleaned = clean_tokens(tokens)
    seen: set[Tuple[str, ...]] = set()
    for _start, _end, group in groups:
        if is_attachable(group) and group not in seen:
            seen.add(group)
            yield group
    n = len(groups)
    for i in range(n):
        for width in range(2, max_subtree_groups + 1):
            j = i + width
            if j > n:
                break
            start = groups[i][0]
            end = groups[j - 1][1]
            subtree = tuple(cleaned[start:end])
            if is_attachable(subtree) and subtree not in seen:
                seen.add(subtree)
                yield subtree


def merge_population(
    population: Dict[Tuple[str, ...], AttachUnit],
    *,
    new_units: Iterable[AttachUnit],
    population_size: int,
) -> None:
    for unit in new_units:
        cur = population.get(unit.tokens)
        if cur is None or unit.score > cur.score:
            population[unit.tokens] = unit
    if len(population) <= population_size:
        return
    ranked = sorted(population.values(), key=lambda u: (-u.score, len(u.tokens), u.source))
    keep = {u.tokens for u in ranked[:population_size]}
    for key in list(population):
        if key not in keep:
            del population[key]


def load_initial_population(
    bank_dir: str | Path,
    *,
    population_size: int,
) -> Dict[Tuple[str, ...], AttachUnit]:
    bank = Path(bank_dir)
    merged: Dict[Tuple[str, ...], AttachUnit] = {}
    for name in ("motifs.csv", "subtrees.csv"):
        path = bank / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                toks = tuple((row.get("tokens") or "").split())
                if not toks or not is_attachable(toks):
                    continue
                try:
                    mean_score = float(row.get("mean_score") or 0.0)
                    best_score = float(row.get("best_score") or mean_score)
                except ValueError:
                    continue
                score = max(mean_score, best_score)
                rows.append(AttachUnit(tokens=toks, score=score, source=name.removesuffix(".csv")))
        rows.sort(key=lambda u: -u.score)
        merge_population(merged, new_units=rows[:population_size], population_size=population_size)
    if not merged:
        raise RuntimeError(f"No attachable units found in bank: {bank}")
    return merged


def sample_pair(population: Dict[Tuple[str, ...], AttachUnit]) -> Tuple[AttachUnit, AttachUnit]:
    units = list(population.values())
    if len(units) == 1:
        return units[0], units[0]
    weights = np.array([max(1e-8, u.score) for u in units], dtype=float)
    weights = weights / float(weights.sum())
    idx = np.random.choice(len(units), size=2, replace=False, p=weights)
    a, b = units[int(idx[0])], units[int(idx[1])]
    return a, b


def score_if_new(
    *,
    smiles: str,
    oracle: PMOOracle,
    seen: Dict[str, float],
    budget: int,
) -> Optional[float]:
    canon = canonicalize_smiles(smiles)
    if canon is None or canon in seen or oracle.n_calls >= budget:
        return None
    score = float(oracle(canon))
    seen[canon] = score
    return score


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run STAMP attaching PMO baseline.")
    p.add_argument("--task", required=True, choices=PMO_TASKS)
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--max-subtree-groups", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = STAMPDecoderImpl()
    oracle = PMOOracle(args.task, max_calls=args.budget)
    population = load_initial_population(args.bank_dir, population_size=args.population_size)
    seen_smiles: Dict[str, float] = {}
    min_atoms, max_atoms = task_atom_window(args.task)
    t0 = time.time()
    accepted_log: List[Dict[str, object]] = []
    attempts = 0
    max_attempts = max(args.budget * 50, 50000)

    while oracle.n_calls < args.budget and attempts < max_attempts:
        attempts += 1
        left, right = sample_pair(population)
        merged_tokens = list(left.tokens) + list(right.tokens)
        smiles = decode_tokens(decoder, merged_tokens)
        if smiles is None:
            continue
        n_atoms = atom_count(smiles)
        if n_atoms < min_atoms or n_atoms > max_atoms:
            continue
        score = score_if_new(smiles=smiles, oracle=oracle, seen=seen_smiles, budget=args.budget)
        if score is None:
            continue

        child_tokens = decoder.encoder.encode_smiles(smiles).split()
        new_units = [
            AttachUnit(tokens=unit, score=score, source="child")
            for unit in iter_attachable_subtrees(child_tokens, max_subtree_groups=args.max_subtree_groups)
        ]
        merge_population(population, new_units=new_units, population_size=args.population_size)
        accepted_log.append(
            {
                "oracle_call": oracle.n_calls,
                "left_tokens": list(left.tokens),
                "right_tokens": list(right.tokens),
                "child_smiles": smiles,
                "child_score": score,
                "n_atoms": n_atoms,
            }
        )
        if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
            best = max(seen_smiles.values()) if seen_smiles else 0.0
            print(
                f"[attach] calls={oracle.n_calls}/{args.budget} best={best:.4f} "
                f"population={len(population)}",
                flush=True,
            )

    metrics = auc_top_k_from_log(oracle._log, k=10, max_calls=args.budget)
    best_smiles = None
    best_score = 0.0
    if seen_smiles:
        best_smiles, best_score = max(seen_smiles.items(), key=lambda kv: kv[1])

    payload = {
        "method": "stamp_attaching",
        "task": args.task,
        "budget": args.budget,
        "seed": args.seed,
        "bank_dir": args.bank_dir,
        "population_size": args.population_size,
        "max_subtree_groups": args.max_subtree_groups,
        "attempts": attempts,
        "oracle_calls": oracle.n_calls,
        "elapsed_seconds": time.time() - t0,
        "best_reward": best_score,
        "best_smiles": best_smiles,
        "pmo_auc_top10": metrics,
        "accepted_log": accepted_log,
        "final_population": [
            {
                "tokens": list(u.tokens),
                "score": u.score,
                "source": u.source,
            }
            for u in sorted(population.values(), key=lambda x: -x.score)[:25]
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "best_reward": best_score,
                "best_smiles": best_smiles,
                "auc_top10": metrics["auc_top_k"],
                "final_top10": metrics["final_top_k_mean"],
                "oracle_calls": oracle.n_calls,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

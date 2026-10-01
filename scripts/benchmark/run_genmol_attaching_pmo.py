#!/usr/bin/env python3
"""Run the GenMol attaching-only PMO baseline.

This is a direct baseline for Table 5 / Table 16 style comparisons:

    ZINC250k oracle-pretested single-cut fragment bank
      -> sample two fragments from top-100 population
      -> RDKit attach reaction
      -> task-specific size filter
      -> oracle
      -> cut scored child back into fragments and update top population

No fragment remasking. No AO-GPT. No STAMP planner.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PMO_ORACLE_PATH = ROOT / "stamp" / "benchmark" / "mcts_unified" / "pmo_oracle.py"
_PMO_ORACLE_SPEC = importlib.util.spec_from_file_location("stamp_pmo_oracle", _PMO_ORACLE_PATH)
if _PMO_ORACLE_SPEC is None or _PMO_ORACLE_SPEC.loader is None:
    raise ImportError(f"Unable to load PMO oracle module from {_PMO_ORACLE_PATH}")
_PMO_ORACLE_MODULE = importlib.util.module_from_spec(_PMO_ORACLE_SPEC)
_PMO_ORACLE_SPEC.loader.exec_module(_PMO_ORACLE_MODULE)

PMO_TASKS = _PMO_ORACLE_MODULE.PMO_TASKS
PMOOracle = _PMO_ORACLE_MODULE.PMOOracle
auc_top_k_from_log = _PMO_ORACLE_MODULE.auc_top_k_from_log


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def cut(smiles: str) -> Set[str]:
    def cut_nonring(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, ...]]:
        patt = Chem.MolFromSmarts("[*]-;!@[*]")
        if patt is None or not mol.HasSubstructMatch(patt):
            return None
        bis = random.choice(mol.GetSubstructMatches(patt))
        bond = mol.GetBondBetweenAtoms(bis[0], bis[1])
        if bond is None:
            return None
        fragments_mol = Chem.FragmentOnBonds(
            mol,
            [bond.GetIdx()],
            addDummies=True,
            dummyLabels=[(1, 1)],
        )
        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    frags: Set[str] = set()
    for _ in range(3):
        parts = cut_nonring(mol)
        if parts is not None:
            for frag in parts:
                smi = Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
                if smi:
                    frags.add(smi)
    return frags


def attach_fragments(frag1: str, frag2: str) -> Optional[str]:
    rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
    if rxn is None:
        return None
    mol1 = Chem.MolFromSmiles(frag1)
    mol2 = Chem.MolFromSmiles(frag2)
    if mol1 is None or mol2 is None:
        return None
    mols = rxn.RunReactants((mol1, mol2))
    if not mols:
        return None
    child = mols[np.random.randint(len(mols))][0]
    try:
        return Chem.MolToSmiles(child, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def task_atom_window(task: str) -> Tuple[int, int]:
    if task in {
        "albuterol_similarity",
        "isomers_c7h8n2o2",
        "isomers_c9h10n2o2pf2cl",
        "median1",
        "qed",
        "sitagliptin_mpo",
        "zaleplon_mpo",
    }:
        return (10, 30)
    if task in {"gsk3b", "jnk3"}:
        return (30, 80)
    return (20, 40)


def atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumAtoms())


def build_or_load_vocab(
    *,
    task: str,
    zinc_path: str | Path,
    vocab_csv: str | Path,
    score_column: Optional[str],
) -> Path:
    out = Path(vocab_csv)
    if out.exists():
        return out

    from tdc import Oracle as TDCOracle

    zinc = pd.read_csv(zinc_path)
    if score_column and score_column in zinc.columns:
        scores = zinc[score_column].astype(float).tolist()
    else:
        oracle = TDCOracle(name=task)
        scores = []
        for smi in zinc["smiles"].tolist():
            canon = canonicalize_smiles(str(smi))
            if canon is None:
                scores.append(0.0)
                continue
            try:
                scores.append(float(oracle(canon)))
            except Exception:
                scores.append(0.0)

    frag2cnt: Dict[str, int] = defaultdict(int)
    frag2score: Dict[str, float] = defaultdict(float)
    for smi, score in zip(zinc["smiles"].tolist(), scores):
        canon = canonicalize_smiles(str(smi))
        if canon is None:
            continue
        for frag in cut(canon):
            frag2cnt[frag] += 1
            frag2score[frag] += float(score)

    rows = []
    for frag, total in frag2score.items():
        mean = total / max(1, frag2cnt[frag])
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        rows.append(
            {
                "frag": frag,
                "score": mean,
                "size": int(mol.GetNumAtoms()),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frag", "score", "size"])
        writer.writeheader()
        for row in rows[:10000]:
            writer.writerow(row)
    return out


def score_canonical_smiles(
    *,
    canon: str,
    oracle: PMOOracle,
    seen: Dict[str, float],
    budget: int,
) -> Tuple[Optional[float], str]:
    if canon in seen:
        return None, "duplicate"
    if oracle.n_calls >= budget:
        return None, "budget"
    score = float(oracle(canon))
    seen[canon] = score
    return score, "accepted"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GenMol attaching-only PMO baseline.")
    p.add_argument("--task", required=True, choices=PMO_TASKS)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--zinc-path", default="data/zinc250k.csv")
    p.add_argument("--score-column", default=None)
    p.add_argument("--vocab-csv", default=None)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--sample-pool-size", type=int, default=1000)
    p.add_argument("--sample-pool-expand-step", type=int, default=100)
    p.add_argument("--sample-pool-expand-every", type=int, default=50)
    p.add_argument("--max-pair-reuses", type=int, default=1)
    p.add_argument(
        "--sampling-mode",
        choices=("repaired", "raw"),
        default="repaired",
        help="repaired: broadened pool + pair dedup + search for novel child; "
        "raw: mimic GenMol attaching loop dynamics more closely.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-attempts", type=int, default=1000)
    p.add_argument("--stall-log-every", type=int, default=100)
    p.add_argument("--max-no-progress-iters", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_csv = args.vocab_csv or f"results/pmo_fragment_vocab/{args.task}.csv"
    vocab_path = build_or_load_vocab(
        task=args.task,
        zinc_path=args.zinc_path,
        vocab_csv=vocab_csv,
        score_column=args.score_column,
    )
    keep_size = max(args.population_size, args.sample_pool_size)
    df = pd.read_csv(vocab_path).iloc[:keep_size]
    population: List[Tuple[float, str]] = list(zip(df["score"].astype(float), df["frag"].astype(str)))
    oracle = PMOOracle(args.task, max_calls=args.budget)
    min_atoms, max_atoms = task_atom_window(args.task)
    seen_smiles: Dict[str, float] = {}
    seen_pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    accepted_log: List[Dict[str, object]] = []
    stall_log: List[Dict[str, object]] = []
    t0 = time.time()
    pop_frags: List[str] = [frag for _score, frag in population]
    no_progress_iters = 0

    it = 0
    while oracle.n_calls < args.budget:
        it += 1
        child = None
        attempt_stats = {
            "attach_fail": 0,
            "size_fail": 0,
            "duplicate": 0,
            "invalid": 0,
            "pair_repeat": 0,
        }
        if args.sampling_mode == "raw":
            current_pool_size = min(len(pop_frags), args.population_size)
        else:
            current_pool_size = min(
                len(pop_frags),
                args.sample_pool_size,
            )
            if no_progress_iters > 0:
                current_pool_size = min(
                    len(pop_frags),
                    max(
                        current_pool_size,
                        args.population_size
                        + (no_progress_iters // max(1, args.sample_pool_expand_every)) * args.sample_pool_expand_step,
                    ),
                    args.sample_pool_size,
                )
        candidate_frags = pop_frags[:current_pool_size]
        for _ in range(args.sample_attempts):
            frag1, frag2 = random.sample(candidate_frags, 2)
            pair_key = tuple(sorted((frag1, frag2)))
            if args.sampling_mode != "raw" and seen_pairs[pair_key] >= args.max_pair_reuses:
                attempt_stats["pair_repeat"] += 1
                continue
            smiles = attach_fragments(frag1, frag2)
            if smiles is None:
                attempt_stats["attach_fail"] += 1
                continue
            n_atoms = atom_count(smiles)
            if not (min_atoms <= n_atoms <= max_atoms):
                attempt_stats["size_fail"] += 1
                continue
            canon = canonicalize_smiles(smiles)
            if canon is None:
                attempt_stats["invalid"] += 1
                continue
            if args.sampling_mode != "raw" and canon in seen_smiles:
                attempt_stats["duplicate"] += 1
                seen_pairs[pair_key] += 1
                continue
            child = (canon, frag1, frag2, n_atoms, pair_key)
            break
        if child is None:
            no_progress_iters += 1
            if args.stall_log_every > 0 and no_progress_iters % args.stall_log_every == 0:
                best = max(seen_smiles.values()) if seen_smiles else 0.0
                event = {
                    "iter": it,
                    "oracle_calls": oracle.n_calls,
                    "reason": "no_valid_child",
                    "best_reward": best,
                    "attempt_stats": attempt_stats,
                    "sample_pool_size": current_pool_size,
                }
                stall_log.append(event)
                print(f"[genmol-A-stall] {json.dumps(event, sort_keys=True)}", flush=True)
            if args.max_no_progress_iters > 0 and no_progress_iters >= args.max_no_progress_iters:
                break
            continue

        smiles, frag1, frag2, n_atoms, pair_key = child
        score, score_reason = score_canonical_smiles(
            canon=smiles,
            oracle=oracle,
            seen=seen_smiles,
            budget=args.budget,
        )
        if score is None:
            no_progress_iters += 1
            if score_reason in attempt_stats:
                attempt_stats[score_reason] += 1
            if args.stall_log_every > 0 and no_progress_iters % args.stall_log_every == 0:
                best = max(seen_smiles.values()) if seen_smiles else 0.0
                event = {
                    "iter": it,
                    "oracle_calls": oracle.n_calls,
                    "reason": score_reason,
                    "best_reward": best,
                    "attempt_stats": attempt_stats,
                    "child_smiles": smiles,
                    "n_atoms": n_atoms,
                    "sample_pool_size": current_pool_size,
                }
                stall_log.append(event)
                print(f"[genmol-A-stall] {json.dumps(event, sort_keys=True)}", flush=True)
            if args.max_no_progress_iters > 0 and no_progress_iters >= args.max_no_progress_iters:
                break
            continue

        no_progress_iters = 0
        if args.sampling_mode != "raw":
            seen_pairs[pair_key] += 1
        population_frags = {frag for _prop, frag in population}
        elite_floor_idx = min(args.population_size, len(population)) - 1
        elite_floor = population[elite_floor_idx][0] if elite_floor_idx >= 0 else float("-inf")
        if score > elite_floor:
            new_frags = [(score, frag) for frag in cut(smiles) if frag not in population_frags]
            population.extend(new_frags)
            population.sort(reverse=True)
            population = population[: (args.population_size if args.sampling_mode == "raw" else keep_size)]
            pop_frags = [frag for _score, frag in population]

        accepted_log.append(
            {
                "oracle_call": oracle.n_calls,
                "frag1": frag1,
                "frag2": frag2,
                "child_smiles": smiles,
                "child_score": score,
                "n_atoms": n_atoms,
            }
        )
        if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
            best = max(seen_smiles.values()) if seen_smiles else 0.0
            print(
                f"[genmol-A] calls={oracle.n_calls}/{args.budget} best={best:.4f} "
                f"population={len(population)}",
                flush=True,
            )

    metrics = auc_top_k_from_log(oracle._log, k=10, max_calls=args.budget)
    best_smiles = None
    best_score = 0.0
    if seen_smiles:
        best_smiles, best_score = max(seen_smiles.items(), key=lambda kv: kv[1])

    payload = {
        "method": "genmol_attaching_exact",
        "task": args.task,
        "budget": args.budget,
        "seed": args.seed,
        "vocab_csv": str(vocab_path),
        "population_size": args.population_size,
        "sample_pool_size": args.sample_pool_size,
        "sample_pool_expand_step": args.sample_pool_expand_step,
        "sample_pool_expand_every": args.sample_pool_expand_every,
        "max_pair_reuses": args.max_pair_reuses,
        "sampling_mode": args.sampling_mode,
        "oracle_calls": oracle.n_calls,
        "elapsed_seconds": time.time() - t0,
        "best_reward": best_score,
        "best_smiles": best_smiles,
        "pmo_auc_top10": metrics,
        "accepted_log": accepted_log,
        "stall_log": stall_log,
        "terminated_after_no_progress": bool(
            args.max_no_progress_iters > 0 and no_progress_iters >= args.max_no_progress_iters
        ),
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

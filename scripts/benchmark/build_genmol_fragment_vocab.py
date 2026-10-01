#!/usr/bin/env python3
"""Build a GenMol-style oracle-pretested PMO fragment vocabulary.

For one PMO task:
    ZINC250k smiles
      -> task oracle scores (or reuse a score column)
      -> random non-ring single-bond cuts, repeated 3 times
      -> fragment mean score aggregation
      -> top-N fragment CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.mcts_unified.pmo_oracle import PMO_TASKS  # noqa: E402


_WORKER_ORACLE = None
_WORKER_TASK = None


def _set_seed(seed: int) -> None:
    random.seed(seed)


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


def cut(smiles: str, repeats: int = 3, seed: Optional[int] = None) -> Set[str]:
    rng = random if seed is None else random.Random(seed)

    def cut_nonring(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, ...]]:
        patt = Chem.MolFromSmarts("[*]-;!@[*]")
        if patt is None or not mol.HasSubstructMatch(patt):
            return None
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        bis = rng.choice(matches)
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
    for _ in range(repeats):
        parts = cut_nonring(mol)
        if parts is None:
            continue
        for frag in parts:
            smi = Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
            if smi:
                frags.add(smi)
    return frags


def _init_worker(task: str) -> None:
    global _WORKER_ORACLE, _WORKER_TASK
    if _WORKER_TASK == task and _WORKER_ORACLE is not None:
        return
    from tdc import Oracle as TDCOracle

    _WORKER_ORACLE = TDCOracle(name=task)
    _WORKER_TASK = task


def _process_with_score_column(item: Tuple[int, str, float, int]) -> Tuple[bool, float, Set[str]]:
    idx, raw, score, cut_repeats = item
    canon = canonicalize_smiles(raw)
    if canon is None:
        return False, 0.0, set()
    return True, float(score), cut(canon, repeats=cut_repeats, seed=idx)


def _process_with_oracle(item: Tuple[int, str, int]) -> Tuple[bool, float, Set[str]]:
    idx, raw, cut_repeats = item
    canon = canonicalize_smiles(raw)
    if canon is None:
        return False, 0.0, set()
    try:
        score = float(_WORKER_ORACLE(canon))
    except Exception:
        score = 0.0
    return True, score, cut(canon, repeats=cut_repeats, seed=idx)


def _iter_results(
    smiles_list: Sequence[str],
    score_values: Optional[Sequence[float]],
    task: str,
    cut_repeats: int,
    num_workers: int,
    chunksize: int,
) -> Iterable[Tuple[bool, float, Set[str]]]:
    if score_values is not None:
        items = ((i, raw, float(score_values[i - 1]), cut_repeats) for i, raw in enumerate(smiles_list, start=1))
        if num_workers <= 1:
            for item in items:
                yield _process_with_score_column(item)
            return
        with mp.Pool(processes=num_workers) as pool:
            yield from pool.imap(_process_with_score_column, items, chunksize=chunksize)
        return

    items = ((i, raw, cut_repeats) for i, raw in enumerate(smiles_list, start=1))
    if num_workers <= 1:
        _init_worker(task)
        for item in items:
            yield _process_with_oracle(item)
        return
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(task,)) as pool:
        yield from pool.imap(_process_with_oracle, items, chunksize=chunksize)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build GenMol-style PMO fragment vocabulary.")
    p.add_argument("--task", required=True, choices=PMO_TASKS)
    p.add_argument("--input", default="data/zinc250k.csv")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--score-column", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--top-fragments", type=int, default=10000)
    p.add_argument("--cut-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=32)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    smiles_list = df[args.smiles_column].astype(str).tolist()

    score_values: Optional[Sequence[float]]
    if args.score_column and args.score_column in df.columns:
        score_values = [float(x) for x in df[args.score_column].tolist()]
        score_source = f"column:{args.score_column}"
    else:
        score_values = None
        score_source = f"tdc:{args.task}"

    frag2cnt: Dict[str, int] = defaultdict(int)
    frag2score: Dict[str, float] = defaultdict(float)
    t0 = time.time()
    n_valid = 0
    for i, (is_valid, score, frags) in enumerate(
        _iter_results(
            smiles_list=smiles_list,
            score_values=score_values,
            task=args.task,
            cut_repeats=args.cut_repeats,
            num_workers=max(1, args.num_workers),
            chunksize=max(1, args.chunksize),
        ),
        start=1,
    ):
        if is_valid:
            n_valid += 1
        for frag in frags:
            frag2cnt[frag] += 1
            frag2score[frag] += float(score)
        if args.progress_every > 0 and i % args.progress_every == 0:
            dt = time.time() - t0
            print(
                f"[frag-vocab] seen={i} valid={n_valid} unique_frags={len(frag2cnt)} elapsed={dt:.1f}s",
                flush=True,
            )

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
                "count": frag2cnt[frag],
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frag", "score", "size", "count"])
        writer.writeheader()
        for row in rows[: args.top_fragments]:
            writer.writerow(row)

    summary = {
        "task": args.task,
        "input": str(inp),
        "output": str(out),
        "score_source": score_source,
        "n_seen": len(smiles_list),
        "n_valid": n_valid,
        "n_unique_fragments": len(frag2cnt),
        "top_fragments": args.top_fragments,
        "cut_repeats": args.cut_repeats,
        "elapsed_seconds": time.time() - t0,
    }
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

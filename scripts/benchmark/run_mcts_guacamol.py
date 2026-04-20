#!/usr/bin/env python3
"""
Run STAMP-MCTS on the GuacaMol goal-directed benchmark suite.

This is the implementation referenced in ``docs/ao_gpt_improvement.md`` /
``mcts_protype.py``: the prototype MCTS from the repo root is driven by
the trained STAMP AR LM (proposal + rollout policy) and scored against
GuacaMol benchmarks (Celecoxib rediscovery, Troglitazone rediscovery, ...
Scaffold Hop).

Basic usage
-----------

Run one benchmark with a small budget (smoke test)::

    PYTHONPATH=. python scripts/benchmark/run_mcts_guacamol.py \
        --model-dir results/formal_compare_20260418/zinc250k_v2k/ar_samecfg_fix \
        --benchmarks "Celecoxib rediscovery" \
        --budget 200 --output results/mcts_guacamol/smoke.json

Run the full 20-task suite::

    PYTHONPATH=. python scripts/benchmark/run_mcts_guacamol.py \
        --model-dir results/formal_compare_20260418/zinc250k_v2k/ar_samecfg_fix \
        --benchmarks all \
        --budget 10000 \
        --output results/mcts_guacamol/v2_10k.json

Per-benchmark scores follow the "paper" convention: for each task we keep
the top-k molecules (k defined by the benchmark's
``contribution_specification``) and report both the best single score and
the GuacaMol top-k mean (reward over the ContributionSpecification).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# Make repo root importable for ``mcts_protype`` + ``stamp.benchmark.*``.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mcts_protype import MCTS, MCTSConfig, RandomSamplingBaseline  # noqa: E402
from stamp.benchmark.mcts_guacamol import (  # noqa: E402
    ARPolicyModel,
    GuacaMolOracle,
    STAMPDecoderImpl,
    list_guacamol_benchmarks,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark aggregation in the GuacaMol sense
# ---------------------------------------------------------------------------

def _topk_mean_from_results(
    results: Sequence[Dict[str, Any]],
    benchmark,
) -> Dict[str, Any]:
    """Compute the GuacaMol-style score for a benchmark given our oracle
    trajectory. Selects the best-scoring *unique* SMILES (by canonical
    form after RDKit sanitisation) for each ``top_count`` bucket in the
    benchmark's contribution specification, and averages.

    Returns a dict::

        {"final_score": float,
         "top_k_means":  {k: mean_top_k_score},
         "top_molecules": [{smiles, score}, ...]}
    """
    from rdkit import Chem

    spec = benchmark.contribution_specification
    top_counts = sorted({int(c) for c in spec.top_counts})

    best_by_smiles: Dict[str, float] = {}
    for row in results:
        smi = (row.get("smiles") or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        score = float(row.get("reward", 0.0))
        if canon not in best_by_smiles or score > best_by_smiles[canon]:
            best_by_smiles[canon] = score

    sorted_scores = sorted(best_by_smiles.items(), key=lambda x: -x[1])
    top_molecules = [
        {"smiles": smi, "score": sc}
        for smi, sc in sorted_scores[: max(top_counts) if top_counts else 10]
    ]

    top_k_means: Dict[str, float] = {}
    for k in top_counts:
        selected = [sc for _, sc in sorted_scores[:k]]
        if not selected:
            top_k_means[str(k)] = 0.0
        else:
            # GuacaMol pads missing entries with 0 when fewer than k molecules.
            padded = selected + [0.0] * (k - len(selected))
            top_k_means[str(k)] = float(np.mean(padded))

    final_score = float(np.mean(list(top_k_means.values()))) if top_k_means else 0.0
    return {
        "final_score": final_score,
        "top_k_means": top_k_means,
        "top_molecules": top_molecules,
        "unique_valid_count": len(best_by_smiles),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_benchmarks(requested: Sequence[str], version: str) -> List[str]:
    all_names = list_guacamol_benchmarks(version=version)
    if len(requested) == 1 and requested[0].lower() == "all":
        return all_names
    resolved: List[str] = []
    missing: List[str] = []
    lookup = {n.lower(): n for n in all_names}
    for name in requested:
        key = name.lower()
        if key in lookup:
            resolved.append(lookup[key])
        else:
            # substring match
            cands = [n for n in all_names if key in n.lower()]
            if len(cands) == 1:
                resolved.append(cands[0])
            else:
                missing.append(name)
    if missing:
        raise SystemExit(
            f"Unknown benchmark(s): {missing}\nAvailable ({version}):\n  "
            + "\n  ".join(all_names)
        )
    return resolved


def run_single_benchmark(
    benchmark_name: str,
    *,
    decoder: STAMPDecoderImpl,
    policy: ARPolicyModel,
    version: str,
    budget: int,
    mcts_cfg: MCTSConfig,
    run_baseline: bool,
    baseline_fraction: float,
    seed: int,
    early_stop_reward: float,
) -> Dict[str, Any]:
    logger.info(f"[{benchmark_name}] building oracle (version={version})")
    oracle = GuacaMolOracle(benchmark_name, version=version)

    summary: Dict[str, Any] = {
        "benchmark": benchmark_name,
        "version": version,
        "budget": budget,
        "seed": seed,
    }

    # --- Optional LM baseline with a *fraction* of the budget for reference.
    if run_baseline:
        b_budget = max(1, int(budget * baseline_fraction))
        logger.info(f"[{benchmark_name}] baseline sampling ({b_budget} calls)")
        _set_seed(seed)
        t0 = time.time()
        baseline = RandomSamplingBaseline(decoder, policy, oracle).run(
            b_budget, early_stop_reward=early_stop_reward,
        )
        baseline["elapsed_seconds"] = time.time() - t0
        baseline["topk"] = _topk_mean_from_results(
            baseline["all_results"], oracle.benchmark
        )
        summary["baseline"] = {
            "best_reward": baseline["best_reward"],
            "best_smiles": baseline["best_smiles"],
            "budget": b_budget,
            "topk": baseline["topk"],
            "elapsed_seconds": baseline["elapsed_seconds"],
        }
        logger.info(
            f"[{benchmark_name}] baseline best={baseline['best_reward']:.4f} "
            f"final={baseline['topk']['final_score']:.4f}"
        )

    # --- MCTS with full budget.
    _set_seed(seed)
    mcts_cfg.oracle_budget = budget
    mcts_cfg.early_stop_reward = early_stop_reward
    logger.info(
        f"[{benchmark_name}] running MCTS ({budget} calls, "
        f"early_stop>={early_stop_reward})"
    )
    t0 = time.time()
    mcts = MCTS(decoder, policy, oracle, mcts_cfg)
    mcts_result = mcts.search()
    mcts_result["elapsed_seconds"] = time.time() - t0
    mcts_result["topk"] = _topk_mean_from_results(
        mcts_result["all_results"], oracle.benchmark
    )
    summary["mcts"] = {
        "best_reward": mcts_result["best_reward"],
        "best_smiles": mcts_result["best_smiles"],
        "topk": mcts_result["topk"],
        "tree_stats": mcts_result["tree_stats"],
        "elapsed_seconds": mcts_result["elapsed_seconds"],
    }
    logger.info(
        f"[{benchmark_name}] mcts best={mcts_result['best_reward']:.4f} "
        f"final={mcts_result['topk']['final_score']:.4f} "
        f"tree={mcts_result['tree_stats']}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run STAMP-MCTS on GuacaMol goal-directed benchmarks."
    )
    parser.add_argument("--model-dir", required=True, help="Trained STAMP LM dir")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"],
        help="Benchmark names (space-separated), or 'all'.",
    )
    parser.add_argument("--version", default="v2", choices=["v1", "v2", "trivial"])
    parser.add_argument("--budget", type=int, default=10000,
                        help="Oracle budget per benchmark for MCTS. "
                             "GuacaMol standard = 10000; MolRL-MGPT uses up to "
                             "5000 iter * 4 agents * 256 = ~5.1M with early stop.")
    parser.add_argument("--early-stop-reward", type=float, default=1.0,
                        help="Terminate a task once best_reward >= this. "
                             "Set to >1 to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True,
                        help="Output JSON path with all per-benchmark results")

    parser.add_argument("--top-k", type=int, default=20,
                        help="MCTS top-k action proposals per node")
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--k-pw", type=float, default=3.0)
    parser.add_argument("--alpha-pw", type=float, default=0.5)
    parser.add_argument("--max-rollout-steps", type=int, default=16)

    parser.add_argument("--proposal-samples", type=int, default=48)
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--proposal-top-k", type=int, default=50)
    parser.add_argument("--rollout-temperature", type=float, default=0.9)
    parser.add_argument("--rollout-top-k", type=int, default=50)

    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run an LM-only random-sampling baseline")
    parser.add_argument("--baseline-fraction", type=float, default=1.0,
                        help="Baseline budget fraction (of --budget). "
                             "1.0 = same budget; 0.5 = half.")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading STAMP AR policy from {args.model_dir}")
    decoder = STAMPDecoderImpl()
    policy = ARPolicyModel(
        args.model_dir,
        device=args.device,
        proposal_samples=args.proposal_samples,
        proposal_temperature=args.proposal_temperature,
        proposal_top_k=args.proposal_top_k,
        rollout_temperature=args.rollout_temperature,
        rollout_top_k=args.rollout_top_k,
    )

    benchmarks = _resolve_benchmarks(args.benchmarks, args.version)
    logger.info(f"Running {len(benchmarks)} benchmarks: {benchmarks}")

    mcts_cfg = MCTSConfig(
        c_puct=args.c_puct,
        top_k=args.top_k,
        k_pw=args.k_pw,
        alpha_pw=args.alpha_pw,
        max_rollout_steps=args.max_rollout_steps,
        oracle_budget=args.budget,
        log_every=max(50, args.budget // 20),
    )

    per_benchmark: List[Dict[str, Any]] = []
    t_all = time.time()
    for name in benchmarks:
        res = run_single_benchmark(
            name,
            decoder=decoder,
            policy=policy,
            version=args.version,
            budget=args.budget,
            mcts_cfg=mcts_cfg,
            run_baseline=args.run_baseline,
            baseline_fraction=args.baseline_fraction,
            seed=args.seed,
            early_stop_reward=args.early_stop_reward,
        )
        per_benchmark.append(res)
        # Incremental save so a later crash doesn't lose prior benchmarks.
        output_path.write_text(
            json.dumps({
                "config": vars(args),
                "benchmarks": per_benchmark,
            }, indent=2, default=float),
            encoding="utf-8",
        )
        logger.info(f"[{name}] written to {output_path}")

    elapsed = time.time() - t_all
    mcts_finals = [r["mcts"]["topk"]["final_score"] for r in per_benchmark]
    summary = {
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "benchmarks": per_benchmark,
        "summary": {
            "mcts_mean_final_score": float(np.mean(mcts_finals)) if mcts_finals else 0.0,
            "n_benchmarks": len(per_benchmark),
        },
    }
    output_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    logger.info(
        f"Done. {len(per_benchmark)} benchmarks in {elapsed:.1f}s. "
        f"Mean final score = {summary['summary']['mcts_mean_final_score']:.4f}. "
        f"Results: {output_path}"
    )


if __name__ == "__main__":
    main()

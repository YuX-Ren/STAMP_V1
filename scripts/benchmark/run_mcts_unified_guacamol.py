#!/usr/bin/env python3
"""Run **unified** STAMP-MCTS (growth + edit actions on one tree) on the
GuacaMol goal-directed suite.

This driver is parallel to ``run_mcts_guacamol.py`` (growth-only baseline)
but swaps the policy for :class:`UnifiedPolicyModel` and the search for
:class:`UnifiedMCTS` so that once a branch reaches a terminal molecule the
tree can continue expanding via AO-GPT-driven subtree edits.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mcts_protype import MCTSConfig  # noqa: E402
from stamp.benchmark.mcts_guacamol import (  # noqa: E402
    ARPolicyModel,
    GuacaMolOracle,
    STAMPDecoderImpl,
    list_guacamol_benchmarks,
)
from stamp.benchmark.lm import load_trained_model  # noqa: E402
from stamp.benchmark.mcts_unified.edit_policy import AOEditPolicy  # noqa: E402
from stamp.benchmark.mcts_unified.memory import MoleculeMemory  # noqa: E402
from stamp.benchmark.mcts_unified.subtree_pool import SubtreePool  # noqa: E402
from stamp.benchmark.mcts_unified.unified_mcts import UnifiedMCTS  # noqa: E402
from stamp.benchmark.mcts_unified.unified_policy import UnifiedPolicyModel  # noqa: E402
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402
from stamp.vocab.principal_subgraph import PrincipalSubgraphTokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topk_mean_from_memory(memory: MoleculeMemory, benchmark) -> Dict[str, Any]:
    spec = benchmark.contribution_specification
    top_counts = sorted({int(c) for c in spec.top_counts})
    scores = sorted(memory.items(), key=lambda x: -x[1])
    top_mols = [{"smiles": s, "score": float(r)} for s, r in scores[: max(top_counts) if top_counts else 10]]
    out: Dict[str, float] = {}
    for k in top_counts:
        sel = [r for _, r in scores[:k]]
        padded = sel + [0.0] * (k - len(sel))
        out[str(k)] = float(np.mean(padded)) if padded else 0.0
    final = float(np.mean(list(out.values()))) if out else 0.0
    return {
        "final_score": final,
        "top_k_means": out,
        "top_molecules": top_mols,
        "unique_valid_count": len(memory),
    }


def run_one_benchmark(
    name: str,
    *,
    decoder: STAMPDecoderImpl,
    policy: UnifiedPolicyModel,
    version: str,
    budget: int,
    mcts_cfg: MCTSConfig,
    seed: int,
    early_stop_reward: float,
) -> Dict[str, Any]:
    logger.info(f"[{name}] oracle")
    oracle = GuacaMolOracle(name, version=version)

    _set_seed(seed)
    mcts_cfg.oracle_budget = budget
    mcts_cfg.early_stop_reward = early_stop_reward
    memory = MoleculeMemory()
    # Share memory with the policy so it can dedup candidates at proposal time.
    policy.memory = memory
    # Reset pool so cross-benchmark contamination doesn't bias scores.
    policy.pool = type(policy.pool)(
        per_sig_cap=policy.pool.per_sig_cap,
        global_cap=policy.pool.global_cap,
        min_threshold=policy.pool.min_threshold,
        threshold_fraction=policy.pool.threshold_fraction,
    )
    mcts = UnifiedMCTS(
        decoder, policy, oracle, mcts_cfg,
        memory=memory,
        pool=policy.pool,
    )
    t0 = time.time()
    res = mcts.search()
    elapsed = time.time() - t0

    top = _topk_mean_from_memory(memory, oracle.benchmark)
    logger.info(
        f"[{name}] best={res['best_reward']:.4f} final={top['final_score']:.4f} "
        f"oracle={res['oracle_calls']} iters={res['iterations']} "
        f"mem_hits={res['memory_hits']} "
        f"actions G/D/T={res['n_growth_actions']}/{res['n_diffusion_actions']}/{res['n_transplant_actions']} "
        f"pool={res['pool_stats']['n_entries_total']}({res['pool_stats']['n_signatures']}sig) "
        f"tree_nodes={res['tree_stats']['total_nodes']} max_depth={res['tree_stats']['max_depth']} "
        f"elapsed={elapsed:.1f}s"
    )

    return {
        "benchmark": name,
        "budget": budget,
        "seed": seed,
        "best_reward": res["best_reward"],
        "best_smiles": res["best_smiles"],
        "topk": top,
        "tree_stats": res["tree_stats"],
        "oracle_calls": res["oracle_calls"],
        "iterations": res["iterations"],
        "memory_hits": res["memory_hits"],
        "n_growth_actions": res["n_growth_actions"],
        "n_diffusion_actions": res["n_diffusion_actions"],
        "n_transplant_actions": res["n_transplant_actions"],
        "n_harvested": res["n_harvested"],
        "pool_stats": res["pool_stats"],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--growth-model-dir", required=True, help="AR LM dir (growth policy)")
    p.add_argument("--edit-model-dir", required=True, help="AO-GPT LM dir (edit policy)")
    p.add_argument("--stamp-vocab", default=None,
                   help="Path to stamp_vocab.txt (PS-VAE). Default: "
                        "<growth-model-dir>/stamp_vocab.txt")
    p.add_argument("--device", default=None)
    p.add_argument("--benchmarks", nargs="+", default=["all"])
    p.add_argument("--version", default="v2", choices=["v1", "v2", "trivial"])
    p.add_argument("--budget", type=int, default=5000)
    p.add_argument("--early-stop-reward", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)

    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--c-puct", type=float, default=1.4)
    p.add_argument("--k-pw", type=float, default=3.0)
    p.add_argument("--alpha-pw", type=float, default=0.5)
    p.add_argument("--max-rollout-steps", type=int, default=16)

    p.add_argument("--edit-max-span-len", type=int, default=8)
    p.add_argument("--edit-temperature", type=float, default=1.0)
    p.add_argument("--edit-top-k", type=int, default=50)
    p.add_argument("--diffusion-samples-per-span", type=int, default=2)
    p.add_argument("--transplants-per-span", type=int, default=5)
    p.add_argument("--size-delta", type=int, default=4)
    p.add_argument("--prior-softmax-tau", type=float, default=0.5)

    p.add_argument("--pool-per-sig-cap", type=int, default=50)
    p.add_argument("--pool-global-cap", type=int, default=5000)
    p.add_argument("--pool-min-threshold", type=float, default=0.3)
    p.add_argument("--pool-threshold-fraction", type=float, default=0.7)

    args = p.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # STAMP encoder with PS-VAE tokenizer for canonical re-encoding.
    vocab_file = args.stamp_vocab or str(Path(args.growth_model_dir) / "stamp_vocab.txt")
    tokenizer = PrincipalSubgraphTokenizer(vocab_file)
    encoder = STAMPEncoder(motif_tokenizer=tokenizer)
    decoder = STAMPDecoderImpl(encoder=encoder)

    logger.info(f"Loading growth AR policy from {args.growth_model_dir}")
    growth = ARPolicyModel(args.growth_model_dir, device=args.device)
    logger.info(f"Loading edit AO policy from {args.edit_model_dir}")
    edit_model, edit_vocab, _, dev = load_trained_model(args.edit_model_dir, device=args.device)
    edit = AOEditPolicy(
        edit_model, edit_vocab, device=dev,
        temperature=args.edit_temperature, top_k=args.edit_top_k,
    )
    pool = SubtreePool(
        per_sig_cap=args.pool_per_sig_cap,
        global_cap=args.pool_global_cap,
        min_threshold=args.pool_min_threshold,
        threshold_fraction=args.pool_threshold_fraction,
    )
    policy = UnifiedPolicyModel(
        growth=growth,
        edit=edit,
        encoder=encoder,
        pool=pool,
        edit_max_span_len=args.edit_max_span_len,
        edit_temperature=args.edit_temperature,
        edit_top_k=args.edit_top_k,
        diffusion_samples_per_span=args.diffusion_samples_per_span,
        transplants_per_span=args.transplants_per_span,
        size_delta=args.size_delta,
        prior_softmax_temperature=args.prior_softmax_tau,
    )

    benchmarks = (
        list_guacamol_benchmarks(version=args.version)
        if len(args.benchmarks) == 1 and args.benchmarks[0].lower() == "all"
        else list(args.benchmarks)
    )
    mcts_cfg = MCTSConfig(
        c_puct=args.c_puct, top_k=args.top_k, k_pw=args.k_pw,
        alpha_pw=args.alpha_pw, max_rollout_steps=args.max_rollout_steps,
        oracle_budget=args.budget, log_every=max(100, args.budget // 20),
    )

    results: List[Dict[str, Any]] = []
    for name in benchmarks:
        r = run_one_benchmark(
            name,
            decoder=decoder,
            policy=policy,
            version=args.version,
            budget=args.budget,
            mcts_cfg=mcts_cfg,
            seed=args.seed,
            early_stop_reward=args.early_stop_reward,
        )
        results.append(r)
        out_path.write_text(
            json.dumps({"config": vars(args), "benchmarks": results}, indent=2, default=float),
            encoding="utf-8",
        )

    finals = [r["topk"]["final_score"] for r in results]
    summary = {
        "config": vars(args),
        "benchmarks": results,
        "summary": {
            "mean_final_score": float(np.mean(finals)) if finals else 0.0,
            "n_benchmarks": len(results),
        },
    }
    out_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    logger.info(f"Mean final = {summary['summary']['mean_final_score']:.4f}  → {out_path}")


if __name__ == "__main__":
    main()

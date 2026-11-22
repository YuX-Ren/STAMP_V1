#!/usr/bin/env python3
r"""Run STAMP PMO with a shared configuration, no surrogate and no niching.

The selected candidate is the first valid, unscored proposal. Fragment scores,
max-update selection, subtree crossover and mutation are retained from the
supplied optimizer snapshot. No tuning-result reference or
per-task winning configuration is read.

Example:
  PYTHONHASHSEED=0 python scripts/benchmark/run_pmo_baseline.py \
    --task qed --seed 1 \
    --output results/benchmark/pmo_no_surrogate/qed/s1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = Path(__file__).resolve().parent / "configs/pmo_no_surrogate.json"


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text())
    if set(profile) != {"name", "oracle_budget", "generator", "optimizer"}:
        raise ValueError("Profile requires name, oracle_budget, generator and optimizer")
    cfg = profile["optimizer"]
    required = {"use_surrogate": False, "gp_brood_size": 1,
                "surrogate_k_candidates": 1, "target_aware": False,
                "use_scaffold_diversity": False, "novelty_weight": 0.0,
                "use_scaffold_niching": False,
                "tree_crossover_prob": 0.0, "mutation_mode": "ao_gpt"}
    for name, expected in required.items():
        if name not in cfg or cfg[name] != expected:
            raise ValueError(f"No-surrogate baseline requires {name}={expected!r}")
    for name in ("population_size", "molecule_pool_size", "max_iterations",
                 "n_attach_tries", "gp_max_tries", "gp_min_pool_size", "log_every"):
        if type(cfg.get(name)) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(profile["oracle_budget"]) is not int or profile["oracle_budget"] <= 0:
        raise ValueError("oracle_budget must be a positive integer")
    if type(cfg.get("warmup")) is not int or cfg["warmup"] < 0:
        raise ValueError("warmup must be a nonnegative integer")
    for name in ("mutation_prob",):
        value = cfg.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    generator = profile["generator"]
    required_generator = {"temperature", "top_k", "motif_strict_vocab"}
    if not required_generator.issubset(generator) or generator.keys() - required_generator - {"top_p"}:
        raise ValueError("Generator requires temperature, top_k and motif_strict_vocab; top_p is optional")
    if not math.isfinite(generator["temperature"]) or generator["temperature"] <= 0:
        raise ValueError("Generator temperature must be positive and finite")
    if type(generator["top_k"]) is not int or generator["top_k"] < 0:
        raise ValueError("Generator top_k must be a nonnegative integer")
    top_p = generator.get("top_p", 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("Generator top_p must be in (0, 1]")
    if type(generator["motif_strict_vocab"]) is not bool:
        raise ValueError("motif_strict_vocab must be a boolean")
    return profile


def make_optimizer_class(base_class: type, chemistry: Any) -> type:
    """Reuse GenMol's reward updates, replacing brood generation with one proposal."""

    class BaselineOptimizer(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            cfg = kwargs["config"]
            if cfg.use_surrogate or cfg.gp_brood_size != 1 or cfg.surrogate_k_candidates != 1:
                raise ValueError("Baseline requires no surrogate and a single-candidate brood")
            if cfg.use_scaffold_niching:
                raise ValueError("Baseline requires scaffold niching disabled")
            super().__init__(*args, **kwargs)
            if self._surrogate_ranker is not None:
                raise RuntimeError("Baseline unexpectedly constructed a surrogate")
            if self._niches:
                raise RuntimeError("Baseline unexpectedly initialized scaffold niches")
            self._canon_seen: set[str] = set()
            self._stamp_cache.parsed = functools.lru_cache(maxsize=4096)(
                self._stamp_cache.parsed
            )
            self.baseline_stats = {
                "candidate_attempts": 0, "invalid_candidates": 0,
                "history_duplicates": 0, "selected_candidates": 0,
                "empty_iterations": 0,
            }

        @functools.lru_cache(maxsize=8192)
        def _canonical_key(self, smiles: str) -> str | None:
            mol = chemistry.MolFromSmiles(smiles)
            if mol is None:
                return None
            return chemistry.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        def _generate(self) -> str | None:
            for _ in range(self.config.n_attach_tries):
                self.baseline_stats["candidate_attempts"] += 1
                candidate = self._generate_one()
                if candidate is None:
                    continue
                key = self._canonical_key(candidate)
                if key is None:
                    self.baseline_stats["invalid_candidates"] += 1
                    continue
                if key in self._canon_seen:
                    self.baseline_stats["history_duplicates"] += 1
                    continue
                self.baseline_stats["selected_candidates"] += 1
                return key
            self.baseline_stats["empty_iterations"] += 1
            return None

        def _already_scored(self, smiles: str) -> bool:
            key = self._canonical_key(smiles)
            if key is None or key in self._canon_seen:
                return True
            self._canon_seen.add(key)
            return False

        def _update_niches(self, smiles: str, score: float) -> None:
            raise RuntimeError("Scaffold niche updates are disabled in this baseline")

    return BaselineOptimizer


def validate_encoder_runtime(encoder: Any, chemistry: Any) -> None:
    if not hasattr(chemistry.SmilesWriteParams(), "ignoreAtomMapNumbers"):
        raise RuntimeError("RDKit 2025.09.4 is required for the motif atom-order option")
    for smiles in ("CCO", "CC(C)O", "CCCc1ccc(OCC)cc1", "C[C@H](O)Cl"):
        original = chemistry.MolFromSmiles(smiles)
        decoded = encoder.decode_stamp_to_mol(encoder.encode_smiles(smiles))
        if decoded is None or chemistry.MolToSmiles(original, isomericSmiles=True) != (
            chemistry.MolToSmiles(decoded, isomericSmiles=True)
        ):
            raise RuntimeError(f"Encoder runtime roundtrip failed: {smiles}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(output: Path) -> list[Path]:
    return [output, output.with_suffix(".meta.json"),
            output.with_suffix(".oracle.jsonl"), output.with_suffix(".json.tmp")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Optimizer source root (defaults to this source checkout)")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints/ao_generation")
    parser.add_argument("--vocab-dir", type=Path, default=ROOT / "datasets/main/data/pmo/fragment_vocab")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, help="Override oracle cap for smoke tests")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("budget", "max_iterations", "threads"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    profile = load_profile(args.config)
    output = args.output.resolve()
    for path in output_paths(output):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing run artifact: {path}")
    # cut() produces sets, so Python's hash seed must be fixed at process startup.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["STAMP_INFERENCE_FP32"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    snapshot = args.source_root.resolve()
    if not (snapshot / "stamp/benchmark/mcts_unified/genmol_faithful.py").is_file():
        raise SystemExit(f"Missing optimizer source: {snapshot}")
    model_dir = args.model_dir.resolve()
    vocab_dir = args.vocab_dir.resolve()
    inputs = [model_dir / name for name in ("model.pt", "vocab.json", "stamp_vocab.txt", "train_meta.json")]
    inputs.append(vocab_dir / f"{args.task}.csv")
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"Missing baseline input: {path}")
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import rdkit
    import torch
    from rdkit import Chem
    from pmo_sampling import load_pmo_model, make_edit_policy
    from stamp.benchmark.mcts_unified.genmol_faithful import GenMolFaithfulConfig, GenMolFaithfulOptimizer
    from stamp.benchmark.mcts_unified.pmo_oracle import PMO_TASKS, PMOOracle, auc_top_k_from_log
    from stamp.core.pmo_codec import PMOEncoder as STAMPEncoder
    from stamp.vocab.principal_subgraph import PrincipalSubgraphTokenizer

    if args.task not in PMO_TASKS:
        raise SystemExit(f"Unknown PMO task: {args.task}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    budget = args.budget if args.budget is not None else profile["oracle_budget"]
    values = dict(profile["optimizer"])
    if args.max_iterations is not None:
        values["max_iterations"] = args.max_iterations
    fields = {field.name for field in dataclasses.fields(GenMolFaithfulConfig)}
    unknown = values.keys() - fields
    if unknown:
        raise SystemExit(f"Optimizer source does not support baseline fields: {sorted(unknown)}")
    config = GenMolFaithfulConfig(**values)
    settings = profile["generator"]
    encoder = STAMPEncoder(
        motif_tokenizer=PrincipalSubgraphTokenizer(str(model_dir / "stamp_vocab.txt")),
        motif_strict_vocab=settings["motif_strict_vocab"],
    )
    validate_encoder_runtime(encoder, Chem)
    started = time.time()
    model, vocab, device, model_metadata = load_pmo_model(model_dir, device=args.device)
    edit = make_edit_policy(model, vocab, device, settings)
    oracle = PMOOracle(args.task, max_calls=budget)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix(".oracle.jsonl").open("x", buffering=1)

    def traced_oracle(smiles: str) -> float:
        if oracle.n_calls >= budget:
            raise RuntimeError("Attempted to exceed the oracle budget")
        previous = oracle.n_calls
        reward = oracle(smiles)
        if oracle.n_calls != previous + 1:
            raise RuntimeError("Oracle did not record the selected candidate")
        trace.write(json.dumps({"call": oracle.n_calls, "smiles": smiles, "reward": reward,
                                "op": optimizer._last_op, "parents": optimizer._last_parents}) + "\n")
        return reward

    optimizer_type = make_optimizer_class(GenMolFaithfulOptimizer, Chem)
    try:
        optimizer = optimizer_type(task=args.task, vocab_dir=str(vocab_dir), encoder=encoder,
                                   edit_policy=edit, oracle=traced_oracle, config=config)
        source_hashes = {
            str(path.relative_to(snapshot)): sha256(path)
            for path in sorted((snapshot / "stamp").rglob("*.py")) if "__pycache__" not in path.parts
        }
        metadata = {
            "task": args.task, "seed": args.seed, "variant": "no_surrogate", "budget": budget,
            "profile_name": profile["name"], "profile": profile, "source_root": str(snapshot),
            "config": dataclasses.asdict(config), "generator": settings,
            "candidate_selection": "first_valid_unscored_candidate", "surrogate_enabled": False,
            "niching_enabled": False,
            "fragment_scoring": "CSV initialization; max-update from true molecule oracle reward; top-K retention",
            "task_size_policy": "optimizer task-family molecule and initial-fragment size ranges",
            "source_files_sha256": source_hashes,
            "source_snapshot_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "input_sha256": {str(path): sha256(path) for path in inputs},
            "profile_sha256": sha256(args.config), "runner_sha256": sha256(Path(__file__)),
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "versions": {"python": platform.python_version(), "rdkit": rdkit.__version__,
                         "torch": torch.__version__, "numpy": np.__version__},
            "model_metadata": model_metadata,
            "sampling_runtime_sha256": sha256(Path(__file__).with_name("pmo_sampling.py")),
        }
        with output.with_suffix(".meta.json").open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        logging.info("START baseline=%s task=%s seed=%s budget=%s surrogate=off niching=off brood=1",
                     profile["name"], args.task, args.seed, budget)
        result = optimizer.run(budget)
    finally:
        trace.close()
    metric = auc_top_k_from_log(oracle._log, k=10, max_calls=budget)
    if metric["n_oracle_calls"] != result["n_calls"] or metric["n_unique_smiles"] != result["n_calls"]:
        raise RuntimeError("Baseline oracle accounting or canonical deduplication mismatch")
    if optimizer._surrogate_ranker is not None or optimizer.n_surrogate_picks != 0:
        raise RuntimeError("Surrogate unexpectedly active during baseline execution")
    if optimizer._niches or optimizer.config.use_scaffold_niching:
        raise RuntimeError("Scaffold niching unexpectedly active during baseline execution")
    payload = {
        **metadata, "metric": metric, "summary": {"mean_auc_top_10": metric["auc_top_k"]},
        "diagnostics": {key: value for key, value in result.items() if key not in {"oracle_log", "trajectory"}},
        "baseline_stats": optimizer.baseline_stats, "n_surrogate_picks": 0,
        "n_scaffold_niches": 0,
        "budget_exhausted": oracle.n_calls == budget,
        "total_elapsed_seconds": time.time() - started,
        "sampling_diagnostics": getattr(edit, "stats", {}),
    }
    with output.with_suffix(".json.tmp").open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    output.with_suffix(".json.tmp").rename(output)
    logging.info("DONE AUC=%.6f calls=%s/%s selected=%s surrogate_picks=0 elapsed=%.1fs output=%s",
                 metric["auc_top_k"], oracle.n_calls, budget,
                 optimizer.baseline_stats["selected_candidates"], payload["total_elapsed_seconds"], output)


if __name__ == "__main__":
    main()

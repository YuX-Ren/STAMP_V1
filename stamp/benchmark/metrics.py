from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED

from .common import canonicalize_smiles

_PROP_WORKER_SA = None
_QUALITY_SA_THRESHOLD = 4.0
_QUALITY_QED_THRESHOLD = 0.6


def _safe_float(v: Sequence[float], fn, default: float = 0.0) -> float:
    if not v:
        return default
    return float(fn(v))


def _load_sa_scorer():
    try:
        from rdkit.Chem import RDConfig
        import sys

        sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_dir) not in sys.path:
            sys.path.append(str(sa_dir))
        import sascorer  # type: ignore

        return sascorer
    except Exception:
        return None


def _init_property_worker(enable_sa: bool) -> None:
    global _PROP_WORKER_SA
    _PROP_WORKER_SA = _load_sa_scorer() if enable_sa else None


def _canonicalize_worker(payload: Tuple[str, bool]) -> Optional[str]:
    s, require_single_component = payload
    return canonicalize_smiles(s, require_single_component=require_single_component)


def _property_worker(smiles: str) -> Optional[Tuple[float, float, float, Optional[float]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    qed = float(QED.qed(mol))
    sa_val: Optional[float] = None
    if _PROP_WORKER_SA is not None:
        try:
            sa_val = float(_PROP_WORKER_SA.calculateScore(mol))
        except Exception:
            sa_val = None
    return (mw, logp, qed, sa_val)


def _maybe_parallel_canonicalize(
    smiles_list: Sequence[str],
    require_single_component: bool,
    num_workers: int,
    chunksize: int,
) -> list[Optional[str]]:
    if num_workers <= 1:
        return [
            canonicalize_smiles(s, require_single_component=require_single_component)
            for s in smiles_list
        ]
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            mapped = pool.map(
                _canonicalize_worker,
                ((s, require_single_component) for s in smiles_list),
                chunksize=max(1, chunksize),
            )
            return list(mapped)
    except Exception:
        return [
            canonicalize_smiles(s, require_single_component=require_single_component)
            for s in smiles_list
        ]


def _calc_properties(
    smiles_list: Sequence[str],
    num_workers: int = 1,
    chunksize: int = 256,
) -> Dict[str, list[float]]:
    sa = _load_sa_scorer()
    props = {"MW": [], "logP": [], "QED": [], "SA": []}

    if num_workers <= 1:
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            props["MW"].append(float(Descriptors.MolWt(mol)))
            props["logP"].append(float(Crippen.MolLogP(mol)))
            props["QED"].append(float(QED.qed(mol)))
            if sa is not None:
                try:
                    props["SA"].append(float(sa.calculateScore(mol)))
                except Exception:
                    pass
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_property_worker,
                initargs=(sa is not None,),
            ) as pool:
                mapped = pool.map(
                    _property_worker,
                    smiles_list,
                    chunksize=max(1, chunksize),
                )
                for item in mapped:
                    if item is None:
                        continue
                    mw, logp, qed, sa_val = item
                    props["MW"].append(mw)
                    props["logP"].append(logp)
                    props["QED"].append(qed)
                    if sa is not None and sa_val is not None:
                        props["SA"].append(sa_val)
        except Exception:
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                props["MW"].append(float(Descriptors.MolWt(mol)))
                props["logP"].append(float(Crippen.MolLogP(mol)))
                props["QED"].append(float(QED.qed(mol)))
                if sa is not None:
                    try:
                        props["SA"].append(float(sa.calculateScore(mol)))
                    except Exception:
                        pass

    if sa is None:
        props.pop("SA", None)
    return props


def _calc_quality(
    unique_valid_smiles: Sequence[str],
    n_generated: int,
    sa_threshold: float = _QUALITY_SA_THRESHOLD,
    qed_threshold: float = _QUALITY_QED_THRESHOLD,
) -> Dict[str, Any]:
    sa = _load_sa_scorer()
    quality_metrics: Dict[str, Any] = {
        "quality_sa_threshold": float(sa_threshold),
        "quality_qed_threshold": float(qed_threshold),
    }
    if sa is None:
        quality_metrics["n_quality"] = 0
        quality_metrics["quality"] = float("nan")
        return quality_metrics

    n_quality = 0
    for s in unique_valid_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            sa_val = float(sa.calculateScore(mol))
        except Exception:
            continue
        qed_val = float(QED.qed(mol))
        if sa_val <= sa_threshold and qed_val >= qed_threshold:
            n_quality += 1

    quality_metrics["n_quality"] = n_quality
    quality_metrics["quality"] = n_quality / max(1, n_generated)
    return quality_metrics


def _quantile_l1_distance(a: Sequence[float], b: Sequence[float], n_quantiles: int = 21) -> float:
    if not a or not b:
        return float("nan")
    qa = np.quantile(np.asarray(a, dtype=float), np.linspace(0, 1, n_quantiles))
    qb = np.quantile(np.asarray(b, dtype=float), np.linspace(0, 1, n_quantiles))
    return float(np.mean(np.abs(qa - qb)))


def _internal_diversity(smiles_list: Sequence[str], max_pairs: int = 5000, seed: int = 42) -> float:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    rnd = random.Random(seed)

    n = len(fps)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return 0.0

    if max_pairs <= 0:
        max_pairs = 1
    target_pairs = min(max_pairs, total_pairs)

    pairs: list[tuple[int, int]] = []
    if total_pairs <= target_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < target_pairs:
            i = rnd.randrange(n)
            j = rnd.randrange(n - 1)
            if j >= i:
                j += 1
            if i > j:
                i, j = j, i
            seen.add((i, j))
        pairs = list(seen)

    sims = [float(DataStructs.TanimotoSimilarity(fps[i], fps[j])) for i, j in pairs]
    if not sims:
        return 0.0
    return 1.0 - float(np.mean(sims))


def evaluate_generated_smiles(
    generated_smiles: Sequence[str],
    train_smiles: Sequence[str],
    reference_smiles: Optional[Sequence[str]] = None,
    generated_canonical: Optional[Sequence[Optional[str]]] = None,
    assume_train_canonical: bool = False,
    reference_max_samples: Optional[int] = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> Dict[str, Any]:
    if eval_workers <= 0:
        eval_workers = max(1, os.cpu_count() or 1)
    if generated_canonical is None:
        canonicalized = _maybe_parallel_canonicalize(
            generated_smiles,
            require_single_component=True,
            num_workers=eval_workers,
            chunksize=eval_chunksize,
        )
    else:
        canonicalized = list(generated_canonical)

    valid = [c for c in canonicalized if c is not None]

    n_total = len(generated_smiles)
    n_valid = len(valid)
    unique_valid = sorted(set(valid))

    if assume_train_canonical:
        train_set = {s for s in train_smiles if s}
    else:
        # Deduplicate first to reduce canonicalization overhead for very large train sets.
        uniq_train = list({s for s in train_smiles if s})
        train_canon = _maybe_parallel_canonicalize(
            uniq_train,
            require_single_component=True,
            num_workers=eval_workers,
            chunksize=eval_chunksize,
        )
        train_set = {c for c in train_canon if c is not None}
    novel = [s for s in unique_valid if s not in train_set]

    metrics: Dict[str, Any] = {
        "n_generated": n_total,
        "n_valid": n_valid,
        "validity": n_valid / max(1, n_total),
        "n_unique_valid": len(unique_valid),
        "uniqueness": len(unique_valid) / max(1, n_valid),
        "n_novel": len(novel),
        "novelty": len(novel) / max(1, len(unique_valid)),
        "internal_diversity": _internal_diversity(unique_valid),
    }
    metrics.update(_calc_quality(unique_valid, n_generated=n_total))

    gen_props = _calc_properties(
        unique_valid,
        num_workers=eval_workers,
        chunksize=eval_chunksize,
    )
    ref_source = list(reference_smiles or train_smiles)
    if reference_max_samples is not None and reference_max_samples > 0 and len(ref_source) > reference_max_samples:
        rnd = random.Random(42)
        ref_source = rnd.sample(ref_source, reference_max_samples)
    ref_props = _calc_properties(
        ref_source,
        num_workers=eval_workers,
        chunksize=eval_chunksize,
    )

    dist = {}
    for key, vals in gen_props.items():
        dist[key] = _quantile_l1_distance(vals, ref_props.get(key, []))
    metrics["distribution_distance"] = dist

    metrics["generated_properties"] = {
        k: {
            "mean": _safe_float(v, np.mean, default=float("nan")),
            "median": _safe_float(v, np.median, default=float("nan")),
        }
        for k, v in gen_props.items()
    }

    return metrics

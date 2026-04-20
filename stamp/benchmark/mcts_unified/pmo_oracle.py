"""PMO benchmark integration (Practical Molecular Optimization, Gao et al. 2022).

Wraps the ``tdc.Oracle`` family (23 tasks) with the AUC-top-10 metric used
by GenMol and most PMO-era papers.  The metric rewards sample efficiency,
not just final reward — at oracle call ``t``, we compute the top-10 mean
of all molecules scored so far, then integrate across the 10,000-call
budget.

Tasks (``tdc.Oracle`` names, 23 total):

- Binding proxies: ``drd2``, ``gsk3b``, ``jnk3``
- Drug-likeness: ``qed``
- Rediscovery: ``celecoxib_rediscovery``, ``troglitazone_rediscovery``,
  ``thiothixene_rediscovery``
- Similarity: ``albuterol_similarity``, ``mestranol_similarity``
- Median molecules: ``median1``, ``median2``
- MPO: ``osimertinib_mpo``, ``fexofenadine_mpo``, ``ranolazine_mpo``,
  ``perindopril_mpo``, ``amlodipine_mpo``, ``sitagliptin_mpo``,
  ``zaleplon_mpo``
- Structural: ``valsartan_smarts``, ``deco_hop``, ``scaffold_hop``
- Isomers: ``isomers_c7h8n2o2``, ``isomers_c9h10n2o2pf2cl``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mcts_protype import Oracle  # noqa: E402


# The canonical PMO suite used by GenMol / PMO papers.
PMO_TASKS: List[str] = [
    "qed",
    "drd2",
    "gsk3b",
    "jnk3",
    "celecoxib_rediscovery",
    "troglitazone_rediscovery",
    "thiothixene_rediscovery",
    "albuterol_similarity",
    "mestranol_similarity",
    "median1",
    "median2",
    "osimertinib_mpo",
    "fexofenadine_mpo",
    "ranolazine_mpo",
    "perindopril_mpo",
    "amlodipine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    "valsartan_smarts",
    "deco_hop",
    "scaffold_hop",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
]


class PMOOracle(Oracle):
    """Wrap a ``tdc.Oracle`` for use in our MCTS / linear pipelines.

    The TDC oracle returns a float for each SMILES (in [0, 1] for PMO).
    Invalid SMILES return 0.
    """

    def __init__(self, task_name: str, max_calls: int = 10_000) -> None:
        from tdc import Oracle as TDCOracle
        self._task_name = task_name
        self._oracle = TDCOracle(name=task_name)
        self._max_calls = int(max_calls)
        self._n_calls = 0
        # Time-series of (call_idx, smiles, score) — the only stateful
        # accounting we need to compute AUC top-k.
        self._log: List[tuple] = []

    # ------------------------------------------------------------------
    # Oracle protocol
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._task_name

    def __call__(self, smiles: str) -> float:
        if not smiles:
            return 0.0
        try:
            score = float(self._oracle(smiles))
        except Exception:
            return 0.0
        if not np.isfinite(score):
            return 0.0
        score = max(0.0, min(1.0, score))
        self._n_calls += 1
        self._log.append((self._n_calls, smiles, score))
        return score

    # ------------------------------------------------------------------
    # Diagnostics / metric
    # ------------------------------------------------------------------
    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def max_calls(self) -> int:
        return self._max_calls

    def reset(self) -> None:
        self._n_calls = 0
        self._log.clear()


# ---------------------------------------------------------------------------
# AUC top-k metric
# ---------------------------------------------------------------------------

def auc_top_k_from_log(
    log: Sequence[tuple],
    k: int = 10,
    max_calls: int = 10_000,
    dedup_smiles: bool = True,
) -> Dict[str, float]:
    """Compute AUC top-k from a ``[(call_idx, smiles, score), ...]`` log.

    Follows the PMO / GenMol convention::

        AUC_top_k = (1 / max_calls) * Σ_{t=1..max_calls} TopK_mean(buffer_t)

    where ``buffer_t`` is the set of molecules scored in calls ``1..t``.
    If ``dedup_smiles``, only the best score per canonical SMILES counts.
    """
    from rdkit import Chem

    # Build the running top-k buffer.
    best_by_smi: Dict[str, float] = {}
    curve: List[float] = []

    # Sort log by call_idx to be safe (should already be in order).
    sorted_log = sorted(log, key=lambda x: x[0])
    next_call = 1
    buffer_sorted: List[float] = []

    for call_idx, smi, score in sorted_log:
        # Fill any gaps in the call index with the previous top-k mean.
        while next_call < call_idx and next_call <= max_calls:
            top_mean = (
                float(np.mean(buffer_sorted[:k]))
                if buffer_sorted else 0.0
            )
            curve.append(top_mean)
            next_call += 1

        if next_call > max_calls:
            break

        # Update buffer.
        canon = smi
        if dedup_smiles and smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                except Exception:
                    canon = smi
        prev = best_by_smi.get(canon, -1.0)
        if score > prev:
            best_by_smi[canon] = float(score)
            # Rebuild sorted buffer.
            buffer_sorted = sorted(best_by_smi.values(), reverse=True)

        # Record top-k mean at this call.
        top_mean = (
            float(np.mean(buffer_sorted[:k]))
            if buffer_sorted else 0.0
        )
        curve.append(top_mean)
        next_call += 1

    # Pad the remainder of the budget with the last value (GenMol convention).
    if curve:
        last = curve[-1]
    else:
        last = 0.0
    while next_call <= max_calls:
        curve.append(last)
        next_call += 1
    curve = curve[:max_calls]

    auc = float(np.mean(curve)) if curve else 0.0
    return {
        "auc_top_k": auc,
        "top_k": k,
        "max_calls": max_calls,
        "n_oracle_calls": len(sorted_log),
        "n_unique_smiles": len(best_by_smi),
        "final_top_k_mean": float(np.mean(buffer_sorted[:k])) if buffer_sorted else 0.0,
        "best_score": float(buffer_sorted[0]) if buffer_sorted else 0.0,
    }

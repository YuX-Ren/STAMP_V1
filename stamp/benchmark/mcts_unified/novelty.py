"""Novelty-aware population update."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def _fp(smi: str, radius: int = 2, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


@dataclass
class NoveltyScorer:
    k_neighbors: int = 10
    radius: int = 2
    n_bits: int = 1024

    def novelty_bonuses(self, frags: Sequence[str]) -> List[float]:
        fps = [_fp(f, radius=self.radius, n_bits=self.n_bits) for f in frags]
        out: List[float] = []
        for i, fp_i in enumerate(fps):
            if fp_i is None:
                out.append(0.0)
                continue
            sims: List[float] = []
            for j, fp_j in enumerate(fps):
                if i == j or fp_j is None:
                    continue
                sims.append(float(DataStructs.TanimotoSimilarity(fp_i, fp_j)))
            if not sims:
                out.append(1.0)
                continue
            sims.sort(reverse=True)
            topk = sims[: self.k_neighbors]
            mean_sim = sum(topk) / len(topk)
            out.append(max(0.0, 1.0 - mean_sim))
        return out


def rank_by_composite(
    population: List[Tuple[float, str]],
    novelty_weight: float,
    scorer: NoveltyScorer,
    *,
    score_gated: bool = True,
    pop_gap_tau: float = 0.4,
) -> List[Tuple[float, str]]:
    if novelty_weight <= 0 or len(population) < 2:
        return sorted(population, reverse=True)
    frags = [f for _, f in population]
    novelty = scorer.novelty_bonuses(frags)
    scores = [p[0] for p in population]
    try:
        med = statistics.median(scores)
    except statistics.StatisticsError:
        med = 0.0
    gap = max(scores) - med
    gap_factor = max(0.0, 1.0 - gap / max(pop_gap_tau, 1e-6))

    def _composite(i: int) -> float:
        s = population[i][0]
        nov = novelty[i]
        frag_gate = max(0.0, 1.0 - s) if score_gated else 1.0
        return s + novelty_weight * gap_factor * frag_gate * nov

    indexed = list(range(len(population)))
    indexed.sort(key=lambda i: -_composite(i))
    return [population[i] for i in indexed]

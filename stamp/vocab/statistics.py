from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .stamp_vocab_types import CandidateMotif, CandidateStat, BuildConfig


def collect_motif_statistics(
    candidates_per_mol: Iterable[List[CandidateMotif]],
) -> Dict[Tuple[str, str], CandidateStat]:
    stats: Dict[Tuple[str, str], CandidateStat] = {}

    for candidates in candidates_per_mol:
        seen_in_mol = set()
        for cand in candidates:
            key = (cand.internal_subgraph_hash, cand.schema_hash)
            if key not in stats:
                stats[key] = CandidateStat(
                    core_hash=cand.internal_subgraph_hash,
                    schema_hash=cand.schema_hash,
                    canonical_smarts=cand.canonical_smarts,
                    atom_count=len(cand.atom_indices),
                    bond_count=cand.bond_count,
                    port_definition=cand.ports,
                )
            st = stats[key]
            st.occ_count += 1
            st.compression_gain_sum += cand.compression_gain
            st.tree_gain_sum += cand.tree_gain
            st.recoverability_sum += cand.recoverability_score
            st.ambiguity_sum += cand.ambiguity_penalty
            seen_in_mol.add(key)

        for key in seen_in_mol:
            stats[key].mol_coverage += 1

    return stats


def score_candidate_stats(stats: Dict[Tuple[str, str], CandidateStat], cfg: BuildConfig) -> None:
    for st in stats.values():
        freq = math.log1p(st.occ_count) + cfg.lambda_cov * math.log1p(st.mol_coverage)
        avg_comp = st.compression_gain_sum / max(1, st.occ_count)
        avg_tree = st.tree_gain_sum / max(1, st.occ_count)
        avg_rec = st.recoverability_sum / max(1, st.occ_count)
        avg_amb = st.ambiguity_sum / max(1, st.occ_count)
        st.total_score = (
            cfg.alpha * freq
            + cfg.beta * avg_comp
            + cfg.gamma * avg_tree
            + cfg.delta * avg_rec
            - cfg.eta * avg_amb
        )


def aggregate_core_stats(
    cand_stats: Dict[Tuple[str, str], CandidateStat]
) -> Dict[str, dict]:
    core_stats: Dict[str, dict] = defaultdict(
        lambda: {
            "frequency": 0,
            "mol_coverage": 0,
            "score_sum": 0.0,
            "schema_freq": defaultdict(int),
            "schema_count": 0,
        }
    )

    for (_, schema_hash), st in cand_stats.items():
        c = core_stats[st.core_hash]
        c["frequency"] += st.occ_count
        c["mol_coverage"] += st.mol_coverage
        c["score_sum"] += st.total_score
        c["schema_freq"][schema_hash] += st.occ_count
        c["schema_count"] += 1

    return core_stats

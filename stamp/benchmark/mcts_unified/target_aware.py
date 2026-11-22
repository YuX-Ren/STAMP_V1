"""Target-aware augmentations for GenMol-faithful PMO optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PMO_TARGETS: Dict[str, str] = {
    "celecoxib_rediscovery": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "troglitazone_rediscovery": "Cc1c(C)c2c(c(C)c1O)CC[C@@](C)(COc1ccc(C[C@@H]3SC(=O)NC3=O)cc1)O2",
    "thiothixene_rediscovery": "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3/C(=C\\CCN4CCN(C)CC4)c2c1",
    "albuterol_similarity": "CC(C)(C)NC[C@H](O)c1ccc(O)c(CO)c1",
    "mestranol_similarity": "COc1ccc2c(c1)CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@H]1CC[C@@]2(C#C)O",
}


def target_for_task(task_name: str) -> Optional[str]:
    return PMO_TARGETS.get(task_name)


def _fp(smi: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


@dataclass
class PlateauTracker:
    stall_calls: int = 500
    tolerance: float = 1e-4
    _best: float = -1.0
    _last_improved_at: int = 0

    def update(self, best_reward: float, n_calls: int) -> None:
        if best_reward > self._best + self.tolerance:
            self._best = float(best_reward)
            self._last_improved_at = int(n_calls)

    def is_plateau(self, n_calls: int) -> bool:
        return (n_calls - self._last_improved_at) >= self.stall_calls


class TargetAwareHelper:
    def __init__(
        self,
        task_name: str,
        *,
        similarity_guard_delta: float = 0.10,
        zero_in_stall_calls: int = 500,
    ) -> None:
        self.task_name = task_name
        self.target_smi = target_for_task(task_name)
        self.target_fp = _fp(self.target_smi) if self.target_smi else None
        self.similarity_guard_delta = float(similarity_guard_delta)
        self.plateau = PlateauTracker(stall_calls=int(zero_in_stall_calls))

    def target_fragments(self) -> List[str]:
        if not self.target_smi:
            return []
        from .genmol_faithful import cut

        return list(cut(self.target_smi))

    @property
    def has_target(self) -> bool:
        return self.target_fp is not None

    def parent_child_tanimoto(self, parent_smi: str, child_smi: str) -> float:
        pfp = _fp(parent_smi)
        cfp = _fp(child_smi)
        if pfp is None or cfp is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(pfp, cfp))

    def accept_mutation(self, parent_smi: str, child_smi: str) -> bool:
        if self.similarity_guard_delta <= 0.0:
            return True
        sim = self.parent_child_tanimoto(parent_smi, child_smi)
        return sim >= (1.0 - self.similarity_guard_delta)

    def note_oracle_call(self, best_reward: float, n_calls: int) -> None:
        self.plateau.update(best_reward, n_calls)

    def in_zero_in_phase(self, n_calls: int) -> bool:
        return self.plateau.is_plateau(n_calls)

"""Canonical-SMILES reward cache shared across the MCTS tree.

Cheap dedup + pseudo-oracle: when an action produces a molecule we've
already scored, return the cached reward instead of spending another
oracle call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from rdkit import Chem


@dataclass
class MoleculeMemory:
    """A simple ``canonical_smiles → reward`` cache.

    Also records (smiles, reward) insertion order so we can pull the
    top-k for downstream reporting.
    """

    _cache: Dict[str, float] = field(default_factory=dict)

    def canon(self, smiles: Optional[str]) -> Optional[str]:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return None

    def lookup(self, smiles: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
        """Return ``(canonical_smiles, cached_reward)``.  Reward is ``None``
        if the molecule has not been scored yet.
        """
        canon = self.canon(smiles)
        if canon is None:
            return None, None
        return canon, self._cache.get(canon)

    def record(self, smiles: Optional[str], reward: float) -> Optional[str]:
        canon = self.canon(smiles)
        if canon is None:
            return None
        prev = self._cache.get(canon, -1.0)
        if reward > prev:
            self._cache[canon] = float(reward)
        return canon

    def __len__(self) -> int:
        return len(self._cache)

    def items(self):
        return self._cache.items()

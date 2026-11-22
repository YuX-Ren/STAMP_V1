"""STAMP-native chemistry filters & ranking modules (Method 11)."""

from __future__ import annotations
from typing import List, Tuple

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from ...core.stamp_codec import STAMPEncoder
from .subtree import parse_terminal_sequence
from .subtree_attribution import SubtreeAttribution
from .transplant import extract_donor_records


# ============================================================================
# Method 11: Subtree-attribution surrogate (ridge regression on bag-of-subtrees)
# ============================================================================

class SurrogateRanker:
    """Wraps SubtreeAttribution to provide candidate ranking during generation.

    - During `_score_and_record`: observe (subtree_features, oracle) pair
    - At generation time: score K candidates by sum of subtree contributions,
      pick the highest-predicted candidate
    """

    def __init__(self, encoder: STAMPEncoder, **attribution_kwargs) -> None:
        self.encoder = encoder
        self.attribution = SubtreeAttribution(**attribution_kwargs)

    def _features(self, smi: str) -> List[Tuple[str, ...]]:
        try:
            stamp = self.encoder.encode_smiles(smi)
            parsed = parse_terminal_sequence(stamp.split(), self.encoder)
            if parsed is None:
                return []
            return [tuple(body) for sig, body, _ in extract_donor_records(parsed, include_root=False, max_span_len=12)]
        except Exception:
            return []

    def observe(self, smi: str, score: float) -> None:
        feats = self._features(smi)
        if feats:
            self.attribution.observe(feats, score)

    def predict(self, smi: str) -> float:
        feats = self._features(smi)
        if not feats:
            return 0.0
        return float(sum(self.attribution.score_subtree(f) for f in feats))

    @property
    def is_ready(self) -> bool:
        return self.attribution.is_ready

    def stats(self) -> dict:
        return self.attribution.stats()

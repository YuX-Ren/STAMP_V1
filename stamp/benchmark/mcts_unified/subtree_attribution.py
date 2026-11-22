"""Cross-molecule subtree attribution via ridge regression.

Given oracle log of (molecule, score) pairs and per-molecule subtree decomposition,
fit a linear model:

    score_i ≈ intercept + Σ_j β_j * x_{ij}

where x_{ij} ∈ {0, 1} indicates whether subtree j (canonical token tuple) appears in
molecule i. β_j is the learned per-subtree contribution, decorrelated across the
co-occurrence matrix.

This gives **real attribution** (not heuristic): β_j answers the question
"what is the average score lift contributed by subtree j, controlling for which
other subtrees co-occur with it?".

Usage in the RAP optimizer:
- After every oracle call, .observe(smiles, parsed, score)
- Periodically (every refit_every calls) the model retrains in-place
- .score_subtree(tokens) returns β_j (0.0 for unseen subtrees)
- .predicted_delta(removed_tokens, added_tokens) returns the expected
  Δscore from a transplant (used for counterfactual filtering).

Implementation notes:
- We cap unique subtrees at max_features (default 1000) to keep refit cheap.
  Selection: keep the top-K most frequent subtrees, with the rest folded into
  a single "rare" feature so contribution mass isn't lost.
- Ridge regression with alpha=1.0 (closed-form solve via numpy).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SubtreeAttribution:
    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        refit_every: int = 200,
        max_features: int = 1000,
        min_observations: int = 20,
    ) -> None:
        self.ridge_alpha = float(ridge_alpha)
        self.refit_every = int(refit_every)
        self.max_features = int(max_features)
        self.min_observations = int(min_observations)
        # Storage: list of (feature_set, score) per molecule
        self._records: List[Tuple[frozenset, float]] = []
        self._subtree_counter: Counter = Counter()  # cumulative observation count per subtree
        # Learned model
        self._beta: Dict[Tuple[str, ...], float] = {}
        self._intercept: float = 0.0
        self._last_refit: int = 0
        self._n_subtrees_in_model: int = 0

    @property
    def n_observations(self) -> int:
        return len(self._records)

    @property
    def is_ready(self) -> bool:
        """True if we have enough data to use the learned model (vs heuristic fallback)."""
        return self.n_observations >= self.min_observations and bool(self._beta)

    def observe(self, subtree_token_tuples: Sequence[Tuple[str, ...]], score: float) -> None:
        """Record a (molecule, score) pair where the molecule decomposes into the
        given subtree token tuples (canonical, no leading [B_*]).

        Triggers a refit once enough observations have accrued.
        """
        feats = frozenset(subtree_token_tuples)
        self._records.append((feats, float(score)))
        for f in feats:
            self._subtree_counter[f] += 1

        if (
            self.n_observations - self._last_refit >= self.refit_every
            and self.n_observations >= self.min_observations
        ):
            self.refit()

    def refit(self) -> None:
        """Re-fit ridge regression on all accumulated observations.

        Selects the top-K most frequent subtrees as features. Other subtrees
        contribute via a single 'rare' feature.
        """
        n_total = len(self._subtree_counter)
        if n_total == 0:
            return
        # Pick top-K subtrees by frequency. Keep only those with count >= 2 (singletons too noisy).
        candidates = [(f, c) for f, c in self._subtree_counter.most_common() if c >= 2]
        candidates = candidates[: self.max_features - 1]  # leave 1 slot for rare bucket
        feature_index: Dict[Tuple[str, ...], int] = {f: i for i, (f, _) in enumerate(candidates)}
        rare_idx = len(feature_index)
        K = rare_idx + 1
        N = len(self._records)

        X = np.zeros((N, K), dtype=np.float32)
        y = np.zeros(N, dtype=np.float32)
        for i, (feats, score) in enumerate(self._records):
            y[i] = score
            saw_rare = False
            for f in feats:
                idx = feature_index.get(f)
                if idx is not None:
                    X[i, idx] = 1.0
                else:
                    saw_rare = True
            if saw_rare:
                X[i, rare_idx] = 1.0

        # Ridge: beta = (X^T X + αI)^-1 X^T (y - y_mean), with intercept = y_mean
        y_mean = float(y.mean())
        y_centered = y - y_mean
        XtX = X.T @ X
        XtX += self.ridge_alpha * np.eye(K, dtype=np.float32)
        Xty = X.T @ y_centered
        try:
            beta_arr = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return
        self._intercept = y_mean
        self._beta = {f: float(beta_arr[i]) for f, i in feature_index.items()}
        # Rare bucket is its own coefficient — apply it for any unseen subtree at query time.
        self._rare_coef = float(beta_arr[rare_idx])
        self._n_subtrees_in_model = len(self._beta)
        self._last_refit = self.n_observations
        logger.debug(
            f"[attribution] refit: N={N} K={K} y_mean={y_mean:.4f} "
            f"beta_min={beta_arr.min():.4f} beta_max={beta_arr.max():.4f} "
            f"rare_coef={self._rare_coef:.4f}"
        )

    def score_subtree(self, tokens: Sequence[str]) -> float:
        """Return learned contribution coefficient for a subtree.

        Falls back to rare-bucket coefficient for unseen subtrees, and to 0 if model
        not yet fitted.
        """
        key = tuple(tokens)
        if key in self._beta:
            return self._beta[key]
        return getattr(self, "_rare_coef", 0.0)

    def predicted_delta(
        self,
        removed_tokens: Sequence[str],
        added_tokens: Sequence[str],
    ) -> float:
        """Predicted Δscore if we replace ``removed_tokens`` with ``added_tokens``.

        This is the simplest counterfactual under the additive model:
        Δscore ≈ β[added] - β[removed]
        """
        return self.score_subtree(added_tokens) - self.score_subtree(removed_tokens)

    def stats(self) -> Dict[str, float]:
        return {
            "n_obs": self.n_observations,
            "n_unique_subtrees": len(self._subtree_counter),
            "n_in_model": self._n_subtrees_in_model,
            "intercept": self._intercept,
            "rare_coef": getattr(self, "_rare_coef", 0.0),
            "is_ready": int(self.is_ready),
        }

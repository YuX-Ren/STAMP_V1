"""SubtreePool — harvest high-scoring subtrees, query by port signature.

Pool semantics
--------------
- Keyed by :class:`PortSignature`. Each bucket is sorted by score descending.
- Per-signature cap: 50 entries (LRU on lowest score).
- Global cap: 5000 entries total (when exceeded, evict from the largest bucket).
- Adaptive harvest threshold: ``max(0.7 * current_best, 0.3)``.

Query
-----
``query(signature, target_size, size_delta=4, top_k=5)`` returns up to
``top_k`` (donor_tokens, donor_score, donor_atom_count) tuples whose port
signature matches and whose heavy-atom count is within ``size_delta`` of
the requested size.

Each donor's tokens are stored as a tuple of strings (the canonical flat
sequence of the subtree, **without** any leading ``[B_*]`` token —
that branch marker is owned by the parent context, not the subtree).
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .port_signature import PortSignature, heavy_atom_count_from_tokens


@dataclass
class DonorEntry:
    tokens: Tuple[str, ...]    # flat subtree tokens (no leading [B_*])
    score: float               # donor molecule's oracle reward
    atom_count: int            # heavy atoms in the subtree
    source_smiles: str = ""    # the canonical SMILES of the donor molecule
    n_uses: int = 0            # how many times we've transplanted from this entry

    def key(self) -> Tuple[str, ...]:
        return self.tokens


@dataclass
class SubtreePool:
    per_sig_cap: int = 50
    global_cap: int = 5000
    min_threshold: float = 0.3
    threshold_fraction: float = 0.7

    pool: Dict[PortSignature, List[DonorEntry]] = field(default_factory=lambda: defaultdict(list))
    _seen: Dict[Tuple[PortSignature, Tuple[str, ...]], DonorEntry] = field(default_factory=dict)
    current_best: float = 0.0

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    @property
    def total_count(self) -> int:
        return sum(len(v) for v in self.pool.values())

    @property
    def harvest_threshold(self) -> float:
        return max(self.threshold_fraction * self.current_best, self.min_threshold)

    def update_best(self, reward: float) -> None:
        if reward > self.current_best:
            self.current_best = float(reward)

    # ------------------------------------------------------------------
    # Harvesting
    # ------------------------------------------------------------------
    def harvest(
        self,
        donors: Iterable[Tuple[PortSignature, Sequence[str], int]],
        molecule_score: float,
        source_smiles: str = "",
    ) -> int:
        """Harvest a batch of (signature, tokens, atom_count) tuples
        extracted from a single molecule with score ``molecule_score``.

        Returns number of new entries added (after dedup).
        """
        self.update_best(molecule_score)
        if molecule_score < self.harvest_threshold:
            return 0
        added = 0
        for sig, tokens, size in donors:
            if sig is None:
                continue
            tokens_t = tuple(tokens)
            if not tokens_t:
                continue
            seen_key = (sig, tokens_t)
            if seen_key in self._seen:
                # Already in pool — promote score if higher.
                entry = self._seen[seen_key]
                if molecule_score > entry.score:
                    entry.score = float(molecule_score)
                    entry.source_smiles = source_smiles
                    self._reinsert(sig, entry)
                continue
            entry = DonorEntry(
                tokens=tokens_t,
                score=float(molecule_score),
                atom_count=int(size),
                source_smiles=source_smiles,
            )
            self._insert(sig, entry)
            self._seen[seen_key] = entry
            added += 1
        self._enforce_global_cap()
        return added

    def _insert(self, sig: PortSignature, entry: DonorEntry) -> None:
        bucket = self.pool[sig]
        # Keep bucket sorted by descending score using bisect on negative key.
        scores = [-e.score for e in bucket]
        i = bisect.bisect_right(scores, -entry.score)
        bucket.insert(i, entry)
        if len(bucket) > self.per_sig_cap:
            evicted = bucket.pop()  # lowest score
            self._seen.pop((sig, evicted.tokens), None)

    def _reinsert(self, sig: PortSignature, entry: DonorEntry) -> None:
        bucket = self.pool[sig]
        try:
            bucket.remove(entry)
        except ValueError:
            return
        scores = [-e.score for e in bucket]
        i = bisect.bisect_right(scores, -entry.score)
        bucket.insert(i, entry)

    def _enforce_global_cap(self) -> None:
        while self.total_count > self.global_cap:
            # Evict from largest bucket.
            largest_sig = max(self.pool.keys(), key=lambda k: len(self.pool[k]))
            bucket = self.pool[largest_sig]
            if not bucket:
                break
            evicted = bucket.pop()  # lowest score in that bucket
            self._seen.pop((largest_sig, evicted.tokens), None)
            if not bucket:
                self.pool.pop(largest_sig, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        signature: PortSignature,
        target_size: int,
        *,
        size_delta: int = 4,
        top_k: int = 5,
    ) -> List[DonorEntry]:
        bucket = self.pool.get(signature, [])
        out: List[DonorEntry] = []
        for entry in bucket:  # already score-sorted descending
            if abs(entry.atom_count - int(target_size)) <= int(size_delta):
                out.append(entry)
                if len(out) >= top_k:
                    break
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "n_signatures": len(self.pool),
            "n_entries_total": self.total_count,
            "current_best": self.current_best,
            "harvest_threshold": self.harvest_threshold,
        }

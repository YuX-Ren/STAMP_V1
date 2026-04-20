"""
Glue code that plugs the trained STAMP AR language model and RDKit/GuacaMol
scoring into the MCTS skeleton in ``mcts_protype.py`` (at the repo root).

Three concrete classes are provided:

- ``STAMPDecoderImpl``: wraps :class:`stamp.core.stamp_codec.STAMPEncoder` so
  that a STAMP token list can be decoded back to a SMILES string with the
  built-in R4 safety fallbacks.  Also supplies an approximate ``is_terminal``
  via a lightweight open-port counter.

- ``ARPolicyModel``: wraps a trained :class:`TinyDecoderLM` (loaded via
  :func:`stamp.benchmark.lm.load_trained_model`).  The MCTS prototype needs
  two interfaces:

    * ``top_k_actions`` — return up to *k* candidate *motif-groups*
      (one motif token + its ``[J_i]``/``[B_n]``/``[S_*]``/``[Ring]`` tokens)
      with their LM probabilities.  We sample a small batch of short
      continuations from the LM, extract the first motif-group from each,
      and aggregate by empirical frequency.

    * ``rollout`` — greedy/low-temperature LM completion of a partial
      STAMP prefix.

- ``GuacaMolOracle``: wraps a goal-directed benchmark's scoring function
  from the ``guacamol`` package, with a ``scipy.histogram`` shim for newer
  SciPy versions.

The runner in ``scripts/benchmark/run_mcts_guacamol.py`` stitches these
together and drives the MCTS from ``mcts_protype.MCTS``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger

# Quiet RDKit warnings during sampling — we'll see invalid SMILES as reward=0.
RDLogger.DisableLog("rdApp.*")

from ..core.stamp_codec import STAMPEncoder
from .lm import (
    LMModel,
    TokenVocab,
    load_trained_model,
    sample_model_sequences,
)


# Make ``mcts_protype`` importable regardless of cwd.  The file lives at the
# repository root alongside ``stamp/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_protype import Action, Oracle, PolicyModel, STAMPDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for STAMP token structure
# ---------------------------------------------------------------------------

def _is_motif_token(tok: str) -> bool:
    """True for motif SMILES tokens (not bracketed structural tokens)."""
    if not tok:
        return False
    if tok.startswith("<"):  # <pad>/<bos>/<eos>/<unk>/<mask>
        return False
    if tok.startswith("[") and not _is_atom_bracket(tok):
        # Structural like [J_i], [B_n], [S_*], [Ring], [END], [BACK]
        return False
    return True


def _is_atom_bracket(tok: str) -> bool:
    """Heuristic: bracketed *atoms* inside SMILES motifs look like [nH], [C@H],
    [N+], etc.  STAMP structural tokens start with J_/B_/S_/Ring/END/BACK.
    """
    if not tok.startswith("[") or not tok.endswith("]"):
        return False
    body = tok[1:-1]
    return not body.startswith(("J_", "B_", "S_", "Ring", "END", "BACK"))


def parse_next_group(tokens: Sequence[str], start: int) -> Optional[int]:
    """Return the end-exclusive index of the next motif-group, or ``None``.

    A motif-group = optional ``[B_n]`` + motif token + zero or more
    ``[J_i]`` each optionally followed by ``[S_*]`` and ``[Ring]``.
    """
    i = start
    n = len(tokens)
    # Optional leading branch token
    if i < n and tokens[i].startswith("[B_"):
        i += 1
    if i >= n:
        return None
    if not _is_motif_token(tokens[i]):
        return None
    i += 1
    while i < n and tokens[i].startswith("[J_"):
        i += 1
        if i < n and tokens[i].startswith("[S_"):
            i += 1
        if i < n and tokens[i] == "[Ring]":
            i += 1
    return i


def count_open_ports(tokens: Sequence[str]) -> int:
    """Lightweight port-stack counter.

    Rough bookkeeping (ignoring ring closures, which partially cancel out):
        open = sum(joints) - (num_motifs - 1)

    The stack machine always *opens* once per ``[J_i]`` and *consumes* one
    port when attaching a non-root motif.  For Mcts ``is_terminal`` this
    approximation is fine — it under-counts when ring closures appear, but
    the LM naturally learns to close things off.
    """
    motifs = 0
    joints = 0
    for tok in tokens:
        if tok.startswith("[J_"):
            joints += 1
        elif _is_motif_token(tok):
            motifs += 1
    if motifs == 0:
        return -1
    return joints - (motifs - 1)


# ---------------------------------------------------------------------------
# STAMPDecoder implementation
# ---------------------------------------------------------------------------

class STAMPDecoderImpl(STAMPDecoder):
    """Real STAMP decoder backed by ``stamp.core.stamp_codec.STAMPEncoder``."""

    def __init__(self, encoder: Optional[STAMPEncoder] = None) -> None:
        self.encoder = encoder or STAMPEncoder()

    def decode_to_smiles(self, tokens: List[str]) -> Optional[str]:
        cleaned = [t for t in tokens if t and not t.startswith("<")]
        if not cleaned:
            return None
        # The decoder tolerates an explicit ``[END]`` terminator — append if
        # missing so partial sequences fall through the R4 safety path cleanly.
        if "[END]" not in cleaned:
            cleaned = cleaned + ["[END]"]
        try:
            mol = self.encoder.decode_stamp_to_mol(" ".join(cleaned))
        except Exception:
            return None
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        frags = Chem.GetMolFrags(mol)
        if len(frags) != 1:
            return None
        try:
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return None
        return smi if smi else None

    def is_terminal(self, tokens: List[str]) -> bool:
        if "[END]" in tokens:
            return True
        open_ports = count_open_ports(tokens)
        if open_ports < 0:
            return False
        return open_ports == 0


# ---------------------------------------------------------------------------
# Policy model implementation
# ---------------------------------------------------------------------------

class ARPolicyModel(PolicyModel):
    """Wraps a trained ``TinyDecoderLM`` for MCTS.

    Parameters
    ----------
    model_dir : path to ``{model.pt, vocab.json}``.
    device    : ``"cuda" / "cuda:0" / "cpu"``.  Auto-selects by default.
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        *,
        proposal_samples: int = 48,
        proposal_temperature: float = 1.0,
        proposal_top_k: int = 50,
        proposal_new_tokens: int = 8,
        rollout_temperature: float = 0.9,
        rollout_top_k: int = 50,
        rollout_new_tokens: int = 48,
    ) -> None:
        self.model, self.vocab, self.rep, self.device = load_trained_model(
            model_dir, device=device
        )
        self.model.eval()
        self.proposal_samples = int(proposal_samples)
        self.proposal_temperature = float(proposal_temperature)
        self.proposal_top_k = int(proposal_top_k)
        self.proposal_new_tokens = int(proposal_new_tokens)
        self.rollout_temperature = float(rollout_temperature)
        self.rollout_top_k = int(rollout_top_k)
        self.rollout_new_tokens = int(rollout_new_tokens)

    # ------------------------------------------------------------------
    # Internal sampling helper
    # ------------------------------------------------------------------
    def _sample(
        self,
        prefix_tokens: Sequence[str],
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> List[List[str]]:
        max_seq_len = int(self.model.cfg.max_seq_len)
        # Reserve room for prompt + new tokens; drop excess prompt from front
        # (rare — STAMP sequences are short).
        prompt = list(prefix_tokens)
        # +1 for BOS inside the sampler.
        if len(prompt) + 1 >= max_seq_len:
            keep = max_seq_len - 1 - max(1, max_new_tokens)
            prompt = prompt[-max(1, keep):]
        budget = min(max_new_tokens, max_seq_len - len(prompt) - 1)
        if budget <= 0:
            return [list(prompt) for _ in range(n_samples)]
        seqs = sample_model_sequences(
            self.model,
            self.vocab,
            n_samples=n_samples,
            max_new_tokens=budget,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt,
            device=self.device,
        )
        return seqs

    # ------------------------------------------------------------------
    # PolicyModel API
    # ------------------------------------------------------------------
    def top_k_actions(self, prefix_tokens: List[str], k: int) -> List[Action]:
        n_samples = max(self.proposal_samples, k * 2)
        # A single motif-group is at most ~6–8 tokens.  Give ourselves a
        # little slack so the parser finds a complete group.
        max_new = max(self.proposal_new_tokens, 10)
        seqs = self._sample(
            prefix_tokens,
            n_samples=n_samples,
            max_new_tokens=max_new,
            temperature=self.proposal_temperature,
            top_k=self.proposal_top_k,
        )
        prefix_len = len(prefix_tokens)
        counter: Dict[Tuple[str, ...], int] = {}
        for seq in seqs:
            if len(seq) <= prefix_len:
                # EOS produced immediately → propose a terminal action.
                key: Tuple[str, ...] = ("[END]",)
            else:
                tail = seq[prefix_len:]
                if not tail:
                    continue
                # Explicit STAMP terminator in the sampled tail.
                if tail[0] == "[END]":
                    key = ("[END]",)
                else:
                    end_idx = parse_next_group(tail, 0)
                    if end_idx is None or end_idx == 0:
                        continue
                    key = tuple(tail[:end_idx])
            counter[key] = counter.get(key, 0) + 1
        if not counter:
            return []
        total = sum(counter.values())
        actions = [
            Action(tokens=tokens, prior=cnt / total)
            for tokens, cnt in counter.items()
        ]
        actions.sort(key=lambda a: -a.prior)
        return actions[:k]

    def rollout(self, prefix_tokens: List[str], max_steps: int) -> List[str]:
        # Each motif-group ≤ ~6 tokens, plus a terminator.
        new_tokens_budget = min(
            self.rollout_new_tokens, max(8, int(max_steps) * 6 + 2)
        )
        seqs = self._sample(
            prefix_tokens,
            n_samples=1,
            max_new_tokens=new_tokens_budget,
            temperature=self.rollout_temperature,
            top_k=self.rollout_top_k,
        )
        if not seqs:
            return list(prefix_tokens)
        return seqs[0]


# ---------------------------------------------------------------------------
# GuacaMol oracle
# ---------------------------------------------------------------------------

def _patch_scipy_histogram() -> None:
    """Older ``guacamol`` imports ``scipy.histogram`` which was removed
    in modern SciPy — alias it to :func:`numpy.histogram` once, lazily.
    """
    import numpy as _np
    import scipy

    if not hasattr(scipy, "histogram"):
        scipy.histogram = _np.histogram  # type: ignore[attr-defined]


class GuacaMolOracle(Oracle):
    """Wrap a GuacaMol goal-directed benchmark scoring function.

    Parameters
    ----------
    benchmark_name : full benchmark name, e.g. ``"Celecoxib rediscovery"``.
    version        : ``"v1"`` / ``"v2"`` / ``"trivial"``.
    """

    def __init__(self, benchmark_name: str, version: str = "v2") -> None:
        _patch_scipy_histogram()
        from guacamol.benchmark_suites import goal_directed_benchmark_suite

        suite = goal_directed_benchmark_suite(version_name=version)
        lookup = {b.name: b for b in suite}
        if benchmark_name not in lookup:
            raise ValueError(
                f"Benchmark '{benchmark_name}' not found in version '{version}'.\n"
                f"Available: {sorted(lookup)}"
            )
        self._benchmark = lookup[benchmark_name]
        self._scoring = self._benchmark.objective
        self._name = benchmark_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def benchmark(self):
        return self._benchmark

    def __call__(self, smiles: str) -> float:
        if not smiles:
            return 0.0
        try:
            score = float(self._scoring.score(smiles))
        except Exception:
            return 0.0
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return max(0.0, min(1.0, score))


def list_guacamol_benchmarks(version: str = "v2") -> List[str]:
    """Return benchmark names for the given version."""
    _patch_scipy_histogram()
    from guacamol.benchmark_suites import goal_directed_benchmark_suite

    return [b.name for b in goal_directed_benchmark_suite(version_name=version)]

"""Faithful reproduction of GenMol's PMO optimizer adapted to our AO-GPT."""

from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from ...core.stamp_codec import STAMPEncoder
from .edit_policy import AOEditPolicy
from .subtree import edit_candidate_spans, parse_terminal_sequence
from .novelty import NoveltyScorer, rank_by_composite
from .target_aware import TargetAwareHelper
from .tree_crossover import tree_crossover
from .chemistry_filters import SurrogateRanker


def _try_load_sa_scorer():
    """Lazily import RDKit SAScorer (1=easy, 10=hard). Returns callable(mol) -> float, or None."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_path) not in sys.path:
            sys.path.append(str(sa_path))
        import sascorer  # type: ignore
        return sascorer.calculateScore
    except Exception:
        return None

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ..oracle import Oracle

logger = logging.getLogger(__name__)

_ATTACH_RXN = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")


# ----------------------------------------------------------------------
# LRU caches for the hot inference path.
# Population parents are reused 100+ times across crossover/mutation so a
# 4096-entry cache hits >90% in steady state.
# ----------------------------------------------------------------------
class _STAMPCache:
    """Per-encoder cache of (smi -> stamp_str) and (smi -> ParsedMolecule).

    Keyed by canonical SMILES so cache survives across SMILES re-canon.
    """
    def __init__(self, encoder: STAMPEncoder, max_size: int = 4096) -> None:
        self.encoder = encoder
        self.max_size = int(max_size)
        self._encode: Dict[str, str] = {}
        self._parsed: Dict[str, object] = {}  # ParsedMolecule

    def encode(self, smi: str) -> str:
        v = self._encode.get(smi)
        if v is not None:
            return v
        v = self.encoder.encode_smiles(smi)
        if len(self._encode) >= self.max_size:
            self._encode.pop(next(iter(self._encode)))
        self._encode[smi] = v
        return v

    def parsed(self, smi: str):
        v = self._parsed.get(smi)
        if v is not None:
            return v
        try:
            stamp = self.encode(smi)
            v = parse_terminal_sequence(stamp.split(), self.encoder)
        except Exception:
            # encode_smiles can raise (e.g. fused rings outside the motif vocab).
            return None
        if v is None:
            return None
        if len(self._parsed) >= self.max_size:
            self._parsed.pop(next(iter(self._parsed)))
        self._parsed[smi] = v
        return v


def attach(frag1: str, frag2: str) -> Optional[Chem.Mol]:
    m1 = Chem.MolFromSmiles(frag1)
    m2 = Chem.MolFromSmiles(frag2)
    if m1 is None or m2 is None:
        return None
    products = _ATTACH_RXN.RunReactants((m1, m2))
    if not products:
        return None
    idx = np.random.randint(len(products))
    prod = products[idx][0]
    try:
        Chem.SanitizeMol(prod)
    except Exception:
        return None
    return prod


def cut(smiles: str, n_tries: int = 3) -> set:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    smarts = Chem.MolFromSmarts("[*]-;!@[*]")
    if not mol.HasSubstructMatch(smarts):
        return set()
    frags: set = set()
    matches = mol.GetSubstructMatches(smarts)
    for _ in range(n_tries):
        if not matches:
            break
        bis = random.choice(matches)
        bond = mol.GetBondBetweenAtoms(bis[0], bis[1])
        if bond is None:
            continue
        try:
            fragments_mol = Chem.FragmentOnBonds(
                mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)]
            )
            sub = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            for f in sub:
                smi = Chem.MolToSmiles(f)
                if "[1*]" in smi or "*" in smi:
                    frags.add(smi)
        except Exception:
            continue
    return frags


SMALL_TASKS = {
    "albuterol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "median1",
    "qed",
    "sitagliptin_mpo",
    "zaleplon_mpo",
}
LARGE_TASKS = {"gsk3b", "jnk3"}


def task_size_range(task: str) -> Tuple[int, int]:
    """Molecule heavy-atom range per task family.

    Reverted default to [20, 40] (faithful's original value, validated at sum=17.72).
    f-RAG's [20, 50] hurt C-group rediscovery/similarity tasks in Phase C 2026-04-30.

    SMALL_TASKS  → [10, 30]
    LARGE_TASKS  → [30, 80]
    default      → [20, 40]
    """
    if task in SMALL_TASKS:
        return 10, 30
    if task in LARGE_TASKS:
        return 30, 80
    return 20, 40


def task_fragment_size_range(task: str) -> Tuple[int, int]:
    """Fragment heavy-atom range per task family.

    Empirical finding (Phase C 23-task sweep, 2026-04-30): f-RAG's filter [10,16] for default
    tasks heavily degrades rediscovery/similarity/SMARTS tasks (celecoxib -0.28, mestranol -0.25,
    valsartan -0.20). Disabled for B/C groups; kept only for SMALL_TASKS where validated.

    SMALL_TASKS  → [5, 12]    (sitagliptin_mpo, qed, median1, zaleplon_mpo, isomers, albuterol)
    LARGE_TASKS  → (0, 0)     no filter (faithful's behavior)
    default      → (0, 0)     no filter (faithful's behavior)
    """
    if task in SMALL_TASKS:
        return 5, 12
    return 0, 0


class SELFIESMutation:
    def __init__(self, n_mutations_min: int = 1, n_mutations_max: int = 3) -> None:
        import selfies as sf

        self.sf = sf
        self._alphabet = list(sf.get_semantic_robust_alphabet())
        self.n_mutations_min = int(n_mutations_min)
        self.n_mutations_max = int(n_mutations_max)

    def mutate(self, smiles: str) -> Optional[str]:
        try:
            selfies_str = self.sf.encoder(smiles)
        except Exception:
            return smiles
        tokens = list(self.sf.split_selfies(selfies_str))
        if len(tokens) < 3:
            return smiles

        n_ops = random.randint(self.n_mutations_min, self.n_mutations_max)
        for _ in range(n_ops):
            op = random.choice(["replace", "insert", "delete"])
            if op == "replace" and tokens:
                i = random.randrange(len(tokens))
                tokens[i] = random.choice(self._alphabet)
            elif op == "insert":
                i = random.randrange(len(tokens) + 1)
                tokens.insert(i, random.choice(self._alphabet))
            elif op == "delete" and len(tokens) > 3:
                i = random.randrange(len(tokens))
                tokens.pop(i)

        try:
            new_smi = self.sf.decoder("".join(tokens))
        except Exception:
            return smiles
        if not new_smi:
            return smiles
        mol = Chem.MolFromSmiles(new_smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return smiles
        if len(Chem.GetMolFrags(mol)) != 1:
            return smiles
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles


class STAMPAtomMutation:
    def __init__(
        self,
        encoder: STAMPEncoder,
        edit: AOEditPolicy,
        n_mask_min: int = 1,
        n_mask_max: int = 3,
    ) -> None:
        self.encoder = encoder
        self.edit = edit
        self.n_mask_min = int(n_mask_min)
        self.n_mask_max = int(n_mask_max)

    def mutate(self, smiles: str) -> Optional[str]:
        try:
            stamp = self.encoder.encode_smiles(smiles)
        except Exception:
            return smiles
        tokens = stamp.split()
        if len(tokens) < 3:
            return smiles
        n_mask = random.randint(self.n_mask_min, self.n_mask_max)
        new_tokens = self.edit.resample_random_k(tokens, n_mask=n_mask)
        if list(new_tokens) == list(tokens):
            return smiles
        try:
            mol = self.encoder.decode_stamp_to_mol(" ".join(list(new_tokens) + ["[END]"]))
        except Exception:
            return smiles
        if mol is None or mol.GetNumAtoms() == 0:
            return smiles
        if len(Chem.GetMolFrags(mol)) != 1:
            return smiles
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles


class STAMPMutation:
    def __init__(self, encoder: STAMPEncoder, edit: AOEditPolicy, cache: Optional[_STAMPCache] = None) -> None:
        self.encoder = encoder
        self.edit = edit
        self.cache = cache or _STAMPCache(encoder)

    def mutate(self, smiles: str) -> Optional[str]:
        try:
            parsed = self.cache.parsed(smiles)
        except Exception:
            return smiles
        if parsed is None:
            return smiles
        spans = edit_candidate_spans(parsed, include_root=False)
        if not spans:
            return smiles
        span = random.choice(spans)
        try:
            new_flat = self.edit.resample_span(parsed.flat_tokens, span.token_start, span.token_end)
        except (ValueError, RuntimeError):
            # AO-GPT max_seq_len overflow or other model-side error: skip mutation.
            return smiles
        if list(new_flat) == list(parsed.flat_tokens):
            return smiles
        try:
            mol = self.encoder.decode_stamp_to_mol(" ".join(list(new_flat) + ["[END]"]))
        except Exception:
            return smiles
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            return smiles
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles


class MultiSpanSTAMPMutation:
    """Mask K disjoint subtree spans simultaneously and let AO-GPT fill them
    jointly (each masked position attends to all observed + previously-sampled
    masked siblings).

    Picks K spans that are non-overlapping at the tree level, sampled
    proportionally to span length so larger branches don't dominate (smaller
    branches get a fair shot). Falls back to single-span if fewer than K
    non-overlapping spans are available.
    """

    def __init__(
        self,
        encoder: STAMPEncoder,
        edit: AOEditPolicy,
        *,
        n_spans: int = 2,
        max_span_len: int = 10,
        cache: Optional[_STAMPCache] = None,
    ) -> None:
        self.encoder = encoder
        self.edit = edit
        self.n_spans = max(1, int(n_spans))
        self.max_span_len = int(max_span_len)
        self.cache = cache or _STAMPCache(encoder)

    def _pick_disjoint_spans(self, spans):
        """Sample up to ``self.n_spans`` non-overlapping spans, weighted by
        1/(span_len + 1) so smaller spans get more representation."""
        usable = [s for s in spans if s.span_len <= self.max_span_len]
        if not usable:
            return []
        random.shuffle(usable)
        chosen: list = []
        for span in usable:
            ok = True
            for existing in chosen:
                if not (span.token_end <= existing.token_start or span.token_start >= existing.token_end):
                    ok = False
                    break
            if ok:
                chosen.append(span)
            if len(chosen) >= self.n_spans:
                break
        return chosen

    def mutate(self, smiles: str) -> Optional[str]:
        try:
            parsed = self.cache.parsed(smiles)
        except Exception:
            return smiles
        if parsed is None:
            return smiles
        spans = edit_candidate_spans(parsed, include_root=False)
        if not spans:
            return smiles
        chosen = self._pick_disjoint_spans(spans)
        if not chosen:
            return smiles
        span_pairs = [(s.token_start, s.token_end) for s in chosen]
        try:
            new_flat = self.edit.resample_multi_span(parsed.flat_tokens, span_pairs)
        except (ValueError, RuntimeError):
            return smiles
        if list(new_flat) == list(parsed.flat_tokens):
            return smiles
        try:
            mol = self.encoder.decode_stamp_to_mol(" ".join(list(new_flat) + ["[END]"]))
        except Exception:
            return smiles
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            return smiles
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles


@dataclass
class GenMolFaithfulConfig:
    population_size: int = 100
    warmup: int = 1000
    max_iterations: int = 30000
    n_attach_tries: int = 1000
    log_every: int = 200
    min_size: int = 20
    max_size: int = 40
    mutation_mode: str = "ao_gpt"
    selfies_n_min: int = 1
    selfies_n_max: int = 3
    ao_atom_n_min: int = 1
    ao_atom_n_max: int = 3
    target_aware: bool = False
    target_frag_score_bonus: float = 0.001
    similarity_guard_delta: float = 0.10
    similarity_guard_retries: int = 3
    zero_in_stall_calls: int = 1_000_000_000
    zero_in_n_mask: int = 1
    novelty_weight: float = 0.0
    novelty_k_neighbors: int = 10
    novelty_gap_tau: float = 0.4
    novelty_auto_scale: bool = False
    parent_sampling_tau: float = 0.0
    tree_crossover_prob: float = 0.0
    tree_crossover_size_delta: int = 4
    molecule_pool_size: int = 100
    # ---- Multi-span remasking knobs (active only when mutation_mode is ao_gpt_multi or mixed_multi) ----
    multi_span_k: int = 2
    multi_span_max_len: int = 10
    chain_steps: int = 1
    # f-RAG-style hyperparameters (NeurIPS 2024)
    fragment_min_atoms: int = 0   # filter vocab CSV: drop fragments with < N heavy atoms
    fragment_max_atoms: int = 0   # filter vocab CSV: drop fragments with > N heavy atoms; 0 disables
    mutation_prob: float = 1.0    # apply mutation with this probability per crossover (f-RAG uses 0.1)
    # ---- STAMP-native chemistry methods (M11) ----
    # M11: Subtree-attribution surrogate ranking
    use_surrogate: bool = False
    surrogate_k_candidates: int = 4
    surrogate_refit_every: int = 200
    surrogate_min_obs: int = 30
    surrogate_ridge_alpha: float = 1.0
    # P2: Multi-subpop scaffold diversity (FAIR — uses only Murcko scaffolds, not target)
    use_scaffold_diversity: bool = False
    scaffold_dominance_threshold: float = 0.5   # if a scaffold is > X of recent population, force exploration
    scaffold_window_calls: int = 300            # window size for scaffold counting
    scaffold_force_diff_rate: float = 0.3       # prob of forcing new scaffold when dominance triggered
    scaffold_force_diff_max_tries: int = 30     # how many gen attempts to find a non-dominant scaffold
    # ---- STAMP-GP Layer 1: port-aware subtree crossover + brood selection ----
    # Replaces GenMol's attach() with chemistry-aware tree_crossover that swaps
    # port-signature-compatible subtrees between two parent molecules.
    use_stamp_gp: bool = False
    gp_brood_size: int = 4               # K candidates per generation step
    gp_size_delta: int = 4               # max heavy-atom diff between swapped subtrees
    gp_min_pool_size: int = 30           # min _mol_pool size before GP activates
    gp_attach_fallback: bool = True      # if all crossover attempts fail, fall back to attach()
    gp_max_tries: int = 3                # tree_crossover internal retries per call
    gp_match_mode: str = "port_sig"      # 'port_sig' (default) or 'topo_hash' (strict tree isomorphism)
    gp_n_swaps: int = 1                  # number of simultaneous disjoint subtree swaps (multi-point crossover)
    # ---- STAMP-GP Layer 3: Scaffold Niching ----
    # Maintain N Murcko-scaffold niches in _mol_pool to resist scaffold collapse.
    use_scaffold_niching: bool = False
    niche_capacity: int = 8              # top-K mols per niche
    max_niches: int = 25                 # max distinct scaffolds tracked
    niche_intra_prob: float = 0.6        # P(both parents from same niche)
    # ---- STAMP-design tuning knobs (defaults disable the filters) ----
    # Acceptance filters applied to every candidate molecule before scoring.
    # When a candidate violates a hard limit, it's dropped and the loop tries another.
    # Soft penalties are applied with probabilistic rejection.
    mw_max: float = 0.0          # hard upper bound on MolWt; 0 disables
    mw_alpha: float = 0.0        # soft-rejection probability slope above 0.85 * mw_max
    sa_max: float = 0.0          # hard upper bound on SA score (1=easy, 10=hard); 0 disables
    sa_alpha: float = 0.0        # soft-rejection probability slope above sa_max - 1.0
    qed_min: float = 0.0         # hard lower bound on QED (drug-likeness); 0 disables
    filter_warmup_calls: int = 100  # don't apply filters until this many oracle calls
    filter_max_retries: int = 6     # if a candidate fails filters, try generating again up to N times


class GenMolFaithfulOptimizer:
    def __init__(
        self,
        task: str,
        vocab_dir: str,
        encoder: STAMPEncoder,
        edit_policy: AOEditPolicy,
        oracle: Oracle,
        config: Optional[GenMolFaithfulConfig] = None,
    ) -> None:
        self.task = task
        self.encoder = encoder
        self.oracle = oracle
        base_cfg = config or GenMolFaithfulConfig()
        # Shared LRU cache for STAMP encode + parse across mutators / GP path.
        self._stamp_cache = _STAMPCache(encoder, max_size=4096)
        self._stamp_mutator = STAMPMutation(encoder, edit_policy, cache=self._stamp_cache)
        self._ao_atom_mutator = None
        self._selfies_mutator = None
        self._multi_span_mutator = None
        need_multi = base_cfg.mutation_mode in {"ao_gpt_multi", "mixed_multi"}
        if need_multi:
            self._multi_span_mutator = MultiSpanSTAMPMutation(
                encoder,
                edit_policy,
                n_spans=base_cfg.multi_span_k,
                max_span_len=base_cfg.multi_span_max_len,
                cache=self._stamp_cache,
            )
        if base_cfg.mutation_mode in {"ao_atom", "mixed_ao"}:
            self._ao_atom_mutator = STAMPAtomMutation(
                encoder,
                edit_policy,
                n_mask_min=base_cfg.ao_atom_n_min,
                n_mask_max=base_cfg.ao_atom_n_max,
            )
        if base_cfg.mutation_mode in {"selfies", "mixed"}:
            self._selfies_mutator = SELFIESMutation(
                n_mutations_min=base_cfg.selfies_n_min,
                n_mutations_max=base_cfg.selfies_n_max,
            )
        self._mutation_mode = base_cfg.mutation_mode

        min_sz, max_sz = task_size_range(task)
        base_cfg.min_size = min_sz
        base_cfg.max_size = max_sz
        # Auto-apply task-family fragment size if user hasn't overridden (both 0).
        if base_cfg.fragment_min_atoms == 0 and base_cfg.fragment_max_atoms == 0:
            f_min, f_max = task_fragment_size_range(task)
            base_cfg.fragment_min_atoms = f_min
            base_cfg.fragment_max_atoms = f_max
            logger.info(
                f"[genmol/{task}] auto fragment-size [{f_min}, {f_max}] (matches f-RAG)"
            )
        self.config = base_cfg

        vocab_csv = Path(vocab_dir) / f"{task}.csv"
        if not vocab_csv.exists():
            raise FileNotFoundError(f"Vocab CSV not found: {vocab_csv}")
        df = pd.read_csv(vocab_csv)
        # f-RAG-style fragment-size filter (drop fragments with atom count outside [min, max]).
        if self.config.fragment_min_atoms > 0 or self.config.fragment_max_atoms > 0:
            min_a = max(0, int(self.config.fragment_min_atoms))
            max_a = int(self.config.fragment_max_atoms) if self.config.fragment_max_atoms > 0 else 10**9

            def _atom_count(smi: str) -> int:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    return 0
                # Count heavy atoms excluding the [1*] dummy attachment atom.
                return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 0 and a.GetSymbol() != "*")

            atom_counts = df["frag"].apply(_atom_count)
            n_before = len(df)
            df = df[(atom_counts >= min_a) & (atom_counts <= max_a)].reset_index(drop=True)
            logger.info(
                f"[genmol/{task}] fragment-size filter [{min_a}, {max_a}]: kept {len(df)}/{n_before}"
            )
        df = df.iloc[: self.config.population_size]
        self.population: List[Tuple[float, str]] = list(zip(df["score"], df["frag"]))
        self.population.sort(reverse=True)

        if self.config.novelty_auto_scale and self.population:
            vocab_top = float(self.population[0][0])
            scale = max(0.0, 1.0 - vocab_top)
            effective_weight = self.config.novelty_weight * scale
            logger.info(
                f"[genmol/{task}] novelty auto-scale: vocab_top={vocab_top:.3f} "
                f"-> weight {self.config.novelty_weight} * {scale:.3f} = {effective_weight:.4f}"
            )
            self.config.novelty_weight = effective_weight

        self._target = TargetAwareHelper(
            task_name=task,
            similarity_guard_delta=self.config.similarity_guard_delta
            if self.config.target_aware
            else 0.0,
            zero_in_stall_calls=self.config.zero_in_stall_calls,
        )

        self._oracle_log: List[Tuple[int, str, float]] = []
        # Trajectory instrumentation (operator + crossover parents per oracle call).
        # Default-on but cheap; surfaced in run() output as "trajectory".
        self._traj: List[dict] = []
        self._last_op: Optional[str] = None
        self._last_parents: Optional[Tuple[str, str]] = None
        self.n_calls = 0
        self.iter = 0
        self.n_mutations_applied = 0
        self.n_crossovers = 0
        self.n_tree_crossovers = 0
        self.n_tree_crossover_failures = 0
        self.best_reward = 0.0
        self.best_smiles: Optional[str] = None
        self._mol_pool: List[Tuple[float, str]] = []
        # ---- STAMP chemistry filters (M11) ----
        self._surrogate_ranker: Optional[SurrogateRanker] = None
        self.n_surrogate_picks = 0
        # ---- P2: Scaffold diversity tracking (FAIR — Murcko only, no target) ----
        self._scaffold_history: List[str] = []  # Murcko scaffold per scored molecule
        self.n_scaffold_diff_forced = 0
        self.n_scaffold_diff_succeeded = 0
        if self.config.use_stamp_gp:
            logger.info(
                f"[genmol/{task}] STAMP-GP enabled: port-aware tree_crossover, "
                f"brood={self.config.gp_brood_size} size_delta={self.config.gp_size_delta} "
                f"min_pool={self.config.gp_min_pool_size} "
                f"attach_fallback={self.config.gp_attach_fallback}"
            )
        # ---- Scaffold Niching (Layer 3) ----
        # niches: scaffold_smiles -> sorted list of (score, smiles), top-K per niche.
        self._niches: Dict[str, List[Tuple[float, str]]] = {}
        if self.config.use_scaffold_niching:
            logger.info(
                f"[genmol/{task}] scaffold niching enabled, "
                f"max_niches={self.config.max_niches} cap={self.config.niche_capacity} "
                f"intra_prob={self.config.niche_intra_prob}"
            )
        if self.config.use_surrogate:
            self._surrogate_ranker = SurrogateRanker(
                encoder,
                ridge_alpha=self.config.surrogate_ridge_alpha,
                refit_every=self.config.surrogate_refit_every,
                min_observations=self.config.surrogate_min_obs,
            )
            logger.info(f"[genmol/{task}] M11 surrogate ranker enabled, K={self.config.surrogate_k_candidates}")
        # STAMP-design filter bookkeeping
        self._sa_scorer = _try_load_sa_scorer() if (self.config.sa_max > 0.0 or self.config.sa_alpha > 0.0) else None
        self.n_filter_rejects_mw = 0
        self.n_filter_rejects_sa = 0
        self.n_filter_rejects_qed = 0
        self.n_filter_passes = 0

    def _stamp_filter_ok(self, mol: Chem.Mol) -> bool:
        """Apply STAMP-design acceptance filters. Returns True if the candidate passes.

        - MW hard cap (config.mw_max), with soft-rejection ramp via mw_alpha.
        - SA hard cap (config.sa_max), with soft-rejection ramp via sa_alpha.
        - QED hard floor (config.qed_min).

        All filters are disabled until self.config.filter_warmup_calls oracle calls have accrued,
        so the population can warm up unconstrained first.
        """
        if self.n_calls < self.config.filter_warmup_calls:
            return True
        cfg = self.config
        # MW gate
        if cfg.mw_max > 0.0:
            try:
                mw = float(Descriptors.MolWt(mol))
            except Exception:
                mw = 0.0
            if mw > cfg.mw_max:
                self.n_filter_rejects_mw += 1
                return False
            if cfg.mw_alpha > 0.0:
                soft_thr = 0.85 * cfg.mw_max
                if mw > soft_thr:
                    excess = (mw - soft_thr) / max(1.0, cfg.mw_max - soft_thr)
                    if random.random() < cfg.mw_alpha * excess:
                        self.n_filter_rejects_mw += 1
                        return False
        # SA gate
        if cfg.sa_max > 0.0 and self._sa_scorer is not None:
            try:
                sa = float(self._sa_scorer(mol))
            except Exception:
                sa = 5.0
            if sa > cfg.sa_max:
                self.n_filter_rejects_sa += 1
                return False
            if cfg.sa_alpha > 0.0:
                soft_thr = max(1.0, cfg.sa_max - 1.0)
                if sa > soft_thr:
                    excess = (sa - soft_thr) / max(0.5, cfg.sa_max - soft_thr)
                    if random.random() < cfg.sa_alpha * excess:
                        self.n_filter_rejects_sa += 1
                        return False
        # QED floor
        if cfg.qed_min > 0.0:
            try:
                q = float(QED.qed(mol))
            except Exception:
                q = 0.0
            if q < cfg.qed_min:
                self.n_filter_rejects_qed += 1
                return False
        self.n_filter_passes += 1
        return True

    def run(self, budget: int) -> Dict[str, object]:
        t0 = time.time()
        while self.n_calls < budget and self.iter < self.config.max_iterations:
            self.iter += 1
            smiles = self._generate()
            if smiles is None:
                continue
            if self._already_scored(smiles):
                continue
            prop = self.oracle(smiles)
            self.n_calls += 1
            self._oracle_log.append((self.n_calls, smiles, prop))
            improved = prop > self.best_reward
            pa, pb = (self._last_parents or (None, None))
            self._traj.append({
                "call": self.n_calls, "smiles": smiles, "reward": float(prop),
                "op": self._last_op or "?", "parent_a": pa, "parent_b": pb,
                "best_so_far": float(max(prop, self.best_reward)), "improved": bool(improved),
            })
            self._update_population(smiles, prop)
            if improved:
                self.best_reward = prop
                self.best_smiles = smiles
            if self.config.target_aware:
                self._target.note_oracle_call(self.best_reward, self.n_calls)
            if self.config.tree_crossover_prob > 0 or self.config.use_stamp_gp:
                self._update_mol_pool(smiles, prop)
            # M11: feed surrogate ranker
            if self._surrogate_ranker is not None:
                self._surrogate_ranker.observe(smiles, prop)
            # Scaffold niching: insert into Murcko-scaffold-keyed niche
            if self.config.use_scaffold_niching:
                self._update_niches(smiles, prop)
            # P2: track scaffold history
            if self.config.use_scaffold_diversity:
                self._scaffold_history.append(self._murcko(smiles))
            if self.n_calls % max(1, self.config.log_every) == 0:
                tail = ""
                if self.config.use_stamp_gp:
                    tot = self.n_tree_crossovers + self.n_tree_crossover_failures
                    rate = self.n_tree_crossovers / tot if tot > 0 else 0.0
                    tail += (
                        f" gp tc_ok={self.n_tree_crossovers} tc_fail={self.n_tree_crossover_failures} "
                        f"rate={rate:.2f} pool={len(self._mol_pool)}"
                    )
                if self.config.use_scaffold_niching:
                    tail += f" niches={len(self._niches)}"
                logger.info(
                    f"[genmol/{self.task}] calls={self.n_calls}/{budget} "
                    f"best={self.best_reward:.4f} pop_top={self.population[0][0]:.4f} "
                    f"iter={self.iter} xover={self.n_crossovers} mut={self.n_mutations_applied} "
                    f"elapsed={time.time()-t0:.1f}s" + tail
                )
        out: Dict[str, object] = {
            "oracle_log": list(self._oracle_log),
            "trajectory": list(self._traj),
            "n_calls": self.n_calls,
            "iter": self.iter,
            "n_crossovers": self.n_crossovers,
            "n_mutations_applied": self.n_mutations_applied,
            "best_reward": self.best_reward,
            "best_smiles": self.best_smiles,
            "final_population_top": self.population[:10],
            "elapsed_seconds": time.time() - t0,
        }
        if self.config.use_stamp_gp:
            tot = self.n_tree_crossovers + self.n_tree_crossover_failures
            out["stamp_gp_stats"] = {
                "tree_crossovers": int(self.n_tree_crossovers),
                "tree_crossover_failures": int(self.n_tree_crossover_failures),
                "success_rate": (self.n_tree_crossovers / tot) if tot > 0 else 0.0,
                "final_mol_pool_size": len(self._mol_pool),
            }
        if self.config.use_scaffold_niching:
            out["scaffold_niching_stats"] = {
                "n_niches": len(self._niches),
                "top_niches": [
                    {
                        "scaffold": k[:64],
                        "size": len(v),
                        "best": v[0][0] if v else 0.0,
                    }
                    for k, v in sorted(self._niches.items(), key=lambda kv: -kv[1][0][0])[:10]
                ],
            }
        return out

    def _generate_zero_in(self) -> Optional[str]:
        if not self.best_smiles:
            return None
        n_attempts = min(20, self.config.n_attach_tries)
        for _ in range(n_attempts):
            mutated = self._ao_atom_mutator.mutate(self.best_smiles)
            if mutated is None or mutated == self.best_smiles:
                continue
            relaxed_delta = self.config.similarity_guard_delta * 2.0
            if self.config.target_aware and relaxed_delta > 0:
                sim = self._target.parent_child_tanimoto(self.best_smiles, mutated)
                if sim < (1.0 - relaxed_delta):
                    continue
            mol = Chem.MolFromSmiles(mutated)
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            if self.config.min_size <= n_atoms <= self.config.max_size:
                if not self._stamp_filter_ok(mol):
                    continue
                self.n_mutations_applied += 1
                return mutated
        return None

    @staticmethod
    def _murcko(smi: str) -> str:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None: return ""
            return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            return ""

    def _dominant_scaffold(self) -> Optional[str]:
        """Return the dominant scaffold if it exceeds the dominance threshold over the recent window, else None."""
        if not self.config.use_scaffold_diversity:
            return None
        window = self._scaffold_history[-self.config.scaffold_window_calls:]
        if len(window) < 50:
            return None
        from collections import Counter
        counter = Counter(s for s in window if s)
        if not counter:
            return None
        top, count = counter.most_common(1)[0]
        if count / len(window) > self.config.scaffold_dominance_threshold:
            return top
        return None

    def _update_niches(self, smiles: str, score: float) -> None:
        """Maintain Murcko-scaffold-keyed top-K niches.

        Inserts (score, smiles) into the niche keyed by its Murcko scaffold;
        evicts within-niche worst when over capacity, evicts whole-niche
        worst (by best score) when over max_niches.
        """
        scaffold = self._murcko(smiles) or "<no_scaffold>"
        niche = self._niches.setdefault(scaffold, [])
        for i, (s, sm) in enumerate(niche):
            if sm == smiles:
                if score > s:
                    niche[i] = (float(score), smiles)
                    niche.sort(reverse=True)
                return
        niche.append((float(score), smiles))
        niche.sort(reverse=True)
        if len(niche) > self.config.niche_capacity:
            niche.pop()
        # Cap total niches: if over, drop the niche with the lowest top score
        if len(self._niches) > self.config.max_niches:
            worst_key = min(self._niches, key=lambda k: self._niches[k][0][0])
            if worst_key != scaffold:
                del self._niches[worst_key]
            else:
                # The new niche itself is the worst -> drop it
                del self._niches[scaffold]

    def _sample_parents_from_niches(self) -> Optional[Tuple[str, str]]:
        """Pick a (mol_a, mol_b) parent pair from scaffold niches.

        Returns None if niches not yet populated. Otherwise samples 2 mols:
          - with prob niche_intra_prob: both from same niche (within-scaffold evolution)
          - else: one from each of two different niches (cross-scaffold)
        """
        if len(self._niches) < 1:
            return None
        keys = list(self._niches.keys())
        if (
            len(keys) >= 2
            and random.random() >= self.config.niche_intra_prob
        ):
            k1, k2 = random.sample(keys, 2)
            n1 = self._niches[k1]
            n2 = self._niches[k2]
            if not n1 or not n2:
                return None
            return n1[random.randrange(len(n1))][1], n2[random.randrange(len(n2))][1]
        # Within-niche: pick a niche with >= 2 mols
        viable = [k for k in keys if len(self._niches[k]) >= 2]
        if not viable:
            # Fallback: pick any 2 mols across niches
            all_mols = [m for k in keys for _, m in self._niches[k]]
            if len(all_mols) < 2:
                return None
            return tuple(random.sample(all_mols, 2))  # type: ignore
        k = random.choice(viable)
        return tuple(self._niches[k][i][1] for i in random.sample(range(len(self._niches[k])), 2))  # type: ignore

    def _stamp_gp_propose(self) -> Optional[str]:
        """STAMP-GP: pick parent pair from molecule pool by score-softmax and run
        port-signature-aware subtree crossover. Returns the child SMILES or None.

        Genuinely tree-leveraging: the swap respects port_signature (bond order
        + open-port count), so the child is valence-consistent by construction.
        Random within tree_crossover gives different chemistry per call -> brood.
        """
        if len(self._mol_pool) < max(2, self.config.gp_min_pool_size):
            return None
        # Prefer niche-based parent selection when scaffold niching is on.
        m1: Optional[str] = None
        m2: Optional[str] = None
        if self.config.use_scaffold_niching:
            pair = self._sample_parents_from_niches()
            if pair is not None:
                m1, m2 = pair
        if m1 is None or m2 is None:
            if self.config.parent_sampling_tau > 0:
                scores_arr = np.array([s for s, _ in self._mol_pool], dtype=np.float64)
                z = (scores_arr - scores_arr.max()) / max(self.config.parent_sampling_tau, 1e-6)
                w = np.exp(z)
                probs = w / w.sum()
                idx = np.random.choice(len(self._mol_pool), 2, replace=False, p=probs)
                m1 = self._mol_pool[int(idx[0])][1]
                m2 = self._mol_pool[int(idx[1])][1]
            else:
                m1, m2 = random.sample([s for _, s in self._mol_pool], 2)
        child = tree_crossover(
            m1, m2, self.encoder,
            size_delta=self.config.gp_size_delta,
            max_tries=self.config.gp_max_tries,
            match_mode=self.config.gp_match_mode,
            parse_cache=self._stamp_cache,
            n_swaps=self.config.gp_n_swaps,
        )
        if child is None or child == m1 or child == m2:
            return None
        self._last_parents = (m1, m2)  # record crossover parents for trajectory
        return child

    def _generate_one(self) -> Optional[str]:
        """Produce a single candidate molecule via crossover + mutation + filters."""
        if len(self.population) < 2:
            return None
        self._last_op = None
        self._last_parents = None
        used_tree = False
        smi: Optional[str] = None
        # ---- STAMP-GP path: always try port-aware tree crossover first ----
        if self.config.use_stamp_gp:
            child = self._stamp_gp_propose()
            if child is not None:
                smi = child
                used_tree = True
                self._last_op = "xover"
                self.n_tree_crossovers += 1
            else:
                self.n_tree_crossover_failures += 1
        # ---- Probabilistic tree_crossover (legacy path, only if STAMP-GP off) ----
        if (
            smi is None
            and not self.config.use_stamp_gp
            and self.config.tree_crossover_prob > 0
            and random.random() < self.config.tree_crossover_prob
            and len(self._mol_pool) >= 2
        ):
            m1, m2 = random.sample([s for _, s in self._mol_pool], 2)
            child = tree_crossover(m1, m2, self.encoder, size_delta=self.config.tree_crossover_size_delta)
            if child is not None:
                smi = child
                used_tree = True
                self.n_tree_crossovers += 1
            else:
                self.n_tree_crossover_failures += 1
        # ---- Attach fallback ----
        # If STAMP-GP failed (e.g., pool small or no port-compatible swap) and
        # gp_attach_fallback is on, fall through to the original attach GA.
        attach_allowed = (not self.config.use_stamp_gp) or self.config.gp_attach_fallback
        if not used_tree and attach_allowed:
            if self.config.parent_sampling_tau > 0:
                scores_arr = np.array([s for s, _ in self.population], dtype=np.float64)
                z = (scores_arr - scores_arr.max()) / max(self.config.parent_sampling_tau, 1e-6)
                w = np.exp(z)
                probs = w / w.sum()
                idx = np.random.choice(len(self.population), 2, replace=False, p=probs)
                f1 = self.population[int(idx[0])][1]
                f2 = self.population[int(idx[1])][1]
            else:
                frags = [f for _, f in self.population]
                f1, f2 = random.sample(frags, 2)
            mol = attach(f1, f2)
            if mol is None:
                return None
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                return None
            self._last_op = "attach"
            self.n_crossovers += 1
        if smi is None:
            return None
        do_mutate = (
            self.iter > self.config.warmup
            and random.random() < self.config.mutation_prob
        )
        if do_mutate:
            parent_for_sim = smi
            mutated = self._apply_mutation(smi)
            if mutated is not None and mutated != smi:
                accept = True
                if self.config.target_aware and self._target.similarity_guard_delta > 0:
                    accept = self._target.accept_mutation(parent_for_sim, mutated)
                if accept:
                    smi = mutated
                    self._last_op = (self._last_op or "?") + "+mut"
                    self.n_mutations_applied += 1
        if "." in smi:
            smi = max(smi.split("."), key=len)
        check_mol = Chem.MolFromSmiles(smi)
        if check_mol is None:
            return None
        n_atoms = check_mol.GetNumAtoms()
        if not (self.config.min_size <= n_atoms <= self.config.max_size):
            return None
        if not self._stamp_filter_ok(check_mol):
            return None
        return smi

    def _generate(self) -> Optional[str]:
        if self.config.target_aware and self._target.in_zero_in_phase(self.n_calls):
            out = self._generate_zero_in()
            if out is not None:
                return out
        # P2: scaffold-diversity divert
        dominant = self._dominant_scaffold()
        force_diff = (
            dominant is not None
            and random.random() < self.config.scaffold_force_diff_rate
        )
        if force_diff:
            self.n_scaffold_diff_forced += 1
            for _ in range(self.config.scaffold_force_diff_max_tries):
                cand = self._generate_one()
                if cand is None:
                    continue
                if self._murcko(cand) != dominant:
                    self.n_scaffold_diff_succeeded += 1
                    return cand
            # fall through if we couldn't find a different-scaffold candidate
        # Decide K candidates (brood size).
        #   STAMP-GP on  : K = gp_brood_size (independent of surrogate readiness)
        #   surrogate on : K = surrogate_k_candidates once surrogate is_ready
        #   otherwise    : K = 1
        K = 1
        if self.config.use_stamp_gp:
            K = max(1, int(self.config.gp_brood_size))
        elif self._surrogate_ranker is not None and self._surrogate_ranker.is_ready:
            K = max(1, int(self.config.surrogate_k_candidates))
        candidates: List[str] = []
        attempts = 0
        max_attempts = self.config.n_attach_tries
        while len(candidates) < K and attempts < max_attempts:
            attempts += 1
            cand = self._generate_one()
            if cand is None:
                continue
            candidates.append(cand)
        if not candidates:
            return None
        if K == 1 or self._surrogate_ranker is None or not self._surrogate_ranker.is_ready:
            return candidates[0]
        # M11: surrogate-rank candidates
        scored = [(c, self._surrogate_ranker.predict(c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        self.n_surrogate_picks += 1
        return scored[0][0]

    def _apply_one_mutation(self, smi: str) -> Optional[str]:
        mode = self._mutation_mode
        if mode == "ao_gpt":
            return self._stamp_mutator.mutate(smi)
        if mode == "ao_gpt_multi":
            if self._multi_span_mutator is None:
                return self._stamp_mutator.mutate(smi)
            return self._multi_span_mutator.mutate(smi)
        if mode == "mixed_multi":
            if self._multi_span_mutator is not None and random.random() < 0.5:
                return self._multi_span_mutator.mutate(smi)
            return self._stamp_mutator.mutate(smi)
        if mode == "ao_atom":
            if self._ao_atom_mutator is None:
                return self._stamp_mutator.mutate(smi)
            return self._ao_atom_mutator.mutate(smi)
        if mode == "selfies":
            if self._selfies_mutator is None:
                return self._stamp_mutator.mutate(smi)
            return self._selfies_mutator.mutate(smi)
        if mode == "mixed":
            if random.random() < 0.5 or self._selfies_mutator is None:
                return self._stamp_mutator.mutate(smi)
            return self._selfies_mutator.mutate(smi)
        if mode == "mixed_ao":
            if random.random() < 0.5 or self._ao_atom_mutator is None:
                return self._stamp_mutator.mutate(smi)
            return self._ao_atom_mutator.mutate(smi)
        return self._stamp_mutator.mutate(smi)

    def _apply_mutation(self, smi: str) -> Optional[str]:
        """Apply mutation. If chain_steps > 1, mutate K times sequentially.

        Each chain step takes the previous step's output. If any step returns
        None or fails to validate, the chain breaks early and we return the
        last successful result.
        """
        chain_n = max(1, int(self.config.chain_steps))
        cur = smi
        for _ in range(chain_n):
            mutated = self._apply_one_mutation(cur)
            if mutated is None:
                break
            cur = mutated
        return cur

    def _update_population(self, smiles: str, prop: float) -> None:
        """f-RAG-style fragment library update.

        For every fragment cut from ``smiles``:
          - If already in population, raise its score to ``prop`` if higher (max-update).
          - Else, insert it.
        Existing fragments' scores get credited from every higher-scoring molecule
        they appear in, propagating oracle signal back to the fragment vocabulary.
        """
        new_frags = cut(smiles)
        if not new_frags:
            return
        # Dict for O(1) lookup; preserves last-wins on duplicate keys (cut() returns a set).
        pop_dict: Dict[str, float] = {f: s for s, f in self.population}
        changed = False
        for frag in new_frags:
            cur = pop_dict.get(frag, -1.0)
            if prop > cur:
                pop_dict[frag] = float(prop)
                changed = True
        if not changed:
            return
        combined = [(s, f) for f, s in pop_dict.items()]
        if self.config.novelty_weight > 0:
            scorer = NoveltyScorer(k_neighbors=self.config.novelty_k_neighbors)
            ranked = rank_by_composite(
                combined,
                self.config.novelty_weight,
                scorer,
                pop_gap_tau=self.config.novelty_gap_tau,
            )
        else:
            ranked = sorted(combined, reverse=True)
        self.population = ranked[: self.config.population_size]

    def _update_mol_pool(self, smiles: str, score: float) -> None:
        for i, (s, sm) in enumerate(self._mol_pool):
            if sm == smiles:
                if score > s:
                    self._mol_pool[i] = (float(score), smiles)
                    self._mol_pool.sort(reverse=True)
                return
        self._mol_pool.append((float(score), smiles))
        self._mol_pool.sort(reverse=True)
        if len(self._mol_pool) > self.config.molecule_pool_size:
            self._mol_pool = self._mol_pool[: self.config.molecule_pool_size]

    def _already_scored(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not hasattr(self, "_canon_seen"):
            self._canon_seen: set = set()
        if canon in self._canon_seen:
            return True
        self._canon_seen.add(canon)
        return False

#!/usr/bin/env python3
"""Run a lightweight STAMP motif-planner baseline on PMO/TDC tasks.

This is intentionally simpler than MCTS.  The state is always a complete
molecule represented by a STAMP token sequence.  The planner maintains an
online motif-group bank from scored molecules, proposes local motif edits,
scores each edit with explicit reason features, executes the best candidate,
and updates the population from oracle feedback.

The first version is designed as a runnable baseline scaffold.  It supports
an optional pre-scored seed CSV, but can also bootstrap entirely from LM
samples when a ZINC250k-derived bank is not ready yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.mcts_guacamol import (  # noqa: E402
    ARPolicyModel,
    STAMPDecoderImpl,
    count_open_ports,
    parse_next_group,
)
from stamp.benchmark.mcts_unified.pmo_oracle import (  # noqa: E402
    PMO_TASKS,
    PMOOracle,
    auc_top_k_from_log,
)


@dataclass
class MoleculeRecord:
    smiles: str
    tokens: List[str]
    score: float
    source: str
    step: int


@dataclass
class UnitStats:
    tokens: Tuple[str, ...]
    support: int = 0
    total_score: float = 0.0
    best_score: float = 0.0
    seen_count: int = 0

    @property
    def mean_score(self) -> float:
        return self.total_score / max(1, self.support)


@dataclass
class CandidatePlan:
    action: str
    parent_smiles: str
    parent_score: float
    new_tokens: List[str]
    old_unit: Tuple[str, ...] = ()
    new_unit: Tuple[str, ...] = ()
    reason: Dict[str, Any] = field(default_factory=dict)
    plan_score: float = 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def clean_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if not tok or tok.startswith("<"):
            continue
        out.append(tok)
        if tok == "[END]":
            break
    return out


def split_motif_groups(tokens: Sequence[str]) -> List[Tuple[int, int, Tuple[str, ...]]]:
    cleaned = clean_tokens(tokens)
    groups: List[Tuple[int, int, Tuple[str, ...]]] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "[END]":
            break
        end = parse_next_group(cleaned, i)
        if end is None or end <= i:
            i += 1
            continue
        groups.append((i, end, tuple(cleaned[i:end])))
        i = end
    return groups


def truncate_to_group_count(tokens: Sequence[str], max_groups: int) -> List[str]:
    if max_groups <= 0:
        return []
    groups = split_motif_groups(tokens)
    if not groups:
        return []
    end = groups[min(len(groups), max_groups) - 1][1]
    return clean_tokens(tokens[:end])


def joint_count(unit: Sequence[str]) -> int:
    return sum(1 for t in unit if t.startswith("[J_"))


def token_size_penalty(tokens: Sequence[str], target_len: int, max_len: int) -> float:
    if max_len <= 0:
        return 0.0
    return abs(len(tokens) - target_len) / max(1, max_len)


def decode_tokens(decoder: STAMPDecoderImpl, tokens: Sequence[str]) -> Optional[str]:
    cleaned = clean_tokens(tokens)
    if not cleaned:
        return None
    if "[END]" not in cleaned:
        cleaned = cleaned + ["[END]"]
    return canonicalize_smiles(decoder.decode_to_smiles(cleaned) or "")


class MotifPlanner:
    def __init__(
        self,
        *,
        decoder: STAMPDecoderImpl,
        policy: ARPolicyModel,
        target_len: int,
        max_seq_len: int,
        beam_size: int,
        candidates_per_step: int,
        random_action_rate: float,
        action_types: Sequence[str],
        refine_probability: float,
        refine_patch_max_groups: int,
        refine_min_candidates: int,
        refine_step_candidates: int,
        replace_min_fraction: float,
        max_consecutive_grows: int,
        low_contrib_fraction: float,
        replace_prior_bonus: float,
    ) -> None:
        self.decoder = decoder
        self.policy = policy
        self.target_len = int(target_len)
        self.max_seq_len = int(max_seq_len)
        self.beam_size = int(beam_size)
        self.candidates_per_step = int(candidates_per_step)
        self.random_action_rate = float(random_action_rate)
        self.action_types = tuple(action_types)
        self.refine_probability = float(refine_probability)
        self.refine_patch_max_groups = int(refine_patch_max_groups)
        self.refine_min_candidates = int(refine_min_candidates)
        self.refine_step_candidates = int(refine_step_candidates)
        self.replace_min_fraction = float(replace_min_fraction)
        self.max_consecutive_grows = int(max_consecutive_grows)
        self.low_contrib_fraction = float(low_contrib_fraction)
        self.replace_prior_bonus = float(replace_prior_bonus)
        self.unit_bank: Dict[Tuple[str, ...], UnitStats] = {}
        self.edit_deltas: Dict[str, List[float]] = {}
        self._consecutive_grows = 0

    def observe(self, rec: MoleculeRecord) -> None:
        for _start, _end, unit in split_motif_groups(rec.tokens):
            stat = self.unit_bank.setdefault(unit, UnitStats(tokens=unit))
            stat.support += 1
            stat.total_score += float(rec.score)
            stat.best_score = max(stat.best_score, float(rec.score))

    def load_offline_bank(self, bank_dir: str | Path, *, top_units: int) -> Dict[str, int]:
        bank = Path(bank_dir)
        counts: Dict[str, int] = {}
        loaded_total = 0
        for name in ("motifs.csv", "subtrees.csv"):
            path = bank / name
            if not path.exists():
                continue
            unit_type = name.removesuffix(".csv")
            count = 0
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    toks = tuple((row.get("tokens") or "").split())
                    if not toks:
                        continue
                    try:
                        support = int(float(row.get("support") or 1))
                        mean = float(row.get("mean_score") or 0.0)
                        best = float(row.get("best_score") or mean)
                    except ValueError:
                        continue
                    stat = self.unit_bank.setdefault(toks, UnitStats(tokens=toks))
                    # Treat offline bank statistics as pretest evidence, not
                    # PMO-loop oracle calls.
                    stat.support += max(1, support)
                    stat.total_score += mean * max(1, support)
                    stat.best_score = max(stat.best_score, best)
                    count += 1
                    loaded_total += 1
                    if top_units > 0 and count >= top_units:
                        break
            counts[unit_type] = count
        counts["total"] = loaded_total
        return counts

    def update_delta(self, action: str, unit: Sequence[str], delta: float) -> None:
        key = self.action_signature(action, unit)
        vals = self.edit_deltas.setdefault(key, [])
        vals.append(float(delta))
        if len(vals) > 128:
            del vals[:-128]

    @staticmethod
    def action_signature(action: str, unit: Sequence[str]) -> str:
        head = next((t for t in unit if not t.startswith("[")), "")
        return f"{action}:{head}"

    def _sample_units(self, n: int) -> List[UnitStats]:
        if not self.unit_bank:
            return []
        stats = list(self.unit_bank.values())
        weights = np.array(
            [
                max(1e-4, s.mean_score)
                * math.sqrt(max(1, s.support))
                / math.sqrt(1 + s.seen_count)
                for s in stats
            ],
            dtype=float,
        )
        weights = weights / max(1e-12, float(weights.sum()))
        idx = np.random.choice(len(stats), size=min(n, len(stats)), replace=False, p=weights)
        return [stats[int(i)] for i in idx]

    def unit_prior(self, unit: Sequence[str]) -> float:
        stat = self.unit_bank.get(tuple(unit))
        if stat is None:
            return 0.0
        return float(stat.mean_score)

    def _low_contrib_groups(
        self,
        groups: Sequence[Tuple[int, int, Tuple[str, ...]]],
    ) -> List[Tuple[int, int, Tuple[str, ...]]]:
        if not groups:
            return []
        ranked = sorted(groups, key=lambda g: (self.unit_prior(g[2]), len(g[2])))
        keep = max(1, math.ceil(len(ranked) * self.low_contrib_fraction))
        return ranked[:keep]

    def _scheduled_actions(self) -> List[str]:
        action_choices = list(self.action_types)
        if "refine" in action_choices:
            if random.random() < self.refine_probability:
                n_refine = max(1, min(self.candidates_per_step, self.refine_step_candidates))
                return ["refine"] * n_refine
            action_choices = [a for a in action_choices if a != "refine"]
        if not action_choices:
            return []

        no_grow = self.max_consecutive_grows >= 0 and self._consecutive_grows >= self.max_consecutive_grows
        if no_grow:
            action_choices = [a for a in action_choices if a != "grow"]
        if not action_choices:
            action_choices = ["replace"]

        counts: Dict[str, int] = {a: 0 for a in action_choices}
        total = max(1, self.candidates_per_step)
        replace_floor = 0
        if "replace" in counts:
            replace_floor = max(1, math.ceil(total * self.replace_min_fraction))
            counts["replace"] = min(total, replace_floor)
        if "refine" in counts and self.refine_min_candidates > 0:
            counts["refine"] = min(
                total - sum(v for k, v in counts.items() if k != "refine"),
                max(counts["refine"], self.refine_min_candidates),
            )

        remaining = total - sum(counts.values())
        fill_order = [a for a in ("grow", "replace", "shrink", "refine") if a in counts]
        if no_grow:
            fill_order = [a for a in fill_order if a != "grow"] + (["grow"] if "grow" in counts else [])
        while remaining > 0 and fill_order:
            for action in fill_order:
                if remaining <= 0:
                    break
                if action == "grow" and no_grow:
                    continue
                counts[action] += 1
                remaining -= 1

        scheduled: List[str] = []
        for action, count in counts.items():
            scheduled.extend([action] * count)
        random.shuffle(scheduled)
        return scheduled[:total]

    def record_action(self, action: str) -> None:
        if action == "grow":
            self._consecutive_grows += 1
        else:
            self._consecutive_grows = 0

    def propose(self, parent: MoleculeRecord) -> List[CandidatePlan]:
        groups = split_motif_groups(parent.tokens)
        units = self._sample_units(max(self.candidates_per_step, self.beam_size))
        if not groups or not units:
            return []

        plans: List[CandidatePlan] = []
        low_contrib_groups = self._low_contrib_groups(groups)
        scheduled_actions = self._scheduled_actions()
        if not scheduled_actions:
            return []
        for action in scheduled_actions:
            if action == "replace":
                target_groups = low_contrib_groups or list(groups)
                plan = self._propose_replace(parent, target_groups, random.choice(units))
            elif action == "grow":
                plan = self._propose_grow(parent, groups, random.choice(units))
            elif action == "shrink":
                target_groups = low_contrib_groups or list(groups)
                plan = self._propose_shrink(parent, target_groups)
            else:
                target_groups = low_contrib_groups or list(groups)
                plan = self._propose_refine(parent, target_groups)
            if plan is not None:
                self._score_plan(plan)
                plans.append(plan)

        plans.sort(key=lambda p: -p.plan_score)
        if random.random() < self.random_action_rate:
            random.shuffle(plans)
        return plans[: self.beam_size]

    def _propose_replace(
        self,
        parent: MoleculeRecord,
        groups: Sequence[Tuple[int, int, Tuple[str, ...]]],
        unit_stat: UnitStats,
    ) -> Optional[CandidatePlan]:
        start, end, old = random.choice(list(groups))
        new = list(parent.tokens[:start]) + list(unit_stat.tokens) + list(parent.tokens[end:])
        if len(new) > self.max_seq_len:
            return None
        return CandidatePlan(
            action="replace",
            parent_smiles=parent.smiles,
            parent_score=parent.score,
            new_tokens=new,
            old_unit=old,
            new_unit=unit_stat.tokens,
        )

    def _propose_grow(
        self,
        parent: MoleculeRecord,
        groups: Sequence[Tuple[int, int, Tuple[str, ...]]],
        unit_stat: UnitStats,
    ) -> Optional[CandidatePlan]:
        boundaries = [end for _start, end, _unit in groups]
        insert_at = random.choice(boundaries)
        new = list(parent.tokens[:insert_at]) + list(unit_stat.tokens) + list(parent.tokens[insert_at:])
        if len(new) > self.max_seq_len:
            return None
        return CandidatePlan(
            action="grow",
            parent_smiles=parent.smiles,
            parent_score=parent.score,
            new_tokens=new,
            new_unit=unit_stat.tokens,
        )

    def _propose_shrink(
        self,
        parent: MoleculeRecord,
        groups: Sequence[Tuple[int, int, Tuple[str, ...]]],
    ) -> Optional[CandidatePlan]:
        if len(groups) <= 1:
            return None
        start, end, old = random.choice(list(groups[1:] or groups))
        new = list(parent.tokens[:start]) + list(parent.tokens[end:])
        return CandidatePlan(
            action="shrink",
            parent_smiles=parent.smiles,
            parent_score=parent.score,
            new_tokens=new,
            old_unit=old,
        )

    def _propose_refine(
        self,
        parent: MoleculeRecord,
        groups: Sequence[Tuple[int, int, Tuple[str, ...]]],
    ) -> Optional[CandidatePlan]:
        start, end, old = random.choice(list(groups))
        prefix = clean_tokens(parent.tokens[:start])
        suffix = clean_tokens(parent.tokens[end:])
        sampled = self.policy._sample(
            prefix,
            n_samples=1,
            max_new_tokens=max(8, self.refine_patch_max_groups * 8),
            temperature=self.policy.rollout_temperature,
            top_k=self.policy.rollout_top_k,
        )
        if not sampled:
            return None
        sample_tokens = clean_tokens(sampled[0])
        if len(sample_tokens) < len(prefix):
            return None
        patch = truncate_to_group_count(sample_tokens[len(prefix):], self.refine_patch_max_groups)
        if not patch:
            return None
        new = prefix + patch + suffix
        if len(new) > self.max_seq_len:
            return None
        return CandidatePlan(
            action="refine",
            parent_smiles=parent.smiles,
            parent_score=parent.score,
            new_tokens=new,
            old_unit=old,
            new_unit=tuple(patch),
        )

    def _score_plan(self, plan: CandidatePlan) -> None:
        new_stat = self.unit_bank.get(plan.new_unit)
        old_stat = self.unit_bank.get(plan.old_unit)
        oracle_prior = float(new_stat.mean_score) if new_stat is not None else 0.0
        old_prior = float(old_stat.mean_score) if old_stat is not None else 0.0
        support = int(new_stat.support) if new_stat is not None else 0
        novelty = 1.0 / math.sqrt(1.0 + (new_stat.seen_count if new_stat is not None else 0))
        expected_delta_vals = self.edit_deltas.get(
            self.action_signature(plan.action, plan.new_unit or plan.old_unit),
            [],
        )
        expected_delta = float(np.mean(expected_delta_vals[-16:])) if expected_delta_vals else 0.0
        if plan.action == "replace":
            port_compatible = float(joint_count(plan.old_unit) == joint_count(plan.new_unit))
        elif plan.action == "refine":
            port_compatible = float(joint_count(plan.old_unit) == joint_count(plan.new_unit or plan.old_unit))
        elif plan.action == "shrink":
            port_compatible = 0.5
        else:
            port_compatible = 0.75
        open_ports = count_open_ports(clean_tokens(plan.new_tokens))
        port_penalty = max(0.0, float(open_ports)) * 0.05 if open_ports >= 0 else 0.1
        size_penalty = token_size_penalty(plan.new_tokens, self.target_len, self.max_seq_len)
        if plan.action == "replace":
            prior_gain = oracle_prior - old_prior
            if prior_gain > 0:
                prior_gain += self.replace_prior_bonus
        elif plan.action == "grow":
            # Growth is useful but easy to over-exploit because it imports a
            # high-prior unit without proving the current site needs more mass.
            prior_gain = 0.55 * oracle_prior
        elif plan.action == "shrink":
            prior_gain = max(0.0, 0.75 - old_prior)
        else:
            prior_gain = max(0.1, 0.8 - old_prior) + 0.25 * oracle_prior

        plan.reason = {
            "oracle_prior": oracle_prior,
            "old_oracle_prior": old_prior,
            "prior_gain": prior_gain,
            "expected_delta": expected_delta,
            "port_compatible": bool(port_compatible >= 0.75),
            "port_compatibility_score": port_compatible,
            "open_ports_estimate": int(open_ports),
            "support": support,
            "novelty": novelty,
            "size_penalty": size_penalty,
            "port_penalty": port_penalty,
        }
        plan.plan_score = (
            1.5 * prior_gain
            + 1.0 * expected_delta
            + 0.5 * port_compatible
            + 0.25 * novelty
            - 0.6 * size_penalty
            - port_penalty
        )
        if plan.action == "replace":
            plan.plan_score += 0.15
        elif plan.action == "refine":
            plan.plan_score += 0.1
        elif plan.action == "grow" and self._consecutive_grows > 0:
            plan.plan_score -= min(0.4, 0.08 * self._consecutive_grows)

    def mark_used(self, unit: Sequence[str]) -> None:
        stat = self.unit_bank.get(tuple(unit))
        if stat is not None:
            stat.seen_count += 1


def select_parent(population: Sequence[MoleculeRecord], top_k: int, tau: float) -> MoleculeRecord:
    pool = sorted(population, key=lambda r: -r.score)[: max(1, top_k)]
    scores = np.array([r.score for r in pool], dtype=float)
    scores = scores - float(scores.max())
    weights = np.exp(scores / max(1e-6, tau))
    weights = weights / max(1e-12, float(weights.sum()))
    idx = int(np.random.choice(len(pool), p=weights))
    return pool[idx]


def sample_bank_seed(
    planner: MotifPlanner,
    *,
    max_groups: int,
    max_seq_len: int,
) -> Optional[List[str]]:
    units = planner._sample_units(max(8, max_groups * 4))
    if not units:
        return None
    tokens: List[str] = []
    n_groups = random.randint(1, max(1, max_groups))
    for _ in range(n_groups):
        pick = random.choice(units)
        candidate = tokens + list(pick.tokens)
        if len(candidate) > max_seq_len:
            break
        tokens = candidate
    return tokens or None


def score_if_new(
    *,
    smiles: str,
    oracle: PMOOracle,
    memory: Dict[str, float],
    max_calls: int,
) -> Optional[float]:
    canon = canonicalize_smiles(smiles)
    if canon is None:
        return None
    if canon in memory:
        return None
    if oracle.n_calls >= max_calls:
        return None
    score = float(oracle(canon))
    memory[canon] = score
    return score


def add_record(
    *,
    smiles: str,
    tokens: Sequence[str],
    score: float,
    source: str,
    step: int,
    population: List[MoleculeRecord],
    planner: MotifPlanner,
    max_population: int,
) -> MoleculeRecord:
    rec = MoleculeRecord(
        smiles=smiles,
        tokens=clean_tokens(tokens),
        score=float(score),
        source=source,
        step=int(step),
    )
    population.append(rec)
    population.sort(key=lambda r: -r.score)
    del population[max_population:]
    planner.observe(rec)
    return rec


def load_seed_smiles(path: Optional[str], smiles_column: str, max_rows: int) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    rows: List[str] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = smiles_column in sample.splitlines()[0].split(",")
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                smi = (row.get(smiles_column) or "").strip()
                if smi:
                    rows.append(smi)
                if len(rows) >= max_rows:
                    break
        else:
            for line in f:
                smi = line.strip().split(",")[0]
                if smi:
                    rows.append(smi)
                if len(rows) >= max_rows:
                    break
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run STAMP motif-planner baseline on PMO/TDC tasks.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--task", default="qed", choices=PMO_TASKS)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--seed-smiles", default=None,
                   help="Optional CSV/SMI file used to bootstrap a pretested bank.")
    p.add_argument("--bank-dir", default=None,
                   help="Offline STAMP PMO bank directory produced by build_stamp_pmo_bank.py.")
    p.add_argument("--bank-top-units", type=int, default=2000,
                   help="Maximum units to preload per bank file. <=0 loads all.")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--max-seed-smiles", type=int, default=256)
    p.add_argument("--init-samples", type=int, default=32)
    p.add_argument("--population-size", type=int, default=128)
    p.add_argument("--parent-top-k", type=int, default=32)
    p.add_argument("--parent-tau", type=float, default=0.08)
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--candidates-per-step", type=int, default=32)
    p.add_argument("--action-types", default="grow,replace,shrink",
                   help="Comma-separated planner actions. Keep refine low-frequency.")
    p.add_argument("--refine-probability", type=float, default=0.05,
                   help="Probability of enabling refine in a planning step when action-types includes it.")
    p.add_argument("--refine-patch-max-groups", type=int, default=2,
                   help="Maximum number of motif groups rewritten by one refine patch.")
    p.add_argument("--refine-min-candidates", type=int, default=1,
                   help="Minimum number of refine proposals when refine is enabled for a step.")
    p.add_argument("--refine-step-candidates", type=int, default=2,
                   help="Number of refine candidates proposed in a dedicated refine step.")
    p.add_argument("--replace-min-fraction", type=float, default=0.5,
                   help="Minimum fraction of proposals reserved for replace actions.")
    p.add_argument("--max-consecutive-grows", type=int, default=4,
                   help="After this many accepted grow actions in a row, grow proposals are suppressed.")
    p.add_argument("--low-contrib-fraction", type=float, default=0.35,
                   help="Fraction of lowest-prior motif groups targeted first for replace/shrink.")
    p.add_argument("--replace-prior-bonus", type=float, default=0.2,
                   help="Extra plan-score bonus for beneficial replace actions.")
    p.add_argument("--random-action-rate", type=float, default=0.05)
    p.add_argument("--explore-sample-rate", type=float, default=0.25,
                   help="Probability of spending a step on a fresh LM sample "
                        "instead of an edit plan. Keeps PMO top-k diversity.")
    p.add_argument("--bank-init-max-groups", type=int, default=3,
                   help="Use offline bank units to build 1..N-group seed molecules before any LM init.")
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--target-len", type=int, default=48)
    p.add_argument("--max-seq-len", type=int, default=160)
    p.add_argument("--proposal-samples", type=int, default=8)
    p.add_argument("--proposal-top-k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-attempts-factor", type=int, default=80)
    p.add_argument("--max-plan-attempts-per-step", type=int, default=2,
                   help="Limit how many ranked plans are tried before moving on.")
    p.add_argument("--log-every", type=int, default=50)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = STAMPDecoderImpl()
    policy = ARPolicyModel(
        args.model_dir,
        device=args.device,
        proposal_samples=args.proposal_samples,
        proposal_temperature=args.temperature,
        proposal_top_k=args.proposal_top_k,
        rollout_temperature=args.temperature,
        rollout_top_k=args.proposal_top_k,
        rollout_new_tokens=args.rollout_steps * 6,
    )
    oracle = PMOOracle(args.task, max_calls=args.budget)
    planner = MotifPlanner(
        decoder=decoder,
        policy=policy,
        target_len=args.target_len,
        max_seq_len=args.max_seq_len,
        beam_size=args.beam_size,
        candidates_per_step=args.candidates_per_step,
        random_action_rate=args.random_action_rate,
        action_types=[x.strip() for x in args.action_types.split(",") if x.strip()],
        refine_probability=args.refine_probability,
        refine_patch_max_groups=args.refine_patch_max_groups,
        refine_min_candidates=args.refine_min_candidates,
        refine_step_candidates=args.refine_step_candidates,
        replace_min_fraction=args.replace_min_fraction,
        max_consecutive_grows=args.max_consecutive_grows,
        low_contrib_fraction=args.low_contrib_fraction,
        replace_prior_bonus=args.replace_prior_bonus,
    )
    bank_counts: Dict[str, int] = {}
    if args.bank_dir:
        bank_counts = planner.load_offline_bank(args.bank_dir, top_units=args.bank_top_units)

    memory: Dict[str, float] = {}
    population: List[MoleculeRecord] = []
    plan_log: List[Dict[str, Any]] = []
    t0 = time.time()

    # Optional pretested seed molecules.  These oracle calls are deliberately
    # counted in this generic runner; for GenMol-aligned pretest, build and
    # pass a persisted bank in a later script.
    for smi in load_seed_smiles(args.seed_smiles, args.smiles_column, args.max_seed_smiles):
        if oracle.n_calls >= min(args.budget, args.init_samples):
            break
        try:
            tokens = decoder.encoder.encode_smiles(smi).split()
        except Exception:
            continue
        canon = decode_tokens(decoder, tokens)
        if canon is None:
            continue
        score = score_if_new(smiles=canon, oracle=oracle, memory=memory, max_calls=args.budget)
        if score is None:
            continue
        add_record(
            smiles=canon,
            tokens=tokens,
            score=score,
            source="seed_smiles",
            step=oracle.n_calls,
            population=population,
            planner=planner,
            max_population=args.population_size,
        )

    attempts = 0
    max_attempts = max(args.budget * args.max_attempts_factor, args.init_samples * 10)

    # Prefer deterministic bank-based construction for warm start.
    while (
        oracle.n_calls < min(args.budget, args.init_samples)
        and attempts < max_attempts
        and planner.unit_bank
    ):
        attempts += 1
        tokens = sample_bank_seed(
            planner,
            max_groups=args.bank_init_max_groups,
            max_seq_len=args.max_seq_len,
        )
        if not tokens:
            break
        smiles = decode_tokens(decoder, tokens)
        if smiles is None:
            continue
        score = score_if_new(smiles=smiles, oracle=oracle, memory=memory, max_calls=args.budget)
        if score is None:
            continue
        add_record(
            smiles=smiles,
            tokens=tokens,
            score=score,
            source="bank_init",
            step=oracle.n_calls,
            population=population,
            planner=planner,
            max_population=args.population_size,
        )
        if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
            print(
                f"[planner] calls={oracle.n_calls}/{args.budget} best={max(r.score for r in population):.4f} "
                f"pop={len(population)} mode=bank_init",
                flush=True,
            )

    # Fallback LM init only if bank-based init was insufficient.
    while oracle.n_calls < min(args.budget, args.init_samples) and attempts < max_attempts:
        attempts += 1
        tokens = policy.rollout([], max_steps=args.rollout_steps)
        smiles = decode_tokens(decoder, tokens)
        if smiles is None:
            continue
        score = score_if_new(smiles=smiles, oracle=oracle, memory=memory, max_calls=args.budget)
        if score is None:
            continue
        add_record(
            smiles=smiles,
            tokens=tokens,
            score=score,
            source="lm_init",
            step=oracle.n_calls,
            population=population,
            planner=planner,
            max_population=args.population_size,
        )
        if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
            print(
                f"[planner] calls={oracle.n_calls}/{args.budget} best={max(r.score for r in population):.4f} "
                f"pop={len(population)} mode=lm_init",
                flush=True,
            )

    if not population:
        raise RuntimeError("Failed to initialize any valid scored molecule.")

    attempts = 0
    while oracle.n_calls < args.budget and attempts < max_attempts:
        attempts += 1
        if random.random() < args.explore_sample_rate:
            tokens = policy.rollout([], max_steps=args.rollout_steps)
            smiles = decode_tokens(decoder, tokens)
            if smiles is None:
                continue
            score = score_if_new(smiles=smiles, oracle=oracle, memory=memory, max_calls=args.budget)
            if score is None:
                continue
            add_record(
                smiles=smiles,
                tokens=tokens,
                score=score,
                source="lm_explore",
                step=oracle.n_calls,
                population=population,
                planner=planner,
                max_population=args.population_size,
            )
            if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
                print(
                    f"[planner] calls={oracle.n_calls}/{args.budget} best={max(r.score for r in population):.4f} "
                    f"pop={len(population)} mode=lm_explore",
                    flush=True,
                )
            continue

        parent = select_parent(population, top_k=args.parent_top_k, tau=args.parent_tau)
        plans = planner.propose(parent)
        if not plans:
            continue

        accepted = False
        tried = 0
        for plan in plans:
            if tried >= args.max_plan_attempts_per_step:
                break
            tried += 1
            smiles = decode_tokens(decoder, plan.new_tokens)
            if smiles is None:
                continue
            score = score_if_new(smiles=smiles, oracle=oracle, memory=memory, max_calls=args.budget)
            if score is None:
                continue
            delta = float(score - parent.score)
            planner.update_delta(plan.action, plan.new_unit or plan.old_unit, delta)
            planner.mark_used(plan.new_unit)
            planner.record_action(plan.action)
            rec = add_record(
                smiles=smiles,
                tokens=plan.new_tokens,
                score=score,
                source=f"plan:{plan.action}",
                step=oracle.n_calls,
                population=population,
                planner=planner,
                max_population=args.population_size,
            )
            plan_log.append(
                {
                    "oracle_call": oracle.n_calls,
                    "action": plan.action,
                    "parent_smiles": parent.smiles,
                    "parent_score": parent.score,
                    "child_smiles": rec.smiles,
                    "child_score": rec.score,
                    "delta": delta,
                    "plan_score": plan.plan_score,
                    "old_unit": list(plan.old_unit),
                    "new_unit": list(plan.new_unit),
                    "reason": plan.reason,
                }
            )
            accepted = True
            if args.log_every > 0 and oracle.n_calls % args.log_every == 0:
                print(
                    f"[planner] calls={oracle.n_calls}/{args.budget} best={max(r.score for r in population):.4f} "
                    f"pop={len(population)} action={plan.action}",
                    flush=True,
                )
            break
        if not accepted:
            continue

    metrics = auc_top_k_from_log(oracle._log, k=10, max_calls=args.budget)
    best = max(population, key=lambda r: r.score)
    payload = {
        "method": "stamp_motif_planner",
        "task": args.task,
        "budget": args.budget,
        "seed": args.seed,
        "model_dir": args.model_dir,
        "device": args.device,
        "elapsed_seconds": time.time() - t0,
        "oracle_calls": oracle.n_calls,
        "n_unique_smiles": len(memory),
        "best_reward": best.score,
        "best_smiles": best.smiles,
        "pmo_auc_top10": metrics,
        "config": vars(args),
        "offline_bank": {
            "bank_dir": args.bank_dir,
            "loaded_counts": bank_counts,
        },
        "top_population": [
            {
                "smiles": r.smiles,
                "score": r.score,
                "source": r.source,
                "step": r.step,
            }
            for r in sorted(population, key=lambda x: -x.score)[:25]
        ],
        "plan_log": plan_log,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "best_reward": payload["best_reward"],
                "best_smiles": payload["best_smiles"],
                "auc_top10": metrics["auc_top_k"],
                "final_top10": metrics["final_top_k_mean"],
                "oracle_calls": oracle.n_calls,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

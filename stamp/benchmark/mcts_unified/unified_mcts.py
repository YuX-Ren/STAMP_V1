"""UnifiedMCTS: growth + edit actions on the same MCTS tree.

Inherits from :class:`mcts_protype.MCTS` and overrides:

- ``_expand``   — edit actions are spliced into the parent state rather than
                  appended.  A tree-node attribute ``edit_baseline`` stores
                  the parent's terminal reward for baseline-subtracted PUCT.
- ``_simulate`` — consults a shared :class:`MoleculeMemory` cache before
                  calling the oracle; duplicates still bump ``visit_count``
                  but don't burn oracle budget.
- ``_puct_score`` — for edit-subtree nodes, the exploitation term is
                    ``max(0, Q − edit_baseline)`` so edits compete against
                    their parent's reward, not against absolute zero.

The tree structure otherwise remains PUCT + progressive widening as in the
Phase-1 prototype.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..mcts_guacamol import STAMPDecoderImpl  # type: ignore
from .memory import MoleculeMemory
from .subtree import parse_terminal_sequence
from .subtree_pool import SubtreePool
from .transplant import (
    decode_transplant_action,
    extract_donor_records,
    is_transplant_action,
)
from .unified_policy import decode_edit_action, is_edit_action

# mcts_protype at repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_protype import MCTS, MCTSConfig, Oracle, STAMPState, TreeNode  # noqa: E402

logger = logging.getLogger(__name__)


class UnifiedMCTS(MCTS):
    """MCTS over STAMP with mixed growth + edit actions."""

    def __init__(
        self,
        decoder: STAMPDecoderImpl,
        policy,
        oracle: Oracle,
        config: MCTSConfig,
        *,
        memory: Optional[MoleculeMemory] = None,
        pool: Optional[SubtreePool] = None,
        edit_baseline_subtract: bool = True,
        max_iteration_multiplier: int = 4,
    ) -> None:
        super().__init__(decoder, policy, oracle, config)
        # Note: ``memory or MoleculeMemory()`` is wrong because ``__len__``
        # makes an empty memory falsy; explicit None check.
        self.memory = memory if memory is not None else MoleculeMemory()
        # Pool: prefer the policy's pool so harvest + query share state.
        if pool is not None:
            self.pool = pool
        elif hasattr(policy, "pool"):
            self.pool = policy.pool
        else:
            self.pool = SubtreePool()
        self.edit_baseline_subtract = bool(edit_baseline_subtract)
        self.max_iteration_multiplier = int(max_iteration_multiplier)
        # oracle-call savings due to memory hits
        self.memory_hits = 0
        self.iterations = 0
        # transplant vs diffusion accounting
        self.n_transplant_actions = 0
        self.n_diffusion_actions = 0
        self.n_growth_actions = 0
        self.n_harvested = 0

    # Safety cap: if memory hits dominate and we somehow stop collecting
    # fresh oracle calls, stop after this many iterations even if budget
    # isn't exhausted.
    @property
    def _iteration_hard_cap(self) -> int:
        return self.max_iteration_multiplier * self.config.oracle_budget

    def search(self) -> Dict[str, Any]:
        # Override the base search loop so that:
        # - logging fires only when ``oracle_calls`` actually advances
        # - we bail out cleanly on the iteration cap
        # - early-stop on best_reward >= early_stop_reward still works
        from mcts_protype import STAMPState as _STAMPState  # local import for type
        root = TreeNode(state=_STAMPState(tokens=[], depth=0))
        t0 = time.time()
        last_logged_call = -1

        while self.oracle_calls < self.config.oracle_budget:
            leaf = self._select(root)
            # Always try to expand: terminal molecules carry edit actions.
            leaf = self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)

            if (
                self.oracle_calls != last_logged_call
                and self.oracle_calls > 0
                and self.oracle_calls % max(1, self.config.log_every) == 0
            ):
                last_logged_call = self.oracle_calls
                logger.info(
                    f"Oracle calls: {self.oracle_calls}/{self.config.oracle_budget} | "
                    f"Best: {self.best_reward:.4f} | "
                    f"Tree nodes: {self._count_nodes(root)} | "
                    f"Max depth: {self._max_depth(root)} | "
                    f"Iters: {self.iterations} | MemHits: {self.memory_hits} | "
                    f"Time: {time.time() - t0:.1f}s"
                )

            if (
                self.config.early_stop_reward is not None
                and self.best_reward
                >= (self.config.early_stop_reward - self.config.early_stop_eps)
            ):
                logger.info(
                    f"Early stop at oracle call {self.oracle_calls}: "
                    f"best_reward={self.best_reward:.4f}"
                )
                break

            if self.iterations >= self._iteration_hard_cap:
                logger.info(
                    f"Iteration cap reached: {self.iterations} iters, "
                    f"{self.oracle_calls}/{self.config.oracle_budget} oracle calls, "
                    f"{self.memory_hits} memory hits."
                )
                break

        elapsed = time.time() - t0
        return {
            "best_reward": self.best_reward,
            "best_smiles": self.best_smiles,
            "oracle_calls": self.oracle_calls,
            "elapsed_seconds": elapsed,
            "reward_trajectory": self.reward_trajectory,
            "all_results": self.all_results,
            "iterations": self.iterations,
            "memory_hits": self.memory_hits,
            "memory_size": len(self.memory),
            "n_growth_actions": self.n_growth_actions,
            "n_diffusion_actions": self.n_diffusion_actions,
            "n_transplant_actions": self.n_transplant_actions,
            "n_harvested": self.n_harvested,
            "pool_stats": self.pool.stats(),
            "tree_stats": {
                "total_nodes": self._count_nodes(root),
                "max_depth": self._max_depth(root),
                "root_visit_count": root.visit_count,
                "root_children": len(root.children),
                "root_Q": root.mean_reward,
            },
        }

    # ------------------------------------------------------------------
    # Selection: do NOT short-circuit on terminal — terminal molecules can
    # still be edited.  Use our own ``is_fully_expanded`` that ignores the
    # base TreeNode's terminal short-circuit.
    # ------------------------------------------------------------------
    def _is_fully_expanded(self, node: TreeNode) -> bool:
        if node._candidate_actions is None:
            return False
        allowed = node.allowed_children_count(self.config.k_pw, self.config.alpha_pw)
        return len(node.children) >= allowed

    def _select(self, root: TreeNode) -> TreeNode:
        node = root
        while True:
            if not self._is_fully_expanded(node):
                return node
            if not node.children:
                return node
            node = self._select_child(node)

    # ------------------------------------------------------------------
    # Expansion: edit actions splice, growth actions append.
    # ------------------------------------------------------------------
    def _expand(self, node: TreeNode) -> TreeNode:
        if not node.has_candidates:
            actions = self.policy.top_k_actions(
                node.state.tokens, k=self.config.top_k
            )
            node.set_candidates(actions)

        action = node.pop_untried_action()
        if action is None:
            return node

        if is_edit_action(action):
            _start, _end, full_new = decode_edit_action(action)
            child_state = STAMPState(
                tokens=list(full_new),
                is_terminal=True,
                depth=node.state.depth + 1,
            )
            self.n_diffusion_actions += 1
        elif is_transplant_action(action):
            _start, _end, full_new, _donor_score = decode_transplant_action(action)
            child_state = STAMPState(
                tokens=list(full_new),
                is_terminal=True,
                depth=node.state.depth + 1,
            )
            self.n_transplant_actions += 1
        else:
            child_state = node.state.extend(action)
            child_state.is_terminal = self.decoder.is_terminal(child_state.tokens)
            self.n_growth_actions += 1

        # edit_baseline: for edit-zone descendants (both diffusion AND
        # transplant), baseline = parent node's terminal reward.
        is_edit_or_transplant = is_edit_action(action) or is_transplant_action(action)
        if (
            getattr(node, "_edit_baseline", None) is not None
            and is_edit_or_transplant
        ):
            baseline = node._edit_baseline
        elif node.state.is_terminal and is_edit_or_transplant:
            baseline = node.mean_reward if node.visit_count > 0 else 0.0
        else:
            baseline = 0.0

        child = TreeNode(
            state=child_state,
            parent=node,
            incoming_action=action,
        )
        child._edit_baseline = baseline  # type: ignore[attr-defined]
        node.children[action.key] = child
        return child

    # ------------------------------------------------------------------
    # Simulation: memory-aware oracle call
    # ------------------------------------------------------------------
    def _simulate(self, node: TreeNode) -> float:
        self.iterations += 1
        if self.iterations > self._iteration_hard_cap:
            # Safety: bail out of the loop via a sentinel.  We set
            # oracle_calls to the budget so the base loop exits cleanly.
            logger.info(
                f"UnifiedMCTS: hit iteration cap {self._iteration_hard_cap} "
                f"with {self.oracle_calls}/{self.config.oracle_budget} "
                f"oracle calls, {self.memory_hits} memory hits."
            )
            self.oracle_calls = self.config.oracle_budget
            return 0.0
        if node.state.is_terminal:
            complete_tokens = node.state.tokens
        else:
            complete_tokens = self.policy.rollout(
                node.state.tokens,
                max_steps=self.config.max_rollout_steps,
            )
        smiles = self.decoder.decode_to_smiles(complete_tokens)
        canon, cached = self.memory.lookup(smiles)

        if cached is not None:
            reward = cached
            self.memory_hits += 1
        elif canon is None:
            reward = 0.0
        else:
            reward = self.oracle(canon)
            self.memory.record(canon, reward)
            # Only consumes budget when the oracle was actually called.
            self.oracle_calls += 1

            # Harvest subtrees from the new molecule into the pool.
            try:
                parsed = parse_terminal_sequence(complete_tokens, self.decoder.encoder)
                if parsed is not None:
                    donors = list(extract_donor_records(parsed))
                    self.n_harvested += self.pool.harvest(
                        donors,
                        molecule_score=float(reward),
                        source_smiles=canon,
                    )
            except Exception:
                pass

        action_kind = "growth"
        if node.incoming_action is not None:
            if is_transplant_action(node.incoming_action):
                action_kind = "transplant"
            elif is_edit_action(node.incoming_action):
                action_kind = "diffusion"

        self.all_results.append({
            "oracle_call": self.oracle_calls,
            "smiles": canon or "",
            "reward": reward,
            "depth": node.state.depth,
            "action": action_kind,
            "cached": cached is not None,
        })

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_smiles = canon

        self.reward_trajectory.append((self.oracle_calls, self.best_reward))
        return reward

    # ------------------------------------------------------------------
    # PUCT: baseline-subtract for edit-zone children
    # ------------------------------------------------------------------
    def _puct_score(self, parent: TreeNode, child: TreeNode) -> float:
        baseline = getattr(child, "_edit_baseline", 0.0) or 0.0
        if not self.edit_baseline_subtract:
            baseline = 0.0
        exploitation = max(0.0, child.mean_reward - baseline)

        prior = child.incoming_action.prior if child.incoming_action else 1.0
        exploration = (
            self.config.c_puct
            * prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        return exploitation + exploration

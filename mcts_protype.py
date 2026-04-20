"""
STAMP-MCTS Phase 1 Prototype
=============================

Simplified, runnable MCTS for fragment-tree molecular optimization.
Phase 1 = bare MCTS. No PER, no value network, no persistent tree reuse.

Purpose: Verify that MCTS + STAMP beats random LM sampling on QED.
Exit criterion: MCTS best reward > LM sampling best reward by >= 0.02.

Architecture:
    - Action = one motif-group (motif SMILES + joint indices)
    - Selection = PUCT with LM prior
    - Expansion = LM top-k proposals
    - Rollout = greedy AR completion
    - Backup = mean reward

How to use:
    1. Implement the three abstract classes at the bottom of this file:
       - STAMPDecoder: your existing encoder/decoder with R4 fallbacks
       - PolicyModel: your existing AR LM wrapped to expose top_k_actions and rollout
       - Oracle: the scoring function (QED provided as default)
    2. Run: python mcts_prototype.py --task qed --budget 1000

Dependencies:
    - Python 3.9+
    - numpy
    - rdkit (for QED oracle)
    - torch (for your LM — not used directly in this file)
"""

from __future__ import annotations

import math
import time
import random
import logging
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Sequence
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class Action:
    """
    One atomic MCTS action = one motif-group in STAMP.
    
    Example: Action(tokens=["c1ccccc1", "[J_0]", "[J_3]"], prior=0.12)
    means "place benzene with joints at atom 0 and atom 3, LM assigns 12% prior".
    
    tokens includes the motif + all its [J_i] joints + optional [B_n] branch marker.
    This is the unit of tree depth — one action = one level deeper in MCTS.
    """
    tokens: Tuple[str, ...]    # immutable for use as dict key
    prior: float               # policy prior from LM, used in PUCT
    
    @property
    def key(self) -> str:
        return " ".join(self.tokens)


@dataclass
class STAMPState:
    """
    A state in the MCTS tree = a partial STAMP sequence.
    
    This is a lightweight wrapper. The heavy lifting (stack machine simulation,
    port tracking, validity) lives in STAMPDecoder.
    """
    tokens: List[str]          # STAMP token prefix built so far
    is_terminal: bool = False  # True when port stack is empty (complete molecule)
    depth: int = 0             # number of motif-groups placed (= tree depth)
    
    def copy(self) -> STAMPState:
        return STAMPState(
            tokens=list(self.tokens),
            is_terminal=self.is_terminal,
            depth=self.depth,
        )
    
    def extend(self, action: Action) -> STAMPState:
        """Return a NEW state with action applied. Does NOT mutate self."""
        new_tokens = self.tokens + list(action.tokens)
        return STAMPState(
            tokens=new_tokens,
            is_terminal=False,  # caller should update via decoder
            depth=self.depth + 1,
        )


@dataclass
class TreeNode:
    """
    A node in the MCTS search tree.
    
    Design notes:
    - children is keyed by Action.key (string) for fast lookup
    - candidate_actions is populated lazily on first expansion attempt
    - We track best_reward separately for "best molecule found" reporting
    """
    state: STAMPState
    parent: Optional[TreeNode] = None
    incoming_action: Optional[Action] = None
    
    # MCTS statistics
    visit_count: int = 0               # N(s)
    total_reward: float = 0.0          # W(s) = sum of rewards through this node
    
    # Children
    children: Dict[str, TreeNode] = field(default_factory=dict)
    
    # Action proposals from policy model (populated lazily)
    _candidate_actions: Optional[List[Action]] = None
    _next_untried_idx: int = 0
    
    @property
    def mean_reward(self) -> float:
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count
    
    @property
    def has_candidates(self) -> bool:
        return self._candidate_actions is not None
    
    @property
    def has_untried_actions(self) -> bool:
        if self._candidate_actions is None:
            return True  # haven't even proposed yet
        return self._next_untried_idx < len(self._candidate_actions)
    
    def set_candidates(self, actions: List[Action]) -> None:
        """Set the candidate action list (from policy model top-k)."""
        self._candidate_actions = actions
        self._next_untried_idx = 0
    
    def pop_untried_action(self) -> Optional[Action]:
        """Get the next untried action, or None if all tried."""
        if self._candidate_actions is None:
            return None
        if self._next_untried_idx >= len(self._candidate_actions):
            return None
        action = self._candidate_actions[self._next_untried_idx]
        self._next_untried_idx += 1
        return action
    
    def allowed_children_count(self, k_pw: float, alpha_pw: float) -> int:
        """Progressive widening: max children = ceil(k * N^alpha)."""
        if self._candidate_actions is None:
            return 0
        max_from_pw = math.ceil(k_pw * (self.visit_count + 1) ** alpha_pw)
        max_from_candidates = len(self._candidate_actions)
        return min(max_from_pw, max_from_candidates)
    
    def is_fully_expanded(self, k_pw: float, alpha_pw: float) -> bool:
        """Check if progressive widening limit is reached."""
        if self.state.is_terminal:
            return True
        allowed = self.allowed_children_count(k_pw, alpha_pw)
        return len(self.children) >= allowed


# ============================================================================
# Abstract interfaces — implement these with your code
# ============================================================================

class STAMPDecoder(ABC):
    """
    Your STAMP encoder/decoder with R4 safety fallbacks.
    Wraps the stack machine logic.
    """
    
    @abstractmethod
    def decode_to_smiles(self, tokens: List[str]) -> Optional[str]:
        """
        Decode a STAMP token sequence to a SMILES string.
        Returns None if decoding fails (should be rare with R4 fallbacks).
        Must handle partial sequences gracefully (fill remaining ports with H).
        """
        ...
    
    @abstractmethod
    def is_terminal(self, tokens: List[str]) -> bool:
        """
        Check if the token sequence represents a complete molecule
        (port stack is empty, no open attachment sites).
        """
        ...


class PolicyModel(ABC):
    """
    Your AR language model, wrapped to provide two interfaces:
    1. top_k_actions: propose candidate motif-groups with priors
    2. rollout: complete a partial sequence to terminal
    """
    
    @abstractmethod
    def top_k_actions(self, prefix_tokens: List[str], k: int) -> List[Action]:
        """
        Given a STAMP prefix, return the top-k most probable next actions.
        
        Each action is a complete motif-group:
            motif_token + [J_i] joints + optional [B_n]
        
        The prior probability should be the joint probability of all tokens
        in the action-group under the LM.
        
        Implementation sketch:
            1. Feed prefix_tokens to LM, get logits for next token
            2. Sample/argmax the next motif token
            3. Continue sampling [J_i] tokens until a non-joint token appears
            4. If a [B_n] token appeared before the motif, include it
            5. Compute joint log-prob of the whole group
            6. Repeat to get k candidates (e.g., via beam search)
        
        Returns: List of Action, sorted by prior descending, length <= k
        """
        ...
    
    @abstractmethod
    def rollout(self, prefix_tokens: List[str], max_steps: int) -> List[str]:
        """
        Complete a partial STAMP sequence to a terminal state.
        
        Uses greedy or low-temperature sampling from the LM.
        Stops when the decoder reports is_terminal or max_steps reached.
        
        Returns: the complete token sequence (prefix + generated tokens)
        """
        ...


class Oracle(ABC):
    """Scoring function for complete molecules."""
    
    @abstractmethod
    def __call__(self, smiles: str) -> float:
        """Score a molecule. Returns reward in [0, 1]. Higher is better."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this oracle (for logging)."""
        ...


# ============================================================================
# MCTS Configuration
# ============================================================================

@dataclass
class MCTSConfig:
    """All hyperparameters for Phase 1 MCTS."""
    
    # PUCT selection
    c_puct: float = 1.4          # exploration constant (AlphaZero default)
    
    # Action proposal
    top_k: int = 30              # number of candidate actions per node
    
    # Progressive widening
    k_pw: float = 5.0            # base children count
    alpha_pw: float = 0.5        # growth exponent (sqrt scaling)
    
    # Rollout
    max_rollout_steps: int = 20  # max motif-groups to generate in rollout
    
    # Budget
    oracle_budget: int = 1000    # total oracle calls allowed
    early_stop_reward: float = 1.0   # stop search as soon as best_reward >= this
    early_stop_eps: float = 1e-4     # tolerance around early_stop_reward

    # Logging
    log_every: int = 100         # print progress every N oracle calls


# ============================================================================
# MCTS Core Algorithm
# ============================================================================

class MCTS:
    """
    Phase 1 MCTS on STAMP fragment trees.
    
    Simplified: no PER, no value network, no persistent tree.
    Single trajectory rollout, mean backup, PUCT selection.
    """
    
    def __init__(
        self,
        decoder: STAMPDecoder,
        policy: PolicyModel,
        oracle: Oracle,
        config: MCTSConfig,
    ):
        self.decoder = decoder
        self.policy = policy
        self.oracle = oracle
        self.config = config
        
        # Tracking
        self.oracle_calls = 0
        self.best_reward = 0.0
        self.best_smiles: Optional[str] = None
        self.all_results: List[Dict[str, Any]] = []
        self.reward_trajectory: List[Tuple[int, float]] = []  # (call_num, best_so_far)
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def search(self) -> Dict[str, Any]:
        """
        Run MCTS until oracle budget is exhausted.
        
        Returns a summary dict with best molecule, all results, and diagnostics.
        """
        root = TreeNode(state=STAMPState(tokens=[], depth=0))
        start_time = time.time()
        
        while self.oracle_calls < self.config.oracle_budget:
            # 1. Selection
            leaf = self._select(root)
            
            # 2. Expansion
            if not leaf.state.is_terminal:
                leaf = self._expand(leaf)
            
            # 3. Simulation (rollout + oracle)
            reward = self._simulate(leaf)
            
            # 4. Backpropagation
            self._backpropagate(leaf, reward)
            
            # Logging
            if self.oracle_calls % self.config.log_every == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Oracle calls: {self.oracle_calls}/{self.config.oracle_budget} | "
                    f"Best: {self.best_reward:.4f} | "
                    f"Tree nodes: {self._count_nodes(root)} | "
                    f"Max depth: {self._max_depth(root)} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Early stop: GuacaMol-style "break if the score has achieved 1.000"
            if (
                self.config.early_stop_reward is not None
                and self.best_reward
                >= (self.config.early_stop_reward - self.config.early_stop_eps)
            ):
                logger.info(
                    f"Early stop: best_reward={self.best_reward:.4f} "
                    f">= {self.config.early_stop_reward} at call {self.oracle_calls}."
                )
                break
        
        elapsed = time.time() - start_time
        
        return {
            "best_reward": self.best_reward,
            "best_smiles": self.best_smiles,
            "oracle_calls": self.oracle_calls,
            "elapsed_seconds": elapsed,
            "reward_trajectory": self.reward_trajectory,
            "all_results": self.all_results,
            "tree_stats": {
                "total_nodes": self._count_nodes(root),
                "max_depth": self._max_depth(root),
                "root_visit_count": root.visit_count,
                "root_children": len(root.children),
                "root_Q": root.mean_reward,
            },
        }
    
    # ------------------------------------------------------------------
    # Phase 1: Selection (PUCT)
    # ------------------------------------------------------------------
    
    def _select(self, root: TreeNode) -> TreeNode:
        """
        Traverse from root to a leaf using PUCT selection.
        
        Stops when we reach:
        - A terminal node (complete molecule)
        - A node that is not fully expanded (has untried actions)
        """
        node = root
        while True:
            if node.state.is_terminal:
                return node
            if not node.is_fully_expanded(self.config.k_pw, self.config.alpha_pw):
                return node
            if not node.children:
                return node
            
            # Select best child by PUCT
            node = self._select_child(node)
        
    def _select_child(self, parent: TreeNode) -> TreeNode:
        """Pick the child with highest PUCT score."""
        best_score = -float("inf")
        best_child = None
        
        for child in parent.children.values():
            score = self._puct_score(parent, child)
            if score > best_score:
                best_score = score
                best_child = child
        
        assert best_child is not None, "select_child called on node with no children"
        return best_child
    
    def _puct_score(self, parent: TreeNode, child: TreeNode) -> float:
        """
        PUCT = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
        
        This is the AlphaZero formula without value prior (Phase 1 has no V̂).
        """
        exploitation = child.mean_reward
        
        prior = child.incoming_action.prior if child.incoming_action else 1.0
        
        exploration = (
            self.config.c_puct
            * prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        
        return exploitation + exploration
    
    # ------------------------------------------------------------------
    # Phase 2: Expansion
    # ------------------------------------------------------------------
    
    def _expand(self, node: TreeNode) -> TreeNode:
        """
        Add one new child to the node.
        
        On first visit, query the policy model for top-k candidate actions.
        Then pop the next untried action and create a child node.
        """
        # Lazy init: get candidates from policy model
        if not node.has_candidates:
            actions = self.policy.top_k_actions(
                node.state.tokens, k=self.config.top_k
            )
            node.set_candidates(actions)
        
        # Pop next untried action
        action = node.pop_untried_action()
        if action is None:
            # All actions tried (shouldn't happen if is_fully_expanded works)
            return node
        
        # Create child state
        child_state = node.state.extend(action)
        child_state.is_terminal = self.decoder.is_terminal(child_state.tokens)
        
        # Create child node
        child = TreeNode(
            state=child_state,
            parent=node,
            incoming_action=action,
        )
        
        node.children[action.key] = child
        return child
    
    # ------------------------------------------------------------------
    # Phase 3: Simulation (rollout + oracle)
    # ------------------------------------------------------------------
    
    def _simulate(self, node: TreeNode) -> float:
        """
        Complete the partial molecule and evaluate with the oracle.
        
        This is a single-trajectory rollout (Phase 1, no PER).
        """
        if node.state.is_terminal:
            # Already complete — decode directly
            complete_tokens = node.state.tokens
        else:
            # Rollout using AR policy model
            complete_tokens = self.policy.rollout(
                node.state.tokens,
                max_steps=self.config.max_rollout_steps,
            )
        
        # Decode to SMILES
        smiles = self.decoder.decode_to_smiles(complete_tokens)
        
        if smiles is None:
            # Decoding failed — return 0 reward
            reward = 0.0
        else:
            # Oracle call
            reward = self.oracle(smiles)
        
        # Track
        self.oracle_calls += 1
        self.all_results.append({
            "oracle_call": self.oracle_calls,
            "smiles": smiles or "",
            "reward": reward,
            "depth": node.state.depth,
        })
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_smiles = smiles
        
        self.reward_trajectory.append((self.oracle_calls, self.best_reward))
        
        return reward
    
    # ------------------------------------------------------------------
    # Phase 4: Backpropagation
    # ------------------------------------------------------------------
    
    def _backpropagate(self, node: TreeNode, reward: float) -> None:
        """
        Mean backup: propagate reward from leaf to root.
        
        Every node on the path gets N += 1, W += reward.
        Q(s) = W(s)/N(s) converges to the mean reward of rollouts through s.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent
    
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    
    def _count_nodes(self, root: TreeNode) -> int:
        count = 1
        for child in root.children.values():
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, root: TreeNode) -> int:
        if not root.children:
            return root.state.depth
        return max(self._max_depth(c) for c in root.children.values())


# ============================================================================
# Baseline: Random LM Sampling (for comparison)
# ============================================================================

class RandomSamplingBaseline:
    """
    Simple baseline: generate molecules by sampling from the LM,
    evaluate each with the oracle. No search structure.
    
    This is what GenMol does (conceptually): independent samples
    with no information reuse across trials.
    """
    
    def __init__(
        self,
        decoder: STAMPDecoder,
        policy: PolicyModel,
        oracle: Oracle,
    ):
        self.decoder = decoder
        self.policy = policy
        self.oracle = oracle
    
    def run(
        self,
        budget: int,
        *,
        early_stop_reward: Optional[float] = 1.0,
        early_stop_eps: float = 1e-4,
    ) -> Dict[str, Any]:
        best_reward = 0.0
        best_smiles = None
        all_results = []
        trajectory = []
        start_time = time.time()

        for i in range(budget):
            # Sample a complete molecule from the LM
            tokens = self.policy.rollout([], max_steps=30)
            smiles = self.decoder.decode_to_smiles(tokens)

            if smiles is None:
                reward = 0.0
            else:
                reward = self.oracle(smiles)

            all_results.append({
                "oracle_call": i + 1,
                "smiles": smiles or "",
                "reward": reward,
            })

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

            trajectory.append((i + 1, best_reward))

            if (i + 1) % 100 == 0:
                logger.info(
                    f"[Baseline] {i+1}/{budget} | Best: {best_reward:.4f}"
                )

            if (
                early_stop_reward is not None
                and best_reward >= (early_stop_reward - early_stop_eps)
            ):
                logger.info(
                    f"[Baseline] early stop at call {i+1}: best_reward={best_reward:.4f}"
                )
                break
        
        elapsed = time.time() - start_time
        
        return {
            "best_reward": best_reward,
            "best_smiles": best_smiles,
            "oracle_calls": budget,
            "elapsed_seconds": elapsed,
            "reward_trajectory": trajectory,
            "all_results": all_results,
        }


# ============================================================================
# Concrete implementations for quick testing
# ============================================================================

class QEDOracle(Oracle):
    """QED oracle using RDKit. Fastest oracle for Phase 1 testing."""
    
    @property
    def name(self) -> str:
        return "qed"
    
    def __call__(self, smiles: str) -> float:
        try:
            from rdkit import Chem
            from rdkit.Chem import QED as QEDModule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return QEDModule.qed(mol)
        except Exception:
            return 0.0


class PenalizedLogPOracle(Oracle):
    """Penalized logP oracle (classic optimization benchmark)."""
    
    @property
    def name(self) -> str:
        return "penalized_logp"
    
    def __call__(self, smiles: str) -> float:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            logp = Crippen.MolLogP(mol)
            sa = self._sa_score(mol)
            cycle_score = self._cycle_score(mol)
            
            # Normalize to roughly [0, 1] range
            raw = logp - sa - cycle_score
            return max(0.0, min(1.0, (raw + 5) / 15))
        except Exception:
            return 0.0
    
    def _sa_score(self, mol) -> float:
        try:
            from rdkit.Chem import RDConfig
            import os, sys
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            import sascorer
            return sascorer.calculateScore(mol)
        except Exception:
            return 3.0  # default moderate SA
    
    def _cycle_score(self, mol) -> float:
        from rdkit.Chem import rdMolDescriptors
        cycle_list = mol.GetRingInfo().AtomRings()
        if not cycle_list:
            return 0.0
        max_cycle = max(len(c) for c in cycle_list)
        return max(0, max_cycle - 6)


# ============================================================================
# Mock implementations (for testing without real LM)
# ============================================================================

class MockDecoder(STAMPDecoder):
    """
    Mock decoder for testing MCTS logic without a real STAMP implementation.
    Treats each motif token as a SMILES fragment and concatenates them.
    
    REPLACE THIS with your real STAMPDecoder for actual experiments.
    """
    
    COMMON_MOTIFS = [
        "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
        "C1CCCCC1", "C1CCNCC1", "C1CCCC1", "C1CCC1",
        "CC", "CCC", "CCO", "CCN", "CC=O", "CC(=O)N",
        "CO", "CN", "CF", "CCl", "CBr",
        "NC=O", "NS(=O)=O", "OC=O", "C#N",
    ]
    
    def decode_to_smiles(self, tokens: List[str]) -> Optional[str]:
        # Extract motif tokens (non-bracket tokens)
        motifs = [t for t in tokens if not t.startswith("[")]
        if not motifs:
            return None
        
        # Simple mock: pick the first motif as the molecule
        # (A real decoder would use the stack machine)
        smiles = motifs[0]
        
        # Try to validate with RDKit
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except ImportError:
            return smiles
        
        return None
    
    def is_terminal(self, tokens: List[str]) -> bool:
        # Mock: terminal after placing 5+ motif-groups
        motif_count = sum(1 for t in tokens if not t.startswith("["))
        return motif_count >= 5


class MockPolicy(PolicyModel):
    """
    Mock policy model for testing MCTS logic.
    Returns random motifs with uniform priors.
    
    REPLACE THIS with your real AR LM for actual experiments.
    """
    
    MOTIFS = MockDecoder.COMMON_MOTIFS
    
    def top_k_actions(self, prefix_tokens: List[str], k: int) -> List[Action]:
        # Sample k random motifs with random joints
        actions = []
        used = set()
        
        for _ in range(min(k, len(self.MOTIFS))):
            motif = random.choice(self.MOTIFS)
            while motif in used and len(used) < len(self.MOTIFS):
                motif = random.choice(self.MOTIFS)
            used.add(motif)
            
            # Random joint index (simplified)
            max_j = min(5, len(motif))
            j = random.randint(0, max_j)
            
            tokens = (motif, f"[J_{j}]")
            prior = random.random()
            actions.append(Action(tokens=tokens, prior=prior))
        
        # Normalize priors
        total = sum(a.prior for a in actions)
        if total > 0:
            actions = [
                Action(tokens=a.tokens, prior=a.prior / total)
                for a in actions
            ]
        
        # Sort by prior descending
        actions.sort(key=lambda a: -a.prior)
        return actions
    
    def rollout(self, prefix_tokens: List[str], max_steps: int) -> List[str]:
        tokens = list(prefix_tokens)
        for _ in range(max_steps):
            motif = random.choice(self.MOTIFS)
            j = random.randint(0, min(5, len(motif)))
            tokens.extend([motif, f"[J_{j}]"])
            
            # Stop after enough motifs
            motif_count = sum(1 for t in tokens if not t.startswith("["))
            if motif_count >= random.randint(3, 8):
                break
        
        return tokens


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(
    decoder: STAMPDecoder,
    policy: PolicyModel,
    oracle: Oracle,
    config: MCTSConfig,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run Phase 1 experiment: MCTS vs random sampling baseline.
    
    Returns comparison results.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"{'='*60}")
    logger.info(f"Phase 1 Smoke Test: {oracle.name}")
    logger.info(f"Budget: {config.oracle_budget} oracle calls per method")
    logger.info(f"Seed: {seed}")
    logger.info(f"PUCT c={config.c_puct}, top_k={config.top_k}, "
                f"PW k={config.k_pw} α={config.alpha_pw}")
    logger.info(f"{'='*60}")
    
    # --- Baseline ---
    logger.info("\n--- Running LM sampling baseline ---")
    baseline = RandomSamplingBaseline(decoder, policy, oracle)
    baseline_result = baseline.run(config.oracle_budget)
    logger.info(f"Baseline best: {baseline_result['best_reward']:.4f} "
                f"({baseline_result['best_smiles']})")
    
    # --- MCTS ---
    random.seed(seed)  # reset seed for fair comparison
    np.random.seed(seed)
    
    logger.info("\n--- Running MCTS ---")
    mcts = MCTS(decoder, policy, oracle, config)
    mcts_result = mcts.search()
    logger.info(f"MCTS best:     {mcts_result['best_reward']:.4f} "
                f"({mcts_result['best_smiles']})")
    logger.info(f"Tree stats:    {mcts_result['tree_stats']}")
    
    # --- Verdict ---
    improvement = mcts_result["best_reward"] - baseline_result["best_reward"]
    logger.info(f"\n{'='*60}")
    logger.info(f"MCTS - Baseline = {improvement:+.4f}")
    
    if improvement > 0.02:
        logger.info("PASS: MCTS outperforms sampling by > 0.02.")
        logger.info("Phase 1 exit criterion met. Proceed to Phase 2 (add V̂).")
    elif improvement > 0:
        logger.info("MARGINAL: MCTS slightly better. Tune c_puct, top_k.")
    else:
        logger.info("FAIL: MCTS does not beat random. Debug MCTS logic.")
        logger.info("Common issues: action granularity wrong, prior not normalized,")
        logger.info("rollout too short, terminal detection broken.")
    
    logger.info(f"{'='*60}")
    
    return {
        "baseline": baseline_result,
        "mcts": mcts_result,
        "improvement": improvement,
        "config": {
            "oracle": oracle.name,
            "budget": config.oracle_budget,
            "seed": seed,
            "c_puct": config.c_puct,
            "top_k": config.top_k,
            "k_pw": config.k_pw,
            "alpha_pw": config.alpha_pw,
        },
    }


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save experiment results to JSON."""
    import json
    
    # Make numpy types serializable
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=default)
    logger.info(f"Results saved to {path}")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="STAMP-MCTS Phase 1 Prototype")
    parser.add_argument("--task", type=str, default="qed",
                        choices=["qed", "penalized_logp"],
                        help="Oracle task")
    parser.add_argument("--budget", type=int, default=1000,
                        help="Oracle call budget")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--c_puct", type=float, default=1.4,
                        help="PUCT exploration constant")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of candidate actions per node")
    parser.add_argument("--k_pw", type=float, default=5.0,
                        help="Progressive widening base")
    parser.add_argument("--alpha_pw", type=float, default=0.5,
                        help="Progressive widening exponent")
    parser.add_argument("--max_rollout", type=int, default=20,
                        help="Max rollout steps")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results_{task}_{budget}.json)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock decoder/policy for testing MCTS logic")
    args = parser.parse_args()
    
    # Oracle
    if args.task == "qed":
        oracle = QEDOracle()
    elif args.task == "penalized_logp":
        oracle = PenalizedLogPOracle()
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Decoder and policy
    if args.mock:
        logger.info("Using MOCK decoder and policy (for MCTS logic testing only)")
        decoder = MockDecoder()
        policy = MockPolicy()
    else:
        # =====================================================
        # TODO: Replace with your real implementations
        # =====================================================
        # from your_stamp_code import YourSTAMPDecoder, YourARPolicy
        # decoder = YourSTAMPDecoder(checkpoint="path/to/decoder")
        # policy = YourARPolicy(checkpoint="path/to/lm")
        # =====================================================
        raise NotImplementedError(
            "Real decoder/policy not implemented. "
            "Either pass --mock for testing, or implement "
            "STAMPDecoder and PolicyModel subclasses."
        )
    
    # Config
    config = MCTSConfig(
        c_puct=args.c_puct,
        top_k=args.top_k,
        k_pw=args.k_pw,
        alpha_pw=args.alpha_pw,
        max_rollout_steps=args.max_rollout,
        oracle_budget=args.budget,
    )
    
    # Run
    results = run_experiment(decoder, policy, oracle, config, seed=args.seed)
    
    # Save
    output_path = args.output or f"results_{args.task}_{args.budget}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
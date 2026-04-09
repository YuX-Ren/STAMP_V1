from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PortInfo:
    local_atom_idx: int
    external_bond_type: str
    local_env_code: str
    stereo_role: str
    port_canonical_rank: int = 0


@dataclass
class CandidateMotif:
    atom_indices: Tuple[int, ...]
    internal_subgraph_hash: str
    canonical_smarts: str
    bond_count: int
    ports: List[PortInfo]
    schema_hash: str
    frequency: int = 0
    molecule_coverage: int = 0
    compression_gain: float = 0.0
    tree_gain: float = 0.0
    recoverability_score: float = 0.0
    ambiguity_penalty: float = 0.0
    total_score: float = 0.0


@dataclass
class CoreVocabEntry:
    core_id: int
    core_hash: str
    canonical_smarts: str
    atom_count: int
    bond_count: int
    frequency: int
    schemas: Dict[str, int] = field(default_factory=dict)


@dataclass
class SchemaVocabEntry:
    schema_id: int
    schema_hash: str
    num_ports: int
    port_definition: List[PortInfo]
    frequency: int


@dataclass
class CandidateStat:
    core_hash: str
    schema_hash: str
    canonical_smarts: str
    atom_count: int
    bond_count: int
    port_definition: List[PortInfo]
    occ_count: int = 0
    mol_coverage: int = 0
    compression_gain_sum: float = 0.0
    tree_gain_sum: float = 0.0
    recoverability_sum: float = 0.0
    ambiguity_sum: float = 0.0
    total_score: float = 0.0


@dataclass
class BuildConfig:
    min_core_freq: int = 20
    min_schema_freq: int = 5
    min_core_score: float = 0.0
    max_vocab_size: Optional[int] = 5000
    alpha: float = 1.0
    beta: float = 0.25
    gamma: float = 0.5
    delta: float = 1.0
    eta: float = 1.0
    lambda_cov: float = 0.5

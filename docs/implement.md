([GitHub][1])

---

# STAMP Motif Vocabulary Construction

**Implementation Spec for Codex**

## 1. Goal

Implement a motif vocabulary construction pipeline for **STAMP (Sequential Tree-based Attachment Motif Presentation)**.

The vocabulary should support:

1. **Exact reversibility**: molecule can be reconstructed uniquely from motif decomposition.
2. **Tree-friendly decomposition**: motif graph should be as tree-like as possible.
3. **Compact sequence representation**: reduce token length compared with atom-level serialization.
4. **Stable attachment modeling**: attachment sites are explicit and canonical.
5. **Stereo-aware interfaces**: chirality / bond stereo is stored in attachment schema, not only in motif core.

This is **not** a plain frequent-subgraph miner.
It is a **motif vocabulary builder for reversible tree serialization**.

---

## 2. Core Design

### 2.1 Final token design

Use a **two-level vocabulary**.

#### Core motif vocabulary

Represents internal topology only:

```text
core_id -> canonical connected subgraph
```

Examples:

* benzene
* pyridine
* amide fragment
* fused bicyclic scaffold
* tert-butyl branch motif

#### Attachment schema vocabulary

Represents how a core connects to the outside:

```text
schema_id -> ordered ports + bond types + local stereo constraints
```

Examples of schema information:

* number of ports
* which atom each port belongs to
* canonical port ordering
* allowed external bond type at each port
* aromatic/single/double semantics
* tetrahedral / E-Z stereo constraints induced by attachment

#### Final STAMP token

A token is:

```text
STAMP_token = (core_id, schema_id)
```

---

## 3. Data Structures

## 3.1 Molecule object

Assume RDKit molecule input.

Internal normalized molecule object should expose:

```python
class MolGraph:
    atoms: List[Atom]
    bonds: List[Bond]
```

Each atom should include:

* atom index
* atomic number
* formal charge
* aromatic flag
* chiral tag
* hydrogen count if needed

Each bond should include:

* begin atom
* end atom
* bond type
* aromatic flag
* stereo info

---

## 3.2 Candidate motif

```python
class CandidateMotif:
    atom_indices: Tuple[int, ...]                 # atoms in this motif
    internal_subgraph_hash: str                   # canonical core hash
    ports: List["PortInfo"]                       # attachment points to outside
    schema_hash: str                              # canonical schema hash
    frequency: int = 0
    molecule_coverage: int = 0
    compression_gain: float = 0.0
    tree_gain: float = 0.0
    recoverability_score: float = 0.0
    ambiguity_penalty: float = 0.0
    total_score: float = 0.0
```

---

## 3.3 Port info

```python
class PortInfo:
    local_atom_idx: int               # atom inside motif that connects outside
    external_bond_type: str           # SINGLE / DOUBLE / TRIPLE / AROMATIC
    local_env_code: str               # optional compact descriptor
    stereo_role: str                  # NONE / TETRA_REF / EZ_LEFT / EZ_RIGHT ...
    port_canonical_rank: int
```

---

## 3.4 Core vocabulary entry

```python
class CoreVocabEntry:
    core_id: int
    core_hash: str
    canonical_smarts: str
    atom_count: int
    bond_count: int
    frequency: int
    schemas: Dict[str, int]           # schema_hash -> count
```

---

## 3.5 Schema vocabulary entry

```python
class SchemaVocabEntry:
    schema_id: int
    schema_hash: str
    num_ports: int
    port_definition: List[PortInfo]
    frequency: int
```

---

## 4. Pipeline Overview

```text
Input molecules
    ↓
Normalize molecules
    ↓
Generate candidate decompositions
    ↓
Extract candidate motifs
    ↓
Canonicalize core and schema
    ↓
Aggregate statistics over dataset
    ↓
Compute motif scores
    ↓
Select core vocab
    ↓
Select / merge schema vocab
    ↓
Build fallback mechanism
    ↓
Validate exact reversibility
    ↓
Export vocab + decomposition rules
```

---

## 5. Candidate Generation

Use **multiple candidate sources**.
Do not rely on one decomposition rule only.

## 5.1 Candidate source A: ring-preserving motifs

Generate motifs from:

* simple rings
* fused ring systems
* bridged ring systems if they are stable and common

Rules:

* never split aromatic ring into tiny pieces at this stage
* fused rings should preferentially stay together as one candidate

---

## 5.2 Candidate source B: functional-group motifs

Use a rule-based library of stable local groups:

* amide
* ester
* sulfonamide
* carboxylate
* nitrile
* urea
* carbamate
* phosphate-like groups
* common heteroaromatic fragments

These are only **candidates**, not guaranteed final vocab items.

---

## 5.3 Candidate source C: cut on rotatable single bonds

Generate candidate motifs by cutting:

* non-ring single bonds
* preferably rotatable bonds
* avoid cutting bonds that destroy stable local chemistry

This source improves tree-likeness.

---

## 5.4 Candidate source D: frequency-driven merges

Generate additional candidates by merging adjacent fragments if:

* merged motif appears frequently across dataset
* merge improves compression or tree score

This is the part most similar to PS-style mining, but here it is only one source among several.([GitHub][1])

---

## 6. Canonicalization

This is critical.
Equivalent motifs must map to exactly the same `(core_hash, schema_hash)`.

## 6.1 Core canonicalization

For each candidate motif:

1. Extract induced subgraph.
2. Remove external atoms.
3. Keep only internal topology and internal bond/atom attributes.
4. Compute canonical atom order.
5. Serialize canonical representation.

Recommended outputs:

* canonical SMARTS
* canonical hash string

Example:

```python
core_hash = canonicalize_core(submol)
```

Core canonicalization should ignore:

* external neighbor identity
* attachment order
* external stereo induced by outside atoms

---

## 6.2 Schema canonicalization

Schema must encode:

1. number of ports
2. which internal atom each port is on
3. canonical order of ports
4. external bond type per port
5. aromatic/single/double bond semantics
6. local stereo constraints related to ports

Example:

```python
schema_hash = canonicalize_schema(core_submol, ports)
```

Two candidates with same core but different attachment ordering should get:

* same `core_hash`
* different `schema_hash` if reconstruction differs

---

## 7. Scoring Function

For each candidate motif (m), compute:

```text
Score(m) =
    α * Frequency(m)
  + β * CompressionGain(m)
  + γ * TreeGain(m)
  + δ * Recoverability(m)
  - η * AmbiguityPenalty(m)
```

Implement each term as below.

---

## 7.1 Frequency

```python
Frequency(m) = log(1 + count_occurrences)
```

Also track:

* raw occurrence count
* number of unique molecules containing motif

Possible combined version:

```python
freq = log(1 + occ_count) + lambda_cov * log(1 + mol_coverage)
```

---

## 7.2 CompressionGain

Measure token reduction from using motif instead of atom-level encoding.

Suggested approximation:

```python
compression_gain = atom_count(m) - 1
```

Better version:

```python
compression_gain = avg_saved_tokens_over_occurrences
```

Where saved tokens includes:

* fewer node tokens
* fewer branch tokens
* fewer ring/back-edge tokens

---

## 7.3 TreeGain

This is STAMP-specific and important.

Measure how much replacing motif with a single node makes the motif graph more tree-like.

Suggested metrics:

* decrease in cyclomatic number
* decrease in number of back-edges needed for serialization
* decrease in branching entropy
* improvement in DFS serialization simplicity

Recommended initial version:

```python
tree_gain =
    w1 * (original_cycle_rank - contracted_cycle_rank) +
    w2 * (original_back_edges - contracted_back_edges)
```

Where `contracted_*` means after collapsing motif into one supernode.

---

## 7.4 Recoverability

Hard requirement first, score second.

If a motif decomposition is not exactly reconstructable, reject candidate or heavily penalize it.

Initial binary version:

```python
recoverability = 1.0 if exact_reconstruction_success else 0.0
```

Better version:

* fraction of occurrences reconstructed exactly
* stereo preserved or not
* bond type preserved or not

---

## 7.5 AmbiguityPenalty

Penalize motifs whose schema is unstable.

Examples of ambiguity:

* same `(core, schema)` maps to multiple reconstructions
* port order not stable
* symmetric motif causes inconsistent ordering
* stereo around interface cannot be determined uniquely

Suggested implementation:

```python
ambiguity_penalty =
    p1 * reconstruction_collision_rate +
    p2 * unstable_port_order_rate +
    p3 * stereo_conflict_rate
```

---

## 8. Vocabulary Selection

## 8.1 Core selection

Aggregate all candidates by `core_hash`.

For each core:

* total frequency
* total molecule coverage
* average compression gain
* average tree gain
* schema diversity
* reconstruction failure rate

Select top cores by:

* minimum frequency threshold
* minimum score threshold
* optional max vocab size

Example:

```python
selected_cores = select_top_cores(
    core_stats,
    min_freq=20,
    min_score=2.5,
    max_vocab_size=5000,
)
```

---

## 8.2 Schema selection

For each selected core:

* gather all observed schema hashes
* keep schemas with sufficient support
* merge nearly equivalent schemas if they are reconstruction-equivalent
* otherwise preserve separately

Important:

* schema merging is allowed only if exact reversibility is preserved

Example:

```python
selected_schemas = select_schemas_for_core(
    core_hash,
    min_schema_freq=5,
    preserve_stereo=True,
)
```

---

## 8.3 Rare pattern fallback

For any rare or unseen motif:

* fall back to smaller motifs
* if necessary fall back to atom-level representation

This ensures:

* open-world coverage
* bounded vocab size
* no OOV failure

Fallback order:

```text
large motif
  -> smaller motif
  -> minimal chemically stable fragment
  -> atom-level
```

---

## 9. Tree-First Decomposition Strategy

When decomposing a new molecule using the built vocabulary, use a **tree-biased search**.

## 9.1 Objective

Find a decomposition that:

1. covers all atoms
2. uses selected vocab motifs
3. minimizes serialization complexity
4. preserves exact reconstruction

## 9.2 Search score

For a decomposition (D):

```text
DecompScore(D) =
    a * TotalCompression(D)
  + b * TotalTreeGain(D)
  - c * NumBackEdges(D)
  - d * NumFallbackAtoms(D)
```

## 9.3 Greedy initial implementation

A practical first version:

1. match largest valid vocab motifs first
2. break ties by higher tree gain
3. break ties by higher compression gain
4. avoid overlaps
5. cover remaining atoms with smaller motifs
6. atom-level fallback last

---

## 10. Stereo Handling

For STAMP-v1, place stereo mainly in **schema**, not in core.

## 10.1 Core should contain

* internal bond topology
* internal aromaticity
* internal atom types
* internal charge pattern

## 10.2 Schema should contain

* port order
* port-local reference ordering
* tetrahedral interface constraints
* double-bond E/Z interface constraints
* any attachment-dependent stereo semantics

Rationale:

* many stereo distinctions are induced by external attachments
* putting all stereo into core causes vocabulary explosion

---

## 11. Required Modules

## 11.1 `mol_normalize.py`

Functions:

* sanitize input molecule
* kekulize or preserve aromatic mode consistently
* normalize atom mapping
* normalize charge/aromatic flags
* assign stereochemistry consistently

API:

```python
def normalize_mol(mol) -> MolGraph:
    ...
```

---

## 11.2 `candidate_generation.py`

Functions:

* generate ring candidates
* generate functional group candidates
* generate cut-based candidates
* generate merge-based candidates

API:

```python
def generate_candidates(mol: MolGraph) -> List[CandidateMotif]:
    ...
```

---

## 11.3 `canonicalize.py`

Functions:

* canonicalize core
* canonicalize schema
* handle symmetric motifs robustly

API:

```python
def canonicalize_core(mol, atom_indices) -> str:
    ...

def canonicalize_schema(mol, atom_indices, ports) -> str:
    ...
```

---

## 11.4 `statistics.py`

Aggregate motif statistics over dataset.

API:

```python
def collect_motif_statistics(mols, candidates_per_mol):
    ...
```

Should produce:

* per-core stats
* per-schema stats
* per-(core, schema) stats

---

## 11.5 `tree_metrics.py`

Compute tree-related quality.

API:

```python
def compute_tree_gain(mol, atom_indices) -> float:
    ...

def contracted_cycle_rank(mol, atom_indices) -> int:
    ...
```

---

## 11.6 `reconstruction.py`

Given decomposition:

* reconstruct exact molecule
* verify atom types
* verify bond types
* verify stereo

API:

```python
def reconstruct_from_stamp(decomposition, vocab) -> MolGraph:
    ...

def is_exact_match(mol_a, mol_b) -> bool:
    ...
```

---

## 11.7 `vocab_builder.py`

Main orchestration.

API:

```python
class StampVocabBuilder:
    def fit(self, mols):
        ...
    def save(self, path):
        ...
    def load(self, path):
        ...
```

---

## 12. Output Files

## 12.1 `core_vocab.json`

```json
[
  {
    "core_id": 0,
    "core_hash": "...",
    "canonical_smarts": "...",
    "atom_count": 6,
    "bond_count": 6,
    "frequency": 10234
  }
]
```

## 12.2 `schema_vocab.json`

```json
[
  {
    "schema_id": 0,
    "schema_hash": "...",
    "num_ports": 2,
    "ports": [
      {
        "local_atom_idx": 1,
        "external_bond_type": "AROMATIC",
        "stereo_role": "NONE",
        "port_canonical_rank": 0
      }
    ],
    "frequency": 5043
  }
]
```

## 12.3 `core_schema_map.json`

```json
{
  "core_hash_1": ["schema_hash_a", "schema_hash_b"]
}
```

## 12.4 `decomposition_records.jsonl`

Per molecule:

* selected motifs
* atom coverage
* fallback positions
* reconstruction success
* token count
* tree complexity metrics

---

## 13. Validation Requirements

The implementation must include the following tests.

## 13.1 Exact reconstruction

For each training molecule:

* decompose
* serialize to STAMP tokens
* reconstruct
* compare exact graph identity

Pass condition:

* 100% graph identity on supported chemistry subset

---

## 13.2 Stereo preservation

For chiral and E/Z molecules:

* decomposition must preserve stereo after reconstruction

---

## 13.3 Canonical stability

Equivalent molecules with different atom indexing must yield:

* same motif decomposition up to deterministic ordering
* same token sequence if canonical serialization is applied

---

## 13.4 OOV robustness

For unseen molecules:

* system must not fail
* fallback decomposition must always exist

---

## 14. Recommended Implementation Order

## Phase 1: minimal working version

1. normalization
2. ring + functional group candidates
3. core canonicalization
4. simple schema = ports + bond types
5. frequency + compression scoring
6. reconstruction check
7. fallback to atoms

## Phase 2: STAMP-specific improvements

1. tree gain metric
2. tree-biased decomposition search
3. schema canonical ordering under symmetry
4. stereo-aware schema
5. better ambiguity penalty

## Phase 3: research version

1. merge-based candidate mining
2. learned scoring / reranking
3. domain-adaptive vocab
4. hierarchical vocab levels
5. benchmarking against PS-style vocab and other baselines

---

## 15. Non-Goals for v1

Do **not** implement these in the first version:

* learned tokenizer
* neural scoring of motifs
* joint tokenizer-generator training
* full 3D interface geometry
* protein/macromolecule adaptation
* highly optimized search over all possible decompositions

Start with deterministic and debuggable code.

---

## 16. Pseudocode

```python
def build_stamp_vocab(molecules):
    normalized = [normalize_mol(m) for m in molecules]

    all_candidates = []
    for mol in normalized:
        candidates = generate_candidates(mol)
        for cand in candidates:
            cand.internal_subgraph_hash = canonicalize_core(mol, cand.atom_indices)
            cand.schema_hash = canonicalize_schema(mol, cand.atom_indices, cand.ports)
        all_candidates.append(candidates)

    stats = collect_motif_statistics(normalized, all_candidates)

    for cand_stat in stats.candidate_stats:
        cand_stat.compression_gain = compute_compression_gain(cand_stat)
        cand_stat.tree_gain = compute_tree_gain_dataset_level(cand_stat)
        cand_stat.recoverability_score = estimate_recoverability(cand_stat)
        cand_stat.ambiguity_penalty = estimate_ambiguity(cand_stat)
        cand_stat.total_score = (
            ALPHA * cand_stat.frequency +
            BETA * cand_stat.compression_gain +
            GAMMA * cand_stat.tree_gain +
            DELTA * cand_stat.recoverability_score -
            ETA * cand_stat.ambiguity_penalty
        )

    core_vocab = select_core_vocab(stats)
    schema_vocab = select_schema_vocab(stats, core_vocab)

    vocab = StampVocab(core_vocab, schema_vocab)

    for mol in normalized:
        decomp = decompose_with_vocab(mol, vocab)
        recon = reconstruct_from_stamp(decomp, vocab)
        assert is_exact_match(mol, recon)

    return vocab
```

---

## 17. Success Criteria

STAMP-v1 is considered successful if it achieves:

1. **Exact reconstruction**
2. **Stable canonical tokenization**
3. **Shorter token sequences than atom-level**
4. **Lower serialization complexity due to tree-aware motifs**
5. **Graceful fallback on unseen chemistry**

---

## 18. One-line summary for Codex

Build a deterministic, reversible, tree-aware motif vocabulary pipeline where each token is `(core motif, attachment schema)`, candidates come from ring/functional-group/cut/merge rules, vocab is selected by frequency + compression + tree gain + recoverability, and all outputs must pass exact reconstruction.


[1]: https://github.com/THUNLP-MT/PS-VAE?utm_source=chatgpt.com "PS-VAE: Molecule Generation by Principal Subgraph ..."

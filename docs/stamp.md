# Sequential Tree-based Attachment Motif Presentation (STAMP): A Fragment-Based Molecular Representation with SELFIES-Style Validity Guarantees for LLM-Guided Molecule Generation

## Abstract

We propose **Sequential Tree-based Attachment Motif Presentation (STAMP)**, a novel molecular representation framework that unifies three previously orthogonal design goals: (1) fragment (motif) identity uniqueness from fragSMILES, (2) hierarchical topological encoding from t-SMILES, and (3) 100% sequence validity guarantees from SELFIES. The key insight is a **three-token grammar** — pure canonical motif tokens, external joint tokens `[J_i]` specifying connection points, and SELFIES-style branch tokens `[B_n]` encoding tree topology — operated by a deterministic stack machine. Building on this representation, we further propose a **reasoning-guided molecule growth** framework in which a large language model (LLM) interleaves natural language reasoning (`[THINK]`) with molecular assembly actions (`[ACT]`), supported by a checkpoint-based backtrack mechanism (`[BACK]`) that enables deliberate self-correction during generation. Crucially, the tree-structured encoding enables a unified **`[MASK]`-based editing** mechanism where both de novo generation and molecular optimization (lead optimization, scaffold decoration, fragment linking, core hopping) reduce to the same operation: conditional completion of masked subtree spans. The combined system transforms molecular design into a multi-step, auditable, chemically-grounded reasoning process.

---

## 1. Introduction

### 1.1 The Representation Bottleneck

Effective molecular representation remains a limiting factor for AI-driven molecular discovery. Existing approaches each solve part of the problem but leave critical gaps:

- **SMILES** is compact and widely supported, but atom-level tokenization loses fragment-level semantics, generates invalid strings, and offers no structural hierarchy.
- **SELFIES** guarantees 100% validity through a stack-based derivation, but operates at the atom level and does not encode fragment-level chemical knowledge.
- **t-SMILES** introduces tree-based hierarchical encoding of fragments, preserving molecular topology. However, its three variants (TSSA, TSDY, TSID) all modify fragment SMILES by inserting dummy atoms (`*`, `[n*]`) or sharing atoms, destroying the canonical identity of motifs.
- **fragSMILES** preserves motif identity — each fragment is a pure canonical SMILES string with no modification. But it uses custom `<index>` syntax for cross-fragment connections (a remote-pairing mechanism that can fail), and encodes only a flat linear traversal with no hierarchical topology.

No existing method simultaneously achieves motif uniqueness, tree topology, and validity guarantees.

### 1.2 The Generation Bottleneck

Current LLM-based molecular generation typically operates as a one-pass autoregressive process over SMILES or SELFIES strings. This has two fundamental limitations:

1. **No deliberation**: Once a token is generated, it becomes a fixed condition for all subsequent tokens. If the model "realizes" at token 50 that the choice at token 10 was suboptimal, it has no recourse.
2. **No chemical grounding**: Each token decision (choosing the next character in a SMILES string) has no direct chemical interpretation. The model cannot reason about "what functional group should go at this position" because the generation unit is an atom or character, not a meaningful chemical substructure.

### 1.3 Our Contribution

We address both bottlenecks with a unified framework:

1. **STAMP encoding** — a three-token grammar that keeps motifs as pure canonical SMILES (uniqueness), uses a SELFIES-style stack machine for assembly (validity), and derives from a fragment tree via DFS (topology).
2. **Reasoning-guided molecule growth** — an LLM generation loop that interleaves natural language reasoning with fragment-level assembly actions, supported by a `[BACK]` mechanism for deliberate self-correction.
3. **Unified [MASK]-based editing** — a molecular optimization framework where editing an existing molecule (replacing a branch, swapping a core, inserting a linker) reduces to the same conditional completion operation as de novo generation, enabled by the tree-to-substring correspondence of the encoding.

---

## 2. STAMP Encoding

### 2.1 Design Principles

The encoding is governed by three principles:

**Principle 1 — Motif-connection separation.** A motif token is a pure canonical SMILES string, identical across all molecules containing that substructure. Connection information is carried entirely by separate tokens external to the motif. This directly inherits fragSMILES's core insight: a benzene ring is always `c1ccccc1`, regardless of where or how it connects.

**Principle 2 — Local-only connection semantics.** Connection tokens specify which atom *within the current motif* serves as a connection point. They do not reference any other fragment. Inter-fragment bonding is performed automatically by the stack machine. This eliminates remote-pairing failures — the fundamental weakness of both t-SMILES's dummy atoms and fragSMILES's `<index>` tags.

**Principle 3 — Operational assembly.** Inspired by SELFIES, the encoding is a sequence of *operations* on a state machine, not a static description requiring global consistency. Any token sequence maps to a valid molecule.

### 2.2 Three Token Types

| Token type | Notation | Example | Semantics |
|---|---|---|---|
| **Motif** | canonical SMILES | `c1ccccc1`, `CC(=O)N` | A chemical fragment. Pure canonical form, no modifications. One motif = one vocabulary entry. |
| **Joint** | `[J_i]` | `[J_0]`, `[J_3]` | "The *i*-th heavy atom in the preceding motif's canonical SMILES is a connection point." Local to the current motif; does not reference other fragments. |
| **Branch** | `[B_n]` | `[B_1]`, `[B_2]` | SELFIES-style operator: "The next *n* motif-joint groups form a branch subtree." Encodes tree topology. |

**Vocabulary sizes:**
- Motif vocabulary: determined by the fragmentation algorithm (typically 2K–15K entries for drug-like molecules).
- Joint vocabulary: `[J_0]` through `[J_19]` (20 tokens, sufficient for fragments up to 20 heavy atoms).
- Branch vocabulary: `[B_1]` through `[B_5]` (5 tokens, sufficient for most molecular trees).

Total grammar overhead: ~25 special tokens.

### 2.3 Fragmentation

Molecular fragmentation follows the fragSMILES protocol: cleavage at all **exocyclic single bonds**. This provides a good tradeoff between token count and vocabulary size, producing fragments that correspond to chemically meaningful building blocks (rings, functional groups, linkers). Alternative fragmentation schemes (BRICS, rotatable bonds, MMPA, Scaffold) are also compatible — the framework is agnostic to the specific fragmentation algorithm, inheriting the flexibility of t-SMILES.

Each fragment is converted to its RDKit canonical SMILES. No dummy atoms, atom map numbers, or other modifications are applied. The resulting canonical string is the motif token.

### 2.4 Fragment Tree Construction

Given the set of fragments and their connectivity (the reduced molecular graph):

1. **Root selection**: Choose the fragment with the highest degree (number of connections to other fragments). Break ties by selecting the fragment with the most heavy atoms.
2. **Tree construction**: From the reduced graph, build a rooted tree using the selected root.
3. **Child ordering**: At each node, order children such that the subtree with more nodes comes first (DFS left-priority). This ensures a unique canonical DFS traversal.

### 2.5 Stack Machine Derivation

The encoding is generated by a DFS pre-order traversal of the fragment tree, and decoded by a deterministic stack machine.

**State**: `(partial molecule graph G, port stack S, branch save stack B)`

**Derivation rules:**

**R1 — Root motif.** The first motif token in the sequence is placed as the root. Read all subsequent `[J_i]` tokens: each specifies an output port (a connection point available for child fragments). Push ports onto S in reverse order (so the first-listed port is on top).

**R2 — Child motif.** For non-root motifs: read the motif token and its `[J_i]` tokens. The *first* `[J_i]` designates the **input port** — pop the top of S and form a bond between the popped port and this input port. Remaining `[J_i]` tokens designate output ports — push onto S in reverse order.

**R3 — Branch.** `[B_n]` saves the current stack S onto B, pops the top port from S as the active port for the branch. The next *n* motif-joint groups are processed within the branch context. After processing *n* groups, restore S from B.

**R4 — Safety fallbacks (SELFIES-style validity guarantee).**

| Condition | Fallback |
|---|---|
| `[J_i]` where *i* ≥ atom count of current motif | Use `i mod atom_count` |
| `[J_i]` points to a saturated atom (no free valence) | Skip to next atom with free valence |
| Motif encountered when S is empty | Place as disconnected component |
| `[B_n]` where *n* exceeds remaining tokens | Truncate to actual remaining length |
| Sequence ends with ports remaining on S | Fill remaining ports with implicit H |

**These four fallback rules guarantee that any arbitrary sequence of tokens from the vocabulary maps to a chemically valid molecule.** This is the SELFIES-level guarantee, lifted to the fragment level.

### 2.6 Canonicalization

Four rules jointly ensure that each molecule maps to exactly one STAMP string:

1. Root = highest-degree fragment (ties broken by atom count).
2. DFS child ordering = larger subtree first.
3. Motif strings = RDKit canonical SMILES.
4. `[J_i]` indices = determined by atom order in canonical SMILES.

Relaxing rules 1–2 enables data augmentation (alternative root choices and child orderings), analogous to SMILES randomization.

### 2.7 Worked Example

**Molecule**: 4-(acetylamino)biphenyl. SMILES: `CC(=O)Nc1ccc(-c2ccccc2)cc1`

**Step 1 — Fragmentation** (exocyclic single bonds):

| Fragment | Canonical SMILES | Connections |
|---|---|---|
| F_A (central phenyl) | `c1ccccc1` | C1 → F_C, C4 → F_B |
| F_B (side phenyl) | `c1ccccc1` | C1 → F_A |
| F_C (acetamido) | `CC(=O)N` | N → F_A |

Note: F_A and F_B have *identical* motif tokens — the same benzene ring canonical SMILES. This is the motif uniqueness property.

**Step 2 — Fragment tree**: F_A as root (degree 2), F_B and F_C as children.

**Step 3 — Encoding** (DFS pre-order with branch):

```
c1ccccc1 [J_0][J_3] [B_1] CC(=O)N [J_3] c1ccccc1 [J_0]
```

**Step 4 — Derivation trace:**

| Token | Action | Port stack |
|---|---|---|
| `c1ccccc1` | Place root F_A | — |
| `[J_0]` | C1 is output port → push A@C1 | [A@C1] |
| `[J_3]` | C4 is output port → push A@C4 | [A@C4, A@C1] |
| `[B_1]` | Save stack, pop A@C4 for branch | active=A@C4, saved=[A@C1] |
| `CC(=O)N` | Place F_C in branch | — |
| `[J_3]` | N is input port → bond C:N ↔ A:C4. Branch done, restore. | [A@C1] |
| `c1ccccc1` | Place F_B on main chain | — |
| `[J_0]` | C1 is input port → bond B:C1 ↔ A:C1. Stack empty. Done. | [] |

### 2.8 Comparison with Existing Methods

| Property | TSID | TSDY | TSSA | fragSMILES | SELFIES | **STAMP** |
|---|---|---|---|---|---|---|
| Motif purity | ✗ `[n*]` inserted | ✗ `*` inserted | ✗ shared atoms | ✓ canonical | atom-level | **✓ canonical** |
| Connection mechanism | Cross-fragment ID pairing | Positional implicit pairing | Shared atom | Cross-fragment `<index>` | Stack-automatic | **Stack-automatic, local `[J_i]`** |
| Topology encoding | BFS + `&^` + φ | BFS + `&^` + φ | BFS + `&^` | Flat linear traversal | `[Branch]` `[Ring]` | **DFS + `[B_n]`** |
| 100% validity | ✗ | ✗ | ✗ | ✗ | ✓ | **✓** |
| Standard parser compatible | ✗ | ✗ | ✗ | ✗ | ✗ (custom) | ✗ (custom, but simple) |

---

## 3. LLM-Guided Reasoning and Molecule Growth

### 3.1 Motivation: Molecule Design as Deliberative Search

The STAMP stack machine maintains, at every step, a **port stack** that precisely describes *which positions on the partial molecule are still open for growth*. This port stack is the molecular analog of a "frontier" in tree search. At each growth step, the LLM's task is: *given the current frontier, the partial molecule's properties, and the design objective, choose what to place where.*

This maps directly onto deliberative reasoning frameworks (chain-of-thought, tree-of-thought) from the NLP literature:

| Molecular concept | Reasoning analog |
|---|---|
| Fragment tree node | Decision point |
| Port stack (growth frontier) | Open subproblems |
| `[B_n]` branch | Subproblem decomposition |
| `[BACK]` | Self-correction / backtracking |
| `[END]` | Solution found |

### 3.2 Extended Token Set for Generation

The generation vocabulary extends STAMP with three additional control tokens:

| Token | Type | Semantics |
|---|---|---|
| `[THINK]` | Reasoning | Marks the beginning of a natural language reasoning segment. The LLM explains its current assessment and planned action. |
| `[BACK]` | Control | Reverts the last motif placement. Restores the stack machine to the checkpoint saved before that placement. |
| `[END]` | Control | Terminates generation. Remaining open ports are filled with implicit hydrogen. |

The full token vocabulary is thus: `{motif tokens} ∪ {[J_0]...[J_19]} ∪ {[B_1]...[B_5]} ∪ {[THINK], [BACK], [END]}`.

### 3.3 Generation Loop

At each step of molecule growth, the LLM receives a structured context and produces an interleaved reasoning-action output:

**Input context (updated each step):**

```
[GOAL]    Target properties (e.g., logP > 3.0, MW < 400, QED > 0.7)
[STACK]   Current open ports (growth frontier)
[PARTIAL] STAMP sequence generated so far
[PROPS]   Computed properties of the partial molecule
[DEPTH]   Number of motifs placed / budget remaining
[HISTORY] Previous [THINK] and [ACT] segments
```

**Output (one step):**

```
[THINK] <natural language reasoning about current state and strategy>
[ACT]   <one of: motif [J_i]+ | [B_n] | [BACK] | [END]>
```

**Example generation trace:**

```
[THINK] Target: logP>3, MW<400. Start with aromatic core for drug-likeness.
[ACT]   c1ccccc1 [J_0][J_3]

[THINK] Phenyl root placed. Two growth points: C1 and C4. C4 side chain
        should increase lipophilicity. Adding another phenyl gives ~+1.5 logP.
[ACT]   [B_1] c1ccccc1 [J_0]

[THINK] Biphenyl scaffold done. logP≈3.2, MW=154. Now C1: need small capping
        group. Try acetamide for H-bond capacity.
[ACT]   CC(=O)N [J_3]

[THINK] logP dropped to 2.1 — too polar. Acetamide hurts lipophilicity.
        Backtrack and try something less polar.
[ACT]   [BACK]

[THINK] Replace with methyl: small MW increase, +0.3 logP.
[ACT]   C [J_0]

[THINK] Final: logP≈3.5, MW=168, QED=0.82. All targets met.
[ACT]   [END]
```

**Clean output** (stripping reasoning and backtrack traces): `c1ccccc1 [J_0][J_3] [B_1] c1ccccc1 [J_0] C [J_0]`

This clean sequence is a standard STAMP encoding, directly decodable by the stack machine.

### 3.4 Backtrack Mechanism

#### 3.4.1 Checkpoint Stack

The stack machine is extended with a **checkpoint stack C**. Before each motif placement, the complete state (port stack S, branch save stack B, partial molecule graph G) is pushed onto C as a snapshot. `[BACK]` pops the most recent snapshot and restores the full state.

#### 3.4.2 Backtrack Granularity

`[BACK]` reverts exactly **one motif placement** — not one token, not one entire subtree. This granularity aligns with the natural unit of chemical decision-making: "place this functional group at this position."

- **Finer** (atom-level) would be chemically meaningless — half a motif is not a valid chemical decision.
- **Coarser** (subtree-level) would be too aggressive — the LLM may be satisfied with the first two fragments of a branch but not the third.

#### 3.4.3 Interaction with Branches

Three rules govern `[BACK]` behavior near branch boundaries:

- **Rule 1**: `[BACK]` inside a branch only reverts within that branch. It does not escape the branch scope.
- **Rule 2**: `[BACK]` at a branch entry point (checkpoint before `[B_n]`) reverts the entire branch including all motifs placed within it.
- **Rule 3**: Consecutive `[BACK]` tokens revert to progressively earlier checkpoints. A generation budget penalty prevents infinite backtrack loops.

#### 3.4.4 The Nature of [BACK]

`[BACK]` is fundamentally different from "generate-then-filter" approaches. It gives the LLM **evaluation-based regret**: the ability to act, observe the consequences (via property computation on the partial molecule), and deliberately undo the action. This is System 2 reasoning — deliberate, evaluative, self-correcting — as opposed to the System 1 one-shot generation of standard autoregressive models.

### 3.5 Search Modes

The framework supports three search strategies at inference time:

| Mode | Mechanism | Use case |
|---|---|---|
| **Linear** | No `[BACK]`. One `[THINK]+[ACT]` per step. | High-throughput virtual screening |
| **Beam + BACK** | *k* parallel paths, each with independent backtrack. | Balanced exploration-exploitation |
| **MCTS + BACK** | LLM as rollout policy; MCTS on fragment tree. `[BACK]` for backpropagating bad paths. | Lead optimization, precision design |

---

## 4. Molecular Editing and Optimization

### 4.1 The Optimization Problem

De novo generation builds molecules from scratch. But in practice, the most common task in drug discovery is **lead optimization**: starting from a known active compound and making targeted modifications — replacing a side chain, swapping a ring system, inserting a linker — while preserving the core scaffold and its desirable properties.

This requires a fundamentally different capability: **locating a specific substructure within an existing molecule, excising it, and regrowing a new substructure in its place.** Existing string-based representations make this difficult because the relationship between string position and molecular substructure is opaque in SMILES/SELFIES. In STAMP, however, the DFS pre-order traversal guarantees that **any subtree of the fragment tree corresponds to a contiguous substring** of the encoding. This makes substructure editing a substring replacement operation.

### 4.2 The [MASK] Token and Edit Operations

We introduce a single new token `[MASK]` that marks positions in the sequence where the LLM should generate new content. All editing operations reduce to: **replace a contiguous span of tokens with [MASK], then let the LLM fill in the [MASK] by growing new structure.**

Four editing patterns emerge from different choices of what to mask:

**Leaf replacement.** Mask the last motif+joints in a branch (a leaf node). The simplest case — the LLM generates a replacement fragment (or a new subtree) with no downstream dependencies.

```
Original:  ...kept... c1ccccc1 [J_0]
Masked:    ...kept... [MASK]
Edited:    ...kept... c1ccncc1 [J_0]       ← phenyl → pyridine
```

**Branch replacement.** Mask an entire `[B_n]`...branch-end span. The LLM generates a completely new branch, potentially with different depth and structure.

```
Original:  ...kept... [B_1] CC(=O)N [J_3] ...kept...
Masked:    ...kept... [MASK] ...kept...
Edited:    ...kept... [B_1] CS(=O)(=O)N [J_5] ...kept...   ← acetamide → sulfonamide
```

**Mid-node replacement.** Mask only a motif+joints (not its children). The children subtrees remain in place, but are re-wired to the new motif's connection points.

```
Original:  c1ccccc1 [J_0][J_3] [B_1] child_A child_B
Masked:    [MASK] [B_1] child_A child_B
Edited:    c1ncncc1 [J_0][J_3] [B_1] child_A child_B     ← phenyl → pyrimidine core
```

**Insertion.** Place a [MASK] between two existing tokens to insert a new fragment (with ≥ 2 ports: one connecting to the parent, one to the original child).

```
Original:  parent [J_0] child [J_0]
Masked:    parent [J_0] [MASK] child [J_0]
Edited:    parent [J_0] CC [J_0][J_1] child [J_0]       ← insert ethyl linker
```

### 4.3 Mid-Node Replacement: Re-Wiring Children

When replacing a mid-tree node that has children, the new motif must provide output ports for each existing child subtree. Four cases arise:

| Case | Ports | Behavior |
|---|---|---|
| **Match, same sites** | New motif has same `[J_i]` values | Children connect to identical atom positions. Safest edit. |
| **Match, different sites** | New motif has same port count but different `[J_i]` | Children preserved but connect to different atoms. Intentional regiochemistry change. |
| **More ports** | New motif has extra output ports | Extra ports are pushed onto the stack. LLM can grow additional subtrees at these new positions — edit becomes "replace + expand." |
| **Fewer ports** | New motif has fewer output ports | Excess children are truncated (filled with H) by safety fallback. The LLM should anticipate this in its `[THINK]` step and avoid it when undesirable. |

### 4.4 The [LOCATE] Directive

To specify which part of the sequence to edit, the LLM outputs a `[LOCATE] start:end` directive, where `start` and `end` are 0-indexed token positions (inclusive). The type of edit is determined implicitly by what the range covers:

| Range covers | Edit type |
|---|---|
| Motif + joints only (no children) | Mid-node replacement |
| `[B_n]` + branch contents | Branch replacement |
| Motif + joints + all descendant subtrees | Subtree replacement |
| Zero-width position between tokens | Insertion point |

This means a single `[LOCATE]` syntax handles all four edit operations without additional type labels.

### 4.5 LLM Edit Workflow

The complete edit cycle consists of five phases:

**Phase 1 — Analyze.** The LLM receives the full STAMP sequence, computed molecular properties, and the optimization objective. In a `[THINK]` block, it identifies which substructure is responsible for the property deficit.

**Phase 2 — Locate.** The LLM outputs `[LOCATE] start:end` to select the token span for editing.

**Phase 3 — Cut.** The selected span is replaced with `[MASK]`. The rest of the sequence is preserved as context.

**Phase 4 — Grow.** The LLM generates replacement tokens at the `[MASK]` position, using the same stack machine rules as de novo generation. The preserved context (flanking tokens) constrains what the LLM can generate: the new content must be compatible with the port stack state at the `[MASK]` position.

**Phase 5 — Verify.** Properties of the modified molecule are computed. The LLM decides to `[ACCEPT]` or continue editing (potentially with another `[LOCATE]` on a different region, or `[BACK]` to try an alternative replacement).

**Example — multi-round optimization:**

```
[TASK]    optimize
[INPUT]   c1ccccc1 [J_0][J_3] [B_1] CC(=O)N [J_3] c1ccccc1 [J_0]
[PROPS]   logP=3.2, logS=-5.1, MW=211
[GOAL]    logS > -4, maintain MW < 300

[THINK]   logS=-5.1 is too low. The biphenyl system (two phenyl rings)
          is the main driver of poor solubility. The terminal phenyl
          at positions 5:6 contributes ~1.5 logP. Replacing with
          pyridine should improve solubility while maintaining planarity.

[LOCATE]  5:6
[CUT]     c1ccccc1 [J_0][J_3] [B_1] CC(=O)N [J_3] [MASK]
[GROW]    c1ccccc1 [J_0][J_3] [B_1] CC(=O)N [J_3] c1ccncc1 [J_0]
[EVAL]    logP=2.5, logS=-3.8, MW=212
[ACCEPT]

[THINK]   logS improved to -3.8 (target met). But let's see if we can
          do better. The acetamide branch at positions 2:4 could be
          replaced with a more polar group. Try methylsulfonamide.

[LOCATE]  2:4
[CUT]     c1ccccc1 [J_0][J_3] [MASK] c1ccncc1 [J_0]
[GROW]    c1ccccc1 [J_0][J_3] [B_1] CS(=O)(=O)N [J_5] c1ccncc1 [J_0]
[EVAL]    logP=1.8, logS=-2.9, MW=248
[ACCEPT]
```

### 4.6 Unification: All Tasks as Conditional Completion

A key insight is that de novo generation and molecular optimization are not separate frameworks — they are **special cases of the same operation**: filling in `[MASK]` tokens within a (possibly empty) STAMP context.

| Task | Input pattern | [MASK] position |
|---|---|---|
| De novo generation | `[MASK]` | Entire sequence |
| Leaf replacement | `...kept... [MASK]` | Terminal motif |
| Branch replacement | `...kept [MASK] kept...` | One branch span |
| Core swap | `[MASK] kept... kept...` | Root motif only |
| Scaffold decoration | `scaffold [MASK] [MASK] [MASK]` | Multiple leaf positions |
| Fragment linking | `fragA [J] [MASK] fragB [J]` | Between two fragments |
| Fragment growing | `frag [J] [MASK]` | Extension from one fragment |
| Multi-site optimization | `[MASK] kept [MASK] kept [MASK]` | Multiple non-contiguous positions |

This unification means a single model, trained on a single objective (conditional completion of `[MASK]` regions), can perform all major molecular design tasks. The task type is determined entirely by the input pattern — no architectural changes or task-specific heads are needed.

### 4.7 Training Data for Edit Tasks

Edit training data can be generated systematically from existing molecule datasets:

1. **Matched molecular pair (MMP) mining**: Extract molecular pairs from ChEMBL where a single structural change produces a measurable property change. Convert both molecules to STAMP, align their sequences, and identify the differing substring. This becomes a (masked_input, target_output, property_change) training triple.

2. **Synthetic edit trajectories**: Take any molecule, convert to STAMP, randomly mask a subtree, and train the model to reconstruct the original. This is a fragment-level masked language modeling objective.

3. **Retrosynthetic decomposition**: Use retrosynthetic analysis to identify the key bonds in a molecule, mask one reagent's contribution, and train the model to reconstruct it — effectively learning fragment-level chemical logic.

---

## 5. Training Strategy

### 5.1 Four-Phase Training Pipeline

#### Phase 1: Language Modeling on Molecular Sequences

- **Data**: Convert molecules from ChEMBL, ZINC, QM9 into STAMP encodings (no reasoning, no backtrack).
- **Objective**: Standard next-token prediction on STAMP sequences.
- **Outcome**: The LLM learns motif vocabulary, grammar rules, and chemical co-occurrence patterns.

#### Phase 2: Reasoning Distillation (Supervised Fine-Tuning)

- **Data construction**: For each STAMP sequence from Phase 1, insert `[THINK]` annotations at each motif placement using a rule-based system:
  - Compute Δ(logP), Δ(MW), Δ(QED), Δ(SA_score) before and after each placement.
  - Generate natural language explanations: *"Adding {motif_name} changes logP from {x} to {y} because {chemical rationale}."*
  - This annotation can be fully automated using RDKit property calculators and template-based text generation.
- **Objective**: SFT on the annotated sequences. The LLM learns to produce reasoning that is consistent with chemical property changes.
- **Outcome**: The LLM generates chemically grounded reasoning at each decision step.

#### Phase 3: Reinforcement Learning with Backtrack

- **Reward**: Property predictor scores on the final molecule (multi-objective: logP, QED, SA, docking score, etc.) minus a per-`[BACK]` penalty to discourage excessive backtracking.
- **Data sources for backtrack trajectories**:
  - *Monte Carlo search traces*: Use the Phase 2 model to perform beam search/MCTS, recording pruned paths as `[ACT]...[BACK]` sequences.
  - *Counterfactual augmentation*: For known optimal molecules, substitute suboptimal motifs at certain steps, compute property degradation, then append `[BACK]` + correct choice.
  - *RL self-play*: The model explores `[BACK]` usage during PPO/DPO training.
- **Outcome**: The LLM learns *when* to backtrack (recognizing property drift from targets) and *what* to try instead.

#### Phase 4: Edit and Optimization Training

- **Data sources**:
  - *Matched molecular pairs (MMPs)*: From ChEMBL activity cliffs and SAR databases, extract molecule pairs differing by a single substructure change. Convert both to STAMP, compute the diff as a `[LOCATE]→[CUT]→[GROW]` edit trajectory. Annotate with `[THINK]` explaining the structure-activity reasoning.
  - *Fragment-level masked language modeling*: Randomly mask subtrees in existing molecules; train the model to reconstruct the original. This teaches the model the distribution of "what fits here" given surrounding context.
  - *Retrosynthetic decomposition*: Mask one reagent's contribution in a known synthesis, train reconstruction.
  - *Synthetic optimization trajectories*: For molecules with known SAR series, generate multi-round edit sequences showing iterative optimization with property improvement at each step.
- **Objective**: Conditional completion of `[MASK]` regions, with `[THINK]` annotations justifying each edit decision and `[EVAL]` steps verifying property changes.
- **Outcome**: The LLM learns to analyze an existing molecule, identify modifiable regions, propose replacements, and evaluate outcomes — the complete medicinal chemist workflow.

### 5.2 Tokenization Strategies

Two granularity options for different downstream tasks:

**Coarse-grained (fragment-level)**: Each motif canonical SMILES is a single token; `[J_i]`, `[B_n]`, `[THINK]`, `[BACK]`, `[END]` are special tokens. Vocabulary = motif vocab + ~30 special tokens. Best for generation tasks where fragment-level decisions are the natural unit.

**Fine-grained (hybrid)**: Motif SMILES are tokenized at the character level (or via BPE), but `[J_i]`, `[B_n]`, and control tokens remain as single tokens. Best for property prediction tasks where intra-fragment atomic detail matters, while retaining the structural signals from joint and branch tokens.

### 5.3 Data Augmentation

- **Root randomization**: Choose different fragments as root (relaxing canonicalization rule 1).
- **Child order randomization**: Vary which subtree is placed in a `[B_n]` branch vs. on the main chain (relaxing rule 2).
- **Fragmentation variation**: Apply different fragmentation algorithms (exocyclic bonds, BRICS, rotatable bonds) to the same molecule, generating different fragment-level descriptions.

---

## 6. Theoretical Analysis

### 6.1 Validity Guarantee

**Theorem.** *For any sequence of tokens drawn from the STAMP vocabulary, the stack machine derivation produces a chemically valid molecular graph.*

*Proof sketch.* The four safety fallback rules (R4) handle all possible error conditions exhaustively: out-of-range joint indices are wrapped modulo atom count; saturated atoms are skipped; empty-stack motifs become disconnected components; over-length branches are truncated. Since each motif token is a valid canonical SMILES (representing a valid molecular subgraph), and each bond formed by the stack machine connects two atoms with free valence (enforced by fallback R4.2), the resulting graph preserves chemical validity at every step. □

### 6.2 Expressiveness

**Claim.** *STAMP can represent any molecule that can be represented by fragSMILES under the same fragmentation scheme.*

This follows from the fact that any reduced molecular graph can be rooted and traversed as a tree, and the joint tokens `[J_i]` can specify any atom within any fragment as a connection point.

### 6.3 Uniqueness

**Claim.** *Under the four canonicalization rules (Section 2.6), each molecule maps to exactly one STAMP string.*

This follows from: (1) canonical root selection is deterministic, (2) canonical child ordering is deterministic, (3) RDKit canonical SMILES is unique per fragment, and (4) `[J_i]` indices are determined by canonical atom ordering.

### 6.4 Sequence Length

For a molecule with *k* fragments and *e* inter-fragment bonds (= *k* − 1 for a tree):

- **Motif tokens**: *k*
- **Joint tokens**: 2*e* + (number of terminal output ports on root) ≤ 2*k*
- **Branch tokens**: at most *k* − 1 (one per non-root branch entry)
- **Virtual nodes** (φ): **zero** (unlike t-SMILES which requires φ padding for full binary tree)

Total: O(*k*) tokens. In practice, for ZINC-250K, the average token count is comparable to fragSMILES (~17 motif tokens) plus ~15 structural tokens, for ~32 total tokens — shorter than SMILES (~44) and SELFIES (~37).

---

## 7. Application Scenarios

### 7.1 De Novo Generation

In the linear (no-backtrack) mode, STAMP functions as a drop-in replacement for SMILES/SELFIES in any autoregressive generation model. The validity guarantee eliminates post-hoc filtering, and the fragment-level tokenization reduces sequence length and search space.

### 7.2 Property Prediction (ADMET/QSAR)

The `[B_n]` tokens provide explicit hierarchical grouping signals to Transformer attention layers. A natural two-level attention architecture: intra-fragment attention on characters within a motif, and inter-fragment attention mediated by `[J_i]`/`[B_n]` tokens. This structural prior should be particularly beneficial for learning fragment-level contributions to molecular properties.

### 7.3 Lead Optimization

The tree-search generation mode with `[BACK]` is directly suited for lead optimization: start from a known active scaffold, grow at specific positions, evaluate after each addition, and backtrack if the modification hurts activity. The `[THINK]` trace provides a human-readable audit trail of the optimization rationale.

### 7.4 Multi-Objective Optimization

The `[THINK]` step can explicitly track multiple objectives (potency, selectivity, ADMET, synthesizability) and reason about tradeoffs. When objectives conflict, the LLM can articulate the tradeoff in natural language before committing to a decision — or `[BACK]` and try a different balance.

---

## 8. Discussion

### 8.1 Why Not Just Put `:n` Inside the Motif?

An earlier design iteration annotated connection points directly within fragment SMILES using atom map numbers (e.g., `c1cc([c:2])cc([c:1])c1`). This breaks motif uniqueness: the same benzene ring produces different token strings depending on which atoms are connection points. The `[J_i]` external token design resolves this by keeping all connection information outside the motif string.

### 8.2 Relationship to TSDY

TSDY (t-SMILES with dummy atom but without ID) is the most similar existing method. The critical differences are: (1) TSDY inserts `*` into fragment SMILES, destroying canonical form; (2) TSDY relies on positional pairing across fragments, which can fail; (3) TSDY requires virtual nodes (φ) for full binary tree padding; (4) TSDY has no validity guarantee.

### 8.3 Relationship to fragSMILES

fragSMILES shares the motif-purity principle but uses cross-fragment `<index>` pairing — a remote dependency that can produce invalid molecules if indices are inconsistent. STAMP replaces this with local `[J_i]` + stack-automatic bonding, eliminating the remote dependency entirely.

### 8.4 The Isomorphism: Molecule Tree ↔ Search Tree ↔ Reasoning Chain

The deepest insight of this framework is a three-way structural correspondence:

- Each **node** in the molecular fragment tree corresponds to a **state** in the search tree and a **decision step** in the reasoning chain.
- A **branch** (`[B_n]`) in the molecule corresponds to **subproblem decomposition** in reasoning.
- **`[BACK]`** corresponds to **self-correction** in reasoning and **pruning** in search.

This means that advances in LLM reasoning (chain-of-thought, tree-of-thought, self-reflection, process reward models) can be directly transferred to molecular generation, with STAMP providing the chemical validity guarantee throughout.

### 8.5 Limitations and Future Work

- **Fragmentation dependence**: Like all fragment-based methods, the quality of the representation depends on the fragmentation algorithm. Exploring learnable fragmentation is an important direction.
- **Ring formation**: The current tree-based structure does not natively support generating novel ring systems that span multiple fragments. An extension with a `[Ring]` token (analogous to SELFIES) could address this.
- **Backtrack budget**: Unlimited backtracking could cause infinite loops or excessive computation. The generation budget penalty in Phase 3 training addresses this, but optimal budget strategies deserve further study.
- **Evaluation**: Systematic benchmarking against SMILES, SELFIES, t-SMILES, and fragSMILES on standard generation and prediction benchmarks (ChEMBL, ZINC, QM9) with multiple model architectures is needed to validate the approach.

---

## 9. Conclusion

STAMP resolves a longstanding trilemma in molecular representation: maintaining fragment identity uniqueness, encoding hierarchical topology, and guaranteeing sequence validity. By separating motif identity (canonical SMILES tokens) from connection specification (external `[J_i]` tokens) and topology (SELFIES-style `[B_n]` operators), it achieves all three simultaneously.

The reasoning-guided molecule growth framework further transforms molecular generation from a one-shot autoregressive process into a deliberative, multi-step reasoning process — where an LLM can think, act, evaluate, and self-correct at the granularity of chemically meaningful fragments. This aligns AI molecular design with how medicinal chemists actually work: examining a scaffold, reasoning about what to add, evaluating the result, and revising if needed.

---

## Appendix A: Formal Grammar

**Core encoding grammar:**

```
sequence     := root_group (branch | child_group)* [END]
root_group   := MOTIF joint+
child_group  := MOTIF joint+
branch       := [B_n] (child_group){1..n}
joint        := [J_i]       where i ∈ {0, 1, ..., 19}
MOTIF        := any string in the motif vocabulary (canonical SMILES)
```

**Generation with reasoning and backtrack:**

```
gen_sequence := gen_step+
gen_step     := [THINK] text [ACT] action
action       := root_group | child_group | branch | [BACK] | [END]
```

**Editing and optimization:**

```
edit_sequence := edit_step+
edit_step     := [THINK] text [ACT] edit_action
edit_action   := [LOCATE] range [CUT] [GROW] fill_tokens [EVAL] [ACCEPT]
               | [LOCATE] range [CUT] [GROW] fill_tokens [EVAL] [BACK]
range         := start_idx ":" end_idx
fill_tokens   := (root_group | child_group | branch)+
```

**Unified conditional completion (covers both generation and editing):**

```
masked_input  := (token | [MASK])+
completion    := for each [MASK] in masked_input: fill_tokens
```

## Appendix B: Stack Machine Pseudocode

```python
def decode(tokens):
    G = empty_graph()
    S = []           # port stack
    B = []           # branch save stack
    C = []           # checkpoint stack (for BACK)
    
    i = 0
    is_root = True
    
    while i < len(tokens):
        tok = tokens[i]
        
        if tok == '[BACK]':
            if C:
                G, S, B = C.pop()   # restore last checkpoint
            i += 1
            continue
        
        if tok == '[END]':
            break
        
        if tok.startswith('[B_'):
            n = int(tok[3:-1])
            B.append(list(S))      # save stack
            active = S.pop() if S else None
            if active:
                S_branch = [active]
            # process next n child_groups within branch
            # ... (n groups consumed, then restore S from B)
            i += 1
            continue
        
        if tok in MOTIF_VOCAB:
            C.append(snapshot(G, S, B))   # checkpoint
            frag = parse_motif(tok)
            G.add_fragment(frag)
            
            # read subsequent [J_i] tokens
            joints = []
            while i+1 < len(tokens) and tokens[i+1].startswith('[J_'):
                i += 1
                j = int(tokens[i][3:-1])
                j = j % frag.atom_count          # safety: wrap
                j = nearest_unsaturated(frag, j)  # safety: valence
                joints.append(j)
            
            if is_root:
                for j in reversed(joints):
                    S.append(Port(frag, j))
                is_root = False
            else:
                if joints:
                    input_port = Port(frag, joints[0])
                    if S:
                        parent_port = S.pop()
                        G.add_bond(parent_port, input_port)
                    for j in reversed(joints[1:]):
                        S.append(Port(frag, j))
        
        i += 1
    
    # fill remaining ports with H
    for port in S:
        G.add_implicit_H(port)
    
    return G
```

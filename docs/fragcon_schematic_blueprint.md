# Fragment-constrained generation — figure blueprint

Low-fidelity layout spec for the STAMP fragment-constrained figures. These are **two
separate standalone figures**, both placed near `tab:fragcon` (mechanism before the table,
opening the subsection like Fig4 opens PMO; quality bars beside/after the table).

- Build script: `results/stamp_paper/figs/build_fragcon_schematic.py`
  (run with `…/envs/stamp-train-rdkit-2025_09_4/bin/python`).
- Figure 1 (mechanism): `results/stamp_paper/figs/fragcon_mechanism.{svg,pdf,png}`
- Figure 2 (quality vs. baselines): `results/stamp_paper/figs/fragcon_quality.{svg,pdf,png}`
- Drop-in molecule renders: `results/stamp_paper/figs/fragcon_assets/*.png`
- The SVGs keep editable `<text>` — open in Illustrator and restyle to the Fig1/Fig4
  hand-vector house style; the blueprint only fixes layout, token strings, and colour roles.

**Scaffold morphing is dropped.** Under STAMP's (and GenMol's) formulation it is the
*identical* infilling task to linker design, so it would be a duplicate row and bar. Four
tasks remain: linker design, motif extension, scaffold decoration, superstructure gen.

## Core conclusion (the one sentence the figures must defend)

> Every fragment-constrained task is the **same** STAMP operation — freeze the seed's
> motif-groups, infill a masked group-span with an any-order model, decode with the
> stack machine — and it is valid-by-construction because each motif-group is one
> contiguous token span. The four tasks differ **only** in where the mask sits; and on
> that single operation STAMP beats the fragment baselines on quality (Figure 2).

This is the twin of Fig4 (subtree crossover): both rest on "motif-group = contiguous
token span". Fig4 manipulates spans for *optimization*; Figure 1 masks spans for
*constrained generation*.

## Colour roles (keep consistent with the rest of the paper)

| Role | Edge | Fill | Meaning |
|---|---|---|---|
| Seed / kept group | `#0F4D92` (blue_main) | `#DCE6F2` pale blue | given, frozen |
| Generated / masked group | `#B2182B` (ours crimson) | `#F6DAD7` pale crimson, hatched | produced by STAMP |
| Ink (text, arrows) | `#4D4D4D` neutral | — | — |

Seed = blue (given), STAMP output = crimson (the hero colour used for "ours" elsewhere).
In the product render, the seed substructure is highlighted pale blue to show it survives.

## Figure 1 — the four tasks, each as a STAMP masking operation

File: `fragcon_mechanism.{svg,pdf,png}`. Header: *STAMP casts every fragment-constrained
task as one masking operation — freeze the seed motif-groups (blue), any-order infill the
masked group-span (red), stack-machine decode.* Three columns: **input** (seed) →
**STAMP group mask** → **decoded molecule** (product, seed highlighted pale blue).

Each row = one task: `seed render → keep/mask group pattern → product render`.
Patterns (■ = seed/kept blue box, ▨ = generated/masked red-hatched box). Examples are real
molecules; the seed substructure is highlighted in the product to show it survives.

| Task | Pattern | Worked example (seed → product) | Mask placement |
|---|---|---|---|
| Linker design | ■ ▨ ■ | benzene + piperidine → phenethylpiperidine | infill the linker between two fragments |
| Motif extension | ■ ▨ | indole → indole-3-acetic acid | append a group at the growing edge |
| Scaffold decoration | ■ ▨ ■ ▨ | pyridine → 6-aminonicotinic acid | infill masks at open attachment ports |
| Superstructure gen. | ■ ▨ ▨ | furan → furyl-benzoic acid | grow generated groups around the seed |

Linker uses a clean two-fragment example (benzene + piperidine joined by an ethylene
linker), where the generated span is unambiguously the connector — clearer than a drug-sized
backbone. For reference when redrawing, the codec splits each molecule into motif-groups
(motif token + trailing `[J_i]`/`[B_n]`); the blueprint draws each row left-to-right, but in
Illustrator a row may be redrawn as a small tree (masked group as root, seed leaves) to match
the Fig1(d)/Fig4 tree idiom and tie it harder to Fig4 (same "motif-group = contiguous span").

## Figure 2 — quality vs. baselines

File: `fragcon_quality.{svg,pdf,png}`. Grouped bar chart, **Quality (%) from `tab:fragcon`**,
x = four tasks, three methods (SAFE-GPT grey, GenMol blue, STAMP crimson/hero):

| Task | SAFE-GPT | GenMol | STAMP |
|---|---|---|---|
| Linker | 21.7 | 21.9 | **36.6** |
| Motif ext. | 18.6 | 30.1 | **55.1** |
| Decoration | 10.0 | **31.8** | 30.1 |
| Superstructure | 14.3 | 34.8 | **58.0** |

STAMP wins 3/4; decoration is the one near-tie. This carries the headline `Q`; keep the full
V/U/Q/Div/Dist numbers in `tab:fragcon`. Optionally add a Uniqueness panel (STAMP also leads).

## Suggested captions (draft)

The mechanism figure is a single figure with four sub-panels **(a)–(d)**, one per task
(`fig:fragcon_mech`, a wide `figure*`); the quality bars are a separate single-column
`figure` (`fig:fragcon_q`). Both are inserted in `main.tex`: the mechanism figure opens the
fragment-constrained paragraph (before `tab:fragcon`), the quality bars follow the table.

**Mechanism (Fig.~\ref{fig:fragcon_mech}).** *Fragment-constrained generation is a single
STAMP masking operation.* Each task is the same operation with a different mask placement: the
seed is encoded as STAMP motif-groups (blue, retained), the region to generate is replaced by
a masked group-span (red), infilled by an any-order model with bidirectional context, and
decoded by the stack machine. Because each motif-group is one contiguous token span, the seed
is preserved exactly and every product is valid by construction (worked examples; seed
highlighted). **(a)** Linker design — freeze two fragments, generate the linker. **(b)** Motif
extension — freeze the core, append a group at the edge. **(c)** Scaffold decoration — freeze
the scaffold, generate substituents at open ports. **(d)** Superstructure generation — freeze
the seed subtree, grow new groups around it. Scaffold morphing is omitted: it reduces to the
identical infilling task as linker design.

**Quality (Fig.~\ref{fig:fragcon_q}).** *STAMP leads fragment-constrained quality.* Quality
(Q, `tab:fragcon`) against SAFE-GPT and GenMol across the four distinct tasks; STAMP wins on
three of four (scaffold decoration is a near-tie).

## Illustrator refinement checklist

- Restyle boxes to the Fig1/Fig4 stroke weight and corner radius; replace hatch fill with
  the dashed-outline "to-generate" idiom if Fig4 uses one (keep crimson = generated).
- Redraw molecules from the SVG/PNG assets (or re-render larger via the script) and align
  baselines; consider the tree form for Figure 1 rows per the note above.
- Figure 1: tighten row note text so it clears the seed render and divider rules (cosmetic).
- Figure 1: the decoration ▨/■ box count is schematic — set it to the real port count of the
  chosen scaffold in the final draft.
- Figure 2 is already publication-styled (Nature rcParams); mainly drop the `(Table 1)`
  suffix once the table number is fixed, and confirm the baseline ordering matches the table.

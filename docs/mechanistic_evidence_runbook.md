# Mechanistic Evidence Runbook

This runbook implements the narrowed evidence package:

1. STAMP edit decomposition into motif, joint, and branch changes.
2. Cross-language edit-locality comparison using complete representations.
3. Operator-level subtree crossover validity/efficiency analysis.

These scripts do not perform STAMP-internal token-family ablations. Removing
motif, joint, or branch tokens makes the representation non-invertible, so the
implemented analyses characterize complete molecular languages and complete
STAMP trees.

## 1. STAMP Edit Decomposition

Input: CSV/TSV with paired molecules.

Required columns by default:

- `pair_id`
- `edit_class`
- `smiles_a`
- `smiles_b`

### MoleculeACE cliff pairs

Build a pair CSV from MoleculeACE benchmark files:

```bash
python scripts/benchmark/build_moleculeace_cliff_pairs.py \
  --moleculeace-dir /tmp/MoleculeACE/MoleculeACE/Data/benchmark_data \
  --output results/mechanistic/moleculeace_cliff_pairs.csv \
  --min-tanimoto 0.9 \
  --min-activity-diff 1.0 \
  --cliff-filter both
```

The output keeps `pair_id,edit_class,smiles_a,smiles_b` for the edit-analysis
script and also includes `target`, `activity_diff`, `tanimoto_morgan_r2`,
`split_a`, `split_b`, and the original `cliff_mol` flags.

Use caps when a target produces too many highly similar cliffs:

```bash
python scripts/benchmark/build_moleculeace_cliff_pairs.py \
  --moleculeace-dir /tmp/MoleculeACE/MoleculeACE/Data/benchmark_data \
  --output results/mechanistic/moleculeace_cliff_pairs_top.csv \
  --max-pairs-per-target 500 \
  --max-pairs-per-molecule 5
```

Run:

```bash
python scripts/benchmark/analyze_edit_mechanism.py \
  --pairs results/mechanistic/moleculeace_cliff_pairs.csv \
  --output-dir results/mechanistic/edit_mechanism \
  --representations smiles,stamp,selfies,safe,fragsmiles,group-selfies
```

With a motif vocab:

```bash
python scripts/benchmark/analyze_edit_mechanism.py \
  --pairs data/paired_edits.csv \
  --output-dir results/mechanistic/edit_mechanism_vocab \
  --stamp-vocab /path/to/motif_vocab.txt \
  --stamp-force-ring-aware-encode \
  --representations smiles,stamp,selfies,safe,fragsmiles,group-selfies
```

Outputs:

- `stamp_edit_decomposition_pairs.csv`: pair-level motif/joint/branch metrics.
- `stamp_edit_decomposition_by_edit_class.csv`: heatmap-ready summary.
- `cross_language_edit_locality_pairs.csv`: pair-level token locality metrics.
- `cross_language_edit_locality_by_representation.csv`: representation summary.
- `errors.csv`: skipped pairs or unavailable optional representations.

Key columns:

- `delta_motif`: motif insertions, deletions, and substitutions.
- `delta_joint`: changed joint/stereo/ring signatures on retained motifs.
- `delta_branch`: parent relation changes, degree changes, branch-token changes,
  and inserted/deleted subtree roots.
- `stamp_subtree_contiguous_edit`: whether the changed token region is contained
  in one STAMP subtree span on both sides of the pair.

## 2. Operator-Level Subtree Crossover

Input: `.smi`, CSV, or TSV with molecules.

Run:

```bash
python scripts/benchmark/eval_subtree_crossover_operator.py \
  --input data/chembl_sample.smi \
  --output-dir results/mechanistic/subtree_operator \
  --n-trials 5000 \
  --max-mols 50000 \
  --max-span-len 16 \
  --size-delta 4 \
  --operators stamp-subtree,stamp-random-span,smiles-substring
```

Outputs:

- `operator_trials.csv`: one row per sampled edit.
- `operator_summary.csv`: valid, strict-valid, fallback-free, uniqueness,
  parent similarity, and Murcko core-retention rates.
- `input_errors.csv`: molecules skipped during parsing or STAMP encoding.

Interpretation:

- `stamp-subtree` swaps compatible STAMP subtree spans and adjusts branch counts
  along enclosing branch ancestors.
- `stamp-random-span` swaps arbitrary STAMP token spans and exposes how much of
  the validity comes from the decoder fallback rather than tree-aligned edits.
- `smiles-substring` is a deliberately weak string-splice baseline.

Use `strict_valid_rate` and `fallback_free_rate` when making the
valid-by-construction argument; plain `valid_rate` includes decoder recovery.

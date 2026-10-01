# GenMol A Reproduction Matrix

Goal: explain the gap between our current exact attaching baseline and GenMol's reported `Attaching (A)` PMO result.

Current facts:

- Our repaired exact A: `15.4232` sum AUC top-10, `seed=1`, `budget=10000`
- GenMol paper A: `17.641`
- Gap: `-2.2178`

## Hypotheses

We need to separate two possible causes:

1. `vocab mismatch`
   - Our rebuilt PMO fragment vocab differs from the published GenMol vocab.

2. `sampling dynamics mismatch`
   - Our repaired attaching loop differs from GenMol's original top-100 uniform attach loop.

## Matrix

Use four representative tasks:

- `qed`
- `valsartan_smarts`
- `mestranol_similarity`
- `celecoxib_rediscovery`

These cover:

- stable high-performing task
- sparse SMARTS task
- strong similarity failure case
- rediscovery failure case

### Axis A: Vocabulary source

1. `rebuilt_vocab`
   - `results/pmo_fragment_vocab/<task>.csv`

2. `official_vocab`
   - `results/pmo_fragment_vocab_genmol_official/<task>.csv`

### Axis B: Sampling policy

1. `genmol_raw_sampling`
   - top-100 only
   - uniform fragment pair sampling
   - no `seen_pairs`
   - early-return attach loop

2. `repaired_sampling`
   - top-1000 sampling pool
   - `seen_pairs`
   - keep searching within a step for a novel legal child
   - stall diagnostics enabled

## Priority runs

Run these first:

1. `official_vocab + repaired_sampling`
2. `rebuilt_vocab + repaired_sampling`

If the gap remains large under both, vocab is not the main blocker.

Then run:

3. `official_vocab + genmol_raw_sampling`

Only for tasks that do not stall immediately. This isolates sampling-policy drift.

## Expected interpretation

- If `official_vocab` closes most of the gap:
  - main issue is vocab distribution mismatch

- If `official_vocab` does not close the gap:
  - main issue is attaching dynamics / environment sensitivity

- If `genmol_raw_sampling` with official vocab improves over repaired sampling:
  - our repair changed the method materially

- If `genmol_raw_sampling` collapses while repaired sampling works:
  - paper A likely depends on environment-specific dynamics that are not stable under our stack

## Output files

Store probe outputs under:

- `results/baselines/genmol_attaching_official_vocab_<task>_budget10000_seed1_cpu.json`
- `results/baselines/genmol_attaching_exact_<task>_budget10000_seed1_cpu.json`

Summaries should go to:

- `results/baselines/genmol_A_probe_summary.md`
- `results/baselines/genmol_A_probe_summary.csv`

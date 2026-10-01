# GenMol A-Only Reproduction Findings

## Scope

This note isolates GenMol PMO **Attaching-only (A)** behavior and explains why our reproduced `A` score remains below the paper value.

We compared four variants:

1. Our `repaired` attaching wrapper
2. Our `raw` attaching wrapper
3. GenMol official PMO vocab
4. GenMol original PMO entrypoint with `--disable-remask`

All comparisons used the same 23 PMO tasks, `budget=10000` unless stated otherwise.

## Main Findings

### 1. Vocab mismatch is not the main cause

Using GenMol official vocab did **not** improve attaching-only performance.

Representative tasks:

| Task | rebuilt vocab + repaired sampling | official vocab + repaired sampling |
| --- | ---: | ---: |
| `qed` | 0.9417 | 0.9385 |
| `valsartan_smarts` | 0.4113 | 0.2992 |
| `mestranol_similarity` | 0.5598 | 0.5397 |
| `celecoxib_rediscovery` | 0.5672 | 0.5619 |

So the paper gap cannot be explained mainly by fragment vocabulary reconstruction.

### 2. Metric mismatch is not the main cause

Recomputing our logs with GenMol's `top_auc` implementation changed the 23-task sum only slightly:

- our metric sum: `15.4232`
- GenMol metric on the same accepted logs: `15.3778`

This is too small to explain the paper gap to `17.641`.

### 3. Environment mismatch is not the main cause

We built a GenMol-aligned environment:

- Python `3.10`
- `PyTDC 0.4.1`
- `rdkit 2022.09.5`
- `pandas 2.1.0`
- `numpy 1.26.4`

Running official vocab + repaired sampling in that environment gave:

| Task | modern env | genmol-align env |
| --- | ---: | ---: |
| `qed` | 0.9385 | 0.9381 |
| `valsartan_smarts` | 0.2992 | 0.2442 |

Environment alignment did not recover the paper value.

### 4. Our raw wrapper matches the public GenMol raw A-only path

We modified the local GenMol clone to allow `--disable-remask`, so the original PMO entrypoint can run without a checkpoint.

On `qed`, `budget=100`:

| Implementation | AUC top-10 |
| --- | ---: |
| our raw wrapper | 0.8877 |
| GenMol original entrypoint (`--disable-remask`) | 0.8820 |

These are close enough to treat the two raw A-only paths as effectively equivalent.

### 5. The public raw A-only path collapses on sparse tasks

On `valsartan_smarts`, `budget=100`, the original GenMol entrypoint with `--disable-remask` produced:

- `auc_top10 = 0.000`
- `avg_top1 = 0.000`

This is incompatible with the paper's Table 16 attaching score for `valsartan_smarts`.

Our own raw `qed` diagnostics also showed collapse under strict top-100 dynamics:

- raw top-100 diagnostic run stopped at `693` unique oracle calls
- stall pattern was dominated by `duplicate` and `size_fail`

## Conclusion

The remaining gap is **not** mainly due to:

- vocab reconstruction
- AUC implementation
- Python / PyTDC / RDKit version mismatch
- our wrapper inventing a weaker raw attaching loop

The strongest evidence now is:

> The publicly executable raw GenMol A-only path does not reliably reproduce the paper-level `A` behavior, especially on sparse PMO tasks.

Therefore, the paper's reported `Attaching (A)` results likely depend on execution details not fully exposed by the public raw A-only path.

## Key Artifacts

- Repaired exact-A results: `results/baselines/genmol_attaching_exact_*_budget10000_seed1_cpu.json`
- Official-vocab probe: `results/baselines/genmol_attaching_official_vocab_*_budget10000_seed1_cpu.json`
- GenMol-aligned raw wrapper:
  - `results/baselines/genmol_attaching_exact_qed_budget100_seed1_cpu_genmol_align_raw.json`
  - `results/baselines/genmol_attaching_exact_qed_budget10000_seed1_cpu_genmol_align_raw_diag.json`
- Original GenMol entrypoint observations:
  - `results/baselines/genmol_repo_qed_budget100_obs/`
  - `results/baselines/genmol_repo_valsartan_budget100_obs/`

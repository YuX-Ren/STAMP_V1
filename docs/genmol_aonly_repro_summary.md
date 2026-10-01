# GenMol A-Only Reproduction Summary

## Bottom Line

We cannot attribute the GenMol `Attaching (A)` gap mainly to our implementation.

The public raw A-only path, the official vocab, and the aligned environment were all tested. None of them recovered the paper-level `A` behavior.

## Key Comparisons

### Official vocab does not fix the gap

| Task | rebuilt vocab + repaired sampling | official vocab + repaired sampling |
| --- | ---: | ---: |
| `qed` | 0.9417 | 0.9385 |
| `valsartan_smarts` | 0.4113 | 0.2992 |
| `mestranol_similarity` | 0.5598 | 0.5397 |
| `celecoxib_rediscovery` | 0.5672 | 0.5619 |

### GenMol-aligned environment does not fix the gap

| Task | modern env | GenMol-aligned env |
| --- | ---: | ---: |
| `qed` | 0.9385 | 0.9381 |
| `valsartan_smarts` | 0.2992 | 0.2442 |

Aligned environment:
- Python `3.10`
- `PyTDC 0.4.1`
- `rdkit 2022.09.5`

### Our raw wrapper matches the public GenMol raw path

`qed`, `budget=100`:

| Implementation | AUC top-10 |
| --- | ---: |
| our raw wrapper | 0.8877 |
| GenMol original entrypoint (`--disable-remask`) | 0.8820 |

This is close enough to treat both as effectively the same raw A-only behavior.

## Critical Evidence

### Public GenMol raw A-only fails on `valsartan_smarts`

`valsartan_smarts`, `budget=100`, original GenMol entrypoint with `--disable-remask`:

- `auc_top10 = 0.000`
- `avg_top1 = 0.000`

So the public raw A-only path itself does not reproduce the paper's strong attaching result on this sparse task.

### Raw top-100 attaching collapses

Our raw `qed` diagnostic run under strict top-100 dynamics:

- stopped at `693` unique oracle calls
- dominated by `duplicate` and `size_fail`

This matches the failure mode seen when we force raw GenMol-style sampling.

## Conclusion

The remaining explanation is:

> The paper-level `Attaching (A)` results rely on execution details that are not fully recoverable from the public raw A-only path.

The gap is **not** mainly explained by:

- rebuilt vs official vocab
- our AUC implementation
- Python / PyTDC / RDKit version mismatch
- our wrapper inventing a weaker raw attaching loop

## References

- Full note: [genmol_aonly_repro_findings.md](genmol_aonly_repro_findings.md)

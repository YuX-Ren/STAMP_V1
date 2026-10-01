# Phase 2 — STAMP+AO-exclusive test-time methods (results summary)

- Date: 2026-04-22
- Model: `ao_gpt_30m_d768_l12_e10`, epoch 9 (pure AR baseline)
- Dataset: 30M STAMP molecules, `split_random == 'train'`, `all_pass == True`
- N = 1024 samples per run (two seeds: 42, 123)
- Checkpoint dir: `/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009`

## Metric definition — GenMol-aligned

Matches `NVIDIA-Digital-Bio/genmol` `scripts/exps/denovo/run.py`:

```python
df = pd.DataFrame({'smiles': samples, 'qed': qed(samples), 'sa': sa(samples)})
validity = len(df) / num_samples
df = df.drop_duplicates('smiles')
uniqueness = len(df) / len(samples)
df = df[df['qed'] >= 0.6]
df = df[df['sa'] <= 4]
quality = len(df) / num_samples   # this is the headline "GenMol score"
```

Key differences vs earlier V×U×Q product reporting:
- **Quality = GenMol single-intersection formula** (dedup → filter QED≥0.6 ∧ SA≤4 → divide by N), **not** V × U × Q (which assumes unique/quality independence and over-estimates by ~0.3 pp).
- **Thresholds** closed interval: QED ≥ 0.6, SA ≤ 4 (vs earlier strict >0.6, <4).
- **Uniqueness denominator**: `len(samples)` (raw-SMILES `drop_duplicates`).
  For STAMP, raw and canonical dedup are empirically equivalent (uRaw − uCanon ≤ 0.2 pp across all tested configs) because the stack-machine decoder always produces canonical SMILES; there is no SAFE-style "same molecule, different DFS ordering → different string" inflation.

## Results (GenMol-aligned, two-seed mean)

| Config | Q% | U% | V×U×Q | **GenMol** | QED | SA |
|---|---:|---:|---:|---:|---:|---:|
| MW gate (mw=368, boost_th=280) — baseline | 84.52 | 94.53 | 79.90 | **79.64** | 0.741 | 2.94 |
| + depth-T (T: 0.5 → 0.8 over expected_len=14) | 84.28 | 94.24 | 79.42 | 79.42 | 0.738 | 2.89 |
| **+ SA-motif penalty α = 0.3** | 89.26 | 89.99 | 80.34 | **80.08** | 0.733 | 2.69 |
| + SA α = 0.5 | 92.87 | 85.89 | 79.77 | 79.50 | 0.737 | 2.58 |
| + SA α = 1.0 | 91.60 | 75.73 | 69.38 | 69.1  | 0.737 | 2.53 |
| + SA α = 2.0 | 92.23 | 67.09 | 61.88 | 61.6  | 0.766 | 2.71 |
| + depth-T + SA α = 0.5 | 92.68 | 85.89 | 79.60 | 79.3  | 0.734 | 2.54 |
| + depth-T + SA α = 1.0 | 92.24 | 76.61 | 70.67 | 70.4  | 0.735 | 2.52 |
| + depth-T + SA α = 2.0 | 92.29 | 65.77 | 60.70 | 60.4  | 0.763 | 2.73 |

Paper reference: **GenMol Quality = 84.6%** (Table 1, de novo). Gap: ~4.5 pp.

## Findings

1. **MW gate alone is the strongest stable addition.** Motif-MW table enables O(1) per-token structural rejection (mask any motif that would push running MW past 368). Forbids over-large candidates without retraining. Lifts the plain-AR baseline from ~76.7 to **79.64** GenMol.
2. **SA-motif penalty α = 0.3 is the only compound improvement** (+0.44 pp over MW-only). Penalty is a precomputed `log(p_bad / p_good)` table on motifs mined offline from ~2 M train molecules (good: QED>0.7 ∧ SA<3; bad: QED<0.5 ∨ SA>4.5). 2052 motifs have log-ratio > +0.5 (penalized), 8739 have < −0.5 (de-facto preferred).
   - Variance is high: per-seed ΔGenMol = −1.7 / +3.1 pp, so the +0.44 mean is within noise.
3. **Higher α breaks uniqueness.** As α ∈ {0.5, 1.0, 2.0}: Q keeps rising (85 → 93), but U collapses (94 → 86 → 76 → 67). GenMol is unique-bound, so net drops.
4. **depth-T (rising temperature with step index)** is neutral-to-harmful alone and shows **no synergy with SA penalty**. Likely too crude a proxy for actual tree depth.
5. **Uniqueness calibration with GenMol settled.** STAMP's canonical = raw under the stack-machine decoder, so there is no hidden gap.

## Method contributions (paper-ready framing)

- **MW gate**: token-level structural filter leveraging motif-MW table. STAMP-exclusive (SAFE/SMILES tokenizers are character-level, cannot sum per-token MW).
- **SA-motif penalty**: data-driven log-ratio bias on motif logits; requires motif-level vocab (STAMP) and decoupled from model weights (no retraining). Offline mine + single `np.save` on `log_ratio_penalty.npy`.
- **AO rescore (explored separately)**: ≤ M random permutations of the generated sequence, rescored by NLL under AO-GPT; select min/max/std. Improves pure quality to ~86% but drops U sharply, so was shelved in favor of guided-sampling methods that trade less unique.

Artifacts:
- `/scratch/yuxuan.ren/.../eval_epoch_0009/phase2_sa_mine/log_ratio_penalty.npy` — V=12572, clipped to ±3.
- `/scratch/yuxuan.ren/.../eval_epoch_0009/phase2_all_layers/all_layers_results.json` — raw per-config numbers.
- `/scratch/yuxuan.ren/.../eval_epoch_0009/phase2_raw_unique/raw_unique_results.json` — cross-check of raw vs canonical uniqueness.

## Not pursued (candidates for further work)

- **Method #5 — branch-level repair.** Identify low-contribution branches (by per-branch QED/SA decomposition) and regenerate only those, keeping the molecular backbone. STAMP-exclusive (needs tree structure).
- **AO rescore × MW gate combination.** Independent bumps (~+5 pp each), but they likely target the same "drug-like mode" so did not compound in earlier tries; worth a more careful compose (rescore only within MW-gate-filtered top-K).
- **Oracle-guided beam / MCTS with MW gate integrated as hard constraint.** Likely next step if more GenMol uplift is needed.

## Next milestone (in progress)

- Fix bugs in `stamp/benchmark/lm.py::TinyDiffusionLM` sampling path (EOS suppression, `eos_id` initialization, AdaLN misuse on MDM denoiser).
- Train a small MDLM baseline (d=512 / l=8 / 3 ep) from scratch to verify the fixed path produces valid molecules.
- If promising, move to Eso-LM (arXiv 2506.01928) style hybrid (α₀=1 with causal shuffled attention) for α₀-varying eval.

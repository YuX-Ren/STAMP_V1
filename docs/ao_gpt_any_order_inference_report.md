# AO-GPT Pretraining and Any-Order Inference Report

Date: 2026-06-04

Final publishable summary has been consolidated separately:

```text
docs/publishable_results_20260612.md
```

This file is retained as the AO-GPT / any-order inference diagnostic log.

## Current Status

The rerun job is now active:

```text
job_id: 501411.hopper-m-02
name: zinc_l3ao4_0604
state: R
node: hopper-30, 4 x H200
output_dir: /scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g_rerun_20260604
```

This job is aligned to the previous checkpoint:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g
```

Key aligned settings:

```text
MODEL_TYPE=ao_gpt
STAMP_PARQUET=/scratch/yuxuan.ren/data/zinc_l3_full_raw/*.parquet
MOTIF_FALLBACK=1
FREQ_CUTOFF=5000
MAX_ATOMIC_MOTIFS=12000
BATCH_SIZE=256
GLOBAL_BATCH_SIZE=1024
MAX_SEQ_LEN=256
SAVE_EVERY_STEPS=10000
CHECKPOINT_TOP_K_BY_QUALITY=10
KEEP_LAST_STEP_CKPTS=0
CONDITION_MODE=none
```

## Training Objective Diagnosis

The previous checkpoint is not a pure GPT despite the good AR-generation quality.

Evidence from the checkpoint:

```text
model_type=ao_gpt
cfg.model_type=ao_gpt
cfg.ao_random_ratio=0.9
target_pos_emb.weight exists
none_emb.weight exists
```

Evidence from code:

- `scripts/benchmark/train_ao_gpt_parquet.py` passes `ao_random_ratio` into `LMConfig`.
- Non-llama training uses `_compute_ao_gpt_loss(...)`.
- `_compute_ao_gpt_loss(...)` samples `_sample_ao_orders(labels, random_ratio=cfg.ao_random_ratio)`.

Important nuance:

This training objective is **any-order next-token slot prediction**, not direct denoising over an arbitrary placeholder canvas.

The data pipeline builds shifted pairs:

```text
x[j] = ids[j]
y[j] = ids[j + 1]
```

Then AO training permutes prediction slots:

```text
orders = _sample_ao_orders(y)
logits = model(x, orders=orders)
targets = y[orders]
```

So the learned objective is:

```text
given clean x slots in an order, predict the corresponding next-token y slots
```

It is not:

```text
given a canvas full of <unk>/<pad> placeholders, directly predict x[pos]
```

## Incorrect Smoke Test To Avoid Over-Interpreting

I added CLI support for:

```text
--ao-decode-mode ar|ao_canvas_confidence
```

and ran a small smoke test on:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree
```

Results:

```text
mode=ar
N=64
V=98.44
U=98.44
Q=85.94

mode=ao_canvas_confidence
N=64
V=23.44
U=18.75
Q=3.12
```

This low `ao_canvas_confidence` result should **not** be interpreted as proving the model was trained as pure GPT. It mainly shows that the current naive canvas sampler in `stamp/benchmark/lm.py` is mismatched to the AO next-token objective.

## Existing Any-Order Inference To Reuse

There is already better any-order inference code in this repository. It should be reused instead of the naive `ao_canvas_confidence` path.

Relevant scripts:

```text
scripts/hpc/sample_random_order_ar.sh
scripts/hpc/sample_confidence_fixed_len.sh
scripts/hpc/grid_fine_confidence.sh
```

### `sample_random_order_ar.sh`

This implements fixed-length random-order AR in y-space.

Key properties:

- Samples a target sequence length from the empirical training length distribution.
- Keeps `BOS` fixed.
- Fixes `EOS` at the sampled target length.
- Builds a random order over y-indices.
- Sequentially fills `y[orders[k]]`.
- Uses only already-filled real-token context.
- Avoids placeholder/PAD pollution.

This is much closer to the actual AO next-token training objective.

### `sample_confidence_fixed_len.sh`

This implements MaskGIT-style confidence-guided AO decoding in y-space.

Key properties:

- Fixed target length.
- `EOS` is fixed.
- `y[0]` is forced first, matching the first-valid-first training constraint.
- Observed slots come before masked slots in the order.
- Each step selects a high-confidence y-slot and fills it.

This is also aligned to the shifted y-space objective better than direct x-canvas filling.

## Current Caveat Before Reuse

The older inference scripts were written for previous AO checkpoints and mostly assume direct STAMP token decoding.

For the current top12k hybrid checkpoint, reuse requires one extra adaptation:

```text
hybrid vocab tokens -> standard STAMP tokens -> STAMP decoder
```

The helper already exists in:

```text
scripts/benchmark/eval_denovo_taskaware.py
```

Relevant functions:

```text
_load_hybrid_vocab
_hybrid_tokens_to_stamp_tokens
```

So the correct next implementation is not to invent a new sampler, but to port the y-space samplers from:

```text
sample_random_order_ar.sh
sample_confidence_fixed_len.sh
```

into a reusable Python path, then combine them with hybrid-vocab decoding.

## `sample_random_order_ar.sh` Hybrid Smoke

I did not run the original shell script directly because it decodes generated tokens as direct STAMP tokens:

```text
rep = build_representation('stamp')
rep.decode(rep.detokenize(toks))
```

That is wrong for the current top12k hybrid checkpoint. I added a minimal adapted runner that preserves the old fixed-length random-order AR y-space algorithm and only changes the decode path:

```text
scripts/benchmark/eval_random_order_ar_hybrid.py
scripts/hpc/sample_random_order_ar_hybrid_smoke.pbs
```

Smoke job:

```text
job_id: 501449.hopper-m-02
model: /scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree
output: /scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree/random_order_ar_hybrid_smoke_20260604
N: 128
MAX_LEN: 96
```

Results:

```text
AR T=0.6 K=10:
  V=98.44  U=97.66  Q=89.84  U|V=99.21  Q|V=92.06  legacy_GenMol=89.91  D=0.860

random-order AR T=0.6 K=10:
  V=48.44  U=46.09  Q=3.91  U|V=95.16  Q|V=8.06  legacy_GenMol=3.72  D=0.859

random-order AR T=0.7 K=10:
  V=47.66  U=43.75  Q=3.12  U|V=91.80  Q|V=6.56  legacy_GenMol=2.87  D=0.875
```

Interpretation:

- The old random-order AR path can run on the top12k hybrid checkpoint after decode adaptation.
- It is still much worse than identity-order AR on validity and quality.
- The failure is therefore not only a hybrid decoding issue; the fixed-length random-order AR inference policy is still mismatched or under-tuned for this checkpoint.

## Recommended Next Step

1. Add reusable samplers to `stamp/benchmark/lm.py` or a new benchmark utility:

```text
ao_random_order_ar_fixed_length
ao_confidence_fixed_length
```

2. Add modes to `eval_denovo_taskaware.py`:

```text
--ao-decode-mode ar
--ao-decode-mode ao_random_order_ar_fixed_length
--ao-decode-mode ao_confidence_fixed_length
```

3. Re-run validation on the existing checkpoint:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree
```

4. Compare:

```text
AR identity-order
fixed-length random-order AR
fixed-length confidence-guided AO
```

5. Only after that decide whether the current AO objective is sufficient or whether the pretraining objective needs to change toward true in-place infilling.

## Practical Conclusion

The old checkpoint is AO-trained, but my first any-order validation used the wrong inference path.

The right path is to reuse the existing y-space any-order samplers and adapt them to hybrid top12k decoding. The naive `ao_canvas_confidence` result should be kept only as a diagnostic of objective mismatch, not as the final AO inference result.

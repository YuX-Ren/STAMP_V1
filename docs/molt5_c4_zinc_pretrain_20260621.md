# MolT5 C4 + ZINC100M Pretraining

This run follows the MolT5 pretraining data composition: C4 text mixed 50/50
with 100M ZINC SMILES.

## Data

- C4 source: `allenai/c4`
- C4 mode: streaming via HuggingFace `datasets`
- ZINC source: `/scratch/yuxuan.ren/data/molformer_encoded_msring/zinc15/*.parquet`
- ZINC molecule column: `smiles`
- ZINC cap: `100000000` rows sampled/streamed from local ZINC15 parquet shards
- Model/tokenizer init: `/scratch/yuxuan.ren/models/t5-v1_1-base`

## Scripts

- Optional local C4 download: `scripts/hpc/download_c4_molt5_pretrain.pbs`
- Train mixed MolT5: `scripts/hpc/pretrain_molt5_c4_zinc_mixed.pbs`
- Trainer: `scripts/benchmark/train_molt5_c4_zinc_mixed.py`

The trainer uses two separate dataloaders and constructs every micro-batch as:

```text
batch = 50% C4 span-corruption examples + 50% ZINC SMILES span-corruption examples
```

It uses the HuggingFace T5 tokenizer and T5 sentinel tokens (`<extra_id_*>`) for
standard T5 denoising.

## Submitted Jobs

- Failed local C4 download: `513061.hopper-m-02`
- Active mixed MolT5 pretraining: `513073.hopper-m-02`
- Dependency: none. The active job streams C4 instead of requiring local C4 files.

Note: `513062.hopper-m-02` was deleted because it used the earlier `t5-small`
default. `513063.hopper-m-02` was submitted with a dependency on the failed C4
download and therefore did not run. The active training job uses T5.1.1 base and
C4 streaming.

The full local C4 download failed with `Disk quota exceeded` after 190 train
shards, so partial files under `/scratch/yuxuan.ren/data/c4` were removed.

Default training output:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/molt5_c4_zinc100m_t5v11base_4g_<timestamp>
```

## Smoke Test

A 1-step CPU smoke test was run with:

- C4 validation shard: `/scratch/yuxuan.ren/data/c4_hf_probe/en/c4-validation.00000-of-00008.json.gz`
- ZINC shard: `/scratch/yuxuan.ren/data/molformer_encoded_msring/zinc15/shard_00000_msring.parquet`
- Random tiny T5 config

The smoke test completed one train step and one eval step, reporting separate
`c4_ppl` and `zinc_ppl`, confirming the mixed data path works.

T5.1.1 base local load was also checked after installing `sentencepiece` in
`stamp-train-rdkit-2025_09_4`.

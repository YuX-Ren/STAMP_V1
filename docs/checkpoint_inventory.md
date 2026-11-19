# STAMP 本地 checkpoint 清单

已检索本地模型目录，找到 617 个候选权重文件，包含中间训练步、任务微调模型和存储副本。以下 10 个与仓库实验说明对应的模型已发布到 [Hugging Face bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1)，权重合计 **8.604 GiB**，含配套文件约 **8.676 GiB**。已核实匿名公开访问、62 个模型配套文件的远端大小和全部 10 份权重的 SHA-256。路径、实际字节数及配套文件见 [机器可读清单](checkpoint_inventory.json)。本次未运行模型推理。

## 生成和片段补全

以下目录均位于 `/scratch/yuxuan.ren/data/ckpt_stamp/`，权重文件为目录内的 `legacy_tree/model.pt`。

| 模型 | 目录名 | 权重大小 |
| --- | --- | ---: |
| AO-GPT de novo generation | `zinc_l3_full_aogpt_uncond_top12k_4g_rerun_20260604` | 1.253 GiB |
| Fragment infill；linker、motif extension、superstructure | `fcstamp_inplace_balanced_1m_aogpt_top12k_1g` | 1.253 GiB |
| Scaffold decoration | `fcstamp_task_scaffold_super_redesign_shared300k_stereo_from_mixed_len256_ep10` | 1.253 GiB |

每个模型需配套同目录的 `stamp_vocab.txt`、`hybrid_vocab.json`、`vocab.json`、`train_meta.json` 和 `history.json`。

生成模型的 `model.pt` 是指向 `checkpoints/step_00300000.pt` 的符号链接。发布时必须包含实际权重内容。其存储副本还位于 `/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g_rerun_20260604/legacy_tree/`；不需要重复上传。

任务对应关系来源：[生成实验说明](publishable_results_20260612.md)及 `results/fcstamp_genmol_final_20260604/manifest.json`。

## PubChem 编码器

| 模型 | 权重路径 | 权重大小 |
| --- | --- | ---: |
| MLM | `/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666/pretrained_roberta.pt` | 0.136 GiB |
| MTR | `/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666_mtr/pretrained_roberta.pt` | 0.122 GiB |

配套同目录的 `vocab.json`、`pretrain_meta.json` 和 `pretrain_history.json`，以及两者共同引用的 motif 词表：

```text
/scratch/yuxuan.ren/data/pubchem10m/vocab_psvae_ring_v10000_n20k/vocab_top4666.txt
```

已使用 `torch.load(weights_only=True, mmap=True)` 检查两份权重：均包含 encoder、任务 head、配置、词表路径和编码模式；MTR 还包含 descriptor names、mean 和 std。两者记录的 `stamp_force_ring_aware_encode` 均为 `False`。发布后需将配置中的本地词表路径映射到随包词表。

## MolT5

以下模型均位于 `/scratch/Projects/CFP-03/CFP03-CF-130/yuxuan.ren/ckpt/`。每份 `model.safetensors` 大小均为 **0.918 GiB**。

| 模型 | 相对目录 |
| --- | --- |
| 预训练 500k | `molt5_c4_zinc_singleview_zinc_curated30k_t5v11base_4g_20260706_041748/checkpoints/step_00500000/` |
| Mol2text，初始化自 350k | `molt5_chebi20_mol2text_zc30k/best/` |
| Text2mol，初始化自 350k | `molt5_chebi20_text2mol_zc30k/best/` |
| Mol2text，初始化自 500k | `molt5_chebi20_mol2text_zc30k_from500k/best/` |
| Text2mol，初始化自 500k | `molt5_chebi20_text2mol_zc30k_from500k/best/` |

配套 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`，以及下游模型的 `finetune_meta.json`。STAMP 解码另需：

```text
/scratch/yuxuan.ren/data/ckpt_stamp/zinc_curated_vocab_30k_legacy/motif_vocab.txt
```

350k 预训练 checkpoint 未在此次检索范围内找到；对应的两份下游 `best/` 权重仍存在，且 `finetune_meta.json` 明确记录初始化自 350k。它们对应 [7 月训练报告](molt5_stamp_chebi20_20260708.md)中的模型。500k 初始化的下游模型是另一组模型，不能直接替代原报告对应的权重。

500k 预训练目录还包含约 1.836 GiB 的 `training_state.pt`；它不计入上述 8.604 GiB 推理权重总量。若需要恢复训练，再额外发布该文件。

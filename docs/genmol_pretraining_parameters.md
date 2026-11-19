# GenMol 规模预训练与 fragment infill 训练参数

## 预训练数据规模

| 参数 | 实际值 |
| --- | ---: |
| 保留训练记录数 | 425,457,500 |
| 保留验证记录数 | 51,000,653 |
| 合计保留记录数 | 476,458,153 |
| 实际抽样验证记录数 | 20,000 |

以上为训练元数据记录的样本行数。

来源：`/scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree/train_meta.json`。

## Fragment infill checkpoint 训练参数

Checkpoint：`/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/model.pt`。

| 参数 | 实际值 |
| --- | --- |
| 初始化 checkpoint | `zinc_l3_full_aogpt_uncond_top12k_4g/legacy_tree/checkpoints/step_00300000.pt` |
| 训练数据 | `/scratch/yuxuan.ren/data/fcstamp_inplace_finetune_balanced_1m.parquet` |
| 构造样本总数 | 1,000,000 |
| 训练样本数 | 980,127 |
| 验证样本数 | 19,873；实际抽样 8,000 |
| 任务分配 | linker design 333,334；motif extension 333,333；scaffold decoration 333,333 |
| 数据构造方式 | splice；每个分子构造 1 个任务样本 |
| 数据划分字段 | `split_random`；训练 `train`，验证 `val` |
| 模型与目标 | AO-GPT；任意顺序自回归交叉熵，`ao_random_ratio=0.9` |
| Transformer 层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| FFN 维度 | 3,072 |
| 参数量 | 112,057,459 |
| 词表大小 | 12,659 |
| 最大序列长度 | 256；动态 batch padding |
| Dropout | 0.1 |
| Epochs | 3 |
| 每个 epoch 优化步数 | 3,829；合计 11,487 |
| Batch size | 每卡 256；全局 256；梯度累积 1 |
| GPU 数量 | 1 |
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight decay | 0.1 |
| 梯度裁剪 | 1.0 |
| 混合精度 | BF16 |
| 随机种子 | 42 |
| Motif fallback | 开启；频率阈值 5,000；最多 12,000 个 atomic motifs；fallback 概率范围 0.02–0.15 |
| Checkpoint 保存 | 每 500 步；保留最近 4 个 step checkpoint；保存 `model.pt` |

来源：checkpoint 同目录的 `train_meta.json`、`history.json`；数据文件旁的 `.summary.json`；训练实现 `scripts/benchmark/train_ao_gpt_parquet.py`。

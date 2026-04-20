# STAMP Diffusion vs AR: 实验总结

2026-04-16 ~ 2026-04-17 完整实验记录。

---

## 1. 实验目标

在 STAMP + PS-VAE motif 词表框架下，对比 autoregressive (AR) GPT 和 masked diffusion LM (MDLM) 的分子生成质量，探索提升 diffusion 生成质量的方法。

---

## 2. 基础设施

### 2.1 PS-VAE Motif Vocab

从原子级 singleton 切换到 pre-merged ring/bond cluster 起点的 BPE mining：

- **Pre-merge bond constraints**：ring bond / double bond 两端的原子在 BPE 开始前就合并成一个 group
- **Mining 和 tokenization 共用同一初始状态**：消除 vocab 覆盖率问题

| vocab_size | 实际 vocab | avg 序列长 | C 占比 | [J_0] 占比 |
|---|---|---|---|---|
| 500 | 643 | 29.9 | 12.1% | 41.0% |
| **2000** | **2000** | **11.5** | **1.7%** | **13.2%** |

vocab_size=2000 是所有实验的默认配置。

### 2.2 Decoder R4.5 Fallback

解决 AR 生成的断连分子问题：
- R4.5a：无 joint token 的 motif → 合成默认 port
- R4.5b：port stack 为空 → 找价键未饱和的原子
- R4.5c：仍无法连接 → 移除该 fragment

效果：validity 95.7% → 99.8%（t=1.0）→ 100%（t≤0.9）。

### 2.3 Diffusion 采样向量化

`_sample_model_sequences_diffusion` 的 per-row topk loop：
- 原始：Python for-loop，186ms/sample
- 向量化后：batch topk，12ms/sample（**15x 加速**）

---

## 3. 模型架构

### 3.1 统一架构

AR 和 diffusion 共用 `TinyDecoderLM`，区别仅在 `bidirectional` 参数：
- AR：`bidirectional=False`（causal attention mask）
- Diffusion：`bidirectional=True`（full attention，无 mask）

**无 time embedding**。主流 MDLM（Sahoo et al. 2024）不使用 timestep embedding，模型只看到"哪些位置是 `<mask>`"，不知道自己在第几步。初始实现错误地加了 `time_mlp`（DDPM 风格），已移除。

### 3.2 权重迁移

架构完全一致 → AR→Diffusion 权重迁移是精确 `load_state_dict`，无需任何适配。

---

## 4. 完整实验结果

### 4.1 AR GPT Baseline

配置：d_model=256, 4 layers, 10 epochs, vocab_size=2000

| temp | top_k | Valid | Uniq | Novel | **Quality** | IntDiv | MW | SA |
|------|-------|-------|------|-------|-------------|--------|-----|-----|
| 0.8 | 50 | 100% | 99.9% | 99.8% | **81.4%** | 0.860 | 331 | - |
| **0.9** | **50** | **100%** | **99.9%** | **99.7%** | **80.7%** | **0.867** | **322** | **2.87** |
| 1.0 | 50 | 100% | 99.9% | 99.7% | 79.6% | 0.870 | 322 | 2.89 |

### 4.2 Diffusion 实验矩阵

所有 diffusion 模型：d_model=256, 4 layers, vocab_size=2000, bidirectional

| 实验 | epochs | init | 其他改动 | t | Valid | Uniq | **Qual** | SA |
|------|--------|------|---------|---|-------|------|----------|-----|
| **标准 MDLM** | 10 | random | - | 1.0 | 93.6% | 89.1% | **38.6%** | 3.79 |
| 标准 MDLM | 10 | random | - | 1.5 | 93.8% | 95.1% | 35.0% | - |
| 标准 MDLM | 10 | random | - | 2.0 | 89.5% | 96.9% | 26.8% | - |
| 大模型 | 30 | random | d=512, 6L, cosine | 1.0 | 86.1% | 85.9% | 36.2% | 3.56 |
| 无 time-emb | 10 | AR | 移除 time_mlp | 1.0 | 93.3% | 91.8% | 38.5% | 3.80 |
| ChemRole noise | 10 | AR | motif exp=0.5, struct exp=1.5 | 1.0 | 92.7% | 86.6% | 31.1% | 3.87 |
| Eso-LMs | 10 | AR | sort_idx + mixed mask | 1.0 | 14.5% | 89.2% | 4.7% | 2.95 |

### 4.3 对比旧 vocab（vocab_size=500）

| 模型 | vocab | t | Valid | Uniq | Qual |
|------|-------|---|-------|------|------|
| AR | 500 | 0.9 | 100% | 100% | 79.0% |
| Diff | 500 | 1.0 | 98.7% | **14.6%** | 9.6% |
| Diff | 500 | 2.0 | 95.7% | 95.5% | 43.8% |

vocab_size=500 的 diffusion 严重 mode collapse（uniqueness 14.6%），需要 t=2.0 才到 95% uniqueness。vocab_size=2000 在 t=1.0 就达到 89%。

---

## 5. 关键发现

### 5.1 vocab_size 是 diffusion mode collapse 的根因

vocab_size=500 → seed 1647 > vocab_size → BPE 不运行 → tokenizer 只认识单个环和原子 → `C` 占 12%、`[J_0]` 占 41% → diffusion 学 marginal distribution → mode collapse。

vocab_size=2000 → BPE 有 353 个 cross-bond merge → `C` 降到 1.7%、`[J_0]` 降到 13% → 频率自然均衡 → mode collapse 基本解决。

### 5.2 AR 仍显著优于 Diffusion

同等配置下 quality 差距：80.7% vs 38.6%（2x）。差距主要来自 SA score：
- AR SA mean = 2.87
- Diffusion SA mean = 3.79

Diffusion 生成的分子合成难度系统性偏高。原因：并行预测缺乏 AR 的序列化全局一致性约束 → 局部合法但全局不兼容的官能团组合（如 `[SH]=O` 接在异常位置）。

### 5.3 诊断方法论

**模型能力验证**：mask 掉苯环位置后 → 模型预测苯环 prob=89.6%（rank 1）。全 mask 状态 → `C` 以 65% 压倒苯环的 16%。→ 模型有能力但初始状态被高频 token 锁定。

**高温验证**：t=2.0 时 uniqueness 从 14.6% → 95.5%。→ 问题在 token 频率不均衡，不在模型容量。

### 5.4 尝试过但效果有限的方法

| 方法 | 假设 | 结果 | 分析 |
|------|------|------|------|
| 移除 time embedding | time_mlp 让模型在不同 t 学不同策略 → 高 t 保守 | 无改善 (38.5% vs 38.6%) | time embedding 不是瓶颈 |
| 更大模型 (25M params, 30 ep) | 模型容量不足 | 无改善 (36.2%) | 不是容量问题 |
| STAMP-level BPE | 频率均衡化 | uniqueness 84% 但 validity 降到 76% | 额外 tokenizer 层引入解码问题 |
| Chemical-role noise schedule | motif 先 mask → 模型学 decoration | 退化 (31.1%) | 非均匀 alpha 的梯度权重不稳定 |
| Eso-LMs (sort_idx + mixed mask) | AR+MDM 联合训练 | validity 14.5% | 训练/推理 mask 不一致，实现不完整 |

### 5.5 有效的方法

| 方法 | 改善 |
|------|------|
| **vocab_size 500→2000** | uniqueness 14.6%→89.1%（根本解） |
| **高温采样 (t=2.0)** | uniqueness 14.6%→95.5%（对 v500 有效） |
| **Decoder R4.5 fallback** | validity 95.7%→99.8% |
| **采样向量化** | 31min→2min (15x) |

---

## 6. 结论

1. **STAMP + PS-VAE vocab + AR GPT** 是一个强基线：validity 100%, uniqueness 100%, quality 80.7%。

2. **Masked diffusion 在 STAMP 上的 quality gap（~40pp vs AR）是架构性的**，不能靠增大模型、调 noise schedule、或简单的 Eso-LMs 式训练消除。根因是 bidirectional masked prediction 缺乏 AR 的序列化全局一致性约束。

3. **vocab_size 是 diffusion 可用性的前提**：必须足够大（≥2000）让 BPE 均衡 token 频率，否则 mode collapse。

4. 正确实现 Eso-LMs（sort_idx + any-order AR + KV cache）可能是缩小 AR/diffusion gap 的最有前途的方向，但需要更仔细的工程实现（训练和推理的 attention mask 必须严格一致）。

---

## 7. 后续方向

- **Eso-LMs 完整实现**：参照原始 repo 的 `algo.py` + `dit.py`，正确实现 sort_idx + causal mask + KV cache 采样
- **Chemical-role 前向过程（改进版）**：用 per-token 连续 noise schedule 替代二值 role exponent，避免梯度权重问题
- **其他数据集**：moses / guacamol / qm9 验证结论的普适性
- **Conditional generation**：加入性质约束（SA、QED）作为 classifier-free guidance

---

## 8. 保留的模型和结果

```
results/stamp_ar_baseline/
├── vocab/
│   ├── zinc250k_psvae.txt              # vocab_size=500
│   └── zinc250k_psvae_v2k.txt          # vocab_size=2000
├── models/
│   ├── zinc250k_stamp_ar/              # AR v500, 10ep
│   ├── zinc250k_stamp_ar_v2k/          # AR v2k, 10ep  ← 推荐
│   ├── zinc250k_stamp_diffusion/       # Diff v500, 10ep
│   ├── zinc250k_stamp_diffusion_v2k/           # Diff v2k std, 10ep
│   ├── zinc250k_stamp_diffusion_v2k_large/     # Diff v2k large, 30ep
│   └── zinc250k_stamp_diffusion_v2k_chemrole/  # Diff v2k chemrole, 10ep
└── eval/
    ├── zinc250k_stamp_ar_t{0.8,0.9,1.0}_k{0,50}/
    ├── zinc250k_stamp_ar_v2k_t{0.9,1.0}_k50/
    └── zinc250k_stamp_diffusion_v2k_*/
```

---

## 9. 复现命令

```bash
# Mine vocab
PYTHONPATH=. python scripts/experiments_cli.py mine-vocab \
    --input splits/train.smiles --output vocab_v2k.txt \
    --vocab-size 2000 --max-mining-mols 50000 --workers 16

# Train AR
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/experiments_cli.py train-lm \
    --train splits/train.smiles --valid splits/valid.smiles \
    --representation stamp --stamp-vocab vocab_v2k.txt \
    --model-type decoder --output-dir model_ar/ \
    --d-model 256 --n-heads 8 --n-layers 4 --d-ff 1024 \
    --epochs 10 --batch-size 128 --lr 3e-4 --max-seq-len 64 \
    --amp auto --torch-compile --encode-workers 16

# Train Diffusion (with AR init)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/benchmark/train_stamp_diffusion.py \
    --train splits/train.smiles --valid splits/valid.smiles \
    --stamp-vocab vocab_v2k.txt --init-from model_ar/ \
    --output-dir model_diff/ \
    --d-model 256 --n-heads 8 --n-layers 4 --d-ff 1024 \
    --epochs 10 --batch-size 128 --lr 1e-4 --max-seq-len 64 \
    --diffusion-steps 128 --diffusion-tokens-per-step 0 \
    --amp auto --torch-compile --encode-workers 16

# Eval AR (推荐 t=0.9 k=50)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/experiments_cli.py sample-eval \
    --model-dir model_ar/ \
    --train splits/train.smiles --reference splits/test.smiles \
    --output-dir eval_ar/ --n-samples 10000 \
    --temperature 0.9 --top-k 50 \
    --decode-batch-size 512 --eval-workers 8 --assume-train-canonical

# Eval Diffusion (推荐 t=1.0, tps=2)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/benchmark/eval_stamp_diffusion.py \
    --model-dir model_diff/ \
    --train splits/train.smiles --reference splits/test.smiles \
    --output-dir eval_diff/ --n-samples 10000 --max-new-tokens 20 \
    --temperature 1.0 --top-k 0 \
    --diffusion-steps 64 --diffusion-tokens-per-step 2 \
    --decode-batch-size 512 --eval-workers 8 --assume-train-canonical
```

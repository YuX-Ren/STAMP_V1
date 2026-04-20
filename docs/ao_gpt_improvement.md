# AO-GPT 改进文档

更新时间：2026-04-18

---

## 1. 目标

本轮工作决定放弃 `diffusion llm` 作为主线，转而把 `AO-GPT`（any-order autoregressive GPT）作为 STAMP 上的主生成模型，并重点学习论文

- GenMol: `https://arxiv.org/pdf/2501.06158`

中的 **fragment-constrained molecule generation** 思路。

这里的核心不是普通 de novo generation，而是：

1. 给定一个分子或 fragment 上下文；
2. 保留不变的 fragment/token 上下文；
3. 对被选中的 fragment span 做 mask；
4. 让模型在 **原位（in-place）** 把被 mask 的片段补回去。

这要求模型天然支持 **infilling / completion**，而不是简单的 left-to-right suffix continuation。

---

## 2. 对 AO-GPT 的判断

### 2.1 旧版 AO-GPT 的主要问题

旧版 `AO-GPT` 不是“方法不行”，而是实现有明显问题：

1. **训练乱序，但推理是 fully left-to-right**
   - 训练时：`_compute_ao_gpt_loss()` 里会随机 permutation
   - 推理时：`_sample_model_sequences_decoder()` 给的是 identity order
   - 结果：train / inference mismatch

2. **旧版 permutation 污染了目标**
   - pad / ignore 位被一起打乱
   - BOS / EOS 的顺序约束没有处理好
   - 造成无意义的 target alignment

3. **统一模型前向存在对齐错误**
   - logits 和 target 错位一位
   - decoder / ao / diffusion 都受影响

4. **completion 流程本质上还是 GPT suffix continuation**
   - `prompt + <SEP> + target`
   - 这是 prefix-to-suffix，不是 AO 天然适配的 in-place infill

### 2.2 当前判断

AO-GPT 适合继续做，原因是：

1. 形式上比 diffusion 更适合“有完整上下文、只修一部分”的任务
2. AO 允许在保留左右上下文的情况下，对 masked region 用任意顺序重建
3. STAMP 的 fragment/group 结构天然适合把 subtree / branch / motif span 转成局部 infill 问题

---

## 3. 已完成的 AO-GPT 修复

### 3.1 基础训练/采样修复

已完成的关键修复集中在 `stamp/benchmark/lm.py`：

1. **修正前向对齐**
   - `target_cond` 改为前置 zero slot
   - 输出 logits 改为 `logits[:, 1:, :]`

2. **修正 AO permutation**
   - 只对有效 token 位置做 permutation
   - 第一个有效位置固定在最前，保留 BOS 可见性
   - 最后一个有效位置固定在最后，保持 EOS 对齐

3. **保留 decoder / ao 共用模型**
   - 仍使用统一 `TinyDecoderLM`
   - 避免为 AO 再做一套单独架构

4. **正式全量复训并验证**
   - 在 `zinc250k v2k vocab`、`200k/20k/20k split`、`d_model=256, 4L, 10 epoch` 下复训

### 3.2 正式结果

正式同配置对照结果：

- AR：
  `results/formal_compare_20260418/zinc250k_v2k/eval/ar_samecfg_fix_t0.9_k50/denovo_metrics.json`
- AO：
  `results/formal_compare_20260418/zinc250k_v2k/eval/ao_samecfg_fix_t0.9_k50/denovo_metrics.json`

关键指标：

| Model | Validity | Uniqueness | Novelty | Quality | IntDiv |
|---|---:|---:|---:|---:|---:|
| AR samecfg fix | 1.0000 | 0.9979 | 0.9936 | 0.7987 | 0.8663 |
| AO samecfg fix | 1.0000 | 0.9993 | 0.9986 | 0.8088 | 0.8680 |

结论：

- 旧版 AO 差，主要是实现错误
- 修复后 AO 已经追平并略高于 AR

---

## 4. 为什么不继续沿用旧 completion 路线

旧 completion 路线使用：

- `prompt_repr + <SEP> + target_repr`

训练形式是：

- 给 prompt，预测 suffix target

这本质上还是：

- **GPT continuation**

而不是：

- **in-place infilling**

如果目标是学好 GenMol / fragment-constrained generation，旧 completion 路线不够自然，因为：

1. 被保留的右侧上下文并不会真正参与“原位重建”
2. 被 mask 的 span 不是在原始序列里补回去，而是被当成一个新 suffix 续写
3. 这更像 SAFE-GPT 的 prefix generation，不像 AO 天然该做的 masked reconstruction

---

## 5. 当前新的 AO-GPT infill 设计

### 5.1 总体思路

新的设计是：

1. 从完整 STAMP 序列中选取连续 fragment-group span
2. 在原序列中把这段 span 替换成逐 token `<mask>`
3. 保留所有上下文 token 原位不动
4. 训练时：
   - 输入 = masked sequence
   - loss 只落在 masked positions
   - masked positions 按 any-order 重建
5. 推理时：
   - 仍输入完整 masked sequence
   - 观察到的 token 先固定
   - 剩余 `<mask>` 位按置信度挑选、逐步原位填回

这是 **AO-native infill**，不再是 suffix completion。

### 5.2 新增的数据格式

`build-fragment-infill` 会输出：

- `source_smiles`
- `source_repr`
- `prompt_repr`
- `masked_repr`
- `target_repr`
- `mask_start_group`
- `mask_group_count`

其中：

- `prompt_repr`：老式单 `<mask>` prompt，给兼容旧 completion 用
- `masked_repr`：新式逐 token `<mask>` 的原位序列，给 AO in-place infill 用
- `target_repr`：被遮住的真实 span

### 5.3 新增的训练入口

新增命令：

- `train-ao-fragment-infill`

其训练目标不是 `prompt -> suffix`，而是：

- `masked_repr -> masked positions`

实现位置：

- `stamp/benchmark/tasks.py::train_ao_fragment_infill`

### 5.4 新增的评估入口

新增命令：

- `eval-fragment-infill`

评估逻辑：

1. 输入 `masked_repr`
2. 模型原位填回全部 `<mask>`
3. 得到完整 `pred_full_repr`
4. 反解回分子
5. 计算：
   - validity
   - quality
   - exact target match
   - exact full match

实现位置：

- `stamp/benchmark/tasks.py::evaluate_fragment_infill`

---

## 6. 当前实现状态

### 6.1 已打通

以下链路已经通过 tiny smoke test：

1. `build-fragment-infill`
2. `train-ao-fragment-infill`
3. `eval-fragment-infill`

也就是说，新的 AO in-place infill 主干已经可用。

### 6.2 还没有做的正式实验

尚未完成的部分：

1. `zinc250k` 全量 fragment infill 训练
2. 正式的 infill 成功率 / exact reconstruction / quality 对照
3. 与 linker / scaffold / superstructure 等 task-specific 数据集对齐
4. 更接近 GenMol 的 fragment-constrained benchmark 复现

---

## 7. 当前局限

### 7.1 仍是 token-level any-order，不是 group-level any-order

虽然 mask span 是按 fragment-group 采样的，但当前生成仍在 token 级别进行：

- `<mask>` token 逐个被填回

因此它已经比 suffix GPT 自然很多，但仍未做到：

- “一次把一个完整 fragment-group 作为原子动作填回”

### 7.2 数据集还是随机 mask，不是 task-specific mask

当前 `build-fragment-infill` 是：

- 随机选连续 fragment-group span 做 mask

这足够做预训练，但还不等于真实任务中的：

- linker design
- scaffold decoration
- motif extension
- superstructure generation

### 7.3 还没有引入显式 attachment-point constraint

当前 AO infill 主要依赖 STAMP 本身的结构文法和上下文约束，还没额外加入：

- 更强的 attachment-site legality bias
- group-level rewiring penalty
- task-specific candidate reranking

---

## 8. 建议的下一步

### 阶段 A：先做通用 AO fragment infill 预训练

建议优先做：

1. 用 `zinc250k train` 构建大规模 `fragment infill` 数据
2. 训练 `AO-GPT in-place infill` 基础模型
3. 评估：
   - validity
   - quality
   - exact target match rate
   - exact full match rate
   - 保留上下文的恢复率

建议命令：

```bash
PYTHONPATH=. python scripts/benchmark/experiments_cli.py build-fragment-infill \
  --input results/stamp_diffusion_20260416_130732/prep/zinc250k/splits/train.smiles \
  --output results/ao_infill/zinc250k_infill.csv \
  --representation stamp \
  --stamp-vocab results/stamp_ar_baseline/vocab/zinc250k_psvae_v2k.txt \
  --max-mols 200000 \
  --min-mask-groups 1 \
  --max-mask-groups 3
```

```bash
PYTHONPATH=. python scripts/benchmark/experiments_cli.py train-ao-fragment-infill \
  --csv results/ao_infill/zinc250k_infill.csv \
  --representation stamp \
  --stamp-vocab results/stamp_ar_baseline/vocab/zinc250k_psvae_v2k.txt \
  --output-dir results/ao_infill/model \
  --d-model 256 --n-heads 8 --n-layers 4 --d-ff 1024 \
  --epochs 10 --batch-size 128 --lr 3e-4 --max-seq-len 96 \
  --model-type ao_gpt --device cuda
```

```bash
PYTHONPATH=. python scripts/benchmark/experiments_cli.py eval-fragment-infill \
  --model-dir results/ao_infill/model \
  --csv results/ao_infill/zinc250k_infill.csv \
  --output-dir results/ao_infill/eval \
  --max-eval 1000 --n-samples-per-prompt 1 \
  --max-new-tokens 64 --temperature 0.9 --top-k 50
```

### 阶段 B：做 task-specific fragment-constrained generation

在通用 infill 预训练基础上，再做：

1. linker generation
2. scaffold morphing
3. motif extension
4. scaffold decoration
5. superstructure generation

这些任务都可以统一成：

- 给定部分 fragment context
- 在一个或多个 span 上做原位 infill

### 阶段 C：进一步增强 AO infill 的“自然性”

可以继续做但暂时不急：

1. **group-level decoding**
   - 每步生成一个完整 fragment-group，而不是单 token

2. **context-aware masked order**
   - 当前是 observed first + masked later
   - 后续可以把 masked region 内的 order 设计成更贴合 attachment / tree frontier 的顺序

3. **task-conditioned reranking**
   - 在不改变主模型的前提下，用 attachment validity / scaffold preservation / exact-match-aware 分数做 rerank

---

## 9. 当前结论

1. AO-GPT 已经在 de novo generation 上证明自己不是问题：修正后正式结果和 AR 同级。

2. 旧的 `completion` 路线不适合作为 AO 的最终 infill 方案，因为它本质上还是 GPT suffix continuation。

3. 新的 **AO in-place infill** 路线已经在仓库中打通，从形式上更符合 fragment-constrained generation 的目标。

4. 下一步最值得做的不是再讨论 diffusion，而是：
   - 用大规模数据先把 AO 的通用 fragment infill 学稳
   - 再接具体任务做约束生成


下面给你一份**可直接开跑的轻量实验 checklist**。它的目标不是刷 SOTA，而是用尽量小的预算，把 **STAMP 相比 SELFIES / Group SELFIES / SAFE / fragSMILES / t-SMILES** 的优劣分出来。相关工作里，Group SELFIES 重点看 distribution learning 和 random sampling quality，SAFE 特别适合 scaffold decoration / linker design，t-SMILES 强调 low-resource、novelty 和 balanced performance，fragSMILES 主打 chemically meaningful fragments 与独立表达 breaking bonds。([RSC出版][1])

# 0. 先定实验假设

建议你把实验写成验证这 4 条：

* **H1**：STAMP 在相同模型和训练预算下，更容易学，表现为更短或更规整的序列、更快收敛、更高 valid decode。
* **H2**：STAMP 在 constrained generation / completion 上优于 atom-level 表示，因为它显式编码了 attachment 和 tree structure。
* **H3**：STAMP 在 low-resource setting 下更稳。
* **H4**：STAMP 的优势来自表示法，而不是模型规模。

---

# 1. 参赛表示法

先不要全铺太开，建议分两轮。

## 第一轮

* SMILES
* SELFIES
* SAFE
* t-SMILES
* STAMP

## 第二轮

再补：

* Group SELFIES
* fragSMILES

这样第一轮更容易尽快出趋势，而第二轮再补 fragment-based baselines。SAFE 本身就是 fragment block sequence，t-SMILES 是 tree-based fragment representation，Group SELFIES 是 robust fragment-based string，fragSMILES 是 chemically meaningful fragment string。([RSC出版][2])

---

# 2. 数据与模型

## 数据

先用一个主数据集就够：

* **MOSES 或过滤后的 ZINC 子集**

轻量版建议：

* train: 100k
* valid: 10k
* test: 10k

再加 low-resource 子集：

* 10k
* 50k
* 100k

## 模型

统一一个小 decoder-only Transformer：

* 12–30M 参数
* 相同层数、hidden size、batch size、学习率
* 只换 tokenizer / representation

这样能最大程度把差异归因到表示法，而不是架构。

---

# 3. 实验 A：Representation Statistics

这个一定要做，而且很便宜。

## 统计项

对每种表示法在同一批分子上统计：

* Avg char length
* Median char length
* Avg token length
* Median token length
* P95 token length
* Max token length
* Vocabulary size
* Valid encode/decode rate
* Special-token ratio
* Avg branch depth / max depth（适用于树式表示）
* Avg attachment-token count

## 你特别该加的两个比值

* `char_len(repr) / char_len(SMILES)`
* `token_len(repr) / token_len(SMILES)`

## 输出形式

一张表：

| Repr | Avg char | Avg tok | P95 tok | Max tok | Vocab | Decode valid | Special ratio |
| ---- | -------: | ------: | ------: | ------: | ----: | -----------: | ------------: |

再加一张图：

* token length histogram
* 或 cumulative distribution of token length

这组指标很重要，因为 t-SMILES 的工作就专门分析 token distribution 和结构复杂度；Group SELFIES 也把 sequence-level efficiency 和 sampling quality 联系起来。([Nature][3])

---

# 4. 实验 B：De Novo Generation

## 训练

每种表示法训练同一个 LM，next-token prediction。

## 采样

每个模型采样 10k 分子。

## 指标

* Validity
* Uniqueness
* Novelty
* Internal diversity
* Property distribution matching：

  * MW
  * logP
  * QED
  * SA

## 你真正关心的

不是谁某一个指标第一，而是：

* STAMP 是否接近 SELFIES 的稳定性
* STAMP 是否不明显牺牲 novelty / diversity
* STAMP 是否比 SAFE / t-SMILES / fragSMILES 更稳或更容易学

Group SELFIES 已报告 common datasets 上的 distribution learning 改善，SAFE 和 t-SMILES 也都把 de novo generation 当作主评测之一。([RSC出版][1])

---

# 5. 实验 C：Scaffold Decoration

这是我最推荐优先跑的。

SAFE 特别强调 scaffold decoration、linker design、motif extension 这类任务，SAFE-based models 在 scaffold decoration 和 linker design 上相对 SMILES-based approaches 有明显优势。([RSC出版][2])

## 数据构造

从训练集分子里抽 scaffold 和 side chains，构造：

* 输入：scaffold + mask
* 输出：完整分子

## 方法

最轻量的做法就是 **prefix / masked completion**，不用复杂 conditional head。

## 指标

* Validity
* Scaffold preservation rate
* Completion success rate
* Diversity among completions
* Property drift from training distribution

## 你预期的卖点

如果 STAMP 真有优势，这一项应该很能打，因为它天然适合 subtree-aware completion。

---

# 6. 实验 D：Linker Design

SAFE 也把 linker design 当核心任务之一。([RSC出版][2])

## 数据构造

把分子切成：

* fragment A
* linker
* fragment B

输入是 A 和 B，输出补 linker。

## 指标

* Validity
* 两端 fragment 保留率
* 连接成功率
* Diversity
* Avg generated linker length

这是非常适合 STAMP 的任务，因为 attachment semantics 会更直观。

---

# 7. 实验 E：Low-Resource

t-SMILES 的主卖点之一就是 low-resource 下更稳。([Nature][3])

## 设置

对 10k / 50k / 100k 子集重复训练。

## 只看 4 个指标

* Validity
* Novelty
* Scaffold decoration success
* Linker design success

## 要画的图

横轴：数据量
纵轴：指标

如果 STAMP 真提供更强 inductive bias，这张图会很好看。

---

# 8. 建议的最小跑法

如果你想一周内先出第一版结果，按这个来：

## Phase 1

只比较：

* STAMP
* SELFIES
* SAFE
* t-SMILES

## 只跑

* Representation statistics
* De novo generation
* Scaffold decoration
* Low-resource

这四组已经足够说明很多问题。

## Phase 2

再补：

* Group SELFIES
* fragSMILES
* Linker design

---

# 9. 你可以直接照着做的 markdown checklist

```markdown
# STAMP Lightweight Benchmark

## Data
- [ ] Prepare one common molecular dataset (MOSES or filtered ZINC subset)
- [ ] Build train/valid/test split: 100k / 10k / 10k
- [ ] Build low-resource subsets: 10k / 50k / 100k

## Representations
- [ ] SMILES
- [ ] SELFIES
- [ ] SAFE
- [ ] t-SMILES
- [ ] STAMP
- [ ] Group SELFIES (phase 2)
- [ ] fragSMILES (phase 2)

## Shared model setup
- [ ] One decoder-only Transformer
- [ ] Same parameter count
- [ ] Same optimizer / lr / batch size / steps
- [ ] Same sampling settings for all methods

## Experiment A: Representation statistics
- [ ] Avg char length
- [ ] Median char length
- [ ] Avg token length
- [ ] Median token length
- [ ] P95 token length
- [ ] Max token length
- [ ] Vocabulary size
- [ ] Valid encode/decode rate
- [ ] Special-token ratio
- [ ] Avg branch depth / max depth (if applicable)
- [ ] Avg attachment-token count (if applicable)

## Experiment B: De novo generation
- [ ] Train LM on each representation
- [ ] Sample 10k molecules per model
- [ ] Compute validity
- [ ] Compute uniqueness
- [ ] Compute novelty
- [ ] Compute internal diversity
- [ ] Compare MW/logP/QED/SA distributions

## Experiment C: Scaffold decoration
- [ ] Build scaffold-completion dataset
- [ ] Train prefix/masked completion model
- [ ] Evaluate validity
- [ ] Evaluate scaffold preservation
- [ ] Evaluate completion success rate
- [ ] Evaluate diversity

## Experiment D: Linker design
- [ ] Build fragment-linker-fragment dataset
- [ ] Train completion model
- [ ] Evaluate validity
- [ ] Evaluate fragment preservation
- [ ] Evaluate connection success
- [ ] Evaluate diversity

## Experiment E: Low-resource
- [ ] Repeat B/C on 10k subset
- [ ] Repeat B/C on 50k subset
- [ ] Repeat B/C on 100k subset
- [ ] Plot performance vs dataset size

## Tables/Figures
- [ ] Table 1: Representation statistics
- [ ] Table 2: De novo generation metrics
- [ ] Table 3: Scaffold decoration / linker design success
- [ ] Figure 1: Token length histogram
- [ ] Figure 2: Training loss/perplexity curves
- [ ] Figure 3: Low-resource scaling plot
```

---

# 10. 你最值得加的一句实验主张

你后面写论文时，可以围绕这句展开：

> We evaluate STAMP not only as a generative string representation, but also as a structural completion language. Therefore, beyond standard de novo generation metrics, we emphasize constrained completion tasks and representation-efficiency statistics, where the inductive bias of the representation is more directly exposed.



[1]: https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e?utm_source=chatgpt.com "Group SELFIES: a robust fragment-based molecular string ..."
[2]: https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00019f?utm_source=chatgpt.com "Gotta be SAFE: a new framework for molecular design"
[3]: https://www.nature.com/articles/s41467-024-49388-6?utm_source=chatgpt.com "t-SMILES: a fragment-based molecular representation ..."

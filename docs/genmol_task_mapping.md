# GenMol 原始任务到 STAMP 任务的映射

更新时间：2026-04-18

---

## 1. 目的

本文件用于明确回答两个问题：

1. GenMol 原始 fragment-constrained generation 的任务到底是什么；
2. 这些任务目前被如何翻译成 STAMP / AO-GPT 可用的输入格式。

重点不是理论描述，而是：

- 原始数据长什么样；
- 原始代码怎么用；
- 当前 STAMP 版本到底喂给模型什么 prompt。

---

## 2. GenMol 原始数据

原始任务数据文件：

- `https://raw.githubusercontent.com/NVIDIA-Digital-Bio/genmol/refs/heads/main/data/fragments.csv`

本地副本：

- `/tmp/genmol_repo/data/fragments.csv`

列定义：

- `name`
- `smiles`
- `linker_design`
- `motif_extension`
- `scaffold_decoration`
- `superstructure_generation`

注意：

- 原始 `fragments.csv` **没有单独的 `scaffold_morphing` 列**
- 在 GenMol 代码中，`scaffold_morphing` 被并入 `linker_design`

---

## 3. GenMol 原始评估代码

原始任务评估入口：

- `/tmp/genmol_repo/scripts/exps/frag/run.py`

原始采样器：

- `/tmp/genmol_repo/src/genmol/sampler.py`

原始任务逻辑：

1. `linker_design`
   - `Sampler.fragment_linking()`
2. `linker_design_onestep`
   - `Sampler.fragment_linking_onestep()`
3. `motif_extension`
   - `Sampler.fragment_completion()`
4. `scaffold_decoration`
   - `Sampler.fragment_completion()`
5. `superstructure_generation`
   - `Sampler.fragment_completion()`

原始口径指标：

- `Validity`
- `Uniqueness`
- `Diversity`
- `Distance`
- `Quality`

其中：

- `Quality = # {QED >= 0.6 and SA <= 4} / num_samples`
- `Distance = 原始分子 vs 生成结果` 的平均 Tanimoto distance

---

## 4. GenMol 原始默认设置

默认超参文件：

- `/tmp/genmol_repo/scripts/exps/frag/hparams_v2.yaml`

### 4.1 通用设置

- `num_samples = 100`

### 4.2 各任务设置

| Task | softmax_temp | randomness | gamma | min_add_len |
|---|---:|---:|---:|---:|
| `linker_design_onestep` | 1.2 | 3.0 | 0.0 | 30 |
| `linker_design` | 1.2 | 3.0 | 0.0 | 30 |
| `motif_extension` | 1.2 | 1.0 | 0.3 | 24 |
| `scaffold_decoration` | 1.5 | 2.0 | 0.3 | 24 |
| `superstructure_generation` | 1.5 | 2.0 | 0.4 | 24 |

---

## 5. GenMol 原始任务样例

以下表格直接来自 `fragments.csv`。

| name | linker_design | motif_extension | scaffold_decoration | superstructure_generation |
|---|---|---|---|---|
| BARICITINIB | `[11*]C1(CC#N)CN(S(=O)(=O)CC)C1.[13*]c1ncnc2[nH]ccc12` | `[11*]C1(CC#N)CN(S(=O)(=O)CC)C1` | `[1*]N1CC([2*])(n2cc(-c3ncnc4[nH]ccc34)cn2)C1` | `c1nc(-c2cnn(C3CNC3)c2)c2cc[nH]c2n1` |
| CYCLOTHIAZIDE | `[13*]C1Nc2cc(Cl)c(S(N)(=O)=O)cc2S(=O)(=O)N1.[12*]C1CC2C=CC1C2` | `[13*]C1Nc2cc(Cl)c(S(N)(=O)=O)cc2S(=O)(=O)N1` | `[1*]c1cc2c(cc1[2*])NC(C1CC3C=CC1C3)NS2(=O)=O` | `O=S1(=O)NC(C2CC3C=CC2C3)Nc2ccccc21` |
| ELIGLUSTAT | `[11*]N1CCCC1.[17*]c1ccc2c(c1)OCCO2` | `[17*]c1ccc2c(c1)OCCO2` | `[1*][C@H](CN1CCCC1)[C@H]([2*])c1ccc2c(c1)OCCO2` | `c1cc2c(cc1CCCN1CCCC1)OCCO2` |
| ERLOTINIB | `[2*]C#C.[14*]OCCOC` | `[14*]OCCOC` | `[1*]c1cccc(Nc2ncnc3cc([2*])c([3*])cc23)c1` | `c1ccc(Nc2ncnc3ccccc23)cc1` |
| FUTIBATINIB | `[8*][C@H]1CCN(C(=O)C=C)C1.[12*]c1cc(OC)cc(OC)c1` | `[8*][C@H]1CCN(C(=O)C=C)C1` | `[1*]N1CC[C@H](n2nc(C#Cc3cc([2*])cc([3*])c3)c3c([4*])ncnc32)C1` | `C(#Cc1nn([C@H]2CCNC2)c2ncncc12)c1ccccc1` |

其余分子可在：

- `/tmp/genmol_repo/data/fragments.csv`

查看。

---

## 6. 当前 STAMP 映射逻辑

当前映射脚本：

- [`scripts/benchmark/eval_ao_genmol_fragments.py`](../scripts/benchmark/eval_ao_genmol_fragments.py)

映射核心逻辑：

1. 读取 `fragments.csv` 中的原始 fragment/query；
2. 将 query 中的 dummy atom 去掉；
3. 在原始完整分子 `smiles` 上做 substructure match；
4. 将匹配到的原子映射回 STAMP fragment-group；
5. 保留这些 group 作为上下文；
6. 其余 group 全部 mask；
7. 用 AO-GPT 做 full-context in-place infill。

这意味着：

- 现在的 STAMP 任务是 **从原始完整分子中保留 query 对应 group**
- 而不是直接用 SAFE 字符串本身做生成 prompt

---

## 7. 各任务对应的 STAMP 输入样例

以下样例基于 `fragments.csv` 中 BARICITINIB 等分子生成。

### 7.1 Linker Design

原始 query：

```text
[11*]C1(CC#N)CN(S(=O)(=O)CC)C1.[13*]c1ncnc2[nH]ccc12
```

当前 STAMP prompt 示例：

```text
C1CNC1 [J_0] [J_2] [J_0] [B_2] C [J_0] [J_0] C#N [J_0] [B_1] CCS(=O)=O [J_2] <mask> C1=CC2=C(N=CN=C2)N1 [J_7]
```

解释：

- 保留了两侧 fragment 对应的 STAMP group
- 中间 linker 区域整体被 mask

### 7.2 Motif Extension

原始 query：

```text
[11*]C1(CC#N)CN(S(=O)(=O)CC)C1
```

当前 STAMP prompt 示例：

```text
C1CNC1 [J_0] [J_2] [J_0] [B_2] C [J_0] [J_0] C#N [J_0] [B_1] CCS(=O)=O [J_2] <mask>
```

解释：

- 保留给定 motif；
- 让模型往外补剩余结构。

### 7.3 Scaffold Decoration

原始 query：

```text
[1*]N1CC([2*])(n2cc(-c3ncnc4[nH]ccc34)cn2)C1
```

当前 STAMP prompt 示例：

```text
C1CNC1 [J_0] [J_2] [J_0] <mask> C1=CNN=C1 [J_2] [J_0] C1=CC2=C(N=CN=C2)N1 [J_7]
```

解释：

- 保留核心 scaffold 对应的 group；
- 让模型补 decoration。

### 7.4 Superstructure Generation

原始 query：

```text
c1nc(-c2cnn(C3CNC3)c2)c2cc[nH]c2n1
```

当前 STAMP prompt 示例：

```text
C1CNC1 [J_0] [J_2] [J_0] <mask> C1=CNN=C1 [J_2] [J_0] C1=CC2=C(N=CN=C2)N1 [J_7]
```

解释：

- 保留给定 substructure；
- 让模型补完整分子外部剩余部分。

---

## 8. 这套 STAMP 映射“对不对”？

### 8.1 比较合理的部分

以下 3 类任务，当前 STAMP 映射是合理的：

- `motif_extension`
- `scaffold_decoration`
- `superstructure_generation`

原因：

- 它们原始语义就是“保留一个给定子结构/骨架/片段，再补剩余分子”
- 转成 STAMP 的 full-context infill 很自然

### 8.2 还不完全等价的部分

#### `linker_design`

当前做法：

- 保留两侧 fragment
- mask 中间 linker 的 STAMP group

这在任务语义上是合理的，但与 GenMol 原始 SAFE 版并不完全等价：

- GenMol 原始 `fragment_linking()` 是 SAFE 上的序列混合与 linking 过程
- 当前 STAMP 版是 **完整树上下文中的 in-place infill**

因此：

- 语义上接近
- 采样机制不完全相同

#### `linker_design_onestep`

当前 STAMP 还没有为 GenMol 原始 `linker_design_onestep` 单独建一份独立任务输入。

原因：

- 原始 benchmark 数据里没有单独一列
- 原代码只是换了另一种 SAFE 采样函数

如果要严格对齐，需要另外实现：

- STAMP one-step linker prompt

### 8.3 `scaffold_morphing`

GenMol 原始数据里没有单独的 `scaffold_morphing` 列。  
原代码里它直接归入 `linker_design`。

因此：

- 如果按 GenMol 原始 benchmark 严格来，`scaffold_morphing` 不应单独作为一列任务
- 我们之前单独拆出来的 `scaffold_morphing`，只是 STAMP 侧的扩展任务，不是 GenMol 原始 benchmark 本体

---

## 9. 当前推荐做法

如果目标是 **strict fair comparison with GenMol**：

1. 只使用原始 4 个任务：
   - `linker_design`
   - `motif_extension`
   - `scaffold_decoration`
   - `superstructure_generation`
2. 不单独报告 `scaffold_morphing`
3. 单独注明：
   - `linker_design_onestep` 还未严格实现

如果目标是 **STAMP-native 扩展 benchmark**：

1. 保留原始 4 个任务
2. 额外加入：
   - `scaffold_morphing`
3. 但要明确写出它不是 GenMol 原始 benchmark 列

---

## 10. 结论

一句话总结：

- 当前 STAMP 映射里，`motif_extension / scaffold_decoration / superstructure_generation` 基本是对的；
- `linker_design` 语义接近，但采样机制还不是原始 SAFE 版本；
- `scaffold_morphing` 不是 GenMol 原始数据列，而是我们额外扩展出来的任务。

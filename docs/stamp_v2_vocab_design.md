# STAMP V2 Vocab 设计

更新时间：2026-04-19

---

## 1. 问题定义

`STAMP v1` 的 vocab / tokenizer 隐含一个强假设：

- motif 是 **互不重叠** 的 connected subgraph
- motif 之间只通过 **单键 attachment** 连接

这使得 v1 的 vocab 可以直接建立在：

- 非环单键 cut
- 或 PS-VAE 风格的 disjoint principal subgraph

之上。

但一旦引入：

- `[ATT_SPIRO]`

这个假设就不成立了，因为两个 motif 可以：

- **共享一个原子**

这意味着：

1. `v2` 的 motif decomposition 不再是 partition，而是 **overlap-aware cover**
2. vocab 不应只统计 motif core，还要统计 **attachment schema**
3. 后续 merge/BPE 不能只做 disjoint union，还要支持 **shared-atom union**

---

## 2. 设计原则

### 2.1 不污染 v1

所有 v2 vocab 代码都在：

- `stamp_v2/vocab/`

不修改：

- `stamp/vocab/`
- `stamp/core/`
- 现有 benchmark 与结论

### 2.2 先做 seed inventory，再做 merge

当前阶段不直接重写完整 mining/BPE，而是先把更基础的一层做对：

1. 给定一个分子
2. 抽出 **v2 seed motifs**
3. 标出 motif 之间的 attachment kind
4. 统计 corpus-level core/schema 频次

这是后续真正做 v2 merge 的前提。

---

## 3. V2 vocab 的对象

### 3.1 Core vocab

core vocab 仍然是 motif 本体：

- `motif_smiles`

但它不再要求整个分子切完后 motif 彼此不重叠。

### 3.2 Schema vocab

schema vocab 记录的是 motif 的接口形状：

- 哪些 local atom 是 port
- 每个 port 对应哪种 attachment kind
- 同一个 atom 上 attachment 的 multiplicity

当前原型的 attachment kind 只有：

- `bond`
- `spiro`

例如一个 schema signature 可能长成：

```text
0:spiro:1|0:bond:1|3:bond:1
```

表示：

- local atom `0` 上有一个 `spiro` attachment
- local atom `0` 上还有一个 `bond` attachment
- local atom `3` 上有一个 `bond` attachment

---

## 4. 当前 seed decomposition 原型

实现位置：

- `stamp_v2/vocab/seed_inventory.py`

### 4.1 Seed motif 来源

当前原型把 seed motif 分成三类：

1. **fused ring system**
   - 由 SSSR ring 先列出来
   - 若两个 ring 共享 `>=2` 个 atom，则合并成同一个 seed motif
   - 这覆盖 fused / bridged ring 的大部分情况

2. **non-ring unsaturated cluster**
   - 非环的 double/triple bond 连通块
   - 例如 `C=O`、`N=N`、`C#N`

3. **singleton atom**
   - 不属于以上两类的剩余原子

### 4.2 为什么这是 overlap-aware 的

对 spiro 结构：

- 两个 ring 只共享 `1` 个 atom
- 它们不会被 merge 成一个 fused ring seed
- 因此会保留成两个 motif
- 且二者都包含那个 shared atom

这就构成了：

- overlapping motifs

而不是 v1 的 disjoint partition。

---

## 5. 当前 attachment typing 原型

### 5.1 `spiro`

若两个 seed motif：

- atom set 有交集
- 且交集大小正好为 `1`

则记为：

- `ATTACHMENT_SPIRO`

shared atom 在两个 motif 各自的 local index 会被记录进 schema。

### 5.2 `bond`

若分子里有一个：

- 非环单键

连接 motif A 和 motif B 的外部原子，则记为：

- `ATTACHMENT_BOND`

### 5.3 这一步还没做的事

当前还没区分更复杂的 attachment：

- fused-edge sharing
- bridge attachment
- 多原子 shared interface

所以当前原型只适合作为：

- `bond + spiro` 的最小 v2 vocab 基座

---

## 6. 当前代码接口

### 6.1 单分子 inspection

```bash
PYTHONPATH=. python scripts/stamp_v2/inspect_seed_motifs.py \
  "C1CCC2(CC1)CC2"
```

输出：

- motif 列表
- 每个 motif 的 `schema_signature`
- motif 之间的 attachment occurrence

### 6.2 语料级 inventory

```bash
PYTHONPATH=. python scripts/stamp_v2/build_seed_vocab_inventory.py \
  --input data/zinc250k.csv \
  --smiles-column smiles \
  --output-dir results/stamp_v2_seed_inventory \
  --max-mols 10000 \
  --show-progress
```

输出文件：

- `meta.json`
- `core_vocab.tsv`
- `schema_vocab.tsv`

其中：

- `core_vocab.tsv`
  - `motif_smiles`
  - `atom_count`
  - `occurrence_count`
  - `molecule_count`

- `schema_vocab.tsv`
  - `motif_smiles`
  - `schema_signature`
  - `attachment_profile`
  - `occurrence_count`
  - `molecule_count`

---

## 7. 当前和 v1 mining 的本质差异

`v1`：

- 对 molecule 做 disjoint cuts
- motif 本身就是最终 token 候选
- vocab 基本只看 motif 频次

`v2`：

- motif 先变成 overlap-aware seed cover
- vocab 至少要拆成：
  - `core`
  - `schema`
- 同一个 `motif_smiles` 在不同 `schema` 下应被视作不同使用情境

因此，`v2` 不是简单“给 v1 词表多加一个 token”。

---

## 8. 后续真正的 v2 mining 应怎么接

当前 inventory 只是 seed 阶段。

后续若要做真正可训练的 v2 tokenizer / mining，建议按这个顺序：

1. **先冻结 seed extraction**
   - 先把 overlap-aware seed 定义稳定下来

2. **再定义 motif incidence graph**
   - 节点：seed motif occurrence
   - 边：`bond` / `spiro`

3. **再定义 merge candidate**
   - `bond merge`：两个 disjoint motif 沿 bond union
   - `spiro merge`：两个 motif 沿 shared atom union

4. **vocab 不只存 core 频率**
   - 至少存：
     - merged motif core
     - merged motif schema
     - attachment-typed neighbor statistics

5. **encode 时不能只贪心 core**
   - tokenizer 需要 joint score：
     - core frequency
     - schema compatibility
     - attachment kind compatibility

---

## 9. 当前原型的局限

1. ring seed 依赖 SSSR，复杂桥环上可能不够稳定
2. 还没有真正实现 v2 BPE merge
3. 还没有把 inventory 接进 `STAMPV2Encoder.encode_mol()`
4. 还没有处理 `fused/shared-edge` 这类比 `spiro` 更复杂的 shared interface
5. schema 目前只记录 local port 统计，不含 parent/child role

---

## 10. 当前结论

由于引入了 shared-atom attachment，`STAMP v2` 的 vocab 构建必须从：

- **partition vocabulary**

转成：

- **overlap-aware core + schema vocabulary**

当前已落地的独立原型就是这第一步：

- 能从单分子抽出 overlap-aware seed motifs
- 能区分 `bond` 和 `spiro`
- 能把这些 motif 统计成可检查的 `core/schema` vocab inventory

这为后续真正的 `STAMP v2 tokenizer/mining` 提供了基础数据结构。

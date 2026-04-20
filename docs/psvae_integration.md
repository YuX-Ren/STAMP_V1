# PS-VAE Principal Subgraph Tokenizer 接入 STAMP

2026-04-15 完成阶段 a+b，2026-04-16 完成阶段 c、性能优化、decoder R4.5 fallback、AR baseline 训练与调优。本文档记录改动的动机、算法、实现、踩过的坑、验证结果以及后续计划，供回看和 review。

---

## 1. 背景和动机

### 1.1 发现的问题

当天一开始排查代码，发现 STAMP 的 "motif vocab" 实际上**根本没有驱动 encoder 的切分**：

- `stamp/core/stamp_codec.py::STAMPEncoder.encode_mol` 固定调用 `_find_cut_bonds(mol)`，规则硬编码为 `exocyclic_single`（只切非环单键），和任何词表无关。
- `stamp/vocab/vocab_builder.py::StampVocabBuilder.fit` 通过 `generate_candidates` 收集候选 motif，但 `candidate_generation.generate_candidates` 内部也是用 `enc._find_cut_bonds(mol)` 跑同一套 exocyclic 切法 —— 只不过把切出来的片段按频率统计出 "top-N 作为 vocab"。
- 结果：**encoder 和 vocab builder 用的是完全相同的切分规则**；vocab 只是对这些固定切分结果的频率过滤，不能让 encoder 识别更大粒度的 motif，也不能闭合 LM 的 token 空间。
- 更糟的是 `stamp/vocab/canonicalize.py::canonicalize_core` 存到 `canonical_smarts` 字段里的其实是 canonical **SMILES**（调用的是 `MolFragmentToSmiles`，不是 `MolToSmarts`），字段命名完全是误导。`docs/implement.md:732` 也延续了这个误命名。

换句话说：**vocab 是旁观者，不是 tokenizer**。

### 1.2 用户的想法

用 [THUNLP-MT/PS-VAE](https://github.com/THUNLP-MT/PS-VAE) 的 principal subgraph mining（Kong et al., "Molecule Generation by Principal Subgraph Mining and Assembling", ICML 2022）作为 motif 词表构建和 tokenization 方式 —— 它用类 BPE 的迭代合并能产出跨越 `exocyclic_single` 粒度的大 motif（稠环、环+连接段），真正让词表驱动切分。

### 1.3 本次工作范围

- **阶段 a**：在 `stamp/vocab/` 下实现 PS-VAE 风格 mining + tokenizer，保留旧 `StampVocabBuilder` 不动。
- **阶段 b**：`STAMPEncoder` 接入新的 tokenizer，替换切键路径。tree / port / joint / branch 逻辑完全不动。
- **阶段 c（未做）**：LM 侧 `TokenVocab` 从 PS-VAE vocab seed。
- **未做**：scripts CLI wrapper、`canonical_smarts` → `canonical_smiles` 字段重命名。

---

## 2. PS-VAE 算法

从 `https://raw.githubusercontent.com/THUNLP-MT/PS-VAE/main/ps/mol_bpe.py` 拉取完整源码阅读，关键要点：

### 2.1 Mining（`graph_bpe`）

1. **初始化**：每个分子转成 `MolInSubgraph`，每个原子是一个 size-1 subgraph，key 用原子 idx。
2. **种子 vocab**：`MAX_VALENCE.keys()`（`{B, Br, C, Cl, F, I, N, O, P, S, Na}`），频率从语料里用 `cnt_atom`（字符级）统计初值。
3. **迭代合并**（类 BPE）：
   a. 对每个分子跑 `get_nei_subgraphs`：遍历每个当前 subgraph、每个原子的邻居，若邻居属于另一个 subgraph 且 `nei_idx > aid`（去重避免双计），形成合并候选 pair `(pid1, pid2)` 并构造合并后的 union subgraph。
   b. 每个候选用 `get_submol` + `mol2smi` 算 canonical SMILES。
   c. 全语料累加这些候选 SMILES 的频次。
   d. 取 argmax 频次的 SMILES 作为 `merge_smi`，加入词表。
   e. 所有分子应用 `merge(merge_smi)`：凡是当前 subgraph 状态下的 `(pid1, pid2)` 其合并 canonical SMILES 等于 `merge_smi` 的，就合并成一个新 upid（单调递增），旧 pid 删除。
4. 直到词表大小达到 `vocab_size`。

### 2.2 Tokenize（`Tokenizer.tokenize`）

关键发现：**tokenizer 不是回放 merge history**！它在推断时重跑一遍贪心算法，用 vocab 里存储的**频率**作为 priority 查表：

1. 新分子初始化为全原子 singleton `MolInSubgraph`。
2. 循环：计算当前所有邻居合并候选 → 在候选里挑一个 **vocab 内频率最高**的 → 合并。
3. 若没有候选命中 vocab，退出循环。
4. 返回最终 subgraph 列表 `[(smi, [atom_ids]), ...]`。
5. 原子级 fallback 天然具备：种子 vocab 包含所有元素符号，任何新分子都能被完全分解到 atom 级。

### 2.3 词表文件格式

纯文本：第一行是 `{"kekulize": true}` 的 JSON 头，之后每行 `<smi>\t<atom_num>\t<freq>`。PS-VAE 原版兼容。

---

## 3. 实现

### 3.1 新增模块

#### `stamp/vocab/_chem_utils.py`

PS-VAE 原版 `utils/chem_utils.py` 的移植。关键差别：

- `mol2smi` 默认 `canonical=True, isomeric=True` —— 为了保留手性。PS-VAE 原版只开 `canonical=True`，对手性不敏感。
- 其他（`smi2mol` / `get_submol` / `cnt_atom` / `MAX_VALENCE`）完全照搬。

#### `stamp/vocab/principal_subgraph.py`

核心文件。三个导出：

- **`MolInSubgraph`**：原版移植，逻辑和 PS-VAE 一致，但 `mol2smi` 换成带 isomeric 的版本。`upid_cnt` 从 `mol.GetNumAtoms()` 开始单调递增作为新 subgraph 的 pid。
- **`mine_principal_subgraphs(smiles_iter, config, output_path)`**：`graph_bpe` 的移植，用 `PSVAEMineConfig` 打包参数（`vocab_size`, `kekulize`, `num_workers`, `show_progress`）。多进程用 `mp.Pool` + `pool.map(_freq_cnt, mols)`。每轮全量 map-reduce 统计候选频率，再广播合并。输出 PS-VAE 格式词表文件。
- **`PrincipalSubgraphTokenizer(vocab_path)`**：加载词表文件，暴露 `decompose(mol) → List[List[int]]`（每个 principal subgraph 的原子 idx 列表）和 `decompose_smiles(smi)`。`prepare_mol(smi)` 是个便利方法，按 vocab 的 kekulize 模式构造 RDKit mol。

注意 PS-VAE 原版 `Tokenizer` 返回的是 `Molecule(rdkit_mol, group_idxs, kekulize)` 这个容器类，STAMP 不需要它，我直接返回 `group_idxs`。

### 3.2 encoder 接入

`stamp/core/stamp_codec.py::STAMPEncoder`：

- `__init__` 新增 `motif_tokenizer: Optional[PrincipalSubgraphTokenizer] = None` 参数。
- `encode_mol` 分叉：若 tokenizer 提供，走新路径；否则保持老的 `_find_cut_bonds` 路径。
- 新路径（详见第 4 节的坑和修复）：
  1. 在 aromatic 形式下 `AssignStereochemistry`，抓 `_CIPCode` 和 bond stereo 作为参考。
  2. 复制 mol，kekulize（若 tokenizer.kekulize=True），重新 `AssignStereochemistry`。
  3. 把 step 1 捕获的 `_CIPCode` 和 bond stereo 写回 kekulized work_mol，覆盖 step 2 的值。
  4. `motif_tokenizer.decompose(work_mol)` → `groups`。
  5. `_cut_bonds_from_groups(work_mol, groups)` → 只保留"非环单键且跨 group"的 cut bond。
  6. `_build_fragments_and_edges(work_mol, cut_bond_ids)` → 现有代码。
- 新增静态方法 `_cut_bonds_from_groups`（见第 4.1 节）。
- 剩下的 `_emit_node` / port / joint / branch 逻辑**完全没改**。

### 3.3 `stamp/vocab/__init__.py`

导出新符号：`MolInSubgraph`, `PrincipalSubgraphTokenizer`, `PSVAEMineConfig`, `mine_principal_subgraphs`。

### 3.4 测试脚本

`scripts/benchmark/psvae_roundtrip_test.py`：从 zinc250k.csv 抽样 → mine PS-VAE vocab → 对比 legacy 和 PS-VAE 两条路径的 roundtrip 通过率。

用法：

```bash
PYTHONPATH=. python scripts/benchmark/psvae_roundtrip_test.py \
    --data data/zinc250k.csv \
    --mine-n 5000 --test-n 30000 --vocab-size 500 \
    --workers 8 \
    --vocab-out results/psvae/zinc250k_vocab_v2.txt
```

`--skip-mine` 可以复用已有词表直接跑 roundtrip。

---

## 4. 踩的坑和修复

### 4.1 坑 1：singleton fallback 导致双键/环键被切

**症状**：第一版实现跑 alanine `N[C@@H](C)C(=O)O`，decode 回来变成 `N[C@@H](C)C(O)O` —— 羧基的 C=O 变成 C-O，双键丢失。

**根因**：PS-VAE tokenizer 的 atom-level fallback 会把未命中词表的原子留成 size-1 subgraph。对于 alanine 这种训练集里没有的分子，所有原子都是 singleton，所有键（包括 C=O 和 C-C 环键）都成了 "跨 group" 的 cut bond。STAMP decoder `decode_stamp_to_mol` 在 `_try_add_bond_with_fallback` 里**硬编码用 `Chem.BondType.SINGLE`** 重建所有 cut bond，双键于是静默退化成单键。

STAMP grammar（见 `docs/stamp.md`）本来就约定：motif 间的 joint token 只承载单键连接，键级信息不在 token stream 里。这条约束来自老的 `_find_cut_bonds` 只切 exocyclic single bond 的设计。

**修复**：在 `_cut_bonds_from_groups` 里**过滤**掉无效 cut（非单键 或 在环键），让这些 bond 两端的 group 在 `_build_fragments_and_edges` 的连通分量构建阶段自动"连回去"，变成一条大 fragment，emit 为该 fragment 的 canonical SMILES（可能是 vocab 外的 UNK，但结构正确）。

代码：

```python
for bond in mol.GetBonds():
    if bond.GetBondType() != Chem.BondType.SINGLE:
        continue
    if bond.IsInRing():
        continue
    if atom_to_group[a] != atom_to_group[b]:
        cut_bond_ids.append(bond.GetIdx())
```

**启示**：tokenizer 给出的 group 数是 encoder 实际 fragment 数的**上界**；评估词表覆盖率时要分两个指标看。

### 4.2 坑 2：Kekulize 翻转 CIP 和 bond E/Z

**症状**：修完坑 1 之后跑 10,000 分子 roundtrip，legacy 100%，PS-VAE 9990/10000（99.90%）。**5 个失败全部是手性翻转**（`[C@H]` ↔ `[C@@H]`），legacy 路径对这些分子全对。

**排查过程**：

1. 确认失败不是 pre-existing bug —— legacy 路径 100% 通过这 5 个分子 → 是 PS-VAE 接入引入的新问题。
2. 注意到 PS-VAE 路径为了和 kekulized vocab 对齐做了 `Chem.Kekulize(work_mol)`，legacy 没有 —— 差别就在这里。
3. 直接对比：同一分子 `O=C(C1CC1)N1CCC[C@H](Cn2cc[nH+]c2-c2cc3n(n2)CC[NH2+]C3)C1` 在 aromatic 和 kekulized 形式下分别 `AssignStereochemistry` 再读 `_CIPCode`：

   ```
   aromatic:  atom 9 (C) CIP=S  tag=CHI_TETRAHEDRAL_CCW
   kekulized: atom 9 (C) CIP=R  tag=CHI_TETRAHEDRAL_CCW
   ```

**根因**：底层 `ChiralTag` 完全不变（CCW），但 CIP 优先级规则依赖邻居的**键级/重复原子**。aromatic 键在 CIP 里按 1.5 阶处理，kekulized 后展成交替单双键，同一个邻居的"重复"方式就变了，导致 R/S 翻转。

后果链：
- encode 时 kekulized mol 上读到 CIP="R" → 发出 `[S_R]` 标签
- decoder 最终 `Chem.SanitizeMol` 会把分子芳香化
- aromatic 形式下该原子 tag CW 对应 CIP="R"、tag CCW 对应 CIP="S"
- `_restore_chirality` 尝试 CW 第一次命中 "R" → 停在 CW
- 但原分子在 aromatic 形式下应该是 CCW —— 翻了

**修复**：立体标签捕获必须在**和 decoder 最终重建一致的 aromatic 参考系**下做：

```python
aro_mol = Chem.Mol(mol)
Chem.AssignStereochemistry(aro_mol, force=True, cleanIt=True)
cip_by_idx = {a.GetIdx(): a.GetProp("_CIPCode") for a in aro_mol.GetAtoms() if a.HasProp("_CIPCode")}
bond_stereo_by_idx = {
    b.GetIdx(): (int(b.GetStereo()), tuple(b.GetStereoAtoms()))
    for b in aro_mol.GetBonds()
    if b.GetStereo() in (Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ)
}

work_mol = Chem.Mol(mol)
if self.motif_tokenizer.kekulize:
    Chem.Kekulize(work_mol, clearAromaticFlags=True)
Chem.AssignStereochemistry(work_mol, force=True, cleanIt=True)  # 初始化属性
for idx, code in cip_by_idx.items():
    work_mol.GetAtomWithIdx(idx).SetProp("_CIPCode", code)  # 覆盖
for bond_idx, (st_int, atoms) in bond_stereo_by_idx.items():
    b = work_mol.GetBondWithIdx(bond_idx)
    b.SetStereoAtoms(atoms[0], atoms[1])
    b.SetStereo(Chem.rdchem.BondStereo(st_int))
```

修完跑 10,000 roundtrip：4/5 的 tetrahedral 失败修好，剩 1 个是 E/Z 反转 → bond stereo 也按同样方式 override。最后 10,000 roundtrip 99.99%。

**启示**：任何跨 aromatic/kekulized 传递 stereo 标签的 pipeline，都必须保证 encoder 和 decoder 两端的芳香化状态一致。RDKit 的 `_CIPCode` 和 `bond.GetStereo()` 都不是不变量，依赖当前 mol 的键级表达方式。已作为 `feedback` 类型记忆存入 `memory/feedback_kekulize_cip_stereo.md`。

### 4.3 坑 3 (未修)：跨片段的复杂 stereo 拓扑

**症状**：30,000 roundtrip 剩 1 个失败，都是这个分子：
`C/[S@](=N\S(=O)(=O)c1cccc2nsnc12)c1ncccc1C(F)(F)F`

PS-VAE 把它切成 `O=S(=O)/N=S [J_4] [S_R] [J_4] [S_R]` 等片段 —— S=N 双键**在片段内部**，但 E/Z 参考原子（`GetStereoAtoms()`）却指向片段外。decoder 看到 `O=S(=O)/N=S` 单独解析时，左侧参考原子缺失，无法正确恢复 E/Z。

legacy 路径对这个分子的切法不同，恰好把 S=N 周边的所有 stereo 参考原子保留在同一个 motif 里（motif 是 `C\S=N/S(=O)=O`），于是 canonical SMILES 里的 `\`/`/` 标记就够了，不需要跨片段恢复。

**这是结构性限制**，不是 PS-VAE 接入引入的 bug：STAMP 的 joint token 不承载键级或 stereo 拓扑信息，所以"stereo 参考原子跨片段边界"这类极端 case 本来就无法 roundtrip。要真正修需要扩 grammar 加 bond-type / stereo-ref token，属于 STAMP v2 级别的工作。

**决定**：在 30k 分子里 0.003% 的比例下，留作 known issue，不处理。

---

## 5. 验证结果

### 5.1 小规模（10 个手写 test 分子）

legacy 路径和 PS-VAE 路径均 **10/10** 通过，覆盖：

- 完整匹配大 motif：`c1ccccc1CCO`、`CCOc1ccccc1`、`Nc1ccccc1`
- 混合 matched/unmatched：`CCOCC`、`c1ccc(Cl)cc1Br`
- 稠环：`c1ccc2ccccc2c1`（萘）
- 手性中心：`N[C@@H](C)C(=O)O`（alanine）
- 取代酯：`CC(=O)Oc1ccc(N)cc1`、`O=C(Cl)c1ccc(F)cc1`
- 杂环：`c1ccncc1C(=O)O`（吡啶甲酸）

### 5.2 大规模（zinc250k 30,000 分子）

PS-VAE vocab 在 5,000 个训练分子上 mine，`vocab_size=500`，`kekulize=True`，`num_workers=8`：

```
  mined 500 entries in 217.7s
Loaded tokenizer: 500 motifs, kekulize=True

=== Legacy STAMP encoder (exocyclic_single) ===
[legacy] 30000/30000 pass  (100.00%)  failures=0 errors=0  time=110.4s

=== PS-VAE enabled STAMP encoder ===
[psvae ] 29999/30000 pass  (100.00%)  failures=1 errors=0  time=364.3s
   FAIL input: C/[S@](=N\S(=O)(=O)c1cccc2nsnc12)c1ncccc1C(F)(F)F
```

- **Legacy**: 30000/30000 (100.00%)
- **PS-VAE**: 29999/30000 (99.997%) —— 唯一失败是 4.3 节的结构性 case
- **PS-VAE 耗时 ~3.3x legacy**：tokenize 阶段每个分子要重跑一次 greedy BPE，合理范围

### 5.3 词表 sample（来自 `results/psvae/zinc250k_vocab_v2.txt`）

头部几条（按原子数排序）：

```
O=C(NC1=CC=CC=C1)C1=CC=CC=C1    15    ...   ← benzanilide, 最大的 principal subgraph
...
C1=CC=CC=C1                     6     ...   ← benzene
CC(=O)                          3     ...
CC                              2     ...
C                               1     ...
Br                              1     0
...
```

词表最上方是 benzanilide（`O=C(NC1=CC=CC=C1)C1=CC=CC=C1`），说明 zinc250k 里 N-苯基苯甲酰胺骨架是一个高频大 motif —— PS-VAE 把它整块识别为单个 token，legacy 路径会把它切成 3 个片段（两个苯环 + 酰胺连接段），信息密度完全不同。

---

## 6. 文件改动清单

**新增**：

- `stamp/vocab/_chem_utils.py`（54 行）：PS-VAE util 移植，`mol2smi` 强制 isomeric
- `stamp/vocab/principal_subgraph.py`（236 行）：`MolInSubgraph` + `mine_principal_subgraphs` + `PrincipalSubgraphTokenizer`
- `scripts/benchmark/psvae_roundtrip_test.py`（126 行）：roundtrip 测试脚本
- `results/psvae/zinc250k_vocab_v2.txt`：500-entry PS-VAE 词表，从 5000 个 zinc250k 分子 mine
- `docs/psvae_integration.md`：本文档
- `/playpen-shared/mufan/.claude/projects/.../memory/project_psvae_stamp_integration.md`：项目类记忆
- `/playpen-shared/mufan/.claude/projects/.../memory/feedback_kekulize_cip_stereo.md`：feedback 类记忆

**修改**：

- `stamp/core/stamp_codec.py`：
  - `STAMPEncoder.__init__` 加 `motif_tokenizer` 参数
  - `encode_mol` 新增 PS-VAE 分支：aromatic-frame stereo 捕获 → kekulize → override → decompose → cut bonds → build fragments
  - 新增 `_cut_bonds_from_groups` 静态方法（过滤非单键/环键）
  - `from __future__` 下加 `TYPE_CHECKING` + `PrincipalSubgraphTokenizer` 前向声明，避免循环 import
- `stamp/vocab/__init__.py`：导出新符号
- `/playpen-shared/mufan/.claude/projects/.../memory/MEMORY.md`：新增两条记忆索引

**未动**：

- `stamp/vocab/vocab_builder.py` 旧 builder、`stamp/vocab/candidate_generation.py`、`stamp/vocab/canonicalize.py` —— 保持 baseline 可用
- `stamp/benchmark/lm.py::TokenVocab` —— 阶段 c 才处理
- `docs/implement.md` 里 `canonical_smarts` 的误命名 —— 未修

---

## 7. Known issues / 下一步

1. **`canonical_smarts` 误命名**：旧 builder 和 `docs/implement.md` 里存的其实是 canonical SMILES。新路径不依赖这个字段，可以先放着；如果要做 alias 重命名，建议加兼容字段而不是硬改，避免破坏已保存的词表文件。
2. **sulfur stereocenter 跨片段边界**：4.3 节提到的 1/30000 case，结构性限制。要真正支持需要扩 STAMP grammar。
3. **阶段 c (LM TokenVocab seeding)**：PS-VAE vocab 应该直接作为 `stamp/benchmark/lm.py::TokenVocab` 的种子 token 集合，而不是从训练语料频率里构建。改动局限在 `TokenVocab.build` 加一个 `seed_tokens` 参数。
4. **scripts CLI wrapper**：应该把 `mine_principal_subgraphs` + `PrincipalSubgraphTokenizer` 接到 `scripts/experiments_cli.py` 或新增 `scripts/vocab/mine_psvae.py`，让 benchmark pipeline 能直接用。
5. **多进程 mining 性能**：当前 `mp.Pool` 每轮 map 都要 pickle `MolInSubgraph`（含 rdkit Mol），开销不小。2000 mols / 289 iter 耗时 55s，5000 mols / 489 iter 耗时 218s。扩到 250k zinc 全量估计几个小时。要加速可以考虑每个 worker 持久化自己的 mol list、只在 merge 阶段同步 SMILES 字符串。
6. **词表和老的 `candidate_generation` 耦合**：阶段 c 之后，老的 `StampVocabBuilder` 就彻底是 baseline，可以考虑在 CLI 层提供统一的 `VocabBuilder` 接口，底层实现二选一。

---

## 8. 快速上手

```bash
# mine 词表
PYTHONPATH=. python -c "
from stamp.vocab.principal_subgraph import mine_principal_subgraphs, PSVAEMineConfig
from stamp.vocab.vocab_builder import iter_smiles_from_file
cfg = PSVAEMineConfig(vocab_size=500, kekulize=True, num_workers=8, show_progress=True)
smis = list(iter_smiles_from_file('data/zinc250k.csv', smiles_column='smiles', max_mols=5000))
mine_principal_subgraphs(smis, cfg, 'results/psvae/my_vocab.txt')
"

# 用 PS-VAE tokenizer encode 分子
PYTHONPATH=. python -c "
from stamp.core.stamp_codec import STAMPEncoder
from stamp.vocab.principal_subgraph import PrincipalSubgraphTokenizer

tok = PrincipalSubgraphTokenizer('results/psvae/my_vocab.txt')
enc = STAMPEncoder(motif_tokenizer=tok)
print(enc.encode_smiles('O=C(Nc1ccccc1)c1ccccc1'))
"

# 全量 roundtrip 测试
PYTHONPATH=. python scripts/benchmark/psvae_roundtrip_test.py \
    --data data/zinc250k.csv \
    --mine-n 5000 --test-n 30000 --vocab-size 500 \
    --workers 8 \
    --vocab-out results/psvae/zinc250k_vocab_v2.txt
```

---

## 9. 后续改进（2026-04-16）

### 9.1 Mining 性能优化

原 `pool.map(_freq_cnt, mols)` 每轮序列化全部 MolInSubgraph → 200k mols 每轮 ~28s。

三层优化：
1. **Persistent worker**（`_bpe_worker_loop`）：worker 持有本地分片，只通信 freq dict + merge_smi 字符串，零 MolInSubgraph 序列化。
2. **`MolFragmentToSmiles` 替代 `get_submol + mol2smi`**：跳过中间 Mol 创建，~30% 加速。
3. **Incremental freq + `max_mining_mols` 子采样**：只重算 dirty mol 的 freq；大语料自动子采样（默认 50k）。

效果：200k mols, 500 vocab, 16 workers → **58s**（原估数小时）。

### 9.2 Pre-merge bond constraints

**问题**：PS-VAE BPE 从原子级 singleton 开始，不关心 bond 类型。Ring/double bond 两端的原子可能在不同 group → encoder 的 `_cut_bonds_from_groups` 不能切 → 二次合并 → 产生 OOV fragment → LM vocab 膨胀到 92k。

**修复（分两步）**：

**Step 1**（tokenizer 端）：`PrincipalSubgraphTokenizer.decompose()` 返回前用 union-find 合并通过非单键/环键连接的 groups（`_merge_stamp_incompatible`）。vocab 从 92k → 8.6k。

**Step 2**（mining 端）：mining 也从 pre-merged groups 开始 BPE（`_pre_merge_bond_constraints`），而非原子级 singleton。这样 mining 和 tokenization 的初始状态一致。效果：
- Vocab 从 8.6k → **~1500**
- 零 UNK
- 高频 seed fragment（C, N, C=O, 苯环, 吡啶, 噻吩等）自动成为 base token
- BPE 只在 exocyclic single bond 上合并（和 STAMP 约束一致）

`PSVAEMineConfig` 新增 `min_seed_freq: int = 2` 参数，过滤掉只出现 1 次的稀有环系（如 freq=1 的稠合四环）。

### 9.3 Decoder R4.5 fallback

**问题**：AR 模型生成的 token 序列可能导致 port_stack 耗尽或 motif 无 joint → fragment 断连 → validity 从 95.7% 被断连拉低。

**修复**（`stamp/core/stamp_codec.py::decode_stamp_to_mol`）：
- **R4.5a**：motif 后无 `[J_*]` → 在 atom 0 合成默认 port
- **R4.5b**：port_stack 为空 → `_find_fallback_port` 在现有分子里找价键未饱和的原子
- **R4.5c**：fallback 也连不上（价键全满） → **移除该 fragment**，保持连通性

效果：validity 95.7% → **99.8%**（temp=1.0）→ **100%**（temp≤0.9）。Roundtrip 零影响。

### 9.4 TokenVocab seeding + closed_vocab（已废弃）

曾尝试 `closed_vocab=True` 强制只用 PS-VAE motif 做 LM token，其他映射为 UNK。但由于 pre-merge 后一些 ring system 的 canonical SMILES 不在 PS-VAE vocab 里（它们是合并后新产生的），导致 13% UNK。

最终方案：不用 closed_vocab。所有 encoder 实际产出的 fragment SMILES 都进 LM vocab（`min_freq=1`），PS-VAE motif 作为 seed 占据前排 token ID。LM vocab ~1500，零 UNK。

---

## 10. AR GPT Baseline 训练与评估

### 10.1 训练配置

```
数据集: zinc250k (train=200k, valid=20k, test=20k)
PS-VAE vocab: 1647 seed fragments (min_seed_freq=2, max_mining_mols=50k)
LM vocab: ~1500 tokens (seed + structural + training data fragments)
模型: decoder (autoregressive GPT)
  d_model=256, n_heads=8, n_layers=4, d_ff=1024 (~7M params)
训练: 10 epochs, batch=128, lr=3e-4, AMP, torch.compile
```

### 10.2 训练 loss

```
ep1:  train=1.163  valid=0.919
ep5:  train=0.818  valid=0.805
ep10: train=0.767  valid=0.776
```

### 10.3 采样参数 sweep

在同一个模型上测试 6 组采样参数（temp × top_k），每组 10k 样本：

| temp | top_k | Valid | Uniq | Novel | **Quality** | IntDiv | MW_d | QED_d | SA_d |
|------|-------|-------|------|-------|-------------|--------|------|-------|------|
| 0.8 | 50 | 100% | 99.9% | 99.8% | **81.4%** | 0.860 | 15.6 | 0.021 | 0.323 |
| 0.8 | 0 | 100% | 99.9% | 99.8% | 80.1% | 0.862 | 15.8 | 0.017 | 0.253 |
| 0.9 | 50 | 100% | 100% | 99.8% | **79.0%** | 0.867 | 13.6 | 0.018 | 0.224 |
| 0.9 | 0 | 100% | 100% | 99.8% | 74.8% | 0.871 | 12.5 | 0.009 | 0.153 |
| 1.0 | 50 | 100% | 100% | 99.9% | 73.9% | 0.872 | 17.2 | 0.010 | 0.101 |
| 1.0 | 0 | 99.8% | 100% | 99.8% | 68.2% | 0.879 | 16.6 | 0.005 | 0.048 |

### 10.4 分析与推荐

**Quality-Diversity tradeoff**：
- 降温（1.0→0.8）：quality +13pp, IntDiv -0.02
- top_k=50 vs 0：quality +3~8pp（过滤极端 tail 采样）

**推荐配置**：
- **最高 quality**: temp=0.8, top_k=50 → Quality 81.4%, IntDiv 0.860
- **最佳平衡**: temp=0.9, top_k=50 → Quality 79.0%, IntDiv 0.867
- **最高 diversity**: temp=1.0, top_k=0 → Quality 68.2%, IntDiv 0.879

**Distribution distance**：
- 低温提高 QED/SA quality 但增大 MW distance（生成分子略偏大）
- temp=0.9 在 MW distance 和 quality 间平衡最好
- QED distance 在所有配置下都极小（<0.022），说明模型学到了 drug-likeness 分布

### 10.5 对比旧 diffusion

| 指标 | AR GPT (t=0.9, k=50) | Old Diffusion |
|---|---|---|
| Validity | **100%** | 69.3% |
| Uniqueness | **100%** | 28.0% |
| Quality | **79.0%** | 14.0% |
| IntDiv | **0.867** | 0.830 |
| MW mean | **331** | 205 |

AR baseline 全面碾压旧 diffusion（主要归功于 vocab 修复 + decoder R4.5）。

### 10.6 复现命令

```bash
# 1. Mine vocab
PYTHONPATH=. python scripts/experiments_cli.py mine-vocab \
    --input splits/train.smiles --output vocab.txt \
    --vocab-size 500 --max-mining-mols 50000 --workers 16

# 2. Train AR
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/experiments_cli.py train-lm \
    --train splits/train.smiles --valid splits/valid.smiles \
    --representation stamp --stamp-vocab vocab.txt \
    --model-type decoder --output-dir model/ \
    --d-model 256 --n-heads 8 --n-layers 4 --d-ff 1024 \
    --epochs 10 --batch-size 128 --lr 3e-4 \
    --amp auto --torch-compile --encode-workers 16

# 3. Sample & Eval (推荐配置)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/experiments_cli.py sample-eval \
    --model-dir model/ \
    --train splits/train.smiles --reference splits/test.smiles \
    --output-dir eval/ --n-samples 10000 \
    --temperature 0.9 --top-k 50 \
    --decode-batch-size 512 --eval-workers 8 --assume-train-canonical
```

---

## 11. Diffusion LM 实验与分析

### 11.1 vocab_size 对 diffusion 的关键影响

vocab_size=500 时 PS-VAE 的 1647 个 seed fragment 已超过 vocab_size → BPE merge 不运行 → tokenizer 只能识别单个环系和原子 → 训练数据中 `C` 占 12%、`[J_0]` 占 41% → diffusion 学到的主要是高频 token 的 marginal distribution → **mode collapse**。

vocab_size=2000 让 BPE 有 353 个 merge 位 → 合并出 `CC`、`NC=O`、`CC1=CC=CC=C1` 等跨单键组合 → `C` 降到 1.7%、`[J_0]` 降到 13%、序列从 30 → 11.5 token → **频率自然均衡化**。

| vocab_size | 实际 vocab | avg 序列长 | C 占比 | [J_0] 占比 |
|---|---|---|---|---|
| 500 | 643 | 29.9 | 12.1% | 41.0% |
| **2000** | **2000** | **11.5** | **1.7%** | **13.2%** |
| 3000 | 3000 | 10.6 | 1.3% | 12.2% |

### 11.2 诊断过程

1. **初始观察**：vocab_size=500 的 diffusion 生成 uniqueness=11%（mode collapse），所有样本都是 Boc-amine 骨架变体。

2. **模型能力验证**：mask 掉苯环位置后，模型预测苯环 prob=89.6%（rank 1）。但全 mask 初始状态下，`C` 以 65% 压倒性胜出（苯环 16%）。→ 模型会认苯环，但初始采样被高频 token 锁定。

3. **高温采样验证**：temp=2.0 时 uniqueness 从 11% → 95.5%（quality 从 10% → 44%）。→ 问题在于 token 频率不均衡，不是模型容量。

4. **vocab_size=2000 验证**：temp=1.0 就达到 uniqueness=89%（vs v500 的 14.6%）。→ vocab_size 是根本解。

5. **BPE 尝试**（已废弃）：在 STAMP token 上加一层 BPE → uniqueness 84% 但 validity 降到 76%。过于复杂且引入新问题，不如直接增大 vocab_size。

### 11.3 完整对比结果（zinc250k, 10k 样本）

**AR GPT Baseline**（d_model=256, 4 layers, 10 epochs, vocab_size=500/1647 seeds）：

| temp | top_k | Valid | Uniq | Novel | Quality | IntDiv | MW | SA |
|------|-------|-------|------|-------|---------|--------|-----|-----|
| 0.8 | 50 | 100% | 99.9% | 99.8% | **81.4%** | 0.860 | 331 | - |
| **0.9** | **50** | **100%** | **100%** | **99.8%** | **79.0%** | **0.867** | **331** | **2.86** |
| 1.0 | 50 | 100% | 100% | 99.9% | 73.9% | 0.872 | 331 | - |
| 1.0 | 0 | 99.8% | 100% | 99.8% | 68.2% | 0.879 | 331 | - |

**Diffusion vocab_size=500**（d_model=256, 4 layers, 10 epochs）：

| temp | tps | Valid | Uniq | Qual | IntDiv | MW | SA |
|------|-----|-------|------|------|--------|-----|-----|
| 0.9 | 2 | 98.8% | 11.4% | 7.4% | 0.673 | 262 | - |
| 1.0 | 2 | 98.7% | 14.6% | 9.6% | 0.679 | 262 | - |
| 2.0 | 2 | 95.7% | 95.5% | 43.8% | 0.824 | 318 | 3.08 |

**Diffusion vocab_size=2000, 小模型**（d_model=256, 4 layers, 10 epochs）：

| temp | Valid | Uniq | Qual | IntDiv | MW | SA |
|------|-------|------|------|--------|-----|-----|
| 1.0 | 93.6% | 89.1% | 38.6% | 0.895 | 233 | 3.79 |
| 1.5 | 93.8% | 95.1% | 35.0% | 0.908 | 256 | - |
| 2.0 | 89.5% | 96.9% | 26.8% | 0.912 | 287 | - |

**Diffusion vocab_size=2000, 大模型**（d_model=512, 6 layers, 30 epochs, cosine scheduler）：

| temp | Valid | Uniq | Qual | IntDiv | MW | SA |
|------|-------|------|------|--------|-----|-----|
| 1.0 | 86.1% | 85.9% | 36.2% | 0.916 | 220 | 3.56 |
| 1.5 | 86.9% | 93.1% | 33.1% | 0.923 | 233 | 3.71 |
| 2.0 | 88.6% | 97.1% | 27.6% | 0.920 | 278 | 3.95 |

### 11.4 分析结论

1. **vocab_size 是 diffusion mode collapse 的根因**：增大 vocab 从 500 → 2000，temp=1.0 uniqueness 从 14.6% → 89.1%，无需额外的 loss reweighting 或 BPE。

2. **AR 仍然显著优于 diffusion**：AR quality=79% vs diffusion 最高 38.6%。差距主要来自 SA score（AR 2.86 vs diffusion 3.56-3.79），diffusion 生成的分子合成难度系统性偏高。

3. **SA 偏高的原因**：diffusion 并行填充 token 位置，缺乏 AR 的序列化上下文 → 局部合法但全局不兼容的官能团组合（如 `[SH]=O`、带电 N 接异常位置）。

4. **增大模型（25M vs 7M）+ 更多 epoch（30 vs 10）+ cosine scheduler 收效有限**：SA 从 3.79 → 3.56 仅小幅改善，quality 无明显提升。说明这是 masked diffusion 架构的固有局限而非模型容量问题。

5. **diversity 方面 diffusion 有优势**：IntDiv 0.92 vs AR 0.87（diffusion 生成更多样化的结构），但质量-多样性的 tradeoff 上 AR 更优。

### 11.5 后续方向

- **Semi-autoregressive**：先 AR 生成 tree skeleton（root motif + 主链），再 diffusion 并行填充 branch 细节
- **Classifier-free guidance**：训练时加入 SA/QED 条件，采样时引导模型生成低 SA 分子
- **Motif-aware masking**：训练时只 mask motif 位置，保留 structural token 可见，让模型学习"给定骨架预测 motif"
- **其他数据集验证**：在 moses/guacamol/qm9 上对比 AR vs diffusion

### 11.6 采样性能优化

原始 diffusion 采样有 Python per-row loop 瓶颈（`torch.topk` per sample），用 batch topk 向量化后 **15x 加速**（186ms/sample → 12ms/sample，10k 样本 31min → 2min）。改动在 `stamp/benchmark/lm.py::_sample_model_sequences_diffusion` 的 transfer_index 计算。

---

## 12. 文件清理记录

2026-04-16 清理了以下中间实验目录：
- BPE 实验（`stamp_diffusion_bpe*`）— 方向废弃
- 中间采样 sweep（`hightemp_*`, `random_*`, `low_confidence_*`）
- 旧 diffusion 训练结果（`stamp_diffusion_20260416_*`中的 models/eval/vocab）
- 临时文件和训练日志

保留的结果目录：
- `models/zinc250k_stamp_ar` — AR baseline (10 epoch, d_model=256)
- `models/zinc250k_stamp_diffusion` — Diffusion v500 (10 epoch, d_model=256)
- `models/zinc250k_stamp_diffusion_v2k` — Diffusion v2k (10 epoch, d_model=256)
- `models/zinc250k_stamp_diffusion_v2k_large` — Diffusion v2k (30 epoch, d_model=512, cosine)
- 对应的 eval 目录（AR 6 组 sweep + diffusion 各配置）

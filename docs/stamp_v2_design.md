# STAMP V2 设计草案

更新时间：2026-04-19

---

## 1. 目标

本设计文件描述一个 **与现有 `stamp/` 主线完全隔离** 的 `STAMP v2` 原型。

动机来自两个现象：

1. 在 GenMol 原始 fragment benchmark 上，某些 query fragment 会落到 STAMP v1 词表之外；
2. 这些 query 往往对应 ring / bridge / spiro 类复杂 motif，无法用 v1 的 `exocyclic_single` cut rule 继续 fallback 成更细粒度的 STAMP token。

因此，v2 的目标不是“改良 v1 benchmark”，而是：

- 引入新的 attachment 语义；
- 允许某些原本不可拆的 motif 用更细粒度的 token 组合表达；
- 在独立目录中实现，不污染 v1 的结论。

---

## 2. 与 V1 的关系

### 2.1 目录隔离

v2 原型代码位于：

- `stamp_v2/`

v1 主线代码仍位于：

- `stamp/`

这意味着：

- v2 不替换现有 codec / benchmark / vocab 逻辑；
- v1 的实验结果与结论不受影响；
- 任何 v2 评估都必须显式调用 `stamp_v2` 目录中的实现。

### 2.2 当前阶段

当前阶段只做：

1. grammar 设计
2. decoder 原型
3. 最小 CLI / self-check

暂不做：

1. v2 tokenizer mining
2. v2 training
3. 替换现有 benchmark 管线

### 2.3 当前已落地文件

当前独立原型已放在：

- `stamp_v2/core/stamp_codec_v2.py`
- `stamp_v2/vocab/seed_inventory.py`
- `scripts/stamp_v2/selfcheck.py`
- `scripts/stamp_v2/stamp_to_smiles_v2.py`
- `scripts/stamp_v2/inspect_seed_motifs.py`
- `scripts/stamp_v2/build_seed_vocab_inventory.py`

约束如下：

- `stamp_v2` 不修改 `stamp/` 现有实现；
- `STAMPV2Encoder` 目前只把 **decode** 升格到 v2；
- `encode_mol()` 仍沿用 v1 路径，因此不会自动产出 `[ATT_SPIRO]`；
- v2 目前是 **手工构造 token 序列可解码** 的状态，还不是完整 tokenizer / training system。

---

## 3. V1 的限制

### 3.1 V1 的连接语义只有一种

v1 只有：

- motif token
- joint token `[J_i]`
- branch token `[B_n]`

其中 fragment 之间的连接语义只有：

- “通过一个单键连接两个 fragment 的外部端口”

这对应 v1 中的：

- `_find_cut_bonds()` 只切 `exocyclic single bond`
- decoder 中 `_try_add_bond_with_fallback()` 只做 `BondType.SINGLE`

### 3.2 V1 不能自然表达 shared-atom attachment

对于这类结构：

- spiro system
- shared-atom bicyclic attachment
- 某些桥环 / 稠环交汇原子

如果想把它们拆成两个 motif，再用 v1 重建，会遇到问题：

- 两个 motif 不是“通过单键连起来”
- 它们是“共用一个原子”

这不是 `[J_i]` 能表达的。

---

## 4. V2 的核心扩展

### 4.1 新 attachment token

v2 引入 attachment token：

- `[ATT_BOND]`
- `[ATT_SPIRO]`

语义：

| Token | 含义 |
|---|---|
| `[ATT_BOND]` | 当前 motif 的输入端口与父节点 active port 通过单键相连 |
| `[ATT_SPIRO]` | 当前 motif 的输入端口与父节点 active port 不是加一根单键，而是 **合并为同一个原子** |

其中：

- `[ATT_BOND]` 是 v1 默认连接语义的显式化
- `[ATT_SPIRO]` 是 v2 新增语义

### 4.2 新的 child-group 语法

v1 child-group 可理解为：

```text
MOTIF [J_input] [J_output_1] [J_output_2] ...
```

v2 扩展为：

```text
MOTIF [ATT_BOND|ATT_SPIRO] [J_input] [J_output_1] [J_output_2] ...
```

若省略 attachment token，则默认：

- `[ATT_BOND]`

### 4.3 `[ATT_SPIRO]` 的具体语义

对非 root motif：

1. 从 port stack 弹出父节点 active port；
2. 读取当前 motif 的第一个 `[J_i]`，得到 child input port；
3. 如果 attachment 类型是 `[ATT_SPIRO]`：
   - 不在父子之间加单键；
   - 而是将 child input atom 与 parent active atom **合并**；
   - child motif 的其余邻接边改挂到 parent active atom 上；
   - child input atom 从图中删除。

这样两个 motif 就可以通过“共享一个原子”连接。

---

## 5. Decoder 原型范围

当前 v2 原型的重点是：

- **decode 支持 `[ATT_SPIRO]`**

当前已实现：

- `decode_stamp_to_mol()` 可解析可选 attachment token
- 支持：
  - `[ATT_BOND]`
  - `[ATT_SPIRO]`
- 非 root motif 可显式声明 attachment 类型
- root motif 上若出现 attachment token，会在 non-strict 模式下忽略并记入 report
- `scripts/stamp_v2/selfcheck.py` 提供最小 smoke test：
  - 显式 `[ATT_BOND]` 与隐式 bond 行为一致
  - `[ATT_SPIRO]` 会生成与 `[ATT_BOND]` 不同的分子图
- `stamp_v2/vocab/seed_inventory.py` 提供 overlap-aware seed vocab 原型：
  - fused ring seed
  - spiro overlap detection
  - `bond/spiro` schema inventory

不保证：

- `encode_mol()` 已能自动识别并产生 `[ATT_SPIRO]`

原因：

- encode 侧需要新的 fragment decomposition / edge typing；
- 这是比 decoder 更大的设计面。

因此当前 v2 状态应理解为：

- **grammar + decoder prototype**

而不是完整可训练系统。

### 5.1 最小使用方式

运行自检：

```bash
PYTHONPATH=. python scripts/stamp_v2/selfcheck.py
```

解码单条 v2 序列：

```bash
PYTHONPATH=. python scripts/stamp_v2/stamp_to_smiles_v2.py \
  "C1CCC1 [J_0] C1CC1 [ATT_SPIRO] [J_0]"
```

该脚本会验证：

1. `CC [J_1] O [J_0]` 与 `CC [J_1] O [ATT_BOND] [J_0]` 都解码成 `CCO`
2. `C1CCC1 [J_0] C1CC1 [ATT_SPIRO] [J_0]` 与 bond 版本解码成不同图结构
3. spiro 版本的原子数比 bond 版本少 1，符合“共用一个原子”的语义

---

## 6. 需要改的结构

### 6.1 Edge 类型

v1 的 `FragmentEdge` 只有：

- `left / right`
- `left_local_atom / right_local_atom`
- stereo label

v2 需要增加：

- `attachment_kind`

例如：

```text
attachment_kind in {"bond", "spiro"}
```

### 6.2 Decoder 连接逻辑

v1：

- `_try_add_bond_with_fallback()`

v2 需要新增：

- `_try_spiro_merge_with_fallback()`

逻辑是：

1. 检查 parent atom 和 child input atom 是否可兼容；
2. 将 child input atom 的邻居重新连接到 parent atom；
3. 删除 child input atom；
4. 修正 fragment 内其余 port 的 atom index。

### 6.3 Grammar 解析

v2 需要新增：

- `_is_attachment()`
- `_parse_attachment()`

并在 `decode_stamp_to_mol()` 的 motif-group 解析中支持：

- motif 后首先读取可选 `[ATT_*]`

---

## 7. 为什么先只做 `[ATT_SPIRO]`

你提出的需求是：

> 引入 attachment token，`[ATT_SPIRO]` 表示某两个 motif 共用一个原子

这是最明确、最典型、也最能解释当前 OOV motif fallback 受阻的问题。

相比之下：

- fused ring
- bridged bicyclic
- shared bond / shared edge

这些结构还需要更复杂的 attachment 类型，例如：

- `[ATT_FUSED]`
- `[ATT_BRIDGE]`

所以 v2 第一阶段先只做：

- `[ATT_SPIRO]`

作为最小可验证扩展。

---

## 8. 当前预期收益

如果 v2 成功，最直接的收益是：

1. 一部分当前在 v1 中只能整体保留的大 ring motif，
   可能在 v2 中可以拆成更小的可训练子结构；
2. GenMol 原始 benchmark 中某些 OOV motif 可以变成
   “in-vocab 小 motif + attachment token”；
3. 这会降低 `built:true but OOV-heavy` 的失败比例。

---

## 9. 当前风险

### 9.1 不保证更好

这条路线是合理的，但 **不保证一定优于 v1**。

可能的风险包括：

1. grammar 复杂化，模型学习难度上升；
2. spiro merge 的 decoder fallback 过于宽松，会引入新的歧义；
3. tokenizer mining 若不配套 redesign，encoder 仍未必会自然用到新 token。

### 9.2 不能污染 v1 结果

因为是否有效尚未验证，v2 必须保持：

- 独立目录
- 独立脚本
- 独立结果

不能影响：

- `stamp/`
- `docs/ao_gpt_improvement.md`
- 现有 `zinc250k` 和 `GenMol` fair comparison 结论

---

## 10. 当前建议的开发顺序

### 阶段 A

完成独立 `stamp_v2` decoder 原型：

1. 新 attachment token 常量
2. 新 parser
3. `_try_spiro_merge_with_fallback()`
4. 最小 decode self-check

### 阶段 B

补充 encode 侧最小原型：

1. 允许 edge typing
2. 手工构造几个 spiro 测试分子
3. 验证 encode/decode roundtrip

### 阶段 C

再考虑是否值得做：

1. v2 tokenizer mining
2. v2 AO-GPT training
3. v2 benchmark 重跑

---

## 11. 当前结论

一句话总结：

- 你提出的 `[ATT_SPIRO]` 是对的；
- 这已经不是 benchmark 调参，而是 **STAMP v2 grammar 扩展**；
- 当前应在 `stamp_v2/` 独立推进，不能污染 v1 结论。

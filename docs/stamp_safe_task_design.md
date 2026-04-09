# STAMP 任务重设计（参考 SAFE）

## 1. 背景与问题

当前 ExpC/ExpD 原始设计存在两个核心问题：

1. **ExpC（scaffold completion）缺少连接位点约束**
- 旧流程输入是 `scaffold_smiles`（无 attachment 信息），模型只能“猜”从哪里长侧链。
- 对 STAMP 这种 motif/tree 表示来说，缺少位点约束会降低条件生成可控性。

2. **ExpD（linker design）任务定义不贴合 STAMP**
- 旧流程是 `fragment_a.fragment_b -> smiles`，即直接预测整分子。
- 这把“链接子生成”和“整体拼接”混在一个学习目标里，任务过重。
- 且 STAMP 单次编码不支持 disconnected 分子作为单一输入（已通过拆分编码修复过一次，但任务层面仍不理想）。

## 2. SAFE 的可迁移设计思想

参考 SAFE 的 scaffold/linker 设计（`safe/sample.py`）：

- 使用**attachment-aware 条件输入**（保留 `*` 位点）
- 先做 **completion**（生成受约束片段）
- 再做 **sanitize + 约束过滤**
- linker 任务中，优先学习“**linker 本身**”，最终分子通过后处理拼接得到

这套思想对 STAMP 同样适用：

- STAMP 可以编码单个带 dummy 的分子/片段（如 `[1*]...`）
- 因此可把“连接位点”显式暴露给模型，而不是隐式猜测

## 3. 新的 STAMP 任务定义

### 3.1 ExpC：Scaffold Decoration（STAMP 专用）

数据列从旧版：
- `smiles`
- `scaffold_smiles`

扩展为：
- `smiles`
- `scaffold_smiles`
- `scaffold_prompt`（在 scaffold 上加 dummy attachment 的条件 scaffold）

训练目标：
- **输入**：`scaffold_prompt`
- **输出**：`smiles`

说明：
- `scaffold_prompt` 通过原分子与 scaffold 对齐后，在 scaffold 对外连接边界处补 `*`。

### 3.2 ExpD：Linker Generation（STAMP 专用）

数据列从旧版：
- `smiles`
- `fragment_a`
- `fragment_b`
- `linker`

扩展为：
- `smiles`
- `fragment_a`
- `fragment_b`
- `linker`
- `fragment_a_ap`
- `fragment_b_ap`
- `linker_ap`

其中 `_ap` 列使用标准化 attachment 标注：
- `fragment_a_ap` 使用 `[1*]`
- `fragment_b_ap` 使用 `[2*]`
- `linker_ap` 同时含 `[1*]` 与 `[2*]`

训练目标：
- **输入**：`{fragment_a_ap}.{fragment_b_ap}`
- **输出**：`linker_ap`

评估流程：
1. 先解码 `pred_repr -> pred_linker`
2. 用 `(fragment_a_ap, fragment_b_ap, pred_linker)` 进行组装，得到 `pred_smiles`
3. 再计算 `validity / fragment_preservation / connection_success`

这与 SAFE 的“先生成 linker，再组装”的任务结构一致。

## 4. 已实现代码位置

### 4.1 任务核心实现
- `stamp/benchmark/tasks.py`

关键新增/修改：
- `_add_scaffold_attachment_dummies`：构造 `scaffold_prompt`
- `_normalize_linker_triplet`：生成 `_ap` 标注列
- `_assemble_labeled_linker_fragments`：由 `fragment_a_ap + fragment_b_ap + linker_ap` 组装最终分子
- `build_scaffold_completion_dataset`：输出新增 `scaffold_prompt`
- `build_linker_design_dataset`：输出新增 `fragment_a_ap / fragment_b_ap / linker_ap`
- `evaluate_linker_design`：自动根据 `meta.target_key` 切换到 linker 模式（先预测 linker，再组装）

### 4.2 运行脚本修改
- `scripts/qsub_runbook_all_tasks.pbs`

已切换为：
- ExpC 训练：`--prompt-key scaffold_prompt`
- ExpD 训练：
  - `--prompt-key fragment_a_ap`
  - `--prompt-template '{fragment_a_ap}.{fragment_b_ap}'`
  - `--target-key linker_ap`

## 5. 复跑建议（只重跑 C/D）

建议不要重跑 A/B，直接重跑 C/D：

1. `build-scaffold`（生成含 `scaffold_prompt` 的新 CSV）
2. `train-completion`（ExpC，`prompt_key=scaffold_prompt`）
3. `eval-scaffold`
4. `build-linker`（生成含 `_ap` 列的新 CSV）
5. `train-completion`（ExpD，`target_key=linker_ap`）
6. `eval-linker`（自动进入 linker 模式并组装评估）

## 6. 验收清单

建议你校验以下点：

1. **数据列是否正确**
- ExpC CSV 头包含 `scaffold_prompt`
- ExpD CSV 头包含 `fragment_a_ap`, `fragment_b_ap`, `linker_ap`

2. **ExpD 训练目标是否正确**
- 模型目录 `meta.json` 中 `target_key == linker_ap`

3. **ExpD 评估模式是否正确**
- 运行日志出现 `mode=linker`
- 预测表中包含 `pred_linker` 列

4. **组装函数可逆性抽查**
- 用真值 `(fragment_a_ap, fragment_b_ap, linker_ap)` 组装应能恢复原 `smiles`（随机抽样检查）

## 7. 备注

- 本次 smoke test 仅验证“流程可跑通”，并非最终质量结论。
- 最终指标仍受模型规模、训练轮数、采样参数（temperature/top-k）影响。
- 对 ExpC/ExpD 的采样建议使用更保守参数（如 `temperature=0.7`, `top_k=1`）提升 validity 稳定性。


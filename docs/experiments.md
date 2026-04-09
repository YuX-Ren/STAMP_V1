# STAMP Lightweight Benchmark Experiments

这个文档对应 `docs/experiments_checklist.md`，并给出可直接运行的代码入口。

## Code Entry

主入口脚本：

- `python scripts/experiments_cli.py <subcommand> ...`
- 实际文件位置：`scripts/benchmark/experiments_cli.py`（`scripts/experiments_cli.py` 为兼容包装器）

核心实现模块：

- `stamp/experiments.py`（兼容入口）
- `stamp/benchmark/*.py`（实际拆分实现）

## Subcommands (Checklist Mapping)

### 0 / 2 Data

- `prepare-data`
- 功能：清洗、去重、切分 train/valid/test，并生成 low-resource 子集。

示例：

```bash
python scripts/experiments_cli.py prepare-data \
  --input data/zinc250k.csv \
  --smiles-column smiles \
  --output-dir results/benchmark \
  --train-size 100000 --valid-size 10000 --test-size 10000 \
  --low-resource-sizes 10000,50000,100000
```

输出：

- `results/benchmark/splits/train.smiles`
- `results/benchmark/splits/valid.smiles`
- `results/benchmark/splits/test.smiles`
- `results/benchmark/splits/low_resource/train_*.smiles`

### 3 Experiment A: Representation Statistics

- `repr-stats`
- 指标：Avg/Median char length, Avg/Median/P95/Max token length, vocab size, encode/decode valid rate, special-token ratio, STAMP branch depth / attachment count, ratio vs SMILES。

示例：

```bash
python scripts/experiments_cli.py repr-stats \
  --input results/benchmark/splits/train.smiles \
  --output-dir results/benchmark/expA_repr_stats \
  --representations smiles,stamp,selfies,safe,t-smiles,group-selfies,fragsmiles \
  --max-mols 20000
```

输出：

- `representation_stats.json`
- `representation_stats_table.csv`

### 4 Experiment B: De Novo Generation

1) 训练统一 decoder-only Transformer：`train-lm`

```bash
python scripts/experiments_cli.py train-lm \
  --train results/benchmark/splits/train.smiles \
  --valid results/benchmark/splits/valid.smiles \
  --representation stamp \
  --output-dir results/benchmark/expB_stamp_lm \
  --epochs 10 --batch-size 64 --max-seq-len 256 \
  --device cuda
```

2) 采样与评估：`sample-eval`

```bash
python scripts/experiments_cli.py sample-eval \
  --model-dir results/benchmark/expB_stamp_lm \
  --train results/benchmark/splits/train.smiles \
  --reference results/benchmark/splits/test.smiles \
  --output-dir results/benchmark/expB_stamp_denovo \
  --n-samples 10000 \
  --decode-batch-size 1024 \
  --device cuda
```

指标：Validity / Uniqueness / Novelty / Internal diversity / MW-logP-QED-SA 分布距离。
注意：当前 `validity` 使用“单连通分子”判定，多片段（含 `.`）将视为 invalid。

补充：若要对齐你引用的 MOSES GPT-20M 设置，可在训练命令加：

```bash
--preset moses_gpt20m
```

该 preset 等价于：
- `d_model=768`
- `n_heads=8`
- `n_layers=6`
- `epochs=10`
- `lr=5e-4`

训练时会打印 `device` 和 `n_model_params`，并写入 `train_meta.json`。

### 5 Experiment C: Scaffold Decoration

1) 构建 scaffold 数据：`build-scaffold`

```bash
python scripts/experiments_cli.py build-scaffold \
  --input results/benchmark/splits/train.smiles \
  --output results/benchmark/expC_scaffold/train_scaffold.csv
```

2) 训练 completion 模型：`train-completion`

```bash
python scripts/experiments_cli.py train-completion \
  --csv results/benchmark/expC_scaffold/train_scaffold.csv \
  --representation stamp \
  --prompt-key scaffold_smiles \
  --target-key smiles \
  --output-dir results/benchmark/expC_scaffold/stamp_model \
  --epochs 10 \
  --device cuda
```

3) 评估：`eval-scaffold`

```bash
python scripts/experiments_cli.py eval-scaffold \
  --model-dir results/benchmark/expC_scaffold/stamp_model \
  --csv results/benchmark/expC_scaffold/train_scaffold.csv \
  --output-dir results/benchmark/expC_scaffold/stamp_eval \
  --max-eval 1000 \
  --device cuda
```

指标：Validity / Scaffold preservation / Completion success / Diversity。

### 6 Experiment D: Linker Design

1) 构建 linker 数据：`build-linker`

```bash
python scripts/experiments_cli.py build-linker \
  --input results/benchmark/splits/train.smiles \
  --output results/benchmark/expD_linker/train_linker.csv
```

2) 训练 completion 模型（prompt=`fragment_a.fragment_b`，target=`smiles`）：

```bash
python scripts/experiments_cli.py train-completion \
  --csv results/benchmark/expD_linker/train_linker.csv \
  --representation stamp \
  --prompt-key fragment_a \
  --prompt-template '{fragment_a}.{fragment_b}' \
  --target-key smiles \
  --output-dir results/benchmark/expD_linker/stamp_model \
  --epochs 10 \
  --device cuda
```

3) 评估：`eval-linker`

```bash
python scripts/experiments_cli.py eval-linker \
  --model-dir results/benchmark/expD_linker/stamp_model \
  --csv results/benchmark/expD_linker/train_linker.csv \
  --output-dir results/benchmark/expD_linker/stamp_eval \
  --max-eval 1000 \
  --device cuda
```

指标：Validity / Fragment preservation / Connection success / Diversity。

### 7 Experiment E: Low-Resource

- `run-low-resource`
- 会在 `10k/50k/100k` 子集上重复训练和 de novo 评估。

```bash
python scripts/experiments_cli.py run-low-resource \
  --split-dir results/benchmark/splits \
  --output-dir results/benchmark/expE_low_resource \
  --representations smiles,stamp,selfies \
  --sizes 10000,50000,100000 \
  --epochs 5 --n-samples 2000 \
  --decode-batch-size 1024 \
  --device cuda
```

输出：

- `low_resource_scaling.csv`
- `low_resource_summary.json`

### 8/9 Tables

- `make-tables`

```bash
python scripts/experiments_cli.py make-tables \
  --repr-stats-json results/benchmark/expA_repr_stats/representation_stats.json \
  --denovo-metrics-json \
    results/benchmark/expB_smiles_denovo/denovo_metrics.json \
    results/benchmark/expB_stamp_denovo/denovo_metrics.json \
  --output-md results/benchmark/summary_tables.md
```

### 11 Experiment F: Property Prediction

- `property-pred`
- 输入带标签 CSV，输出性质预测回归指标：`RMSE / MAE / R² / Pearson r`。

```bash
python scripts/experiments_cli.py property-pred \
  --input data/delaney-processed.csv \
  --smiles-column smiles \
  --target-column "measured log solubility in mols per litre" \
  --representations smiles,stamp,selfies,safe,group-selfies,fragsmiles \
  --test-size 0.2 \
  --output-dir results/benchmark/expF_property_prediction
```

输出：

- `property_prediction_metrics.json`
- `property_prediction_metrics_table.csv`

### 10 Plots

- `make-plots`
- 自动从一个 run root 扫描 `expA/B/C/D/E` 结果并生成 PNG 图。

```bash
python scripts/experiments_cli.py make-plots \
  --run-root results/benchmark
```

输出：

- `results/benchmark/plots/plots_summary.json`
- `results/benchmark/plots/expA/*.png`
- `results/benchmark/plots/expB/*.png`（若存在 de novo 结果）
- `results/benchmark/plots/expC/*.png`（若存在 scaffold 结果）
- `results/benchmark/plots/expD/*.png`（若存在 linker 结果）
- `results/benchmark/plots/expE/*.png`（若存在 low-resource 结果）
- `results/benchmark/plots/expF/*.png`（若存在 property prediction 结果）

## Notes

- 当前内置可直接运行的表示法：`smiles`, `stamp`, `safe`, `group-selfies`, `fragsmiles`。
- 训练/采样相关命令支持 `--device`；推荐显式设为 `--device cuda` 以确保全部跑在 GPU 上。
- `safe`、`group-selfies`、`fragsmiles` 使用官方实现（支持 vendored fallback）。
- `safe` 若缺少 pip 包，可直接将官方仓库放到 `third_party/safe`（并安装最小依赖 `datamol`、`loguru`）。
- `group-selfies` 若缺少 pip 包，可直接将官方仓库放到 `third_party/group-selfies`。
- 若环境里还未安装，可执行：`pip install git+https://github.com/f48r1/chemicalgof.git`。
- `selfies` 仍依赖外部包（需 `pip install selfies`）。
- `t-smiles` 目前仍是 placeholder。
- 训练已使用动态 padding（按 batch 最大长度补齐），并在日志/`train_meta.json` 输出 `train_effective_max_seq_len` 与 `valid_effective_max_seq_len`。
- 所有输出默认写到 `results/benchmark/...`，可通过参数覆盖。

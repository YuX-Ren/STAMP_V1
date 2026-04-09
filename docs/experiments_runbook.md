# STAMP 实验清单与执行步骤（可直接运行）

本文件给出一套从 0 到 1 的实验执行清单，覆盖 `A/B/C/D/E`，并附完整命令。  
默认使用数据集：`data/zinc250k.csv`

## 0. 实验清单

- `A` 表示统计（Representation Statistics）
- `B` De Novo Generation（训练 + 采样评估）
- `C` Scaffold Decoration（构建数据 + 训练 + 评估）
- `D` Linker Design（构建数据 + 训练 + 评估）
- `E` Low-Resource Scaling（10k/50k/100k）
- `F` 汇总表格（论文/报告 markdown 表）

## 1. 一次性设置（建议先执行）

```bash
cd /home/svu/yuxuan.ren/STAMP

# 可按需修改
export DATA_CSV="data/zinc250k.csv"
export SMILES_COL="smiles"
export RUN_ROOT="results/benchmark_$(date +%Y%m%d_%H%M%S)"
export REPRS="smiles stamp selfies group-selfies fragsmiles safe"
export ENCODE_WORKERS=32
export ENCODE_CHUNKSIZE=256
export NUM_DATA_WORKERS=8
export EVAL_WORKERS=16
export EVAL_CHUNKSIZE=512
export REPR_WORKERS=6

mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
```

## 2. Step A: 准备数据切分

```bash
python scripts/experiments_cli.py prepare-data \
  --input "$DATA_CSV" \
  --smiles-column "$SMILES_COL" \
  --output-dir "$RUN_ROOT" \
  --train-size 100000 \
  --valid-size 10000 \
  --test-size 10000 \
  --low-resource-sizes 10000,50000,100000
```

输出重点：
- `$RUN_ROOT/splits/train.smiles`
- `$RUN_ROOT/splits/valid.smiles`
- `$RUN_ROOT/splits/test.smiles`
- `$RUN_ROOT/splits/low_resource/train_10000.smiles`
- `$RUN_ROOT/splits/low_resource/train_50000.smiles`
- `$RUN_ROOT/splits/low_resource/train_100000.smiles`

## 3. Step B: 实验 A（表示统计）

```bash
python scripts/experiments_cli.py repr-stats \
  --input "$RUN_ROOT/splits/train.smiles" \
  --output-dir "$RUN_ROOT/expA_repr_stats" \
  --representations "$(echo $REPRS | tr ' ' ',')" \
  --workers "$REPR_WORKERS" \
  --max-mols 20000
```

输出重点：
- `$RUN_ROOT/expA_repr_stats/representation_stats.json`
- `$RUN_ROOT/expA_repr_stats/representation_stats_table.csv`

## 4. Step C: 实验 B（De Novo）

对每种表示法跑：`train-lm -> sample-eval`

```bash
for r in $REPRS; do
  python scripts/experiments_cli.py train-lm \
    --train "$RUN_ROOT/splits/train.smiles" \
    --valid "$RUN_ROOT/splits/valid.smiles" \
    --representation "$r" \
    --output-dir "$RUN_ROOT/expB_${r}_lm" \
    --encode-workers "$ENCODE_WORKERS" \
    --encode-chunksize "$ENCODE_CHUNKSIZE" \
    --num-data-workers "$NUM_DATA_WORKERS" \
    --prefetch-factor 4 \
    --amp auto \
    --torch-compile \
    --epochs 10 \
    --batch-size 256 \
    --max-seq-len 256 \
    --device cuda

  python scripts/experiments_cli.py sample-eval \
    --model-dir "$RUN_ROOT/expB_${r}_lm" \
    --train "$RUN_ROOT/splits/train.smiles" \
    --reference "$RUN_ROOT/splits/test.smiles" \
    --output-dir "$RUN_ROOT/expB_${r}_denovo" \
    --n-samples 10000 \
    --decode-batch-size 1024 \
    --eval-workers 16 \
    --eval-chunksize 512 \
    --reference-max-samples 50000 \
    --assume-train-canonical \
    --device cuda
done
```

每个表示法输出重点：
- `$RUN_ROOT/expB_<repr>_lm/train_meta.json`
- `$RUN_ROOT/expB_<repr>_denovo/denovo_metrics.json`
- `$RUN_ROOT/expB_<repr>_denovo/denovo_samples.tsv`

### 4.1 论文同款（MOSES GPT-20M）设置

对应你给的设定：`6 layers / 8 heads / hidden=768 / epochs=10 / lr=5e-4`。  
CLI 里可直接用 `--preset moses_gpt20m` 一键应用。

```bash
# 全表示法统一跑 moses_gpt20m
for r in $REPRS; do
  echo "=== train+sample moses_gpt20m: $r ==="
  python scripts/experiments_cli.py train-lm \
    --train "$RUN_ROOT/splits/train.smiles" \
    --valid "$RUN_ROOT/splits/valid.smiles" \
    --representation "$r" \
    --output-dir "$RUN_ROOT/expB_${r}_lm_moses20m" \
    --encode-workers "$ENCODE_WORKERS" \
    --encode-chunksize "$ENCODE_CHUNKSIZE" \
    --num-data-workers "$NUM_DATA_WORKERS" \
    --prefetch-factor 4 \
    --amp auto \
    --torch-compile \
    --preset moses_gpt20m \
    --batch-size 256 \
    --max-seq-len 256 \
    --device cuda || { echo "[WARN] train failed for $r"; continue; }

  python scripts/experiments_cli.py sample-eval \
    --model-dir "$RUN_ROOT/expB_${r}_lm_moses20m" \
    --train "$RUN_ROOT/splits/train.smiles" \
    --reference "$RUN_ROOT/splits/test.smiles" \
    --output-dir "$RUN_ROOT/expB_${r}_denovo_moses20m" \
    --n-samples 10000 \
    --decode-batch-size 1024 \
    --eval-workers 16 \
    --eval-chunksize 512 \
    --reference-max-samples 50000 \
    --assume-train-canonical \
    --device cuda || { echo "[WARN] sample-eval failed for $r"; continue; }
done
```

说明：
- 训练日志会打印 `device` 和 `n_model_params`，可直接核对是否在 CUDA 以及参数规模。
- `selfies` 需要先安装依赖：`pip install selfies`。
- 当前仓库中的 `safe`、`group-selfies`、`fragsmiles` 使用官方实现（支持 vendored fallback）。
- 若未安装 `chemicalgof`，请先执行：`pip install git+https://github.com/f48r1/chemicalgof.git`。
- 若未安装 `group_selfies`，请克隆官方仓库到 `third_party/group-selfies`（当前仓库已兼容该路径）。
- 若未安装 `safe-mol`，请克隆官方仓库到 `third_party/safe` 并安装最小依赖：`pip install datamol loguru`。

## 5. Step D: 实验 C（Scaffold Decoration）

先建任务数据，再训练与评估。默认先跑 STAMP。

```bash
python scripts/experiments_cli.py build-scaffold \
  --input "$RUN_ROOT/splits/train.smiles" \
  --output "$RUN_ROOT/expC_scaffold/train_scaffold.csv" \
  --workers "$ENCODE_WORKERS" \
  --chunksize "$ENCODE_CHUNKSIZE"

python scripts/experiments_cli.py train-completion \
  --csv "$RUN_ROOT/expC_scaffold/train_scaffold.csv" \
  --representation stamp \
  --prompt-key scaffold_smiles \
  --target-key smiles \
  --output-dir "$RUN_ROOT/expC_scaffold/stamp_model" \
  --encode-workers "$ENCODE_WORKERS" \
  --encode-chunksize "$ENCODE_CHUNKSIZE" \
  --num-data-workers "$NUM_DATA_WORKERS" \
  --prefetch-factor 4 \
  --amp auto \
  --torch-compile \
  --epochs 10 \
  --device cuda

python scripts/experiments_cli.py eval-scaffold \
  --model-dir "$RUN_ROOT/expC_scaffold/stamp_model" \
  --csv "$RUN_ROOT/expC_scaffold/train_scaffold.csv" \
  --output-dir "$RUN_ROOT/expC_scaffold/stamp_eval" \
  --max-eval 1000 \
  --eval-workers "$EVAL_WORKERS" \
  --eval-chunksize "$EVAL_CHUNKSIZE" \
  --device cuda
```

输出重点：
- `$RUN_ROOT/expC_scaffold/train_scaffold.csv`
- `$RUN_ROOT/expC_scaffold/stamp_eval/scaffold_completion_metrics.json`

## 6. Step E: 实验 D（Linker Design）

```bash
python scripts/experiments_cli.py build-linker \
  --input "$RUN_ROOT/splits/train.smiles" \
  --output "$RUN_ROOT/expD_linker/train_linker.csv" \
  --workers "$ENCODE_WORKERS" \
  --chunksize "$ENCODE_CHUNKSIZE"

python scripts/experiments_cli.py train-completion \
  --csv "$RUN_ROOT/expD_linker/train_linker.csv" \
  --representation stamp \
  --prompt-key fragment_a \
  --prompt-template '{fragment_a}.{fragment_b}' \
  --target-key smiles \
  --output-dir "$RUN_ROOT/expD_linker/stamp_model" \
  --encode-workers "$ENCODE_WORKERS" \
  --encode-chunksize "$ENCODE_CHUNKSIZE" \
  --num-data-workers "$NUM_DATA_WORKERS" \
  --prefetch-factor 4 \
  --amp auto \
  --torch-compile \
  --epochs 10 \
  --device cuda

python scripts/experiments_cli.py eval-linker \
  --model-dir "$RUN_ROOT/expD_linker/stamp_model" \
  --csv "$RUN_ROOT/expD_linker/train_linker.csv" \
  --output-dir "$RUN_ROOT/expD_linker/stamp_eval" \
  --max-eval 1000 \
  --eval-workers "$EVAL_WORKERS" \
  --eval-chunksize "$EVAL_CHUNKSIZE" \
  --device cuda
```

输出重点：
- `$RUN_ROOT/expD_linker/train_linker.csv`
- `$RUN_ROOT/expD_linker/stamp_eval/linker_design_metrics.json`

## 7. Step F: 实验 E（Low-Resource）

```bash
python scripts/experiments_cli.py run-low-resource \
  --split-dir "$RUN_ROOT/splits" \
  --output-dir "$RUN_ROOT/expE_low_resource" \
  --representations "$(echo $REPRS | tr ' ' ',')" \
  --sizes 10000,50000,100000 \
  --epochs 5 \
  --batch-size 64 \
  --max-seq-len 256 \
  --n-samples 2000 \
  --decode-batch-size 1024 \
  --encode-workers "$ENCODE_WORKERS" \
  --encode-chunksize "$ENCODE_CHUNKSIZE" \
  --eval-workers "$EVAL_WORKERS" \
  --eval-chunksize "$EVAL_CHUNKSIZE" \
  --num-data-workers "$NUM_DATA_WORKERS" \
  --prefetch-factor 4 \
  --amp auto \
  --torch-compile \
  --device cuda
```

输出重点：
- `$RUN_ROOT/expE_low_resource/low_resource_scaling.csv`
- `$RUN_ROOT/expE_low_resource/low_resource_summary.json`

## 8. Step G: 汇总报告表

```bash
python scripts/experiments_cli.py make-tables \
  --repr-stats-json "$RUN_ROOT/expA_repr_stats/representation_stats.json" \
  --denovo-metrics-json \
    "$RUN_ROOT/expB_smiles_denovo/denovo_metrics.json" \
    "$RUN_ROOT/expB_stamp_denovo/denovo_metrics.json" \
    "$RUN_ROOT/expB_selfies_denovo/denovo_metrics.json" \
    "$RUN_ROOT/expB_group-selfies_denovo/denovo_metrics.json" \
    "$RUN_ROOT/expB_fragsmiles_denovo/denovo_metrics.json" \
  --output-md "$RUN_ROOT/summary_tables.md"
```

输出重点：
- `$RUN_ROOT/summary_tables.md`

## 9. 快速验收（建议）

```bash
ls -lh "$RUN_ROOT"/expA_repr_stats
ls -lh "$RUN_ROOT"/expB_*_denovo
ls -lh "$RUN_ROOT"/expC_scaffold/stamp_eval
ls -lh "$RUN_ROOT"/expD_linker/stamp_eval
ls -lh "$RUN_ROOT"/expE_low_resource
```

## 10. 生成图表（推荐）

```bash
python scripts/experiments_cli.py make-plots \
  --run-root "$RUN_ROOT"
```

输出重点：
- `$RUN_ROOT/plots/plots_summary.json`
- `$RUN_ROOT/plots/expA/*.png`
- `$RUN_ROOT/plots/expB/*.png`（如果你跑了 de novo）
- `$RUN_ROOT/plots/expC/*.png`（如果你跑了 scaffold）
- `$RUN_ROOT/plots/expD/*.png`（如果你跑了 linker）
- `$RUN_ROOT/plots/expE/*.png`（如果你跑了 low-resource）

## 11. 备注

- 当前项目中可直接完整训练评估的表示法：`smiles, stamp, selfies, group-selfies, fragsmiles`。
- 模型训练与采样命令默认按 `--device cuda` 执行；若无 GPU 会直接报错，避免误落到 CPU 慢速运行。
- 训练使用动态 padding（按 batch 最大长度补齐）；日志会打印 `train_effective_max_seq_len` / `valid_effective_max_seq_len` 供你检查。
- `group-selfies` 与 `fragsmiles` 已可直接运行。
- `selfies` 需要先安装依赖：`pip install selfies`。
- `safe` 已切换为官方 SAFE codec（不再兼容历史占位编码格式）。
- `t-smiles` 目前仍是 placeholder；若加入 `REPRS`，相关训练/采样会报未实现。

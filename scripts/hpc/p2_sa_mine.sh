#!/bin/bash
# Offline mine: motif frequencies in "good" (QED>0.7 & SA<3) vs "bad" (SA>4) pools,
# compute per-motif log-ratio penalty table for inference.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
PARQUET=/scratch/yuxuan.ren/data/dataset_stamp.parquet
OUT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009/phase2_sa_mine

source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

python - <<PY
import sys, json, time
sys.path.insert(0, '.')
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache
ckpt_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009')
_, vocab = _load_scan_cache(ckpt_dir)
V = len(vocab.itos)
print(f'[mine] vocab_size={V}', flush=True)

# Thresholds
QED_GOOD = 0.7; SA_GOOD = 3.0
QED_BAD  = 0.5; SA_BAD  = 4.5  # OR condition for bad

pf = pq.ParquetFile('${PARQUET}')
print(f'[mine] parquet row_groups={pf.num_row_groups}', flush=True)

good_doc_freq = np.zeros(V, dtype=np.int64)  # # molecules in good pool that contain motif
bad_doc_freq  = np.zeros(V, dtype=np.int64)
good_total = 0; bad_total = 0
total_seen = 0

t0 = time.time()
MAX_RG = 8   # about 2M rows
for rg in range(min(MAX_RG, pf.num_row_groups)):
    tab = pf.read_row_group(rg, columns=['stamp_smiles', 'qed', 'sa_score', 'split_random', 'all_pass'])
    stamp = tab['stamp_smiles'].to_pylist()
    qed = tab['qed'].to_pylist()
    sa  = tab['sa_score'].to_pylist()
    split = tab['split_random'].to_pylist()
    apass = tab['all_pass'].to_pylist()
    for i in range(len(stamp)):
        if split[i] != 'train' or not apass[i] or stamp[i] is None: continue
        q, s = float(qed[i]), float(sa[i])
        is_good = q > QED_GOOD and s < SA_GOOD
        is_bad  = q < QED_BAD or s > SA_BAD
        if not (is_good or is_bad): continue
        toks = stamp[i].split()
        # Unique motif ids in this sample
        ids = set()
        for t in toks:
            tid = vocab.stoi.get(t, vocab.unk_id)
            ids.add(tid)
        if is_good:
            for tid in ids: good_doc_freq[tid] += 1
            good_total += 1
        else:
            for tid in ids: bad_doc_freq[tid] += 1
            bad_total += 1
        total_seen += 1
    print(f'[mine]   rg {rg+1}/{MAX_RG}: total_seen={total_seen} good={good_total} bad={bad_total}  '
          f'elapsed={time.time()-t0:.0f}s', flush=True)

print(f'\n[mine] Done: total_seen={total_seen} good={good_total} bad={bad_total}', flush=True)

# Compute log-ratio penalty
eps = 10.0  # Laplace smoothing on doc count
p_good = (good_doc_freq + eps) / (good_total + eps * V)
p_bad  = (bad_doc_freq  + eps) / (bad_total  + eps * V)
log_ratio = np.log(p_bad) - np.log(p_good)   # positive = bad signal

# Clip extremes to prevent exploding penalties
log_ratio = np.clip(log_ratio, -3.0, 3.0)

# Save
np.save(Path('${OUT_DIR}') / 'log_ratio_penalty.npy', log_ratio)
# Also save human-readable top-50 bad and top-50 good motifs
top_bad = np.argsort(log_ratio)[::-1][:50]
top_good = np.argsort(log_ratio)[:50]
sample = []
print('\n[mine] Top-20 SA-bad motifs (penalty > 0):', flush=True)
for idx in top_bad[:20]:
    tok = vocab.itos[idx]
    print(f'  id={int(idx)} tok={tok!r} log_ratio={log_ratio[idx]:+.3f} '
          f'p_bad={p_bad[idx]:.4f} p_good={p_good[idx]:.4f}', flush=True)
print('\n[mine] Top-20 GOOD motifs (penalty < 0, prefer):', flush=True)
for idx in top_good[:20]:
    tok = vocab.itos[idx]
    print(f'  id={int(idx)} tok={tok!r} log_ratio={log_ratio[idx]:+.3f} '
          f'p_bad={p_bad[idx]:.4f} p_good={p_good[idx]:.4f}', flush=True)

# Stats
n_pos = int((log_ratio > 0.5).sum())
n_neg = int((log_ratio < -0.5).sum())
print(f'\n[mine] penalty distribution: {n_pos} motifs with ratio>+0.5 (bad), '
      f'{n_neg} with ratio<-0.5 (good), total V={V}', flush=True)
print(f'[mine] saved log_ratio to {Path("${OUT_DIR}") / "log_ratio_penalty.npy"}', flush=True)
PY
echo "[done]"

#!/bin/bash
# Bottleneck diagnostic for AO-GPT training. Run this inside a PBS interactive
# session (or as a batch job) on a GPU node.
set -u

REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

echo "[remote] host=$(hostname)"
echo "[remote] pwd=$(pwd)"
echo "[remote] date=$(date)"
nvidia-smi -L || true

source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

# nvidia-smi sampler in the background (GPU util / SM / memory)
nvidia-smi dmon -s umc -d 1 -c 180 > /tmp/diag_nvsmi.log 2>&1 &
DMON_PID=$!
trap "kill $DMON_PID 2>/dev/null || true" EXIT

python - <<'PY'
import sys, time
sys.path.insert(0, '.')
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _compute_ao_gpt_loss,
    build_dynamic_lm_collate_fn,
)
from scripts.benchmark.train_ao_gpt_parquet import (
    _ParquetTokenPairDataset, _load_scan_cache,
)

dev = torch.device('cuda:0')
print(f'[diag] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}', flush=True)

scan_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8')
scan_summary, vocab = _load_scan_cache(scan_dir)
print(f'[diag] vocab={len(vocab.itos)}  train_rows={scan_summary.train_rows:,}', flush=True)

cfg = LMConfig(
    model_type='ao_gpt', d_model=512, n_heads=8, n_layers=8, d_ff=2048,
    dropout=0.1, max_seq_len=64, batch_size=4096, lr=5e-4, weight_decay=0.01,
    epochs=1, seed=42, num_data_workers=8, prefetch_factor=6, amp='bf16',
    torch_compile=False, tf32=True,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def make_ds():
    ds = _ParquetTokenPairDataset(
        parquet_path='/scratch/yuxuan.ren/data/dataset_stamp.parquet',
        vocab=vocab, text_column='stamp_smiles', split_column='split_random',
        split_value='train', require_all_pass=True,
        max_seq_len=cfg.max_seq_len, max_rows=None, seed=42,
        rank=0, world_size=1,
    )
    ds.set_epoch(0)
    return ds

collate_fn = build_dynamic_lm_collate_fn(vocab.pad_id, pad_to_multiple_of=8)

def make_loader(num_workers, prefetch):
    ds = make_ds()
    kwargs = dict(
        batch_size=cfg.batch_size, collate_fn=collate_fn,
        pin_memory=True, num_workers=num_workers,
    )
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch
        kwargs['persistent_workers'] = True
    return DataLoader(ds, **kwargs)

model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
n_params = sum(p.numel() for p in model.parameters())
print(f'[diag] model params={n_params/1e6:.2f}M', flush=True)

def measure(name, fn, n_batches=20, warmup=3, nw=8, pf=6):
    loader = make_loader(nw, pf)
    it = iter(loader)
    for _ in range(warmup):
        fn(next(it))
    torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0
    last_seq = 0
    for _ in range(n_batches):
        batch = next(it)
        fn(batch)
        tokens += batch[0].numel()
        last_seq = batch[0].shape[1]
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'[{name:18s}] nw={nw} {n_batches}b in {dt:6.2f}s => '
          f'{n_batches/dt:5.2f} it/s  {tokens/dt/1e6:6.2f} Mtok/s  seq={last_seq}', flush=True)
    del loader, it

def f_data(batch):
    x, y = batch
    x = x.to(dev, non_blocking=True)
    y = y.to(dev, non_blocking=True)
    torch.cuda.synchronize()

def f_forward(batch):
    x, y = batch
    x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _ = model(x)
    torch.cuda.synchronize()

def f_forward_loss(batch):
    x, y = batch
    x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _ = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
    torch.cuda.synchronize()

def f_full(batch):
    x, y = batch
    x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()

print('---- phase breakdown (same loader, nw=8) ----', flush=True)
measure('data_only',     f_data)
measure('forward_only',  f_forward)
measure('forward_loss',  f_forward_loss)
measure('full_step',     f_full)

print('---- num_workers sweep (full_step) ----', flush=True)
for nw in (4, 8, 12, 16):
    measure(f'full_nw{nw}', f_full, n_batches=20, warmup=3, nw=nw, pf=6)
PY

echo '---- nvidia-smi dmon (middle/end samples) ----'
sed -n '5,40p;60,90p' /tmp/diag_nvsmi.log
echo "[remote] done=$(date)"

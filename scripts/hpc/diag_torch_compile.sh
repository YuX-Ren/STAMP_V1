#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
echo "[remote] host=$(hostname) date=$(date)"
nvidia-smi -L || true
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<'PY'
import sys, time
sys.path.insert(0, '.')
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _compute_ao_gpt_loss,
    build_dynamic_lm_collate_fn, _sample_ao_orders,
)
from scripts.benchmark.train_ao_gpt_parquet import (
    _ParquetTokenPairDataset, _load_scan_cache,
)

dev = torch.device('cuda:0')
print(f'[diag2] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}', flush=True)

scan_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8')
_, vocab = _load_scan_cache(scan_dir)
print(f'[diag2] vocab={len(vocab.itos)}', flush=True)

cfg = LMConfig(
    model_type='ao_gpt', d_model=512, n_heads=8, n_layers=8, d_ff=2048,
    dropout=0.1, max_seq_len=64, batch_size=4096, lr=5e-4, weight_decay=0.01,
    epochs=1, seed=42, num_data_workers=8, prefetch_factor=6, amp='bf16',
    torch_compile=False, tf32=True,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def make_loader():
    ds = _ParquetTokenPairDataset(
        parquet_path='/scratch/yuxuan.ren/data/dataset_stamp.parquet',
        vocab=vocab, text_column='stamp_smiles', split_column='split_random',
        split_value='train', require_all_pass=True,
        max_seq_len=cfg.max_seq_len, max_rows=None, seed=42, rank=0, world_size=1,
    )
    ds.set_epoch(0)
    return DataLoader(
        ds, batch_size=cfg.batch_size,
        collate_fn=build_dynamic_lm_collate_fn(vocab.pad_id, pad_to_multiple_of=8),
        pin_memory=True, num_workers=8, prefetch_factor=6, persistent_workers=True,
    )

# ---- Verify vectorized _sample_ao_orders is fast on GPU ----
print('---- sanity: _sample_ao_orders on GPU ----', flush=True)
labels = torch.randint(0, 1000, (cfg.batch_size, 40), device=dev)
labels[:, -4:] = -100  # tail pad
for _ in range(3): _sample_ao_orders(labels)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): _sample_ao_orders(labels)
torch.cuda.synchronize()
ms = (time.time() - t0) * 1000 / 50
print(f'  _sample_ao_orders: {ms:.3f} ms/call at bs={cfg.batch_size} seq=40', flush=True)

# ---- Measure full_step with and without torch.compile ----
def build_model():
    m = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg).to(dev)
    o = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    return m, o

def bench(name, model, opt, loader, n_batches=30, warmup=5):
    it = iter(loader)
    for _ in range(warmup):
        batch = next(it)
        x, y = batch
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0; last_seq = 0
    for _ in range(n_batches):
        batch = next(it)
        x, y = batch
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.numel(); last_seq = x.shape[1]
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'[{name:30s}] {n_batches}b in {dt:6.2f}s => {n_batches/dt:5.2f} it/s  '
          f'{tokens/dt/1e6:6.2f} Mtok/s  seq={last_seq}', flush=True)

loader = make_loader()

# Baseline (no compile)
print('---- baseline: no torch.compile ----', flush=True)
m0, o0 = build_model()
bench('baseline_nocompile', m0, o0, loader)

# torch.compile: mode='default'
print('---- torch.compile(default) ----', flush=True)
m1, o1 = build_model()
m1c = torch.compile(m1)  # default mode
# warmup benchmark will also drive the first-time compile (can be slow)
t_compile_start = time.time()
bench('compile_default_warmup', m1c, o1, loader, n_batches=5, warmup=0)
print(f'  (compile+warmup wallclock: {time.time()-t_compile_start:.1f}s)', flush=True)
bench('compile_default', m1c, o1, loader)

# torch.compile: mode='reduce-overhead' (cuda graphs)
print('---- torch.compile(reduce-overhead) ----', flush=True)
m2, o2 = build_model()
try:
    m2c = torch.compile(m2, mode='reduce-overhead')
    t_compile_start = time.time()
    bench('compile_reduce_warmup', m2c, o2, loader, n_batches=5, warmup=0)
    print(f'  (compile+warmup wallclock: {time.time()-t_compile_start:.1f}s)', flush=True)
    bench('compile_reduce', m2c, o2, loader)
except Exception as e:
    print(f'  reduce-overhead failed: {type(e).__name__}: {e}', flush=True)

# torch.compile: mode='max-autotune' (tries triton kernels, slowest compile)
print('---- torch.compile(max-autotune) ----', flush=True)
m3, o3 = build_model()
try:
    m3c = torch.compile(m3, mode='max-autotune')
    t_compile_start = time.time()
    bench('compile_max_warmup', m3c, o3, loader, n_batches=5, warmup=0)
    print(f'  (compile+warmup wallclock: {time.time()-t_compile_start:.1f}s)', flush=True)
    bench('compile_max_autotune', m3c, o3, loader)
except Exception as e:
    print(f'  max-autotune failed: {type(e).__name__}: {e}', flush=True)

print('[diag2] done', flush=True)
PY

echo "[remote] end=$(date)"

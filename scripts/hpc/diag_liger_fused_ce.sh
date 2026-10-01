#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
echo "[remote] host=$(hostname) date=$(date)"
nvidia-smi -L || true
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

# Install Liger Kernel (CPU-side install — no GPU compilation)
if ! python -c 'import liger_kernel' 2>/dev/null; then
    echo "[remote] installing liger-kernel..."
    pip install --no-cache-dir liger-kernel 2>&1 | tail -5
fi
python -c "import liger_kernel; print(f'[remote] liger_kernel={liger_kernel.__version__}')"

python - <<'PY'
import sys, time, math
sys.path.insert(0, '.')
import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _compute_ao_gpt_loss, _sample_ao_orders,
    build_dynamic_lm_collate_fn,
)
from scripts.benchmark.train_ao_gpt_parquet import (
    _ParquetTokenPairDataset, _load_scan_cache,
)

dev = torch.device('cuda:0')
print(f'[diag3] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}', flush=True)

_, vocab = _load_scan_cache(Path('/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8'))
V = len(vocab.itos); D = 512

cfg = LMConfig(
    model_type='ao_gpt', d_model=D, n_heads=8, n_layers=8, d_ff=2048,
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

# ---- Helpers ------------------------------------------------------------

def build_model():
    m = TinyDecoderLM(vocab_size=V, cfg=cfg).to(dev)
    # Drop LM head bias so Liger's fused linear-CE path works (Liger bug: when
    # bias is provided, grad_bias is None and its detach() NPEs). Common practice
    # in modern LMs anyway. Replace with bias-less Linear sharing trained weight.
    new_head = nn.Linear(cfg.d_model, V, bias=False).to(dev)
    new_head.weight = m.head.weight  # share parameter identity
    m.head = new_head
    o = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    return m, o

def forward_hidden(model, x, orders):
    """Run TinyDecoderLM but return post-final_layer hidden, not logits.
    Mirrors TinyDecoderLM.forward; must stay in sync if that changes."""
    from stamp.benchmark.lm import _ADALN_COND_DIM
    B, t = x.shape
    dev_ = x.device
    pos = torch.arange(0, t + 1, dtype=torch.long, device=dev_)
    if orders is None:
        orders = torch.arange(t, device=dev_).unsqueeze(0).expand(B, t)
    idx = model._shuffle(x, orders)
    tok_emb = model.tok_emb(idx)
    none_emb = model.none_emb(torch.zeros(B, 1, dtype=torch.long, device=dev_))
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)
    pe = model.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)
    pe_prefix = pe[:, :1]
    pe_post = model._shuffle(pe[:, 1:], orders)
    tpe = model.target_pos_emb(pos[:t]).unsqueeze(0).expand(B, -1, -1)
    tpe_shuffled = model._shuffle(tpe, orders)
    tpe_zeros = torch.zeros(B, 1, _ADALN_COND_DIM, device=dev_)
    target_cond = torch.cat([tpe_zeros, tpe_shuffled], dim=1)
    h = model.drop(tok_emb + torch.cat([pe_prefix, pe_post], dim=1))
    use_causal = not model.bidirectional
    for block in model.blocks:
        h = block(h, target_cond, is_causal=use_causal)
    h = model.final_layer(h, target_cond)
    return h[:, 1:, :]                       # (B, t, d_model)

def loss_liger(model, x, y):
    """Liger fused-linear CE path: no logits materialization."""
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    if not hasattr(loss_liger, 'fn'):
        loss_liger.fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    orders = _sample_ao_orders(y, random_ratio=0.9)
    hidden = forward_hidden(model, x, orders)                # (B, t, d)
    bsz = x.size(0)
    batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand_as(orders)
    targets = y[batch_idx, orders]                           # (B, t)
    return loss_liger.fn(
        model.head.weight,                                   # (V, d)
        hidden.reshape(-1, hidden.size(-1)),                 # (B*t, d)
        targets.reshape(-1),                                 # (B*t,)
        None,                                                # no bias
    )

def loss_standard(model, x, y):
    return _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)

def bench(name, loss_fn, model, opt, loader, n_batches=30, warmup=5):
    it = iter(loader)
    for _ in range(warmup):
        batch = next(it); x, y = batch
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = loss_fn(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens = 0; last_seq = 0
    for _ in range(n_batches):
        batch = next(it); x, y = batch
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = loss_fn(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.numel(); last_seq = x.shape[1]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f'[{name:28s}] {n_batches}b in {dt:6.2f}s => {n_batches/dt:5.2f} it/s  '
          f'{tokens/dt/1e6:6.2f} Mtok/s  seq={last_seq}  peak={peak_gb:.2f}GB', flush=True)

# ---- Equivalence check on a small batch ---------------------------------
print('---- equivalence check ----', flush=True)
m_eq, _ = build_model()
torch.manual_seed(0)
x_eq = torch.randint(0, V, (8, 20), device=dev)
y_eq = x_eq.clone()
# Force identity ordering so both paths shuffle identically
# (random_ratio=0 in _sample_ao_orders → identity gate)
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    torch.manual_seed(1); l_std = _compute_ao_gpt_loss(m_eq, x_eq, y_eq, cfg, random_ratio=0.0)
    torch.manual_seed(1)
    orders = _sample_ao_orders(y_eq, random_ratio=0.0)
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    lce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    h = forward_hidden(m_eq, x_eq, orders)
    t = y_eq[torch.arange(8, device=dev).unsqueeze(1).expand_as(orders), orders]
    l_lig = lce(m_eq.head.weight, h.reshape(-1, D), t.reshape(-1), None)
print(f'  standard loss = {l_std.item():.6f}', flush=True)
print(f'  liger    loss = {l_lig.item():.6f}', flush=True)
print(f'  |diff| = {abs(l_std.item() - l_lig.item()):.6f}', flush=True)

loader = make_loader()

# ---- Benchmarks --------------------------------------------------------
print('---- benchmarks (vectorized _sample_ao_orders, bs=4096) ----', flush=True)

print('[A] standard CE, no compile', flush=True)
mA, oA = build_model(); bench('standard_nocompile', loss_standard, mA, oA, loader)

print('[B] standard CE, compile(default)', flush=True)
mB, oB = build_model()
mBc = torch.compile(mB)
t0 = time.time(); bench('standard_compile_warmup', loss_standard, mBc, oB, loader, n_batches=3, warmup=0)
print(f'  (compile+warmup wall: {time.time()-t0:.1f}s)', flush=True)
bench('standard_compile', loss_standard, mBc, oB, loader)

print('[C] liger fused CE, no compile', flush=True)
mC, oC = build_model(); bench('liger_nocompile', loss_liger, mC, oC, loader)

print('[D] liger fused CE, compile(default)', flush=True)
mD, oD = build_model()
mDc = torch.compile(mD)
t0 = time.time(); bench('liger_compile_warmup', loss_liger, mDc, oD, loader, n_batches=3, warmup=0)
print(f'  (compile+warmup wall: {time.time()-t0:.1f}s)', flush=True)
bench('liger_compile', loss_liger, mDc, oD, loader)

# ---- Liger at bigger batch (memory headroom test) ----
print('[E] liger, bigger batch (bs=8192)', flush=True)
cfg.batch_size = 8192
mE, oE = build_model()
# can't reuse loader; build a fresh one at new bs
loader2 = DataLoader(
    _ParquetTokenPairDataset(
        parquet_path='/scratch/yuxuan.ren/data/dataset_stamp.parquet',
        vocab=vocab, text_column='stamp_smiles', split_column='split_random',
        split_value='train', require_all_pass=True,
        max_seq_len=64, max_rows=None, seed=42, rank=0, world_size=1,
    ),
    batch_size=8192,
    collate_fn=build_dynamic_lm_collate_fn(vocab.pad_id, pad_to_multiple_of=8),
    pin_memory=True, num_workers=8, prefetch_factor=6, persistent_workers=True,
)
# warm the iter
it2 = iter(loader2); next(it2)
del it2
try:
    bench('liger_nocompile_bs8k', loss_liger, mE, oE, loader2)
except torch.cuda.OutOfMemoryError as e:
    print(f'  OOM at bs=8192 (liger, no compile): {e}', flush=True)

print('[diag3] done', flush=True)
PY

echo "[remote] end=$(date)"

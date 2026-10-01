#!/bin/bash
# Phase 2 Layer 1+2: depth-aware temperature + MW-gated early stop.
# Both STAMP-exclusive (use motif-level vocab + STAMP grammar depth).
# No QED/SA oracle during inference.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_depth_mw}"
: "${N:=1024}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

python - <<PY
import sys, json, time, random, re
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _normalize_model_type, _trim_special_tail,
    sample_model_sequences, _sample_next_tokens,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}'); out_dir = Path('${OUT_DIR}')
_, vocab = _load_scan_cache(ckpt_dir)
ckpt = torch.load(ckpt_dir / 'model.pt', map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
cfg.model_type = _normalize_model_type(str(ckpt.get('model_type', cfg.model_type)))
dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[depthmw] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

# ---- Precompute motif-token MW lookup ----
is_struct_re = re.compile(r'^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring|END|BACK)\]$')
motif_mw = torch.zeros(len(vocab.itos), dtype=torch.float32, device=dev)
for idx, tok in enumerate(vocab.itos):
    if tok in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}: continue
    if is_struct_re.match(tok): continue  # structural token: mw=0
    try:
        m = Chem.MolFromSmiles(tok)
        if m is not None:
            motif_mw[idx] = float(Descriptors.MolWt(m))
    except Exception:
        pass
print(f'[depthmw] motif MW table: {int((motif_mw > 0).sum())}/{len(vocab.itos)} tokens have MW>0', flush=True)

# ---- Token-type indicators ----
is_branch_open = torch.zeros(len(vocab.itos), dtype=torch.bool, device=dev)
is_back = torch.zeros(len(vocab.itos), dtype=torch.bool, device=dev)
for idx, tok in enumerate(vocab.itos):
    if tok.startswith('[B_'):
        is_branch_open[idx] = True
    if tok == '[BACK]':
        is_back[idx] = True

N = ${N}

def decode_and_score(toks):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                c = Chem.MolToSmiles(m)
                return c, float(QED.qed(m)), float(sascorer.calculateScore(m)), m
    except Exception:
        pass
    return None, 0.0, 10.0, None

def score_token_seqs(seqs):
    canon=[]; mols=[]
    for toks in seqs:
        c, q, s, m = decode_and_score(toks)
        canon.append(c); mols.append(m)
    valid_m = [m for m in mols if m is not None]
    if not valid_m: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    qeds = np.array([QED.qed(m) for m in valid_m])
    sas  = np.array([sascorer.calculateScore(m) for m in valid_m])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in valid_m]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(mols); validity = len(valid_m)/total
    canons_nonnull = [c for c in canon if c]
    unique_rate = len(set(canons_nonnull))/len(canons_nonnull) if canons_nonnull else 0.0
    quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<42s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

def generate_layered(
    n_samples, max_new_tokens=64,
    base_temp=0.6, base_topk=10,
    depth_t_enabled=False, depth_t_min=0.5, depth_t_max=0.8, expected_len=14,
    mw_gate_enabled=False, mw_max=420, mw_boost_threshold=350, mw_boost=2.0,
):
    """AR sampling with optional Layer 1 (depth-T) + Layer 2 (MW gate)."""
    ids_list = [[vocab.bos_id] for _ in range(n_samples)]
    depth = [0] * n_samples
    running_mw = [0.0] * n_samples
    finished = [False] * n_samples

    for step in range(max_new_tokens):
        if all(finished): break
        active_idx = [i for i in range(n_samples) if not finished[i]]
        max_ctx = max(len(ids_list[i]) for i in active_idx)
        ctx = torch.full((len(active_idx), max_ctx), vocab.pad_id, dtype=torch.long, device=dev)
        for j, i in enumerate(active_idx):
            ctx[j, :len(ids_list[i])] = torch.tensor(ids_list[i], dtype=torch.long, device=dev)
        if max_ctx > cfg.max_seq_len:
            ctx = ctx[:, -cfg.max_seq_len:]
        with torch.no_grad():
            if cfg.model_type in ('ao_gpt', 'ao_gpt_pp'):
                ao_orders = torch.arange(ctx.size(1), device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx, orders=ao_orders).float()[:, -1, :]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx).float()[:, -1, :]

        # Suppress specials
        for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logits.size(-1):
                logits[:, int(sup)] = float('-inf')

        for b, i in enumerate(active_idx):
            cur_logits = logits[b].clone()
            # Layer 2: MW gate
            if mw_gate_enabled:
                running = running_mw[i]
                # forbid tokens whose motif MW would push over max_mw
                mw_after = running + motif_mw
                mask_over = mw_after > mw_max
                cur_logits[mask_over] = float('-inf')
                # boost EOS when running MW is high
                if running > mw_boost_threshold:
                    cur_logits[vocab.eos_id] += mw_boost
            # Layer 1: depth-aware temperature (per-sample)
            if depth_t_enabled:
                # Use token position / expected_len as depth proxy (simpler than actual tree depth)
                pos = step
                frac = min(1.0, pos / float(expected_len))
                temp = depth_t_min + (depth_t_max - depth_t_min) * frac
            else:
                temp = base_temp
            # Sample
            tok = _sample_next_tokens(cur_logits.unsqueeze(0), temperature=temp, top_k=base_topk).item()
            tok = int(tok)
            ids_list[i].append(tok)
            # Update bookkeeping
            running_mw[i] += float(motif_mw[tok].item())
            if tok == vocab.eos_id:
                finished[i] = True
    out = []
    for row in ids_list:
        trimmed = _trim_special_tail(row[1:], vocab)
        out.append(vocab.decode_ids(trimmed))
    return out

# ---- Run 4 configs ----
configs = [
    ('baseline AR (no gates)', dict()),
    ('Layer1: depth-T', dict(depth_t_enabled=True)),
    ('Layer2: MW gate', dict(mw_gate_enabled=True)),
    ('L1 + L2', dict(depth_t_enabled=True, mw_gate_enabled=True)),
    ('L2 mw_max=380', dict(mw_gate_enabled=True, mw_max=380)),
    ('L2 mw_max=500 (looser)', dict(mw_gate_enabled=True, mw_max=500)),
    ('L2 boost=4.0 (stronger EOS)', dict(mw_gate_enabled=True, mw_boost=4.0)),
    ('L1 wider T (0.5-1.0)', dict(depth_t_enabled=True, depth_t_max=1.0)),
]
results = []
for name, kw in configs:
    t0 = time.time()
    seqs = generate_layered(n_samples=N, **kw)
    dt = time.time() - t0
    s = score_token_seqs(seqs)
    results.append(dict(name=name, dt_s=dt, **s))
    print(fmt(f'{name} ({dt:.1f}s)', s), flush=True)

(out_dir / 'depth_mw_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP by Quality ----', flush=True)
for r in sorted(results, key=lambda r: -r['quality']):
    print(fmt(r['name'], r), flush=True)
print('\n---- TOP by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol']):
    print(fmt(r['name'], r), flush=True)
print(f'\n[depthmw] wrote {out_dir}/depth_mw_results.json', flush=True)
PY
echo "[remote] end=$(date)"

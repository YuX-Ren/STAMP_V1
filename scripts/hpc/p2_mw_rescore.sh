#!/bin/bash
# MW gate + AO rescore combo. Also fine-scan mw_max thresholds.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_mw_rescore}"
: "${N:=1024}"
: "${POOL:=4096}"
: "${M_PERMS:=16}"

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
    sample_model_sequences, _sample_next_tokens, _sample_ao_orders,
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
print(f'[mw-rescore] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

is_struct_re = re.compile(r'^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring|END|BACK)\]$')
motif_mw = torch.zeros(len(vocab.itos), dtype=torch.float32, device=dev)
for idx, tok in enumerate(vocab.itos):
    if tok in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}: continue
    if is_struct_re.match(tok): continue
    try:
        m = Chem.MolFromSmiles(tok)
        if m is not None:
            motif_mw[idx] = float(Descriptors.MolWt(m))
    except Exception: pass
print(f'[mw-rescore] motif MW table: {int((motif_mw > 0).sum())}/{len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

def generate_mw_gated(n_samples, mw_max=380, boost=2.0, boost_th=350,
                      base_temp=0.6, base_topk=10, max_new_tokens=64):
    ids_list = [[vocab.bos_id] for _ in range(n_samples)]
    running_mw = [0.0] * n_samples
    finished = [False] * n_samples
    for step in range(max_new_tokens):
        if all(finished): break
        active = [i for i in range(n_samples) if not finished[i]]
        max_ctx = max(len(ids_list[i]) for i in active)
        ctx = torch.full((len(active), max_ctx), vocab.pad_id, dtype=torch.long, device=dev)
        for j, i in enumerate(active):
            ctx[j, :len(ids_list[i])] = torch.tensor(ids_list[i], dtype=torch.long, device=dev)
        if max_ctx > cfg.max_seq_len:
            ctx = ctx[:, -cfg.max_seq_len:]
        with torch.no_grad():
            if cfg.model_type in ('ao_gpt','ao_gpt_pp'):
                ao = torch.arange(ctx.size(1), device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx, orders=ao).float()[:, -1, :]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx).float()[:, -1, :]
        for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logits.size(-1):
                logits[:, int(sup)] = float('-inf')
        for b, i in enumerate(active):
            cl = logits[b].clone()
            r = running_mw[i]
            # MW gate
            mask_over = (r + motif_mw) > mw_max
            cl[mask_over] = float('-inf')
            if r > boost_th:
                cl[vocab.eos_id] += boost
            tok = int(_sample_next_tokens(cl.unsqueeze(0), temperature=base_temp, top_k=base_topk).item())
            ids_list[i].append(tok)
            running_mw[i] += float(motif_mw[tok].item())
            if tok == vocab.eos_id: finished[i] = True
    out = []
    for row in ids_list:
        trimmed = _trim_special_tail(row[1:], vocab)
        out.append(vocab.decode_ids(trimmed))
    return out

def compute_ao_nll(token_seqs, M=M, BSZ=512):
    POOL_ = len(token_seqs)
    bos, eos, pad = vocab.bos_id, vocab.eos_id, vocab.pad_id
    id_seqs = [[bos] + vocab.encode_tokens(toks) + [eos] for toks in token_seqs]
    max_len = min(max(len(s) for s in id_seqs), cfg.max_seq_len)
    lengths = torch.tensor([min(len(s), max_len) for s in id_seqs], dtype=torch.long)
    padded = torch.full((POOL_, max_len), pad, dtype=torch.long)
    for i, s in enumerate(id_seqs):
        L = min(len(s), max_len); padded[i, :L] = torch.tensor(s[:L], dtype=torch.long)
    padded = padded.to(dev)
    x = padded[:, :-1].contiguous(); y = padded[:, 1:].contiguous()
    seq_mask_y = torch.arange(max_len - 1, device=dev).unsqueeze(0) < (lengths.to(dev).unsqueeze(1) - 1)
    labels = torch.where(seq_mask_y, y, torch.full_like(y, -100))
    all_nlls = torch.zeros(M, POOL_, device=dev)
    for m_idx in range(M):
        with torch.no_grad():
            for s in range(0, POOL_, BSZ):
                e = min(s + BSZ, POOL_)
                orders = _sample_ao_orders(labels[s:e], random_ratio=1.0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x[s:e], orders=orders).float()
                bi = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
                targets = y[s:e][bi, orders]; lab_sh = labels[s:e][bi, orders]
                log_probs = F.log_softmax(logits, dim=-1)
                nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                valid = (lab_sh != -100).float()
                all_nlls[m_idx, s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
    return all_nlls

def score_subset(token_seqs_sub):
    canon = []
    for toks in token_seqs_sub:
        try:
            smi = rep.decode(rep.detokenize(toks))
            if smi:
                m = Chem.MolFromSmiles(smi)
                canon.append(Chem.MolToSmiles(m) if m is not None else None)
            else:
                canon.append(None)
        except Exception: canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(canon); validity = len(valid)/total
    unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<48s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

results = []

# ---- Part A: fine MW threshold scan ----
print('\n==== Part A: fine MW threshold scan (first 1024 of 1024 each) ====', flush=True)
for mw_max in (340, 350, 360, 370, 380, 390, 400, 420, 450):
    t0 = time.time()
    seqs = generate_mw_gated(n_samples=N, mw_max=mw_max)
    dt = time.time() - t0
    s = score_subset(seqs)
    results.append(dict(name=f'MW={mw_max}', dt_s=dt, **s))
    print(fmt(f'MW={mw_max} ({dt:.0f}s)', s), flush=True)

# ---- Part B: MW gate (best) + AO rescore ----
# Use best mw_max from Part A, generate 4096 pool, rescore
best = max([r for r in results if r['name'].startswith('MW=')], key=lambda r: r['genmol'])
best_mw = int(best['name'].split('=')[1])
print(f'\n==== Part B: Best MW={best_mw} + AO rescore (pool=4096) ====', flush=True)
t0 = time.time()
seqs_pool = generate_mw_gated(n_samples=POOL, mw_max=best_mw)
print(f'  pool {POOL} in {time.time()-t0:.0f}s', flush=True)
t0 = time.time()
all_nlls = compute_ao_nll(seqs_pool)
print(f'  {M} perms in {time.time()-t0:.1f}s', flush=True)
nll_min = all_nlls.min(dim=0).values
nll_std = all_nlls.std(dim=0)
max_8 = all_nlls[:8].max(dim=0).values

def top(sig, k=N): return torch.argsort(sig, descending=True)[:k].cpu().tolist()

# various rescores on MW-gated pool
s = score_subset([seqs_pool[i] for i in range(N)])
results.append(dict(name=f'MW={best_mw} first-N', **s)); print(fmt(f'MW={best_mw} first-N', s))

s = score_subset([seqs_pool[i] for i in top(nll_min)])
results.append(dict(name=f'MW={best_mw} + bot-N min_16', **s)); print(fmt(f'MW={best_mw} + bot-N min_16', s))

s = score_subset([seqs_pool[i] for i in top(nll_min - 0.2 * nll_std)])
results.append(dict(name=f'MW={best_mw} + min-0.2*std', **s)); print(fmt(f'MW={best_mw} + min-0.2*std', s))

outer = nll_min - 0.5 * nll_std
idx_in = top(outer, k=int(1.3*N))
inner_set = torch.tensor(idx_in, device=dev)
rel = torch.argsort(max_8[inner_set], descending=True)[:N].cpu().tolist()
sel = [idx_in[i] for i in rel]
s = score_subset([seqs_pool[i] for i in sel])
results.append(dict(name=f'MW={best_mw} + (min-0.5std) top1.3N → max_8', **s))
print(fmt(f'MW={best_mw} + (min-0.5std) top1.3N → max_8', s))

(out_dir / 'mw_rescore_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-5 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:5]:
    print(fmt(r['name'], r), flush=True)
print('\n---- TOP-5 by Quality ----', flush=True)
for r in sorted(results, key=lambda r: -r['quality'])[:5]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[mw-rescore] wrote {out_dir}/mw_rescore_results.json', flush=True)
PY
echo "[remote] end=$(date)"

#!/bin/bash
# Fine mw_max scan + multi-seed variance check
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_mw_fine}"
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
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _normalize_model_type, _trim_special_tail, _sample_next_tokens,
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

is_struct_re = re.compile(r'^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring|END|BACK)\]$')
motif_mw = torch.zeros(len(vocab.itos), dtype=torch.float32, device=dev)
for idx, tok in enumerate(vocab.itos):
    if tok in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}: continue
    if is_struct_re.match(tok): continue
    try:
        m = Chem.MolFromSmiles(tok)
        if m is not None: motif_mw[idx] = float(Descriptors.MolWt(m))
    except Exception: pass
print(f'[mw-fine] motif MW table ready', flush=True)

def generate_mw_gated(n_samples, mw_max=370, boost=2.0, boost_th=300,
                      base_temp=0.6, base_topk=10, seed=None, max_new_tokens=64):
    if seed is not None: torch.manual_seed(seed)
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
        if max_ctx > cfg.max_seq_len: ctx = ctx[:, -cfg.max_seq_len:]
        with torch.no_grad():
            if cfg.model_type in ('ao_gpt','ao_gpt_pp'):
                ao = torch.arange(ctx.size(1), device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx, orders=ao).float()[:, -1, :]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx).float()[:, -1, :]
        for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logits.size(-1): logits[:, int(sup)] = float('-inf')
        for b, i in enumerate(active):
            cl = logits[b].clone()
            r = running_mw[i]
            mask_over = (r + motif_mw) > mw_max
            cl[mask_over] = float('-inf')
            if r > boost_th: cl[vocab.eos_id] += boost
            tok = int(_sample_next_tokens(cl.unsqueeze(0), temperature=base_temp, top_k=base_topk).item())
            ids_list[i].append(tok)
            running_mw[i] += float(motif_mw[tok].item())
            if tok == vocab.eos_id: finished[i] = True
    out = []
    for row in ids_list:
        trimmed = _trim_special_tail(row[1:], vocab)
        out.append(vocab.decode_ids(trimmed))
    return out

def score_subset(seqs):
    canon = []
    for toks in seqs:
        try:
            smi = rep.decode(rep.detokenize(toks))
            if smi:
                m = Chem.MolFromSmiles(smi)
                canon.append(Chem.MolToSmiles(m) if m is not None else None)
            else: canon.append(None)
        except Exception: canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(canon); validity = len(valid)/total; unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<42s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

N = ${N}
results = []

# Part 1: fine scan with 2 seeds per value
print('\n==== Fine MW scan (2 seeds each) ====', flush=True)
MW_VALS = [360, 365, 368, 370, 372, 375, 378, 382, 385, 388, 392, 395]
for mw_max in MW_VALS:
    for seed in (42, 123):
        t0 = time.time()
        seqs = generate_mw_gated(n_samples=N, mw_max=mw_max, seed=seed)
        dt = time.time() - t0
        s = score_subset(seqs)
        results.append(dict(name=f'MW={mw_max} seed={seed}', dt_s=dt, **s))
        print(fmt(f'MW={mw_max} seed={seed} ({dt:.0f}s)', s), flush=True)

# Part 2: best MW + boost_th scan
# Find best mw_max from Part 1 (mean over seeds)
by_mw = {}
for r in results:
    mw = int(r['name'].split()[0].split('=')[1])
    by_mw.setdefault(mw, []).append(r['genmol'])
avg_gm = {mw: np.mean(v) for mw, v in by_mw.items()}
best_mw = max(avg_gm, key=avg_gm.get)
print(f'\n[mw-fine] best mw_max (by avg GenMol): MW={best_mw}  avg={avg_gm[best_mw]:.2f}', flush=True)

# Also test boost_th variations at best mw_max
print('\n==== Boost_th variants at best MW ====', flush=True)
for boost_th in (280, 300, 320, 340):
    t0 = time.time()
    seqs = generate_mw_gated(n_samples=N, mw_max=best_mw, boost_th=boost_th, seed=42)
    dt = time.time() - t0
    s = score_subset(seqs)
    results.append(dict(name=f'MW={best_mw} boost_th={boost_th}', dt_s=dt, **s))
    print(fmt(f'MW={best_mw} boost_th={boost_th} ({dt:.0f}s)', s), flush=True)

(out_dir / 'mw_fine_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-5 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:5]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[mw-fine] wrote {out_dir}/mw_fine_results.json', flush=True)
PY
echo "[remote] end=$(date)"

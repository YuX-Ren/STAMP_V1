#!/bin/bash
# Compare unique under canonical vs raw (non-canonical) SMILES dedup,
# matching GenMol's raw-SMILES drop_duplicates approach.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
CKPT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009
PENALTY_NPY=$CKPT_DIR/phase2_sa_mine/log_ratio_penalty.npy
OUT_DIR=$CKPT_DIR/phase2_raw_unique
N=1024

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
stamp_codec = rep.codec

is_struct_re = re.compile(r'^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring|END|BACK)\]$')
motif_mw = torch.zeros(len(vocab.itos), dtype=torch.float32, device=dev)
for idx, tok in enumerate(vocab.itos):
    if tok in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}: continue
    if is_struct_re.match(tok): continue
    try:
        m = Chem.MolFromSmiles(tok)
        if m is not None: motif_mw[idx] = float(Descriptors.MolWt(m))
    except Exception: pass
log_ratio_np = np.load('${PENALTY_NPY}')
sa_penalty = torch.tensor(log_ratio_np, dtype=torch.float32, device=dev)

def generate(n_samples, max_new_tokens=64,
             mw_max=368, boost=2.0, boost_th=280,
             base_temp=0.6, base_topk=10,
             sa_penalty_w=0.0, seed=42):
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
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
        if sa_penalty_w > 0:
            logits = logits - sa_penalty_w * sa_penalty.unsqueeze(0)
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

def score_both(seqs):
    """Return metrics under canonical dedup AND raw (non-canonical) dedup + GenMol Quality."""
    raw_smi = []    # non-canonical SMILES (matches GenMol raw drop_duplicates)
    canon_smi = []  # canonical SMILES (our strict dedup)
    for toks in seqs:
        try:
            text = rep.detokenize(toks)
            mol = stamp_codec.decode_stamp_to_mol(text)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raw_smi.append(None); canon_smi.append(None); continue
            raw = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
            canon = Chem.MolToSmiles(mol, canonical=True,  isomericSmiles=True)
            raw_smi.append(raw if raw else None)
            canon_smi.append(canon if canon else None)
        except Exception:
            raw_smi.append(None); canon_smi.append(None)

    total = len(seqs)
    valid_raw = [s for s in raw_smi if s]
    valid_canon = [s for s in canon_smi if s]
    validity = len(valid_raw) / total

    # Uniqueness: two formulations
    uniq_raw = len(set(valid_raw)) / len(valid_raw) if valid_raw else 0.0
    uniq_canon = len(set(valid_canon)) / len(valid_canon) if valid_canon else 0.0

    # Quality (our old product): validity * unique * (qmask over valid mean)
    mols = [Chem.MolFromSmiles(s) for s in valid_canon]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])

    # Strict thresholds as we used before (>0.6 and <4)
    qmask_strict = (qeds > 0.6) & (sas < 4.0)
    # GenMol's thresholds (>=0.6 and <=4)
    qmask_genmol = (qeds >= 0.6) & (sas <= 4.0)

    # GenMol-style Quality: dedup first (raw), then filter QED>=0.6 & SA<=4, /N
    # Build df-like raw dedup
    seen = set(); keep_idx = []
    for i, s in enumerate(raw_smi):
        if s is None: continue
        if s in seen: continue
        seen.add(s); keep_idx.append(i)
    # For each kept index, recompute qed/sa on the canonical of the same mol
    # (we reuse valid_canon ordering; need mapping raw_smi idx -> valid_canon pos)
    # Simpler: recompute on raw->mol each time (consistent with GenMol using raw samples to TDC oracle)
    genmol_hits = 0
    for i in keep_idx:
        m = Chem.MolFromSmiles(raw_smi[i])
        if m is None: continue
        q = QED.qed(m); s = sascorer.calculateScore(m)
        if q >= 0.6 and s <= 4.0:
            genmol_hits += 1
    genmol_quality = genmol_hits / total  # GenMol's single-number score

    # Canonical-dedup variant of the same formula (our strict match)
    seen2 = set(); keep2 = []
    for i, s in enumerate(canon_smi):
        if s is None: continue
        if s in seen2: continue
        seen2.add(s); keep2.append(i)
    canon_hits = 0
    for i in keep2:
        m = Chem.MolFromSmiles(canon_smi[i])
        if m is None: continue
        q = QED.qed(m); s = sascorer.calculateScore(m)
        if q >= 0.6 and s <= 4.0:
            canon_hits += 1
    canon_quality_geneq = canon_hits / total

    # Product-form (our old metric)
    old_qmean = float(qmask_strict.mean())
    old_genmol = validity * uniq_canon * old_qmean  # original product

    return dict(
        validity=validity*100,
        uniq_raw=uniq_raw*100,
        uniq_canon=uniq_canon*100,
        qed_mean=float(qeds.mean()),
        sa_mean=float(sas.mean()),
        quality_ours_product=old_qmean*100,
        genmol_score_raw=genmol_quality*100,          # GenMol apples-to-apples (raw dedup)
        genmol_score_canon=canon_quality_geneq*100,   # strict canonical dedup variant
        genmol_score_ours_product=old_genmol*100,     # our old product-form number
    )

N = ${N}
results = []
configs = [
    ('MW gate baseline (mw=368,bt=280)', dict()),
    ('+ SA-pen a=0.3',                   dict(sa_penalty_w=0.3)),
]
for name, kw in configs:
    for seed in (42, 123):
        t0 = time.time()
        seqs = generate(n_samples=N, seed=seed, **kw)
        dt = time.time() - t0
        s = score_both(seqs)
        full = f'{name} seed={seed}'
        results.append(dict(name=full, dt_s=dt, **s))
        print(f'[{full:<38s}] '
              f'valid={s["validity"]:>5.1f}  uRaw={s["uniq_raw"]:>5.2f}  uCanon={s["uniq_canon"]:>5.2f}  '
              f'GM_raw={s["genmol_score_raw"]:>5.2f}  GM_canon={s["genmol_score_canon"]:>5.2f}  '
              f'GM_old={s["genmol_score_ours_product"]:>5.2f}  '
              f'QED={s["qed_mean"]:.3f}  SA={s["sa_mean"]:.2f}', flush=True)

(out_dir / 'raw_unique_results.json').write_text(json.dumps(results, indent=2))
print('\n---- Averages (2 seeds) ----', flush=True)
by_name = {}
for r in results:
    base = r['name'].rsplit(' seed=', 1)[0]
    by_name.setdefault(base, []).append(r)
for base, rs in by_name.items():
    print(f'[{base:<38s}]  '
          f'uRaw={np.mean([r["uniq_raw"] for r in rs]):.2f}  '
          f'uCanon={np.mean([r["uniq_canon"] for r in rs]):.2f}  '
          f'GM_raw={np.mean([r["genmol_score_raw"] for r in rs]):.2f}  '
          f'GM_canon={np.mean([r["genmol_score_canon"] for r in rs]):.2f}  '
          f'GM_old={np.mean([r["genmol_score_ours_product"] for r in rs]):.2f}',
          flush=True)
print(f'\n[raw-unique] wrote {out_dir}/raw_unique_results.json', flush=True)
PY
echo "[remote] end=$(date)"

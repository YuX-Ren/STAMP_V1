#!/bin/bash
# A-tune v2: extend oversample 4x-8x, explore (nll - α*std) α sweep
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore_tune2}"
: "${N:=1024}"
: "${POOL:=8192}"
: "${M_PERMS:=8}"
: "${TEMP:=0.6}"
: "${TOPK:=10}"
: "${SAMPLE_CHUNK:=2048}"   # sample in chunks to avoid peak mem

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

python - <<PY
import sys, json, time, random
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, sample_model_sequences, _normalize_model_type,
    _sample_ao_orders,
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
print(f'[A-tune2] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}
TEMP = ${TEMP}; TOPK = ${TOPK}; CHUNK = ${SAMPLE_CHUNK}

# ---- Generate draft pool in chunks to avoid OOM ----
t0 = time.time()
token_seqs = []
for cs in range(0, POOL, CHUNK):
    ce = min(cs + CHUNK, POOL)
    with torch.no_grad():
        seqs = sample_model_sequences(
            model, vocab, n_samples=ce-cs, max_new_tokens=64,
            temperature=TEMP, top_k=TOPK, device=dev, ao_decode_mode='ar',
        )
    token_seqs.extend(seqs)
    torch.cuda.empty_cache()
print(f'[A-tune2] generated {POOL} drafts in {time.time()-t0:.1f}s (chunks of {CHUNK})', flush=True)

bos, eos, pad = vocab.bos_id, vocab.eos_id, vocab.pad_id
id_seqs = [[bos] + vocab.encode_tokens(toks) + [eos] for toks in token_seqs]
max_len = min(max(len(s) for s in id_seqs), cfg.max_seq_len)
lengths = torch.tensor([min(len(s), max_len) for s in id_seqs], dtype=torch.long)
padded = torch.full((POOL, max_len), pad, dtype=torch.long)
for i, s in enumerate(id_seqs):
    L = min(len(s), max_len); padded[i, :L] = torch.tensor(s[:L], dtype=torch.long)
padded = padded.to(dev)
x = padded[:, :-1].contiguous(); y = padded[:, 1:].contiguous()
seq_mask_y = torch.arange(max_len - 1, device=dev).unsqueeze(0) < (lengths.to(dev).unsqueeze(1) - 1)
labels = torch.where(seq_mask_y, y, torch.full_like(y, -100))

BSZ = 512
all_nlls = torch.zeros(M, POOL, device=dev)
t_rescore = time.time()
for m_idx in range(M):
    with torch.no_grad():
        for s in range(0, POOL, BSZ):
            e = min(s + BSZ, POOL)
            x_c = x[s:e]; y_c = y[s:e]; lab_c = labels[s:e]
            orders = _sample_ao_orders(lab_c, random_ratio=1.0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_c, orders=orders).float()
            batch_idx = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
            targets = y_c[batch_idx, orders]; lab_sh = lab_c[batch_idx, orders]
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = (lab_sh != -100).float()
            all_nlls[m_idx, s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
print(f'[A-tune2] computed NLL for {M} perms x {POOL} samples in {time.time()-t_rescore:.1f}s', flush=True)

mean_nll = all_nlls.mean(dim=0)
std_nll  = all_nlls.std(dim=0)

def score_subset(idx_list):
    seqs = [token_seqs[i] for i in idx_list]
    smiles = []
    for toks in seqs:
        try: smiles.append(rep.decode(rep.detokenize(toks)))
        except Exception: smiles.append(None)
    canon = []
    for smi in smiles:
        if not smi: canon.append(None); continue
        try:
            m = Chem.MolFromSmiles(smi)
            canon.append(Chem.MolToSmiles(m) if m is not None else None)
        except Exception:
            canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100, unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

def fmt(name, s):
    return f'[{name:<32s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}'

results = []
print('\n---- oversample sweep (bot-N by mean_nll) ----', flush=True)
for os_mult in (4, 5, 6, 7, 8):
    pool_size = os_mult * N
    if pool_size > POOL: continue
    sub = mean_nll[:pool_size]
    idx = torch.argsort(sub, descending=True)[:N].cpu().tolist()
    s = score_subset(idx)
    results.append(dict(name=f'bot-N OS={os_mult}x', **s))
    print(fmt(f'bot-N OS={os_mult}x mean_nll', s), flush=True)

print('\n---- combined (nll - alpha*std), OS=4x ----', flush=True)
for alpha in (0.5, 1.0, 2.0, 3.0):
    pool_size = 4 * N
    sub_n = mean_nll[:pool_size]; sub_s = std_nll[:pool_size]
    metric = sub_n - alpha * sub_s
    idx = torch.argsort(metric, descending=True)[:N].cpu().tolist()
    s = score_subset(idx)
    results.append(dict(name=f'bot-N (nll-{alpha}*std) OS=4x', **s))
    print(fmt(f'bot-N (nll-{alpha}*std) OS=4x', s), flush=True)

print('\n---- combined (nll - alpha*std), OS=8x ----', flush=True)
for alpha in (0.5, 1.0, 2.0):
    pool_size = 8 * N
    if pool_size > POOL: continue
    sub_n = mean_nll[:pool_size]; sub_s = std_nll[:pool_size]
    metric = sub_n - alpha * sub_s
    idx = torch.argsort(metric, descending=True)[:N].cpu().tolist()
    s = score_subset(idx)
    results.append(dict(name=f'bot-N (nll-{alpha}*std) OS=8x', **s))
    print(fmt(f'bot-N (nll-{alpha}*std) OS=8x', s), flush=True)

(out_dir / 'tune2_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[A-tune2] wrote {out_dir}/tune2_results.json', flush=True)
PY
echo "[remote] end=$(date)"

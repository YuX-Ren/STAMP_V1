#!/bin/bash
# B: mask-and-regenerate refinement. AR draft -> identify lowest-confidence
# position per sample -> swap into "second-to-last AO order slot" (full
# context except the target) -> resample. Repeat R iterations.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/refine}"
: "${N:=1024}"
: "${TEMP:=0.6}"
: "${TOPK:=10}"
: "${R_STEPS:=3}"     # refine iterations
: "${TEMP_REFINE:=1.2}"      # temperature used for refill (higher = more exploration)
: "${TOPK_REFINE:=50}"       # top-k for refill
: "${EXCLUDE_DRAFT:=0}"      # 1 = forbid the original draft token during refill

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
    _sample_next_tokens,
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
print(f'[refine] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; TEMP = ${TEMP}; TOPK = ${TOPK}; R = ${R_STEPS}
TEMP_REFINE = ${TEMP_REFINE}; TOPK_REFINE = ${TOPK_REFINE}; EXCLUDE_DRAFT = bool(${EXCLUDE_DRAFT})
print(f'[refine] draft: T={TEMP} K={TOPK}  refill: T={TEMP_REFINE} K={TOPK_REFINE} exclude_draft={EXCLUDE_DRAFT}', flush=True)

# ---- Draft generation (AR) ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=N, max_new_tokens=64,
        temperature=TEMP, top_k=TOPK, device=dev, ao_decode_mode='ar',
    )
print(f'[refine] generated {N} drafts in {time.time()-t0:.1f}s  (T={TEMP} K={TOPK})', flush=True)

# Build padded id tensor
bos, eos, pad = vocab.bos_id, vocab.eos_id, vocab.pad_id
id_seqs = [[bos] + vocab.encode_tokens(toks) + [eos] for toks in token_seqs]
max_len = min(max(len(s) for s in id_seqs), cfg.max_seq_len)
ids = torch.full((N, max_len), pad, dtype=torch.long, device=dev)
lengths = torch.zeros(N, dtype=torch.long, device=dev)
for i, s in enumerate(id_seqs):
    L = min(len(s), max_len)
    ids[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=dev)
    lengths[i] = L
print(f'[refine] padded to L={max_len}', flush=True)

# Score helper (tokens -> canonical SMILES -> metrics)
def ids_to_tokens(id_row, length):
    # Strip BOS/EOS/pad
    out_ids = id_row[1:length-1].tolist()  # between BOS and EOS
    return vocab.decode_ids(out_ids)

def score_ids(ids_mat, lens):
    seqs = [ids_to_tokens(ids_mat[i], int(lens[i])) for i in range(ids_mat.size(0))]
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
    if not valid:
        return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims = []; n = len(fps); pairs = min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100, unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

print('[refine] scoring baseline draft...', flush=True)
s0 = score_ids(ids, lengths)
print(f'[baseline]  q={s0["quality"]:.2f}%  u={s0["unique"]:.2f}%  d={s0["div"]:.3f}  QED={s0["qed"]:.3f}  SA={s0["sa"]:.2f}', flush=True)

# ---- Iterative refine ----
# Definitions in y-space (indices 0..max_len-2):
#   y[t] = ids[t+1]; first_valid_y = 0; last_valid_y = lengths - 2 (EOS position in y)
for r_iter in range(1, R+1):
    t_iter = time.time()
    x = ids[:, :-1].contiguous()
    y = ids[:, 1:].contiguous()
    L_y = y.size(1)
    # Valid mask in y-space (excludes pad)
    y_valid_mask = torch.arange(L_y, device=dev).unsqueeze(0) < (lengths - 1).unsqueeze(1)  # (N, L_y)
    last_valid_y = (lengths - 2).clamp(min=1)  # (N,), y-index of EOS

    # Per-token log-prob via identity order
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits_id = model(x).float()                             # (N, L_y, V)
    log_probs = F.log_softmax(logits_id, dim=-1)
    token_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)      # (N, L_y)
    # Protect: invalid (pad), first valid (idx 0), last valid (EOS)
    token_lp = token_lp.masked_fill(~y_valid_mask, float('inf'))
    token_lp[:, 0] = float('inf')
    bi = torch.arange(N, device=dev)
    token_lp[bi, last_valid_y] = float('inf')
    # Lowest-confidence position per sample
    mask_pos = token_lp.argmin(dim=-1)                               # (N,), y-index
    target_pos = (last_valid_y - 1).clamp(min=1)                     # where to slot it

    # Build orders: identity with mask_pos <-> target_pos swapped
    orders = torch.arange(L_y, device=dev).unsqueeze(0).expand(N, L_y).contiguous()
    # swap
    a = orders[bi, mask_pos].clone()
    b_ = orders[bi, target_pos].clone()
    orders[bi, mask_pos] = b_
    orders[bi, target_pos] = a

    # Forward and fetch logit at order output index target_pos (predicts y[mask_pos])
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x, orders=orders).float()                 # (N, L_y, V)
    logit_refill = logits[bi, target_pos, :].clone()                 # (N, V)
    # Forbid specials and EOS for this middle refill
    for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.eos_id):
        if 0 <= int(sup) < logit_refill.size(-1):
            logit_refill[:, int(sup)] = float('-inf')
    # Optionally forbid the original draft token to force a change
    if EXCLUDE_DRAFT:
        old_tok_for_excl = ids[bi, mask_pos + 1]
        logit_refill[bi, old_tok_for_excl] = float('-inf')
    new_tok = _sample_next_tokens(logit_refill, temperature=TEMP_REFINE, top_k=TOPK_REFINE)

    # --- Diagnostics ---
    old_tok = ids[bi, mask_pos + 1].clone()
    n_same_mp_tp = int((mask_pos == target_pos).sum().item())
    n_changed = int((new_tok != old_tok).sum().item())
    # Log top-1 of refill distribution vs drafted token
    top1 = logit_refill.argmax(dim=-1)
    n_top1_eq_draft = int((top1 == old_tok).sum().item())
    print(f'[iter {r_iter} diag] mp==tp: {n_same_mp_tp}/{N}  changed: {n_changed}/{N}  '
          f'top1==draft: {n_top1_eq_draft}/{N}', flush=True)
    # Print 3 examples
    for demo in range(min(3, N)):
        ot = vocab.itos[int(old_tok[demo])]; nt = vocab.itos[int(new_tok[demo])]
        mp = int(mask_pos[demo]); tp = int(target_pos[demo])
        print(f'  ex{demo}: mp={mp} tp={tp}  draft={ot!r} new={nt!r}', flush=True)

    # Write back: ids[b, mp+1] = new_tok
    ids[bi, mask_pos + 1] = new_tok

    s = score_ids(ids, lengths)
    dt_iter = time.time() - t_iter
    print(f'[iter {r_iter}]   q={s["quality"]:.2f}%  u={s["unique"]:.2f}%  d={s["div"]:.3f}  '
          f'QED={s["qed"]:.3f}  SA={s["sa"]:.2f}   ({dt_iter:.2f}s)', flush=True)

# Dump final
with (out_dir / 'refine.json').open('w') as f:
    json.dump(dict(T=TEMP, K=TOPK, R=R, N=N, baseline=s0, final=s), f, indent=2)
print(f'[refine] wrote {out_dir}/refine.json', flush=True)
PY
echo "[remote] end=$(date)"

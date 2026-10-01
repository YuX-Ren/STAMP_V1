#!/bin/bash
# P1.A: Fixed-length confidence-guided decoding (MaskGIT-style for AO-GPT).
# - Sample target lengths from empirical train-length distribution
# - Init [BOS, PAD..., EOS] per sample
# - Iteratively unmask N most-confident middle positions via full AO context
# - First-valid (y[0]) always predicted at order pos 0 (stay in training dist)
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/conf_decoding}"
: "${N:=1024}"

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
    LMConfig, TinyDecoderLM, _normalize_model_type, _trim_special_tail,
    _apply_top_k_logits, sample_model_sequences,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}'); out_dir = Path('${OUT_DIR}')
scan_summary, vocab = _load_scan_cache(ckpt_dir)
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
print(f'[conf] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

# Empirical sequence length distribution (in ids, including BOS+EOS)
# scan_summary.train_lengths is lengths of token sequences (valid_sequences were 1+len+1)
# valid_sequences are length-prefixed with BOS+EOS so its length = orig_len + 2
# We need ids-length including BOS and EOS.
# scan_cache.train_lengths are the original token counts (without BOS/EOS).
train_lens = np.array(scan_summary.train_lengths)
# Convert to "ids length" = token_count + 2 (BOS + EOS)
ids_lens_dist = train_lens + 2
print(f'[conf] length dist (ids): mean={ids_lens_dist.mean():.1f}  median={np.median(ids_lens_dist)}  '
      f'min={ids_lens_dist.min()} max={ids_lens_dist.max()}', flush=True)


def confidence_decode(
    model, vocab, n_samples, lengths, *,
    max_len=64, temperature=1.0, gumbel_r=0.5, unmask_per_step=1, top_k=0, dev=dev,
):
    """MaskGIT-style decoding with EOS fixed and first-valid-first constraint.

    lengths: (N,) desired ids length per sample (BOS..EOS inclusive).
    Returns list of token-string sequences (List[List[str]]).
    """
    assert max_len >= int(lengths.max().item()), "max_len must cover longest sample"
    L_y = max_len - 1  # y-space length
    rows = torch.arange(n_samples, device=dev)
    lengths = lengths.to(dev).long()
    eos_pos = lengths - 1                          # ids-index of EOS
    last_valid_y = eos_pos - 1                     # y-index of EOS (i.e. last valid y)

    # Init ids with [BOS, PAD..., EOS, PAD..]
    ids = torch.full((n_samples, max_len), vocab.pad_id, dtype=torch.long, device=dev)
    ids[:, 0] = vocab.bos_id
    ids[rows, eos_pos] = vocab.eos_id

    # filled mask in y-space: True where y[i] is known (EOS only at start)
    filled = torch.zeros(n_samples, L_y, dtype=torch.bool, device=dev)
    filled[rows, last_valid_y] = True
    idx_L = torch.arange(L_y, device=dev)
    position_in_valid = idx_L.unsqueeze(0) <= last_valid_y.unsqueeze(1)  # True for 0..last_valid_y
    # Out-of-range (beyond EOS): treat as "not to predict"
    needs_pred = position_in_valid & ~filled

    t0 = time.time()
    step = 0
    while True:
        if not bool(needs_pred.any().item()):
            break
        step += 1
        # --- Build orders ---
        # Key-based argsort: y[0]=0, observed middle=1+i/L, masked middle=2+i/L, last=4, beyond=5+i/L
        L_y_range_f = idx_L.unsqueeze(0).expand(n_samples, L_y).float()
        is_first = (idx_L.unsqueeze(0) == 0)
        is_last = (idx_L.unsqueeze(0) == last_valid_y.unsqueeze(1))
        is_beyond = ~position_in_valid
        is_first_and_valid = is_first & position_in_valid
        is_obs_mid = position_in_valid & filled & ~is_first & ~is_last
        is_mask_mid = needs_pred & ~is_first & ~is_last
        is_first_mask = is_first_and_valid & ~filled  # y[0] unfilled
        is_first_obs = is_first_and_valid & filled    # y[0] filled
        keys = torch.full((n_samples, L_y), 100.0, device=dev)
        keys[is_first_mask] = 0.0
        keys[is_first_obs] = 0.0
        keys[is_obs_mid] = 1.0 + L_y_range_f[is_obs_mid] / float(L_y)
        keys[is_mask_mid] = 2.0 + L_y_range_f[is_mask_mid] / float(L_y)
        keys[is_last] = 4.0
        keys[is_beyond] = 5.0 + L_y_range_f[is_beyond] / float(L_y)
        orders = keys.argsort(dim=-1)             # (N, L_y)
        inv_orders = torch.argsort(orders, dim=-1)

        # --- Forward ---
        x_in = ids[:, :-1].contiguous()           # (N, L_y)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_in, orders=orders).float()  # (N, L_y, V)

        # Scaled logits for SAMPLING (temperature divide), and confidence is top-1 prob
        if top_k > 0:
            logits_f = _apply_top_k_logits(logits, top_k=top_k)
        else:
            logits_f = logits
        # For scoring confidence: use softmax at temperature
        logits_scaled = logits_f / max(temperature, 1e-6)
        probs = F.softmax(logits_scaled, dim=-1)
        top1_prob = probs.max(dim=-1).values      # (N, L_y) at each ORDER position

        # Gumbel noise on log-confidence (Chang et al. 2022)
        if gumbel_r > 0:
            u = torch.rand_like(top1_prob).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            conf_score = top1_prob.clamp(min=1e-20).log() + gumbel_r * gumbel
        else:
            conf_score = top1_prob.clamp(min=1e-20).log()

        # Map from order-position space to y-position space
        batch_idx = rows.unsqueeze(1).expand(n_samples, L_y)
        conf_y = conf_score[batch_idx, inv_orders]   # (N, L_y) — confidence to predict y[i]

        # Enforce: if y[0] unfilled, MUST be predicted this step (not driven by confidence)
        # Otherwise, pick the most confident from needs_pred
        y0_unfilled = is_first_mask.squeeze(-1) if is_first_mask.dim() == 2 else is_first_mask
        # is_first_mask shape (N, L_y); we want per-sample bool whether y[0] unfilled
        y0_must_pick = is_first_mask[:, 0]        # (N,)

        conf_y = conf_y.masked_fill(~needs_pred, float('-inf'))
        # For samples where y[0] must be picked, force pick index 0
        target_y_idx = conf_y.argmax(dim=-1)      # (N,)
        target_y_idx = torch.where(y0_must_pick, torch.zeros_like(target_y_idx), target_y_idx)

        # Fetch logit at the target y-position
        target_order_pos = inv_orders[rows, target_y_idx]
        target_logits = logits_f[rows, target_order_pos, :].clone()  # (N, V)
        # Suppress specials (never sample BOS/PAD/EOS/UNK for middle)
        for sup in (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < target_logits.size(-1):
                target_logits[:, int(sup)] = float('-inf')
        target_logits_scaled = target_logits / max(temperature, 1e-6)
        target_probs = F.softmax(target_logits_scaled, dim=-1)
        sampled_tok = torch.multinomial(target_probs, num_samples=1).squeeze(-1)

        # Write back ids[b, target_y_idx[b] + 1] = sampled_tok[b], but ONLY if the target
        # was genuinely in needs_pred. For samples fully done, no-op.
        any_active = needs_pred.any(dim=-1)       # (N,)
        ids[rows, target_y_idx + 1] = torch.where(any_active, sampled_tok, ids[rows, target_y_idx + 1])
        filled[rows, target_y_idx] = filled[rows, target_y_idx] | any_active
        needs_pred = position_in_valid & ~filled

    dt = time.time() - t0
    print(f'[conf]   decode done in {dt:.1f}s ({step} steps)', flush=True)

    out = []
    all_ids = ids[:, 1:].cpu().tolist()            # strip BOS
    for row in all_ids:
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def score(seqs):
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
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0., genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    N_total = len(canon)
    validity = len(valid) / N_total
    unique_rate = len(set(valid)) / len(valid)
    quality = float(qmask.mean())
    # GenMol formula: valid AND unique AND qed+sa on total
    genmol = validity * unique_rate * quality
    return dict(
        validity=validity*100, unique=unique_rate*100,
        quality=quality*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
        genmol=genmol*100,
    )

def fmt(name, s):
    return (f'[{name:<32s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  v={s["validity"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

N = ${N}
# Sample target lengths once for all configs (fair comparison)
rng = np.random.default_rng(42)
sampled_lengths_np = rng.choice(ids_lens_dist, size=N, replace=True)
sampled_lengths = torch.tensor(np.clip(sampled_lengths_np, 4, 63), dtype=torch.long, device=dev)
print(f'[conf] sampled per-sample target lengths: mean={sampled_lengths.float().mean().item():.1f}  '
      f'p90={sampled_lengths.float().quantile(0.9).item():.0f}', flush=True)

results = []

# Reference: plain AR (current production)
print('\n==== Reference: AR T=0.6 K=10 ====', flush=True)
t0 = time.time()
with torch.no_grad():
    seqs_ar = sample_model_sequences(
        model, vocab, n_samples=N, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'  sampled in {time.time()-t0:.1f}s', flush=True)
s = score(seqs_ar); results.append(dict(name='AR T=0.6 K=10', **s)); print(fmt('AR T=0.6 K=10', s), flush=True)

# Confidence decoding sweep
print('\n==== Confidence decoding (fixed-length, MaskGIT-AO-GPT) ====', flush=True)
for temp, r in [(0.5, 0.5), (0.5, 1.0), (0.7, 0.5), (0.7, 1.0), (1.0, 0.5), (1.0, 1.0)]:
    print(f'\n-- T={temp}, r={r}, unmask_per_step=1 --', flush=True)
    t0 = time.time()
    seqs = confidence_decode(model, vocab, N, sampled_lengths, max_len=64,
                             temperature=temp, gumbel_r=r, unmask_per_step=1, top_k=0, dev=dev)
    print(f'  total time {time.time()-t0:.1f}s', flush=True)
    s = score(seqs); results.append(dict(name=f'conf T={temp} r={r}', **s))
    print(fmt(f'conf T={temp} r={r}', s), flush=True)

(out_dir / 'conf_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[conf] wrote {out_dir}/conf_results.json', flush=True)
PY
echo "[remote] end=$(date)"

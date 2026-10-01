#!/bin/bash
# P2 combo: partial-QED guided drafting (STAMP-exclusive) + AO multi-order rescore.
# Both signals are uniquely STAMP+AO; GenMol/SAFE cannot replicate either stage.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_combo}"
: "${N:=1024}"
: "${POOL:=4096}"
: "${M_PERMS:=16}"

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
    sample_model_sequences, _sample_ao_orders,
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
print(f'[combo] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

def partial_mol_and_score(ids_list):
    try:
        tokens = vocab.decode_ids(ids_list)
        tokens_clean = [t for t in tokens if t not in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}]
        stamp_text = rep.detokenize(tokens_clean)
        if not stamp_text.strip(): return None, 0.0, 10.0
        mol = stamp_codec.decode_stamp_to_mol(stamp_text)
        if mol is None: return None, 0.0, 10.0
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags: return None, 0.0, 10.0
        main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return main, float(QED.qed(main)), float(sascorer.calculateScore(main))
    except Exception:
        return None, 0.0, 10.0

def partial_qed_guided_ar(n_samples, max_new_tokens=64, qw=5.0, sw=1.0,
                           cand_K=5, base_topk=10, base_temp=0.6,
                           guidance_start_step=2, dev=dev):
    ids_list = [[vocab.bos_id] for _ in range(n_samples)]
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
            if cfg.model_type in ('ao_gpt','ao_gpt_pp'):
                ao_orders = torch.arange(ctx.size(1), device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx, orders=ao_orders).float()[:, -1, :]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx).float()[:, -1, :]
        for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logits.size(-1):
                logits[:, int(sup)] = float('-inf')
        scaled = logits / base_temp
        topk_lp, topk_ids = torch.topk(scaled, k=min(base_topk, scaled.size(-1)), dim=-1)
        cand_lp = topk_lp[:, :cand_K]; cand_id = topk_ids[:, :cand_K]
        use_guidance = step >= guidance_start_step
        for b, i in enumerate(active_idx):
            if not use_guidance:
                top_vals, top_inds = torch.topk(torch.softmax(scaled[b] - scaled[b].max(), dim=-1),
                                                k=min(base_topk, scaled.size(-1)))
                top_vals = top_vals / top_vals.sum()
                tok = int(top_inds[torch.multinomial(top_vals, num_samples=1).item()].item())
            else:
                scores = torch.empty(cand_K, device=dev)
                for k in range(cand_K):
                    cand_tok = int(cand_id[b, k].item())
                    lp_k = float(cand_lp[b, k].item())
                    probe = ids_list[i][1:] + [cand_tok]
                    _, q, sa = partial_mol_and_score(probe)
                    scores[k] = lp_k + qw * q - sw * sa
                probs = torch.softmax(scores - scores.max(), dim=0)
                pick = int(torch.multinomial(probs, num_samples=1).item())
                tok = int(cand_id[b, pick].item())
            ids_list[i].append(tok)
            if tok == vocab.eos_id: finished[i] = True
    out = []
    for row_ids in ids_list:
        row = row_ids[1:]; row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out

def score_subset(idx_list, token_seqs):
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
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
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
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  SA={s["sa"]:.2f}')

def analyze_pool(pool_name, token_seqs):
    """Pad, compute AO NLL, run rescore strategies. Returns list of results."""
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

    BSZ = 512
    all_nlls = torch.zeros(M, POOL_, device=dev)
    for m_idx in range(M):
        with torch.no_grad():
            for s in range(0, POOL_, BSZ):
                e = min(s + BSZ, POOL_)
                x_c = x[s:e]; y_c = y[s:e]; lab_c = labels[s:e]
                orders = _sample_ao_orders(lab_c, random_ratio=1.0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x_c, orders=orders).float()
                bi = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
                targets = y_c[bi, orders]; lab_sh = lab_c[bi, orders]
                log_probs = F.log_softmax(logits, dim=-1)
                nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                valid = (lab_sh != -100).float()
                all_nlls[m_idx, s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
    nll_min = all_nlls.min(dim=0).values
    nll_std = all_nlls.std(dim=0)
    max_8 = all_nlls[:8].max(dim=0).values

    def top(sig, k=N): return torch.argsort(sig, descending=True)[:k].cpu().tolist()

    out_results = []
    # Baseline first-N
    s = score_subset(list(range(min(N, POOL_))), token_seqs)
    out_results.append(dict(name=f'{pool_name}:first-N', **s)); print(fmt(f'{pool_name}:first-N', s))
    # Pure min_16
    s = score_subset(top(nll_min), token_seqs)
    out_results.append(dict(name=f'{pool_name}:min_16', **s)); print(fmt(f'{pool_name}:min_16', s))
    # min - 0.2*std
    s = score_subset(top(nll_min - 0.2 * nll_std), token_seqs)
    out_results.append(dict(name=f'{pool_name}:min-0.2std', **s)); print(fmt(f'{pool_name}:min-0.2std', s))
    # 2-stage winner
    outer = nll_min - 0.5 * nll_std
    idx_in = top(outer, k=int(1.3 * N))
    inner_set = torch.tensor(idx_in, device=dev)
    rel = torch.argsort(max_8[inner_set], descending=True)[:N].cpu().tolist()
    sel = [idx_in[i] for i in rel]
    s = score_subset(sel, token_seqs)
    out_results.append(dict(name=f'{pool_name}:(min-0.5std) top1.3N→max_8', **s))
    print(fmt(f'{pool_name}:(min-0.5std) top1.3N→max_8', s))
    # Oracle filter + dedup
    t0_o = time.time()
    qed_p = np.full(POOL_, -1.0); sa_p = np.full(POOL_, 10.0); canon_p = [None]*POOL_
    for i, toks in enumerate(token_seqs):
        try:
            smi = rep.decode(rep.detokenize(toks))
            if smi:
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    canon_p[i] = Chem.MolToSmiles(m)
                    qed_p[i] = QED.qed(m); sa_p[i] = sascorer.calculateScore(m)
        except Exception: pass
    pass_mask = (qed_p > 0.6) & (sa_p < 4.0) & np.array([c is not None for c in canon_p])
    print(f'[{pool_name}] pass count: {int(pass_mask.sum())}/{POOL_} ({pass_mask.mean()*100:.2f}%)', flush=True)
    if pass_mask.sum() >= N:
        pass_idx = np.where(pass_mask)[0]
        seen = set(); uniq_pass = []
        for i in pass_idx:
            c = canon_p[i]
            if c and c not in seen:
                seen.add(c); uniq_pass.append(int(i))
        print(f'[{pool_name}] unique passing: {len(uniq_pass)}', flush=True)
        if len(uniq_pass) >= N:
            sel = uniq_pass[:N]
            s = score_subset(sel, token_seqs)
            out_results.append(dict(name=f'{pool_name}:oracle pass+dedup', **s))
            print(fmt(f'{pool_name}:oracle pass+dedup', s))
            # Top-N by QED among unique-pass
            uniq_pass_q = sorted(uniq_pass, key=lambda i: -qed_p[i])
            s = score_subset(uniq_pass_q[:N], token_seqs)
            out_results.append(dict(name=f'{pool_name}:oracle+dedup+QED-top', **s))
            print(fmt(f'{pool_name}:oracle+dedup+QED-top', s))
        else:
            print(f'[{pool_name}] NOT ENOUGH unique passing: need {N}', flush=True)
    return out_results

# ---- Pool A: plain AR ----
print('\n==== Pool A: plain AR drafts ====', flush=True)
t0 = time.time()
with torch.no_grad():
    seqs_A = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'  {POOL} drafts in {time.time()-t0:.1f}s', flush=True)
results_A = analyze_pool('AR', seqs_A)

# ---- Pool B: partial-QED guided AR ----
print('\n==== Pool B: partial-QED guided AR (qw=5, sw=1) ====', flush=True)
t0 = time.time()
seqs_B = partial_qed_guided_ar(n_samples=POOL, qw=5.0, sw=1.0, dev=dev)
print(f'  {POOL} drafts in {time.time()-t0:.1f}s', flush=True)
results_B = analyze_pool('Guided', seqs_B)

all_res = results_A + results_B
(out_dir / 'combo_results.json').write_text(json.dumps(all_res, indent=2))

print('\n==== HEAD-TO-HEAD (AR vs Guided) ====', flush=True)
by_strat = {}
for r in all_res:
    parts = r['name'].split(':', 1)
    pool = parts[0]; strat = parts[1]
    by_strat.setdefault(strat, {})[pool] = r
for strat, d in sorted(by_strat.items()):
    if 'AR' in d and 'Guided' in d:
        a = d['AR']; b = d['Guided']
        delta_gm = b['genmol'] - a['genmol']
        delta_q = b['quality'] - a['quality']
        print(f'  [{strat:<32s}] AR: GM={a["genmol"]:>6.2f} q={a["quality"]:>6.2f} u={a["unique"]:>6.2f} | '
              f'Guided: GM={b["genmol"]:>6.2f} q={b["quality"]:>6.2f} u={b["unique"]:>6.2f} | '
              f'ΔGM={delta_gm:+.2f}  Δq={delta_q:+.2f}', flush=True)

print(f'\n[combo] wrote {out_dir}/combo_results.json', flush=True)
PY
echo "[remote] end=$(date)"

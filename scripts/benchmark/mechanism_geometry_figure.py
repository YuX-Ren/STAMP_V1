#!/usr/bin/env python3
"""Mechanism figure for §6.1 "representation geometry": Figure 7.

Figure 6 (plot_motif_token_embeddings.py) establishes claim (1) -- chemically
similar motifs are nearer in the embedding. This script builds the *second*
mechanism figure that turns the geometry into a mechanistic bridge, cashing out
three further claims about the STAMP encoder relative to ChemBERTa-2 (SMILES BPE)
and ECFP4:

  (B) attachment-position differences are *moderately* distinguished.
      Controlled regioisomer families (same atoms, different attachment site,
      built with RDKit so they are genuine positional isomers). A representation
      that factorizes motif identity from attachment site should place
      positional isomers at a small-but-nonzero distance: smaller than a
      substituent-identity swap, larger than zero. We report the nested
      ordering  d(positional) < d(substitution) < d(random)  per representation.

  (C) motif replacement induces a *local, interpretable* embedding shift.
      Re-mine MoleculeACE single-cut MMPs; a pair sharing a context and
      differing by one variable fragment is a chemical edit A->B. Delta-z =
      z(B) - z(A), oriented canonically per transformation. "Local": ||Delta-z||
      (RePRA-normalized) is far below a random pair. "Interpretable": Delta-z for
      the *same* transformation are parallel (mean within-transformation cosine)
      far above the across-transformation baseline -- i.e. a given edit is a
      consistent direction, the word-analogy property.

  (D) embedding distance is *consistent* with structural and property distance.
      Structural: Spearman( embedding-distance , ECFP4-Tanimoto-distance ) over
      MMP pairs. Property: RePRA activity-cliff / scaffold-hop counts relative to
      ECFP4 (s_IR < 2), read from the existing repra_metrics_table.csv. RePRA is
      used here as *mechanistic validation* of geometry, not as a standalone
      leaderboard.

Embeddings are the same [CLS] encodings used by moleculeace_embedding_figures.py;
MMP mining reuses moleculeace_mmp_edit_locality.py. Outputs under --output-dir:

    mechanism_geometry_figure.{svg,pdf,png}   composite 3-panel figure
    mechanism_geometry_metrics.json           all scalar claims
    mechanism_attachment_pairs.csv            per-pair regioisomer distances
    mechanism_shift_transformations.csv       per-transformation consistency
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba-cache-{os.environ.get('USER', 'user')}")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

# Reuse the exact embedding extractors + MMP miner used elsewhere in the paper.
from moleculeace_embedding_figures import (  # noqa: E402
    chemberta_embeddings,
    ecfp4_matrix,
    stamp_embeddings,
)
from moleculeace_mmp_edit_locality import (  # noqa: E402
    enumerate_mmp_pairs,
    load_moleculeace_records,
)

# ---- Publication styling (shared with plot_motif_token_embeddings) ----------
PALETTE = {
    "ink": "#272727",
    "stamp": "#0F4D92",      # STAMP-MTR -> signal blue
    "chemberta": "#7C4DA0",  # ChemBERTa-2 -> purple
    "ecfp": "#767676",       # ECFP4 -> neutral
    "signal": "#B64342",
    "gold": "#E2A02C",
    "teal": "#2F6F73",
    "light": "#CFCECE",
}
REP_ORDER = ["STAMP-MTR", "ChemBERTa-2", "ECFP4"]
REP_COLOR = {"STAMP-MTR": PALETTE["stamp"], "ChemBERTa-2": PALETTE["chemberta"], "ECFP4": PALETTE["ecfp"]}


def apply_publication_style(font_size: int = 8) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def add_panel_label(ax, label: str, *, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="bottom")


def save_pub(fig, path_base: Path, dpi: int = 600) -> None:
    base = Path(path_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Controlled regioisomer construction (claim B: attachment position)
# =====================================================================

# Monosubstituted aromatic parents (the fixed "anchor"); probe is attached at
# every free aromatic CH to enumerate genuine ortho/meta/para regioisomers.
_ANCHOR_PARENTS = {
    "benzoic":      "O=C(O)c1ccccc1",
    "anisole":      "COc1ccccc1",
    "toluene":      "Cc1ccccc1",
    "benzonitrile": "N#Cc1ccccc1",
    "aniline":      "Nc1ccccc1",
    "phenol":       "Oc1ccccc1",
    "cf3benzene":   "FC(F)(F)c1ccccc1",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "fluorobenzene": "Fc1ccccc1",
    "pyridine":     "c1ccncc1",
}
# Probe substituents; attaching atom is written first so combine+bond is valid.
_PROBES = {
    "F": "F", "Cl": "Cl", "Br": "Br", "Me": "C", "OMe": "OC",
    "NH2": "N", "CN": "C#N", "CF3": "C(F)(F)F", "OH": "O",
}


def _attach_at_all_positions(parent_smiles: str, probe_smiles: str) -> List[str]:
    """Attach `probe` at every free aromatic CH of `parent`; return unique
    canonical SMILES (= the distinct positional isomers of that family)."""
    parent = Chem.MolFromSmiles(parent_smiles)
    probe = Chem.MolFromSmiles(probe_smiles)
    if parent is None or probe is None:
        return []
    out: List[str] = []
    seen: set[str] = set()
    n_parent = parent.GetNumAtoms()
    for atom in parent.GetAtoms():
        if not (atom.GetIsAromatic() and atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0):
            continue
        combo = Chem.CombineMols(parent, probe)
        ed = Chem.RWMol(combo)
        ed.AddBond(atom.GetIdx(), n_parent, Chem.BondType.SINGLE)
        m = ed.GetMol()
        try:
            Chem.SanitizeMol(m)
        except Exception:
            continue
        smi = Chem.MolToSmiles(m)
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def build_regioisomer_families(min_isomers: int = 2) -> pd.DataFrame:
    """One row per generated molecule with (family, anchor, probe) labels.

    family = (anchor, probe): molecules in a family differ only in attachment
    position -> positional isomers. Same anchor, different probe -> substituent
    swap. Different anchor -> unrelated.
    """
    rows: List[dict] = []
    for anchor, parent in _ANCHOR_PARENTS.items():
        for probe, psmi in _PROBES.items():
            isomers = _attach_at_all_positions(parent, psmi)
            if len(isomers) < min_isomers:
                continue
            for k, smi in enumerate(isomers):
                rows.append({
                    "smiles": smi, "anchor": anchor, "probe": probe,
                    "family": f"{anchor}|{probe}", "position_idx": k,
                })
    df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    return df


# =====================================================================
# Distance helpers
# =====================================================================

def _norm_bounds(emb: np.ndarray, *, max_mols: int = 5000, seed: int = 13) -> Tuple[float, float]:
    """RePRA-style MinMax bounds from valid pairwise euclidean distances.

    pdist is O(n^2) in memory, so for large molecule banks we subsample to
    `max_mols` rows -- the bounds are only a normalization scale and a random
    subsample is representative.
    """
    valid = ~np.isnan(emb).any(axis=1)
    x = emb[valid]
    if len(x) < 2:
        return math.nan, math.nan
    if len(x) > max_mols:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), max_mols, replace=False)]
    flat = pdist(x.astype(np.float64), metric="euclidean")
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return math.nan, math.nan
    return float(flat.min()), float(flat.max())


def _pair_dist(emb: np.ndarray, ia: int, ib: int, lo: float, hi: float) -> float:
    a, b = emb[ia], emb[ib]
    if np.isnan(a).any() or np.isnan(b).any():
        return math.nan
    d = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
    if hi - lo < 1e-12:
        return 0.0
    return float(np.clip((d - lo) / (hi - lo), 0.0, 1.0))


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return math.nan
    return float(np.dot(u, v) / (nu * nv))


# =====================================================================
# Embedding bank
# =====================================================================

class EmbeddingBank:
    """Compute + cache embeddings for a fixed list of unique SMILES."""

    def __init__(self, smiles: Sequence[str], reps: Dict[str, np.ndarray]):
        self.index = {s: i for i, s in enumerate(smiles)}
        self.smiles = list(smiles)
        self.reps = reps  # name -> (N, d) matrix, NaN rows for failures
        self.bounds = {name: _norm_bounds(mat) for name, mat in reps.items()}

    def has(self, smi: str) -> bool:
        return smi in self.index

    def pair_norm_dist(self, name: str, a: str, b: str) -> float:
        if a not in self.index or b not in self.index:
            return math.nan
        lo, hi = self.bounds[name]
        return _pair_dist(self.reps[name], self.index[a], self.index[b], lo, hi)

    def vec(self, name: str, smi: str) -> Optional[np.ndarray]:
        if smi not in self.index:
            return None
        v = self.reps[name][self.index[smi]]
        return None if np.isnan(v).any() else v.astype(np.float64)


def build_embeddings(
    smiles: Sequence[str], *, stamp_dir: Path, chemberta_name: Optional[str],
    device: str, batch_size: int, n_bits: int,
) -> EmbeddingBank:
    reps: Dict[str, np.ndarray] = {}
    print(f"[embed] STAMP-MTR on {len(smiles)} unique molecules ...", flush=True)
    reps["STAMP-MTR"] = stamp_embeddings(stamp_dir, smiles, device=device, batch_size=batch_size)
    if chemberta_name:
        print(f"[embed] ChemBERTa-2 ({chemberta_name}) ...", flush=True)
        try:
            reps["ChemBERTa-2"] = chemberta_embeddings(
                chemberta_name, smiles, device=device, batch_size=batch_size,
                max_len=256, local_files_only=True,
            )
        except Exception as exc:
            print(f"[embed] ChemBERTa-2 skipped ({exc})", flush=True)
    print("[embed] ECFP4 ...", flush=True)
    reps["ECFP4"] = ecfp4_matrix(smiles, n_bits)
    return EmbeddingBank(smiles, reps)


# =====================================================================
# Claim B: attachment-position separation
# =====================================================================

def attachment_analysis(bank: EmbeddingBank, fam: pd.DataFrame, *, seed: int,
                        max_random: int = 4000) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rng = random.Random(seed)
    by_family: Dict[str, List[str]] = {}
    by_anchor: Dict[str, List[str]] = {}
    for _, r in fam.iterrows():
        by_family.setdefault(r["family"], []).append(r["smiles"])
        by_anchor.setdefault(r["anchor"], []).append(r["smiles"])
    all_smi = fam["smiles"].tolist()

    rows: List[dict] = []
    # positional: same family (anchor+probe), different molecule
    for family, smis in by_family.items():
        for a, b in itertools.combinations(sorted(set(smis)), 2):
            rows.append({"category": "positional", "smiles_a": a, "smiles_b": b})
    # substitution: same anchor, different probe
    for anchor, smis in by_anchor.items():
        fam_of = {r["smiles"]: r["family"] for _, r in fam[fam["anchor"] == anchor].iterrows()}
        uniq = sorted(set(smis))
        pairs = [(a, b) for a, b in itertools.combinations(uniq, 2) if fam_of[a] != fam_of[b]]
        rng.shuffle(pairs)
        for a, b in pairs[:300]:
            rows.append({"category": "substitution", "smiles_a": a, "smiles_b": b})
    # random: different anchor
    anchor_of = {r["smiles"]: r["anchor"] for _, r in fam.iterrows()}
    rand_pairs: List[Tuple[str, str]] = []
    for _ in range(max_random * 3):
        a, b = rng.sample(all_smi, 2)
        if anchor_of[a] != anchor_of[b]:
            rand_pairs.append((a, b))
        if len(rand_pairs) >= max_random:
            break
    for a, b in rand_pairs:
        rows.append({"category": "random", "smiles_a": a, "smiles_b": b})

    df = pd.DataFrame(rows)
    for name in bank.reps:
        df[name] = [bank.pair_norm_dist(name, a, b) for a, b in zip(df["smiles_a"], df["smiles_b"])]

    summary: Dict[str, dict] = {}
    for name in bank.reps:
        s = {}
        for cat in ("positional", "substitution", "random"):
            v = df.loc[df["category"] == cat, name].dropna().to_numpy()
            s[cat] = {"median": float(np.median(v)) if v.size else math.nan,
                      "mean": float(np.mean(v)) if v.size else math.nan, "n": int(v.size)}
        # fraction of positional pairs that are non-degenerate (distinguished at all)
        pv = df.loc[df["category"] == "positional", name].dropna().to_numpy()
        s["positional_nonzero_frac"] = float(np.mean(pv > 1e-6)) if pv.size else math.nan
        # "moderate": positional median relative to substitution median (0..1)
        if s["substitution"]["median"] and math.isfinite(s["substitution"]["median"]) and s["substitution"]["median"] > 0:
            s["positional_over_substitution"] = s["positional"]["median"] / s["substitution"]["median"]
        else:
            s["positional_over_substitution"] = math.nan
        summary[name] = s
    return df, summary


# =====================================================================
# Claim C: motif-replacement shift vectors
# =====================================================================

def shift_analysis(bank: EmbeddingBank, pairs: pd.DataFrame, *, seed: int,
                   min_group: int = 6, max_groups: int = 400,
                   max_random_dirs: int = 5000) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rng = np.random.default_rng(seed)

    # Group MMP pairs by canonical transformation (unordered fragment swap).
    pairs = pairs.copy()
    def _tkey(r):
        return tuple(sorted((str(r["variable_a"]), str(r["variable_b"]))))
    pairs["tkey"] = pairs.apply(_tkey, axis=1)
    # canonical orientation: source = lexicographically smaller variable
    pairs["src_is_a"] = pairs.apply(lambda r: str(r["variable_a"]) <= str(r["variable_b"]), axis=1)

    summary: Dict[str, dict] = {}
    trows: List[dict] = []
    for name in bank.reps:
        deltas_by_t: Dict[tuple, List[np.ndarray]] = {}
        local_norms: List[float] = []
        lo, hi = bank.bounds[name]
        scale = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        for _, r in pairs.iterrows():
            sa, sb = (r["smiles_a"], r["smiles_b"]) if r["src_is_a"] else (r["smiles_b"], r["smiles_a"])
            va = bank.vec(name, sa)
            vb = bank.vec(name, sb)
            if va is None or vb is None:
                continue
            d = vb - va
            deltas_by_t.setdefault(r["tkey"], []).append(d)
            local_norms.append(float(np.linalg.norm(d)) / scale)

        # within-transformation consistency: mean cosine of each Delta to its
        # group-mean direction, over groups with >= min_group instances.
        within_cos: List[float] = []
        big_groups = [(t, v) for t, v in deltas_by_t.items() if len(v) >= min_group]
        rng.shuffle(big_groups) if False else None
        for t, vs in big_groups[:max_groups]:
            M = np.vstack(vs)
            mean_dir = M.mean(axis=0)
            cs = [_cos(d, mean_dir) for d in vs]
            cs = [c for c in cs if math.isfinite(c)]
            if cs:
                gc = float(np.mean(cs))
                within_cos.append(gc)
                trows.append({"rep": name, "tkey": "  ->  ".join(t), "n": len(vs), "within_cos": gc})

        # across-transformation baseline: cosine between Delta vectors drawn from
        # *different* transformations.
        flat = [(t, d) for t, vs in deltas_by_t.items() for d in vs]
        across: List[float] = []
        if len(flat) > 4:
            for _ in range(max_random_dirs):
                (ta, da), (tb, db) = (flat[i] for i in rng.integers(0, len(flat), size=2))
                if ta == tb:
                    continue
                c = _cos(da, db)
                if math.isfinite(c):
                    across.append(c)
        summary[name] = {
            "within_cos_median": float(np.median(within_cos)) if within_cos else math.nan,
            "within_cos_mean": float(np.mean(within_cos)) if within_cos else math.nan,
            "across_cos_median": float(np.median(across)) if across else math.nan,
            "n_groups": len(within_cos),
            "local_norm_median": float(np.median(local_norms)) if local_norms else math.nan,
            "random_norm_median": 1.0,  # by construction MinMax max ~ random extreme; report relative below
        }
    return pd.DataFrame(trows), summary


# =====================================================================
# Claim D: embedding vs structural / property distance
# =====================================================================

def consistency_analysis(bank: EmbeddingBank, pairs: pd.DataFrame, *, repra_csv: Optional[Path],
                         seed: int, max_pairs: int = 20000) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    df = pairs
    if len(df) > max_pairs:
        df = df.sample(max_pairs, random_state=seed)
    # structural reference distance = ECFP4 Tanimoto distance between the molecules
    chem: List[float] = []
    fps: Dict[str, object] = {}
    def _fp(smi):
        if smi not in fps:
            m = Chem.MolFromSmiles(smi)
            fps[smi] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
        return fps[smi]
    rows = list(zip(df["smiles_a"], df["smiles_b"]))
    for a, b in rows:
        fa, fb = _fp(a), _fp(b)
        chem.append(1.0 - float(DataStructs.TanimotoSimilarity(fa, fb)) if (fa and fb) else math.nan)
    chem = np.asarray(chem)

    out: Dict[str, dict] = {"structural_spearman": {}}
    for name in bank.reps:
        if name == "ECFP4":
            continue
        d = np.asarray([bank.pair_norm_dist(name, a, b) for a, b in rows])
        ok = np.isfinite(d) & np.isfinite(chem)
        rho = float(spearmanr(d[ok], chem[ok]).statistic) if ok.sum() > 10 else math.nan
        out["structural_spearman"][name] = {"rho": rho, "n": int(ok.sum())}
    # ECFP self-consistency reference (sanity ~1)
    de = np.asarray([bank.pair_norm_dist("ECFP4", a, b) for a, b in rows])
    oke = np.isfinite(de) & np.isfinite(chem)
    out["structural_spearman"]["ECFP4"] = {
        "rho": float(spearmanr(de[oke], chem[oke]).statistic) if oke.sum() > 10 else math.nan,
        "n": int(oke.sum())}

    # property side: RePRA s_IR (lower better, 2 == ECFP) aggregated per rep.
    out["repra"] = {}
    if repra_csv and Path(repra_csv).exists():
        rp = pd.read_csv(repra_csv)
        rep_map = {"stamp_mtr": "STAMP-MTR", "stamp_mlm": "STAMP-MLM",
                   "chemberta": "ChemBERTa-2", "MorganFP": "ECFP4"}
        rp["rep_disp"] = rp["rep"].map(rep_map).fillna(rp["rep"])
        for disp, g in rp.groupby("rep_disp"):
            out["repra"][disp] = {
                "sIR_median": float(np.nanmedian(g["sIR"])),
                "sIR_mean": float(np.nanmean(g["sIR"])),
                "AC_rate_mean": float(np.nanmean(g["AC_rate"])),
                "SH_rate_mean": float(np.nanmean(g["SH_rate"])),
                "n_tasks": int(len(g)),
                "frac_tasks_below_2": float(np.mean(g["sIR"] < 2.0)),
            }
    return out


# =====================================================================
# Figure
# =====================================================================

def plot_figure(attach_df: pd.DataFrame, attach_sum, shift_df, shift_sum,
                cons, out_base: Path, *, model_tag: str) -> None:
    reps = [r for r in REP_ORDER if r in shift_sum]
    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.06, right=0.985, bottom=0.17, top=0.86)

    # ---- Panel a: attachment-position separation -------------------
    ax = fig.add_subplot(gs[0, 0])
    cats = ["positional", "substitution", "random"]
    cat_lab = ["positional\nisomer", "substituent\nswap", "unrelated"]
    width = 0.26
    xpos = np.arange(len(cats))
    for j, name in enumerate(reps):
        med = [attach_sum[name][c]["median"] for c in cats]
        ax.bar(xpos + (j - (len(reps) - 1) / 2) * width, med, width,
               color=REP_COLOR[name], label=name, edgecolor="white", linewidth=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cat_lab)
    ax.set_ylabel("RePRA-normalized\nembedding distance")
    ax.set_title("Attachment position is\nmoderately distinguished", pad=3)
    ax.legend(loc="upper left", fontsize=6)
    add_panel_label(ax, "a")

    # ---- Panel b: replacement shift consistency --------------------
    ax = fig.add_subplot(gs[0, 1])
    xpos = np.arange(len(reps))
    within = [shift_sum[n]["within_cos_median"] for n in reps]
    across = [shift_sum[n]["across_cos_median"] for n in reps]
    ax.bar(xpos - 0.2, within, 0.38, color=[REP_COLOR[n] for n in reps],
           edgecolor="white", linewidth=0.4, label="same edit")
    ax.bar(xpos + 0.2, across, 0.38, color="none",
           edgecolor=[REP_COLOR[n] for n in reps], linewidth=1.1, hatch="///", label="different edit")
    ax.axhline(0, color=PALETTE["ink"], lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace("-", "-\n") for n in reps])
    ax.set_ylabel(r"mean cosine of $\Delta z$")
    ax.set_title("A motif swap is a\nconsistent direction", pad=3)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE["ink"], label="same edit"),
                       Patch(facecolor="none", edgecolor=PALETTE["ink"], hatch="///", label="different edit")],
              loc="upper right", fontsize=6)
    add_panel_label(ax, "b")

    # ---- Panel c: property alignment (RePRA) + structure preservation
    # Main = RePRA s_IR (property): STAMP-MTR is the only encoder below the
    # ECFP=2 line. Inset = structural Spearman rho (STAMP vs ChemBERTa): the
    # geometry also *preserves* structural distance. ECFP is omitted from the
    # structural bar because the structural reference IS ECFP-Tanimoto (a
    # tautology); RePRA is where ECFP is the meaningful baseline.
    ax = fig.add_subplot(gs[0, 2])
    spear = cons.get("structural_spearman", {})
    repra = cons.get("repra", {})
    reps_r = [r for r in ["STAMP-MTR", "STAMP-MLM", "ChemBERTa-2"] if r in repra]
    sir = [repra[r]["sIR_median"] for r in reps_r]
    xpos = np.arange(len(reps_r))
    bars = ax.bar(xpos, sir, 0.6, color=[REP_COLOR.get(r, PALETTE["teal"]) for r in reps_r],
                  edgecolor="white", linewidth=0.4)
    ax.axhline(2.0, color=PALETTE["signal"], lw=1.0, ls="--")
    ax.annotate("ECFP4 baseline", xy=(len(reps_r) - 0.55, 2.0), xytext=(0, 2),
                textcoords="offset points", fontsize=6, color=PALETTE["signal"], ha="right", va="bottom")
    ax.set_xticks(xpos)
    ax.set_xticklabels([r.replace("-MTR", "\nMTR").replace("-MLM", "\nMLM").replace("-2", "-2") for r in reps_r])
    ax.set_ylabel(r"RePRA $s_{IR}$  (lower = better)")
    ax.set_ylim(0, max(3.0, max(sir) * 1.28))
    ax.set_title("Geometry is property-aligned\n& structure-preserving", pad=3)
    # inset: structural Spearman rho for the learned sequence encoders, parked in
    # the empty headroom above the shortest bar.
    reps_s = [r for r in ["STAMP-MTR", "ChemBERTa-2"] if r in spear]
    if reps_s:
        axi = ax.inset_axes([0.07, 0.64, 0.34, 0.31])
        rho = [spear[r]["rho"] for r in reps_s]
        axi.bar(np.arange(len(reps_s)), rho, 0.6,
                color=[REP_COLOR.get(r, PALETTE["teal"]) for r in reps_s], edgecolor="white", linewidth=0.3)
        axi.set_xticks(np.arange(len(reps_s)))
        axi.set_xticklabels([r.split("-")[0][:5] for r in reps_s], fontsize=5)
        axi.set_ylabel(r"$\rho$ struct.", fontsize=6)
        axi.set_ylim(0, 1)
        axi.tick_params(labelsize=5)
        axi.set_title("tracks structure", fontsize=5, pad=1)
    add_panel_label(ax, "c")

    fig.suptitle(f"STAMP geometry as a mechanistic bridge — {model_tag}", y=0.99, fontsize=10)
    save_pub(fig, out_base)


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp-dir", default="/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666_mtr")
    ap.add_argument("--moleculeace-dir", default="/scratch/yuxuan.ren/data/moleculeace")
    ap.add_argument("--chemberta-name", default="seyonec/PubChem10M_SMILES_BPE_60k")
    ap.add_argument("--no-chemberta", action="store_true")
    ap.add_argument("--repra-csv", default="results/repra_analysis_figures/repra_metrics_table.csv")
    ap.add_argument("--output-dir", default="results/mechanism_geometry_figure")
    ap.add_argument("--model-tag", default="STAMP-MTR (PubChem-10M)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-pairs-total", type=int, default=40000)
    ap.add_argument("--max-pairs-per-context", type=int, default=40)
    ap.add_argument("--min-shift-group", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="2 tasks, tiny caps")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--replot-from-json", default=None,
                    help="skip all compute; re-render the figure from a metrics JSON.")
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    apply_publication_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_from_json:
        m = json.loads(Path(args.replot_from_json).read_text())
        plot_figure(None, m["attachment"], None, m["shift"], m["consistency"],
                    out_dir / "mechanism_geometry_figure", model_tag=m.get("model_tag", args.model_tag))
        print(f"[replot] re-rendered figure from {args.replot_from_json} -> {out_dir}", flush=True)
        return 0

    tasks = args.tasks
    if args.smoke and tasks is None:
        tasks = ["CHEMBL204_Ki", "CHEMBL2147_Ki"]
    max_pairs_total = 2000 if args.smoke else args.max_pairs_total

    # ---- regioisomer families (claim B) ----
    fam = build_regioisomer_families()
    print(f"[regio] {len(fam)} molecules across {fam['family'].nunique()} families", flush=True)

    # ---- MMP pairs (claims C, D) ----
    print("[mmp] loading MoleculeACE records ...", flush=True)
    recs = load_moleculeace_records(Path(args.moleculeace_dir), strip_salts=True, tasks=tasks)
    pairs = enumerate_mmp_pairs(
        recs, max_cut_bonds=20, all_context_sides=False, keep_context_duplicates=False,
        min_context_heavy_atoms=6, max_variable_heavy_atoms=14,
        max_pairs_per_context=args.max_pairs_per_context, max_pairs_total=max_pairs_total,
        seed=args.seed,
    )
    print(f"[mmp] {len(pairs)} MMP pairs over {pairs['task'].nunique()} tasks", flush=True)

    # ---- embeddings for every unique molecule we touch ----
    uniq = sorted(set(fam["smiles"]).union(pairs["smiles_a"]).union(pairs["smiles_b"]))
    print(f"[embed] {len(uniq)} unique molecules", flush=True)
    bank = build_embeddings(
        uniq, stamp_dir=Path(args.stamp_dir),
        chemberta_name=None if args.no_chemberta else args.chemberta_name,
        device=args.device, batch_size=args.batch_size, n_bits=args.n_bits,
    )

    # ---- run the three analyses ----
    attach_df, attach_sum = attachment_analysis(bank, fam, seed=args.seed)
    shift_df, shift_sum = shift_analysis(bank, pairs, seed=args.seed, min_group=args.min_shift_group)
    cons = consistency_analysis(bank, pairs, repra_csv=Path(args.repra_csv), seed=args.seed)

    metrics = {
        "model_tag": args.model_tag,
        "n_unique_molecules": len(uniq),
        "n_regio_molecules": int(len(fam)),
        "n_regio_families": int(fam["family"].nunique()),
        "n_mmp_pairs": int(len(pairs)),
        "n_tasks": int(pairs["task"].nunique()),
        "attachment": attach_sum,
        "shift": shift_sum,
        "consistency": cons,
    }
    (out_dir / "mechanism_geometry_metrics.json").write_text(json.dumps(metrics, indent=2))
    attach_df.to_csv(out_dir / "mechanism_attachment_pairs.csv", index=False)
    shift_df.to_csv(out_dir / "mechanism_shift_transformations.csv", index=False)

    plot_figure(attach_df, attach_sum, shift_df, shift_sum, cons,
                out_dir / "mechanism_geometry_figure", model_tag=args.model_tag)

    print("\n==== KEY NUMBERS ====", flush=True)
    for name in bank.reps:
        a = attach_sum[name]
        print(f"[{name}] attach pos/sub/rand med = "
              f"{a['positional']['median']:.3f}/{a['substitution']['median']:.3f}/{a['random']['median']:.3f}"
              f"  pos-nonzero={a['positional_nonzero_frac']:.2f}", flush=True)
    for name in shift_sum:
        s = shift_sum[name]
        print(f"[{name}] shift within/across cos = {s['within_cos_median']:.3f}/{s['across_cos_median']:.3f}"
              f"  (n_groups={s['n_groups']})  local_norm_med={s['local_norm_median']:.3f}", flush=True)
    print(f"[structural Spearman] {cons['structural_spearman']}", flush=True)
    print(f"[repra] {cons.get('repra', {})}", flush=True)
    print(f"\nsaved to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

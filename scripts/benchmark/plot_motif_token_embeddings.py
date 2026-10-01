#!/usr/bin/env python3
"""Visualize the STAMP **motif embedding space** of a pretrained encoder.

Two embedding sources are supported:

    contextual  (default)  encode ``[bos, motif, eos]`` and take the encoder's
                           hidden state at the motif position. This is the
                           chemically meaningful object.
    lookup                 the raw ``tok_emb.weight`` row (input embedding
                           table). NOTE: this table sits on a near-uniform norm
                           shell with only ~1.7x chemical-neighbor enrichment,
                           so its 2D projection looks like a featureless blob.

Why contextual: on STAMP-MTR the contextual motif embeddings give ~4.9x
kNN-Tanimoto enrichment over a random baseline (vs ~1.7x for the lookup table),
i.e. embedding nearest-neighbors are genuinely similar fragments.

Outputs (under --output-dir):
    motif_embedding_by_property.png      motif UMAP, 4 chemistry panels
    motif_knn_tanimoto_enrichment.png    are embedding-neighbors similar? curve
    motif_neighbor_examples.png          query motifs + nearest neighbors (mols)
    motif_token_embedding_table.csv      token, freq, props, 2D coords
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba-cache-{os.environ.get('USER', 'user')}")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

# --- Publication styling (nature-figure skill) -----------------------------
# Restrained palette: one neutral, one signal, one accent family.
PALETTE = {
    "ink": "#272727",
    "teal": "#2F6F73",
    "blue": "#0F4D92",
    "signal": "#B64342",
    "gold": "#E2A02C",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}
# Sequential colormaps reused across panels so the figure reads as one family.
CMAP_SEQ = "viridis"


def apply_publication_style(font_size: int = 8) -> None:
    """Nature-style rcParams with editable vector text. Call once before plotting."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",   # keep text as <text> nodes in SVG
        "pdf.fonttype": 42,       # editable TrueType in PDF
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size + 2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def save_pub(fig, path_base: Path, dpi: int = 600) -> None:
    """Export editable SVG + PDF + raster PNG (nature-figure export contract)."""
    base = Path(path_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label: str, *, x: float = -0.10, y: float = 1.06) -> None:
    """Nature-style bold panel letter at the top-left of an axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="bottom")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.roberta_property import load_pretrained_roberta  # noqa: E402

_JOINT_RE = re.compile(r"^\[J_\d+\]$")
_BRANCH_RE = re.compile(r"^\[B_\d+\]$")
_SCHEMA_TOKENS = {"[Ring]", "[S_S]", "[S_R]", "[S_E]", "[S_Z]"}

# Curated, recognizable query fragments for the neighbor-example grid; any that
# are absent from the vocab are silently skipped and topped up by frequency.
_QUERY_MOTIFS = [
    "C1=CC=CC=C1",     # benzene
    "C1=CC=NC=C1",     # pyridine
    "C1CCCCC1",        # cyclohexane
    "O=CO",            # carboxylic acid
    "NC=O",            # amide
    "CCO",             # ethanol-ish
    "FC1=CC=CC=C1",    # fluorobenzene
    "O=S(=O)O",        # sulfonyl
]


def classify_token(tok: str) -> str:
    if tok.startswith("<") and tok.endswith(">"):
        return "special"
    if _JOINT_RE.match(tok):
        return "joint"
    if _BRANCH_RE.match(tok):
        return "branch"
    if tok in _SCHEMA_TOKENS:
        return "schema"
    if Chem.MolFromSmiles(tok) is not None:
        return "motif"
    return "other"


def motif_properties(mol: Chem.Mol) -> Dict[str, float]:
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))
    return {
        "n_heavy": mol.GetNumHeavyAtoms(),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_hetero": n_hetero,
        "mol_weight": float(Descriptors.MolWt(mol)),
    }


def load_freqs(ckpt_dir: Path) -> Dict[str, int]:
    """Read motif training frequencies from the vocab file recorded in the ckpt.

    The pretrained ckpt stores ``stamp_vocab`` as a path to a TSV
    (smiles \\t n_atoms \\t count). Returns {} if unavailable.
    """
    ck = torch.load(ckpt_dir / "pretrained_roberta.pt", map_location="cpu", weights_only=False)
    sv = ck.get("stamp_vocab")
    if not isinstance(sv, str) or not Path(sv).exists():
        return {}
    freq: Dict[str, int] = {}
    for ln in Path(sv).read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 3 and parts[2].isdigit():
            freq[parts[0]] = int(parts[2])
    return freq


def contextual_motif_embeddings(model, vocab, token_ids: Sequence[int], *, device: str, batch_size: int) -> np.ndarray:
    dev = torch.device(device)
    model.to(dev).eval()
    d = model.tok_emb.embedding_dim
    out = np.zeros((len(token_ids), d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(token_ids), batch_size):
            chunk = list(token_ids[s : s + batch_size])
            x = torch.tensor(
                [[vocab.bos_id, c, vocab.eos_id] for c in chunk], dtype=torch.long, device=dev
            )
            h = model(x)  # (b, 3, d); pos 1 == motif token
            out[s : s + len(chunk)] = h[:, 1, :].float().cpu().numpy()
    return out


def _preprocess(emb: np.ndarray, *, center: bool, pca_pre: int, seed: int) -> np.ndarray:
    """Mean-center + optional PCA-whiten to spread a collapsed embedding.

    Contextual embeddings carry a strong shared direction; removing it (center)
    and denoising to a few whitened PCs stops UMAP from piling everything into
    one central mass.
    """
    x = emb.astype(np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    if pca_pre and pca_pre < min(x.shape):
        x = PCA(n_components=pca_pre, whiten=True, random_state=seed).fit_transform(x)
    return x


def _despread(xy: np.ndarray) -> np.ndarray:
    """Rank-transform each axis to a uniform marginal, so a dense core can't be
    squeezed into a corner by a handful of far outliers. Preserves neighbor
    ordering (monotonic per axis) while filling the canvas evenly."""
    out = np.empty_like(xy, dtype=np.float32)
    n = len(xy)
    for c in range(xy.shape[1]):
        ranks = np.argsort(np.argsort(xy[:, c]))
        out[:, c] = (ranks / max(n - 1, 1)).astype(np.float32)
    return out


def fit_2d(
    emb: np.ndarray, *, method: str, seed: int, n_neighbors: int,
    min_dist: float = 0.15, spread: float = 1.0, metric: str = "cosine",
    center: bool = False, pca_pre: int = 0, uniformize: bool = False,
) -> Tuple[np.ndarray, str]:
    x = _preprocess(emb, center=center, pca_pre=pca_pre, seed=seed)
    # After whitening, cosine is meaningless; use euclidean.
    eff_metric = "euclidean" if pca_pre else metric
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, max(2, len(x) - 1)),
                min_dist=min_dist,
                spread=max(spread, min_dist),
                metric=eff_metric,
                random_state=seed,
            )
            xy = reducer.fit_transform(x).astype(np.float32)
            return (_despread(xy) if uniformize else xy), "umap"
        except Exception as exc:  # pragma: no cover
            print(f"[warn] UMAP failed ({exc}); falling back to PCA", flush=True)
    x2 = StandardScaler().fit_transform(x)
    xy = PCA(n_components=2, random_state=seed).fit_transform(x2).astype(np.float32)
    return (_despread(xy) if uniformize else xy), "pca"


def knn_tanimoto_curve(Z: np.ndarray, fps: List, *, ks: Sequence[int], n_query: int, seed: int) -> Dict[int, Tuple[float, float]]:
    """Mean ECFP4 Tanimoto of the top-k embedding (cosine) neighbors vs random."""
    rng = np.random.default_rng(seed)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    n = len(Z)
    idx = np.arange(n)
    q = idx if n <= n_query else rng.choice(idx, n_query, replace=False)
    kmax = max(ks)
    nn_sims = {k: [] for k in ks}
    rand_sims: List[float] = []
    for qi in q:
        d = Zn[qi] @ Zn.T
        order = np.argsort(-d)
        nn = [j for j in order if j != qi][:kmax]
        for k in ks:
            for j in nn[:k]:
                nn_sims[k].append(DataStructs.TanimotoSimilarity(fps[qi], fps[j]))
        for j in rng.choice(idx, kmax, replace=False):
            rand_sims.append(DataStructs.TanimotoSimilarity(fps[qi], fps[j]))
    rand_mean = float(np.mean(rand_sims))
    return {k: (float(np.mean(v)), rand_mean) for k, v in nn_sims.items()}


def plot_motif_panels(motif: pd.DataFrame, out_path: Path, *, model_tag: str, enrich: float) -> None:
    panels = [
        ("n_aromatic_rings", "aromatic-ring count", "cividis"),
        ("n_rings", "ring count", "plasma"),
        ("n_hetero", "heteroatom count", "magma"),
        ("n_heavy", "heavy-atom count", "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.6))
    for ax, (col, label, cmap) in zip(axes.ravel(), panels):
        vals = motif[col].to_numpy(dtype=float)
        vmax = max(float(np.nanpercentile(vals, 98)), 1.0) if len(vals) else 1.0
        sc = ax.scatter(
            motif["x"], motif["y"], c=vals, s=12, cmap=cmap, alpha=0.85,
            edgecolors="none", vmin=float(np.nanmin(vals)) if len(vals) else 0.0, vmax=vmax,
        )
        ax.set_title(f"motif tokens by {label}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.82, label=label)
    fig.suptitle(
        f"STAMP contextual motif-embedding space — {model_tag}\n"
        f"n={len(motif)} motifs · kNN-Tanimoto enrichment = {enrich:.2f}x vs random",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_enrichment(curve: Dict[int, Tuple[float, float]], out_path: Path, *, model_tag: str, mode: str) -> None:
    ks = sorted(curve)
    nn = [curve[k][0] for k in ks]
    rand = curve[ks[0]][1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(ks, nn, marker="o", color="#2f6f73", label="embedding kNN")
    ax.axhline(rand, color="#b84a62", linestyle="--", label=f"random baseline ({rand:.3f})")
    ax.set_xlabel("number of nearest embedding neighbors k")
    ax.set_ylabel("mean ECFP4 Tanimoto to neighbors")
    ax.set_title(f"Do similar motifs cluster? — {model_tag} ({mode} embedding)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def ecfp_nn_edges(fps: List, *, k: int, min_tanimoto: float) -> List[Tuple[int, int, float]]:
    """Top-k ECFP4 nearest-neighbor edges (i<j, Tanimoto >= threshold)."""
    edges: Dict[Tuple[int, int], float] = {}
    n = len(fps)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1.0
        for j in np.argsort(-sims)[:k]:
            t = float(sims[j])
            if t < min_tanimoto:
                break
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges[(a, b)] = t
    return [(a, b, t) for (a, b), t in edges.items()]


def plot_umap_ecfp_edges(
    df: pd.DataFrame, edges: List[Tuple[int, int, float]], out_path: Path, *, model_tag: str, min_tanimoto: float
) -> None:
    """Embedding UMAP with ECFP nearest-neighbor edges overlaid.

    Short edges that stay inside clusters ⇒ the embedding layout agrees with
    ECFP chemical distance; long edges crossing the plot ⇒ disagreement.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    import matplotlib as mpl

    xy = df[["x", "y"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#b9c0c2", alpha=0.6, edgecolors="none", zorder=1)
    if edges:
        segs = np.array([[xy[a], xy[b]] for a, b, _ in edges])
        tani = np.array([t for _, _, t in edges])
        lengths = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1)
        med_len = float(np.median(lengths))
        # Per-edge alpha: short (chemistry-agreeing) edges opaque, long edges fade.
        p90 = float(np.percentile(lengths, 90)) or 1.0
        alpha = np.clip(0.75 * (1.0 - lengths / p90), 0.04, 0.75)
        norm = Normalize(vmin=min_tanimoto, vmax=1.0)
        rgba = mpl.colormaps["inferno"](norm(tani))
        rgba[:, 3] = alpha
        order = np.argsort(-lengths)  # draw long (faint) first, short (bold) on top
        lc = LineCollection(segs[order], colors=rgba[order], linewidths=0.7, zorder=2)
        ax.add_collection(lc)
        sm = ScalarMappable(cmap="inferno", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.82, label="ECFP4 Tanimoto of edge")
    else:
        med_len = float("nan")
    ax.set_title(
        f"Embedding UMAP with ECFP nearest-neighbor edges — {model_tag}\n"
        f"{len(edges)} edges (Tanimoto ≥ {min_tanimoto:g}); median edge length = {med_len:.2f}"
    )
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_umap_ecfp_anchors(
    df: pd.DataFrame, fps: List, anchors: List[int], smis: List[str], out_path: Path, *, model_tag: str
) -> None:
    """Color the embedding UMAP by ECFP Tanimoto to each anchor motif.

    If the bright (chemically near) points form a contiguous region, the
    embedding locally preserves ECFP distance.
    """
    xy = df[["x", "y"]].to_numpy(dtype=float)
    n = len(anchors)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, ai in zip(axes.ravel(), anchors):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[ai], fps))
        order = np.argsort(sims)  # draw far (dim) first, near (bright) on top
        # Contrast: saturate at the 99.5th-pct Tanimoto so the near-island pops;
        # size grows with similarity so chemically-near motifs are emphasized.
        vmax = max(float(np.percentile(sims, 99.5)), 0.3)
        sizes = 8 + 60 * np.clip(sims / max(vmax, 1e-9), 0, 1) ** 2
        ax.axis("on")
        sc = ax.scatter(xy[order, 0], xy[order, 1], c=sims[order], s=sizes[order], cmap="viridis",
                        vmin=0.0, vmax=vmax, edgecolors="none")
        ax.scatter([xy[ai, 0]], [xy[ai, 1]], s=240, marker="*", c="#d11f4a", edgecolors="white", linewidths=1.0, zorder=3)
        ax.set_title(f"ECFP Tanimoto to  {smis[ai][:22]}  (★)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.85, label=f"Tanimoto (vmax={vmax:.2f})")
    fig.suptitle(f"ECFP chemical distance over the embedding UMAP — {model_tag}", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_neighbor_examples(
    Z: np.ndarray, smis: List[str], freqs: np.ndarray, queries: List[int], out_path: Path, *, k: int
) -> None:
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    mols, legends = [], []
    for qi in queries:
        d = Zn[qi] @ Zn.T
        order = [j for j in np.argsort(-d) if j != qi][:k]
        row = [qi] + order
        for col, j in enumerate(row):
            m = Chem.MolFromSmiles(smis[j])
            mols.append(m)
            tag = "QUERY" if col == 0 else f"sim={DataStructs.TanimotoSimilarity(_fp(smis[qi]), _fp(smis[j])):.2f}"
            legends.append(f"{tag}\n{smis[j][:18]}")
    img = Draw.MolsToGridImage(mols, molsPerRow=k + 1, subImgSize=(150, 130), legends=legends)
    img.save(out_path)


def _fp(smi: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048)


def _draw_edges(ax, xy: np.ndarray, edges: List[Tuple[int, int, float]], min_tanimoto: float):
    """Overlay ECFP nearest-neighbor edges; long edges fade so agreement reads clearly."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if not edges:
        return None
    segs = np.array([[xy[a], xy[b]] for a, b, _ in edges])
    tani = np.array([t for _, _, t in edges])
    lengths = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1)
    p90 = float(np.percentile(lengths, 90)) or 1.0
    alpha = np.clip(0.75 * (1.0 - lengths / p90), 0.04, 0.75)
    norm = Normalize(vmin=min_tanimoto, vmax=1.0)
    rgba = mpl.colormaps["inferno"](norm(tani))
    rgba[:, 3] = alpha
    order = np.argsort(-lengths)
    lc = LineCollection(segs[order], colors=rgba[order], linewidths=0.6, zorder=2)
    ax.add_collection(lc)
    return norm


def plot_nature_main_figure(
    df: pd.DataFrame, curve: Dict[int, Tuple[float, float]], edges: List[Tuple[int, int, float]],
    Z: np.ndarray, fps: List, motif_smi: List[str], queries: List[int],
    out_base: Path, *, model_tag: str, enrich: float, min_tanimoto: float,
    repra_csv: Optional[Path] = None, suptitle: bool = True,
) -> None:
    """Composite publication figure on embedding-space geometry only:
    (a) hero UMAP by aromaticity, (b) kNN-Tanimoto enrichment, (c) ECFP-edge
    agreement, (d) nearest-neighbor molecule grid.

    Encodes the core claim: contextual motif embeddings are organized by chemistry.
    The downstream RePRA s_IR metric (repra_csv) is intentionally NOT included here
    — it is a property-prediction result, not an embedding-space property, and
    lives in its own table/figure.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec

    xy = df[["x", "y"]].to_numpy(dtype=float)
    fig = plt.figure(figsize=(10.5, 7.2))  # full journal width
    # 6-col grid: row0 = a|b (half each); row1 = c (1/3) | d (2/3).
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.05], hspace=0.40, wspace=0.55)

    # (a) Hero: UMAP colored by aromatic-ring count.
    ax_a = fig.add_subplot(gs[0, 0:3])
    arom = df["n_aromatic_rings"].to_numpy(dtype=float)
    sc = ax_a.scatter(xy[:, 0], xy[:, 1], c=arom, s=5, cmap=CMAP_SEQ, alpha=0.85,
                      vmin=0, vmax=2, edgecolors="none")
    ax_a.set_title("Motif embedding (UMAP)", pad=4)
    ax_a.set_xlabel("UMAP-1"); ax_a.set_ylabel("UMAP-2")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    cb = fig.colorbar(sc, ax=ax_a, shrink=0.82, ticks=[0, 1, 2], pad=0.02)
    cb.set_label("aromatic rings", fontsize=7)
    add_panel_label(ax_a, "a")

    # (b) kNN-Tanimoto enrichment curve.
    ax_b = fig.add_subplot(gs[0, 3:6])
    ks = sorted(curve)
    nn = [curve[k][0] for k in ks]
    rand = curve[ks[0]][1]
    ax_b.plot(ks, nn, marker="o", ms=4, lw=1.6, color=PALETTE["teal"], label="embedding kNN")
    ax_b.axhline(rand, ls="--", lw=1.1, color=PALETTE["signal"], label=f"random ({rand:.3f})")
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("nearest neighbors, k"); ax_b.set_ylabel("mean ECFP4 Tanimoto")
    ax_b.set_title("Neighbors are chemically similar", pad=4)
    ax_b.grid(axis="y", alpha=0.2)
    ax_b.legend(loc="upper right")
    ax_b.annotate(f"{enrich:.1f}× over chance", xy=(ks[len(ks)//2], nn[len(ks)//2]),
                  xytext=(0.42, 0.55), textcoords="axes fraction", fontsize=7,
                  color=PALETTE["ink"])
    add_panel_label(ax_b, "b")

    # (c) ECFP nearest-neighbor edges over the UMAP (bottom-left).
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_c.scatter(xy[:, 0], xy[:, 1], s=4, c=PALETTE["neutral_light"], alpha=0.6, edgecolors="none", zorder=1)
    norm = _draw_edges(ax_c, xy, edges, min_tanimoto)
    if norm is not None:
        sm = ScalarMappable(cmap="inferno", norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_c, shrink=0.82, pad=0.02)
        cb.set_label("ECFP4 Tanimoto", fontsize=7)
    ax_c.set_title("Layout tracks ECFP distance", pad=4)
    ax_c.set_xlabel("UMAP-1"); ax_c.set_ylabel("UMAP-2")
    ax_c.set_xticks([]); ax_c.set_yticks([])
    add_panel_label(ax_c, "c")

    # (d) Nearest-neighbor molecule grid for a few query motifs (bottom-right, wide).
    ax_d = fig.add_subplot(gs[1, 2:6])
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    k = 5
    qsel = queries[:5]
    mols, legends = [], []
    for qi in qsel:
        d = Zn[qi] @ Zn.T
        order = [j for j in np.argsort(-d) if j != qi][:k]
        for col, j in enumerate([qi] + order):
            mols.append(Chem.MolFromSmiles(motif_smi[j]))
            legends.append("query" if col == 0 else
                           f"{DataStructs.TanimotoSimilarity(fps[qi], fps[j]):.2f}")
    grid = Draw.MolsToGridImage(mols, molsPerRow=k + 1, subImgSize=(150, 120), legends=legends)
    ax_d.imshow(np.asarray(grid))
    ax_d.set_title("Embedding neighbors of query motifs", pad=4)
    ax_d.axis("off")
    add_panel_label(ax_d, "d", x=-0.02)

    if suptitle:
        fig.suptitle(
            f"STAMP contextual motif embeddings are organized by chemistry — {model_tag}",
            y=0.995, fontsize=10,
        )
    save_pub(fig, out_base)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-dir", default="/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666_mtr")
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--output-dir", default="results/motif_token_embedding_figures")
    ap.add_argument("--embedding-mode", choices=["contextual", "lookup"], default="contextual")
    ap.add_argument("--projection", choices=["umap", "pca"], default="umap")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.5,
                    help="UMAP min_dist; larger spreads points apart (anti-collapse).")
    ap.add_argument("--spread", type=float, default=1.0,
                    help="UMAP spread; scales the whole layout (larger = less cramped).")
    ap.add_argument("--no-uniformize", dest="uniformize", action="store_false",
                    help="disable the rank-transform that fills the canvas evenly (on by default).")
    ap.set_defaults(uniformize=True)
    ap.add_argument("--no-center", action="store_true",
                    help="disable mean-centering before projection.")
    ap.add_argument("--pca-pre", type=int, default=50,
                    help="PCA-whiten to this many dims before UMAP (0=off). Decollapses dense cores.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--knn-queries", type=int, default=600, help="random motifs sampled for the enrichment curve.")
    ap.add_argument("--neighbor-k", type=int, default=6)
    ap.add_argument("--repra-csv", default="results/repra_analysis_figures/repra_metrics_table.csv",
                    help="RePRA metrics table for composite panel (e); skipped if missing.")
    ap.add_argument("--no-figure-suptitle", action="store_true",
                    help="omit the composite suptitle (use when the LaTeX caption carries the title).")
    ap.add_argument("--ecfp-edge-k", type=int, default=1, help="ECFP nearest neighbors per motif to draw as edges.")
    ap.add_argument("--ecfp-edge-min-tanimoto", type=float, default=0.4,
                    help="only draw an ECFP edge if Tanimoto >= this.")
    ap.add_argument("--top-frequent-only", type=int, default=0,
                    help="if >0, restrict to the N most frequent motifs (clearest structure).")
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    apply_publication_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_tag or Path(args.pretrain_dir).name

    model, vocab, _rep, _cfg, _dev = load_pretrained_roberta(args.pretrain_dir, device=args.device)
    freq = load_freqs(Path(args.pretrain_dir))
    itos = list(vocab.itos)

    # Collect motif tokens (+ properties) in vocab order.
    motif_ids: List[int] = []
    motif_smi: List[str] = []
    props: List[dict] = []
    for i, tok in enumerate(itos):
        if classify_token(tok) != "motif":
            continue
        mol = Chem.MolFromSmiles(tok)
        motif_ids.append(i)
        motif_smi.append(tok)
        rec = {"token_id": i, "token": tok, "freq": int(freq.get(tok, 0))}
        rec.update(motif_properties(mol))
        props.append(rec)
    df = pd.DataFrame(props)
    print(f"[load] {model_tag}: {len(motif_ids)} motif tokens, mode={args.embedding_mode}", flush=True)

    # Optionally restrict to the most-trained motifs (clearest structure).
    if args.top_frequent_only and args.top_frequent_only < len(df):
        keep = df.sort_values("freq", ascending=False).head(args.top_frequent_only).index.to_numpy()
        keep.sort()
        df = df.iloc[keep].reset_index(drop=True)
        motif_ids = [motif_ids[i] for i in keep]
        motif_smi = [motif_smi[i] for i in keep]
        print(f"[filter] kept top-{args.top_frequent_only} by frequency", flush=True)

    # Embeddings.
    if args.embedding_mode == "contextual":
        Z = contextual_motif_embeddings(model, vocab, motif_ids, device=args.device, batch_size=args.batch_size)
    else:
        Z = model.tok_emb.weight.detach().cpu().numpy()[np.asarray(motif_ids)].astype(np.float32)

    fps = [_fp(s) for s in motif_smi]
    curve = knn_tanimoto_curve(Z, fps, ks=[1, 3, 5, 10, 20], n_query=args.knn_queries, seed=args.seed)
    enrich = curve[10][0] / max(curve[10][1], 1e-9)
    print(f"[knn] Tanimoto@10={curve[10][0]:.3f} random={curve[10][1]:.3f} enrichment={enrich:.2f}x", flush=True)

    xy, used = fit_2d(
        Z, method=args.projection, seed=args.seed, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, spread=args.spread, center=not args.no_center,
        pca_pre=args.pca_pre, uniformize=args.uniformize,
    )
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    print(f"[projection] {used} center={not args.no_center} pca_pre={args.pca_pre} "
          f"min_dist={args.min_dist} n_neighbors={args.n_neighbors}", flush=True)

    plot_motif_panels(df, out_dir / "motif_embedding_by_property.png", model_tag=model_tag, enrich=enrich)
    plot_enrichment(curve, out_dir / "motif_knn_tanimoto_enrichment.png", model_tag=model_tag, mode=args.embedding_mode)

    # Overlay ECFP chemical distance onto the embedding UMAP.
    edges = ecfp_nn_edges(fps, k=args.ecfp_edge_k, min_tanimoto=args.ecfp_edge_min_tanimoto)
    plot_umap_ecfp_edges(df, edges, out_dir / "motif_umap_ecfp_edges.png",
                         model_tag=model_tag, min_tanimoto=args.ecfp_edge_min_tanimoto)
    print(f"[ecfp-edges] {len(edges)} edges at Tanimoto>={args.ecfp_edge_min_tanimoto}", flush=True)

    # Neighbor-example grid: curated queries present in this motif set, topped up
    # with the most frequent motifs.
    smi_to_local = {s: i for i, s in enumerate(motif_smi)}
    queries = [smi_to_local[s] for s in _QUERY_MOTIFS if s in smi_to_local]
    if len(queries) < args.neighbor_k:
        for j in df.sort_values("freq", ascending=False).index:
            if j not in queries:
                queries.append(int(j))
            if len(queries) >= max(args.neighbor_k, 6):
                break
    plot_neighbor_examples(Z, motif_smi, df["freq"].to_numpy(), queries[:8],
                           out_dir / "motif_neighbor_examples.png", k=args.neighbor_k)
    plot_umap_ecfp_anchors(df, fps, queries[:4], motif_smi,
                           out_dir / "motif_umap_ecfp_anchors.png", model_tag=model_tag)

    # Publication composite (nature-figure): editable SVG + PDF + PNG.
    plot_nature_main_figure(
        df, curve, edges, Z, fps, motif_smi, queries,
        out_dir / "motif_embedding_nature_figure", model_tag=model_tag,
        enrich=enrich, min_tanimoto=args.ecfp_edge_min_tanimoto,
        repra_csv=Path(args.repra_csv) if args.repra_csv else None,
        suptitle=not args.no_figure_suptitle,
    )
    print(f"[nature] wrote composite SVG/PDF/PNG to {out_dir}/motif_embedding_nature_figure.*", flush=True)

    df.to_csv(out_dir / "motif_token_embedding_table.csv", index=False)
    print(f"saved motif-embedding figures + table to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

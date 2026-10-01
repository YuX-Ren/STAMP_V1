"""Reusable Nature-journal matplotlib styling for STAMP benchmark figures.

Usage
-----
    from stamp.benchmark.plot_style import (
        apply_nature_style, despine, style_colorbar,
        NATURE, COL_SINGLE, COL_DOUBLE,
    )

    apply_nature_style()                       # set global rcParams once
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    ...
    despine(ax)                                # remove top/right spines
    style_colorbar(cbar)                       # hairline outline + tick styling

`apply_nature_style()` is idempotent; call it before creating any figure.
The palette in `NATURE` is colour-blind-safe (RdBu/Set2 derived) and is what
the benchmark figures use for "ours" vs. baseline comparisons.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

# Nature column widths (mm -> inches). Figures should be authored at final
# print size so font sizes land at the journal's 5-7 pt requirement.
COL_SINGLE = 89.0 / 25.4   # ~3.50 in, single column
COL_ONEHALF = 120.0 / 25.4  # ~4.72 in, 1.5 column
COL_DOUBLE = 183.0 / 25.4  # ~7.20 in, double column

# Muted, colour-blind-safe palette. `ours` reads as the hero series.
NATURE = {
    "ours": "#B2182B",      # muted crimson  (STAMP / proposed method)
    "baseline": "#2166AC",  # muted blue     (reference / prior work)
    "accent": "#1B7837",    # muted green
    "highlight": "#E08214",  # muted orange
    "neutral": "#4D4D4D",   # axis / text grey
    "grid": "#000000",      # grid colour (used at low alpha)
    "muted": "#878787",
    "series": ("#B2182B", "#2166AC", "#1B7837", "#E08214", "#762A83", "#01665E"),
}

# Canonical Nature-figure palette (matches the `nature-figure` skill family) for
# when you want figures consistent with that house style. `ours`/`baseline` map to
# its hero red and primary blue; use green/red only for directional (gain/drop) cues.
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#FFD700",
}
# Use when colour itself carries meaning (hero / baseline / positive variant).
DEFAULT_COLORS = (
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
)

_PREFERRED_FONTS = ["Helvetica", "Helvetica Neue", "Arial", "Nimbus Sans", "DejaVu Sans"]


def _resolve_font() -> str:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _PREFERRED_FONTS if f in available), "DejaVu Sans")


def nature_rc(base_font_size: float = 9.0) -> dict:
    """Return the rcParams dict for the Nature style (does not mutate global state)."""
    family = _resolve_font()
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [family],
        "font.size": base_font_size,
        "axes.linewidth": 0.8,
        "axes.labelsize": base_font_size + 1,
        "axes.titlesize": base_font_size + 2,
        "axes.titleweight": "regular",
        "axes.labelcolor": NATURE["neutral"],
        "axes.edgecolor": NATURE["neutral"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base_font_size - 0.5,
        "ytick.labelsize": base_font_size - 0.5,
        "xtick.color": NATURE["neutral"],
        "ytick.color": NATURE["neutral"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": base_font_size - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Embed fonts as editable text in vector output (Illustrator-friendly).
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def apply_nature_style(base_font_size: float = 9.0) -> None:
    """Apply the Nature style to the global matplotlib rcParams (idempotent)."""
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401  (ensures pyplot import side effects)

    matplotlib.rcParams.update(nature_rc(base_font_size))


@contextmanager
def nature_style(base_font_size: float = 9.0) -> Iterator[None]:
    """Context manager that applies the Nature style only within the block."""
    import matplotlib.pyplot as plt

    with plt.rc_context(nature_rc(base_font_size)):
        yield


def despine(ax, *, top: bool = True, right: bool = True,
            left: bool = False, bottom: bool = False) -> None:
    """Hide the requested spines (top/right by default) and recolour the rest."""
    for name, hide in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        ax.spines[name].set_visible(not hide)
    for name in ("left", "bottom"):
        if ax.spines[name].get_visible():
            ax.spines[name].set_color(NATURE["neutral"])
    ax.tick_params(colors=NATURE["neutral"])


def soft_grid(ax, *, axis: str = "y", alpha: float = 0.08) -> None:
    """Add faint reference gridlines and push them below the data."""
    ax.grid(True, axis=axis, color=NATURE["grid"], alpha=alpha, linewidth=0.6)
    ax.set_axisbelow(True)


def style_colorbar(cbar, label: str | None = None, *, labelsize: float = 9.0) -> None:
    """Apply hairline outline + neutral tick styling to a colorbar."""
    if label is not None:
        cbar.set_label(label, fontsize=labelsize)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(NATURE["neutral"])
    cbar.ax.tick_params(width=0.6, length=3, labelsize=labelsize - 1, colors=NATURE["neutral"])


def style_legend(legend, *, text_color: str = "#222222") -> None:
    """Recolour legend text to the standard near-black."""
    for text in legend.get_texts():
        text.set_color(text_color)


def save_publication(fig, out_path, *, formats=("svg", "pdf", "png"),
                     dpi: int = 600, close: bool = True) -> list[str]:
    """Export a figure for submission, SVG first (editable text), then PDF/PNG/TIFF.

    `out_path` may carry an extension (stripped) or not. Parent dirs are created.
    SVG/PDF are vector with embedded editable text (see ``apply_nature_style``);
    raster formats (png/tiff) are written at ``dpi`` (600 = Nature submission grade).
    """
    from pathlib import Path

    base = Path(out_path)
    base = base.with_suffix("") if base.suffix else base
    base.parent.mkdir(parents=True, exist_ok=True)
    raster = {"png", "tif", "tiff", "jpg", "jpeg"}
    saved: list[str] = []
    for fmt in formats:
        path = f"{base}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in raster:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return saved

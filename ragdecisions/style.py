"""Shared visual style so every chart in the study looks like one family.

The same dark surface and finding-first titles as the experimentation toolkit.
Each retriever keeps one colour everywhere, the reranker is shown as a filled
marker rather than a new colour, and the acid lime highlight is reserved for
the thing the reader should look at.
"""

from pathlib import Path

import matplotlib as mpl

SURFACE = "#111116"
INK = "#F4F1EA"
INK_2 = "#B9B5C2"
MUTED = "#85818F"
GRID = "#24232B"
BASELINE = "#3A3944"
HIGHLIGHT = "#D4FF3F"

# One colour per retriever. Keyword search is a quiet grey on purpose: it is the free baseline.
RETRIEVER_COLORS = {
    "bm25": "#8A8796",
    "minilm": "#3B8FE0",
    "bge-small": "#C4831C",
    "bge-base": "#7B68FF",
}
RETRIEVER_LABELS = {
    "bm25": "Keyword search (BM25)",
    "minilm": "MiniLM-L6",
    "bge-small": "bge-small",
    "bge-base": "bge-base",
}

DATASET_COLORS = {
    "multihop_rag": "#F0563F",
    "natural_questions": "#3BB39B",
}

# Floors that any configuration has to beat.
FLOOR_COLOR = "#55535F"

FONT_STACK = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]
FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def use_style() -> None:
    """Apply the study style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "ytick.major.size": 0,
        "xtick.major.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.solid_capstyle": "round",
        "text.parse_math": False,  # so dollar signs in labels print as dollars
    })


def headline(fig, title: str, subtitle: str = "") -> None:
    """Big finding-first title with a quieter subtitle, left aligned."""
    from matplotlib.transforms import offset_copy

    fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=17,
             fontweight="bold", color=INK)
    if subtitle:
        below = offset_copy(fig.transFigure, fig=fig, y=-30, units="points")
        fig.text(0.02, 0.98, subtitle, ha="left", va="top", fontsize=11,
                 color=MUTED, transform=below)


def save(fig, name: str) -> None:
    """Save a figure to docs/figures for use in the README."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.3)

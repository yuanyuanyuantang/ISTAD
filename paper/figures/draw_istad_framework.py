# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) model schematic → SankeyDiagram/plot_SankeyDiagram.py (structure incompatible; cross-type inherit) → param inherit
# (b) hypergraph mechanism → SankeyDiagram/plot_SankeyDiagram.py (structure incompatible; cross-type inherit) → param inherit
# (c) method extension → SankeyDiagram/plot_SankeyDiagram.py (structure incompatible; cross-type inherit) → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


SKILL_ROOT = Path("/home/zenghaoyang/.codex/skills/academic-figure-skill")
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
from compose import compose_figure  # noqa: E402


BLUE = CATEGORICAL[0]
RED = CATEGORICAL[1]
GREEN = CATEGORICAL[2]
ORANGE = CATEGORICAL[3]
PURPLE = CATEGORICAL[4]
DARK_GREY = CATEGORICAL[5]
LIGHT_BLUE = "#EAF2F8"
LIGHT_GREEN = "#EAF4EC"
LIGHT_ORANGE = "#FFF3E5"
LIGHT_PURPLE = "#F2EAF5"
LIGHT_RED = "#F8E8E8"
LIGHT_GREY = "#F3F3F3"
EDGE_GREY = "#777777"

# Exact double-column canvas: 183 mm x 143.8 mm.
figsize = (7.204724, 5.661417)


def _setup(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(
        0.53, 0.965, title, ha="center", va="top",
        fontsize=8.5, fontweight="bold", color=BLACK,
    )


def _box(
    ax, x, y, w, h, text, face, edge, fontsize=6.2, linewidth=0.85,
    linestyle="-", radius=0.018, text_color=BLACK, weight="normal",
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        linestyle=linestyle, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=text_color, fontweight=weight,
        linespacing=1.15, zorder=3,
    )
    return patch


def _arrow(
    ax, start, end, color=EDGE_GREY, linewidth=0.85,
    linestyle="-", connectionstyle="arc3,rad=0", alpha=1.0,
    mutation_scale=7,
):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle, alpha=alpha, zorder=1,
        shrinkA=1.5, shrinkB=1.5,
    )
    ax.add_patch(arrow)
    return arrow


def panel_a(ax, spec=None):
    """Hero panel: the complete ISTAD scoring pathway."""
    _setup(ax, "ISTAD overview")
    ax.text(
        0.53, 0.930, "Normal-only fitting • no anomaly labels",
        ha="center", va="center", fontsize=5.5, color=DARK_GREY,
    )

    _box(ax, 0.30, 0.855, 0.42, 0.055, "Normalized window  X", LIGHT_GREY, DARK_GREY, 6.5, weight="bold")

    _box(ax, 0.065, 0.735, 0.39, 0.065, "Ridge VAR(1) predictor", LIGHT_BLUE, BLUE, 6.2, weight="bold")
    _box(ax, 0.555, 0.735, 0.39, 0.065, "Causal Conv1D + HGAT-Lite", LIGHT_GREEN, GREEN, 6.2, weight="bold")
    _arrow(ax, (0.48, 0.855), (0.26, 0.802), color=BLUE)
    _arrow(ax, (0.54, 0.855), (0.75, 0.802), color=GREEN)

    _box(ax, 0.065, 0.625, 0.39, 0.060, "Signed innovations  r(t)", LIGHT_BLUE, BLUE)
    _box(
        ax, 0.555, 0.615, 0.39, 0.080,
        "Dynamic sparse incidence\n" + r"$H(t)\in\mathbb{R}^{C\times M}$  •  Top-k",
        LIGHT_GREEN, GREEN,
    )
    _arrow(ax, (0.26, 0.735), (0.26, 0.687), color=BLUE)
    _arrow(ax, (0.75, 0.735), (0.75, 0.697), color=GREEN)

    _box(ax, 0.065, 0.500, 0.39, 0.075, "Feature evidence  E(t)\nstandardized |r| or squared r", LIGHT_BLUE, BLUE, 5.9)
    _box(
        ax, 0.555, 0.490, 0.39, 0.085,
        "Hyperedge evidence\n" + r"$H(t)^{\mathsf{T}}E(t)$",
        LIGHT_GREEN, GREEN, 6.1,
    )
    _arrow(ax, (0.26, 0.625), (0.26, 0.577), color=BLUE)
    _arrow(ax, (0.75, 0.615), (0.75, 0.577), color=GREEN)
    _arrow(
        ax, (0.455, 0.537), (0.555, 0.537), color=PURPLE,
        linestyle="--", linewidth=1.0,
    )

    _box(ax, 0.065, 0.385, 0.39, 0.060, "Training ECDF  →  q(t)", LIGHT_BLUE, BLUE, 6.2)
    _box(ax, 0.555, 0.385, 0.39, 0.060, "Training ECDF  →  g(t)", LIGHT_GREEN, GREEN, 6.2)
    _arrow(ax, (0.26, 0.500), (0.26, 0.447), color=BLUE)
    _arrow(ax, (0.75, 0.490), (0.75, 0.447), color=GREEN)

    _box(ax, 0.285, 0.275, 0.45, 0.060, "Normal-tail stability gate", LIGHT_GREY, DARK_GREY, 6.2)
    _arrow(ax, (0.26, 0.385), (0.41, 0.337), color=BLUE)
    _arrow(ax, (0.75, 0.385), (0.61, 0.337), color=GREEN)

    _box(
        ax, 0.245, 0.145, 0.53, 0.082,
        "Rank-safe fusion\n" + r"$s=(q+\epsilon g)/(1+\epsilon),\quad \epsilon=0.5/(N+1)$",
        LIGHT_PURPLE, PURPLE, 6.2, linewidth=1.1, weight="bold",
    )
    _arrow(ax, (0.51, 0.275), (0.51, 0.229), color=PURPLE, linewidth=1.0)

    _box(ax, 0.335, 0.035, 0.35, 0.060, "Anomaly score  s(t)", LIGHT_RED, RED, 6.5, linewidth=1.0, weight="bold")
    _arrow(ax, (0.51, 0.145), (0.51, 0.097), color=RED, linewidth=1.0)
    ax.text(
        0.51, 0.012, "HGAT refines ties without reversing distinct VAR ranks",
        ha="center", va="bottom", fontsize=5.2, color=PURPLE,
    )


def panel_b(ax, spec=None):
    """Mechanism panel: low-rank dynamic node-edge-node propagation."""
    _setup(ax, "HGAT-Lite relation learning")
    ax.text(0.14, 0.845, "nodes", ha="center", fontsize=5.7, color=DARK_GREY)
    ax.text(0.50, 0.845, "hyperedges", ha="center", fontsize=5.7, color=DARK_GREY)
    ax.text(0.86, 0.845, "messages", ha="center", fontsize=5.7, color=DARK_GREY)

    node_y = [0.73, 0.61, 0.49, 0.37, 0.25]
    edge_y = [0.62, 0.39]
    output_y = [0.69, 0.53, 0.37, 0.21]
    for idx, y in enumerate(node_y, start=1):
        ax.add_patch(Circle((0.14, y), 0.033, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.9, zorder=3))
        ax.text(0.14, y, f"h{idx}", ha="center", va="center", fontsize=5.3, color=BLUE, zorder=4)

    for idx, y in enumerate(edge_y, start=1):
        ax.add_patch(Ellipse((0.50, y), 0.13, 0.085, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.0, zorder=3))
        ax.text(0.50, y, f"z{idx}", ha="center", va="center", fontsize=5.6, color=GREEN, fontweight="bold", zorder=4)

    left_edges = [
        (0, 0, 0.90), (1, 0, 0.55), (2, 0, 0.35),
        (1, 1, 0.28), (2, 1, 0.55), (3, 1, 0.85), (4, 1, 0.42),
    ]
    for ni, ei, strength in left_edges:
        _arrow(
            ax, (0.176, node_y[ni]), (0.43, edge_y[ei]), color=GREEN,
            linewidth=0.45 + 0.75 * strength, alpha=0.28 + 0.55 * strength,
            mutation_scale=5,
        )

    for idx, y in enumerate(output_y, start=1):
        ax.add_patch(Circle((0.86, y), 0.029, facecolor=LIGHT_PURPLE, edgecolor=PURPLE, linewidth=0.85, zorder=3))
        ax.text(0.86, y, f"m{idx}", ha="center", va="center", fontsize=5.1, color=PURPLE, zorder=4)

    right_edges = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)]
    for ei, oi in right_edges:
        _arrow(
            ax, (0.57, edge_y[ei]), (0.825, output_y[oi]), color=PURPLE,
            linewidth=0.65, alpha=0.55, mutation_scale=5,
        )

    ax.text(0.31, 0.775, "Top-k H(t)", ha="center", fontsize=5.4, color=GREEN, fontweight="bold")
    ax.text(0.68, 0.775, "reuse H(t)", ha="center", fontsize=5.4, color=PURPLE, fontweight="bold")
    _box(
        ax, 0.08, 0.075, 0.84, 0.075,
        r"$H_{im}(t)\propto\exp(h_i(t)^{\mathsf{T}}d_m/\sqrt{r})$"
        + "\nnode → edge → node, one shared incidence",
        LIGHT_GREY, DARK_GREY, 5.5,
    )


def panel_c(ax, spec=None):
    """Scope panel: the BR-KAN innovation-domain projection ablation."""
    _setup(ax, "BR-KAN relation extension")
    border = FancyBboxPatch(
        (0.045, 0.08), 0.91, 0.80,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#FFFCF7", edgecolor=ORANGE, linewidth=1.0,
        linestyle="--", zorder=0,
    )
    ax.add_patch(border)
    ax.text(
        0.50, 0.835, "ABLATION EXTENSION ONLY",
        ha="center", va="center", fontsize=5.7, fontweight="bold", color="#9A5A00",
    )

    _box(
        ax, 0.19, 0.710, 0.62, 0.072,
        "Signed VAR innovation  " + r"$\xi=r/\sigma$",
        LIGHT_ORANGE, ORANGE, 6.0,
    )
    _box(ax, 0.30, 0.600, 0.40, 0.060, "Causal Conv1D", LIGHT_GREY, DARK_GREY, 6.0)
    _arrow(ax, (0.50, 0.710), (0.50, 0.662), color=ORANGE)

    _box(
        ax, 0.075, 0.385, 0.38, 0.150,
        "Linear projection\nv=Wu\nMAIN ISTAD",
        LIGHT_BLUE, BLUE, 5.8, linewidth=0.95, weight="bold",
    )
    _box(
        ax, 0.535, 0.370, 0.40, 0.180,
        "BR-KAN projection\n"
        + r"$v=Wu+0.1\tanh\phi_B(u)$" + "\n"
        + r"$\phi_B(u)=\sum_k c_kB_k(u)$" + "\nOPTIONAL",
        LIGHT_ORANGE, ORANGE, 5.2, linewidth=0.95, linestyle="--", weight="bold",
    )
    _arrow(ax, (0.43, 0.600), (0.28, 0.537), color=BLUE)
    _arrow(ax, (0.57, 0.600), (0.73, 0.552), color=ORANGE, linestyle="--")

    _box(ax, 0.25, 0.215, 0.50, 0.075, "Same HGAT-Lite", LIGHT_GREEN, GREEN, 6.2, weight="bold")
    _arrow(ax, (0.28, 0.385), (0.42, 0.292), color=BLUE)
    _arrow(ax, (0.73, 0.370), (0.58, 0.292), color=ORANGE, linestyle="--")
    ax.text(
        0.50, 0.145, "c=0 at initialization  •  bounded ±0.1  •  +64 parameters",
        ha="center", va="center", fontsize=5.25, color="#8A5600",
    )
    ax.text(
        0.50, 0.095, "Evaluated separately; does not replace the main method",
        ha="center", va="bottom", fontsize=5.2, color=DARK_GREY,
    )


def main():
    output_prefix = Path(__file__).resolve().parent / "fig1_istad_framework"
    fig = compose_figure(
        panel_funcs=[panel_a, panel_b, panel_c],
        panel_types=["schematic", "mechanism diagram", "schematic extension"],
        fig_width_mm=183,
        hero_idx=0,
        archetype="schematic_led",
        output_prefix=str(output_prefix),
        panel_labels=["a", "b", "c"],
        journal="Knowledge-Based Systems",
    )
    fig.set_size_inches(*figsize, forward=True)
    # Re-export to an exact 183 mm canvas after composition.  The skill's
    # baseline tight export is retained above; this overwrite prevents the
    # whitespace trimmer from changing the journal-width physical dimensions.
    width_in, height_in = figsize
    exact_bbox = Bbox.from_bounds(0, 0, width_in, height_in)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches=exact_bbox, dpi=300)
    fig.savefig(f"{output_prefix}.png", bbox_inches=exact_bbox, dpi=300)
    fig.savefig(f"{output_prefix}.svg", bbox_inches=exact_bbox, dpi=300)
    plt.close(fig)
    print(f"Exported editable SVG: {output_prefix}.svg")


if __name__ == "__main__":
    main()

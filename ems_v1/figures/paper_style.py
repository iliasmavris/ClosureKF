"""
Figure Design Language v1.0 for GMD / ClosureKF paper.

Import this in every plotting script:
    from paper_style import apply_gmd_style, COLORS, LINESTYLES, MARKERS
    from paper_style import SINGLE_COL, DOUBLE_COL
    apply_gmd_style()

Palette: Okabe-Ito (colorblind-safe).  Two-tier scheme:
  Tier A (highlighted): physics, closure variants, strong ML refs
  Tier B (de-emphasized): simple baselines in gray family

Model hierarchy (current manuscript):
  6-term full library -> 2-term (b2+d2) intermediate -> 1-term (d2-only) recommended
  Blue = intermediate 2-term, Vermillion = recommended d2-only
"""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Tier A: primary models (always colored)
# ---------------------------------------------------------------------------
COLORS = {
    "physics_kf": "#222222",       # dark gray / black
    "closure_2t": "#0072B2",       # blue  -- intermediate (b2+d2)
    "closure_1t": "#D55E00",       # vermillion -- recommended (d2-only)
    "mlp":        "#009E73",       # green
    "gru":        "#CC79A7",       # reddish purple

    # Tier B: de-emphasized baselines (gray family by default)
    "rf":          "#E69F00",      # orange -- promote to Tier A if key comparator
    "ridge":       "#999999",      # medium gray
    "ar10":        "#999999",      # medium gray
    "persistence": "#BDBDBD",      # light gray
    "mean_inc":    "#BDBDBD",      # light gray
    "null":        "#BDBDBD",      # light gray
}

# ---------------------------------------------------------------------------
# Linestyles -- Tier B baselines get distinct dashes to separate them
# ---------------------------------------------------------------------------
LINESTYLES = {
    # gate diagnostics
    "gate_on":     "-",
    "gate_off":    "--",

    # Tier B baseline differentiation (all gray, different dashes)
    "persistence": (0, (6, 2)),    # long dash
    "mean_inc":    (0, (2, 2)),    # short dash
    "ridge":       ":",            # dotted
    "ar10":        "-.",           # dash-dot
}

# ---------------------------------------------------------------------------
# Markers -- only for discrete horizons / sampled points
# ---------------------------------------------------------------------------
MARKERS = {
    "physics_kf": "o",             # circle
    "closure_2t": "s",             # square
    "closure_1t": "^",             # triangle
}

# ---------------------------------------------------------------------------
# Figure sizes (inches) -- build to GMD column widths
# ---------------------------------------------------------------------------
SINGLE_COL = (3.35, 2.6)          # ~85 mm wide
DOUBLE_COL = (6.9, 3.2)           # ~175 mm wide


def apply_gmd_style():
    """Apply GMD/ClosureKF rcParams.  Call once at script start."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.size":          9,
        "axes.labelsize":     10,
        "axes.titlesize":     10,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "lines.linewidth":    2.0,
        "axes.grid":          True,
        "grid.alpha":         0.18,
        "grid.linestyle":     "-",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     False,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,      # embed TrueType
        "ps.fonttype":        42,
    })

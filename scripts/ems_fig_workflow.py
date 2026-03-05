"""
Generate fig_workflow.pdf: Closure-discovery workflow diagram.

A programmatic figure showing the 4-stage pipeline:
  Stage 1: Physics-only training
  Stage 2: Closure training (frozen physics)
  Stage 3: Parsimony reduction (5t -> 2t -> 1t)
  Stage 4: Diagnostic verification + lockbox

Output: manuscript_ems_v1/figures/fig_workflow.pdf (and .png)

NO data dependencies. Pure diagram.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "manuscript_ems_v1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Also save to v11.1 figures for graphicspath
FIG_DIR2 = ROOT / "final_lockbox_v11_1_alpha_fix" / "seed1" / "figures"

# ---- Layout parameters ----
fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-1.0, 6.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
C_DATA = '#4DBEEE'     # light blue - data
C_STAGE = '#77AC30'    # green - training stages
C_AUDIT = '#EDB120'    # amber - audits
C_RESULT = '#D95319'   # orange-red - outputs
C_ARROW = '#333333'
C_GENERIC = '#A2A2A2'  # grey - generic labels


def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False,
             style='round', alpha=0.85):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"{style},pad=0.15",
        facecolor=color, edgecolor='#333333', linewidth=1.0,
        alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=3, wrap=True)


def draw_diamond(ax, x, y, size, text, color, fontsize=7):
    """Draw a diamond (audit checkpoint)."""
    s = size / 2
    diamond = plt.Polygon(
        [(x, y+s), (x+s, y), (x, y-s), (x-s, y)],
        facecolor=color, edgecolor='#333333', linewidth=1.0,
        alpha=0.85, zorder=2
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=3)


def arrow(ax, x1, y1, x2, y2, text='', fontsize=7):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                linewidth=1.5, connectionstyle='arc3,rad=0'),
                zorder=1)
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.2
        ax.text(mx, my, text, ha='center', va='bottom', fontsize=fontsize,
                color='#555555', style='italic')


# ============================================================
# Row 1 (y=5): Data preparation
# ============================================================
draw_box(ax, 1.0, 5.0, 2.0, 0.8, 'Raw data\n(300 Hz)', C_DATA, fontsize=8)
arrow(ax, 2.1, 5.0, 3.4, 5.0)
draw_box(ax, 4.5, 5.0, 2.0, 0.8, 'Anti-alias\nfilter + 10 Hz', C_DATA, fontsize=8)
arrow(ax, 5.6, 5.0, 6.9, 5.0)
draw_box(ax, 8.0, 5.0, 2.0, 0.8, 'Clean splits\n(train/val/test)', C_DATA,
         fontsize=8, bold=True)
arrow(ax, 9.1, 5.0, 10.4, 5.0)
draw_diamond(ax, 11.2, 5.0, 0.8, 'MD5', C_AUDIT, fontsize=7)

# ============================================================
# Row 2 (y=3.3): Training stages
# ============================================================
# Stage 1
draw_box(ax, 1.5, 3.3, 2.5, 1.0,
         'Stage 1\nPhysics-only\n(KF, all params)', C_STAGE,
         fontsize=8, bold=True)
arrow(ax, 2.85, 3.3, 4.0, 3.3)

# Freeze audit
draw_diamond(ax, 4.7, 3.3, 0.9, 'Freeze\naudit', C_AUDIT, fontsize=6)
arrow(ax, 5.4, 3.3, 6.3, 3.3)

# Stage 2
draw_box(ax, 7.5, 3.3, 2.5, 1.0,
         'Stage 2\nClosure training\n(EKF, frozen physics)', C_STAGE,
         fontsize=8, bold=True)
arrow(ax, 8.85, 3.3, 10.0, 3.3)

# Alpha audit
draw_diamond(ax, 10.7, 3.3, 0.9, 'Alpha\naudit', C_AUDIT, fontsize=6)

# Connect data to Stage 1
arrow(ax, 8.0, 4.55, 1.5, 3.85)

# ============================================================
# Row 3 (y=1.5): Parsimony + diagnostics
# ============================================================
draw_box(ax, 1.5, 1.5, 2.5, 1.0,
         'Tolerance rule\n5t -> 2t\n(forward selection)', C_STAGE,
         fontsize=8, bold=True)
arrow(ax, 2.85, 1.5, 4.3, 1.5)

draw_box(ax, 5.5, 1.5, 2.2, 1.0,
         'Ablation\n2t -> 1t\n(operational test)', C_STAGE,
         fontsize=8, bold=True)
arrow(ax, 6.7, 1.5, 7.9, 1.5)

draw_box(ax, 9.2, 1.5, 2.4, 1.0,
         'Diagnostics\nACF, NIS, PSD\ncov90, DxR2', C_RESULT,
         fontsize=8, bold=True)
arrow(ax, 10.5, 1.5, 11.5, 1.5)

draw_box(ax, 12.3, 1.5, 1.4, 1.0,
         'Lockbox\n(frozen)', C_RESULT,
         fontsize=9, bold=True, alpha=0.95)

# Connect Stage 2 to parsimony
arrow(ax, 7.5, 2.75, 1.5, 2.05)

# ============================================================
# Row 4 (y=-0.2): Generic workflow labels
# ============================================================
ax.text(1.5, -0.3, 'Configurable:\ncandidate library,\ntolerance thresholds',
        ha='center', va='top', fontsize=6.5, color=C_GENERIC,
        style='italic', bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#F0F0F0', alpha=0.5))

ax.text(5.5, -0.3, 'Configurable:\nleave-one-out\non any subset',
        ha='center', va='top', fontsize=6.5, color=C_GENERIC,
        style='italic', bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#F0F0F0', alpha=0.5))

ax.text(9.2, -0.3, 'Configurable:\nmetric set,\nhorizon range',
        ha='center', va='top', fontsize=6.5, color=C_GENERIC,
        style='italic', bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#F0F0F0', alpha=0.5))

ax.text(12.3, -0.3, 'Hash-based\nprovenance',
        ha='center', va='top', fontsize=6.5, color=C_GENERIC,
        style='italic', bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#F0F0F0', alpha=0.5))

# ============================================================
# Legend
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor=C_DATA, edgecolor='#333', label='Data preparation'),
    mpatches.Patch(facecolor=C_STAGE, edgecolor='#333', label='Training / selection'),
    mpatches.Patch(facecolor=C_AUDIT, edgecolor='#333', label='Audit checkpoint'),
    mpatches.Patch(facecolor=C_RESULT, edgecolor='#333', label='Output / lockbox'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
          framealpha=0.8, edgecolor='#999')

# Title
ax.text(6.75, 6.2, 'Closure-Discovery Workflow', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(6.75, 5.8, '(generalizable to any state-space model with systematic model error)',
        ha='center', va='center', fontsize=8, color='#666666', style='italic')

plt.tight_layout()

# Save
out_pdf = FIG_DIR / "fig_workflow.pdf"
out_png = FIG_DIR / "fig_workflow.png"
fig.savefig(str(out_pdf), dpi=300, bbox_inches='tight')
fig.savefig(str(out_png), dpi=150, bbox_inches='tight')

# Also save to v11.1 figures path (for graphicspath)
out_pdf2 = FIG_DIR2 / "fig_workflow.pdf"
out_png2 = FIG_DIR2 / "fig_workflow.png"
fig.savefig(str(out_pdf2), dpi=300, bbox_inches='tight')
fig.savefig(str(out_png2), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf2}")
print(f"Saved: {out_png2}")
print("Done.")

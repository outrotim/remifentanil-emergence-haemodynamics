#!/usr/bin/env python3
"""Generate STROBE flow diagram for Study 2 (OIH) — Figure 1."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def draw_box(ax, x, y, w, h, text, fontsize=9, bold=False, color='#E8F4FD', edgecolor='#2C3E50'):
    """Draw a box with text."""
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edgecolor, linewidth=1.2
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True,
            bbox=dict(facecolor='none', edgecolor='none'))

def draw_arrow(ax, x1, y1, x2, y2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

def draw_side_box(ax, x, y, w, h, text, fontsize=8):
    """Draw an exclusion box on the side."""
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor='#FDEDEC', edgecolor='#C0392B', linewidth=1.0
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color='#922B21')

def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Title
    ax.text(5, 15.5, 'Figure 1. STROBE Study Flow Diagram',
            ha='center', va='center', fontsize=13, fontweight='bold')

    # === Main flow boxes (center column) ===
    cx = 4.2  # center x
    bw = 3.8  # box width
    bh = 0.55  # box height

    # Box 0: VitalDB total
    y0 = 14.5
    draw_box(ax, cx, y0, bw, bh,
             'VitalDB database\nn = 6,388 noncardiac surgical cases',
             fontsize=9, bold=True, color='#D5F5E3')

    # Box 1: General anaesthesia
    y1 = 13.3
    draw_box(ax, cx, y1, bw, bh,
             'General anaesthesia\nn = 6,043')
    draw_arrow(ax, cx, y0 - bh/2, cx, y1 + bh/2)
    draw_side_box(ax, 8.0, (y0+y1)/2, 2.5, 0.45,
                  'Non-GA excluded\nn = 345', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y0+y1)/2), xytext=(cx+bw/2, (y0+y1)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 2: RFTN TCI
    y2 = 12.1
    draw_box(ax, cx, y2, bw, bh,
             'Remifentanil TCI data available\nn = 4,753')
    draw_arrow(ax, cx, y1 - bh/2, cx, y2 + bh/2)
    draw_side_box(ax, 8.0, (y1+y2)/2, 2.5, 0.45,
                  'No RFTN TCI\nn = 1,290', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y1+y2)/2), xytext=(cx+bw/2, (y1+y2)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 3: BIS
    y3 = 10.9
    draw_box(ax, cx, y3, bw, bh,
             'BIS monitoring data available\nn = 4,498')
    draw_arrow(ax, cx, y2 - bh/2, cx, y3 + bh/2)
    draw_side_box(ax, 8.0, (y2+y3)/2, 2.5, 0.45,
                  'No BIS data\nn = 255', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y2+y3)/2), xytext=(cx+bw/2, (y2+y3)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 4: Adults
    y4 = 9.7
    draw_box(ax, cx, y4, bw, bh,
             'Adults (age \u2265 18 years)\nn = 4,485')
    draw_arrow(ax, cx, y3 - bh/2, cx, y4 + bh/2)
    draw_side_box(ax, 8.0, (y3+y4)/2, 2.5, 0.45,
                  'Age < 18\nn = 13', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y3+y4)/2), xytext=(cx+bw/2, (y3+y4)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 5: Duration
    y5 = 8.5
    draw_box(ax, cx, y5, bw, bh,
             'Surgical duration \u2265 60 min\nn = 4,453')
    draw_arrow(ax, cx, y4 - bh/2, cx, y5 + bh/2)
    draw_side_box(ax, 8.0, (y4+y5)/2, 2.5, 0.45,
                  'Duration < 60 min\nn = 32', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y4+y5)/2), xytext=(cx+bw/2, (y4+y5)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 6: ASA
    y6 = 7.3
    draw_box(ax, cx, y6, bw, bh,
             'ASA I-IV (V excluded)\nn = 4,444')
    draw_arrow(ax, cx, y5 - bh/2, cx, y6 + bh/2)
    draw_side_box(ax, 8.0, (y5+y6)/2, 2.5, 0.45,
                  'ASA V\nn = 9', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y5+y6)/2), xytext=(cx+bw/2, (y5+y6)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # Box 7: QC
    y7 = 6.1
    draw_box(ax, cx, y7, bw, bh,
             'Haemodynamic data quality check passed\nn = 4,443',
             bold=True, color='#D5F5E3')
    draw_arrow(ax, cx, y6 - bh/2, cx, y7 + bh/2)
    draw_side_box(ax, 8.0, (y6+y7)/2, 2.5, 0.45,
                  'QC failure\nn = 1', fontsize=8)
    ax.annotate('', xy=(8.0-1.25, (y6+y7)/2), xytext=(cx+bw/2, (y6+y7)/2),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    # === Analysis populations ===
    y8 = 4.8
    draw_box(ax, cx, y8, bw, 0.7,
             'FINAL COHORT\nn = 4,443\n(RFTN extraction: 4,377 successful;\n5 outliers set to missing \u2192 4,372 for dose-response)',
             fontsize=8, bold=True, color='#D4EFDF', edgecolor='#1B7A3D')
    draw_arrow(ax, cx, y7 - bh/2, cx, y8 + 0.35)

    # Analysis subsets
    y9 = 3.3
    # Left: HR/MAP rebound subset
    draw_box(ax, 2.3, y9, 3.0, 0.7,
             'Primary endpoints\n(complete-case analytic sample)\nHR rebound: n = 2,854\nMAP rebound: n = 2,842',
             fontsize=7.5, color='#FEF9E7', edgecolor='#B7950B')

    # Right: Full cohort analyses (each endpoint listed separately)
    draw_box(ax, 6.8, y9, 3.2, 0.95,
             'Secondary / exploratory endpoints\nFTN rescue: n = 4,275\nNHD: n = 4,267\nOWSI partial: n = 4,275\nOWSI complete-case: n = 2,847',
             fontsize=7.0, color='#FEF9E7', edgecolor='#B7950B')

    draw_arrow(ax, cx-0.5, y8 - 0.35, 2.3, y9 + 0.35)
    draw_arrow(ax, cx+0.5, y8 - 0.35, 6.8, y9 + 0.35)

    # IPTW subset
    y10 = 2.0
    draw_box(ax, cx, y10, 4.5, 0.6,
             'IPTW analysis (covariate-complete)\nn = 2,866 (PS model: AUC = 0.851;\nESS = 1,332 after trimming)',
             fontsize=7.5, color='#F4ECF7', edgecolor='#7D3C98')
    draw_arrow(ax, 2.3, y9 - 0.35, cx-0.3, y10 + 0.3)
    draw_arrow(ax, 6.8, y9 - 0.35, cx+0.3, y10 + 0.3)

    # Taper analysis
    y11 = 0.9
    draw_box(ax, cx, y11, 4.5, 0.55,
             'Taper dynamics analysis\nn = 2,637 (TCI pump data + post-surgical HR/MAP)',
             fontsize=7.5, color='#F4ECF7', edgecolor='#7D3C98')
    draw_arrow(ax, cx, y10 - 0.3, cx, y11 + 0.275)

    plt.tight_layout()

    outdir = Path(__file__).parent / 'main_figures'
    fig.savefig(outdir / 'Figure_1_STROBE_flow.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(outdir / 'Figure_1_STROBE_flow.png', dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {outdir}")

if __name__ == '__main__':
    main()

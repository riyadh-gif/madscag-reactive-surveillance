"""Figure 2: the proposed two-stage system.

Laid out for the full text width at a minimum glyph size of 10 pt, which is
what the journal requires. Since the canvas width is fixed by the page, the
room for larger type has to come from wider boxes and shorter labels rather
than a bigger canvas, so each line is kept to about twenty characters.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10.0})
OUT = Path(os.environ["FIGDIR"])

W, H = 100.0, 56.0
fig, ax = plt.subplots(figsize=(6.8, 4.35))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TITLE, BODY, LEAD = 10.5, 10.0, 6.4


def box(cx, cy, w, title, lines, fc="white"):
    """Draw a box centred on (cx, cy), height derived from the line count."""
    h = 5.5 + LEAD * len(lines)
    x, y = cx - w / 2.0, cy - h / 2.0
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec="0.2", lw=0.9))
    ax.text(cx, y + h - 3.4, title, ha="center", va="center",
            fontsize=TITLE, fontweight="bold")
    for i, t in enumerate(lines):
        ax.text(cx, y + h - 7.6 - i * LEAD, t, ha="center", va="center",
                fontsize=BODY)
    return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)


def arrow(a, b, rad=0.0, frm="r", to="l"):
    p = {"r": (a["r"], a["cy"]), "l": (a["l"], a["cy"])}[frm]
    q = {"l": (b["l"], b["cy"]), "r": (b["r"], b["cy"])}[to]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 lw=0.9, color="0.25",
                                 connectionstyle="arc3,rad=%g" % rad))


win = box(12, 28, 22, "12-period window",
          ["climate, 86", "obs. history, 3",
           "neighbour lags, 5", "pressure, 2"], fc="0.95")
tab = box(36, 28, 21, "Tabular vector",
          ["last, mean, s.d.", "plus 8 summaries", "$266$ columns"], fc="0.95")
sa = box(60.5, 40, 26, "Stage A",
         ["boosted classifier", r"all rows $\rightarrow\ \hat{p}$"])
sb = box(60.5, 15, 26, "Stage B",
         ["boosted regressor", r"observed rows $\rightarrow\ \hat{s}$"])
b1 = box(87, 46, 25, "Blind-spot set",
         [r"$\hat{p}$ low, $\hat{s}$ high"], fc="0.93")
b2 = box(87, 28, 25, "Patrol ranking",
         [r"ordered by $\hat{s}$"], fc="0.93")
b3 = box(87, 10, 25, "Conformal interval",
         [r"$\hat{s} \pm \hat{q}_{0.9}$"], fc="0.93")

arrow(win, tab)
arrow(tab, sa, rad=-0.22)
arrow(tab, sb, rad=0.22)
arrow(sa, b1, rad=-0.12)
arrow(sb, b1, rad=0.30)
arrow(sb, b2, rad=-0.18)
arrow(sb, b3, rad=0.12)

# the separation note sits below the diagram rather than between the stage
# boxes, where larger type would collide with them
ax.text(50, 0.8, "no shared parameters; neither stage is refitted on the "
                 "other's output",
        ha="center", va="center", fontsize=10.0, style="italic", color="0.35")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / ("fig_system." + ext), dpi=400, bbox_inches="tight")
print("ditulis:", OUT / "fig_system.pdf")

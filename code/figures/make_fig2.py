"""Figure 2: the proposed two-stage system."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8})
OUT = Path(os.environ["FIGDIR"])

W, H = 100.0, 46.0
fig, ax = plt.subplots(figsize=(7.1, 3.15))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TITLE, BODY, LEAD = 7.8, 6.7, 3.5


def box(cx, cy, w, title, lines, fc="white"):
    """Draw a box centred on (cx, cy), height derived from the line count."""
    h = 6.0 + LEAD * len(lines)
    x, y = cx - w / 2.0, cy - h / 2.0
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec="0.2", lw=0.9))
    ax.text(cx, y + h - 2.9, title, ha="center", va="center",
            fontsize=TITLE, fontweight="bold")
    for i, t in enumerate(lines):
        ax.text(cx, y + h - 6.4 - i * LEAD, t, ha="center", va="center",
                fontsize=BODY)
    return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)


def arrow(a, b, rad=0.0, frm="r", to="l"):
    p = {"r": (a["r"], a["cy"]), "l": (a["l"], a["cy"])}[frm]
    q = {"l": (b["l"], b["cy"]), "r": (b["r"], b["cy"])}[to]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 lw=0.85, color="0.25",
                                 connectionstyle="arc3,rad=%g" % rad))


win = box(11, 23, 20, "12-period window",
          ["climate, 86", "observation history, 3",
           "neighbour lags, 5", "outbreak pressure, 2"], fc="0.95")
tab = box(34, 23, 18, "Tabular vector",
          ["last, mean, s.d. of each", "covariate, plus 8", "summaries, $266$ in all"],
          fc="0.95")
sa = box(58, 35, 20, "Stage A",
         ["boosted classifier", "fitted on all rows", r"output $\hat{p}$"])
sb = box(58, 11, 20, "Stage B",
         ["boosted regressor", "observed rows only", r"output $\hat{s}$"])
b1 = box(85, 38, 24, "Blind-spot set", [r"$\hat{p}$ low, $\hat{s}$ high"], fc="0.93")
b2 = box(85, 23, 24, "Patrol ranking", [r"ordered by $\hat{s}$"], fc="0.93")
b3 = box(85, 8, 24, "Conformal interval", [r"$\hat{s} \pm \hat{q}_{0.9}$"], fc="0.93")

arrow(win, tab)
arrow(tab, sa, rad=-0.22)
arrow(tab, sb, rad=0.22)
arrow(sa, b1, rad=-0.12)
arrow(sb, b1, rad=0.30)
arrow(sb, b2, rad=-0.18)
arrow(sb, b3, rad=0.12)

ax.text(58, 23, "no shared parameters;\nneither stage is refitted\non the other's output",
        ha="center", va="center", fontsize=6.3, style="italic", color="0.35")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / ("fig_system." + ext), dpi=400, bbox_inches="tight")
print("ditulis:", OUT / "fig_system.pdf")

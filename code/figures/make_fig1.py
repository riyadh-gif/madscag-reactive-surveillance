"""Figures 1 and 2: the reactive patrol loop, and the enrichment it produces.

IJIES does not accept one figure carrying two lettered panels, so the loop
schematic and the arrival-gradient bars are drawn and saved as two separate
figures. The gradient is computed from the raw register at draw time, so it
cannot drift away from the numbers quoted in the text.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11.6, "axes.linewidth": 0.7})

DATA = Path(os.environ["MADSCAG_INPUT"]) / "dataset" / "datasiap" / "panel_cuaca.csv"
OUT = Path(os.environ["FIGDIR"])

p = pd.read_csv(DATA, low_memory=False, usecols=["kab", "kec", "desa", "t", "diamati", "ltsj"])
p["vid"] = p.kab + "|" + p.kec + "|" + p.desa
p = p.sort_values(["vid", "t"]).reset_index(drop=True)
p["obs"] = (p.diamati > 0).astype(int)
p["atk"] = (p.ltsj.fillna(0) > 0).astype(int)

gaps = []
for _, g in p.groupby("vid", sort=False):
    run, out = 0, np.full(len(g), -1)
    for i, v in enumerate(g.obs.values):
        if v == 1:
            out[i] = run; run = 0
        else:
            run += 1
    gaps.append(pd.Series(out, index=g.index))
p["gap"] = pd.concat(gaps)
o = p[p.obs == 1]
cats = [(0, "0"), (1, "1"), (2, "2"), (3, "3")]
rate, nobs, lab = [], [], []
for k, s in cats:
    m = o.gap == k
    rate.append(o.loc[m, "atk"].mean()); nobs.append(int(m.sum())); lab.append(s)
m = o.gap >= 4
rate.append(o.loc[m, "atk"].mean()); nobs.append(int(m.sum())); lab.append("$\geq$4")

figA, axA = plt.subplots(figsize=(6.8, 3.35))
figB, axB = plt.subplots(figsize=(6.8, 3.60))

# ---- figure 1: the loop --------------------------------------------------
axA.set_xlim(-0.6, 10.4); axA.set_ylim(-1.35, 2.3); axA.axis("off")
per = np.arange(10)
axA.text(-0.5, 2.05, "Village A, visited every period", fontsize=10.9)
for x in per:
    axA.plot(x, 1.45, "o", ms=9.8, mfc="0.15", mec="0.15")
axA.text(-0.5, 0.42, "Village B, skipped then revisited", fontsize=10.9)
for x in per:
    if x < 4:
        axA.plot(x, -0.15, "o", ms=9.8, mfc="0.15", mec="0.15")
    elif x < 8:
        axA.plot(x, -0.15, "o", ms=9.8, mfc="white", mec="0.15", mew=0.9)
    else:
        axA.plot(x, -0.15, "o", ms=11.9, mfc="0.45", mec="black", mew=1.1)
axA.add_patch(Rectangle((4.15, 0.52), 3.5, 0.62, fc="white", ec="0.3", lw=0.8))
axA.text(5.9, 0.83, "farmer reports damage", ha="center", va="center", fontsize=10.2)
axA.add_patch(FancyArrowPatch((7.65, 0.83), (8.55, 0.12), arrowstyle="-|>",
                              mutation_scale=9, lw=0.9, color="0.2",
                              connectionstyle="arc3,rad=-0.3"))
axA.annotate("", xy=(9.9, -0.85), xytext=(-0.2, -0.85),
             arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"))
axA.text(4.9, -1.18, "survey periods (half-months)", ha="center", fontsize=10.2)
for xx, tt in [(0.15, "visited"), (3.3, "skipped"), (6.4, "visited after a gap")]:
    axA.plot(xx, -0.62, "o", ms=8.4,
             mfc={"visited": "0.15", "skipped": "white"}.get(tt, "0.45"),
             mec="0.15", mew=0.9)
    axA.text(xx + 0.25, -0.66, tt, fontsize=10.0, va="center")

# ---- figure 2: the gradient ----------------------------------------------
xs = np.arange(len(rate))
axB.bar(xs, rate, width=0.62, color="0.35", edgecolor="black", linewidth=0.7)
for x, r, n in zip(xs, rate, nobs):
    axB.text(x, r + 0.022, "%.3f" % r, ha="center", fontsize=10.2)
    axB.text(x, 0.035, "n=%s" % format(n, ","), ha="center", fontsize=10.0,
             color="white", rotation=90, va="bottom")
axB.axhline(rate[0], ls=(0, (4, 3)), lw=0.8, color="0.25")
axB.set_xticks(xs); axB.set_xticklabels(lab)
axB.set_xlabel("periods unvisited before arrival", fontsize=10.9)
axB.set_ylabel("$P(\mathrm{attack} \mid \mathrm{visited})$", fontsize=10.9)
axB.set_ylim(0, 1.06)
axB.tick_params(labelsize=10.2, length=2.5)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

OUT.mkdir(parents=True, exist_ok=True)
for f, name in ((figA, "fig_reactivity_loop"), (figB, "fig_reactivity_gap")):
    f.tight_layout()
    for ext in ("pdf", "png"):
        f.savefig(OUT / (name + "." + ext), dpi=400, bbox_inches="tight")
    print("ditulis:", OUT / (name + ".pdf"))
print("laju:", ["%.4f" % r for r in rate], "n:", nobs)

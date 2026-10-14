"""Figures 5 and 6: severity captured as a function of the patrol budget.

Read from eval_v7/capture_curve.csv at draw time, so the curves and the
numbers quoted in Section 4.4 cannot drift apart. IJIES does not accept one
figure carrying two lettered panels, so the four capture curves and the
s-hat minus aggregate difference are saved as two separate figures. The
difference needs its own axis because at the scale of the capture curves a
two-point gap is invisible, and the point of Section 3.7 is precisely how
large that gap is and where it sits.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11.6, "axes.linewidth": 0.7})

SRC = Path(os.environ["MADSCAG_WORK"]) / "eval_v7" / "capture_curve.csv"
OUT = Path(os.environ["FIGDIR"])

d = pd.read_csv(SRC)
piv = d.pivot(index="frac", columns="policy", values="severity_capture")
f = piv.index.to_numpy()

style = [
    ("model_sev",   "rank by $\\hat{s}$",             "#1b4965", "-",  2.0, "o"),
    ("model_agg",   "rank by $\\hat{p}\\hat{s}$",     "#5fa8d3", "--", 1.4, "s"),
    ("persistence", "persistence",                     "#c1666b", "-.", 1.4, "^"),
    ("random",      "random allocation",               "0.45",    ":",  1.4, None),
]

figA, axA = plt.subplots(figsize=(6.8, 4.05))
figB, axB = plt.subplots(figsize=(6.8, 3.85))

for col, lab, c, ls, lw, mk in style:
    axA.plot(f, piv[col], ls, color=c, lw=lw, label=lab,
             marker=mk, ms=3.6, markevery=2)

axA.axvline(0.50, color="0.75", lw=0.6, zorder=0)
axA.annotate("half budget", xy=(0.50, 0.045), fontsize=10.0, color="0.45",
             rotation=90, va="bottom", ha="right")
axA.set_xlabel("patrol budget $f$ (share of observed villages retained)")
axA.set_ylabel("severity captured $C(f)$")
axA.set_xlim(0.02, 0.98); axA.set_ylim(0, 1.0)
axA.legend(frameon=False, fontsize=10.2, loc="upper left", handlelength=2.4)
axA.tick_params(labelsize=10.2, length=2.5)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

diff = (piv["model_sev"] - piv["model_agg"]) * 100
axB.axhline(0, color="0.6", lw=0.7)
axB.plot(f, diff, "-", color="#1b4965", lw=1.6, marker="o", ms=3.6)
axB.fill_between(f, 0, diff, where=(diff > 0), color="#1b4965", alpha=0.12)
k = int(np.argmax(diff.to_numpy()))
axB.set_ylim(min(-0.4, float(diff.min()) - 0.2), float(diff.max()) + 0.75)
axB.text(0.96, float(diff.max()) + 0.55,
         f"peak {diff.iloc[k]:.1f} points at $f={f[k]:.2f}$",
         fontsize=10.0, color="#1b4965", ha="right", va="top")
axB.set_xlabel("patrol budget $f$")
axB.set_ylabel("$C_{\\hat{s}} - C_{\\hat{p}\\hat{s}}$  (points)")
axB.set_xlim(0.02, 0.98)
axB.tick_params(labelsize=10.2, length=2.5)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

OUT.mkdir(parents=True, exist_ok=True)
for figobj, name in ((figA, "fig_capture_curves"), (figB, "fig_capture_diff")):
    figobj.tight_layout()
    for ext in ("pdf", "png"):
        figobj.savefig(OUT / (name + "." + ext), dpi=400, bbox_inches="tight")
    print("ditulis:", OUT / (name + ".pdf"))
print("C(0.50):", {c: round(float(piv[c].loc[0.50]), 4) for c in piv.columns})
print("max diff:", round(float(diff.max()), 3), "pts at f =", float(f[k]))

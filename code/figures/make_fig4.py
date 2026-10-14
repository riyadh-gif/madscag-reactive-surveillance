"""Figures 7 and 8: the nine ablations read as paired-seed differences.

Each ablation is paired with the base run that shares its seed, so every
point is one (seed, horizon) cell and the run-to-run variation cancels
instead of being averaged away. Twelve cells per arm: four seeds times
three horizons.

IJIES does not accept one figure carrying two lettered panels, so the two
metrics are saved as two separate figures. The arm order is identical in
both, so an arm sits at the same height in each and the two can still be
read against one another.

Two metrics are plotted rather than one, because a single metric hides
the result. The two components that survive the sign test survive it on
different quantities, and each on the quantity it governs: the
missingness encoder on attendance, the zero-inflated head on the
severity output. Scoring only the composite error would report the
second and miss the first.

Sign counts are printed beside each arm rather than a mean with an error
bar, because with four seeds a mean invites a significance reading the
design cannot support. An arm is resolved only when all twelve cells
agree.
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

SRC = Path(os.environ["MADSCAG_WORK"]) / "eval_v7" / "metrics_full.csv"
OUT = Path(os.environ["FIGDIR"])
BASE = "BASE_v7"

# (column, output file stem, axis label, symlog threshold, x-limits)
PANELS = [
    ("auc_a", "fig_ablation_attendance",
     "change in attendance AUC\n(left of zero = removal hurts)",
     0.002, (-0.035, 0.016)),
    ("mae_c", "fig_ablation_error",
     "change in composite MAE\n(right of zero = removal hurts)",
     0.01, (-0.045, 0.90)),
]

LABEL = {
    "NO_ZI_B_v7":     "zero-inflated head",
    "NO_MPE_v7":      "missingness encoder",
    "NO_SPATIAL_v7":  "neighbour channel",
    "NO_PRESSURE_v7": "pressure channel",
    "NO_DIFF_v7":     "diffusion loss term",
    "BETA_OBS_v7":    "observation modulator (on)",
    "NO_GATE_v7":     "zero-inflation gate",
    "NO_ANOM_v7":     "anomaly channel",
    "NO_OBSFLAG_v7":  "observation flag (control)",
}
# one fixed row order across both panels, so an arm sits at the same
# height in each and the two can be read against one another
ORDER = list(LABEL)

d = pd.read_csv(SRC)
u = d[d.split == "uji"]

OUT.mkdir(parents=True, exist_ok=True)

for metric, stem, xlabel, thr, xlim in PANELS:
    fig, ax = plt.subplots(figsize=(6.8, 4.30))
    base = u[u.variant == BASE].set_index(["seed", "horizon"])[metric]
    ax.axvline(0, color="0.55", lw=0.8, zorder=1)
    for i, v in enumerate(reversed(ORDER)):
        col = u[u.variant == v].set_index(["seed", "horizon"])[metric]
        ix = col.index.intersection(base.index)
        diff = (col.loc[ix] - base.loc[ix]).to_numpy()
        n, pos, neg = len(diff), int((diff > 0).sum()), int((diff < 0).sum())
        zero = int((diff == 0).sum())
        if zero == n:
            c, mk, tag = "#8a6d3b", "D", "identical"
        elif max(pos, neg) == n:
            c, mk, tag = "#1b4965", "o", f"{n} / {n}"
        else:
            c, mk, tag = "0.6", "o", f"{pos}/{neg}"
        ax.scatter(diff, np.full(n, i) + np.linspace(-0.16, 0.16, n),
                   s=22, color=c, alpha=0.75, marker=mk, linewidths=0, zorder=3)
        ax.plot([diff.mean()] * 2, [i - 0.30, i + 0.30], color=c, lw=2.0,
                solid_capstyle="butt", zorder=4)
        ax.text(1.015, i, tag, transform=ax.get_yaxis_transform(),
                va="center", fontsize=10.0, color=c)
    ax.set_xscale("symlog", linthresh=thr, linscale=0.9)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=10.0)
    ax.tick_params(labelsize=10.0, length=2.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[v] for v in reversed(ORDER)], fontsize=10.0)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)

    h = [plt.Line2D([], [], marker="o", ls="", color="#1b4965", ms=5.6,
                    label="all twelve cells agree in sign"),
         plt.Line2D([], [], marker="o", ls="", color="0.6", ms=5.6,
                    label="sign not resolved by four seeds"),
         plt.Line2D([], [], marker="D", ls="", color="#8a6d3b", ms=5.6,
                    label="no path to the output")]
    fig.legend(handles=h, frameon=False, fontsize=10.0, ncol=1,
               loc="lower center", bbox_to_anchor=(0.55, -0.10))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / (stem + "." + ext), dpi=400, bbox_inches="tight")
    print("ditulis:", OUT / (stem + ".pdf"))
    plt.close(fig)
for metric, *_ in PANELS:
    base = u[u.variant == BASE].set_index(["seed", "horizon"])[metric]
    print(f"--- {metric} ---")
    for v in ORDER:
        s = u[u.variant == v].set_index(["seed", "horizon"])[metric]
        ix = s.index.intersection(base.index)
        df = (s.loc[ix] - base.loc[ix])
        print(f"  {v:16s} +{int((df>0).sum()):2d} -{int((df<0).sum()):2d} "
              f"={int((df==0).sum()):2d}  mean={df.mean():+.4f}")

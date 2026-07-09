#!/usr/bin/env python3
"""figs/kernel_landscape.png -- transition-kernel landscape, THREE properties:
  x  = explicitness of the one-step transition kernel K(y|x)  (implicit -> exact closed-form)
  y  = smile + dynamics flexibility                            (no smile -> free dynamics)
  fill = exactness of MARGINAL fit: solid=exact (any arb-free smile), light=constrained,
         hollow=parametric/approximate; grey hollow = N/A (no smile, Black-Scholes).
De-hyped (no highlight box, plain markers). SANOS is a real discrete Markov martingale (explicit,
rigid, exact-marginal); SANOS-Evolve replaces its rigid coupling with a free-dynamics GM kernel."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "..", "figs"); os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C0, GREY = "#1f6f8b", "#8a8a8a"                     # teal edge, grey
EXFILL, PALE = "#8ec6da", "#ddeef5"                 # light-blue fill (exact), very pale (constrained)

fig, ax = plt.subplots(figsize=(9.0, 6.1))

XB = [2.85, 5.2, 7.5]          # explicitness tier boundaries
YB = [2.7, 5.05, 7.25]         # flexibility tier boundaries
XC = [1.6, 4.0, 6.4, 8.7]      # tier centres
YC = [1.4, 4.0, 6.1, 8.4]
for xb in XB:
    ax.plot([xb, xb], [0, 10.0], color="#e2e2e2", lw=0.8, ls=(0, (4, 3)), zorder=0)
for yb in YB:
    ax.plot([0, 10.4], [yb, yb], color="#e2e2e2", lw=0.8, ls=(0, (4, 3)), zorder=0)


def marker(x, y, fit, s=86, z=3):
    if fit == "exact":
        ax.scatter([x], [y], s=s, facecolors=EXFILL, edgecolors=C0, linewidths=1.4, zorder=z)
    elif fit == "constrained":
        ax.scatter([x], [y], s=s, facecolors=PALE, edgecolors=C0, linewidths=1.4, zorder=z)
    elif fit == "approx":
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=C0, linewidths=1.5, zorder=z)
    else:  # na
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=GREY, linewidths=1.4, zorder=z)


# (x, y, label, fit, ha, dx)
fam = [
    (1.7, 8.4, "Schrodinger bridge / MOT",      "exact",       "left",  0.30),
    (2.7, 7.4, "perturbed OT (JPM)",            "exact",       "left",  0.30),
    (1.7, 6.2, "LSV (backbone prior)",          "exact",       "left",  0.30),
    (2.5, 6.7, "Quintic OU (2-factor SV)",      "approx",      "left",  0.28),
    (4.1, 6.0, "Markov-functional / Bass",      "exact",       "left",  0.30),
    (4.0, 3.6, "LVG (Carr-Nadtochiy)",          "exact",       "left",  0.30),
    (6.0, 6.4, "affine (Heston) / Fourier LSV", "approx",      "left",  0.30),
    (6.0, 4.7, "Gram-Charlier",                 "approx",      "left",  0.30),
    (6.8, 4.1, "heat-kernel expansion",         "approx",      "left",  0.30),
    (6.4, 3.2, "Levy / additive",               "approx",      "left",  0.30),
    (2.1, 3.3, "BM mixture LV",                 "constrained", "left",  0.30),
    (8.7, 1.4, "Black-Scholes",                 "na",          "right", -0.30),
    (8.7, 3.5, "SANOS (discrete local vol)",    "exact",       "left",  0.30),
    (8.7, 8.0, "SANOS-Evolve (this work)",      "exact",       "left",  0.30),
]
JOINT = {"Schrodinger bridge / MOT", "perturbed OT (JPM)", "Quintic OU (2-factor SV)"}
for x, y, lab, fit, ha, dx in fam:
    marker(x, y, fit)
    if lab in JOINT:                       # black x = joint SPX-VIX calibration (hard targets)
        ax.scatter([x], [y], s=26, marker="x", color="black", linewidths=1.1, zorder=6)
    ax.text(x + dx, y, lab, fontsize=7.6, ha=ha, va="center", color="#222222")

# fill legend (bottom-left interior)
lx, lyc = 0.5, [1.85, 1.35, 0.85, 0.35]
marker(lx, lyc[0], "exact", s=72, z=4)
ax.text(lx + 0.35, lyc[0], "exact marginals (any arbitrage-free smile)", fontsize=7, va="center", color="#333")
marker(lx, lyc[1], "constrained", s=72, z=4)
ax.text(lx + 0.35, lyc[1], "constrained marginals", fontsize=7, va="center", color="#333")
marker(lx, lyc[2], "approx", s=72, z=4)
ax.text(lx + 0.35, lyc[2], "parametric / approximate marginals", fontsize=7, va="center", color="#333")
ax.scatter([lx], [lyc[3]], s=26, marker="x", color="black", linewidths=1.1, zorder=4)
ax.text(lx + 0.35, lyc[3], "joint SPX–VIX calibration (both smiles hard targets)", fontsize=7, va="center", color="#333")

# axes as arrows + ordinal tier labels
ax.set_xlim(-2.25, 11.4); ax.set_ylim(-2.0, 10.4)
ax.set_xticks([]); ax.set_yticks([])
ax.annotate("", xy=(11.0, 0.0), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
ax.annotate("", xy=(0.0, 10.2), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)

xtl = ["implicit\n(PDE / opt.)", "root-search /\n1-D solve", "transform /\nseries", "exact closed-\nform kernel"]
for xc, t in zip(XC, xtl):
    ax.text(xc, -0.45, t, ha="center", va="top", fontsize=7, color="#555")
ax.text(5.1, -1.65, r"explicitness of the transition kernel $K(y\,|\,x)$  →", ha="center", va="top", fontsize=9)

ytl = ["no smile", "smile only\n(rigid dyn.)", "smile +\ncoupled dyn.", "smile +\nfree dyn."]
for yc, t in zip(YC, ytl):
    ax.text(-0.45, yc, t, ha="right", va="center", fontsize=7, color="#555")
ax.text(-2.05, 5.0, "smile + dynamics flexibility  →", ha="center", va="center", rotation=90, fontsize=9)

fig.tight_layout()
out = os.path.join(FIGS, "kernel_landscape.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print("wrote", out)

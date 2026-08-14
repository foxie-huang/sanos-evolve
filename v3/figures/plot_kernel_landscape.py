#!/usr/bin/env python3
"""figs/kernel_landscape.png -- schematic positioning of transition-kernel constructions.

Redesigned from v2/data/make_landscape_fig.py. Three changes matter, all to stop the figure
claiming more than the paper does:

  y-axis   was "smile + dynamics flexibility" (top row "free dynamics"), which contradicted the
           paper's own evidence -- bounded static-preserving SSR range, two-timescale decay floor,
           frozen conditional shape, a seven-free-parameter family explicitly NOT claimed dense.
           Now: the RELATIONSHIP between marginal fit and dynamic selection. Categorical, checkable,
           and it puts MOT/bridges and SANOS-Evolve in the same band for a true reason.
  brace    the paper's actual distinction -- WHAT selects the dynamics -- was carried only by dot
           labels. A brace now splits the top band into exogenous-criterion vs observed-readout
           selection, so the geometry carries the claim.
  fill     was "exactness of the marginal fit", which a filled SANOS-Evolve marker could be read as
           claiming exact propagation to any marginal. The paper reports a finite propagation
           residual (3 call-bp / 26 IV-bp). Fill now records the SCOPE of the marginal family.

Also: the joint SPX/VIX cross overlay is gone (folded into labels, freeing the brace as the one
strong non-axis device), Bass is dropped from the scatter and kept in the prose (it does not belong
on the same point as Markov-functional -- they differ on the very axis being plotted), and the
encyclopedic points (Gram-Charlier, heat-kernel, LVG, Levy, BM mixture) are dropped so that every
remaining dot supports the paper's positioning.

    python3 plot_kernel_landscape.py     # -> manuscript_v3/figs/kernel_landscape.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import sys  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

FIGS = os.path.normpath(os.path.join(HERE, "..", "manuscript_v3", "figs"))
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C0, GREY = "#1f6f8b", "#8a8a8a"
FILL_EXT, FILL_CON = "#8ec6da", "#ddeef5"

fig, ax = plt.subplots(figsize=(9.4, 6.4))

XB = [2.9, 5.3, 7.6]                     # explicitness tier boundaries
YB = [2.6, 4.9, 7.1]                     # relationship tier boundaries
for xb in XB:
    ax.plot([xb, xb], [0, 10.2], color="#e4e4e4", lw=0.8, ls=(0, (4, 3)), zorder=0)
for yb in YB:
    ax.plot([0, 10.6], [yb, yb], color="#e4e4e4", lw=0.8, ls=(0, (4, 3)), zorder=0)


def marker(x, y, scope, s=92, z=3):
    """fill = SCOPE of the marginal family, not exactness of the fit."""
    if scope == "external":        # externally prescribed, nonparametric, convex-ordered
        ax.scatter([x], [y], s=s, facecolors=FILL_EXT, edgecolors=C0, linewidths=1.4, zorder=z)
    elif scope == "constrained":   # flexible but structurally constrained class
        ax.scatter([x], [y], s=s, facecolors=FILL_CON, edgecolors=C0, linewidths=1.4, zorder=z)
    elif scope == "parametric":    # parametric / approximate family
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=C0, linewidths=1.5, zorder=z)
    else:                          # no smile
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=GREY, linewidths=1.4, zorder=z)


#  BAND CENTRES: every dot sits on its band centre, so within-band height carries no meaning --
#  EXCEPT in the top band, where the two dots are deliberately offset and the caption declares what
#  the offset means (how completely the selection determines the coupling: a transport criterion
#  fixes the whole coupling, a finite readout vector narrows a restricted family to a
#  readout-equivalence class within it).
BAND = {"top_hi": 9.35, "top_lo": 8.45, "joint": 6.00, "inherited": 3.70, "nosmile": 1.35}

#      x     y                 label                            scope        pos  sub
fam = [
    (1.75, BAND["top_hi"], "martingale optimal transport",      "external",   "r",
     "an exogenous cost or reference law\nfixes the whole coupling"),
    (9.05, BAND["top_lo"], "SANOS-Evolve (this work)",          "external",   "l",
     "two readout term structures constrain a readout-equivalence\nclass in a seven-free-parameter family"),
    (1.75, BAND["joint"], "SLV, prescribed backbone",           "constrained","a", None),
    (5.40, BAND["joint"], "Markov-functional",                  "constrained","a", None),
    (3.60, BAND["joint"], "Quintic OU",                         "parametric", "b",
     "SPX and VIX surfaces, with skew\nand SSR targeted jointly"),
    (7.30, BAND["joint"], "affine (Heston) LSV",                "parametric", "b", None),
    (9.05, BAND["inherited"], "SANOS (discrete local volatility)", "external", "l",
     "coupling determined by the marginals"),
    (9.05, BAND["nosmile"], "Black\u2013Scholes",               "na",         "l", None),
]
for x, y, lab, scope, pos, sub in fam:
    marker(x, y, scope)
    if pos in ("r", "l"):
        dx, ha = (0.32, "left") if pos == "r" else (-0.32, "right")
        ax.text(x + dx, y, lab, ha=ha, va="center", fontsize=9.2)
        if sub:
            ax.text(x + dx, y - 0.40, sub, ha=ha, va="top", fontsize=7.4,
                    color="#555", linespacing=1.3)
    elif pos == "a":                                   # label above the dot, centred
        ax.text(x, y + 0.30, lab, ha="center", va="bottom", fontsize=9.2)
        if sub:
            ax.text(x, y + 0.66, sub, ha="center", va="bottom", fontsize=7.4,
                    color="#555", linespacing=1.3)
    else:                                              # label below the dot, centred
        ax.text(x, y - 0.30, lab, ha="center", va="top", fontsize=9.2)
        if sub:
            ax.text(x, y - 0.66, sub, ha="center", va="top", fontsize=7.4,
                    color="#555", linespacing=1.3)

# ---- the brace: what selects the dynamics, inside the top band ---------------------------------
ax.annotate("", xy=(0.75, 7.50), xytext=(3.6, 7.50),
            arrowprops=dict(arrowstyle="-", color="#8a8a8a", lw=1.0,
                            connectionstyle="bar,fraction=-0.12"))
ax.text(2.15, 7.23, "exogenous selection criterion", ha="center", va="top",
        fontsize=8.2, style="italic", color="#444")
ax.annotate("", xy=(7.6, 7.50), xytext=(10.0, 7.50),
            arrowprops=dict(arrowstyle="-", color="#333", lw=1.2,
                            connectionstyle="bar,fraction=-0.14"))
ax.text(8.8, 7.23, "observed dynamic readouts", ha="center", va="top",
        fontsize=8.2, style="italic", color="#111")

# ---- axes ---------------------------------------------------------------------------------------
ax.set_xlim(0.0, 10.9); ax.set_ylim(0.0, 10.2)
ax.set_xticks([1.45, 4.1, 6.45, 9.05])
ax.set_xticklabels(["implicit\n(PDE / optimisation /\nsimulation)", "numerical solve /\n1-D inversion",
                    "transform /\nseries", "explicit finite\ntransition law"], fontsize=8.4)
ax.set_yticks([1.35, 3.70, 6.00, 8.90])
ax.set_yticklabels(["no smile\ndynamics", "dynamics largely\ninherited from the\nsmile specification",
                    "smile and dynamics\njointly calibrated",
                    "marginals fixed;\ndynamics selected\nseparately"], fontsize=8.4)
ax.set_xlabel("computational explicitness of the one-step transition law  $\\rightarrow$", fontsize=9.6)
ax.set_ylabel("relationship between marginal fit and dynamic selection  $\\rightarrow$", fontsize=9.6)
ax.tick_params(length=0)

# ---- legend: scope of the marginal family --------------------------------------------------------
leg_y, leg_x = 2.35, 0.95
for i, (scope, txt) in enumerate([
        ("external",    "externally prescribed, nonparametric convex-ordered marginals"),
        ("constrained", "flexible but structurally constrained marginal class"),
        ("parametric",  "parametric or approximate marginal family")]):
    marker(leg_x, leg_y - 0.52 * i, scope, s=64, z=4)
    ax.text(leg_x + 0.30, leg_y - 0.52 * i, txt, ha="left", va="center", fontsize=7.8, color="#333")

fig.tight_layout()
out = os.path.join(FIGS, "kernel_landscape.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print("wrote", out)
print(f"  {len(fam)} constructions; joint SPX/VIX folded into labels; Bass dropped from the scatter")

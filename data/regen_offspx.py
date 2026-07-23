#!/usr/bin/env python3
"""Isolated regenerator for figs/real_offspx.png (Figure `fig:real-offspx`) ONLY, so the other three
paper figures (real_spx / real_hedge / real_dynamics) are never rewritten. This is a verbatim copy of
the FIGURE-2 block of generate_figs.py (all panels hard-coded, no live data); panel (a) is the
multi-start cross-regime RMS matching tab:crossregime. Keep in sync with generate_figs.py fig 2."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "..", "figs"); os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C0, C1, C2 = "#1f6f8b", "#b02a37", "#9a6b00"

fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))

# (a) SPX cross-regime fit RMS  (multi-start, w_vov=0.8; matches tab:crossregime)
yr = ["2015", "2019", "2020", "2022", "2023"]
spx_ssr = [0.6, 2.3, 3.8, 3.2, 9.2]; spx_vix = [1.2, 2.6, 9.0, 22.2, 22.6]
x = np.arange(len(yr)); w = 0.38
ax[0, 0].bar(x - w / 2, spx_ssr, w, color=C0, label="SSR RMS")
ax[0, 0].bar(x + w / 2, spx_vix, w, color=C1, label="VIX vov RMS")
ax[0, 0].set(title="(a) SPX SSR+vov fit by regime", ylabel="RMS (%)"); ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(yr)
ax[0, 0].legend(frameon=False); ax[0, 0].axhline(5, ls=":", c="gray", lw=0.8)
ax[0, 0].annotate("two-factor\nvov ceiling", (3.0, 22.2), (1.15, 18.5), fontsize=7.5, color=C2,
                  arrowprops=dict(arrowstyle="->", color=C2, lw=0.8))

# (b) NDX off-SPX fit RMS
nyr = ["2018", "2020", "2021", "2022"]; ndx_ssr = [8.9, 11.8, 5.0, 6.4]; ndx_vov = [4.9, 8.1, 18.7, 1.1]
xn = np.arange(len(nyr))
ax[0, 1].bar(xn - w / 2, ndx_ssr, w, color=C0, label="SSR RMS")
ax[0, 1].bar(xn + w / 2, ndx_vov, w, color=C1, label="vov RMS")
ax[0, 1].set(title="(b) NDX off-SPX fit (no VIX)", ylabel="RMS (%)"); ax[0, 1].set_xticks(xn); ax[0, 1].set_xticklabels(nyr)
ax[0, 1].legend(frameon=False)
ax[0, 1].annotate("vov shape\nceiling", (2, 19.5), (2.4, 15), fontsize=7.5, color=C2,
                  arrowprops=dict(arrowstyle="->", color=C2, lw=0.8))

# (c) NDX vs SPX realised vol-of-vol (cross-sectional)
ax[1, 0].bar([0], [1.16], 0.5, color=C0, label="SPX")
ax[1, 0].bar([1], [1.62], 0.5, color=C1, label="NDX")
ax[1, 0].set(title="(c) Realised vol-of-vol: NDX $>$ SPX", ylabel="realised vov (30d)")
ax[1, 0].set_xticks([0, 1]); ax[1, 0].set_xticklabels(["SPX", "NDX"]); ax[1, 0].set_ylim(0, 1.9)
for xi, v in [(0, 1.16), (1, 1.62)]:
    ax[1, 0].text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)

# (d) E[nu|z] second-order leverage correction (from-spot ATM, SPX 2015)
mw = [4, 8, 13]; lead = [0.097, 0.107, 0.122]; corr = [0.108, 0.119, 0.133]; real = [0.122, 0.129, 0.135]
ax[1, 1].plot(mw, real, "o-", c=C0, label="realised")
ax[1, 1].plot(mw, corr, "s--", c=C1, label=r"$E[\nu|z]$-corrected")
ax[1, 1].plot(mw, lead, "^:", c=C2, label="leading order")
ax[1, 1].set(title=r"(d) From-spot ATM: $E[\nu|z]$ correction", xlabel="maturity (weeks)", ylabel="from-spot ATM vol")
ax[1, 1].legend(frameon=False)

fig.tight_layout(); fig.savefig(os.path.join(FIGS, "real_offspx.png"), bbox_inches="tight"); plt.close(fig)
print("wrote figs/real_offspx.png (isolated; other 3 figures untouched)")

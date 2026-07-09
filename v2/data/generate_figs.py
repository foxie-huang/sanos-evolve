#!/usr/bin/env python3
"""Generate the real-data empirical figures for disc_SLV.tex (figs/real_spx.png, figs/real_offspx.png)
from the actual run outputs. Numbers not in JSON are taken verbatim from the logged runs."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "..", "figs"); os.makedirs(FIGS, exist_ok=True)   # disc_SLV/figs (paper root)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C0, C1, C2 = "#1f6f8b", "#b02a37", "#9a6b00"

R34 = json.load(open(os.path.join(HERE, "real_pipeline_34_results.json")))

# ---------- FIGURE 1: SPX real-data calibration ----------
fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))

# (a) SSR term structure, 2019 (diag_2023)
wk = [1, 2, 4, 8, 13]
ssr_emp = [1.912, 1.706, 1.654, 1.629, 1.568]; ssr_mdl = [1.852, 1.748, 1.703, 1.630, 1.532]
ax[0, 0].plot(wk, ssr_emp, "o-", c=C0, label="realised")
ax[0, 0].plot(wk, ssr_mdl, "s--", c=C1, label="model")
ax[0, 0].set(title="(a) SSR term structure  (SPX 2019)", xlabel="maturity (weeks)", ylabel="SSR")
ax[0, 0].legend(frameon=False)

# (b) VIX vol-of-vol term structure, 2019
dte = [10, 17, 24, 31, 45, 80, 108, 136, 171]
vov_d = [1.177, 1.064, 1.03, 0.989, 0.879, 0.724, 0.655, 0.592, 0.543]
vov_m = [1.121, 1.111, 1.045, 0.984, 0.884, 0.729, 0.652, 0.594, 0.538]
ax[0, 1].plot(dte, vov_d, "o-", c=C0, label="VIX-implied (data)")
ax[0, 1].plot(dte, vov_m, "s--", c=C1, label="model")
ax[0, 1].set(title="(b) VIX vol-of-vol  (SPX 2019)", xlabel="VIX-option dte", ylabel=r"vov ($\xi$)")
ax[0, 1].legend(frameon=False)

# (c) forward-skew roll-down (step 4), a few 2015 dates
xs = [1, 2, 3, 4]; labs = ["1m2m", "1m3m", "3m6m", "6m1y"]
for date, c in [("2015-06-01", C0), ("2015-07-01", C1), ("2015-08-03", C2)]:
    ax[1, 0].plot(xs, [s["fwd_skew"] for s in R34[date]["step4"]], "o-", c=c, label=date[5:])
ax[1, 0].set(title="(c) Forward skew rolls down", xlabel="forward window", ylabel="forward ATM skew")
ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(labs); ax[1, 0].legend(frameon=False, title="2015")

# (d) marginal-propagation residual (step 3), 2015-06-01
s3 = R34["2015-06-01"]["step3"]; dtes = [r["dte"] for r in s3]; cbp = [r["call_bp"] for r in s3]
ax[1, 1].plot(dtes, cbp, "o-", c=C0)
ax[1, 1].axvspan(0, 190, color=C0, alpha=0.06); ax[1, 1].text(90, max(cbp) * 0.9, r"liquid $\leq$6m", ha="center", fontsize=8, color=C0)
ax[1, 1].set(title="(d) Marginal residual $d(\\mu_j\\mathcal{K}_j,\\mu_{j+1})$", xlabel="expiry (dte)", ylabel="call-price residual (bp)")

fig.tight_layout(); fig.savefig(os.path.join(FIGS, "real_spx.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIGURE 2: cross-regime + off-SPX ----------
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

# ---------- FIGURE 3: SSR-consistent hedging ----------
import hedging_backtest as HB
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
Rg, sr, ssr = HB.sweep("2019")
ax[0].plot(Rg, sr, c=C0, lw=1.8)
for R, lab, c in [(-1, "sticky-$\\delta$", C2), (0, "Black", "gray"), (1.6, "SSR", C1)]:
    ax[0].axvline(R, ls="--", c=c, lw=1)
    ax[0].text(R, 1.03, lab, rotation=90, va="bottom", ha="right", fontsize=8, color=c)
ax[0].axhline(1.0, ls=":", c="gray", lw=0.8)
ax[0].set(title="(a) Hedging variance vs $R$  (SPX 2019, 1m ATM)", xlabel="SSR-delta parameter $R$",
          ylabel="hedging std / Black")
red = HB.reductions(["2016", "2017", "2019", "2021"]); yr = list(red); x = np.arange(len(yr)); w = 0.38
ax[1].bar(x - w / 2, [red[y]["red_ssr"] * 100 for y in yr], w, color=C0, label="SSR-consistent")
ax[1].bar(x + w / 2, [red[y]["red_sd"] * 100 for y in yr], w, color=C1, label="sticky-delta")
ax[1].axhline(0, c="k", lw=0.6)
ax[1].set(title="(b) Variance reduction vs Black", ylabel="reduction (%)")
ax[1].set_xticks(x); ax[1].set_xticklabels(yr); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "real_hedge.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIGURE 4: smile dynamics (sticky-moneyness check, from sticky_check.py) ----------
SD = json.load(open(os.path.join(HERE, "sticky_check_results.json")))
labs = ["1wk", "1m", "2m", "3m"]; wks = [1, 4, 8, 13]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(wks, [SD[l]["SSR"] for l in labs], "o-", c=C0, zorder=3, label="model")
for y, lab, c in [(0, "sticky-moneyness", C2), (1, "sticky-strike", "gray"), (2, "local vol", C1)]:
    ax[0].axhline(y, ls="--", c=c, lw=1); ax[0].text(13.3, y, lab, va="center", fontsize=7.5, color=c)
ax[0].set(title="(a) Level: SSR term structure", xlabel="maturity (weeks)", ylabel="SSR",
          xlim=(0, 19), ylim=(-0.2, 2.35)); ax[0].legend(frameon=False, loc="center right")
d = SD["1m"]; k = np.linspace(-0.05, 0.05, 101); r = -0.01                      # 1m smile response to a -1% move
ax[1].plot(k * 100, r * (d["beta"] + d["d_skew"] * k + 0.5 * d["d_curv"] * k ** 2) * 100, c=C0, lw=1.8, label="model")
ax[1].axhline(0, ls="--", c=C2, lw=1.2, label="sticky-moneyness")
ax[1].plot(k * 100, r * (d["skew0"] + d["curv0"] * k) * 100, ls="--", c="gray", lw=1.2, label="sticky-strike")
ax[1].set(title=r"(b) Shape: 1m smile shift on $-1\%$ spot", xlabel="log-moneyness $k$ (%)",
          ylabel=r"$\Delta\sigma(k)$ (vol pts)"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "real_dynamics.png"), bbox_inches="tight"); plt.close(fig)
print("wrote figs/real_spx.png, figs/real_offspx.png, figs/real_hedge.png, figs/real_dynamics.png")

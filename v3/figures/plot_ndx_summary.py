#!/usr/bin/env python3
"""One-panel NDX summary for the body of the paper.

Replaces the nine-panel NDX SSR grid in Section 5.4 (that grid moves to the supplement). The panel
carries the whole finding of the subsection in one view:

    the SSR channel transfers   -> every year's SSR RMS falls inside its own HAC sampling band
    the forward-variance does not -> three steep-decay years sit far above their band

Numbers are read from the SAME record that backs Table 9, not transcribed from it:

    ts2_ndx_<date>.json              uniform two-stage panel, exact-joint forward-variance block (7 years)
    ts2allseed_ndx_2016-06-01.json   2016: basin selected at the reported n_F=17 (see Appendix H)
    ndx_figdata.json                 per-year Newey-West HAC floors

The 2016 exception is asserted rather than assumed: the record must contain a surviving seed, and
the one quoted is the lowest stage-2 objective among them.

    python3 plot_ndx_summary.py            # -> manuscript_v3/figs/ndx_summary.png
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

OUT  = os.path.normpath(os.path.join(HERE, "..", "manuscript_v3", "figs", "ndx_summary.png"))

# SOURCE: the shipped panel record, not per-date files transcribed by hand. `_c9` is the eight-tenor
# fit on the corrected realised target (const-maturity + physical bound + splice + churn removal,
# then the observation-operator correction); see Appendix D. The panel years are the fit dates, so
# 2018 is present and 2023 is not -- that follows the record rather than the earlier figure's set.
PANEL = os.path.join(HERE, "panel_c9_NDX.json")
_p = json.load(open(PANEL))["dates"]
DATES = sorted(_p)
YEARS = [d[:4] for d in DATES]
REGIME = {"2012": "flattest", "2016": "steep", "2017": "steep", "2018": "", "2019": "moderate",
          "2020": "COVID", "2021": "melt-up", "2022": "bear", "2024": "low-vol"}

ssr = np.array([_p[d]["ssr_rms"] for d in DATES])
vov = np.array([_p[d]["vov_rms"] for d in DATES])
_jh = json.load(open(os.path.join(HERE, "ssr_joint_hac.json")))["NDX"]
flr = np.array([_jh[y]["floor_joint"] for y in YEARS])

# The claim the panel makes; assert it rather than trust it.
assert (ssr < flr).all(), f"a year's SSR RMS is NOT inside its HAC s.e.: {dict(zip(YEARS, ssr - flr))}"

# The forward-variance target carries ~24% RMS of its own roughness (Appendix D), so the residual is
# drawn against THAT rather than against zero -- the earlier figure annotated three years as outliers
# against an implicit floor of zero, which the measurement does not support.
ROUGH = 24.0
print(f"SSR RMS  {ssr.min():.1f}-{ssr.max():.1f}%   HAC floors {flr.min():.1f}-{flr.max():.1f}%")
print(f"vov RMS  {vov.min():.1f}-{vov.max():.1f}%   target roughness {ROUGH:.0f}%")

x = np.arange(len(YEARS))
C_BAND, C_SSR, C_VOV = "#9fbfcc", "#1f6f8b", "#b02a37"
fig, ax = plt.subplots(figsize=(7.2, 3.4))

ax.bar(x, flr, width=0.74, color=C_BAND, alpha=.45, lw=0, zorder=1,
       label="realised-SSR target s.e. (Newey-West joint HAC)")
ax.plot(x, ssr, "o", color=C_SSR, ms=7, zorder=3, label="SSR fit RMS")
ax.plot(x, vov, "s", color=C_VOV, ms=6.5, zorder=3, mfc="white", mew=1.6,
        label="variance-of-variance fit RMS")
ax.axhline(ROUGH, color=C_VOV, ls=(0, (5, 3)), lw=1.0, alpha=.55, zorder=2,
           label="forward-variance target roughness (panel average)")

ax.set_xticks(x)
ax.set_xticklabels([f"{y}\n{REGIME[y]}" for y in YEARS], fontsize=8)
ax.set_ylabel("fit RMS (%)", fontsize=9)
ax.set_ylim(0, max(vov.max(), flr.max(), ROUGH) * 1.18)
ax.grid(axis="y", alpha=.25, zorder=0)
ax.tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)

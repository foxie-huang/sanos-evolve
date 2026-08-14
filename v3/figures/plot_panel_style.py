#!/usr/bin/env python3
r"""THE standard SPX/NDX term-structure panel. Every SSR/vov figure uses this -- do not reinvent it.

Drawing grammar copied from v3_scripts/figures/plot_ts2_panel.py so the figures read as the same
family: 3x3 grid, log-x SSR with week labels, JOINT-HAC band + errorbars, diffusive SSR->2 line,
linear-x vov with no band (a single-day Q read carries no sampling error), regime + RMS in each
title, legend in panel 0 only, shared y-label.

Differences from v2, both deliberate:
  * ONE model series (_ref), not theta@7 vs theta@17 -- keeps the @17 styling (red squares, dashed).
  * Year set is ours: 2018 present, 2023 absent. 2018 has no published regime label, left blank
    rather than invented.

Band is `ssr_joint_hac.json` -> se_joint, NOT the block bootstrap: the delta-method se treats the
regression slope and the skew denominator as independent and they are not; ssr_joint_hac HACs their
joint long-run covariance. Point estimates unchanged, only the half-width.

Parameterised 2026-08-10. It used to hardcode `panel_ref_SPX.json` and was re-pointed by `sed`-ing a
copy per panel, which is how a "standard style" quietly forks. Defaults reproduce the original _ref
figures byte-for-byte; everything else is a flag.

    python3 plot_panel_style.py                                    # the _ref figures, unchanged
    python3 plot_panel_style.py --panel panel_g9_SPX.json --prefix g9 \
        --note "gated $\lambda$" --gate                            # per-date gate marked in titles
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.environ.get("SANOS_DATA") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"))

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="panel_ref_SPX.json")
ap.add_argument("--prefix", default=None, help="output stem; default keeps the _ref filenames")
ap.add_argument("--note", default=None, help="extra clause appended to both suptitles")
ap.add_argument("--gate", action="store_true",
                help="mark each panel with which lambda variant the per-date gate ran")
# --compare overlays a SECOND model panel in the established two-model grammar (primary red dashed,
# comparison gold dotted) already used for the 2016 like-for-like figure. Same drawing rules, one
# extra series -- not a new look.
ap.add_argument("--compare", default=None, help="second panel_*.json to overlay as the comparison")
ap.add_argument("--compare-label", dest="clabel", default="comparison")
ap.add_argument("--label", default="model", help="legend label for the primary series")
ap.add_argument("--ticker", default="SPX", help="selects the HAC band block and the regime labels")
# --vovcurve draws the model's vov across the WHOLE tenor grid instead of only the fitted anchors.
# The curve is NOT computed here: plot_panel_style stays a pure-matplotlib reader, and the kernel
# evaluation lives in diagnostics/plot_ndx_oos_kf.py, which dumps ndx_oos_curves<TAG>.json. Pass that
# file. Its reach is whatever LADDER that run used -- 56 rungs to get 365d, 42 stops at 267d.
ap.add_argument("--vovcurve", default=None, help="ndx_oos_curves<TAG>.json: model vov across tenors")
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
A = ap.parse_args()
OUT = A.out

plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C_OBS, C_MOD, C_CMP = "#1f6f8b", "#b02a37", "#c98500"
WKS = [1, 2, 4, 8, 13]
WLAB = ["1wk", "2wk", "1m", "2m", "3m"]

_P = json.load(open(os.path.join(D, A.panel)))
R = _P["dates"]
CMP = json.load(open(os.path.join(D, A.compare)))["dates"] if A.compare else {}
GATE = _P.get("lam_gate", {}) if A.gate else {}
VC = json.load(open(os.path.join(D, A.vovcurve) if not os.path.isabs(A.vovcurve) else A.vovcurve)) \
    if A.vovcurve else {}
_TK = A.ticker.upper()
JH = json.load(open(os.path.join(D, "ssr_joint_hac.json")))[_TK]
CT = json.load(open(os.path.join(D, "corrected_targets.json")))[_TK]
# Regime labels per ticker. NDX labels follow the v3 paper's own table; 2018 has none there (the
# paper's NDX panel does not include it) so it is left blank rather than invented.
REG = {"2012": "flattest", "2016": "steep", "2017": "steep/calm", "2018": "",
       "2019": "moderate", "2020": "COVID", "2021": "high-flat", "2022": "bear grind",
       "2024": "steep low-vol"} if _TK == "SPX" else {
       "2012": "flattest", "2016": "steep", "2017": "steep", "2018": "",
       "2019": "moderate", "2020": "COVID", "2021": "melt-up", "2022": "bear",
       "2024": "low-vol"}
YRS = [d[:4] for d in R]
BY = {d[:4]: r for d, r in R.items()}
GT = {d[:4]: v for d, v in GATE.items()}
BC = {d[:4]: r for d, r in CMP.items()}


_EST = {}                      # yr -> max % disagreement between the fitted target and the band centre


def tag(yr):
    """Which lambda the gate ran at this date. Empty unless --gate, so the default look is untouched."""
    if yr not in GT or GT[yr] is None:
        return ""
    return "   [$\\lambda$ smoothed]" if GT[yr] else "   [$\\lambda$ raw]"


def grid(kind, fname, suptitle):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6))
    for ax, yr in zip(axes.flat, YRS):
        r = BY[yr]
        if kind == "ssr":
            x = np.array(WKS, float)
            tgt = np.array(r["ssr_target"], float)
            se = np.array(JH[yr]["se_joint"], float)
            # The band is the sampling error of the estimator behind JH[yr]["R"]. Where the FITTED
            # target was produced by a DIFFERENT estimator the two are not the same object, and the
            # panel must say so rather than draw a band that silently belongs to something else.
            # This is real and documented: on NDX 2017 a large anti-leverage one-week print survives
            # the filters, so the target is the Huber-robust beta (1.766 at 1wk) while `R` is still
            # OLS (1.414, and inverted 1wk<2wk) -- 24.9% apart. Every other date agrees to 0.0%.
            if not np.allclose(tgt, JH[yr]["R"], rtol=1e-6):
                _EST[yr] = 100 * float(np.max(np.abs(tgt / np.array(JH[yr]["R"], float) - 1)))
            m = np.array(r["ssr"], float)
            c = np.array(BC[yr]["ssr"], float) if yr in BC else None
            ax.fill_between(x, tgt - se, tgt + se, color=C_OBS, alpha=0.15, lw=0)
            ax.errorbar(x, tgt, yerr=se, fmt="o", color=C_OBS, ms=4, capsize=2, lw=1,
                        label="realised $\\pm$HAC")
            if c is not None:
                ax.plot(x, c, "^:", color=C_CMP, ms=4, lw=1.3, label=A.clabel)
            ax.plot(x, m, "s--", color=C_MOD, ms=4, lw=1.4, label=A.label)
            ax.axhline(2.0, color="#555", ls=(0, (5, 3)), lw=0.7, alpha=0.5, zorder=0,
                       label="diffusive short-time limit (SSR$\\to$2)" if ax is axes.flat[0] else None)
            ax.set_xscale("log"); ax.set_xticks(WKS); ax.set_xticklabels(WLAB, fontsize=7)
            _r = (f"{BC[yr]['ssr_rms']:.1f}% $\\to$ {r['ssr_rms']:.1f}%" if yr in BC
                  else f"{r['ssr_rms']:.1f}%")
            _rb = "  [robust-$\\beta$]" if yr in _EST else ""
            ax.set_title(f"{yr}  {REG.get(yr,'')}{tag(yr)}{_rb}\nSSR RMS  {_r}   "
                         f"target s.e. {JH[yr]['floor_joint']:.1f}%", fontsize=8)
            _all = np.concatenate([m, c]) if c is not None else m
            lo = min(tgt.min() - se.max(), _all.min()) - 0.15
            hi = max(tgt.max() + se.max(), _all.max()) + 0.15
            ax.set_ylim(lo, hi)
        else:
            x = np.array(r["vov_tenor_d"], float)
            tgt = np.array(r["vov_target"], float); m = np.array(r["vov"], float)
            if yr in VC:
                # LOG-X here, unlike the 2-anchor default: the grid spans 14d-365d (26x) and a linear
                # axis crushes everything short of 90d. Flagged in the suptitle, not silent.
                v = VC[yr]
                To = np.array(v["tenor_obs"], float)
                raw = np.array(v["realised_raw"], float)
                cor = np.array(v["realised_cor"], float) if v.get("realised_cor") else None
                ax.plot(To, raw, "o", mfc="none", color="#9aa5ab", ms=3.5, lw=1,
                        label="realised RAW (T-day var)" if ax is axes.flat[0] else None)
                if cor is not None:
                    ax.plot(To, cor, "o", mfc="none", color=C_OBS, ms=4, lw=1,
                            label="realised, readout object" if ax is axes.flat[0] else None)
                ax.plot(np.array(v["tenor"], float), np.array(v["model"], float), "-", color=C_MOD,
                        lw=1.5, label=A.label + " (all tenors)" if ax is axes.flat[0] else None)
                ax.set_xscale("log")
                ax.set_xticks([14, 30, 90, 180, 365])
                ax.set_xticklabels(["14d", "30d", "90d", "180d", "1y"], fontsize=6.5)
                # the anchors ride on top -- they are the only vov the objective actually saw
                _al = "FITTED anchors (" + ", ".join(f"{int(v)}d" for v in x) + ")"
                ax.plot(x, tgt, "o", color="#0b3d52", ms=5.5, zorder=5,
                        label=_al if ax is axes.flat[0] else None)
                ax.plot(x, m, "s", color=C_MOD, ms=5, zorder=6,
                        label="model at the anchors" if ax is axes.flat[0] else None)
            else:
                # UNCHANGED default path -- verified byte-identical against the committed _ref figures
                ax.plot(x, tgt, "o-", color=C_OBS, ms=3.5, lw=1, label="realised/implied")
                if yr in BC:
                    ax.plot(np.array(BC[yr]["vov_tenor_d"], float), np.array(BC[yr]["vov"], float),
                            "^:", color=C_CMP, ms=3.5, lw=1.3, label=A.clabel)
                ax.plot(x, m, "s--", color=C_MOD, ms=3.5, lw=1.4, label=A.label)
            ax.set_xlabel("dte", fontsize=7)
            _r = (f"{BC[yr]['vov_rms']:.1f}% $\\to$ {r['vov_rms']:.1f}%" if yr in BC
                  else f"{r['vov_rms']:.1f}%")
            ax.set_title(f"{yr}  {REG.get(yr,'')}{tag(yr)}\nvov RMS  {_r}", fontsize=8)
        if ax is axes.flat[0]:
            ax.legend(frameon=False, fontsize=6.5 if A.compare else 7, loc="upper right")
    # NOT "vol-of-vol": on SPX this axis is the VIX ATM implied volatility, which is the
    # observation the forward-variance block actually reads. Off index the same slot holds the
    # corrected realised strip, so the axis is labelled per ticker.
    fig.supylabel("SSR" if kind == "ssr"
                  else (r"VIX ATM implied vol  $\xi$" if _TK == "SPX"
                        else r"realised fwd-var  $\xi$"), fontsize=10)
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.01, 0, 1, 0.99])
    p = os.path.join(OUT, fname)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    for _y, _d in sorted(_EST.items()):
        print(f"    NOTE {_y}: fitted target differs from the HAC band centre by {_d:.1f}% -- "
              f"different estimators; panel flagged robust-beta, band is the OLS one.")


# The suptitle is for a reader of the paper, not a reader of the shell history: the operative
# configuration lives in the replication manifest, so name the panel rather than the env flags.
BASE = r"nine SPX regimes, shipped two-stage fits"
if A.vovcurve:
    # NOT "ladder 56": the vov curve's reach is a property of the run that built --vovcurve, which
    # this script cannot see. Stating it here printed "ladder 56" over curves built at LADDER=42.
    BASE = BASE.replace("ladder 42", "vov curve to " + f"{max(json.load(open(os.path.join(D, A.vovcurve) if not os.path.isabs(A.vovcurve) else A.vovcurve))[next(iter(VC))]['tenor']):.0f}d") \
           + ", vov panel on LOG tenor"
if A.note:
    BASE += ", " + A.note
_ssr = f"{A.prefix}_ssr.png" if A.prefix else "ref_ssr_v2style.png"
_vov = f"{A.prefix}_vov.png" if A.prefix else "ref_vov_v2style.png"
# Name the OBSERVATION, not the concept: on SPX the panel is the VIX ATM implied-volatility term
# structure; off index it is the corrected realised forward-variance strip. "vol-of-vol" named
# neither and contradicted the manuscript's own caption.
_VL = ("VIX ATM implied-volatility term structure" if _TK == "SPX"
       else "corrected realised forward-variance strip")
grid("ssr", _ssr, _TK + r" realised SSR term structure ($\pm$joint HAC) vs model — " + BASE)
grid("vov", _vov, f"{_TK} {_VL} vs model — " + BASE)

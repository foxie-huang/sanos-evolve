#!/usr/bin/env python3
"""NDX vol-of-vol OUT OF SAMPLE, on the CORRECTED model -- the fit saw 30d and 90d only.

Same figure and the same drawing grammar as `plot_ndx_vov_oos.py` (log-log, model as a continuous
line in tenor, fitted anchors as filled markers, out-of-sample tenors as open markers with their
rvov_spread(w=63) bars, red title where the 2-point slope exceeds the 1/sqrt(T) ceiling). TWO things
differ, and both are the point of running it again:

  * **VOVLEV=1.** The old script evaluates `lam_fns=None`, i.e. the UNLEVERED readout -- the version
    this project set out to correct. Here the model is drawn through the leveraged readout with the
    lambda ladder, so the curve is the corrected model.
  * **kap_s FREE.** The old record pins kap_s at 0.9956 and carries 7 parameters; the kernel_fast
    fits carry 8 with kap_s fitted, which is what sets the long-end decay the OOS tenors probe.

Everything OFF 30d/90d is a genuine prediction: the fit never saw it. The realised side comes from
`.ndx_oos_cache/` (built by diagnostics/ndx_vov_oos.py) and is not recomputed here.

MODEL CURVE IS CAPPED at the ladder. The leveraged readout propagates n_opt = round(tau/dt) steps and
then reads a 30d VIX window (n_var=4), so it needs n_opt+n_var <= LADDER rungs. At LADDER=42 that is
~267d; beyond it `lam_fns` clamps to the last rung and the reading is an extrapolation, so the line
stops rather than pretending. Realised points beyond the cap are still drawn -- they are data.

    python3 plot_ndx_oos_kf.py                 # tag _n9
    python3 plot_ndx_oos_kf.py --tag _n9 --out /tmp
"""
import argparse
import glob
import json
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_n9")
ap.add_argument("--out", default=None)
# --cm takes the REALISED side from the constant-maturity cache too, so both the target and the fit
# are on the same construction. Without it the model comes from a CM fit but the points are the
# snapped series -- two different objects on one axis.
ap.add_argument("--cm", action="store_true", help="realised side from .ndx_cm_cache (constant maturity)")
# --pq draws the realised series ALSO on the object the readout computes. vix.py:155 -- the readout is
# an option expiring at tau on a 30-DAY forward variance (tau_var FIXED); the realised series at tenor
# T is the daily vol of the T-DAY variance. Different objects, gap widening with T. spx_pq_vov.json
# measures that gap on SPX, where both sides exist. Both series are drawn: hiding the raw one would
# bury the size of the correction.
ap.add_argument("--pq", action="store_true", help="also draw realised x SPX Q/P (the readout's object)")
ap.add_argument("--compare", default=None, help="second tag to overlay as a comparison model curve")
A = ap.parse_args()

os.environ.setdefault("LADDER", "42")
os.environ.setdefault("VOVLAMTEN", "avg")
sys.argv = [sys.argv[0], "cpu"]

import torch                                                     # noqa: E402
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernel_fast"))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                              # noqa: E402
import consts, fkernel as kernel, vix as VX                      # noqa: E402
import calibrate_slv_exact_ts as C                               # noqa: E402
import end_to_end as E                                           # noqa: E402

OUT = A.out or os.path.join(_P.DATA, "figs_ndx")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
C_OBS, C_MOD, C_FIT, C_CMP, C_RAW = "#1f6f8b", "#b02a37", "#0b3d52", "#c98500", "#9aa5ab"
GRID = np.array([7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365], float)
FITTED = (30, 90)

_sub, _pat = ((".ndx_cm_cache", "vov_cm_NDX_*.npz") if A.cm
              else (".ndx_oos_cache", "vov_oos_NDX_*.npz"))
OOS = {os.path.basename(f).split("_")[-1][:4]: dict(np.load(f))
       for f in glob.glob(os.path.join(_P.DATA, _sub, _pat))}
FITS = sorted(glob.glob(os.path.join(_P.DATA, f"fit_kf{A.tag}_*_ndx.json")))
if not FITS:
    raise SystemExit(f"no fit_kf{A.tag}_*_ndx.json -- run the NDX panel first")
PQ = json.load(open(os.path.join(_P.DATA, "spx_pq_vov.json"))) if A.pq else None


def pq_ratio(yr, T):
    """Per-tenor SPX Q/P for this year; NaN where Q was unbracketed (those points simply do not draw)."""
    z = PQ[yr]; Tp = [int(x) for x in z["tenors"]]
    q = np.asarray(z["q_mean"], float); pp = np.asarray(z["p_rvov"], float)
    r = np.full(len(T), np.nan)
    for i, t in enumerate(T):
        if int(t) in Tp:
            j = Tp.index(int(t))
            if np.isfinite(q[j]) and np.isfinite(pp[j]) and pp[j] > 0:
                r[i] = q[j] / pp[j]
    return r


def curve(tag, date, gs):
    j = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}_ndx.json")))
    ctx, _c, _n = E.ctx_rebuilt(date, "NDX")
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    with torch.no_grad():
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        return np.array([float(VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT,
                                           lam_fns=LAM, us0=u0)[1]) for d in gs])


def theilsen(x, y):
    return float(np.median([(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(len(x)), 2)]))


CURVES = {}
n = len(FITS)
rows = int(np.ceil(n / 3))
fig, axes = plt.subplots(rows, 3, figsize=(11, 2.9 * rows + 0.5))
tab = []
cap = None
for ax, p in zip(axes.flat, FITS):
    j = json.load(open(p))
    date = j["date"]; yr = date[:4]
    ctx, _c, _n = E.ctx_rebuilt(date, "NDX")
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    cap = (len(LAM) - n_var) * K.dt * 365.0            # last tenor the ladder actually supports
    th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    with torch.no_grad():
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        gs = GRID[GRID <= cap]
        mv = np.array([float(VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT,
                                         lam_fns=LAM, us0=u0)[1]) for d in gs])
    d = OOS[yr]; T = np.asarray(d["tenors"], float); rv = np.asarray(d["rvov"], float)
    sp = np.asarray(d["spread"], float); fit = np.isin(T, FITTED)
    first = ax is axes.flat[0]
    if A.compare:
        ax.plot(gs, curve(A.compare, date, gs), ":", color=C_CMP, lw=1.5,
                label=f"model {A.compare}" if first else None)
    ax.plot(gs, mv, "-", color=C_MOD, lw=1.6,
            label=(f"model {A.tag}, VOVLEV=1" if A.compare else "model, VOVLEV=1 (fitted $\\theta$)")
            if first else None)
    _lab = ("realised (const-mat), OUT of sample" if A.cm else "realised, OUT of sample")
    if A.pq:
        # RAW stays on the plot, greyed: the correction is large and must be visible, not assumed.
        ax.plot(T[~fit], rv[~fit], "o", mfc="none", color=C_RAW, ms=5, lw=1,
                label="realised RAW (T-day variance)" if first else None)
        ax.plot(T[fit], rv[fit], "o", color=C_RAW, ms=6, alpha=0.6, label=None)
        r = pq_ratio(yr, T); rvc, spc = rv * r, sp * r
        ax.errorbar(T[~fit], rvc[~fit], yerr=spc[~fit], fmt="o", mfc="none", color=C_OBS, ms=5, lw=1,
                    capsize=2, label="realised x SPX Q/P (30d variance)" if first else None)
        ax.errorbar(T[fit], rvc[fit], yerr=spc[fit], fmt="o", color=C_FIT, ms=7, lw=1.2, capsize=2,
                    zorder=5, label="FITTED anchors (30d, 90d)" if first else None)
        rv_t = rvc                                   # titles score against the object being tracked
    else:
        ax.errorbar(T[~fit], rv[~fit], yerr=sp[~fit], fmt="o", mfc="none", color=C_OBS, ms=5, lw=1,
                    capsize=2, label=_lab if first else None)
        ax.errorbar(T[fit], rv[fit], yerr=sp[fit], fmt="o", color=C_FIT, ms=7, lw=1.2, capsize=2,
                    zorder=5, label="realised, FITTED (30d, 90d)" if first else None)
        rv_t = rv
    ok = np.isfinite(rv_t) & (rv_t > 0)
    pr = -theilsen(np.log(T[ok]), np.log(rv_t[ok]))               # robust 11-tenor decay exponent
    p2 = np.log(rv_t[T == 30][0] / rv_t[T == 90][0]) / np.log(3.0)   # the 2-point one the fit sees
    tab.append((yr, p2, pr, j["vov_rms"]))
    CURVES[yr] = dict(tenor=gs.tolist(), model=mv.tolist(),
                      realised_raw=rv.tolist(), realised_cor=(rv_t.tolist() if A.pq else None),
                      tenor_obs=T.tolist())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([7, 30, 90, 365]); ax.set_xticklabels(["7d", "30d", "90d", "1y"], fontsize=7.5)
    ax.set_title(f"{yr}   $p$ 2-pt {p2:.2f} | 11-tnr {pr:.2f}   (fit RMS {j['vov_rms']:.1f}%)",
                 fontsize=8.5, color=("#b02a37" if p2 > 0.5 else "black"))
for ax in axes.flat[n:]:
    ax.axis("off")
axes.flat[0].legend(fontsize=7, loc="best")
for a in (axes[:, 0] if rows > 1 else [axes[0]]):
    a.set_ylabel("vol-of-vol")
fig.suptitle(f"NDX vol-of-vol OUT OF SAMPLE{' (constant-maturity realised)' if A.cm else ''}"
             f"{', realised rescaled to the readout object by SPX Q/P' if A.pq else ''}, "
             "corrected model (VOVLEV=1, $\\kappa_S$ free) -- fitted on "
             f"30d/90d ONLY, drawn to {cap:.0f}d (ladder {len(LAM)}).\nRed title: the 2-point slope exceeds "
             "the model's $1/\\sqrt{T}$ ceiling", fontsize=10, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
f = os.path.join(OUT, f"ndx_oos_kf{A.tag}{'_cm' if A.cm else ''}{'_pq' if A.pq else ''}"
                 f"{('_vs' + A.compare) if A.compare else ''}.png")
fig.savefig(f, bbox_inches="tight")
print("wrote", f)
P2 = np.array([t[1] for t in tab]); PR = np.array([t[2] for t in tab])
print(f"  model curve capped at {cap:.0f}d (ladder {len(LAM)} rungs - {n_var} VIX-window rungs)")
print(f"  p 2-point  mean {P2.mean():.3f}   {int((P2 > 0.5).sum())}/{n} above the 1/sqrt(T) ceiling")
print(f"  p 11-tenor mean {PR.mean():.3f}   {int((PR > 0.5).sum())}/{n} above")
json.dump(CURVES, open(os.path.join(_P.DATA, f"ndx_oos_curves{A.tag}.json"), "w"), indent=1)
json.dump([dict(year=y, p_2pt=a, p_11tenor=b, vov_rms=c) for y, a, b, c in tab],
          open(os.path.join(_P.DATA, f"ndx_oos_kf{A.tag}_exponents.json"), "w"), indent=1)

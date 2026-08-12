#!/usr/bin/env python3
"""End-to-end: kernel fit on the rebuilt static layer vs the shipped one, same date, same objective.

Everything downstream of the marginals is untouched -- same targets, same `J.fit_date`, same start
point, same weights. Only two objects change, and they are the only two the marginals ever reach:

    sig_ref   recomputed from the blended 6m marginal of the joint fit
    LT        the leverage table, rebuilt on the alpha-blend and re-fitted to the same
              density-weighted log-polynomial form `leverage_at` uses, so `lev_torch` is unchanged

The safety clip (0.2, 5.0) and the calendar floor are deliberately left in place: the point is to
show they stop binding, not to remove them.

    python3 end_to_end.py 2020-06-01
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

# (was sys.path.insert(0, HERE) -- HERE is now the DATA dir; sibling modules come
#  from _paths, which puts the v3_scripts code dirs ahead of it.)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "poc")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "data")))
sys.argv = [sys.argv[0], "cpu"] + sys.argv[1:]

import torch                                                        # noqa: E402
torch.set_default_dtype(torch.float32); torch.set_default_device("cpu")
torch.set_num_threads(1)
import sanos_true as ST, sanos_leverage as SL                       # noqa: E402
import static_payload as SP                                         # noqa: E402
import calibrate_joint_torch as J                                   # noqa: E402
import calibrate_slv_exact_ts as C                                  # noqa: E402
import calibrate_ndx as CN                                          # noqa: E402
import discslv_torch as DTt                                         # noqa: E402
from discslv_slv import Epi_V                                       # noqa: E402
from discslv_2f import TwoFactorSV                                  # noqa: E402
from orats_sanos import orats_chain_df                              # noqa: E402

DEG = int(os.environ.get("DEG", "1"))    # log-lambda polynomial degree (shipped leverage_at uses 1)
SAFETY = (0.2, 5.0)
FREEZE_OFF = os.environ.get("NOFREEZE", "") == "1"   # force recomputation of the static layer
_LADDER = int(os.environ.get("LADDER") or 0) or max(J.NS)   # weeks of lambda to build
CACHE = os.path.join(HERE, "staticcache"); os.makedirs(CACHE, exist_ok=True)


def joint_cached(date, ticker, eta):
    """(K, V, qs, Ts) from the joint LP, cached per (ticker, date, eta) -- the LP is ~4-10s and is
    independent of theta, so re-solving it inside every fit iteration is pure waste."""
    # v3 = SANOS as written: quoted-PRICE bid/ask band mapped by parity into normalised
    # undiscounted call units, and inverse-Vega weights (Section 4.2's own alternative to 1/(A-B)).
    # v2 = monotonicity screen drops rather than coerces. Key bumped so stale solutions cannot be
    # silently reused -- the LP inputs changed, not just the screen.
    f = os.path.join(CACHE, f"{ticker}_{date}_eta{eta:.3f}_v3.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["K"], z["V"], list(z["qs"]), z["Ts"]
    df = orats_chain_df(f"{ST.ORATS}/SPX-NDX-RUT-VIX_{date}.json.gz", ticker)
    slices = [s for s in (ST.market_slice(g) for _, g in sorted(df.groupby("dte"))) if s is not None]
    out = ST.fit_joint(slices, eta=eta, verbose=False)
    if out is None:
        raise RuntimeError(f"{ticker} {date}: joint LP failed")
    K, V, qs, keep = out
    Ts = np.array([slices[i]["tau"] for i in keep])       # taus of the SURVIVING expiries only
    np.savez(f, K=K, V=V, qs=np.array(qs), Ts=Ts, keep=keep)
    return K, V, list(qs), Ts


class Lev:
    """Same interface as slv_wire._Lev, so lev_torch consumes it unchanged."""
    def __init__(self, coef, zmax, safety=SAFETY):
        self.coef, self.zmax, self.safety = np.asarray(coef, float), float(zmax), safety


def _ladder_build_overridden():
    """LAMDSG/LAMH change how SL.leverage BUILDS each rung, and the frozen payload path skips that
    code entirely -- so serving the cache would make them SILENTLY INERT. This is the same class of
    bug as the `_smooth_ladder` bypass fixed above, except it CANNOT be repaired after loading:
    the derivative estimator has to run while the ladder is being constructed. Recompute instead."""
    if int(os.environ.get("LAMDSG") or 0) >= 3:
        return True
    if abs(float(os.environ.get("LAMH") or 1.0) - 1.0) > 1e-12:
        return True
    return False

def rebuilt_static(date, ticker, eta=SL.ETA, deg=None):
    """(sig_ref, {k: Lev}, n clipped, n pillars) from the joint LP + alpha-blend.

    Served from the frozen 40-float payload when one exists for this (eta, deg) -- see
    static_payload.py. Only sig_ref and LT ever reach `model_torch`, and both are CONSTANTS with
    respect to theta, so serving them from disk is not an approximation: no gradient was ever going
    to flow back into the LP. `load` returns None rather than raising if the payload is missing or
    was frozen under different (eta, deg), so we fall back to recomputation instead of silently
    fitting the wrong static layer.
    """
    deg = DEG if deg is None else deg
    if not FREEZE_OFF and not _ladder_build_overridden():
        hit = SP.load(date, ticker, eta=eta, deg=deg)
        # The payload records how many rungs it was FROZEN with (_meta[2]); SP.load validates eta and
        # deg but NOT that count. Serving a 42-rung ladder to a LADDER=56 request would silently cap
        # the vov readout back at 267d while every label claimed 365d -- the same silent-inertness
        # class as the ladder-build bypass above. Fall through to recomputation instead.
        if hit is not None and len(hit[1]) < _LADDER:
            hit = None
        if hit is not None:
            # The payload caches the RAW static layer; smoothing is a post-processing step on it, so
            # it must be applied HERE too. Until 2026-08-10 this path returned early and LAMSMOOTH/
            # LAMSLOPE were SILENTLY INERT unless NOFREEZE=1 -- a setting that changes the model and
            # reports nothing. Verified safe: the payload ladder is bit-identical (0.000e+00 over all
            # 42 weeks x 2 coefficients) to the recomputed one, so frozen+smoothed == NOFREEZE=1
            # +smoothed, at a fraction of the cost.
            s_, lev_, cl_, np_ = hit
            return s_, _smooth_ladder(lev_, date), cl_, np_
    K, V, qs, Ts = joint_cached(date, ticker, eta)
    q6, v6 = SL.blend(K, V, qs, Ts, 0.5, eta)
    m1 = float(np.sum(q6 * K)); m2 = float(np.sum(q6 * K ** 2) * np.exp(v6))
    sig_ref = float(np.sqrt(max(np.log(max(m2 / m1 ** 2, 1 + 1e-12)), 1e-12)) / np.sqrt(0.5))
    kw0 = dict(zip(J.NAMES, C.X0_MAP["dense"]))
    EV = Epi_V(TwoFactorSV(gbar=J.solve_gbar(kw0, sig_ref, dt=J.DT), dt=J.DT,
                           n_f=5, n_s=3, n_l=5, **kw0))
    lev, clipped = {}, 0
    # LADDER=N builds lambda out to week N instead of stopping at max(J.NS)=13 (=91 days).
    # WHY: the VOVLEV=1 VIX readout needs weeks 1..37 (262d) -- 3 of 9 vov tenors run past the ladder
    # at 2016/2018 and 6 of 12 at 2022/2024, and the propagation runs up to 24 of 37 steps on the
    # FROZEN week-13 slice, which is the flat tail in the model's vov (handoff 6e.1). The option
    # surface covers 934-1662 days, so the data supports it.
    #
    # STRICTLY ADDITIVE: slices 1..13 are computed from the same inputs by the same code, and
    # `readouts.ssr_ts` loops `range(K.nmax)` with nmax = max(NS) = 13, so SSR never indexes past 12
    # and every SSR number is unchanged. Only the leveraged VIX beyond 91 days sees anything new.
    #
    # NOTE: the frozen payload holds 13 weeks, so this needs NOFREEZE=1 until the payload is re-frozen.
    for k in range(1, _LADDER + 1):
        zg, lam, bad = SL.leverage(K, V, qs, Ts, k * SL.DT, EV=EV, eta=eta)
        assert bad == 0, f"{ticker} {date} week {k}: calendar slope non-positive at {bad} nodes"
        qm, vm = SL.blend(K, V, qs, Ts, k * SL.DT, eta)
        wt = np.sqrt(SL.density(K, qm, vm, np.exp(zg)) * np.exp(zg) + 1e-12)   # as leverage_at
        coef = np.polyfit(zg, np.log(np.maximum(lam, 1e-300)), deg, w=wt)
        clipped += int(((lam < SAFETY[0]) | (lam > SAFETY[1])).sum())
        lev[k] = Lev(coef, float(zg[-1]))
    lev = _smooth_ladder(lev, date)
    return sig_ref, lev, clipped, len(Ts)


# LAMSMOOTH=<deg> mollifies the lambda ladder ALONG T. Off by default (empty/0 = raw ladder).
#
# WHY. lambda_k is fitted INDEPENDENTLY per week from dC/dT of a PIECEWISE-LINEAR blend, so dC/dT
# genuinely jumps at every pillar (6e.1b) and the ladder inherits it. At 2016 lambda's SLOPE is
# -17.75 at week 7 against -7.44 at week 8 -- a 2.4x spike, 58.1% week jump -- and the VIX window
# that ends on week 7 reads it, producing the +22.3% vov kink at 29d (6e.13). Panel-wide the ladder's
# week-2..12 slope roughness predicts the model's vov non-smoothness at corr = +0.942, and only 2012
# and 2016 are badly affected (18-19% mean jump vs 5-11% elsewhere).
#
# WHY THIS IS SAFE, unlike the two recorded dead ends. lambda is the discrete Gyongy overlay, so
# smoothing it costs REPRICING ACCURACY at the pillars -- but it CANNOT introduce arbitrage: the
# kernel is martingale independently of lambda (`recompress` re-locks it every step, and `step`
# applies the log-sum-exp normaliser A). The B-spline blend failed because it changed the MARGINALS
# (negative q, calendar arbitrage, 338% repricing); the pillar-aware stencil (6e.2) changed dC/dT
# itself and was worse on all four dates. This touches neither -- only the kernel's leverage.
#
# The roughness is an ARTIFACT OF THE INTERPOLATION, not of the data: C(T) between pillars is
# unobserved and `blend`'s linearity is a modelling choice. Gate any setting on the repricing bar
# (verify_blend.py, ~26 ATM vol bp of bid-ask).
#
# Each week is only two numbers (polyfit deg=1: slope, intercept), so this smooths two sequences of
# length _LADDER as polynomials in log T. deg 0 = constant (maximum smoothing), higher = closer to raw.
_LAMSMOOTH = int(os.environ.get("LAMSMOOTH") or 0)
# LAMKEEP=<n> leaves the FIRST n and LAST n weeks at their raw values. A polynomial fit carries its
# largest error at the endpoints, week 1 has the steepest lambda and matters most for the front of
# the surface, and the final week is pure extrapolation -- so anchoring both ends keeps the smoother
# from distorting where it is least constrained. Costs a seam between the kept and smoothed regions;
# whether that seam is worse than the distortion is an empirical question, hence the knob.
_LAMKEEP = int(os.environ.get("LAMKEEP") or 0)
# LAMSG=<odd window> applies a SAVITZKY-GOLAY filter (local polynomial in a sliding window) to each
# log-lambda coefficient sequence instead of the GLOBAL polynomial of LAMSMOOTH. Order LAMSGDEG
# (default 2). Takes precedence over LAMSMOOTH when both are set.
#
# WHY LOCAL. LAMSMOOTH fits ONE polynomial across all 42 weeks, so a jump at week 7 is "fixed" by
# distorting weeks 1-42, and its error is largest at the endpoints -- which is why LAMKEEP had to
# exist at all. That is a projection onto a low-dimensional global space, not a mollification. SG is
# the standard local mollifier: it fits a low-order polynomial in a window around each week and keeps
# only the centre value, so it adapts where the ladder is rough and leaves smooth stretches alone.
# `mode="interp"` fits the end windows directly, so the edges need no separate protection.
#
# The defect being mollified IS local: `blend` is piecewise-linear in T, so dC/dT is piecewise
# CONSTANT with jumps at pillars (6e.1b), and the ladder inherits a jump only where the +-DT/2
# stencil straddles one. At 2016 that is weeks 7-9 (30.1/58.1/24.2%) against 1.0-6.7% elsewhere.
# LAMSLOPE=1 smooths ONLY the z-SLOPE coefficients of log lambda and leaves the CONSTANT term raw.
# polyfit returns [c_deg, ..., c_1, c_0]; c_0 is log lambda(0), i.e. the LEVEL, and the rest are the
# shape in z. The two do different jobs: the LEVEL sets each week's variance, which is exactly what
# the Gyongy overlay exists to get right, while the SLOPE sets the leverage skew -- and 6e.13 showed
# the vov spike tracks the SLOPE (2016 week 7: -17.75 vs -7.44 at week 8, a 2.4x jump) not the level.
# So smoothing the slope alone should keep the vov gain while leaving every week's variance untouched,
# which is the part of the marginal the repricing question is about.
_LAMSLOPE = os.environ.get("LAMSLOPE") == "1"
_LAMSG = int(os.environ.get("LAMSG") or 0)
_LAMSGDEG = int(os.environ.get("LAMSGDEG") or 2)

# LAMDATES=<path.json> makes the smoothing decision PER DATE from a table {date: true|false}.
#
# WHY A TABLE AND NOT A THRESHOLD. The obvious gate is "smooth only where the ladder is rough", and
# it does not work: across the 9 SPX dates, THIRTEEN smoothness statistics -- ladder-side (RMS of what
# the mollifier removes; mean/max relative weekly slope jump over weeks 2-12, over 1-42, over the tail;
# readout-weighted by how many fitted tenors read each week), and residual-side (oscillatory amplitude
# of the pre-smoothing vov residual, its ratio to total, sign flips, mean |second difference|) -- ALL
# OVERLAP between the six dates smoothing helps and the three it hurts. The best of them reaches
# corr = -0.73 against the outcome, carried entirely by 2012/2016 being both roughest and most
# improved; among the other seven there is no signal (2020/2021/2022 have the SMOOTHEST ladders,
# R_dev 0.033-0.038, and gain 10-34%, while 2017 is third-roughest and loses). Any threshold
# misclassifies at least four of nine. See handoff 6e.16.
#
# Nor is the loss an optimiser artifact: 2019 and 2024 were refit COLD under the same smoothing
# (`_sl9c`, njev 10-12 against the warm run's 2-3) and landed within 0.2pp of the warm result. The
# degradation is a property of the smoothed objective, not of where the search started.
#
# So the decision has to be MEASURED, not predicted: fit both ways, keep the lower DATA cost, and
# record which. `lam_gate.py` builds the table and reports the margin against the staircase noise
# floor. A date missing from the table RAISES -- defaulting silently is exactly the failure the
# frozen-payload bypass above already cost us once.
_LAMDATES = os.environ.get("LAMDATES", "")
_LAMTAB = None


def _lam_on(date):
    """Is lambda smoothing enabled for `date`? True for every date when no table is given."""
    global _LAMTAB
    if not _LAMDATES:
        return True
    if _LAMTAB is None:
        import json
        with open(_LAMDATES) as fh:
            t = json.load(fh)
        _LAMTAB = t.get("smooth", t)
    if date not in _LAMTAB:
        raise KeyError(f"LAMDATES table {_LAMDATES} has no entry for {date}. Add it (true/false) "
                       f"or unset LAMDATES -- refusing to guess which model to run.")
    return bool(_LAMTAB[date])


def _smooth_ladder(lev, date=None):
    """Refit each log-lambda coefficient as a degree-_LAMSMOOTH polynomial in log(week)."""
    if date is not None and not _lam_on(date):
        return lev
    if _LAMSG:
        return _sg_ladder(lev)
    if not _LAMSMOOTH:
        return lev
    ks = sorted(lev)
    if len(ks) < _LAMSMOOTH + 2:
        return lev
    x = np.log(np.asarray(ks, float))
    C = np.array([lev[k].coef for k in ks], float)          # (nweek, deg+1)
    S = np.empty_like(C)
    for j in range(C.shape[1]):
        S[:, j] = np.polyval(np.polyfit(x, C[:, j], _LAMSMOOTH), x)
    if _LAMSLOPE:
        S[:, -1] = C[:, -1]                 # keep log lambda(0) -- the LEVEL -- exactly as fitted
    if _LAMKEEP:
        n = min(_LAMKEEP, len(ks) // 2)
        S[:n] = C[:n]; S[len(ks) - n:] = C[len(ks) - n:]
    return {k: Lev(S[i], lev[k].zmax) for i, k in enumerate(ks)}

def _sg_ladder(lev):
    """Savitzky-Golay each log-lambda coefficient across weeks. Local; edges via mode='interp'."""
    from scipy.signal import savgol_filter
    ks = sorted(lev)
    w = _LAMSG if _LAMSG % 2 == 1 else _LAMSG + 1          # savgol needs an odd window
    if len(ks) < w or w <= _LAMSGDEG:
        return lev
    C = np.array([lev[k].coef for k in ks], float)
    S = np.empty_like(C)
    for j in range(C.shape[1]):
        S[:, j] = savgol_filter(C[:, j], w, _LAMSGDEG, mode="interp")
    if _LAMSLOPE:
        S[:, -1] = C[:, -1]
    return {k: Lev(S[i], lev[k].zmax) for i, k in enumerate(ks)}


def ctx_rebuilt(date, ticker):
    """The shipped context with only sig_ref and LT replaced."""
    ctx = J.build_date_ctx(date) if ticker == "SPX" else CN.build_ctx_ndx(date)
    sig_ref, lev, clipped, npil = rebuilt_static(date, ticker, deg=DEG)
    ctx = dict(ctx)
    ctx["sig_ref"] = float(sig_ref)
    # LEVZFAC widens the range over which the FITTED log-linear leverage is applied, without
    # refitting it. sanos_leverage builds lambda on +-KSD*sd (KSD=1.5) and lev_torch clamps its
    # argument there, so at week 1 lambda is frozen outside +-0.02..0.05 against a one-step return sd
    # of ~0.031 -- the outer third of the week-1 return distribution sees a constant leverage, which
    # caps the spot->vol response the SSR regression measures. Extrapolating the degree-1 fit is
    # bounded by `safety` (0.2, 5.0), so this cannot run away. Refitting on a wider window is the
    # separate experiment (SL.KSD), and changes the fit itself rather than only its domain.
    zfac = float(os.environ.get("LEVZFAC", "1.0"))
    ctx["LT"] = {n: [DTt.lev_torch(lev[k + 1].coef, zfac * lev[k + 1].zmax, lev[k + 1].safety)
                     for k in range(n)] for n in sorted(set(list(J.NS) + [_LADDER]))}
    return ctx, clipped, npil


def report(tag, ctx, x0):
    t0 = time.time()
    res = J.fit_date(ctx, x0)
    th = res.x if hasattr(res, "x") else res
    with torch.no_grad():
        m = J.model_torch(torch.tensor(th, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                          ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
    n = len(J.NS)
    ssr, vov = m[:n], m[n:]
    tgt_s = np.asarray(ctx["emp"], float); tgt_v = np.asarray(ctx["vov_d"], float)
    rms = lambda a, b: 100 * np.sqrt(np.mean(((a - b) / b) ** 2))
    print(f"  {tag:16s} SSR RMS {rms(ssr, tgt_s):6.2f}%   vov RMS {rms(vov, tgt_v):6.2f}%   "
          f"sig_ref {ctx['sig_ref']:.4f}   [{time.time()-t0:.0f}s]")
    return ssr, tgt_s


if __name__ == "__main__":
    date = sys.argv[2] if len(sys.argv) > 2 else "2020-06-01"
    x0 = np.asarray(C.X0_MAP["ts"], float)
    for tk in ("SPX", "NDX"):
        print(f"\n===== {tk} {date} =====")
        old = J.build_date_ctx(date) if tk == "SPX" else CN.build_ctx_ndx(date)
        new, clipped, npil = ctx_rebuilt(date, tk)
        print(f"  rebuilt static layer: {npil} pillars, "
              f"{clipped}/{max(J.NS)*SL.NZ} leverage nodes outside the (0.2,5.0) clip")
        s_old, tgt = report("shipped", old, x0)
        s_new, _ = report("rebuilt", new, x0)
        print(f"  {'target SSR':16s} " + " ".join(f"{v:6.3f}" for v in tgt))
        print(f"  {'  shipped':16s} " + " ".join(f"{v:6.3f}" for v in s_old))
        print(f"  {'  rebuilt':16s} " + " ".join(f"{v:6.3f}" for v in s_new))

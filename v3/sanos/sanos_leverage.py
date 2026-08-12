#!/usr/bin/env python3
"""SANOS eq.(2) alpha-blend interpolation, and the leverage rebuilt on top of it.

`slv_interp.interp_marginal` takes the NEAREST pillar's shape and rescales it to an interpolated
total variance. The shape is therefore piecewise constant with a jump at every pillar midpoint, and
`leverage_at` central-differences across +-dt/2 -- so whenever that window straddles a midpoint it
differences two different shapes. Measured: 11 of 12 floored leverage nodes sit on a straddle.

SANOS instead blends BOTH bracketing pillars (eq. 2):

    Chat(T,K) = a(T) sum_i q_j^i Call(K_j^i, K, V(T)) + (1-a(T)) sum_i q_{j-1}^i Call(K_{j-1}^i, K, V(T))
    V(T)      = a(T) V_j + (1-a(T)) V_{j-1}

with a(T) rising 0 -> 1 across (T_{j-1}, T_j], and N_0 = 1, q_0 = delta at the forward below the
first pillar. On the common model-strike grid the fit already uses, this collapses to

    q(T) = (1-a) q_{j-1} + a q_j       component variance  eta * V(T)

which makes two things exact rather than approximate:

  martingale     sum_i q_i(T) K_i = (1-a) + a = 1, since each pillar satisfies K'q = 1;
  calendar       U q(T) is monotone in a because U q_{j-1} <= U q_j (the LP's own constraint), and
                 V(T) rises with T, and Call is increasing in variance -- so d_T C >= 0 everywhere,
                 not merely at the pillars.

The Dupire numerator therefore cannot go negative, so the 1e-14 floor never fires and the leverage
cannot rail at the 0.2 safety bound for want of a calendar slope.

    python3 sanos_leverage.py NDX 2020-06-01
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

import sanos_true as ST                                            # noqa: E402
from sanos_lp import bs_call                                       # noqa: E402
from orats_sanos import orats_chain_df                             # noqa: E402
from scipy.stats import norm                                       # noqa: E402

DT = 1.0 / 52.0
ETA = ST.ETA        # re-export, NOT a second declaration. sanos_true owns eta; see its docstring.
KSD, NZ = 1.5, 25   # the leverage window and node count used downstream


_PILLAR_AWARE = os.environ.get("PILLARAWARE", "0") == "1"


def _segment(Ts, T):
    """The segment of `blend`'s own piecewise-linear interpolation that contains T.

    `blend` interpolates between consecutive pillars (and from a Dirac at T = 0 below the first), so
    these are exactly the intervals on which C(T) is smooth.
    """
    if T >= Ts[-1]:
        return float(Ts[-1]), float("inf")
    j = int(np.searchsorted(Ts, T, side="left"))
    return (0.0 if j == 0 else float(Ts[j - 1])), float(Ts[j])



# ---- B-SPLINE BLEND (BLEND=bspline; default "linear" = SANOS eq.2 unchanged) -------------------
# WHY. eq.2's two-pillar convex blend is piecewise-LINEAR in T, so C(T) kinks at every pillar and
# dC/dT genuinely JUMPS there. lambda inherits it: pooled over four SPX dates, consecutive weeks
# STAYING inside a pillar interval move dlam/dz by 2.8%, weeks CROSSING one by 11.9% (4.2x). Under
# VOVLEV=1 each vov tenor reads one lambda slice, so that lands directly in the model's vov curve --
# and roughening the ladder degrades vov on 4/4 dates with a dose-response of 1.32 +- 0.32 (6e.1b).
# Decomposing the vov-only floor at 2016: the model's SMOOTH component already fits the target to
# 3.78%, and all of the 13.58% floor is jaggedness theta cannot cancel because lambda is exogenous.
#
# A two-pillar blend can NEVER be C1 across a pillar whatever a(T) is, because the STENCIL switches:
# d/dT[(1-a)q_{j-1} + a q_j] = a'(q_j - q_{j-1}) on the left and a'(q_{j+1} - q_j) on the right --
# different vectors. Wider support is the only fix, so: a B-spline partition of unity.
#
# WHAT IT PRESERVES, all three by construction and all three verified numerically in `verify_blend`:
#   martingale    sum_i q_i(T) K_i = 1     because sum_j B_j(T) = 1 and every pillar has K'q_j = 1
#   positivity    q(T) >= 0                because B_j >= 0, so q(T) is a CONVEX combination
#   calendar      d_T C >= 0 everywhere    B-splines are variation-diminishing, so monotone control
#                                          values (the LP's own Uq_{j-1} <= Uq_j) give a monotone
#                                          curve; V(T) = sum_j B_j(T) V_j is monotone for the same
#                                          reason, and Call is increasing in variance
# Degree 2 is the default: C1 is enough to kill the dC/dT jump, support spans 3 pillars (mild), and
# it overshoots less than cubic.
#
# THE COST, and it is real: in control-point form the curve does NOT interpolate the pillars, so the
# model no longer reproduces the LP marginal exactly at T_j. `verify_blend` measures that repricing
# error -- it has to sit inside bid-ask or this is not usable.
_BLEND = os.environ.get("BLEND", "linear")
_BDEG = int(os.environ.get("BLENDDEG", "2"))
_BCACHE = {}


def _bweights(Ts, T, p):
    """Partition-of-unity weights over the augmented sites [0, Ts...]; index 0 is the T=0 Dirac."""
    from scipy.interpolate import BSpline
    key = (id(Ts), len(Ts), p)
    kv = _BCACHE.get(key)
    if kv is None:
        x = np.concatenate([[0.0], np.asarray(Ts, float)])
        n = len(x); pp = min(p, n - 1)
        interior = np.array([x[j:j + pp].mean() for j in range(1, n - pp)]) if n - pp - 1 > 0 \
            else np.array([])
        t = np.concatenate([np.full(pp + 1, x[0]), interior, np.full(pp + 1, x[-1])])
        assert len(t) == n + pp + 1, (len(t), n, pp)
        kv = _BCACHE[key] = (x, t, pp)
    x, t, pp = kv
    Tc = float(min(max(T, x[0]), x[-1]))
    return np.asarray(BSpline.design_matrix([Tc], t, pp).todense()).ravel()



def _ctrl(K, V, qs, Ts, p):
    """Control points c with sum_j B_j(T_i) c_j = q_i -- one banded solve, cached per pillar set."""
    key = ("ctrl", id(Ts), len(Ts), p)
    hit = _BCACHE.get(key)
    if hit is not None:
        return hit
    sites = np.concatenate([[0.0], np.asarray(Ts, float)])
    A = np.array([_bweights(Ts, float(t), p) for t in sites])          # (n, n), banded
    q0 = np.zeros_like(np.asarray(qs[0], float)); q0[int(np.argmin(np.abs(K - 1.0)))] = 1.0
    Q = np.vstack([q0] + [np.asarray(x, float) for x in qs])           # (n, nK)
    Vv = np.concatenate([[0.0], np.asarray(V, float)])
    hit = _BCACHE[key] = (np.linalg.solve(A, Q), np.linalg.solve(A, Vv))
    return hit


def blend(K, V, qs, Ts, T, eta=ETA):
    """(mixture weights on K, component variance) at maturity T -- SANOS eq.(2), or the B-spline
    partition of unity generalising it when BLEND=bspline (see the note above)."""
    if _BLEND in ("bspline", "bspline-interp"):
        w = _bweights(Ts, T, _BDEG)
        if _BLEND == "bspline-interp":
            # CONTROL-POINT form reprices the pillars 20-400% wrong (verify_blend) because it
            # APPROXIMATES them. Solve instead for control points that INTERPOLATE:
            #     sum_j B_j(T_i) c_j = q_i         (A c = q, A banded, one small solve per date)
            # Repricing at the pillars is then 0 by construction, and the martingale SURVIVES
            # EXACTLY: apply K' to A c = q; A's rows sum to 1 (partition of unity) so A 1 = 1, and
            # K'q_i = 1 at every pillar, hence K'c_j = 1. What is NOT automatic any more is q >= 0
            # and monotone Uq -- interpolation can overshoot -- so verify_blend measures both.
            C = _ctrl(K, V, qs, Ts, _BDEG)
            return (w @ C[0]), eta * float(w @ C[1])
        q0 = np.zeros_like(qs[0]); q0[int(np.argmin(np.abs(K - 1.0)))] = 1.0
        Q = np.zeros_like(qs[0]); Vt = 0.0
        for j, wj in enumerate(w):
            if wj == 0.0:
                continue
            Q = Q + wj * (q0 if j == 0 else qs[j - 1])
            Vt = Vt + wj * (0.0 if j == 0 else V[j - 1])
        return Q, eta * Vt
    if T >= Ts[-1]:
        return qs[-1], eta * V[-1]
    j = int(np.searchsorted(Ts, T, side="left"))
    if j == 0:                                   # below the first pillar: blend from the Dirac
        q0 = np.zeros_like(qs[0]); q0[int(np.argmin(np.abs(K - 1.0)))] = 1.0
        T0, V0 = 0.0, 0.0
    else:
        q0, T0, V0 = qs[j - 1], Ts[j - 1], V[j - 1]
    a = (T - T0) / (Ts[j] - T0)
    return (1 - a) * q0 + a * qs[j], eta * ((1 - a) * V0 + a * V[j])


def call(K, q, var, strikes):
    """Undiscounted call prices at `strikes` for the mixture (q on anchors K) at component variance."""
    if var <= 1e-14:
        return np.array([float(np.sum(q * np.maximum(K - k, 0.0))) for k in np.atleast_1d(strikes)])
    s = np.sqrt(var)
    out = []
    for k in np.atleast_1d(strikes):
        d1 = (np.log(np.maximum(K, 1e-12) / k) + 0.5 * var) / s
        out.append(float(np.sum(q * (K * norm.cdf(d1) - k * norm.cdf(d1 - s)))))
    return np.array(out)


def density(K, q, var, strikes):
    """Risk-neutral density of the mixture at `strikes` (lognormal components anchored at K)."""
    s = np.sqrt(max(var, 1e-14))
    out = []
    for k in np.atleast_1d(strikes):
        d = (np.log(k / np.maximum(K, 1e-12)) + 0.5 * var) / s
        out.append(float(np.sum(q * np.exp(-0.5 * d ** 2) / (k * s * np.sqrt(2 * np.pi)))))
    return np.array(out)


def leverage(K, V, qs, Ts, T, EV=1.0, eta=ETA):
    """lambda(z) on the +-KSD window at maturity T, with NO floor and NO safety clip.

    Returns (z grid, lambda, n_nonpositive_numerator) so the caller can see whether the calendar
    slope ever needed rescuing -- under eq.(2) it should not.
    """
    # LAMH=<mult> widens the calendar stencil to h = mult*DT/2. `blend` is PIECEWISE-LINEAR in T, so
    # dC/dT is piecewise-CONSTANT with jumps at pillars -- and a central difference over width 2h IS
    # the average of dC/dT over [T-h, T+h]. Widening h is therefore a boxcar MOLLIFICATION of the
    # derivative at its source, local by construction, with no boundary artifacts and no effect on
    # the marginals. This is the OPPOSITE of the 6e.2 dead end, which NARROWED the stencil to stay
    # inside one linear segment (worse on all four dates). Default 1.0 = the original +-DT/2.
    h = float(os.environ.get("LAMH") or 1.0) * DT / 2
    # PILLAR-AWARE CALENDAR DERIVATIVE (PILLARAWARE=0 restores the fixed +-DT/2 stencil).
    #
    # `blend` is PIECEWISE-LINEAR in T -- a = (T-T0)/(Ts[j]-T0) on both the weights and the variance
    # -- so C(T) has a KINK at every pillar and dC/dT genuinely jumps there. A fixed +-3.5 day
    # stencil that STRADDLES a pillar therefore averages the slopes either side, and the answer
    # depends on where the pillar happens to sit inside the window rather than on the surface.
    #
    # Measured on SPX 2016 (26 pillars, only 14 inside 100d): as the window slides off pillar 52
    # onto 59 across k = 7,8,9 the fitted dlam/dz swings +30.1%, -58.1%, +24.2%, while weeks 10-13,
    # whose windows hold no pillar or a stable one, move -1.0%, -1.3%, -2.1%, -5.9%. Across the four
    # SPX dates the ladder's roughness rank-orders the VOVLEV=1 vov miss exactly (2016 18.6% rough /
    # 40.8% vov; 2018 9.6/15.1; 2024 7.5/9.4; 2022 5.4/5.0).
    #
    # Clipping the stencil to the segment that contains T keeps the difference inside ONE linear
    # piece, so it estimates the derivative of the surface rather than the placement of the grid.
    # Where the segment is wide this is the original central difference unchanged.
    if _PILLAR_AWARE:
        T0, T1 = _segment(Ts, T)
        lo, hi = max(T - h, T0), min(T + h, T1)
        if not (hi > lo):                      # degenerate segment: fall back rather than divide by 0
            lo, hi = max(T - h, 1e-6), T + h
    else:
        lo, hi = max(T - h, 1e-6), T + h
    ql, vl = blend(K, V, qs, Ts, lo, eta)
    qh, vh = blend(K, V, qs, Ts, hi, eta)
    qm, vm = blend(K, V, qs, Ts, T, eta)
    m1 = float(np.sum(qm * K)); m2 = float(np.sum(qm * K ** 2) * np.exp(vm))
    sd = np.sqrt(max(np.log(max(m2 / m1 ** 2, 1 + 1e-12)), 1e-12))
    zg = np.linspace(-KSD * sd, KSD * sd, NZ)
    ks = np.exp(zg)
    # LAMDSG=<npts> estimates dC/dT by a LOCAL POLYNOMIAL (Savitzky-Golay) fit to C(T) across npts
    # samples spanning +-h, instead of the 2-point central difference. The linear coefficient of a
    # local polynomial fit IS the derivative at the centre, and it is a far better estimator than any
    # finite difference when the function has kinks -- which C(T) does, because `blend` is
    # piecewise-linear in T so dC/dT is piecewise CONSTANT and jumps at every pillar (6e.1b).
    #
    # This acts AT THE SOURCE and is LOCAL, unlike the two things already tried: LAMH widens the same
    # 2-point difference (a boxcar average of dC/dT -- helps 2016, barely helps 2012), and
    # LAMSMOOTH/LAMSG smooth lambda AFTER the fact, downstream of the derivative. Degree LAMDSGDEG
    # (default 2). npts < 3 or degree >= npts falls back to the 2-point difference.
    _nsg = int(os.environ.get("LAMDSG") or 0)
    _psg = int(os.environ.get("LAMDSGDEG") or 2)
    if _nsg >= 3 and _psg < _nsg:
        _n = _nsg if _nsg % 2 == 1 else _nsg + 1
        _Tj = T + np.linspace(-h, h, _n)
        _Cj = np.array([call(K, *blend(K, V, qs, Ts, max(t, 1e-6), eta), ks) for t in _Tj])
        _A = np.vander(_Tj - T, _psg + 1)          # highest power first; column -2 is the linear term
        _cf, *_ = np.linalg.lstsq(_A, _Cj, rcond=None)
        num = _cf[-2]                               # d/dT of the local fit, evaluated at T
    else:
        num = (call(K, qh, vh, ks) - call(K, ql, vl, ks)) / (hi - lo)
    den = 0.5 * ks ** 2 * density(K, qm, vm, ks)
    bad = int((num <= 0).sum())
    lam2 = num / np.maximum(den, 1e-300) * DT / EV
    return zg, np.sqrt(np.maximum(lam2, 0.0)), bad


if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "NDX"
    date = sys.argv[2] if len(sys.argv) > 2 else "2020-06-01"
    df = orats_chain_df(f"{ST.ORATS}/SPX-NDX-RUT-VIX_{date}.json.gz", tk)
    slices, dtes = [], []
    for dte, g in sorted(df.groupby("dte")):
        s = ST.market_slice(g)
        if s is not None:
            slices.append(s); dtes.append(int(dte))
    t0 = time.time()
    out = ST.fit_joint(slices, eta=ETA)
    if out is None:
        print("LP failed"); sys.exit(1)
    K, V, qs = out
    Ts = np.array([s["tau"] for s in slices])
    print(f"{tk} {date}: {len(slices)} pillars, eta={ETA}, fit {time.time()-t0:.1f}s\n")

    # (1) calendar monotonicity of the BLENDED surface, on a fine T grid between pillars
    Tg = np.linspace(2 / 365, min(Ts[-1], 1.0), 200)
    ks = np.exp(np.linspace(-0.5, 0.5, 81))
    C = np.array([call(K, *blend(K, V, qs, Ts, T), ks) for T in Tg])
    dmin = float(np.min(np.diff(C, axis=0)))
    mart = max(abs(float(np.sum(blend(K, V, qs, Ts, T)[0] * K)) - 1.0) for T in Tg)
    print(f"calendar monotonicity off-pillar: min dC/dT step = {dmin:+.3e}  "
          f"({'MONOTONE' if dmin >= -1e-12 else 'VIOLATED'})")
    print(f"martingale off-pillar:            max |E[K]-1| = {mart:.2e}\n")

    # (2) the leverage, with no floor and no clip
    print(f"{'week':>5s} {'dte':>5s} {'lam min':>9s} {'lam max':>9s} {'dlam/dz':>9s} {'num<=0':>8s}")
    print("-" * 52)
    nbad = 0
    for k in range(1, 14):
        zg, lam, bad = leverage(K, V, qs, Ts, k * DT)
        nbad += bad
        sl = np.polyfit(zg, np.log(np.maximum(lam, 1e-300)), 1)[0]
        print(f"{k:5d} {int(round(k*DT*365)):5d} {lam.min():9.4f} {lam.max():9.4f} {sl:9.3f} {bad:8d}")
    print(f"\nnodes needing the 1e-14 calendar floor: {nbad}/{13*NZ}"
          f"   ({'NONE -- floor never fires' if nbad == 0 else 'STILL FLOORED'})")

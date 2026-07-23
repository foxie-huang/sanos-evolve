#!/usr/bin/env python3
"""
Held-amplitude SSR re-fit: hold (nu_f, nu_s) at the VIX-identified vol-of-vol amplitudes and fit the
remaining 6 params (nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s) to the SSR term structure. The clean
decoupling the whole thread has been building toward -- VIX identifies the vol-of-vol, the SSR identifies
the leverage/comovement and the timescales. If the SSR still fits with the amplitudes VIX-pinned, nu is
IDENTIFIED (not floated) and the fit is jointly consistent with both markets.

    OOS_DATE=2015-06-01 python3 calibrate_slv_held.py [n_workers]

Reuses the parallel-Jacobian harness: the main process reconstructs the full 8-vector (held amplitudes +
free params) and dispatches it to the pool's _eval, so the fast Jacobian machinery is unchanged.
"""
import sys, os, time, glob
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_slv_exact_ts as C                                   # noqa: E402  exact_vec_ts, NAMES, NS, DT, WREL, LO, HI, X0_MAP
import calibrate_slv_exact_ts_par as P                               # noqa: E402  build_ctx, _init, _eval, OOS_DATE, YR, OUT

VIX_AMP = {"2015-06-01": (0.204, 0.415), "2019-06-03": (0.546, 0.167)}   # VIX-identified (nu_f, nu_s) per date
FREE = [2, 3, 4, 5, 6, 7]                                            # nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s

if __name__ == "__main__":
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    date = P.OOS_DATE
    if date not in VIX_AMP:
        sys.exit(f"no VIX amplitudes for {date}; run calibrate_vix.py first")
    nu_f, nu_s = VIX_AMP[date]
    full0 = C.X0_MAP["ts"].copy(); full0[0], full0[1] = nu_f, nu_s   # start: VIX amplitudes + theta_ts free params
    x0 = full0[FREE].copy(); lo = C.LO[FREE]; hi = C.HI[FREE]

    chain, sig_ref = P.build_ctx()
    from empirical_ssr import empirical_ssr                           # noqa: E402
    emp, nd = empirical_ssr(sorted(glob.glob(f"{P.OUT}/SPX-NDX-RUT-VIX_{P.YR}-*.json.gz")), ns=C.NS, dt=C.DT)
    print(f"HELD (VIX): nu_f={nu_f:.3f}  nu_s={nu_s:.3f}   date={date}   fitting {[C.NAMES[i] for i in FREE]}", flush=True)
    print(f"emp SSR ({nd} {P.YR}): {np.round(emp, 3)}  EQUAL % weight", flush=True)

    def reconstruct(x):
        full = full0.copy(); full[FREE] = x; return full

    _last = {}
    def ev(x):
        full = reconstruct(np.asarray(x, float)); key = full.tobytes()
        if _last.get("k") != key:
            _last["k"] = key; _last["v"] = C.exact_vec_ts(full, chain, sig_ref)
        return _last["v"]

    def resid(x):
        return C.WREL * (ev(x) - emp) / emp

    pool = ProcessPoolExecutor(max_workers=NW, initializer=P._init)

    def jac(x, *a):
        x = np.asarray(x, float); f0 = ev(x)
        h = 5e-2 * np.maximum(np.abs(x), 1.0)
        steps = np.where(x + h <= hi, h, -h)
        xs = [reconstruct(x + steps[i] * np.eye(len(x))[i]) for i in range(len(x))]   # full 8-vectors to workers
        fps = list(pool.map(P._eval, xs))
        J = np.zeros((len(emp), len(x)))
        for i in range(len(x)):
            J[:, i] = C.WREL * (fps[i] - f0) / emp / steps[i]
        return J

    t0 = time.time()
    res = least_squares(resid, x0, jac=jac, bounds=(lo, hi), max_nfev=60, xtol=1e-6, ftol=1e-6, verbose=2)
    pool.shutdown()
    full = reconstruct(res.x); mod = C.exact_vec_ts(full, chain, sig_ref, nz=15)
    print(f"\nHELD-AMPLITUDE SSR fit  date={date}  ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(C.NAMES, full)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in C.LABELS))
    print(f"{'ssr':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{'emp':6}" + "".join(f"{v:8.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))

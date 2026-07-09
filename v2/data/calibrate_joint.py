#!/usr/bin/env python3
"""
JOINT SSR + VIX calibration. The sequential route (VIX->amplitudes, hold, SSR->rest) breaks: the SSR's
free params (nu_l, kap_s, lam_f, ...) ALSO move the VIX vol-of-vol, so holding only (nu_f,nu_s) doesn't
keep the VIX consistent. Fit ALL 8 params to a COMBINED objective -- the SSR term structure AND the VIX
vol-of-vol -- so nu is identified by VIX and the leverage by the SSR, JOINTLY (the paper's eq:objective +
eq:Rvov soft target). Pooled Jacobian; each worker eval returns concat[ssr(5), vov(n_vix)].

    OOS_DATE=2015-06-01 python3 calibrate_joint.py [n_workers] [w_vov]
"""
import sys, os, time, glob
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_slv_exact_ts as C                                   # noqa: E402
import calibrate_slv_exact_ts_par as P                               # noqa: E402
from discslv_2f import TwoFactorSV                                    # noqa: E402
from slv_wire import solve_gbar                                      # noqa: E402
from vix_readout import model_vix_ivol, data_vix                      # noqa: E402

MIN_DTE = 7
_W = {}


def _vix_targets(sig_ref):
    spot, dv = data_vix(P.OOS_DATE); dv = [d for d in dv if d[0] >= MIN_DTE]
    return spot, np.array([d[0] for d in dv], float), np.array([d[2] for d in dv])


def _model(theta, chain, sig_ref, spot, vdtes, nz=C.NZ):
    """concat[ SSR term structure (exact), VIX vol-of-vol at each expiry ] for one theta."""
    ssr = C.exact_vec_ts(np.asarray(theta, float), chain, sig_ref, nz=nz)
    kw = dict(zip(C.NAMES, np.asarray(theta, float)))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=C.DT), dt=C.DT, n_f=5, n_s=3, n_l=5, **kw)
    vov = np.array([model_vix_ivol(K, sig_ref, d / 365.0, spot=spot)[1] for d in vdtes])
    return np.concatenate([ssr, vov])


def _jinit():
    _W["chain"], _W["sig_ref"] = P.build_ctx()                       # also populates C._CACHE for the SSR leverage
    _W["spot"], _W["vdtes"], _ = _vix_targets(_W["sig_ref"])


def _jeval(theta):
    return _model(theta, _W["chain"], _W["sig_ref"], _W["spot"], _W["vdtes"])


if __name__ == "__main__":
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    W_VOV = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8         # relative weight on the VIX block
    date = P.OOS_DATE
    chain, sig_ref = P.build_ctx()
    spot, vdtes, vov_d = _vix_targets(sig_ref)
    from empirical_ssr import empirical_ssr                           # noqa: E402
    emp, nd = empirical_ssr(sorted(glob.glob(f"{P.OUT}/SPX-NDX-RUT-VIX_{P.YR}-*.json.gz")), ns=C.NS, dt=C.DT)
    n_ssr = len(emp); wv = W_VOV * np.sqrt(n_ssr / len(vdtes))        # balance the two blocks by point count
    WARM = {"2015-06-01": np.array([0.204, 0.415, 1.101, -0.300, 0.625, 2.455, 0.921, 2.575]),    # held-fit theta
            "2020-06-01": np.array([0.208, 0.411, 1.070, -0.303, 0.633, 2.092, 0.937, 2.706])}    # small-nu_f basin (2015-warm)
    x0 = WARM.get(date, C.X0_MAP["ts"]).copy()                        # warm start where the cold start hit a local min
    print(f"JOINT SSR+VIX  date={date}  NW={NW}  w_vov={W_VOV} (block wt {wv:.2f})", flush=True)
    print(f"emp SSR: {np.round(emp,3)}   VIX dtes: {vdtes.astype(int)}", flush=True)

    def split(v):
        return v[:n_ssr], v[n_ssr:]

    _last = {}
    def ev(x):
        key = np.asarray(x, float).tobytes()
        if _last.get("k") != key:
            _last["k"] = key; _last["v"] = _model(x, chain, sig_ref, spot, vdtes)
        return _last["v"]

    def resid(x):
        s, v = split(ev(x))
        return np.concatenate([C.WREL * (s - emp) / emp, wv * (v - vov_d) / vov_d])

    pool = ProcessPoolExecutor(max_workers=NW, initializer=_jinit)

    def jac(x, *a):
        x = np.asarray(x, float); f0 = ev(x)
        h = 5e-2 * np.maximum(np.abs(x), 1.0)
        steps = np.where(x + h <= C.HI, h, -h)
        xs = [x + steps[i] * np.eye(len(x))[i] for i in range(len(x))]
        fps = list(pool.map(_jeval, xs))
        J = np.zeros((n_ssr + len(vdtes), len(x)))
        for i in range(len(x)):
            ds, dv = split((fps[i] - f0) / steps[i])
            J[:, i] = np.concatenate([C.WREL * ds / emp, wv * dv / vov_d])
        return J

    t0 = time.time()
    res = least_squares(resid, x0, jac=jac, bounds=(C.LO, C.HI), max_nfev=80, xtol=1e-6, ftol=1e-6, verbose=2)
    pool.shutdown()
    s, v = split(_model(res.x, chain, sig_ref, spot, vdtes, nz=15))
    print(f"\nJOINT fit  date={date}  ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={val:.3f}" for n, val in zip(C.NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in C.LABELS))
    print(f"{'ssr':6}" + "".join(f"{val:8.3f}" for val in s))
    print(f"{'emp':6}" + "".join(f"{val:8.3f}" for val in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(s, emp)))
    print(f"\nVIX vol-of-vol  RMS err {np.sqrt(np.mean(((v-vov_d)/vov_d)**2))*100:.0f}%")
    print(f"  {'dte':>5}{'data':>8}{'model':>8}{'err':>7}")
    for i, d in enumerate(vdtes):
        print(f"  {int(d):>5}{vov_d[i]:>8.3f}{v[i]:>8.3f}{100*(v[i]-vov_d[i])/vov_d[i]:>6.0f}%")

#!/usr/bin/env python3
"""
Parallel-Jacobian calibration (reuses calibrate_slv_exact_ts.exact_vec_ts unchanged).

WHY not pure numpy: each eval propagates nf*ns*nz independent sigma_ATM(z,f,s) chains, and every chain
must be recompressed on its own (recompress_2f merges its mixture to nk components) -- packing them into
one array would merge distinct grid points together. So the grid is independent chains, not a tensor.
But the optimizer's finite-difference Jacobian (8 params -> 8 independent theta-evals per iteration) IS
embarrassingly parallel -> evaluate those columns across a ProcessPool, using the idle cores.

Usage:  python3 calibrate_slv_exact_ts_par.py <dense|low> [n_workers]
        python3 calibrate_slv_exact_ts_par.py verify          # check a worker eval == the serial value
"""
import sys, os, time, glob
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_slv_exact_ts as C                              # exact_vec_ts, NAMES, NS, DT, WREL, LO, HI, X0_MAP, _CACHE
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402
from discslv_2f import TwoFactorSV                              # noqa: E402
from discslv_slv import Epi_V                                   # noqa: E402

OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
OOS_DATE = os.environ.get("OOS_DATE", "2015-06-01")                # override date (statics + year target) for OOS
DATE = OUT + f"/SPX-NDX-RUT-VIX_{OOS_DATE}.json.gz"; YR = OOS_DATE[:4]
_CTX = {}


def build_ctx():
    """(chain, sig_ref) + populate C._CACHE with the theta-invariant per-step leverage. Deterministic ->
    identical in the main process and every worker (EV is pinned to sig_ref^2*dt by the gamma-bar reset)."""
    chain = sanos_chain(DATE); sig_ref = ref_vol(chain)
    kw0 = dict(zip(C.NAMES, C.X0_MAP["dense"]))
    K0 = TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=C.DT), dt=C.DT, n_f=5, n_s=3, n_l=5, **kw0)
    EV0 = Epi_V(K0)
    C._CACHE.clear(); C._CACHE.update({k: leverage_at(chain, k * C.DT, EV0, dt=C.DT) for k in range(1, max(C.NS) + 1)})
    return chain, sig_ref


def _init():
    _CTX["chain"], _CTX["sig_ref"] = build_ctx()


def _eval(x):
    return C.exact_vec_ts(np.asarray(x, float), _CTX["chain"], _CTX["sig_ref"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dense"

    if mode == "verify":                                       # worker path must equal the serial value
        chain, sig_ref = build_ctx()
        serial = C.exact_vec_ts(C.X0_MAP["dense"], chain, sig_ref, nz=9)
        with ProcessPoolExecutor(max_workers=1, initializer=_init) as pool:
            worker = list(pool.map(_eval, [C.X0_MAP["dense"]]))[0]
        print("serial:", np.round(serial, 4))
        print("worker:", np.round(worker, 4))
        print("max|diff| =", float(np.max(np.abs(serial - worker))), "->", "OK" if np.allclose(serial, worker) else "MISMATCH")
        sys.exit(0)

    START = mode; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    X0 = C.X0_MAP[START]
    chain, sig_ref = build_ctx()
    from empirical_ssr import empirical_ssr                     # noqa: E402
    emp, nd = empirical_ssr(sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{YR}-*.json.gz")), ns=C.NS, dt=C.DT)
    print(f"START={START}  NW={NW}  X0={np.round(X0, 3)}", flush=True)
    print(f"empirical SSR ({nd} {YR}): {np.round(emp, 3)}  EQUAL % weight (parallel Jacobian, EXACT beta + term-structure)", flush=True)

    _last = {}
    def ev(x):                                                 # 1-entry memo: resid(x) then jac(x) share the base eval
        key = np.asarray(x, float).tobytes()
        if _last.get("k") != key:
            _last["k"] = key; _last["v"] = C.exact_vec_ts(np.asarray(x, float), chain, sig_ref)
        return _last["v"]

    def resid(x):
        return C.WREL * (ev(x) - emp) / emp

    pool = ProcessPoolExecutor(max_workers=NW, initializer=_init)

    def jac(x, *a):                                            # parallel forward-difference (rel step 5e-2)
        x = np.asarray(x, float); f0 = ev(x)
        h = 5e-2 * np.maximum(np.abs(x), 1.0)
        steps = np.where(x + h <= C.HI, h, -h)                 # backward diff at an upper bound
        xs = [x + steps[i] * np.eye(len(x))[i] for i in range(len(x))]
        fps = list(pool.map(_eval, xs))
        J = np.zeros((len(emp), len(x)))
        for i in range(len(x)):
            J[:, i] = C.WREL * (fps[i] - f0) / emp / steps[i]
        return J

    t0 = time.time()
    res = least_squares(resid, X0, jac=jac, bounds=(C.LO, C.HI), max_nfev=60, xtol=1e-6, ftol=1e-6, verbose=2)
    pool.shutdown()
    mod = C.exact_vec_ts(res.x, chain, sig_ref, nz=15)
    print(f"\nEXACT-beta + TERM-STRUCTURE (parallel), START={START}  ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(C.NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in C.LABELS))
    print(f"{'ts':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{'emp':6}" + "".join(f"{v:8.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))

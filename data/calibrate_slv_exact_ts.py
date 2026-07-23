#!/usr/bin/env python3
"""
Calibrate 1wk-3m with the EXACT beta AND the per-step TERM-STRUCTURE leverage (fused_ssr_exact_ts):
each propagation step k gets leverage_at((k+1)*dt) instead of one frozen leverage_at(T). The frozen-
leverage dense re-fit hit the belly to -16/-17% by pumping vol-of-vol (nu_f 0.87); bolting the term-
structure leverage on at that theta then OVERSHOT the short-mid (2wk +15%, 1m +7%) while nailing the
belly (2m +1%, 3m -5%) -- the tell that theta needs re-balancing AROUND the ts leverage. This re-fit
lets the optimizer dial the SV back down where the ts leverage now carries the belly, for (hopefully)
a flat error across the whole curve. The honest best-faithful-fit.

The leverage cache is THETA-INVARIANT: solve_gbar resets gamma-bar so EV = sigma_ref^2 * dt for every
theta (paper Sec.748), so leverage_at(., EV) is the same map every eval -> build it ONCE.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact_ts               # noqa: E402
discslv_slv.propagate = propagate_vec
from empirical_ssr import empirical_ssr                               # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]; NZ = 9
WREL = np.array([1.0, 1.0, 1.0, 1.0, 1.0])   # EQUAL % weight: residual = (model-emp)/emp -> equal % error target,
#   short end no longer sacrificed (the old belly-heavy [1,1.5,2,1.5,1] on ABSOLUTE residuals spent the well-
#   determined 1wk to buy the belly; with the belly now reachable, that was backwards).
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
LO = np.array([0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([1.20, 1.20, 1.50, 0.0, 8.0, 8.0, 1.0, 4.0])                  # nu_l HI 1.0->1.5 (2015 fit railed it)
X0_MAP = {                                                                 # multi-start: don't trust one basin
    "dense": np.array([0.872, 0.532, 0.831, -0.652, 0.489, 2.088, 0.769, 2.991]),   # high-nu (frozen dense re-fit)
    "low":   np.array([0.280, 0.293, 0.465, -2.113, 1.164, 2.460, 0.992, 2.518]),   # low-nu/high-skew (sparse fit, fit 1wk +2%)
    "ts":    np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780]),   # 2015 dense+ts+equal-wt fit (well-behaved OOS start)
}
_CACHE = {}


def exact_vec_ts(x, chain, sig_ref, nz=NZ):
    kw = dict(zip(NAMES, x)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    out = []
    for n in NS:
        lam_fns = [_CACHE[k + 1] for k in range(n)]                  # step k -> maturity (k+1)*dt (theta-invariant)
        out.append(fused_ssr_exact_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, 16, DT, nz=nz)[0])
    return np.array(out)


if __name__ == "__main__":
    START = sys.argv[1] if len(sys.argv) > 1 else "dense"
    X0 = X0_MAP[START]
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    date = OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"; yr = "2015"
    chain = sanos_chain(date); sig_ref = ref_vol(chain)
    kw0 = dict(zip(NAMES, X0)); K0 = TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0)
    EV0 = Epi_V(K0)                                                   # theta-invariant EV (gamma-bar reset)
    _CACHE.update({k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, max(NS) + 1)})
    emp, nd = empirical_ssr(sorted(__import__("glob").glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=NS, dt=DT)
    print(f"START={START}  X0={np.round(X0,3)}", flush=True)
    print(f"empirical SSR ({nd} {yr}): {np.round(emp, 3)}  EQUAL % weight {WREL}  (EXACT beta + TERM-STRUCTURE leverage, weekly)", flush=True)
    t0 = time.time()
    res = least_squares(lambda x: WREL * (exact_vec_ts(x, chain, sig_ref) - emp) / emp, X0, bounds=(LO, HI),
                        diff_step=5e-2, max_nfev=60, xtol=1e-6, ftol=1e-6, verbose=2)
    mod = exact_vec_ts(res.x, chain, sig_ref, nz=15)
    print(f"\nEXACT-beta + TERM-STRUCTURE calibration ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in LABELS))
    print(f"{'ts':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{'emp':6}" + "".join(f"{v:8.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))

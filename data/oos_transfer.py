#!/usr/bin/env python3
"""
Out-of-sample transfer test (the backtest premise): hold theta_ts FIXED (fit on 2015), swap in another
date's SANOS chain (statics = marginals + leverage), evaluate the fused SSR with the term-structure
leverage, and compare to THAT year's realized empirical SSR. Only the vol LEVEL adapts (sig_ref -> gbar
via the gamma-bar reset); the dynamics SHAPE (nu/kap/lam_skew) is held at theta_ts. If it still fits,
theta transfers across regimes and a backtest can hold theta fixed + re-run only the cheap per-date
statics. If it misses, the SSR term structure is regime-dependent (itself a finding).

    python3 oos_transfer.py 2019-06-03
"""
import sys, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact_ts               # noqa: E402
discslv_slv.propagate = propagate_vec
from empirical_ssr import empirical_ssr                               # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
THETA_TS = np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780])   # 2015 dense+ts+equal-wt fit
EMP_2015 = np.array([2.027, 1.658, 1.595, 1.538, 1.457])
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))

if __name__ == "__main__":
    date = sys.argv[1]; yr = date[:4]
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
    kw = dict(zip(NAMES, THETA_TS))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    cache = {k: leverage_at(chain, k * DT, EV, dt=DT) for k in range(1, max(NS) + 1)}

    emp, nd = empirical_ssr(sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=NS, dt=DT)
    mod = np.array([fused_ssr_exact_ts(K, [cache[k + 1] for k in range(n)], n, EV, nub, Vlr, tiltr, 16, DT, nz=15)[0]
                    for n in NS])

    print(f"OOS TRANSFER: theta_ts (2015 fit) HELD, applied to {date} statics  (sig_ref={sig_ref:.3f})")
    print("theta_ts: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, THETA_TS)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in LABELS))
    print(f"{'model':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{f'emp{yr}':6}" + "".join(f"{v:8.3f}" for v in emp) + f"   ({nd} days)")
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))
    print(f"\n{'emp2015':6}" + "".join(f"{v:8.3f}" for v in EMP_2015) + "   (in-sample target, for reference)")
    print(f"{'shift':6}" + "".join(f"{100*(a-b)/b:7.0f}%" for a, b in zip(emp, EMP_2015)) + f"   ({yr} realized vs 2015 realized")

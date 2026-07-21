#!/usr/bin/env python3
"""Diagnose the 2019 OOS miss: is theta regime-dependent, or did the 2019-06-03 SANOS leverage come out
flat (a statics artifact)? Compare, per maturity, the leverage skew dlam/dz and the chain coverage for
the in-sample 2015-06-01 vs the OOS 2019-06-03. A flat 2019 dlam/dz => statics issue, not theta."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V                                         # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
THETA_TS = np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780])
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))

for date in ["2015-06-01", "2019-06-03"]:
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
    kw = dict(zip(NAMES, THETA_TS)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K)
    print(f"\n=== {date}  sig_ref={sig_ref:.3f}  chain expiries={len(chain)}  (T range {chain[0][0]*365:.0f}-{chain[-1][0]*365:.0f} dte) ===")
    print(f"{'mat':>5}{'dlam/dz':>9}{'lam(-.05)':>10}{'lam(0)':>8}{'lam(+.05)':>10}")
    for n, lab in zip(NS, LABELS):
        lam = leverage_at(chain, n * DT, EV, dt=DT)
        dsl = float((lam(0.02) - lam(-0.02)) / 0.04)
        print(f"{lab:>5}{dsl:>9.1f}{float(lam(-0.05)):>10.3f}{float(lam(0.0)):>8.3f}{float(lam(0.05)):>10.3f}")

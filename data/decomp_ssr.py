#!/usr/bin/env python3
"""
Proper three-way SSR decomposition (forward-from-Dupire LV, not a residual):
  LV    = real leverage, kernel OFF (nu_f=nu_s=nu_l->0) -> the pure local-vol backbone the Dupire
          leverage dictates. This is the leverage's OWN SSR contribution, computed forward.
  SV    = FLAT leverage (lambda=1), real theta -> the pure kernel (stochastic-vol) contribution.
  Total = real leverage, real theta (the fused model).
  inter = Total - LV - SV  -> the leverage x kernel interaction (the "genuine coupling"), measured
          explicitly instead of being hidden inside a Total-SV residual.

Blocks: theta_ts @2015 (in-sample), theta_ts @2019 (transfer -> shows LV drop at fixed theta),
theta_2019 @2019 (re-fit -> shows SV raised to compensate). All with the term-structure leverage.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact_ts               # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]; NZ = 15
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
THETA_TS = np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780])   # 2015 fit
THETA_2019 = np.array([0.786, 0.375, 0.955, -0.275, 0.671, 2.123, 0.834, 2.093]) # 2019 re-fit
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
flat = lambda z: np.ones_like(np.asarray(z, float))


def kernel(theta, sig_ref):
    kw = dict(zip(NAMES, theta)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    return K, Epi_V(K)


def decomp(date, theta, label):
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
    K, EV = kernel(theta, sig_ref); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    tlv = theta.copy(); tlv[0:3] = 1e-3                              # nu->0: kernel off -> pure local vol (LV)
    K0, EV0 = kernel(tlv, sig_ref); nub0 = nu_bar(K0, EV0); Vlr0, tiltr0 = raw_increment(K0)
    cache = {k: leverage_at(chain, k * DT, EV, dt=DT) for k in range(1, max(NS) + 1)}
    print(f"\n=== {label}   date={date}  sig_ref={sig_ref:.3f} ===")
    print(f"{'mat':>4}{'Total':>8}{'LV(Dup)':>9}{'SV':>7}{'inter':>8}{'LV+SV':>8}")
    for n, lab in zip(NS, LABELS):
        lam_fns = [cache[k + 1] for k in range(n)]
        tot = fused_ssr_exact_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, 16, DT, nz=NZ)[0]
        sv = fused_ssr_exact_ts(K, [flat] * n, n, EV, nub, Vlr, tiltr, 16, DT, nz=NZ)[0]
        lv = fused_ssr_exact_ts(K0, lam_fns, n, EV0, nub0, Vlr0, tiltr0, 16, DT, nz=NZ)[0]
        print(f"{lab:>4}{tot:>8.3f}{lv:>9.3f}{sv:>7.3f}{tot-lv-sv:>8.3f}{lv+sv:>8.3f}", flush=True)


if __name__ == "__main__":
    decomp("2015-06-01", THETA_TS, "theta_ts @ 2015 (in-sample)")
    decomp("2019-06-03", THETA_TS, "theta_ts @ 2019 (TRANSFER)")
    decomp("2019-06-03", THETA_2019, "theta_2019 @ 2019 (re-fit)")

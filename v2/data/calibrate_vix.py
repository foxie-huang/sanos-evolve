#!/usr/bin/env python3
"""
Identify the SLOW factor (nu_s, kap_s) from the VIX vol-of-vol term structure -- the observable the SSR
alone cannot separate. 2-parameter least_squares fit of (nu_s, kap_s) to the model-vs-data VIX ATM
implied-vol curve, holding the other 6 params at theta_ts. Per date (2015 in-sample, 2019 OOS). The
fitted (nu_s, kap_s) then feed the held-slow SSR re-fit so nu is IDENTIFIED, not floated.
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv                                                    # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from discslv_2f import TwoFactorSV                                    # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar                 # noqa: E402
from vix_readout import model_vix_ivol, data_vix                      # noqa: E402

DT = 1.0 / 52.0
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
THETA_TS = np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780])
INU_F, INU_S = 0, 1                                                 # the two vol-of-vol AMPLITUDES (fast, slow)
OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"


def vov_curve(x, dtes, sig_ref, spot, base=THETA_TS):
    """Model VIX vol-of-vol at each expiry for x=[nu_f, nu_s] (the fast/slow AMPLITUDES), rest held at
    base. Short expiries pin nu_f (fast), long pin nu_s (slow); kap_s is impotent for the shape so it
    is NOT a fit target here."""
    th = base.copy(); th[INU_F], th[INU_S] = x
    kw = dict(zip(NAMES, th)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    return np.array([model_vix_ivol(K, sig_ref, d / 365.0, spot=spot)[1] for d in dtes])


MIN_DTE = 7                                                         # drop ultra-short (<1wk) VIX options (noisy/discrete)


def fit_date(date):
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
    spot, dv = data_vix(date)
    dtes = np.array([d[0] for d in dv]); vov_d = np.array([d[2] for d in dv])
    use = dtes >= MIN_DTE                                           # full term structure: short->nu_f, long->nu_s
    x0 = np.array([THETA_TS[INU_F], THETA_TS[INU_S]])
    res = least_squares(lambda x: vov_curve(x, dtes[use], sig_ref, spot) - vov_d[use],
                        x0, bounds=([0.02, 0.05], [1.2, 1.2]), xtol=1e-10, diff_step=3e-2)
    return res.x, dtes, vov_d, vov_curve(res.x, dtes, sig_ref, spot), vov_curve(x0, dtes, sig_ref, spot), use


if __name__ == "__main__":
    for date in ["2015-06-01", "2019-06-03"]:
        x, dtes, vd, vm, v0, use = fit_date(date)
        print(f"\n=== {date}  VIX vol-of-vol fit  (amplitudes nu_f, nu_s; dte>={MIN_DTE}) ===")
        print(f"  nu_f: {THETA_TS[INU_F]:.3f} -> {x[0]:.3f}    nu_s: {THETA_TS[INU_S]:.3f} -> {x[1]:.3f}")
        print(f"  {'dte':>5}{'data':>8}{'model@ts':>10}{'model@fit':>11}{'fit.err':>9}")
        for i, d in enumerate(dtes):
            if use[i]:
                print(f"  {d:>5}{vd[i]:>8.3f}{v0[i]:>10.3f}{vm[i]:>11.3f}{100*(vm[i]-vd[i])/vd[i]:>8.0f}%", flush=True)

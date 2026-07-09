#!/usr/bin/env python3
"""Diagnose the abnormal 2023-06-01 fit (SSR 34.5% / VIX 24.1%, lam_skew railed) vs a clean year (2019).
Checks: empirical-SSR target (degenerate?), SANOS chain coverage, ATM vol/skew term structure, VIX vov,
and the per-maturity/expiry model-vs-target residual at the fitted theta."""
import sys, os, json, glob
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_joint_torch as J
import calibrate_slv_exact_ts as C
from empirical_ssr import empirical_ssr
from vix_readout import data_vix
from discslv_slv import atm_skew_of
from slv_interp import interp_marginal
from slv_wire import sanos_chain

TH = json.load(open(os.path.join(HERE, "robustness_years.json")))

for date, yr in [("2019-06-03", "2019"), ("2023-06-01", "2023")]:
    print(f"\n{'='*60}\n{date}   (fit: SSR {TH[date]['fit_ssr']}% / VIX {TH[date]['fit_vix']}%)\n{'='*60}")
    emp, nd = empirical_ssr(sorted(glob.glob(f"{J.OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=C.NS, dt=C.DT)
    print(f"  EMPIRICAL SSR target ({nd} days, {C.LABELS}): {np.round(emp, 3)}")
    chain = sanos_chain(f"{J.OUT}/SPX-NDX-RUT-VIX_{date}.json.gz")
    print(f"  SANOS chain: {len(chain)} expiries, T {chain[0][0]:.3f}..{chain[-1][0]:.2f}y")
    print("  real ATM vol/skew term structure:")
    for wk in [4, 8, 13, 26, 52]:
        av, sk = atm_skew_of(interp_marginal(chain, wk * C.DT), wk * C.DT)
        print(f"     {wk:2d}wk: vol {av:.4f}  skew {sk:+.2f}")
    spot, dv = data_vix(date); dv = [d for d in dv if d[0] >= 7]
    print(f"  VIX: spot {spot:.3f}  vov: " + " ".join(f"{int(d[0])}d={d[2]:.3f}" for d in dv))
    # model vs target at the fitted theta
    theta = np.array(TH[date]["theta"], float)
    ctx = J.build_date_ctx(date)
    m = J.model_torch(torch.tensor(theta, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                      ctx["spot"], ctx["vdtes"]).detach().numpy()
    ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
    print(f"  theta = {np.round(theta,3)}")
    print(f"  SSR  model {np.round(s,3)}  vs emp {np.round(ctx['emp'],3)}  ->err {np.round(100*(s-ctx['emp'])/ctx['emp'],0)}%")
    print(f"  VIX  model {np.round(v,3)}  vs data {np.round(ctx['vov_d'],3)} ->err {np.round(100*(v-ctx['vov_d'])/ctx['vov_d'],0)}%")

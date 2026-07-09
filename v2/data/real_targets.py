#!/usr/bin/env python3
"""
real_targets.py -- extract the REAL calibration targets from ORATS data, to swap in for
calibrate_2f's hardcoded canonical ones. No new model code: reuse the SANOS fit
(orats_sanos) + the kernel's own _atm_skew readout (discslv_2f).

statics_targets(path): per date -> (ATM vol(T_j), ATM skew(T_j)) from the real SANOS
marginals, at the model maturity grid NS=[4,13,26,52] steps (1m/3m/6m/1y at dt=1/52),
by log-linear interpolation of the fitted term structure onto those maturities.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from sanos_lp import prep_expiry, sanos_fit          # noqa: E402
from discslv import GMM                              # noqa: E402
from discslv_2f import _atm_skew                     # noqa: E402
from orats_sanos import orats_chain_df, eta_for_dte  # noqa: E402

DT = 1.0 / 52.0
NS = [4, 13, 26, 52]                                 # calibrate_2f's grid: 1m/3m/6m/1y


def date_smiles(path, ticker="SPX"):
    """Per maturity: (T_years, ATM vol, ATM skew) from the real SANOS-fitted marginal."""
    df = orats_chain_df(path, ticker)
    out = []
    for dte, g in df.groupby("dte"):
        import sanos_lp; sanos_lp.ETA = eta_for_dte(int(dte))
        e = prep_expiry(g)
        if e.get("n", 0) < 20:
            continue
        q, _ = sanos_fit(e)
        if q is None:
            continue
        # SANOS marginal as a GMM on the unit forward (mean log(kappa), width sqrt(V))
        W = np.asarray(q); MU = np.log(e["kappa"]); SG = np.full_like(W, np.sqrt(e["V"]))
        MU = MU - np.log(np.sum(W * np.exp(MU + 0.5 * SG ** 2)))    # martingale lock
        vol, skew = _atm_skew(GMM(W, MU, SG, F=1.0), e["tau"])
        out.append((float(e["tau"]), float(vol), float(skew)))
    return sorted(out)


def statics_targets(path, ticker="SPX", ns=NS, dt=DT):
    """ATM vol/skew at the model maturities (n*dt), interpolated from the date's term structure."""
    sm = date_smiles(path, ticker)
    T = np.array([s[0] for s in sm]); vol = np.array([s[1] for s in sm]); skew = np.array([s[2] for s in sm])
    Tt = np.array(ns) * dt
    return dict(T=Tt, iv=np.interp(Tt, T, vol), sk=np.interp(Tt, T, skew), raw=sm)


if __name__ == "__main__":
    OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
    path = sys.argv[1] if len(sys.argv) > 1 else OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
    t = statics_targets(path)
    print(f"{os.path.basename(path)}  real statics targets (interp to 1m/3m/6m/1y):")
    print("  maturities :", "  ".join(f"{n}n={n*DT:.3f}y" for n in NS))
    print("  ATM vol    :", np.round(t["iv"], 4), "   | canonical was 0.17 0.17 0.17 0.17")
    print("  ATM skew   :", np.round(t["sk"], 3), "   | canonical was -0.50 -0.40 -0.34 -0.28")
    print("  (raw term structure: " + ", ".join(f"{T:.3f}y:{v:.3f}/{s:+.2f}" for T, v, s in t["raw"][:8]) + " ...)")

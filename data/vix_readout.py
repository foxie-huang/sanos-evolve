#!/usr/bin/env python3
"""
Forward-variance (VIX) readout: the SEPARATE dynamic observable that identifies the vol-of-vol, which
the SSR-only calibration left floating. Model side = the kernel's atomic regime law (paper eq:vix-atomic:
VIX_T=sqrt(vbar_k) on regime k w.p. p_k), with mean-reversion damping over the 30d variance window
(fast factor damped, slow barely). Data side = VIX option ATM implied vol (the vol-of-VIX). Compared at
theta_ts: if the model's implied vol-of-vol is far from what VIX prices, nu was under-identified.

Regime variance (discslv_2f L40): V(f,s,l)=exp(gbar+nu_f zf[f]+nu_s zs[s]+nu_l zl[l])*dt; l-averaged
annualized rate = exp(gbar+nu_f zf[f]+nu_s zs[s]+nu_l^2/2). The dispersion across (f,s) IS the vol-of-vol.
"""
import sys, os, json, gzip
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV, stationary_pi                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar                 # noqa: E402
from slv_dtscale import factor_autocorr                              # noqa: E402

DT = 1.0 / 52.0
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
THETA_TS = np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780])
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
TAU_VAR = 30.0 / 365.0                                               # VIX 30-day forward-variance window


def model_vix_ivol(K, sig_ref, tau_opt, spot=None):
    """Model VIX ATM implied vol at option expiry tau_opt -- FAITHFUL (no affine/damping approximation).
      1. Per-regime annualized forward-variance rate v_fs = sig_ref^2 * nu_bar (the FULL one-step
         increment variance V-bar = E_l[V]+Var_l(d), level-matched to sig_ref^2 by the gamma-bar reset).
      2. 30d forward variance carried by a regime = EXACT multi-step expected variance,
         Y[f,s] = (1/n_var) sum_{m=1}^{n_var} (Pf^m @ v @ (Ps^m)^T)[f,s], with Pf,Ps the l-averaged
         factor transitions -- this IS the discrete E_t[int v du]/tau, no exponential-damping shortcut.
      3. Terminal regime law at the option expiry = central (ATM) regime propagated n_opt steps (widens
         from a point toward stationary -> the vol-of-vol term structure).
      4. Atomic VIX law VIX=sqrt(Y) w.p. p; ATM option E_p[(VIX-F)^+]; invert Black over tau_opt."""
    from discslv_slv import Epi_V, nu_bar
    EV = Epi_V(K); v = (sig_ref ** 2) * nu_bar(K, EV)               # (n_f,n_s) forward-variance rate per regime
    # EXACT joint regime transition on the product state (f,s), index f*n_s+s. The branch l is COMMON
    # to both factors, so the joint does NOT factorise into the branch-averaged marginals.
    Pj = sum(K.wl[l] * np.kron(K.Tf[l], K.Ts[l]) for l in range(K.n_l))
    n_var = max(1, int(round(TAU_VAR / K.dt)))                       # steps in the 30d variance window
    vv = v.reshape(-1); Y = np.zeros_like(vv); Pm = np.eye(K.n_f * K.n_s)
    for _ in range(n_var):                                           # exact m-step-ahead expected variance, m=1..n_var
        Pm = Pm @ Pj; Y = Y + Pm @ vv
    Y = (Y / n_var).reshape(v.shape)
    vix = np.sqrt(Y)                                                 # atomic VIX per regime
    n_opt = max(1, int(round(tau_opt / K.dt)))                       # steps to option expiry
    fc = K.n_f // 2                                                  # fast factor reverts fast -> start central
    s0 = K.n_s // 2 if spot is None else int(np.argmin(np.abs(vix[fc, :] - spot)))   # slow regime = today's vol state
    p = np.linalg.matrix_power(Pj, n_opt)[fc * K.n_s + s0].reshape(v.shape)          # terminal regime law (joint)
    F = float(np.sum(p * vix))                                       # VIX forward = E_p[VIX]
    Catm = float(np.sum(p * np.maximum(vix - F, 0.0)))              # ATM VIX call (undiscounted, on the forward)
    iv = brentq(lambda s: F * (2 * norm.cdf(s * np.sqrt(tau_opt) / 2) - 1) - Catm, 1e-4, 8.0)
    return F, iv, float(np.sum(p * Y))


def data_vix(date):
    """Per VIX expiry: forward via put-call PARITY regression (C-P = DF*(F-K) over central strikes) and
    the ATM implied vol interpolated at K=F on the OTM smile (call IV above F, put IV below). Faithful --
    not a nearest-strike proxy."""
    rows = json.load(gzip.open(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz", "rt"))["strikes"]
    vix = [r for r in rows if r["ticker"] == "VIX"]
    spot = float(vix[0]["spotPrice"]) / 100.0                        # VIX index level as a vol (14.52 -> 0.1452)
    out = []
    for dte in sorted(set(r["dte"] for r in vix)):
        e = sorted([r for r in vix if r["dte"] == dte], key=lambda r: r["strike"])
        Kk = np.array([r["strike"] for r in e], float)
        cmid = np.array([0.5 * (r.get("callBidPrice", np.nan) + r.get("callAskPrice", np.nan)) for r in e])
        pmid = np.array([0.5 * (r.get("putBidPrice", np.nan) + r.get("putAskPrice", np.nan)) for r in e])
        civ = np.array([r.get("callMidIv", np.nan) for r in e]); piv = np.array([r.get("putMidIv", np.nan) for r in e])
        m = np.isfinite(cmid) & np.isfinite(pmid) & (cmid > 0) & (pmid > 0)
        if m.sum() < 5:
            continue
        Km, d = Kk[m], (cmid - pmid)[m]                             # d = C-P = DF*(F-K) = DF*F - DF*K
        lo, hi = np.quantile(Km, [0.2, 0.8]); w = (Km >= lo) & (Km <= hi)     # central strikes only (robust)
        if w.sum() >= 3:
            Km, d = Km[w], d[w]
        b, a = np.linalg.lstsq(np.vstack([np.ones_like(Km), Km]).T, d, rcond=None)[0]   # d = b + a*K, a=-DF
        DF = max(-a, 1e-6); F = b / DF                              # forward from parity intercept/slope
        iv_otm = np.where(Kk >= F, civ, piv)                       # OTM wing: call above F, put below
        ok = np.isfinite(iv_otm) & (iv_otm > 0)
        atm_iv = float(np.interp(F, Kk[ok], iv_otm[ok]))           # ATM vol-of-VIX = IV interpolated at K=F
        out.append((dte, F / 100.0, atm_iv))                        # dte, VIX fwd (as vol), VIX ATM impvol
    return spot, out


if __name__ == "__main__":
    for date in ["2015-06-01", "2019-06-03"]:
        chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
        kw = dict(zip(NAMES, THETA_TS)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
        spot, dv = data_vix(date)
        print(f"\n=== {date}  sig_ref={sig_ref:.3f}  VIX spot={spot:.3f} ===")
        print(f"{'dte':>5}{'fwd_d':>8}{'fwd_m':>8}{'vov_data':>10}{'vov_model':>11}{'m/d':>7}")
        for dte, Fd, iv_d in dv:
            Fm, iv_m, _ = model_vix_ivol(K, sig_ref, dte / 365.0, spot=spot)
            print(f"{dte:>5}{Fd:>8.3f}{Fm:>8.3f}{iv_d:>10.3f}{iv_m:>11.3f}{iv_m/iv_d:>7.2f}", flush=True)

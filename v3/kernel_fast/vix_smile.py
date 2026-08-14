#!/usr/bin/env python3
"""Per-expiry VIX forward and ATM smile slope d(IV)/d(log K).

*** THE CONSUMER OF THIS MODULE (refit.py VOVMNY) IS OFF PERMANENTLY -- see the DO NOT USE block
    there. The moneyness correction is out of scope (it makes the fit joint SPX/VIX), it creates a
    flat direction between nu and the model's VIX forward, and it changed nothing measurable.
    This module is retained because its MEASUREMENTS are sound and reusable as diagnostics:
    the market VIX forward and the ATM smile slope per expiry. Use it to STATE the instrument
    caveat, not to correct for it. ***

Originally written for the moneyness-matched vov comparison (handoff 6e.27e).

WHY. Both sides of the vov residual take ATM implied vol AT THEIR OWN FORWARD, and the model's
forward runs ~11.7% below the market's (6e.27). The VIX smile slopes UP -- measured +0.35 to +0.99 in
d(IV)/d(log K), mean +0.6 -- BECAUSE VOL-OF-VOL RISES WITH VIX LEVEL. So the instrument reads the
model's dispersion at a displaced point of a level-dependent smile, worth -9.1% of ATM vol on average
and up to -26%. That is the same order as the entire vov fit residual, so the vol-of-vol amplitude nu
identified through it can carry a bias of that size.

This module supplies (F_market, slope) per expiry so `refit.py` can translate the MODEL's ATM vol
from the model's own forward to the market's:

    vov_adj = iv_model + slope * log(F_market / F_model)

CORRECTING THE MODEL, NOT THE TARGET -- deliberately. Reading the market smile AT the model's forward
would make the TARGET depend on theta, and since the smile slopes up the optimiser could lower
F_model to drag the target down after it. That is a self-referential degeneracy. Correcting the model
side leaves the target exactly as it was, so the comparison is unchanged wherever the forwards agree.

FIRST ORDER, and the approximation is named: the translation uses the MARKET's smile slope, i.e. it
asks "what would the model's vol be if its law sat at the market's level, given the market's measured
level-dependence". It does not use the model's own smile (which would need off-ATM VIX option prices
off a 5-node atomic law -- numerically fragile at 12% out of the money).

    python3 vix_smile.py 2016-06-01
"""
import gzip
import json
import os
import sys

import numpy as np

ORATS = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
NEAR = 0.35            # |log(K/F)| window for the local slope fit
MIN_QUOTES = 5


def _expiry(e):
    """(F, K grid, OTM IV grid) for one expiry's rows, or None. Forward by put-call parity
    regression over central strikes -- the same construction `vix_readout.data_vix` uses, so the
    forward here is the one the vov TARGET was read at."""
    K = np.array([r["strike"] for r in e], float)
    civ = np.array([r.get("callMidIv", np.nan) for r in e], float)
    piv = np.array([r.get("putMidIv", np.nan) for r in e], float)
    cm = np.array([0.5 * (r.get("callBidPrice", np.nan) + r.get("callAskPrice", np.nan)) for r in e])
    pm = np.array([0.5 * (r.get("putBidPrice", np.nan) + r.get("putAskPrice", np.nan)) for r in e])
    m = np.isfinite(cm) & np.isfinite(pm) & (cm > 0) & (pm > 0)
    if m.sum() < MIN_QUOTES:
        return None
    Km, d = K[m], (cm - pm)[m]
    lo, hi = np.quantile(Km, [0.2, 0.8])
    w = (Km >= lo) & (Km <= hi)
    if w.sum() >= 3:
        Km, d = Km[w], d[w]
    b, a = np.linalg.lstsq(np.vstack([np.ones_like(Km), Km]).T, d, rcond=None)[0]
    DF = max(-a, 1e-6)
    F = b / DF
    iv = np.where(K >= F, civ, piv)                 # OTM wing: call above F, put below
    ok = np.isfinite(iv) & (iv > 0)
    if ok.sum() < MIN_QUOTES:
        return None
    return F, K[ok], iv[ok]


def smile_at(date, dtes, near=NEAR):
    """(F_market, slope) arrays aligned to `dtes`, in VOL units (F/100, as the vov target is).

    Missing or unusable expiries get slope 0 and F_market = nan -> the caller's correction term is
    then exactly zero there, i.e. it falls back to the unadjusted comparison rather than guessing.
    """
    rows = json.load(gzip.open(f"{ORATS}/SPX-NDX-RUT-VIX_{date}.json.gz", "rt"))["strikes"]
    vix = [r for r in rows if r["ticker"] == "VIX"]
    by = {}
    for dte in sorted(set(r["dte"] for r in vix)):
        e = sorted([r for r in vix if r["dte"] == dte], key=lambda r: r["strike"])
        got = _expiry(e)
        if got is None:
            continue
        F, K, iv = got
        lk = np.log(K / F)
        sel = np.abs(lk) < near
        if sel.sum() < 4:
            sel = np.abs(lk) < 2 * near
        if sel.sum() < 4:
            continue
        by[int(dte)] = (F / 100.0, float(np.polyfit(lk[sel], iv[sel], 1)[0]))
    Fm = np.full(len(dtes), np.nan)
    sl = np.zeros(len(dtes))
    for i, d in enumerate(dtes):
        if int(d) in by:
            Fm[i], sl[i] = by[int(d)]
    return Fm, sl


if __name__ == "__main__":
    _dates = [a for a in sys.argv[1:] if a[:2] == "20"] or ["2016-06-01"]
    _H = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
    sys.argv = [sys.argv[0], "cpu"]
    import _paths as _P                                          # noqa: E402
    import end_to_end as E                                       # noqa: E402
    for date in _dates:
        ctx, _c, _p = E.ctx_rebuilt(date, "SPX")
        VD = [int(x) for x in ctx["vdtes"]]
        Fm, sl = smile_at(date, VD)
        print(f"\n  {date}   {len(VD)} vov tenors")
        print(f"    {'dte':>5s} {'F_mkt':>8s} {'dIV/dlogK':>10s}")
        for d, f, s in zip(VD, Fm, sl):
            print(f"    {d:5d} {f:8.4f} {s:10.3f}")


def model_fwd(ctx, date, dev, tag="_n9"):
    """The MODEL's VIX forward per vov tenor at a REFERENCE theta (default the shipped `_n9` fit).

    Used only by VOVMNYLEVEL, to turn the per-tenor forward gap into ONE number per date. It must be
    a FIXED reference, not the live theta: making the shift depend on theta would reintroduce exactly
    the self-referential target the model-side correction exists to avoid.
    """
    import torch
    import consts, fkernel as kernel, vix as VX
    import calibrate_slv_exact_ts as C
    import _paths as _P
    f = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}.json")))
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    th = torch.tensor([f["theta"][n] for n in C.NAMES_N] + [f["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
    return np.array([float(VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT, lam_fns=LAM, us0=u0)[0])
                     for d in ctx["vdtes"]])

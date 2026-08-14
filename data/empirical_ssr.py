#!/usr/bin/env python3
"""
empirical_ssr.py -- the REALIZED SSR from the ORATS series (the SPX time-series regression
the poc hardcodes as {1:1.9, 4:1.5, 13:1.3, 26:1.1, 52:1.0}). This is the one genuinely-new
piece: data analysis, not a model algo.

Per constant maturity T (the calibrate_2f grid 1m/3m/6m/1y): build the ATM-vol series
Sigma_ATM(t,T) and the spot series, then
    SSR(T) = Cov(dSigma_ATM(T), r) / Var(r) / mean(skew(T)),   r = daily log spot return,
matching ssr_2f's definition (beta / ATM skew). ATM vol + skew are read from the raw smile
by a local quadratic fit near the money (reuses orats_loader; no SANOS LP for a market quantity).
"""
import sys, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from orats_loader import load_day                       # noqa: E402

DT = 1.0 / 52.0
NS = [4, 13, 26, 52]; TT = np.array(NS) * DT            # 1m/3m/6m/1y


def atm_vol_skew(s, ksd=1.5):
    """Local ATM vol + skew from a loaded smile s: quadratic fit of OTM mid-IV over an SD-SCALED
    window |logm| < ksd*sigma_ATM*sqrt(T). A fixed wide window (was |logm|<0.10 = +-5 SD at 1wk)
    averaged the steep ATM skew with the flat wings and under-read it, inflating the short-end SSR;
    the SD-scaled window recovers the SANOS-marginal skew."""
    lm = np.asarray(s["logm"], float)
    iv = np.array([v if v else np.nan for v in s["iv"]], float)
    ok = np.isfinite(iv)
    m0 = ok & (np.abs(lm) < 0.04)
    atm0 = np.median(iv[m0]) if m0.sum() >= 3 else (np.median(iv[ok]) if ok.any() else np.nan)
    if not np.isfinite(atm0):
        return None
    win = float(np.clip(ksd * atm0 * np.sqrt(max(s.get("T", 0.08), 1e-6)), 0.025, 0.10))
    m = ok & (np.abs(lm) < win)
    if m.sum() < 4:
        return None
    c = np.polyfit(lm[m], iv[m], 2)                     # c = [quad, lin, const]
    return float(np.polyval(c, 0.0)), float(c[1])       # (ATM vol = iv(0), skew = d iv / d logm at 0)


def date_row(path, tt=TT, ticker="SPX"):
    """(spot, ATMvol[tt], skew[tt]) for one date, interpolated to the maturities tt.
    spot = MEDIAN stockPrice across the day's smiles. ORATS stockPrice is unreliable (varies across
    strikes; badly corrupt in 2022-2023 with prints up to +20%), which injected spurious +-20% one-day
    returns that inflated realised vol (2023: 55% vs true ~13%) and collapsed the realised SSR (2023:
    ~0.1 vs true ~1.25). orats_loader.load_day already medians across all rows, so the smiles share one
    clean day spot -- taking the median here guards date_row independently. ATM vol/skew use the clean
    parity-forward F and are unaffected."""
    day = load_day(path, [ticker]).get(ticker, {})
    T, vol, sk, allsp = [], [], [], []
    for exp, s in day.items():
        if s.get("spot"):
            allsp.append(s["spot"])
        if not s["F"] or not s["T"]:
            continue
        vs = atm_vol_skew(s)
        if vs:
            T.append(s["T"]); vol.append(vs[0]); sk.append(vs[1])
    if len(T) < 3 or not allsp:
        return None
    o = np.argsort(T); T = np.array(T)[o]; vol = np.array(vol)[o]; sk = np.array(sk)[o]
    return float(np.median(allsp)), np.interp(tt, T, vol), np.interp(tt, T, sk)


def empirical_ssr(paths, ns=NS, dt=DT, ticker="SPX"):
    tt = np.array(ns) * dt
    rows = [r for r in (date_row(p, tt, ticker) for p in sorted(paths)) if r]
    spots = np.array([r[0] for r in rows]); vols = np.vstack([r[1] for r in rows]); sks = np.vstack([r[2] for r in rows])
    ret = np.diff(np.log(spots))                        # daily log spot return
    dvol = np.diff(vols, axis=0)                         # daily ATM-vol change per maturity
    # DATA FILTER, two rules, per maturity (only bite where corruption exists; clean data -> unchanged):
    #  (1) POSITIVE-SKEW day: an index smile is always NEGATIVE-skew, so a positive fitted ATM skew = corrupt
    #      short-dated quadratic (thin options; e.g. 2016-01-15 skew +1.13). Drop the day.
    #  (2) INVERTED SHORT END: a thin-weekly interpolation artifact prints an absurd short-dated ATM vol (e.g.
    #      NDX 2017-10-27 1wk vol 31% = 2.5x the 1m on a calm rally day; skew NEGATIVE so rule (1) misses it).
    #      These anti-leverage vol-spikes dragged NDX 2012/16/17 1wk SSR to 0.75-1.02 (true ~1.5-1.75). Flag a
    #      SHORT maturity whose ATM vol exceeds 1.5x the 1m ATM vol. (Real crashes lift the whole curve, so the
    #      1wk/1m ratio stays <1.5; the artifact is an ISOLATED short-end spike.) Liquid SPX -> essentially no hits.
    ref = min(2, len(ns) - 1)                           # index of the 1m maturity (the short-end reference)
    ssr = np.empty(len(ns))
    for j in range(len(ns)):
        bad = sks[:, j] > 0.0                           # rule (1)
        if j < ref:                                     # rule (2): only the short maturities (1wk, 2wk)
            bad = bad | (vols[:, j] > 1.5 * vols[:, ref])
        keep = (~bad[:-1]) & (~bad[1:])                 # exclude any daily change touching a bad day
        ssr[j] = np.cov(dvol[keep, j], ret[keep])[0, 1] / np.var(ret[keep]) / np.mean(sks[~bad, j])
    return ssr, len(rows)


if __name__ == "__main__":
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    yr = sys.argv[1] if len(sys.argv) > 1 else "2015"
    paths = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz"))
    ssr, n = empirical_ssr(paths)
    print(f"{yr}: {n} dates | empirical realized SSR (1m/3m/6m/1y): {np.round(ssr, 2)}")
    print(f"  (poc hardcoded SPX target: ~1.5 / 1.3 / 1.1 / 1.0)")

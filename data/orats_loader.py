#!/usr/bin/env python3
"""
orats_loader.py -- turn a raw ORATS hist/strikes day file into clean, calibration-
ready smiles, with the model-free forward/discount recovered by put-call parity.

    from orats_loader import load_day
    day = load_day(".../SPX-NDX-RUT-VIX_2015-06-15.json.gz", tickers=["SPX"])
    smile = day["SPX"]["2015-07-17"]     # {T,F,DF,spot,strike,logm,smvVol,callMidIv,...}

Per (ticker, expiry): strike grid, the smile `iv` (OTM-side mid IV -- put below F,
call above F), call/put mid IVs, mid prices, deltas, log-moneyness log(K/F), the
parity forward F & discount DF, and ORATS `smvVol` (smoothed) WHERE AVAILABLE.
NOTE: smvVol is populated only in recent years (0 in the older history), so `iv`
(raw mid IVs) is the primary vol surface. Only strikes with a two-sided call AND
put quote feed the forward regression.

CLI:  python3 orats_loader.py <day.json.gz>
"""
import json, gzip
from math import log
import numpy as np


def _mid(bid, ask):
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def parity_forward(strike, cmid, pmid):
    """Recover (F, DF) per expiry from  C - P = DF*(F - K)  by regressing (C-P) on K:
    intercept = DF*F, slope = -DF.  Returns (None, None) if <3 usable strikes."""
    K, y = [], []
    for k, c, p in zip(strike, cmid, pmid):
        if c is not None and p is not None:
            K.append(k); y.append(c - p)
    if len(K) < 3:
        return None, None
    K = np.asarray(K, float); y = np.asarray(y, float)
    A = np.column_stack([np.ones_like(K), K])          # y = (DF*F)*1 + (-DF)*K
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    DF = -b1
    if not np.isfinite(DF) or DF <= 0:
        return None, None
    return b0 / DF, DF


def load_day(path, tickers=None):
    """Load one day's file -> {ticker: {expiry: smile_dict}}."""
    rows = json.load(gzip.open(path, "rt"))["strikes"]
    groups = {}
    sp = {}                                                # per-ticker stockPrice samples (all rows)
    for r in rows:
        t = r["ticker"]
        if tickers and t not in tickers:
            continue
        groups.setdefault((t, r["expirDate"][:10]), []).append(r)
        v = r.get("stockPrice")
        if v is not None:
            sp.setdefault(t, []).append(v)
    # Robust day spot = MEDIAN stockPrice across ALL of the ticker's rows. ORATS records stockPrice per
    # row and it is unreliable -- it varies across strikes, and is badly corrupt in 2022-2023 with bad
    # prints up to +20%. The old code used rs[0].stockPrice (one arbitrary row), which injected spurious
    # +-20% one-day return jumps that inflated realised vol and collapsed the realised-SSR target. The
    # median is robust; F/DF and the mid-IV smile use put-call parity and are unaffected either way.
    day_spot = {t: float(np.median(v)) for t, v in sp.items() if v}
    out = {}
    for (t, exp), rs in groups.items():
        rs.sort(key=lambda r: r["strike"])
        K = [r["strike"] for r in rs]
        cmid = [_mid(r.get("callBidPrice"), r.get("callAskPrice")) for r in rs]
        pmid = [_mid(r.get("putBidPrice"), r.get("putAskPrice")) for r in rs]
        F, DF = parity_forward(K, cmid, pmid)
        dte = rs[0].get("dte")
        cIv = [r.get("callMidIv") for r in rs]
        pIv = [r.get("putMidIv") for r in rs]
        # the smile = OTM-side mid IV (put below F, call above F); primary vol surface
        iv = ([(p if k < F else c) for k, c, p in zip(K, cIv, pIv)] if F else None)
        out.setdefault(t, {})[exp] = {
            "T": (dte / 365.0 if dte else None),
            "dte": dte,
            "F": F, "DF": DF,
            "spot": day_spot.get(t),                       # robust day spot (median across all rows)
            "strike": K,
            "logm": ([log(k / F) for k in K] if F else None),
            "iv": iv,
            "callMidIv": cIv,
            "putMidIv": pIv,
            "smvVol": [r.get("smvVol") for r in rs],
            "delta": [r.get("delta") for r in rs],
            "cmid": cmid, "pmid": pmid,
        }
    return out


if __name__ == "__main__":
    import sys
    d = load_day(sys.argv[1])
    for t in sorted(d):
        exps = sorted(e for e in d[t] if d[t][e]["T"])
        e0 = exps[len(exps) // 3]
        s = d[t][e0]
        fr = f"{s['F']:.2f}" if s["F"] else "n/a"
        print(f"{t}: {len(exps)} expiries | e.g. {e0} T={s['T']:.3f} F={fr} "
              f"spot={s['spot']} strikes={len(s['strike'])}")

#!/usr/bin/env python3
"""
Off-SPX vol-of-vol readout demo (paper Sec. 'Extracting the readout, and what stands in for VIX').
For NDX (no VIX in our pipeline) build the forward-variance term structure from NDX's OWN option strip
(model-free VIX-formula / log-contract), take the REALISED variance-of-variance (P) as the vov target, and
compare to SPX -- which additionally has the sharp Q VIX-implied vov. Built-in validation: the SPX var-swap
sqrt(V) reconstructed from the SPX strip should track the actual VIX index level. Records wall-time.
    python3 ndx_vov_readout.py [YEAR]
"""
import gzip, json, glob, os, sys, time
import numpy as np

OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"


def expiry(recs, ticker, target):
    tk = [r for r in recs if r["ticker"] == ticker]
    dtes = sorted(set(r["dte"] for r in tk if r["dte"] >= 7))
    if not dtes:
        return None
    dt = min(dtes, key=lambda x: abs(x - target))
    return [r for r in tk if r["dte"] == dt]


def cmid(r): return 0.5 * (r["callBidPrice"] + r["callAskPrice"])
def pmid(r): return 0.5 * (r["putBidPrice"] + r["putAskPrice"])


def var_swap(ch, T):
    """Model-free annualised variance (VIX formula) from the OTM strip; returns sqrt(V) (VIX-equivalent)."""
    ch = sorted(ch, key=lambda r: r["strike"])
    spot = ch[0]["spotPrice"]
    k0 = min(ch, key=lambda r: abs(r["strike"] - spot))
    F = k0["strike"] + (cmid(k0) - pmid(k0))                        # parity forward (discounting ~1 at 30-90d)
    Ks = np.array([r["strike"] for r in ch], float)
    Q, ok = [], []
    for r in ch:                                                    # OTM leg, liquidity-filtered (bid>0)
        if r["strike"] < F:
            q, good = pmid(r), r["putBidPrice"] > 0
        else:
            q, good = cmid(r), r["callBidPrice"] > 0
        Q.append(q); ok.append(good and q > 0)
    Q = np.array(Q); ok = np.array(ok); dK = np.gradient(Ks)
    K0 = Ks[Ks <= F].max() if (Ks <= F).any() else F
    sig2 = (2.0 / T) * np.sum((dK / Ks ** 2 * Q)[ok]) - (1.0 / T) * (F / K0 - 1.0) ** 2
    return np.sqrt(max(sig2, 1e-8))


def rvov(x):                                                        # realised vol-of-vol: ann. vol of the vol level
    return float(np.std(np.diff(np.log(x))) * np.sqrt(252))


def rvov_spread(x, w=63):                                           # quarterly sub-window spread = P estimate noise
    qs = [rvov(x[i:i + w]) for i in range(0, len(x) - w, w)]
    return float(np.std(qs)) if len(qs) > 1 else float("nan")


if __name__ == "__main__":
    yr = sys.argv[1] if len(sys.argv) > 1 else "2019"
    files = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz"))
    t0 = time.time(); rows = []
    for f in files:
        try:
            recs = json.load(gzip.open(f))["strikes"]
            chS, chN = expiry(recs, "SPX", 30), expiry(recs, "NDX", 30)
            chS9, chN9 = expiry(recs, "SPX", 90), expiry(recs, "NDX", 90)
            vix = [r for r in recs if r["ticker"] == "VIX"]
            if not (chS and chN and chS9 and chN9 and vix):
                continue
            rows.append((var_swap(chS, 30 / 365), var_swap(chN, 30 / 365), vix[0]["spotPrice"] / 100,
                         var_swap(chS9, 90 / 365), var_swap(chN9, 90 / 365)))
        except Exception:
            continue
    A = np.array(rows); wall = time.time() - t0
    print(f"{yr}: {len(A)} days processed  (wall {wall:.0f}s)\n")
    print(f"VALIDATION  SPX own-strip sqrt(V) 30d mean = {A[:,0].mean()*100:5.2f}   vs  VIX index mean = "
          f"{A[:,2].mean()*100:5.2f}   (should match)")
    print(f"\nREALISED vol-of-vol  (P: annualised vol of the 30d variance-swap level)")
    print(f"  {'':4}{'rvov':>8}{'quarterly spread':>18}")
    print(f"  {'SPX':4}{rvov(A[:,0]):>8.3f}{'+-'+format(rvov_spread(A[:,0]),'.3f'):>18}   "
          f"(VIX-index cross-check {rvov(A[:,2]):.3f})")
    print(f"  {'NDX':4}{rvov(A[:,1]):>8.3f}{'+-'+format(rvov_spread(A[:,1]),'.3f'):>18}")
    print(f"\nvov TERM STRUCTURE  (30d vs 90d realised vov -> slow timescale)")
    print(f"  SPX  30d {rvov(A[:,0]):.3f}   90d {rvov(A[:,3]):.3f}   ratio {rvov(A[:,3])/rvov(A[:,0]):.2f}")
    print(f"  NDX  30d {rvov(A[:,1]):.3f}   90d {rvov(A[:,4]):.3f}   ratio {rvov(A[:,4])/rvov(A[:,1]):.2f}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ndx_vov_{yr}.npy"), A)

#!/usr/bin/env python3
"""
orats_sanos.py -- bridge ORATS day files to the SANOS static-marginal LP (poc/sanos_lp.py).

Turns an ORATS chain into the (type, strike, bid, ask, impliedVolatility, dte) frame that
the existing prep_expiry/sanos_fit expect, then fits the SANOS marginal per expiry and
reports the round-trip IV error -- i.e. runs paper Algorithm 1 on real data, reusing the
poc/ code unchanged.

    python3 orats_sanos.py <day.json.gz> [TICKER]
"""
import sys, os, json, gzip
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "poc"))
from sanos_lp import prep_expiry, sanos_fit          # noqa: E402
from discslv import bs_call, bs_implied_vol          # noqa: E402


def orats_chain_df(path, ticker="SPX"):
    """ORATS day file -> DataFrame[type,strike,bid,ask,impliedVolatility,dte,volume,ivBid,ivAsk]."""
    rows = json.load(gzip.open(path, "rt"))["strikes"]
    rec = []
    for r in rows:
        if r["ticker"] != ticker:
            continue
        K, dte = r["strike"], r["dte"]
        rec.append(("call", K, r.get("callBidPrice"), r.get("callAskPrice"), r.get("callMidIv"),
                    dte, r.get("callVolume"), r.get("callBidIv"), r.get("callAskIv"), r.get("callOpenInterest")))
        rec.append(("put",  K, r.get("putBidPrice"),  r.get("putAskPrice"),  r.get("putMidIv"),
                    dte, r.get("putVolume"),  r.get("putBidIv"),  r.get("putAskIv"), r.get("putOpenInterest")))
    df = pd.DataFrame(rec, columns=["type", "strike", "bid", "ask", "impliedVolatility",
                                    "dte", "volume", "ivBid", "ivAsk", "oi"])
    return df.dropna(subset=["bid", "ask", "impliedVolatility"])


def eta_for_dte(dte):
    """SANOS smoothness by maturity, per Buehler-Horvath Figs 3-4: eta=0.06 fits
    well-sampled maturities within bid/ask; eta=0.25 for sparse long-dated (>2y)."""
    return 0.25 if dte > 730 else 0.06


def fit_day(path, ticker="SPX", eta_fn=eta_for_dte):
    """Run the SANOS LP per expiry (eta set per-expiry); return fit-quality dicts."""
    import sanos_lp
    df = orats_chain_df(path, ticker)
    out = []
    for dte, g in df.groupby("dte"):
        sanos_lp.ETA = eta_fn(int(dte))
        e = prep_expiry(g)
        if e.get("n", 0) < 20:                                       # SANOS: >=20 active options/expiry
            continue
        q, M = sanos_fit(e)
        if q is None:
            out.append(dict(dte=int(dte), F=e["F"], atm_iv=e["atm_iv"], n=e["n"], fail=True))
            continue
        cfit = np.maximum(M.T @ q, 1e-12)
        ivfit = np.array([bs_implied_vol(cfit[l], 1.0, e["kappa"][l], e["tau"]) for l in range(len(cfit))])
        d = (ivfit - e["iv"]) * 1e4
        core = np.abs(np.log(e["kappa"])) < 0.2
        inband = (float(np.mean((cfit >= e["c_lo"] - 1e-9) & (cfit <= e["c_hi"] + 1e-9)))
                  if "c_lo" in e else float("nan"))                  # fraction of strikes fit within bid/ask
        out.append(dict(dte=int(dte), F=e["F"], atm_iv=e["atm_iv"], n=e["n"], inband=inband,
                        core_bp=(float(np.sqrt(np.mean(d[core] ** 2))) if core.any() else float("nan")),
                        rmse_bp=float(np.sqrt(np.mean(d ** 2))), max_bp=float(np.max(np.abs(d)))))
    return out


if __name__ == "__main__":
    path = sys.argv[1]; tk = sys.argv[2] if len(sys.argv) > 2 else "SPX"
    res = fit_day(path, tk)
    print(f"{tk}: {len(res)} expiries fit  ({os.path.basename(path)})")
    print(f"{'dte':>5} {'F':>9} {'ATMiv':>7} {'N':>4} {'inband':>7} {'core_bp':>8} {'rmse_bp':>8} {'max_bp':>8}")
    for r in res:
        if r.get("fail"):
            print(f"{r['dte']:>5} {r['F']:>9.1f} {r['atm_iv']:>7.3f} {r['n']:>4}   FAIL"); continue
        print(f"{r['dte']:>5} {r['F']:>9.1f} {r['atm_iv']:>7.3f} {r['n']:>4} {r['inband']*100:>6.0f}% "
              f"{r['core_bp']:>8.1f} {r['rmse_bp']:>8.1f} {r['max_bp']:>8.1f}")
    good = [r for r in res if not r.get("fail")]
    if good:
        print(f"median WITHIN-BAND {np.median([r['inband'] for r in good])*100:.0f}%  |  "
              f"median core-RMSE {np.median([r['core_bp'] for r in good]):.1f} bp  over {len(good)} expiries")

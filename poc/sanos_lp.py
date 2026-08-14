"""
sanos_lp.py -- Algorithm 1: the SANOS LP on a real option chain (IV-space, discount-free).

Per expiry, fit a log-normal-mixture marginal to the smile:
    c(k) = sum_i q^i * Call(kappa^i, k, V_j),    sum_i q^i = 1,   sum_i q^i kappa^i = 1,   q^i >= 0.
Everything is on the UNIT FORWARD martingale Z=S/F (discount drops out). Anchors kappa^i = the
quoted OTM moneyness; the backbone V_j is a SMALL smoothing width (so the q^i are essentially the
risk-neutral density, the SANOS reading) rather than the ATM variance. Targets are pure-call prices
from the market implied vols. We report the round-trip IV error.

Run from poc/ :  python3 sanos_lp.py
"""
import os
import numpy as np
import pandas as pd
import cvxpy as cp
from discslv import bs_call, bs_implied_vol

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "options_SPX.csv")
ETA = 0.25     # SANOS smoothness: component variance = ETA * ATM variance (paper default 0.25)
LIQ = 100      # liquidity floor: >=100 lots traded or held (Buehler-Horvath SPX default)
LIQ_FALLBACK = None   # if set (e.g. 1) and the >=LIQ set has <LIQ_MIN strikes, relax to this floor (thin index weeklies)
LIQ_MIN = 20   # min strikes before the fallback engages


def forward(df_e):
    """Forward F by put-call parity regression  C-P = DF*(F-K)  over central strikes (robust)."""
    calls = df_e[df_e.type == "call"].groupby("strike").first()
    puts = df_e[df_e.type == "put"].groupby("strike").first()
    common = calls.index.intersection(puts.index)
    K = common.values.astype(float)
    d = (0.5 * (calls.loc[common, "bid"] + calls.loc[common, "ask"])
         - 0.5 * (puts.loc[common, "bid"] + puts.loc[common, "ask"])).values
    lo, hi = np.quantile(K, [0.25, 0.75]); w = (K >= lo) & (K <= hi)
    if w.sum() >= 3:
        K, d = K[w], d[w]
    b, a = np.linalg.lstsq(np.vstack([np.ones_like(K), K]).T, d, rcond=None)[0]   # d = b + a K
    DF = -a
    if not (1e-3 < DF <= 1.0 + 1e-6):
        DF = 1.0
    return float(b / DF) if DF > 0 else float(K[np.argmin(np.abs(d))])


def prep_expiry(df_e):
    """OTM anchors with the SANOS liquidity filters (Buehler-Horvath 2026, sec. after
    Fig 1): >=100 lots traded on the day (if a `volume` column is present) and
    Vega/sqrt(T) >= 0.1%. When per-strike bid/ask IVs (`ivBid`,`ivAsk`) are supplied,
    also returns the pure-call band [c_lo, c_hi] so sanos_fit can fit WITHIN bid/ask."""
    F = forward(df_e); tau = float(df_e.dte.iloc[0]) / 365.0
    has_vol = "volume" in df_e.columns; has_oi = "oi" in df_e.columns
    has_band = "ivBid" in df_e.columns and "ivAsk" in df_e.columns
    cand = []
    for _, r in df_e.iterrows():
        otm = (r.type == "call" and r.strike >= F) or (r.type == "put" and r.strike < F)
        if not (otm and r.bid > 0 and r.ask > r.bid and np.isfinite(r.impliedVolatility) and 0.02 < r.impliedVolatility < 1.5):
            continue
        ivb = iva = np.nan
        if has_band and np.isfinite(r.ivBid) and np.isfinite(r.ivAsk) and 0 < r.ivBid < r.ivAsk:
            ivb, iva = float(r.ivBid), float(r.ivAsk)
        v = float(r.volume) if (has_vol and np.isfinite(r.volume)) else 0.0
        o = float(r.oi) if (has_oi and np.isfinite(r.oi)) else 0.0
        cand.append((float(r.strike), float(r.impliedVolatility), ivb, iva, v, o))
    cand = sorted(set(cand))
    if has_vol or has_oi:                                                       # liquidity: traded OR held
        strong = [c for c in cand if c[4] >= LIQ or c[5] >= LIQ]                # SANOS default: >=100 lots traded or held
        if LIQ_FALLBACK is not None and len(strong) < LIQ_MIN:                  # thin index weeklies: relax to any live interest
            cand = [c for c in cand if c[4] >= LIQ_FALLBACK or c[5] >= LIQ_FALLBACK]
        else:
            cand = strong                                                       # default path: identical to the strict >=100 filter
    rows = [c[:4] for c in cand]
    if not rows:
        return dict(kappa=np.array([]), n=0, F=F, tau=tau, V=0.0)
    K = np.array([x[0] for x in rows]); iv = np.array([x[1] for x in rows])
    ivb = np.array([x[2] for x in rows]); iva = np.array([x[3] for x in rows])
    kappa = K / F; lk = np.log(kappa); w0 = iv * np.sqrt(tau)
    vega = (np.asarray(bs_call(1.0, kappa, w0 + 1e-4)) - np.asarray(bs_call(1.0, kappa, w0))) / 1e-4   # Vega/sqrt(T)=phi(d)
    keep = (np.abs(lk) < 0.5) & (vega >= 1e-3)                                  # SANOS: Vega/sqrt(T) >= 0.1%
    kappa, iv, ivb, iva, lk = kappa[keep], iv[keep], ivb[keep], iva[keep], lk[keep]
    if not len(kappa):
        return dict(kappa=np.array([]), n=0, F=F, tau=tau, V=0.0)
    atm = iv[np.abs(lk) < 0.05]
    atm_iv = float(np.median(atm)) if len(atm) else float(iv[np.argmin(np.abs(lk))])
    V = ETA * (atm_iv ** 2) * tau
    w = iv * np.sqrt(tau)
    out = dict(kappa=kappa, iv=iv, w=w, c=np.asarray(bs_call(1.0, kappa, w)), V=V,
               F=F, tau=tau, atm_iv=atm_iv, n=len(kappa))
    ivb_f = np.where(np.isfinite(ivb), ivb, iv)                                 # missing bid/ask IV -> zero-width (fit mid)
    iva_f = np.where(np.isfinite(iva), iva, iv)
    out["c_lo"] = np.asarray(bs_call(1.0, kappa, ivb_f * np.sqrt(tau)))
    out["c_hi"] = np.asarray(bs_call(1.0, kappa, iva_f * np.sqrt(tau)))
    return out


def sanos_fit(e):
    """SANOS LP: fit the model call prices WITHIN the bid/ask band [c_lo, c_hi] via a
    soft slack penalty (vega-weighted), on the eta-smoothed design, s.t. simplex +
    martingale. Falls back to fit-to-mid when no band is supplied (e.g. Yahoo data)."""
    kap, V = e["kappa"], e["V"]; s = np.sqrt(V)
    M = np.asarray(bs_call(kap[:, None], kap[None, :], s))             # M[i,l] = Call(kap_i, kap_l, s)
    vega = np.maximum(np.asarray(bs_call(1.0, kap, e["w"] + 1e-4) - bs_call(1.0, kap, e["w"])) / 1e-4, 1e-7)
    q = cp.Variable(len(kap), nonneg=True)
    fit = M.T @ q
    if "c_lo" in e and "c_hi" in e:
        obj = cp.sum(cp.multiply(vega, cp.pos(e["c_lo"] - fit) + cp.pos(fit - e["c_hi"])))   # within bid/ask (SANOS)
    else:
        obj = cp.sum(cp.multiply(vega, cp.abs(fit - e["c"])))                                 # fallback: fit-to-mid
    cp.Problem(cp.Minimize(obj), [cp.sum(q) == 1, q @ kap == 1]).solve(solver=cp.CLARABEL)
    return (None if q.value is None else np.asarray(q.value)), M


def main():
    df = pd.read_csv(DATA)
    print(f"{'dte':>4} {'F':>8} {'ATMiv':>7} {'N':>4} {'sqrtV':>7} {'core':>9} {'full':>9} {'max(bp)':>9}")
    for dte, g in df.groupby("dte"):
        e = prep_expiry(g)
        if len(e["kappa"]) < 5:
            continue
        q, M = sanos_fit(e)
        if q is None:
            print(f"{dte:>4} {e['F']:>8.1f} {e['atm_iv']:>7.3f} {len(e['kappa']):>4}   solver FAIL")
            continue
        cfit = np.maximum(M.T @ q, 1e-12)
        ivfit = np.array([bs_implied_vol(cfit[l], 1.0, e["kappa"][l], e["tau"]) for l in range(len(cfit))])
        d = (ivfit - e["iv"]) * 1e4
        core = np.abs(np.log(e["kappa"])) < 0.2
        print(f"{dte:>4} {e['F']:>8.1f} {e['atm_iv']:>7.3f} {len(e['kappa']):>4} {np.sqrt(e['V']):>7.4f} "
              f"{np.sqrt(np.mean(d[core]**2)):>9.1f} {np.sqrt(np.mean(d**2)):>9.1f} {np.max(np.abs(d)):>9.1f}")


if __name__ == "__main__":
    main()

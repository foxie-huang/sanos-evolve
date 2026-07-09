#!/usr/bin/env python3
"""
Hedging backtest (paper Sec. 'The SSR-consistent hedge ratio', eq:ssr-delta) -- the empirical centrepiece.
Delta-hedge a rolling 1-month ATM SPX option once per day; the one-parameter delta family
    Delta(R) = Delta_BS + Vega * R * skew / S
spans the standard deltas: R=0 is Black / sticky-strike, R=-1 is sticky-delta (smile roll), and R=SSR is the
SSR-consistent (minimum-variance) delta. The 1-day hedging P&L (option value change minus the delta hedge, BS
Taylor form) has variance minimised at R = the realised SSR; the SSR-consistent delta reduces the hedging
variance vs Black. Reports the variance vs R, the minimising R, the reduction, and turnover. Records wall-time.
    python3 hedging_backtest.py [YEAR ...]
"""
import sys, os, glob, time
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from empirical_ssr import date_row                                   # noqa: E402

OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
DT = 1.0 / 52.0
R_MODEL = 1.6                                                        # calibrated model 1m SSR (structural, OOS)


def series(year, wk):
    tt = np.array([wk * DT])
    rows = [date_row(p, tt, "SPX") for p in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))]
    rows = [r for r in rows if r]
    S = np.array([r[0] for r in rows]); sig = np.array([r[1][0] for r in rows]); sk = np.array([r[2][0] for r in rows])
    ok = np.isfinite(S) & np.isfinite(sig) & np.isfinite(sk) & (sig > 0.01)
    return S[ok], sig[ok], sk[ok]


def backtest(year, wk=4):
    S, sig, sk = series(year, wk)
    dS = np.diff(S); dsig = np.diff(sig); Sm, sigm, skm = S[:-1], sig[:-1], sk[:-1]
    ret = dS / Sm; T = wk * DT; dt = 1.0 / 252.0
    d1 = 0.5 * sigm * np.sqrt(T); phi = norm.pdf(d1)
    vega = Sm * phi * np.sqrt(T); gamma = phi / (Sm * sigm * np.sqrt(T)); theta = -Sm * phi * sigm / (2 * np.sqrt(T))
    delta_bs = norm.cdf(d1)
    beta = np.cov(dsig, ret)[0, 1] / np.var(ret); ssr_real = beta / np.mean(skm)   # realised 1m SSR this window

    def resid(R):                                                    # 1-day delta-hedged P&L, BS Taylor form
        return 0.5 * gamma * dS ** 2 + vega * dsig - R * vega * skm * ret - theta * dt

    def turnover(R):                                                 # mean |d Delta| * S (daily rebalancing)
        d = delta_bs + vega * R * skm / Sm
        return float(np.mean(np.abs(np.diff(d)) * Sm[:-1]))

    Rg = np.linspace(-1.0, 3.0, 81); std = np.array([np.std(resid(R)) for R in Rg])
    v0 = np.std(resid(0.0)); Ropt = Rg[np.argmin(std)]
    return dict(n=len(dS), ssr_real=float(ssr_real), Ropt=float(Ropt),
                red_ssr=float(1 - np.std(resid(R_MODEL)) / v0),      # reduction at the model SSR (OOS structural)
                red_opt=float(1 - std.min() / v0),                   # reduction at the in-sample optimum
                red_sd=float(1 - np.std(resid(-1.0)) / v0),          # sticky-delta vs Black
                turn_bs=turnover(0.0), turn_ssr=turnover(R_MODEL))


def sweep(year, wk=4):
    """(R grid, hedging std normalised to Black, realised SSR) for the variance-vs-R figure."""
    S, sig, sk = series(year, wk)
    dS = np.diff(S); dsig = np.diff(sig); Sm, sigm, skm = S[:-1], sig[:-1], sk[:-1]
    ret = dS / Sm; T = wk * DT; dt = 1.0 / 252.0
    d1 = 0.5 * sigm * np.sqrt(T); phi = norm.pdf(d1)
    vega = Sm * phi * np.sqrt(T); gamma = phi / (Sm * sigm * np.sqrt(T)); theta = -Sm * phi * sigm / (2 * np.sqrt(T))
    beta = np.cov(dsig, ret)[0, 1] / np.var(ret); ssr_real = beta / np.mean(skm)

    def resid(R):
        return 0.5 * gamma * dS ** 2 + vega * dsig - R * vega * skm * ret - theta * dt

    Rg = np.linspace(-1.5, 3.5, 101); v0 = np.std(resid(0.0))
    return Rg, np.array([np.std(resid(R)) / v0 for R in Rg]), float(ssr_real)


def reductions(years):
    return {y: backtest(y) for y in years}


if __name__ == "__main__":
    years = sys.argv[1:] or ["2016", "2017", "2019", "2021"]
    t0 = time.time()
    print(f"SSR-consistent hedging backtest  (1m ATM SPX, daily rehedge; delta model SSR R={R_MODEL})\n")
    print(f"{'year':>6}{'days':>6}{'realised SSR':>13}{'R*':>6} | {'var-reduction vs Black':>24} | turnover")
    print(f"{'':>31} | {'SSR   optimal   stickyd':>24} |  BS / SSR")
    for y in years:
        r = backtest(y)
        print(f"{y:>6}{r['n']:>6}{r['ssr_real']:>13.2f}{r['Ropt']:>6.2f} | "
              f"{r['red_ssr']*100:>5.0f}%  {r['red_opt']*100:>6.0f}%  {r['red_sd']*100:>7.0f}% | "
              f"{r['turn_bs']:>6.1f}/{r['turn_ssr']:.1f}")
    print(f"\n(positive var-reduction = SSR-consistent delta hedges better than Black; sticky-delta negative = worse)")
    print(f"wall {time.time()-t0:.0f}s")

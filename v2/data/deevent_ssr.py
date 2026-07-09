#!/usr/bin/env python3
"""
Daily event-conditional realized SSR -- the EOD screen for the de-eventing DYNAMICS channel.

The clean-flank bridge on the CALENDAR axis (no intraday): regress the daily constant-maturity ATM-vol
change on the daily return, split into EVENT-day transitions (FOMC / CPI / NFP announcement days) vs CLEAN
days. In  dSigma = a + b*r :  the event vol-crush is the intercept a (return-independent, drops out of the
slope), the leverage is the slope b. Time-homogeneity gives the baseline b_clean from the event-free days;
the make-or-break is the EXCESS  dSSR = SSR_event - SSR_clean  (SSR = b/skew). A clean positive dSSR = the
event carries extra spot-vol leverage on top of the diffusion (the dynamics channel is populated on index
macro); dSSR ~ 0 = "variance, not skew / go to earnings". Per-group regressions -> heteroskedasticity-safe
SEs. Records wall-time.
    python3 deevent_ssr.py [TICKER] [YEAR ...]        (default SPX 2015..2024)
"""
import sys, os, glob, time, json
import numpy as np
from datetime import date as D, timedelta

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from empirical_ssr import date_row, DT                                     # noqa: E402
from deevent_termstruct import FOMC, CPI, OUT                              # noqa: E402

WKS = [1, 2, 4]; TT = np.array(WKS) * DT; LAB = ["1w", "2w", "1m"]         # constant maturities


def nfp_dates(years):
    """NFP release = first Friday of each month (~8:30am; move is in that day's close-to-close)."""
    out = []
    for y in years:
        for m in range(1, 13):
            d = D(y, m, 1); d += timedelta((4 - d.weekday()) % 7)          # first Friday
            out.append(d.isoformat())
    return out


def build_series(years, ticker):
    """Daily (date, spot, ATMvol[WKS], skew[WKS]); cached to deevent_ssr_<ticker>.json."""
    cache = os.path.join(HERE, f".deevent_ssr_{ticker}.json")
    have = json.load(open(cache)) if os.path.exists(cache) else {}
    rows = {}
    for y in years:
        for p in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{y}-*.json.gz")):
            d = os.path.basename(p).split("_")[1][:10]
            if d in have:
                rows[d] = have[d]; continue
            r = date_row(p, TT, ticker)
            if r:
                rows[d] = [float(r[0]), [float(x) for x in r[1]], [float(x) for x in r[2]]]
    json.dump({**have, **rows}, open(cache, "w"))
    return dict(sorted(rows.items()))


def slope(x, y):
    """OLS slope b of y~a+b*x with its heteroskedasticity-naive SE."""
    x = np.asarray(x); y = np.asarray(y); n = len(x)
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean(); e = y - a - b * x
    se = np.sqrt((np.sum(e ** 2) / (n - 2)) / np.sum((x - x.mean()) ** 2))
    return b, se, a


def analyze(rows, events_by_type):
    dates = list(rows)
    spot = np.array([rows[d][0] for d in dates])
    vol = np.array([rows[d][1] for d in dates]); sk = np.array([rows[d][2] for d in dates])
    ret = np.diff(np.log(spot)); dvol = np.diff(vol, axis=0)
    td = dates[1:]; ske = sk[1:]                                            # transition ends on td
    ok = np.all(np.isfinite(dvol), 1) & np.isfinite(ret) & (np.abs(ret) < 0.15)
    ev_any = {t for ev in events_by_type.values() for t in ev}
    clean = np.array([ok[i] and td[i] not in ev_any for i in range(len(td))])

    out = {}
    for name, evset in events_by_type.items():
        m = np.array([ok[i] and td[i] in evset for i in range(len(td))])
        out[name] = {}
        if m.sum() < 5:                                                    # too few events to regress
            for lab in LAB:
                out[name][lab] = dict(n=int(m.sum()), ssr_c=float("nan"), ssr_e=float("nan"),
                                      dssr=float("nan"), t=float("nan"), crush=float("nan"))
            continue
        for j, lab in enumerate(LAB):
            be, see, ae = slope(ret[m], dvol[m, j])
            bc, sec, ac = slope(ret[clean], dvol[clean, j])
            ssr_e = be / np.mean(ske[m, j]); ssr_c = bc / np.mean(ske[clean, j])
            dse = np.sqrt(see ** 2 + sec ** 2)
            out[name][lab] = dict(n=int(m.sum()), ssr_c=ssr_c, ssr_e=ssr_e,
                                  dssr=(be - bc) / np.mean(ske[clean, j]),
                                  t=(be - bc) / dse, crush=ae * 100)
    return out, int(clean.sum())


if __name__ == "__main__":
    args = sys.argv[1:]; ticker = "SPX"
    if args and args[0].isalpha():
        ticker = args.pop(0).upper()
    years = [int(y) for y in args] or list(range(2015, 2025))
    t0 = time.time()
    rows = build_series(years, ticker)
    evs = {"FOMC": set(FOMC), "CPI": set(CPI), "NFP": set(nfp_dates(years))}
    out, nclean = analyze(rows, evs)

    print(f"Daily event-conditional realized SSR -- {ticker} {years[0]}-{years[-1]} ({len(rows)} days, {nclean} clean)\n")
    print("event-leverage EXCESS = SSR_event - SSR_clean; t = t-stat of the slope difference; crush = mean event-day dSigma\n")
    print(f"{'event':>5} {'mat':>4} {'n':>4} | {'SSR_clean':>9} {'SSR_event':>9} {'dSSR':>7} {'t':>6} | {'crush %':>8}")
    for name in ("FOMC", "CPI", "NFP"):
        for lab in LAB:
            r = out[name][lab]
            print(f"{name:>5} {lab:>4} {r['n']:>4} | {r['ssr_c']:>9.2f} {r['ssr_e']:>9.2f} {r['dssr']:>7.2f} "
                  f"{r['t']:>6.1f} | {r['crush']:>8.2f}")
        print()
    json.dump(out, open(os.path.join(HERE, f"deevent_ssr_results_{ticker}.json"), "w"), indent=1)
    print(f"wall {time.time()-t0:.0f}s")

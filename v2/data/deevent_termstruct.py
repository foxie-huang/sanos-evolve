#!/usr/bin/env python3
"""
Real-data de-eventing, Tier 1: the scheduled-event variance lump in the EOD term structure.

For each trading day, fit the ATM total-variance term structure w(T)=sigma_ATM(T)^2*T over the FRONT
expiries as  w(T) = J + r*T:  the intercept J is the additive event lump (all expiries spanning an
imminent event carry it; sqrt(J) = the implied event move), r the diffusive variance rate. J is computed
WITHOUT the event calendar (just the front-loaded excess variance) and then validated AGAINST it -- so the
calendar alignment (Tier 1a) is non-circular. Overlaying FOMC/CPI, J(t) should spike on event-eves and be
~0 otherwise. Records wall-time.
    python3 deevent_termstruct.py [YEAR ...] [TICKER]
"""
import sys, os, glob, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from orats_loader import load_day                                          # noqa: E402
from empirical_ssr import atm_vol_skew                                     # noqa: E402

OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"

FOMC = ("2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14 "
        "2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13 "
        "2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 "
        "2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11 "
        "2020-01-29 2020-03-18 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 "
        "2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15 "
        "2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 "
        "2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 "
        "2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18").split()

# CPI monthly-print release dates 2015-2024, from the FRED API (release_id=10, Consumer Price Index),
# deduped to the monthly print (the later of the two February dates; the earlier Feb date each year is the
# annual seasonal-factor revision, not a new-data print). 22/24 match the prior hand-entered 2022-23 dates.
CPI = ("2015-01-16 2015-02-26 2015-03-24 2015-04-17 2015-05-22 2015-06-18 2015-07-17 2015-08-19 "
       "2015-09-16 2015-10-15 2015-11-17 2015-12-15 2016-01-20 2016-02-19 2016-03-16 2016-04-14 "
       "2016-05-17 2016-06-16 2016-07-15 2016-08-16 2016-09-16 2016-10-18 2016-11-17 2016-12-15 "
       "2017-01-18 2017-02-15 2017-03-15 2017-04-14 2017-05-12 2017-06-14 2017-07-14 2017-08-11 "
       "2017-09-14 2017-10-13 2017-11-15 2017-12-13 2018-01-12 2018-02-14 2018-03-13 2018-04-11 "
       "2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12 "
       "2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 "
       "2019-09-12 2019-10-10 2019-11-13 2019-12-11 2020-01-14 2020-02-13 2020-03-11 2020-04-10 "
       "2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10 "
       "2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 "
       "2021-09-14 2021-10-13 2021-11-10 2021-12-10 2022-01-12 2022-02-10 2022-03-10 2022-04-12 "
       "2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13 "
       "2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 "
       "2023-09-13 2023-10-12 2023-11-14 2023-12-12 2024-01-11 2024-02-13 2024-03-12 2024-04-10 "
       "2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11").split()


def lump(path, ticker="SPX", dlo=2, dhi=20):
    """J (intercept) and diffusive vol sqrt(r) of w(T)=J+r*T over front expiries in [dlo,dhi] dte."""
    day = load_day(path, [ticker]).get(ticker, {})
    T, w = [], []
    for exp, s in day.items():
        if not s["T"] or not s["F"] or s["dte"] is None or not (dlo <= s["dte"] <= dhi):
            continue
        vs = atm_vol_skew(s)
        if vs and vs[0] > 0.01:
            T.append(s["T"]); w.append(vs[0] ** 2 * s["T"])
    if len(T) < 4:
        return None
    T = np.array(T); w = np.array(w); o = np.argsort(T); T, w = T[o], w[o]
    r, J = np.polyfit(T, w, 1)                                             # w = r*T + J
    return float(J), float(np.sqrt(max(r, 1e-8)))


def j_series(year, ticker="SPX"):
    rows = []
    for p in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz")):
        d = os.path.basename(p).split("_")[1][:10]
        r = lump(p, ticker)
        if r:
            rows.append((d, r[0], r[1]))
    return rows


def _pre_event(date, cal, lo=1, hi=4):
    """True if `date` is 1..4 calendar days before any event in cal (the lump builds pre-release)."""
    from datetime import date as D
    dt = D.fromisoformat(date)
    return any(0 < (D.fromisoformat(e) - dt).days <= hi for e in cal)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    ticker = "SPX"
    if args and args[-1].isalpha():
        ticker = args.pop().upper()
    years = args or ["2023"]
    t0 = time.time()
    print(f"De-eventing Tier 1 -- event variance lump J in the {ticker} term structure (implied move sqrt(J))\n")
    allrows = []
    for yr in years:
        rows = j_series(yr, ticker)
        allrows += rows
        preF = [J for d, J, r in rows if _pre_event(d, FOMC)]
        preC = [J for d, J, r in rows if _pre_event(d, CPI) and not _pre_event(d, FOMC)]
        other = [J for d, J, r in rows if not _pre_event(d, FOMC) and not _pre_event(d, CPI)]
        mv = lambda xs: (np.sqrt(max(np.mean(xs), 0)) * 100 if xs else float("nan"))
        print(f"{yr}: {len(rows)} days | mean implied-move sqrt(J): "
              f"pre-FOMC {mv(preF):.2f}% (n={len(preF)})  pre-CPI {mv(preC):.2f}% (n={len(preC)})  "
              f"other {mv(other):.2f}% (n={len(other)})")
    # Tier 1a: are the highest-J days event-eves?
    top = sorted(allrows, key=lambda r: -r[1])[:12]
    print(f"\nTop-12 J days (Tier 1a -- should be event-eves):")
    for d, J, r in top:
        tag = "FOMC-eve" if _pre_event(d, FOMC) else ("CPI-eve" if _pre_event(d, CPI) else "?")
        print(f"  {d}  implied move {np.sqrt(max(J,0))*100:5.2f}%   diff vol {r*100:4.1f}%   {tag}")
    json.dump({d: [J, r] for d, J, r in allrows}, open(os.path.join(HERE, f"deevent_J_{ticker}.json"), "w"))
    print(f"\nwall {time.time()-t0:.0f}s")

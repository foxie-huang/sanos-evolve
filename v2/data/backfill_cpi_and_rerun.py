#!/usr/bin/env python3
"""
Backfill 2015-2021 (and 2024) CPI RELEASE dates from the FRED API, then rerun the event-conditional SSR
regime split with the complete CPI list. The disc_SLV sandbox blocks stlouisfed.org / bls.gov (WebFetch 403,
no curl network), so this must run on a machine with network + a FRED key.

    FRED_API_KEY=<key> python3 backfill_cpi_and_rerun.py          # (key likely already in your shell env)

It (1) resolves the authoritative CPI release_id from the CPIAUCSL series and asserts the release name is CPI,
(2) pulls its release dates, (3) sanity-checks them (weekday, ~12/yr, mid-month), (4) writes them to
cpi_dates_backfill.json AND prints a paste-ready block for deevent_termstruct.CPI, (5) reruns the pooled +
per-event-type regime split (deevent_ssr machinery) with the union of the existing 2022-23 CPI and the backfill.
Records nothing to the source automatically -- verify the dates, then paste them into deevent_termstruct.CPI.
"""
import os, sys, json, urllib.request, urllib.parse
from datetime import date as D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
KEY = os.environ.get("FRED_API_KEY", "")
if not KEY:
    sys.exit("set FRED_API_KEY (your macro_oracle key works)")


def fred(path, **params):
    q = urllib.parse.urlencode({**params, "api_key": KEY, "file_type": "json"})
    with urllib.request.urlopen(f"https://api.stlouisfed.org/fred/{path}?{q}", timeout=30) as r:
        return json.load(r)


# --- 1-2. authoritative CPI release_id + name, then its release dates 2015-2021 (+ 2024) ---------------------
rid = fred("series/release", series_id="CPIAUCSL")["releases"][0]["id"]
name = fred("release", release_id=rid)["releases"][0]["name"]
assert "Consumer Price Index" in name, f"FRED release {rid} is {name!r}, not CPI -- aborting"
raw = fred("release/dates", release_id=rid, realtime_start="2015-01-01", realtime_end="2024-12-31",
           limit=1000, sort_order="asc", include_release_dates_with_no_data="false")["release_dates"]
want = {d["date"] for d in raw if d["date"][:4] in {str(y) for y in range(2015, 2022)} | {"2024"}}
cpi_bf = sorted(want)

# --- 3. sanity checks (fail loudly rather than pollute the test with a wrong day) ---------------------------
byyr = {}
for d in cpi_bf:
    dt = D.fromisoformat(d)
    assert dt.weekday() < 5, f"{d} is not a weekday"
    assert 8 <= dt.day <= 20, f"{d} not mid-month (CPI is ~10th-15th)"   # loose guard
    byyr[d[:4]] = byyr.get(d[:4], 0) + 1
print(f"FRED release_id={rid}  name={name!r}")
print(f"backfilled {len(cpi_bf)} CPI dates; per year: {byyr}")
assert all(10 <= n <= 12 for n in byyr.values()), f"unexpected per-year counts {byyr}"
json.dump(cpi_bf, open(os.path.join(HERE, "cpi_dates_backfill.json"), "w"))
print("wrote cpi_dates_backfill.json\npaste-ready for deevent_termstruct.CPI:\n  " + " ".join(cpi_bf))

# --- 4. rerun the regime split with the COMPLETE CPI list ---------------------------------------------------
import deevent_termstruct as DTm
from deevent_ssr import nfp_dates, LAB
CPI_FULL = sorted(set(DTm.CPI) | set(cpi_bf))
rows = json.load(open(os.path.join(HERE, ".deevent_ssr_SPX.json"))); dates = sorted(rows)
spot = np.array([rows[d][0] for d in dates]); vol = np.array([rows[d][1] for d in dates]); sk = np.array([rows[d][2] for d in dates])
ret = np.diff(np.log(spot)); dvol = np.diff(vol, axis=0); td = dates[1:]; ske = sk[1:]
ok = np.all(np.isfinite(dvol), 1) & np.isfinite(ret) & (np.abs(ret) < 0.15)
yr = np.array([int(td[i][:4]) for i in range(len(td))])
EVS = {"FOMC": set(DTm.FOMC), "CPI": set(CPI_FULL), "NFP": set(nfp_dates(range(2015, 2025)))}
evany = set().union(*EVS.values())


def hc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); xc = x - x.mean(); sxx = np.sum(xc ** 2)
    b = np.sum(xc * (y - y.mean())) / sxx; e = y - y.mean() - b * xc
    return b, np.sqrt(np.sum(xc ** 2 * e ** 2) / sxx ** 2)


def cell(mev, mcl, j):
    be, see = hc(ret[mev], dvol[mev, j]); bc, sec = hc(ret[mcl], dvol[mcl, j])
    skc = np.nanmean(ske[mcl, j]); skm = np.nanmean(ske[mev, j])
    return int(mev.sum()), bc / skc, be / skm, (be - bc) / skc, (be - bc) / np.sqrt(see ** 2 + sec ** 2)


print("\n=== CPI (now full 2015-24) event-conditional SSR by regime ===")
print(f"{'regime':>15} {'mat':>3} {'n':>4} | {'SSR_c':>6} {'SSR_e':>6} {'dSSR':>7} {'t_HC':>6}")
for rn, (y0, y1) in {"2015-19 normal": (2015, 2019), "2020-21 stress": (2020, 2021), "2022-24 grind": (2022, 2024)}.items():
    inr = (yr >= y0) & (yr <= y1) & ok
    cln = inr & np.array([td[i] not in evany for i in range(len(td))])
    m = inr & np.array([td[i] in EVS["CPI"] for i in range(len(td))])
    for j in (0, 2):
        n, sc, se, ds, t = cell(m, cln, j)
        print(f"{rn:>15} {LAB[j]:>3} {n:>4} | {sc:>6.2f} {se:>6.2f} {ds:>7.2f} {t:>6.1f}")
    print()

print("=== pooled all-macro (FOMC+CPI+NFP), normal regime, with full CPI ===")
inr = (yr >= 2015) & (yr <= 2019) & ok; cln = inr & np.array([td[i] not in evany for i in range(len(td))])
m = inr & np.array([td[i] in evany for i in range(len(td))])
for j in (0, 2):
    n, sc, se, ds, t = cell(m, cln, j)
    print(f"  {LAB[j]}: n={n}  SSR_c {sc:.2f}  SSR_e {se:.2f}  dSSR {ds:+.2f}  t_HC {t:.1f}")

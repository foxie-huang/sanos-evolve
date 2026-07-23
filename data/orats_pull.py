#!/usr/bin/env python3
"""
orats_pull.py -- resumable per-symbol ORATS EOD historical puller.

Pulls ORATS `hist/strikes` (full option chain, EOD ~14min-before-close, with the
SMV smoothed surface + raw bid/ask + greeks) for one or more tickers across a date
range, ONE trading day per request, writing one gzipped-JSON file per day. Only the
symbols you ask for -- no 5,000-symbol bulk, no external drive (SPX full history is
~1-2 GB).

Verified 2026-06: `hist/strikes` serves SPX back to 2008-01-02 (521 rows) through
2024 (10,519 rows), expiries out to LEAPS. 42 fields per row incl.
callBidPrice/callAskPrice, putBidPrice/putAskPrice, call/put Bid|Mid|Ask Iv,
smvVol, callValue/putValue (theos), delta/gamma/theta/vega/rho/phi/driftlessTheta,
stockPrice/spotPrice, residualRate, OI and volume.

  Forward/rate note: hist/strikes carries `residualRate` + stock/spot price but not
  an explicit per-expiry forward. Get the clean forward either (a) from the
  `hist/cores` endpoint (per-expiry ATM forward + ATM vol), or (b) by put-call
  parity from callValue-putValue at each strike. Add --cores to also pull cores.

Usage:
    export ORATS_TOKEN=...                     # NEVER hard-code the token
    python3 orats_pull.py --start 2015-01-01 --end 2024-12-31 --tickers SPX
    python3 orats_pull.py --start 2007-01-01 --tickers SPX,NDX,RUT,VIX   # <=10/req

Resumable: a day whose output (or .empty marker) already exists is skipped, so you
can stop/restart freely within the trial window. Holidays return 0 rows -> a marker
file is written so they aren't retried.
"""
import os, sys, json, gzip, time, argparse, urllib.request, urllib.error
from datetime import date, timedelta

BASE = os.environ.get("ORATS_BASE", "https://api.orats.io/datav2")
TOKEN = os.environ.get("ORATS_TOKEN")


def fetch(endpoint, params, retries=6):
    """GET with exponential backoff; return the parsed JSON 'data' list (or [])."""
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{endpoint}?token={TOKEN}&{q}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read())
            return d.get("data", []) if isinstance(d, dict) else (d or [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []                                    # no data for this date (holiday / pre-listing)
            if e.code in (401, 403):
                sys.exit(f"AUTH ERROR {e.code}: token invalid/expired -- stopping (rotate + rerun to resume).")
            wait = 2 ** attempt                              # 429 / 5xx / other -> retry
            print(f"    retry {attempt+1}/{retries} in {wait}s (HTTP {e.code})", file=sys.stderr, flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError) as e:
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} in {wait}s ({type(e).__name__})", file=sys.stderr, flush=True)
            time.sleep(wait)
    print(f"    GAVE UP after {retries} retries: {endpoint} {params}", file=sys.stderr, flush=True)
    return None


def trading_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:            # Mon-Fri; holidays handled via empty response
            yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--tickers", default="SPX", help="comma list, <=10 (e.g. SPX,NDX,RUT,VIX)")
    ap.add_argument("--out", default="orats_eod")
    ap.add_argument("--cores", action="store_true", help="also pull hist/cores (per-expiry forward/ATM vol)")
    ap.add_argument("--throttle", type=float, default=0.2, help="seconds between requests")
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("set ORATS_TOKEN in the environment first")

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    os.makedirs(args.out, exist_ok=True)
    tlabel = args.tickers.replace(",", "-")
    n_pull = n_done = n_holiday = n_fail = 0
    for d in trading_days(start, end):
        ds = d.isoformat()
        fp = os.path.join(args.out, f"{tlabel}_{ds}.json.gz")
        if os.path.exists(fp) or os.path.exists(fp + ".empty"):
            n_done += 1
            continue
        rows = fetch("hist/strikes", {"ticker": args.tickers, "tradeDate": ds})
        if rows is None:                              # persistent failure: log, retry on next run
            with open(os.path.join(args.out, "_failures.log"), "a") as fl:
                fl.write(ds + "\n")
            n_fail += 1
            time.sleep(args.throttle)
            continue
        if not rows:
            open(fp + ".empty", "w").close()          # holiday / no data
            n_holiday += 1
        else:
            payload = {"strikes": rows}
            if args.cores:
                payload["cores"] = fetch("hist/cores", {"ticker": args.tickers, "tradeDate": ds})
                time.sleep(args.throttle)
            with gzip.open(fp, "wt") as f:
                json.dump(payload, f)
            n_pull += 1
            if n_pull % 50 == 0:
                print(f"  {ds}: {n_pull} days pulled ({len(rows)} rows today)", flush=True)
        time.sleep(args.throttle)
    print(f"done: {n_pull} pulled, {n_done} already present, {n_holiday} empty/holiday, {n_fail} failed (see _failures.log)")


if __name__ == "__main__":
    main()

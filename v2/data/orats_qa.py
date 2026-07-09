#!/usr/bin/env python3
"""
orats_qa.py -- QA a directory of ORATS day files produced by orats_pull.py.

Checks: date coverage + gaps (weekdays with neither data nor a holiday marker),
per-year data-day counts, per-ticker row counts, load integrity, and the parity-
forward recovery rate. Run after the pull completes (also works mid-pull).

    python3 orats_qa.py /path/to/orats_eod
"""
import sys, os, glob, re
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orats_loader import load_day

DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json\.gz$")
EMPTY_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json\.gz\.empty$")


def weekdays(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d.isoformat()
        d += timedelta(days=1)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "orats_eod"
    data = {m.group(1): p for p in glob.glob(os.path.join(root, "*.json.gz"))
            if (m := DATE_RE.search(p))}
    empt = {m.group(1) for p in glob.glob(os.path.join(root, "*.empty"))
            if (m := EMPTY_RE.search(p))}
    if not data:
        sys.exit("no data files in " + root)

    dates = sorted(data)
    lo, hi = dates[0], dates[-1]
    print(f"range {lo} .. {hi}   data days: {len(data)}   holiday markers: {len(empt)}")

    have = set(data) | empt
    gaps = [wd for wd in weekdays(date.fromisoformat(lo), date.fromisoformat(hi))
            if wd not in have]
    print(f"GAPS (weekdays missing both data + holiday marker): {len(gaps)}"
          + (f"  e.g. {gaps[:10]}" if gaps else "  -> none, coverage complete"))

    yr = {}
    for dt in dates:
        yr[dt[:4]] = yr.get(dt[:4], 0) + 1
    print("per-year data days:", " ".join(f"{y}:{c}" for y, c in sorted(yr.items())))

    print("integrity / per-ticker rows / SPX forward recovery (~3 sampled days/year):")
    bad = 0
    for y in sorted(yr):
        yd = [dt for dt in dates if dt[:4] == y]
        for dt in yd[:: max(1, len(yd) // 3)][:3]:
            try:
                day = load_day(data[dt])
            except Exception as e:
                print(f"   {dt}: LOAD FAILED ({type(e).__name__})"); bad += 1; continue
            rows = {t: sum(len(s["strike"]) for s in day[t].values()) for t in sorted(day)}
            spx = day.get("SPX", {})
            fok = sum(1 for s in spx.values() if s["F"])
            print(f"   {dt}: rows {rows} | SPX fwd {fok}/{len(spx)} expiries")
    print(f"load failures: {bad}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
The clean-flank cumulant bridge (the DESIGNED de-eventing procedure) on a real snapshot.

Setup: one EOD chain before an event, expiries T1<T2<event<T3<T4. Spot- & time-homogeneity make the
risk-neutral cumulants of the log-return ADD across independent forward windows, so "propagate mu2 through
K^diff(theta) -> mu3^diff" is, in cumulant space, just "add the diffusive forward-cumulant increment." The
CLEAN flanking windows (event in neither endpoint -> the event lump cancels in the increment) calibrate the
smooth diffusive forward-cumulant rate kappa_n^diff'(T); the event-crossing window [T2,T3] carries diffusive
+ event, so
      J_n = Delta kappa_n([T2,T3]) - rate_n^diff(T_cross) * (T3 - T2)
is the event cumulant residual  mu3 (-) mu3^diff:  J2 = variance lump (sqrt = implied move), J3 = event
skew-cumulant, J4 = event kurtosis. Model-free cumulants via Bakshi-Kapadia-Madan (BKM) on the OTM mid-price
strip (the marginals' cumulants directly). This is the STATICS channel of the bridge (memory: statics =
marginal-moment residual = J + event-skew). Records wall-time.
    python3 deevent_bridge.py [EVENT_DATE=2024-05-01] [TICKER=SPX] [BACKDAYS=9]
"""
import sys, os, glob, time
import numpy as np
from datetime import date as D, timedelta

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from orats_loader import load_day                                          # noqa: E402
from deevent_termstruct import OUT, lump                                   # noqa: E402


def bkm(s, xclip=0.35):
    """Model-free forward cumulants (kappa2=V, kappa3=W, kappa4=X) to this expiry from the OTM strip."""
    K = np.asarray(s["strike"], float); F = s["F"]; DF = s["DF"]
    if not F or not DF:
        return None
    O = np.array([(p if k < F else c) if (p is not None and c is not None) else np.nan
                  for k, c, p in zip(K, s["cmid"], s["pmid"])], float)
    x = np.log(K / F)
    ok = np.isfinite(O) & (O > 0) & (np.abs(x) < xclip)
    K, O, x = K[ok], O[ok], x[ok]
    if len(K) < 8:
        return None
    o = np.argsort(K); K, O, x = K[o], O[o], x[o]; Of = O / DF             # forward OTM value
    V = np.trapezoid(2 * (1 - x) / K ** 2 * Of, K)
    W = np.trapezoid((6 * x - 3 * x ** 2) / K ** 2 * Of, K)
    X = np.trapezoid((12 * x ** 2 - 4 * x ** 3) / K ** 2 * Of, K)
    return float(V), float(W), float(X)


def snapshot(event, backdays):
    """First existing EOD file on/before event-backdays (scan back for weekends/holidays)."""
    ev = D.fromisoformat(event)
    for d in range(backdays, backdays + 6):
        p = f"{OUT}/SPX-NDX-RUT-VIX_{(ev - timedelta(d)).isoformat()}.json.gz"
        if os.path.exists(p):
            return p, (ev - timedelta(d))
    return None, None


def bridge(event="2024-05-01", ticker="SPX", backdays=9):
    p, snap = snapshot(event, backdays)
    if not p:
        return None
    day = load_day(p, [ticker]).get(ticker, {})
    ev_T = (D.fromisoformat(event) - snap).days / 365.0
    rows = []
    for exp, s in day.items():
        if not s["T"] or not s["F"] or s["dte"] is None or not (2 <= s["dte"] <= 55):
            continue
        c = bkm(s)
        if c:
            rows.append((s["T"], exp, c[0], c[1], c[2]))                   # (T, expiry, k2, k3, k4)
    rows.sort()
    if len(rows) < 6:
        return None
    T = np.array([r[0] for r in rows])
    K = {"J2": np.array([r[2] for r in rows]), "J3": np.array([r[3] for r in rows]),
         "J4": np.array([r[4] for r in rows])}
    # bridge via a STEP regression: kappa_n(T) = b*T + c*T^2  (smooth diffusive, through origin)
    #                                          + J_n * 1[T >= event]  (the additive event lump).
    # Uses ALL expiries at once (robust) instead of fragile adjacent differences; J_n = the step = mu3 (-) mu3^diff.
    H = (T >= ev_T).astype(float); npre = int((H == 0).sum()); npost = int(H.sum())
    if npre < 2 or npost < 3:
        return None
    A = np.column_stack([T, T ** 2, H])
    J = {n: float(np.linalg.lstsq(A, K[n], rcond=None)[0][2]) for n in K}
    J2, J3, J4 = J["J2"], J["J3"], J["J4"]
    Jint = lump(p, ticker)
    return dict(event=event, snap=snap.isoformat(), npre=npre, npost=npost,
                J2=J2, J3=J3, J4=J4, move=np.sqrt(max(J2, 0)),
                skew=(J3 / J2 ** 1.5 if J2 > 0 else float("nan")),
                exkurt=(J4 / J2 ** 2 if J2 > 0 else float("nan")),
                Jint=(Jint[0] if Jint else float("nan")))


if __name__ == "__main__":
    ev = sys.argv[1] if len(sys.argv) > 1 else "2024-05-01"
    tk = sys.argv[2] if len(sys.argv) > 2 else "SPX"
    bd = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    t0 = time.time()
    events = ev.split(",")
    print(f"Clean-flank cumulant bridge -- {tk}  (event cumulant residual mu3 (-) mu3^diff)\n")
    print(f"{'event':>11} {'snap':>11} {'pre/post':>9} | {'move%':>6} {'ev-skew':>8} {'ev-exkurt':>9} | {'J2':>9} {'J2_intcpt':>10}")
    for e in events:
        r = bridge(e, tk, bd)
        if not r:
            print(f"{e:>11}  (no snapshot / too few expiries)"); continue
        print(f"{r['event']:>11} {r['snap']:>11} {r['npre']:>4}/{r['npost']:<4} | {r['move']*100:>6.2f} {r['skew']:>8.2f} "
              f"{r['exkurt']:>9.2f} | {r['J2']:>9.2e} {r['Jint']:>10.2e}")
    print(f"\n(move=sqrt(J2)=implied event move; ev-skew=J3/J2^1.5; J2_intcpt=crude term-structure intercept for sanity)")
    print(f"wall {time.time()-t0:.1f}s")

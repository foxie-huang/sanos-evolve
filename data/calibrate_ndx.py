#!/usr/bin/env python3
"""
OFF-SPX NDX calibration -- fit theta to NDX with OWN-STRIP + REALISED targets only (no VIX, no cliquets),
the paper's off-SPX recipe (Sec. 'what stands in for VIX'). Statics = NDX SANOS marginals; leverage/SSR
target = NDX REALISED SSR (own strip); vov target = NDX REALISED variance-of-variance (realised vol of the
own-strip variance-swap level; tenors from NDXTENORS, default [30,90]d). Ridge-stabilised joint fit
(reuses calibrate_joint_torch.fit_date). Records wall-time.

The two caveats this file used to carry are no longer open questions, they are MEASURED and correctable:
  * "the vov is P (realised), matched to the model's Q readout" -- the gap is not mainly a risk premium.
    It is the SPOT-vs-FUTURE damping plus an object mismatch, measured on SPX by spx_pq_vov.py and
    applied with NDXVOVPQ=1.
  * "model vov term structure (option-expiry) vs realised (variance-tenor) is an approximate alignment"
    -- this is exactly right and it is the LARGER effect. vix.py:155: the readout is an option at tau on
    a 30-DAY forward variance, while realised at tenor T is the daily vol of the T-day variance. The
    mismatch grows monotonically with T (Q/P 0.333 at 7d to 1.220 at 180d) and is what made the NDX
    decay look unreachably steep.
    python3 calibrate_ndx.py [cpu|mps] [DATE ...]
"""
import sys, os, glob, gzip, json, time
import numpy as np
import torch

DEVICE = "mps"
if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "mps"):
    DEVICE = sys.argv.pop(1)
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J                                   # noqa: E402
import calibrate_slv_exact_ts as C                                  # noqa: E402
from empirical_ssr import empirical_ssr                             # noqa: E402
from ndx_vov_readout import expiry, var_swap, rvov                  # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at  # noqa: E402
from discslv_2f import TwoFactorSV                                  # noqa: E402
from discslv_slv import Epi_V                                       # noqa: E402
import discslv_torch as DTt                                         # noqa: E402

OUT = J.OUT; NAMES = J.NAMES; DT = J.DT; NS = C.NS
CACHE = os.path.join(HERE, ".ndx_cache"); os.makedirs(CACHE, exist_ok=True)
# NDXTENORS widens the vov anchor set. The 2-tenor default is inherited from the v3 paper, where it
# was forced by data defects; those defects are now diagnosed and repaired (roll -> const-maturity,
# object -> Q/P, bad prints -> screen), and coverage is measured at >=93% from 14d and ~100% from 45d.
# At 2 anchors the fit is 7 residuals against 7 free parameters -- EXACTLY determined, so the residual
# cannot reveal misspecification and in-sample vov RMS means very little. 14..180d gives 13 vs 7.
# EXCLUDE 7d (26% mean coverage; 1 trading day in 2012/2016/2017) and 270/365d (no bracketed VIX
# option, so no Q/P, and the raw long wing is erratic -- see 6e.33).
TENORS = [int(x) for x in (os.environ.get("NDXTENORS") or "30,90").split(",") if x.strip()]


def ndx_ssr_cached(year):
    k = os.path.join(CACHE, f"ssr_{year}.npz")
    if os.path.exists(k):
        d = np.load(k); return d["ssr"], int(d["nd"])
    paths = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    ssr, nd = empirical_ssr(paths, ns=NS, dt=DT, ticker="NDX")
    np.savez(k, ssr=ssr, nd=nd); return ssr, nd


# NDXVOVCM=1 takes the vov target from the CONSTANT-MATURITY series instead of the nearest-expiry
# snapped one. Default OFF -- the snapped targets are what every published NDX number was fitted to.
#
# WHY. `ndx_vov_readout.expiry()` snaps to the NEAREST listed expiry, so the daily variance-swap
# series sits at a dte that JUMPS (30d target: 32d, 31d, ... 17d, then 45d), and `rvov` --
# std(diff(log(x))) -- counts every jump as a vol move when it is a step ALONG the term structure.
# Measured: roll severity falls monotonically with tenor (0.168 at 7d to 0.009 at 365d) and within
# each year rvov correlates with it across tenors at +0.550, positive in 10/10 years.
#
# The constant-maturity series interpolates between the two BRACKETING expiries linearly in TOTAL
# VARIANCE, so it sits at exactly the target tenor every day and cannot roll. Built by
# `diagnostics/ndx_vov_cm.py` into `.ndx_cm_cache/` -- a SEPARATE cache; this must never write to
# `.ndx_cache/vov_<year>.npz`, which holds the snapped targets the published fits used.
#
# WHAT IT DOES AND DOES NOT FIX (measured over 10 years, see handoff):
#   FIXES the curve's self-consistency -- the 90d point vs its 60/120d neighbours goes -18.7% -> -9.5%,
#     and the 2-point and 11-tenor decay exponents converge (0.616->0.566 and 0.334->0.382).
#   DOES NOT fix reachability -- 7/10 dates stay below the model's ~0.62 decay floor. 2020 is
#     transformed (ratio 0.524->0.709); the melt-up years 2021/2023/2024 barely move.
# Adopt it because the targets are more correct, NOT because the fit will improve.
NDX_VOV_CM = os.environ.get("NDXVOVCM", "0") == "1"
CM_CACHE = os.path.normpath(os.path.join(HERE, "..", "..", "vix_joint_refit", ".ndx_cm_cache"))

# ---- NDXVOVPQ: put the realised NDX target on the same OBJECT the model's readout computes --------
#
# THE MISMATCH. The model reads vov through `VX.vix_ivol` -- the IV of a VIX-STYLE OPTION, i.e. an
# option on a FUTURE. On SPX the target is exactly that (VIX ATM option IV), so readout and target
# are the same object. On NDX there is no liquid vol-option market, so the target is the realised vol
# of the var-swap SPOT level, and the two differ by the spot-vs-future damping: a future's vol is
# pulled in by mean reversion relative to spot. That is not a modelling choice, it is an object gap.
#
# THE RATIO IS MEASURED, NOT ASSUMED. SPX is the one ticker carrying both sides, so
# `diagnostics/spx_pq_vov.py` builds Q (VIX ATM option IV) and P (realised vov of the CONSTANT-MATURITY
# SPX strip -- this same construction, applied to SPX) and takes Q/P per year per tenor. Measured over
# 10 years: 0.769 at 30d (below 1 in 10 of 10 years) and 1.030 at 90d. The 30d anchor is the inflated
# one, which is why the realised construction demands a decay at the model's 1/sqrt(T) floor (ratio
# 0.619, p=0.442) while the implied construction demands 0.823 (p=0.179). Feed SPX realised targets
# and SPX inherits the same unreachable decay -- it is a property of the CONSTRUCTION, not of NDX.
#
# WHAT IT ASSUMES, and it is the whole risk: that the SPX ratio transports to NDX. Both are US
# large-cap index vol markets with the same macro drivers, but this is not measurable on NDX (no
# liquid NDX vol options) and so cannot be verified on the ticker it is applied to. PER-YEAR ratios
# are used rather than the 10-year constant so the correction tracks regime; the constant is the
# robustness check, not the default.
#
# REQUIRES NDXVOVCM=1 or NDXVOVSCR=1. The ratio's denominator is a constant-maturity P, so applying
# it to the snapped series would divide out one construction and multiply in another. NOTE the
# residual mismatch when pairing with NDXVOVSCR: the Q/P denominator is the SPLICED, UNSCREENED SPX
# estimator, so the ratio is off by however much the splice+screen moves SPX. Bounded and stated
# rather than assumed away -- see the handoff for the measured size.

# NDXVOVSCR: the const-maturity series with the SPLICE fixed (increments only between adjacent
# trading days) and single-day bad prints removed by a rolling-median screen. Built by
# diagnostics/ndx_vov_screen.py into its own cache. Required for multi-tenor fitting: the objective
# is a sum of RELATIVE residuals, so one bad print (2018 45d reads 9.7 against neighbours near 1.5)
# would dominate the entire fit.
NDX_VOV_SCR = os.environ.get("NDXVOVSCR", "0") == "1"
SCR_CACHE = os.path.normpath(os.path.join(HERE, "..", "..", "vix_joint_refit", ".ndx_scr_cache"))
NDX_VOV_PQ = os.environ.get("NDXVOVPQ", "0") == "1"
PQ_JSON = os.path.normpath(os.path.join(HERE, "..", "..", "vix_joint_refit", "spx_pq_vov.json"))
if NDX_VOV_PQ and not (NDX_VOV_CM or NDX_VOV_SCR):
    raise SystemExit("NDXVOVPQ=1 requires NDXVOVCM=1 or NDXVOVSCR=1 -- the Q/P ratio was measured "
                     "against a CONSTANT-MATURITY realised P (spx_pq_vov.py calls "
                     "ndx_vov_cm.cm_level), so applying it to the snapped series mixes two "
                     "constructions.")
if NDX_VOV_SCR and NDX_VOV_CM:
    raise SystemExit("NDXVOVSCR=1 and NDXVOVCM=1 are two different realised series -- pick one.")


def pq_ratio(year):
    """Per-year, per-tenor SPX Q/P. Raises rather than defaulting to 1.0 -- a silent no-op correction
    would be indistinguishable from an applied one in the fit record."""
    if not os.path.exists(PQ_JSON):
        raise SystemExit(f"NDXVOVPQ=1 but {PQ_JSON} is missing -- run diagnostics/spx_pq_vov.py first")
    d = json.load(open(PQ_JSON))
    if str(year) not in d:
        raise SystemExit(f"NDXVOVPQ=1 but spx_pq_vov.json has no year {year}")
    z = d[str(year)]
    T = [int(x) for x in z["tenors"]]
    out = []
    for t in TENORS:
        if t not in T:
            raise SystemExit(f"NDXVOVPQ=1 but spx_pq_vov.json has no tenor {t}d (has {T})")
        i = T.index(t)
        q, p = float(z["q_mean"][i]), float(z["p_rvov"][i])
        if not (np.isfinite(q) and np.isfinite(p) and p > 0 and q > 0):
            raise SystemExit(f"NDXVOVPQ=1 but {year} {t}d has a non-finite Q/P (q={q}, p={p})")
        out.append(q / p)
    return np.array(out)


def ndx_vov_cached(year):
    if NDX_VOV_SCR:
        f = os.path.join(SCR_CACHE, f"vov_scr_NDX_{year}.npz")
        if not os.path.exists(f):
            raise SystemExit(f"NDXVOVSCR=1 but {f} is missing -- run diagnostics/ndx_vov_screen.py first")
        z = np.load(f)
        T = [int(x) for x in z["tenors"]]
        missing = [t for t in TENORS if t not in T]
        if missing:
            raise SystemExit(f"NDXVOVSCR=1 but the screened cache lacks tenors {missing} (has {T})")
        rv = np.array([float(z["rvov"][T.index(t)]) for t in TENORS])
        if not np.all(np.isfinite(rv)):
            bad = [t for t, v in zip(TENORS, rv) if not np.isfinite(v)]
            raise SystemExit(f"NDXVOVSCR=1 but tenors {bad} have a non-finite screened rvov in {year}")
        return rv * pq_ratio(year) if NDX_VOV_PQ else rv
    if NDX_VOV_CM:
        f = os.path.join(CM_CACHE, f"vov_cm_NDX_{year}.npz")
        if not os.path.exists(f):
            raise SystemExit(f"NDXVOVCM=1 but {f} is missing -- run diagnostics/ndx_vov_cm.py first")
        z = np.load(f)
        T = [int(x) for x in z["tenors"]]
        rv = np.array([float(z["rvov"][T.index(t)]) for t in TENORS])
        return rv * pq_ratio(year) if NDX_VOV_PQ else rv
    k = os.path.join(CACHE, f"vov_{year}.npz")
    if os.path.exists(k):
        return np.load(k)["rvov"]
    series = {t: [] for t in TENORS}
    for f in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz")):
        try:
            recs = json.load(gzip.open(f))["strikes"]
            for t in TENORS:
                ch = expiry(recs, "NDX", t)
                if ch:
                    series[t].append(var_swap(ch, ch[0]["dte"] / 365))
        except Exception:
            continue
    rv = np.array([rvov(np.array(series[t])) for t in TENORS])
    np.savez(k, rvov=rv); return rv


def ndx_spot(date):                                                # VXN-equivalent: own-strip sqrt(V) at 30d
    recs = json.load(gzip.open(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"))["strikes"]
    ch = expiry(recs, "NDX", 30)
    return float(var_swap(ch, ch[0]["dte"] / 365))


def build_ctx_ndx(date):
    yr = date[:4]
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz", ticker="NDX")
    sig_ref = ref_vol(chain)
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    lev = {k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, max(NS) + 1)}
    C._CACHE.clear(); C._CACHE.update(lev)
    LT = {n: [DTt.lev_torch(lev[k + 1].coef, lev[k + 1].zmax, lev[k + 1].safety) for k in range(n)] for n in NS}
    emp, nd = ndx_ssr_cached(yr); vov_d = ndx_vov_cached(yr)
    return dict(date=date, sig_ref=float(sig_ref), LT=LT, spot=ndx_spot(date),
                vdtes=np.array(TENORS, float), vov_d=vov_d, emp=emp, nd=nd, cache_hit=True)


if __name__ == "__main__":
    dates = sys.argv[1:] or ["2018-06-01", "2020-06-01", "2021-06-01", "2022-06-01"]
    anchor = np.asarray(C.X0_MAP["ts"], float)
    print(f"OFF-SPX NDX fit [{DEVICE}] -- own-strip SANOS + realised SSR + realised var-of-var (no VIX)", flush=True)
    t_all = time.time(); rows = []
    for date in dates:
        if not os.path.exists(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"):
            print(f"  {date}: MISSING"); continue
        tc = time.time(); ctx = build_ctx_ndx(date); t_ctx = time.time() - tc
        res = J.fit_date_multistart(ctx, anchor, J.W_REG, max_nfev=40)
        m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                          ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
        ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
        se = 100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2))
        ve = 100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2))
        rows.append((date, ctx, se, ve, res, s, v))
        print(f"\n[{date}]  ctx {t_ctx:.1f}s  multistart {res.wall:.0f}s / {res.n_starts} seeds  "
              f"(cost best {min(res.costs):.4f} vs worst {max(res.costs):.4f})", flush=True)
        print(f"   NDX realised SSR {np.round(ctx['emp'],2)}   realised vov {np.round(ctx['vov_d'],2)}"
              f"   (VXN-eq spot {ctx['spot']*100:.1f})", flush=True)
        print(f"   theta " + "  ".join(f"{n}={x:.3f}" for n, x in zip(NAMES, res.x)), flush=True)
        print(f"   model SSR {np.round(s,2)}   model vov {np.round(v,2)}", flush=True)
        print(f"   SSR RMS {se:.1f}%   vov RMS {ve:.1f}%", flush=True)
    print(f"\n=== OFF-SPX NDX: {len(rows)} dates in {time.time()-t_all:.0f}s [{DEVICE}] ===")
    print(f"{'date':>12}{'SSR%':>7}{'vov%':>7}{'fit_s':>7}{'evals':>7}")
    for date, ctx, se, ve, res, s, v in rows:
        print(f"{date:>12}{se:>7.1f}{ve:>7.1f}{res.wall:>7.0f}{res.nfev:>7d}")

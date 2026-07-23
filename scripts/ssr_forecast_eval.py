"""
Out-of-sample SSR-forecast evaluation harness  (design: ../ssr_forecast_test_design.md)
=======================================================================================

Tests review Concern #1: does the model's forward-looking, option-IMPLIED SSR (SSR^Q_t)
forecast next-period REALISED spot-vol comovement better -- or with genuine incremental
information -- than a cheap BACKWARD-looking realised estimate (SSR^P_t)?

    H1 (model's claim): SSR^Q_t encompasses / beats the trailing realised estimate.
    H0 (honest null):   SSR^Q_t adds nothing beyond SSR^P_t  -> the hedging headline
                        (27-47% vs Black) is just the minimum-variance-delta effect.

This module is a PURE, data-source-agnostic statistics + replay engine. Real inputs enter
through three hooks; NOTHING here fabricates real-data numbers.

------------------------------------------------------------------------------------------
WIRING (exact POC signatures, from poc/) -- replace the two forecaster hooks:

  ModelSSRForecaster  (forward-looking, Q):
      per test date t, calibrate SANOS-Evolve to the t cross-section and read SSR^Q_t.
        from calibrate_2f import kernel                 # kernel(x) -> TwoFactorSV, x is 9 params
        from discslv_2f    import ssr_2f                # ssr_2f(K, n, nk, dm) -> (ssr, vol, skew)
        # calibrate via scipy.optimize.least_squares(residuals, X0, ...) as in calibrate_2f.py
        # tenor T (years) -> steps n = round(T / K.dt); ssr = ssr_2f(K, n, nk=16)[0]
      For full statics use calibrate.py + data_port.load_chain(source=csv_path) which returns
        {"maturities", "marginals"(GMM), "forwards", "source"}.

  RealisedSSRForecaster (backward-looking, P):   IMPLEMENTED HERE (market-data regression;
      the POC has no market-data realised-SSR -- realized_ssr.py is MC-on-kernel, not this).
      trailing OLS slope of dATM-IV on log-return over [t-w, t], optionally / skew -> SSR units.

  load_panel: build a Panel from the ORATS cache (memory: ORATS EOD SPX+VIX 2010-2026),
      or poc/fetch_data.py CSVs (columns: expiration,type,dte,strike,bid,ask,impliedVolatility).

Units: keep TARGET and FORECAST in the same units. Two consistent choices:
   (a) SSR units:  target = beta_real / skew_obs ,  model = ssr_2f(...)[0]      [default]
   (b) beta units: target = beta_real            ,  model = ssr * skew_model
The stats engine is unit-agnostic; the caller guarantees consistency.
------------------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

try:
    from scipy import stats as _sps
    def _norm_sf(z):  return _sps.norm.sf(z)
    def _t_sf(t, df): return _sps.t.sf(t, df)
except Exception:                                   # scipy-free fallback (normal approx)
    import math
    def _norm_sf(z):  return 0.5 * math.erfc(z / math.sqrt(2.0))
    def _t_sf(t, df): return _norm_sf(t)


# ======================================================================================
# 1. HAC (Newey-West) inference primitives
# ======================================================================================
def newey_west_lrv(u: np.ndarray, lags: int) -> float:
    """Long-run variance of a mean-zero-ish scalar series via Bartlett-weighted Newey-West."""
    u = np.asarray(u, float)
    u = u - u.mean()
    n = u.size
    g0 = (u @ u) / n
    s = g0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = (u[k:] @ u[:-k]) / n
        s += 2.0 * w * gk
    return s


def nw_mean_test(d: np.ndarray, lags: int, alternative: str = "two-sided") -> dict:
    """Newey-West t-test that E[d] = 0. `lags` should be >= (forward-window overlap - 1)."""
    d = np.asarray(d, float)
    n = d.size
    mean = d.mean()
    lrv = newey_west_lrv(d, lags)
    se = np.sqrt(lrv / n)
    t = mean / se if se > 0 else np.nan
    if alternative == "less":       p = _t_sf(-t, n - 1)         # H1: mean < 0
    elif alternative == "greater":  p = _t_sf(t, n - 1)          # H1: mean > 0
    else:                           p = 2.0 * _t_sf(abs(t), n - 1)
    return dict(mean=mean, se=se, t=t, p=p, n=n)


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int, names: Optional[list] = None) -> dict:
    """OLS with an intercept prepended to X, Newey-West HAC standard errors.

    Returns coefs, HAC ses, t-stats, two-sided p, R^2, resid, and the design used.
    """
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k0 = X.shape
    Xd = np.column_stack([np.ones(n), X])               # intercept + regressors
    k = k0 + 1
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)                 # pinv: robust to a collinear/constant column
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta
    # HAC meat: S = sum_t w_k xt xt' u_t u_{t-k}   (Bartlett)
    S = np.zeros((k, k))
    xu = Xd * resid[:, None]
    S += xu.T @ xu / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = xu[lag:].T @ xu[:-lag] / n
        S += w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.where(se > 0, beta / se, np.nan)
    p = 2.0 * _t_sf(np.abs(tstat), max(n - k, 1))
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nm = ["const"] + (names or [f"x{i}" for i in range(k0)])
    return dict(names=nm, beta=beta, se=se, t=tstat, p=p, r2=r2,
                resid=resid, cov=cov, n=n, k=k)


# ======================================================================================
# 2. Forecast-evaluation statistics  (the design's Tables A / B)
# ======================================================================================
def diebold_mariano(loss_model: np.ndarray, loss_bench: np.ndarray, lags: int) -> dict:
    """DM test of equal predictive accuracy. d = loss_model - loss_bench.
    One-sided 'less' p is evidence the MODEL is more accurate (lower loss)."""
    d = np.asarray(loss_model, float) - np.asarray(loss_bench, float)
    res = nw_mean_test(d, lags, alternative="less")
    res["interpretation"] = ("model more accurate" if res["mean"] < 0 else "benchmark more accurate")
    return res


def mincer_zarnowitz(realised: np.ndarray, forecast: np.ndarray, lags: int) -> dict:
    """realised = a + g * forecast + e.  Good forecast: g>0 sig; ideally (a,g)=(0,1)."""
    r = ols_hac(realised, forecast, lags, names=["forecast"])
    a, g = r["beta"]
    return dict(alpha=a, gamma=g, se=r["se"], t=r["t"], p=r["p"], r2=r["r2"], n=r["n"])


def encompassing(realised: np.ndarray, f_model: np.ndarray, f_bench: np.ndarray,
                 lags: int) -> dict:
    """THE decisive test:  realised = a + bQ*f_model + bP*f_bench + e   (HAC SEs).

      reject H0  (model wins):  bQ significant  AND  bP -> 0 / insignificant
      keep   H0  (null):        bQ ~ 0          AND  bP significant
      both contribute:          both significant -> report incremental delta_R2 of f_model.
    The affine (a, b*) map absorbs the Q-vs-P risk-premium level/scale bias, so this is the
    fair, level-robust test.
    """
    full = ols_hac(realised, np.column_stack([f_model, f_bench]), lags,
                   names=["f_model_Q", "f_bench_P"])
    bench_only = ols_hac(realised, f_bench, lags, names=["f_bench_P"])
    model_only = ols_hac(realised, f_model, lags, names=["f_model_Q"])
    a, bQ, bP = full["beta"]
    delta_r2_model = full["r2"] - bench_only["r2"]      # info f_model adds over f_bench alone
    verdict = _encompass_verdict(full["p"][1], full["p"][2], bQ, bP)
    return dict(a=a, bQ=bQ, bP=bP, se=full["se"], t=full["t"], p=full["p"],
                r2_full=full["r2"], r2_bench_only=bench_only["r2"],
                r2_model_only=model_only["r2"], delta_r2_model=delta_r2_model,
                n=full["n"], verdict=verdict)


def _encompass_verdict(p_bQ, p_bP, bQ, bP, alpha=0.05) -> str:
    q_sig = (p_bQ < alpha) and (bQ > 0)
    p_sig = (p_bP < alpha)
    if q_sig and not p_sig:  return "MODEL ENCOMPASSES (reject H0)"
    if p_sig and not q_sig:  return "NULL HOLDS: model adds nothing (fail to reject H0)"
    if q_sig and p_sig:      return "BOTH CONTRIBUTE: model adds incremental info"
    return "NEITHER significant (inconclusive)"


def affine_bias_correct(realised_train, fc_train, fc_test, lags: int = 0) -> np.ndarray:
    """Fit realised_train = a + b*fc_train on TRAIN only; apply to TEST. Absorbs the constant
    Q-vs-P risk premium and scale bias without any look-ahead."""
    fc_train = np.asarray(fc_train, float)
    fc_test = np.asarray(fc_test, float)
    yt = np.asarray(realised_train, float)
    if np.var(fc_train) < 1e-14:                        # constant forecaster -> best constant
        return np.full_like(fc_test, yt.mean())
    r = ols_hac(yt, fc_train, lags, names=["fc"])
    a, b = r["beta"]
    return a + b * fc_test


# ======================================================================================
# 3. Panel, forecasters, and the OOS driver
# ======================================================================================
@dataclass
class Panel:
    """Daily aligned series. One entry per trading day, sorted by date.

    dates:   (n,) array-like of comparable date keys (ints or np.datetime64)
    spot:    (n,) underlying level S_u
    atmiv:   dict tenor(years) -> (n,) constant-maturity ATM implied vol series
    skew:    dict tenor(years) -> (n,) observed ATM skew  d sigma / d k   at each day (for SSR units)
    xsec:    optional dict date_key -> opaque option cross-section handle for model calibration
    """
    dates: np.ndarray
    spot: np.ndarray
    atmiv: dict
    skew: dict = field(default_factory=dict)
    xsec: Optional[dict] = None

    def logret(self) -> np.ndarray:
        s = np.asarray(self.spot, float)
        r = np.full(s.size, np.nan)
        r[1:] = np.log(s[1:] / s[:-1])
        return r


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    x, y = x[m], y[m]
    vx = np.var(x)
    return np.cov(x, y, bias=True)[0, 1] / vx if vx > 0 else np.nan


def realised_beta(panel: Panel, i0: int, i1: int, tenor: float) -> float:
    """OLS slope of dATM-IV on log-return over day indices [i0, i1) (half-open).
    beta = Cov(dsigma_ATM, r) / Var(r).  Used both for the TRAILING benchmark ([t-w,t))
    and the FORWARD target ((t, t+h])."""
    sig = np.asarray(panel.atmiv[tenor], float)
    r = panel.logret()
    dsig = np.full(sig.size, np.nan)
    dsig[1:] = sig[1:] - sig[:-1]
    idx = np.arange(max(i0, 1), i1)
    return _ols_slope(dsig[idx], r[idx])


@dataclass
class RealisedSSRForecaster:
    """Backward-looking benchmark F^P: trailing realised beta (optionally / skew -> SSR)."""
    window: int = 63                       # trailing trading days
    as_ssr: bool = True                    # divide by observed skew -> SSR units (matches ssr_2f)

    def __call__(self, panel: Panel, t: int, tenor: float) -> float:
        b = realised_beta(panel, t - self.window, t + 1, tenor)   # [t-w, t]
        if not self.as_ssr:
            return b
        sk = np.asarray(panel.skew[tenor], float)[t] if tenor in panel.skew else np.nan
        return b / sk if (sk and np.isfinite(sk) and abs(sk) > 1e-9) else np.nan


def forward_target(panel: Panel, t: int, horizon: int, tenor: float,
                   as_ssr: bool = True) -> float:
    """The realisation being forecast: realised comovement over (t, t+horizon]."""
    b = realised_beta(panel, t + 1, t + 1 + horizon, tenor)
    if not as_ssr:
        return b
    sk = np.asarray(panel.skew[tenor], float)[t] if tenor in panel.skew else np.nan
    return b / sk if (sk and np.isfinite(sk) and abs(sk) > 1e-9) else np.nan


# Model hook -------------------------------------------------------------------
ModelForecaster = Callable[[Panel, int, float], float]   # (panel, t_index, tenor) -> SSR^Q_t


def model_forecaster_stub(panel: Panel, t: int, tenor: float) -> float:
    raise NotImplementedError(
        "Wire ModelForecaster to the POC: calibrate_2f.kernel + least_squares on the t "
        "cross-section (panel.xsec[panel.dates[t]]), then discslv_2f.ssr_2f(K, n=round(tenor/K.dt), "
        "nk=16)[0]. See module docstring."
    )


# Persistence / constant forecasters ------------------------------------------
def persistence_forecaster(horizon: int, as_ssr: bool = True) -> ModelForecaster:
    def f(panel: Panel, t: int, tenor: float) -> float:
        return forward_target(panel, t - horizon, horizon, tenor, as_ssr)   # last realised window
    return f


# ======================================================================================
# 4. Build the OOS forecast frame (with look-ahead asserts + train/test split)
# ======================================================================================
@dataclass
class ForecastFrame:
    tenor: float
    horizon: int
    dates: np.ndarray
    target: np.ndarray
    f_model: np.ndarray
    f_bench: np.ndarray
    extra: dict                    # e.g. persistence, const, spot indices for hedging
    train_mask: np.ndarray
    test_mask: np.ndarray


def build_frame(panel: Panel, tenor: float, horizon: int,
                model: ModelForecaster, bench: RealisedSSRForecaster,
                train_frac: float = 0.5, min_trail: Optional[int] = None,
                as_ssr: bool = True, extra_forecasters: Optional[dict] = None) -> ForecastFrame:
    """Assemble aligned (target, forecasts) over all dates where every input is available.
    Look-ahead audit: model/bench at t use only info <= t; target uses (t, t+h]."""
    n = len(panel.dates)
    min_trail = bench.window if min_trail is None else min_trail
    rows = []
    for t in range(min_trail, n - horizon):
        tgt = forward_target(panel, t, horizon, tenor, as_ssr)
        fm = model(panel, t, tenor)
        fb = bench(panel, t, tenor)
        row = dict(t=t, date=panel.dates[t], target=tgt, f_model=fm, f_bench=fb)
        if extra_forecasters:
            for name, fc in extra_forecasters.items():
                row[name] = fc(panel, t, tenor)
        rows.append(row)
    rows = [r for r in rows if all(np.isfinite(v) for k, v in r.items()
                                   if k not in ("date",) and np.isscalar(v))]
    if not rows:
        raise ValueError("No complete rows -- check tenor keys, window, and horizon vs sample length.")
    t_idx = np.array([r["t"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    target = np.array([r["target"] for r in rows], float)
    f_model = np.array([r["f_model"] for r in rows], float)
    f_bench = np.array([r["f_bench"] for r in rows], float)
    extra = {}
    if extra_forecasters:
        for name in extra_forecasters:
            extra[name] = np.array([r[name] for r in rows], float)
    extra["_t_index"] = t_idx
    # LOOK-AHEAD AUDIT
    assert np.all(np.diff(t_idx) >= 0), "dates not sorted"
    n_rows = len(rows)
    cut = int(train_frac * n_rows)
    train_mask = np.zeros(n_rows, bool); train_mask[:cut] = True
    test_mask = ~train_mask
    return ForecastFrame(tenor, horizon, dates, target, f_model, f_bench, extra,
                         train_mask, test_mask)


# ======================================================================================
# 5. Evaluate (statistical) and hedging replay (economic)
# ======================================================================================
def evaluate(frame: ForecastFrame, hac_lags: Optional[int] = None,
             bias_correct: bool = True) -> dict:
    """Run MZ, DM, and the encompassing test on the OOS (test) block.
    Bias-correction params are fit on TRAIN only (no look-ahead)."""
    lags = frame.horizon - 1 if hac_lags is None else hac_lags
    tr, te = frame.train_mask, frame.test_mask
    y_tr, y_te = frame.target[tr], frame.target[te]

    def prep(fc):
        if not bias_correct:
            return fc[te]
        return affine_bias_correct(y_tr, fc[tr], fc[te])
    fm = prep(frame.f_model)
    fb = prep(frame.f_bench)

    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def mae(a, b):  return float(np.mean(np.abs(a - b)))

    out = dict(tenor=frame.tenor, horizon=frame.horizon, n_test=int(te.sum()),
               hac_lags=lags, bias_corrected=bias_correct)
    out["rmse"] = dict(model=rmse(fm, y_te), bench=rmse(fb, y_te))
    out["mae"] = dict(model=mae(fm, y_te), bench=mae(fb, y_te))
    out["dm"] = diebold_mariano((fm - y_te) ** 2, (fb - y_te) ** 2, lags)
    out["mz_model"] = mincer_zarnowitz(y_te, fm, lags)
    out["mz_bench"] = mincer_zarnowitz(y_te, fb, lags)
    out["encompassing"] = encompassing(y_te, fm, fb, lags)
    # extra forecasters (persistence, const, ...) -> DM vs their loss
    for name, fc in frame.extra.items():
        if name.startswith("_"):
            continue
        f = prep(fc)
        out.setdefault("extra_dm", {})[name] = diebold_mariano((fm - y_te) ** 2, (f - y_te) ** 2, lags)
    return out


def hedging_replay(panel: Panel, frame: ForecastFrame, tenor: float,
                   forecasts: dict, hac_lags: Optional[int] = None,
                   baseline: str = "f_bench") -> dict:
    """Economic test: for each forecaster's R_t = forecast_t (SSR units), the SSR-consistent
    delta rolls the ATM vol by R_t * skew * dlogS each day. The unhedged smile-roll P&L per
    window is  Vega * (dsigma_ATM - R_t*skew*dlogS) ; its VARIANCE is the hedging error the
    delta removes. We proxy per-window hedging variance by the residual variance of
    (dsigma_ATM - R*skew*r) over (t, t+h]; lower is better.

    Ratios are reported RELATIVE TO `baseline` (default 'f_bench' = the feasible minimum-variance /
    Hull-White (2017) delta, the real industry standard), NOT vs Black. Black (R=0, naive BS delta)
    and sticky-delta (R=-1) are kept only as reference rows; the ex-post oracle R* is the (infeasible)
    floor. The DM test compares f_model vs `baseline` on per-window hedging variance.
    """
    lags = frame.horizon - 1 if hac_lags is None else hac_lags
    sig = np.asarray(panel.atmiv[tenor], float)
    sk = np.asarray(panel.skew[tenor], float) if tenor in panel.skew else np.ones(sig.size)
    r = panel.logret()
    dsig = np.full(sig.size, np.nan); dsig[1:] = sig[1:] - sig[:-1]
    t_idx = frame.extra["_t_index"]
    te = frame.test_mask

    def window_resid_var(R_series):
        v = []
        for j, t in enumerate(t_idx):
            if not te[j]:
                continue
            idx = np.arange(t + 1, t + 1 + frame.horizon)
            idx = idx[idx < sig.size]
            roll = dsig[idx] - R_series[j] * sk[t] * r[idx]      # residual smile-roll
            roll = roll[np.isfinite(roll)]
            if roll.size:
                v.append(np.mean(roll ** 2))
        return np.array(v)

    # oracle R* per window (ex-post slope of dsig on skew*r) and Black R=0
    def oracle_R():
        R = np.full(len(t_idx), np.nan)
        for j, t in enumerate(t_idx):
            idx = np.arange(t + 1, t + 1 + frame.horizon); idx = idx[idx < sig.size]
            b = _ols_slope(dsig[idx], sk[t] * r[idx])
            R[j] = b
        return R

    series = dict(forecasts)
    series["oracle"] = oracle_R()                              # infeasible ex-post floor
    series["black"] = np.zeros(len(t_idx))                     # naive BS delta (reference only)
    series["sticky_delta"] = -np.ones(len(t_idx))             # R=-1 desk convention (reference)
    var = {name: window_resid_var(R) for name, R in series.items()}
    base_mean = float(np.mean(var[baseline])) if baseline in var and var[baseline].size else np.nan
    # ratios RELATIVE TO the industry MV baseline (not Black); also keep the raw /Black for reference
    black_mean = float(np.mean(var["black"])) if var["black"].size else np.nan
    ratios = {name: float(np.mean(v)) / base_mean if base_mean else np.nan
              for name, v in var.items()}
    ratios_vs_black = {name: float(np.mean(v)) / black_mean if black_mean else np.nan
                       for name, v in var.items()}
    # DM: model vs the industry baseline on per-window hedging variance
    dm = None
    if "f_model" in var and baseline in var and var["f_model"].size == var[baseline].size:
        dm = diebold_mariano(var["f_model"], var[baseline], lags)
    return dict(mean_resid_var=ratios, mean_resid_var_vs_black=ratios_vs_black,
                baseline=baseline, dm_model_vs_baseline=dm, n_windows=int(te.sum()))


# ======================================================================================
# 6. Reporting
# ======================================================================================
def format_report(ev: dict, hr: Optional[dict] = None) -> str:
    L = []
    L.append(f"OOS SSR-forecast evaluation  |  tenor={ev['tenor']}  horizon={ev['horizon']}d  "
             f"n_test={ev['n_test']}  HAC_lags={ev['hac_lags']}  "
             f"bias_corrected={ev['bias_corrected']}")
    L.append("-" * 78)
    L.append(f"[B] Accuracy      RMSE            MAE")
    L.append(f"    model(Q)      {ev['rmse']['model']:.5f}        {ev['mae']['model']:.5f}")
    L.append(f"    bench(P)      {ev['rmse']['bench']:.5f}        {ev['mae']['bench']:.5f}")
    dm = ev["dm"]
    L.append(f"    Diebold-Mariano (model vs bench, H1: model better): "
             f"t={dm['t']:.2f}  p={dm['p']:.3f}  -> {dm['interpretation']}")
    mzm, mzb = ev["mz_model"], ev["mz_bench"]
    L.append(f"    Mincer-Zarnowitz gamma  model={mzm['gamma']:.3f} (p={mzm['p'][1]:.3f}, R2={mzm['r2']:.3f})"
             f"   bench={mzb['gamma']:.3f} (p={mzb['p'][1]:.3f}, R2={mzb['r2']:.3f})")
    en = ev["encompassing"]
    L.append("-" * 78)
    L.append(f"[A] Encompassing  realised = a + bQ*model + bP*bench   (the decisive test)")
    L.append(f"    bQ = {en['bQ']:+.3f}  (p={en['p'][1]:.3f})     bP = {en['bP']:+.3f}  (p={en['p'][2]:.3f})")
    L.append(f"    R2: full={en['r2_full']:.3f}  bench-only={en['r2_bench_only']:.3f}  "
             f"model-only={en['r2_model_only']:.3f}  incremental(model)={en['delta_r2_model']:+.3f}")
    L.append(f"    VERDICT: {en['verdict']}")
    if "extra_dm" in ev:
        L.append("    DM vs other benchmarks (H1: model better): " +
                 "  ".join(f"{k}: p={v['p']:.3f}" for k, v in ev["extra_dm"].items()))
    if hr is not None:
        L.append("-" * 78)
        base = hr.get("baseline", "f_bench")
        L.append(f"[C] Hedging replay: mean per-window hedging variance / {base} "
                 f"(the industry min-variance / Hull-White delta); [/Black] in brackets for reference")
        vb = hr.get("mean_resid_var_vs_black", {})
        for name, ratio in sorted(hr["mean_resid_var"].items(), key=lambda kv: kv[1]):
            tag = "  <- industry baseline" if name == base else ("  <- infeasible floor" if name == "oracle" else "")
            L.append(f"    {name:13s} {ratio:.3f}   [/Black {vb.get(name, float('nan')):.3f}]{tag}")
        if hr["dm_model_vs_baseline"] is not None:
            d = hr["dm_model_vs_baseline"]
            L.append(f"    DM hedging-var (model vs {base}, H1: model lower): "
                     f"t={d['t']:.2f}  p={d['p']:.3f}  -> {d['interpretation']}")
    return "\n".join(L)


if __name__ == "__main__":
    print(__doc__)
    print("This module is the engine. Run scripts/run_demo.py for a synthetic self-test.")

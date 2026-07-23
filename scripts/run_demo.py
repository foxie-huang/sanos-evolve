"""
Synthetic self-test + output-shape demo for the SSR-forecast harness.
=====================================================================

NO REAL DATA. This builds a synthetic ground-truth process where the answer is KNOWN, to prove
the statistics behave and to show the exact tables the paper will produce:

  * "informative" world: a latent, persistent comovement state theta_t drives realised comovement;
    the model forecast F^Q sees theta_t contemporaneously (a LEADING signal), while the cheap
    benchmark F^P is a trailing OLS estimate (LAGGING). The harness SHOULD find that F^Q
    encompasses F^P (reject H0).
  * "null" world: same panel, but F^Q is pure noise. The harness SHOULD find the NULL holds
    (F^Q adds nothing; the lagged-but-real F^P wins).

If both expectations hold, the engine's encompassing / DM / MZ logic is trustworthy for the real
study.  Run:  python3 run_demo.py
"""
import numpy as np
import ssr_forecast_eval as E

RNG = np.random.default_rng(20260707)
N_DAYS   = 2200          # ~8.7y of trading days (cf. ORATS 2015-2023)
TENOR    = 1.0 / 12      # 1-month tenor (years); label only in this synthetic demo
HORIZON  = 21            # forward window (trading days ~ 1 month)
TRAIL_W  = 63            # trailing window for the cheap benchmark (~3 months)
SD_R     = 0.010         # daily log-return sd
SD_EPS   = 0.004         # idiosyncratic daily ATM-vol-change noise
SD_QMOD  = 0.15          # model-signal noise (informative world): F^Q = theta + N(0, SD_QMOD)


def make_world(seed):
    """Return a Panel plus the latent theta and a model-forecast array for the informative world."""
    rng = np.random.default_rng(seed)
    # latent persistent comovement state theta_t (AR(1)), beta units
    mu, phi, sd_eta = 1.5, 0.99, 0.05
    theta = np.empty(N_DAYS); theta[0] = mu
    for t in range(1, N_DAYS):
        theta[t] = mu + phi * (theta[t - 1] - mu) + sd_eta * rng.standard_normal()
    # daily returns and ATM-vol changes: dsigma_u = beta_u * r_u + eps_u,  beta_u = theta_u
    r = SD_R * rng.standard_normal(N_DAYS)
    dsig = theta * r + SD_EPS * rng.standard_normal(N_DAYS)
    spot = 100.0 * np.exp(np.cumsum(r))
    # ATM-IV LEVEL as a running sum of the daily changes; the harness only ever uses diffs,
    # so no clipping (clipping the level would corrupt diff(atmiv) and destroy the dsigma signal).
    atmiv = 0.15 + np.cumsum(dsig)
    skew = np.full(N_DAYS, 1.0)                             # unit skew -> SSR == beta units here
    panel = E.Panel(dates=np.arange(N_DAYS), spot=spot,
                    atmiv={TENOR: atmiv}, skew={TENOR: skew})
    # model forecast (informative): contemporaneous, noisy view of theta
    f_model_info = theta + SD_QMOD * rng.standard_normal(N_DAYS)
    return panel, theta, f_model_info


def array_forecaster(arr):
    """Wrap a precomputed per-date array as a ModelForecaster(panel, t, tenor)."""
    def f(panel, t, tenor):
        return float(arr[t]) if 0 <= t < arr.size else np.nan
    return f


def run_world(panel, f_model_arr, label, as_ssr=False):
    bench = E.RealisedSSRForecaster(window=TRAIL_W, as_ssr=as_ssr)
    extras = {
        "persistence": E.persistence_forecaster(HORIZON, as_ssr=as_ssr),
        "const": (lambda p, t, ten: 1.5),                  # fixed R (cf. paper's structural 1.6)
    }
    frame = E.build_frame(panel, TENOR, HORIZON,
                          model=array_forecaster(f_model_arr), bench=bench,
                          train_frac=0.5, as_ssr=as_ssr, extra_forecasters=extras)
    ev = E.evaluate(frame, bias_correct=True)
    # hedging replay: feed each forecaster's per-row R series (test rows handled inside)
    fc_series = {"f_model": frame.f_model, "f_bench": frame.f_bench,
                 "persistence": frame.extra["persistence"], "const": frame.extra["const"]}
    hr = E.hedging_replay(panel, frame, TENOR, fc_series)
    print("\n" + "=" * 78)
    print(f"WORLD: {label}")
    print("=" * 78)
    print(E.format_report(ev, hr))
    return ev, hr


def main():
    panel, theta, f_info = make_world(seed=1)

    # ---- World 1: informative model (F^Q leads) -> expect MODEL ENCOMPASSES -----------
    ev1, _ = run_world(panel, f_info, "informative model (F^Q = theta + noise, leading)")

    # ---- World 2: null model (F^Q = pure noise) -> expect NULL HOLDS -------------------
    f_null = 1.5 + np.std(theta) * RNG.standard_normal(theta.size)
    ev2, _ = run_world(panel, f_null, "null model (F^Q = pure noise)")

    # ---- self-test assertions ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("SELF-TEST")
    print("=" * 78)
    v1, v2 = ev1["encompassing"], ev2["encompassing"]
    ok1 = (v1["p"][1] < 0.05 and v1["bQ"] > 0)                        # model info significant
    ok2 = (v2["p"][1] > 0.05)                                          # null model not significant
    ok3 = ev1["dm"]["p"] < 0.10                                        # DM favours model in world 1
    print(f"  world1 bQ p={v1['p'][1]:.3f} (want <0.05, sign +)   -> {'PASS' if ok1 else 'FAIL'}")
    print(f"  world1 DM p={ev1['dm']['p']:.3f} (want <0.10)          -> {'PASS' if ok3 else 'FAIL'}")
    print(f"  world2 bQ p={v2['p'][1]:.3f} (want >0.05)            -> {'PASS' if ok2 else 'FAIL'}")
    print(f"  world1 incremental R2(model) = {v1['delta_r2_model']:+.3f}")
    print(f"  world2 incremental R2(model) = {v2['delta_r2_model']:+.3f}")
    allok = ok1 and ok2 and ok3
    print(f"\n  {'*** SELF-TEST PASSED ***' if allok else '### SELF-TEST FAILED ###'}  "
          f"(harness distinguishes an informative Q signal from noise)")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

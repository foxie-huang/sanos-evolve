# SSR-forecast test — scripts

Reference implementation for the out-of-sample SSR-forecast test
(design: [`../ssr_forecast_test_design.md`](../ssr_forecast_test_design.md)).
Addresses review Concern #1: does the model's forward-looking option-implied SSR forecast
next-period realised comovement better than a cheap backward-looking realised estimate?

| file | what it is |
|---|---|
| `ssr_forecast_eval.py` | The engine (pure, data-source-agnostic): Newey–West HAC OLS, Diebold–Mariano, Mincer–Zarnowitz, **forecast-encompassing** (the decisive test), train-only affine bias-correction, hedging-variance replay, look-ahead asserts. |
| `run_demo.py` | Synthetic self-test — **no real data**. Builds a known-answer world (Q leads P → model should win) and a null world (Q = noise → model should lose), and asserts the statistics behave. Exit 0 = pass. |
| `wire_orats.py` | **Real-data run.** Builds the panel from `v2/data/empirical_ssr.date_row` (ORATS SPX EOD), uses the Doeff–Kamal skew-decay SSR as the option-implied F^Q (the honest, non-circular ℚ signal — see findings), runs the harness, caches to `.orats_cache/`, saves `results_*.json`. `python3 wire_orats.py 2015 2019`. Findings: `../ssr_forecast_findings.md`. |
| `hullwhite.py` | **Faithful industry baseline.** The cross-strike **Hull–White (2017)** minimum-variance delta: builds the constant-delta 1m vol surface from ORATS and estimates ∂E[Δσ]/∂S ≈ (a+bδ+cδ²)/(S√τ) by a trailing pooled regression across strikes. Re-runs the test with it as the baseline. `python3 hullwhite.py 2015 2019 63`. Confirms the `wire_orats.py` result (98.8% correlated with the simple ATM MV delta). |
| `wire_poc.py` | Fill-in template wiring F^Q to the *full* POC model (`v2/poc/calibrate_2f.kernel` + `discslv_2f.ssr_2f`) — a heavier variant of `wire_orats.py`. Not runnable until the two TODOs are filled. |

## Run the self-test
```sh
cd scripts && python3 run_demo.py          # prints Tables A/B/C, then PASS/FAIL
```
Requires `numpy` (and `scipy` if present; falls back to a normal approx otherwise).

## Wire to real data (three hooks)
1. **`load_orats_panel()`** in `wire_poc.py` → build a daily `Panel` (spot, constant-maturity ATM
   IV + skew per tenor; stash each date's option cross-section in `Panel.xsec`).
2. **`PocModelForecaster`** → calibrate SANOS-Evolve at each date `t` (`calibrate_2f`, look-ahead-safe)
   and read `ssr_2f(K, n=round(T/K.dt))[0]`. Cache θ by date (~60–120 s/date).
3. **`RealisedSSRForecaster`** (already implemented) → trailing OLS of ΔATM-IV on log-return.

Then run per tenor T ∈ {1w, 1m, 3m} and report Tables A/B/C. Pre-registered decision rule and
confounds are in the design doc (§7–§8). The engine is unit-agnostic — keep target and forecast in
the same units (SSR: model native vs realised β/skew; the harness default).

## What a result means
- **Encompassing bQ significant (+):** the option-implied SSR carries genuine forward-looking
  information → supports the paper's claim.
- **bQ ≈ 0, bP significant:** null holds → the hedging headline is the minimum-variance-delta effect;
  report that honestly (the model's other contributions stand regardless).

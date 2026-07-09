# SSR-forecast test — REAL-DATA findings (SPX EOD)

**Run:** `scripts/wire_orats.py` (+ `hullwhite.py`) on ORATS SPX EOD 2015–2023. Design:
`ssr_forecast_test_design.md`. Target = 1-month **forward realised** spot–vol comovement (SSR units);
horizon 21d; trailing window 63d (swept); HAC lags 20; 50/50 train/test, train-only bias-correction.
Baseline = the **minimum-variance / Hull–White (2017) delta** (not Black). F^Q = option-implied
**Doeff–Kamal skew-decay SSR** (H+3/2 from |skew(τ)|~τ^{H−1/2}); F^P = trailing realised SSR = the MV delta.

---

## ⚠️ FIRST, A DATA BUG (found while analysing the "grind") — affects the paper's real-data results

**ORATS `stockPrice` is corrupt in 2022–2023.** It varies across strikes within a day, with bad prints
up to +20% (e.g. 2023-03-06 ranges 4050→4685 for a spot really ~4050; the clean **parity-forward F**
that day is 4050.7). The standard pipeline (`v2/data/empirical_ssr.date_row` → `orats_loader`) picks one
row's `stockPrice`, so on those days it grabs a bad value → spurious ±20% one-day return spikes.

Consequences of the bad spot (returns only — the mid-IV smile, hence ATM vol/skew, is clean):

| | corrupt spot | **clean spot** (median stockPrice) | reality |
|---|---|---|---|
| 2022 realised vol | 0.60 | **0.24** | ~24% ✓ |
| 2023 realised vol | 0.55 | **0.13** | ~13% ✓ (low-vol year) |
| 2022 realised SSR | 0.22 | **1.30** | normal |
| 2023 realised SSR | 0.10 | **1.25** | normal |

**The "2023 grind: realised SSR → 0.1, ℙ/ℚ divergence, out of scope" premise is ~90% this data bug.**
The paper's realised-SSR fit *targets* for 2022–23 come through the same corrupt path, so the reported
"34.5% SSR RMS, degenerate grind" is very likely mostly the bug, not economics. **Fix:** use the median
`stockPrice` (or the parity-forward F) for the spot/return series — one-line change in
`empirical_ssr.date_row` / `orats_loader`. `scripts/wire_orats.clean_row` and `hullwhite._day_features`
now do this.

**Clean per-year picture** (realised SSR is normal every year; only a *mild* genuine 2022–23 effect):

| yr | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|---|---|
| realised vol | .15 | .13 | .07 | .17 | .12 | .31 | .13 | **.24** | **.13** |
| realised SSR | 1.67 | 1.55 | 1.45 | 1.78 | 1.65 | 1.50 | 1.82 | **1.30** | **1.25** |
| option-implied (DK) | 1.63 | 1.67 | 1.66 | 1.74 | 1.69 | 1.67 | 1.64 | 1.76 | 1.84 |

2022–23 realised SSR (~1.27) is modestly *below* the 2015–21 norm (~1.6) with a slightly steeper
option-implied skew, so the ℙ/ℚ gap widens to ~0.5 (vs ~0.1) — a real but small effect, not a collapse.

---

## Test results (clean spot)

| | 2015–2019 | 2015–2023 (incl. former "grind") |
|---|---|---|
| **Forecasting** — corr(F^Q, fwd realised) | +0.15 | −0.06 (≈0) |
| corr(F^P, fwd realised) | −0.11 | +0.02 |
| encompassing verdict | inconclusive | inconclusive |
| **Hedging ÷ MV/Hull–White delta (=1.00)** — F^Q | **0.76** | **0.89** |
| — const R=1.5 | 0.80 | 0.85 |
| — Black (reference) | 2.66 | 2.49 |
| — oracle floor | 0.53 | 0.65 |
| DM (F^Q vs MV, H1 model lower) | p=0.023 | p=0.155 |

**The former grind blow-up is gone.** With clean spot, F^Q hedges ~11% *better* than the MV/Hull–White
delta over the full 2015–2023 sample (was 4× *worse* with corrupt spot); no regime reversal.

---

## Conclusions (corrected)

1. **Forecasting: the honest NULL, robust to the data fix.** Neither the option-implied SSR nor the
   trailing estimate forecasts next-period realised comovement — ~0 correlation, encompassing
   inconclusive, in both windows and both corrupt/clean. Forward comovement is essentially
   unforecastable at this horizon; the option-implied SSR is near-constant (~1.7, sd 0.09) and adds no
   information over the cheap estimate.

2. **Hedging: a modest, stability-driven edge over the industry MV delta — now across the whole sample.**
   The stable SSR delta beats the MV/Hull–White delta by ~11–24% (DM p=0.02 in 2015–19, p=0.16 over
   2015–23; robust to the MV window 42–252d and to the cross-strike Hull–White form — 98.8–99.4%
   correlated with the simple ATM MV). **`const ≈ f_model` throughout, so the edge is stability, not
   option-implied prediction** — equity-index comovement mean-reverts, so any trailing MV estimate
   chases noise while a stable delta does not.

3. **There is no "grind ℙ/ℚ divergence."** It was a spot-data artifact. The mild genuine 2022–23 effect
   (realised SSR ~1.27, ℙ/ℚ gap ~0.5) is well within the model's normal operating range.

---

## What this means for the paper (Concern #1)

- **Quote the hedge edge vs the minimum-variance / Hull–White delta (~11–24%), not "27–47% below Black."**
  Black is a straw man; against the real baseline the edge is modest and stability-driven.
- **The edge is stability, not forward-looking prediction** (encompassing null; `const` matches `f_model`).
- **Re-examine the "grind out of scope" claim in the paper.** It rests on a corrupt realised-SSR target;
  the clean 2022–23 realised SSR is normal (~1.25). A re-fit with clean targets would likely fit the
  grind fine — worth checking before the paper leans on the ℙ/ℚ-divergence narrative.
- **Honest §8 sentence** (to replace "listed below"): *Against the minimum-variance/Hull–White delta,
  the SSR-consistent hedge lowers variance ~11–24% (robust to the MV estimation window and the
  cross-strike form). The improvement reflects the stability of a fixed/option-anchored SSR against a
  noisy trailing estimate, not forward-looking prediction: the option-implied SSR is near-constant and
  does not encompass the trailing estimate out-of-sample.*

---

## Caveats

1. **RMSE / DM-accuracy is misleading** for a near-constant forecaster; trust the correlation signs, the
   encompassing verdict, and the hedging replay.
2. **The clean spot is `median(stockPrice)`** (robust to the tail bad prints); the parity-forward F gives
   the same. A production fix belongs in the shared pipeline (`empirical_ssr`/`orats_loader`).
3. F^Q is the lightweight skew-decay proxy (full-model F^Q via `wire_poc.py` is heavier; same answer
   since on SPX EOD the model SSR is anchored to the skew-decay when no ℚ SSR observable exists).
4. Single underlying (SPX), EOD, 1-month tenor/horizon.

## Reproduce
```sh
cd scripts && python3 wire_orats.py 2015 2019        # clean spot (median stockPrice)
              python3 wire_orats.py 2015 2023         # incl. former "grind" — now normal
              python3 hullwhite.py 2015 2023 63       # faithful cross-strike Hull-White baseline
```

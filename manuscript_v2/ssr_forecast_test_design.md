# Out-of-sample SSR-forecast test — design

**Status:** design + reference harness (`scripts/`). Not yet run on real data (needs the POC
calibration engine wired to the hooks in `scripts/ssr_forecast_eval.py`).
**Addresses:** review Concern #1 — the hedging headline (27–47% vs Black) is the *expected*
minimum-variance-delta effect; this test isolates whether the model's forward-looking,
option-implied SSR earns its keep over a cheap backward-looking realised estimate. It is the
"decisive head-to-head, listed below" that §8 of `disc_SLV_v2.tex` now promises.

---

## 1. The claim under test

The paper's delta family is $\Delta(R)=\Delta_{\mathrm{BS}}+\mathrm{Vega}\cdot R\,\mathcal S/S$.
Hedging at $R=$ the realised SSR is, by construction, the minimum-variance delta, so beating the
Black delta ($R=0$) is a textbook result and **not** evidence for SANOS-Evolve specifically.

The model's actual, testable value proposition is narrower and sharper:

> **H₁ (the model's claim).** The time-$t$ *option-implied* SSR, $\mathrm{SSR}^{\mathbb Q}_t$,
> calibrated from the time-$t$ option cross-section, forecasts the *next-period realised*
> spot–vol comovement better — or at least with genuine incremental information — than a cheap
> backward-looking estimate $\mathrm{SSR}^{\mathbb P}_{t}$ built from a trailing window of
> realised data.

> **H₀ (the honest null).** $\mathrm{SSR}^{\mathbb Q}_t$ adds no incremental predictive content
> for realised comovement beyond $\mathrm{SSR}^{\mathbb P}_{t}$. If H₀ is not rejected, the
> paper's hedging headline is the minimum-variance-delta effect and nothing more; the
> forward-looking machinery is not earning its keep for hedging/forecasting.

The test must be **capable of cleanly returning H₀** — that is the point. It is not a strawman:
there is an a-priori mechanism by which the model *could* win — the option-implied SSR is
forward-looking and can re-price spot–vol comovement *ahead* of a regime shift that a trailing
realised estimate only catches with a lag. The test decides whether that mechanism actually pays.

---

## 2. The object being forecast

Work at a fixed horizon/tenor $T$ (report several; see §6). Define the **spot–vol comovement
slope** over a forward window $[t,t+h]$ from realised daily data:
$$
\beta^{\mathrm{real}}_{[t,t+h]}
= \frac{\mathrm{Cov}\!\big(\Delta\widehat\sigma_{\mathrm{ATM},T},\ r\big)}{\mathrm{Var}(r)}
\quad\text{over days } u\in[t,t+h],
$$
where $r_u=\Delta\log S_u$ is the daily log-return and $\widehat\sigma_{\mathrm{ATM},T,u}$ is the
constant-maturity-$T$ ATM implied vol on day $u$. The realised SSR is
$\mathrm{SSR}^{\mathrm{real}}_{[t,t+h]}=\beta^{\mathrm{real}}_{[t,t+h]}/\mathcal S_{T}$ with
$\mathcal S_T$ the ATM skew.

**Forecast the slope $\beta$, not the ratio, as the primary target.** The skew $\mathcal S_T$ is
observable at $t$ from the smile and is common to the model SSR and the realised SSR (both divide
by the same $\mathcal S$). Forecasting the raw regression slope $\beta$ avoids injecting
skew-estimation noise into both sides and keeps the comparison apples-to-apples. Report the SSR
(= $\beta/\mathcal S$) as a secondary, more interpretable target.

**The $\mathbb Q$-vs-$\mathbb P$ wedge is real and must be handled, not ignored.**
$\mathrm{SSR}^{\mathbb Q}$ is risk-neutral; $\mathrm{SSR}^{\mathrm{real}}$ is physical. There is an
SSR/comovement risk premium, so the *levels* need not match — the paper says as much (the grind
regime has realised SSR → 0.1 while the Q-model is pinned near $H+\tfrac32$). A raw level
comparison is therefore *unfair to the model*, which never claimed to predict $\mathbb P$ levels.
The fair question is whether $\mathrm{SSR}^{\mathbb Q}_t$ carries **incremental information about
the time-variation** of realised comovement. Two devices make the test fair (§4):
1. an affine bias-correction $\beta^{\mathrm{real}}\approx a+b\,\beta^{\mathbb Q}$ fit on a
   *training* block only and applied out-of-sample (absorbs the constant risk premium and any
   scale bias);
2. the encompassing regression, whose coefficients absorb the same affine map automatically.

Report the raw level gap separately — it *is* the estimated SSR risk premium, a result of
independent interest.

---

## 3. The forecasters

All are formed using information available **at or before $t$** (no look-ahead).

| symbol | forecaster | direction | source |
|---|---|---|---|
| $F^{\mathbb Q}$ | $\mathrm{SSR}^{\mathbb Q}_t$ / $\beta^{\mathbb Q}_t$ — SANOS-Evolve calibrated to the $t$ cross-section | forward-looking, risk-neutral | `calibrate_2f` → `model_ssr(θ, T)` |
| $F^{\mathbb P}$ | trailing realised: OLS slope of $\Delta\widehat\sigma_{\mathrm{ATM}}$ on $r$ over $[t-w,t]$ | backward-looking, physical | `realized_ssr.py` on a trailing window |
| $F^{\text{persist}}$ | last realised value $\beta^{\mathrm{real}}_{[t-h,t]}$ | backward, naive | realised, previous window |
| $F^{\text{const}}$ | a fixed constant (full-sample mean $\bar\beta$; and the paper's structural $R{=}1.6$ in SSR units) | none | the paper's own replay baseline |
| $F^{\text{DK}}$ | Doeff–Kamal static-implied SSR $=H+\tfrac32$ from the skew-decay exponent at $t$ | forward, static-only | skew term-structure at $t$ |
| — refs — | $F^{\text{black}}{=}0$ (sticky-strike), $F^{\text{sticky-}\delta}{=}-1$ | — | endpoints of $\Delta(R)$ |

**Decisive pair:** $F^{\mathbb Q}$ vs $F^{\mathbb P}$. Everything else is context. $F^{\mathbb P}$
is exactly the "cheap alternative — a trailing realised-SSR delta" named in the paper.

---

## 4. Statistical evaluation (forecast accuracy)

Let $\{\,(\hat\beta^{(m)}_t,\ \beta^{\mathrm{real}}_{[t,t+h]})\,\}_{t\in\mathcal T_{\text{OOS}}}$ be
the aligned forecast/realisation pairs for method $m$ over the out-of-sample dates.

1. **Loss.** Out-of-sample RMSE and MAE of each $\hat\beta^{(m)}$ against
   $\beta^{\mathrm{real}}$, after the train-only affine bias-correction of §2. Report both
   corrected and raw.

2. **Diebold–Mariano (equal predictive accuracy).** For the loss differential
   $d_t = \ell(\hat\beta^{\mathbb Q}_t)-\ell(\hat\beta^{\mathbb P}_t)$ (squared or absolute),
   test $\mathbb E[d_t]=0$ vs one-sided $<0$ (model better). Use **HAC/Newey–West** variance with
   lag $\ge h-1$ because forward windows overlap. This is the headline accuracy statistic.

3. **Mincer–Zarnowitz (forecast efficiency).** Regress
   $\beta^{\mathrm{real}}_{[t,t+h]}=\alpha+\gamma\,\hat\beta^{(m)}_t+\varepsilon_t$; a good forecast
   has $\gamma>0$ significant, and ideally $(\alpha,\gamma)\approx(0,1)$ *after* bias-correction.
   Report $R^2$ per method (predictive $R^2$ for realised comovement).

4. **Forecast encompassing — THE decisive test.** Run
   $$
   \beta^{\mathrm{real}}_{[t,t+h]}
   = a + b_{\mathbb Q}\,\hat\beta^{\mathbb Q}_t
       + b_{\mathbb P}\,\hat\beta^{\mathbb P}_t + \varepsilon_t
   \quad(\text{HAC SEs}).
   $$
   - **Model wins (reject H₀):** $b_{\mathbb Q}$ significant *and* $b_{\mathbb P}$ driven toward
     zero / insignificant → $\mathrm{SSR}^{\mathbb Q}$ encompasses the cheap estimate.
   - **Model adds nothing (fail to reject H₀):** $b_{\mathbb Q}\approx 0$ while $b_{\mathbb P}$
     significant → the forward-looking forecast is redundant; the honest null holds.
   - **Both contribute:** both significant → the model adds *incremental* information; report the
     partial contribution ($\Delta R^2$ from adding $F^{\mathbb Q}$ to a model with $F^{\mathbb P}$).
   The affine map $(a,\ b_\cdot)$ absorbs the $\mathbb Q$–$\mathbb P$ risk premium, so this is the
   fair, level-robust test.

---

## 5. Economic evaluation (hedging) — extends the paper's Table 5 / Fig 8

The paper hedges a rolling 1-month ATM SPX option daily at a **fixed** $R=1.6$. Replace the fixed
$R$ with each **feasible time-varying forecast** $R^{(m)}_t=\hat\beta^{(m)}_t/\mathcal S_t$ and
recompute realised hedging-error variance over $[t,t+h]$.

- **The industry baseline is the minimum-variance / Hull–White (2017) delta, NOT Black.** No desk
  hedges an index with the naive Black delta ($R{=}0$); comparing to it is a straw man (any
  skew-adjusted delta beats it — this is the paper's "27–47% below Black"). The real baseline is the
  feasible **minimum-variance delta**: the empirical regression of $\Delta\sigma$ on $\Delta S$ (Hull–
  White 2017), which is exactly $F^{\mathbb P}$ here. **Normalise hedging variance to $F^{\mathbb P}$**,
  and demote Black and sticky-delta ($R{=}-1$) to reference rows.
- **Feasible comparison (the point):** hedging variance under $F^{\mathbb Q}$ vs the MV/Hull–White
  delta $F^{\mathbb P}$, via a HAC one-sided DM test on the per-window squared-hedging-error
  differential. If indistinguishable, the option-implied SSR buys no hedging improvement over the
  industry standard. **Sweep the MV-delta estimation window** (the real Hull–White uses a long,
  smooth window; a hobbled short window is not a fair baseline) — the option-implied delta must beat
  the *best* MV delta, not a noisy one.
- **Bracketing references:** the ex-post oracle $R^\star_t$ (infeasible, minimises realised hedging
  variance) as the floor; Black ($R{=}0$) and sticky-delta ($R{=}-1$) as naive reference rows.
- Keep the paper's transaction-cost treatment; report gross and net.

---

## 6. Data, sample, horizons

- **Data:** ORATS SPX EOD 2015–2023 (the existing dataset; `data_port.py` / `fetch_data.py`).
  Constant-maturity ATM IV series from the calibrated surface (consistent ATM proxy across days).
- **Tenors $T$:** 1w, 1m, 3m (the SSR is tenor-dependent; the paper's term structure runs
  $2.0\to1.4$). 1m is the primary (matches the hedging replay); report the others.
- **Forward window $h$:** align $h$ to $T$ where sensible (e.g. 1m target ↔ ~21-day forward
  window). Report sensitivity to $h$.
- **Trailing window $w$** for $F^{\mathbb P}$: a primary $w$ (e.g. 63 trading days) plus a
  robustness sweep ($w\in\{21,42,63,126\}$) — the model must beat the *best* cheap estimate, not a
  deliberately hobbled one.
- **OOS protocol:** rolling/expanding origin. Any bias-correction $(a,b)$ and any hyperparameter is
  fit on the training block only, then frozen for the OOS block. No calibration at $t$ may use
  data after $t$.

---

## 7. Pre-registered decision rule

State before running (avoids the retro-fit the whole test exists to prevent):

- **Primary (statistical):** reject H₀ iff, in the §4.4 encompassing regression on the **full
  sample**, $b_{\mathbb Q}$ is significant at 5% (HAC) with the correct sign **and** its inclusion
  raises OOS predictive $R^2$ over the $F^{\mathbb P}$-only model by a pre-set margin (e.g. ≥2 pts).
- **Corroborating:** DM test (§4.2) favours the model one-sided at 5%.
- **Economic:** hedging-variance difference (§5) favours the model one-sided at 5% HAC.
- **Full sample is primary; no regime cherry-picking.** Report the regime breakdown
  (calm/moderate/high-vol/grind) as *secondary*. The grind (realised SSR→0, below the local-vol
  floor) is a known $\mathbb P$–$\mathbb Q$ divergence; pre-commit to reporting it, not excluding
  it, and interpret a model loss there as expected, not hidden.
- **If H₀ holds:** state plainly in the paper that the hedging edge is the minimum-variance-delta
  effect and the forward-looking claim is unsupported at these horizons — and keep the model's
  *other* contributions (arbitrage-free-by-construction, closed-form/deterministic, exact SSR
  readout) which do not depend on this test.

---

## 8. Confounds and robustness

- **Overlapping forward windows** → HAC everywhere; consider non-overlapping-window robustness.
- **ATM-vol proxy noise** → use the model/SANOS constant-maturity ATM IV, not raw nearest-strike
  quotes; verify against an independent CMV series.
- **Skew normalisation** → forecast $\beta$ primarily (skew common to both); SSR secondary.
- **Regime dependence** → full-sample primary + regime cut (§7).
- **Best-case cheap benchmark** → sweep $w$; also include an AR(1)/EWMA realised-SSR forecaster so
  $F^{\mathbb P}$ is a *strong* backward-looking baseline, not a lagged strawman.
- **Look-ahead audit** → assert every feature's timestamp $\le t$; unit-test in the harness.
- **Multiple testing** across tenors → report all; Bonferroni/BH note if claiming any single tenor.
- **Sensitivity to bias-correction window** → report OOS results with and without correction.

---

## 9. Deliverables (what goes in the paper)

- **Table A** — encompassing regression: $(a,b_{\mathbb Q},b_{\mathbb P})$ with HAC SEs, OOS
  $R^2$, per tenor. *The decisive table.*
- **Table B** — OOS accuracy: RMSE/MAE and DM $p$-values, $F^{\mathbb Q}$ vs $F^{\mathbb P}$ vs
  naïve, per tenor.
- **Table C** — hedging replay with time-varying forecasts: variance vs Black, gap to oracle,
  $F^{\mathbb Q}$ vs $F^{\mathbb P}$ difference test. Supersedes/augments the current Table 5.
- **Figure** — time series of $\mathrm{SSR}^{\mathbb Q}_t$, trailing $\mathrm{SSR}^{\mathbb P}_t$,
  and forward-realised, with regimes shaded; visually shows whether Q leads P around turns.
- **One honest paragraph** in §8 reporting the outcome — win, null, or partial — replacing the
  current "listed below" promissory note.

---

## 10. Harness (`scripts/`)

- `ssr_forecast_eval.py` — the evaluation engine: forecaster protocol, HAC DM test,
  Mincer–Zarnowitz, encompassing regression, hedging-variance replay, look-ahead asserts. Pure and
  data-source-agnostic; real inputs enter through three hooks (`ModelSSRForecaster`,
  `RealisedSSRForecaster`, `load_panel`) documented at the top of the file.
- `run_demo.py` — runs the whole harness on a **synthetic** ground-truth process where the answer
  is known (Q leads P by construction, plus a null variant where Q is pure noise), to (a) prove the
  statistics behave — the encompassing test picks the informative forecaster, the null variant
  correctly fails to reject — and (b) show the exact output shape the paper tables will take. **No
  real-data numbers are produced or implied here.**
- Wiring: replace the two forecaster hooks with calls into `v2/poc/calibrate_2f.py` +
  `discslv_2f` (model) and `v2/poc/realized_ssr.py` (realised); point `load_panel` at the ORATS
  cache. See the interface notes at the top of `ssr_forecast_eval.py`.

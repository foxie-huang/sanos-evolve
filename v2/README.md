# SANOS-Evolve (disc_SLV) — v2 code roadmap

Discrete **stochastic-local-volatility** for SPX: SANOS arbitrage-free Gaussian-mixture **marginals (statics)** +
a two-timescale GM martingale **kernel (dynamics)**, fused by discrete Dupire  `σ_LV² = σ²_Dupire / E[ν|z]`.
Paper = `../disc_SLV.tex` (calibration objective `eq:objective` §850, algorithm `alg:calib` §901,
de-eventing `sec:deevent`).

- **Code:** `v2/data/` = working scripts (ORATS-integrated) · `v2/poc/` = model core + synthetic PoC.
- **Data:** `/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod/SPX-NDX-RUT-VIX_YYYY-MM-DD.json.gz`
  — EOD full chains, SPX/NDX/RUT/VIX bundled, 2010–2026. **SPX expiries: daily since ~mid-2022; M/W/F 2016–2022; Fridays earlier.**
- **Conventions that bite:**
  - Weekly step `DT = 1/52`. The SSR is **dt-dependent** (kernel transitions use κ/λ per-step, no clean OU split) → use a fixed dt convention.
  - **ORATS `dte = calendar-days + 1`** → an event's maturity is `T = ((event − snap).days + 1)/365`, NOT `.days/365` (off-by-one bracket bug).
  - Long runs: `nohup python3 X.py … > log 2>&1 & disown` (NOT the harness background — that dies on restart). Data is at the path above; token from `ORATS_TOKEN` env (never hardcode; rotate if pasted).
  - `least_squares(verbose=2)` under nohup is **block-buffered** — no progress lines until the process exits. Silence ≠ hang.
  - **Multi-date runs (pool/control/aggregates) MUST parallelize** — dates are independent and each is ~10s (SANOS-LP-bound); ~2 min parallel vs ~13 min serial. macOS/Py3.14 multiprocessing = **spawn**, so every ad-hoc `ProcessPoolExecutor` script MUST put its driver code under `if __name__ == "__main__":` — without the guard, each spawned worker re-imports the module, re-runs the pool code, and nests pools → `BrokenProcessPool` (this, NOT BLAS-fork, is the crash; a serial fallback is the wrong reaction). The `deevent_bridge_pool/control.py` drivers already have the guard. Also set `OMP_NUM_THREADS=1` (hygiene: avoids BLAS oversubscription with N workers).

---

## ROADMAP — "to do X, run Y"

| Goal | Script / call |
|---|---|
| Pull ORATS history | `orats_pull.py` |
| Clean smiles for a date | `orats_loader.load_day(path, [ticker])` → `{expiry: {T,F,DF,strike,iv,cmid,pmid,...}}` |
| SANOS arb-free marginals (a date) | `slv_wire.sanos_chain(path, ticker)` → `[(T, (W,MU,SG)), …]` |
| Realized (ℙ) SSR of a **period** | `empirical_ssr.empirical_ssr(sorted(glob(year)), ns=NS, ticker=…)` → SSR term structure |
| Observed SSR, SPX & NDX by regime | `python3 ssr_market.py [years…]` |
| **Fit θ — SSR only (parallel)** | `OOS_DATE=YYYY-MM-DD python3 calibrate_slv_exact_ts_par.py ts 12` (~13 min; prints θ + SSR err) |
| Fit θ — SSR only (serial) | `python3 calibrate_slv_exact_ts.py [dense\|low\|ts]` |
| Fit θ — SSR + VIX (joint) | `calibrate_joint.py` (numpy) · `calibrate_joint_torch.py` (MPS jacrev, faster on the full joint) |
| **Fit θ — full `eq:objective` (marginal band-loss + SSR + VoV), numpy** | `OOS_DATE=YYYY-MM-DD python3 calibrate_full.py [nw] [w_marg] [w_vov]` — the faithful `alg:calib` (true-FD Jacobian, process pool); `… calibrate_full.py test` prints block magnitudes first |
| **Fit θ — full `eq:objective`, TORCH (jacfwd AD + multi-start)** | `OOS_DATE=YYYY-MM-DD python3 calibrate_full_torch.py [verify\|fit\|fit1] [w_marg] [w_vov]` — forward 18× faster, `jacfwd` 5.5× faster than `jacrev`; **`fit` = multi-start** (the correct de-event θ); `verify` checks it vs numpy to ~1e-5. `DEV=cpu\|mps` |
| Model SSR term structure at a θ | `slv_fast.fused_ssr_exact_ts(K, lam_fns, n, …)` · driver `slv_termstruct_exact.py` |
| Model marginal at n·dt (from spot) | `curv_diag.model_marg(chain, sig, theta, n)` → `(W,MU,SG)` |
| Interpolate a marginal chain to T | `slv_interp.interp_marginal(chain, T)` |
| Cumulants of a GM marginal | κ₂,₃,₄ from `(W,MU,SG)` central moments (see `curv_diag.exkurt`, `deevent_bridge_slv.cumulants`) |
| VIX / vol-of-vol readout | `vix_readout.py` · off-SPX stand-in: `ndx_vov_readout.py` |
| Hedging backtest (SSR-δ vs Black) | `python3 hedging_backtest.py [years…]` |
| Sticky-moneyness / smile dynamics | `python3 sticky_check.py [half npts]` |
| De-event — term-structure lump J | `python3 deevent_termstruct.py [years…] [ticker]` |
| De-event — event-conditional SSR | `python3 deevent_ssr.py [ticker] [years…]` |
| De-event — bridge (per event) | `DBG=1 python3 deevent_bridge_slv.py [events,csv] [ticker] [backdays] [flank\|spot]` — `flank` (default) = one-step-from-flank, skew-faithful; `spot` = from-spot (skew cold-starts) |
| De-event — pool / control | `python3 deevent_bridge_pool.py` · `deevent_bridge_control.py` |
| Paper figures | `python3 generate_figs.py` → `../figs/*.png` |

---

## THE KERNEL & ITS CALIBRATION  (the part I keep getting wrong — read the paper, not the drivers)

**9 structural knobs**  `θ = (γ̄, ν_F, ν_S, ν_L, λ_skew, λ_F, λ_S, κ_F, κ_S)`. γ̄ (log-variance level) **IS one of the nine** —
`slv_wire.solve_gbar` pins it to σ_ref (the γ̄-reset makes `E_π[V̄]=σ_ref²·dt`), but the paper counts it.
Code order: `NAMES = [nu_f, nu_s, nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s]` (+ gbar solved).

**The paper's objective (`eq:objective`) has THREE terms:**
1. **marginal digital band-loss** — fit the Gyöngy-fused *propagated* marginal `G^S_θ` to the market bid-ask band at `μ_{j+1}` (this is *"fit the kernel to reproduce the T_j→T_{j+1} transition"*),
2. **`β_SSR ·` SSR term-structure**,
3. **`β_vov ·` vol-of-vol readout**.
Martingale enforced *exactly* (per-fibre lock), so its soft term is dropped. `alg:calib`: per-maturity, warm-started
across j; the leverage lands the marginal by construction, recompression re-locks the martingale.

**WHICH SCRIPT FITS WHICH TERMS:**
- `calibrate_slv_exact_ts(_par)` = the **β_SSR term ALONE**. Fitting the SSR alone lets θ drift off the marginal.
- `calibrate_joint(_torch)` = **SSR + VoV** (no marginal band-loss).
- **`calibrate_full` (numpy) / `calibrate_full_torch` (jacfwd + multi-start) = the FULL 3-term `eq:objective`** — the
  marginal band-loss + SSR + VoV. This is the faithful `alg:calib` and the only correct fit for a de-eventing bridge.

**⚠️ E[ν|z] COLLAPSE BUG (found & fixed 2026-07-03; the torch port surfaced it).** `slv_fast.propagate_vec` (used by the
from-spot marginal propagators — `calibrate_full.model_marginals`, `deevent_bridge_slv.model_chain`, `curv_diag`/`curv_test`
— via the `discslv_slv.propagate=propagate_vec` monkeypatch) called the **scalar** `E_nu_given_z` with the whole means-array,
so `z−MU=0` elementwise and `float(np.sum(...))` **collapsed the local conditional into one spot-frozen E[ν] for every
component** (true per-component variance is 36–48% lower). Uncaught because `propagate_vec` was only validated against
STATE-INDEPENDENT leverage (its `__main__`) and the SSR path IS state-independent. **Fix:** `discslv_slv.E_nu_given_z_vec`
(per-query, vectorized) now used by all four callers; torch (always per-component) matches fixed numpy to 5.6e-5. **Any
result predating the fix that used a from-spot leveraged marginal (the pre-fix 3.05% marginal, the 16.8% bridge) is stale.**

**θ is regime-dependent — refit per regime, never transfer.** Validated θ:
- 2015 calm `θ_2015joint` (ν_f≈0.21)
- 2019 `θ_2019joint = [0.695, 0.360, 0.917, −0.282, 0.814, 2.004, 0.800, 2.685]` (SSR~2% + VIX~2%);
  `θ_2019_ssr_ts = [0.786, 0.375, 0.955, −0.275, 0.671, 2.123, 0.834, 2.093]` (SSR-ts only, ~1%)
- 2020 high-vol (ν_f≈0.14)
- 2015 dense/ts warm-starts in `calibrate_slv_exact_ts.X0_MAP`.

**θ affects the marginal variance** (through the E[ν|z] leverage correction) — it is **not** θ-invariant (an earlier
claim, retracted). Higher vol-of-vol → more marginal variance.

---

## PIPELINE (flow)

```
orats_pull → orats_loader (clean smiles) → orats_sanos / sanos_lp  (SANOS LP: marginals μ_j, statics)
   → slv_wire (leverage σ_LV = Dupire / E[ν|z], gbar)  +  discslv_2f (TwoFactorSV kernel, 9 knobs)
   → slv_fast.propagate_vec (fused forward map)  +  slv_interp (interpolate maturities)
   → READOUTS:  fused_ssr_exact_ts (SSR) · vix_readout (vov) · hedging_backtest (hedge)
   → calibrate_* (fit θ to the readouts — SEE CAVEAT above)
   → generate_figs (paper figures)
```

---

## DE-EVENTING  (this session — verdict: **variance, not skew**, control-validated)

Two channels + the model bridge:
- **Static variance (Tier 1):** `deevent_termstruct.py` — the event lump J = intercept of `w(T)=σ²T = J + r·T`; ~1–1.5% FOMC move, calendar-aligned. Cross-section (SPX/NDX/RUT) leans **multiplicative** (√J/vol ≈ const).
- **Dynamics:** `deevent_ssr.py` — event-conditional realized SSR (event-day vs clean-day Δσ↔r slope). Index macro = **variance-dominated** (SSR *drops* ~0.3 at the release).
- **Model bridge:** `deevent_bridge_slv.py` (disc_SLV kernel), μ₃ (SANOS) ⊖ μ₃^diff. TWO propagations: `build` = from-spot (variance fine, but skew **cold-starts** from a point mass → under-built at short T); `build_flank` = **ONE-STEP-FROM-FLANK (default, skew-faithful)** — anchor μ₂ (SANOS @ last clean expiry, carries the market skew), `lift_marginal` to a regime state via stationary π, ONE time-rescaled step (dt=Δ=T₃−T₂, event-free flank leverage), **`onestep_expand` = NO recompress** (one recompress flattens a high-skew anchor: nk16 −1.29 vs no-recompress −1.76). Per-date (2019): clean controls skew-gap ~0.03, FOMC −0.60. Set `THETA` from a **`calibrate_full`/`calibrate_full_torch` fit** (full 3-term), NOT the SSR-only fit; E[ν|z] is per-component (post-fix). `deevent_bridge.py` = model-free BKM (variance robust, skew noisy). `deevent_bridge_pool/control.py` = pool + non-event control (now `build_flank`; use the **UNFILTERED signed** means — the drivers' `J2>0` filter biases the control skew). **2022–24 aggregate (48 ev vs 30 ctrl):** variance = clean signal (event J2 3.3× control, net √≈0.99% move, matches termstruct); skew = **no significant event signal after control** (net J3 t≈1.2) — "variance not skew" now confirmed ROBUST to the propagation method, not a from-spot artifact.
- Full design/status: memory `sanos_evolve_deeventing_program.md`; paper `sec:deevent`.

---

## STALE / SUPERSEDED
- `poc/discslv_gsv.py` (single OU factor) → `poc/discslv_2f.py` (two-factor).
- `calibrate_slv_exact_ts` / `calibrate_joint` (partial-loss) → `calibrate_full` / `calibrate_full_torch` (full `eq:objective`) for any faithful / de-eventing work.
- `KM_TF_INTEGRATION`-era assumptions superseded by the current calibration; `run_poc.py` L-BFGS-B is the fit-then-project the paper rejects (the real fit is `least_squares`/TRF).

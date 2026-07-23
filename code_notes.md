# SANOS-Evolve (disc_SLV) — code guide

Discrete **stochastic-local-volatility** for SPX: SANOS arbitrage-free Gaussian-mixture **marginals
(statics)** + a two-timescale GM martingale **kernel (dynamics)**, fused by the discrete Gyongy
identity `sigma_LV^2 = sigma^2_Dupire / E[nu|z]`. The calibration objective (`eq:objective`) and
algorithm (`alg:calib`) are defined in `disc_SLV_v2.tex`.

- **Code:** `data/` = working scripts (ORATS-integrated) · `poc/` = model core + synthetic PoC.
- **Data:** `$ORATS_EOD_DIR/SPX-NDX-RUT-VIX_YYYY-MM-DD.json.gz` — EOD full chains, SPX/NDX/RUT/VIX
  bundled, 2010–2026. SPX expiries: daily since ~mid-2022; M/W/F 2016–2022; Fridays earlier.

## Conventions that bite
- Weekly step `DT = 1/52`. The SSR is **dt-dependent** (kernel transitions use kappa/lambda per-step),
  so fix a dt convention.
- **ORATS `dte = calendar-days + 1`** → an expiry's maturity is `T = ((expiry − snap).days + 1)/365`,
  not `.days/365` (off-by-one bracket bug).
- Long runs: `nohup python3 X.py … > log 2>&1 & disown`. Data at `$ORATS_EOD_DIR`; the API token comes
  from the `ORATS_TOKEN` env var (never hard-code it).
- **Multi-date runs must parallelize** — dates are independent and ~10 s each (SANOS-LP-bound).
  macOS / Py3.14 multiprocessing uses **spawn**, so any `ProcessPoolExecutor` script must guard its
  driver with `if __name__ == "__main__":` (otherwise each spawned worker re-imports the module,
  re-runs the pool, and nests pools → `BrokenProcessPool`). Set `OMP_NUM_THREADS=1` to avoid BLAS
  oversubscription.

## Roadmap — "to do X, run Y"

| Goal | Script / call |
|---|---|
| Pull ORATS history | `orats_pull.py` |
| Clean smiles for a date | `orats_loader.load_day(path, [ticker])` → `{expiry: {T,F,DF,strike,iv,cmid,pmid,...}}` |
| SANOS arb-free marginals (a date) | `slv_wire.sanos_chain(path, ticker)` → `[(T, (W,MU,SG)), …]` |
| Realized (ℙ) SSR of a period | `empirical_ssr.empirical_ssr(sorted(glob(year)), ns=NS, ticker=…)` |
| Fit θ — SSR only | `calibrate_slv_exact_ts.py` / `calibrate_slv_exact_ts_par.py` (parallel) |
| Fit θ — SSR + VoV | `calibrate_joint.py` (numpy) · `calibrate_joint_torch.py` (MPS autodiff) |
| **Fit θ — full `eq:objective` (marginal band-loss + SSR + VoV)** | `calibrate_full.py` (numpy) · `calibrate_full_torch.py` (jacfwd + multi-start) — the faithful `alg:calib` |
| Model SSR term structure at a θ | `slv_fast.fused_ssr_exact_ts(...)` · driver `slv_termstruct_exact.py` |
| Interpolate a marginal chain to T | `slv_interp.interp_marginal(chain, T)` |
| VIX / vol-of-vol readout | `vix_readout.py` · off-SPX stand-in `ndx_vov_readout.py` |
| Hedging backtest (SSR-δ vs Black) | `hedging_backtest.py` |
| Paper figures | `generate_figs.py` → `figs/*.png` |

## The kernel and its calibration

**9 structural knobs** `θ = (γ̄, ν_F, ν_S, ν_L, λ_skew, λ_F, λ_S, κ_F, κ_S)`. γ̄ (the log-variance
level) is pinned by `slv_wire.solve_gbar` to σ_ref but counts as one of the nine.

**`eq:objective` has three terms:** (1) the **marginal digital band-loss** — fit the Gyöngy-fused
propagated marginal to the market bid–ask band at `μ_{j+1}`; (2) `β_SSR ·` the SSR term structure;
(3) `β_vov ·` the vol-of-vol readout. The martingale is enforced exactly (per-fibre lock), so its
soft term is dropped; `alg:calib` runs per maturity, warm-started across j, and the leverage lands
the marginal by construction.

**Which script fits which terms:**
- `calibrate_slv_exact_ts(_par)` = the **SSR term alone** (θ can drift off the marginal).
- `calibrate_joint(_torch)` = **SSR + VoV** (no marginal band-loss).
- **`calibrate_full` / `calibrate_full_torch` = the full 3-term `eq:objective`** — the only correct
  fit when the marginal must be matched.

**θ is regime-dependent — refit per regime, never transfer.** θ affects the marginal variance through
the E[ν|z] leverage correction (higher vol-of-vol → more marginal variance).

## Pipeline

```
orats_pull → orats_loader (clean smiles) → sanos_lp (marginals μ_j, statics)
   → slv_wire (leverage σ_LV = Dupire / E[ν|z], gbar) + discslv_2f (two-factor kernel, 9 knobs)
   → slv_fast.propagate_vec (fused forward map) + slv_interp (interpolate maturities)
   → readouts: fused_ssr_exact_ts (SSR) · vix_readout (vov) · hedging_backtest (hedge)
   → calibrate_* (fit θ) → generate_figs (paper figures)
```

## Current vs superseded
- `poc/discslv_gsv.py` (single OU factor) → `poc/discslv_2f.py` (two-factor).
- `calibrate_slv_exact_ts` / `calibrate_joint` (partial loss) → `calibrate_full` /
  `calibrate_full_torch` (full `eq:objective`) for any faithful marginal-matching work.

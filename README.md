# SANOS-Evolve — a discrete stochastic-local-volatility model for European-option smile dynamics

This repository accompanies the working paper *SANOS-Evolve* by **Shaosai Huang** (Kspectra Research Inc.):
the manuscript, the figures, and the calibration/evaluation code behind the empirical study.

## Paper

This is a **working paper** (comments welcome). The current version is
[`disc_SLV_v3.pdf`](disc_SLV_v3.pdf) (source [`disc_SLV_v3.tex`](disc_SLV_v3.tex)); the previous
version [`disc_SLV_v2.pdf`](disc_SLV_v2.pdf) is kept for reference.

**v3 is a substantial revision, not an increment.** The carried volatility factors are continuous
AR(1) processes with a closed-form k-step law, where v2 discretised them onto a node ladder; the
readouts, the leverage overlay and the admissibility claims were rewritten to match what the code
does. Results in v2 should not be compared term-by-term with v3.

## What the model is

SANOS-Evolve is a discrete stochastic-local-volatility construction — arbitrage-free **SANOS
marginals** (the static smile) plus a fitted **martingale transition kernel** with a two-timescale
latent-volatility regime (the dynamics). It is translation-invariant in log-spot, which gives a
**closed-form skew-stickiness ratio (SSR)** as an exact realised covariance. The vol-of-vol is
*identified* from a forward-variance **readout** (VIX for SPX; the option strip's own curvature
otherwise) rather than by jointly calibrating a volatility-index options market, so the construction
stays portable across underlyings. SPX is the worked example.

## Layout

Everything lives at the repository root:

```
disc_SLV_v3.tex, .pdf   manuscript, current version (source + compiled)
disc_SLV_v2.tex, .pdf   manuscript, previous version
figs_v3/                figures for v3
v3/                     the v3 engine — see below
ssr_forecast_*.md       method notes (SSR-forecast test design + findings)
scripts/                out-of-sample harness — ssr_forecast_eval.py (Newey–West HAC / Diebold–Mariano /
                        encompassing), wire_orats.py (real-data run), hullwhite.py (cross-strike
                        Hull–White minimum-variance-delta baseline), run_demo.py (no-data self-test)
data/                   calibration + empirical — calibrate_joint_torch.py (two-timescale kernel,
                        GPU autodiff), fit_summary_ms.py (multi-start cross-regime SSR + vol-of-vol),
                        empirical_ssr.py / orats_loader.py (ORATS readers), generate_figs.py,
                        hedging_backtest.py (SSR-consistent hedging replay), vix_*.py diagnostics
poc/                    core model engine (SANOS LP, discrete-SLV kernel, VIX readout)
figs/                   manuscript figures (v2)
```

### The v3 engine

`v3/` holds exactly what the v3 paper uses — the import closure of its five entry points, not the
whole working tree:

```
v3/kernel_fast/         the production path: consts, fkernel, propagate, readouts, vix, refit
v3/diagnostics/         emit_jacobian_table.py (Table 12), leverage_remainder.py (the finite-step
                        leverage diagnostic), kernel_hedge_test.py (the smile-roll replay)
v3/figures/             the scripts that generate the paper's figures
v3/artifacts/           manifest.json (every production constant + a per-record fingerprint and
                        reproduction check), the readout-Jacobian spectrum, the leverage-remainder
                        summaries
v3/artifacts/records/   the 18 shipped fit records (9 SPX, 9 NDX) behind every reported number
```

Two entry points need the option chains and cannot be run from this repository alone:
`v3/kernel_fast/refit.py` (it calibrates against them) and `v3/diagnostics/kernel_hedge_test.py`
(it replays a rolling SPX option position). The other three run against the artifacts shipped here.

## Data

The empirical studies use **proprietary ORATS / CBOE end-of-day option data**, which is **not
included** here (it is licensed and not redistributable). Point the loaders at your own ORATS EOD store
by setting the `ORATS_EOD_DIR` environment variable (files `SPX-NDX-RUT-VIX_YYYY-MM-DD.json.gz`); it
defaults to `~/orats_eod`. Without the data the code is reference-only.

## Build the manuscript

```sh
latexmk -pdf disc_SLV_v2.tex      # or: pdflatex -interaction=nonstopmode disc_SLV_v2.tex  (run 2–3x)
```

The bibliography is inline (`thebibliography`), so no `.bib` is needed; figures resolve via
`\graphicspath{{./}}` → `figs/`.

## Reproduce (with your own data)

- `scripts/run_demo.py` — **synthetic self-test, no data required** (`python3 scripts/run_demo.py`,
  exit 0 = pass).
- `scripts/` — the out-of-sample SSR-forecast + Hull–White hedging harness
  (see [`scripts/README.md`](scripts/README.md)); needs `ORATS_EOD_DIR`.
- `data/` — two-timescale kernel calibration (`fit_summary_ms.py`, `calibrate_joint_torch.py`),
  the SSR-consistent hedging replay (`hedging_backtest.py`), and the figures (`generate_figs.py`).

## Citation

```bibtex
@techreport{huang2026sanos,
  author      = {Huang, Shaosai},
  title       = {{SANOS-Evolve}: A Discrete Stochastic-Local-Volatility Model for European-Option Smile Dynamics},
  institution = {Kspectra Research},
  year        = {2026},
  type        = {Working paper}
}
```

## License

The **code** (the Python scripts under `scripts/`, `data/`, and `poc/`) is released under the
**MIT License** — see [`LICENSE`](LICENSE). The **manuscripts and figures** (`disc_SLV_v3.tex`,
`disc_SLV_v3.pdf`, `figs_v3/`, `disc_SLV_v2.tex`, `disc_SLV_v2.pdf`, `figs/`) are © the author, all
rights reserved, and are **not** covered by the
MIT License.

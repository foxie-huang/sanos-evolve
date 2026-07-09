# SANOS-Evolve — a discrete stochastic-local-volatility model for European-option smile dynamics

**Working paper — private preview.** This repository accompanies the working paper *SANOS-Evolve*
(`manuscript_v2/`): the manuscript, the figures, and the calibration/evaluation code behind the
empirical study. Contents will be updated and the repository made public in a later revision.

## What the model is

SANOS-Evolve is a discrete stochastic-local-volatility construction — arbitrage-free **SANOS
marginals** (the static smile) plus a fitted **martingale transition kernel** with a two-timescale
latent-volatility regime (the dynamics). It is translation-invariant in log-spot, which gives a
**closed-form skew-stickiness ratio (SSR)** as an exact realised covariance. The vol-of-vol is
*identified* from a forward-variance **readout** (VIX for SPX; the option strip's own curvature
otherwise) rather than by jointly calibrating a volatility-index options market, so the construction
stays portable across underlyings. SPX is the worked example.

## Layout

```
manuscript_v2/               the manuscript
├── disc_SLV_v2.tex          source
├── disc_SLV_v2.pdf          compiled
├── ssr_forecast_*.md        method notes (SSR-forecast test design + findings)
└── scripts/                 out-of-sample evaluation harness
    ├── ssr_forecast_eval.py   forecast engine (Newey–West HAC, Diebold–Mariano, encompassing)
    ├── wire_orats.py          real-data SSR-forecast run
    ├── hullwhite.py           cross-strike Hull–White minimum-variance delta baseline
    └── run_demo.py, wire_poc.py, results_*.json
v2/
├── data/                    calibration + empirical scripts
│   ├── calibrate_joint_torch.py   two-timescale kernel calibration (GPU autodiff)
│   ├── fit_summary_ms.py          multi-start cross-regime SSR + vol-of-vol fits
│   ├── empirical_ssr.py, orats_loader.py   ORATS smile / SSR readers
│   ├── generate_figs.py, regen_offspx.py   figure generation
│   ├── hedging_backtest.py                 SSR-consistent hedging replay
│   ├── vix_*.py, vov_nfactor_probe.py      vol-of-vol diagnostics
│   └── *.json                     cached fit results
└── poc/                     core model engine (SANOS LP, discrete-SLV kernel, VIX readout)
figs/                        manuscript figures
```

## Data

The empirical studies use **proprietary ORATS / CBOE end-of-day option data**, which is **not
included** here. The loaders (`orats_loader.py`, `wire_orats.py`) expect a local ORATS EOD store —
point them at your own copy. Without it the code is reference-only. External-data-acquisition helpers
(Finnhub / FMP) are omitted; where API keys are used they are read from environment variables and are
never hard-coded.

## Build the manuscript

```sh
cd manuscript_v2 && latexmk -pdf disc_SLV_v2.tex      # or: pdflatex -interaction=nonstopmode disc_SLV_v2.tex  (twice)
```

## Reproduce (with your own data)

See `manuscript_v2/scripts/README.md` for the out-of-sample SSR-forecast + Hull–White hedging
pipeline, and `v2/data/` for calibration (`fit_summary_ms.py`) and the figures (`generate_figs.py`).

## Status

Active research code — not yet a curated reproducibility package; a cleaned release will follow.

## License

The **code** (the Python scripts under `manuscript_v2/scripts/` and `v2/`) is released under the
**MIT License** — see [`LICENSE`](LICENSE). The **manuscript and figures** (`manuscript_v2/*.tex`,
`manuscript_v2/*.pdf`, `figs/`) are © the authors, all rights reserved, and are **not** covered by the
MIT License.

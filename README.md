# SANOS-Evolve — a discrete stochastic-local-volatility model for European-option smile dynamics

This repository accompanies the working paper *SANOS-Evolve* by **Shaosai Huang** (Kspectra Research Inc.):
the manuscript (`manuscript_v2/`), the figures, and the calibration/evaluation code behind the empirical
study.

## Paper

This is a **working paper** (comments welcome). The compiled manuscript is in this repository:
[`manuscript_v2/disc_SLV_v2.pdf`](manuscript_v2/disc_SLV_v2.pdf), with source in
[`manuscript_v2/disc_SLV_v2.tex`](manuscript_v2/disc_SLV_v2.tex). Preprint versions are posted on SSRN and
preprints.org.

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
└── scripts/                 out-of-sample evaluation harness (curated)
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
included** here (it is licensed and not redistributable). Point the loaders at your own ORATS EOD store
by setting the `ORATS_EOD_DIR` environment variable (files `SPX-NDX-RUT-VIX_YYYY-MM-DD.json.gz`); it
defaults to `~/orats_eod`. Without the data the code is reference-only.

## Build the manuscript

```sh
cd manuscript_v2 && latexmk -pdf disc_SLV_v2.tex      # or: pdflatex -interaction=nonstopmode disc_SLV_v2.tex  (run 2–3×)
```

The bibliography is inline (`thebibliography`), so no `.bib` is needed; figures resolve via
`\graphicspath{{../}}` → `figs/`.

## Reproduce (with your own data)

- `manuscript_v2/scripts/run_demo.py` — **synthetic self-test, no data required** (`python3 run_demo.py`,
  exit 0 = pass).
- `manuscript_v2/scripts/` — the out-of-sample SSR-forecast + Hull–White hedging harness
  (see [`scripts/README.md`](manuscript_v2/scripts/README.md)); needs `ORATS_EOD_DIR`.
- `v2/data/` — two-timescale kernel calibration (`fit_summary_ms.py`, `calibrate_joint_torch.py`),
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

The **code** (the Python scripts under `manuscript_v2/scripts/` and `v2/`) is released under the
**MIT License** — see [`LICENSE`](LICENSE). The **manuscript and figures** (`manuscript_v2/*.tex`,
`manuscript_v2/*.pdf`, `figs/`) are © the author, all rights reserved, and are **not** covered by the
MIT License.

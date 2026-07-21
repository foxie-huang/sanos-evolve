# SANOS-Evolve — proof-of-concept (paper §12)

A runnable, end-to-end skeleton of the *SANOS-Evolve* discrete stochastic-local-volatility
pipeline for SPX: SANOS-style Gaussian-mixture marginals (statics) + a two-timescale
latent-regime Gaussian-mixture **martingale kernel** (dynamics). Everything is closed
form — Black–Scholes enters as the MGF of a truncated Gaussian, so every mixture price is
a finite weighted sum of Black calls.

## Files
- `discslv.py` — the library: Black-Scholes (MGF form) + implied vol; `GMM` in
  log-moneyness (call, CDF, convex-order check); `TwoTimescaleKernel` (coefficient map,
  exact per-fibre martingale normalisation, GH-quadrature propagation, forward-start smile,
  SSR); synthetic convex-ordered marginals.
- `run_poc.py` — the driver running the six-step §12 protocol + the Method A/B ablation.
- `run_extended.py` — extended demo: multi-step propagation + §11 recompression (§§8, 11)
  and the §7 VIX vol-of-vol layer.
- `intraday.py` — intraday / 0DTE algorithm suite (§13): calibration loop + SSR hedger +
  skew/SSR signal, on a synthetic intraday world.
- `fetch_data.py` — pull real SPX/SPY option chains from Yahoo (`^SPX`) into `../data/`.
- `check_finnhub.py`, `probe_fmp.py` — probe a data API's option entitlements.
- `requirements.txt` — numpy, scipy.

## Run
```bash
pip install -r requirements.txt
python run_poc.py          # single-step Sec. 12 pipeline + Method A/B ablation
python run_extended.py     # multi-step + recompression (Sec. 8/11) + VIX layer (Sec. 7)
python intraday.py         # intraday / 0DTE algo suite (Sec. 13): hedger, signal
```

## What it demonstrates (maps to paper §12)
1. **Marginals + convex order** — synthetic SANOS-stand-in marginals; the calendar
   no-arbitrage precondition of Theorem 1 is checked (violation ≈ 0).
2. **One admissible kernel** — fitted to the next marginal at the digital/CDF rung;
   `lam_skew` is set by the marginal, and the kernel is martingale **by construction**.
   Marginal KS ≈ 1e-3.
3. **Martingale check** — `forward(propagated μ₀K) = 1.0` to machine precision.
4. **Forward-start smile** — closed-form forward IV by return-moneyness, with a negative
   ATM skew.
5. **SSR moves with the state** — `dΣ_ATM/d log S ≠ 0`; the model SSR hits the imposed
   view (1.30).
6. **Ablation (the headline claim)** — Method A (sticky-delta, `lam_mov = 0`, SSR ≈ 0) vs
   Method B (`lam_mov` fit, SSR = 1.30) share the *same* marginal fit and static skew, so
   the SSR is a **structural, (nearly) marginal-free dynamic degree of freedom**.

## What is real vs. stubbed
**Real machinery:** the two-knob kernel (`lam_skew` = leverage mean-tilt → forward skew;
`lam_mov` = spot-coupling → SSR), the exact per-fibre martingale normalisation, closed-form
mixture-of-Black pricing, the forward-start smile, and the SSR finite-difference.

**Stubbed / simplified (the explicit next steps):**
- *Marginals are synthetic* — μ₀ is a symmetric martingale mixture, μ₁ = μ₀ convolved with
  an independent **skewed** martingale increment (guarantees convex order and a realistic
  skew). Swap in real SANOS LP output at `discslv.synthetic_marginals`.
- *Multi-step + recompression now implemented* (`run_extended.py`): a time-homogeneous
  generator propagates a maturity chain, and moment-merge recompression + a global
  martingale re-lock keeps the component budget flat (18 vs a 6.8M blow-up at 5 steps)
  while preserving convex order and the forward. State-dependent weights use source-mean
  collocation; the constrained *kernel-based* projection (§11) is the cleaner variant still
  to add.
- *SSR is an imposed view* — the synthetic increment is spot-independent (true SSR ≈ 0), so
  the PoC imposes the SSR view via `lam_mov` and shows it costs ~nothing on the marginal.
  Real calibration would take the SSR from forward-start quotes or a desk view.
- *VIX layer now implemented* (`run_extended.py`): model-implied VIX future / smile /
  vol-of-vol from the regime law and damped regime variances, with the nu_s <-> vol-of-vol
  identification (the R_vov target) and the slow-dominated damping. **Remaining:** real
  SANOS marginals and the hedging-P&L backtest (§12).

## Caveats
- The numbers are illustrative (synthetic data); they validate the *pipeline*, not a
  market fit.
- The SSR sign/convention follows Bergomi: SSR = 1 is sticky-strike, 0 is sticky-delta;
  `lam_mov = 0` gives the sticky-delta limit (SSR ≈ 0).

## Intraday / 0DTE algorithm suite (`intraday.py`)
Re-scopes the model to the 0DTE / intraday-expiry surface (paper §13) and runs four
research algorithms on one calibration loop, demonstrated on a synthetic intraday world (a
live 0DTE feed plugs in behind the same interface). **Research / decision-support — it
produces hedge ratios and signals, not orders.**

- **Calibration loop** — per-snapshot ATM vol / skew / rolling model-SSR / vol-of-vol /
  regime at ~0.03 ms/snapshot (real-time).
- **SSR hedger + backtest** — the *fixed-strike* delta
  `Δ = Δ_BS + vega · skew · (SSR − 1)/S` (so SSR=1 is sticky-strike = Black, SSR=0 is
  sticky-delta). On the synthetic 0DTE world (realised SSR ≈ 1.6) the hedging-P&L variance
  bottoms at the realised SSR, and the **SSR-correct delta beats Black/sticky-strike and
  clearly beats sticky-delta** — the central claim.
- **Skew/SSR signal** — model vs realised intraday SSR → mean-reversion signal.

Honest findings: the win over *sticky-delta* is large; the win over *Black/sticky-strike*
is modest and grows with how far the realised SSR sits above 1 (it is high, ~1.6–1.9, in
the 0DTE regime); a noisy rolling-SSR estimate erodes the edge, arguing for a longer window
or the structural (kernel) SSR. Real use needs a live 0DTE feed, the intraday
seasonality/jump structure of §13, transaction-cost calibration, and out-of-sample
validation. **Not order execution.**

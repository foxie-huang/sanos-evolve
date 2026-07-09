# v2 — smoothed-recompress + AD (IN PROGRESS, 2026-07)

Started as an identical copy of v1. Plan:
1. Redesign `recompress_2f` as vectorized + differentiable (soft-assignment / fixed-grid) — kills the
   discrete cluster-merge seams (speed ~2x, smoother objective, AD-ready).
2. Port the forward map to JAX for AD gradients (8-eval finite-difference Jacobian -> ~1 pass).
Target: ~1 min/fit for the per-date backtest (cached target, warm start). Active development lives here.

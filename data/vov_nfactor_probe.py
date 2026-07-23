#!/usr/bin/env python3
"""Representability probe: can K OU timescales trace the grind vov term structure at all?
Fit the DATA vov(tau) with the model's own damping family  Y(tau)=c0 + sum_i c_i * D(kappa_i*tau),
D(x)=(1-e^-x)/x, for K=1,2,3 factors, UNCONSTRAINED (no SSR coupling) -> best case per K.
If K=2 already nails it, the paper's ~22% miss is the joint-SSR tension, not a factor deficit;
if K=2 plateaus and K=3 drops sharply, a 3rd (slow) factor genuinely buys the long end.
Data vov copied from vix_maturity_breakdown.py (polished-fit run)."""
import numpy as np
from scipy.optimize import least_squares

GRIND = {
 "2022": (np.array([8,15,22,29,50,78,113,141,169,204,232,260]),
          np.array([.823,.873,.875,.919,.847,.809,.751,.709,.677,.643,.616,.580])),
 "2023": (np.array([7,14,21,28,35,49,77,112,140,168,203,231,259]),
          np.array([.907,.857,.773,.771,.759,.749,.706,.661,.618,.603,.570,.549,.526])),
}

def D(x): return np.where(x < 1e-9, 1.0, (1.0 - np.exp(-np.clip(x, 0, 60))) / np.where(x < 1e-9, 1.0, x))
def model(p, tau, K):
    c0, cs, ks = p[0], p[1:1+K], p[1+K:1+2*K]
    return c0 + sum(cs[i] * D(ks[i] * tau) for i in range(K))
def hfit(tau, y): return -np.polyfit(np.log(tau), np.log(y), 1)[0]

# kappa seed sets (1/yr): geometric spreads spanning ~2wk .. ~10yr timescales
SEEDS = {1: [[3.0], [0.5], [12.0]],
         2: [[12.0, 1.0], [20.0, 0.3], [6.0, 0.1]],
         3: [[20.0, 3.0, 0.3], [30.0, 4.0, 0.15], [12.0, 1.5, 0.1], [40.0, 2.0, 0.1]]}

for yr, (dte, vov) in GRIND.items():
    tau = dte / 365.0
    print(f"\n=== {yr} grind (data vov h={hfit(tau, vov):.3f}, {vov[0]:.2f}@{dte[0]}d -> {vov[-1]:.2f}@{dte[-1]}d) ===")
    for K in (1, 2, 3):
        best = None
        for ks0 in SEEDS[K]:
            p0 = np.array([vov.min()] + [max(vov.max() - vov.min(), 0.1)] * K + ks0, float)
            lo = [0.0] + [0.0] * K + [0.02] * K
            hi = [2.0] + [5.0] * K + [400.0] * K
            try:
                r = least_squares(lambda p: (model(p, tau, K) - vov) / vov, p0,
                                  bounds=(lo, hi), max_nfev=4000, method="trf")
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        yv = model(best.x, tau, K)
        rms = 100 * np.sqrt(np.mean(((yv - vov) / vov) ** 2))
        long_rms = 100 * np.sqrt(np.mean(((yv - vov)[dte >= 140] / vov[dte >= 140]) ** 2))
        ks = np.sort(best.x[1+K:1+2*K])
        taus = ", ".join(f"{1/k*12:.1f}mo" for k in ks)   # timescales 1/kappa in months
        print(f"  K={K}: rel-RMS {rms:5.1f}%   long-end(>=140d) {long_rms:5.1f}%   "
              f"model-h {hfit(tau, yv):.3f}   timescales [{taus}]")
print("\n[done]")

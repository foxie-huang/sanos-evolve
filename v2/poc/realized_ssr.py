"""
realized_ssr.py -- the REALIZED SSR: the model's prediction for the SPX time-series regression.

The empirical SPX SSR regresses daily ATM-vol changes on returns: SSR(T)=beta_T/skew_T.  The matching
model quantity is NOT the comparative-statics bump (bump spot + reset the regime to slow_weights),
which over-counts the regime response.  It is the response along the realized one-step dynamics:
the current regime is a definite state b (held), the spot moves, and we measure how the T-vol responds.

Because a one-step regime change b->b' is uncorrelated with the return at z=0 (the transition target
slow_weights(0)=prior has no spot tilt), the r-correlated part of the realized vol change is the spot-
LEVEL response of the T-vol holding the initial regime.  Hence:

    realized_SSR(T) = E_{b~prior}[ d sigma_ATM(T; z,b)/dz |_0 ] / E_{b~prior}[ skew(T; 0,b) ]

with sigma_ATM(T; z,b) propagated from the DEFINITE initial regime b (point mass), via exact MC (the
GH closed-form damps the response, so MC is required).  Contrast: comparative-statics resets the
initial regime to slow_weights(z+-h) -> extra d(regime)/dz term -> the high, rising curve.

Run from poc/ :  python3 realized_ssr.py
"""
import numpy as np
import warnings
from discslv import TwoTimescaleKernel
from ssr_demo import BASE, LAM_MOV_0, ssr_at
from mc_check import simulate, _iv, mc_stats

warnings.filterwarnings("ignore")
NPATHS = 300_000
NBATCH = 8
SEED = 7


def realized_ssr(K, n, h=8e-3, dm=6e-3, npaths=NPATHS, nbatch=NBATCH, seed=SEED):
    """E_b[ d ATMvol(T)/dz | regime b held ] / E_b[ skew(T;b) ], exact MC, batched stderr."""
    T = n * K.dt
    prior = K.regime_prior()
    bs = npaths // nbatch
    num_b = np.zeros((K.n_slow, nbatch))
    skw_b = np.zeros((K.n_slow, nbatch))
    for b in range(K.n_slow):
        zp = simulate(K, n, +h, seed, npaths, init_b=b)        # spot +h, regime fixed at b (CRN)
        zm = simulate(K, n, -h, seed, npaths, init_b=b)        # spot -h, regime fixed at b
        z0 = simulate(K, n, 0.0, seed, npaths, init_b=b)       # for the per-regime skew
        Sp, Sm, S0 = np.exp(zp), np.exp(zm), np.exp(z0)
        for i in range(nbatch):
            s = slice(i * bs, (i + 1) * bs)
            Fp, Fm, F0 = Sp[s].mean(), Sm[s].mean(), S0[s].mean()
            num_b[b, i] = (_iv(Sp[s], Fp, Fp, T) - _iv(Sm[s], Fm, Fm, T)) / (2 * h)
            skw_b[b, i] = (_iv(S0[s], F0, F0 * np.exp(dm), T) - _iv(S0[s], F0, F0 * np.exp(-dm), T)) / (2 * dm)
    num = (prior[:, None] * num_b).sum(0)                       # prior-weighted, per batch
    skw = (prior[:, None] * skw_b).sum(0)
    ssr = num / skw
    return float(ssr.mean()), float(ssr.std(ddof=1) / np.sqrt(nbatch))


def main():
    K = TwoTimescaleKernel(lam_mov=LAM_MOV_0, **BASE)
    spx = {1: 1.9, 4: 1.5, 13: 1.3, 26: 1.1, 52: 1.0}
    print("SSR term structure: REALIZED (matches the SPX regression) vs comparative-statics vs SPX\n")
    print(f"{'n':>3} {'mat':>4} | {'realized (MC)':>15} | {'comp-stat (MC)':>14} | {'closed-form':>11} | {'SPX':>4}")
    print("-" * 70)
    for n, lab in [(1, "1w"), (4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]:
        rm, re = realized_ssr(K, n)
        cs = mc_stats(K, n, seed=SEED, npaths=NPATHS, nbatch=NBATCH, h=8e-3)[2][0]
        cf = ssr_at(K, n)
        print(f"{n:>3} {lab:>4} | {rm:>9.2f} +/- {re:.2f} | {cs:>14.2f} | {cf:>11.2f} | {spx[n]:>4.1f}", flush=True)


if __name__ == "__main__":
    main()

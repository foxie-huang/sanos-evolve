"""
retune_check.py -- can a pure level re-tune (lower lam_mov) fit the contango SPX SSR target?

The variance-curve test showed our contango-conditional (b0=0) realized SSR has the RIGHT rising
shape but is too HIGH (1.65 -> 2.15 vs the recent SPX contango benchmark ~0.9 -> ~1.6). lam_mov is
the SSR LEVEL knob. Here we sweep lam_mov and read the b0=0 (contango) realized SSR term structure;
if some lam_mov lands it on the target band, we can fit SPX with NO structural change and the
two-factor kernel becomes an optional strengthening extension.

Target band (recent SPX, contango): Quintic 6-May-2024 ~0.9->1.6; Gatheral 2023 per-maturity
1m=1.47 3m=1.45 6m=1.51 1y=1.60. Both rising; we aim the model between them.

Run from poc/ :  python3 retune_check.py
"""
import numpy as np
import warnings
from discslv import TwoTimescaleKernel
from ssr_demo import BASE
from mc_check import simulate, _iv

warnings.filterwarnings("ignore")
NP = 150_000
SEED = 7
MATS = [(4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]


def contango_ssr(K, n, b=0, h=8e-3, dm=6e-3, npaths=NP):
    """Realized SSR conditional on the current regime being b (b=0 = low vol = contango curve)."""
    T = n * K.dt
    zp = simulate(K, n, +h, SEED, npaths, init_b=b)
    zm = simulate(K, n, -h, SEED, npaths, init_b=b)
    z0 = simulate(K, n, 0.0, SEED, npaths, init_b=b)
    Sp, Sm, S0 = np.exp(zp), np.exp(zm), np.exp(z0)
    Fp, Fm, F0 = Sp.mean(), Sm.mean(), S0.mean()
    skew = (_iv(S0, F0, F0 * np.exp(dm), T) - _iv(S0, F0, F0 * np.exp(-dm), T)) / (2 * dm)
    num = (_iv(Sp, Fp, Fp, T) - _iv(Sm, Fm, Fm, T)) / (2 * h)
    return num / skew


def main():
    print("Re-tune lam_mov -> contango-conditional (b0=0) realized SSR vs SPX target\n")
    print(f"{'lam_mov':>8} | " + "  ".join(f"{lab:>5}" for _, lab in MATS))
    print("-" * 40)
    for lm in [-5, -6, -7, -8, -10, -12]:
        K = TwoTimescaleKernel(lam_mov=lm, **BASE)
        row = [contango_ssr(K, n) for n, _ in MATS]
        print(f"{lm:>8} | " + "  ".join(f"{x:5.2f}" for x in row), flush=True)
    print("-" * 40)
    print(f"{'SPX~Quintic':>8} |  ~0.9   ~1.2   ~1.4   ~1.6   (rising, contango)")
    print(f"{'Gatheral23':>8} |  1.47   1.45   1.51   1.60")


if __name__ == "__main__":
    main()

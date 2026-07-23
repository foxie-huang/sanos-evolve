"""
node_convergence.py -- Gauss-Hermite node-count convergence study for the two-factor kernel.

Question (Arthur, 2026-06-28): how are n_f, n_s, n_l determined -- is there a level
beyond which the discretisation error is small enough?

Answer (this script): the per-node variance V = exp(nu*zeta)*dt is LOGNORMAL in the GH
node, so GH (polynomial-exact) resolves E[V] only SLOWLY, at a rate governed by the
vol-of-vol nu.  n_l (small nu_l=0.14) is converged at 5; n_f, n_s (nu~0.45-0.50) are NOT
converged even at 9/7 -- the ATM-vol LEVEL climbs monotonically with node count.

DECISION (kept): 5/3/5 is a deliberately COARSE modelling choice, not a converged grid.
Calibration absorbs the discretisation -- gbar and the leverages are fit AT the chosen grid,
so the ATM/SSR targets are hit at any count and refining merely re-shifts theta.  Bumping
the counts does not buy convergence (the lognormal level never settles) and costs ~4x, so
we keep 5/3/5 and frame the continuous limit as a structural relationship, not an accuracy
claim.  (disc_SLV.tex sec:twoscale + math_flow.html 2.0a reworded accordingly.)

Run from poc/ :  python3 node_convergence.py
"""
import numpy as np, warnings, time
warnings.filterwarnings('ignore')
from discslv_2f import TwoFactorSV, ssr_2f

DT = 1 / 52.0
TH = [-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34]   # calibrated theta (calibrate_2f.py)


def kern(nf, ns, nl):
    return TwoFactorSV(gbar=TH[0], nu_f=TH[1], nu_s=TH[2], nu_l=TH[3], lam_skew=TH[4],
                       lam_f=TH[5], lam_s=TH[6], kap_f=TH[7], kap_s=TH[8], dt=DT,
                       n_f=nf, n_s=ns, n_l=nl)


def reads(nf, ns, nl, nk=12, names=('vol1m', 'skew1m', 'SSR1m', 'SSR1y')):
    """Key readouts at one grid (same calibrated theta), for the chosen node counts."""
    K = kern(nf, ns, nl)
    r1, v1, k1 = ssr_2f(K, 4, nk=nk)        # 1m: SSR, ATM vol, ATM skew
    ry = ssr_2f(K, 52, nk=nk)[0]            # 1y SSR
    return np.array([v1, k1, r1, ry])


def isolated_sweep(ref=(9, 6, 9), nk=14):
    """Hold two factors at a high reference, sweep the third -> isolates each factor's
    convergence.  Result (2026-06-28): n_l converged at 5 (~0.005); n_s=3 coarse (~0.09);
    n_f=5 coarse (~0.05), both still decreasing."""
    RF, RS, RL = ref
    names = ['ATMvol1m', 'skew1m', 'SSR1m', 'SSR1y']
    base = reads(RF, RS, RL, nk=nk)
    print(f"reference {ref}:", dict(zip(names, np.round(base, 4))))
    for lab, vals in [('n_f', [2, 3, 4, 5, 6]), ('n_s', [2, 3, 4, 5]), ('n_l', [3, 5, 7, 9])]:
        print(f"\n=== sweep {lab} (others at {ref}) ===")
        print(f"{lab:>4} " + "".join(f"{n:>9}" for n in names) + "   max|err vs ref|")
        for n in vals:
            out = reads(n, RS, RL, nk=nk) if lab == 'n_f' else \
                  reads(RF, n, RL, nk=nk) if lab == 'n_s' else reads(RF, RS, n, nk=nk)
            print(f"{n:>4} " + "".join(f"{o:>9.4f}" for o in out) + f"   {np.abs(out - base).max():.2e}")


def combo_sweep(nk=12):
    """n_f in {7,8,9} x n_s in {6,7}, n_l=5, vs the 5/3/5 baseline and the 9/7/5 ceiling.
    Result (2026-06-28): vol(1m) climbs monotonically 0.161 -> 0.23 and is STILL rising at
    9/7 (lognormal/GH slow convergence); SSR(1m) settles to ~1.41 across the high combos
    (vs 1.48 at 5/3).  The ~0.07 gap from 5/3 is absorbed by recalibration."""
    names = ['vol1m', 'skew1m', 'SSR1m', 'SSR1y']
    combos = [(5, 3), (7, 6), (8, 6), (9, 6), (7, 7), (8, 7), (9, 7)]
    res = {}
    for (nf, ns) in combos:
        t = time.time(); res[(nf, ns)] = reads(nf, ns, 5, nk=nk)
        print(f"  n_f={nf} n_s={ns} n_l=5  " +
              " ".join(f"{nm}={v:+.4f}" for nm, v in zip(names, res[(nf, ns)])) +
              f"  [{time.time() - t:.0f}s]")
    ref, base = res[(9, 7)], res[(5, 3)]
    print(f"\n{'nf/ns':>6} " + "".join(f"{n:>9}" for n in names) + "   d(9/7)   d(5/3)")
    for (nf, ns) in combos:
        r = res[(nf, ns)]
        print(f"{nf}/{ns:<4} " + "".join(f"{v:>9.4f}" for v in r) +
              f"   {np.abs(r - ref).max():.3f}    {np.abs(r - base).max():.3f}")


if __name__ == "__main__":
    print("### isolated sweep (each factor vs high reference) ###")
    isolated_sweep()
    print("\n### combo sweep (n_f=7,8,9 x n_s=6,7, n_l=5) ###")
    combo_sweep()

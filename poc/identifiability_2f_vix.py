"""
identifiability_2f_vix.py -- Does the closed-form VIX-vol-of-vol term structure (@30d, @90d)
resolve the soft (lambda,kappa) direction of the 9-knob identifiability?

Compares the elasticity-scaled SVD spectrum of:
  (A) the 9 smile/SSR observables (the current Step-6 map), vs
  (B) those 9 PLUS VIXvov@30d and VIXvov@90d  -> 11 observables.

VIX-vov is closed form: the tau-forward variance from regime (f,s) is the tau-average of the
expected regime variance under powers of the marginal regime transition Pbar = sum_l w_l Tf(l) x Ts(l);
VIXvov(tau) is the stationary-pi dispersion of sqrt(that).  Finite sum over the GH grid, no MC.

Run from poc/ :  python3 identifiability_2f_vix.py
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from discslv_2f import stationary_pi
from identifiability_2f import kernel, obs, XSTAR, NAMES, OBS

TAUS = {"vixvov30": 30.0 / 365.0, "vixvov90": 90.0 / 365.0}


def vix_vov(K, tau):
    """Closed-form VIX vol-of-vol at horizon tau: stationary-pi std of the tau-forward vol."""
    NF, NS = K.n_f, K.n_s
    M = NF * NS
    # annualized regime variance v_{f,s} = E_l[V_{f,s,l}] / dt, flat index f*NS+s
    v = np.array([(K.wl * K.Vl[f, s]).sum() / K.dt for f in range(NF) for s in range(NS)])
    # marginal regime transition Pbar[(f,s) -> (f',s')] = sum_l w_l Tf(l,f) (x) Ts(l,s)
    Pbar = np.zeros((M, M))
    for f in range(NF):
        for s in range(NS):
            i = f * NS + s
            for l in range(K.n_l):
                Pbar[i] += K.wl[l] * np.outer(K.Tf[l, f], K.Ts[l, s]).ravel()
    N = max(1, int(round(tau / K.dt)))                       # steps in the tau window
    acc = np.zeros(M); cur = v.copy()
    for _ in range(N):                                       # acc = sum_{k=0}^{N-1} E[v at step k]
        acc += cur; cur = Pbar @ cur
    vix = np.sqrt(np.maximum(acc / N, 1e-12))                # tau-forward vol per starting regime
    pi = stationary_pi(K).ravel()
    m = float((pi * vix).sum())
    return float(np.sqrt((pi * (vix - m) ** 2).sum()))       # stationary dispersion = vol-of-vol


def vixvec(x):
    K = kernel(x)
    return np.array([vix_vov(K, t) for t in TAUS.values()])


def elasticity(J, O):
    return J * np.abs(XSTAR)[None, :] / (np.abs(O)[:, None] + 1e-9)


def report(tag, Jt):
    U, sv, Vt = np.linalg.svd(Jt)
    eff = int((sv > 1e-2 * sv[0]).sum())
    print(f"\n[{tag}]  sigma = " + "  ".join(f"{s:.3f}" for s in sv))
    print(f"        eff_rank(1%) {eff}/9    cond {sv[0]/sv[-1]:.0f}")
    soft = sorted(zip(NAMES, Vt[-1]), key=lambda t: -abs(t[1]))[:5]
    print("        softest vec: " + "  ".join(f"{n}:{v:+.2f}" for n, v in soft))
    return sv


def main():
    print("Building the 9-obs Jacobian once, plus the cheap VIX-vov rows ...", flush=True)
    O9 = obs(XSTAR)
    vix0 = vixvec(XSTAR)
    print("  O9(theta*)   =", dict(zip(OBS, np.round(O9, 3))), flush=True)
    print(f"  VIXvov@30d/90d = {vix0[0]:.4f} / {vix0[1]:.4f}", flush=True)

    J9 = np.zeros((9, 9)); Jv = np.zeros((2, 9))
    for j in range(9):
        h = 0.03 * max(abs(XSTAR[j]), 0.1)
        xp = XSTAR.copy(); xp[j] += h
        xm = XSTAR.copy(); xm[j] -= h
        J9[:, j] = (obs(xp) - obs(xm)) / (2 * h)             # expensive (smiles)
        Jv[:, j] = (vixvec(xp) - vixvec(xm)) / (2 * h)       # cheap (regime grid)
        print(f"  d/d{NAMES[j]:<8} done", flush=True)

    svA = report("A  9 obs: vol/skew/curv/SSR", elasticity(J9, O9))
    J11 = np.vstack([J9, Jv]); O11 = np.concatenate([O9, vix0])
    svB = report("B  + VIXvov@30d,@90d (11 obs)", elasticity(J11, O11))

    print("\n=== soft-direction lift (sigma_8, sigma_9) ===")
    print(f"  sigma_8: {svA[7]:.4f} -> {svB[7]:.4f}   ({svB[7]/max(svA[7],1e-9):.1f}x)")
    print(f"  sigma_9: {svA[8]:.4f} -> {svB[8]:.4f}   ({svB[8]/max(svA[8],1e-9):.1f}x)")
    print(f"  eff_rank: {int((svA>1e-2*svA[0]).sum())}/9  ->  {int((svB>1e-2*svB[0]).sum())}/9")


if __name__ == "__main__":
    main()

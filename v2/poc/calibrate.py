"""
calibrate.py -- Algorithm 5 (Stage 2): joint calibration of the kernel's structural knobs.

Fit theta = (gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s) so the propagated chain reproduces,
at every maturity T_j (n_j = round(T_j/dt) kernel steps):

    ATM vol  Sigma_ATM(T_j)     <- statics (from SANOS marginals)
    ATM skew  dSigma/dlogK(T_j) <- statics
    SSR(T_j)  dSigma_ATM/dlogS / skew  <- dynamics target

This is the JOINT fit that makes the statics/dynamics decoupling EXACT: when the SSR target
changes, lam_mov moves to hit it and lam_skew (+ nu_f) co-adjust to hold the marginal skew,
so the statics stay put.  (The leading-order demo in ssr_demo.py could not do this -- moving
lam_mov alone drifted the skew.)

Demonstration (main): take the baseline kernel's OWN statics as the fixed target (so they are
in-family), then calibrate theta for three SSR levels (0.7x, 1.0x, 1.3x baseline).  Each
solution should reproduce the SAME (vol, skew) term structures but its OWN SSR term structure.

Run from poc/ :  python3 calibrate.py
"""
import numpy as np
import warnings
from scipy.optimize import least_squares
from discslv import TwoTimescaleKernel, recompress_joint, marginal_from_joint

warnings.filterwarnings("ignore")
DT = 1.0 / 26.0           # biweekly kernel step (fewer steps to reach each maturity)
NK = 6                    # recompression components per regime
NAMES = ["gbar", "nu_s", "nu_f", "lam_skew", "lam_mov", "eps_s"]
LO = np.array([np.log(0.005), 0.05, 0.05, -3.5, -25.0, 0.02])
HI = np.array([np.log(0.20), 1.20, 1.20, 0.00, 0.00, 0.60])
X0 = np.array([np.log(0.04), 0.45, 0.45, -1.2, -12.0, 0.10])


def kernel(x):
    return TwoTimescaleKernel(gbar=x[0], nu_s=x[1], nu_f=x[2], lam_skew=x[3],
                              lam_mov=x[4], eps_s=x[5], dt=DT)


def horizon_smile(K, n, z0=0.0):
    pb = K.slow_weights(z0)
    comps = [(pb[b], z0, 1e-4, b) for b in range(K.n_slow)]
    for _ in range(n):
        comps = recompress_joint(K.propagate(comps), NK, K.n_slow)
    return marginal_from_joint(comps)


def observables(K, ns, h=6e-3, dm=6e-3):
    """Return (atm_iv[], skew[], ssr[]) at each n in ns."""
    iv_, sk_, ssr_ = [], [], []
    for n in ns:
        T = n * K.dt
        g0 = horizon_smile(K, n, 0.0); F0 = g0.forward()
        iv = lambda g, F, k: float(g.implied_vol(F * k, T)[0])
        atm = iv(g0, F0, 1.0)
        sk = (iv(g0, F0, np.exp(dm)) - iv(g0, F0, np.exp(-dm))) / (2 * dm)
        gu, gd = horizon_smile(K, n, +h), horizon_smile(K, n, -h)
        dvol = (iv(gu, gu.forward(), 1.0) - iv(gd, gd.forward(), 1.0)) / (2 * h)
        iv_.append(atm); sk_.append(sk); ssr_.append(dvol / sk if abs(sk) > 1e-9 else np.nan)
    return np.array(iv_), np.array(sk_), np.array(ssr_)


def residuals(x, ns, tgt, w):
    iv, sk, ssr = observables(kernel(x), ns)
    return np.concatenate([w[0] * (iv - tgt["iv"]), w[1] * (sk - tgt["sk"]), w[2] * (ssr - tgt["ssr"])])


def calibrate(ns, tgt, w, x0=X0, max_nfev=120):
    res = least_squares(residuals, x0, bounds=(LO, HI), args=(ns, tgt, w),
                        diff_step=2e-2, max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
    return res.x


def main():
    Tdays = [30, 89, 180, 271, 361]
    ns = [max(1, int(round(t / 365.0 / DT))) for t in Tdays]
    w = np.array([200.0, 60.0, 20.0])                    # weights: prioritise statics, then SSR

    iv0, sk0, ssr0 = observables(kernel(X0), ns)          # baseline statics = the fixed target
    print("Algorithm 5: joint calibration (statics held fixed, SSR dialed)\n")
    print("fit maturities :", "  ".join(f"{t}d(n={n})" for t, n in zip(Tdays, ns)))
    print("target statics : ATM vol", np.round(iv0, 4))
    print("                 ATM skew", np.round(sk0, 3))
    print("baseline SSR    :", np.round(ssr0, 2), "\n")

    levels = {"low (0.7x)": 0.7, "mid (1.0x)": 1.0, "high (1.3x)": 1.3}
    sols, x_prev = {}, X0
    for name, f in levels.items():
        tgt = dict(iv=iv0, sk=sk0, ssr=f * ssr0)
        x = calibrate(ns, tgt, w, x0=x_prev)             # warm-start from previous solution
        x_prev = x; sols[name] = x

    # ---- report: same statics, different SSR ----
    print(f"{'SSR target':>12} | {'theta (gbar nu_s nu_f lam_skew lam_mov eps_s)':<44} | fit quality")
    print("-" * 100)
    for name, f in levels.items():
        x = sols[name]; iv, sk, ssr = observables(kernel(x), ns)
        th = f"{x[0]:5.2f} {x[1]:4.2f} {x[2]:4.2f} {x[3]:6.2f} {x[4]:6.2f} {x[5]:5.3f}"
        iv_err = np.sqrt(np.mean((iv - iv0) ** 2)) * 1e4
        sk_err = np.sqrt(np.mean((sk - sk0) ** 2))
        ssr_err = np.sqrt(np.mean((ssr - f * ssr0) ** 2))
        print(f"{name:>12} | {th:<44} | vol {iv_err:4.0f}bp  skew {sk_err:.3f}  SSRrmse {ssr_err:.3f}")

    print("\nAchieved SSR term structures (the controllable dynamics):")
    print(f"{'maturity':>12} |" + "".join(f"{t}d".rjust(8) for t in Tdays))
    for name, f in levels.items():
        _, _, ssr = observables(kernel(sols[name]), ns)
        print(f"{name:>12} |" + "".join(f"{v:8.2f}" for v in ssr))
    print(f"{'(target mid)':>12} |" + "".join(f"{v:8.2f}" for v in ssr0))

    print("\nStatics check -- vol & skew across the three solutions (should coincide = decoupled):")
    print(f"{'maturity':>12} |" + "".join(f"{t}d".rjust(8) for t in Tdays))
    for name in levels:
        iv, sk, _ = observables(kernel(sols[name]), ns)
        print(f"{name+' vol':>12} |" + "".join(f"{v:8.4f}" for v in iv))
    for name in levels:
        iv, sk, _ = observables(kernel(sols[name]), ns)
        print(f"{name+' skew':>12} |" + "".join(f"{v:8.3f}" for v in sk))

    # ---- plot: SSR diverges (controllable), skew coincides (statics held) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ty = np.array(Tdays) / 365.0
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for name in levels:
            iv, sk, ssr = observables(kernel(sols[name]), ns)
            a1.plot(Ty, ssr, "o-", label=f"{name}")
            a2.plot(Ty, sk, "o-", label=f"{name}")
        a1.set_ylabel("SSR"); a1.legend(); a1.grid(alpha=0.3)
        a1.set_title("Algorithm 5: SSR dialed (controllable dynamics)")
        a2.set_ylabel("ATM skew"); a2.set_xlabel("maturity (years)"); a2.grid(alpha=0.3)
        a2.set_title("marginal skew coincides  ->  statics held fixed (exact decoupling)")
        fig.tight_layout(); fig.savefig("calibrate_decoupling.png", dpi=130)
        print("\nsaved plot -> poc/calibrate_decoupling.png")
    except Exception as ex:
        print(f"\n(plot skipped: {ex})")


if __name__ == "__main__":
    main()

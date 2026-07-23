"""
ssr_demo.py -- the controllable-SSR demonstration (SANOS-Evolve dynamics layer).

The marginals (statics) come from SANOS (here the synthetic stand-in via data_port).
The two-timescale kernel adds the DYNAMICS. This script shows three things:

  1. TERM STRUCTURE  -- the model's SSR (skew-stickiness ratio) across maturities emerges
                        with the right SPX-like shape: high (~1.9) at the short end,
                        decaying toward ~1 at one year.
  2. CONTROLLABILITY -- lam_mov sets the SSR LEVEL; eps_s (slow-factor persistence) sets the
                        DECAY SLOPE. The dynamics are dial-able independently of the statics.
  3. DECOUPLING (honest) -- at the ATM/short-step level lam_mov is a pure response knob
                        (marginal-neutral, since lam_mov * zeta * 0 = 0). In the MULTI-STEP
                        discrete model a weak secondary coupling remains (collocation at the
                        drifted component means), so the marginal SKEW drifts as lam_mov moves.
                        Exact fixed-statics control needs a joint calibration (co-adjust lam_skew);
                        that is what Algorithm 5 (calibrate) does.

The SSR at horizon T is read off the n-step conditional smile (n = round(T/dt) kernel steps):
    SSR(T) = ( d Sigma_ATM / d log S ) / ( d Sigma / d log K )|_ATM ,
both by finite difference on the propagated mixture. The regime is initialised at the
leverage-tilted slow_weights(z0), so SSR(n=1) reproduces the kernel's own one-step ssr().

Run from poc/ :  python3 ssr_demo.py
"""
import numpy as np
import warnings
from discslv import TwoTimescaleKernel, recompress_joint, marginal_from_joint
from data_port import load_chain

warnings.filterwarnings("ignore")
DT = 1.0 / 52.0                       # weekly kernel step
NK = 8                                # recompression: components kept per regime

# baseline structural knobs (tuned to an SPX-like SSR term structure)
BASE = dict(gbar=np.log(0.04), nu_s=0.45, nu_f=0.45, lam_skew=-1.2, eps_s=0.10, dt=DT)
LAM_MOV_0 = -12.0


def spx_ssr_ref(T):
    """Approximate empirical SPX SSR term structure (anchors ~1.9@1wk -> ~1.05@1y)."""
    return 1.0 + 0.95 * np.exp(-np.asarray(T) / 0.20)


def horizon_smile(K, n, z0=0.0):
    """n-step conditional spot law from spot z0, regime init = leverage-tilted slow_weights(z0)."""
    pb = K.slow_weights(z0)
    comps = [(pb[b], z0, 1e-4, b) for b in range(K.n_slow)]
    for _ in range(n):
        comps = recompress_joint(K.propagate(comps), NK, K.n_slow)
    return marginal_from_joint(comps)


def ssr_at(K, n, h=6e-3, dm=6e-3):
    """SSR of the n-step smile = (d ATM vol / d log S) / (ATM skew)."""
    T = n * K.dt
    gu, gd, g0 = horizon_smile(K, n, +h), horizon_smile(K, n, -h), horizon_smile(K, n, 0.0)
    iv = lambda g, k: float(g.implied_vol(g.forward() * k, T)[0])
    dvol = (iv(gu, 1.0) - iv(gd, 1.0)) / (2 * h)
    skew = (iv(g0, np.exp(dm)) - iv(g0, np.exp(-dm))) / (2 * dm)
    return dvol / skew if abs(skew) > 1e-9 else float("nan")


def marginal_atm(K, n, dm=6e-3):
    """Unconditional (z0=0) marginal: ATM vol and ATM skew -- the 'statics' the SSR knob must not move."""
    T = n * K.dt
    g = horizon_smile(K, n, 0.0); F = g.forward()
    iv = lambda k: float(g.implied_vol(F * k, T)[0])
    return iv(1.0), (iv(np.exp(dm)) - iv(np.exp(-dm))) / (2 * dm)


def main():
    ch = load_chain("synthetic")
    T = np.array(ch["maturities"])
    n = np.maximum(1, np.round(T / DT).astype(int))
    lbl = [f"{int(round(t*365))}d" for t in T]

    # ---- 1+2: SSR term structure for three lam_mov (level control) ----
    lams = [LAM_MOV_0 + 4, LAM_MOV_0, LAM_MOV_0 - 4]    # -8, -12, -16
    curves = {lm: np.array([ssr_at(TwoTimescaleKernel(lam_mov=lm, **BASE), ni) for ni in n]) for lm in lams}
    ref = spx_ssr_ref(T)

    print("SANOS-Evolve : SSR term structure (synthetic SPX-like chain, dt=1/52)\n")
    hdr = "maturity |" + "".join(f"{l:>8}" for l in lbl)
    print(hdr); print("-" * len(hdr))
    for lm in lams:
        tag = "  (baseline)" if lm == LAM_MOV_0 else ""
        print(f"lam={lm:+5.0f} |" + "".join(f"{v:8.2f}" for v in curves[lm]) + tag)
    print(f"SPX  ref |" + "".join(f"{v:8.2f}" for v in ref))

    # ---- 2b: eps_s controls the decay slope ----
    print("\nDecay-slope control via eps_s (lam_mov = baseline):")
    for eps in [0.05, 0.10, 0.20]:
        bb = dict(BASE); bb["eps_s"] = eps
        cur = np.array([ssr_at(TwoTimescaleKernel(lam_mov=LAM_MOV_0, **bb), ni) for ni in n])
        print(f"eps={eps:4.2f} |" + "".join(f"{v:8.2f}" for v in cur))

    # ---- 3: honest decoupling check at one maturity ----
    j = int(np.argmin(np.abs(T - 0.25)))                # ~3m
    print(f"\nDecoupling check at {lbl[j]} -- does the SSR knob (lam_mov) disturb the statics?")
    print("lam_mov |  SSR    ATM_iv   ATM_skew")
    for lm in lams:
        K = TwoTimescaleKernel(lam_mov=lm, **BASE)
        s = ssr_at(K, n[j]); iv, sk = marginal_atm(K, n[j])
        print(f"{lm:+6.0f}  | {s:5.2f}  {iv:7.4f}  {sk:+7.4f}")
    print("  -> ATM vol ~invariant (clean), ATM skew drifts (leading-order decoupling); a joint")
    print("     calibration co-adjusts lam_skew to hold the marginal exactly (Algorithm 5).")

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lm in lams:
            ax.plot(T, curves[lm], "o-", label=f"model  lam_mov={lm:+.0f}")
        ax.plot(T, ref, "k--", lw=2, label="SPX reference")
        ax.set_xlabel("maturity (years)"); ax.set_ylabel("SSR")
        ax.set_title("SANOS-Evolve: controllable SSR term structure"); ax.legend(); ax.grid(alpha=0.3)
        out = "ssr_term_structure.png"; fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\nsaved plot -> poc/{out}")
    except Exception as ex:
        print(f"\n(plot skipped: {ex})")


if __name__ == "__main__":
    main()

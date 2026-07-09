"""
soft_dir_band.py -- the precise statics/dynamics decoupling band at nk=24, the rigorous way.

Compute (nk=24) the Jacobian of the static smile (vol1m, vol1y, skew1m) and of SSR(1y) w.r.t. all 9
knobs.  Project the SSR(1y) gradient onto the static null-space -> the steepest STATIC-PRESERVING SSR
direction u.  Line-search along +/- u, Newton-correcting the statics back with the static pseudo-
inverse, and record SSR(1y) while the statics stay held.  The SSR(1y) range over held steps is the
decoupling band -- the Step-6 soft direction made concrete (all 9 knobs, nk=24).

Run from poc/ :  python3 soft_dir_band.py   (~6-8 min)
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from discslv_2f import TwoFactorSV, ssr_2f

DT = 1.0 / 52.0; NK = 24
NAMES = ["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
B = np.array([-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])
H = np.array([0.08, 0.05, 0.05, 0.04, 0.10, 0.08, 0.10, 0.08, 0.12])
LO = np.array([np.log(0.002), 0.05, 0.05, 0.05, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([np.log(0.15), 1.2, 1.2, 1.0, 0.0, 8.0, 8.0, 2.0, 4.0])


def kern(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], nu_l=x[3], lam_skew=x[4],
                       lam_f=x[5], lam_s=x[6], kap_f=x[7], kap_s=x[8], dt=DT, n_f=5, n_s=3, n_l=5)


def obs(x):                                   # statics=(vol1m,vol1y,skew1m), and SSR(1y)
    K = kern(x); _, v1, k1 = ssr_2f(K, 4, nk=NK); r4, v4, _ = ssr_2f(K, 52, nk=NK)
    return np.array([v1, v4, k1]), r4


def main():
    st0, ssr0 = obs(B)
    print(f"baseline (nk={NK}): statics(vol1m,vol1y,sk1m)={np.round(st0,4)}  SSR1y={ssr0:.3f}", flush=True)

    Js = np.zeros((3, 9)); gS = np.zeros(9)
    for j in range(9):
        xp = B.copy(); xp[j] += H[j]; xm = B.copy(); xm[j] -= H[j]
        sp, rp = obs(xp); sm, rm = obs(xm)
        Js[:, j] = (sp - sm) / (2 * H[j]); gS[j] = (rp - rm) / (2 * H[j])
        print(f"  jacobian col {NAMES[j]:9s} done", flush=True)

    Jpinv = np.linalg.pinv(Js)                       # 9x3
    P = np.eye(9) - Jpinv @ Js                        # static null-space projector
    pg = P @ gS; rate = np.linalg.norm(pg); u = pg / rate
    print(f"\n||proj(grad SSR1y) on static null-space|| = {rate:.4f}  (SSR1y change per unit static-preserving knob step)", flush=True)
    print("steepest static-preserving direction (knob loadings):", flush=True)
    for nm, v in sorted(zip(NAMES, u), key=lambda t: -abs(t[1])):
        if abs(v) > 0.08:
            print(f"    {nm:9s} {v:+.2f}", flush=True)

    def hold_eval(alpha):
        x = np.clip(B + alpha * u, LO, HI)
        for _ in range(4):                            # Newton-restore statics (converge skew too)
            st, _ = obs(x); x = np.clip(x - Jpinv @ (st - st0), LO, HI)
        st, ssr = obs(x); return st, ssr

    print("\nline-search along +/- u (statics Newton-held, 4 steps):", flush=True)
    band = []
    for a in [-0.7, -0.5, -0.35, -0.2, 0.0, 0.15, 0.3, 0.45]:
        st, ssr = hold_eval(a)
        verr = max(abs(st[0] - st0[0]), abs(st[1] - st0[1])) * 1e4; kerr = abs(st[2] - st0[2])
        held = verr < 15 and kerr < 0.02
        if held:
            band.append(ssr)
        print(f"  alpha={a:+.2f}: SSR1y={ssr:.3f}   held(vol {verr:3.0f}bp, sk {kerr:.3f}) = {held}", flush=True)

    print(f"\nFULLY-TIGHT BAND  SSR(1y) @ held statics (nk=24, 9 knobs): "
          f"[{min(band):.3f}, {max(band):.3f}]   width {max(band)-min(band):.3f}", flush=True)


if __name__ == "__main__":
    main()

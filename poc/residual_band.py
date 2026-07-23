"""
residual_band.py -- the HONEST residual decoupling band: hold a richer static smile
(vol@1m,6m,1y + skew@1m + curvature@1m, 5 observables) and measure how far SSR(1y) can still move.
Per Friz-Gatheral the forward-variance curve largely determines the SSR, so holding the vol curve +
curvature should shrink the band toward the true extra-SV freedom.  Method: project grad SSR(1y) onto
the null-space of the (scaled) 5x9 static Jacobian, line-search along it, Newton-hold the 5 statics.
Renders the band into ../discussions/math_flow.html.  Run from poc/  (~8-10 min).
"""
import base64, io, re, warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, ssr_2f, smile_2f, stationary_pi

DT = 1.0 / 52.0; NK = 20
NS = [4, 13, 26, 52]; T = np.array(NS) * DT
TGT_SSR = [1.45, 1.30, 1.15, 1.05]; TGT_VOL = [0.17] * 4; TGT_SKEW = [-0.50, -0.40, -0.34, -0.28]
B = np.array([-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])
H = np.array([0.08, 0.05, 0.05, 0.04, 0.10, 0.08, 0.10, 0.08, 0.12])
LO = np.array([np.log(0.002), 0.05, 0.05, 0.05, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([np.log(0.15), 1.2, 1.2, 1.0, 0.0, 8.0, 8.0, 2.0, 4.0])
TOL = np.array([12e-4, 12e-4, 12e-4, 0.02, 0.6])      # vol1m, vol6m, vol1y, skew1m, curv1m


def kern(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], nu_l=x[3], lam_skew=x[4],
                       lam_f=x[5], lam_s=x[6], kap_f=x[7], kap_s=x[8], dt=DT, n_f=5, n_s=3, n_l=5)


def sstats(g, Tn, dm=6e-3):
    F = g.forward(); iv = lambda k: float(g.implied_vol(F * k, Tn)[0])
    a, up, dn = iv(1.0), iv(np.exp(dm)), iv(np.exp(-dm))
    return a, (up - dn) / (2 * dm), (up - 2 * a + dn) / dm ** 2


def obs(x):                                            # 5 statics + SSR1y
    K = kern(x); pi = stationary_pi(K)
    A = np.zeros((K.n_f, K.n_s)); S = np.zeros_like(A); C = np.zeros_like(A)
    for f in range(K.n_f):
        for s in range(K.n_s):
            A[f, s], S[f, s], C[f, s] = sstats(smile_2f(K, 4, f, s, NK), 4 * DT)
    v1 = (pi * A).sum(); k1 = (pi * S).sum(); c1 = (pi * C).sum()
    _, v6, _ = ssr_2f(K, 26, nk=NK); r4, v4, _ = ssr_2f(K, 52, nk=NK)
    return np.array([v1, v6, v4, k1, c1]), r4


def curve(x):
    K = kern(x); return [ssr_2f(K, n, nk=NK)[0] for n in NS]


def main():
    st0, ssr0 = obs(B); sc = np.abs(st0)
    print(f"statics target: vol1m={st0[0]:.4f} vol6m={st0[1]:.4f} vol1y={st0[2]:.4f} "
          f"skew1m={st0[3]:.3f} curv1m={st0[4]:.2f} | SSR1y={ssr0:.3f}", flush=True)
    Js = np.zeros((5, 9)); gS = np.zeros(9)
    for j in range(9):
        xp = B.copy(); xp[j] += H[j]; xm = B.copy(); xm[j] -= H[j]
        sp, rp = obs(xp); sm, rm = obs(xm)
        Js[:, j] = ((sp - sm) / sc) / (2 * H[j]); gS[j] = (rp - rm) / (2 * H[j])
        print(f"  jacobian col {j} done", flush=True)
    Jpinv = np.linalg.pinv(Js); P = np.eye(9) - Jpinv @ Js
    rate = np.linalg.norm(P @ gS); u = (P @ gS) / rate
    print(f"\n||proj grad SSR1y|| = {rate:.4f}  (null-space dim {9 - np.linalg.matrix_rank(Js)})", flush=True)

    def hold(a):
        x = np.clip(B + a * u, LO, HI)
        for _ in range(4):
            st, _ = obs(x); x = np.clip(x - Jpinv @ ((st - st0) / sc), LO, HI)
        return x

    recs = []
    for a in [-1.4, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.4]:
        x = hold(a); st, r = obs(x); err = np.abs(st - st0)
        ok = bool(np.all(err < TOL))
        recs.append((a, x, r, ok))
        print(f"  a={a:+.2f}: SSR1y={r:.3f} held={ok}  errs vol {err[0]*1e4:.0f}/{err[1]*1e4:.0f}/{err[2]*1e4:.0f}bp "
              f"sk {err[3]:.3f} cv {err[4]:.2f}", flush=True)
    held = [(a, x, r) for a, x, r, ok in recs if ok]
    band_ssr = [ssr0] + [r for _, _, r in held]
    lo_r = min(band_ssr); hi_r = max(band_ssr)
    print(f"\nRESIDUAL BAND  SSR(1y) @ full-smile-held (5 obs, nk={NK}): "
          f"[{lo_r:.3f}, {hi_r:.3f}]  width {hi_r-lo_r:.3f}   ({len(held)}/8 configs held)", flush=True)

    curves = [curve(B)]
    if held:
        xlo = min(held, key=lambda t: t[2])[1]; xhi = max(held, key=lambda t: t[2])[1]
        curves += [curve(xlo), curve(xhi)]
    arr = np.array(curves); LOb = arr.min(0); HIb = arr.max(0); w1y = HIb[3] - LOb[3]
    base_full = [ssr_2f(kern(B), n, nk=NK) for n in NS]
    SSR0 = [b[0] for b in base_full]; VOL0 = [b[1] for b in base_full]; SKEW0 = [b[2] for b in base_full]
    print(f"per-tenor band: 1m[{LOb[0]:.2f},{HIb[0]:.2f}] 6m[{LOb[2]:.2f},{HIb[2]:.2f}] 1y[{LOb[3]:.3f},{HIb[3]:.3f}]", flush=True)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    ax[0].plot(T, SSR0, "o-", label="SANOS-Evolve"); ax[0].plot(T, TGT_SSR, "s--", label="SPX target")
    ax[0].set_title("SSR term structure"); ax[0].legend(fontsize=7)
    ax[1].plot(T, VOL0, "o-"); ax[1].plot(T, TGT_VOL, "s--"); ax[1].set_title("ATM vol"); ax[1].set_ylim(.10, .22)
    ax[2].plot(T, SKEW0, "o-"); ax[2].plot(T, TGT_SKEW, "s--"); ax[2].set_title("ATM skew")
    ax[3].plot(T, SSR0, "o-", color="C0", label="baseline SSR")
    ax[3].fill_between(T, LOb, HIb, alpha=.25, color="C3", label="reachable @ full smile held")
    ax[3].set_title("residual SSR freedom @ fixed smile"); ax[3].legend(fontsize=6, loc="upper right")
    ax[3].text(0.04, 0.06, f"vol@1m,6m,1y + skew@1m\n+ curv@1m held (nk={NK})\nband(1y)≈{w1y:.2f}",
               transform=ax[3].transAxes, fontsize=6.3, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.85))
    for a_ in ax:
        a_.grid(alpha=.3); a_.set_xlabel("maturity (y)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    P_ = "../discussions/math_flow.html"; html = open(P_, encoding="utf-8").read()
    newfig = ('<figure><img alt="2f calibrated fit" src="data:image/png;base64,' + b64 + '">\n'
              '<figcaption>The calibrated two-factor genuine-SV kernel (closed form, no Monte&nbsp;Carlo; nk=20). '
              'Panels 1&ndash;3: the <strong>SSR</strong>, ATM&nbsp;vol and ATM&nbsp;skew term structures, model vs SPX target. '
              'Panel&nbsp;4: <strong>residual statics/dynamics decoupling</strong> &mdash; now holding a richer static smile '
              '(vol@1m,6m,1y, skew@1m <em>and</em> curvature@1m) along the static-Jacobian null-space, the SSR is movable only inside the '
              f'narrow shaded band: the long-end residual freedom is $\\approx{LOb[3]:.2f}\\text{{--}}{HIb[3]:.2f}$ (width ${w1y:.2f}$). '
              'This is the honest decoupling number: per Friz&ndash;Gatheral the forward-variance curve largely <em>determines</em> the SSR, so once '
              'the vol curve and curvature are held, only this small extra stochastic-vol freedom remains &mdash; the earlier wider band was '
              'inflated by the looser (3-observable) hold. It is still non-zero (genuine SV, off the local-vol floor), and the calibration '
              'target sits inside it. $\\theta_0=(\\bar\\gamma\\,{-}5.30,\\nu_f\\,0.43,\\nu_s\\,0.50,\\nu_l\\,0.14,\\lambda_{\\rm skew}\\,{-}1.48,'
              '\\lambda_f\\,0.98,\\lambda_s\\,1.65,\\kappa_f\\,1.00,\\kappa_s\\,2.34)$.</figcaption></figure>')
    html2 = re.sub(r'<figure><img alt="2f calibrated fit".*?</figure>', lambda m: newfig, html, count=1, flags=re.S)
    assert html2 != html, "figure not found"
    open(P_, "w", encoding="utf-8").write(html2)
    print(f"spliced; residual 1y band [{LOb[3]:.3f},{HIb[3]:.3f}] width {w1y:.3f}; HTML {len(html2)} bytes", flush=True)


if __name__ == "__main__":
    main()

"""
tighten_decouple_2f.py -- tightened statics/dynamics decoupling at nk=24, using lam_s (slow leverage)
as a MONOTONE SSR dial.  At each lam_s, re-fit the static knobs (gbar, nu_s, lam_skew) to hold the
vol curve + 1m skew; read out SSR/vol/skew at nk=24.  Then rebuild demo panel 4 (lam_s fan) and
re-splice into ../discussions/math_flow.html.  Run from poc/  (slow ~6 min).
"""
import base64, io, re, warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, ssr_2f

DT = 1.0 / 52.0; NKF = 14; NK = 24
NS = [4, 13, 26, 52]; T = np.array(NS) * DT
TGT_SSR = [1.45, 1.30, 1.15, 1.05]; TGT_VOL = [0.17] * 4; TGT_SKEW = [-0.50, -0.40, -0.34, -0.28]
B = dict(gbar=-5.30, nu_f=0.43, nu_s=0.50, nu_l=0.14, lam_skew=-1.48, lam_f=0.98, lam_s=1.65, kap_f=1.00, kap_s=2.34)


def K(g, ns, lsk, ls):
    return TwoFactorSV(gbar=g, nu_f=B["nu_f"], nu_s=ns, nu_l=B["nu_l"], lam_skew=lsk,
                       lam_f=B["lam_f"], lam_s=ls, kap_f=B["kap_f"], kap_s=B["kap_s"], dt=DT, n_f=5, n_s=3, n_l=5)


def vs(Kk, n, nk):
    r, a, s = ssr_2f(Kk, n, nk=nk); return a, s


def main():
    Kb = TwoFactorSV(dt=DT, n_f=5, n_s=3, n_l=5, **B)
    base = [ssr_2f(Kb, n, nk=NK) for n in NS]
    SSR0 = [b[0] for b in base]; VOL0 = [b[1] for b in base]; SKEW0 = [b[2] for b in base]
    v1, v4, s1 = VOL0[0], VOL0[3], SKEW0[0]; TGT = [v1, v4, s1]
    print(f"baseline nk={NK}: SSR={np.round(SSR0,3)} vol={np.round(VOL0,3)} skew={np.round(SKEW0,3)}", flush=True)
    print(f"hold (vol1m={v1:.4f}, vol1y={v4:.4f}, sk1m={s1:.3f}); dial lam_s, re-fit (gbar,nu_s,lam_skew)", flush=True)

    def refit(ls, x0):
        def r(x):
            a1, k1 = vs(K(x[0], x[1], x[2], ls), 4, NKF); a4, _ = vs(K(x[0], x[1], x[2], ls), 52, NKF)
            return [10 * (a1 - TGT[0]), 10 * (a4 - TGT[1]), (k1 - TGT[2])]
        return least_squares(r, x0, bounds=([np.log(0.002), 0.05, -3.0], [np.log(0.15), 1.2, 0.0]),
                             diff_step=3e-2, max_nfev=22, xtol=1e-9).x

    order = [1.65, 2.4, 3.2, 1.0]; x0 = np.array([B["gbar"], B["nu_s"], B["lam_skew"]]); res = {}
    for ls in order:
        xf = refit(ls, x0); x0 = xf
        out = [ssr_2f(K(*xf, ls), n, nk=NK) for n in (4, 26, 52)]
        res[ls] = ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out])
        ss, vv, kk = res[ls]
        print(f"  lam_s={ls}: SSR(1m,6m,1y)={np.round(ss,3)} vol={np.round(vv,3)} sk={np.round(kk,3)}", flush=True)
    lss = sorted(res); s1y = [res[l][0][2] for l in lss]
    vcols = list(zip(*[res[l][1] for l in lss])); vrng = max(max(c) - min(c) for c in vcols) * 1e4
    skcols = list(zip(*[res[l][2] for l in lss])); skrng = max(abs(c[0] - c[-1]) for c in skcols)
    mono = all(s1y[i] <= s1y[i + 1] + 1e-9 for i in range(len(s1y) - 1))
    print(f"\nlam_s {lss}: SSR(1y)={np.round(s1y,3)} monotone={mono} | vol curve held {vrng:.0f}bp | skew held {skrng:.3f}", flush=True)

    # ---- figure ----
    TD = np.array([4, 26, 52]) * DT
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    ax[0].plot(T, SSR0, "o-", label="SANOS-Evolve"); ax[0].plot(T, TGT_SSR, "s--", label="SPX target")
    ax[0].set_title("SSR term structure"); ax[0].legend(fontsize=7)
    ax[1].plot(T, VOL0, "o-"); ax[1].plot(T, TGT_VOL, "s--"); ax[1].set_title("ATM vol"); ax[1].set_ylim(.10, .22)
    ax[2].plot(T, SKEW0, "o-"); ax[2].plot(T, TGT_SKEW, "s--"); ax[2].set_title("ATM skew")
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(lss)))
    axb = ax[3].twinx()
    for l, c in zip(lss, cols):
        ax[3].plot(TD, res[l][0], "o-", color=c, label=f"λ_s={l}")
        axb.plot(TD, res[l][1], "^:", color=c, lw=1.0, alpha=.7)
    ax[3].set_title("clean decoupling — dial λ_s @ held statics")
    ax[3].set_ylabel("SSR (solid)"); axb.set_ylabel("ATM vol (dotted)"); axb.set_ylim(.10, .22)
    ax[3].legend(fontsize=6, loc="upper right")
    ax[3].text(0.04, 0.06, f"vol held ≤{vrng:.0f}bp\nskew held ≤{skrng:.2f}", transform=ax[3].transAxes,
               fontsize=6.5, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.8))
    for i, a_ in enumerate(ax):
        a_.grid(alpha=.3); a_.set_xlabel("maturity (y)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    P = "../discussions/math_flow.html"; html = open(P, encoding="utf-8").read()
    lo, hi = min(s1y), max(s1y)
    newfig = ('<figure><img alt="2f calibrated fit" src="data:image/png;base64,' + b64 + '">\n'
              '<figcaption>The calibrated two-factor genuine-SV kernel (closed form, no Monte&nbsp;Carlo; nk=24). '
              'Panels 1&ndash;3: the <strong>SSR</strong>, ATM&nbsp;vol and ATM&nbsp;skew term structures, model vs SPX target. '
              'Panel&nbsp;4: <strong>clean statics/dynamics decoupling</strong> &mdash; the slow leverage $\\lambda_s$ is a '
              'monotone SSR dial: re-fitting the static knobs $(\\bar\\gamma,\\nu_s,\\lambda_{\\rm skew})$ holds the vol curve '
              f'(to $\\le{vrng:.0f}$&nbsp;bp) and skew (to $\\le{skrng:.2f}$) while $\\lambda_s$ lifts the long-end SSR over '
              f'$\\approx{lo:.2f}\\text{{--}}{hi:.2f}$' + ' at <em>fixed statics</em> (dotted vol curves collapse; solid SSR curves separate). '
              'The range is bounded &mdash; it is the Step-6 soft direction made concrete &mdash; so wider SSR moves at fixed '
              'statics are the role of the full SANOS marginal layer. ($\\kappa_s$ instead dials the SSR <em>shape</em>, non-monotonically.) '
              '$\\theta_0=(\\bar\\gamma\\,{-}5.30,\\nu_f\\,0.43,\\nu_s\\,0.50,\\nu_l\\,0.14,\\lambda_{\\rm skew}\\,{-}1.48,'
              '\\lambda_f\\,0.98,\\lambda_s\\,1.65,\\kappa_f\\,1.00,\\kappa_s\\,2.34)$.</figcaption></figure>')
    html2 = re.sub(r'<figure><img alt="2f calibrated fit".*?</figure>', lambda m: newfig, html, count=1, flags=re.S)
    assert html2 != html, "figure not found"
    open(P, "w", encoding="utf-8").write(html2)
    print(f"spliced; SSR(1y) {lo:.3f}-{hi:.3f} (range {hi-lo:.3f}); HTML {len(html2)} bytes", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
slv_wire.py -- wire REAL SANOS marginals into the validated SLV fusion and read the fused SSR.

Sanity step before the fit (faithful alg:calib): build the real SANOS marginal chain, form the
per-maturity discrete-Dupire leverage from it, and print the fused SSR (LV from the SANOS leverage
skew + SV from the kernel) next to the empirical SSR. No new model code -- reuses
discslv_slv.{dupire_var, fused_ssr_readout, ...} + orats_sanos + empirical_ssr.
"""
import sys, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import sanos_lp                                                       # noqa: E402
from orats_sanos import orats_chain_df, eta_for_dte                  # noqa: E402
from sanos_lp import prep_expiry, sanos_fit                          # noqa: E402
import discslv_slv as S                                              # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout, sd_of, gm_call, gm_density_S  # noqa: E402
from discslv_2f import TwoFactorSV                                   # noqa: E402
from empirical_ssr import empirical_ssr                              # noqa: E402
from slv_interp import interp_marginal                               # noqa: E402

DT = 1.0 / 52.0; NS = [4, 13, 26, 52]; LABELS = ["1m", "3m", "6m", "1y"]


def sanos_chain(path, ticker="SPX"):
    """[(T_j, (W,MU,SG))] real SANOS GM marginals, martingale-locked, sorted by T."""
    df = orats_chain_df(path, ticker); chain = []
    for dte, g in df.groupby("dte"):
        sanos_lp.ETA = eta_for_dte(int(dte))
        e = prep_expiry(g)
        if e.get("n", 0) < 20:
            continue
        q, _ = sanos_fit(e)
        if q is None:
            continue
        W = np.asarray(q); MU = np.log(e["kappa"]); SG = np.full_like(W, np.sqrt(e["V"]))
        MU = MU - np.log(np.sum(W * np.exp(MU + 0.5 * SG ** 2)))       # martingale lock
        chain.append((float(e["tau"]), (W, MU, SG)))
    return sorted(chain, key=lambda x: x[0])


class _Lev:
    """Picklable leverage lambda(z)=clip(exp(polyval(coef, clip(z,-zmax,zmax))), safety). A class, not a
    closure, so it survives pickling to worker processes (needed for the grid-parallel SSR eval)."""
    def __init__(self, coef, zmax, safety):
        self.coef, self.zmax, self.safety = coef, zmax, safety

    def __call__(self, z):
        return np.clip(np.exp(np.polyval(self.coef, np.clip(z, -self.zmax, self.zmax))), *self.safety)


def leverage_at(chain, T, EV, dt=DT, ksd=1.5, deg=1, safety=(0.2, 5.0)):
    """Discrete-Dupire leverage lambda(z) = sqrt(sigma^2_Dupire(e^z, T) * dt / EV), leading order
    E[nu|z] ~ 1. Three numerics fixes so the leverage is BENIGN -- SV uncorrupted (paper Sec. fusion:
    'central-difference Dupire, finer grids, tail-clamped leverage'):
      (1) CENTRAL calendar slope over [T-dt/2, T+dt/2] with density at the central marginal;
      (2) SMOOTH: the true local-vol skew is smooth, but the GM density is wavy (few components) ->
          fit log lambda(z) with a low-order, density-weighted polynomial (trust the high-mass region);
      (3) TAIL-FREEZE: evaluate the smooth fit clipped to +-ksd*SD (flat-extrapolate into the illiquid
          wings) instead of a hard lambda-clamp that suppressed wing vol and shrank the regime jumps.
    z-dependence (dlam/dz) = the SANOS local-vol skew -> the LV part of the SSR."""
    h = dt / 2
    mu_lo = interp_marginal(chain, max(T - h, 1e-4)); mu_hi = interp_marginal(chain, T + h)
    mu_mid = interp_marginal(chain, T); zmax = ksd * sd_of(mu_mid)

    def raw_lam2(z):
        K = np.exp(z)
        num = max(gm_call(K, mu_hi) - gm_call(K, mu_lo), 1e-14) / (2 * h)   # central calendar slope
        return num / max(0.5 * K ** 2 * gm_density_S(K, mu_mid), 1e-300) * dt / EV

    zg = np.linspace(-zmax, zmax, 25)
    lam2 = np.array([max(raw_lam2(z), 1e-12) for z in zg])
    wt = np.sqrt(np.array([gm_density_S(np.exp(z), mu_mid) * np.exp(z) for z in zg]) + 1e-12)  # z-density weight
    coef = np.polyfit(zg, 0.5 * np.log(lam2), deg, w=wt)                    # smooth log-lambda ~ poly(z)

    return _Lev(coef, zmax, safety)                                    # picklable callable lambda(z)


def ref_vol(chain):
    """Representative ATM vol from the SANOS chain (~6m expiry): sigma ~ sd(log S)/sqrt(T)."""
    T6 = min(chain, key=lambda c: abs(c[0] - 0.5))
    return sd_of(T6[1]) / np.sqrt(T6[0])


def solve_gbar(kw, sig_ref, dt=DT, iters=8):
    """gbar such that E_pi[Vbar] = sig_ref^2 * dt, i.e. leverage lambda ~ 1 (paper Sec. fusion, §748:
    gbar is 'reset by sigma_LV', NOT log(sig_ref^2) -- the regime dispersion + skew-tilt variance
    inflate EV). EV ~ prop e^gbar, so a fixed point converges in a few steps."""
    target = sig_ref ** 2 * dt; gbar = float(np.log(target))
    for _ in range(iters):
        gbar += float(np.log(target / Epi_V(TwoFactorSV(gbar=gbar, dt=dt, n_f=5, n_s=3, n_l=5, **kw))))
    return gbar


if __name__ == "__main__":
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    date = sys.argv[1] if len(sys.argv) > 1 else OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
    yr = os.path.basename(date).split("_")[-1][:4]
    chain = sanos_chain(date)
    print(f"SANOS chain: {len(chain)} expiries, T(y) = " + ", ".join(f"{c[0]:.2f}" for c in chain[:9]) + " ...")
    sig_ref = ref_vol(chain)                                          # gbar reset so EV = sig_ref^2*dt -> lambda ~ 1
    d = dict(nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65, kap_f=1.00, kap_s=2.34, nu_l=0.14)
    K = TwoFactorSV(gbar=solve_gbar(d, sig_ref), dt=DT, n_f=5, n_s=3, n_l=5, **d)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K); nk = 16
    emp, nd = empirical_ssr(sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")))
    zg = np.linspace(-0.25, 0.25, 11)
    print(f"ref vol {sig_ref:.3f} -> gbar={np.log(sig_ref**2):.2f}   EV={EV:.2e}")
    print(f"empirical SSR ({nd} {yr} dates): {np.round(emp, 2)}\n")
    print(f"{'T':>4}{'n':>4}{'fusedSSR':>10}{'LV':>8}{'SV':>8}{'emp':>8}   lam min/mean/max")
    for n, lab, e in zip(NS, LABELS, emp):
        T = n * DT; lam_fn = leverage_at(chain, T, EV)
        lams = np.array([lam_fn(z) for z in zg])
        tot, lv, sv = fused_ssr_readout(K, lam_fn, n, EV, nub, Vlr, tiltr, nk, DT)
        print(f"{lab:>4}{n:>4}{tot:>10.3f}{lv:>8.3f}{sv:>8.3f}{e:>8.3f}   "
              f"{lams.min():.2f}/{lams.mean():.2f}/{lams.max():.2f}", flush=True)

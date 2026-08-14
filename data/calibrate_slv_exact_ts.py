#!/usr/bin/env python3
"""
Calibrate 1wk-3m with the EXACT beta AND the per-step TERM-STRUCTURE leverage (fused_ssr_exact_ts):
each propagation step k gets leverage_at((k+1)*dt) instead of one frozen leverage_at(T). The frozen-
leverage dense re-fit hit the belly to -16/-17% by pumping vol-of-vol (nu_f 0.87); bolting the term-
structure leverage on at that theta then OVERSHOT the short-mid (2wk +15%, 1m +7%) while nailing the
belly (2m +1%, 3m -5%) -- the tell that theta needs re-balancing AROUND the ts leverage. This re-fit
lets the optimizer dial the SV back down where the ts leverage now carries the belly, for (hopefully)
a flat error across the whole curve. The honest best-faithful-fit.

The leverage cache is THETA-INVARIANT: solve_gbar resets gamma-bar so EV = sigma_ref^2 * dt for every
theta (paper Sec.748), so leverage_at(., EV) is the same map every eval -> build it ONCE.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact_ts               # noqa: E402
discslv_slv.propagate = propagate_vec
from empirical_ssr import empirical_ssr                               # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]; NZ = 9
WREL = np.array([1.0, 1.0, 1.0, 1.0, 1.0])   # EQUAL % weight: residual = (model-emp)/emp -> equal % error target,
#   short end no longer sacrificed (the old belly-heavy [1,1.5,2,1.5,1] on ABSOLUTE residuals spent the well-
#   determined 1wk to buy the belly; with the belly now reachable, that was backwards).
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
LO = np.array([0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([1.20, 1.20, 1.50, 0.0, 8.0, 8.0, 1.0, 4.0])                  # nu_l HI 1.0->1.5 (2015 fit railed it)
X0_MAP = {                                                                 # multi-start: don't trust one basin
    "dense": np.array([0.872, 0.532, 0.831, -0.652, 0.489, 2.088, 0.769, 2.991]),   # high-nu (frozen dense re-fit)
    "low":   np.array([0.280, 0.293, 0.465, -2.113, 1.164, 2.460, 0.992, 2.518]),   # low-nu/high-skew (sparse fit, fit 1wk +2%)
    "ts":    np.array([0.696, 0.290, 0.999, -0.462, 0.439, 2.465, 0.903, 2.780]),   # 2015 dense+ts+equal-wt fit (well-behaved OOS start)
}
# ======================================================================================
# NORMALISED PARAMETERISATION (SANOS_REFERENCE.md 16.9). Added alongside; the vectors above
# still drive the old abscissa-indexed kernel.
#
#     u' = kap u + sqrt(1-kap^2) ( rho z_l + sqrt(1-rho^2) eps )      Var(u) = 1 by construction
#
# lam_f/lam_s (unbounded loadings, LO/HI 0..8) become BOUNDED correlations rho_f/rho_s, and kappa
# is now exactly the autocorrelation instead of a number the old chain realised only 0.9-95% of.
#
# kap_s is NOT FITTED. It is fixed at Bergomi's k2 ~ 0.23/yr. The reason is IDENTIFICATION, not
# representation: 16.9's kernel represents kappa_s = 0.9956 exactly and at the same cost as 0.5,
# so the 16.8 argument ("a chain cannot represent kappa >= 0.99") no longer applies. What does
# apply is 16.0's measurement -- a two-factor decay fit to the 9-12 vov targets is DEGENERATE in
# all nine years, because a 37-week window has no leverage on a ~150-week timescale. Left free,
# kap_s absorbs other misfits; that is how it reached > 1 in 8/9 years, which has no stationary
# limit at all. Fixing it also frees nu_s, which ranged 0.10-0.60 with no stable value while the
# two were entangled.
KAP_S_FIXED = 0.9956                      # Bergomi k2 ~ 0.23/yr, ~157wk
KAP_S_BRACKET = (0.98, 0.999)             # 16.0's robustness bracket: 2x in surviving weight at 260d

NAMES_N = ["nu_f", "nu_s", "nu_l", "lam_skew", "rho_f", "rho_s", "kap_f"]     # 7 fitted, not 8
LO_N = np.array([0.05, 0.05, 0.10, -3.0, 0.00, 0.00, 0.05])
HI_N = np.array([3.00, 3.00, 1.50,  0.0, 0.99, 0.99, 0.995])
#   nu_*  HI 1.2 -> 3.0: nu now multiplies a UNIT-variance factor, so it absorbs the sqrt(Var)
#         the old truncated chain silently swallowed. Remapped seeds sit at 0.46-1.38.
#   rho_* LO 0 preserves the old lam_* >= 0 constraint exactly (rho = lam/sqrt(1+lam^2) is monotone).
#   kap_f HI 1.0 -> 0.995: |kap| >= 1 has NO stationary law, and `s_f = sqrt(clamp(1-kap^2,1e-12))`
#         would silently produce a diverging factor rather than erroring. kap_f RAILED at the old
#         1.0 in 4/9 years, which 16.0 attributes to the truncation -- so whether it still rails
#         under the fixed kernel is a real test, not a nuisance.

# Seeds remapped from X0_MAP by MATCHING WHAT THE OLD CHAIN REALISED, not its nominal parameters.
# This matters: the old seeds were fitted AGAINST the truncated chain, so a formula-faithful remap
# (rho = lam/sqrt(1+lam^2), nu' = nu sqrt((1+lam^2)/(1-kap^2))) propagates the compensation --
# "low" would map to nu_f = 3.40 for a chain that only ever delivered ~0.56. So instead:
#     kap_f  <- the REALISED ac of the old n_f=5 chain      (e.g. ts: nominal 0.903 -> 0.834)
#     nu_f   <- nu_f_old * the REALISED sd                  (e.g. ts: 0.696 * 1.82 = 1.269)
#     rho_f  <- realised corr(z', z_l) / sqrt(1 - kap_f^2)
# kap_s has no stationary image at all (2.5-3.0 in every seed), so nu_s uses the realised slow sd
# and kap_s takes KAP_S_FIXED. Generating script: vix_joint_refit/remap_seeds.py
X0_MAP_N = {
    "dense": np.array([1.3847, 0.8176, 0.8310, -0.6520, 0.4270, 0.9019, 0.7372]),
    "low":   np.array([0.5855, 0.4618, 0.4650, -2.1130, 0.6752, 0.9264, 0.8113]),
    "ts":    np.array([1.2689, 0.4572, 0.9990, -0.4620, 0.3743, 0.9267, 0.8341]),
}


def theta9_n(x7, gbar, kap_s=None):
    """7 fitted params + solved gbar + the FIXED kap_s -> the 9-vector build_kernel_n consumes."""
    x7 = np.asarray(x7, float)
    return np.concatenate([[gbar], x7[:6], [x7[6]], [KAP_S_FIXED if kap_s is None else kap_s]])


# ---------------------------------------------------------------- 8-parameter variant: kap_s FITTED
# NAMES_N/LO_N/HI_N/X0_MAP_N are LEFT AT 7 ON PURPOSE. Eleven scripts zip NAMES_N against theta dicts
# in already-written fit JSONs, all of which have 7 keys -- widening the shared constant would raise
# KeyError on every recorded panel (ndx_panel_record, panel_reb_record, fit_norm_*_ct, every plot).
# The 8-param path is opt-in and additive; `_th9_n` and `fit_date` dispatch on len(theta).
#
# WHY FIT IT. kap_s was pinned at 0.9956 (157wk) on the grounds that a ~37-week window cannot
# identify a ~150-week timescale -- but that argues against HAVING a 157-week factor, not for pinning
# one there. Measured: a 5-rung ladder (0.90/0.95/0.98/0.99/0.9956) on SPX has an interior cost
# minimum at 0.98 (34wk), improving 8 of 9 years, mean 1.85x, so the direction is NOT degenerate with
# nu_s. And kap_s carries ~25x the vov-shape leverage of any other parameter (7.7% vs 0.3% per 2%
# perturbation) while being the only one held fixed.
NAMES_N8 = NAMES_N + ["kap_s"]
# LO 0.50 = 1.0wk half-life, HI 0.998 = 346wk.
# HI MUST BRACKET THE INCUMBENT: KAP_S_FIXED = 0.9956 is Bergomi's k2 ~ 0.23/yr (~157wk), so a bound
# below it would forbid the fit from returning to the literature anchor and bias the test against the
# value we are testing. 0.998 clears it with room. HI must still stay < 1: |kap| >= 1 has no
# stationary distribution (the old kap_s range [0.5, 4.0] is illegal here -- it was a softmax
# coefficient, not an autocorrelation).
# LO 0.50 spans v2's fitted slow half-lives (0.7-45wk) and fully overlaps the fitted kap_f range
# (0.40-0.92), so the two factors CAN swap roles -- the model is symmetric under
# (nu_f,rho_f,kap_f) <-> (nu_s,rho_s,kap_s). Harmless for the fit (identical mixture), but theta must
# be read with the factors SORTED before comparing across years.
LO_N8 = np.concatenate([LO_N, [0.50]])
HI_N8 = np.concatenate([HI_N, [0.998]])
# seeds start at the ladder optimum, not at the old pin
X0_MAP_N8 = {k: list(v) + [0.98] for k, v in X0_MAP_N.items()}


_CACHE = {}


def exact_vec_ts(x, chain, sig_ref, nz=NZ):
    kw = dict(zip(NAMES, x)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    out = []
    for n in NS:
        lam_fns = [_CACHE[k + 1] for k in range(n)]                  # step k -> maturity (k+1)*dt (theta-invariant)
        out.append(fused_ssr_exact_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, 16, DT, nz=nz)[0])
    return np.array(out)


if __name__ == "__main__":
    START = sys.argv[1] if len(sys.argv) > 1 else "dense"
    X0 = X0_MAP[START]
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    date = OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"; yr = "2015"
    chain = sanos_chain(date); sig_ref = ref_vol(chain)
    kw0 = dict(zip(NAMES, X0)); K0 = TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0)
    EV0 = Epi_V(K0)                                                   # theta-invariant EV (gamma-bar reset)
    _CACHE.update({k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, max(NS) + 1)})
    emp, nd = empirical_ssr(sorted(__import__("glob").glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=NS, dt=DT)
    print(f"START={START}  X0={np.round(X0,3)}", flush=True)
    print(f"empirical SSR ({nd} {yr}): {np.round(emp, 3)}  EQUAL % weight {WREL}  (EXACT beta + TERM-STRUCTURE leverage, weekly)", flush=True)
    t0 = time.time()
    res = least_squares(lambda x: WREL * (exact_vec_ts(x, chain, sig_ref) - emp) / emp, X0, bounds=(LO, HI),
                        diff_step=5e-2, max_nfev=60, xtol=1e-6, ftol=1e-6, verbose=2)
    mod = exact_vec_ts(res.x, chain, sig_ref, nz=15)
    print(f"\nEXACT-beta + TERM-STRUCTURE calibration ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in LABELS))
    print(f"{'ts':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{'emp':6}" + "".join(f"{v:8.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))

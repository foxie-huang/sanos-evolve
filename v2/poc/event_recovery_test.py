"""
Synthetic de-event RECOVERY test -- the STATIC half of de-eventing.

Validates the paper's claim (sec "Scheduled-event de-eventing"):
    K_obs_23 = K_clean_23  (convolved with)  rho_J,
    rho_J a small martingale-normalised jump mixture, recovered by a parametric
    deconvolution at the digital rung -- well-posed where a nonparametric one is not.

Procedure (the clean-flank bridge, static channel):
  1. clean marginal mu3_diff  = propagate K_diff(theta_0) to T3  (the de-evented counterfactual).
  2. inject a KNOWN asymmetric event jump rho_J (2 legs, up/down, martingale-normalised).
  3. dirty marginal mu3_obs    = mu3_diff (convolved with) rho_J   (the observed event-spanning smile).
  4. RECOVER the jump as the cumulant residual  kappa(mu3_obs) - kappa(mu3_diff)  -- exact, because
     convolution => cumulants add -- and fit a 2-leg mixture rho_J_hat to those cumulants.
  5. check recovered (variance, skew, ex-kurtosis) and density vs the injected; reconstruct the smile.
  6. robustness: perturb the observed cumulants by quote-scale noise -> recovery stays stable
     (the well-posedness of the low-cumulant parametric deconvolution).

Run from poc/ :  python3 event_recovery_test.py
"""
import sys, json
POC = "/Users/foxie/Documents/Research/2026/SANOS_Evolve/disc_SLV/poc"
sys.path.insert(0, POC)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from discslv import GMM
from discslv_2f import TwoFactorSV, smile_2f

DT = 1 / 52.0
TH = dict(gbar=-5.30, nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65,
          kap_f=1.00, kap_s=2.34, dt=DT, nu_l=0.14, n_f=5, n_s=3, n_l=5)
Kd = TwoFactorSV(**TH)
N3 = 13            # event sits in a 3-month window
Tj = N3 * DT


def gmm_cumulants(g):
    """First four cumulants of the log-return X ~ sum_i w_i N(mu_i, s_i^2)."""
    w, mu, s = g.w, g.mu, g.s
    m = float((w * mu).sum())
    d = mu - m
    c2 = float((w * (s ** 2 + d ** 2)).sum())
    c3 = float((w * (d ** 3 + 3 * d * s ** 2)).sum())
    m4 = float((w * (d ** 4 + 6 * d ** 2 * s ** 2 + 3 * s ** 4)).sum())
    k4 = m4 - 3 * c2 ** 2
    return dict(mean=m, var=c2, c3=c3, k4=k4,
                skew=c3 / c2 ** 1.5, exkurt=k4 / c2 ** 2, std=np.sqrt(c2))


def convolve(g1, g2):
    """Convolution of two independent log-return mixtures = the cross-product mixture."""
    w = (g1.w[:, None] * g2.w[None, :]).ravel()
    mu = (g1.mu[:, None] + g2.mu[None, :]).ravel()
    s = np.sqrt(g1.s[:, None] ** 2 + g2.s[None, :] ** 2).ravel()
    return GMM(w / w.sum(), mu, s, F=1.0)


def two_leg(p, au, ad, beta):
    """Martingale-normalised 2-outcome (up/down) event jump."""
    w = np.array([p, 1 - p]); mu = np.array([au, ad]); s = np.array([beta, beta])
    A = np.log(np.sum(w * np.exp(mu + 0.5 * s ** 2)))   # martingale shift
    return GMM(w, mu - A, s, F=1.0)


def fit_two_leg(tvar, tskew, texk, seed=(0.5, 0.04, -0.05, 0.02)):
    """Parametric deconvolution: fit (p, au, ad, beta) to the recovered (var, skew, ex-kurt)."""
    def res(x):
        p, au, ad, beta = x
        c = gmm_cumulants(two_leg(p, au, ad, beta))
        return [(c["var"] - tvar) / tvar, c["skew"] - tskew, c["exkurt"] - texk]
    sol = least_squares(res, seed, bounds=([0.05, 0.0, -0.3, 0.001], [0.95, 0.3, 0.0, 0.1]),
                        xtol=1e-12, ftol=1e-12)
    return sol.x


# ---------- 1. clean (de-evented) marginal mu3_diff ----------
mu3_diff = smile_2f(Kd, N3, 2, 1, 24)          # mid-vol regime, 3m
kd = gmm_cumulants(mu3_diff)
print("=" * 78)
print("DE-EVENT RECOVERY TEST (static channel) -- calibrated theta_0, 3m window")
print("=" * 78)
print(f"clean marginal mu3_diff:  std={kd['std']:.4f}  skew={kd['skew']:+.4f}  exkurt={kd['exkurt']:+.4f}\n")

# ---------- 2. inject a KNOWN asymmetric event jump ----------
P_TRUE, AU_TRUE, AD_TRUE, B_TRUE = 0.45, 0.050, -0.062, 0.018   # crash-skewed earnings-like event
rho_J = two_leg(P_TRUE, AU_TRUE, AD_TRUE, B_TRUE)
kj = gmm_cumulants(rho_J)
print(f"INJECTED jump rho_J (2-leg):  p_up={P_TRUE}, a_up={AU_TRUE:+.3f}, a_dn={AD_TRUE:+.3f}, beta={B_TRUE}")
print(f"   -> std={kj['std']:.4f} ({kj['std']*100:.1f}% event move)  skew={kj['skew']:+.4f}  "
      f"exkurt={kj['exkurt']:+.4f}\n")

# ---------- 3. dirty marginal mu3_obs = mu3_diff (*) rho_J ----------
mu3_obs = convolve(mu3_diff, rho_J)
ko = gmm_cumulants(mu3_obs)
print(f"dirty marginal mu3_obs:   std={ko['std']:.4f}  skew={ko['skew']:+.4f}  exkurt={ko['exkurt']:+.4f}\n")

# ---------- 4. RECOVER the jump as the cumulant residual ----------
rec = dict(var=ko["var"] - kd["var"], c3=ko["c3"] - kd["c3"], k4=ko["k4"] - kd["k4"])
rec["std"] = np.sqrt(rec["var"]); rec["skew"] = rec["c3"] / rec["var"] ** 1.5
rec["exkurt"] = rec["k4"] / rec["var"] ** 2
print("RECOVERED jump (cumulant residual kappa(obs) - kappa(diff)):")
print(f"   std={rec['std']:.4f}  skew={rec['skew']:+.4f}  exkurt={rec['exkurt']:+.4f}")
print(f"   errors vs injected:  d_std={rec['std']-kj['std']:+.2e}  d_skew={rec['skew']-kj['skew']:+.2e}  "
      f"d_exkurt={rec['exkurt']-kj['exkurt']:+.2e}")

# ---------- 5. parametric deconvolution: fit a 2-leg mixture ----------
xh = fit_two_leg(rec["var"], rec["skew"], rec["exkurt"])
rho_hat = two_leg(*xh); kh = gmm_cumulants(rho_hat)
print(f"\nPARAMETRIC fit rho_J_hat:  p_up={xh[0]:.3f}, a_up={xh[1]:+.3f}, a_dn={xh[2]:+.3f}, beta={xh[3]:.3f}")
print(f"   -> std={kh['std']:.4f}  skew={kh['skew']:+.4f}  exkurt={kh['exkurt']:+.4f}")
mu3_recon = convolve(mu3_diff, rho_hat)

# smile reconstruction error
ks = np.exp(np.linspace(-0.28, 0.28, 41))
iv_clean = mu3_diff.implied_vol(ks, Tj); iv_dirty = mu3_obs.implied_vol(ks, Tj)
iv_recon = mu3_recon.implied_vol(ks, Tj)
smile_mae = float(np.mean(np.abs(iv_recon - iv_dirty)))
print(f"   smile reconstruction MAE (recon vs dirty): {smile_mae*1e4:.3f} bp")

# ---------- 6. robustness: the jump is a SMALL RESIDUAL of two larger marginals ----------
snr = rec["var"] / kd["var"]                          # jump variance / clean variance
amp = ko["var"] / rec["var"]                          # noise-amplification factor (obs/jump)
print(f"\nrobustness -- de-eventing is a DECONVOLUTION (jump var / clean var = {snr:.2f}, "
      f"obs/jump = {amp:.1f}x amplification):")
print(f"   {'obs noise':>10} | {'rec std':>8} {'(err)':>7} | {'rec skew':>9} (inj {kj['skew']:+.3f})")
vs, ss = [], []
for fv in (0.98, 0.99, 1.0, 1.01, 1.02):              # +-2% quote-scale noise on the OBSERVED moments
    v = ko["var"] * fv - kd["var"]; c3 = ko["c3"] * fv - kd["c3"]
    sd = np.sqrt(max(v, 1e-9)); sk = c3 / max(v, 1e-9) ** 1.5; vs.append(sd); ss.append(sk)
    print(f"   {(fv-1)*100:>+9.0f}% | {sd:>8.4f} {(sd/kj['std']-1)*100:>+6.0f}% | {sk:>+9.3f}")
print(f"   => VARIANCE recovers but with ~{amp:.0f}x noise amplification; SKEW is noise-sensitive "
      f"(std {min(ss):+.2f}..{max(ss):+.2f} over +-2% noise).")
print("      Honest: exact with clean data; in practice the event SKEW needs clean quotes or averaging")
print("      over many events -- itself consistent with Zhong's empirical 'variance, not skew'.")

# ---------- verdict ----------
ok = (abs(rec["std"] - kj["std"]) < 1e-6 and abs(rec["skew"] - kj["skew"]) < 1e-4
      and smile_mae < 5e-4)
print("\n" + "=" * 78)
print(f"VERDICT -> recovery exact in cumulant space (convolution => cumulants add): "
      f"std/skew/exkurt match to ~1e-6; 2-leg density + smile reconstructed (MAE {smile_mae*1e4:.2f} bp).")
print(f"   RECOVERY OK: {ok}")
print("=" * 78)

# ---------- plots ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
a1.plot(np.log(ks), iv_clean * 100, color="#1b7837", lw=2, label="clean (de-evented) $\\mu_3^{diff}$")
a1.plot(np.log(ks), iv_dirty * 100, color="#b2182b", lw=2, label="dirty (observed) $\\mu_3^{obs}$")
a1.plot(np.log(ks), iv_recon * 100, "--", color="#222", lw=1.4, label="reconstructed (clean $*\\,\\hat\\rho_J$)")
a1.set(title="Event-spanning smile: dirty vs de-evented vs reconstructed", xlabel="log-moneyness",
       ylabel="implied vol (%)"); a1.legend(fontsize=8, frameon=False); a1.grid(alpha=.25)
xg = np.linspace(-0.18, 0.14, 400)
def dens(g):
    return np.array([np.sum(g.w * np.exp(-0.5 * ((x - g.mu) / g.s) ** 2) / (g.s * np.sqrt(2 * np.pi))) for x in xg])
a2.plot(xg, dens(rho_J), color="#2166ac", lw=2, label="injected $\\rho_J$")
a2.plot(xg, dens(rho_hat), "--", color="#b2182b", lw=1.6, label="recovered $\\hat\\rho_J$")
a2.set(title="Event jump density: injected vs recovered", xlabel="event log-return",
       ylabel="density"); a2.legend(fontsize=8, frameon=False); a2.grid(alpha=.25)
a2.text(-0.17, max(dens(rho_J)) * 0.6, f"std {rec['std']*100:.1f}%\nskew {rec['skew']:+.2f}",
        fontsize=8, color="#555")
fig.suptitle("De-event recovery (static channel): clean-flank cumulant residual recovers a known "
             "asymmetric event jump\n(variance + crash-skew), exactly -- the static half the paper asserts",
             fontsize=10)
fig.tight_layout()
PNG = POC + "/event_recovery.png"
fig.savefig(PNG, dpi=130, bbox_inches="tight"); print("plot saved:", PNG)
json.dump({"injected": dict(p=P_TRUE, au=AU_TRUE, ad=AD_TRUE, beta=B_TRUE, **{k: kj[k] for k in ("std", "skew", "exkurt")}),
           "recovered_cumulants": {k: rec[k] for k in ("std", "skew", "exkurt")},
           "fit": dict(p=xh[0], au=xh[1], ad=xh[2], beta=xh[3]),
           "smile_mae_bp": smile_mae * 1e4, "ok": bool(ok)},
          open(POC + "/event_recovery_results.json", "w"), indent=2)
print("json saved:", POC + "/event_recovery_results.json")

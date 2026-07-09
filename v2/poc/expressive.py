"""
expressive.py -- Algorithm 5 (Stage 3): the expressive layer  phi = G(theta) + delta.

Stage 2 fits the 6 STRUCTURAL knobs theta.  The general kernel has per-component affine
parameters (weight/mean/variance), ~6 per fast node -> a 6K-2 expressive space.  The structural
map G(theta) is a single point in that space; it fits the (vol, skew, SSR) term structures but
leaves a RESIDUAL smile surface (curvature / wings the 6 knobs cannot bend).

Stage 3 adds a correction delta on top of G(theta), fit to the residual by a WIENER FILTER:
linearise the smile surface,  r ~= J delta ;  SVD J = U S V^T ;
    delta_hat(tau) = V * [ S / (S^2 + tau^-2) ] * (U^T r)        (Tikhonov / Wiener shrinkage)
    edf(tau)       = sum_i S_i^2 / (S_i^2 + tau^-2)              (effective # active directions)
Small-singular-value directions (under-identified by the data) are shrunk toward the structural
prior delta = 0; well-identified directions move to fit the data.  Sweeping tau traces the
bias-variance frontier; edf says how many expressive directions the data actually supports.

Expressive knobs used here: per-fast-node log-variance dg_l and skew-tilt ds_l (2 * n_fast),
a concrete slice of the full 6K-2 set.  The kernel stays an exact martingale (global log-sum lock).

Run from poc/ :  python3 expressive.py
"""
import numpy as np
import warnings
from discslv import TwoTimescaleKernel
from data_port import load_chain
from calibrate import DT, kernel as struct_kernel, observables, calibrate, horizon_smile

warnings.filterwarnings("ignore")
CGRID = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])      # standardized log-moneyness (in ATM-total-vol units)
IDX = [1, 3, 5, 6, 7]                                 # chain maturities used: 30,89,180,271,361 d


class ExpressiveKernel(TwoTimescaleKernel):
    """Structural kernel + per-fast-node corrections dg (log-variance) and ds (skew tilt)."""
    def __init__(self, x, dg=None, ds=None):
        super().__init__(gbar=x[0], nu_s=x[1], nu_f=x[2], lam_skew=x[3],
                         lam_mov=x[4], eps_s=x[5], dt=DT)
        self.dg = np.zeros(self.n_fast) if dg is None else np.asarray(dg, float)
        self.ds = np.zeros(self.n_fast) if ds is None else np.asarray(ds, float)

    def _increment(self, z0, b):
        g = self.gbar + self.nu_s * self.zeta_s[b] + self.nu_f * self.zeta_f + self.delta * z0 + self.dg
        sig2 = np.exp(g) * self.dt; sig = np.sqrt(sig2)
        wl = self.fast_weights(z0)
        mtil = -0.5 * sig2 + (self.lam_skew * self.zeta_f + self.ds) * sig
        A = np.log(np.sum(wl * np.exp(mtil + 0.5 * sig2)))        # global martingale lock (incl. dg, ds)
        return wl, mtil - A, sig2


def smile_surface(K, ns, vj):
    """IV at standardized moneyness c*vj for each maturity; flat vector (len = M*len(CGRID))."""
    out = []
    for n, v in zip(ns, vj):
        T = n * K.dt; g = horizon_smile(K, n, 0.0); F = g.forward()
        for c in CGRID:
            out.append(float(g.implied_vol(F * np.exp(c * v), T)[0]))
    return np.array(out)


def target_surface(chain):
    """Smile surface of the synthetic chain (the richer target the structural model must approximate)."""
    out, vj, iv0, sk0 = [], [], [], []
    for j in IDX:
        T = chain["maturities"][j]; mu = chain["marginals"][j]; F = mu.forward()
        atm = float(mu.implied_vol(F, T)[0]); v = atm * np.sqrt(T)
        vj.append(v); iv0.append(atm)
        dm = 6e-3
        sk0.append((float(mu.implied_vol(F * np.exp(dm), T)[0]) - float(mu.implied_vol(F * np.exp(-dm), T)[0])) / (2 * dm))
        for c in CGRID:
            out.append(float(mu.implied_vol(F * np.exp(c * v), T)[0]))
    return np.array(out), np.array(vj), np.array(iv0), np.array(sk0)


def main():
    ch = load_chain("synthetic")
    Ts = [ch["maturities"][j] for j in IDX]
    ns = [max(1, int(round(T / DT))) for T in Ts]
    tgt, vj, iv0, sk0 = target_surface(ch)

    # ---- Stage 2: structural fit to the target statics (vol + skew), SSR free ----
    theta = calibrate(ns, dict(iv=iv0, sk=sk0, ssr=np.zeros_like(iv0)),
                      w=np.array([200.0, 60.0, 0.0]), max_nfev=80)
    base = smile_surface(ExpressiveKernel(theta), ns, vj)
    r = tgt - base
    rmse0 = np.sqrt(np.mean(r ** 2)) * 1e4
    print(f"Stage 3: expressive layer phi = G(theta) + delta   (target = synthetic chain smile surface)\n")
    print(f"structural theta : gbar {theta[0]:.2f}  nu_s {theta[1]:.2f}  nu_f {theta[2]:.2f}  "
          f"lam_skew {theta[3]:.2f}  lam_mov {theta[4]:.2f}  eps_s {theta[5]:.3f}")
    print(f"structural residual (smile surface) : {rmse0:.1f} bp RMSE over {len(tgt)} IV points\n")

    # ---- Stage 3: Jacobian of the surface wrt (dg, ds), SVD, Wiener filter ----
    nf = ExpressiveKernel(theta).n_fast
    p0 = np.zeros(2 * nf); eps = 1e-3
    surf = lambda p: smile_surface(ExpressiveKernel(theta, dg=p[:nf], ds=p[nf:]), ns, vj)
    J = np.zeros((len(tgt), 2 * nf))
    for i in range(2 * nf):
        pp = p0.copy(); pp[i] = eps
        J[:, i] = (surf(pp) - base) / eps
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    rU = U.T @ r

    print("singular spectrum of the expressive Jacobian (identifiable directions):")
    print("  " + "  ".join(f"{s:.2e}" for s in S))
    print(f"  -> {np.sum(S > S[0]*1e-3)} directions above 1e-3 * S_max (rest are data-blind)\n")

    print(f"{'tau':>8} {'edf':>6} {'residual(bp)':>13}   (ONE-STEP Wiener frontier -- linearisation-limited)")
    for tau in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        filt = S / (S ** 2 + 1.0 / tau ** 2)
        dhat = Vt.T @ (filt * rU)
        edf = float(np.sum(S ** 2 / (S ** 2 + 1.0 / tau ** 2)))
        rmse = np.sqrt(np.mean((tgt - surf(dhat)) ** 2)) * 1e4
        print(f"{tau:>8.1f} {edf:>6.2f} {rmse:>13.1f}")

    # ---- Gauss-Newton with Wiener regularization (re-linearise each step) ----
    print("\nGauss-Newton refinement (re-linearise each step; Wiener tau = 30):")
    phi = np.zeros(2 * nf); tau = 30.0; rprev = rmse0
    for it in range(6):
        b = surf(phi); rr = tgt - b
        Ji = np.zeros((len(tgt), 2 * nf))
        for i in range(2 * nf):
            pp = phi.copy(); pp[i] += eps
            Ji[:, i] = (surf(pp) - b) / eps
        Ui, Si, Vti = np.linalg.svd(Ji, full_matrices=False)
        dh = Vti.T @ ((Si / (Si ** 2 + 1.0 / tau ** 2)) * (Ui.T @ rr))
        step, rc = 1.0, rprev                                     # damped accept-if-better
        while step > 1e-2:
            rc = np.sqrt(np.mean((tgt - surf(phi + step * dh)) ** 2)) * 1e4
            if rc <= rprev + 1e-9:
                break
            step *= 0.5
        if rc > rprev + 1e-9:
            break
        phi = phi + step * dh; rprev = rc
        edf = float(np.sum(Si ** 2 / (Si ** 2 + 1.0 / tau ** 2)))
        print(f"  iter {it}: residual {rc:6.1f} bp   edf {edf:.2f}   step {step:.2f}")
    dhat_b = phi; rmse_b = rprev
    print(f"\nresidual {rmse0:.1f} -> {rmse_b:.1f} bp  ({100*(1-rmse_b/rmse0):.0f}% of structural misfit removed)")
    print(f"expressive correction  dg = {np.round(dhat_b[:nf],3)}")
    print(f"                       ds = {np.round(dhat_b[nf:],3)}")

    # ---- plot ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        taus = np.logspace(-1, 2.7, 40)
        edfs = [float(np.sum(S**2/(S**2+1/t**2))) for t in taus]
        rmses = []
        for t in taus:
            dh = Vt.T @ ((S/(S**2+1/t**2)) * rU)
            rmses.append(np.sqrt(np.mean((tgt - surf(dh))**2))*1e4)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.semilogx(taus, rmses, "-", label="one-step Wiener")
        a1.axhline(rmse0, ls="--", color="gray", label="structural (delta=0)")
        a1.axhline(rmse_b, ls=":", color="green", label="Gauss-Newton refined")
        a1.set_xlabel("Wiener tau"); a1.set_ylabel("smile residual (bp)"); a1.set_title("bias-variance frontier"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(edfs, rmses, "o-"); a2.set_xlabel("effective dof"); a2.set_ylabel("smile residual (bp)")
        a2.set_title("residual vs expressive complexity"); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("expressive_wiener.png", dpi=130)
        print("\nsaved plot -> poc/expressive_wiener.png")
    except Exception as ex:
        print(f"\n(plot skipped: {ex})")


if __name__ == "__main__":
    main()

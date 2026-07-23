"""
discslv.py -- SANOS-Evolve proof-of-concept library.

A discrete stochastic-local-volatility model for SPX:
  * statics  : SANOS-style Gaussian-mixture marginals in log-moneyness z = log(S/F);
  * dynamics : a two-timescale latent-regime Gaussian-mixture *martingale* kernel
               between consecutive marginals.

Everything is closed form. Black-Scholes enters as the moment-generating function of a
truncated Gaussian, so every mixture price is a finite weighted sum of Black calls.

SCAFFOLD STATUS
---------------
* The kernel, the exact martingale normalisation, the closed-form pricing, the
  forward-start smile and the SSR finite-difference are the REAL machinery.
* The marginals here are SYNTHETIC (a martingale lognormal mixture with increasing
  variance, hence in convex order) standing in for SANOS LP output. Swap in real
  SANOS marginals at `synthetic_marginals`.
* State-dependent kernel weights are evaluated at the source-component mean
  (collocation); multi-step propagation would add the constrained recompression of
  paper Sec. 11. The PoC runs a single step, so no recompression is needed.

Maps to paper Sec. 12 (proof-of-concept protocol).
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm


# --------------------------------------------------------------------------------------
# Black-Scholes as the MGF of a truncated Gaussian
# --------------------------------------------------------------------------------------
def bs_call(F, K, tot_vol):
    """Black call price: forward F, strike K, total volatility tot_vol = sigma*sqrt(T).
    Vectorised over K (and/or F)."""
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    s = np.maximum(np.asarray(tot_vol, float), 1e-12)
    d1 = (np.log(F / K) + 0.5 * s ** 2) / s
    d2 = d1 - s
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def bs_implied_vol(price, F, K, T, lo=1e-4, hi=5.0, tol=1e-10, maxit=200):
    """Invert a Black call price to an annualised implied volatility by bisection."""
    intrinsic = max(F - K, 0.0)
    price = min(max(price, intrinsic + 1e-14), F - 1e-14)

    def f(sig):
        return float(bs_call(F, K, sig * np.sqrt(T))) - price

    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:                      # price outside [f(lo), f(hi)] -> nearest bound
        return a if abs(fa) < abs(fb) else b
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def softmax(x):
    x = np.asarray(x, float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# --------------------------------------------------------------------------------------
# Gaussian mixture in log-moneyness  z = log(S / F)
# --------------------------------------------------------------------------------------
class GMM:
    """Mixture sum_i w_i N(mu_i, s_i^2) for z = log(S/F).  s_i is the *total* vol to T.

    The price-space forward of component i is F * exp(mu_i + s_i^2/2); the mixture is a
    martingale (E[S] = F) iff sum_i w_i exp(mu_i + s_i^2/2) = 1.
    """

    def __init__(self, w, mu, s, F=1.0):
        self.w = np.asarray(w, float)
        self.mu = np.asarray(mu, float)
        self.s = np.asarray(s, float)
        self.F = float(F)

    @property
    def n(self):
        return len(self.w)

    def comp_forwards(self):
        return self.F * np.exp(self.mu + 0.5 * self.s ** 2)

    def forward(self):
        return float(np.sum(self.w * self.comp_forwards()))

    def martingale_normalize(self):
        """Shift every component mean by -A so the total forward is exactly F."""
        A = np.log(np.sum(self.w * np.exp(self.mu + 0.5 * self.s ** 2)))
        self.mu = self.mu - A
        return self

    def call(self, K):
        """Call prices on strikes K -- a finite mixture of Black calls."""
        K = np.atleast_1d(np.asarray(K, float))
        out = np.zeros_like(K)
        for wi, Fi, si in zip(self.w, self.comp_forwards(), self.s):
            out = out + wi * bs_call(Fi, K, si)
        return out

    def cdf(self, K):
        """Risk-neutral CDF P(S <= K) = sum_i w_i Phi((log(K/F) - mu_i)/s_i)."""
        K = np.atleast_1d(np.asarray(K, float))
        z = np.log(K / self.F)
        out = np.zeros_like(K)
        for wi, mui, si in zip(self.w, self.mu, self.s):
            out = out + wi * norm.cdf((z - mui) / max(si, 1e-12))
        return out

    def implied_vol(self, K, T):
        """Mixture implied-vol smile on strikes K (invert the mixture call price)."""
        K = np.atleast_1d(np.asarray(K, float))
        c = self.call(K)
        return np.array([bs_implied_vol(ci, self.F, Ki, T) for ci, Ki in zip(c, K)])


def convex_order_violation(gmm_a, gmm_b, Kgrid):
    """Max violation of  gmm_a <=_cx gmm_b  (calendar no-arb): want call_b >= call_a."""
    ca = gmm_a.call(Kgrid)
    cb = gmm_b.call(Kgrid)
    return float(np.max(np.maximum(ca - cb, 0.0)))


# --------------------------------------------------------------------------------------
# The two-timescale latent-regime martingale kernel
# --------------------------------------------------------------------------------------
class TwoTimescaleKernel:
    """One-step kernel on the joint state (z, b): z = log-moneyness, b = slow vol regime.

    Coefficient map (paper Sec. 5), with TWO leverage knobs that separate the two
    properties of equity smile dynamics:

        gamma_{(b,l)} = gbar + nu_s zeta^s_b + nu_f zeta^f_l        (log-variance level)
        sigma^2_{(b,l)}(z) = exp(gamma_{(b,l)} + delta z) dt        (component step variance)

        * lam_skew  -- leverage MEAN-TILT  ell_l = lam_skew * zeta^f_l * sigma_l
                       (eq. heston-coeff): high-vol nodes drift down -> a genuine
                       forward-smile SKEW even at the money (the SSR *denominator*).
        * lam_mov   -- spot-coupling of the weights, w_l(z) = softmax(eta^f_l + lam_mov zeta^f_l z)
                       and likewise for the slow transition: the smile MOVES with spot
                       (the SSR *numerator*, d Sigma_ATM / d log S).

    Each fibre is martingale-normalised exactly (the global per-fibre log-sum shift
    A(z,b)), so E[e^{Z'} | z, b] = e^z and the kernel is a martingale by construction.
    lam_mov = 0  ->  smile rides with spot  ->  sticky-delta  ->  SSR = 0.
    """

    def __init__(self, gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s, dt,
                 n_slow=3, n_fast=5, delta=0.0):
        self.gbar, self.nu_s, self.nu_f = gbar, nu_s, nu_f
        self.lam_skew, self.lam_mov = lam_skew, lam_mov
        self.eps_s, self.dt, self.delta = eps_s, dt, delta
        self.n_slow, self.n_fast = n_slow, n_fast
        zs, ws = hermegauss(n_slow)        # standardized slow factor nodes + Gaussian weights
        zf, wf = hermegauss(n_fast)        # standardized fast factor nodes + Gaussian weights
        self.zeta_s, self.zeta_f = zs, zf
        self.eta_s = np.log(ws / ws.sum())  # base log-weights -> Gaussian factor when lam_mov = 0
        self.eta_f = np.log(wf / wf.sum())

    # ---- weights ----
    def fast_weights(self, z0):
        return softmax(self.eta_f + self.lam_mov * self.zeta_f * z0)

    def slow_weights(self, z0):
        return softmax(self.eta_s + self.lam_mov * self.zeta_s * z0)

    def regime_prior(self):
        return softmax(self.eta_s)

    # ---- fast-node increment mixture at state (z0, regime b) ----
    def _increment(self, z0, b):
        """Returns (w_l, d_l, sigma2_l): fast weights, martingale-locked means, variances
        for the log-return increment.  E[exp(increment)] = 1 exactly (global log-sum lock)."""
        g = self.gbar + self.nu_s * self.zeta_s[b] + self.nu_f * self.zeta_f + self.delta * z0
        sig2 = np.exp(g) * self.dt
        sig = np.sqrt(sig2)
        wl = self.fast_weights(z0)
        mtil = -0.5 * sig2 + self.lam_skew * self.zeta_f * sig      # leverage mean-tilt -> skew
        A = np.log(np.sum(wl * np.exp(mtil + 0.5 * sig2)))          # global martingale lock
        return wl, mtil - A, sig2

    # ---- propagation of a joint state (list of (w, mu, s, b)) ----
    def propagate(self, comps):
        out = []
        for (w, mu, s, b) in comps:
            z0 = mu                                   # collocation point for state-dep. terms
            wl, d, sig2 = self._increment(z0, b)
            base = np.zeros(self.n_slow); base[b] = 1.0
            Pbp = (1.0 - self.eps_s) * base + self.eps_s * self.slow_weights(z0)
            for l in range(self.n_fast):
                new_mu = mu + d[l]
                new_s = np.sqrt(s ** 2 + sig2[l])
                for bp in range(self.n_slow):
                    ww = w * wl[l] * Pbp[bp]
                    if ww > 1e-12:
                        out.append((ww, new_mu, new_s, bp))
        tot = sum(c[0] for c in out)
        return [(c[0] / tot, c[1], c[2], c[3]) for c in out]

    # ---- forward-start smile and SSR ----
    def forward_start_call(self, z0, k):
        """E[(S'/S - k)^+] at state z0 (regime + fast weights leverage-tilted to z0)."""
        pb = self.slow_weights(z0)
        c = 0.0
        for b in range(self.n_slow):
            wl, d, sig2 = self._increment(z0, b)
            for l in range(self.n_fast):
                Fret = np.exp(d[l] + 0.5 * sig2[l])    # return-forward of node l
                c += pb[b] * wl[l] * float(bs_call(Fret, k, np.sqrt(sig2[l])))
        return c

    def forward_iv(self, z0, k):
        return bs_implied_vol(self.forward_start_call(z0, k), 1.0, k, self.dt)

    def forward_skew(self, z0=0.0, dm=2e-3):
        """ATM forward-smile skew d Sigma / d log k at k = 1 (the SSR denominator)."""
        up = self.forward_iv(z0, np.exp(+dm))
        dn = self.forward_iv(z0, np.exp(-dm))
        return (up - dn) / (2 * dm)

    def ssr(self, z0=0.0, h=2e-3, dm=2e-3):
        """Bergomi SSR = (dSigma_ATM/d log S) / (ATM skew), by central finite difference.
        lam_mov = 0  ->  sticky-delta  ->  SSR = 0."""
        up = self.forward_iv(z0 + h, 1.0)
        dn = self.forward_iv(z0 - h, 1.0)
        dvol_dz = (up - dn) / (2 * h)
        skew = self.forward_skew(z0, dm)
        if abs(skew) < 1e-9:
            return float("nan")
        return dvol_dz / skew


# --------------------------------------------------------------------------------------
# Joint-state helpers
# --------------------------------------------------------------------------------------
def lift_to_joint(gmm, kernel):
    """Lift a spot marginal to a joint (z, b) state using the regime prior."""
    pb = kernel.regime_prior()
    comps = []
    for wi, mui, si in zip(gmm.w, gmm.mu, gmm.s):
        for b in range(kernel.n_slow):
            comps.append((wi * pb[b], mui, si, b))
    return comps


def marginal_from_joint(comps, F=1.0):
    """Marginalise the joint state over the regime b -> spot GMM."""
    w = np.array([c[0] for c in comps])
    mu = np.array([c[1] for c in comps])
    s = np.array([c[2] for c in comps])
    return GMM(w, mu, s, F=F)


# --------------------------------------------------------------------------------------
# Synthetic marginals (stand-in for SANOS LP output)
# --------------------------------------------------------------------------------------
def convolve(a, b, F=1.0):
    """Convolve two GMMs in z (independent sum of log-returns).  b is the increment."""
    w = np.outer(a.w, b.w).ravel()
    mu = (a.mu[:, None] + b.mu[None, :]).ravel()
    s = np.sqrt(a.s[:, None] ** 2 + b.s[None, :] ** 2).ravel()
    return GMM(w, mu, s, F=F)


def _skewed_increment(dt):
    """A skewed, independent, *martingale* log-return increment (3-component GMM):
    a down-shifted high-vol 'crash' component gives the marginal a negative skew.
    Being a martingale increment, convolving mu0 with it preserves convex order."""
    w = np.array([0.15, 0.70, 0.15])
    vol = np.array([0.34, 0.16, 0.13])           # annualised; high-vol crash component
    s = vol * np.sqrt(dt)
    mu = -0.5 * s ** 2 + np.array([-0.05, 0.0, 0.0])   # extra down-shift -> negative skew
    return GMM(w, mu, s, F=1.0).martingale_normalize()


def synthetic_marginals(T1=0.25, T2=0.50, F=1.0):
    """Stand-in for SANOS LP output.  mu0 is a symmetric martingale mixture at T1; mu1 is
    mu0 convolved with an independent *skewed* martingale increment, so mu1 carries a
    realistic negative skew AND is guaranteed to be in convex order above mu0 (calendar
    no-arb).  The increment is spot-independent, so the data's intrinsic SSR is ~0
    (sticky-delta); the proof of concept then *imposes* an SSR view via the kernel's
    spot-coupling and shows it costs ~nothing on the marginal.  Returns (mu0, mu1, T1, T2)."""
    w = np.array([0.20, 0.60, 0.20])
    vols = np.array([0.22, 0.17, 0.14])          # T1 per-component annualised vols
    s0 = vols * np.sqrt(T1)
    mu0 = GMM(w, -0.5 * s0 ** 2, s0, F=F)        # symmetric martingale base
    mu1 = convolve(mu0, _skewed_increment(T2 - T1), F=F)
    return mu0, mu1, T1, T2


def synthetic_chain(maturities, F=1.0):
    """A convex-ordered chain of marginals across several maturities: mu_0 symmetric,
    each successive marginal = previous convolved with a skewed martingale increment over
    the maturity gap.  Returns a list [mu_0, ..., mu_M]."""
    w = np.array([0.20, 0.60, 0.20])
    vols = np.array([0.22, 0.17, 0.14])
    s0 = vols * np.sqrt(maturities[0])
    mus = [GMM(w, -0.5 * s0 ** 2, s0, F=F)]
    for j in range(1, len(maturities)):
        dt = maturities[j] - maturities[j - 1]
        mus.append(convolve(mus[-1], _skewed_increment(dt), F=F))
    return mus


# --------------------------------------------------------------------------------------
# Recompression (paper Sec. 11): keep the per-fibre component budget flat
# --------------------------------------------------------------------------------------
def merge_gmm(w, mu, s, n_keep):
    """Reduce a 1-D GMM (in z) to n_keep components by greedy moment-preserving merges of
    the least-disruptive adjacent pair.  Each merge preserves the pair's 0th/1st/2nd
    moments; the forward E[e^z] is re-locked by the caller."""
    w = list(map(float, w)); mu = list(map(float, mu)); s2 = list(np.asarray(s, float) ** 2)
    while len(w) > n_keep:
        order = np.argsort(mu)
        w = [w[i] for i in order]; mu = [mu[i] for i in order]; s2 = [s2[i] for i in order]
        cost = [(mu[j + 1] - mu[j]) ** 2 * w[j] * w[j + 1] / (w[j] + w[j + 1])
                for j in range(len(w) - 1)]
        j = int(np.argmin(cost))
        W = w[j] + w[j + 1]
        M = (w[j] * mu[j] + w[j + 1] * mu[j + 1]) / W
        V = (w[j] * (s2[j] + mu[j] ** 2) + w[j + 1] * (s2[j + 1] + mu[j + 1] ** 2)) / W - M ** 2
        w = w[:j] + [W] + w[j + 2:]; mu = mu[:j] + [M] + mu[j + 2:]; s2 = s2[:j] + [V] + s2[j + 2:]
    return np.array(w), np.array(mu), np.sqrt(np.maximum(np.array(s2), 1e-12))


def recompress_joint(comps, n_keep, n_slow):
    """Recompress each regime's spot mixture to n_keep components, then re-apply the
    global exponential-moment martingale lock (Sec. 11, moment-merge variant).  The
    carried regimes are preserved; only the spot resolution is compressed."""
    out = []
    for b in range(n_slow):
        wb = np.array([c[0] for c in comps if c[3] == b])
        if wb.size == 0:
            continue
        mub = np.array([c[1] for c in comps if c[3] == b])
        sb = np.array([c[2] for c in comps if c[3] == b])
        pb = wb.sum()
        wn, mn, sn = merge_gmm(wb / pb, mub, sb, n_keep)
        out.extend((pb * wi, mi, si, b) for wi, mi, si in zip(wn, mn, sn))
    A = np.log(sum(c[0] * np.exp(c[1] + 0.5 * c[2] ** 2) for c in out))   # global martingale re-lock
    return [(c[0], c[1] - A, c[2], c[3]) for c in out]


def regime_law(comps, n_slow):
    """Marginal regime probabilities p_b from a joint state."""
    p = np.zeros(n_slow)
    for (w, _, _, b) in comps:
        p[b] += w
    return p / p.sum()


# --------------------------------------------------------------------------------------
# VIX layer (paper Sec. 7): model-implied 30-day forward variance from the vol regime
# --------------------------------------------------------------------------------------
def damping(kappa, tau):
    """30-day forward-variance damping of a mean-reverting factor: (1-e^{-k tau})/(k tau)."""
    return (1.0 - np.exp(-kappa * tau)) / (kappa * tau)


def vix_layer(kernel, comps, tau_vix=30.0 / 365.0):
    """Model-implied VIX at the propagated state.  The slow regime b carries an
    instantaneous variance v_b = exp(gbar + nu_s zeta^s_b); the 30-day forward variance is
    the damped value vbar_b = v_inf + (v_b - v_inf) D(kappa_s, tau).  VIX is atomic:
    VIX_T = sqrt(vbar_b) with probability p_b.  Returns (p, vix, kappa_s, D_s, D_f)."""
    p = regime_law(comps, kernel.n_slow)
    v_inf = np.exp(kernel.gbar)
    v_b = np.exp(kernel.gbar + kernel.nu_s * kernel.zeta_s)
    kappa_s = -np.log(max(1.0 - kernel.eps_s, 1e-9)) / kernel.dt
    kappa_f = 30.0                                   # conventional fast rate (per year)
    D_s, D_f = damping(kappa_s, tau_vix), damping(kappa_f, tau_vix)
    vbar = v_inf + (v_b - v_inf) * D_s
    vix = np.sqrt(np.maximum(vbar, 1e-12))
    return p, vix, kappa_s, D_s, D_f


def vix_future(p, vix):
    return float(np.sum(p * vix))


def vix_call(p, vix, K):
    K = np.atleast_1d(np.asarray(K, float))
    return np.array([float(np.sum(p * np.maximum(vix - k, 0.0))) for k in K])


def vix_vov(p, vix):
    """Vol-of-vol proxy: standard deviation of VIX_T under the regime law p."""
    m = float(np.sum(p * vix))
    return float(np.sqrt(np.sum(p * (vix - m) ** 2)))

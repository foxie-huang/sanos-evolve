#!/usr/bin/env python3
"""
discslv_torch.py (v2) -- the fused forward map in PyTorch, differentiable in theta for AD gradients.

STAGE 1: kernel build + fused propagate + recompress, ported from v1's numpy (TwoFactorSV + slv_fast.
propagate_vec + discslv_2f.recompress_2f). The recompress keeps v1's EXACT adaptive equal-weight-chunk
assignment but STOP-GRADIENTs it (a detached membership matrix); only the moment-matching is
differentiable. Forward values match v1 bit-close; gradients flow through theta -> (kernel, increments)
-> the propagated marginal. float64 to match numpy.

Validate:  python3 discslv_torch.py   (compares kernel arrays + one propagate step to v1)
"""
import os

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

# WARNING -- this is a GLOBAL set by import side-effect, and calibrate_joint_torch.py sets
# float32. Whichever module is imported LAST wins. Today calibrate_joint_torch imports THIS module
# first, so float32 is what production runs on -- but that is a property of the import graph, not a
# decision. Reorder the imports and the numerical config silently flips. float64 here is deliberate
# for standalone use (matching the numpy reference); float32 is deliberate for the fit (2.5x on
# jacrev, and MPS has no float64 at all). Do not "tidy" either one without checking the other.
torch.set_default_dtype(torch.float64)
PNAMES = ["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]


Q_VIX = int(os.environ.get("Q_VIX", "5"))        # GH nodes for the spot integral in the leveraged VIX
# Which leverage tenor the leveraged VIX uses. The VIX at expiry T is a 30d variance swap spanning
# [T, T+30d], so the local vol that belongs to it sits at maturity ~T+15d FROM TODAY -- not 30d from
# today. LT stops at 13wk while vdtes reach 260d, so every long tenor extrapolates whichever rule is
# used; this switch exists to measure how much that choice matters.
_VOVLAMTEN = os.environ.get("VOVLAMTEN", "expiry")    # expiry | mid | fix30
# VIXFIX=0 restores the pre-2026-08-08 leveraged VIX readout in vix_ivol_n, which scaled
# vol-of-vol by ~rho and returned EXACTLY ZERO at rho = 0. Kept only for A/B.
_VIXFIX = os.environ.get("VIXFIX", "1") == "1"
# SLOWFROZEN=1: the slow factor becomes a STATIC GH quadrature instead of a Markov chain. Ts becomes
# the identity, so kap_s and lam_s no longer act, and the s-nodes carry a frozen draw whose amplitude
# nu_s absorbs. Motivated by 16.0b/16.0c: a chain cannot represent kappa_s >= 0.99 at any n, but the
# factor is then too slow to move over the horizon, so freezing costs 0.1-0.75% on ATM vol.
_SLOWFROZEN = os.environ.get("SLOWFROZEN", "0") == "1"


def lev_torch(coef, zmax, safety):
    """Torch leverage lambda(MU) = clip(exp(polyval(coef, clip(MU,-zmax,zmax))), safety). coef from v1's
    _Lev (theta-invariant statics), so this is differentiable in MU (the state) but constant in theta."""
    c = np.asarray(coef, float); lo, hi = safety

    def lam(mu):
        z = torch.clamp(mu, -zmax, zmax)
        p = torch.zeros_like(z)
        for a in c:                                        # Horner polyval (a scalars cast to z's dtype/device)
            p = p * z + float(a)
        return torch.clamp(torch.exp(p), lo, hi)
    # Slope of log-lambda, exposed for the Jensen correction in propagate_batch. The propagator
    # freezes lambda at each mixture component's MEAN, but lambda varies across the component's own
    # width: measured lam(MU+SG)/lam(MU) = 0.90 at step 1 falling to 0.72 by step 13, with the mean
    # component width EXCEEDING the spread of component means (68% of total variance is intra-
    # component). For a degree-1 log-lambda against a Gaussian component the correction is exact:
    #     E[lambda]   = lambda(MU) * exp(c1^2 SG^2 / 2)
    #     E[lambda^2] = lambda(MU)^2 * exp(2 c1^2 SG^2)
    # Only set for deg 1 -- a higher-degree fit is not log-linear and the identity does not hold.
    lam.c1 = float(c[0]) if len(c) == 2 else 0.0
    lam.zmax = float(zmax)
    return lam


def build_kernel(theta, dt, n_f=5, n_s=3, n_l=5):
    """theta: torch tensor [gbar,nu_f,nu_s,nu_l,lam_skew,lam_f,lam_s,kap_f,kap_s]. Returns a dict of torch
    kernel arrays (differentiable in theta): Vl,D (n_f,n_s,n_l), tilt (n_f,n_s,n_l), Tf (n_l,n_f,n_f),
    Ts (n_l,n_s,n_s), and the fixed nodes/weights."""
    gbar, nu_f, nu_s, nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s = theta
    dd = dict(dtype=theta.dtype, device=theta.device)                # kernel follows theta's dtype/device
    zf = torch.as_tensor(hermegauss(n_f)[0], **dd); wf = torch.as_tensor(hermegauss(n_f)[1], **dd); wf = wf / wf.sum(); ef = torch.log(wf)
    zs = torch.as_tensor(hermegauss(n_s)[0], **dd); ws = torch.as_tensor(hermegauss(n_s)[1], **dd); ws = ws / ws.sum(); es = torch.log(ws)
    zl = torch.as_tensor(hermegauss(n_l)[0], **dd); wl = torch.as_tensor(hermegauss(n_l)[1], **dd); wl = wl / wl.sum()
    g = gbar + nu_f * zf[:, None, None] + nu_s * zs[None, :, None] + nu_l * zl[None, None, :]   # (nf,ns,nl)
    Vl = torch.exp(g) * dt
    sigl = torch.sqrt(Vl)
    tilt = lam_skew * zl[None, None, :] * sigl
    mtil = -0.5 * Vl + tilt
    A = torch.log((wl[None, None, :] * torch.exp(mtil + 0.5 * Vl)).sum(-1, keepdim=True))
    D = mtil - A
    Tf = torch.stack([torch.stack([torch.softmax(ef + kap_f * zf[f] * zf + lam_f * zl[l] * zf, 0)
                                   for f in range(n_f)]) for l in range(n_l)])   # (nl,nf,nf)
    if _SLOWFROZEN:
        Ts = torch.eye(n_s, **dd).unsqueeze(0).expand(n_l, n_s, n_s).contiguous()   # frozen: no s-transitions
    else:
        Ts = torch.stack([torch.stack([torch.softmax(es + kap_s * zs[s] * zs + lam_s * zl[l] * zs, 0)
                                       for s in range(n_s)]) for l in range(n_l)])   # (nl,ns,ns)
    return dict(Vl=Vl, D=D, tilt=tilt, Tf=Tf, Ts=Ts, wl=wl, ws=ws, zf=zf, zs=zs, zl=zl, n_f=n_f, n_s=n_s, n_l=n_l, dt=dt)


def _merge(w, mu, sg, nk):
    """Detached-membership recompression: v1's sort -> equal-weight chunks (STOP-GRADIENT) -> per-chunk
    0/1/2 moment-match (differentiable). Same forward value as _fast_merge; gradient = the within-cell part."""
    n = w.shape[0]
    with torch.no_grad():                                  # discrete assignment: no grad
        order = torch.argsort(mu)
        cw = torch.cumsum(w[order], 0)
        q = torch.linspace(0.0, float(cw[-1]), nk + 1)[1:-1]
        edges = torch.searchsorted(cw, q, right=True)
        chunk = torch.searchsorted(edges, torch.arange(n), right=True)          # chunk id per sorted position
        memb = torch.zeros(n, nk); memb[order, chunk] = 1.0                      # original component -> its chunk
    Wj = memb.t() @ w
    keep = Wj > 1e-15
    Ws = torch.where(keep, Wj, torch.ones_like(Wj))
    Mj = (memb.t() @ (w * mu)) / Ws
    E2 = (memb.t() @ (w * (sg ** 2 + mu ** 2))) / Ws
    Vj = torch.clamp(E2 - Mj ** 2, min=1e-12)
    return Wj[keep], Mj[keep], torch.sqrt(Vj)[keep]


def recompress(W, MU, SG, F, S, nk, n_f, n_s):
    """Per-(f,s)-regime detached-membership recompression + global martingale re-lock (matches v1)."""
    Wo, MUo, SGo, Fo, So = [], [], [], [], []
    for f in range(n_f):
        for s in range(n_s):
            m = (F == f) & (S == s)
            if int(m.sum()) == 0:
                continue
            w, mu, sg = W[m], MU[m], SG[m]
            if w.shape[0] > nk:
                w, mu, sg = _merge(w, mu, sg, nk)
            Wo.append(w); MUo.append(mu); SGo.append(sg)
            Fo.append(torch.full((w.shape[0],), f)); So.append(torch.full((w.shape[0],), s))
    W = torch.cat(Wo); MU = torch.cat(MUo); SG = torch.cat(SGo)
    F = torch.cat(Fo).long(); S = torch.cat(So).long()
    A = torch.log((W * torch.exp(MU + 0.5 * SG ** 2)).sum())                     # global re-lock
    return W, MU - A, SG, F, S


def propagate(state, ker, lam, nk):
    """One fused step (ports slv_fast.propagate_vec): leveraged increment -> expand over (l,f',s') -> recompress."""
    W, MU, SG, F, S = state
    nl, nf, ns_ = ker["n_l"], ker["n_f"], ker["n_s"]; wl = ker["wl"]
    Vlr = ker["Vl"][F, S]                                  # (N, nl)
    lm = torch.clamp(lam(MU), min=1e-6)                    # (N,) leverage at component means
    Vl = lm[:, None] ** 2 * Vlr
    mtil = -0.5 * Vl + lm[:, None] * ker["tilt"][F, S]
    A = torch.log((wl[None, :] * torch.exp(mtil + 0.5 * Vl)).sum(-1))            # (N,)
    Dl = mtil - A[:, None]
    Tfi = ker["Tf"][:, F, :].permute(1, 0, 2)              # (N, nl, nf)
    Tsi = ker["Ts"][:, S, :].permute(1, 0, 2)              # (N, nl, ns)
    w = W[:, None, None, None] * wl[None, :, None, None] * Tfi[:, :, :, None] * Tsi[:, :, None, :]
    mu_b = (MU[:, None] + Dl)[:, :, None, None].expand(-1, -1, nf, ns_)
    sg_b = torch.sqrt(SG[:, None] ** 2 + Vl)[:, :, None, None].expand(-1, -1, nf, ns_)
    fp = torch.arange(nf)[None, None, :, None].expand(w.shape)
    sp = torch.arange(ns_)[None, None, None, :].expand(w.shape)
    W2 = w.reshape(-1); W2 = W2 / W2.sum()
    return recompress(W2, mu_b.reshape(-1), sg_b.reshape(-1), fp.reshape(-1), sp.reshape(-1), nk, nf, ns_)


# ======================================================================================
# NORMALISED KERNEL (SANOS_REFERENCE.md 16.9). Added ALONGSIDE the abscissa-indexed path
# above -- ~45 call sites read the old kernel dict, so nothing here replaces it yet.
#
# WHY. The old path has two independent defects, both hitting BOTH persistent factors:
#   16.0   TRUNCATION. z has stationary variance (1+lam^2)/(1-kap^2) = 1.55..219 across the
#          fitted panel, but is carried on abscissas spanning +-2.86. The chain realises
#          sd 1.48-2.08 whatever theta asks for: 0.9%-95% of the intended amplitude, 9/9 years.
#   16.0d  TRANSITION WIDTH. The GH rule's abscissas are the roots of He_n, placed for the
#          STATIONARY density N(0,1), but the transition density is N(kap z, 1-kap^2), of
#          width sqrt(1-kap^2). 11% error on ONE integral at kap_f = 0.857; 100% at kap_s.
#          NOT fixed by normalising -- the normalised chain degenerates to P = I instead.
#
# Both vanish here, for one reason: the factor process is linear-Gaussian and EXOGENOUS
# (nothing in the price feeds back into it), so its law is carried as a bivariate GAUSSIAN
# and updated in closed form. No abscissas, so nothing to truncate; no rule applied to the
# transition, so no integrand width can be wrong.
#
# theta = [gbar, nu_f, nu_s, nu_l, lam_skew, rho_f, rho_s, kap_f, kap_s]  -- same COUNT as
# the old theta, but lam_f/lam_s become BOUNDED correlations rho_f/rho_s, and kappa is now
# exactly the autocorrelation rather than a number the chain fails to realise.
#
#     u' = kap u + sqrt(1-kap^2) ( rho z_l + sqrt(1-rho^2) eps )       Var(u) = 1 by construction
#
# State becomes (W, MU, SG, mf, ms, Vff, Vss, Vfs): the two integer abscissa indices F, S are
# replaced by the factors' per-component Gaussian. Branching is n_x n_l = 15, against 75.
# ======================================================================================

def build_kernel_n(theta, dt, n_l=5, n_x=3):
    """Normalised kernel. Returns only what the propagator needs -- no Vl, D, tilt, Tf, Ts:
    those existed solely because the factors were abscissa indices."""
    gbar, nu_f, nu_s, nu_l, lam_skew, rho_f, rho_s, kap_f, kap_s = theta
    dd = dict(dtype=theta.dtype, device=theta.device)
    zl = torch.as_tensor(hermegauss(n_l)[0], **dd); wl = torch.as_tensor(hermegauss(n_l)[1], **dd)
    zx = torch.as_tensor(hermegauss(n_x)[0], **dd); wx = torch.as_tensor(hermegauss(n_x)[1], **dd)
    s_f = torch.sqrt(torch.clamp(1 - kap_f ** 2, min=1e-12))
    s_s = torch.sqrt(torch.clamp(1 - kap_s ** 2, min=1e-12))
    return dict(gbar=gbar, nu_f=nu_f, nu_s=nu_s, nu_l=nu_l, lam_skew=lam_skew,
                rho_f=rho_f, rho_s=rho_s, kap_f=kap_f, kap_s=kap_s,
                s_f=s_f, s_s=s_s,
                q_f=s_f ** 2 * (1 - rho_f ** 2), q_s=s_s ** 2 * (1 - rho_s ** 2),
                zl=zl, wl=wl / wl.sum(), zx=zx, wx=wx / wx.sum(),
                n_l=n_l, n_x=n_x, dt=dt)


def stationary_n(ker):
    """Stationary law of (u_f, u_s): bivariate normal, UNIT marginals, closed form. Replaces the
    linear solve in `stationary_pi` -- there is no chain to solve for.

        corr = sqrt((1-kap_f^2)(1-kap_s^2)) rho_f rho_s / (1 - kap_f kap_s)

    Automatically a valid correlation: the bound reduces to (kap_f - kap_s)^2 >= 0, equality iff
    the two factors share a kappa. Verified to 2.5e-16 (continuous_carry_test.py step 2b)."""
    c = (ker["s_f"] * ker["s_s"] * ker["rho_f"] * ker["rho_s"]
         / torch.clamp(1 - ker["kap_f"] * ker["kap_s"], min=1e-12))
    return torch.clamp(c, -1.0 + 1e-9, 1.0 - 1e-9)


def init_state_n(ker, mu0=0.0, sg0=1e-6):
    """One component, factors on their stationary law."""
    dd = dict(dtype=ker["zl"].dtype, device=ker["zl"].device)
    one = torch.ones(1, **dd)
    return (one.clone(), torch.full((1,), float(mu0), **dd), torch.full((1,), float(sg0), **dd),
            torch.zeros(1, **dd), torch.zeros(1, **dd),
            one.clone(), one.clone(), stationary_n(ker).reshape(1))


def propagate_n(state, ker, lam, nk, nb_f=5, nb_s=3, n_p=5):
    """One step. Branches over (x, l) only -- n_x n_l = 15 against the old n_f n_s n_l = 75.

    X = nu_f u_f + nu_s u_s is the factors' combined contribution to log-variance. Inside a
    component (u_f, u_s) is Gaussian, so X is Gaussian and exp(X) is LOGNORMAL -- the price
    increment is a lognormal-variance mixture of normals and is not itself normal. Conditioning
    on X at n_x GH abscissas makes it exactly Gaussian per branch, so the ONLY approximation on
    the price side is that quadrature. Unlike the old transition quadrature this one is
    legitimate: X's own density is what the abscissas integrate against, and it is Gaussian.
    Converged at n_x = 3 (price_side_test.py step 3).

    n_p RESOLVES 16.10. The old propagator evaluates the leverage ONCE per component, at its mean:
    `lm = lam(MU)`, applied across the component's whole width SG. Within a component, volatility
    then does not depend on where the price lands -- and skew IS that dependence. Measured cost
    -40 to -43% on ATM skew across five dates. It is f(E[x]) where the model needs E[f(x)], and no
    moment correction reaches it: `_clamped_ratio` fixes E[lambda] exactly but still yields ONE vol
    per component, so it cannot produce a price/vol covariance.

    With n_p > 1 the component's OWN width is resolved on GH abscissas p_q = MU + SG z_q (weights
    w_q), lambda is evaluated at each, and each sub-branch carries only the fresh diffusion V. The
    old SG is not discarded -- it is re-expressed as the SPREAD of the p_q means, which GH
    reproduces exactly in the first two moments for n_p >= 2. That is the point: it converts
    WITHIN-component variance, which refinement cannot reach (`sg_b = sqrt(SG^2 + Vl)` only ever
    accumulates), into BETWEEN-component variance, which it can.

    n_p = 1 keeps the old lambda-at-the-mean behaviour, for measuring the difference.
    """
    W, MU, SG, mf, ms, Vff, Vss, Vfs = state
    nu_f, nu_s, nu_l = ker["nu_f"], ker["nu_s"], ker["nu_l"]
    zl, wl, zx, wx = ker["zl"], ker["wl"], ker["zx"], ker["wx"]
    nx, nl = ker["n_x"], ker["n_l"]
    dd = dict(dtype=MU.dtype, device=MU.device)
    mX = nu_f * mf + nu_s * ms
    vX = torch.clamp(nu_f ** 2 * Vff + 2 * nu_f * nu_s * Vfs + nu_s ** 2 * Vss, min=1e-16)
    sX = torch.sqrt(vX)
    x = mX[:, None] + sX[:, None] * zx[None, :]                                  # (N, nx)
    g = ker["gbar"] + x[:, :, None] + nu_l * zl[None, None, :]                    # (N, nx, nl)
    sig = torch.sqrt(torch.exp(g) * ker["dt"])                                    # (N, nx, nl)
    if n_p > 1:
        zp = torch.as_tensor(hermegauss(n_p)[0], **dd)
        wp = torch.as_tensor(hermegauss(n_p)[1], **dd); wp = wp / wp.sum()
        p = MU[:, None] + SG[:, None] * zp[None, :]                              # (N, np) own width
        lmp = torch.clamp(lam(p), min=1e-6)                                      # lambda AT the price
        sg4 = sig[:, None, :, :]                                                 # (N, 1, nx, nl)
        lm4 = lmp[:, :, None, None]                                              # (N, np, 1, 1)
        V = (lm4 * sg4) ** 2
        w3 = wx[None, None, :, None] * wl[None, None, None, :]
        # normaliser PER (component, p): the martingale must hold at each resolved price
        A = torch.log((w3 * torch.exp(ker["lam_skew"] * zl[None, None, None, :] * lm4 * sg4)
                       ).sum((2, 3)))                                            # (N, np)
        MUn = p[:, :, None, None] + (-0.5 * V + lm4 * ker["lam_skew"]
                                     * zl[None, None, None, :] * sg4 - A[:, :, None, None])
        SGn = torch.sqrt(V)                        # old SG now lives in the spread of the p_q means
        Wn = W[:, None, None, None] * wp[None, :, None, None] * w3
    else:
        lm = torch.clamp(lam(MU), min=1e-6)
        V = (lm[:, None, None] * sig) ** 2
        w3 = wx[None, :, None] * wl[None, None, :]
        A = torch.log((w3 * torch.exp(ker["lam_skew"] * zl[None, None, :] * lm[:, None, None] * sig)
                       ).sum((1, 2)))
        MUn = MU[:, None, None] + (-0.5 * V + lm[:, None, None] * ker["lam_skew"]
                                   * zl[None, None, :] * sig - A[:, None, None])
        SGn = torch.sqrt(SG[:, None, None] ** 2 + V)
        Wn = W[:, None, None] * w3
    # factor posterior given X = x (bivariate normal conditioning), then the exact AR(1) update
    bf = (nu_f * Vff + nu_s * Vfs) / vX
    bs = (nu_f * Vfs + nu_s * Vss) / vX
    d = sX[:, None] * zx[None, :]                                                # x - mX
    cmf = mf[:, None] + bf[:, None] * d
    cms = ms[:, None] + bs[:, None] * d
    cVff = Vff - bf * (nu_f * Vff + nu_s * Vfs)
    cVss = Vss - bs * (nu_f * Vfs + nu_s * Vss)
    cVfs = Vfs - bf * (nu_f * Vfs + nu_s * Vss)
    mfn = ker["kap_f"] * cmf[:, :, None] + ker["s_f"] * ker["rho_f"] * zl[None, None, :]  # (N,nx,nl)
    msn = ker["kap_s"] * cms[:, :, None] + ker["s_s"] * ker["rho_s"] * zl[None, None, :]
    if n_p > 1:                       # the factors do not depend on the PRICE sub-abscissa: broadcast
        mfn, msn = mfn[:, None], msn[:, None]
    sh = MUn.shape
    lead = lambda v: v.reshape((-1,) + (1,) * (len(sh) - 1))          # (N,) -> (N,1,..,1)
    Vffn = lead(ker["kap_f"] ** 2 * cVff + ker["q_f"]).expand(sh)
    Vssn = lead(ker["kap_s"] ** 2 * cVss + ker["q_s"]).expand(sh)
    Vfsn = lead(ker["kap_f"] * ker["kap_s"] * cVfs).expand(sh)
    flat = lambda t: t.reshape(-1)
    Wn = Wn.expand(sh)
    W2 = flat(Wn); W2 = W2 / W2.sum()
    return recompress_n((W2, flat(MUn.expand(sh)), flat(SGn.expand(sh)),
                         flat(mfn.expand(sh)), flat(msn.expand(sh)),
                         flat(Vffn), flat(Vssn), flat(Vfsn)), nk, nb_f, nb_s)


def _bands_n(w, x, nb):
    """Equal-weight bands by sorted x, membership STOP-GRADIENT (as `_merge`)."""
    with torch.no_grad():
        if x.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=x.device)
        o = torch.argsort(x)
        cw = torch.cumsum(w[o], 0)
        q = cw[-1] * (torch.arange(1, nb, dtype=w.dtype, device=w.device) / nb)
        e = torch.searchsorted(cw.contiguous(), q.contiguous(), right=True)
        ch = torch.searchsorted(e.contiguous(),
                                torch.arange(x.numel(), device=x.device), right=True)
        b = torch.empty(x.numel(), dtype=torch.long, device=x.device)
        b[o] = ch
        return b


def recompress_n(state, nk, nb_f, nb_s):
    """Cells on (mf, ms), merge on MU inside. The FACTOR block is moment-matched by the law of
    total covariance, so its mean AND covariance are preserved EXACTLY under any partition --
    cells are therefore a price-side knob only (16.9 step 2b, exact to 2.5e-16)."""
    W, MU, SG, mf, ms, Vff, Vss, Vfs = state
    with torch.no_grad():
        b1 = _bands_n(W, mf, nb_f)
        cell = torch.zeros_like(b1)
        # No `if bool(m.any())` guard on the two loops. It was inherited from the OLD `recompress`,
        # where it WAS required: that loop ran over genuine (f, s) regime indices and the old
        # `_merge` had no empty-input check, so an empty regime would have failed inside it. The
        # port added a `numel() == 0` guard to `_bands_n` (returning zeros(0)), which makes the
        # outer check redundant -- an all-False mask assignment is a no-op. Removal verified
        # bit-identical (0.000e+00) on SSR/beta/skew/vov and a raw 13-step chain, over 2 dates x 3
        # seeds. Each guard was a bool() device->host SYNC, 20 per call and 2925 calls per model
        # evaluation = 58,500 syncs; free on CPU, ruinous on MPS. This does NOT make the function
        # MPS-ready on its own -- the boolean mask indexing below (W[m], ms[m], cell[m]=) still has
        # a data-dependent shape, and `[keep]` at the return still makes the output ragged. See
        # HANDOFF_kernel_normalisation.md 6b.
        for f in range(nb_f):
            m = b1 == f
            cell[m] = f * nb_s + _bands_n(W[m], ms[m], nb_s)
        sub = torch.zeros_like(cell)
        for c in range(nb_f * nb_s):
            m = cell == c
            sub[m] = _bands_n(W[m], MU[m], nk)
        key = cell * nk + sub
        nc = nb_f * nb_s * nk
    # Segment-sum by index_add, NOT a dense (N, nc) one-hot. Identical result and identical gradient
    # (membership is stop-grad, the sums are differentiable in the values), but O(N) instead of
    # O(N*nc) -- the one-hot is a ~17e9-entry tensor at N~2e6, nc~8700 and will hang the machine.
    # `_merge_batch` already carries this fix and its comment; this is the same trap.
    def seg(x):
        # NOTE the `.to(out.dtype)` below is a NO-OP -- `out` is built with dtype=x.dtype, so the
        # cast can never convert anything. It was added against "the VOVLEV=1 path promotes some
        # intermediates to float64", which was a real symptom with the wrong remedy: the promotion
        # was in the TANGENT, not the primal, and `.dtype` reports only the primal. Root cause was
        # upstream in calibrate_joint_torch.solve_gbar_torch_n, where a Python-float `target`
        # (a double) divided a float32 dual and upcast its tangent; fixed there 2026-08-08.
        # Left as-is deliberately: harmless, and every existing fit was produced with it.
        out = torch.zeros(nc, dtype=x.dtype, device=x.device)
        return out.index_add(0, key.to(x.device), x.to(out.dtype))
    Wj = seg(W)
    keep = Wj > 1e-15
    Ws = torch.where(keep, Wj, torch.ones_like(Wj))
    avg = lambda v: seg(W * v) / Ws
    M = avg(MU)
    E2 = avg(SG ** 2 + MU ** 2)
    Mf, Ms = avg(mf), avg(ms)
    Ff = avg(Vff + mf * mf) - Mf * Mf          # law of total (co)variance
    Fs = avg(Vss + ms * ms) - Ms * Ms
    Fc = avg(Vfs + mf * ms) - Mf * Ms
    # MARTINGALE RE-LOCK, RELATIVE. The old form was `A = log(post_forward)`, i.e. it forced
    # E[S] = 1 after EVERY step. Starting from log-price z0 the forward should be exp(z0), so that
    # form SUBTRACTED THE STARTING LEVEL: after one step lambda(MU) saw MU ~ 0 instead of z0, the
    # price level was forgotten, and dsigma_ATM/dz collapsed like 1/n. Measured against MC
    # (price_side_test.py::mc_sigma_at): propagator/MC = 1.001 at 1wk but 0.100 at 13wk.
    # Correcting only the recompression DRIFT -- log(post) - log(pre) -- keeps the level.
    Vout = torch.clamp(E2 - M * M, min=1e-12)
    pre = torch.log((W * torch.exp(MU + 0.5 * SG ** 2)).sum())
    post = torch.log((Wj[keep] * torch.exp(M[keep] + 0.5 * Vout[keep])).sum())
    A = post - pre
    return (Wj[keep], M[keep] - A, torch.sqrt(Vout)[keep],
            Mf[keep], Ms[keep],
            torch.clamp(Ff, min=1e-12)[keep], torch.clamp(Fs, min=1e-12)[keep], Fc[keep])


def stationary_pi(ker):
    """Stationary (f,s) distribution by a differentiable linear solve (pi P = pi, sum pi = 1).

    With a FROZEN slow factor Ts is the identity, so the joint chain is reducible in s and the solve is
    singular -- any s-distribution is stationary. The correct answer is then the product of the fast
    factor's own stationary law with the slow factor's QUADRATURE weights.
    """
    nf, ns = ker["n_f"], ker["n_s"]; nfs = nf * ns
    if _SLOWFROZEN:
        Pf = torch.einsum("l,lac->ac", ker["wl"], ker["Tf"])
        dd = dict(dtype=Pf.dtype, device=Pf.device)
        A = torch.cat([(Pf.t() - torch.eye(nf, **dd))[:-1], torch.ones(1, nf, **dd)], 0)
        b = torch.zeros(nf, **dd); b[-1] = 1.0
        pf = torch.linalg.solve(A, b)
        return pf[:, None] * ker["ws"][None, :]
    P = torch.einsum("l,lac,lbd->abcd", ker["wl"], ker["Tf"], ker["Ts"]).reshape(nfs, nfs)
    dd = dict(dtype=P.dtype, device=P.device)
    A = torch.cat([(P.t() - torch.eye(nfs, **dd))[:-1], torch.ones(1, nfs, **dd)], 0)
    b = torch.zeros(nfs, **dd); b[-1] = 1.0
    return torch.linalg.solve(A, b).reshape(nf, ns)


def lev_increment(Vlr_fs, tilt_fs, wl, lam0):
    """One-step return moments (Dl, Vl) at the ATM leverage lam0 (ports lev_increment)."""
    Vl = lam0 ** 2 * Vlr_fs
    mtil = -0.5 * Vl + lam0 * tilt_fs
    A = torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum())
    return mtil - A, Vl


def interp_batch(query, xp, yp):
    """Linear interp of EACH row of yp (M, nz) at all `query` points, edge-clipped like np.interp.
    Fully vectorized (shared brackets since xp is common) -- no .item() sync. Differentiable in yp, query."""
    q = query.reshape(-1)
    idx = torch.searchsorted(xp, q).clamp(1, xp.shape[0] - 1)
    x0 = xp[idx - 1]; x1 = xp[idx]
    t = (q - x0) / (x1 - x0)                                          # (Nq,)
    out = yp[:, idx - 1] + t[None, :] * (yp[:, idx] - yp[:, idx - 1]) # (M, Nq)
    out = torch.where(q[None, :] < xp[0], yp[:, :1], out)
    out = torch.where(q[None, :] > xp[-1], yp[:, -1:], out)
    return out.reshape(yp.shape[0], *query.shape)


def fused_ssr_ts(theta, lam_fns, n, dt, nz=13, zmax=0.12, Q=5, nk=16, n_f=5, n_s=3, n_l=5):
    """Full exact-beta term-structure SSR in torch (ports slv_fast.fused_ssr_exact_ts). Differentiable in
    theta -> SSR. sigma_ATM(z,f,s) grid via propagate+atm_skew; analytic covariance sum over the one-step
    transition (nodes l x GH sub-quadrature q x dest regimes)."""
    ker = build_kernel(theta, dt, n_f, n_s, n_l)
    zg = torch.linspace(-zmax, zmax, nz); iz0 = nz // 2
    sig_rows = []; skw0 = []
    for f in range(n_f):
        for s in range(n_s):
            row = []
            for iz in range(nz):
                st = (torch.ones(1), zg[iz].reshape(1).clone(), torch.full((1,), 1e-4), torch.tensor([f]), torch.tensor([s]))
                for k in range(n):
                    st = propagate(st, ker, lam_fns[k], nk)
                a, sk = atm_skew(st[0], st[1], st[2], n * dt)
                row.append(a)
                if iz == iz0:
                    skw0.append(sk)
            sig_rows.append(torch.stack(row))
    sigg = torch.stack(sig_rows).reshape(n_f, n_s, nz)                 # (nf,ns,nz)
    skw0 = torch.stack(skw0).reshape(n_f, n_s)
    pi = stationary_pi(ker); lam0 = lam_fns[0](torch.zeros(1))[0]
    zq_np, wq_np = hermegauss(Q); wq_np = wq_np / wq_np.sum()
    zq = torch.tensor(zq_np); wq = torch.tensor(wq_np); wl = ker["wl"]
    wlq = wl[:, None] * wq[None, :]                                   # (nl, Q)
    siggflat = sigg.reshape(n_f * n_s, nz)
    beta_num = torch.zeros(()); var = torch.zeros(())
    for f in range(n_f):                                             # per starting regime; l/q/dest all vectorized
        for s in range(n_s):
            Dl, Vl = lev_increment(ker["Vl"][f, s], ker["tilt"][f, s], wl, lam0)   # (nl,)
            mean_r = (wl * Dl).sum()
            var = var + pi[f, s] * ((wl * (Dl ** 2 + Vl)).sum() - mean_r ** 2)
            sd = torch.sqrt(torch.clamp(Vl, min=1e-16))
            r = Dl[:, None] + sd[:, None] * zq[None, :]              # (nl, Q) one-step return at node l, sub-point q
            SIG = interp_batch(r, zg, siggflat).reshape(n_f, n_s, n_l, Q)          # sigma_ATM at r per dest (fp,sp)
            P = ker["Tf"][:, f, :][:, :, None] * ker["Ts"][:, s, :][:, None, :]    # (nl, nf, ns) dest weights
            sig_dest = torch.einsum("lab,ablq->lq", P, SIG)          # (nl, Q)
            cov = (wlq * (sig_dest - sigg[f, s, iz0]) * (r - mean_r)).sum()
            beta_num = beta_num + pi[f, s] * cov
    beta = beta_num / var; skew = (pi * skw0).sum()
    return beta / skew, beta, skew


_SQRT2 = 2.0 ** 0.5
_SQRT2PI = (2.0 * np.pi) ** 0.5


def _stat_nodes_n(ker, na, nb):
    """Quadrature over the STATIONARY bivariate law of (u_f, u_s): unit marginals, corr = c.
    This is the pi-average the old `stationary_pi` linear solve provided -- and it is a legitimate
    quadrature (integrated and discarded), unlike the abscissas the old kernel CARRIED (16.0d).

    RECTANGULAR in (u_f, u_s) -- the correlation is carried in the WEIGHTS, not by shearing the grid.

    An earlier version sheared it (u_s = c za + sqrt(1-c^2) zb) and interpolated in (za, zb). That is
    WRONG here: zb depends on BOTH u_f and u_s, so after the transform the destination coordinates
    are no longer separable, yet `fused_ssr_ts_n` uses them as independent tensor axes -- the two
    factor innovations get cross-paired and the factor-change channel is scrambled. Symptom: beta
    converged to ~-1.04 against the ~-1.68 a flat SSR needs, and REFINING THE GRID DID NOT HELP,
    which is what ruled resolution out and pointed here.

    With a rectangular grid the weight is the bivariate normal density over the product of marginals:
        w_ij = wa_i wb_j * phi_2(u_i, u_j; c) / (phi(u_i) phi(u_j))
    renormalised to sum to 1.
    """
    dd = dict(dtype=ker["zl"].dtype, device=ker["zl"].device)
    za = torch.as_tensor(hermegauss(na)[0], **dd); wa = torch.as_tensor(hermegauss(na)[1], **dd)
    zb = torch.as_tensor(hermegauss(nb)[0], **dd); wb = torch.as_tensor(hermegauss(nb)[1], **dd)
    wa = wa / wa.sum(); wb = wb / wb.sum()
    c = stationary_n(ker)
    uf = za[:, None].expand(na, nb).contiguous()
    us = zb[None, :].expand(na, nb).contiguous()
    q = torch.clamp(1 - c ** 2, min=1e-12)
    # log of the Gaussian-copula density ratio; the marginal factors cancel against wa, wb
    lr = -0.5 * torch.log(q) - (c ** 2 * (uf ** 2 + us ** 2) - 2 * c * uf * us) / (2 * q)
    W = (wa[:, None] * wb[None, :]) * torch.exp(lr - lr.max())
    return uf, us, W / W.sum(), za, zb


def _stat_nodes_n_OLD_SHEARED(ker, na, nb):
    """Kept only to document the bug above. Do not use."""
    dd = dict(dtype=ker["zl"].dtype, device=ker["zl"].device)
    za = torch.as_tensor(hermegauss(na)[0], **dd); wa = torch.as_tensor(hermegauss(na)[1], **dd)
    zb = torch.as_tensor(hermegauss(nb)[0], **dd); wb = torch.as_tensor(hermegauss(nb)[1], **dd)
    wa = wa / wa.sum(); wb = wb / wb.sum()
    c = stationary_n(ker)
    uf = za[:, None].expand(na, nb)
    us = c * za[:, None] + torch.sqrt(torch.clamp(1 - c ** 2, min=0.0)) * zb[None, :]
    return uf.contiguous(), us.expand(na, nb).contiguous(), (wa[:, None] * wb[None, :]), za, zb


def _interp_lin(xp, q):
    """Linear-interp weights of query q onto sorted abscissas xp: (idx0, idx1, t), edge-clipped."""
    i = torch.searchsorted(xp.contiguous(), q.reshape(-1).contiguous()).clamp(1, xp.shape[0] - 1)
    x0, x1 = xp[i - 1], xp[i]
    t = torch.clamp((q.reshape(-1) - x0) / (x1 - x0), 0.0, 1.0)
    return (i - 1).reshape(q.shape), i.reshape(q.shape), t.reshape(q.shape)


def fused_ssr_ts_multi_n(theta9, lam_fns, NS, dt, **kw):
    """All tenors in NS from ONE propagation chain per grid point, snapshotted -- the `_n` twin of
    `fused_ssr_ts_multi`. Bit-identical to calling `fused_ssr_ts_n` per tenor (the per-tenor chains
    are prefixes of the longest), at sum(NS) -> max(NS) propagation steps: 28 -> 13, a 2.15x cut.

    Returns a stacked tensor of SSR values, one per tenor in NS.
    """
    ssr, _, _ = _ssr_core_n(theta9, lam_fns, NS, dt, **kw)      # ONE call -- it is already multi-tenor
    return torch.stack(ssr)


def fused_ssr_ts_n(theta9, lam_fns, n, dt, **kw):
    """Single-tenor SSR. Thin wrapper on the multi-tenor core; returns (SSR, beta, skew)."""
    ssr, beta, skew = _ssr_core_n(theta9, lam_fns, [n], dt, **kw)
    return ssr[0], beta[0], skew[0]


def _ssr_core_n(theta9, lam_fns, NS, dt, nz=13, zmax=0.12, Q=5, nk=16,
                n_l=5, n_x=3, n_p=5, na=5, nb=5, ne=3, nb_f=5, nb_s=3):
    """SSR term structure on the normalised kernel (16.9). Port of `fused_ssr_ts`.

    WHAT CHANGED. The old routine looped over (f, s) regimes, read `ker["Vl"][f,s]` / `ker["tilt"][f,s]`
    as lookups, took destination weights from `Tf[:,f,:] kron Ts[:,s,:]`, and pi-averaged with the
    `stationary_pi` linear solve. All four were consequences of the factors being CARRIED abscissa
    indices. Here:

      * the (f, s) loop becomes a QUADRATURE over the stationary bivariate Gaussian (`_stat_nodes_n`)
        -- 25 evaluations at na=5, nb=5. This was 15 (na=5, nb=3) to match the old n_f*n_s "so cost
        is unchanged", but nb=3 left a 27.9% worst-case error in the pi-average: the integrand is
        exp(nu_f u_f + nu_s u_s), not a polynomial, so the rule must resolve a function of SCALE nu_s
        (1.16-2.13 across the panel). nb=7 brings it to 0.118%; raising na instead does nothing;
      * Vl and tilt are computed from (u_f, u_s) directly, no lookup;
      * destination weights become the one-step factor law, which is Gaussian: given the branch z_l,
        u' = kap u + s (rho z_l + sqrt(1-rho^2) eps), quadratured over eps with `ne` abscissas;
      * pi is the GH weight product -- no linear solve, and no `_SLOWFROZEN` special case.

    sigma_ATM is now a function of (z, u_f, u_s) rather than (z, f, s), so the destination lookup is
    a trilinear interpolation instead of a regime index.

    lam0 NOTE. `lev_increment` evaluates the one-step return at log-price 0 with lambda(0). That is a
    POINT, not a component with width, so 16.10 does not apply to it -- and SSR is by definition a
    local at-the-money regression, so a point evaluation is what the definition asks for. 16.10
    enters this routine through `sigg`/`skw0`, which are built by the propagator; passing n_p fixes
    them, and beta and skew then share the same treatment (which is why the beta/skew cancellation
    question is live again rather than settled against).
    """
    ker = build_kernel_n(theta9, dt, n_l=n_l, n_x=n_x)
    dd = dict(dtype=ker["zl"].dtype, device=ker["zl"].device)
    zg = torch.linspace(-zmax, zmax, nz, **dd); iz0 = nz // 2
    UF, US, PI, za, zb = _stat_nodes_n(ker, na, nb)
    zl, wl = ker["zl"], ker["wl"]
    zero = torch.zeros(1, **dd)

    # ---- sigma_ATM(z, u_f, u_s, tenor): ONE chain per grid point, SNAPSHOTTED at every tenor.
    # The per-tenor chains are prefixes of the longest, so this is bit-identical to propagating each
    # tenor separately, at sum(NS) -> max(NS) steps (28 -> 13 for NS = [1,2,4,8,13]).
    nT = len(NS); nmax = max(NS); snap = {v: i for i, v in enumerate(NS)}
    sig = torch.zeros(nT, na, nb, nz, **dd)
    skw0 = torch.zeros(nT, na, nb, **dd)
    for a in range(na):
        for b in range(nb):
            for iz in range(nz):
                st = (torch.ones(1, **dd), zg[iz].reshape(1).clone(), torch.full((1,), 1e-4, **dd),
                      UF[a, b].reshape(1).clone(), US[a, b].reshape(1).clone(),
                      zero.clone(), zero.clone(), zero.clone())
                for k in range(nmax):
                    st = propagate_n(st, ker, lam_fns[k], nk, nb_f=nb_f, nb_s=nb_s, n_p=n_p)
                    if (k + 1) not in snap:
                        continue
                    t = snap[k + 1]; n = k + 1
                # ATM means AT-THE-FORWARD. Since the re-lock in `recompress_n` is now RELATIVE, the
                # state correctly carries its level and the forward is exp(z0), not 1 -- so `atm_skew`
                # (which prices at strike 1) must be fed a forward-centred MU. Without this the
                # measured dsigma/dz is the moneyness effect and comes out with the WRONG SIGN.
                    Wn_ = st[0] / st[0].sum()
                    lF = torch.log((Wn_ * torch.exp(st[1] + 0.5 * st[2] ** 2)).sum())
                    v, sk = atm_skew(Wn_, st[1] - lF, st[2], n * dt)
                    sig[t, a, b, iz] = v
                    if iz == iz0:
                        skw0[t, a, b] = sk

    # ---- one-step return moments at ATM, per stationary node. lambda(0): a point, see docstring.
    lam0 = lam_fns[0](zero)[0]
    X = ker["nu_f"] * UF + ker["nu_s"] * US                                   # (na,nb)
    gg = torch.clamp(ker["gbar"] + X[..., None] + ker["nu_l"] * zl, min=-40.0, max=12.0)
    Vraw = torch.exp(gg) * dt                                                 # (na,nb,nl)
    Vl = lam0 ** 2 * Vraw
    sdl = torch.sqrt(torch.clamp(Vl, min=1e-16))
    mtil = -0.5 * Vl + ker["lam_skew"] * zl * sdl
    A = torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum(-1, keepdim=True))
    Dl = mtil - A
    mean_r = (wl * Dl).sum(-1)                                                # (na,nb)
    var = (PI * ((wl * (Dl ** 2 + Vl)).sum(-1) - mean_r ** 2)).sum()

    # ---- destination: one-step factor law, Gaussian, quadratured over the fresh innovation
    ze = torch.as_tensor(hermegauss(ne)[0], **dd)
    we = torch.as_tensor(hermegauss(ne)[1], **dd); we = we / we.sum()
    zq = torch.as_tensor(hermegauss(Q)[0], **dd)
    wq = torch.as_tensor(hermegauss(Q)[1], **dd); wq = wq / wq.sum()
    r = Dl[..., None] + sdl[..., None] * zq                                   # (na,nb,nl,Q)
    kf, ks, sf, ss = ker["kap_f"], ker["kap_s"], ker["s_f"], ker["s_s"]
    ef = sf * torch.sqrt(torch.clamp(1 - ker["rho_f"] ** 2, min=0.0))
    es = ss * torch.sqrt(torch.clamp(1 - ker["rho_s"] ** 2, min=0.0))
    ufn = kf * UF[..., None, None] + sf * ker["rho_f"] * zl[:, None] + ef * ze     # (na,nb,nl,ne)
    usn = ks * US[..., None, None] + ss * ker["rho_s"] * zl[:, None] + es * ze
    # trilinear interpolation of sig over (za, zb, zg). The grid is RECTANGULAR in (u_f, u_s), so
    # the two factor coordinates interpolate independently -- no transform, nothing to cross-pair.
    ia0, ia1, ta = _interp_lin(za, ufn)
    ib0, ib1, tb = _interp_lin(zb, usn)
    iz_0, iz_1, tz = _interp_lin(zg, r)
    beta_all, skew_all, ssr_all = [], [], []
    for t in range(nT):
      beta_num = torch.zeros((), **dd)
      for a in range(na):
       for b in range(nb):
            # (nl, ne) factor destinations x (nl, Q) returns -> (nl, ne, ne', Q); ef/es innovations
            # are independent, so the two factor axes are separate quadratures.
            A0, A1, TA = ia0[a, b], ia1[a, b], ta[a, b]                        # (nl,ne)
            B0, B1, TB = ib0[a, b], ib1[a, b], tb[a, b]
            Z0, Z1, TZ = iz_0[a, b], iz_1[a, b], tz[a, b]                      # (nl,Q)
            sg_t = sig[t]
            g = lambda A, B, Z: sg_t[A[:, :, None, None], B[:, None, :, None], Z[:, None, None, :]]
            s000 = g(A0, B0, Z0); s001 = g(A0, B0, Z1)
            s010 = g(A0, B1, Z0); s011 = g(A0, B1, Z1)
            s100 = g(A1, B0, Z0); s101 = g(A1, B0, Z1)
            s110 = g(A1, B1, Z0); s111 = g(A1, B1, Z1)
            wA = TA[:, :, None, None]; wB = TB[:, None, :, None]; wZ = TZ[:, None, None, :]
            s_dest = ((1 - wA) * ((1 - wB) * ((1 - wZ) * s000 + wZ * s001)
                                  + wB * ((1 - wZ) * s010 + wZ * s011))
                      + wA * ((1 - wB) * ((1 - wZ) * s100 + wZ * s101)
                              + wB * ((1 - wZ) * s110 + wZ * s111)))           # (nl,ne,ne,Q)
            w4 = (wl[:, None, None, None] * we[None, :, None, None]
                  * we[None, None, :, None] * wq[None, None, None, :])
            cov = (w4 * (s_dest - sig[t, a, b, iz0]) * (r[a, b][:, None, None, :] - mean_r[a, b])).sum()
            beta_num = beta_num + PI[a, b] * cov
      beta = beta_num / var
      skew = (PI * skw0[t]).sum()
      beta_all.append(beta); skew_all.append(skew); ssr_all.append(beta / skew)
    return ssr_all, beta_all, skew_all


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _phi(x):
    return torch.exp(-0.5 * x ** 2) / _SQRT2PI


def gm_call(Kstrike, W, MU, SG):
    """Closed-form GM call price (sum of Black calls), unit-forward. Differentiable. Ports gm_call."""
    F = torch.exp(MU + 0.5 * SG ** 2)
    d1 = (torch.log(F / Kstrike) + 0.5 * SG ** 2) / SG
    return (W * (F * _Phi(d1) - Kstrike * _Phi(d1 - SG))).sum()


def _bs_call(Kstrike, u):
    d1 = (-torch.log(Kstrike) + 0.5 * u ** 2) / u
    return _Phi(d1) - Kstrike * _Phi(d1 - u)                          # unit-forward Black call, total-vol u


def _implied_u(Kstrike, C, u0, iters=8):
    """Differentiable BS implied total-vol by unrolled Newton (vega = phi(d1)); matches v1's bisection."""
    u = u0
    for _ in range(iters):
        d1 = (-torch.log(Kstrike) + 0.5 * u ** 2) / u
        vega = torch.clamp(_phi(d1), min=1e-10)
        u = torch.clamp(u - (_bs_call(Kstrike, u) - C) / vega, min=1e-6)
    return u


def atm_skew(W, MU, SG, T, dm=6e-3):
    """ATM implied vol + ATM skew of a GM marginal -- CLOSED-FORM + differentiable (ports atm_skew_of).
    ATM guess by probit u=2*sqrt(2)*erfinv(C_atm); the three IVs (0,+-dm) by differentiable Newton;
    skew = central difference of the inverted IVs (same dm as v1)."""
    sqrtT = float(T) ** 0.5
    one = torch.ones((), dtype=MU.dtype)
    Catm = gm_call(one, W, MU, SG)
    u0 = 2.0 * _SQRT2 * torch.erfinv(torch.clamp(Catm, -0.999, 0.999))   # ATM total-vol guess (exact at ATM)
    Klo = torch.exp(torch.tensor(-dm)); Khi = torch.exp(torch.tensor(dm))
    u_a = _implied_u(one, Catm, u0)
    u_lo = _implied_u(Klo, gm_call(Klo, W, MU, SG), u0)
    u_hi = _implied_u(Khi, gm_call(Khi, W, MU, SG), u0)
    sig_atm = u_a / sqrtT
    skew = (u_hi - u_lo) / sqrtT / (2 * dm)
    return sig_atm, skew


def kstep_law_n(ker, m, V, k):
    """k-step law of (u_f, u_s) given a Gaussian (m, V) today. CLOSED FORM -- this is what replaces
    `matrix_power(Pj, n_opt)` and the kron product in `vix_ivol`. m = (mf, ms), V = (Vff, Vss, Vfs),
    all tensors broadcastable together.

        V_ff,k = kap_f^2k V_ff,0 + (1 - kap_f^2k)
        V_fs,k = (kap_f kap_s)^k V_fs,0 + s_f s_s rho_f rho_s (1 - (kap_f kap_s)^k)/(1 - kap_f kap_s)

    Sanity: at the stationary law (V_ff,0 = 1, V_fs,0 = the 16.9 corr) both are invariant in k, as
    they must be. The z_l cross term is the shared shock accumulating over the k steps -- it does NOT
    factorise, which is the same thing the old code's comment says about Pbar != kron of marginals.
    """
    mf, ms = m
    Vff, Vss, Vfs = V
    kf, ks = ker["kap_f"], ker["kap_s"]
    af, as_ = kf ** k, ks ** k
    cross = ker["s_f"] * ker["s_s"] * ker["rho_f"] * ker["rho_s"]
    kk = kf * ks
    geo = (1 - kk ** k) / torch.clamp(1 - kk, min=1e-12)
    return ((af * mf, as_ * ms),
            (af ** 2 * Vff + (1 - af ** 2),
             as_ ** 2 * Vss + (1 - as_ ** 2),
             (kk) ** k * Vfs + cross * geo))


def _vfull_n(ker, mX, vX):
    """E[ full one-step increment variance ] over X = nu_f u_f + nu_s u_s ~ N(mX, vX), unlevered.
    Mirrors the old `Vfull = E_l[Vl] + Var_l(D)` but with the regime lookup replaced by a quadrature
    on X's OWN density (abscissas rescaled to it -- see 16.0d on why that distinction is the bug)."""
    zl, wl, zx, wx = ker["zl"], ker["wl"], ker["zx"], ker["wx"]
    x = mX[..., None] + torch.sqrt(torch.clamp(vX, min=1e-16))[..., None] * zx        # (..., nx)
    # Guard the log-variance. At large na/nb the stationary abscissas reach |u| ~ 4.7, so
    # nu_f u_f + nu_s u_s can hit +-6 and exp() overflows to inf -> NaN downstream. The bound is
    # generous (e^{+-12} on the variance) and only ever binds on grid points the fit never visits.
    gg = torch.clamp(ker["gbar"] + x[..., None] + ker["nu_l"] * zl, min=-40.0, max=12.0)
    Vl = torch.exp(gg) * ker["dt"]                                                     # (..., nx, nl)
    mtil = -0.5 * Vl + ker["lam_skew"] * zl * torch.sqrt(Vl)
    m1 = (wl * mtil).sum(-1)
    m2 = (wl * mtil ** 2).sum(-1)
    Vfull = (wl * Vl).sum(-1) + (m2 - m1 ** 2)                                        # (..., nx)
    return (wx * Vfull).sum(-1)                                                       # integrate X out


def vix_n(ker, sig_ref, m, V, n_var):
    """VIX (30d forward vol) as a function of the factor law (m, V) today. The old code walked the
    regime chain `n_var` times with a matrix power; here each step's law is closed form."""
    nu_f, nu_s = ker["nu_f"], ker["nu_s"]
    tot = 0.0
    for k in range(1, n_var + 1):
        (mf, ms), (Vff, Vss, Vfs) = kstep_law_n(ker, m, V, k)
        mX = nu_f * mf + nu_s * ms
        vX = nu_f ** 2 * Vff + 2 * nu_f * nu_s * Vfs + nu_s ** 2 * Vss
        tot = tot + _vfull_n(ker, mX, vX)
    # normalise so the STATIONARY expectation is sig_ref^2, exactly as the old `v = sig_ref^2 Vfull/EV`
    c = stationary_n(ker)
    z0 = torch.zeros((), dtype=ker["zl"].dtype, device=ker["zl"].device)
    one = torch.ones((), dtype=z0.dtype, device=z0.device)
    mXs = z0
    vXs = nu_f ** 2 + 2 * nu_f * nu_s * c + nu_s ** 2
    EV = _vfull_n(ker, mXs, vXs)
    return torch.sqrt(torch.clamp(sig_ref ** 2 * (tot / n_var) / EV, min=1e-16))


def vix_ivol_n(ker, sig_ref, tau_opt, spot, tau_var=30.0 / 365.0, lam_fns=None,
               nk_vix=16, n_p=5, nq=7):
    """Normalised-kernel VIX ATM implied vol (16.9). Port of `vix_ivol`.

    WHAT GOT SIMPLER. The old routine needed the joint regime chain: a kron product per branch, a
    matrix power for the terminal law, and an argmin over slow-regime abscissas to match spot. All
    three are gone -- the k-step factor law is closed form (`kstep_law_n`), so the terminal law is a
    bivariate Gaussian and 'today's slow level' is a scalar solve on a continuous u_s instead of a
    pick from 3 abscissas. That last point matters: the old spot match could only land on one of
    n_s = 3 values.

    lam_fns given -> the LEVERAGED readout (VOVLEV). Same motivation as before: instantaneous
    variance is lambda(x)^2 exp(g), so the 30d variance swap rate carries lambda too, and the
    unlevered path above levers it on the SSR side but not the vov side. It needs the joint
    (log-price, factor) law at expiry, which `propagate_n` supplies -- now with n_p resolving
    lambda across each component's width (16.10) rather than freezing it at the mean.
    """
    dd_ = dict(dtype=ker["zl"].dtype, device=ker["zl"].device)
    dt = ker["dt"]
    n_var = max(1, int(round(tau_var / dt)))
    n_opt = max(1, int(round(tau_opt / dt)))
    c0 = stationary_n(ker)
    one = torch.ones((), **dd_)
    # today's slow level: solve VIX(u_f = 0, u_s) = spot on a CONTINUOUS u_s (old code picked one of
    # n_s = 3 abscissas by argmin). Bisection, no grad -- it is a state choice, not a parameter.
    with torch.no_grad():
        lo, hi = torch.full((), -6.0, **dd_), torch.full((), 6.0, **dd_)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            vm = vix_n(ker, sig_ref, (torch.zeros((), **dd_), mid),
                       (one * 0.0, one * 0.0, one * 0.0), n_var)
            hi = torch.where(vm > spot, mid, hi)
            lo = torch.where(vm > spot, lo, mid)
        us0 = 0.5 * (lo + hi)
    m0 = (torch.zeros((), **dd_), us0)
    V0 = (torch.zeros((), **dd_), torch.zeros((), **dd_), torch.zeros((), **dd_))
    if lam_fns is None:
        # terminal factor law: bivariate Gaussian, closed form. Quadrature over it, Cholesky-rotated.
        (mfT, msT), (Vff, Vss, Vfs) = kstep_law_n(ker, m0, V0, n_opt)
        zq = torch.as_tensor(hermegauss(nq)[0], **dd_)
        wq = torch.as_tensor(hermegauss(nq)[1], **dd_); wq = wq / wq.sum()
        sf = torch.sqrt(torch.clamp(Vff, min=1e-16))
        rho = Vfs / torch.clamp(sf * torch.sqrt(torch.clamp(Vss, min=1e-16)), min=1e-16)
        ss = torch.sqrt(torch.clamp(Vss, min=1e-16))
        uf = mfT + sf * zq[:, None]
        us = msT + ss * (rho * zq[:, None] + torch.sqrt(torch.clamp(1 - rho ** 2, min=0.0))
                         * zq[None, :])
        w2 = wq[:, None] * wq[None, :]
        z = torch.zeros_like(uf)
        vixT = vix_n(ker, sig_ref, (uf, us), (z, z, z), n_var)
        F = (w2 * vixT).sum()
        Catm = (w2 * torch.clamp(vixT - F, min=0.0)).sum()
    else:
        st = init_state_n(ker)
        st = (st[0], st[1], st[2], st[3] * 0.0, st[3] * 0.0 + us0,
              st[5] * 0.0, st[6] * 0.0, st[7] * 0.0)
        for k in range(n_opt):
            st = propagate_n(st, ker, lam_fns[min(k, len(lam_fns) - 1)], nk_vix, n_p=n_p)
        W, MU, SG, mf, ms, Vff, Vss, Vfs = st
        zq = torch.as_tensor(hermegauss(Q_VIX)[0], **dd_)
        wq = torch.as_tensor(hermegauss(Q_VIX)[1], **dd_); wq = wq / wq.sum()
        x = MU[:, None] + SG[:, None] * zq[None, :]                     # (N,Q) log-price at expiry
        if _VOVLAMTEN == "avg":
            # AVERAGE lambda^2 OVER THE VIX'S OWN WINDOW rather than reading one slice.
            # The VIX at expiry tau covers [tau, tau+30d] = steps n_opt .. n_opt+n_var, and
            # instantaneous variance is lambda(x)^2 exp(g), so what multiplies is the RMS of lambda
            # across those steps -- the mean of lambda^2 then a sqrt. More correct, and it smooths
            # by construction: consecutive tenors share n_var-1 of their n_var slices.
            #
            # It matters because the ladder is fitted per week and jumps 11.9% whenever consecutive
            # weeks cross a pillar (2.8% when they do not) -- `blend` is piecewise-linear in T so
            # dC/dT is discontinuous (handoff 6e.1b). One slice per tenor puts that straight into the
            # model's vov curve and no theta can cancel it.
            idx = [min(n_opt - 1 + m, len(lam_fns) - 1) for m in range(n_var)]
            l2 = None
            for i in idx:
                li2 = torch.clamp(lam_fns[i](x), min=1e-6) ** 2
                l2 = li2 if l2 is None else l2 + li2
            lam_x = torch.sqrt(l2 / float(len(idx)))
        else:
            if _VOVLAMTEN == "fix30":      li = min(n_var, len(lam_fns) - 1)
            elif _VOVLAMTEN == "mid":      li = min(n_opt + n_var // 2 - 1, len(lam_fns) - 1)
            else:                          li = min(n_opt - 1, len(lam_fns) - 1)
            lam_x = torch.clamp(lam_fns[li](x), min=1e-6)
        wj = (W / W.sum())[:, None] * wq[None, :]
        lam_x = lam_x / (wj * lam_x).sum()                              # LEVEL pinned by sig_ref
        if _VIXFIX:
            # Expand each component's factor law N(m, V) onto the SAME Gauss-Hermite rule the
            # unlevered branch above uses, and evaluate vix_n with ZERO variance at each node.
            #
            # WHY. `vix_n(m, V)` reads V as remaining uncertainty about the factor AT THE VALUATION
            # DATE and folds it into the variance-swap rate as convexity. At the option expiry the
            # factor is REALISED, so that uncertainty is zero -- which is why the lam_fns is None
            # branch passes (z, z, z) after placing its nodes. Passing V here converted genuine
            # DISPERSION of the VIX into a convexity correction on its MEAN.
            #
            # It matters because the NORMALISED kernel (16.9) carries each factor as a Gaussian
            # (m, V) per component: only the return branch zeta_l moves m ACROSS components, while
            # the fresh innovation eps goes INSIDE V. So the old form retained only the rho-share of
            # the vol-of-vol. Measured ratio to the unlevered readout at lambda == 1, tenor 8d:
            # 0.000 at rho=0, 0.275 at rho=0.239, 0.944 at rho=0.99 -- i.e. EXACTLY ZERO vol-of-vol
            # at zero leverage, for a model with unchanged nu_f, nu_s. After the fix: 1.000, 1.004,
            # 0.956. The FORWARD was always right (0.25% at 8d), which is how this survived so long.
            #
            # The pre-16.9 `vix_ivol` below does NOT have this defect and needs no change: it carries
            # the factor as a discrete regime index (Fi, Si) with no within-component variance, so
            # all of its factor uncertainty is already across components.
            zf = torch.as_tensor(hermegauss(nq)[0], **dd_)
            wf = torch.as_tensor(hermegauss(nq)[1], **dd_); wf = wf / wf.sum()
            sfc = torch.sqrt(torch.clamp(Vff, min=1e-16))
            ssc = torch.sqrt(torch.clamp(Vss, min=1e-16))
            rhoc = Vfs / torch.clamp(sfc * ssc, min=1e-16)
            orc = torch.sqrt(torch.clamp(1 - rhoc ** 2, min=0.0))
            ufc = mf[:, None, None] + sfc[:, None, None] * zf[None, :, None]
            usc = (ms[:, None, None] + ssc[:, None, None]
                   * (rhoc[:, None, None] * zf[None, :, None] + orc[:, None, None] * zf[None, None, :]))
            zc = torch.zeros_like(usc)
            vixT = vix_n(ker, sig_ref, (ufc, usc), (zc, zc, zc), n_var)  # (N, nq, nq)
            vx = lam_x[:, None, None, :] * vixT[..., None]               # (N, nq, nq, Q)
            wfull = wj[:, None, None, :] * (wf[:, None] * wf[None, :])[None, :, :, None]
        else:
            vixT = vix_n(ker, sig_ref, (mf, ms), (Vff, Vss, Vfs), n_var)   # the OLD, broken form
            vx = lam_x * vixT[:, None]
            wfull = wj
        F = (wfull * vx).sum(); Catm = (wfull * torch.clamp(vx - F, min=0.0)).sum()
    iv = (2.0 / tau_opt ** 0.5) * _SQRT2 * torch.erfinv(torch.clamp(Catm / F, -0.999, 0.999))
    return F, iv


def vix_ivol(ker, sig_ref, tau_opt, spot, tau_var=30.0 / 365.0, lam_fns=None, nk_vix=16):
    """Torch VIX ATM implied vol (differentiable), ports vix_readout.model_vix_ivol: per-regime forward
    variance v = sig_ref^2 * Vbar/E_pi[Vbar]; 30d forward variance by exact multi-step transition; terminal
    regime law = spot-matched slow regime propagated tau_opt steps; atomic VIX ATM option, probit inversion."""
    wl = ker["wl"]; dt = ker["dt"]; nf, ns = ker["n_f"], ker["n_s"]
    dd_ = dict(dtype=wl.dtype, device=wl.device)
    EVl = (wl * ker["Vl"]).sum(-1); md = (wl * ker["D"]).sum(-1); md2 = (wl * ker["D"] ** 2).sum(-1)
    Vfull = EVl + (md2 - md ** 2)                                     # (nf,ns) full one-step increment variance
    pi = stationary_pi(ker); EV = (pi * Vfull).sum()
    v = sig_ref ** 2 * Vfull / EV                                    # forward-variance rate per regime
    # EXACT joint regime transition on the product state (f,s), index f*ns+s. The branch index l is
    # COMMON to both factors, so the joint transition does NOT factorise into the branch-averaged
    # marginals: Pbar = sum_l w_l (Tf^(l) kron Ts^(l)) != (sum_l w_l Tf^(l)) kron (sum_l w_l Ts^(l)).
    # The two agree only when lam_f=lam_s=0 (branch drops out of both transitions).
    Pj = sum(wl[l] * torch.kron(ker["Tf"][l], ker["Ts"][l]) for l in range(ker["n_l"]))
    n_var = max(1, int(round(tau_var / dt)))
    vv = v.reshape(-1); Y = torch.zeros_like(vv)
    Pm = torch.eye(nf * ns, dtype=v.dtype, device=v.device)
    for _ in range(n_var):
        Pm = Pm @ Pj; Y = Y + Pm @ vv                                # 30d forward variance, exact multi-step
    vix = torch.sqrt((Y / n_var).reshape(nf, ns))
    n_opt = max(1, int(round(tau_opt / dt))); fc = nf // 2
    with torch.no_grad():
        s0 = int(torch.argmin(torch.abs(vix[fc, :] - spot)))         # today's slow regime = VIX spot
    if lam_fns is None:
        p = torch.matrix_power(Pj, n_opt)[fc * ns + s0].reshape(nf, ns)  # terminal regime law (joint)
        F = (p * vix).sum(); Catm = (p * torch.clamp(vix - F, min=0.0)).sum()
    else:
        # LEVERAGED VIX (lam_fns given). Instantaneous variance in this model is lambda(x)^2 * exp(g),
        # so the 30d variance swap rate -- the VIX -- carries lambda too. The regime-only path above
        # uses ker["Vl"] RAW, i.e. un-levered, while propagate() levers it (lm**2 * Vlr): the same
        # variance is levered on the SSR path and not on the vov path. A constant lambda would cancel
        # in `v = sig_ref^2 * Vfull / EV`, so only the VARIATION of lambda matters, and capturing it
        # needs the joint (log-price, regime) law at expiry rather than the regime chain alone.
        st = (torch.ones(1, **dd_), torch.zeros(1, **dd_), torch.full((1,), 1e-4, **dd_),
              torch.full((1,), fc, dtype=torch.long), torch.full((1,), s0, dtype=torch.long))
        for k in range(n_opt):
            st = propagate(st, ker, lam_fns[min(k, len(lam_fns) - 1)], nk_vix)
        W, MU, SG, Fi, Si = st
        zq = torch.as_tensor(hermegauss(Q_VIX)[0], **dd_)
        wq = torch.as_tensor(hermegauss(Q_VIX)[1], **dd_); wq = wq / wq.sum()
        x = MU[:, None] + SG[:, None] * zq[None, :]                  # (N,Q) log-price at expiry
        if _VOVLAMTEN == "fix30":      li = min(n_var, len(lam_fns) - 1)
        elif _VOVLAMTEN == "mid":      li = min(n_opt + n_var // 2 - 1, len(lam_fns) - 1)
        else:                          li = min(n_opt - 1, len(lam_fns) - 1)
        lam_x = torch.clamp(lam_fns[li](x), min=1e-6)
        wj = (W / W.sum())[:, None] * wq[None, :]
        lam_x = lam_x / (wj * lam_x).sum()                           # LEVEL is pinned by sig_ref;
        vx = lam_x * vix[Fi, Si][:, None]                            # only lambda's dispersion enters
        F = (wj * vx).sum(); Catm = (wj * torch.clamp(vx - F, min=0.0)).sum()
    iv = (2.0 / tau_opt ** 0.5) * _SQRT2 * torch.erfinv(torch.clamp(Catm / F, -0.999, 0.999))   # probit ATM inversion
    return F, iv


if __name__ == "__main__":
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
    sys.path.insert(0, POC); sys.path.insert(0, HERE)
    import discslv_slv
    from slv_fast import propagate_vec
    discslv_slv.propagate = propagate_vec
    from discslv_2f import TwoFactorSV
    from discslv_slv import Epi_V, nu_bar, raw_increment
    from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at

    DT = 1.0 / 52.0
    kw = dict(nu_f=0.208, nu_s=0.411, nu_l=1.070, lam_skew=-0.303, lam_f=0.633, lam_s=2.092, kap_f=0.937, kap_s=2.706)
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    chain = sanos_chain(OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"); sig = ref_vol(chain)
    gbar = solve_gbar(kw, sig, dt=DT)
    K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    lamv1 = leverage_at(chain, 4 * DT, EV, dt=DT)

    theta = torch.tensor([gbar, kw["nu_f"], kw["nu_s"], kw["nu_l"], kw["lam_skew"],
                          kw["lam_f"], kw["lam_s"], kw["kap_f"], kw["kap_s"]], requires_grad=True)
    ker = build_kernel(theta, DT)
    lamt = lev_torch(lamv1.coef, lamv1.zmax, lamv1.safety)

    print("=== kernel-array match (torch vs v1) ===")
    for nm, a, b in [("Vl", ker["Vl"], K.Vl), ("D", ker["D"], K.D), ("Tf", ker["Tf"], K.Tf), ("Ts", ker["Ts"], K.Ts)]:
        print(f"  {nm:>3}  max|diff| = {float((a.detach().numpy() - b).__abs__().max()):.2e}")

    st_np = (np.array([1.0]), np.array([0.0]), np.array([1e-4]), np.array([2], np.intp), np.array([1], np.intp))
    st_t = (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1e-4]), torch.tensor([2]), torch.tensor([1]))
    for step in range(4):
        st_np, _ = propagate_vec(K, st_np, (lambda mc: lamv1(mc) ** 2), EV, nub, Vlr, tiltr, 16)
        st_t = propagate(st_t, ker, lamt, 16)
    Wn, Mn, Sn, Fn, Sn2 = st_np
    Wt, Mt, St, Ft, St2 = (x.detach().numpy() for x in st_t)
    on = np.lexsort((Mn, Sn2, Fn)); ot = np.lexsort((Mt, St2, Ft))
    print(f"\n=== 4-step marginal match (sorted by regime,mean) ===  n_comp v1={len(Wn)} torch={len(Wt)}")
    print(f"  max|dW|={np.abs(Wn[on]-Wt[ot]).max():.2e}  max|dMU|={np.abs(Mn[on]-Mt[ot]).max():.2e}  "
          f"max|dSG|={np.abs(Sn[on]-St[ot]).max():.2e}  regimes match={np.array_equal(Fn[on],Ft[ot]) and np.array_equal(Sn2[on],St2[ot])}")

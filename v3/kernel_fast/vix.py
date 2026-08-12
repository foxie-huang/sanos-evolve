"""VIX ATM implied volatility. Port of `discslv_torch.kstep_law_n` / `vix_n` / `vix_ivol_n`.

BOTH paths. `lam_fns=None` is the unlevered readout (the module default in the reference);
`lam_fns` given is the LEVERAGED one, i.e. VOVLEV=1 -- instantaneous variance is lambda(x)^2 exp(g),
so the 30d variance-swap rate carries lambda too, and the unlevered form levers the SSR side but not
the vov side. Handoff 7 step 5 records one 2024 refit at VOVLEV=1 going cost 0.109 -> 0.035 and vov
RMS 24.1% -> 12.2%, so this is the readout that matters.

Same scalar discipline as the rest of the package: no Python number ever touches a tensor, because
under jacfwd on MPS that raises from either side.
"""
import os

import torch

from consts import scalar
from fkernel import stationary_corr, vfull
from propagate import step

# VIXFIX=0 restores the pre-2026-08-08 leveraged readout, which scaled vol-of-vol by ~rho.
_VIXFIX = os.environ.get("VIXFIX", "1") == "1"
# VOVLAMTEN selects WHICH lambda slice the leveraged VIX reads: expiry (default) | mid |
# fix30 | avg. See `_leveraged` -- "avg" averages lambda^2 over the VIX window.
_LAMTEN = os.environ.get("VOVLAMTEN", "expiry")


def kstep_law(ker, m, V, k):
    """k-step law of (u_f, u_s) from a Gaussian (m, V) today. Closed form -- this is what replaces
    the old `matrix_power(Pj, n_opt)` and its kron product.

        V_ff,k = kap_f^2k V_ff,0 + (1 - kap_f^2k)
        V_fs,k = (kap_f kap_s)^k V_fs,0 + s_f s_s rho_f rho_s (1 - (kap_f kap_s)^k)/(1 - kap_f kap_s)

    The z_l cross term is the shared shock accumulating over k steps; it does NOT factorise, the same
    point the reference makes about Pbar != kron of the branch-averaged marginals.
    """
    K = ker["K"]
    mf, ms = m
    Vff, Vss, Vfs = V
    kf, ks = ker["kap_f"], ker["kap_s"]
    af, as_ = kf.pow(k), ks.pow(k)
    kk = kf * ks
    cross = ker["s_f"] * ker["s_s"] * ker["rho_f"] * ker["rho_s"]
    geo = (K.one - kk.pow(k)) / torch.clamp(K.one - kk, min=1e-12)
    return ((af * mf, as_ * ms),
            (af.pow(2) * Vff + (K.one - af.pow(2)),
             as_.pow(2) * Vss + (K.one - as_.pow(2)),
             kk.pow(k) * Vfs + cross * geo))


def vix(ker, sig_ref, m, V, n_var):
    """30-day forward vol as a function of the factor law (m, V) today."""
    K = ker["K"]
    nu_f, nu_s = ker["nu_f"], ker["nu_s"]
    tot = None
    for k in range(1, n_var + 1):
        (mf, ms), (Vff, Vss, Vfs) = kstep_law(ker, m, V, k)
        mX = nu_f * mf + nu_s * ms
        vX = nu_f.pow(2) * Vff + nu_f * nu_s * Vfs * K.two + nu_s.pow(2) * Vss
        t = vfull(ker, mX, vX)
        tot = t if tot is None else tot + t
    c = stationary_corr(ker)
    vXs = nu_f.pow(2) + nu_f * nu_s * c * K.two + nu_s.pow(2)
    EV = vfull(ker, K.zero, vXs)                       # stationary normalisation, as the reference
    s2 = scalar(EV, sig_ref * sig_ref)
    return torch.sqrt(torch.clamp(s2 * (tot / scalar(EV, float(n_var))) / EV, min=1e-16))


def init_state(ker):
    """One component, factors on their stationary law. (B=1, N=1) for the batch-first propagator."""
    K = ker["K"]
    one = torch.ones(1, 1, dtype=K.dtype, device=K.device)
    z = torch.zeros(1, 1, dtype=K.dtype, device=K.device)
    return (one.clone(), z.clone(), torch.full((1, 1), 1e-6, dtype=K.dtype, device=K.device),
            z.clone(), z.clone(), one.clone(), one.clone(),
            stationary_corr(ker).reshape(1, 1))


def vix_ivol(ker, sig_ref, tau_opt, spot, tau_var=30.0 / 365.0, lam_fns=None,
             nk_vix=16, lamten=None, us0=None):
    """(forward, ATM implied vol) of the VIX law. `lam_fns` given => the LEVERAGED (VOVLEV=1) form.

    `us0` may be passed in: it is the solve of VIX(u_f=0, u_s) = spot and depends only on
    (ker, sig_ref, spot, n_var) -- NOT on tau_opt. Recomputing its 60-step bisection once per tenor
    is 12x redundant, which is invisible on CPU and dominates on MPS. Use `solve_us0` once and pass
    the result.
    """
    """(forward, ATM implied vol) of the atomic VIX law. Returns tensors."""
    K = ker["K"]
    dt = K.dt
    n_var = max(1, int(round(tau_var / dt)))
    n_opt = max(1, int(round(tau_opt / dt)))
    z = K.zero
    if us0 is None:
        us0 = solve_us0(ker, sig_ref, spot, n_var)

    if lam_fns is not None:
        return _leveraged(ker, sig_ref, tau_opt, us0, n_var, n_opt, lam_fns, nk_vix,
                          lamten or _LAMTEN)

    (mfT, msT), (Vff, Vss, Vfs) = kstep_law(ker, (z, us0), (z, z, z), n_opt)
    zq, wq = K.zq_vix, K.wq_vix        # hoisted into Consts: no numpy, no H2D copy per call
    sf = torch.sqrt(torch.clamp(Vff, min=1e-16))
    ss = torch.sqrt(torch.clamp(Vss, min=1e-16))
    rho = Vfs / torch.clamp(sf * ss, min=1e-16)
    uf = mfT + sf * zq[:, None]
    us = msT + ss * (rho * zq[:, None]
                     + torch.sqrt(torch.clamp(K.one - rho.pow(2), min=0.0)) * zq[None, :])
    w2 = wq[:, None] * wq[None, :]
    zz = torch.zeros_like(uf)
    vixT = vix(ker, sig_ref, (uf, us), (zz, zz, zz), n_var)
    F = (w2 * vixT).sum()
    Catm = (w2 * torch.clamp(vixT - F, min=0.0)).sum()
    sq2 = 2.0 ** 0.5
    iv = torch.erfinv(torch.clamp(Catm / F, -0.999, 0.999)) * scalar(F, 2.0 / tau_opt ** 0.5 * sq2)
    return F, iv


def solve_us0(ker, sig_ref, spot, n_var):
    """Today's slow level: VIX(u_f = 0, u_s) = spot on a CONTINUOUS u_s (the old code could only pick
    one of n_s = 3 abscissas). Bisection under no_grad -- a state choice, not a parameter, so jacfwd
    never traces these 60 iterations."""
    K = ker["K"]
    z = K.zero
    with torch.no_grad():
        lo = torch.full((), -6.0, dtype=K.dtype, device=K.device)
        hi = torch.full((), 6.0, dtype=K.dtype, device=K.device)
        for _ in range(60):
            mid = (lo + hi) * K.half
            gt = vix(ker, sig_ref, (z, mid), (z, z, z), n_var) > spot
            hi = torch.where(gt, mid, hi)
            lo = torch.where(gt, lo, mid)
        return (lo + hi) * K.half


def _leveraged(ker, sig_ref, tau_opt, us0, n_var, n_opt, lam_fns, nk_vix, lamten):
    """VOVLEV=1: propagate the JOINT (log-price, factor) law to the option expiry, then weight the
    per-component VIX by lambda evaluated across the spot distribution."""
    K = ker["K"]
    # `step` reads its component budget from Consts, so a differing nk_vix would be SILENTLY
    # ignored. Fail loudly instead of diverging from the reference without saying so.
    if nk_vix != K.nk:
        raise ValueError(f"nk_vix={nk_vix} != Consts.nk={K.nk}; rebuild Consts(nk={nk_vix})")
    st = init_state(ker)
    st = (st[0], st[1], st[2], st[3] * K.zero, st[3] * K.zero + us0,
          st[5] * K.zero, st[6] * K.zero, st[7] * K.zero)
    for k in range(n_opt):
        st = step(st, ker, lam_fns[min(k, len(lam_fns) - 1)])
    W, MU, SG, mf, ms, Vff, Vss, Vfs = (t[0] for t in st)          # single chain -> drop the B axis
    zq, wq = K.zq_lev, K.wq_lev        # Q_VIX = 5 here, NOT nq = 7 -- see Consts
    x = MU[:, None] + SG[:, None] * zq[None, :]                    # (N, Q) log-price at expiry
    if lamten == "avg":
        # AVERAGE lambda^2 OVER THE VIX'S OWN WINDOW instead of reading one slice.
        #
        # The VIX at expiry tau covers [tau, tau + 30d], i.e. propagation steps n_opt .. n_opt+n_var.
        # Reading a SINGLE slice at n_opt-1 is both less correct and badly behaved: the ladder is
        # fitted independently per week and jumps 11.9% whenever consecutive weeks cross a pillar
        # (2.8% when they do not), because `blend` is piecewise-linear in T so dC/dT is discontinuous
        # -- handoff 6e.1b. One slice per tenor puts that discontinuity straight into the model's vov
        # term structure, and no theta can cancel it: the vov-only floor at 2016 is 13.58% of which
        # 13.89pp is jaggedness, while the model's SMOOTH component already fits to 3.78%.
        #
        # Averaging is also the more correct object, not merely the smoother one. Instantaneous
        # variance is lambda(x)^2 exp(g) and VIX^2 is its average over the window, so what multiplies
        # is the RMS of lambda across those steps -- hence the mean of lambda^2, then a sqrt. It
        # smooths by construction: consecutive tenors share n_var-1 of their n_var slices.
        idx = [min(n_opt - 1 + m, len(lam_fns) - 1) for m in range(n_var)]
        l2 = None
        for i in idx:
            li2 = torch.clamp(lam_fns[i](x), min=1e-6).pow(2)
            l2 = li2 if l2 is None else l2 + li2
        lam_x = torch.sqrt(l2 / float(len(idx)))
    else:
        if lamten == "fix30":
            li = min(n_var, len(lam_fns) - 1)
        elif lamten == "mid":
            li = min(n_opt + n_var // 2 - 1, len(lam_fns) - 1)
        else:
            li = min(n_opt - 1, len(lam_fns) - 1)
        lam_x = torch.clamp(lam_fns[li](x), min=1e-6)
    wj = (W / W.sum())[:, None] * wq[None, :]
    lam_x = lam_x / (wj * lam_x).sum()                             # LEVEL pinned by sig_ref

    if not _VIXFIX:
        # THE OLD, BROKEN PATH -- kept only for A/B. See the module docstring.
        vixT = vix(ker, sig_ref, (mf, ms), (Vff, Vss, Vfs), n_var)
        vx = lam_x * vixT[:, None]
        wfull = wj
    else:
        # Expand each component's factor law N(m, V) onto Gauss--Hermite nodes and evaluate `vix`
        # with ZERO variance at each node -- exactly what the unlevered path does.
        #
        # WHY. `vix(m, V)` reads V as remaining uncertainty about the factor AT THE VALUATION DATE
        # and folds it into the variance-swap rate as convexity. At the option expiry the factor is
        # REALISED, so that uncertainty is zero; passing V there converts genuine dispersion of the
        # VIX into a convexity correction on its mean. Since the normalised kernel carries the fresh
        # innovation eps INSIDE V (only the return branch zeta_l moves m across components), the old
        # form kept just the rho-share of the vol-of-vol: measured ratio to the unlevered readout
        # 0.000 at rho=0, 0.275 at rho=0.239, 0.944 at rho=0.99. At rho=0 it reported EXACTLY ZERO
        # vol-of-vol for a model with unchanged nu_f, nu_s.
        zf, wf = K.zq_vix, K.wq_vix                                # nq = 7, the unlevered rule
        sf = torch.sqrt(torch.clamp(Vff, min=1e-16))
        ss = torch.sqrt(torch.clamp(Vss, min=1e-16))
        rho = Vfs / torch.clamp(sf * ss, min=1e-16)
        one = torch.sqrt(torch.clamp(K.one - rho.pow(2), min=0.0))
        uf = mf[:, None, None] + sf[:, None, None] * zf[None, :, None]          # (N, nv, 1)
        us = (ms[:, None, None]
              + ss[:, None, None] * (rho[:, None, None] * zf[None, :, None]
                                     + one[:, None, None] * zf[None, None, :]))  # (N, nv, nv)
        zz = torch.zeros_like(us)
        vixT = vix(ker, sig_ref, (uf, us), (zz, zz, zz), n_var)                  # (N, nv, nv)
        vx = lam_x[:, None, None, :] * vixT[..., None]                           # (N, nv, nv, Q)
        wfull = wj[:, None, None, :] * (wf[:, None] * wf[None, :])[None, :, :, None]

    F = (wfull * vx).sum()
    Catm = (wfull * torch.clamp(vx - F, min=0.0)).sum()
    iv = torch.erfinv(torch.clamp(Catm / F, -0.999, 0.999)) * scalar(F, 2.0 / tau_opt ** 0.5 * 2.0 ** 0.5)
    return F, iv

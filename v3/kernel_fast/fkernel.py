"""Kernel construction and the stationary bivariate quadrature. Port of `discslv_torch`'s
`build_kernel_n` / `stationary_n` / `_stat_nodes_n` / `_epi_v_n` / `_vfull_n` / `solve_gbar_torch_n`.

Every constant comes from `Consts`; no numpy, no scalar on the left of `-` or `/`.
"""
import torch


def build_kernel(theta9, K):
    """theta9 = [gbar, nu_f, nu_s, nu_l, lam_skew, rho_f, rho_s, kap_f, kap_s]."""
    gbar, nu_f, nu_s, nu_l, lam_skew, rho_f, rho_s, kap_f, kap_s = theta9
    s_f = torch.sqrt(torch.clamp(K.one - kap_f.pow(2), min=1e-12))
    s_s = torch.sqrt(torch.clamp(K.one - kap_s.pow(2), min=1e-12))
    return dict(gbar=gbar, nu_f=nu_f, nu_s=nu_s, nu_l=nu_l, lam_skew=lam_skew,
                rho_f=rho_f, rho_s=rho_s, kap_f=kap_f, kap_s=kap_s,
                s_f=s_f, s_s=s_s,
                q_f=s_f.pow(2) * (K.one - rho_f.pow(2)),
                q_s=s_s.pow(2) * (K.one - rho_s.pow(2)),
                K=K)


def stationary_corr(ker):
    """corr(u_f, u_s) of the stationary bivariate law. Both marginals are unit variance by
    construction, so the AR(1) cross-moment recursion closes on the correlation alone:
        c = kf ks c + s_f s_s rho_f rho_s   =>   c = s_f s_s rho_f rho_s / (1 - kf ks)."""
    num = ker["s_f"] * ker["s_s"] * ker["rho_f"] * ker["rho_s"]
    den = torch.clamp(ker["K"].one - ker["kap_f"] * ker["kap_s"], min=1e-12)
    return torch.clamp(num / den, min=-0.999, max=0.999)


def stat_nodes(ker):
    """Rectangular (u_f, u_s) grid with the correlation carried in the WEIGHTS, not by shearing.

    Shearing is WRONG here and was a real bug: u_s = c*za + sqrt(1-c^2)*zb makes zb depend on both
    factors, so the destination coordinates stop being separable while the SSR readout still uses
    them as independent tensor axes -- the two innovations get cross-paired. Symptom was beta
    converging to ~-1.04 against the ~-1.68 a flat SSR needs, and refining the grid not helping.
    """
    K = ker["K"]
    c = stationary_corr(ker)
    uf = K.za[:, None].expand(K.na, K.nb).contiguous()
    us = K.zb[None, :].expand(K.na, K.nb).contiguous()
    q = torch.clamp(K.one - c.pow(2), min=1e-12)
    # log Gaussian-copula density ratio; the marginal factors cancel against wa, wb.
    lr = torch.log(q) * K.neg_half - (c.pow(2) * (uf.pow(2) + us.pow(2))
                                     - uf * us * c * K.two) / (q * K.two)
    W = (K.wa[:, None] * K.wb[None, :]) * torch.exp(lr - lr.max())
    return uf, us, W / W.sum(), K.za, K.zb


def vfull(ker, mX, vX):
    """E[full one-step increment variance] over X ~ N(mX, vX), unlevered: E_l[V] + Var_l(mtil).

    Mirrors `_vfull_n` term for term. Note it uses the RAW `mtil`, not the martingale-locked drift
    `D = mtil - A`: A is constant across l, so Var_l is identical either way, and matching the
    reference's arithmetic exactly keeps the comparison bit-clean.
    """
    K = ker["K"]
    x = mX[..., None] + torch.sqrt(torch.clamp(vX, min=1e-16))[..., None] * K.zx
    gg = torch.clamp(ker["gbar"] + x[..., None] + ker["nu_l"] * K.zl, min=K.lo_g, max=K.hi_g)
    Vl = torch.exp(gg) * K.dt_t                                         # (..., nx, nl)
    mtil = Vl * K.neg_half + ker["lam_skew"] * K.zl * torch.sqrt(Vl)
    m1 = (K.wl * mtil).sum(-1)
    m2 = (K.wl * mtil.pow(2)).sum(-1)
    Vfull = (K.wl * Vl).sum(-1) + (m2 - m1.pow(2))                      # (..., nx)
    return (K.wx * Vfull).sum(-1)


def epi_v(ker):
    """E_pi[full one-step increment variance] under the stationary bivariate law."""
    c = stationary_corr(ker)
    K = ker["K"]
    vXs = ker["nu_f"].pow(2) + ker["nu_f"] * ker["nu_s"] * c * K.two + ker["nu_s"].pow(2)
    return vfull(ker, K.zero, vXs)


def solve_gbar(theta7or8, sig_ref, K, kap_s_fixed=None, iters=8):
    """Fixed-point for gbar so that E_pi[V] = sig_ref^2 dt. `sig_ref` is a Python float; it becomes
    a 0-d device tensor before any division -- a bare float on the LEFT of `/` raises under jacfwd
    on MPS."""
    tgt = torch.as_tensor(sig_ref * sig_ref * K.dt, dtype=theta7or8.dtype, device=theta7or8.device)
    gbar = torch.log(tgt)
    for _ in range(iters):
        ker = build_kernel(th9(theta7or8, gbar, K, kap_s_fixed), K)
        gbar = gbar + torch.log(tgt / epi_v(ker))
    return gbar


def th9(theta, gbar, K=None, kap_s_fixed=None):   # K unused; kept for call-site symmetry
    """[gbar] + theta, appending kap_s from the pin when theta is length 7."""
    if theta.shape[0] == 8:
        return torch.cat([gbar.reshape(1), theta])
    ks = torch.as_tensor(kap_s_fixed, dtype=theta.dtype, device=theta.device)
    return torch.cat([gbar.reshape(1), theta, ks.reshape(1)])

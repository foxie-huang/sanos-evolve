#!/usr/bin/env python3
"""THE GATE. Everything in kernel_fast must reproduce `discslv_torch`; this measures how well.

Run order matters: each stage isolates one thing, so a failure names its own cause.
  1  scalars   -- solve_gbar / stationary / stat_nodes vs the reference, float64
  2  grid      -- sigma_ATM grid vs the reference's (a,b,iz) triple loop
  3  SSR       -- the term structure, the only quantity with a decision threshold
  4  MPS       -- does it run at all, does jacfwd survive, how much does it move, how fast

    python3 verify.py [DATE]
"""
import os
import sys
import time

DATE = sys.argv[1] if len(sys.argv) > 1 else "2022-06-01"
sys.argv = [sys.argv[0], "cpu"]
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import _paths                                               # noqa: E402,F401
import consts                                               # noqa: E402
import fkernel as kernel                                    # noqa: E402
import readouts                                             # noqa: E402
import vix as VX                                            # noqa: E402
import discslv_torch as D                                   # noqa: E402
import calibrate_joint_torch as J                           # noqa: E402
import calibrate_slv_exact_ts as C                          # noqa: E402
import end_to_end as E                                      # noqa: E402
import discslv_torch_batched as TB                          # noqa: E402

# `readouts.atm_skew` is the MPS-clean port of TB.atm_skew_batch (which uses numpy float64 scalars
# MPS refuses). Stage 0 proves they agree, so the SIGMA GRID stays the only thing that differs
# between ref and new, and CPU and MPS legs run identical code.
ATM = readouts.atm_skew

ctx, _, _ = E.ctx_rebuilt(DATE, "SPX")
LAM, SIG = ctx["LT"][13], ctx["sig_ref"]
KS = C.KAP_S_FIXED
print(f"kernel_fast gate  |  SPX {DATE}\n{'=' * 74}")

torch.manual_seed(0)
_W = torch.rand(3, 240, dtype=torch.float64); _W = _W / _W.sum(1, keepdim=True)
_MU = torch.randn(3, 240, dtype=torch.float64) * 0.02
_SG = 0.02 + 0.05 * torch.rand(3, 240, dtype=torch.float64)
_a, _b = TB.atm_skew_batch(_W, _MU, _SG, 0.25), readouts.atm_skew(_W, _MU, _SG, 0.25)
print(f"\n0. atm_skew port vs TB.atm_skew_batch   vol |d| {float((_a[0]-_b[0]).abs().max()):.3e}"
      f"   skew |d| {float((_a[1]-_b[1]).abs().max()):.3e}")


def theta(dev, dt):
    return torch.tensor(list(C.X0_MAP_N["ts"]) + [KS], dtype=dt, device=dev)


# ---- 1. scalars, float64 ------------------------------------------------------------------------
th = theta("cpu", torch.float64)
K64 = consts.Consts("cpu", torch.float64)
gb_r = J.solve_gbar_torch_n(th, SIG, J.DT)
gb_n = kernel.solve_gbar(th, SIG, K64, KS)
kr = D.build_kernel_n(J._th9_n(th, gb_r), J.DT, n_x=J.N_X)
kn = kernel.build_kernel(kernel.th9(th, gb_n, K64, KS), K64)
a, b = D._stat_nodes_n(kr, K64.na, K64.nb), kernel.stat_nodes(kn)
print(f"\n1. SCALARS (float64)")
print(f"   solve_gbar   |d| = {abs(float(gb_r - gb_n)):.3e}")
print(f"   epi_v        |d| = {abs(float(J._epi_v_n(kr) - kernel.epi_v(kn))):.3e}")
print(f"   stationary   |d| = {abs(float(D.stationary_n(kr) - kernel.stationary_corr(kn))):.3e}")
for i, n in enumerate(["uf", "us", "PI"]):
    print(f"   stat_nodes {n:2s} |d| = {float((a[i] - b[i]).abs().max()):.3e}")

# ---- 2 & 3. grid and SSR, float64, against the reference's own helpers ---------------------------
t0 = time.time()
ssr_r, _, _ = D._ssr_core_n(J._th9_n(th, gb_r), LAM, J.NS, J.DT, nz=K64.nz, n_x=K64.n_x,
                            n_p=K64.n_p, na=K64.na, nb=K64.nb)
t_ref = time.time() - t0
ssr_r = np.array([float(x) for x in ssr_r])
t0 = time.time()
ssr_n = readouts.ssr_ts(kn, LAM, D._interp_lin, ATM)
t_new = time.time() - t0
ssr_n = ssr_n.detach().numpy()
d = np.abs((ssr_n - ssr_r) / ssr_r) * 100
print(f"\n2/3. SSR TERM STRUCTURE (float64 CPU)   ref {t_ref:6.2f}s   new {t_new:6.2f}s"
      f"   {t_ref / t_new:5.2f}x")
print("   tenor   " + "  ".join(f"{l:>9}" for l in J.LABELS))
print("   ref     " + "  ".join(f"{x:9.5f}" for x in ssr_r))
print("   new     " + "  ".join(f"{x:9.5f}" for x in ssr_n))
print("   d %     " + "  ".join(f"{x:9.4f}" for x in d))
print(f"   MAX |d SSR| = {d.max():.4f}%   (panel SSR RMS 0.9-4.9%)")

# ---- 4. MPS -------------------------------------------------------------------------------------
# ---- 3b. vix_ivol, both branches -----------------------------------------------------------------
print(f"\n3b. VIX ATM IV, all {len(ctx['vdtes'])} tenors (float64 CPU)")
_us0 = VX.solve_us0(kn, SIG, ctx["spot"], max(1, int(round((30.0 / 365.0) / K64.dt))))
for _tag, _lam in (("unlevered  (VOVLEV=0)", None), ("LEVERAGED  (VOVLEV=1)", LAM)):
    _mx = 0.0
    for _d in ctx["vdtes"]:
        _a = D.vix_ivol_n(kr, SIG, float(_d) / 365.0, ctx["spot"], lam_fns=_lam, n_p=K64.n_p)
        _b = VX.vix_ivol(kn, SIG, float(_d) / 365.0, ctx["spot"], lam_fns=_lam, us0=_us0)
        _mx = max(_mx, abs(float(_a[1] - _b[1])) / abs(float(_a[1])))
    print(f"    {_tag}   max rel |d iv| = {_mx:.3e}"
          + ("   (closed form: never touches the propagator)" if _lam is None
             else "   (via the propagator: the banding difference)"))

print(f"\n4. MPS  (available={torch.backends.mps.is_available()})")
if torch.backends.mps.is_available():
    res = {}
    for dev in ("cpu", "mps"):
        Kd = consts.Consts(dev, torch.float32)
        thd = theta(dev, torch.float32)

        def f(t, Kd=Kd):
            g = kernel.solve_gbar(t, SIG, Kd, KS)
            kk = kernel.build_kernel(kernel.th9(t, g, Kd, KS), Kd)
            ssr = readouts.ssr_ts(kk, LAM, D._interp_lin, ATM)
            # VOVLEV=1, and us0 solved ONCE: it does not depend on tau_opt, so the reference's
            # per-tenor bisection is 12x redundant -- free on CPU, dominant on MPS.
            u0 = VX.solve_us0(kk, SIG, ctx["spot"], 4)
            vv = torch.stack([VX.vix_ivol(kk, SIG, float(dd) / 365.0, ctx["spot"],
                                          lam_fns=LAM, us0=u0)[1] for dd in ctx["vdtes"]])
            return torch.cat([ssr, vv])          # the FULL readout, as model_torch_n returns
        sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
        try:
            f(thd); sync()
            t0 = time.time(); s = f(thd); sync(); tf = time.time() - t0
            t0 = time.time(); jc = torch.func.jacfwd(f)(thd); sync(); tj = time.time() - t0
            sn, jn = s.detach().cpu().numpy(), jc.detach().cpu().numpy()
            res[dev] = (tf, tj, sn, jn)
            print(f"   {dev.upper():4s} fwd {tf:7.3f}s  jacfwd {tj:8.3f}s ({tj / tf:4.1f}x)"
                  f"  finite={bool(np.isfinite(jn).all())}")
        except Exception as ex:
            print(f"   {dev.upper():4s} FAILED  {type(ex).__name__}: {str(ex)[:100]}")
    if len(res) == 2:
        (cf, cj, cs, _), (mf, mj, ms, _) = res["cpu"], res["mps"]
        print(f"\n   MPS speedup  fwd {cf / mf:5.2f}x   jacfwd {cj / mj:5.2f}x")
        print(f"   max rel |CPU-MPS| on SSR: {np.abs((ms - cs) / cs).max():.2e}")
        print(f"   GN iteration  CPU {cf + cj:6.2f}s -> MPS {mf + mj:5.2f}s")
        print(f"   fit 22nfev+9njev  CPU {22 * cf + 9 * cj:5.0f}s "
              f"({(22 * cf + 9 * cj) / 60:.1f} min) -> MPS {22 * mf + 9 * mj:4.0f}s "
              f"({(22 * mf + 9 * mj) / 60:.2f} min)")
        print(f"   vs the reference loop fit (957 s = 16 min): "
              f"{957 / (22 * mf + 9 * mj):.0f}x")

#!/usr/bin/env python3
"""
The PRINCIPLED + ACCURATE term-structure leverage fix, measured with the EXACT beta.

Earlier the term-structure hypothesis (each propagation step gets its maturity-appropriate leverage,
steep short / flat long, instead of ONE frozen T-leverage at every step) was tested only with the
leading-order READOUT (fused_ssr_readout_ts), which we since found is off 7-36%. This re-measures the
belly recovery with fused_ssr_exact_ts -- the de-sampled-MC exact beta -- so the number is real, not
an artifact of the discredited approximation.

At a FIXED theta (isolating the leverage-application effect), compare:
    single = fused_ssr_exact     with one frozen leverage_at(T)          (the current model)
    ts     = fused_ssr_exact_ts  with lam_fns[k] = leverage_at((k+1)*dt) (per-step term structure)
against the empirical SPX SSR. The single-vs-ts gap at 2m/3m = how much of the belly the principled
term-structure leverage genuinely claws back, and the residual to emp = the true capacity wall.
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                          # noqa: E402
import discslv_slv                                                         # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                       # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact, fused_ssr_exact_ts    # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at         # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]
EMP = [2.027, 1.658, 1.595, 1.538, 1.457]                                  # exact-beta empirical target (252 2015)
date = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.872, nu_s=0.532, nu_l=0.831, lam_skew=-0.652, lam_f=0.489, lam_s=2.088, kap_f=0.769, kap_s=2.991)  # DENSE-chain re-fit theta
K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)

t0 = time.time()
nmax = max(NS)
print(f"building leverage term structure: leverage_at(k*dt) for k=1..{nmax} ...", flush=True)
lam_cache = {k: leverage_at(chain, k * DT, EV, dt=DT) for k in range(1, nmax + 1)}
print(f"  dlam/dz by step maturity: " +
      "  ".join(f"{k}={float((lam_cache[k](0.02)-lam_cache[k](-0.02))/0.04):.1f}" for k in range(1, nmax + 1)),
      flush=True)

print(f"\n{'mat':>4}{'single':>9}{'ts':>9}{'emp':>7}{'s.err':>8}{'ts.err':>8}{'recov':>8}")
for n, lab, e in zip(NS, LABELS, EMP):
    single = leverage_at(chain, n * DT, EV, dt=DT)                         # one frozen T-leverage (current)
    lam_fns = [lam_cache[k + 1] for k in range(n)]                         # step k -> maturity (k+1)*dt
    ss = fused_ssr_exact(K, single, n, EV, nub, Vlr, tiltr, 16, DT, nz=13)[0]
    st = fused_ssr_exact_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, 16, DT, nz=13)[0]
    se, te = (ss - e) / e * 100, (st - e) / e * 100
    recov = (st - ss) / (e - ss) * 100 if abs(e - ss) > 1e-9 else float("nan")   # % of gap closed
    print(f"{lab:>4}{ss:>9.3f}{st:>9.3f}{e:>7.3f}{se:>7.0f}%{te:>7.0f}%{recov:>7.0f}%", flush=True)
print(f"\n({time.time()-t0:.0f}s)  single=frozen leverage_at(T), ts=per-step leverage_at((k+1)*dt), both EXACT beta")

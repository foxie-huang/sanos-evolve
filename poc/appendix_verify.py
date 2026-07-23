"""
appendix_verify.py -- numerical verification of every checkable appendix formula (2026-07-06).
  [C1] lognormal facts used in App C's mollify step: Var(L)=s^2(e^{s2}-1), E|L-s| <= s*sqrt(e^{s2}-1)
  [C2] Lemma 2 end-to-end: barycentric quantise + mean-matched mollify on a bimodal test law;
       forward exact to machine, W1(eta,rho) <= claimed bound, W1 -> 0 with refinement
  [C3] Prop 3 per-fibre bound on a martingale fibre
  [E1] App E level: u0 = 2*Phi^{-1}((1+C_mix(0))/2) vs bisection-inverted BS IV
  [E2] App E mixture sums eq:glide-mix vs finite differences of the closed-form mixture price
  [E3] App E Black reference eq:glide-bs vs finite differences of C_BS
  [E4] App E u', u'' (eq:glide-uu) vs finite differences of the bisection IV
  [A1] App A pricing-map entry and CDF formula vs numerical integration
Persists this printout to appendix_verify_results.txt. Wall time printed at the end.
"""
import time, io, sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

T0 = time.time()
rng = np.random.default_rng(7)
buf = io.StringIO()
def P(*a):
    print(*a); print(*a, file=buf)

def bs_call(F, K, u):          # undiscounted Black, total vol u
    if u <= 0: return max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * u * u) / u
    return F * norm.cdf(d1) - K * norm.cdf(d1 - u)

# ---------- [C1] lognormal facts ----------
s, s2 = 1.37, 0.41 ** 2
Z = rng.standard_normal(4_000_000)
L = s * np.exp(np.sqrt(s2) * Z - s2 / 2)          # mean-matched lognormal
var_th = s * s * (np.exp(s2) - 1)
mad_bound = s * np.sqrt(np.exp(s2) - 1)
P(f"[C1] E[L]={L.mean():.6f} (target {s})  Var={L.var():.6f} vs s^2(e^s2-1)={var_th:.6f}  "
  f"E|L-s|={np.abs(L - s).mean():.6f} <= bound {mad_bound:.6f}  "
  f"{'OK' if np.abs(L-s).mean() <= mad_bound and abs(L.var()-var_th) < 3e-3 else 'FAIL'}")

# ---------- [C2] Lemma 2 end-to-end ----------
# test law rho: bimodal lognormal mixture (weights .6/.4), forward F
w_r = np.array([0.6, 0.4]); m_r = np.array([-0.25, 0.30]); v_r = np.array([0.18**2, 0.30**2])
F_r = float(np.sum(w_r * np.exp(m_r + v_r / 2)))
grid = np.linspace(1e-6, 12.0, 600_001)          # spot grid for CDFs / W1
def cdf_rho(x):  # lognormal mixture CDF in s
    x = np.maximum(x, 1e-300)
    return sum(w * norm.cdf((np.log(x) - m) / np.sqrt(v)) for w, m, v in zip(w_r, m_r, v_r))
def w1(G1, G2): return np.trapezoid(np.abs(G1 - G2), grid)

def lemma2(N, M, sig):
    edges = np.linspace(0.0, M, N + 1)
    # closed-form cell masses and partial first moments of the lognormal mixture:
    #   mass(a,b) = sum w [Phi(l(b)) - Phi(l(a))],  l(x) = (ln x - m)/sqrt(v)
    #   int_a^b s drho = sum w e^{m+v/2} [Phi(l(b)-sqrt(v)) - Phi(l(a)-sqrt(v))]
    def mass_mom(a, b):
        a = max(a, 1e-300)
        ms = mo = 0.0
        for w, m, v in zip(w_r, m_r, v_r):
            la = (np.log(a) - m) / np.sqrt(v); lb = (np.log(b) - m) / np.sqrt(v)
            ms += w * (norm.cdf(lb) - norm.cdf(la))
            mo += w * np.exp(m + v / 2) * (norm.cdf(lb - np.sqrt(v)) - norm.cdf(la - np.sqrt(v)))
        return ms, mo
    wj, sj = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ms, mo = mass_mom(a, b)
        if ms > 1e-14: wj.append(ms); sj.append(mo / ms)
    ms_t, mo_t = mass_mom(M, 1e6)                              # tail cell (M, inf)
    if ms_t > 1e-14: wj.append(ms_t); sj.append(mo_t / ms_t)   # true tail barycentre
    wj = np.array(wj); sj = np.array(sj)
    fwd_atomic = float(np.sum(wj * sj))
    # mollify: mean-matched lognormals, log-var sig^2
    Geta = sum(w * norm.cdf((np.log(np.maximum(grid, 1e-300) / s_) + sig**2 / 2) / sig)
               for w, s_ in zip(wj, sj))
    fwd_eta = float(np.sum(wj * sj))                                  # mean-matched => same forward
    W = w1(Geta, cdf_rho(grid))
    _, mo_tail = mass_mom(M, 1e6)
    bound = (edges[1] - edges[0]) + 2 * mo_tail + F_r * np.sqrt(np.exp(sig**2) - 1)
    return fwd_eta, W, bound

for (N, M, sig) in [(20, 6.0, 0.20), (60, 8.0, 0.08), (200, 10.0, 0.03)]:
    fwd, W, bound = lemma2(N, M, sig)
    P(f"[C2] N={N:3d} M={M:4.1f} sig={sig:.2f}:  forward err={abs(fwd-F_r):.2e}  "
      f"W1(eta,rho)={W:.5f} <= bound {bound:.5f}  {'OK' if W <= bound + 1e-6 else 'FAIL'}")

# ---------- [C3] Prop 3 per-fibre bound ----------
sf = 1.9; vf = 0.35 ** 2                                   # martingale fibre: lognormal mean sf
w_r, m_r, v_r = np.array([1.0]), np.array([np.log(sf) - vf / 2]), np.array([vf])
F_r = sf
fwd, W, bound = lemma2(80, 12.0, 0.05)
P(f"[C3] fibre mean {sf}: forward err={abs(fwd-F_r):.2e}  W1={W:.5f} <= bound {bound:.5f}  "
  f"{'OK' if W <= bound + 1e-6 and abs(fwd - F_r) < 1e-8 else 'FAIL'}")

# ---------- App E: random martingale GMs ----------
def glide(omega, mm, vv):
    Fk = np.exp(mm + vv / 2); dlt = mm / np.sqrt(vv)
    C0 = float(np.sum(omega * (Fk * norm.cdf(dlt + np.sqrt(vv)) - norm.cdf(dlt))))
    u0 = 2 * norm.ppf(0.5 + 0.5 * C0)
    dC = float(np.sum(-omega * norm.cdf(dlt)))                       # dC_mix/dx at 0
    d2C = float(np.sum(omega * (norm.pdf(dlt) / np.sqrt(vv) - norm.cdf(dlt))))
    dB = -norm.cdf(-u0 / 2)                                          # dC_BS/dx at (0,u0)
    d2B = -norm.cdf(-u0 / 2) + norm.pdf(u0 / 2) / u0
    V = norm.pdf(u0 / 2); Vx = 0.5 * norm.pdf(u0 / 2); Vu = -u0 / 4 * norm.pdf(u0 / 2)
    u1 = (dC - dB) / V
    u2 = (d2C - d2B - 2 * Vx * u1 - Vu * u1 ** 2) / V
    return u0, u1, u2

def mixprice(omega, mm, vv, x):
    Fk = np.exp(mm + vv / 2)
    return float(np.sum([w * bs_call(F, np.exp(x), np.sqrt(v)) for w, F, v in zip(omega, Fk, vv)]))

def iv(omega, mm, vv, x):
    c = mixprice(omega, mm, vv, x)
    return brentq(lambda u: bs_call(1.0, np.exp(x), u) - c, 1e-6, 4.0, xtol=1e-14)

errs = np.zeros((30, 5))
for t in range(30):
    n = rng.integers(2, 6)
    omega = rng.dirichlet(np.ones(n)); vv = rng.uniform(0.02, 0.25, n) ** 2
    mm = rng.normal(0, 0.15, n) - vv / 2
    mm -= np.log(np.sum(omega * np.exp(mm + vv / 2)))                # martingale-normalise
    u0, u1, u2 = glide(omega, mm, vv)
    h = 1e-4
    ivm, iv0, ivp = iv(omega, mm, vv, -h), iv(omega, mm, vv, 0.0), iv(omega, mm, vv, h)
    errs[t, 0] = abs(u0 - iv0)
    errs[t, 1] = abs(u1 - (ivp - ivm) / (2 * h))
    errs[t, 2] = abs(u2 - (ivp - 2 * iv0 + ivm) / h ** 2)
    dCn = (mixprice(omega, mm, vv, h) - mixprice(omega, mm, vv, -h)) / (2 * h)
    d2Cn = (mixprice(omega, mm, vv, h) - 2 * mixprice(omega, mm, vv, 0) + mixprice(omega, mm, vv, -h)) / h ** 2
    Fk = np.exp(mm + vv / 2); dlt = mm / np.sqrt(vv)
    errs[t, 3] = abs(float(np.sum(-omega * norm.cdf(dlt))) - dCn)
    errs[t, 4] = abs(float(np.sum(omega * (norm.pdf(dlt) / np.sqrt(vv) - norm.cdf(dlt)))) - d2Cn)
P(f"[E1] level  |u0 - bisection IV|      max {errs[:,0].max():.2e}   {'OK' if errs[:,0].max() < 1e-10 else 'FAIL'}")
P(f"[E4] skew   |u1 - FD(IV)|            max {errs[:,1].max():.2e}   {'OK' if errs[:,1].max() < 1e-5 else 'FAIL'}")
P(f"[E4] curv   |u2 - FD(IV)|            max {errs[:,2].max():.2e}   {'OK' if errs[:,2].max() < 1e-3 else 'FAIL'}")
P(f"[E2] dC_mix sum vs FD                max {errs[:,3].max():.2e}   {'OK' if errs[:,3].max() < 1e-6 else 'FAIL'}")
P(f"[E2] d2C_mix sum vs FD               max {errs[:,4].max():.2e}   {'OK' if errs[:,4].max() < 1e-4 else 'FAIL'}")
# [E3] Black reference vs FD of C_BS in x at fixed u
u0t = 0.23; h = 1e-5
dB_num = (bs_call(1, np.exp(h), u0t) - bs_call(1, np.exp(-h), u0t)) / (2 * h)
d2B_num = (bs_call(1, np.exp(h), u0t) - 2 * bs_call(1, 1, u0t) + bs_call(1, np.exp(-h), u0t)) / h ** 2
V_num = (bs_call(1, 1, u0t + h) - bs_call(1, 1, u0t - h)) / (2 * h)
Vx_num = ((bs_call(1, np.exp(h), u0t + h) - bs_call(1, np.exp(-h), u0t + h)) -
          (bs_call(1, np.exp(h), u0t - h) - bs_call(1, np.exp(-h), u0t - h))) / (4 * h * h)
Vu_num = (bs_call(1, 1, u0t + h) - 2 * bs_call(1, 1, u0t) + bs_call(1, 1, u0t - h)) / h ** 2
e3 = [abs(-norm.cdf(-u0t / 2) - dB_num),
      abs((-norm.cdf(-u0t / 2) + norm.pdf(u0t / 2) / u0t) - d2B_num),
      abs(norm.pdf(u0t / 2) - V_num),
      abs(0.5 * norm.pdf(u0t / 2) - Vx_num),
      abs(-u0t / 4 * norm.pdf(u0t / 2) - Vu_num)]
P(f"[E3] BS refs (dC,d2C,V,Vx,Vu) vs FD  max {max(e3):.2e}   {'OK' if max(e3) < 1e-4 else 'FAIL'}")

# ---------- [A1] App A pricing map + CDF ----------
a, hbw, Fj, K = -0.12, 0.21, 100.0, 96.0
zg = np.linspace(a - 10 * hbw, a + 10 * hbw, 400_001)
pay = np.maximum(Fj * np.exp(zg) - K, 0.0) * norm.pdf((zg - a) / hbw) / hbw
num_price = np.trapezoid(pay, zg)
map_price = bs_call(Fj * np.exp(a + hbw ** 2 / 2), K, hbw)
zK = np.log(K / Fj)                                 # integrate the density up to the kink exactly
zg2 = np.linspace(a - 10 * hbw, zK, 400_001)
num_cdf = np.trapezoid(norm.pdf((zg2 - a) / hbw) / hbw, zg2)
map_cdf = norm.cdf((zK - a) / hbw)
P(f"[A1] pricing-map entry vs integral   |diff|={abs(num_price-map_price):.2e}   "
  f"{'OK' if abs(num_price-map_price) < 1e-6 else 'FAIL'}")
P(f"[A1] CDF formula vs integral         |diff|={abs(num_cdf-map_cdf):.2e}   "
  f"{'OK' if abs(num_cdf-map_cdf) < 1e-8 else 'FAIL'}")

wall = time.time() - T0
P(f"wall {wall:.1f}s")
open("appendix_verify_results.txt", "w").write(buf.getvalue())

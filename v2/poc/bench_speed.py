"""Wall-clock benchmark of the SANOS-Evolve discrete-LSV pipeline (for the speed section)."""
import sys, time, subprocess
POC = "/Users/foxie/Documents/Research/2026/SANOS_Evolve/disc_SLV/poc"
sys.path.insert(0, POC)
import numpy as np

def cpu():
    try:
        return subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        return "unknown CPU"
print("CPU:", cpu(), "| numpy", np.__version__, "\n")

def bench(fn, n, name, warm=2):
    for _ in range(warm): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    ts=sorted(ts); med=ts[len(ts)//2]
    unit = ("us", 1e6) if med<1e-3 else (("ms",1e3) if med<1.0 else ("s",1.0))
    print(f"  {name:48s} {med*unit[1]:9.3f} {unit[0]:3s}  (median/{n})")
    return med

from discslv_2f import TwoFactorSV, ssr_2f, smile_2f, recompress_2f
DT=1/52.0
X0=[np.log(0.04),0.55,0.55,0.5,-0.8,3.0,3.0,0.2,2.0]
def kernel(x):
    return TwoFactorSV(gbar=x[0],nu_f=x[1],nu_s=x[2],lam_skew=x[4],lam_f=x[5],lam_s=x[6],
                       kap_f=x[7],kap_s=x[8],dt=DT,nu_l=x[3],n_f=5,n_s=3,n_l=5)

print("=== KERNEL build + propagation (closed-form GM, no MC/PDE) ===")
bench(lambda: kernel(X0), 50, "kernel construction (precompute V,d,P)")
K=kernel(X0)
# steady-state one step: warm to ~16 comps/regime then time propagate+recompress
W=np.array([1.0]);MU=np.array([0.0]);SG=np.array([1e-4]);F=np.array([0],np.intp);S=np.array([0],np.intp)
for _ in range(6):
    W,MU,SG,F,S=K.propagate(W,MU,SG,F,S); W,MU,SG,F,S=recompress_2f(W,MU,SG,F,S,16,K.n_f,K.n_s)
ncomp=len(W); exp=ncomp*K.n_l*K.n_f*K.n_s
def one_step():
    w,mu,sg,f,s=K.propagate(W,MU,SG,F,S); recompress_2f(w,mu,sg,f,s,16,K.n_f,K.n_s)
bench(one_step, 200, f"one step: propagate {ncomp}->{exp} + recompress->{ncomp}")
bench(lambda: smile_2f(K,52,0,0,16), 30, "full 1y density (52 steps, nk=16, 1 regime)")

print("\n=== READOUT (closed-form: pricing, ATM vol/skew, SSR -- no re-sim) ===")
bench(lambda: ssr_2f(K,52,16), 10, "SSR + ATM vol + skew @1y (all 15 regimes)")
from calibrate_2f import observables, residuals, X0 as CX0, LO, HI
bench(lambda: observables(X0), 10, "observables @4 maturities (=1 calibration eval)")

print("\n=== DYNAMICS CALIBRATION (9 knobs -> SSR/vol/skew term structure) ===")
from scipy.optimize import least_squares
t=time.perf_counter()
res=least_squares(residuals, CX0, bounds=(LO,HI), diff_step=3e-2, max_nfev=160, xtol=1e-8, ftol=1e-8)
dtc=time.perf_counter()-t
print(f"  full 9-knob fit: {res.nfev} evals, {dtc:.2f} s, final cost {res.cost:.1f}")

print("\n=== STATICS: SANOS convex LP on the real SPX chain ===")
try:
    import pandas as pd
    from sanos_lp import prep_expiry, sanos_fit, DATA
    df=pd.read_csv(DATA)
    es=[]
    for _,g in df.groupby("dte"):
        e=prep_expiry(g)
        if len(e["kappa"])>=5: es.append(e)
    mid=es[len(es)//2]
    bench(lambda: sanos_fit(mid), 20, f"1 expiry LP solve (cvxpy/CLARABEL, N={len(mid['kappa'])})")
    t=time.perf_counter()
    for e in es: sanos_fit(e)
    print(f"  full chain: {len(es)} expiries in {(time.perf_counter()-t)*1e3:.1f} ms"
          f"  ({(time.perf_counter()-t)*1e3/len(es):.1f} ms/expiry)")
    print("  (note: cvxpy canonicalisation dominates; a raw LP solver would be faster)")
except Exception as ex:
    print("  statics LP skipped:", repr(ex))
print("\nDONE.")

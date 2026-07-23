"""
sanos_validate.py -- validate the SANOS LP on CLEAN synthetic data.

The real-chain fit is blocked by data quality (its IV column disagrees with its bid/ask by 90-324 bp).
Here we generate an exactly arbitrage-free chain from a KNOWN marginal (a multi-component GM on the
unit forward, so F=1 and the smile/IV are exact), put a TIGHT +/-10bp IV band on it, and run the SANOS
recipe (eta*V_j components, anchors with strike conditioning, vega-weighted L1, simplex + martingale).
If it lands within the band, the LP is faithful and the real-data failure is confirmed as data quality.

Run from poc/ :  python3 sanos_validate.py
"""
import numpy as np
import cvxpy as cp
import warnings
from discslv import bs_call, bs_implied_vol, synthetic_marginals

warnings.filterwarnings("ignore")
ETA = 0.25          # SANOS smoothness
HALFSPR = 0.0010    # +/-10 bp IV half-spread (tight, SPX-like)


def main():
    _, mu1, _, T = synthetic_marginals()                 # true marginal = a 9-component GM, maturity T
    kq = np.linspace(0.78, 1.30, 53)                     # quote moneyness (F = 1)
    c_true = np.asarray(mu1.call(kq))
    iv_true = mu1.implied_vol(kq, T)
    ok = np.isfinite(iv_true) & (iv_true > 0.02) & (c_true > 1e-7)
    kq, c_true, iv_true = kq[ok], c_true[ok], iv_true[ok]
    atm_iv = float(iv_true[np.argmin(np.abs(kq - 1.0))])
    V = atm_iv ** 2 * T                                  # ATM total variance

    c_bid = np.asarray(bs_call(1.0, kq, np.maximum(iv_true - HALFSPR, 1e-3) * np.sqrt(T)))
    c_ask = np.asarray(bs_call(1.0, kq, (iv_true + HALFSPR) * np.sqrt(T)))

    ka = np.linspace(0.62, 1.55, 60)                     # anchors: wider grid (strike conditioning)
    s = np.sqrt(ETA * V)
    M = np.asarray(bs_call(ka[:, None], kq[None, :], s))  # M[i,l] = Call(anchor_i, quote_l, sqrt(eta V))
    w = iv_true * np.sqrt(T)
    vega = np.maximum(np.asarray(bs_call(1.0, kq, w + 1e-4) - bs_call(1.0, kq, w)) / 1e-4, 1e-7)

    q = cp.Variable(len(ka), nonneg=True)
    obj = cp.sum(cp.multiply(vega, cp.abs(M.T @ q - c_true)))     # vega-weighted L1 to mid (SANOS)
    cp.Problem(cp.Minimize(obj), [cp.sum(q) == 1, q @ ka == 1]).solve(solver=cp.CLARABEL)

    cfit = np.maximum(M.T @ q.value, 1e-12)
    ivfit = np.array([bs_implied_vol(cfit[l], 1.0, kq[l], T) for l in range(len(kq))])
    d = (ivfit - iv_true) * 1e4
    core = np.abs(np.log(kq)) < 0.2
    inband = (cfit >= c_bid - 1e-12) & (cfit <= c_ask + 1e-12)

    print(f"clean synthetic chain: true = {mu1.n}-comp GM, T={T}, ATM iv={atm_iv:.3f}")
    print(f"  quotes={len(kq)}  anchors={len(ka)}  eta={ETA}  band=+/-{HALFSPR*1e4:.0f}bp IV")
    print(f"  core ivRMSE = {np.sqrt(np.mean(d[core]**2)):.2f} bp")
    print(f"  full ivRMSE = {np.sqrt(np.mean(d**2)):.2f} bp   max = {np.max(np.abs(d)):.2f} bp")
    print(f"  within +/-{HALFSPR*1e4:.0f}bp band: {100*np.mean(inband):.0f}%   (q>=0 sum={q.value.sum():.3f})")


if __name__ == "__main__":
    main()

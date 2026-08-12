"""Device-resident constants. Built ONCE per (device, dtype, config); never inside a step.

Two reasons this exists rather than calling `hermegauss` where it is needed, as the reference does:
numpy in the hot path is a host->device copy per call on MPS and an unconditional graph break under
Dynamo. Everything here is a tensor on the target device before any step runs.

It also holds the small scalar constants as 0-d tensors. See README: a bare Python scalar on the
LEFT of `-` or `/` with a dual tensor raises TypeError under jacfwd on MPS.
"""
import torch
from numpy.polynomial.hermite_e import hermegauss


def _gh(n, dtype, device, normalise=True):
    z, w = hermegauss(n)
    zt = torch.as_tensor(z, dtype=dtype, device=device)
    wt = torch.as_tensor(w, dtype=dtype, device=device)
    return zt, (wt / wt.sum() if normalise else wt)


class Consts:
    """Quadrature nodes/weights and scalars for one (device, dtype, resolution) combination."""

    __slots__ = ("device", "dtype", "n_l", "n_x", "n_p", "na", "nb", "ne", "Q", "nk",
                 "nb_f", "nb_s", "nz", "zmax", "dt", "NS", "nmax", "snap",
                 "zl", "wl", "zx", "wx", "zp", "wp", "za", "wa", "zb", "wb",
                 "ze", "we", "zq", "wq", "zq_vix", "wq_vix", "nq", "zq_lev", "wq_lev", "q_vix",
                 "zg", "iz0",
                 "half", "one", "two", "zero", "neg_half", "dt_t", "eps_v", "eps_w", "lo_g", "hi_g")

    def __init__(self, device="cpu", dtype=torch.float32, *, n_l=5, n_x=3, n_p=5, na=5, nb=5,
                 ne=3, Q=5, nq=7, q_vix=5, nk=16, nb_f=5, nb_s=3, nz=9, zmax=0.12, dt=1.0 / 52.0,
                 NS=(1, 2, 4, 8, 13)):
        dev = torch.device(device)
        self.device, self.dtype = dev, dtype
        self.n_l, self.n_x, self.n_p = n_l, n_x, n_p
        self.na, self.nb, self.ne, self.Q, self.nq = na, nb, ne, Q, nq
        self.q_vix = q_vix
        self.nk, self.nb_f, self.nb_s = nk, nb_f, nb_s
        self.nz, self.zmax, self.dt = nz, zmax, dt
        self.NS = tuple(NS); self.nmax = max(NS)
        self.snap = {v: i for i, v in enumerate(self.NS)}

        # ALL weights normalised to sum 1 -- `build_kernel_n` stores `wl/wl.sum()` AND
        # `wx/wx.sum()`. (He weights sum to sqrt(2*pi), so leaving wx raw silently scales every
        # variance integral by 2.5066; caught by a 2.8e-4 gbar mismatch against the reference.)
        self.zl, self.wl = _gh(n_l, dtype, dev)
        self.zx, self.wx = _gh(n_x, dtype, dev)
        self.zp, self.wp = _gh(n_p, dtype, dev)
        self.za, self.wa = _gh(na, dtype, dev)
        self.zb, self.wb = _gh(nb, dtype, dev)
        self.ze, self.we = _gh(ne, dtype, dev)
        self.zq, self.wq = _gh(Q, dtype, dev)
        # The reference uses DIFFERENT node counts in the two VIX branches and it is easy to miss:
        #   unlevered  -> hermegauss(nq),    nq = 7      (discslv_torch.py:813)
        #   LEVERAGED  -> hermegauss(Q_VIX), Q_VIX = 5   (discslv_torch.py:833)
        # Using nq for both silently over-resolves the leveraged spot integral.
        self.zq_vix, self.wq_vix = _gh(nq, dtype, dev)       # unlevered terminal-law quadrature
        self.zq_lev, self.wq_lev = _gh(q_vix, dtype, dev)    # LEVERAGED spot-integral quadrature
        self.zg = torch.linspace(-zmax, zmax, nz, dtype=dtype, device=dev)
        self.iz0 = nz // 2

        c = lambda v: torch.as_tensor(v, dtype=dtype, device=dev)     # noqa: E731
        self.zero, self.one, self.two, self.half = c(0.0), c(1.0), c(2.0), c(0.5)
        self.neg_half, self.dt_t = c(-0.5), c(dt)
        self.eps_v, self.eps_w = c(1e-12), c(1e-15)
        self.lo_g, self.hi_g = -40.0, 12.0                            # clamp bounds: plain floats OK

    def like(self, device=None, dtype=None):
        """A twin on another device/dtype, same resolution."""
        return Consts(device or self.device, dtype or self.dtype,
                      n_l=self.n_l, n_x=self.n_x, n_p=self.n_p, na=self.na, nb=self.nb,
                      ne=self.ne, Q=self.Q, nq=self.nq, q_vix=self.q_vix, nk=self.nk, nb_f=self.nb_f, nb_s=self.nb_s,
                      nz=self.nz, zmax=self.zmax, dt=self.dt, NS=self.NS)

    @property
    def nc(self):
        return self.nb_f * self.nb_s * self.nk

    def __repr__(self):
        return (f"Consts({self.device}, {self.dtype}, na={self.na} nb={self.nb} nz={self.nz} "
                f"n_x={self.n_x} n_p={self.n_p} nc={self.nc})")


_SCALARS = {}


def scalar(t, v):
    """A 0-d tensor of value `v` matching t's dtype/device, cached.

    Needed because under `jacfwd` ON MPS a Python float may not touch a dual tensor from EITHER
    side: `2.0 * x`, `x * 2.0` and `x.add(1.0)` all raise TypeError. (On CPU they are fine, which is
    why this only shows up when the device changes.) Values are unchanged, so CPU results stay
    bit-identical.
    """
    key = (t.dtype, t.device, float(v))
    r = _SCALARS.get(key)
    if r is None:
        r = _SCALARS[key] = torch.as_tensor(float(v), dtype=t.dtype, device=t.device)
    return r

"""B1: FORWARD BYTES before / after the fix.  SHA-256 of R and T (and the
orders) for every fixture, NumPy eager, JAX eager and JAX under jax.jit;
``SG_TAG=pre`` on the base tree, ``SG_TAG=post`` on the fixed one, compared
by ``b1_compare.py``.

    python b1_fwd_bytes.py
"""
import hashlib

from _h import dump, jax, jnp, np

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_2d

P, WL = 1.2, 1.0
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
BASE, DIRN = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3))
RECT = np.ones((15, 15))
RECT[3:12, 5:10] = 3.2                      # a non-square (C2v) pillar


def rc(cell, **kw):
    def run(xp, t):
        eps = (xp.asarray(cell) + t * xp.asarray(DIRN)).astype(complex)
        o, R, T = rcwa_efficiency_2d(P, P, eps, 1.45, 1.0, 0.45, WL,
                                     n_orders_x=3, n_orders_y=3, **kw)
        return o, R, T
    return run


def pm(**kw):
    def run(xp, t):
        a = kw.get("angle", 0.0)
        kk = dict(kw)
        kk["angle"] = (xp.asarray(a) + 0.0 * t) if xp is jnp else a
        o, R, T = pmm_efficiency_1d(P, xp.asarray(2.0 + 0j) + 0.0 * t, 1.0,
                                    1.45, 1.0, 0.45, 0.5, WL, degree=12,
                                    stabilize=False, **kk)
        return o, R, T
    return run


FIX = {
    "rcwa_sym_te": rc(BASE, polarization="te"),
    "rcwa_sym_tm": rc(BASE, polarization="tm"),
    "rcwa_sym_li_te": rc(BASE, polarization="te", formulation="li"),
    "rcwa_rect_te": rc(RECT, polarization="te"),
    "rcwa_sym_oblique_tm": rc(BASE, polarization="tm", theta=0.2),
    "rcwa_rect_conical_te": rc(RECT, polarization="te", theta=0.2, phi=0.3),
    "rcwa_uniform_te": rc(np.full((15, 15), 2.25), polarization="te"),
    "pmm1d_te_0": pm(polarization="te"),
    "pmm1d_tm_0": pm(polarization="tm"),
    "pmm1d_te_0.2": pm(polarization="te", angle=0.2),
    "pmm1d_tm_0.2": pm(polarization="tm", angle=0.2),
}


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


out = {}
for name, run in FIX.items():
    o, R, T = run(np, 0.0)
    rec = {"numpy": sha(o, R, T)}
    o, R, T = run(jnp, 0.0)
    rec["jax_eager"] = sha(R, T)
    R, T = jax.jit(lambda t: run(jnp, t)[1:])(0.0)
    rec["jax_jit"] = sha(R, T)
    rec["R0"] = float(np.asarray(R).ravel()[0])
    out[name] = rec
    print(name, rec, flush=True)
print(dump("b1_fwd_bytes.json", out))

"""G2: FORWARD BYTES before / after the round-2 routing of ``rcwa_jones_2d``
and ``RCWAStack``.  SHA-256 of R, T (and the zeroth-order Jones) for every
fixture, NumPy eager, JAX eager and JAX under jax.jit; ``SG_TAG=pre`` on the
base tree, ``SG_TAG=r2post`` on the fixed one, compared by
``g2_compare.py``.

    python g2_fwd_bytes_rcwa.py

Fixtures (g1's cell: 15 x 15, centre 4 / sides 1.5 / corners 1, n_orders
3 x 3, P 1.2, wl 1, n_substrate 1, n_superstrate 1.45; t = 0 enters as
``base + t * x-sides`` so the jit input is traced):
  jones_a        eps * I tensor cell, normal incidence
  jones_c        the same at theta 0.2, phi 0.3
  jones_oop      the centre block a uniaxial LC (n_o 1.5, n_e 1.7, director
                 tilted 30 deg OUT of the x-y plane: polar 60 deg from +z,
                 azimuth 0), the rest eps * I; normal incidence
  jones_oop_c    jones_oop at theta 0.2, phi 0.3
  stack1         RCWAStack, the scalar cell, depth 0.45
  stack2         + a uniform spacer eps 2.25, thickness 0.1
  stack2p        + a second patterned layer (the base cell, 0.2)
  stack1_c       stack1 at theta 0.2, phi 0.3
NB JAX eager on a concrete in-plane tensor cell runs rcwa_jones_2d's
IN-PLANE branch; under jit the traced cell runs the general (out-of-plane)
branch.  The RCWAStack homogeneous-mode cache is cleared before every call
(it keeps arrays built inside a jit trace -- see g1 section 0).
"""
import hashlib

from _h import dump, jax, jnp, np

import lumenairy.elements.rcwa._core as RCcore
from lumenairy.elements.rcwa import RCWAStack, rcwa_jones_2d
from lumenairy.elements.rcwa._core import uniaxial_tensor

P, WL, D = 1.2, 1.0, 0.45
NSUB, NSUP = 1.0, 1.45
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
c3 = np.zeros((3, 3))
c3[1, 1] = 1.0
BASE, DX, CEN = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3, c3))
I3 = np.eye(3)
LC = uniaxial_tensor(1.5, 1.7, np.deg2rad(60.0))
ISO_T = BASE[:, :, None, None] * I3
OOP_T = np.where(CEN[:, :, None, None] > 0, LC[None, None], ISO_T)


def jones(cell_t, **kw):
    def run(xp, t):
        cell = (xp.asarray(cell_t) + t * xp.asarray(DX)[:, :, None, None]
                * xp.asarray(I3)).astype(complex)
        o, R, T, J = rcwa_jones_2d(P, P, cell, NSUB, NSUP, D, WL,
                                   n_orders_x=3, n_orders_y=3, **kw)
        return o, R, T, J
    return run


def stack(second=None, theta=0.0, phi=0.0):
    def run(xp, t):
        RCcore._clear_rcwa_caches()
        eps = (xp.asarray(BASE) + t * xp.asarray(DX)).astype(complex)
        st = RCWAStack(P, period_y=P, n_superstrate=NSUP, n_substrate=NSUB,
                       n_orders=3, n_orders_y=3)
        st.add_layer(D, eps_cell=eps)
        if second == "spacer":
            st.add_layer(0.1, eps=2.25)
        elif second == "patterned":
            st.add_layer(0.2, eps_cell=xp.asarray(BASE).astype(complex))
        st.set_source(WL, theta=theta, phi=phi)
        res = st.solve()
        o, R, T = res.efficiencies()
        return o, R, T, res.jones_reflection()
    return run


FIX = {
    "jones_a": jones(ISO_T),
    "jones_c": jones(ISO_T, theta=0.2, phi=0.3),
    "jones_oop": jones(OOP_T),
    "jones_oop_c": jones(OOP_T, theta=0.2, phi=0.3),
    "stack1": stack(),
    "stack2": stack("spacer"),
    "stack2p": stack("patterned"),
    "stack1_c": stack(theta=0.2, phi=0.3),
}


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


out = {}
for name, run in FIX.items():
    rec = {}
    for mode in ("numpy", "jax_eager", "jax_jit"):
        try:
            if mode == "numpy":
                o, R, T, J = run(np, 0.0)
                rec[mode] = sha(o, R, T, J)
            elif mode == "jax_eager":
                o, R, T, J = run(jnp, 0.0)
                rec[mode] = sha(R, T, J)
            else:
                R, T, J = jax.jit(lambda t, run=run: run(jnp, t)[1:])(0.0)
                rec[mode] = sha(R, T, J)
                rec["R0"] = float(np.asarray(R).ravel()[0])
        except Exception as e:
            rec[mode] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    out[name] = rec
    print(name, rec, flush=True)
print(dump("g2_fwd_bytes_rcwa.json", out))

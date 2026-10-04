"""H: how many eigendecompositions does a routed twin's JITTED forward (and
jit(jacrev)) actually execute?  ``_jax_cluster_routed`` runs the solve twice
under a trace (record pass + the rule's consumer); its block comment says
the record pass's downstream is dead code that XLA removes.  The d4-d7
``capture`` (an ORDERED debug callback, an effect XLA must keep) sees every
eig twice on the new tree, so it cannot answer this.  Here: count the eig
custom calls (``geev``) in the lowered StableHLO (before optimisation) and in
the compiled HLO (after DCE), PRE vs POST.

    python h_eigcount.py
"""
import re

from _h import dump, jax, jnp, np

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import PMM2DStackHybrid, PMMStack, pmm_jones_1d

GEEV = re.compile(r"geev", re.IGNORECASE)


def counts(fn, x):
    low = jax.jit(fn).lower(jnp.asarray(x))
    pre = len(GEEV.findall(low.as_text()))
    post = len(GEEV.findall(low.compile().as_text()))
    return pre, post


XEXY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def f_bm(d):
    e = 2.56 * jnp.eye(3, dtype=complex) + d * jnp.asarray(XEXY)
    R, T, Jr, _ = berreman_jones_1d(
        [(e, jnp.asarray(0.3e-6)), (jnp.asarray(2.1 + 0j),
                                    jnp.asarray(0.1e-6))],
        jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j), jnp.asarray(1e-6))
    return jnp.concatenate([R, T, jnp.real(jnp.ravel(Jr))])


def f_pj(a):
    _o, R, T, _J = pmm_jones_1d(1.2, 4.0 * np.eye(3), np.eye(3), 1.0, 1.45,
                                0.45, 0.5, 1.0, angle=a, degree=12,
                                stabilize=False)
    return jnp.concatenate([jnp.ravel(jnp.asarray(R)),
                            jnp.ravel(jnp.asarray(T))])


def f_ps(a):
    st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=12)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.add_layer(0.15e-6, segments=[(0.1, 1.0), (0.3, 2.25), (0.6, 1.0)])
    st.set_source(1.0e-6, angle=a)
    _o, R, T, _J = st.solve()
    return jnp.concatenate([jnp.ravel(jnp.asarray(R)),
                            jnp.ravel(jnp.asarray(T))])


C4 = np.full((6, 6), 1.0 + 0j)
C4[0:3, 0:3] = 4.0


def f_s2(th):
    st = PMM2DStackHybrid(1.2e-6, n_substrate=1.0, n_superstrate=1.45,
                          degree=7, n_orders=3, formulation="laurent")
    st.add_layer(0.3e-6, eps_cell=C4)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(1.0e-6, theta=th)
    _o, R, T, _J = st.solve()
    return jnp.concatenate([jnp.ravel(jnp.asarray(R)),
                            jnp.ravel(jnp.asarray(T))])


out = {}
for name, fn in (("berreman_exy", f_bm), ("pmmjones1d_angle", f_pj),
                 ("pmmstack_shared_2L_angle", f_ps),
                 ("stack2d_c4v_laurent_theta", f_s2)):
    rec = {}
    for lab, g in (("fwd", fn), ("jacrev", jax.jacrev(fn))):
        try:
            rec[lab] = dict(zip(("lowered", "compiled"), counts(g, 0.0)))
        except Exception as e:      # recorded, not hidden
            rec[lab] = {"error": repr(e)[:400]}
    out[name] = rec
    print(name, rec, flush=True)
print(dump("h_eigcount.json", out))

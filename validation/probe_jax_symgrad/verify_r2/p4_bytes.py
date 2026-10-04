"""Claims (2) and (3): forward values of every routed entry and of the
controls, NumPy / JAX eager / jax.jit, saved per tree (LUMROOT) for a byte
comparison PRE (7c0bc8bd) vs POST (HEAD).  python p4_bytes.py <tag>"""
import sys
import warnings

from _vc import BUILD, jax, jnp, np

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import PMM2DStackHybrid, PMMStack, pmm_efficiency_1d, pmm_jones_1d
from lumenairy.elements.rcwa import RCWAStack, rcwa_efficiency_1d, rcwa_efficiency_2d, rcwa_jones_2d

S = 24
CROSS = np.ones((S, S), complex)
CROSS[8:16, :] = 0.0
CROSS[:, 8:16] = 0.0
ARMX = np.zeros((S, S))
ARMX[8:16, 16:24] = 1.0
M = CROSS == 0.0
OUT = {}
NOTES = {}


def cross(eps_a):
    return np.where(M, eps_a, 1.44 + 0j)


def tens_iso(e):
    return e[:, :, None, None] * np.eye(3, dtype=complex)[None, None]


def tens_aniso(e):
    t = tens_iso(e).copy()
    t[:, :, 1, 1] *= 0.8
    t[:, :, 0, 1] = t[:, :, 1, 0] = np.where(M, 0.3, 0.0)
    return t


def tens_tilt(e):
    t = tens_aniso(e).copy()
    t[:, :, 0, 2] = t[:, :, 2, 0] = np.where(M, 0.4, 0.0)
    return t


def put(name, arrs):
    OUT[name] = np.concatenate([np.ravel(np.asarray(a)).astype(complex)
                                for a in arrs])


def three(name, fn, x):
    """NumPy / eager JAX / jitted JAX of fn(x, xp)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        put(name + "/np", fn(x, np))
    NOTES[name + "/np"] = sorted({str(i.message)[:90] for i in w
                                  if "notice" in str(i.message).lower()
                                  or "scope" in str(i.message).lower()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        put(name + "/jax", fn(jnp.asarray(x), jnp))
    NOTES[name + "/jax"] = sorted({str(i.message)[:90] for i in w
                                   if "notice" in str(i.message).lower()
                                   or "scope" in str(i.message).lower()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        put(name + "/jit", jax.jit(lambda y: tuple(fn(y, jnp)))(
            jnp.asarray(x)))
    NOTES[name + "/jit"] = sorted({str(i.message)[:90] for i in w
                                   if "notice" in str(i.message).lower()
                                   or "scope" in str(i.message).lower()})


# ---- rcwa_efficiency_2d
for th, ph in ((0.0, 0.0), (0.2, 0.3)):
    for pol in ("te", "tm"):
        def f(x, xp, th=th, ph=ph, pol=pol):
            e = xp.asarray(cross(4.0 + 0.3j)) + x * xp.asarray(ARMX)
            _o, R, T = rcwa_efficiency_2d(1.3, 1.3, e, 1.5, 1.0, 0.3, 1.0,
                                          theta=th, phi=ph, polarization=pol,
                                          n_orders_x=5, n_orders_y=5)
            return [R, T]
        three(f"eff2d/{pol}/{th}", f, 0.0)

# ---- rcwa_jones_2d: iso / in-plane aniso (exy) / tilted, laurent & li
for kind, tf in (("iso", tens_iso), ("aniso", tens_aniso),
                 ("tilt", tens_tilt)):
    for form in ("laurent", "li"):
        for th, ph in ((0.0, 0.0), (0.2, 0.3)):
            def f(x, xp, tf=tf, form=form, th=th, ph=ph):
                t = xp.asarray(tf(cross(5.0 + 0.2j))) + x * xp.asarray(
                    tens_iso(ARMX.astype(complex)))
                _o, R, T, J = rcwa_jones_2d(1.3, 1.3, t, 1.5, 1.0, 0.3, 1.0,
                                            theta=th, phi=ph, n_orders_x=5,
                                            n_orders_y=5, formulation=form)
                return [R, T, J]
            try:
                three(f"jones2d/{kind}/{form}/{th}", f, 0.0)
            except Exception as e:  # noqa: BLE001
                NOTES[f"jones2d/{kind}/{form}/{th}"] = f"RAISE {type(e).__name__}: {e}"[:200]


# ---- RCWAStack: two patterned layers + spacer, normal and conical
for th, ph in ((0.0, 0.0), (0.2, 0.3)):
    def f(x, xp, th=th, ph=ph):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.2, eps_cell=xp.asarray(cross(6.25 + 0j))
                     + x * xp.asarray(ARMX))
        st.add_layer(0.1, eps=2.1)
        st.add_layer(0.15, eps_tensor_cell=xp.asarray(tens_aniso(
            cross(3.0 + 0.2j))))
        st.set_source(1.0, theta=th, phi=ph)
        _o, R, T = st.solve().efficiencies()
        return [R, T]
    three(f"rcwastack/{th}", f, 0.0)


# ---- Berreman: off-plane tensor (traced route), in-plane tensor, scalars
def berr(kind, angle):
    def f(x, xp):
        if kind == "offplane":
            e1 = xp.asarray(np.array([[3.0, 0.2, 0.4], [0.2, 2.8, 0],
                                      [0.4, 0, 3.1]], complex)) + x
        elif kind == "inplane":
            e1 = xp.asarray(np.diag([3.0, 2.6, 2.9]).astype(complex)) + x
        else:
            e1 = xp.asarray(3.0 + 0.1j) + x
        if xp is jnp:
            return list(berreman_jones_1d(
                [(e1, jnp.asarray(0.25e-6)), (jnp.asarray(2.0 + 0j),
                                              jnp.asarray(0.15e-6))],
                jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j),
                jnp.asarray(0.9e-6), angle=angle))
        return list(berreman_jones_1d([(e1, 0.25e-6), (2.0, 0.15e-6)], 1.5,
                                      1.0, 0.9e-6, angle=angle))
    return f


for kind in ("offplane", "inplane", "scalar"):
    for ang in (0.0, 0.35):
        three(f"berreman/{kind}/{ang}", berr(kind, ang), 0.0)

# ---- 1-D PMM
for ang in (0.0, 0.2):
    for pol in ("te", "tm"):
        def f(x, xp, ang=ang, pol=pol):
            _o, R, T = pmm_efficiency_1d(0.9, xp.asarray(5.0 + 0.2j) + x, 1.2,
                                         1.5, 1.0, 0.3, 0.4, 1.0, angle=ang,
                                         polarization=pol, degree=10,
                                         stabilize=False)
            return [R, T]
        three(f"pmmeff1d/{pol}/{ang}", f, 0.0)

    def f(x, xp, ang=ang):
        er = xp.asarray(np.diag([5.0 + 0.2j, 4.0 + 0.2j, 4.5 + 0.2j])) + x
        _o, R, T, J = pmm_jones_1d(0.9, er, 1.2 * np.eye(3, dtype=complex),
                                   1.5, 1.0, 0.3, 0.4, 1.0, angle=ang,
                                   degree=10, stabilize=False)
        return [R, T, J]
    three(f"pmmjones1d/{ang}", f, 0.0)

for grids in ("shared", "per-layer"):
    for ang in (0.0, 0.2):
        def f(x, xp, grids=grids, ang=ang):
            st = PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0,
                          degree=10, layer_grids=grids)
            st.add_layer(0.25e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
            st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
            st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                           (0.15, 3.0), (0.5, 1.5)])
            st.set_source(0.8e-6, angle=(ang + x) if xp is jnp else ang)
            _o, R, T, J = st.solve()
            return [R, T, J]
        three(f"pmmstack/{grids}/{ang}", f, 0.0)

# ---- hybrid 2-D stack: concrete (np), traced layout (jit) laurent / li
S8 = 8
C1 = np.full((S8, S8), 1.0 + 0j)
C1[0:4, 0:4] = 3.0
LAY = np.zeros((S8, S8), np.int64)
for i in range(4):
    for j in range(4):
        LAY[i, j] = 1 + 4 * i + j
for form in ("laurent", "li"):
    def hy(x, layout, form=form):
        st = PMM2DStackHybrid(1.1e-6, n_substrate=1.5, n_superstrate=1.0,
                              degree=7, n_orders=3, formulation=form)
        kw = {} if layout is None else {"region_layout": layout}
        st.add_layer(0.25e-6, eps_cell=x, **kw)
        st.add_layer(0.1e-6, eps=2.1)
        st.set_source(0.95e-6, theta=0.0)
        _o, R, T, J = st.solve()
        return [R, T, J]
    put(f"hybrid/{form}/np", hy(C1, None))
    put(f"hybrid/{form}/jit_layout", jax.jit(lambda c: tuple(hy(c, LAY)))(
        jnp.asarray(C1)))

# ---- control: 1-D RCWA
for pol in ("te", "tm"):
    def f(x, xp, pol=pol):
        _o, R, T = rcwa_efficiency_1d(0.9, xp.asarray(2.2 + 0.05j) + x, 1.1,
                                      1.5, 1.0, 0.3, 0.4, 1.0, angle=0.1,
                                      polarization=pol, n_orders=11)
        return [R, T]
    three(f"rcwa1d/{pol}", f, 0.0)

tag = sys.argv[1]
np.savez(f"p4_bytes_{tag}_{BUILD}.npz", **{k.replace("/", "|"): v
                                            for k, v in OUT.items()})
import json  # noqa: E402

json.dump(NOTES, open(f"p4_notes_{tag}_{BUILD}.json", "w"), indent=1)
print(len(OUT), "entries")

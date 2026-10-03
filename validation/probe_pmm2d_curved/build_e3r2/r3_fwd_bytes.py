"""R3: the twin's FORWARD values are byte-identical to the E3 build's
(``d4e92eb5``): SHA-256 of R, T and the Jones matrix on the E3-2 fixture set
plus the V-E3-1 / V-E3-2 references, each three ways -- eager at the
reference parameters, eager with a traced-type (JAX) shape parameter, and
under ``jax.jit``.

    LUM_TREE=<tree> python r3_fwd_bytes.py M LABEL

writes ``r3_fwd_bytes_<LABEL>_M<M>_<build>.json``; ``r3_compare.py`` diffs
two labels.
"""
import hashlib
import sys

from _r2 import P, PMM2DStackPure, dump, jax, jnp, np

from lumenairy.elements.pmm import Circle, Ellipse, FilletRect, Rect, SinusoidalWall

M, LABEL = int(sys.argv[1]), sys.argv[2]
_EYE = np.eye(3, dtype=complex)
_LC = np.diag([2.25, 2.89, 2.4]).astype(complex)
_LC[0, 1] = _LC[1, 0] = 0.2
_MU_G = np.array([[1.2, 0.1j, 0], [-0.1j, 1.2, 0], [0, 0, 1.0]], complex)


def circ(r=0.36, cx=0.6):
    return [dict(thickness=0.5, shapes=[Circle(cx, 0.6, r, 4.0)],
                 background_eps=1.0)]


FIX = {
    "scalar_pillar": (dict(layers=[dict(thickness=0.4, eps_cell=np.array(
        [[4.0, 1.0], [1.0, 1.0]], complex))]), None),
    "scalar_conical_lossy": (dict(layers=[dict(thickness=0.4,
                                               eps_cell=np.array(
        [[4.0 + 0.5j, 1.0], [1.0, 1.0]]))], theta=0.25, phi=0.4), None),
    "tensor_lc": (dict(layers=[dict(thickness=0.4, eps_cell=np.where(
        np.arange(4).reshape(2, 2, 1, 1) == 0, _LC, 2.0 * _EYE))]), None),
    "magnetic_tensor": (dict(layers=[dict(
        thickness=0.4, eps_cell=np.array([[4.0, 1.0], [1.0, 1.0]], complex),
        mu_cell=np.where(np.arange(4).reshape(2, 2, 1, 1) == 0, _MU_G,
                         _EYE))]), None),
    "multilayer": (dict(layers=[dict(thickness=0.2, eps=2.1),
                                dict(thickness=0.3, eps_cell=np.array(
                                    [[4.0 + 0.3j, 1.0], [1.0, 1.0]])),
                                dict(thickness=0.15, eps=_LC)],
                        theta=0.15, phi=0.2), None),
    "uniform_only": (dict(layers=[dict(thickness=0.2, eps=2.1),
                                  dict(thickness=0.3, eps=_LC)],
                          theta=0.15, phi=0.2), None),
    "circle": (dict(layers=circ()),
               lambda v: [Circle(0.6, 0.6, v, 4.0)]),
    "circle_conical": (dict(layers=circ(), theta=0.3, phi=0.4),
                       lambda v: [Circle(0.6, 0.6, v, 4.0)]),
    "fillet": (dict(layers=[dict(thickness=0.4, shapes=[FilletRect(
        0.6, 0.6, 0.6, 0.5, 0.1, 4.0)], background_eps=1.0)]),
        lambda v: [FilletRect(0.6, 0.6, 0.6, 0.5, v * 0.1 / 0.36, 4.0)]),
    "sinusoid": (dict(layers=[dict(thickness=0.4, shapes=[SinusoidalWall(
        "x", 0.6, 0.08, eps=2.25)], background_eps=1.0)]),
        lambda v: [SinusoidalWall("x", 0.6, v * 0.08 / 0.36, eps=2.25)]),
    "square_pillar": (dict(layers=[dict(thickness=0.45, shapes=[Rect(
        0.6, 0.6, 0.5, 0.5, 3.5)], background_eps=1.0)]),
        lambda v: [Rect(0.6, 0.6, v * 0.5 / 0.36, 0.5, 3.5)]),
    "ellipse_sym": (dict(layers=[dict(thickness=0.45, shapes=[Ellipse(
        0.6, 0.6, 0.33, 0.33, 3.5)], background_eps=1.0)]),
        lambda v: [Ellipse(0.6, 0.6, v * 0.33 / 0.36, 0.33, 3.5)]),
    "rect_oblique": (dict(layers=[dict(thickness=0.45, shapes=[Rect(
        0.6, 0.6, 0.5, 0.4, 3.5)], background_eps=1.0)], theta=0.3),
        lambda v: [Rect(0.6, 0.6, v * 0.5 / 0.36, 0.4, 3.5)]),
    "circle_tensor_magnetic_two_layer": (dict(layers=[
        dict(thickness=0.3, shapes=[Circle(0.6, 0.6, 0.36, _LC)],
             background_eps=2.0),
        dict(thickness=0.2, shapes=[Circle(0.6, 0.6, 0.36, 4.0, mu=_MU_G)],
             background_eps=1.0)]), None),
}


def build(fx):
    kw = {k: v for k, v in fx.items() if k != "layers"}
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend="jax")
    for L in fx["layers"]:
        st.add_layer(**L)
    st.set_source(1.0, theta=kw.get("theta", 0.0), phi=kw.get("phi", 0.0))
    return st


def h(arrs):
    return [hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes())
            .hexdigest()[:16] for a in arrs]


out = {"M": M, "label": LABEL}
for name, (fx, shp) in FIX.items():
    st = build(fx)
    rec = {"eager_ref": h(st.solve()[1:])}
    if shp is not None:
        tw = st.jax_twin()

        def f(v, tw=tw, st=st, shp=shp):
            p = tw.params()
            p["layers"][0]["shapes"] = shp(v)
            return st.solve(params=p)[1:]
        v0 = 0.36
        rec["eager_traced"] = h(f(jnp.asarray(v0)))
        rec["jit"] = h(jax.jit(f)(v0))
        rec["T_jit"] = np.asarray(jax.jit(f)(v0)[1]).ravel()[:4].tolist()
    out[name] = rec
    print(name, rec, flush=True)
print(dump(f"r3_fwd_bytes_{LABEL}_M{M}.json", out))

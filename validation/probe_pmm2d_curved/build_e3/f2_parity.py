"""E3-2 FORWARD PARITY: the twin vs the NumPy stack on >= 20 fixtures, with
the bar DERIVED from the round-off of the pipeline's stages.

    python f2_parity.py M [fixture ...]

For every fixture three solves:
  numpy    -- the shipped stack (QZ on the pencil);
  numpy_se -- the SAME NumPy stack with the pencil eig replaced by the
              standard eig of G^-1 L (the reduction the twin uses), i.e. the
              shipped arithmetic with ONE stage changed to an equally exact
              algorithm: the difference numpy_se - numpy is the round-off
              reach of the eig stage on this fixture;
  twin     -- the JAX twin (eager), whose remaining differences from
              numpy_se are the summation orders of the vectorised quadrature
              and of the XLA kernels.
Per fixture: max |dR|, |dT| over every order and |dJones| for twin-numpy,
twin-numpy_se and numpy_se-numpy, plus the operator-stage parity of the
patterned layer (max |L_twin - L_numpy| / max |L_numpy|).
Output f2_parity_M<M>.json.
"""
import sys

import scipy.linalg as _sla
from _e3common import CM, TS, WL, P, PMM2DStackPure, StagJaxTwin, absd, dump, jnp, np  # noqa

from lumenairy.elements.pmm import Circle, Ellipse, FilletRect, Rect, SinusoidalWall
from lumenairy.elements.rcwa._core import uniaxial_tensor

M = int(sys.argv[1])
ONLY = set(sys.argv[2:])
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
GYRO = np.array([[2.25, 0.5j, 0], [-0.5j, 2.25, 0], [0, 0, 2.0]], complex)
MU_G = np.array([[1.6, 0.4j, 0], [-0.4j, 1.6, 0], [0, 0, 1.2]], complex)
EYE = np.eye(3, dtype=complex)


def tcell(host, incl, n=2):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    c[0, 0] = incl
    return c


pil2 = np.array([[4.0, 1.0], [1.0, 1.0]], complex)


def stk(n_orders=2, cmap=None, **src):
    kw = {} if cmap is None else {"cmap": cmap}
    s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                       n_orders=n_orders, **kw)
    return s, dict(theta=src.get("theta", 0.0), phi=src.get("phi", 0.0))


def F(layers, **kw):
    """layers: list of add_layer kwargs dicts."""
    def build():
        s, src = stk(**kw)
        for L in layers:
            s.add_layer(**L)
        s.set_source(WL, **src)
        return s
    return build


cms = CM.SeparableStretch(np.array([0.0, 0.3, 0.9, P]),
                          np.array([0.0, 0.4, 0.8, P]),
                          fx=CM.SineStretch(0.05 * P), period_x=P, period_y=P)
stripe = np.ones((3, 3), complex)
stripe[1, :] = 4.0 + 0.5j
circ = lambda e=4.0, **k: Circle(0.6, 0.6, 0.36, e, **k)  # noqa: E731
FIX = {
    "scalar_pillar": F([dict(thickness=0.4, eps_cell=pil2)]),
    "scalar_rect_walls": F([dict(thickness=0.4, shapes=[Rect(0.6, 0.55, 0.5,
                                                            0.4, 4.0)],
                                 background_eps=1.0)]),
    "scalar_oblique": F([dict(thickness=0.4, eps_cell=pil2)], theta=0.2),
    "scalar_conical": F([dict(thickness=0.4, eps_cell=pil2)], theta=0.25,
                        phi=0.4),
    "lossy_scalar": F([dict(thickness=0.4, eps_cell=np.array(
        [[4.0 + 0.5j, 1.0], [1.0, 1.0]]))]),
    "tensor_lc": F([dict(thickness=0.4, eps_cell=tcell(2.0 * EYE, LC))]),
    "tensor_lc_conical": F([dict(thickness=0.4,
                                 eps_cell=tcell(2.0 * EYE, LC))],
                           theta=0.25, phi=0.4),
    "tensor_gyro": F([dict(thickness=0.4, eps_cell=tcell(EYE, GYRO))]),
    "magnetic_scalar": F([dict(thickness=0.4, eps_cell=pil2,
                               mu_cell=np.ones((2, 2), complex) * 1.3)]),
    "magnetic_tensor": F([dict(thickness=0.4, eps_cell=pil2,
                               mu_cell=tcell(EYE, MU_G))]),
    "multilayer": F([dict(thickness=0.2, eps=2.1),
                     dict(thickness=0.3, eps_cell=np.array(
                         [[4.0 + 0.3j, 1.0], [1.0, 1.0]])),
                     dict(thickness=0.15, eps=LC)], theta=0.15, phi=0.2),
    "mapped_stretch_stripe": F([dict(thickness=0.4, eps_cell=stripe)],
                               cmap=cms),
    "circle": F([dict(thickness=0.5, shapes=[circ()], background_eps=1.0)]),
    "circle_conical": F([dict(thickness=0.5, shapes=[circ()],
                              background_eps=1.0)], theta=0.3, phi=0.4),
    "circle_lossy_oblique": F([dict(thickness=0.5,
                                    shapes=[circ(2.25 + 0.1j)],
                                    background_eps=1.0)], theta=0.25),
    "circle5": F([dict(thickness=0.5, shapes=[Circle(0.6, 0.6, 0.36, 4.0,
                                                     core=0.5)],
                       background_eps=1.0)]),
    "fillet": F([dict(thickness=0.4, shapes=[FilletRect(0.6, 0.6, 0.6, 0.5,
                                                        0.1, 4.0)],
                      background_eps=1.0)]),
    "ellipse_rot": F([dict(thickness=0.4, shapes=[Ellipse(
        0.6, 0.6, 0.35, 0.25, 4.0, angle=0.3)], background_eps=1.0)]),
    "sinusoid": F([dict(thickness=0.4, shapes=[SinusoidalWall(
        "x", 0.6, 0.08, eps=2.25)], background_eps=1.0)]),
    "two_layer_merged": F([dict(thickness=0.3, shapes=[circ(4.0 + 0.2j)],
                                background_eps=1.0),
                           dict(thickness=0.2, eps=2.1),
                           dict(thickness=0.25, shapes=[Rect(
                               0.6, 0.6, 0.3, 0.3, 2.25)],
                                background_eps=1.0)]),
    "circle_tensor_lc": F([dict(thickness=0.5, shapes=[circ(LC)],
                                background_eps=2.0)]),
    "circle_magnetic": F([dict(thickness=0.5, shapes=[circ(4.0, mu=MU_G)],
                               background_eps=1.0)]),
}


class _SE:
    """scipy.linalg stand-in whose eig(L, G) is the standard eig of G^-1 L
    (the twin's reduction) -- every other attribute is scipy.linalg's."""

    def __getattr__(self, k):
        return getattr(_sla, k)

    @staticmethod
    def eig(a, b=None, **kw):
        if b is None:
            return _sla.eig(a, **kw)
        return np.linalg.eig(np.linalg.solve(b, a))


out = {"M": M, "fixtures": {}}
for name, build in FIX.items():
    if ONLY and name not in ONLY:
        continue
    o, R, T, J = build().solve()
    TS.sla = _SE()
    try:
        o2, R2, T2, J2 = build().solve()
    finally:
        TS.sla = _sla
    st = build()
    st.backend = "jax"
    tw = st.jax_twin()
    _o, Rj, Tj, Jj = tw.solve()
    Rj, Tj, Jj = (np.asarray(a) for a in (Rj, Tj, Jj))
    # operator stage: the first non-uniform layer's shadow vs its NumPy ref
    opd = None
    for rec in tw.layers:
        if "ref" in rec:
            from lumenairy.elements.pmm._jax_twod_staggered import _shadow
            ref = rec["ref"]
            lp = tw.params()["layers"][tw.layers.index(rec)]
            if "shapes" in lp:
                lp = dict(lp, eps=rec["eps"], mu=rec.get("mu"))
            ec, mc = tw._layer_cells(rec, lp)
            sh = _shadow(ref, jnp, ec, mc, tw.cmap_ref)
            opd = float(np.max(np.abs(np.asarray(sh.Lmat) - ref.Lmat))
                        / np.max(np.abs(ref.Lmat)))
            break
    rec = {"twin_numpy": [absd(Rj, R), absd(Tj, T), absd(Jj, J)],
           "twin_numpy_se": [absd(Rj, R2), absd(Tj, T2), absd(Jj, J2)],
           "numpy_se_numpy": [absd(R2, R), absd(T2, T), absd(J2, J)],
           "op_rel": opd, "closure": float(np.max(np.abs(
               R.sum(1) + T.sum(1) - 1.0)))}
    out["fixtures"][name] = rec
    print(f"{name:24s} twin-np {rec['twin_numpy'][0]:.1e} "
          f"{rec['twin_numpy'][1]:.1e} {rec['twin_numpy'][2]:.1e} | se-np "
          f"{rec['numpy_se_numpy'][0]:.1e} {rec['numpy_se_numpy'][1]:.1e} "
          f"{rec['numpy_se_numpy'][2]:.1e} | twin-se "
          f"{rec['twin_numpy_se'][1]:.1e} | op {opd}", flush=True)
dump(f"f2_parity_M{M}.json", out)

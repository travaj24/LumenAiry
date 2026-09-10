"""SLANT on the PER-LAYER (L2 mortar) path of the pure staggered 2-D PMM.

The build that added ``PMM2DStackPure(layer_grids='per-layer')``
(``docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md``) carried ``slant``
through the new cascade unchanged and listed the dedicated gate as an OPEN
ITEM: *"it is exercised only by the shared-path suites plus the per-layer
plumbing; a dedicated per-layer slanted gate is NOT in this build and is the
obvious next measurement."*  This file is that gate.  Measurements and the
verdicts live in ``docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md``
and the reproducer is ``validation/probe_verify_mortar/v4_slant_mixed.py``.

Why the per-layer path needs its own slanted gate, in one line: a slanted
layer returns DISTINCT forward/backward mode sets and puts the whole stack on
the GENERALIZED cascade, so it exercises
``_interface_smatrix_general_mortar_2d`` -- a different function from the
square in-plane mortar -- and it additionally carries the frame-anchor
transmission phase, which is a far-field correction the mortar knows nothing
about.  Neither is covered by a vertical per-layer test.

Shapes, per ``docs/TESTING_STANDARDS.md``:

* the bit-identity claim is SAME-BUILD two-arm by sha256;
* the conforming-identity bar is DERIVED AT RUNTIME from the stack's own
  ``eps * cond_2(G)`` (it grows with ``M``);
* the accuracy claim is scored against an INDEPENDENT engine (the shipped 1-D
  inclined-coordinate solver ``pmm_efficiency_1d_slanted``) and barred against
  the VERTICAL control measured in the same test, not against an absolute
  number -- the vertical arm is what this basis / oracle pair can do on this
  geometry, and the slanted arm has to track it;
* the slant SIGN carries a two-sided FAIL-BEFORE, per order, because a slanted
  grating is not x-mirror-symmetric and the sign is observable there -- while
  the wrong-sign arm conserves energy to the same digits, which is the
  lossless trap and is asserted rather than merely written down.

MEASURED ON THIS BUILD (2026-09-11, Windows 11 / CPython 3.14.6 / numpy 2.4.4
/ scipy 1.17.1 scipy-openblas, OMP/OPENBLAS/MKL = 1), split ``N=2 | N=4`` at
``M_A = 7`` / ``M_B = 5``, orders -1..+1, both polarizations:

    slant  0 deg normal   TE 1.815e-06   TM 8.947e-04   closure 9.54e-07
    slant  0 deg obl 22   TE 3.343e-06   TM 9.708e-05   closure 6.09e-09
    slant 15 deg normal   TE 7.043e-07   TM 6.854e-04   closure 9.13e-07
                  WRONG SIGN  TE 9.215e-02  (130843x)   closure 9.13e-07
    slant 15 deg obl 22   TE 1.983e-06   TM 1.807e-04   closure 2.07e-08
                  WRONG SIGN  TE 1.888e-02  (9523x)     closure 2.65e-08
    slant 28 deg normal   TE 5.611e-06   TM 6.815e-04   closure 8.97e-07
                  WRONG SIGN  TE 2.106e-01  (37526x)    closure 8.97e-07
    slant 28 deg obl 22   TE 2.471e-06   TM 2.018e-04   closure 3.52e-08
                  WRONG SIGN  TE 3.123e-02  (12643x)    closure 3.58e-08

and the modal ladder at 28 deg / obl 22: TE 7.698e-04 -> 2.471e-06 ->
1.190e-07 at ``(M_A, M_B) = (5,4) / (7,5) / (9,6)`` with closure 1.70e-05 ->
3.52e-08 -> 4.08e-10.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Granet2DTransverseE,
)

_P = 0.80
_WL = 1.0
_DEP = 0.28
_NR, _NG = 2.0, 1.0
_NSUP, _NSUB = 1.0, 1.5
_ORD = (-1, 0, 1)

# duty 1/2 on TWO grids the pair is genuinely non-conforming on: N = 2 has its
# only wall at P/2, N = 4 has walls at P/4, P/2, 3P/4.
_CELL2 = np.array([[_NR ** 2, _NR ** 2], [_NG ** 2, _NG ** 2]], dtype=_C)
_CELL4 = np.array([[_NR ** 2] * 4, [_NR ** 2] * 4,
                   [_NG ** 2] * 4, [_NG ** 2] * 4], dtype=_C)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _gram_cond(N, M):
    """``cond_2`` of the two V1 / V2 block field Grams on one grid -- the
    round-off scale that DERIVES the conforming-identity bar (it grows with
    ``M``, so a fixed constant would fail at large ``M``)."""
    sol = Granet2DTransverseE(_P, _P, N, N, M, np.full((N, N), 2.0 + 0j),
                              k0=2 * np.pi / _WL)
    G = -sol.Rmat
    qq = sol.q ** 2
    return max(float(np.linalg.cond(G[:qq, :qq])),
               float(np.linalg.cond(G[qq:, qq:])))


def _oracle(phi_deg, theta, pol, deg=22):
    """The shipped 1-D engine -- a DIFFERENT formulation (inclined
    coordinates) on a different basis, so it is an independent oracle."""
    if phi_deg == 0.0:
        o, R, T = pmm_efficiency_1d(_P, _NR, _NG, _NSUB, _NSUP, _DEP, 0.5,
                                    _WL, angle=theta, polarization=pol,
                                    degree=deg, far_field_orders=15)
    else:
        o, R, T = pmm_efficiency_1d_slanted(
            _P, _NR, _NG, _NSUB, _NSUP, _DEP, 0.5, _WL, np.deg2rad(phi_deg),
            angle=theta, polarization=pol, degree=deg, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def _split(slant, MA, MB, theta):
    """The slanted grating SPLIT into two layers at the SAME slant on
    NON-CONFORMING grids -- which is what puts the shear through
    ``_interface_smatrix_general_mortar_2d``."""
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=max(MA, MB), n_orders=3,
                        layer_grids="per-layer")
    st.add_layer(_DEP / 2, eps_cell=_CELL2, slant=slant, n_modes=MA)
    st.add_layer(_DEP / 2, eps_cell=_CELL4, slant=slant, n_modes=MB)
    st.set_source(_WL, theta=theta, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        o, R, T, _J = st.solve()
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    return ({m: (float(R[0, i]), float(T[0, i])) for m, i in idx.items()},
            {m: (float(R[1, i]), float(T[1, i])) for m, i in idx.items()},
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


def _perorder(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in _ORD if m in a and m in b)


# ==========================================================================
def test_slanted_per_layer_on_coinciding_grids_is_bit_exact_vs_shared():
    """A SLANTED per-layer stack whose grids COINCIDE takes the shipped
    generalized square interface and reproduces ``layer_grids='shared'`` byte
    for byte -- the bypass contract, on the generalized cascade.

    ... and with the bypass DISABLED (``force_mortar=True``) the SAME stack
    driven through ``_interface_smatrix_general_mortar_2d`` reproduces it to
    round-off.  The bar is DERIVED at runtime from this grid's own
    ``eps * cond_2(G)``: MEASURED 7.033e-15 (N=2, M=6) and 7.034e-15
    (N=4, M=5) against derived bars of 1.7e-11 and 1.4e-11, i.e. ~2000x
    inside.  A fixed constant would not survive a modal-count change."""
    t = float(np.tan(np.deg2rad(22.0)))
    for M, N, cell in ((6, 2, _CELL2), (5, 4, _CELL4)):
        def _build(lg):
            st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP,
                                n_substrate=_NSUB, n_modes=M, n_orders=2,
                                layer_grids=lg)
            extra = {"n_modes": M} if lg == "per-layer" else {}
            st.add_layer(_DEP / 2, eps_cell=cell, slant=(t, 0.0), **extra)
            st.add_layer(_DEP / 2, eps_cell=cell, slant=(t, 0.0), **extra)
            st.set_source(_WL, theta=0.19, phi=0.0)
            return st

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a = _build("shared").solve()
            b = _build("per-layer").solve()
            f = _build("per-layer")._solve_per_layer(
                jones=True, retain_internal=False, force_mortar=True)
        for x, y in zip(a[1:], b[1:]):
            assert _h(x) == _h(y), (N, M, float(np.max(np.abs(x - y))))
        scale = max(float(np.max(np.abs(a[1]))), float(np.max(np.abs(a[2]))))
        got = max(float(np.max(np.abs(a[1] - f[1]))),
                  float(np.max(np.abs(a[2] - f[2])))) / scale
        bar = 10.0 * np.finfo(float).eps * _gram_cond(N, M)
        assert got < bar, (N, M, got, bar)


@pytest.mark.parametrize("phi_deg, theta_deg", [(15.0, 0.0), (28.0, 22.0)])
def test_slanted_split_across_non_conforming_grids_matches_the_1d_oracle(
        phi_deg, theta_deg):
    """THE GATE.  A y-uniform slanted grating split across NON-CONFORMING
    per-layer grids must reproduce the shipped 1-D inclined-coordinate solver
    ORDER FOR ORDER -- and the slant SIGN is pinned two-sided on that same
    oracle.

    The bar is the VERTICAL control measured in this test, not an absolute
    number: the same split at slant 0 against ``pmm_efficiency_1d`` is what
    this basis, this mortar and this oracle can do on this geometry, and the
    slanted arm has to TRACK it.  MEASURED (WIN, ``M_A = 7`` / ``M_B = 5``):
    the vertical control reads 1.815e-06 at normal and 3.343e-06 at 22 deg;
    the slanted arms read 7.043e-07 / 5.611e-06 (normal, 15 / 28 deg) and
    1.983e-06 / 2.471e-06 (22 deg), i.e. 0.39x - 3.1x the control.  Barred at
    20x, with the measured worst 3.1x inside it.

    The WRONG-SIGN arm is the fail-before: 9.215e-02 (15 deg normal) up to
    2.106e-01 (28 deg normal) -- 9.5e+03x to 1.3e+05x the right-sign arm --
    while conserving energy to the SAME digits."""
    theta = np.deg2rad(theta_deg)
    t = float(np.tan(np.deg2rad(phi_deg)))
    orc_te = _oracle(phi_deg, theta, "te")
    ctl_te = _oracle(0.0, theta, "te")
    _tm_v, te_v, _c_v = _split(None, 7, 5, theta)
    vertical = _perorder(te_v, ctl_te)
    _tm_p, te_p, clo_p = _split((t, 0.0), 7, 5, theta)
    right = _perorder(te_p, orc_te)
    _tm_m, te_m, clo_m = _split((-t, 0.0), 7, 5, theta)
    wrong = _perorder(te_m, orc_te)
    # 1. the slanted arm TRACKS the vertical control through the mortar
    assert right < 20.0 * vertical, (phi_deg, theta_deg, right, vertical)
    # 2. ... and the sign is observable per order, by decades
    assert wrong > 1e-3, (phi_deg, theta_deg, wrong)
    assert wrong > 100.0 * right, (phi_deg, theta_deg, wrong, right)
    # 3. ... while ENERGY carries essentially no information about it -- the
    #    lossless trap, reproduced through the mortar.  MEASURED: 9.13e-07 vs
    #    9.13e-07 at 15 deg normal and 3.52e-08 vs 3.58e-08 at 28 deg / 22
    #    deg, i.e. the two closures agree to 2 % while the answers sit
    #    decades apart.  Barred at a factor 3, with the measured 1.02x inside.
    assert clo_m < 3.0 * clo_p and clo_p < 3.0 * clo_m, (clo_p, clo_m)
    assert clo_p < 1e-5 and clo_m < 1e-5, (clo_p, clo_m)


def test_slanted_split_converges_through_the_mortar():
    """The gate above is a single modal setting; this is the SHAPE claim --
    the split's per-order error against the 1-D oracle collapses with the two
    layers' modal counts, so the agreement is convergence and not a
    coincidence at one rung.

    MEASURED (28 deg slant, 22 deg incidence): TE 7.698e-04 -> 2.471e-06 at
    ``(M_A, M_B) = (5,4) -> (7,5)``, with closure 1.70e-05 -> 3.52e-08; the
    build-doc ladder continues to 1.190e-07 / 4.08e-10 at ``(9,6)``.  The
    DECISION barred here is 'the finer rung is at least a decade tighter on
    BOTH the observable and the closure', with the measured 312x / 483x
    inside it."""
    theta = np.deg2rad(22.0)
    t = float(np.tan(np.deg2rad(28.0)))
    orc = _oracle(28.0, theta, "te")
    errs, clos = [], []
    for MA, MB in ((5, 4), (7, 5)):
        _tm, te, clo = _split((t, 0.0), MA, MB, theta)
        errs.append(_perorder(te, orc))
        clos.append(clo)
    assert errs[1] < 0.1 * errs[0], errs
    assert clos[1] < 0.1 * clos[0], clos

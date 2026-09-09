"""Stage B -- OUT-OF-PLANE anisotropic permittivity for the PURE (no-floor)
staggered 2-D PMM: the first-order ``4 q^2`` generator behind
``pmm_jones_2d_staggered`` and the tensor ``PMM2DStackPure``.

Formulation and GO verdict:
``docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`` candidate (a).
Integration measurements, and the table each bar below cites:
``docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md``.

EVERY bar here is DERIVED from a measurement made during this build on
2026-09-09 (Windows 11, py3.14.6, numpy 2.4.4, scipy 1.17.1, OMP=1), stated in
the assertion's comment with its build-doc table row.  Nothing pins a
cross-build value: the comparisons are two-arm, same-build, and the
oracle-referenced bars carry decades of gap on both sides.

Two physics degeneracies shape the design and are load-bearing (S6 / S5 T3 of
the experiment):

* the dispersion relation is invariant under ``eps -> eps^T``, so NO
  eigenvalue or energy measurement can see an ``e13``/``e31`` swap -- the
  gates therefore carry a NON-RECIPROCAL (``e13 = conj(e31)``, Hermitian,
  still lossless) tensor compared against Berreman's FIELDS;
* negating the out-of-plane block is an EXACT symmetry whenever the director
  azimuth equals the incidence azimuth -- so every fail-before control here
  uses a director azimuth (25 deg) DIFFERENT from the incidence azimuth
  (40 deg).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_1d,
    pmm_jones_2d,
)
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _homog_geom_cache,
    _region_modes,
    _region_modes_oop,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402
from lumenairy.elements.rcwa.oned import (  # noqa: E402
    rcwa_jones_1d_segments,
)

# --------------------------------------------------------------------------- #
# fixtures -- grids <= (3,3), M <= 8 (the plan's test-cost rule)
# --------------------------------------------------------------------------- #
_WL = 1.0e-6
_P = 1.2e-6                       # patterned cells: px = py = 1.2 lam
_DEP = 0.4e-6
_PU = 0.9e-6                      # uniform slab: px = py = 0.9 lam
_DEPU = 0.35e-6
_NSUB, _NSUP = 1.5, 1.0

#: tilted-director LC, out of plane.  Director AZIMUTH 25 deg -- deliberately
#: different from the 40-deg conical incidence azimuth used below (S6).
_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
_OOP_LOSSY = uniaxial_tensor(1.5 + 0.02j, 1.7 + 0.02j, np.deg2rad(35.0),
                             phi=np.deg2rad(25.0))
#: NON-RECIPROCAL: Hermitian (hence lossless) but NOT symmetric -- the only
#: tensor that can see an e13 <-> e31 swap (S15).
_NONREC = np.array(_OOP, dtype=complex)
_NONREC[0, 2] = _OOP[0, 2] + 0.22j
_NONREC[2, 0] = np.conj(_NONREC[0, 2])
#: IN-PLANE controls (cross terms EXACTLY zero, not merely small)
_LC = np.array(uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55), dtype=complex)
_LC[0, 2] = _LC[1, 2] = _LC[2, 0] = _LC[2, 1] = 0.0
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_ISO = np.eye(3, dtype=complex)

_CONICAL = (np.deg2rad(25.0), np.deg2rad(40.0))


def _cell(host, pillar, n=2):
    """``(n, n, 3, 3)`` cell: ``pillar`` in segment (0, 0), ``host`` elsewhere."""
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:, :] = host
    c[0, 0] = pillar
    return c


def _uniform(t33, n=2):
    return np.broadcast_to(t33, (n, n, 3, 3)).copy()


def _lcell(pillar, host=_ISO):
    """A (3,3) L -- three pixels sharing a RE-ENTRANT 270-degree corner, and
    NOT 180-degree symmetric up to a lattice shift (which every stripe and
    every single-pixel pillar is; those cells are blind to the orientation
    conventions this file gates)."""
    c = np.empty((3, 3, 3, 3), dtype=complex)
    c[:, :] = host
    for i, j in ((0, 0), (1, 0), (0, 1)):
        c[i, j] = pillar
    return c


def _solve(ec, M, theta=0.0, phi=0.0, n_orders=3, px=_P, dep=_DEP):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d_staggered(px, px, ec, _NSUB, _NSUP, dep, _WL,
                                      degree=M, n_orders=n_orders,
                                      theta=theta, phi=phi)


def _force_oop(flag):
    """Context-free dispatch override: patch the RELATIVE off-plane test the
    assembly reads at call time, so the SAME cell can be driven down BOTH
    branches (the only way to make the reduction a two-arm claim)."""
    orig = TS._tile_is_offplane
    TS._tile_is_offplane = (lambda t: True) if flag else orig
    return orig


def _per_order(o_a, R_a, T_a, o_b, R_b, T_b):
    """``(dR, dT)`` = max over the orders BOTH solves carry."""
    idx = {tuple(int(v) for v in r): j for j, r in enumerate(np.asarray(o_b))}
    dR = dT = 0.0
    for i, r in enumerate(np.asarray(o_a)):
        j = idx.get(tuple(int(v) for v in r))
        if j is None:
            continue
        dR = max(dR, float(np.max(np.abs(np.asarray(R_a)[:, i]
                                         - np.asarray(R_b)[:, j]))))
        dT = max(dT, float(np.max(np.abs(np.asarray(T_a)[:, i]
                                         - np.asarray(T_b)[:, j]))))
    return dR, dT


def _exact_kz_roots(eps, u, v):
    """The four exact ``kz/k0`` roots of ``det(k k^T - |k|^2 I + eps) = 0`` for
    transverse ``(u, v)`` in ``k0`` units, expanded in EXACT polynomial
    arithmetic (each entry is a polynomial of degree <= 2 in ``kz``).  A
    sampled-and-fitted determinant conditions its Vandermonde at ~1e-8, decades
    above the bar this gate needs."""
    from numpy.polynomial.polynomial import polyadd, polymul, polysub
    e = np.asarray(eps, dtype=complex)
    P = np.empty((3, 3), dtype=object)
    P[0, 0] = np.array([e[0, 0] - v * v, 0.0, -1.0], dtype=complex)
    P[0, 1] = np.array([e[0, 1] + u * v], dtype=complex)
    P[0, 2] = np.array([e[0, 2], u], dtype=complex)
    P[1, 0] = np.array([e[1, 0] + u * v], dtype=complex)
    P[1, 1] = np.array([e[1, 1] - u * u, 0.0, -1.0], dtype=complex)
    P[1, 2] = np.array([e[1, 2], v], dtype=complex)
    P[2, 0] = np.array([e[2, 0], u], dtype=complex)
    P[2, 1] = np.array([e[2, 1], v], dtype=complex)
    P[2, 2] = np.array([e[2, 2] - u * u - v * v], dtype=complex)
    det = polyadd(
        polysub(polymul(P[0, 0], polysub(polymul(P[1, 1], P[2, 2]),
                                         polymul(P[1, 2], P[2, 1]))),
                polymul(P[0, 1], polysub(polymul(P[1, 0], P[2, 2]),
                                         polymul(P[1, 2], P[2, 0])))),
        polymul(P[0, 2], polysub(polymul(P[1, 0], P[2, 1]),
                                 polymul(P[1, 1], P[2, 0]))))
    return np.roots(det[::-1])


# --------------------------------------------------------------------------- #
# G1 -- the in-plane reduction, and the dispatch floor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,host,pillar", [
    ("in-plane uniaxial pillar", _ISO * 4.0, _LC),
    ("gyrotropic pillar", _ISO * 4.0, _GYRO),
])
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), _CONICAL])
def test_g1_oop_generator_reduces_to_the_inplane_path(name, host, pillar,
                                                      theta, phi):
    """With the cross terms EXACTLY zero the 4 q^2 generator must reproduce the
    Stage-A 2 q^2 path -- driven on the SAME cell, dispatch forced.

    This is also what pins ``_OOP_H_GAUGE``, the constant relating the
    generator's ``G = i Z0 H`` state to the Eq.-25 partner the ISOTROPIC
    half-spaces carry: a wrong value cancels inside a pure out-of-plane stack
    and does NOT cancel here, so this comparison is its only gate.

    MEASURED 2026-09-09 (build doc T4, probe g2_inplane_reduction.py) over four
    cells x two mounts x M in {5, 6}: the worst R/T/Jones difference is
    4.49e-14 and the worst spectral distance 1.38e-12.  Bars 5e-12 and 1.5e-10
    (x1e2), which sit ~4 decades under the smallest thing the reduction could
    plausibly break -- the O(1e-8) movement an M-step of the discretization
    itself produces on these cells."""
    ec = _cell(host, pillar)
    orig = _force_oop(False)
    try:
        arm_in = _solve(ec, 6, theta=theta, phi=phi)
        _force_oop(True)
        arm_oop = _solve(ec, 6, theta=theta, phi=phi)
    finally:
        TS._tile_is_offplane = orig
    dR = float(np.max(np.abs(arm_in[1] - arm_oop[1])))
    dT = float(np.max(np.abs(arm_in[2] - arm_oop[2])))
    dJ = float(np.max(np.abs(arm_in[3] - arm_oop[3])))
    assert dR < 5e-12 and dT < 5e-12, (name, dR, dT)
    assert dJ < 5e-12, (name, dJ)

    # ... and the SPECTRA: the generator's 4 q^2 eigenvalue set must be the
    # E-form's {+q, -q}, not merely a set giving the same observables.
    k0 = 2.0 * np.pi / _WL
    a0x = _NSUP * np.sin(theta) * np.cos(phi) * k0
    a0y = _NSUP * np.sin(theta) * np.sin(phi) * k0
    sol_in = Granet2DTransverseE(_P, _P, 2, 2, 6, ec, alpha0x=a0x,
                                 alpha0y=a0y, k0=k0)
    orig = _force_oop(True)
    try:
        sol_oop = Granet2DTransverseE(_P, _P, 2, 2, 6, ec, alpha0x=a0x,
                                      alpha0y=a0y, k0=k0)
        _Wf, _Vf, lam_f, _Wb, _Vb, lam_b = _region_modes_oop(sol_oop)
    finally:
        TS._tile_is_offplane = orig
    _W, _V, lam_in, _g2 = _region_modes(sol_in)
    ref = np.concatenate([1j * lam_in, -1j * lam_in])
    got = np.concatenate([1j * lam_f, 1j * lam_b])
    dist = float(np.max(np.min(np.abs(got[:, None] - ref[None, :]), axis=1)))
    assert dist < 1.5e-10, (name, dist)


def test_g1_dispatch_floor_is_relative_and_bit_identical():
    """A physically IN-PLANE cell built by ROTATING a diagonal tensor carries
    ~1e-16 float noise in the xz/yz slots, so the dispatch floor is RELATIVE
    (``1e-12 * scale``, shared verbatim with the hybrid).  Below it the cell
    must stay on the in-plane path and be BIT-IDENTICAL to the clean cell --
    a same-build two-arm claim, no tolerance involved; above it the cell must
    route to the generator.

    MEASURED 2026-09-09 (build doc T4): a 1e-16 stray leaves ``offplane``
    False and R/T/Jones byte-equal; 1e-3 sets it True."""
    base = _cell(_ISO * 4.0, _LC)
    stray = base.copy()
    stray[0, 0, 0, 2] = 1e-16
    big = base.copy()
    big[0, 0, 0, 2] = 1e-3
    k0 = 2.0 * np.pi / _WL
    assert Granet2DTransverseE(_P, _P, 2, 2, 5, stray, k0=k0).offplane is False
    assert Granet2DTransverseE(_P, _P, 2, 2, 5, big, k0=k0).offplane is True
    a = _solve(base, 5)
    b = _solve(stray, 5)
    assert a[1].tobytes() == b[1].tobytes()
    assert a[2].tobytes() == b[2].tobytes()
    assert a[3].tobytes() == b[3].tobytes()


# --------------------------------------------------------------------------- #
# G2 -- the uniform-slab dispersion of the ASSEMBLED generator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,t33", [("tilt35 azim25", _OOP),
                                      ("NON-RECIPROCAL", _NONREC)])
def test_g2_uniform_slab_dispersion_and_sum_of_roots(name, t33):
    """The generator's spectrum must contain the EXACT quartic roots of
    ``det(k k^T - |k|^2 I + eps) = 0`` on the harmonics the basis resolves, at
    normal AND conical incidence, for a symmetric AND a NON-RECIPROCAL tensor.

    The discriminating quantity is the SUM of the fundamental harmonic's four
    roots: it is exactly zero for any in-plane tensor and non-zero only through
    the out-of-plane coupling (the asymmetric extraordinary pair), so a
    generator with the wrong relative ``i`` -- or the wrong rotation gauge --
    cannot produce it.  At NORMAL incidence that sum is zero by symmetry, which
    is why the conical mount is the one that gates.

    MEASURED 2026-09-09 (build doc T3): the fundamental's residual is 1.5e-14
    (conical, M=8) and 2.4e-14 (normal), while the roots for the OPPOSITE
    transverse wavevector -- what a wrong rotation gauge would produce -- sit
    at 9.16e-02 (8.41e-02 non-reciprocal).  Bar 1e-11 (x~500 above the
    measurement, 9 decades below the negative control)."""
    k0 = 2.0 * np.pi / _WL
    cell = _uniform(t33)
    for theta, phi in ((0.0, 0.0), _CONICAL):
        kx0 = _NSUP * np.sin(theta) * np.cos(phi)
        ky0 = _NSUP * np.sin(theta) * np.sin(phi)
        sol = Granet2DTransverseE(_PU, _PU, 2, 2, 8, cell, alpha0x=kx0 * k0,
                                  alpha0y=ky0 * k0, k0=k0)
        _Wf, _Vf, lam_f, _Wb, _Vb, lam_b = _region_modes_oop(sol)
        spec = np.concatenate([1j * lam_f, 1j * lam_b])
        roots = _exact_kz_roots(t33, kx0, ky0)
        d = float(np.max(np.min(np.abs(roots[:, None] - spec[None, :]),
                                axis=1)))
        assert d < 1e-11, (name, theta, d)
        # the sum-of-roots discriminator, and the wrong-gauge control
        wrong = _exact_kz_roots(t33, -kx0, -ky0)
        d_wrong = float(np.max(np.min(np.abs(wrong[:, None] - spec[None, :]),
                                      axis=1)))
        if theta != 0.0:
            # the out-of-plane coupling makes the +/- k_t configurations
            # genuinely different: 9.2e-02 measured, 9 decades above the bar
            assert abs(np.sum(roots)) > 1e-2, (name, np.sum(roots))
            assert d_wrong > 1e-3, (name, d_wrong)
        else:
            assert abs(np.sum(roots)) < 1e-12       # symmetric at k_t = 0


# --------------------------------------------------------------------------- #
# G3 -- Berreman 4x4 on a UNIFORM out-of-plane slab
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,t33", [
    ("lossless tilt35 azim25", _OOP),
    ("NON-RECIPROCAL tilt35 azim25", _NONREC),
    ("lossy tilt35 azim25", _OOP_LOSSY),
])
@pytest.mark.parametrize("mount,theta,phi", [
    ("normal", 0.0, 0.0),
    ("oblique 25", np.deg2rad(25.0), 0.0),
    ("conical 25/40", *_CONICAL),
])
def test_g3_uniform_oop_slab_matches_berreman(name, t33, mount, theta, phi):
    """The exact oracle for a uniform layer at any incidence, on R, T AND the
    reflection Jones -- lossless, LOSSY (whose absorption deficit must be
    reproduced, not merely bounded) and NON-RECIPROCAL (the only case that can
    see an ``e13``/``e31`` swap).

    MEASURED 2026-09-09 (build doc T2, probe g1_rot_sign_berreman.py) at M=8:
    worst dR 1.99e-14, dT 1.32e-13, dJones 8.54e-14 over all nine
    tensor x mount combinations, with the order leak at 1e-26 (a uniform cell
    puts NOTHING in a non-zero diffraction order).  Bars 1e-11 / 1e-11 --
    ~2 decades above the measurement and 7 decades below the fail-before
    controls of ``test_g7_*`` (1.4e-04 .. 4.5e-03)."""
    Rb, Tb, Jrb, _Jt = berreman_jones_1d([(t33, _DEPU)], _NSUB, _NSUP, _WL,
                                         angle=theta, phi=phi)
    o, R, T, J = _solve(_uniform(t33), 8, theta=theta, phi=phi, px=_PU,
                        dep=_DEPU)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert float(np.max(np.abs(R.sum(axis=1) - Rb))) < 1e-11
    assert float(np.max(np.abs(T.sum(axis=1) - Tb))) < 1e-11
    assert float(np.max(np.abs(J - Jrb))) < 1e-11
    leak = float(np.max(np.abs(np.delete(R, p0, axis=1)))
                 + np.max(np.abs(np.delete(T, p0, axis=1))))
    assert leak < 1e-20, leak      # measured 1e-26


def test_g3_berreman_convergence_is_two_sided_in_m():
    """Not just "small at M=8": the residual must DROP with M, which is what
    separates a converging discretization from a coincidence.  A UNIFORM region
    at oblique is degree-limited in this basis (the Bloch phase must be
    resolved), so this is the ladder that shows it converging spectrally.

    MEASURED 2026-09-09 (build doc T2), conical 25/40, non-reciprocal tensor:
    dR 2.38e-13 (M=6) -> 2.06e-14 (M=8); oblique 25, lossless: 7.93e-12 ->
    7.97e-15, three decades over two degrees.  The claim asserted is the DROP
    (with a x0.5 margin so a build's last bits cannot flip it) plus the M=8
    value under 1e-11."""
    theta, phi = np.deg2rad(25.0), 0.0
    Rb, _Tb, Jrb, _Jt = berreman_jones_1d([(_OOP, _DEPU)], _NSUB, _NSUP, _WL,
                                          angle=theta, phi=phi)
    res = []
    for M in (5, 6, 8):
        _o, R, _T, J = _solve(_uniform(_OOP), M, theta=theta, phi=phi,
                              px=_PU, dep=_DEPU)
        res.append(max(float(np.max(np.abs(R.sum(axis=1) - Rb))),
                       float(np.max(np.abs(J - Jrb)))))
    assert res[1] < 0.5 * res[0], res      # measured 2.0e-08 -> 1.9e-11
    assert res[2] < 0.5 * res[1], res      # measured 1.9e-11 -> 8.0e-15
    assert res[2] < 1e-11, res


# --------------------------------------------------------------------------- #
# G4 -- y-uniform out-of-plane STRIPE against the 1-D engines, per order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mount,theta", [("normal", 0.0),
                                         ("oblique 25", np.deg2rad(25.0))])
def test_g4_oop_stripe_matches_the_1d_engines_per_order(mount, theta):
    """A y-uniform out-of-plane stripe is a cell BOTH 1-D engines can solve
    exactly, so the comparison is per order and both polarizations against two
    INDEPENDENT engines whose own mutual spread is the bar.

    MEASURED 2026-09-09 (build doc T5): ``rcwa_jones_1d_segments(41)`` and
    ``pmm_jones_1d(degree=18)`` differ from each other by dR 3.2e-06 /
    dT 3.8e-05 / dJones 7.3e-06 (normal) and 3.3e-06 / 3.2e-05 / 1.3e-05
    (oblique).  The staggered arm reaches dR 1.59e-05 / dT 1.04e-04 at M=8
    (normal) and 7.4e-06 / 1.15e-04 (oblique), falling monotonically from
    1.2e-04 / 8.3e-04 at M=5 -- this air-groove cell has four DIELECTRIC
    CORNERS, where the shipped basis converges ALGEBRAICALLY, so the floor is
    the module's documented corner cap and NOT an out-of-plane defect (the
    same cell with the cross terms zeroed caps the same way).  Bar 3x the
    measured M=8 residual: 5e-04 on R/T/Jones -- deterministic discretization
    error, whose cross-build spread is ~1e-12."""
    ridge, groove = _OOP, _ISO
    ec = np.zeros((2, 2, 3, 3), dtype=complex)
    ec[0, :] = ridge
    ec[1, :] = groove
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_r, R_r, T_r, J_r = rcwa_jones_1d_segments(
            _P, [(0.5, ridge), (0.5, groove)], _NSUB, _NSUP, _DEP, _WL,
            n_orders=31, theta=theta)
    o_r = np.asarray(o_r)
    ir = {int(m): j for j, m in enumerate(o_r)}
    o_s, R_s, T_s, J_s = _solve(ec, 8, theta=theta, phi=0.0)
    o_s = np.asarray(o_s)
    common = [m for m in range(-3, 4) if m in ir]
    for m in common:
        i = int(np.where((o_s[:, 0] == m) & (o_s[:, 1] == 0))[0][0])
        assert float(np.max(np.abs(R_s[:, i] - R_r[:, ir[m]]))) < 5e-4, m
        assert float(np.max(np.abs(T_s[:, i] - T_r[:, ir[m]]))) < 5e-4, m
    assert float(np.max(np.abs(J_s - J_r))) < 5e-4

    # Y-MOMENTUM: a y-invariant cell scatters NOTHING into (m, n != 0).  The
    # out-of-plane blocks A23/A32 carry a y-derivative and a y-mixed mass, and
    # a mis-placed one breaks exactly this.  MEASURED 1e-27 .. 1e-26 (T5);
    # bar 1e-20, which is 6 decades above the measurement and 20 below any
    # physical leak.
    sel = o_s[:, 1] != 0
    assert float(np.max(np.abs(R_s[:, sel]))) < 1e-20
    assert float(np.max(np.abs(T_s[:, sel]))) < 1e-20


def test_g4_stripe_agrees_with_the_second_1d_engine_too():
    """The second independent 1-D engine (``pmm_jones_1d``, a spectral-element
    solver rather than a Fourier one), so the G4 claim does not rest on one
    oracle family.  Same cell, same bar, normal incidence only (cost)."""
    ridge, groove = _OOP, _ISO
    ec = np.zeros((2, 2, 3, 3), dtype=complex)
    ec[0, :] = ridge
    ec[1, :] = groove
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_p, R_p, T_p, J_p = pmm_jones_1d(_P, ridge, groove, _NSUB, _NSUP,
                                          _DEP, 0.5, _WL, theta=0.0,
                                          degree=18, far_field_orders=21,
                                          stabilize=False)
    o_p = np.asarray(o_p)
    ip = {int(m): j for j, m in enumerate(o_p)}
    o_s, R_s, T_s, J_s = _solve(ec, 8)
    o_s = np.asarray(o_s)
    for m in [m for m in range(-3, 4) if m in ip]:
        i = int(np.where((o_s[:, 0] == m) & (o_s[:, 1] == 0))[0][0])
        assert float(np.max(np.abs(R_s[:, i] - R_p[:, ip[m]]))) < 5e-4, m
        assert float(np.max(np.abs(T_s[:, i] - T_p[:, ip[m]]))) < 5e-4, m


# --------------------------------------------------------------------------- #
# G5 -- a genuinely 2-D out-of-plane cell, PER ORDER, and the no-floor property
# --------------------------------------------------------------------------- #
def test_g5_2d_reentrant_corner_cell_matches_the_fourier_oracles():
    """The (3,3) L cell -- a RE-ENTRANT 270-degree corner, and the only shape
    in this file that is NOT 180-degree symmetric up to a lattice shift, so it
    is the only one that can see the assembly's orientation conventions at all.

    This is the gate GATE 0 of the integration brief demanded.  MEASURED
    2026-09-09 (build doc T6): the two Fourier oracles' own mutual spread on
    this cell is dR 4.84e-05 (``pmm_jones_2d(13, laurent)`` vs
    ``rcwa_jones_2d(9)``) and 6.40e-05 (its 'li' rule), and ``rcwa``'s own
    7 -> 9 movement is 3.46e-05 -- neither Fourier arm is converged here, a
    re-entrant corner being the slowest-converging feature a Fourier basis has.
    The staggered arm sits AT that spread: dR 6.62e-05 (M=5), 7.24e-05 (M=6),
    7.58e-05 (M=7), self-moving only 9.3e-06 over the last step while its own
    energy closure falls to 5.7e-10.  Bar 4e-04 on R -- 5x the measurement and
    5x the oracle spread, and 1.5 decades under the 4.5e-03 the WRONG
    orientation convention produced on this same cell (the prototype's open
    item, probe g0c)."""
    ec = _lcell(_OOP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine = np.repeat(np.repeat(ec, 11, axis=0), 11, axis=1)
        o_r, R_r, T_r, J_r = rcwa_jones_2d(_P, _P, fine, _NSUB, _NSUP, _DEP,
                                           _WL, n_orders_x=7, n_orders_y=7)
    o_s, R_s, T_s, J_s = _solve(ec, 6)
    dR, dT = _per_order(o_s, R_s, T_s, o_r, R_r, T_r)
    assert dR < 4e-4, dR
    assert dT < 4e-3, dT           # measured 1.17e-03 (T carries the bulk)
    assert float(np.max(np.abs(J_s - J_r))) < 4e-3


def test_g5_no_fourier_floor_two_sided():
    """The property the pure solver exists for, kept by the out-of-plane path:
    the staggered result does not depend on the far-field order count, and the
    hybrid's does.

    MEASURED 2026-09-09 (build doc T6): on the (3,3) L cell the staggered arm
    moves 2.55e-15 (normal) and 3.94e-09 (conical) when ``n_orders`` goes
    3 -> 8, while the hybrid's two ``E_z`` elimination rules -- both rigorous,
    differing only in the Fourier factorization -- disagree by 7.73e-04 at
    ``n_orders = 13``.  Eleven decades of contrast.  Bars: staggered < 1e-10,
    hybrid > 1e-5, with the hybrid arm measured on THIS build so the claim is
    not a pin on a remembered number."""
    ec = _lcell(_OOP)
    a = _solve(ec, 6, n_orders=3)
    b = _solve(ec, 6, n_orders=8)
    moved = max(_per_order(a[0], a[1], a[2], b[0], b[1], b[2]))
    assert moved < 1e-10, moved
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h7 = pmm_jones_2d(_P, _P, ec, _NSUB, _NSUP, _DEP, _WL, degree=7,
                          n_orders=7, stabilize=True)
        h9 = pmm_jones_2d(_P, _P, ec, _NSUB, _NSUP, _DEP, _WL, degree=7,
                          n_orders=9, stabilize=True)
    hyb_moved = max(_per_order(h7[0], h7[1], h7[2], h9[0], h9[1], h9[2]))
    assert hyb_moved > 1e-5, hyb_moved
    assert hyb_moved > 1e4 * moved, (moved, hyb_moved)


# --------------------------------------------------------------------------- #
# G6 -- cascade closure, two-sided, over a DEPTH ladder
# --------------------------------------------------------------------------- #
def test_g6_hermitian_oop_closes_at_three_depths_and_does_not_grow():
    """A mis-classified growing mode does not show at one depth -- it shows as
    ``exp(+|Re lam| k0 L)``.  So this is a DEPTH ladder (0.25, 1 and 3
    wavelengths) on a Hermitian (hence exactly lossless) out-of-plane tensor,
    plus the forward-set growth factor, plus the split count.

    MEASURED 2026-09-09 (build doc T7): the uniform Hermitian cell closes to
    6.8e-14 / 8.3e-14 / 2.1e-13 (normal) and 8.8e-14 / 1.3e-13 / 2.2e-13
    (conical) -- flat in depth, which is the whole point -- with the split
    exactly 2q^2 / 2q^2 (288/288 at M=7) and ``max exp(-Re(lam_f) k0 L)``
    exactly 1.0000e+00 at 3 wavelengths.  Bars: closure 1e-10 (~3 decades
    above the measurement, ~7 below the 1e-3 an out-of-plane sign error
    produces) and growth <= 1 + 1e-12."""
    cell = _uniform(_OOP)
    k0 = 2.0 * np.pi / _WL
    for theta, phi in ((0.0, 0.0), _CONICAL):
        for dep_lam in (0.25, 1.0, 3.0):
            _o, R, T, _J = _solve(cell, 7, theta=theta, phi=phi, px=_P,
                                  dep=dep_lam * _WL)
            tot = R.sum(axis=1) + T.sum(axis=1)
            assert float(np.max(np.abs(tot - 1.0))) < 1e-10, (theta, dep_lam,
                                                              tot)
        sol = Granet2DTransverseE(
            _P, _P, 2, 2, 7, cell,
            alpha0x=_NSUP * np.sin(theta) * np.cos(phi) * k0,
            alpha0y=_NSUP * np.sin(theta) * np.sin(phi) * k0, k0=k0)
        _Wf, _Vf, lam_f, _Wb, _Vb, lam_b = _region_modes_oop(sol)
        q = sol.q * sol.q
        assert lam_f.size == 2 * q and lam_b.size == 2 * q
        grow = float(np.max(np.exp(-np.real(lam_f) * k0 * 3.0 * _WL)))
        assert grow <= 1.0 + 1e-12, grow


def test_g6_lossy_oop_closes_below_one_and_absorbs_monotonically():
    """The other side of the two-sided claim: a NON-Hermitian tensor must close
    BELOW 1, by a margin, and its absorption must rise with depth (an unstable
    cascade shows as a non-monotone or out-of-range deficit).

    MEASURED 2026-09-09 (build doc T7): the lossy pillar absorbs 2.38e-02 /
    1.61e-01 / 4.72e-01 at 0.25 / 1 / 3 wavelengths (normal) and 2.95e-02 /
    1.68e-01 / 4.34e-01 (conical), with the forward growth factor 8.68e-01 and
    8.89e-01 -- strictly below 1 at every depth."""
    cell = _cell(_ISO, _OOP_LOSSY)
    absorbed = []
    for dep_lam in (0.25, 1.0, 3.0):
        _o, R, T, _J = _solve(cell, 6, dep=dep_lam * _WL)
        absorbed.append(float(np.min(1.0 - R.sum(axis=1) - T.sum(axis=1))))
    assert 0.0 < absorbed[0] < absorbed[1] < absorbed[2] < 1.0, absorbed
    assert absorbed[0] > 1e-3, absorbed        # a real deficit, not roundoff


# --------------------------------------------------------------------------- #
# G7 -- fail-before controls: every out-of-plane term is load-bearing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("control", ["drop", "negate", "transpose"])
def test_g7_oop_block_controls_move_the_berreman_residual(control):
    """Each mis-assembly of the out-of-plane block must move the Berreman
    residual by the decades the prototype measured.  The tensor is
    NON-RECIPROCAL and the director azimuth (25 deg) differs from the incidence
    azimuth (40 deg) -- WITHOUT both of those, two of these three controls are
    exact symmetries and the gate silently asserts nothing (S6).

    MEASURED 2026-09-09 (build doc T9), conical 25/40, M=8, non-reciprocal:
    reference dR 2.06e-14 / dJones 8.54e-14; drop 1.04e-03 / 3.22e-03; negate
    1.68e-03 / 4.49e-03; transpose 1.67e-03 / 4.49e-03.  Bar: the control must
    exceed 1e-4 on dR -- one decade under the smallest control and 10 decades
    above the reference."""
    t33 = np.array(_NONREC, dtype=complex)
    if control == "drop":
        t33[0, 2] = t33[1, 2] = t33[2, 0] = t33[2, 1] = 0.0
    elif control == "negate":
        t33[0, 2] *= -1
        t33[1, 2] *= -1
        t33[2, 0] *= -1
        t33[2, 1] *= -1
    else:
        t33[0, 2], t33[2, 0] = _NONREC[2, 0], _NONREC[0, 2]
        t33[1, 2], t33[2, 1] = _NONREC[2, 1], _NONREC[1, 2]
    theta, phi = _CONICAL
    Rb, _Tb, Jrb, _Jt = berreman_jones_1d([(_NONREC, _DEPU)], _NSUB, _NSUP,
                                          _WL, angle=theta, phi=phi)
    _o, R, _T, J = _solve(_uniform(t33), 6, theta=theta, phi=phi, px=_PU,
                          dep=_DEPU)
    dR = float(np.max(np.abs(R.sum(axis=1) - Rb)))
    dJ = float(np.max(np.abs(J - Jrb)))
    assert dR > 1e-4, (control, dR)
    assert dJ > 1e-4, (control, dJ)


def test_g7_rotation_gauge_is_two_sided():
    """``_OOP_ROT_SIGN`` is the 180-degree rotation the shipped basis carries.
    It is unobservable at NORMAL incidence on a uniform cell (both signs give
    the same answer -- the rotation maps that configuration onto itself), and
    it is decisive at CONICAL: exactly the asymmetry this claim needs.

    MEASURED 2026-09-09 (build doc T2, probe g1_rot_sign_berreman.py), M=6,
    non-reciprocal tensor: at conical 25/40 the shipped -1 gives dR 2.38e-13
    and the flipped +1 gives 1.68e-03; at normal both give ~1e-14.  This drives
    the flipped sign through the SHIPPED code rather than re-deriving it, so
    the negative arm is a property of the build under test."""
    theta, phi = _CONICAL
    Rb, _Tb, Jrb, _Jt = berreman_jones_1d([(_NONREC, _DEPU)], _NSUB, _NSUP,
                                          _WL, angle=theta, phi=phi)
    cell = _uniform(_NONREC)
    orig = TS._OOP_ROT_SIGN
    try:
        _o, R, _T, J = _solve(cell, 6, theta=theta, phi=phi, px=_PU,
                              dep=_DEPU)
        good = max(float(np.max(np.abs(R.sum(axis=1) - Rb))),
                   float(np.max(np.abs(J - Jrb))))
        TS._OOP_ROT_SIGN = -orig
        _o, R2, _T2, J2 = _solve(cell, 6, theta=theta, phi=phi, px=_PU,
                                 dep=_DEPU)
        bad = max(float(np.max(np.abs(R2.sum(axis=1) - Rb))),
                  float(np.max(np.abs(J2 - Jrb))))
        # normal incidence: BOTH signs must agree (the rotation is a symmetry
        # of a uniform cell at k_t = 0 -- so this axis cannot gate the sign,
        # which is why the conical mount is the one that does)
        _o, R3, _T3, J3 = _solve(cell, 6, px=_PU, dep=_DEPU)
        TS._OOP_ROT_SIGN = orig
        _o, R4, _T4, J4 = _solve(cell, 6, px=_PU, dep=_DEPU)
    finally:
        TS._OOP_ROT_SIGN = orig
    assert good < 1e-11, good
    assert bad > 1e-4, bad
    assert bad > 1e7 * good, (good, bad)
    assert float(np.max(np.abs(R3 - R4))) < 1e-11
    assert float(np.max(np.abs(J3 - J4))) < 1e-11


# --------------------------------------------------------------------------- #
# G8 -- stacks
# --------------------------------------------------------------------------- #
def test_g8a_one_oop_layer_equals_two_half_thickness_layers():
    """The consistency identity available for a mixed cascade: one out-of-plane
    layer of depth ``d`` is the same physics as two of ``d/2``, so the
    generalized interface (a genuine A|A modal match, not a trivial one) and
    the generalized propagation must compose exactly.

    MEASURED 2026-09-09 (build doc T8a): per-order 8.88e-16 (normal) and
    3.33e-16 (conical), Jones 4.06e-16 / 2.81e-16.  Bar 1e-11 -- five decades
    above the measurement, and far below any physical difference."""
    ec = _cell(_ISO, _OOP)
    for theta, phi in ((0.0, 0.0), (np.deg2rad(20.0), np.deg2rad(35.0))):
        one = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                             n_modes=6, n_orders=3)
        one.add_layer(_DEP, eps_cell=ec).set_source(_WL, theta=theta, phi=phi)
        two = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                             n_modes=6, n_orders=3)
        two.add_layer(_DEP / 2, eps_cell=ec).add_layer(_DEP / 2, eps_cell=ec)
        two.set_source(_WL, theta=theta, phi=phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = one.solve(jones=True)
            b = two.solve(jones=True)
        assert max(_per_order(a[0], a[1], a[2], b[0], b[1], b[2])) < 1e-11
        assert float(np.max(np.abs(a[3] - b[3]))) < 1e-11


def test_g8b_uniform_oop_multilayer_matches_berreman_multilayer():
    """Three UNIFORM out-of-plane layers -- lossless, NON-RECIPROCAL and lossy
    -- against the exact Berreman multilayer at oblique and conical incidence.
    A uniform region is degree-limited in this basis, so the claim is two-sided
    in M as in G3.

    MEASURED 2026-09-09 (build doc T8b): dR 8.26e-12 (M=6) -> 7.47e-15 (M=8) at
    oblique 25 and 3.06e-13 -> 1.25e-14 at conical 25/40, with dJones
    2.34e-11 -> 3.10e-14 and 1.25e-12 -> 6.50e-14.  Bar 1e-11 at M=8."""
    layers = [(_OOP, 0.30e-6), (_NONREC, 0.22e-6), (_OOP_LOSSY, 0.17e-6)]
    for theta, phi in ((np.deg2rad(25.0), 0.0), _CONICAL):
        Rb, Tb, Jrb, _Jt = berreman_jones_1d(layers, _NSUB, _NSUP, _WL,
                                             angle=theta, phi=phi)
        st = PMM2DStackPure(_PU, _PU, n_superstrate=_NSUP, n_substrate=_NSUB,
                            n_modes=8, n_orders=3)
        for t33, d in layers:
            st.add_layer(d, eps=t33)
        st.set_source(_WL, theta=theta, phi=phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, J = st.solve(jones=True)
        assert float(np.max(np.abs(R.sum(axis=1) - Rb))) < 1e-11
        assert float(np.max(np.abs(T.sum(axis=1) - Tb))) < 1e-11
        assert float(np.max(np.abs(J - Jrb))) < 1e-11


def test_g8c_layer_absorption_closes_on_a_mixed_oop_stack():
    """``retain_internal`` / :meth:`layer_absorption` on the generalized
    cascade -- the C3 contract ``sum_i A_i == 1 - sum R - sum T``, on a stack
    mixing a LOSSY out-of-plane layer with an in-plane one.  A cross-machinery
    check: the per-layer numbers come from the internal Gram flux, the deficit
    from the Rayleigh far field.

    MEASURED 2026-09-09 (build doc T8d): the residual falls 9.14e-04 (M=5) ->
    7.62e-05 (M=6) -> 4.14e-06 (M=7) -> 2.09e-07 (M=8) against an absorbed
    fraction of 3.6e-02, i.e. it is DISCRETIZATION-limited, not a formula
    error -- so the claim asserted is the drop plus the M=7 value.  On a
    LOSSLESS mixed stack every per-layer absorption is <= 5.3e-14 (T8c)."""
    ec_o = _cell(_ISO, _OOP_LOSSY)
    ec_i = _cell(_ISO, _LC)
    theta, phi = np.deg2rad(20.0), np.deg2rad(35.0)
    res = []
    for M in (5, 7):
        st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                            n_modes=M, n_orders=3)
        st.add_layer(0.30e-6, eps_cell=ec_o).add_layer(0.20e-6, eps_cell=ec_i)
        st.set_source(_WL, theta=theta, phi=phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = st.solve(jones=True, retain_internal=True)
            A = st.layer_absorption()
        deficit = 1.0 - R.sum(axis=1) - T.sum(axis=1)
        assert float(np.min(deficit)) > 1e-2, deficit    # a real absorber
        res.append(float(np.max(np.abs(A.sum(axis=0) - deficit))))
    assert res[1] < 0.05 * res[0], res       # measured 9.1e-04 -> 4.1e-06
    assert res[1] < 1e-4, res


def test_g8c_lossless_mixed_stack_absorbs_nothing():
    """The other side: a LOSSLESS mixed (out-of-plane | in-plane | uniform)
    stack must have every per-layer absorption at roundoff, and its closure
    must be the discretization's alone.

    MEASURED 2026-09-09 (build doc T8c): max per-layer |A_i| = 8.44e-15 at
    M=6 (5.3e-14 at M=8), closure 7.29e-05 at M=6."""
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=6, n_orders=3)
    st.add_layer(0.25e-6, eps_cell=_cell(_ISO, _OOP))
    st.add_layer(0.20e-6, eps_cell=_cell(_ISO, _LC))
    st.add_layer(0.15e-6, eps=2.25)
    st.set_source(_WL, theta=np.deg2rad(20.0), phi=np.deg2rad(35.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, _J = st.solve(jones=True, retain_internal=True)
        A = st.layer_absorption()
    assert float(np.max(np.abs(A))) < 1e-11
    assert float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1))) < 1e-3


# --------------------------------------------------------------------------- #
# G9 -- guards: what must still refuse, and refuse loudly
# --------------------------------------------------------------------------- #
def test_g9_zero_ezz_still_raises_on_the_oop_path():
    """``e33 == 0`` breaks BOTH ``E_z`` eliminations (the in-plane Schur solve
    against ``Meps33`` and the out-of-plane ``A33`` solve), so it must raise on
    either branch and name the reason."""
    bad = _cell(_ISO, _OOP)
    bad[0, 0, 2, 2] = 0.0
    with pytest.raises(ValueError, match="e_zz"):
        pmm_jones_2d_staggered(_P, _P, bad, _NSUB, _NSUP, _DEP, _WL, degree=5)
    with pytest.raises(ValueError, match="e_zz"):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps_cell=bad)
    bad_u = np.array(_OOP, dtype=complex)
    bad_u[2, 2] = 0.0
    with pytest.raises(ValueError, match="e_zz"):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps=bad_u)


def test_g9_mode_entries_refuse_the_wrong_assembly():
    """The two region-mode entries are NOT interchangeable -- an out-of-plane
    assembly has no second-order pencil and distinct forward/backward sets --
    so each must refuse the other's solver loudly rather than return a
    silently wrong basis.  Likewise the shared eps-free geometric eig."""
    k0 = 2.0 * np.pi / _WL
    sol_oop = Granet2DTransverseE(_P, _P, 2, 2, 4, _uniform(_OOP), k0=k0)
    sol_in = Granet2DTransverseE(_P, _P, 2, 2, 4, _uniform(_LC), k0=k0)
    assert sol_oop.offplane is True and sol_in.offplane is False
    with pytest.raises(ValueError, match="_region_modes_oop"):
        _region_modes(sol_oop)
    with pytest.raises(ValueError, match="_region_modes"):
        _region_modes_oop(sol_in)
    with pytest.raises(ValueError, match="uniform SCALAR region"):
        _homog_geom_cache(sol_oop)
    # the out-of-plane assembly must not leave stale second-order operators
    assert sol_oop.Lmat is None and sol_oop.Rmat is None
    assert sol_oop.Stt is None and sol_oop.Schur is None
    assert sol_oop.dimtot == 4 * sol_oop.q ** 2


def test_g9_scalar_entry_and_shapes_still_refuse():
    """The scalar efficiency entry stays scalar-only, and the shape / square
    grid guards are unchanged by the out-of-plane branch."""
    from lumenairy.elements.pmm.twod_staggered import (
        pmm_efficiency_2d_staggered,
    )
    with pytest.raises(ValueError, match="pmm_jones_2d_staggered"):
        pmm_efficiency_2d_staggered(_P, _P, _cell(_ISO, _OOP), _NSUB, _NSUP,
                                    _DEP, _WL, degree=5)
    with pytest.raises(ValueError, match=r"Nx, Ny, 3, 3"):
        pmm_jones_2d_staggered(_P, _P, np.ones((2, 2, 2, 2), complex), _NSUB,
                               _NSUP, _DEP, _WL, degree=5)
    with pytest.raises(ValueError, match="eps_cell must be SQUARE"):
        pmm_jones_2d_staggered(_P, _P, _cell(_ISO, _OOP)[:1], _NSUB, _NSUP,
                               _DEP, _WL, degree=5)

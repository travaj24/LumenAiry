"""MAGNETIC (permeability-tensor) anisotropy for the PURE (no-floor) staggered
2-D PMM: ``pmm_jones_2d_staggered(..., mu_cell=)`` and
``PMM2DStackPure.add_layer(..., mu=/mu_cell=)``.

Gates G1-G7 of the magnetic build.  EVERY bar below is DERIVED from a
measurement made during this build on 2026-09-10 (Windows 11, py3.14, numpy
2.x, OMP/OPENBLAS/MKL = 1); the measurement and its table row in
``docs/audits/BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`` are cited in each
assertion's comment.  Nothing here pins a cross-build value: every comparison
is two-arm and same-build, and the oracle-referenced bars carry decades of gap
on both sides.

Formulation: Granet, J. Opt. Soc. Am. A 40, 652 (2023).  The paper's general
equations already carry a permeability through ``chi_t = [mu_t]^-1``:
``R = C[chi_t]C`` (Eq. 24 / A39), ``K_tz = C[chi_t][d2; -d1]`` (Eq. 21 / A43)
and ``S_tt`` weighted by ``chi33`` (Eq. 20 / A42).  The shipped solver
implemented their ``chi_t = I``, ``chi33 = 1`` reduction.  The paper uses
``exp(+i w t)``; this module is PUBLIC ``exp(-i w t)`` end to end.

The two independent physics oracles used here need no external engine:

* the ANALYTIC Airy / characteristic-matrix formula with the wave impedance
  ``Z = sqrt(mu/eps)`` for a uniform slab (G2), whose CONVENTION is pinned on
  the nonmagnetic arm before it is trusted on a magnetic one, and
* electromagnetic DUALITY ``(E, H, eps, mu) -> (Z0 H, -E/Z0, mu, eps)`` (G3,
  G4), exact for self-dual (vacuum) half-spaces.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_1d,
)
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _homog_geom_cache,
    _region_modes,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_1d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

# --------------------------------------------------------------------------- #
# fixtures (grids <= (3,3), M <= 8 -- the test-cost rule)
# --------------------------------------------------------------------------- #
_WL, _P, _DEP = 0.55e-6, 0.70e-6, 0.28e-6
_EYE = np.eye(3, dtype=complex)
_LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_LC2 = uniaxial_tensor(1.15, 1.35, np.pi / 2, phi=-0.30)
#: HERMITIAN gyrotropic PERMEABILITY (m12 = -m21 = +0.4i in the PUBLIC gauge):
#: lossless, and the discriminator for the C-rotated mixed blocks
_MU_GYRO = np.array([[1.6, 0.4j, 0.0], [-0.4j, 1.6, 0.0], [0.0, 0.0, 1.2]],
                    dtype=complex)
#: LOSSY permeability: Im(m11) > 0 is an anti-Hermitian part = absorption
_MU_LOSSY = np.array([[1.6 + 0.25j, 0.0, 0.0], [0.0, 1.6, 0.0],
                      [0.0, 0.0, 1.2]], dtype=complex)
_S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
_S2 = np.array([[0, 1], [1, 0]], dtype=float)


def _cell(host, incl, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = incl
    return c


def _uni(t33, n=2):
    return np.ascontiguousarray(np.broadcast_to(t33, (n, n, 3, 3)))


def _transpose_cell(A):
    B = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = _S3 @ A[j, i] @ _S3
    return B


def _idx(orders):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(orders))}


# =========================================================================== #
# G1 -- the nonmagnetic path is UNTOUCHED, and mu = 1 through the magnetic
#       path reproduces it.
# =========================================================================== #
def test_g1_mu_none_dispatch_is_bit_identical():
    """``mu_cell=None`` must be a DISPATCH, not a rewrite: the assembly must
    take no magnetic branch and retain nothing extra.

    Two arms, one build: the shipped call signature (no ``mu_cell`` at all)
    and the explicit ``mu_cell=None``.  MEASURED 2026-09-10 (build doc M1):
    every retained operator is EXACTLY equal (max|diff| = 0.0) and
    ``Ggram_blocks is None`` / ``magnetic is False`` on both, so the
    nonmagnetic solver retains not one byte more than before.
    """
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    kw = dict(alpha0x=0.31, alpha0y=-0.17, k0=2 * np.pi / 0.62)
    a = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw)
    b = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, mu_cell=None, **kw)
    for name in ("Lmat", "Rmat", "Stt", "Schur"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for k in (0, 1):
        assert np.array_equal(a.Et_blocks[k], b.Et_blocks[k]), k
    for s in (a, b):
        assert s.Ggram_blocks is None
        assert s.magnetic is False
        assert s.mu_cell is None
        for dead in ("Curl", "Kzt", "Ktz", "G3", "Meps33"):
            assert not hasattr(s, dead), dead


def test_g1_unit_mu_forced_through_the_magnetic_path():
    """``mu = 1`` driven through the MAGNETIC assembly must reproduce the
    nonmagnetic one -- the reduction that proves the new weights are the
    paper's general form of the shipped ones and not a different operator.

    MEASURED 2026-09-10 (build doc M1), (2,2) M=6 at oblique Bloch phases,
    relative max|diff| between the two arms:

        Lmat 7.95e-15   Rmat 3.28e-17   Stt 4.62e-14   Schur 3.23e-16
        eigenvalue SET (nearest-neighbour) 6.15e-14 relative

    and through the public entry at M=5 on a uniform eps=4 slab: dR 1.41e-15,
    dT 1.33e-15, dJones 8.44e-15.  These are re-summation differences (the
    weighted assembly sums the per-cell krons in a different order), so the
    bar is 1e-12: 16x over the worst measurement and 3.7 decades under the
    smallest genuinely magnetic signal in this file (the M=5 oblique
    discretization residual, 5.2e-09, table M3).
    """
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    kw = dict(alpha0x=0.31, alpha0y=-0.17, k0=2 * np.pi / 0.62)
    a = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw)
    ones = np.ones((2, 2), dtype=complex)
    for mu in (ones, _uni(_EYE)):
        b = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, mu_cell=mu, **kw)
        assert b.magnetic is True
        assert b.Ggram_blocks is not None
        for name in ("Lmat", "Rmat", "Stt", "Schur"):
            A, B = getattr(a, name), getattr(b, name)
            rel = np.max(np.abs(A - B)) / max(float(np.max(np.abs(A))), 1e-30)
            assert rel < 1e-12, (name, rel)
        g2a, g2b = _region_modes(a)[3], _region_modes(b)[3]
        near = np.min(np.abs(g2a[:, None] - g2b[None, :]), axis=1)
        assert np.max(near) / np.max(np.abs(g2a)) < 1e-12
    ec = _uni(4.0 * _EYE)
    o, R, T, J = pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL,
                                        degree=5, n_orders=2)
    o2, R2, T2, J2 = pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL,
                                            mu_cell=ones, degree=5, n_orders=2)
    assert np.max(np.abs(R - R2)) < 1e-12
    assert np.max(np.abs(T - T2)) < 1e-12
    assert np.max(np.abs(J - J2)) < 1e-12


# =========================================================================== #
# G2 -- a uniform ISOTROPIC MAGNETIC slab against the ANALYTIC Airy formula
#       with the wave impedance.
# =========================================================================== #
def _airy(eps, mu, th0, t_over_wl, pol):
    """(R, T, r) of a uniform slab between VACUUM half-spaces, in the module's
    PUBLIC ``exp(-i w t)`` gauge.

        kz   = sqrt(eps*mu - sin^2 th0)          (Im >= 0)
        Y    = kz/mu   (TE)  |  eps/kz  (TM)     relative admittance
        Y0   = cos th0 (TE)  |  1/cos th0 (TM)   vacuum
        M    = [[cos d, -i sin d / Y], [-i Y sin d, cos d]] ,  d = k0 t kz
        [B; C] = M [1; Y0] ;  r = (Y0 B - C)/(Y0 B + C) ;  t = 2 Y0/(Y0 B + C)

    The MINUS signs are the convention bridge: the textbook (+i) matrix belongs
    to ``exp(+i w t)``, where ``Im(eps) > 0`` is GAIN.  The two forms agree for
    real eps and mu, so only a LOSSY arm can pin it -- MEASURED 2026-09-10
    (build doc M2): with (+i) a lossy slab reads dT = 4.4 (it amplifies), with
    (-i) 1e-15.  ``test_g2_the_analytic_oracle_matches_the_nonmagnetic_path``
    below pins the whole formula on the already-validated nonmagnetic arm
    BEFORE any magnetic arm uses it."""
    eps, mu = complex(eps), complex(mu)
    kz = np.sqrt(complex(eps * mu - np.sin(th0) ** 2))
    if kz.imag < 0:
        kz = -kz
    c0 = np.cos(th0)
    Y, Y0 = ((kz / mu, complex(c0)) if pol == "te"
             else (eps / kz, complex(1.0 / c0)))
    d = 2.0 * np.pi * t_over_wl * kz
    M = np.array([[np.cos(d), -1j * np.sin(d) / Y],
                  [-1j * Y * np.sin(d), np.cos(d)]], dtype=complex)
    B, C = M @ np.array([1.0, Y0], dtype=complex)
    r = (Y0 * B - C) / (Y0 * B + C)
    return abs(r) ** 2, abs(2.0 * Y0 / (Y0 * B + C)) ** 2, r


def _slab_residual(eps, mu, th0, M, magnetic=True):
    """max over {R, T, Jones} x {TE, TM} of |engine - analytic| for a uniform
    slab.  ``phi = 0``, so incident E_y is TE (s) and incident E_x is TM (p);
    the library's efficiency ROWS are already the s / p power responses."""
    kw = dict(mu_cell=np.full((2, 2), complex(mu))) if magnetic else {}
    o, R, T, J = pmm_jones_2d_staggered(
        _P, _P, np.full((2, 2), complex(eps)), 1.0, 1.0, _DEP, _WL,
        degree=M, n_orders=2, theta=th0, phi=0.0, **kw)
    p0 = _idx(o)[(0, 0)]
    out = 0.0
    for pol, row in (("tm", 0), ("te", 1)):
        Ra, Ta, ra = _airy(eps, mu, th0, _DEP / _WL, pol)
        out = max(out, abs(float(R[row, p0]) - Ra), abs(float(T[row, p0]) - Ta),
                  abs(complex(J[row, row]) - ra))
    return out


def test_g2_the_analytic_oracle_matches_the_nonmagnetic_path():
    """PIN THE ORACLE FIRST.  The Airy formula above is only usable as a
    magnetic oracle if it reproduces the SHIPPED nonmagnetic solver, whose
    scalar path is already gated -- including the Jones PHASE, which is where
    the exp(-iwt) bridge lives.

    MEASURED 2026-09-10 (build doc M2A) at M=8, eps=4, mu=1: normal incidence
    1.36e-14 (mu_cell omitted) / 2.45e-14 (mu=1 forced through the magnetic
    path), theta=0.35 6.66e-15 / 1.27e-14.  Bar 1e-12 -- 40x over the worst,
    and the failure mode this guards against is O(1e-1..1e0) (the wrong
    time-convention arm reads dT = 4.4).
    """
    for th in (0.0, 0.35):
        assert _slab_residual(4.0, 1.0, th, 8, magnetic=False) < 1e-12
        assert _slab_residual(4.0, 1.0, th, 8) < 1e-12


@pytest.mark.parametrize("eps,mu", [
    (4.0, 2.0),                  # lossless magnetic
    (1.0, 4.0),                  # purely magnetic contrast (eps = vacuum)
    (4.0, 2.0 + 0.3j),           # LOSSY permeability
    (4.0 + 0.2j, 2.0),           # lossy permittivity in a magnetic host
])
def test_g2_uniform_magnetic_slab_vs_analytic(eps, mu):
    """R, T AND the complex Jones of a uniform magnetic slab against the
    analytic impedance formula, at normal and oblique incidence, both
    polarizations.

    MEASURED 2026-09-10 (build doc M2B/M2C) at M=8, worst over {R, T, Jones}
    x {TE, TM} for each pair:

        eps  mu           theta 0        theta 0.35
        4.0  2.0          4.298e-14      3.064e-14
        1.0  4.0          1.870e-14      1.177e-14
        4.0  2.0+0.3j     3.689e-14      3.014e-14
        4.0+0.2j  2.0     3.613e-14      2.766e-14

    Bar 1e-12 = 23x over the worst of those and 4.4 decades under the M=5
    oblique residual (2.414e-08, the smallest genuine discretization signal
    here), so it has a gap on both sides.
    """
    for th in (0.0, 0.35):
        assert _slab_residual(eps, mu, th, 8) < 1e-12


def test_g2_uniform_slab_converges_spectrally_in_m():
    """TWO-SIDED companion: a UNIFORM region is smooth, so the oblique
    residual must FALL steeply with M rather than merely being small at one M.

    MEASURED 2026-09-10 (build doc M2B), eps=4 mu=2 at theta=0.35:
    M=5 2.414e-08 -> M=8 3.064e-14, a factor 7.9e+05.  Bar: a drop of at
    least 1e3, i.e. 790x of headroom.  (At NORMAL incidence both ends already sit at
    round-off, ~1e-15, so there is nothing to converge and the ladder claim is
    made at oblique, where the transverse Bloch phase must be resolved.)
    """
    d5 = _slab_residual(4.0, 2.0, 0.35, 5)
    d8 = _slab_residual(4.0, 2.0, 0.35, 8)
    assert d8 < d5 / 1e3, (d5, d8)


# =========================================================================== #
# G3 -- ELECTROMAGNETIC DUALITY: the exact physics oracle for mu.
# =========================================================================== #
_D2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _sp_drives(theta, phi):
    return ((-np.sin(phi), np.cos(phi)),
            (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi)))


def _eff(amps, drive, kz_inc, kx0, ky0):
    """Per-order efficiency for an arbitrary tangential drive -- the library's
    own projection (``rcwa._core._project_efficiency``) applied to the linear
    combination of the two returned drive rows.  ``|E_inc|^2`` carries the
    tangential norm explicitly: the library hardcodes 1.0 there because ITS
    two drives are unit-TANGENTIAL, while an (s, p) drive is unit in the FULL
    field."""
    ex0, ey0 = drive
    Ex = ex0 * amps["Ex"][0] + ey0 * amps["Ex"][1]
    Ey = ex0 * amps["Ey"][0] + ey0 * amps["Ey"][1]
    kx, ky, kz = amps["kx"], amps["ky"], amps["kz"]
    Ez = -(kx * Ex + ky * Ey) / np.where(np.abs(kz) < 1e-12, 1.0, kz)
    einc = abs(ex0) ** 2 + abs(ey0) ** 2
    if kz_inc != 0:
        einc = einc + ((kx0 * ex0 + ky0 * ey0) / kz_inc) ** 2
    e = (np.real(kz / kz_inc)
         * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2) / einc)
    return np.where(np.real(kz) > 0, np.real(e), 0.0)


def _jones_sp(amps, p0, theta, phi):
    ct, sp, cp = np.cos(theta), np.sin(phi), np.cos(phi)
    out = np.zeros((2, 2), dtype=complex)
    for col, drive in enumerate(_sp_drives(theta, phi)):
        Ex = drive[0] * amps["Ex"][0][p0] + drive[1] * amps["Ex"][1][p0]
        Ey = drive[0] * amps["Ey"][0][p0] + drive[1] * amps["Ey"][1][p0]
        out[0, col] = -sp * Ex + cp * Ey
        out[1, col] = (cp * Ex + sp * Ey) / ct
    return out


def _run(eps_c, mu_c, theta, phi, M):
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.0, n_modes=M,
                        n_orders=3)
    if mu_c is None:
        st.add_layer(_DEP, eps_cell=eps_c)
    else:
        st.add_layer(_DEP, eps_cell=eps_c, mu_cell=mu_c)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return (np.asarray(o), st.per_order_amplitudes("reflection"),
            st.per_order_amplitudes("transmission"))


def _duality(eps_c, mu_c, theta, phi, M):
    """(per-order R/T residual, Jones residual, no-rotation control) between a
    grating and its DUAL (eps and mu exchanged), with vacuum half-spaces.

    Duality maps the incident and outgoing polarizations by
    ``D = [[0,-1],[1,0]]`` in the (s, p) frame -- a UNIT-magnitude map at any
    incidence, which is why the comparison is made there and not in the lab
    (x, y) tangential basis.  The Jones rule carries a minus because the
    incident p-hat here is the optics one, ``-(k^ x s^)``, while the reflected
    p-hat is ``+(k^_r x s^)``."""
    o, refA, trnA = _duality._run(eps_c, mu_c, theta, phi, M)
    dual_eps = mu_c if mu_c is not None else _uni(_EYE, eps_c.shape[0])
    o2, refB, trnB = _duality._run(dual_eps, eps_c, theta, phi, M)
    assert np.array_equal(o, o2)
    sd, pd = _sp_drives(theta, phi)
    kzi, kx0, ky0 = refA["kz_inc"], refA["kx0"], refA["ky0"]
    d = 0.0
    for A, B in ((refA, refB), (trnA, trnB)):
        for drA, drB in ((sd, pd), (pd, sd)):
            d = max(d, float(np.max(np.abs(_eff(A, drA, kzi, kx0, ky0)
                                           - _eff(B, drB, kzi, kx0, ky0)))))
    p0 = _idx(o)[(0, 0)]
    JA, JB = _jones_sp(refA, p0, theta, phi), _jones_sp(refB, p0, theta, phi)
    return (d, float(np.max(np.abs(JB + _D2 @ JA @ np.linalg.inv(_D2)))),
            float(np.max(np.abs(JB - JA))))


_duality._run = _run


def test_g3_duality_uniform_tensor_pair_is_spectral():
    """A UNIFORM (eps, mu) tensor pair carries no corner, so both arms are
    spectrally accurate and duality must hold at round-off.

    MEASURED 2026-09-10 (build doc M3B) at conical incidence
    (theta 0.30, phi 0.70): M=5 5.22e-09 -> M=8 5.87e-14 on R/T and 1.76e-13
    on the Jones (at NORMAL incidence 1.6e-13 / 4.2e-14 at every M).  Bar
    1e-11 = 57x over the worst M=8 reading; the no-rotation control sits at
    3.01e-01, i.e. 10 decades above the bar, so the ROTATION is what is being
    measured and not a coincidence of magnitudes.
    """
    ue, um = _uni(_LC), _uni(_LC2)
    d, dJ, ctrl = _duality(ue, um, 0.30, 0.70, 8)
    assert d < 1e-11, d
    assert dJ < 1e-11, dJ
    assert ctrl > 1e-2, ctrl


def test_g3_duality_patterned_cell_ladder():
    """The staggered DISCRETIZATION is not self-dual (E3 lives in V3 while H3
    lands in Vw), so a PATTERNED cell's two duality arms agree only to
    discretization accuracy -- the claim is therefore TWO-SIDED: the mismatch
    must SHRINK with M, and the M=8 mismatch must sit under a derived bar.

    Arm A here is ``mu = 1``, i.e. the SHIPPED (already gated) anisotropic
    path, so its dual -- eps = vacuum, mu = the whole pattern -- is the
    magnetic assembly measured against a verified one.

    MEASURED 2026-09-10 (build doc M3A), LC host + isotropic pillar:

        incidence            M=5        M=8      ratio
        normal            5.991e-03  2.512e-04    23.9x
        conical 0.30/0.70 1.542e-03  7.038e-05    21.9x

    (the probe's third incidence, theta 0.30 / phi 0, reads
    4.988e-03 -> 4.266e-05, 117x; the test takes the max over the two above.)

    Bars: 3x the worst M=8 reading -> 7.6e-4 (this is deterministic
    discretization error, not build noise -- it is the same shape as the
    shipped G4 stripe ladder, whose BLAS-path sensitivity was measured at
    ~2e-09 relative), and a ladder drop of at least 5x (measured 21.9x at
    worst, so 4.4x of headroom).
    """
    ec = _cell(_LC, 4.0 * _EYE)
    d5 = max(_duality(ec, None, th, ph, 5)[0]
             for th, ph in ((0.0, 0.0), (0.30, 0.70)))
    d8s = [_duality(ec, None, th, ph, 8) for th, ph in ((0.0, 0.0),
                                                        (0.30, 0.70))]
    d8 = max(x[0] for x in d8s)
    assert d8 < 7.6e-4, d8
    assert max(x[1] for x in d8s) < 7.6e-4
    assert d8 < d5 / 5.0, (d5, d8)
    # the rotation is load-bearing: the unrotated Jones comparison is decades
    # away (measured 2.94e-01 / 5.47e-01)
    assert min(x[2] for x in d8s) > 1e-2


# =========================================================================== #
# G4 -- a y-uniform MAGNETIC stripe against the 1-D engines, through duality.
#
# ENGINE CENSUS (2026-09-10): NO 1-D diffraction engine in the library accepts
# a permeability -- rcwa_jones_1d / rcwa_efficiency_1d / PMMStack /
# pmm_jones_1d are nonmagnetic, and berreman.py's layer matrix states mu = 1.
# (The only user-settable mu anywhere is eme_2d_vector's ``mu_xy``, a WAVEGUIDE
# MODE solver: it returns propagation constants, not diffraction orders.)  So
# the 1-D check runs through duality: the ELECTRIC stripe both 1-D engines can
# solve IS the dual of the magnetic one this engine solves.
# =========================================================================== #
_G4_P, _G4_DEP, _G4_TH = 0.90e-6, 0.30e-6, 0.22
_G4_RIDGE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_G4_GROOVE = 2.10 * np.eye(3, dtype=complex)


def _g4_stripe(ridge, groove):
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = groove
    c[0, :] = ridge
    return c


@pytest.fixture(scope="module")
def g4_oracles():
    kw = dict(angle=_G4_TH)
    return (pmm_jones_1d(_G4_P, _G4_RIDGE, _G4_GROOVE, 1.0, 1.0, _G4_DEP, 0.5,
                         _WL, degree=16, stabilize=False, **kw),
            rcwa_jones_1d(_G4_P, _G4_RIDGE, _G4_GROOVE, 1.0, 1.0, _G4_DEP, 0.5,
                          _WL, n_orders=40, **kw))


def _g4_residual(M, oracle):
    """(R/T residual, Jones residual, y-forbidden leak, unswapped control) of
    the MAGNETIC stripe (eps = vacuum, mu = the stripe) against a 1-D ELECTRIC
    solve, with the duality row swap.

    R/T need no rescaling (each library row is already that drive's power
    response), but the JONES is a TANGENTIAL-amplitude matrix and duality maps
    tangential amplitudes by ``A = [[0, -kz], [1/kz, 0]]`` in a classical
    mount, so the rule is ``J_dual = -A J A^-1`` (it collapses to ``C J C`` at
    normal incidence)."""
    o, R, T, J = pmm_jones_2d_staggered(
        _G4_P, _G4_P, _g4_stripe(_EYE, _EYE), 1.0, 1.0, _G4_DEP, _WL,
        mu_cell=_g4_stripe(_G4_RIDGE, _G4_GROOVE), degree=M, n_orders=3,
        theta=_G4_TH, phi=0.0)
    o1, R1, T1, J1 = oracle
    sel = np.asarray(o)[:, 1] == 0
    two = {int(m): i for i, m in zip(np.arange(len(o))[sel],
                                     np.asarray(o)[sel, 0])}
    one = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
    common = [m for m in one if m in two]
    dRT = max(max(abs(R[r][two[m]] - R1[1 - r][one[m]]),
                  abs(T[r][two[m]] - T1[1 - r][one[m]]))
              for m in common for r in (0, 1))
    unsw = max(abs(R[r][two[m]] - R1[r][one[m]])
               for m in common for r in (0, 1))
    kz = np.cos(_G4_TH)
    A = np.array([[0.0, -kz], [1.0 / kz, 0.0]])
    dJ = float(np.max(np.abs(J + A @ J1 @ np.linalg.inv(A))))
    forb = float(max(np.max(np.abs(R[:, ~sel])), np.max(np.abs(T[:, ~sel]))))
    return float(dRT), dJ, forb, float(unsw)


def test_g4_magnetic_stripe_matches_both_1d_engines(g4_oracles):
    """A y-uniform MAGNETIC stripe has no corner in the 2-D field, so the
    staggered solve converges cleanly onto the exact 1-D result -- reached
    through duality, since no 1-D engine in the library takes a permeability.

    MEASURED 2026-09-10 (build doc M4), theta = 0.22:

        M      per-order max|dR|,|dT|      max|dJones|
        5            2.769e-03              1.442e-03
        6            1.607e-04              5.240e-05
        7            2.149e-05              2.316e-05
        8            7.630e-06              1.718e-05

    and at M=8 against the SECOND oracle (rcwa_jones_1d, 81 orders):
    6.757e-06 / 1.519e-05.  The two 1-D oracles agree with EACH OTHER to
    8.74e-07 on R/T and 1.98e-06 on the Jones, i.e. 34x / 35x under the bars
    below, so the oracle's own floor is not what is being measured.

    Bars ~4x the worst M=8 reading: 3e-5 on R/T (3.9x over 7.630e-06) and
    7e-5 on the Jones (4.1x over 1.718e-05).  The
    UNSWAPPED comparison (duality ignored) reads 4.07e-02 -- 3 decades above
    the bar, so the swap is load-bearing.
    """
    for oracle in g4_oracles:
        dRT, dJ, _forb, unsw = _g4_residual(8, oracle)
        assert dRT < 3e-5, dRT
        assert dJ < 7e-5, dJ
        assert unsw > 1e-3, unsw


def test_g4_ladder_and_y_momentum(g4_oracles):
    """TWO-SIDED companion + transverse-momentum conservation.

    MEASURED 2026-09-10 (build doc M4): the R/T residual falls 2.769e-03 ->
    7.630e-06 from M=5 to M=8 (363x); bar = a drop of at least 20x (18x of
    headroom).  A y-uniform cell cannot scatter into n != 0: the leak into
    those orders measures 2.87e-28 .. 4.68e-26 over the whole ladder -- a
    round-off floor, not a convergent quantity -- so the bar is 1e-20, 6
    decades above it and 17 decades below the smallest physical order here.
    """
    d5 = _g4_residual(5, g4_oracles[0])[0]
    d8, _dJ, forb, _u = _g4_residual(8, g4_oracles[0])
    assert d8 < d5 / 20.0, (d5, d8)
    assert forb < 1e-20, forb


# =========================================================================== #
# G5 -- lossless closure with a HERMITIAN permeability, the lossy-mu deficit,
#       and the tripwire's magnetic predicate.
# =========================================================================== #
def _closure(ec, mc, M, theta=0.0, phi=0.0, record=False):
    st = PMM2DStackPure(_P, _P, n_modes=M, n_orders=3)
    st.add_layer(_DEP, eps_cell=ec, mu_cell=mc)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T, J = st.solve()
    fired = [x for x in w if "energy closure" in str(x.message)]
    tot = [float(R[r].sum() + T[r].sum()) for r in (0, 1)]
    return (tot, len(fired)) if record else tot


def test_g5_hermitian_mu_is_lossless():
    """A HERMITIAN permeability absorbs nothing -- the Poynting dissipation
    term carries the anti-Hermitian parts of BOTH eps and mu -- so
    ``sum R + sum T = 1`` is exact for a Hermitian (eps, mu) pair between
    lossless half-spaces.  The permeability here is GYROTROPIC
    (``m12 = -m21 = +0.4i``), which is Hermitian and therefore lossless
    despite being complex: the case a naive ``Im(mu) != 0 -> lossy`` predicate
    would get wrong.

    MEASURED 2026-09-10 (build doc M5A), |sum R + sum T - 1| at M=8:
    uniform LC eps + gyrotropic mu 1.07e-13 (normal) / 2.91e-14 (conical);
    patterned 1.93e-09 (normal) / 1.48e-06 (conical).  Bars: 1e-11 for the
    uniform arm (94x over the measurement) and 1e-4 for the patterned one
    (68x), the latter being corner-capped algebraic convergence, not
    round-off.
    """
    for th, ph in ((0.0, 0.0), (0.25, 0.60)):
        tot = _closure(_uni(_LC), _uni(_MU_GYRO), 8, th, ph)
        assert max(abs(v - 1.0) for v in tot) < 1e-11, tot
        totp = _closure(_cell(_LC, 4.0 * _EYE), _cell(_MU_GYRO, 1.2 * _EYE),
                        8, th, ph)
        assert max(abs(v - 1.0) for v in totp) < 1e-4, totp


def test_g5_patterned_closure_improves_with_m():
    """TWO-SIDED companion: the patterned closure defect must be CONVERGENT,
    not a fixed offset.  MEASURED 2026-09-10 (build doc M5A) at conical
    incidence: M=6 1.732e-03 -> M=8 1.480e-06, a factor 1170.  Bar: at least
    10x (117x of headroom).
    """
    d6 = max(abs(v - 1.0) for v in _closure(
        _cell(_LC, 4.0 * _EYE), _cell(_MU_GYRO, 1.2 * _EYE), 6, 0.25, 0.60))
    d8 = max(abs(v - 1.0) for v in _closure(
        _cell(_LC, 4.0 * _EYE), _cell(_MU_GYRO, 1.2 * _EYE), 8, 0.25, 0.60))
    assert d8 < d6 / 10.0, (d6, d8)


def test_g5_lossy_mu_closes_below_one():
    """The other side of the same claim: an ANTI-HERMITIAN mu part
    (``Im(m11) = +0.25`` in the PUBLIC gauge) must absorb, i.e. close BELOW 1
    by a visible margin -- and the tripwire must stay SILENT there, because no
    unity claim exists to violate.

    MEASURED 2026-09-10 (build doc M5B): sum R + T = 0.953739 / 0.552147 at
    M=6 and 0.953733 / 0.552172 at M=8 (the two incident polarizations), i.e.
    a deficit of 0.046 that is stable to 6e-06 in M.  Bar: below 0.99 -- 4.6x
    inside the measured deficit and decades outside the ~1e-6 closure defect
    of the corresponding LOSSLESS cell.
    """
    tot, fired = _closure(_cell(_LC, 4.0 * _EYE), _cell(_MU_LOSSY, 1.2 * _EYE),
                          8, record=True)
    assert max(tot) < 0.99, tot
    assert min(tot) > 0.0, tot
    assert fired == 0, tot


def test_g5_tripwire_recognises_a_hermitian_mu_as_lossless():
    """The closure tripwire must (a) stay silent on a RESOLVED Hermitian-mu
    solve and (b) FIRE on an under-resolved one -- the two-sided form of "the
    predicate now includes mu".

    MEASURED 2026-09-10 (build doc M5C), LC/iso eps + gyro/1.2 mu at
    theta 0.25 / phi 0.60: M=8 closes to 1e-06 and warns 0 times; M=3 closes
    to 0.849 -- 17x the shipped 5e-2 window -- and warns twice (once per
    incident polarization).  The M=3 arm is the ENGINEERED under-resolution
    the standards ask for, not a hoped-for one: M=3 is the minimum the basis
    admits.
    """
    tot8, fired8 = _closure(_cell(_LC, 4.0 * _EYE),
                            _cell(_MU_GYRO, 1.2 * _EYE), 8, 0.25, 0.60,
                            record=True)
    assert fired8 == 0, tot8
    tot3, fired3 = _closure(_cell(_LC, 4.0 * _EYE),
                            _cell(_MU_GYRO, 1.2 * _EYE), 3, 0.25, 0.60,
                            record=True)
    assert max(abs(v - 1.0) for v in tot3) > 0.3, tot3
    assert fired3 == 2, (fired3, tot3)


# =========================================================================== #
# G9 -- the layer_absorption budget with a LOSSY MAGNETIC layer.
# =========================================================================== #
def _absorption_closure(M):
    """max over the two incident polarizations of
    ``|sum_i A_i - (1 - sum R - sum T)|`` for a two-layer stack whose FIRST
    layer is magnetic and lossy.  The two sides come from different machinery
    -- the internal block-Gram flux quadrature vs the Rayleigh far field --
    so this is a cross-machinery identity, not a tautology."""
    st = PMM2DStackPure(_P, _P, n_modes=M, n_orders=3)
    st.add_layer(_DEP, eps_cell=_cell(_LC, 4.0 * _EYE),
                 mu_cell=_cell(_MU_LOSSY, 1.2 * _EYE))
    st.add_layer(0.15e-6, eps=2.25)
    st.set_source(_WL, theta=0.2, phi=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    return float(np.max(np.abs(A.sum(axis=0)
                               - (1.0 - R.sum(axis=1) - T.sum(axis=1)))))


def test_g9_absorption_budget_closes_for_a_magnetic_layer():
    """``retain_internal`` / :meth:`layer_absorption` must keep working when a
    layer is MAGNETIC.  The flux bilinear form is the eps-free block Gram of
    the (nonmagnetic) half-space assembly, which the magnetic path does not
    touch -- but "should still hold" is not a measurement, so it is measured.

    MEASURED 2026-09-10 (build doc M9), lossy-mu layer over an isotropic one at
    theta 0.2 / phi 0.4:

        M      |sum A - (1 - R - T)|
        5           1.285e-05
        6           7.042e-07
        7           2.192e-08
        8           9.867e-10

    (absorbed fractions 0.0553 / 0.4458 at M=8, i.e. the budget is a large
    number, not a rounding artefact.)  Bar 1e-8 at M=8 -- 10x over the
    measurement -- plus the two-sided companion: the residual must fall at
    least 100x from M=5 to M=8 (measured 13,000x).
    """
    d5, d8 = _absorption_closure(5), _absorption_closure(8)
    assert d8 < 1e-8, d8
    assert d8 < d5 / 100.0, (d5, d8)


# =========================================================================== #
# G6 -- the x<->y transpose symmetry, with the mu blocks swapped like the eps
#       blocks.  This is what a swapped m12/m21 placement breaks.
# =========================================================================== #
def _transpose_residual(ea, ma, eb, mb, M=6):
    kw = dict(n_substrate=1.5, n_superstrate=1.0, depth=_DEP, wavelength=_WL,
              degree=M, n_orders=3)
    oA, RA, TA, JA = pmm_jones_2d_staggered(_P, 1.3 * _P, ea, mu_cell=ma, **kw)
    oB, RB, TB, JB = pmm_jones_2d_staggered(1.3 * _P, _P, eb, mu_cell=mb, **kw)
    ia, ib = _idx(oA), _idx(oB)
    dev = max(max(abs(RB[r][ib[(m, n)]] - RA[1 - r][ia[(n, m)]]),
                  abs(TB[r][ib[(m, n)]] - TA[1 - r][ia[(n, m)]]))
              for (m, n) in ib if (n, m) in ia for r in (0, 1))
    return float(dev), float(np.max(np.abs(JB - _S2 @ JA @ _S2)))


def test_g6_transpose_symmetry_with_the_mu_blocks_swapped():
    """Transposing the cell about x = y -- swap the grid axes, the periods,
    AND the tensor components (e11<->e22, e12<->e21, and the SAME on mu) --
    must map order (m, n) -> (n, m) and swap the Jones rows and columns
    exactly, on a symmetric discretization.

    MEASURED 2026-09-10 (build doc M6), LC/iso eps + gyrotropic/1.2 mu at M=6:
    per-order residual 2.80e-14, Jones residual 3.85e-14.  Bar = 1e-11: 260x
    over the measurement and 8 decades under the 5.22e-03 the swapped-block
    control below produces.
    """
    ea, ma = _cell(_LC, 4.0 * _EYE), _cell(_MU_GYRO, 1.2 * _EYE)
    dev, dJ = _transpose_residual(ea, ma, _transpose_cell(ea),
                                  _transpose_cell(ma))
    assert dev < 1e-11 and dJ < 1e-11, (dev, dJ)


def test_g6_mu_mixed_block_placement_control():
    """THE placement control for the PERMEABILITY.  Interchanging ``m12`` and
    ``m21`` in ONE arm must BREAK the transpose symmetry -- and the
    GYROTROPIC mu (``m12 = -m21``) is the discriminator, exactly as the
    gyrotropic eps was for the permittivity blocks: the rotated-LC tensors in
    this file have ``e12 = e21``, so swapping THEM is a no-op (measured
    2.24e-14, i.e. no break at all).

    MEASURED 2026-09-10 (build doc M6) at M=6: correct placement 2.80e-14,
    m12/m21 swapped 5.22e-03 (its Jones residual 1.33e-01) -- 11 decades
    apart.  Bars: correct < 1e-11, swapped > 1e-7 (4.7 decades under the
    measured break, 4 decades over the correct arm): a two-sided, fail-before
    demonstration.
    """
    ea, ma = _cell(_LC, 4.0 * _EYE), _cell(_MU_GYRO, 1.2 * _EYE)
    eb, mb = _transpose_cell(ea), _transpose_cell(ma)
    ok, _ = _transpose_residual(ea, ma, eb, mb)
    maw = ma.copy()
    maw[..., 0, 1], maw[..., 1, 0] = ma[..., 1, 0].copy(), ma[..., 0, 1].copy()
    bad, _ = _transpose_residual(ea, maw, eb, mb)
    eaw = ea.copy()
    eaw[..., 0, 1], eaw[..., 1, 0] = ea[..., 1, 0].copy(), ea[..., 0, 1].copy()
    noop, _ = _transpose_residual(eaw, ma, eb, mb)
    assert ok < 1e-11, ok
    assert bad > 1e-7, bad
    assert noop < 1e-11, noop      # symmetric eps off-diagonal: swap is a no-op


# =========================================================================== #
# FAIL-BEFORE -- every magnetic term, including the R-vs-Gram separation, is
#                load-bearing.  Knocked out THROUGH the shipped code.
# =========================================================================== #
def _knockout_residual(monkeypatch, kind):
    """The G2 analytic residual (uniform isotropic magnetic slab, M=8,
    theta=0.35) with one magnetic term disabled."""
    if kind == "gram":
        orig = Granet2DTransverseE._assemble

        def asm(self):
            orig(self)
            self.Ggram_blocks = None
        monkeypatch.setattr(Granet2DTransverseE, "_assemble", asm)
    else:
        orig_chi = Granet2DTransverseE._chi_maps

        def chi(self):
            c = orig_chi(self)
            if c is None:
                return None
            c11, c12, c21, c22, c33 = c
            if kind == "chi33":
                return c11, c12, c21, c22, np.ones_like(c33)
            if kind == "chi_t":
                return (np.ones_like(c11), np.zeros_like(c12),
                        np.zeros_like(c21), np.ones_like(c22), c33)
            raise AssertionError(kind)
        monkeypatch.setattr(Granet2DTransverseE, "_chi_maps", chi)
    return _slab_residual(4.0, 2.0, 0.35, 8)


def test_failbefore_the_r_vs_gram_separation_is_load_bearing(monkeypatch):
    """THE TRAP.  With ``chi_t = I`` the shipped code uses ONE object
    (``-Rmat``) both as the pencil's right-hand matrix and as the block field
    Gram that recovers the Eq.-25 H partners.  With ``chi_t != I`` these are
    DIFFERENT operators: the pencil needs ``R = C[chi_t]C``, while Eq. 25
    carries no ``chi_t`` at all and must project with ``blockdiag(G1, G2)``.
    Collapsing them applies ``[chi_t]^-1`` to every H partner -- an interface
    error that no eigenvalue and no renormalised energy check can see.

    MEASURED 2026-09-10 (build doc M1b): intact 1.88e-14 against the analytic
    oracle; with the Gram collapsed onto R, 2.15e-01.  Bars: intact < 1e-12,
    collapsed > 1e-4 (3 decades under the measured break, 8 decades over the
    intact arm).
    """
    assert _slab_residual(4.0, 2.0, 0.35, 8) < 1e-12
    assert _knockout_residual(monkeypatch, "gram") > 1e-4


@pytest.mark.parametrize("kind,eq", [
    ("chi33", "Eq. 20/A42 -- S_tt is the chi33-weighted curl-curl"),
    ("chi_t", "Eq. 24/A39 + Eq. 21/A43 -- R and K_tz carry [chi_t]"),
])
def test_failbefore_each_chi_weight_is_load_bearing(monkeypatch, kind, eq):
    """Each permeability weight, removed on its own, must break the analytic
    agreement -- measured through the shipped code, not a forked copy.

    MEASURED 2026-09-10 (build doc M1b), uniform eps=4 mu=2 slab at
    theta=0.35, M=8, against the analytic Airy oracle:

        intact                       1.876e-14
        chi33 -> 1   (S_tt)          7.232e-03
        chi_t -> I   (R, K_tz)       2.747e-02

    Bar 1e-4: 72x under the smaller break and 10 decades over the intact arm.
    (The MIXED chi12 / chi21 blocks cannot be seen by an isotropic mu; they
    are gated by the duality test below, which reads 3.402e-02 with them
    zeroed against 4.219e-14 intact.)
    """
    assert _knockout_residual(monkeypatch, kind) > 1e-4, eq


def test_failbefore_the_mixed_chi_blocks_are_load_bearing(monkeypatch):
    """The C-rotated MIXED blocks (``chi21`` in ``R[1,2]`` / ``K_tz`` row 1,
    ``chi12`` in ``R[2,1]`` / row 2) need an ANISOTROPIC mu to be visible, so
    they are gated by duality on a uniform (eps, mu) tensor pair.

    MEASURED 2026-09-10 (build doc M1b) at conical incidence, M=7: intact
    4.219e-14, mixed blocks zeroed 3.402e-02 -- 12 decades apart.  Bars:
    intact < 1e-11 (as in G3), zeroed > 1e-4.
    """
    ue, um = _uni(_LC), _uni(_LC2)
    assert _duality(ue, um, 0.30, 0.70, 7)[0] < 1e-11
    orig_chi = Granet2DTransverseE._chi_maps

    def chi(self):
        c = orig_chi(self)
        if c is None:
            return None
        c11, c12, c21, c22, c33 = c
        return c11, np.zeros_like(c12), np.zeros_like(c21), c22, c33
    monkeypatch.setattr(Granet2DTransverseE, "_chi_maps", chi)
    assert _duality(ue, um, 0.30, 0.70, 7)[0] > 1e-4


# =========================================================================== #
# G7 -- guards.  Every message names the limitation and the alternative.
# =========================================================================== #
def _oop(t33, i, j, val):
    c = _uni(t33)
    c[..., i, j] = val
    c[..., j, i] = val
    return c


def test_g7_out_of_plane_mu_raises():
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE permeability"):
        pmm_jones_2d_staggered(_P, _P, _uni(4.0 * _EYE), 1.0, 1.0, _DEP, _WL,
                               mu_cell=_oop(_EYE, 0, 2, 0.3), degree=4)


def test_g7_mu_with_an_out_of_plane_eps_raises():
    with pytest.raises(NotImplementedError, match="out-of-plane|OUT-OF-PLANE"):
        pmm_jones_2d_staggered(_P, _P, _oop(4.0 * _EYE, 1, 2, 0.4), 1.0, 1.0,
                               _DEP, _WL, mu_cell=_uni(_EYE), degree=4)
    with pytest.raises(NotImplementedError):
        PMM2DStackPure(_P, _P).add_layer(
            _DEP, eps_cell=_oop(4.0 * _EYE, 1, 2, 0.4), mu_cell=_uni(_EYE))
    with pytest.raises(NotImplementedError):
        PMM2DStackPure(_P, _P).add_layer(
            _DEP, eps=_oop(4.0 * _EYE, 0, 2, 0.4)[0, 0], mu=2.0)


def test_g7_magnetic_half_space_raises():
    """A magnetic half-space changes the wave impedance, hence BOTH the
    Rayleigh flux normalisation and the incident-amplitude overlap.  The
    keyword exists only so that asking for one is LOUD rather than silently
    normalised for vacuum."""
    for kw in ({"mu_superstrate": 2.0}, {"mu_substrate": 1.5}):
        with pytest.raises(NotImplementedError, match="MAGNETIC HALF-SPACE"):
            pmm_jones_2d_staggered(_P, _P, _uni(4.0 * _EYE), 1.0, 1.0, _DEP,
                                   _WL, degree=4, **kw)
        with pytest.raises(NotImplementedError, match="MAGNETIC HALF-SPACE"):
            PMM2DStackPure(_P, _P, **kw)
    # mu = 1 is the nonmagnetic default and must be accepted
    PMM2DStackPure(_P, _P, mu_superstrate=1.0, mu_substrate=1)


def test_g7_shape_and_singularity_guards():
    ec = _uni(4.0 * _EYE)
    bad = [
        (np.ones((2, 2, 2, 2)), "must be \\(Nx, Ny, 3, 3\\)"),
        (np.ones((3, 3)), "must match the eps_cell grid"),
        (np.ones((2, 3)), "must be SQUARE"),
        (np.zeros((2, 2)), "must be nonzero"),
        (np.ones((2, 2, 3)), "2-D \\(Nx, Ny\\) scalar grid"),
    ]
    for mu, msg in bad:
        with pytest.raises(ValueError, match=msg):
            pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL,
                                   mu_cell=mu, degree=4)
    sing = _uni(_EYE)
    sing[..., 0, 1] = sing[..., 1, 0] = 1.0        # det [mu_t] = 0
    with pytest.raises(ValueError, match="SINGULAR"):
        pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL, mu_cell=sing,
                               degree=4)
    zz = _uni(_EYE)
    zz[..., 2, 2] = 0.0
    with pytest.raises(ValueError, match="m_zz must be nonzero"):
        pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL, mu_cell=zz,
                               degree=4)
    with pytest.raises(ValueError, match="at most ONE of mu"):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps=4.0, mu=2.0,
                                         mu_cell=np.ones((2, 2)))
    with pytest.raises(ValueError, match="uniform mu must be a scalar"):
        PMM2DStackPure(_P, _P).add_layer(_DEP, eps=4.0, mu=np.ones((2, 2, 2)))


def test_g7_homog_geom_cache_refuses_a_magnetic_region():
    """The shared eps-free geometric eig does not exist for a magnetic region
    (chi_t and chi33 weight R, K_tz and S_tt, so ``L0_geom`` is not
    geometric).  It must RAISE rather than silently return a wrong basis --
    which is why a uniform magnetic layer takes its own region eig."""
    sol = Granet2DTransverseE(_P / _WL, _P / _WL, 2, 2, 4,
                              np.full((2, 2), 4.0 + 0j), k0=2 * np.pi,
                              mu_cell=np.full((2, 2), 2.0 + 0j))
    with pytest.raises(ValueError, match="MAGNETIC region"):
        _homog_geom_cache(sol)


def test_g7_float_noise_in_m13_does_not_trip_the_block_form_guard():
    """A physically in-plane permeability built by ROTATING a diagonal one
    carries ~1e-16 in the xz/yz/zx/zy slots; a strict ``> 0`` test would
    reject every such tensor.  The guard is RELATIVE (1e-12 * scale), shared
    verbatim with the permittivity.

    MEASURED 2026-09-10 (build doc M7): a 1e-16 stray in m13 solves, and its
    R is BIT-IDENTICAL to the exact-identity mu (max|dR| = 0.0) -- the 2x2
    inverse and chi33 never read the out-of-plane slots once the guard has
    passed.
    """
    ec = _uni(4.0 * _EYE)
    tiny = _uni(_EYE)
    tiny[..., 0, 2] = 1e-16
    kw = dict(degree=5, n_orders=2)
    o, R, T, J = pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL,
                                        mu_cell=tiny, **kw)
    o2, R2, T2, J2 = pmm_jones_2d_staggered(_P, _P, ec, 1.0, 1.0, _DEP, _WL,
                                            mu_cell=_uni(_EYE), **kw)
    assert np.array_equal(R, R2)
    assert np.array_equal(T, T2)


# =========================================================================== #
# API -- the stack builder's magnetic combinations, and the dedupe.
# =========================================================================== #
def test_api_uniform_and_patterned_mu_combinations_agree():
    """``add_layer`` must accept any combination of uniform / patterned on the
    eps and mu sides, and a uniform spec must be exactly the constant cell it
    broadcasts to (the union grid is taken from whichever side is patterned).
    """
    ec = _cell(_LC, 4.0 * _EYE)
    st_a = PMM2DStackPure(_P, _P, n_modes=5, n_orders=2)
    st_a.add_layer(_DEP, eps_cell=ec, mu=_MU_GYRO)
    st_b = PMM2DStackPure(_P, _P, n_modes=5, n_orders=2)
    st_b.add_layer(_DEP, eps_cell=ec, mu_cell=_uni(_MU_GYRO))
    for st in (st_a, st_b):
        st.set_source(_WL, theta=0.2, phi=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oa, Ra, Ta, Ja = st_a.solve()
        ob, Rb, Tb, Jb = st_b.solve()
    assert np.array_equal(Ra, Rb)
    assert np.array_equal(Ta, Tb)
    # uniform eps + PATTERNED mu: the union grid comes from the mu side
    st_c = PMM2DStackPure(_P, _P, n_modes=5, n_orders=2)
    st_c.add_layer(_DEP, eps=4.0, mu_cell=_cell(_MU_GYRO, 1.2 * _EYE))
    st_c.set_source(_WL, theta=0.2, phi=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oc, Rc, Tc, Jc = st_c.solve()
    assert Rc.shape == Ra.shape


def test_api_magnetic_layer_in_a_generalized_cascade():
    """A MAGNETIC layer must also work in a stack that contains an
    OUT-OF-PLANE layer, i.e. on the GENERALIZED S-matrix cascade: a magnetic
    region is still a symmetric second-order one, so it enters as
    ``[[W, W], [V, -V]]`` through ``_modes_as_general`` exactly as an in-plane
    tensor layer does.  The claim is that the two routes COMPOSE, and it is
    checked the only way a mixed stack can be: the lossless closure.

    MEASURED 2026-09-10 (build doc, Section 5): tilted-director out-of-plane
    layer over a uniform-eps MAGNETIC layer, theta 0.2 / phi 0.4 --
    |sum R + sum T - 1| = 7.955e-03 (M=4), 1.057e-03 (M=5), 1.749e-05 (M=6),
    3.081e-06 (M=7).  Bar: 1e-4 at M=7 (32x over the measurement) plus the
    two-sided companion, a drop of at least 50x from M=4 (measured 2582x).
    Both layers are lossless (a real uniform mu and a rotated-real eps), so
    the tripwire also gets to be silent here.
    """
    oop = uniaxial_tensor(1.5, 1.8, 0.6, phi=0.3)      # tilted -> out-of-plane
    ec = _uni(oop)
    ec[0, 0] = 4.0 * _EYE

    def dev(M):
        st = PMM2DStackPure(_P, _P, n_modes=M, n_orders=3)
        st.add_layer(0.20e-6, eps_cell=ec)
        st.add_layer(0.20e-6, eps=4.0, mu_cell=_uni(1.6 * _EYE))
        st.set_source(_WL, theta=0.2, phi=0.4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o, R, T, J = st.solve()
        fired = [x for x in w if "energy closure" in str(x.message)]
        return (max(abs(float(R[r].sum() + T[r].sum()) - 1.0) for r in (0, 1)),
                len(fired))

    d4, _f4 = dev(4)
    d7, f7 = dev(7)
    assert d7 < 1e-4, d7
    assert d7 < d4 / 50.0, (d4, d7)
    assert f7 == 0, d7


def test_api_two_identical_magnetic_layers_share_one_eig(monkeypatch):
    """A magnetic layer cannot ride the shared eps-free geometric eig, so it
    is deduped by ``(eps bytes, mu bytes)`` like a patterned cell: two
    byte-identical magnetic layers must cost ONE region eig, and two layers
    differing ONLY in mu must cost two."""
    calls = []
    orig = TS._region_modes

    def counting(solver):
        calls.append(1)
        return orig(solver)
    monkeypatch.setattr(TS, "_region_modes", counting)
    monkeypatch.setattr("lumenairy.elements.pmm.stack2d_pure._region_modes",
                        counting)
    ec = _uni(4.0 * _EYE)
    st = PMM2DStackPure(_P, _P, n_modes=4, n_orders=2)
    st.add_layer(_DEP, eps_cell=ec, mu_cell=_uni(_MU_GYRO))
    st.add_layer(_DEP, eps_cell=ec, mu_cell=_uni(_MU_GYRO))
    st.set_source(_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    assert len(calls) == 1, len(calls)
    calls.clear()
    st2 = PMM2DStackPure(_P, _P, n_modes=4, n_orders=2)
    st2.add_layer(_DEP, eps_cell=ec, mu_cell=_uni(_MU_GYRO))
    st2.add_layer(_DEP, eps_cell=ec, mu_cell=_uni(1.2 * _EYE))
    st2.set_source(_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st2.solve()
    assert len(calls) == 2, len(calls)

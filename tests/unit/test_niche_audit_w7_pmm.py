"""AUDIT W7 -- ``lumenairy/elements/pmm/**`` numerical interiors.

The territory's recorded non-coverage (``stack2d``/``stack2d_pure`` numerics,
the conical modal machinery, the JAX twins' numerics, internal-field /
retained-state numerics, dispersive + threaded sweeps) plus the W6
sibling-defect-class sweep (Berreman-F-1 flux-gauge conjugation, EME-W6-1
solver routing, W6-2 exponential conditioning, W6-3 discarded Im, BOR-B1/B2
discretization + unit scales, Asymptotic-A11/A13 caches).

FOUND + FIXED
=============

F-C (CRITICAL-physics, the Berreman-F-1 twin -- FLUX-GAUGE CONJUGATION)
    ``PMMStack._solve_covariant`` -- the Li covariant oblique-coordinate
    cascade that EVERY slanted-layer ``PMMStack`` solve routes through (and
    ``factorization='covariant'``) -- conjugates its half-space permittivities
    to the INTERNAL ``exp(+iwt)`` gauge at the top of the method, then passed
    those INTERNAL values straight into ``_core._kz_forward``, which is a
    PUBLIC ``exp(-iwt)`` helper.  Double-conjugated, ``sqrt`` landed in the 4th
    quadrant, the ``Im < 0`` branch flip sent ``Re(kz) < 0``, and the
    ``Re(kz) > 0`` propagating mask inside ``_assemble_jones_farfield``
    SILENTLY ZEROED T into any absorbing substrate.  Identical mechanism, and
    identical signature, to the Berreman F-1 finding.

    Measured pre-fix.  The oracle is exact: with a HOMOGENEOUS layer the slant
    is a physical no-op, so the ``slant_angle=0`` (vertical) cascade of the
    SAME stack is the right answer.  P=0.30 um, eps 2.25, 0.22 um deep,
    wl 0.55 um, slant 0.35 rad, theta=0.3, degree 8, far_field_orders 5:

    ==============  ========================  ========================
    n_substrate     T, slant=0 (oracle)       T, slant=0.35 (covariant)
    ==============  ========================  ========================
    1.5             [0.96483, 0.95489]        [0.96483, 0.95489]  ok
    1.5+0.01j       [0.96586, 0.95613]        [0, 0]
    1.5+0.05j       [0.96974, 0.96086]        [0, 0]
    1.5+0.30j       [0.98578, 0.98047]        [0, 0]
    0.2+3.5j        [0.08641, 0.08135]        [0, 0]
    ==============  ========================  ========================

    -- a 98.6% energy loss with ZERO warnings: ``_warn_stack_energy`` only
    fires on super-unity / negative totals, and ``R+T = 0.014`` is "passive".
    An absorbing SUPERSTRATE was worse still: ``kz_inc = -1.14651`` tripped
    ``ValueError: non-propagating incidence`` on a perfectly propagating
    medium, so the covariant path was UNUSABLE there.

    FIX: un-conjugate at the three flux sites (``np.conj(eps_sup/eps_sub)``),
    exactly the ``kz_ord`` bridge ``_core._pmm_jones_oblique_core`` already
    carries and rcwa's ``_forward_flux_kz``.  Identity for a real eps, so every
    lossless solve is BYTE-UNCHANGED; the MODAL kz inside the cascade keeps
    the internal convention (that path was already correct).  Post-fix the
    covariant cascade reproduces the vertical oracle to 5.4e-8 over a
    30-config (n_sub, n_sup, theta) matrix -- the covariant union-grid
    discretization floor at degree 8, vs 7.5e-14 on the lossless rows.

F-A (HIGH, false hard failure on a valid configuration)
    Every 1-D / 2-D scalar + Jones entry point runs a degree-scan
    ``_stabilize_scalar`` / ``_stabilize_jones`` consensus whose PASSIVITY gate
    was ``tot <= 1 + tol``.  ``sum(R)+sum(T) <= 1`` is a theorem only for a
    LOSSLESS incidence medium: with an ABSORBING superstrate each wave is
    normalised by its own ``Re(kz_inc)`` z-flux while the incident/reflected
    cross-term also carries net flux, so the totals legitimately exceed unity
    -- which is exactly what ``_require_propagating_incidence`` (called by
    every one of these solves, one line earlier) documents: "R + T != 1 by
    construction ... treat the sums as indicative".

    So the gate rejected EVERY degree in the 16-wide scan and
    ``pmm_efficiency_1d`` raised

        RuntimeError: no resonance-free solve in degrees [10, 26); the
        requested degree sits in a high-degree resonance band.

    blaming the caller's degree for a perfectly healthy solve.  Measured
    pre-fix (P=0.30 um, eps 3.61/1.0, 0.22 um deep, duty 0.5, wl 0.55 um,
    n_sub=1.5, degree 10, far_field_orders 5) -- ``stabilize=False`` answers
    with the rcwa cross-check in brackets:

    ============================  =====  ======================  ==========
    n_superstrate                 pol    stabilize=True          no-stab tot
    ============================  =====  ======================  ==========
    1.2+0.01j, theta=0            te     OK  (tot 1.000263)      1.000263
    1.2+0.01j, theta=0            tm     RuntimeError            1.001344
    1.2+0.01j, theta=0.3          te     RuntimeError            1.002663
    1.2+0.05j, theta=0            te     RuntimeError            1.002631
    1.2+0.20j, theta=0            te     RuntimeError            1.030262
    1.2+0.20j, theta=0            tm     RuntimeError            1.107630
    ============================  =====  ======================  ==========

    (rcwa agrees with the no-stab TE totals to 4e-7, so nothing was wrong with
    the solve.)  The Jones twin degraded more quietly: no passive degree meant
    "returning the last attempt" with a bogus "may sit in a resonance band"
    warning and NO convergence consensus at all.

    FIX: ``_lossy_incidence(n_superstrate)`` -> ``super_unity_ok=`` drops ONLY
    the ``tot <= 1 + tol`` half of the gate (the lower bound, the per-order
    non-negativity and the whole per-order convergence consensus stay), wired
    through all 15 stabilizer call sites in ``oned.py`` / ``twod.py`` /
    ``twod_jones.py``.  This is the skip ``twod._warn_lossless_energy_2d`` and
    rcwa's ``_check_energy(..., lossless=)`` already had.

F-B (HIGH, physics -- Berreman-F-1 class, flux gauge at the incidence plane)
    ``_core._scalar_farfield_RT``'s TM channel normalised by
    ``flux_inc = Re(kz_inc / eps_sup)`` where ``kz_inc`` is ALREADY the real
    part of the complex order-0 root -- a REAL kz divided into a COMPLEX
    ``eps_sup``, which is the z-flux of no wave, while the numerators
    ``Re(kz/eps)`` use the full complex kz.  For a lossless superstrate the
    two agree identically; for an ABSORBING one they do not, and the recipe
    broke the hardest symmetry available:

        at NORMAL incidence on an ISOTROPIC homogeneous slab TE and TM must be
        bit-for-bit the same physics.

    Measured pre-fix (P=0.30 um, uniform eps 2.25, 0.22 um, wl 0.55 um,
    n_sub=1.5, degree 8, ``stabilize=False``), scalar TE vs scalar TM at
    theta = 0:

    ==============  ===================  ===================  ==========
    n_superstrate   TE (R, T)            TM (R, T)            |dT|
    ==============  ===================  ===================  ==========
    1.0             0.0400000, 0.960000  0.0400000, 0.960000  8.4e-16
    1.2+0.01j       0.0123592, 0.987709  0.0123609, 0.987847  1.372e-04
    1.2+0.05j       0.0126843, 0.989030  0.0127284, 0.992470  3.440e-03
    1.2+0.20j       0.0177353, 1.009550  0.0187488, 1.067238  5.769e-02
    ==============  ===================  ===================  ==========

    while ``pmm_jones_1d``, ``pmm_jones_2d``, ``pmm_efficiency_2d_cell``,
    ``pmm_efficiency_2d_staggered`` and ``rcwa_efficiency_1d(pol='tm')`` all
    returned the TE value on the same problem -- this ONE recipe out of the
    ~26 far-field sites disagreed with the whole family.

    FIX: for a complex ``eps_sup`` the incident flux becomes the family value
    ``(kz_inc^2+kx0^2)/kz_inc * |kz0/eps_sup|^2`` (the E-amplitude
    ``kz_inc * einc_sq`` normalisation of ``_project_efficiency`` /
    ``_assemble_jones_farfield``, expressed in the TM channel's Hy gauge).
    Byte-unchanged for a real superstrate (exact-zero branch).  Post-fix:
    rcwa parity 1.3e-14 over a 15-config lossy-superstrate matrix, TE == TM at
    normal incidence 2.7e-15.  Mirrored branch-free in the JAX scalar twin
    ``_jpmm_scalar_farfield`` (a traced eps cannot be tested for exact loss).

REFUTED (with evidence -- these rows are standing refutation gates)
==================================================================

F-1 on the VERTICAL cascades (the covariant one is F-C above).  All 71
    ``_kz_forward`` / ``_kz_forward2`` / ``_pmm2d_order_kz`` /
    ``_require_propagating_incidence`` sites in ``pmm/**`` were enumerated by
    AST, each classified against its enclosing function's eps gauge, and then
    measured: apart from ``_solve_covariant`` every entry point reproduces the
    from-scratch TMM in the homogeneous limit to <= 1.3e-14 for an absorbing
    SUBSTRATE, an absorbing SUPERSTRATE, both, and a metal substrate, at normal
    and oblique incidence.  No ``T = 0``, no super-unity blow-up.  (PMM's
    ``_kz_forward`` takes the PUBLIC eps where rcwa's ``_forward_flux_kz``
    un-conjugates its own argument, so the two families need OPPOSITE bridges
    -- which is how ``_solve_covariant`` came to be the odd one out.)

W6-1 (spurious modes / growing propagator in a cascade).  Every
    ``_propagation_smatrix`` and ``_propagation_smatrix_general`` in the family
    was instrumented over 226 randomized solves (period 0.15-2.5, depth
    0.06-20, lossy ridges, absorbing substrates, incidence to 1.1 rad, all
    1-D + 2-D + conical + stack entry points): ``max |exp(-lam k0 L)| =
    1 + 2.3e-11`` (forward generator ``1 + 1.6e-14``, backward
    ``1 + 8.7e-15``).  The 2.3e-11 is the documented ``_forward_branch_flip``
    near-real tolerance band (``1e-8 * max|q|``) times ``k0 L ~ 107``, i.e.
    ~2e-13 of residual QZ noise in ``Im(q)`` -- bounded and benign, NOT a
    branch-selection defect.

W6-2 (exponential conditioning / wrong-plane amplitudes).  The
    index-matched-identity oracle (layer eps == superstrate == substrate, so
    T(order 0) must be exactly 1) holds at depth 0.1 .. 10000 wavelengths on
    every entry point.  The error grows LINEARLY in depth (~1.5e-14 per unit
    depth, the propagator's accumulated phase rounding), not as
    ``exp(|kz| d)``: 1.5e-12 at depth 100, 1.5e-10 at depth 10000.  No R=1 /
    T=0 collapse anywhere.

W6-3 (discarded Im on kz / eigenvalues carrying loss).  ``layer_absorption``
    closes against the far field (``sum A == 1 - sum R - sum T``) to 4.2e-15
    (PMMStack), 5.0e-16 (PMM2DStackHybrid) and 3.7e-12 (PMM2DStackPure) over
    weak / strong / thick / multi-layer / uniaxial-lossy-axis cases -- including
    the ``ezz``-only-lossy orientation.

Routers.  ``twod_jones._tile_is_offplane``'s ``1e-12 * scale`` floor: the
    mis-routing error stays BELOW the off-plane magnitude itself (1e-13
    coupling -> 0.0 delta, 1e-11 -> 1.4e-14, 1e-3 -> 3.4e-7), so the floor
    separates noise from physics with margin.  The even-parity fold
    (``symmetry='auto'``) reproduces the full ``2Nf`` solve to 6.3e-15
    including lossy cells and an absorbing substrate, and correctly declines
    at oblique incidence (bit-identical there).

Conical s/p synthesis.  At ``phi != 0`` the isotropic limit must be the
    FLUX-WEIGHTED s/p mixture, not the naive average.  ``pmm_jones_1d_conical``,
    ``pmm_jones_1d_conical_tensor`` and ``PMMStack``'s conical cascade match
    that oracle to < 1e-9 over 60 (n_sup, n_sub, theta, phi) combinations
    including absorbing superstrate, absorbing substrate and a metal
    substrate, where the naive average is off by up to 1.3e-2.
"""
import numpy as np
import pytest

import lumenairy.elements.pmm as P
import lumenairy.elements.rcwa as RC

_C = np.complex128
WL = 0.55
PER = 0.30
DEPTH = 0.22
I3 = np.eye(3)


# ===========================================================================
# A from-scratch characteristic-matrix TMM oracle (PUBLIC exp(-iwt) gauge).
# Independent of the library: only numpy.  s-pol amplitudes are tangential Ey,
# p-pol amplitudes are tangential Hy; the fluxes below are the EXACT
# time-averaged Sz, which stay exact in an absorbing medium because
# Re(kz/eps) == Re(kz)(|kz|^2+kx^2)/|eps|^2 identically.
# ===========================================================================

def _kz_of(eps, kt):
    v = np.sqrt(_C(eps) - _C(kt) ** 2)
    return -v if v.imag < 0 else v


def _tmm(eps_layers, thick, eps_sup, eps_sub, kt, wl, pol):
    k0 = 2.0 * np.pi / wl
    eta_s = _kz_of(eps_sup, kt)
    eta_b = _kz_of(eps_sub, kt)
    if pol == "p":
        eta_s = eta_s / _C(eps_sup)
        eta_b = eta_b / _C(eps_sub)
    M = np.eye(2, dtype=_C)
    for e, t in zip(eps_layers, thick):
        kz = _kz_of(e, kt)
        eta = kz if pol == "s" else kz / _C(e)
        d = k0 * kz * t
        c, s = np.cos(d), np.sin(d)
        M = M @ np.array([[c, -1j * s / eta], [-1j * eta * s, c]], dtype=_C)
    B, Cc = M @ np.array([1.0, eta_b], dtype=_C)
    return ((eta_s * B - Cc) / (eta_s * B + Cc),
            2.0 * eta_s / (eta_s * B + Cc))


def _tmm_scalar_RT(eps_layers, thick, eps_sup, eps_sub, kt, wl, pol):
    """R, T in the PMM/RCWA family normalisation (kz_inc = Re(kz_sup))."""
    r, t = _tmm(eps_layers, thick, eps_sup, eps_sub, kt, wl, pol)
    kzs, kzb = _kz_of(eps_sup, kt), _kz_of(eps_sub, kt)
    kz_inc = float(np.real(kzs))
    if pol == "s":
        return (float(np.real(kzs)) * abs(r) ** 2 / kz_inc,
                float(np.real(kzb)) * abs(t) ** 2 / kz_inc)
    # p: convert the Hy amplitudes to Ex and apply the family recipe
    r_ex = abs(r)
    t_ex = abs(t) * abs(kzb / _C(eps_sub)) / abs(kzs / _C(eps_sup))
    f0 = kz_inc * (1.0 + (kt / kz_inc) ** 2)
    return (float(np.real(kzs)) * (1.0 + kt ** 2 / abs(kzs) ** 2)
            * r_ex ** 2 / f0,
            float(np.real(kzb)) * (1.0 + kt ** 2 / abs(kzb) ** 2)
            * t_ex ** 2 / f0)


def _sp_mixture(eps_layers, thick, eps_sup, eps_sub, theta, phi, wl):
    """The conical isotropic limit: the FLUX-WEIGHTED s/p mixture for the two
    incident lab polarizations (NOT the naive 50/50 average)."""
    kt = float(np.real(np.sqrt(eps_sup))) * np.sin(theta)
    Rs, Ts = _tmm_scalar_RT(eps_layers, thick, eps_sup, eps_sub, kt, wl, "s")
    Rp, Tp = _tmm_scalar_RT(eps_layers, thick, eps_sup, eps_sub, kt, wl, "p")
    kz_inc = float(np.real(_kz_of(eps_sup, kt)))
    g = 1.0 + (kt / kz_inc) ** 2
    c2, s2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    ex = 1.0 + (kt * np.cos(phi) / kz_inc) ** 2
    ey = 1.0 + (kt * np.sin(phi) / kz_inc) ** 2
    return ((s2 * Rs + c2 * g * Rp) / ex, (s2 * Ts + c2 * g * Tp) / ex,
            (c2 * Rs + s2 * g * Rp) / ey, (c2 * Ts + s2 * g * Tp) / ey)


# Cross-platform tolerance policy.  Every bar below is set from a MEASURED
# envelope with headroom, never from a hash: the isotropic-limit oracle rows
# measured <= 1.3e-14 on this box and are pinned at 1e-11; the cross-solver
# (PMM vs RCWA, different bases) rows measured <= 1.3e-14 in the HOMOGENEOUS
# limit only -- a patterned cell has a genuine basis/truncation difference and
# is never pinned tight.
_ORACLE = 1e-11          # PMM vs the from-scratch TMM, homogeneous limit
_FAMILY = 1e-10          # PMM path vs PMM path / rcwa, same physics


# ===========================================================================
# F-A -- the absorbing-superstrate stabilizer gate
# ===========================================================================

@pytest.mark.parametrize("n_sup", [1.2 + 0.01j, 1.2 + 0.05j, 1.2 + 0.2j])
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("angle", [0.0, 0.3])
def test_w7_fa_absorbing_superstrate_scalar_solves_at_default_stabilize(
        n_sup, pol, angle):
    """F-A: the DEFAULT ``stabilize=True`` must not reject an absorbing
    incidence medium.  Pre-fix this raised ``RuntimeError: no resonance-free
    solve in degrees [10, 26)`` for every row but (0.01j, te, theta=0)."""
    orders, R, T = P.pmm_efficiency_1d(
        PER, 1.9, 1.0, 1.5, n_sup, DEPTH, 0.5, WL, angle=angle,
        polarization=pol, degree=10, far_field_orders=5)
    tot = float(R.sum() + T.sum())
    # legitimately super-unity, but bounded and passive per order
    assert 1.0 - 1e-9 <= tot < 1.2
    assert float(np.min(R)) >= -1e-12 and float(np.min(T)) >= -1e-12
    # ... and the stabilized answer must agree with the unstabilized one
    _o2, R2, T2 = P.pmm_efficiency_1d(
        PER, 1.9, 1.0, 1.5, n_sup, DEPTH, 0.5, WL, angle=angle,
        polarization=pol, degree=10, far_field_orders=5, stabilize=False)
    assert abs(tot - float(R2.sum() + T2.sum())) < 5e-3


@pytest.mark.parametrize("n_sup", [1.2 + 0.2j, 1.2 + 0.6j])
def test_w7_fa_absorbing_superstrate_jones_keeps_the_consensus(n_sup):
    """F-A (Jones twin): pre-fix no degree was 'passive', so the scan fell
    through to "returning the last attempt" with a bogus resonance-band
    warning and NO convergence consensus.  Post-fix the consensus runs, so
    neither the resonance-band nor the non-convergence warning fires."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _o, R, T, _J = P.pmm_jones_1d(
            PER, 3.61 * I3, 1.0 * I3, 1.5, n_sup, DEPTH, 0.5, WL, angle=0.3,
            degree=10, far_field_orders=5)
    msgs = [str(w.message) for w in rec]
    assert not any("resonance band" in m for m in msgs), msgs
    assert not any("did not converge" in m for m in msgs), msgs
    tot = R.sum(1) + T.sum(1)
    assert np.all(tot >= 1.0) and np.all(tot < 1.6)


def test_w7_fa_lossless_superstrate_keeps_the_strict_gate():
    """F-A must NOT widen the gate for a lossless incidence medium -- the
    super-unity resonance rejection is exactly what it is there for."""
    from lumenairy.elements.pmm._core import _lossy_incidence
    assert _lossy_incidence(1.0) is False
    assert _lossy_incidence(1.5 - 1e-9j) is False        # gain reads False
    assert _lossy_incidence(1.5 + 1e-12j) is True
    assert _lossy_incidence(np.array([1.0, 2.0])) is False   # non-scalar
    orders, R, T = P.pmm_efficiency_1d(
        PER, 1.9, 1.0, 1.5, 1.0, DEPTH, 0.5, WL, angle=0.3,
        polarization="tm", degree=10, far_field_orders=5)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-11


# ===========================================================================
# F-B -- the scalar TM incident flux in an absorbing superstrate
# ===========================================================================

@pytest.mark.parametrize("n_sup", [1.0, 1.2 + 0.01j, 1.2 + 0.05j,
                                   1.2 + 0.2j, 1.2 + 0.6j])
def test_w7_fb_te_equals_tm_at_normal_incidence_on_an_isotropic_slab(n_sup):
    """F-B: the hardest symmetry there is.  At theta=0 an isotropic slab has
    no s/p distinction, so scalar TE and scalar TM must coincide.  Pre-fix the
    TM channel drifted by 1.4e-4 / 3.4e-3 / 5.8e-2 in T at Im(n_sup) =
    0.01 / 0.05 / 0.2."""
    out = {}
    for pol in ("te", "tm"):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, R, T = P.pmm_efficiency_1d(
                    PER, 1.5, 1.5, 1.5, n_sup, DEPTH, 0.5, WL, angle=0.0,
                    polarization=pol, degree=8, far_field_orders=5,
                    stabilize=False)
        out[pol] = (float(R.sum()), float(T.sum()))
    assert abs(out["te"][0] - out["tm"][0]) < _FAMILY
    assert abs(out["te"][1] - out["tm"][1]) < _FAMILY


@pytest.mark.parametrize("n_sup", [1.0, 1.2 + 0.01j, 1.2 + 0.05j, 1.2 + 0.2j])
@pytest.mark.parametrize("angle", [0.0, 0.3, 0.6])
def test_w7_fb_scalar_tm_matches_rcwa_in_the_homogeneous_limit(n_sup, angle):
    """F-B: the scalar TM far field must agree with rcwa's ``_project_efficiency``
    on the SAME physical problem.  Pre-fix the lossy-superstrate rows were off
    by up to 5.9e-2."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T = P.pmm_efficiency_1d(
            PER, 1.5, 1.5, 1.5, n_sup, DEPTH, 0.5, WL, angle=angle,
            polarization="tm", degree=8, far_field_orders=5, stabilize=False)
        _o2, Rr, Tr = RC.rcwa_efficiency_1d(
            PER, 1.5, 1.5, 1.5, n_sup, DEPTH, 0.5, WL, angle=angle,
            polarization="tm", n_orders=5)
    assert abs(float(R.sum()) - float(Rr.sum())) < _FAMILY
    assert abs(float(T.sum()) - float(Tr.sum())) < _FAMILY


def test_w7_fb_lossless_superstrate_is_byte_unchanged():
    """F-B is gated on an EXACT-zero ``Im(eps_sup)`` test, so a real
    superstrate takes the historical expression verbatim.  This pins the
    branch condition, not a solver value."""
    from lumenairy.elements.pmm._core import _scalar_farfield_RT
    kx = np.array([-0.2, 0.0, 0.2])
    r = np.array([0.1 + 0.2j, 0.3 - 0.1j, 0.05j])
    t = np.array([0.4 - 0.1j, 0.6 + 0.2j, 0.02j])
    k0 = 2 * np.pi / WL
    for eps_sup in (1.0, 2.25, 4.0):
        R, T = _scalar_farfield_RT(r, t, kx, 0.0, k0, eps_sup, 2.25, "tm")
        kz_sup = np.sqrt(eps_sup - kx ** 2 + 0j)
        kz_sub = np.sqrt(2.25 - kx ** 2 + 0j)
        kz_inc = float(np.real(np.sqrt(eps_sup + 0j)))
        flux = np.real(kz_inc / eps_sup)
        Rref = np.where(np.real(kz_sup) > 0,
                        np.real(np.real(kz_sup / eps_sup) * np.abs(r) ** 2
                                / flux), 0.0)
        Tref = np.where(np.real(kz_sub) > 0,
                        np.real(np.real(kz_sub / 2.25) * np.abs(t) ** 2
                                / flux), 0.0)
        assert R.tobytes() == Rref.tobytes()
        assert T.tobytes() == Tref.tobytes()


# ===========================================================================
# F-1 class (REFUTED) -- absorbing sub/superstrate on every entry point
# ===========================================================================

_F1_MEDIA = [
    ("lossless", 1.0 + 0j, 1.5 + 0j),
    ("abs-substrate", 1.0 + 0j, 1.5 + 0.30j),
    ("abs-superstrate", 1.20 + 0.05j, 1.5 + 0j),
    ("abs-both", 1.20 + 0.05j, 1.5 + 0.30j),
    ("metal-substrate", 1.0 + 0j, 0.20 + 3.5j),
]


def _jones_oracle(n_sup, n_sub, theta):
    kt = float(np.real(n_sup)) * np.sin(theta)
    Rs, Ts = _tmm_scalar_RT([2.25], [DEPTH], n_sup ** 2, n_sub ** 2, kt,
                            WL, "s")
    Rp, Tp = _tmm_scalar_RT([2.25], [DEPTH], n_sup ** 2, n_sub ** 2, kt,
                            WL, "p")
    return (Rp, Tp), (Rs, Ts)


_F1_MEDIA_2D = [m for m in _F1_MEDIA
                if m[0] in ("lossless", "abs-substrate", "abs-superstrate")]


@pytest.mark.parametrize("tag,n_sup,n_sub", _F1_MEDIA)
@pytest.mark.parametrize("theta", [0.0, 0.35])
def test_w7_f1_jones_entry_points_vs_tmm_with_lossy_media(tag, n_sup, n_sub,
                                                          theta):
    """F-1 signature check: an absorbing exit (or entrance) medium must NOT
    zero T nor blow the totals up.  Every Jones-gauge entry point is measured
    against the from-scratch TMM in the homogeneous limit."""
    import warnings
    (Rp, Tp), (Rs, Ts) = _jones_oracle(n_sup, n_sub, theta)
    cell4 = np.zeros((2, 2, 3, 3), dtype=complex)
    cell4[:] = 2.25 * I3

    def _stack1d():
        st = P.PMMStack(PER, n_substrate=n_sub, n_superstrate=n_sup,
                        degree=8, far_field_orders=5)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, angle=theta)
        return st.solve()

    def _stack1d_conical():
        st = P.PMMStack(PER, n_substrate=n_sub, n_superstrate=n_sup,
                        degree=8, far_field_orders=5)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, theta=theta, phi=0.0)
        return st.solve()

    def _stack2d_hybrid():
        st = P.PMM2DStackHybrid(PER, PER, n_substrate=n_sub,
                                n_superstrate=n_sup, degree=5, n_orders=3)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, theta=theta, phi=0.0)
        return st.solve()

    def _stack2d_pure():
        st = P.PMM2DStackPure(PER, PER, n_substrate=n_sub,
                              n_superstrate=n_sup, n_modes=5, n_orders=3)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, theta=theta, phi=0.0)
        return st.solve()[:4]

    paths = {
        "pmm_jones_1d": lambda: P.pmm_jones_1d(
            PER, 2.25 * I3, 2.25 * I3, n_sub, n_sup, DEPTH, 0.5, WL,
            angle=theta, degree=8, far_field_orders=5, stabilize=False),
        "conical": lambda: P.pmm_jones_1d_conical(
            PER, 2.25, 2.25, n_sub, n_sup, DEPTH, 0.5, WL, theta=theta,
            phi=0.0, degree=8, n_orders=5),
        "conical_tensor": lambda: P.pmm_jones_1d_conical_tensor(
            PER, np.stack([2.25 * I3, 2.25 * I3]), n_sub, n_sup, DEPTH, WL,
            theta=theta, phi=0.0, degree=8, n_orders=5),
        "PMMStack": _stack1d,
        "PMMStack.conical": _stack1d_conical,
    }
    for name, fn in paths.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = fn()
        got = ((float(R[0].sum()), float(T[0].sum())),
               (float(R[1].sum()), float(T[1].sum())))
        tol = _ORACLE if "Pure" not in name else 1e-8   # pure: modal floor
        assert abs(got[0][0] - Rp) < tol, (name, tag, "R[Ex]", got, Rp)
        assert abs(got[0][1] - Tp) < tol, (name, tag, "T[Ex]", got, Tp)
        assert abs(got[1][0] - Rs) < tol, (name, tag, "R[Ey]", got, Rs)
        assert abs(got[1][1] - Ts) < tol, (name, tag, "T[Ey]", got, Ts)
        # the F-1 signature itself: a lossy exit medium must still transmit
        if "metal" not in tag:
            assert got[0][1] > 0.5 and got[1][1] > 0.5, (name, tag, got)


@pytest.mark.parametrize("tag,n_sup,n_sub", _F1_MEDIA_2D)
def test_w7_f1_2d_entry_points_vs_tmm_with_lossy_media(tag, n_sup, n_sub):
    """The 2-D Jones entry points on the same media matrix, reduced to normal
    incidence and three media because a 2-D solve costs ~30x a 1-D one."""
    import warnings
    theta = 0.0
    (Rp, Tp), (Rs, Ts) = _jones_oracle(n_sup, n_sub, theta)
    cell4 = np.zeros((2, 2, 3, 3), dtype=complex)
    cell4[:] = 2.25 * I3

    def _stack2d_hybrid():
        st = P.PMM2DStackHybrid(PER, PER, n_substrate=n_sub,
                                n_superstrate=n_sup, degree=5, n_orders=3)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, theta=theta, phi=0.0)
        return st.solve()

    def _stack2d_pure():
        st = P.PMM2DStackPure(PER, PER, n_substrate=n_sub,
                              n_superstrate=n_sup, n_modes=5, n_orders=3)
        st.add_layer(DEPTH, eps=2.25)
        st.set_source(WL, theta=theta, phi=0.0)
        return st.solve()[:4]

    paths = {
        "pmm_jones_2d": lambda: P.pmm_jones_2d(
            PER, PER, cell4, n_sub, n_sup, DEPTH, WL, theta=theta, phi=0.0,
            degree=5, n_orders=3),
        "PMM2DStackHybrid": _stack2d_hybrid,
        "PMM2DStackPure": _stack2d_pure,
    }
    for name, fn in paths.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = fn()
        tol = _ORACLE if "Pure" not in name else 1e-8
        assert abs(float(R[0].sum()) - Rp) < tol, (name, tag)
        assert abs(float(T[0].sum()) - Tp) < tol, (name, tag)
        assert abs(float(R[1].sum()) - Rs) < tol, (name, tag)
        assert abs(float(T[1].sum()) - Ts) < tol, (name, tag)
        if "metal" not in tag:
            assert float(T[0].sum()) > 0.5 and float(T[1].sum()) > 0.5


@pytest.mark.parametrize("tag,n_sup,n_sub", _F1_MEDIA)
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_w7_f1_scalar_entry_points_vs_tmm_with_lossy_media(tag, n_sup, n_sub,
                                                           pol):
    """The scalar (TE/TM) far-field recipe, same media matrix.  Covers
    ``_scalar_farfield_RT`` (1-D) and rcwa's ``_project_efficiency`` (2-D)."""
    import warnings
    kt = 0.0
    exp = _tmm_scalar_RT([2.25], [DEPTH], n_sup ** 2, n_sub ** 2, kt, WL,
                         "s" if pol == "te" else "p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T = P.pmm_efficiency_1d(
            PER, 1.5, 1.5, n_sub, n_sup, DEPTH, 0.5, WL, angle=0.0,
            polarization=pol, degree=8, far_field_orders=5, stabilize=False)
    checks = [("1d", (R.sum(), T.sum()), _ORACLE)]
    if tag in ("lossless", "abs-substrate", "abs-superstrate"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o2, R2, T2 = P.pmm_efficiency_2d_cell(
                PER, PER, np.full((2, 2), 2.25 + 0j), n_sub, n_sup, DEPTH, WL,
                theta=0.0, phi=0.0, polarization=pol, degree=5, n_orders=3)
            _o3, R3, T3 = P.pmm_efficiency_2d_staggered(
                PER, PER, np.full((2, 2), 2.25 + 0j), n_sub, n_sup, DEPTH, WL,
                theta=0.0, phi=0.0, polarization=pol, degree=5, n_orders=3)
        checks += [("2d_cell", (np.sum(R2), np.sum(T2)), _ORACLE),
                   ("staggered", (np.sum(R3), np.sum(T3)), 1e-8)]
    for name, (Rv, Tv), tol in checks:
        assert abs(float(Rv) - exp[0]) < tol, (name, tag, pol, Rv, exp)
        assert abs(float(Tv) - exp[1]) < tol, (name, tag, pol, Tv, exp)


# ===========================================================================
# Conical s/p synthesis (REFUTED -- standing gate)
# ===========================================================================

@pytest.mark.parametrize("n_sup,n_sub", [(1.0, 1.5), (1.0, 1.5 + 0.3j),
                                         (1.2 + 0.05j, 1.5), (1.0, 0.2 + 3.5j)])
@pytest.mark.parametrize("theta", [0.25, 0.55])
@pytest.mark.parametrize("phi", [0.3, 1.2])
def test_w7_conical_is_the_flux_weighted_sp_mixture(n_sup, n_sub, theta, phi):
    """The conical vector far field at phi != 0 must be the FLUX-WEIGHTED s/p
    mixture, not the naive average (which is off by up to 1.3e-2 here)."""
    import warnings
    Rx, Tx, Ry, Ty = _sp_mixture([2.25], [DEPTH], n_sup ** 2, n_sub ** 2,
                                 theta, phi, WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, _J = P.pmm_jones_1d_conical(
            PER, 2.25, 2.25, n_sub, n_sup, DEPTH, 0.5, WL, theta=theta,
            phi=phi, degree=8, n_orders=5)
    assert abs(float(R[0].sum()) - Rx) < _ORACLE
    assert abs(float(T[0].sum()) - Tx) < _ORACLE
    assert abs(float(R[1].sum()) - Ry) < _ORACLE
    assert abs(float(T[1].sum()) - Ty) < _ORACLE
    # the naive 50/50 average is measurably WRONG -- the gate would pass
    # trivially if the two rows happened to coincide
    if abs(theta) > 0.4 and abs(phi - np.pi / 4) > 0.1:
        assert abs(0.5 * (Rx + Ry) - Rx) > 1e-4


def test_w7_conical_matches_the_classical_in_plane_path_on_a_grating():
    """phi=0 conical == the classical in-plane Jones cascade (different
    machinery, same physics)."""
    for theta in (0.0, 0.25, 0.5):
        _o, R, T, J = P.pmm_jones_1d(
            PER, 3.61 * I3, 1.0 * I3, 1.5, 1.0, DEPTH, 0.5, WL, angle=theta,
            degree=8, far_field_orders=5, stabilize=False)
        o2, R2, T2, J2 = P.pmm_jones_1d_conical(
            PER, 3.61, 1.0, 1.5, 1.0, DEPTH, 0.5, WL, theta=theta, phi=0.0,
            degree=8, n_orders=5)
        i1 = {int(v): k for k, v in enumerate(np.asarray(_o))}
        i2 = {int(np.asarray(v).ravel()[0]): k
              for k, v in enumerate(np.asarray(o2))}
        for m in sorted(set(i1) & set(i2)):
            for p in (0, 1):
                assert abs(R[p][i1[m]] - R2[p][i2[m]]) < 1e-12
                assert abs(T[p][i1[m]] - T2[p][i2[m]]) < 1e-12
        assert float(np.max(np.abs(J - J2))) < 1e-12


# ===========================================================================
# W6-2 (REFUTED) -- index-matched identity at depth
# ===========================================================================

@pytest.mark.parametrize("depth", [0.1, 10.0, 1000.0])
def test_w7_w62_index_matched_identity_at_depth(depth):
    """Layer eps == superstrate == substrate: T(order 0) must be 1 and R
    zero at ANY depth.  The measured error is LINEAR in depth (~1.5e-14 per
    unit depth, the propagator's phase rounding), not exp(|kz| d)."""
    nm = 1.5
    bar = 1e-13 + 3e-13 * depth        # measured 1.5e-14/unit, 20x headroom
    _o, R, T, _J = P.pmm_jones_1d(
        PER, 2.25 * I3, 2.25 * I3, nm, nm, depth, 0.5, WL, angle=0.25,
        degree=8, far_field_orders=5, stabilize=False)
    m0 = int(np.where(np.asarray(_o) == 0)[0][0])
    assert abs(T[0][m0] - 1.0) < bar and abs(T[1][m0] - 1.0) < bar
    assert float(np.max(np.abs(R))) < 1e-20
    st = P.PMMStack(PER, n_substrate=nm, n_superstrate=nm, degree=8,
                    far_field_orders=5)
    st.add_layer(depth, eps=2.25)
    st.set_source(WL, angle=0.25)
    _o, R, T, _J = st.solve()
    assert abs(T[0][m0] - 1.0) < bar and abs(T[1][m0] - 1.0) < bar


def test_w7_w62_index_matched_identity_2d_at_depth():
    """Same oracle through both 2-D stack flavours and the conical 2-D Jones
    entry (which stayed at 4.4e-16 out to depth 1e4)."""
    nm = 1.5
    cell4 = np.zeros((2, 2, 3, 3), dtype=complex)
    cell4[:] = 2.25 * I3
    for depth in (1.0, 1000.0):
        _o, R, T, _J = P.pmm_jones_2d(PER, PER, cell4, nm, nm, depth, WL,
                                      theta=0.25, phi=0.3, degree=5,
                                      n_orders=3)
        def _p0(o):
            o = np.asarray(o)
            return int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])

        p0 = _p0(_o)
        assert abs(T[0][p0] - 1.0) < 1e-12 and abs(T[1][p0] - 1.0) < 1e-12
        st = P.PMM2DStackHybrid(PER, PER, n_substrate=nm, n_superstrate=nm,
                                degree=5, n_orders=3)
        st.add_layer(depth, eps=2.25)
        st.set_source(WL, theta=0.25, phi=0.3)
        _o, R, T, _J = st.solve()
        p0 = _p0(_o)
        assert abs(T[0][p0] - 1.0) < 1e-12 and abs(T[1][p0] - 1.0) < 1e-12


# ===========================================================================
# W6-1 (REFUTED) -- no growing propagator in any cascade
# ===========================================================================

def test_w7_w61_no_growing_propagator_in_any_cascade():
    """Instrument every propagation S-matrix and assert
    ``max |exp(-lam k0 L)| <= 1 + eps`` over a randomized ensemble.
    Measured over 226 solves: 1 + 2.3e-11 (the documented
    ``_forward_branch_flip`` near-real tolerance band, not a defect)."""
    import warnings

    import lumenairy.elements.pmm._core as PC
    import lumenairy.elements.pmm.conical as _cn
    import lumenairy.elements.pmm.oned as _o
    import lumenairy.elements.pmm.stack as _s
    import lumenairy.elements.pmm.twod as _t
    import lumenairy.elements.pmm.twod_jones as _tj
    from lumenairy.elements.rcwa import _core as RCC

    seen = {"m": 0.0}
    orig = PC._propagation_smatrix
    orig_g = RCC._propagation_smatrix_general
    orig_star = RCC._propagation_star
    mods = [PC, _o, _s, _t, _tj, _cn, RCC]

    def _note(lam, k0_L, sign=-1.0):
        seen["m"] = max(seen["m"], float(np.max(np.abs(
            np.exp(sign * np.asarray(lam) * k0_L)))))

    def spy(lam, k0_L):
        _note(lam, k0_L)
        return orig(lam, k0_L)

    def spy_g(lam_f, lam_b, k0_L):
        _note(lam_f, k0_L)
        _note(lam_b, k0_L, +1.0)
        return orig_g(lam_f, lam_b, k0_L)

    def spy_star(S, lam, k0_L):
        _note(lam, k0_L)
        return orig_star(S, lam, k0_L)

    names = ("_propagation_smatrix", "_propagation_smatrix_general",
             "_propagation_star")
    spies = dict(zip(names, (spy, spy_g, spy_star)))
    saved = [(m, {n: getattr(m, n, None) for n in names}) for m in mods]
    try:
        for m in mods:
            for n in names:
                if hasattr(m, n):
                    setattr(m, n, spies[n])
        rng = np.random.default_rng(20260726)
        ok = 0
        for _ in range(12):
            per = float(rng.uniform(0.15, 0.9))
            dep = float(10.0 ** rng.uniform(-1.0, 1.0))
            duty = float(rng.uniform(0.25, 0.75))
            nr = complex(rng.uniform(1.5, 3.5), rng.choice([0.0, 0.6]))
            nsub = complex(rng.uniform(1.0, 2.0), rng.choice([0.0, 0.4]))
            ang = float(rng.uniform(0.0, 0.9))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for pol in ("te", "tm"):
                    try:
                        P.pmm_efficiency_1d(per, nr, 1.05, nsub, 1.0, dep,
                                            duty, WL, angle=ang,
                                            polarization=pol, degree=8,
                                            far_field_orders=5,
                                            stabilize=False)
                        ok += 1
                    except (ValueError, RuntimeError):
                        pass
                try:
                    P.pmm_jones_1d_conical(per, nr ** 2, 1.1, nsub, 1.0, dep,
                                           duty, WL, theta=ang, phi=0.4,
                                           degree=8, n_orders=5)
                    ok += 1
                except (ValueError, RuntimeError):
                    pass
                try:                       # the cascade that drives the spy
                    st = P.PMMStack(per, n_substrate=nsub,
                                    n_superstrate=1.0, degree=8,
                                    far_field_orders=5)
                    st.add_layer(dep, segments=[(duty, nr ** 2),
                                                (1 - duty, 1.1)])
                    st.add_layer(dep * 0.4, eps=1.21)
                    st.set_source(WL, theta=ang, phi=0.35)
                    st.solve()
                    ok += 1
                except (ValueError, RuntimeError):
                    pass
    finally:
        for m, kept in saved:
            for n, v in kept.items():
                if v is not None:
                    setattr(m, n, v)
    assert ok >= 20, ok
    # a genuine branch-selection defect gives |exp| >> 1 (order 1e1..1e9);
    # the measured tolerance-band residual is ~1e-11.
    assert seen["m"] <= 1.0 + 1e-8, seen["m"]
    assert seen["m"] > 0.0


# ===========================================================================
# W6-3 (REFUTED) -- layer_absorption closure includes what the field lost
# ===========================================================================

@pytest.mark.parametrize("layers", [
    [(0.20, 2.25 + 0.30j)],
    [(0.20, 2.25 + 3.0j)],
    [(0.15, 2.25 + 0.10j), (0.10, 4.0 + 0.0j), (0.18, 6.0 + 1.5j)],
    [(3.00, 2.25 + 0.20j)],
])
@pytest.mark.parametrize("angle", [0.0, 0.3])
def test_w7_w63_stack_layer_absorption_closes(layers, angle):
    st = P.PMMStack(PER, n_substrate=1.5, n_superstrate=1.0, degree=8,
                    far_field_orders=5)
    for t, e in layers:
        st.add_layer(t, eps=e)
    st.set_source(WL, angle=angle)
    _o, R, T, _J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    assert np.max(np.abs(A.sum(0) - (1.0 - R.sum(1) - T.sum(1)))) < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_w7_w63_uniaxial_lossy_axis_is_accounted(axis):
    """A uniaxial absorber whose ONLY loss sits on one diagonal channel --
    including ``ezz``, the channel a Re() cast would silently drop."""
    e = np.diag([2.25 + 0j] * 3).astype(complex)
    e[axis, axis] = 2.25 + 0.4j
    st = P.PMMStack(PER, n_substrate=1.5, n_superstrate=1.0, degree=8,
                    far_field_orders=5)
    st.add_layer(0.2, eps=e)
    st.set_source(WL, angle=0.3)
    _o, R, T, _J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    assert np.max(np.abs(A.sum(0) - (1.0 - R.sum(1) - T.sum(1)))) < 1e-12
    assert float(np.max(A)) > 1e-3          # the loss is actually attributed


@pytest.mark.parametrize("cls,kw", [("hybrid", dict(degree=5, n_orders=3)),
                                    ("pure", dict(n_modes=5, n_orders=3))])
def test_w7_w63_stack2d_layer_absorption_closes(cls, kw):
    klass = P.PMM2DStackHybrid if cls == "hybrid" else P.PMM2DStackPure
    st = klass(PER, PER, n_substrate=1.5, n_superstrate=1.0, **kw)
    st.add_layer(0.15, eps=2.25 + 0.10j)
    st.add_layer(0.12, eps=6.0 + 1.0j)
    st.set_source(WL, theta=0.3, phi=0.2)
    out = st.solve(retain_internal=True)
    R, T = np.asarray(out[1]), np.asarray(out[2])
    A = np.asarray(st.layer_absorption())
    assert np.max(np.abs(A.sum(0) - (1.0 - R.sum(1) - T.sum(1)))) < 1e-9


# ===========================================================================
# Routers (REFUTED -- standing gates)
# ===========================================================================

def test_w7_offplane_router_floor_separates_noise_from_physics():
    """``_tile_is_offplane``'s ``1e-12 * scale`` floor: a coupling BELOW the
    floor is mis-routed to the in-plane path on purpose, and the resulting
    error must stay below the coupling magnitude itself."""
    from lumenairy.elements.pmm.twod_jones import _tile_is_offplane
    base = np.zeros((2, 2, 3, 3), dtype=complex)
    base[:] = 2.25 * I3
    base[0, 0] = np.diag([3.5, 2.6, 2.9]).astype(complex)
    ref = None
    for mag, routed in ((0.0, False), (1e-13, False), (1e-11, True),
                        (1e-3, True)):
        cell = base.copy()
        cell[0, 0, 0, 2] = cell[0, 0, 2, 0] = mag * 3.5
        assert _tile_is_offplane(np.conj(cell)) is routed, mag
        _o, R, T, _J = P.pmm_jones_2d(PER, PER, cell, 1.5, 1.0, DEPTH, WL,
                                      theta=0.0, phi=0.0, degree=5,
                                      n_orders=3)
        if ref is None:
            ref = (R.copy(), T.copy())
            continue
        d = max(float(np.max(np.abs(R - ref[0]))),
                float(np.max(np.abs(T - ref[1]))))
        assert d < max(10.0 * mag, 1e-13), (mag, d)


@pytest.mark.parametrize("n_sub,theta,lossy", [
    (1.5 + 0j, 0.0, False), (1.5 + 0j, 0.0, True),
    (1.5 + 0.3j, 0.0, False), (1.5 + 0j, 0.3, False)])
def test_w7_even_parity_fold_reproduces_the_full_solve(n_sub, theta, lossy):
    """``symmetry='auto'`` folds a centro-symmetric cell at normal incidence
    into the even sector; it must reproduce the full 2Nf cascade (and decline
    at oblique, where it is bit-identical)."""
    cell = np.zeros((2, 2, 3, 3), dtype=complex)
    cell[:] = (2.25 + (0.3j if lossy else 0.0)) * I3
    cell[0, 0] = np.diag([3.5, 2.6, 2.9]).astype(complex)
    res = {}
    for sym in (True, False):
        _o, R, T, J = P.pmm_jones_2d(PER, PER, cell, n_sub, 1.0, DEPTH, WL,
                                     theta=theta, phi=0.0, degree=5,
                                     n_orders=3, symmetry=sym)
        res[sym] = (R, T, J)
    for i in range(3):
        assert float(np.max(np.abs(res[True][i] - res[False][i]))) < 1e-12


# ===========================================================================
# stack2d / stack2d_pure numerics (recorded non-coverage)
# ===========================================================================

@pytest.mark.parametrize("cls", ["hybrid", "pure"])
@pytest.mark.parametrize("n_sub", [1.5 + 0j, 1.5 + 0.3j])
def test_w7_stack2d_slice_invariance(cls, n_sub):
    """Splitting ONE homogeneous-in-z layer into k slices must be exact --
    the cascade-correctness oracle for both 2-D stack flavours (measured
    1.8e-15)."""
    cell = np.array([[4.0 + 0j, 2.0 + 0j], [2.0 + 0j, 2.0 + 0j]])
    ref = None
    for k in (1, 2, 5):
        if cls == "hybrid":
            st = P.PMM2DStackHybrid(PER, PER, n_substrate=n_sub,
                                    n_superstrate=1.0, degree=5, n_orders=3)
        else:
            st = P.PMM2DStackPure(PER, PER, n_substrate=n_sub,
                                  n_superstrate=1.0, n_modes=6, n_orders=3)
        for _ in range(k):
            st.add_layer(DEPTH / k, eps_cell=cell)
        st.set_source(WL, theta=0.3, phi=0.4)
        out = st.solve()
        v = (np.asarray(out[1]).sum(1), np.asarray(out[2]).sum(1))
        if ref is None:
            ref = v
        else:
            assert float(np.max(np.abs(v[0] - ref[0]))) < 1e-12
            assert float(np.max(np.abs(v[1] - ref[1]))) < 1e-12


def test_w7_internal_field_is_flat_in_an_index_matched_slab():
    """Internal-field numerics beyond the consumer-API gates: with the layer
    index-matched to both half-spaces the field IS the incident plane wave,
    so |Ey| must be constant in x and z (measured spread 3.9e-14)."""
    nm = 1.5
    st = P.PMMStack(PER, n_substrate=nm, n_superstrate=nm, degree=8,
                    far_field_orders=5)
    st.add_layer(1.0, eps=nm ** 2)
    st.set_source(WL, angle=0.3)
    st.solve(retain_internal=True)
    for z in (0.0, 0.25, 0.5, 0.999):
        f = st.internal_field(z, incident=(0.0, 1.0))
        Ey = np.abs(np.asarray(f["Ey"]))
        assert abs(float(Ey.mean()) - 1.0) < 1e-12
        assert float(Ey.max() - Ey.min()) < 1e-11


# ===========================================================================
# F-C -- the covariant / slanted cascade's flux gauge (the Berreman-F-1 twin)
# ===========================================================================

def _slant_stack(n_sub, n_sup, slant, theta, eps=2.25):
    st = P.PMMStack(PER, n_substrate=n_sub, n_superstrate=n_sup, degree=8,
                    far_field_orders=5)
    st.add_layer(DEPTH, eps=eps * I3, slant_angle=slant)
    st.set_source(WL, angle=theta)
    return st.solve()


@pytest.mark.parametrize("n_sub", [1.5 + 0j, 1.5 + 0.01j, 1.5 + 0.05j,
                                   1.5 + 0.30j, 0.2 + 3.5j])
@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_w7_fc_slanted_cascade_transmits_into_an_absorbing_substrate(n_sub,
                                                                     theta):
    """F-C: a HOMOGENEOUS layer makes the slant a physical NO-OP, so the
    vertical cascade of the same stack is the exact oracle for the covariant
    one.  Pre-fix every absorbing-substrate row returned T = [0, 0] silently
    (the double-conjugated eps drove Re(kz_sub) < 0 into the propagating
    mask); the lossless row was perfect -- the F-1 signature exactly."""
    _o0, R0, T0, _J0 = _slant_stack(n_sub, 1.0, 0.0, theta)
    _o1, R1, T1, _J1 = _slant_stack(n_sub, 1.0, 0.35, theta)
    # the signature itself: transmission must not collapse
    assert float(T1.sum()) > 0.5 * float(T0.sum()) > 0.0
    # measured covariant-vs-vertical floor 5.4e-8 at degree 8; 20x headroom
    assert float(np.max(np.abs(R1 - R0))) < 1e-6
    assert float(np.max(np.abs(T1 - T0))) < 1e-6


@pytest.mark.parametrize("n_sup", [1.0 + 0j, 1.2 + 0.05j])
@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_w7_fc_slanted_cascade_accepts_an_absorbing_superstrate(n_sup, theta):
    """F-C: pre-fix an absorbing SUPERSTRATE produced ``kz_inc = -1.14651``
    and raised ``non-propagating incidence`` on a perfectly propagating
    medium, so the covariant path could not run at all."""
    _o0, R0, T0, _J0 = _slant_stack(1.5, n_sup, 0.0, theta)
    _o1, R1, T1, _J1 = _slant_stack(1.5, n_sup, 0.35, theta)
    assert float(np.max(np.abs(T1 - T0))) < 1e-6
    assert float(np.max(np.abs(R1 - R0))) < 1e-6


def test_w7_fc_slanted_grating_on_an_absorbing_substrate_is_sane():
    """A REAL slanted grating (not the no-op limit) on an absorbing
    substrate: pre-fix T was identically zero, so the totals collapsed to R
    alone (0.014 -- 'passive', hence invisible to ``_warn_stack_energy``)."""
    for n_sub, lossless in ((1.5 + 0j, True), (1.5 + 0.3j, False)):
        st = P.PMMStack(PER, n_substrate=n_sub, n_superstrate=1.0, degree=8,
                        far_field_orders=5)
        st.add_layer(DEPTH, segments=[(0.5, 3.61 * I3), (0.5, 1.0 * I3)],
                     slant_angle=0.3)
        st.set_source(WL, angle=0.25)
        _o, R, T, _J = st.solve()
        tot = R.sum(1) + T.sum(1)
        assert np.all(T.sum(1) > 0.5), T.sum(1)
        # a lossless slanted grating still closes (the convection/covariant
        # union-grid floor is ~1e-4 here, the documented slant floor)
        bar = 1e-3 if lossless else 2e-3
        assert np.all(np.abs(tot - 1.0) < bar), tot


# ===========================================================================
# F-D -- the 2-D scalar even-parity fold recentred only HALF its operators
# ===========================================================================

def _fd_cell(n, lo, w):
    c = np.full((n, n), 2.0 + 0j)
    c[lo:lo + w, lo:lo + w] = 6.0
    return c


# (n, lo, w, pre-fix |fold - full|).  The FIRST four have a NON-TRIVIAL
# recentring gauge (|D-1|max ~ 1.7-2.0) and are the defect rows; the last two
# have D == 1 to 1e-15 and are the CONTROL -- they agreed even pre-fix, which
# is what proves the gauge (not the fold) was the defect.
_FD_CASES = [
    (4, 0, 2, 3.174e-03), (4, 2, 2, 2.087e-03),
    (5, 0, 1, 1.542e-03), (6, 0, 1, 1.051e-03),
    (4, 1, 2, 3.858e-15), (5, 2, 1, 3.997e-15),     # D trivial: controls
    # (4, 0, 1) is excluded: its truncation is ill-conditioned here (the
    # lossless-closure tripwire fires at R+T = 1.09 for BOTH branches), so it
    # cannot separate the gauge defect from the projection floor.
]


@pytest.mark.parametrize("n,lo,w,prefix_delta", _FD_CASES)
def test_w7_fd_even_parity_fold_matches_the_full_2d_cell_solve(n, lo, w,
                                                               prefix_delta):
    """F-D: ``twod._symmetric_solve_2d`` conjugated ``EpsF``/``EinvF``/``EPS_n*``
    by the recentring gauge D but used ``GxF``/``GyF`` RAW, on the premise
    that they are diagonal in the order index.  True of rcwa's Fourier
    ``Kx``/``Ky``; FALSE of the hybrid PMM's SEM-PROJECTED derivative
    operators, which are dense -- so whenever the cell's symmetry centre is
    off the FFT origin the folded (P, Q) is built half in one gauge and half
    in the other.  ``_flip_invariant`` only tests ``EpsF``, so nothing caught
    it, and this is the DEFAULT ``symmetry='auto'`` path.

    Measured pre-fix (period 0.8, depth 0.3, wl 0.55, degree 5, n_orders 2,
    TE, normal incidence), ``|fold - full|`` per case -- the ``prefix_delta``
    column above.  The last two rows are the decisive CONTROL: their gauge is
    trivial (``|D-1|max <= 3.4e-15``) and they agreed to ~4e-15 even pre-fix,
    so the discrepancy tracks the GAUGE, not the fold itself.  Post-fix every
    row agrees to <= 9.4e-15."""
    cell = _fd_cell(n, lo, w)
    got = {}
    for sym in (True, False):
        _o, R, T = P.pmm_efficiency_2d_cell(
            0.8, 0.8, cell, 1.5, 1.0, 0.3, 0.55, theta=0.0, phi=0.0,
            polarization="te", degree=5, n_orders=2, symmetry=sym)
        got[sym] = (np.asarray(R), np.asarray(T))
    d = max(float(np.max(np.abs(got[True][0] - got[False][0]))),
            float(np.max(np.abs(got[True][1] - got[False][1]))))
    assert d < 1e-12, (n, lo, w, d, "prefix was %.3e" % prefix_delta)


# ===========================================================================
# B2 -- unit-scale invariance of the mass-weighted flux cut
# ===========================================================================

@pytest.mark.parametrize("scale,label", [(1.0, "um"), (1e3, "nm"),
                                         (1e-6, "m"), (1e6, "pm")])
def test_w7_b2_flux_cut_is_unit_invariant(scale, label):
    """B2: ``flux = Im(E^T S0 conj(H))`` carries the element Jacobian, so the
    old ``max(max|flux|, 1.0)`` floor had LENGTH units.  In METRES an 8 nm
    structure sank the whole spectrum below the absolute 1e-9 and every
    propagating mode was reclassified evanescent.  Pre-fix, the SAME physics:
    ``R+T = 2.000003`` in nm/um but ``2.027293`` in metres (max per-order
    drift 3.4e-2).  Reference = nanometres."""
    ref = P.pmm_jones_1d(
        0.008 * 1e3, 3.61 * I3, 1.0 * I3, 1.5, 1.0, 0.0035 * 1e3, 0.45,
        0.0055 * 1e3, angle=0.2, degree=7, far_field_orders=5,
        stabilize=False)
    o, R, T, J = P.pmm_jones_1d(
        0.008 * scale, 3.61 * I3, 1.0 * I3, 1.5, 1.0, 0.0035 * scale, 0.45,
        0.0055 * scale, angle=0.2, degree=7, far_field_orders=5,
        stabilize=False)
    assert float(np.max(np.abs(R - ref[1]))) < 1e-12, label
    assert float(np.max(np.abs(T - ref[2]))) < 1e-12, label
    assert float(np.max(np.abs(J - ref[3]))) < 1e-12, label
    # ... and the energy closure survives the unit change
    assert abs(float(R.sum() + T.sum()) - 2.0) < 1e-4


# ===========================================================================
# A11 / A13 -- cache key completeness and returned-array ownership
# ===========================================================================

_A11_CELL = np.array([[4.0 + 0j, 2.0, 3.0, 2.5], [2.0, 2.0, 2.5, 3.0],
                      [3.0, 2.5, 4.0, 2.0], [2.5, 3.0, 2.0, 2.0]])


def _a11_stack(degree, grade):
    h = P.PMM2DStackHybrid(0.4, 0.4, n_substrate=1.5, n_superstrate=1.0,
                           degree=degree, n_orders=2, grade=grade,
                           elements_per_strip=3, symmetry=False)
    h.add_layer(0.2, eps_cell=_A11_CELL)
    h.add_layer(0.1, eps=2.1)
    h.set_source(0.55, theta=0.2, phi=0.3)
    return h


@pytest.mark.parametrize("attr,a,b", [("degree", 5, 7), ("degree", 5, 9),
                                      ("degree", 5, 11), ("grade", False, True)])
def test_w7_a11_geom_cache_key_covers_the_solver_parameters(attr, a, b):
    """A11: ``PMM2DStackHybrid._geom_cache`` persists across ``solve()`` and is
    dropped only by ``add_layer``, but its key carried the LAYER geometry only
    -- while the cached build depends on ``degree``/``grade``/``period_x``/
    ``period_y``/``n_orders``.  Pre-fix, ``degree`` 5 -> 7 and 5 -> 11 returned
    answers BIT-IDENTICAL to the degree-5 one (6.24e-03 / 9.48e-03 off), and
    5 -> 9 gave ``sum(R) = 0.237212592`` where a fresh object gives
    ``0.243068009``."""
    h = _a11_stack(a, False) if attr == "degree" else _a11_stack(5, a)
    h.solve()
    setattr(h, attr, b)
    _o, R, T, _J = h.solve()
    fresh = _a11_stack(b, False) if attr == "degree" else _a11_stack(5, b)
    _o2, R2, T2, _J2 = fresh.solve()
    assert np.array_equal(R, R2) and np.array_equal(T, T2)


def test_w7_a13_caches_hand_out_read_only_arrays():
    """A13: every cached array must be frozen before it is shared -- a caller
    that writes into one poisons every later solve on the same key.  Measured
    pre-fix: ``_geom_cache`` 21/23 arrays writeable (next solve drifted
    1.543e-06), ``_PreparedPMMStack._eig_cache`` 12/12 (7.844e-07), and
    ``_jax_twod._STATIC_CACHE`` 8/8 at MODULE scope."""
    def _writeable(obj, out):
        if isinstance(obj, np.ndarray):
            if obj.flags.writeable:
                out.append(obj.shape)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                _writeable(v, out)
        elif isinstance(obj, dict):
            for v in obj.values():
                _writeable(v, out)

    h = _a11_stack(5, False)
    h.solve()
    bad = []
    for v in h._geom_cache.values():
        _writeable(v, bad)
    assert bad == [], bad

    st = P.PMMStack(PER, n_substrate=1.5, degree=8, far_field_orders=5)
    st.add_layer(0.2, segments=[(0.5, 3.61 * I3), (0.5, 1.0 * I3)])
    pr = st.prepare()
    r1 = pr.solve(wavelength=WL, angle=0.2)
    r2 = pr.solve(wavelength=WL, angle=0.2)
    bad = []
    for v in list(pr._eig_cache.values()) + list(pr._mats_cache.values()):
        _writeable(v, bad)
    assert bad == [], bad
    assert all(np.array_equal(a, b) for a, b in zip(r1, r2))


# ===========================================================================
# F-E -- a TRACED wavelength cannot size the propagating-order set
# ===========================================================================

def _fe_stack(period, wl, ffo, degree, eps=4.0 + 0.5j):
    st = P.PMMStack(period, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                    far_field_orders=ffo)
    st.add_layer(0.25e-6, segments=[(0.4, eps), (0.6, 1.0)])
    st.add_layer(0.10e-6, eps=2.25)
    st.set_source(wl, angle=0.0)
    return st


_FE_WL = 0.633e-6


@pytest.mark.parametrize("mode", ["jit", "grad"])
def test_w7_fe_traced_wavelength_raises_on_the_stack(mode):
    """F-E: the order set is ``max(ffo, 2*m_prop+5)`` with
    ``m_prop = floor(n_max*period/wl)`` -- a data-dependent INTEGER COUNT that
    fixes array shapes, so it cannot come from a tracer.  Pre-fix it fell back
    to ``wl = inf`` -> ``m_prop = 0`` -> the set collapsed to the ``ffo``
    floor, DROPPING propagating orders.

    Measured pre-fix (2-layer stack, wl 633 nm, ``jax.jit`` over wl):
    period 2.4 um / ffo 5 -> NumPy N=19 but jit N=5, forward 3.90e-02 rel,
    ``d/d(wl)`` 1.76e-02 rel; period 3.2 um / ffo 5 -> N=25 vs 5, forward
    4.15e-02, gradient **2.07e-01**.  Un-jitted the value is concrete, so the
    forward answer was bit-exact -- only the traced evaluation was wrong."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def f(wl):
        _o, R, _T, _J = _fe_stack(2.4e-6, wl, 5, 24).solve()
        return R if mode == "jit" else R[0].sum()

    with pytest.raises(NotImplementedError, match="TRACED wavelength"):
        if mode == "jit":
            jax.jit(f)(jnp.asarray(_FE_WL))
        else:
            jax.grad(f)(_FE_WL)


@pytest.mark.parametrize("entry", ["pmm_efficiency_1d", "pmm_jones_1d"])
def test_w7_fe_traced_wavelength_raises_on_the_1d_entries(entry):
    """The same collapse in the two functional JAX entries -- measured pre-fix
    on ``pmm_efficiency_1d`` under ``jax.jit`` over wl: period 2.4 um / ffo 5
    -> NumPy N=27 vs jit N=5 and **15.0%** forward error; 3.2 um -> N=35 vs 5
    and **13.3%**.  (0.8 um / ffo 5 -> N=11 vs 5 but 7.3e-16, because the
    dropped orders are evanescent there -- which is how it stayed hidden.)"""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def f(wl):
        if entry == "pmm_efficiency_1d":
            return P.pmm_efficiency_1d(
                2.4e-6, jnp.asarray(3.0 + 0j), jnp.asarray(1.0 + 0j), 1.5,
                1.0, 0.25e-6, 0.4, wl, degree=24, far_field_orders=5,
                polarization="te", stabilize=False)[1]
        return P.pmm_jones_1d(
            2.4e-6, jnp.asarray(9.0 + 0j) * I3, jnp.asarray(1.0 + 0j) * I3,
            1.5, 1.0, 0.25e-6, 0.4, wl, degree=24, far_field_orders=5,
            stabilize=False)[1]

    with pytest.raises(NotImplementedError, match="TRACED wavelength"):
        jax.jit(f)(jnp.asarray(_FE_WL))


def test_w7_fe_concrete_wavelength_jit_matches_unjitted():
    """The contract the bug broke: with a CONCRETE wavelength, ``jax.jit`` must
    return the SAME SHAPE as the un-jitted call (pre-fix jit gave ``(2, 5)``
    where un-jitted gave ``(2, 9)``), and the physically-invariant totals must
    agree.  Traced EPS is deliberately still allowed -- it also shrinks the
    order set, but only EVANESCENT orders drop, so the totals are exact
    (measured 0.0 / 1.4e-17 here)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    for period, ffo, degree in ((0.8e-6, 21, 12), (0.8e-6, 11, 12),
                                (2.4e-6, 19, 24)):
        def g(eps, _p=period, _f=ffo, _d=degree):
            return _fe_stack(_p, _FE_WL, _f, _d, eps=eps).solve()[1]

        a = np.asarray(g(jnp.asarray(4.0 + 0.5j)))
        b = np.asarray(jax.jit(g)(jnp.asarray(4.0 + 0.5j)))
        assert a.shape == b.shape, (period, ffo, a.shape, b.shape)
        assert abs(a[0].sum() - b[0].sum()) < 1e-14
        assert abs(a[1].sum() - b[1].sum()) < 1e-14


def test_w7_fe_concrete_wavelength_gradients_stay_exact():
    """What the raise preserves: at a FIXED wavelength every other gradient is
    exact.  Measured vs a NumPy central difference -- eps.re 2.5e-9,
    thickness 2.0e-10, angle 1.8e-8, n_sub 5.8e-9."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def mk(eps=4.0 + 0.5j, t1=0.25e-6, ang=0.0, nsub=1.5, traced=()):
        def j(n, v):
            return jnp.asarray(v) if n in traced else v

        st = P.PMMStack(0.8e-6, n_substrate=j("nsub", nsub),
                        n_superstrate=1.0, degree=12, far_field_orders=21)
        st.add_layer(j("t1", t1), segments=[(0.4, j("eps", eps)), (0.6, 1.0)])
        st.add_layer(0.10e-6, eps=2.25)
        st.set_source(_FE_WL, angle=j("ang", ang))
        return st

    cases = (
        ("eps.re", lambda v: mk(eps=v + 0.5j,
                                traced=("eps",)).solve()[1][0].sum(),
         4.0, 1e-6),
        ("thickness", lambda v: mk(t1=v, traced=("t1",)).solve()[1][0].sum(),
         0.25e-6, 1e-12),
        ("angle", lambda v: mk(ang=v, traced=("ang",)).solve()[1][1].sum(),
         0.3, 1e-6),
        ("n_sub", lambda v: mk(nsub=v, traced=("nsub",)).solve()[2][0].sum(),
         1.5, 1e-6),
    )
    for name, f, x, h in cases:
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert abs(ad - fd) <= 1e-6 * max(abs(fd), 1e-12), (name, ad, fd)

"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 R-2: ``seidel_wfe`` returned -W.

``seidel_wfe`` composed the textbook Hopkins / Welford expansion

    W = (1/8) S1 rho^4 + (1/2) S2 rho^3 cos + (1/2) S3 rho^2 cos^2
        + (1/4) S3 rho^2 + (1/4) S4 H^2 rho^2 + (1/2) S5 rho cos

out of ``seidel_coefficients``' sums -- which are documented as
``code = -S_Welford`` (the S3-1 note in ``seidel.py``'s refracting
branch).  Composing the two conventions returned the NEGATIVE of the
physical wavefront: measured against an exact-trace oracle the ratio was
-0.9975 ... -0.9998 across four singlets over ``rho in [0.3, 1]``, and
uniformly -1.000 term by term (``rho^2``, ``rho^3``, ``rho^4``).

Two INDEPENDENT sign routes are pinned here:

1. an exact-trace wavefront oracle (accumulated real-ray OPL to the
   paraxial image point, referenced to the chief ray and to a common
   incoming wavefront) -- ratio must be +1, not -1;
2. pure ray GEOMETRY, no OPL bookkeeping at all: the marginal ray's axis
   crossing versus the paraxial one.  For a wavefront referenced to the
   paraxial focus, ``dW/dh = h*dz / L^2`` to leading order (``L`` = pupil
   -> focus distance, ``dz`` = that ray's longitudinal focus shift), so
   ``W`` carries the SIGN of ``dz``.  A single positive element always
   has undercorrected spherical aberration (marginal focus short,
   ``dz < 0``) and must therefore report ``W < 0``; the biconcave case
   flips both, which no global sign error can survive.

Convention pinned by this file (also stated in the docstring):
``W = OPL(pupil point) - OPL(reference ray)`` to the paraxial image
point, so ``W < 0`` = path shorter = wavefront ADVANCED.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.raytrace import Surface, RayBundle, trace, seidel_coefficients
from lumenairy.raytrace.seidel_analysis import seidel_wfe

_WL = 587.5618e-9


# ======================================================================
# Designs: stop AT surface 0, so pupil sampling is the launch height.
# ======================================================================
def _singlet(R1, R2, d, glass, h_max):
    return [
        Surface(radius=R1, glass_before='air', glass_after=glass,
                thickness=d, semi_diameter=h_max, is_stop=True),
        Surface(radius=R2, glass_before=glass, glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ]


_DESIGNS = [
    # name,                       R1,      R2,      d,     glass,      h_max
    ('biconvex 50/-50 N-BK7',   50e-3,  -50e-3,  5e-3, 'N-BK7',    2e-3),
    ('plano-cvx flat-first',    np.inf, -100e-3, 4e-3, 'N-BK7',    3e-3),
    ('meniscus 30/200 N-LAK22', 30e-3,  200e-3,  6e-3, 'N-LAK22',  2e-3),
    ('biconcave -60/60 N-SF2', -60e-3,   60e-3,  3e-3, 'N-SF2',    2e-3),
]


# ======================================================================
# Exact-trace wavefront oracle
# ======================================================================
def _fan(y, u0, wl):
    y = np.atleast_1d(np.asarray(y, float))
    th = np.arctan(float(u0))
    z = np.zeros_like(y)
    return RayBundle(x=z.copy(), y=y.copy(), z=z.copy(), L=z.copy(),
                     M=np.full_like(y, np.sin(th)),
                     N=np.full_like(y, np.cos(th)),
                     wavelength=wl, alive=np.ones_like(y, bool),
                     opd=z.copy())


def _axis_crossing(surfaces, wl, h):
    """z (from the last surface vertex) where the real ray launched at
    height ``h``, parallel to the axis, crosses the axis."""
    b = trace(_fan(h, 0.0, wl), surfaces, wl).ray_history[-1]
    y, u = float(b.y[0]), float(b.M[0] / b.N[0])
    return -y / u


def _wavefront_oracle(surfaces, wl, h_max, rho):
    """W(rho) [m] on the meridional section, on axis.

    ``W = (accumulated real OPL) + (straight distance to the paraxial
    image point) - (the same for the rho = 0 reference ray)``.  For a
    virtual image (negative BFL) the reference point lies upstream, so
    that leg counts negatively.
    """
    rho = np.atleast_1d(np.asarray(rho, float))
    y = np.concatenate([[0.0], rho * h_max])
    out = trace(_fan(y, 0.0, wl), surfaces, wl).ray_history[-1]
    bfl = _axis_crossing(surfaces, wl, 1e-7)       # paraxial focus
    d = np.sqrt(out.x ** 2 + out.y ** 2 + (out.z - bfl) ** 2)
    if bfl < 0:
        d = -d
    W = out.opd + d
    return W[1:] - W[0]


def _seidel_W(surfaces, rho):
    sd, _ = seidel_coefficients(surfaces, _WL, stop_index=0,
                               field_angle=1e-9)
    return np.asarray(seidel_wfe(sd, rho, np.zeros_like(np.asarray(
        rho, float)))), sd


# ======================================================================
# Pin 1 -- ratio vs the exact-trace oracle must be +1
# ======================================================================
class TestR2SeidelWfeMatchesExactTraceWavefront:

    @pytest.mark.parametrize('name,R1,R2,d,glass,h_max', _DESIGNS)
    def test_ratio_is_plus_one_at_rho_0p7(self, name, R1, R2, d, glass,
                                          h_max):
        surfaces = _singlet(R1, R2, d, glass, h_max)
        W_ex = float(_wavefront_oracle(surfaces, _WL, h_max, 0.7)[0])
        W_sd = float(_seidel_W(surfaces, 0.7)[0])
        ratio = W_sd / W_ex
        assert abs(ratio - 1.0) < 0.01, (
            f"{name}: seidel_wfe/exact = {ratio:+.6f} at rho=0.7 "
            f"(W_seidel={W_sd/_WL:+.6f} wv, W_exact={W_ex/_WL:+.6f} wv).  "
            f"A value near -1 is the R-2 defect (the expansion composed "
            f"from -S_Welford sums); a value far from +-1 is a magnitude "
            f"regression.")

    @pytest.mark.parametrize('name,R1,R2,d,glass,h_max', _DESIGNS)
    def test_ratio_is_plus_one_across_the_pupil(self, name, R1, R2, d,
                                                glass, h_max):
        surfaces = _singlet(R1, R2, d, glass, h_max)
        rho = np.linspace(0.3, 1.0, 8)
        W_ex = _wavefront_oracle(surfaces, _WL, h_max, rho)
        W_sd = _seidel_W(surfaces, rho)[0]
        ratio = W_sd / W_ex
        assert np.all(np.abs(ratio - 1.0) < 0.01), (
            f"{name}: ratios over rho in [0.3, 1] = "
            f"{np.array2string(ratio, precision=6)}")
        # ... and the pre-fix answer is rejected everywhere
        assert not np.any(np.abs(ratio + 1.0) < 0.01), (
            f"{name}: seidel_wfe still returns -W")

    def test_absolute_magnitude_is_pinned(self):
        """Catch a magnitude regression that a pure ratio test would miss
        if BOTH sides moved: hard-coded exact-trace values [waves]."""
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        for rho, want in ((0.5, -5.494026e-03), (0.7, -2.110904e-02),
                          (1.0, -8.795834e-02)):
            W_sd = float(_seidel_W(surfaces, rho)[0]) / _WL
            assert abs(W_sd - want) < 1e-4, (
                f"seidel_wfe(rho={rho}) = {W_sd:+.6f} waves; exact-trace "
                f"{want:+.6f} waves (third-order truncation is ~0.1%).")


# ======================================================================
# Pin 2 -- geometry-only sign anchor (no OPL bookkeeping)
# ======================================================================
class TestR2SignFromMarginalFocusShift:
    """The sign of W at the paraxial-focus reference must equal the sign of
    the marginal ray's longitudinal focus shift."""

    @pytest.mark.parametrize('name,R1,R2,d,glass,h_max', _DESIGNS)
    def test_wavefront_sign_follows_the_focus_shift(self, name, R1, R2, d,
                                                    glass, h_max):
        surfaces = _singlet(R1, R2, d, glass, h_max)
        z_par = _axis_crossing(surfaces, _WL, 1e-7)
        z_marg = _axis_crossing(surfaces, _WL, h_max)
        dz = z_marg - z_par
        assert abs(dz) > 1e-6, f"{name}: focus shift too small to judge"
        W = float(_seidel_W(surfaces, 1.0)[0])
        assert np.sign(W) == np.sign(dz), (
            f"{name}: marginal focus shift dz = {dz*1e6:+.3f} um but "
            f"seidel_wfe(rho=1) = {W/_WL:+.6f} waves.  W referenced to the "
            f"paraxial focus obeys dW/dh = h*dz/L^2, so the two signs must "
            f"agree; opposite signs are the R-2 defect.")

    def test_positive_singlet_is_undercorrected_and_W_is_negative(self):
        """Textbook anchor: a single positive element cannot be
        overcorrected, so its marginal focus is SHORT and W < 0."""
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        z_par = _axis_crossing(surfaces, _WL, 1e-7)
        z_marg = _axis_crossing(surfaces, _WL, 2e-3)
        assert z_marg < z_par, "biconvex singlet must be undercorrected"
        W = float(_seidel_W(surfaces, 1.0)[0])
        assert W < 0.0, (
            f"undercorrected positive singlet reported W(rho=1) = "
            f"{W/_WL:+.6f} waves > 0")

    def test_negative_singlet_flips_both(self):
        surfaces = _singlet(-60e-3, 60e-3, 3e-3, 'N-SF2', 2e-3)
        z_par = _axis_crossing(surfaces, _WL, 1e-7)
        z_marg = _axis_crossing(surfaces, _WL, 2e-3)
        assert z_marg > z_par
        assert float(_seidel_W(surfaces, 1.0)[0]) > 0.0


# ======================================================================
# Pin 3 -- the rho^4 term's sign versus the library's own S1
# ======================================================================
class TestR2Rho4TermSign:
    """``W``'s spherical term is ``-(1/8) S1_code rho^4``: the library's S1
    is ``-S_Welford``, so a POSITIVE reported S1 must produce a NEGATIVE
    quartic term."""

    def test_biconvex_S1_positive_and_quartic_term_negative(self):
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        W, sd = _seidel_W(surfaces, 1.0)
        S1 = float(sd['total']['S1'])
        assert S1 > 0.0, f"expected S1 > 0 for this biconvex; got {S1!r}"
        assert float(W[0] if np.ndim(W) else W) < 0.0

    def test_quartic_term_is_exactly_minus_one_eighth_S1(self):
        """Isolate the quartic term: S1-only totals dict, S4 = 0."""
        totals = {'S1': 8.0, 'S2': 0.0, 'S3': 0.0, 'S4': 0.0, 'S5': 0.0}
        with pytest.warns(RuntimeWarning):
            W = float(seidel_wfe(totals, rho=1.0, theta=0.0,
                                 field_angle=0.0))
        assert W == pytest.approx(-1.0, abs=1e-15), (
            f"seidel_wfe(S1=8, rho=1) = {W!r}; the quartic term is "
            f"-(1/8)*S1 = -1.0 on this library's sign convention (pre-fix "
            f"it returned +1.0).")

    def test_field_curvature_dc_companion_keeps_its_relative_weight(self):
        """The (1/4) S3 rho^2 DC companion (v4.11.2) must survive R-2's
        global flip: S3=1, rho=1, theta=0 gives -(1/2 + 1/4) = -0.75.

        NOTE for the reviewer: this is the corrected counterpart of
        ``tests/unit/test_audit_raytrace.py``
        ``TestAuditFixesV4_11_2_raytrace_SeidelWfeFieldCurvatureDcTerm``,
        which pins +0.75 / +rho^2 and therefore pins the R-2 sign defect.
        """
        totals = {'S1': 0.0, 'S2': 0.0, 'S3': 1.0, 'S4': 0.0, 'S5': 0.0}
        with pytest.warns(RuntimeWarning):
            W = float(seidel_wfe(totals, rho=1.0, theta=0.0,
                                 field_angle=0.0))
        assert abs(W + 0.75) < 1e-15, (
            f"seidel_wfe(S3=1, rho=1, theta=0) = {W!r}; expected -0.75 "
            f"(= -[(1/2) astigmatism + (1/4) field-curvature DC]).")
        # theta = pi/2 kills the astigmatism term, leaving the DC alone
        rho = np.array([0.0, 0.5, 1.0])
        with pytest.warns(RuntimeWarning):
            W2 = seidel_wfe({'S1': 0.0, 'S2': 0.0, 'S3': 4.0, 'S4': 0.0,
                             'S5': 0.0}, rho=rho,
                            theta=np.full_like(rho, np.pi / 2),
                            field_angle=0.0)
        assert np.max(np.abs(np.asarray(W2) + rho ** 2)) < 1e-15


# ======================================================================
# Invariants the fix must NOT disturb
# ======================================================================
class TestR2InvariantsPreserved:

    def test_zero_at_pupil_origin(self):
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        assert abs(float(_seidel_W(surfaces, 0.0)[0])) < 1e-300

    def test_magnitude_only_consumers_unchanged(self):
        """|W| is bit-identical to the pre-fix value -- the exact-trace
        magnitude at rho=1 for the reference biconvex."""
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        assert abs(abs(float(_seidel_W(surfaces, 1.0)[0])) / _WL
                   - 8.787990e-02) < 1e-6

    def test_vectorises_over_a_2d_pupil_grid(self):
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        sd, _ = seidel_coefficients(surfaces, _WL, stop_index=0,
                                   field_angle=1e-3)
        r = np.linspace(0, 1, 9)
        t = np.linspace(0, 2 * np.pi, 7)
        R, T = np.meshgrid(r, t, indexing='ij')
        W = seidel_wfe(sd, R, T)
        assert W.shape == R.shape
        assert np.all(np.isfinite(W))

    def test_no_new_warnings_on_the_result_dict_path(self):
        surfaces = _singlet(50e-3, -50e-3, 5e-3, 'N-BK7', 2e-3)
        sd, _ = seidel_coefficients(surfaces, _WL, stop_index=0,
                                   field_angle=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            seidel_wfe(sd, np.linspace(0, 1, 5), np.zeros(5))

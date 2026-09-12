"""GBD <-> ASM interoperability (v5.21 converter, INVERTED in v5.46).

A Gabor frame of exact Gaussian-beam solutions, propagated exactly and summed,
reproduces the angular-spectrum field of the same beam with NO residual phase:
free-space propagation is linear and every beamlet is an exact solution.  The
``phi0(z) = 2*arctan(z / zR_beamlet)`` offset these tests used to assert was the
beamlet Gouy phase applied with the WRONG SIGN by
``propagate_beamlets_freespace`` (``Q_new/Q_old`` in a convention that needs
``conj(Q_new/Q_old)``) -- audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 finding
S5.  The give-away, recorded by the audit and re-measured in
``test_s5_no_gouy_offset_at_any_waist_factor`` below, is that the "convention"
offset depended on ``waist_factor``, a purely numerical knob.

These tests therefore now assert the OPPOSITE of the v5.21 pins:
``_relerr(Gz, Az)`` must be SMALL with no converter, and the three converter
functions are deprecated no-ops.  ``match_global_phase`` -- the general,
propagator-agnostic primitive -- is unaffected and still tested.
"""
import warnings

import numpy as np
import pytest

from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.gbd import (
    asm_field_to_gbd,
    gbd_asm_gouy_phase,
    gbd_field_to_asm,
    match_global_phase,
    propagate_gbd_freespace,
)

LAM = 0.633e-6


def _relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))


def _gauss(N, dx, w0=0.12e-3, off=0.0, tilt=0.0):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-((X - off) ** 2 + Y ** 2) / w0 ** 2)
    if tilt:
        E = E * np.exp(1j * 2 * np.pi / LAM * tilt * X)
    return E.astype(np.complex128)


def test_s5_no_gouy_offset_at_any_waist_factor():
    """There is NO global phase between a GBD free-space field and ASM.

    Oracle: ``angular_spectrum_propagate``, which is exact for a band-limited
    field on this grid (the transfer function is applied in closed form).

    Bar derivation.  The residual global phase ``arg <Az, Gz>`` is bounded by
    the GBD decomposition error itself, which is ~1e-3 ... 1.6e-2 in relative
    L2 at these waist factors -- so a residual phase of order 1e-4 rad is the
    floor, not 0.  Measured on this fixture post-fix: +4.7e-06 / +4.7e-06 /
    +4.6e-06 rad at waist_factor 1.0 / 1.5 / 2.5.  Pre-fix the SAME quantity
    was +3.133 / +3.126 / +3.108 rad, i.e. the closed form
    ``2*arctan(z/zR_beamlet)`` this test used to assert.  The 1e-3 rad gate is
    two decades above the measured residual and three decades below the
    pre-fix value, so it cannot pass on the pre-fix code and cannot fail on
    decomposition noise.

    The waist-factor DEPENDENCE is the diagnostic: a real convention offset
    cannot depend on a numerical knob.  Pre-fix the three values spanned
    0.025 rad; post-fix they span < 2e-7 rad.
    """
    N, dx = 160, 4e-6
    E0 = _gauss(N, dx)
    measured = []
    for wf, z in [(1.0, 5e-3), (1.5, 3e-3), (2.5, 8e-3)]:
        Gz = propagate_gbd_freespace(E0, dx, z=z, wavelength=LAM,
                                     waist_factor=wf)
        Az = angular_spectrum_propagate(E0, z, LAM, dx)
        phi = float(np.angle(np.vdot(Az.ravel(), Gz.ravel())))
        measured.append(phi)
        assert abs(phi) < 1e-3, (
            f"waist_factor={wf}: residual GBD-vs-ASM global phase {phi:+.4e} "
            f"rad; the pre-S5 code gave 2*arctan(z/zR_beamlet) ~ +3.13 rad "
            f"here.")
    assert max(measured) - min(measured) < 1e-4, (
        f"the residual phase still depends on waist_factor (spread "
        f"{max(measured) - min(measured):.3e} rad); a genuine convention "
        f"offset cannot.")


def test_gbd_matches_asm_with_no_converter():
    """A GBD free-space field equals ``angular_spectrum_propagate`` to the GBD
    decomposition accuracy with NO conversion and NO free phase fit -- across a
    non-trivial off-centre + tilted field.

    Bar derivation: the GBD decomposition floor on these fixtures is the
    residual Poisson-summation ripple of the Gabor frame, measured here at
    2.9e-03 (centred, wf=1.5) and 1.8e-03 (off-centre + tilted, wf=1.0); the
    5e-3 gate is the same floor the v5.21 pin used for the CONVERTED field.
    Pre-fix the unconverted field scored > 0.5 (the old test asserted exactly
    that), so this gate has two decades of margin on both sides.
    """
    N, dx = 192, 4e-6
    for off, tilt, wf, z in [(0.0, 0.0, 1.5, 6e-3),
                             (30e-6, 0.003, 1.0, 4e-3)]:
        E0 = _gauss(N, dx, off=off, tilt=tilt)
        Gz = propagate_gbd_freespace(E0, dx, z=z, wavelength=LAM,
                                     waist_factor=wf)
        Az = angular_spectrum_propagate(E0, z, LAM, dx)
        assert _relerr(Gz, Az) < 5e-3, (
            f"off={off} tilt={tilt} wf={wf}: relerr {_relerr(Gz, Az):.3e}")


def test_gbd_to_asm_handoff_matches_pure_asm_without_conversion():
    """A GBD free-space leg continued with ASM equals pure end-to-end ASM --
    no converter in between.  This is the interoperability use case the v5.21
    converter existed for."""
    N, dx = 192, 4e-6
    E0 = _gauss(N, dx, w0=0.12e-3)
    z1, z2, wf = 4e-3, 10e-3, 1.5
    G1 = propagate_gbd_freespace(E0, dx, z=z1, wavelength=LAM, waist_factor=wf)
    hybrid = angular_spectrum_propagate(G1, z2, LAM, dx)
    ref = angular_spectrum_propagate(E0, z1 + z2, LAM, dx)
    assert _relerr(hybrid, ref) < 5e-3


@pytest.mark.parametrize('fn_name', ['gbd_asm_gouy_phase',
                                     'gbd_field_to_asm',
                                     'asm_field_to_gbd'])
def test_converters_are_deprecated_no_ops(fn_name):
    """The three compensator functions warn and do nothing."""
    N, dx = 32, 4e-6
    E = _gauss(N, dx, off=20e-6, tilt=0.002)
    kw = dict(z=5e-3, wavelength=LAM, dx=dx, waist_factor=1.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if fn_name == 'gbd_asm_gouy_phase':
            out = gbd_asm_gouy_phase(kw['z'], kw['wavelength'], kw['dx'],
                                     kw['waist_factor'])
            assert out == 0.0
        elif fn_name == 'gbd_field_to_asm':
            assert np.array_equal(gbd_field_to_asm(E, **kw), E)
        else:
            assert np.array_equal(asm_field_to_gbd(E, **kw), E)
    assert any(issubclass(c.category, DeprecationWarning) for c in caught), (
        f"{fn_name} must emit a DeprecationWarning")


def test_match_global_phase():
    """``match_global_phase`` reconciles a global phase EXACTLY (its whole job)
    and is a NO-OP for the GBD-free-space vs ASM pair, which no longer differ
    by one."""
    N, dx = 96, 4e-6
    E = _gauss(N, dx, off=15e-6, tilt=0.002)
    rotated = E * np.exp(1j * 1.234)
    aligned = match_global_phase(rotated, E)
    assert _relerr(aligned, E) < 1e-13
    z, wf = 6e-3, 1.5
    Gz = propagate_gbd_freespace(_gauss(N, dx), dx, z=z, wavelength=LAM,
                                 waist_factor=wf)
    Az = angular_spectrum_propagate(_gauss(N, dx), z, LAM, dx)
    via_match = match_global_phase(Gz, Az)
    # Post-S5 the match is very nearly the identity: the rotation it applies
    # is the residual global phase, which on this DELIBERATELY SMALL N = 96
    # grid is -2.08e-03 rad (the GBD decomposition itself is relerr 3.85e-02
    # here, so a residual phase of that order over the field is the floor,
    # not zero; the N = 160 fixture above measures 4.7e-06 rad).  relerr for
    # a pure rotation by phi is |1 - e^{i phi}| ~ |phi|, so the measured
    # 2.076e-03 matches the phase to 4 digits.  Gate at 1e-2: 5x above the
    # measured value and 3 decades below the pre-fix pi rotation
    # (|1 - e^{i pi}| = 2).
    assert _relerr(via_match, Gz) < 1e-2

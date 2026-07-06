"""GBD <-> ASM interoperability converter (v5.21).

GBD-reconstructed and ASM fields of the same beam are the SAME field (same
coordinates, forward direction, no conjugate / flip) up to a global beamlet-Gouy
phase phi0(z) = 2*arctan(z / zR_beamlet).  The converter reconciles it so the two
propagators are exactly interchangeable.
"""
import numpy as np

from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.gbd import (
    asm_field_to_gbd,
    gbd_asm_gouy_phase,
    gbd_field_to_asm,
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


def test_gouy_phase_formula():
    """phi0 = 2*arctan(z/zR_b), zR_b = pi (wf*dx)^2 / lambda -- the measured
    GBD/ASM overlap phase matches the closed form to ~1e-7."""
    N, dx = 160, 4e-6
    E0 = _gauss(N, dx)
    for wf, z in [(1.0, 5e-3), (1.5, 3e-3), (2.5, 8e-3)]:
        Gz = propagate_gbd_freespace(E0, dx, z=z, wavelength=LAM,
                                     waist_factor=wf)
        Az = angular_spectrum_propagate(E0, z, LAM, dx)
        measured = float(np.angle(np.vdot(Az.ravel(), Gz.ravel())))
        pred = gbd_asm_gouy_phase(z, LAM, dx, waist_factor=wf)
        assert abs(float(np.angle(np.exp(1j * (measured - pred))))) < 1e-5


def test_gbd_to_asm_makes_fields_match():
    """After removing phi0, a GBD free-space field equals angular_spectrum_
    propagate to the GBD decomposition accuracy (no free phase fit) -- across a
    non-trivial off-centre + tilted field."""
    N, dx = 192, 4e-6
    for off, tilt, wf, z in [(0.0, 0.0, 1.5, 6e-3),
                             (30e-6, 0.003, 1.0, 4e-3)]:
        E0 = _gauss(N, dx, off=off, tilt=tilt)
        Gz = propagate_gbd_freespace(E0, dx, z=z, wavelength=LAM,
                                     waist_factor=wf)
        Az = angular_spectrum_propagate(E0, z, LAM, dx)
        Gz_asm = gbd_field_to_asm(Gz, z=z, wavelength=LAM, dx=dx,
                                  waist_factor=wf)
        # raw GBD vs ASM differs by the global phase (large); converted matches
        assert _relerr(Gz, Az) > 0.5                     # uncorrected: global phase
        assert _relerr(Gz_asm, Az) < 5e-3                # corrected: decomposition floor


def test_converter_round_trip():
    """asm_field_to_gbd is the exact inverse of gbd_field_to_asm."""
    N, dx = 96, 4e-6
    E = _gauss(N, dx, off=20e-6, tilt=0.002)
    z, wf = 5e-3, 1.5
    back = asm_field_to_gbd(
        gbd_field_to_asm(E, z=z, wavelength=LAM, dx=dx, waist_factor=wf),
        z=z, wavelength=LAM, dx=dx, waist_factor=wf)
    assert _relerr(back, E) < 1e-13


def test_gbd_to_asm_handoff_matches_pure_asm():
    """The interoperability use case: a GBD free-space leg, converted to ASM
    convention, then continued with ASM, equals pure end-to-end ASM (up to the
    GBD decomposition accuracy).  This is a correct GBD->ASM handoff."""
    N, dx = 192, 4e-6
    E0 = _gauss(N, dx, w0=0.12e-3)
    z1, z2, wf = 4e-3, 10e-3, 1.5
    # GBD leg to z1, convert to ASM convention, ASM the rest
    G1 = propagate_gbd_freespace(E0, dx, z=z1, wavelength=LAM, waist_factor=wf)
    G1_asm = gbd_field_to_asm(G1, z=z1, wavelength=LAM, dx=dx, waist_factor=wf)
    hybrid = angular_spectrum_propagate(G1_asm, z2, LAM, dx)
    # pure ASM reference (single step; ASM is exact per step)
    ref = angular_spectrum_propagate(E0, z1 + z2, LAM, dx)
    assert _relerr(hybrid, ref) < 5e-3

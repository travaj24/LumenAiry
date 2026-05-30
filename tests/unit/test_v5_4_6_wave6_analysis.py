"""v5.4.6 Wave 6 regression pins: analysis-physics cluster.

- F-10 : apply_detector conserves flux on non-integer pixel ratios.
- F-11 : ImagePlaneWFE RMS removes piston (Strehl unbiased by a constant).
- F-13 : phase_shift_extract is correct for arbitrary (non-equispaced) shifts.
"""
from __future__ import annotations

import numpy as np

import lumenairy as la

# ---- F-13: arbitrary-shift least-squares phase extraction -------------

def test_phase_shift_extract_arbitrary_shifts():
    from lumenairy.analysis.interferometry import phase_shift_extract
    phi = 0.7
    a, b = 2.0, 1.5
    shifts = [0.0, 1.3, 2.0, 4.5, 5.5]  # deliberately non-equispaced
    frames = [a + b * np.cos(phi - s) * np.ones((3, 3)) for s in shifts]
    phase, mod = phase_shift_extract(frames, shifts=shifts, convention='hardware')
    assert np.allclose(phase, phi, atol=1e-9), (
        f"non-equispaced LSQ phase {phase[0, 0]} != {phi}")
    assert np.allclose(mod, b, atol=1e-9)


def test_phase_shift_extract_equispaced_unchanged():
    """The equispaced default path must be bit-preserved by the general
    LSQ (S^T S is diagonal there)."""
    from lumenairy.analysis.interferometry import phase_shift_extract
    phi = -1.1
    frames = [3.0 + 0.8 * np.cos(phi - 2 * np.pi * i / 4) * np.ones((2, 2))
              for i in range(4)]
    phase, _ = phase_shift_extract(frames, convention='hardware')
    assert np.allclose(phase, phi, atol=1e-9)


# ---- F-10: detector flux conservation on non-integer ratios -----------

def test_detector_flux_conservation_non_integer_ratio():
    """For a uniform field, collected signal per unit detector-FOV-area
    must be ratio-independent (conservation).  The pre-fix box-mean
    over-/under-counted by up to ~25% for non-integer ratios."""
    N, dxf = 240, 1e-6
    E = (1.0e8 + 0j) * np.ones((N, N))  # large counts -> Poisson negligible
    per_area = []
    for ratio in (2.0, 2.5, 3.0):
        pp = ratio * dxf
        npx = int(N / ratio)
        img, _, _ = la.apply_detector(
            E, dxf, pp, n_pixels=npx, quantum_efficiency=1.0,
            exposure_time=1.0, read_noise_e=0.0, seed=0)
        per_area.append(float(img.sum()) / (npx * pp) ** 2)
    spread = (max(per_area) - min(per_area)) / np.mean(per_area)
    assert spread < 0.02, (
        f"detector flux not conserved across pixel ratios: spread={spread:.3f}")


# ---- F-11: ImagePlaneWFE RMS is piston-insensitive --------------------

def test_image_plane_wfe_rms_removes_piston():
    """Adding a constant piston to the OPD must not change rms_waves /
    strehl (a piston does not aberrate the wavefront).

    v5.4.7 (audit AUDIT_V5_4_6 gap #3): build ImagePlaneWFE with ALL
    required dataclass fields so this pin actually runs.  The v5.4.6
    version constructed it with 2 of 8 required fields, hit a TypeError,
    and pytest.skip'd every run -- so the F-11 piston-removal fix shipped
    unverified by its own test.
    """
    from lumenairy.analysis.image_plane_wfe import ImagePlaneWFE

    n = 64
    opd_w = np.linspace(-0.1, 0.1, n)  # waves, zero-mean aberration
    alive = np.ones(n, dtype=bool)
    px = np.linspace(-1.0, 1.0, n)
    py = np.zeros(n)

    def _wfe(opd):
        return ImagePlaneWFE(
            px=px.copy(), py=py.copy(), opd_w=opd, alive=alive.copy(),
            chief_idx=0, img_d_m=0.1, wavelength_m=633e-9, aperture_m=1e-3)

    w0 = _wfe(opd_w.copy())
    w1 = _wfe(opd_w + 5.0)  # +5 waves of pure piston
    assert np.isclose(w0.rms_waves, w1.rms_waves), (
        "rms_waves must be invariant to a constant piston (F-11)")
    assert np.isclose(w0.strehl, w1.strehl)
    # Sanity: a pure-piston wavefront (constant OPD) has zero RMS.
    assert np.isclose(_wfe(np.full(n, 3.0)).rms_waves, 0.0, atol=1e-12)

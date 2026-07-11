"""Subsystem-audit remediation campaign (docs/audits/AUDIT_*_2026_07_07..09).

One growing file, sectioned per audit, pinning the findings fixed in the
chronological subsystem sweep that follows the v5.21 delta audit.
"""
import numpy as np
import pytest

# =========================================================================
# AUDIT 2 -- sources/core.py (AUDIT_SOURCES_CORE_2026_07_07)
# =========================================================================
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    create_annular_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    create_multi_field_sources,
    create_top_hat_beam,
)


def _complex_j_ensemble(Ny=6, Nx=6, nr=40, seed=0):
    """A complex-MCF (linearly-phase-tilted + noise) ensemble -- its
    J(r1, r2) phase sign is nonzero, so the SRC-1 conjugation shows up."""
    rng = np.random.default_rng(seed)
    xs = np.arange(Nx)
    base = np.exp(1j * 0.7 * xs)[None, :] * np.ones((Ny, 1))
    return np.stack([
        base * np.exp(1j * rng.normal(0, 0.3))
        + 0.2 * (rng.standard_normal((Ny, Nx))
                 + 1j * rng.standard_normal((Ny, Nx)))
        for _ in range(nr)])


def test_src1_dense_matches_documented_mcf_convention():
    """SRC-1: the dense J is <E(r1) conj(E(r2))> (the documented convention),
    NOT its conjugate.  Pin it directly against the reference build."""
    ens = _complex_j_ensemble()
    nr = ens.shape[0]
    mcf = PartialCoherenceMCF.from_ensemble(
        ens, dx=1e-6, dy=1e-6, wavelength=0.5e-6, max_full_N=8)
    Em = ens.reshape(nr, -1)
    J_ref = (Em.T @ Em.conj()) / nr           # <E(r_i) conj(E(r_j))>
    assert np.allclose(mcf.J_full, J_ref, atol=1e-12)


def test_src1_dense_and_modal_agree_on_complex_j():
    """SRC-1: the dense and (full-rank) modal branches now agree to machine
    precision on a complex-J ensemble -- before the fix the dense branch was
    the conjugate, flipping the coherence phase sign with grid size."""
    ens = _complex_j_ensemble()
    nr = ens.shape[0]
    n_pix = ens.shape[1] * ens.shape[2]
    full_rank = min(nr, n_pix)                         # SVD rank of J
    kw = dict(dx=1e-6, dy=1e-6, wavelength=0.5e-6)
    dense = PartialCoherenceMCF.from_ensemble(ens, max_full_N=8, **kw)
    modal = PartialCoherenceMCF.from_ensemble(
        ens, max_full_N=4, n_modes=full_rank, **kw)    # full rank = exact
    cd = dense.coherence_at(2, 1, 3, 4)
    cm = modal.coherence_at(2, 1, 3, 4)
    assert abs(cd - cm) < 1e-9
    assert np.sign(np.angle(cd)) == np.sign(np.angle(cm))   # same phase sign


@pytest.mark.parametrize("call", [
    lambda: create_hermite_gauss(32, 1e-6, 0.0, 0.5e-6),        # w0 = 0
    lambda: create_hermite_gauss(32, 1e-6, -1e-3, 0.5e-6),      # w0 < 0
    lambda: create_laguerre_gauss(32, 1e-6, 0.0, 0.5e-6),       # w0 = 0
    lambda: create_top_hat_beam(32, 1e-6, 0.5e-6, diameter=0.0),
    lambda: create_top_hat_beam(32, 1e-6, 0.5e-6, diameter=-1e-3),
    lambda: create_annular_beam(32, 1e-6, 0.5e-6,
                                outer_diameter=1e-3, inner_diameter=2e-3),
    lambda: create_annular_beam(32, 1e-6, 0.5e-6,
                                outer_diameter=-1e-3, inner_diameter=0.0),
])
def test_src2_scale_parameter_guards(call):
    """SRC-2: the four factories missing scale-parameter validation now
    reject non-physical waists / diameters / inverted annuli."""
    with pytest.raises(ValueError):
        call()


def test_src2_valid_calls_still_work():
    assert np.isfinite(create_hermite_gauss(32, 1e-6, 1e-3, 0.5e-6)[0]).all()
    o, x, y = create_annular_beam(32, 1e-6, 0.5e-6, outer_diameter=2e-3,
                                  inner_diameter=1e-3)
    assert np.isfinite(o).all()


def test_src3_empty_field_angles_raises():
    """SRC-3: an empty field_angles now raises instead of silently returning
    (sources=[], x=None, y=None)."""
    with pytest.raises(ValueError, match="field_angles is empty"):
        create_multi_field_sources(32, 1e-6, 0.5e-6, [])


# =========================================================================
# AUDIT 3 -- propagators/*.py (AUDIT_PROPAGATORS_KERNELS_2026_07_07)
# =========================================================================


def test_ds1_farfield_propagate_returns_valid_source():
    """DS-1: ``Source.propagate`` at a far distance auto-selects a pitch-
    CHANGING kernel (fraunhofer/sas); pre-fix it wrapped the raw
    ``(E, dx, dy)`` tuple AS the field (``.shape`` then raised) and kept the
    stale input pitch.  Post-fix the returned Source carries a real ndarray
    field and the kernel's output pitch."""
    from lumenairy.sources.core import Source

    N, dx, wl = 64, 8e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (6 * dx) ** 2).astype(np.complex128)
    src = Source(E=E, dx=dx, dy=dx, wavelength=wl)
    out = src.propagate(method='auto', z=5.0)  # N_F << 1 -> far-field kernel
    assert isinstance(out.E, np.ndarray)
    assert out.E.shape == E.shape
    assert np.all(np.isfinite(out.E))
    # A far-field kernel changes the pitch to ~ lambda*z/(N*dx); it must NOT
    # be the stale input pitch.
    assert out.dx != dx


def test_ds1_asm_preserves_anamorphic_dy():
    """DS-1 (anamorphic re-thread): a pitch-PRESERVING kernel keeps the
    input's distinct y-pitch instead of collapsing dy == dx."""
    from lumenairy.sources.core import Source

    N, dx, dy, wl = 32, 5e-6, 7e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (5 * dx) ** 2).astype(np.complex128)
    out = Source(E=E, dx=dx, dy=dy, wavelength=wl).propagate(method='asm',
                                                              z=1e-4)
    assert out.dx == dx
    assert out.dy == dy


def test_hf1_d4sigma_waist_convention():
    """HF-1: the LG-basis waist is the 1/e^2 radius == D4sigma/2, NOT the
    audit's D4sigma/4 (== second-moment sigma == half the waist).  Pin the
    convention on a known Gaussian."""
    from lumenairy.analysis.core import beam_d4sigma

    N, dx = 256, 1e-6
    w0 = 30e-6  # 1/e^2 amplitude radius
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / w0 ** 2).astype(np.complex128)
    d4x, d4y = beam_d4sigma(E, dx=dx)
    # D4sigma == 2 * w0  ->  w0 == d4x / 2 (the fixed factor); d4x/4 would be
    # w0/2, which is what the pre-fix code / audit prescription used.
    assert abs(d4x / 2.0 - w0) / w0 < 0.02
    assert abs(d4x / 4.0 - w0) / w0 > 0.4  # the WRONG factor is far off


def test_vd1_immersion_na_raises():
    """VD-1: an immersion NA (>= 1) is rejected instead of silently clamped
    to the 89.2 deg air cone."""
    from lumenairy.propagators.vector_diffraction import richards_wolf_focus

    N, dx, wl, f = 32, 2e-6, 550e-9, 1e-3
    pupil = np.ones((N, N), dtype=np.complex128)
    with pytest.raises(ValueError, match="NA must be in"):
        richards_wolf_focus(pupil, wl, 1.4, f, dx)
    # A valid air NA still works.
    out = richards_wolf_focus(pupil, wl, 0.6, f, dx)
    assert out is not None


def test_pk1_sas_beyond_zlimit_warns():
    """PK-1: propagating past the SAS validity bound emits a RuntimeWarning
    (was an easy-to-miss stdout print only)."""
    from lumenairy.propagators.sas import scalable_angular_spectrum_propagate

    N, dx, wl = 64, 4e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (8 * dx) ** 2).astype(np.complex128)
    with pytest.warns(RuntimeWarning, match="validity"):
        scalable_angular_spectrum_propagate(E, 10.0, wl, dx)


def test_hfpi2_invalid_sampling_raises():
    """HFPI-2: the ``sampling`` selector is now validated up front (and
    actually dispatched)."""
    from lumenairy.propagators.hfpi import (
        propagate_hfpi_through_prescription,
    )

    N, dx, wl = 16, 4e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    with pytest.raises(ValueError, match="sampling must be"):
        propagate_hfpi_through_prescription(
            E, dx, {'surfaces': []}, wavelength=wl, n_paths=8,
            sampling='bogus')


def test_sy2_anamorphic_pitch_changing_branch_raises():
    """SY-2: the pitch-CHANGING chain branches (fresnel/sas/turbulence)
    assume a square grid; an anamorphic working pitch now raises instead of
    silently mis-resampling the y-axis with the x-ratio."""
    from lumenairy.propagators.system import propagate_through_system

    N, dx, dy, wl = 32, 5e-6, 7e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    elements = [{'type': 'propagate', 'z': 1e-3, 'method': 'fresnel'}]
    with pytest.raises(ValueError, match="square grid pitch"):
        propagate_through_system(E, elements, wl, dx, dy=dy)


def test_dispatch_asm_znone_returns_copy():
    """Dispatch nit: ``method='asm', z=None`` returns a COPY, not the input
    array itself (a caller mutating the output must not corrupt the source)."""
    from lumenairy.propagators.dispatch import _dispatch_to_method

    N, dx, wl = 16, 4e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = _dispatch_to_method('asm', E, z=None, wavelength=wl, dx=dx,
                              prescription=None, output_grid=None,
                              output_dx=None)
    assert out is not E
    out[0, 0] = 999.0
    assert E[0, 0] == 1.0

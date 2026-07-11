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

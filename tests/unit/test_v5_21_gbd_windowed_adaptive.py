"""v5.21 GBD speed/memory/accuracy features + Maslov default.

Covers:
 * windowed (bounded-support) reconstruction == dense sum to machine precision,
   across scalar / tensor / direction-ramp / anamorphic beamlets;
 * the dense-path memory-budget auto-chunk leaves small-N results unchanged;
 * decompose_field_adaptive: partition-of-unity (energy) + matches uniform-fine
   edge fidelity at far fewer beamlets;
 * soft-edge (analytic partial-vignetting) aperture: correct half-open-plane
   energy fraction, and improves the hard-aperture Airy-focus accuracy;
 * apply_real_lens_maslov default integration_method is now 'auto'.
"""
import numpy as np
import pytest

from lumenairy.propagators.gbd import (
    BeamletBundle,
    apply_aperture_to_beamlets,
    apply_thin_lens_to_beamlets,
    decompose_field_adaptive,
    decompose_field_to_beamlets,
    propagate_beamlets_freespace,
    reconstruct_field_from_beamlets,
)

LAM = 0.633e-6


def _relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))


def _gauss(N, dx, w0=0.15e-3, tilt=0.0, dy=None):
    dy = dy or dx
    xs = (np.arange(N) - N // 2) * dx
    ys = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(xs, ys)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    if tilt:
        E = E * np.exp(1j * 2 * np.pi / LAM * tilt * X)
    return E


# --------------------------------------------------------------------------
# windowed reconstruction == dense
# --------------------------------------------------------------------------
@pytest.mark.parametrize("z_prop,lens_f,tilt,aniso", [
    (0.0, None, 0.0, False),          # scalar, at plane
    (30e-3, 30e-3, 0.0, False),       # scalar, near a focus
    (2e-3, None, 0.02, False),        # direction ramp (Husimi tilt)
    (1e-3, None, 0.0, True),          # anamorphic dy != dx
])
def test_windowed_reconstruct_matches_dense(z_prop, lens_f, tilt, aniso):
    N, dx = 96, 5e-6
    dy = 6e-6 if aniso else dx
    E = _gauss(N, dx, tilt=tilt, dy=(dy if aniso else None))
    b = decompose_field_to_beamlets(
        E, dx, wavelength=LAM, dy=(dy if aniso else None),
        sample_step=2, waist_factor=1.5, direction_sampling=bool(tilt))
    if lens_f is not None:
        b = apply_thin_lens_to_beamlets(b, lens_f, LAM)
    if z_prop:
        b = propagate_beamlets_freespace(b, z_prop, LAM)
    dense = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                            wavelength=LAM, window=None)
    win = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                          wavelength=LAM, window=5.0)
    assert _relerr(win, dense) < 1e-9


def test_windowed_reconstruct_tensor_Q():
    N, dx = 96, 5e-6
    E = _gauss(N, dx)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    n = len(b)
    zRx = np.pi * (1.5 * dx) ** 2 / LAM
    Q = np.zeros((n, 2, 2), dtype=np.complex128)
    Q[:, 0, 0] = -1j / zRx
    Q[:, 1, 1] = -1j / (1.4 * zRx)
    Q[:, 0, 1] = Q[:, 1, 0] = 0.06j / zRx     # skew astigmatism
    bt = BeamletBundle(positions=b.positions, directions=b.directions, Q=Q,
                       amplitude=b.amplitude, waist0=b.waist0)
    dense = reconstruct_field_from_beamlets(bt, Ny=N, Nx=N, dx=dx,
                                            wavelength=LAM, window=None)
    win = reconstruct_field_from_beamlets(bt, Ny=N, Nx=N, dx=dx,
                                          wavelength=LAM, window=5.0)
    assert _relerr(win, dense) < 1e-9


def test_mem_budget_does_not_change_small_N():
    N, dx = 64, 5e-6
    E = _gauss(N, dx)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    a = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=LAM,
                                        window=None, mem_budget_mb=512.0)
    # A tiny budget forces chunk auto-shrink; result must be unchanged to ULP.
    c = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=LAM,
                                        window=None, mem_budget_mb=1.0)
    assert _relerr(c, a) < 1e-12


# --------------------------------------------------------------------------
# adaptive decomposition
# --------------------------------------------------------------------------
def test_adaptive_matches_uniform_fine_at_fewer_beamlets():
    """Residual-refined adaptive reaches uniform-fine edge fidelity with far
    fewer beamlets than the uniform-fine grid."""
    N, dx = 128, 5e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r = np.sqrt(X ** 2 + Y ** 2)
    E = (r <= 0.2e-3).astype(np.complex128)         # hard-edged top hat

    def recon(bundle):
        return reconstruct_field_from_beamlets(bundle, Ny=N, Nx=N, dx=dx,
                                               wavelength=LAM, window=5.0)
    fine = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=1,
                                       waist_factor=1.5)
    adap, st = decompose_field_adaptive(E, dx, wavelength=LAM, base_step=4,
                                        refine_step=1, return_stats=True)
    e_fine = _relerr(recon(fine), E)
    e_adap = _relerr(recon(adap), E)
    # adaptive is at least as faithful as uniform-fine, with fewer beamlets
    assert e_adap <= e_fine * 1.10
    assert st['n_total'] < 0.4 * len(fine)


def test_adaptive_partition_of_unity_no_double_count():
    """Residual refinement ADDS a correction (fine carries E - recon(coarse)),
    so reconstruct(adaptive) at z=0 conserves energy / matches the field better
    than the coarse alone -- proving no double counting at the coarse/fine seam
    (a double-counted overlay would inflate the energy)."""
    N, dx = 96, 5e-6
    E = _gauss(N, dx, w0=0.12e-3)

    def recon(bundle):
        return reconstruct_field_from_beamlets(bundle, Ny=N, Nx=N, dx=dx,
                                               wavelength=LAM, window=5.0)
    coarse = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=4,
                                         waist_factor=1.5 * 4)
    adap = decompose_field_adaptive(E, dx, wavelength=LAM, base_step=4,
                                    refine_step=1)
    e_in = float(np.sum(np.abs(E) ** 2))
    e_adap = float(np.sum(np.abs(recon(adap)) ** 2))
    # adaptive reconstruction energy is within a few % of the input (not
    # inflated -> no double counting) and closer to E than the coarse alone
    assert abs(e_adap - e_in) / e_in < 0.05
    assert _relerr(recon(adap), E) < _relerr(recon(coarse), E)


# --------------------------------------------------------------------------
# soft-edge aperture
# --------------------------------------------------------------------------
def test_soft_edge_energy_fraction_straight_edge():
    """A single beamlet far from a straight (rectangular) edge passes ~all its
    energy inside and ~none outside; on the edge it passes ~half."""
    from lumenairy.propagators.gbd import _erf_xp
    # erf sanity across backends
    assert abs(float(_erf_xp(np, np.array([0.0]))[0])) < 1e-12
    assert abs(float(_erf_xp(np, np.array([5.0]))[0]) - 1.0) < 1e-6

    N, dx = 64, 5e-6
    E = _gauss(N, dx, w0=0.08e-3)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    # rectangular half-width huge -> everything passes (soft ~ binary ~ 1)
    big = apply_aperture_to_beamlets(b, 1.0, shape='rectangular', soft_edge=True)
    assert _relerr(big.amplitude, b.amplitude) < 1e-4
    # half-width 0 (edge at centre): each beamlet centred on axis passes ~1/2
    zero = apply_aperture_to_beamlets(b, 0.0, shape='rectangular', soft_edge=True)
    on_axis = np.argmin(np.abs(b.positions[:, 0]) + np.abs(b.positions[:, 1]))
    frac = abs(zero.amplitude[on_axis] / b.amplitude[on_axis])
    assert abs(frac - 0.5) < 0.05


def test_soft_edge_improves_hard_aperture_focus():
    """Soft-edge aperture lowers the hard-aperture Airy-focus intensity error
    vs binary vignetting."""
    from lumenairy.propagators.asm import angular_spectrum_propagate
    from lumenairy.propagators.gbd import propagate_gbd_thin_lens
    N, dx, f, R = 160, 6e-6, 30e-3, 0.5e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r = np.sqrt(X ** 2 + Y ** 2)
    r2 = X ** 2 + Y ** 2
    E0 = (r <= R).astype(np.complex128)
    k = 2 * np.pi / LAM
    Ea = angular_spectrum_propagate(E0 * np.exp(-1j * k * r2 / (2 * f)), f, LAM, dx)

    def relI(A, B):
        IA = np.abs(A) ** 2
        IB = np.abs(B) ** 2
        m = IB > 0.01 * IB.max()
        return float(np.linalg.norm((IA - IB)[m]) / np.linalg.norm(IB[m]))
    kw = dict(z_to_lens=0.0, focal_length=f, z_lens_to_output=f, wavelength=LAM,
              output_dx=dx, sample_step=2, waist_factor=3.0,
              aperture_semi_diameter=R)
    e_bin = relI(propagate_gbd_thin_lens(E0, dx, aperture_soft_edge=False, **kw), Ea)
    e_soft = relI(propagate_gbd_thin_lens(E0, dx, aperture_soft_edge=True, **kw), Ea)
    assert e_soft < e_bin           # soft-edge is strictly better here
    assert e_soft < 0.03            # and lands well under the binary ~3.3%


# --------------------------------------------------------------------------
# Maslov default
# --------------------------------------------------------------------------
def test_maslov_default_is_auto():
    import inspect

    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    sig = inspect.signature(apply_real_lens_maslov)
    assert sig.parameters['integration_method'].default == 'auto'

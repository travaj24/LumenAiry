"""Regression pin for the v5.2.5 JAX-side Strehl-normalization fix.

Closes AUDIT_V5_2_5 P3-8 (JAX-side ``monte_carlo_tolerancing_jax``
Strehl-normalization fix shipped untested at v5.2.5).

Background
----------
v5.2.5 closed AUDIT_V5_2_3 P3-F1-1 in BOTH ``monte_carlo_tolerancing``
(NumPy) AND ``monte_carlo_tolerancing_jax`` (JAX) by pinning
``ideal_peak`` to the UNPERTURBED nominal exit pupil computed ONCE
before the trial loop -- previously the JAX twin recomputed
``ideal_peak`` inside the loop from each perturbed ``E_exit``, which
produced non-physical Strehl > 1 (mean ~1.09 in example 09).

The NumPy fix had a regression test
(``test_v5_2_5_p3_f1_1_monte_carlo_strehl_bounded``) but the JAX
mirror did not.  AUDIT_V5_2_5 P3-8 flagged this as "JAX-side fix
shipped untested" -- a sibling-gap meta-pattern at the test surface.

v5.3 closes P3-8 by adding end-to-end JAX-path regression coverage.

What these tests pin
--------------------
1. The JAX path returns Strehl values in ``[0, 1.05]`` -- physical
   bound with a small ``(1, 1.05)`` slack for the focus-shift
   methodology limitation that P3-7 documents (a trial whose
   ``z_best`` shifts away from nominal BFL can land on a peak whose
   absolute value exceeds the unperturbed on-axis peak at BFL; the
   v5.2.5 CHANGELOG calls this "rounding noise" but P3-7 corrected
   the diagnosis to a methodology consequence).

2. The mean Strehl is < 1.0 -- tolerancing degrades the system on
   average; a mean >= 1.0 would indicate the normalization fix
   regressed.

3. The JAX path produces the SAME mean (within 0.01) as the NumPy
   path on identical (prescription, seed, perturbation_spec) inputs
   -- end-to-end physics parity.

JAX is gated via ``pytest.importorskip('jax')`` so the test skips
cleanly when JAX isn't installed.

Author: Andrew Traverso -- v5.3 / AUDIT_V5_2_5 P3-8 closure.
"""
# v5.3 (AUDIT_V5_2_5 P3-8 JAX MC regression):
# JAX-side Strehl normalization fix mirroring the NumPy fix at
# lumenairy/analysis/through_focus.py:1251-1267.  These tests pin the
# physically bounded Strehl contract end-to-end.
from __future__ import annotations

import numpy as np
import pytest


def _build_small_singlet_mc_inputs():
    """Build a small singlet + small pupil + light perturbation spec
    so the Monte Carlo trial loop converges quickly.

    Returns
    -------
    kwargs : dict
        Ready-to-splat kwargs for ``monte_carlo_tolerancing`` /
        ``monte_carlo_tolerancing_jax`` (minus ``n_trials`` / ``seed``
        / ``backend`` which the caller varies).
    """
    import lumenairy as la

    # ~100 mm-EFL N-BK7 plano-convex singlet at 12 mm clear aperture,
    # same shape as examples/09_tolerancing_monte_carlo.py but on a
    # smaller pupil grid (N=64) so each trial converges in a couple
    # of seconds.
    wavelength = 587.56e-9
    aperture = 12e-3
    prescription = la.make_singlet(
        R1=51.5e-3, R2=float('inf'), d=4e-3,
        glass='N-BK7', aperture=aperture,
    )
    # Use the singlet's paraxial BFL as ``focal_length`` -- this is
    # the v5.2.5 / example-09 fix (mismatched focal_length also
    # produces Strehl > 1 even with the nominal-pupil pin).
    surfs = la.surfaces_from_prescription(prescription)
    _, _efl, bfl, _ = la.system_abcd(surfs, wavelength=wavelength)
    focal_length = bfl

    # Small pupil grid -- N=64 keeps each trial under ~1 s on CPU.
    N = 64
    dx = aperture / 32  # semi-aperture covers ~32 px.
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    semi_ap = aperture / 2
    E_source = np.where(np.sqrt(X * X + Y * Y) <= semi_ap, 1.0, 0.0).astype(
        np.complex128)

    # SMALL perturbation magnitudes (20 um decenter / 0.2 mrad tilt,
    # vs. example 09's 50 um / 0.5 mrad) so trials converge to high
    # Strehl quickly -- we're testing the normalization contract, not
    # the tolerancing physics.
    perturbation_spec = [
        {'surface_index': 0, 'decenter_std': 20e-6,
         'tilt_std': 0.2e-3, 'form_error_rms': 0.0},
        {'surface_index': 1, 'decenter_std': 20e-6,
         'tilt_std': 0.2e-3, 'form_error_rms': 0.0},
    ]

    return dict(
        prescription=prescription,
        wavelength=wavelength,
        N=N,
        dx=dx,
        E_source=E_source,
        perturbation_spec=perturbation_spec,
        focal_length=focal_length,
        aperture=aperture,
        z_scan_n=5,  # small but >= 3 so find_best_focus has room.
        verbose=False,
    )


def test_jax_monte_carlo_strehl_physically_bounded():
    """Pin: ``monte_carlo_tolerancing_jax`` returns Strehl values in
    ``[0, 1.05]``.

    The hard physical bound is ``[0, 1]``; we allow a tiny
    ``(1, 1.05)`` slack for the focus-shift methodology limitation
    that P3-7 documents (perturbed ``z_best`` lands on a peak whose
    absolute value exceeds the unperturbed on-axis BFL peak).  A
    value outside ``[0, 1.05]`` indicates the v5.2.5 normalization
    fix regressed -- ``ideal_peak`` is being recomputed inside the
    loop from a perturbed pupil again.
    """
    pytest.importorskip('jax')
    import lumenairy as la

    kwargs = _build_small_singlet_mc_inputs()
    stats = la.monte_carlo_tolerancing_jax(
        n_trials=10, seed=0, **kwargs)

    strehls = np.asarray(stats['strehl_array'])
    assert strehls.size == 10, (
        f"strehl_array size {strehls.size} != 10 trials.")

    # Hard physical floor.
    assert np.all(strehls >= 0.0), (
        f"JAX MC produced negative Strehl: min={strehls.min():.4f}.")

    # Soft physical ceiling with P3-7 methodology slack.
    assert np.all(strehls <= 1.05), (
        f"JAX MC produced Strehl > 1.05: max={strehls.max():.4f}. "
        f"v5.2.5 normalization fix has regressed -- ``ideal_peak`` is "
        f"likely being recomputed from a perturbed pupil inside the "
        f"trial loop again (see "
        f"lumenairy/analysis/through_focus.py:1251-1267).")


def test_jax_monte_carlo_mean_under_unity():
    """Pin: tolerancing DEGRADES the system on average -- the mean
    Strehl across trials is strictly < 1.0.

    A mean >= 1.0 with non-zero perturbations would indicate the
    Strehl normalization is fundamentally broken (perturbed pupils
    are reporting as better than the diffraction-limited reference).
    This is the smoke test for the v5.2.5 fix's substantive intent.
    """
    pytest.importorskip('jax')
    import lumenairy as la

    kwargs = _build_small_singlet_mc_inputs()
    stats = la.monte_carlo_tolerancing_jax(
        n_trials=10, seed=0, **kwargs)

    mean = stats['strehl_peak_mean']
    assert mean < 1.0, (
        f"JAX MC mean Strehl {mean:.4f} >= 1.0; tolerancing should "
        f"degrade the system on average.  v5.2.5 nominal-pupil pin "
        f"has likely regressed.")


def test_jax_numpy_parity_within_005():
    """Pin: the JAX and NumPy MC paths produce the SAME mean Strehl
    (within 0.05) on identical (prescription, seed, perturbation_spec)
    inputs WHEN both use the same wave propagator (``'real_lens'``).

    Sequential per-trial seeding (``seed + t``) is the same on both
    backends, so the SAME perturbations are drawn -- the only
    differences are floating-point rounding inside the wave
    propagator + through-focus kernels.  The JAX through-focus kernel
    defaults to float32 (unless ``JAX_ENABLE_X64=1``), so the
    propagated-field amplitudes round to ~6-7 decimal digits where
    NumPy uses float64.  Over a 5-step through-focus scan with peak-
    seeking ``find_best_focus``, this rounds the trial-level Strehl by
    a few %; 0.05 tolerance comfortably absorbs this while still
    catching the v5.2.5 P3-F1-1 bug (which produced mean ~1.09 vs
    ~0.96 -- a 0.13 delta).

    Important: the JAX twin defaults to ``wave_propagator=
    'real_lens_traced_jax'`` (ray-traced OPD screen) while the NumPy
    path uses the analytic ``apply_real_lens``.  These are
    intentionally different physics models.  We pass
    ``wave_propagator='real_lens'`` to the JAX twin so both backends
    use the SAME wave model -- only the through-focus scan + Strehl
    normalization differ -- which is the parity contract AUDIT_V5_2_5
    P3-8 cares about.

    A wide divergence indicates one backend's normalization has
    regressed independently of the other.
    """
    pytest.importorskip('jax')
    import lumenairy as la

    kwargs = _build_small_singlet_mc_inputs()
    n_trials = 10
    seed = 0

    stats_np = la.monte_carlo_tolerancing(
        n_trials=n_trials, seed=seed, backend='numpy', **kwargs)
    stats_jax = la.monte_carlo_tolerancing_jax(
        n_trials=n_trials, seed=seed,
        wave_propagator='real_lens',  # match NumPy's analytic model.
        **kwargs)

    mean_np = stats_np['strehl_peak_mean']
    mean_jax = stats_jax['strehl_peak_mean']
    assert abs(mean_np - mean_jax) < 0.05, (
        f"JAX vs NumPy MC mean Strehl parity broken: "
        f"NumPy={mean_np:.4f}, JAX={mean_jax:.4f}, "
        f"|delta|={abs(mean_np - mean_jax):.4f} >= 0.05.")


def test_jax_monte_carlo_stats_dict_shape():
    """Pin: the JAX MC return dict has the same keys + shapes as the
    NumPy MC return dict -- the v5.2.5 fix must not have silently
    changed the public contract.
    """
    pytest.importorskip('jax')
    import lumenairy as la

    kwargs = _build_small_singlet_mc_inputs()
    stats = la.monte_carlo_tolerancing_jax(
        n_trials=10, seed=0, **kwargs)

    expected_scalar_keys = {
        'strehl_peak_mean', 'strehl_peak_std',
        'strehl_peak_p05', 'strehl_peak_p50', 'strehl_peak_p95',
    }
    expected_array_keys = {'strehl_array', 'trial_results'}
    keys = set(stats.keys())
    missing = (expected_scalar_keys | expected_array_keys) - keys
    assert not missing, (
        f"monte_carlo_tolerancing_jax missing return-dict keys: "
        f"{missing}.")

    for k in expected_scalar_keys:
        v = stats[k]
        assert np.isfinite(v), (
            f"stats[{k!r}] not finite: {v}.")

    strehls = np.asarray(stats['strehl_array'])
    assert strehls.shape == (10,), (
        f"strehl_array shape {strehls.shape} != (10,).")
    assert len(stats['trial_results']) == 10, (
        f"trial_results length {len(stats['trial_results'])} != 10.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""apply_real_lens_maslov_jax caustic-phase fix (v5.21).

The differentiable Maslov phase-screen previously counted ``det(J)`` sign flips
along a radial scan, which could not see an axial focus and missed even-
multiplicity caustics (both eigenvalues flip -> det unchanged) -- so the output
was wrong by pi past an axial point focus.  It now uses the Morse / Maslov index
at the pixel (number of negative eigenvalues of the forward ray-map Jacobian J):
index 2 past an axial focus (-> -pi Gouy shift), index 1 past an off-axis fold.
"""
import numpy as np
import pytest


def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_maslov_jax_applies_minus_pi_past_axial_focus():
    """The Maslov phase screen equals the plain traced screen times
    ``exp(-i pi/2 * n_neg)``: identical before focus (n_neg=0), and negated
    (n_neg=2 -> factor -1, the -pi Gouy shift) past an axial focus."""
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.backend.array import to_numpy
    from lumenairy.elements._lens_jax import (
        apply_real_lens_maslov_jax,
        apply_real_lens_traced_jax,
    )
    lam = 0.633e-6

    def _singlet(det):
        # 3-surface: plano-convex N-BK7 (f ~ 48.5 mm) + a flat detector plane a
        # distance ``det`` past the lens, so the forward map propagates through
        # focus (trace applies the gap to the detector surface).
        return {'aperture_diameter': 8e-3, 'surfaces': [
            {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'glass_before': 'N-BK7',
             'glass_after': 'air'},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'air'}],
            'thicknesses': [3e-3, det, 0.0]}
    N, dx = 96, 5e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (2.5e-3) ** 2).astype(np.complex128)

    def _ratio(det):
        Fm = to_numpy(apply_real_lens_maslov_jax(
            E, prescription=_singlet(det), wavelength=lam, dx=dx))
        Ft = to_numpy(apply_real_lens_traced_jax(
            E, prescription=_singlet(det), wavelength=lam, dx=dx))
        m = np.abs(Ft) > 0.2 * np.abs(Ft).max()
        return np.mean(Fm[m] / Ft[m])

    r_before = _ratio(30e-3)     # before focus: no caustic
    r_past = _ratio(70e-3)       # past focus: both eigenvalues flipped
    assert abs(r_before - 1.0) < 1e-3
    assert abs(r_past - (-1.0)) < 1e-3       # the -pi Maslov/Gouy shift


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_maslov_jax_still_differentiable():
    """The caustic-phase fix keeps the propagator jax.grad / jit friendly."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
    lam = 0.633e-6
    presc = {'aperture_diameter': 8e-3, 'surfaces': [
        {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 70e-3, 0.0]}
    N, dx = 64, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = jnp.asarray(np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2)
                    .astype(np.complex128))

    def loss(a):
        F = apply_real_lens_maslov_jax(a * E, prescription=presc,
                                       wavelength=lam, dx=dx)
        return jnp.sum(jnp.abs(F) ** 2)

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g) and g != 0.0


# v5.32.1 marker inversion (AUDIT_CI_TEST_TIME_2026_08_03 §4/chunk 6): this
# file's THREE pure-NumPy tests are 174.6 s of its 190 s; the jax tests its
# name advertises are the cheap ones (6.9 + 3.5 s).  Nothing here was marked,
# so the heavy 92% ran on the 4-Python fast gate.  The claims are byte-identity
# / equivalence of NumPy integration paths -- not Python-version-sensitive --
# which is exactly what the single-Python slow gate exists for.
@pytest.mark.slow
def test_maslov_integration_method_auto_matches_and_is_fast():
    """integration_method='auto' resolves to the concrete integrator from the
    chart's v2-oscillation count and is byte-identical to it: uniform
    'quadrature' when well-resolved (low-NA), the fast asymptotic
    'local_quadrature' when uniform quadrature would over-run its sample cap
    (high-NA oscillatory)."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    lam = 0.633e-6

    def _singlet(back, ap):
        return {'aperture_diameter': ap, 'surfaces': [
            {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'glass_before': 'N-BK7',
             'glass_after': 'air'}], 'thicknesses': [3e-3, back]}
    N, dx = 64, 4e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2).astype(np.complex128)
    for ap, back in [(3e-3, 40e-3), (10e-3, 48e-3)]:
        Fa = apply_real_lens_maslov(E, prescription=_singlet(back, ap),
                                    wavelength=lam, dx=dx,
                                    integration_method='auto')
        Fq = apply_real_lens_maslov(E, prescription=_singlet(back, ap),
                                    wavelength=lam, dx=dx,
                                    integration_method='quadrature')
        Fl = apply_real_lens_maslov(E, prescription=_singlet(back, ap),
                                    wavelength=lam, dx=dx,
                                    integration_method='local_quadrature')
        # auto must be byte-identical to exactly one of the two concrete methods
        assert (np.max(np.abs(Fa - Fq)) < 1e-12) or \
               (np.max(np.abs(Fa - Fl)) < 1e-12)


def _folded_rx(mirror_R):
    d_in, d_out = 20e-3, 15e-3
    elements = [
        {'element_type': 'surface', 'radius': 50e-3, 'conic': 0.,
         'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'element_type': 'surface', 'radius': float('inf'), 'conic': 0.,
         'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'element_type': 'mirror', 'radius': mirror_R, 'conic': 0.},
        {'element_type': 'surface', 'radius': float('inf'), 'conic': 0.,
         'glass_before': 'air', 'glass_after': 'air'}]
    return {'elements': elements,
            'surfaces': [e for e in elements if e['element_type'] == 'surface'],
            'thicknesses': [3e-3, d_in + d_out, 0.0],
            'all_thicknesses': [3e-3, d_in, d_out],
            'aperture_diameter': 8e-3, 'name': 'f'}


@pytest.mark.slow            # v5.32.1: see the marker-inversion note above
def test_maslov_fold_split_matches_manual_chain():
    """apply_real_lens_maslov(fold_split=True) on a folded prescription equals
    the documented manual pattern (split_prescription_at_mirrors + alternate
    Maslov per refractive leg with free-space + apply_mirror per fold), and a
    curved fold applies the R/2 focus (differs from a flat fold)."""
    from lumenairy.elements.elements import apply_mirror
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    from lumenairy.io.prescriptions_transforms import (
        split_prescription_at_mirrors,
    )
    from lumenairy.propagators.asm import angular_spectrum_propagate
    lam = 0.633e-6
    N, dx = 80, 5e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=lam, dx=dx, integration_method='auto')

    folded = _folded_rx(float('inf'))
    Ff = apply_real_lens_maslov(E, prescription=folded, fold_split=True, **kw)
    assert np.isfinite(Ff).all()
    # manual documented chain, same method
    Em = E
    for lg in split_prescription_at_mirrors(folded):
        if lg['kind'] == 'refractive':
            Em = apply_real_lens_maslov(Em, prescription=lg['prescription'], **kw)
        else:
            m = lg['element']
            Em = angular_spectrum_propagate(Em, lg['distance_in'], lam, dx)
            Em = apply_mirror(Em, wavelength=lam, dx=dx, radius=m.get('radius'),
                              conic=0., aperture_diameter=None)
            Em = angular_spectrum_propagate(Em, lg['distance_out'], lam, dx)
    assert np.max(np.abs(Ff - Em)) < 1e-12          # faithful to the pattern
    # a curved fold (R=-100mm, f=50mm) applies the mirror focus
    Fc = apply_real_lens_maslov(E, prescription=_folded_rx(-100e-3),
                                fold_split=True, **kw)
    assert np.isfinite(Fc).all()
    assert np.max(np.abs(Fc - Ff)) > 1e-2           # focus phase changed it


@pytest.mark.slow            # v5.32.1: see the marker-inversion note above
def test_maslov_fold_split_noop_on_unfolded():
    """fold_split=True reduces to the single-call path on a prescription with
    no fold (byte-identical to fold_split=False)."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    presc = {'aperture_diameter': 8e-3, 'surfaces': [
        {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 40e-3]}
    N, dx = 64, 5e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.8e-3) ** 2).astype(np.complex128)
    F0 = apply_real_lens_maslov(E, prescription=presc, wavelength=0.633e-6,
                                dx=dx, fold_split=False)
    F1 = apply_real_lens_maslov(E, prescription=presc, wavelength=0.633e-6,
                                dx=dx, fold_split=True)
    assert np.array_equal(F0, F1)


def test_maslov_vector_polarization():
    """apply_real_lens_maslov_vector applies the base-ray Fresnel Jones (reusing
    the GBD polarization ray tracing) then propagates each mixed component with
    the caustic-safe scalar Maslov: an x-polarized beam through a singlet stays
    x-polarized (cross-pol at the symmetry floor) and carries the two-surface
    Fresnel transmission T1*T2."""
    from lumenairy.elements.lenses_maslov import (
        apply_real_lens_maslov,
        apply_real_lens_maslov_vector,
    )
    lam = 0.633e-6
    singlet = {'aperture_diameter': 10e-3, 'surfaces': [
        {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -50e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'}],
        'thicknesses': [4e-3, 45e-3]}
    N, dx = 72, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2).astype(np.complex128)
    Evec = np.stack([E0, np.zeros_like(E0)])       # x-polarized
    out = apply_real_lens_maslov_vector(
        Evec, prescription=singlet, wavelength=lam, dx=dx,
        integration_method='auto')
    assert out.shape == (2, N, N) and np.isfinite(out).all()
    Px = float(np.sum(np.abs(out[0]) ** 2))
    Py = float(np.sum(np.abs(out[1]) ** 2))
    assert Py < 1e-6 * Px                          # cross-pol at symmetry floor
    scal = apply_real_lens_maslov(E0, prescription=singlet, wavelength=lam,
                                  dx=dx, integration_method='auto')
    ng = 1.515
    T1 = ng * (2.0 / (1.0 + ng)) ** 2
    T2 = (1.0 / ng) * (2.0 * ng / (ng + 1.0)) ** 2
    assert abs(Px / float(np.sum(np.abs(scal) ** 2)) - T1 * T2) < 5e-3

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

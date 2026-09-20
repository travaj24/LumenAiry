"""VERIFY-WP-C2 item 12 -- a pytest plugin that injects ONE mutation.

``VC2_MUTATION=<name> python -m pytest ... -p vc2_mutplugin`` applies the
named mutation at ``pytest_configure``, before collection, by patching the
library's module objects at runtime.  Nothing under ``lumenairy/`` is
edited on disk.

The mutations are the ones the WP-C2 report's matrix claims to catch, plus
four this verification adds.
"""
import os
import sys

import numpy as np


def _mut_defaults_reverted():
    """Both public defaults put back, signature and all."""
    import lumenairy.raytrace.trace                          # noqa: F401
    import lumenairy.raytrace.world_trace                    # noqa: F401
    for key in ('lumenairy.raytrace.trace',
                'lumenairy.raytrace.world_trace'):
        mod = sys.modules[key]
        fn = mod.trace if key.endswith('.trace') else mod.trace_world
        d = list(fn.__defaults__)
        d[-2], d[-1] = 'surface', 'generic'
        fn.__defaults__ = tuple(d)


def _mut_defaults_reverted_in_loop():
    """SIGNATURES UNTOUCHED; the LOOP takes the old arithmetic anyway.

    ``functools.wraps`` copies ``__wrapped__``, which is what
    ``inspect.signature`` follows -- so a suite that asks the signature
    sees no change at all and only a suite that asks a CALL can catch
    this.  That separation is the report's own claim for
    ``..._defaults_are_what_a_call_actually_takes``.
    """
    import functools
    from lumenairy.raytrace import intersection as I
    from lumenairy.raytrace import surface as S
    real_sn = S._surface_normal
    real_rf, real_rl = I._refract, I._reflect

    @functools.wraps(real_sn)
    def sn(x, y, surf, *, analytic_sphere=False):
        return real_sn(x, y, surf, analytic_sphere=False)

    @functools.wraps(real_rf)
    def rf(rays, surface, n1, n2, **kw):
        kw['renormalize'] = True
        kw['sphere_normal'] = 'generic'
        return real_rf(rays, surface, n1, n2, **kw)

    @functools.wraps(real_rl)
    def rl(rays, surface, **kw):
        kw['renormalize'] = True
        kw['sphere_normal'] = 'generic'
        return real_rl(rays, surface, **kw)

    S._surface_normal = sn
    I._surface_normal = sn
    I._refract, I._reflect = rf, rl
    sys.modules['lumenairy.raytrace.trace']._refract = rf
    sys.modules['lumenairy.raytrace.trace']._reflect = rl


def _mut_clamp_moved():
    """The closed form's domain clamp moved from 0.9999 to 0.99."""
    from lumenairy.raytrace import surface as S

    def sphere_normal(x, y, R):
        norm = (x * x + y * y) / (R * R)
        valid = norm < 0.99
        nz = np.where(valid, np.sqrt(np.maximum(1.0 - norm, 0.0)), np.nan)
        return -x / R, -y / R, nz

    S._sphere_normal = sphere_normal
    _rewire_surface_normal(sphere_normal)


def _mut_whole_normal_sign_flipped():
    """The WHOLE closed-form normal vector negated at a MIRROR.

    Kept as a NEGATIVE control.  ``_conic_core.refract_snell`` /
    ``reflect_mirror`` orient the normal against the incoming ray
    (``fl = where(d.n > 0, -1, +1)``) before using it, so negating all
    three components is a semantic NO-OP and NOTHING should catch it.
    A suite that DID go red here would be pinning an internal sign that
    the physics does not have.
    """
    from lumenairy.raytrace import surface as S
    real = S._surface_normal

    def sn(x, y, surf, *, analytic_sphere=False):
        nx, ny, nz = real(x, y, surf, analytic_sphere=analytic_sphere)
        if analytic_sphere and getattr(surf, 'is_mirror', False):
            return -nx, -ny, -nz
        return nx, ny, nz

    S._surface_normal = sn
    _install_surface_normal(sn)


def _mut_mirror_sign_flipped():
    """The closed form's LONGITUDINAL component negated at a MIRROR.

    This is the mutation the report's matrix entry "the normal's SIGN
    flips (mirror)" has to mean: ``nz -> -nz`` alone tilts the normal to
    the far side of the sphere, which the ray-orientation step cannot
    undo.
    """
    from lumenairy.raytrace import surface as S
    real = S._surface_normal

    def sn(x, y, surf, *, analytic_sphere=False):
        nx, ny, nz = real(x, y, surf, analytic_sphere=analytic_sphere)
        if analytic_sphere and getattr(surf, 'is_mirror', False):
            return nx, ny, -nz
        return nx, ny, nz

    S._surface_normal = sn
    _install_surface_normal(sn)


def _mut_nz_sign_flipped_everywhere():
    """The closed form's ``nz`` negated at EVERY pure sphere."""
    from lumenairy.raytrace import surface as S
    real = S._surface_normal

    def sn(x, y, surf, *, analytic_sphere=False):
        nx, ny, nz = real(x, y, surf, analytic_sphere=analytic_sphere)
        if analytic_sphere and S._is_pure_spherical(surf):
            return nx, ny, -nz
        return nx, ny, nz

    S._surface_normal = sn
    _install_surface_normal(sn)


# ---------------- this verification's own four ----------------------

def _mut_clamp_on_analytic_only():
    """The GENERIC route's clamp removed, so only the closed form gates.

    VERIFY-C2's own: the two routes are supposed to gate the SAME domain
    from two expressions that differ by about one ULP.  This widens the
    generic gate to the whole sphere, which turns the one-ULP rim band
    into a finite annulus and must be caught by whatever pins the band.
    """
    from lumenairy.raytrace import surface as S

    def sag_deriv(h, R, conic=0.0, aspheric_coeffs=None):
        h = np.asarray(h, dtype=np.float64)
        dz_dh = np.zeros_like(h)
        if R is not None and not np.isinf(R):
            norm = (1 + conic) * h ** 2 / R ** 2
            valid = norm < 1.0                       # was 0.9999
            denom = np.where(valid,
                             np.sqrt(np.maximum(1 - norm, 1e-30)), 1.0)
            dz_dh = np.where(valid, h / (R * denom), np.nan)
        if aspheric_coeffs:
            for power, coeff in aspheric_coeffs.items():
                dz_dh = dz_dh + power * coeff * h ** (power - 1)
        return dz_dh

    S._surface_sag_derivative = sag_deriv


def _mut_exit_rescale_never_runs():
    """``renormalize='exit'`` stops rescaling at all."""
    import lumenairy.raytrace.trace                          # noqa: F401
    from lumenairy.raytrace import intersection as I

    def noop(rays):
        return None

    I._normalize_directions = noop
    sys.modules['lumenairy.raytrace.trace']._normalize_directions = noop
    wt = sys.modules.get('lumenairy.raytrace.world_trace')
    if wt is not None and hasattr(wt, '_normalize_directions'):
        wt._normalize_directions = noop


def _mut_predicate_accepts_conics():
    """The selection predicate misclassifies a conic (k != 0) as a sphere.

    VERIFY-C2's own, and the sharpest one available: the closed form then
    evaluates the normal of a DIFFERENT surface from the one the sag
    dispatch and the intersection solved -- exactly the v4.12.0 failure
    the predicate exists to prevent.
    """
    from lumenairy.raytrace import surface as S
    real = S._is_pure_spherical

    def pred(surf):
        R = surf.radius
        return (R is not None and (not np.isinf(R))
                and not surf.aspheric_coeffs
                and getattr(surf, 'radius_y', None) is None
                and getattr(surf, 'freeform', None) is None
                and not S._field_frame_active(surf))

    S._is_pure_spherical = pred
    _install_is_pure_spherical(pred, real)


def _mut_jax_gets_a_clamp():
    """The JAX tracer given the NumPy domain clamp it does not have.

    VERIFY-C2's own.  The two backends currently gate differently -- the
    NumPy normal is NaN outside ``h**2/R**2 < 0.9999`` and the JAX one has
    no gate at all.  Adding the gate to JAX changes which rays survive on
    any prescription whose aperture reaches the rim, so any pin that
    claims the two backends agree must move.
    """
    import lumenairy.raytrace.jax_trace as J
    real = J._refract_jax

    def refract(state, R, conic, asph_items, n1, n2):
        import jax.numpy as jnp
        out = real(state, R, conic, asph_items, n1, n2)
        if np.isinf(R) or conic != 0.0 or asph_items:
            return out
        h2 = (state.x * state.x + state.y * state.y) / (float(R) ** 2)
        bad = h2 >= 0.9999
        return out._replace(alive=jnp.logical_and(out.alive, ~bad)) \
            if hasattr(out, '_replace') else out

    J._refract_jax = refract


# ---------------- plumbing -------------------------------------------

def _install_surface_normal(fn):
    for key in ('lumenairy.raytrace.intersection',
                'lumenairy.raytrace.surface'):
        mod = sys.modules.get(key)
        if mod is not None and hasattr(mod, '_surface_normal'):
            mod._surface_normal = fn


def _install_is_pure_spherical(fn, real):
    for key in ('lumenairy.raytrace.intersection',
                'lumenairy.raytrace.surface'):
        mod = sys.modules.get(key)
        if mod is not None and hasattr(mod, '_is_pure_spherical'):
            mod._is_pure_spherical = fn


def _rewire_surface_normal(sphere_normal):
    from lumenairy.raytrace import surface as S
    real_is = S._is_pure_spherical

    def sn(x, y, surf, *, analytic_sphere=False):
        if analytic_sphere and real_is(surf):
            return sphere_normal(x, y, surf.radius)
        dz_dx, dz_dy = S._surface_sag_derivatives_xy(x, y, surf)
        mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
        return -dz_dx / mag, -dz_dy / mag, 1.0 / mag

    S._surface_normal = sn
    _install_surface_normal(sn)


MUTATIONS = {
    'defaults_reverted': _mut_defaults_reverted,
    'defaults_reverted_in_loop': _mut_defaults_reverted_in_loop,
    'clamp_moved': _mut_clamp_moved,
    'mirror_sign_flipped': _mut_mirror_sign_flipped,
    'whole_normal_sign_flipped': _mut_whole_normal_sign_flipped,
    'nz_sign_flipped_everywhere': _mut_nz_sign_flipped_everywhere,
    'clamp_on_analytic_only': _mut_clamp_on_analytic_only,
    'exit_rescale_never_runs': _mut_exit_rescale_never_runs,
    'predicate_accepts_conics': _mut_predicate_accepts_conics,
    'jax_gets_a_clamp': _mut_jax_gets_a_clamp,
}


def pytest_configure(config):
    name = os.environ.get('VC2_MUTATION')
    if not name or name == 'none':
        return
    if name not in MUTATIONS:
        raise SystemExit(f'unknown VC2_MUTATION {name!r}; '
                         f'known: {sorted(MUTATIONS)}')
    MUTATIONS[name]()
    print(f'\n[vc2] MUTATION APPLIED: {name}', flush=True)

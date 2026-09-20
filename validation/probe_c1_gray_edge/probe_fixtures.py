"""WP-C1 probe 2 -- byte identity, archive to archive.

Two claims, both measured against a ``git archive`` of the parent commit
(49ddf4bd) extracted to its own tree, never against a working copy:

  A. With ``edge='hard'`` passed explicitly, every APERTURE-TOUCHING fixture is
     byte-identical to the parent commit's answer with no keyword at all.
     (The way back is exact, not merely close.)
  B. With NO keyword, every NON-APERTURE fixture reachable from here is
     byte-identical to the parent commit's.  (The flip's blast radius stops at
     the aperture.)

Each fixture returns an ndarray; the probe records the SHA-256 of
``arr.tobytes()`` together with dtype and shape, so "identical" means the same
bits, not the same float to some tolerance.

Usage::

    python validation/probe_c1_gray_edge/probe_fixtures.py <tag> <arm>

``arm`` is ``'default'`` (aperture fixtures called with NO ``edge=`` keyword)
or ``'hard'`` (aperture fixtures called with ``edge='hard'``).  Non-aperture
fixtures never take the keyword and are emitted on both arms.

Writes ``validation/probe_c1_gray_edge/fixtures_<tag>_<arm>.json``.
"""
import hashlib
import json
import os
import platform
import sys
import warnings

import numpy as np

import lumenairy

APERTURE_FIXTURES = {}
PLAIN_FIXTURES = {}


def aperture_fixture(fn):
    APERTURE_FIXTURES[fn.__name__] = fn
    return fn


def plain_fixture(fn):
    PLAIN_FIXTURES[fn.__name__] = fn
    return fn


def _field(N=128, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))


# ---------------------------------------------------------------------------
# A.  Aperture-touching fixtures -- every in-library path that reaches
#     elements.apply_aperture, one per public entry point.
# ---------------------------------------------------------------------------

@aperture_fixture
def ap_circular(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_field(), 2e-6, 'circular', {'diameter': 1.7e-4},
                          **kw)


@aperture_fixture
def ap_annular(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_field(), 2e-6, 'annular',
                          {'inner_diameter': 4e-5, 'outer_diameter': 1.7e-4},
                          **kw)


@aperture_fixture
def ap_rectangular(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_field(), 2e-6, 'rectangular',
                          {'width_x': 1.33e-4, 'width_y': 0.91e-4}, **kw)


@aperture_fixture
def ap_decentred_anamorphic(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_field(), 2e-6, 'circular', {'diameter': 1.1e-4},
                          xc=3.7e-6, yc=-1.3e-6, dy=5e-6, **kw)


@aperture_fixture
def ap_complex64(kw):
    from lumenairy.elements.elements import apply_aperture
    E = _field(N=64).astype(np.complex64)
    return apply_aperture(E, 2e-6, 'circular', {'diameter': 6.3e-5}, **kw)


@aperture_fixture
def lyot_stop(kw):
    """``apply_lyot_stop`` exposes no ``edge``; on the 'hard' arm the probe
    calls its DOCUMENTED way back (the equivalent annular
    ``apply_aperture(..., edge='hard')``), which is the claim that matters."""
    from lumenairy.elements.elements import apply_aperture, apply_lyot_stop
    if kw:
        return apply_aperture(_field(), 2e-6, shape='annular',
                              params={'inner_diameter': 3e-5,
                                      'outer_diameter': 1.7e-4}, **kw)
    return apply_lyot_stop(_field(), 2e-6, outer_diameter=1.7e-4,
                           inner_diameter=3e-5)


@aperture_fixture
def algebra_aperture_operator(kw):
    """The algebraic surface has no ``edge`` vocabulary; its documented way
    back is a direct ``apply_aperture`` call, which the 'hard' arm runs."""
    from lumenairy.algebra.apertures import Aperture
    if kw:
        from lumenairy.elements.elements import apply_aperture
        return apply_aperture(_field(), 2e-6, shape='circular',
                              params={'diameter': 1.7e-4}, dy=2e-6, **kw)
    op = Aperture(diameter=1.7e-4, shape='circular')
    E, _dx, _dy = op._apply(_field(), dx=2e-6, dy=2e-6, wavelength=633e-9)
    return E


@aperture_fixture
def jones_field_apply_aperture(kw):
    """``JonesField.apply_aperture`` has no ``edge``; the way back is the
    free function on each component, which the 'hard' arm runs."""
    from lumenairy.elements.polarization import JonesField
    if kw:
        from lumenairy.elements.elements import apply_aperture
        parts = [apply_aperture(_field(seed=s), 2e-6, shape='circular',
                                params={'diameter': 1.7e-4}, dy=2e-6, **kw)
                 for s in (11, 12)]
        return np.concatenate([np.asarray(p).ravel() for p in parts])
    jf = JonesField(_field(seed=11), _field(seed=12), dx=2e-6)
    jf.apply_aperture(shape='circular', params={'diameter': 1.7e-4})
    return np.concatenate([np.asarray(jf.Ex).ravel(),
                           np.asarray(jf.Ey).ravel()])


def _aperture_chain(kw):
    elem = {'type': 'aperture', 'shape': 'circular',
            'params': {'diameter': 1.7e-4}}
    elem.update(kw)          # 'edge' is an element key from v5.49.0
    return [elem]


@aperture_fixture
def system_aperture_element(kw):
    from lumenairy.propagators.system import propagate_through_system
    out = propagate_through_system(_field(N=128), _aperture_chain(kw),
                                   633e-9, dx=2e-6)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


@aperture_fixture
def system_aperture_element_jax_jit(kw):
    """The jit'd JAX kernel's aperture branch (fast path)."""
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        return None
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.propagators.system import propagate_through_system_jax
    out = propagate_through_system_jax(
        jnp.asarray(_field(N=128)), _aperture_chain(kw), 633e-9, 2e-6)
    return np.asarray(out)


@aperture_fixture
def system_aperture_element_jax_eager(kw):
    """The JAX slow path (``verbose=True`` bypasses the jit cache)."""
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        return None
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.propagators.system import propagate_through_system_jax
    out = propagate_through_system_jax(
        jnp.asarray(_field(N=128)), _aperture_chain(kw), 633e-9, 2e-6,
        verbose=True)
    return np.asarray(out)


@aperture_fixture
def rs_spatial_of_apertured_field(kw):
    from lumenairy.elements.elements import apply_aperture
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    N, dx = 256, 2e-6
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 2e-4}, **kw)
    return rayleigh_sommerfeld_propagate(E, z=16e-3, wavelength=633e-9,
                                         dx=dx, kernel='spatial')


@aperture_fixture
def hf_quadrature_of_apertured_field(kw):
    from lumenairy.elements.elements import apply_aperture
    from lumenairy.propagators.hf import propagate_huygens_fresnel_with_opl_callable
    N, dx, lam, z = 128, 4e-6, 633e-9, 5e-3
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 2e-4}, **kw)

    def opl_fn(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / lam

    return propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl_fn, output_grid_x=np.array([0.0, 5e-6]),
        output_grid_y=np.array([0.0]), input_grid_dx=dx)


@aperture_fixture
def jax_apply_aperture(kw):
    """The JAX arm of the SAME function (one xp-parametrised body)."""
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        return None
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.elements.elements import apply_aperture
    E = jnp.asarray(_field(N=64))
    return np.asarray(apply_aperture(E, 2e-6, 'circular',
                                     {'diameter': 6.3e-5}, **kw))


@aperture_fixture
def cupy_apply_aperture(kw):
    import importlib.util
    if importlib.util.find_spec('cupy') is None:
        return None
    import cupy as cp

    from lumenairy.elements.elements import apply_aperture
    E = cp.asarray(_field(N=64))
    return cp.asnumpy(apply_aperture(E, 2e-6, 'circular',
                                     {'diameter': 6.3e-5}, **kw))


# ---------------------------------------------------------------------------
# B.  Non-aperture fixtures -- reachable public entry points that do NOT go
#     through elements.apply_aperture.  Called with no keyword on both arms.
# ---------------------------------------------------------------------------

@plain_fixture
def gaussian_aperture():
    from lumenairy.elements.elements import apply_gaussian_aperture
    return apply_gaussian_aperture(_field(), 2e-6, sigma=5e-5)


@plain_fixture
def apodized_pupil():
    from lumenairy.elements.elements import apply_apodized_pupil
    return apply_apodized_pupil(_field(), 2e-6, 1.7e-4, apodization='cos2')


@plain_fixture
def zernike_aberration():
    from lumenairy.elements.elements import apply_zernike_aberration
    return apply_zernike_aberration(_field(), 2e-6,
                                    {(2, 0): 0.3, (4, 0): -0.2}, 1.0e-4)


@plain_fixture
def thin_lens():
    """``_lens_thin`` builds its OWN mask, so it must not move."""
    from lumenairy.elements import apply_thin_lens
    return apply_thin_lens(_field(), f=0.05, wavelength=633e-9, dx=2e-6)


@plain_fixture
def mirror_with_aperture_diameter():
    from lumenairy.elements import apply_mirror
    return apply_mirror(_field(), 633e-9, 2e-6, radius=0.1,
                        aperture_diameter=1.7e-4)


@plain_fixture
def axicon():
    from lumenairy.elements import apply_axicon
    return apply_axicon(_field(), 0.01, 1.5, 633e-9, 2e-6, 2e-6)


@plain_fixture
def asm_propagate():
    from lumenairy.propagators.asm import angular_spectrum_propagate
    return angular_spectrum_propagate(_field(), z=2e-3, wavelength=633e-9,
                                      dx=2e-6)


@plain_fixture
def rs_transfer_propagate():
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    return rayleigh_sommerfeld_propagate(_field(), z=2e-3, wavelength=633e-9,
                                         dx=2e-6, kernel='transfer')


@plain_fixture
def rs_spatial_gaussian():
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (2e-5) ** 2)).astype(complex)
    return rayleigh_sommerfeld_propagate(E, z=30e-3, wavelength=633e-9,
                                         dx=dx, kernel='spatial')


@plain_fixture
def hf_freespace():
    from lumenairy.propagators.hf import propagate_huygens_fresnel_freespace
    N, dx = 64, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (4e-5) ** 2)).astype(complex)
    return propagate_huygens_fresnel_freespace(
        E, z=5e-3, wavelength=633e-9, dx=dx)


@plain_fixture
def thin_grating_efficiency():
    from lumenairy.elements.thin_grating import thin_grating_efficiency_1d
    _orders, R, T = thin_grating_efficiency_1d(
        period=1e-5, n_ridge=1.5, n_groove=1.0, n_substrate=1.0,
        n_superstrate=1.0, depth=6e-7, duty_cycle=0.5, wavelength=633e-9,
        n_orders=5)
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain_fixture
def rcwa_efficiency_1d():
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    orders, R, T = rcwa_efficiency_1d(
        period=8e-7, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=4e-7, duty_cycle=0.5, wavelength=633e-9,
        n_orders=7, polarization='te')
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain_fixture
def pmm_efficiency_1d():
    from lumenairy.elements.pmm import pmm_efficiency_1d
    orders, R, T = pmm_efficiency_1d(
        period=8e-7, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
        n_superstrate=1.0, depth=4e-7, duty_cycle=0.5, wavelength=633e-9,
        polarization='te')
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain_fixture
def zernike_basis():
    from lumenairy.analysis.zernike import zernike_basis_matrix
    N, dx = 64, 1e-4
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    B, _mask = zernike_basis_matrix(10, X, Y, 3e-3)
    return np.asarray(B, dtype=float)


@plain_fixture
def turbulence_screen():
    import lumenairy as la
    return np.asarray(la.generate_turbulence_screen(64, 1e-3, r0=0.05,
                                                    seed=3), dtype=float)


@plain_fixture
def real_lens_thin():
    import lumenairy as la
    pres = la.make_singlet(R1=0.05, R2=-0.05, d=3e-3, glass='N-BK7',
                           aperture=1e-2)
    N, dx = 128, 1e-4
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (3e-3) ** 2)).astype(complex)
    return np.asarray(la.apply_real_lens(E, prescription=pres,
                                         wavelength=633e-9, dx=dx,
                                         surface_model='thin'))


def _digest(arr):
    a = np.ascontiguousarray(arr)
    return {'sha256': hashlib.sha256(a.tobytes()).hexdigest(),
            'dtype': str(a.dtype), 'shape': list(a.shape),
            'nbytes': int(a.nbytes)}


def main():
    tag = sys.argv[1]
    arm = sys.argv[2]
    assert arm in ('default', 'hard'), arm
    kw = {} if arm == 'default' else {'edge': 'hard'}

    out = {'tag': tag, 'arm': arm,
           'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0],
           'platform': platform.platform(),
           'numpy': np.__version__,
           'aperture': {}, 'plain': {}, 'unavailable': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, fn in APERTURE_FIXTURES.items():
            try:
                v = fn(kw)
            except Exception as e:                       # noqa: BLE001
                out['unavailable'].append(
                    f'aperture:{name}:{type(e).__name__}:{e}')
                continue
            if v is None:
                out['unavailable'].append(f'aperture:{name}:not-applicable')
                continue
            out['aperture'][name] = _digest(v)
        for name, fn in PLAIN_FIXTURES.items():
            try:
                v = fn()
            except Exception as e:                       # noqa: BLE001
                out['unavailable'].append(
                    f'plain:{name}:{type(e).__name__}:{e}')
                continue
            if v is None:
                out['unavailable'].append(f'plain:{name}:not-applicable')
                continue
            out['plain'][name] = _digest(v)

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f'fixtures_{tag}_{arm}.json')
    with open(path, 'w', encoding='cp1252') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"arm={arm}  aperture fixtures={len(out['aperture'])}  "
          f"plain fixtures={len(out['plain'])}  "
          f"unavailable={len(out['unavailable'])}")
    for u in out['unavailable']:
        print(f"  unavailable: {u}")
    print(f"wrote {path}")


if __name__ == '__main__':
    main()

"""VERIFY-C1 probe 2 -- byte identity, archive to archive, on my OWN fixtures.

An independent fixture set from `validation/probe_c1_gray_edge/probe_fixtures.py`
(different grids, different shapes, different seeds, and entry points that probe
does not reach -- notably ``lumenairy.evaluate``, which builds an ``'aperture'``
element from a prescription's STOP surface).

  Claim A (the way back is exact): base tree with NO keyword vs new tree with
    ``edge='hard'`` -- every APERTURE fixture must be byte-identical.
  Claim B (the blast radius stops at the aperture): base tree vs new tree, both
    with NO keyword -- every APERTURE fixture must MOVE and every NON-APERTURE
    fixture must be byte-identical.

Counts: >= 20 aperture fixtures, >= 15 non-aperture.  SHA-256 over
``np.ascontiguousarray(arr).tobytes()`` with dtype and shape, so "identical"
means the same bits.  The probe also records, per fixture, the count of
NEGATIVE-ZERO real and imaginary parts, which is what a byte digest sees and a
tolerance comparison never does.

Usage::

    python -P validation/probe_verify_c1/probe_fixtures_v.py <tag> <arm>

``arm`` is ``default`` (aperture fixtures called with no ``edge=``) or ``hard``
(aperture fixtures called with ``edge='hard'``, or, for an entry point that
exposes no ``edge``, that entry point's DOCUMENTED way back).

Writes ``validation/probe_verify_c1/fx_<tag>_<arm>.json``.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np

import lumenairy

AP = {}
PLAIN = {}


def ap(fn):
    AP[fn.__name__] = fn
    return fn


def plain(fn):
    PLAIN[fn.__name__] = fn
    return fn


def _f(N=96, seed=2027):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))


DX = 1.25e-6
LAM = 1064e-9


def _jnp():
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        return None
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    return jnp


# ------------------------------------------------------------- aperture arms
@ap
def v_circ(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(), DX, 'circular', {'diameter': 9.1e-5}, **kw)


@ap
def v_circ_odd_grid(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(N=97, seed=31), DX, 'circular',
                          {'diameter': 8.3e-5}, **kw)


@ap
def v_annulus(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(seed=5), DX, 'annular',
                          {'inner_diameter': 2.7e-5, 'outer_diameter': 9.1e-5},
                          **kw)


@ap
def v_rect(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(seed=9), DX, 'rectangular',
                          {'width_x': 7.3e-5, 'width_y': 4.1e-5}, **kw)


@ap
def v_rect_anamorphic_grid(kw):
    """dy != dx AND width_x != width_y (an anamorphic stop on an
    anamorphic grid) -- the shape the WP-C1 probe's single decentred
    fixture does not separate."""
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(seed=13), DX, 'rectangular',
                          {'width_x': 7.3e-5, 'width_y': 4.1e-5},
                          dy=1.6 * DX, **kw)


@ap
def v_circ_decentred(kw):
    """Centre at a NON-half-integer pixel offset in both axes."""
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(seed=17), DX, 'circular', {'diameter': 6.7e-5},
                          xc=3.37 * DX, yc=-1.83 * DX, **kw)


@ap
def v_annulus_decentred_anamorphic(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(seed=19), DX, 'annular',
                          {'inner_diameter': 2.1e-5, 'outer_diameter': 7.9e-5},
                          xc=1.7 * DX, yc=0.9 * DX, dy=1.35 * DX, **kw)


@ap
def v_circ_complex64(kw):
    from lumenairy.elements.elements import apply_aperture
    return apply_aperture(_f(N=64, seed=23).astype(np.complex64), DX,
                          'circular', {'diameter': 5.3e-5}, **kw)


@ap
def v_circ_nan_outside(kw):
    """A field carrying NaN and +-inf OUTSIDE the stop -- the A8 contract."""
    from lumenairy.elements.elements import apply_aperture
    E = _f(N=64, seed=29)
    E[0, 0] = np.nan
    E[0, 1] = np.inf
    E[1, 0] = -np.inf
    E[1, 1] = complex(np.nan, np.inf)
    return apply_aperture(E, DX, 'circular', {'diameter': 4.1e-5}, **kw)


@ap
def v_circ_edge_samples_8(kw):
    """``edge_samples`` is not the default; the way back on this arm is the
    hard edge, which ignores it."""
    from lumenairy.elements.elements import apply_aperture
    k = dict(kw)
    if not k:
        k = {'edge_samples': 8}
    return apply_aperture(_f(seed=37), DX, 'circular', {'diameter': 9.1e-5},
                          **k)


@ap
def v_lyot_annular(kw):
    from lumenairy.elements.elements import apply_aperture, apply_lyot_stop
    if kw:
        return apply_aperture(_f(seed=41), DX, shape='annular',
                              params={'inner_diameter': 2.3e-5,
                                      'outer_diameter': 8.7e-5}, **kw)
    return apply_lyot_stop(_f(seed=41), DX, outer_diameter=8.7e-5,
                           inner_diameter=2.3e-5)


@ap
def v_lyot_outer_only(kw):
    from lumenairy.elements.elements import apply_aperture, apply_lyot_stop
    if kw:
        return apply_aperture(_f(seed=43), DX, shape='annular',
                              params={'inner_diameter': 0.0,
                                      'outer_diameter': 8.7e-5}, **kw)
    return apply_lyot_stop(_f(seed=43), DX, outer_diameter=8.7e-5)


@ap
def v_algebra_operator_circ(kw):
    from lumenairy.algebra.apertures import Aperture
    if kw:
        from lumenairy.elements.elements import apply_aperture
        return apply_aperture(_f(seed=47), DX, shape='circular',
                              params={'diameter': 9.1e-5}, dy=DX, **kw)
    E, _dx, _dy = Aperture(diameter=9.1e-5, shape='circular')._apply(
        _f(seed=47), dx=DX, dy=DX, wavelength=LAM)
    return E


@ap
def v_algebra_operator_annular(kw):
    from lumenairy.algebra.apertures import Aperture
    if kw:
        from lumenairy.elements.elements import apply_aperture
        return apply_aperture(_f(seed=53), DX, shape='annular',
                              params={'inner_diameter': 4.55e-5,
                                      'outer_diameter': 9.1e-5},
                              dy=DX, **kw)
    E, _dx, _dy = Aperture(diameter=9.1e-5, shape='annular')._apply(
        _f(seed=53), dx=DX, dy=DX, wavelength=LAM)
    return E


@ap
def v_algebra_operator_square(kw):
    from lumenairy.algebra.apertures import Aperture
    if kw:
        from lumenairy.elements.elements import apply_aperture
        return apply_aperture(_f(seed=59), DX, shape='rectangular',
                              params={'width_x': 7.3e-5, 'width_y': 7.3e-5},
                              dy=DX, **kw)
    E, _dx, _dy = Aperture(diameter=7.3e-5, shape='rectangular')._apply(
        _f(seed=59), dx=DX, dy=DX, wavelength=LAM)
    return E


@ap
def v_jones_field(kw):
    from lumenairy.elements.polarization import JonesField
    if kw:
        from lumenairy.elements.elements import apply_aperture
        parts = [apply_aperture(_f(seed=s), DX, shape='circular',
                                params={'diameter': 9.1e-5}, dy=DX, **kw)
                 for s in (61, 67)]
        return np.concatenate([np.asarray(p).ravel() for p in parts])
    jf = JonesField(_f(seed=61), _f(seed=67), dx=DX)
    jf.apply_aperture(shape='circular', params={'diameter': 9.1e-5})
    return np.concatenate([np.asarray(jf.Ex).ravel(),
                           np.asarray(jf.Ey).ravel()])


def _chain(kw, shape='circular', params=None):
    elem = {'type': 'aperture', 'shape': shape,
            'params': params or {'diameter': 9.1e-5}}
    elem.update(kw)
    return [elem]


@ap
def v_chain_numpy(kw):
    from lumenairy.propagators.system import propagate_through_system
    out = propagate_through_system(_f(N=128, seed=71), _chain(kw), LAM, dx=DX)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


@ap
def v_chain_numpy_rect(kw):
    from lumenairy.propagators.system import propagate_through_system
    out = propagate_through_system(
        _f(N=128, seed=73),
        _chain(kw, 'rectangular',
               {'width_x': 7.3e-5, 'width_y': 4.1e-5}), LAM, dx=DX)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


@ap
def v_chain_jax_jit(kw):
    jnp = _jnp()
    if jnp is None:
        return None
    from lumenairy.propagators.system import propagate_through_system_jax
    return np.asarray(propagate_through_system_jax(
        jnp.asarray(_f(N=128, seed=71)), _chain(kw), LAM, DX))


@ap
def v_chain_jax_eager(kw):
    jnp = _jnp()
    if jnp is None:
        return None
    from lumenairy.propagators.system import propagate_through_system_jax
    return np.asarray(propagate_through_system_jax(
        jnp.asarray(_f(N=128, seed=71)), _chain(kw), LAM, DX, verbose=True))


@ap
def v_chain_jax_eager_annular(kw):
    jnp = _jnp()
    if jnp is None:
        return None
    from lumenairy.propagators.system import propagate_through_system_jax
    return np.asarray(propagate_through_system_jax(
        jnp.asarray(_f(N=128, seed=79)),
        _chain(kw, 'annular',
               {'inner_diameter': 2.7e-5, 'outer_diameter': 9.1e-5}),
        LAM, DX, verbose=True))


@ap
def v_evaluate_prescription_stop(kw):
    """``lumenairy.evaluate`` on a Zemax-shape prescription with a STOP
    surface.  ``_prescription_to_elements`` emits an ``'aperture'`` element
    with NO ``edge`` key, so this entry point takes the new default -- and
    ``evaluate`` exposes no way to name the old one.  On the 'hard' arm the
    probe runs the only way back that exists: rebuild the element list by
    hand and add ``'edge': 'hard'``."""
    import lumenairy as la
    rx = {
        'elements': [
            {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'is_stop': True, 'semi_diameter': 1.2e-3,
             'comment': 'STOP'},
            {'surf_num': 2, 'element_type': 'surface', 'radius': 0.05,
             'glass_after': 'N-BK7', 'semi_diameter': 2e-3},
            {'surf_num': 3, 'element_type': 'surface', 'radius': -0.05,
             'glass_after': 'air', 'semi_diameter': 2e-3},
        ],
        'all_thicknesses': [2e-3, 3e-3, 20e-3],
        'aperture_diameter': 2.4e-3,
    }
    src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=1.0e-3)
    if kw:
        # The ONLY way back: reach into the private builder, inject the
        # key `evaluate` has no argument for, and drive the chain by hand.
        from lumenairy.propagators.system import (
            _prescription_to_elements, propagate_through_system)
        elements = [dict(e, **kw) if e.get('type') == 'aperture' else e
                    for e in _prescription_to_elements(rx)]
        out = propagate_through_system(np.asarray(src.E), elements,
                                       633e-9, dx=40e-6)
        return np.asarray(out[0] if isinstance(out, tuple) else out)
    return np.asarray(la.evaluate(rx, src).field)


@ap
def v_jax_direct(kw):
    jnp = _jnp()
    if jnp is None:
        return None
    from lumenairy.elements.elements import apply_aperture
    return np.asarray(apply_aperture(jnp.asarray(_f(N=64, seed=83)), DX,
                                     'circular', {'diameter': 5.3e-5}, **kw))


@ap
def v_cupy_direct(kw):
    import importlib.util
    if importlib.util.find_spec('cupy') is None:
        return None
    import cupy as cp
    from lumenairy.elements.elements import apply_aperture
    return cp.asnumpy(apply_aperture(cp.asarray(_f(N=64, seed=83)), DX,
                                     'circular', {'diameter': 5.3e-5}, **kw))


@ap
def v_rs_spatial_of_apertured(kw):
    from lumenairy.elements.elements import apply_aperture
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    N, dx = 192, 2.08333e-6
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 1.25e-4}, **kw)
    return rayleigh_sommerfeld_propagate(E, z=9e-3, wavelength=LAM, dx=dx,
                                         kernel='spatial')


@ap
def v_asm_of_apertured(kw):
    from lumenairy.elements.elements import apply_aperture
    from lumenairy.propagators.asm import angular_spectrum_propagate
    N, dx = 128, 2e-6
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 1.25e-4}, **kw)
    return angular_spectrum_propagate(E, z=1.5e-3, wavelength=LAM, dx=dx)


@ap
def v_psf_of_apertured(kw):
    import lumenairy as la
    from lumenairy.elements.elements import apply_aperture
    N, dx = 128, 4e-6
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 2.5e-4}, **kw)
    psf, _dxf = la.compute_psf(E, LAM, 0.05, dx, normalize='power')
    return np.asarray(psf, dtype=float)


# ---------------------------------------------------------- non-aperture arms
@plain
def p_gaussian_aperture():
    from lumenairy.elements.elements import apply_gaussian_aperture
    return apply_gaussian_aperture(_f(seed=101), DX, sigma=3.1e-5)


@plain
def p_apodized_pupil():
    from lumenairy.elements.elements import apply_apodized_pupil
    return apply_apodized_pupil(_f(seed=103), DX, 9.1e-5, apodization='cos2')


@plain
def p_zernike_aberration():
    from lumenairy.elements.elements import apply_zernike_aberration
    return apply_zernike_aberration(_f(seed=107), DX,
                                    {(2, 0): 0.21, (3, 1): -0.13}, 5.5e-5)


@plain
def p_thin_lens():
    from lumenairy.elements import apply_thin_lens
    return apply_thin_lens(_f(seed=109), f=0.037, wavelength=LAM, dx=DX)


@plain
def p_thin_lens_aplanatic():
    """The non-paraxial thin lens has its OWN out-of-domain sentinel and its
    own mask -- it must not move."""
    from lumenairy.elements import apply_thin_lens
    return apply_thin_lens(_f(seed=113), f=0.037, wavelength=LAM, dx=DX,
                           lens_model='aplanatic')


@plain
def p_mirror_with_aperture_diameter():
    from lumenairy.elements import apply_mirror
    return apply_mirror(_f(seed=127), LAM, DX, radius=0.08,
                        aperture_diameter=9.1e-5)


@plain
def p_axicon():
    from lumenairy.elements import apply_axicon
    return apply_axicon(_f(seed=131), 0.008, 1.46, LAM, DX, DX)


@plain
def p_vortex_phase_mask():
    import lumenairy as la
    return np.asarray(la.apply_vortex_phase_mask(_f(seed=137), DX, charge=3))


@plain
def p_asm():
    from lumenairy.propagators.asm import angular_spectrum_propagate
    return angular_spectrum_propagate(_f(seed=139), z=1.5e-3, wavelength=LAM,
                                      dx=DX)


@plain
def p_rs_transfer():
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    return rayleigh_sommerfeld_propagate(_f(seed=149), z=1.5e-3,
                                         wavelength=LAM, dx=DX,
                                         kernel='transfer')


@plain
def p_rs_spatial_gaussian():
    from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (1.7e-5) ** 2)).astype(complex)
    return rayleigh_sommerfeld_propagate(E, z=25e-3, wavelength=LAM, dx=dx,
                                         kernel='spatial')


@plain
def p_hf_freespace():
    from lumenairy.propagators.hf import propagate_huygens_fresnel_freespace
    N, dx = 64, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (3.3e-5) ** 2)).astype(complex)
    return propagate_huygens_fresnel_freespace(E, z=4e-3, wavelength=LAM,
                                               dx=dx)


@plain
def p_fraunhofer_mft():
    import lumenairy as la
    N, dx = 96, 4e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (5e-5) ** 2)).astype(complex)
    return np.asarray(la.fraunhofer_propagate_mft(E, 0.05, LAM, dx, 2e-6, 64))


@plain
def p_thin_grating():
    from lumenairy.elements.thin_grating import thin_grating_efficiency_1d
    _o, R, T = thin_grating_efficiency_1d(
        period=9e-6, n_ridge=1.46, n_groove=1.0, n_substrate=1.0,
        n_superstrate=1.0, depth=7.1e-7, duty_cycle=0.45, wavelength=LAM,
        n_orders=5)
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain
def p_rcwa():
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    _o, R, T = rcwa_efficiency_1d(
        period=7e-7, n_ridge=2.1, n_groove=1.0, n_substrate=1.45,
        n_superstrate=1.0, depth=3.7e-7, duty_cycle=0.42, wavelength=LAM,
        n_orders=9, polarization='tm')
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain
def p_pmm():
    from lumenairy.elements.pmm import pmm_efficiency_1d
    _o, R, T = pmm_efficiency_1d(
        period=7e-7, n_ridge=2.1, n_groove=1.0, n_substrate=1.45,
        n_superstrate=1.0, depth=3.7e-7, duty_cycle=0.42, wavelength=LAM,
        polarization='tm')
    return np.concatenate([np.asarray(R, dtype=float),
                           np.asarray(T, dtype=float)])


@plain
def p_zernike_basis():
    from lumenairy.analysis.zernike import zernike_basis_matrix
    N, dx = 64, 1e-4
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    B, _m = zernike_basis_matrix(12, X, Y, 2.7e-3)
    return np.asarray(B, dtype=float)


@plain
def p_turbulence():
    import lumenairy as la
    return np.asarray(la.generate_turbulence_screen(64, 1e-3, r0=0.07,
                                                    seed=17), dtype=float)


@plain
def p_real_lens_thin():
    import lumenairy as la
    pres = la.make_singlet(R1=0.04, R2=-0.06, d=2.5e-3, glass='N-BK7',
                           aperture=8e-3)
    N, dx = 128, 1e-4
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * (2.3e-3) ** 2)).astype(complex)
    return np.asarray(la.apply_real_lens(E, prescription=pres,
                                         wavelength=LAM, dx=dx,
                                         surface_model='thin'))


@plain
def p_gaussian_beam():
    import lumenairy as la
    E, _dx, _w = la.create_gaussian_beam(64, DX, LAM, w0=2.1e-5)
    return np.asarray(E)


@plain
def p_chain_lens_only():
    """A chain with NO aperture element -- the chain machinery itself must
    not move."""
    from lumenairy.propagators.system import propagate_through_system
    out = propagate_through_system(
        _f(N=128, seed=151),
        [{'type': 'lens', 'f': 0.037}, {'type': 'propagate', 'z': 1e-3}],
        LAM, dx=DX)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


@plain
def p_chain_jax_lens_only():
    jnp = _jnp()
    if jnp is None:
        return None
    from lumenairy.propagators.system import propagate_through_system_jax
    return np.asarray(propagate_through_system_jax(
        jnp.asarray(_f(N=128, seed=151)),
        [{'type': 'lens', 'f': 0.037}, {'type': 'propagate', 'z': 1e-3}],
        LAM, DX))


def _digest(arr):
    a = np.ascontiguousarray(arr)
    d = {'sha256': hashlib.sha256(a.tobytes()).hexdigest(),
         'dtype': str(a.dtype), 'shape': list(a.shape),
         'nbytes': int(a.nbytes)}
    if np.iscomplexobj(a):
        r, i = np.real(a), np.imag(a)
        d['neg_zero_real'] = int(np.count_nonzero(
            (r == 0.0) & (np.signbit(r))))
        d['neg_zero_imag'] = int(np.count_nonzero(
            (i == 0.0) & (np.signbit(i))))
        d['all_finite'] = bool(np.all(np.isfinite(a)))
        d['n_nonfinite'] = int(np.count_nonzero(~np.isfinite(a)))
    return d


def main():
    tag = sys.argv[1]
    arm = sys.argv[2]
    assert arm in ('default', 'hard'), arm
    kw = {} if arm == 'default' else {'edge': 'hard'}
    aperture, plainres, unavailable, errors = {}, {}, [], {}
    for name, fn in sorted(AP.items()):
        try:
            arr = fn(dict(kw))
        except Exception as e:                       # noqa: BLE001
            errors[name] = '%s: %s' % (type(e).__name__, e)
            continue
        if arr is None:
            unavailable.append(name)
            continue
        aperture[name] = _digest(arr)
    for name, fn in sorted(PLAIN.items()):
        try:
            arr = fn()
        except Exception as e:                       # noqa: BLE001
            errors[name] = '%s: %s' % (type(e).__name__, e)
            continue
        if arr is None:
            unavailable.append(name)
            continue
        plainres[name] = _digest(arr)
    out = {
        'tag': tag, 'arm': arm,
        'lumenairy_file': lumenairy.__file__,
        'lumenairy_version': getattr(lumenairy, '__version__', '?'),
        'python': sys.version, 'numpy': np.__version__,
        'platform': platform.platform(),
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS')},
        'aperture': aperture, 'plain': plainres,
        'unavailable': sorted(unavailable), 'errors': errors,
        'n_aperture': len(aperture), 'n_plain': len(plainres),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'fx_%s_%s.json' % (tag, arm))
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('aperture fixtures =', len(aperture), 'plain =', len(plainres))
    print('unavailable =', out['unavailable'])
    print('errors =', out['errors'])
    print('wrote', path)


if __name__ == '__main__':
    main()

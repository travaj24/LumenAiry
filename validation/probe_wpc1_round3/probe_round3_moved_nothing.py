"""WP-C1 ROUND 3 -- did round 3 move any LEGAL answer?

The neutrality half of the round-3 closures (R1-R3 are guard changes;
R4-R8 are documentation).  Same fixture family as
``probe_round2_moved_nothing.py``, from which this is derived, with three
additions that exercise the round-3 edits specifically: an element whose
params do NOT resolve carrying a LEGAL rim on all three chain routes (R1
hoists the reader above that gate, so those three must not move), and a
``numpy.str_`` edge, which the R2 ``isinstance`` test must not swallow.
The exception handler also RECORDS what it caught instead of dropping the
eager key, which cost the round-2 copy one fixture on this box.

Original round-2 preamble:

Round 2's library changes are a refusal (``_validate_edge_kwargs`` hoisted out
of ``apply_aperture``'s body and called from ``_aperture_edge_kwargs`` and
``_prescription_to_elements``) and two ``evaluate`` keywords that default to
``None`` and stamp nothing.  Both are supposed to be answer-neutral for every
input that was already legal.  "Supposed to be" is not a measurement, and the
round-2 addendum does not make this comparison.

This probe digests a fixture family of LEGAL calls -- direct ``apply_aperture``
on three shapes, an anamorphic grid, an odd grid, complex64, a non-finite
field, ``apply_lyot_stop``, the ``algebra.Aperture`` operator,
``JonesField.apply_aperture``, all three chain routes on two element shapes
(unkeyworded, ``edge='hard'``, ``edge_samples=8``) and ``lumenairy.evaluate``
on a STOP prescription -- and is run from inside the ``git archive 7ea01ede``
extraction and from inside the round-2 tree.  Any fixture whose digest differs
is something round 2 moved.

Run from inside the tree under test:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=<tree> python probe_round2_moved_nothing.py <out.json>
"""
import hashlib
import json
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements.elements import apply_aperture, apply_lyot_stop

LAM = 1064e-9


def _d(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return "{0}|{1}|{2}".format(
        hashlib.sha256(a.tobytes()).hexdigest()[:16], a.dtype, a.shape)


def _field(N, dtype=complex, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(dtype)


def fixtures():
    out = {}
    dx = 1.25e-6
    E = _field(64)
    out['circ_default'] = _d(apply_aperture(E, dx, 'circular',
                                            {'diameter': 5.3e-5}))
    out['circ_hard'] = _d(apply_aperture(E, dx, 'circular',
                                         {'diameter': 5.3e-5}, edge='hard'))
    out['circ_gray8'] = _d(apply_aperture(E, dx, 'circular',
                                          {'diameter': 5.3e-5},
                                          edge='gray', edge_samples=8))
    out['rect_anamorphic'] = _d(apply_aperture(
        E, dx, 'rectangular', {'width_x': 4.1e-5, 'width_y': 2.7e-5},
        dy=1.7 * dx, xc=0.3 * dx, yc=-0.7 * dx))
    out['annulus_decentred'] = _d(apply_aperture(
        _field(63), dx, 'annular',
        {'inner_diameter': 1.9e-5, 'outer_diameter': 5.1e-5},
        xc=2.3 * dx, yc=-1.1 * dx))
    out['odd_grid'] = _d(apply_aperture(_field(63), dx, 'circular',
                                        {'diameter': 4.7e-5}))
    out['complex64'] = _d(apply_aperture(_field(64, np.complex64), dx,
                                         'circular', {'diameter': 5.3e-5}))
    Enan = _field(64).copy()
    Enan[3, 5] = np.nan
    Enan[7, 9] = np.inf
    Enan[11, 13] = -np.inf
    out['nonfinite'] = _d(apply_aperture(Enan, dx, 'circular',
                                         {'diameter': 5.3e-5}))
    out['lyot'] = _d(apply_lyot_stop(E, dx, outer_diameter=5.1e-5,
                                     inner_diameter=1.9e-5))
    out['numpy_int_samples'] = _d(apply_aperture(
        E, dx, 'circular', {'diameter': 5.3e-5},
        edge_samples=np.int64(8)))

    from lumenairy.algebra.apertures import Aperture
    op = Aperture(diameter=5.3e-5, shape='circular')
    out['algebra_operator'] = _d(op._apply(E, dx=dx, dy=dx, wavelength=LAM)[0])

    from lumenairy.elements.polarization import JonesField
    jf = JonesField(Ex=E.copy(), Ey=(2.0 * E).copy(), dx=dx)
    jf.apply_aperture('circular', {'diameter': 5.3e-5})
    out['jones_ex'] = _d(jf.Ex)
    out['jones_ey'] = _d(jf.Ey)

    from lumenairy.propagators.system import propagate_through_system, propagate_through_system_jax
    base = {'type': 'aperture', 'shape': 'circular',
            'params': {'diameter': 5.3e-5}}
    for tag, extra in (('plain', {}), ('hard', {'edge': 'hard'}),
                       ('n8', {'edge_samples': 8})):
        el = [dict(base, **extra)]
        npy, _ = propagate_through_system(E, el, LAM, dx=dx)
        out['chain_numpy_' + tag] = _d(npy)
        try:
            import jax
            jax.config.update('jax_enable_x64', True)
            import jax.numpy as jnp
            out['chain_jit_' + tag] = _d(propagate_through_system_jax(
                jnp.asarray(E), el, LAM, dx))
            out['chain_eager_' + tag] = _d(propagate_through_system_jax(
                jnp.asarray(E), el, LAM, dx, verbose=True))
        except Exception as exc:               # noqa: BLE001
            # Record on BOTH keys and keep the message: the round-2 copy set
            # only ``chain_jit_`` here, so a failure in the EAGER call (which
            # runs second) silently dropped ``chain_eager_`` from the census
            # and turned a 23-fixture comparison into a 22-fixture one.
            msg = 'unavailable: {0}: {1}'.format(type(exc).__name__,
                                                 str(exc)[:160])
            out.setdefault('chain_jit_' + tag, msg)
            out.setdefault('chain_eager_' + tag, msg)

    # --- WP-C1 ROUND 3 additions ---------------------------------------
    # R1 hoists ``_aperture_edge_kwargs`` above the ``_resolve_aperture_params``
    # gate in both JAX routes.  An element whose params do NOT resolve is a
    # no-op on every route and must STAY one when its rim is legal.
    unres = [{'type': 'aperture', 'shape': 'circular', 'edge': 'hard'}]
    npy_u, _ = propagate_through_system(E, unres, LAM, dx=dx)
    out['unresolvable_legal_numpy'] = _d(npy_u)
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        out['unresolvable_legal_jit'] = _d(propagate_through_system_jax(
            jnp.asarray(E), unres, LAM, dx))
        out['unresolvable_legal_eager'] = _d(propagate_through_system_jax(
            jnp.asarray(E), unres, LAM, dx, verbose=True))
    except Exception as exc:                   # noqa: BLE001
        msg = 'unavailable: {0}: {1}'.format(type(exc).__name__,
                                             str(exc)[:160])
        out.setdefault('unresolvable_legal_jit', msg)
        out.setdefault('unresolvable_legal_eager', msg)
    # R2's isinstance(edge_samples, (bool, np.bool_)) must not swallow the
    # numpy scalars that are LEGAL, and R3's try/except must not swallow a
    # numpy string edge.
    out['numpy_str_edge'] = _d(apply_aperture(
        E, dx, 'circular', {'diameter': 5.3e-5}, edge=np.str_('gray')))

    rx = {
        'elements': [
            {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
            {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
             'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
            {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
             'glass_after': 'air', 'semi_diameter': 1.6e-3},
        ],
        'all_thicknesses': [1.5e-3, 2.5e-3, 18e-3],
        'aperture_diameter': 1.8e-3,
    }
    src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=0.9e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out['evaluate_default'] = _d(la.evaluate(rx, src).field)
    return out


def main(out_path):
    fx = fixtures()
    payload = {'lumenairy_file': la.__file__,
               'python': sys.version.split()[0],
               'numpy': np.__version__,
               'fixtures': fx}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('lumenairy:', la.__file__)
    print('fixtures:', len(fx))
    for k in sorted(fx):
        print('  {0:24s} {1}'.format(k, fx[k]))


if __name__ == '__main__':
    main(sys.argv[1])

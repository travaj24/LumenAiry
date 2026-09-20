"""WP-C4 round 2, V-C4-D2 -- every public entry point that reaches the MFT,
driven archive-to-archive, with and without the new ``mft_method=`` keyword.

VERIFY-WP-C4 D2 found seven public entry points that MOVE at a captured shape
and expose no one-keyword way back.  This probe drives all of them (plus the
four that already had one, as the control) at a CAPTURED shape and at a
REFUSED shape, digests the raw bytes, and -- on a tree that carries the
keyword -- repeats each captured call with ``mft_method='bluestein'`` and
``mft_method='separable'``.  ``r2_entry_compare.py`` then answers, per entry
point:

* did the no-keyword call MOVE between ``49ddf4bd`` and this branch at the
  captured shape (it must, or the entry point never reached the rule),
* is it identical at the refused shape (it must be, or the flip is wider than
  the rule says), and
* WHICH SPELLING of ``mft_method`` reproduces the base bytes exactly.

The spelling is a measurement and not a preference: it is ``'separable'``
wherever the caller was passing the separable flag into the primitive and
``'bluestein'`` everywhere else, and the point of measuring it is that the
Migration Guide has to name the right one per entry point.

    PYTHONPATH=<tree> python r2_entry.py <tree> <tag>

``tag`` is a free label that lands in the file name (``base`` / ``branch``).
"""
from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'probe_verify_c4'))
import v4lib  # noqa: E402

WL = 1.31e-6
#: Input grid for every driver.  ``N_out = 8`` is exactly ``N/32`` (captured,
#: and work-dense at 132 multiply-adds per kernel entry); ``N_out = 64`` is
#: ``N/4`` (refused by the ratio).
N_IN = 256
N_CAP = 8
N_REF = 64


def _gauss(n, dx, w):
    import numpy as np
    g = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)).astype(
        np.complex128)


def _singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _field_of(out):
    """Every driver's answer reduced to ONE array, whatever wrapper it came
    back in."""
    import numpy as np
    for attr in ('field', 'E', 'envelope'):
        if hasattr(out, attr):
            return np.asarray(getattr(out, attr))
    if isinstance(out, tuple):
        return np.asarray(out[0])
    return np.asarray(out)


# ---------------------------------------------------------------------------
# The drivers.  Each takes ``n_out`` and a route-keyword dict (empty on the
# base tree) and returns ONE array.
# ---------------------------------------------------------------------------

def d_compute_psf(la, n_out, kw):
    import numpy as np
    dx = 4e-6
    pupil = _gauss(N_IN, dx, N_IN * dx / 6.0)
    psf, _ = la.compute_psf(pupil, WL, 50e-3, dx, N_psf=int(n_out),
                            method='mft', dx_psf=2e-6, **kw)
    return np.asarray(psf)


def d_resample_field(la, n_out, kw):
    import numpy as np
    dx_in = 1e-6
    E = _gauss(N_IN, dx_in, N_IN * dx_in / 6.0)
    out, _ = la.resample_field(E, dx_in, dx_in * N_IN / float(n_out),
                               N_out=int(n_out), method='chirpz', **kw)
    return np.asarray(out)


def d_propagate_asm(la, n_out, kw):
    dx = 2e-6
    E = _gauss(N_IN, dx, N_IN * dx / 6.0)
    return _field_of(la.propagate(
        E, z=2e-3, wavelength=WL, dx=dx, method='asm',
        output_grid=(int(n_out), 0.5e-6), **kw))


def d_propagate_fresnel(la, n_out, kw):
    dx = 2e-6
    E = _gauss(N_IN, dx, N_IN * dx / 6.0)
    return _field_of(la.propagate(
        E, z=20e-3, wavelength=WL, dx=dx, method='fresnel',
        output_grid=(int(n_out), 0.5e-6), **kw))


def d_focus_readout(la, n_out, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    rmag, na = 20.0e-3, 0.05
    w = na * rmag
    dx = 2.0 * 4.0 * w / N_IN
    env = _gauss(N_IN, dx, w)
    return np.asarray(C.carrier_referenced_focus_readout(
        env, -rmag, rmag, WL, dx, dx_out=0.25e-6, N_out=int(n_out),
        on_replica='ignore', on_focus_containment='ignore', **kw))


def d_exact_focus_readout(la, n_out, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    rmag, na = 20.0e-3, 0.05
    w = na * rmag
    dx = 2.0 * 4.0 * w / N_IN
    env = _gauss(N_IN, dx, w)
    full = np.asarray(C.carrier_referenced_reconstruct(env, -rmag, WL, dx))
    return np.asarray(C.carrier_referenced_exact_focus_readout(
        full, -rmag, rmag, WL, dx, dx_out=0.25e-6, N_out=int(n_out),
        N_fine=N_IN, dx_fine=dx, on_readout_window='ignore',
        on_replica='ignore', **kw))


def d_re_reference(la, n_out, kw):
    import numpy as np
    from lumenairy.propagators.carrier_field import (
        CarrierField, CarrierSpec, FieldGrid, re_reference)
    dx, R = 0.3e-6, -5.0e-4
    env = _gauss(N_IN, dx, 20e-6)
    fa = CarrierField(env, FieldGrid((N_IN, N_IN), dx),
                      CarrierSpec(R=R), WL)
    gb = FieldGrid((int(n_out), int(n_out)), N_IN * dx / float(n_out))
    out = re_reference(fa, CarrierSpec(R=R * 1.1), gb,
                       on_nyquist='ignore', on_window='ignore', **kw)
    return np.asarray(out.envelope)


def _chain(la, n_out, kw, transport):
    from lumenairy.propagators import carrier as C
    n, dx, w, r_in = N_IN, 60e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    res = C.propagate_traced_carrier_chain(
        _gauss(n, dx, w), [{'prescription': presc, 'gap_before': 20e-3}],
        1.31e-6, dx, r_in=r_in, ray_subsample=16, n_workers=1,
        final_distance=8e-3, final_leg='paraxial', transport=transport,
        traced_kwargs=dict(on_undersample='silent', on_noncollimated='silent'),
        focus_readout=dict(dx_out=0.5e-6, N_out=int(n_out),
                           on_replica='ignore'),
        **kw)
    return _field_of(res)


def d_traced_chain_sziklas(la, n_out, kw):
    return _chain(la, n_out, kw, 'sziklas')


def d_traced_chain_collins(la, n_out, kw):
    return _chain(la, n_out, kw, 'collins')


def d_traced_chain_multi(la, n_out, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    n, dx, w, r_in = N_IN, 60e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    res = C.propagate_traced_carrier_chain_multi(
        [{'field': _gauss(n, dx, w), 'carrier': r_in}],
        [{'prescription': presc, 'gap_before': 20e-3}], 1.31e-6, dx,
        output_grid=dict(dx_out=0.5e-6, N_out=int(n_out),
                         on_replica='ignore'),
        final_distance=8e-3, ray_subsample=16, n_workers=1,
        final_leg='paraxial',
        traced_kwargs=dict(on_undersample='silent', on_noncollimated='silent'),
        **kw)
    return np.asarray(_field_of(res))


#: The four that already had a way back, as the CONTROL: they take the same
#: keyword name through ``**method_kwargs``, so if the split below were an
#: artefact of the driver rather than of the rule, these would show it too.
def d_asm_mft(la, n_out, kw):
    import numpy as np
    dx = 2e-6
    E = _gauss(N_IN, dx, N_IN * dx / 6.0)
    return np.asarray(la.angular_spectrum_propagate_mft(
        E, 2e-3, WL, dx, 0.5e-6, int(n_out),
        **{('method' if k == 'mft_method' else k): v for k, v in kw.items()}))


DRIVERS = {
    'compute_psf': (d_compute_psf, 'bluestein'),
    'resample_field': (d_resample_field, 'bluestein'),
    'propagate_asm': (d_propagate_asm, 'bluestein'),
    'propagate_fresnel': (d_propagate_fresnel, 'bluestein'),
    'carrier_referenced_focus_readout': (d_focus_readout, 'bluestein'),
    'carrier_referenced_exact_focus_readout': (d_exact_focus_readout,
                                               'separable'),
    're_reference': (d_re_reference, 'separable'),
    'propagate_traced_carrier_chain': (d_traced_chain_sziklas, 'bluestein'),
    'propagate_traced_carrier_chain_collins': (d_traced_chain_collins,
                                               'bluestein'),
    'propagate_traced_carrier_chain_multi': (d_traced_chain_multi,
                                             'bluestein'),
    'angular_spectrum_propagate_mft': (d_asm_mft, 'bluestein'),
}


def main(tree, tag):
    la = v4lib.anchor(tree)
    out = {'build': v4lib.build_tag(), 'tag': tag,
           'python': sys.version.split()[0],
           'lumenairy_version': la.__version__,
           'N_in': N_IN, 'N_captured': N_CAP, 'N_refused': N_REF,
           'keys': {}, 'errors': {}}
    for name, (fn, _spell) in DRIVERS.items():
        for label, n_out in (('captured', N_CAP), ('refused', N_REF)):
            arms = [('', {})]
            if tag != 'base':
                if label == 'captured':
                    arms += [('|bluestein', {'mft_method': 'bluestein'}),
                             ('|separable', {'mft_method': 'separable'})]
            for suffix, kw in arms:
                key = f"{name}|{label}{suffix}"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        arr = fn(la, n_out, kw)
                    out['keys'][key] = v4lib.digest_array(arr)
                    print(f"  {key:70s} {out['keys'][key][:16]}", flush=True)
                except Exception as exc:                      # noqa: BLE001
                    out['errors'][key] = f"{type(exc).__name__}: {exc}"
                    print(f"  {key:70s} ERROR {type(exc).__name__}: "
                          f"{str(exc)[:110]}", flush=True)
    out['n_keys'] = len(out['keys'])
    out['n_errors'] = len(out['errors'])
    print(f"{out['n_keys']} keys, {out['n_errors']} errors", flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"r2_entry_{tag}_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

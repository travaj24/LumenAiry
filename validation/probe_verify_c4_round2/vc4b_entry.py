"""VERIFY-WP-C4 ROUND 2, item 4 -- the way back at every entry point, driven
archive-to-archive at MY OWN captured and refused shapes.

Independent of ``r2_entry.py`` in three ways that matter:

* **Different shapes.**  ``512 -> 16`` captured (ratio exactly 1/32, work/entry
  264 -- the same work/entry the shipped suite's smallest firing shape reads)
  and ``512 -> 32`` refused (ratio 1/16).  The branch measured ``256 -> 8`` and
  ``256 -> 64``; a spelling that is an artefact of one input size would not
  survive both.
* **A twelfth entry point.**  ``propagate_carrier_referenced(transport=
  'collins')``, which the branch's own probe does not drive at all (its report
  argues from the signature that the rule cannot fire there).  Driven here so
  that "cannot fire" is a measurement.
* **Two more arms per captured call.**  ``mft_method=None`` and
  ``mft_method='auto'`` against the no-keyword call, which is item 6's simple
  byte check: ``None`` must be byte-identical to naming nothing.

    PYTHONPATH=<tree> python vc4b_entry.py <tree> <tag>

``tag`` is ``base`` (a ``git archive 49ddf4bd`` tree, which has no
``mft_method``) or ``branch``.

Author:  Andrew Traverso
"""
from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

WL = 1.31e-6
N_IN = 512
N_CAP = 16               # 512/32 -- captured; work/entry (512+16)/2 = 264
N_REF = 32               # 512/16 -- refused by the ratio condition


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


def _arr(out):
    import numpy as np
    for attr in ('field', 'E', 'envelope'):
        if hasattr(out, attr):
            return np.asarray(getattr(out, attr))
    if isinstance(out, tuple):
        return np.asarray(out[0])
    return np.asarray(out)


# --------------------------------------------------------------------------
# drivers: (lumenairy, n_out, route-keyword dict) -> one array
# --------------------------------------------------------------------------

def d_compute_psf(la, n, kw):
    import numpy as np
    dx = 4e-6
    psf, _ = la.compute_psf(_gauss(N_IN, dx, N_IN * dx / 6.0), WL, 50e-3, dx,
                            N_psf=int(n), method='mft', dx_psf=1e-6, **kw)
    return np.asarray(psf)


def d_resample_field(la, n, kw):
    import numpy as np
    dx = 1e-6
    out, _ = la.resample_field(_gauss(N_IN, dx, N_IN * dx / 6.0), dx,
                               dx * N_IN / float(n), N_out=int(n),
                               method='chirpz', **kw)
    return np.asarray(out)


def d_propagate_asm(la, n, kw):
    dx = 2e-6
    return _arr(la.propagate(_gauss(N_IN, dx, N_IN * dx / 6.0), z=2e-3,
                             wavelength=WL, dx=dx, method='asm',
                             output_grid=(int(n), 0.4e-6), **kw))


def d_propagate_fresnel(la, n, kw):
    dx = 2e-6
    return _arr(la.propagate(_gauss(N_IN, dx, N_IN * dx / 6.0), z=20e-3,
                             wavelength=WL, dx=dx, method='fresnel',
                             output_grid=(int(n), 0.4e-6), **kw))


def d_focus_readout(la, n, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    rmag, na = 20.0e-3, 0.05
    w = na * rmag
    dx = 2.0 * 4.0 * w / N_IN
    return np.asarray(C.carrier_referenced_focus_readout(
        _gauss(N_IN, dx, w), -rmag, rmag, WL, dx, dx_out=0.2e-6,
        N_out=int(n), on_replica='ignore',
        on_focus_containment='ignore', **kw))


def d_exact_focus_readout(la, n, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    rmag, na = 20.0e-3, 0.05
    w = na * rmag
    dx = 2.0 * 4.0 * w / N_IN
    env = _gauss(N_IN, dx, w)
    full = np.asarray(C.carrier_referenced_reconstruct(env, -rmag, WL, dx))
    return np.asarray(C.carrier_referenced_exact_focus_readout(
        full, -rmag, rmag, WL, dx, dx_out=0.2e-6, N_out=int(n),
        N_fine=N_IN, dx_fine=dx, on_readout_window='ignore',
        on_replica='ignore', **kw))


def d_re_reference(la, n, kw):
    import numpy as np
    from lumenairy.propagators.carrier_field import (
        CarrierField, CarrierSpec, FieldGrid, re_reference)
    dx, R = 0.3e-6, -5.0e-4
    fa = CarrierField(_gauss(N_IN, dx, 20e-6), FieldGrid((N_IN, N_IN), dx),
                      CarrierSpec(R=R), WL)
    out = re_reference(fa, CarrierSpec(R=R * 1.1),
                       FieldGrid((int(n), int(n)), N_IN * dx / float(n)),
                       on_nyquist='ignore', on_window='ignore', **kw)
    return np.asarray(out.envelope)


def _chain(la, n, kw, transport):
    from lumenairy.propagators import carrier as C
    dx, w, r_in = 30e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return _arr(C.propagate_traced_carrier_chain(
        _gauss(N_IN, dx, w),
        [{'prescription': presc, 'gap_before': 20e-3}], WL, dx,
        r_in=r_in, ray_subsample=16, n_workers=1, final_distance=8e-3,
        final_leg='paraxial', transport=transport,
        traced_kwargs=dict(on_undersample='silent',
                           on_noncollimated='silent'),
        focus_readout=dict(dx_out=0.4e-6, N_out=int(n),
                           on_replica='ignore'), **kw))


def d_chain_sziklas(la, n, kw):
    return _chain(la, n, kw, 'sziklas')


def d_chain_collins(la, n, kw):
    return _chain(la, n, kw, 'collins')


def d_chain_multi(la, n, kw):
    import numpy as np
    from lumenairy.propagators import carrier as C
    dx, w, r_in = 30e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return np.asarray(_arr(C.propagate_traced_carrier_chain_multi(
        [{'field': _gauss(N_IN, dx, w), 'carrier': r_in}],
        [{'prescription': presc, 'gap_before': 20e-3}], WL, dx,
        output_grid=dict(dx_out=0.4e-6, N_out=int(n),
                         on_replica='ignore'),
        final_distance=8e-3, ray_subsample=16, n_workers=1,
        final_leg='paraxial',
        traced_kwargs=dict(on_undersample='silent',
                           on_noncollimated='silent'), **kw)))


def d_carrier_referenced_collins(la, n, kw):
    """The twelfth entry point.  Its Collins leg keeps the input's N, so the
    output grid size is NOT ``n`` -- the row exists to measure whether the rule
    can fire here at all, which the branch argues from the signature."""
    import numpy as np
    from lumenairy.propagators import carrier as C
    dx, R = 0.4e-6, -6.0e-4
    env = _gauss(N_IN, dx, 25e-6)
    out = C.propagate_carrier_referenced(
        env, R, 3.0e-4, WL, dx, transport='collins',
        dx_out=dx * float(N_IN) / float(n), on_collins_sampling='ignore', **kw)
    return np.asarray(_arr(out))


def d_asm_mft(la, n, kw):
    """The control that already had a way back -- same keyword content, spelled
    ``method=`` because that name is free on this signature."""
    import numpy as np
    dx = 2e-6
    return np.asarray(la.angular_spectrum_propagate_mft(
        _gauss(N_IN, dx, N_IN * dx / 6.0), 2e-3, WL, dx, 0.4e-6, int(n),
        **{('method' if k == 'mft_method' else k): v for k, v in kw.items()}))


DRIVERS = {
    'compute_psf': d_compute_psf,
    'resample_field': d_resample_field,
    'propagate_asm': d_propagate_asm,
    'propagate_fresnel': d_propagate_fresnel,
    'carrier_referenced_focus_readout': d_focus_readout,
    'carrier_referenced_exact_focus_readout': d_exact_focus_readout,
    're_reference': d_re_reference,
    'propagate_traced_carrier_chain_sziklas': d_chain_sziklas,
    'propagate_traced_carrier_chain_collins': d_chain_collins,
    'propagate_traced_carrier_chain_multi': d_chain_multi,
    'propagate_carrier_referenced_collins': d_carrier_referenced_collins,
    'angular_spectrum_propagate_mft': d_asm_mft,
}


def main(tree, tag):
    la = L.anchor(tree)
    L.single_thread_ffts()
    out = {'build': L.build(), 'tag': tag, 'python': sys.version.split()[0],
           'lumenairy_version': la.__version__, 'N_in': N_IN,
           'N_captured': N_CAP, 'N_refused': N_REF,
           'keys': {}, 'errors': {}}
    for name, fn in DRIVERS.items():
        for label, n_out in (('captured', N_CAP), ('refused', N_REF)):
            arms = [('', {})]
            if tag != 'base' and label == 'captured':
                arms += [('|bluestein', {'mft_method': 'bluestein'}),
                         ('|separable', {'mft_method': 'separable'}),
                         ('|none', {'mft_method': None}),
                         ('|auto', {'mft_method': 'auto'})]
            for suffix, kw in arms:
                key = "%s|%s%s" % (name, label, suffix)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        arr = fn(la, n_out, kw)
                    out['keys'][key] = L.digest(arr)
                    print("  %-72s %s shape=%s" % (key, out['keys'][key][:16],
                                                   arr.shape), flush=True)
                except Exception as exc:                      # noqa: BLE001
                    out['errors'][key] = "%s: %s" % (type(exc).__name__, exc)
                    print("  %-72s ERROR %s: %s"
                          % (key, type(exc).__name__, str(exc)[:110]),
                          flush=True)
    out['n_keys'] = len(out['keys'])
    out['n_errors'] = len(out['errors'])
    print("%d keys, %d errors" % (out['n_keys'], out['n_errors']), flush=True)
    L.write(out, os.path.join(HERE, "vc4b_entry_%s_%s.json" % (tag, L.tag())))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

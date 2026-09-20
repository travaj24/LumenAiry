"""VERIFY-WP-C2 ROUND 2 -- the entry point the ALIAS hides.

``lumenairy/elements/_lens_real.py`` imports the tracer under a local
alias::

    from ..raytrace import (..., trace as _rt_trace)
    ...
    res_fan = _rt_trace(fan, surfs_fan, wavelength)

inside ``_apply_real_lens_impl``, which the exported ``apply_real_lens``
calls one hop away whenever ``seidel_correction=True``.  Neither census --
the shipped one in ``test_c2_analytic_normal_default.py`` nor this
verification's module-qualified one -- sees it, because both decide "this
body traces" by looking for the NAME ``trace``, and the body names
``_rt_trace``.  The transitive walk misses it for the same reason, which
is why ``apply_real_lens`` is absent from the 46-function population in
``validation/probe_c2_round2/r2_transitive_parents_*.json``.

This probe answers three questions with numbers:

1. does ``apply_real_lens(seidel_correction=True)`` actually reach the
   tracer?  (a spy on ``lumenairy.raytrace.trace`` counts the calls and
   records the ``sphere_normal`` / ``renormalize`` each one received)
2. does its ANSWER move at the shipped defaults?  (the same call in a
   PRE-tree process and in a tip process, SHA-256 over the returned
   array's dtype + shape + raw bytes)
3. is there any way back?  (``inspect.signature``)

Usage:  python vr2_alias_entrypoint.py <out.json> [--tag pre|post]
"""
import hashlib
import inspect
import json
import pathlib
import sys

import numpy as np


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def build_field():
    n = 256
    dx = 80.0e-6
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    w0 = 4.0e-3
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    return E, dx


def main(out_path, tag):
    import lumenairy as la
    print('lumenairy.__file__ =', la.__file__)
    from lumenairy.io.prescriptions_builders import make_singlet

    out = {'tag': tag, 'lumenairy_file': la.__file__,
           'python': sys.version.split()[0]}

    # ---- 1. the signature -------------------------------------------
    params = inspect.signature(la.apply_real_lens).parameters
    out['apply_real_lens_has_sphere_normal'] = 'sphere_normal' in params
    out['apply_real_lens_has_renormalize'] = 'renormalize' in params
    out['apply_real_lens_module'] = la.apply_real_lens.__module__

    # ---- 2. the spy --------------------------------------------------
    import lumenairy.raytrace as rt
    real_trace = rt.trace
    seen = []

    def spy(*a, **kw):
        seen.append({'sphere_normal': kw.get('sphere_normal', '<omitted>'),
                     'renormalize': kw.get('renormalize', '<omitted>')})
        return real_trace(*a, **kw)

    E, dx = build_field()
    pres = make_singlet(0.0300, -0.0300, 0.0060, 'N-BK7', 0.0200)
    pres['aperture_diameter'] = 0.0180
    wl = 587.6e-9
    kwargs = dict(prescription=pres, wavelength=wl, dx=dx,
                  seidel_correction=True)

    rt.trace = spy
    try:
        E_spy = la.apply_real_lens(E, **kwargs)
    finally:
        rt.trace = real_trace
    out['spy_trace_calls'] = len(seen)
    out['spy_kwargs'] = seen
    out['digest_spy'] = digest(E_spy)

    # ---- 3. the answer at the library default ------------------------
    E_out = la.apply_real_lens(E, **kwargs)
    out['digest_default'] = digest(E_out)
    out['abs_sum'] = float(np.sum(np.abs(E_out)))

    # control: the same call with the correction OFF must not trace
    seen2 = []

    def spy2(*a, **kw):
        seen2.append(1)
        return real_trace(*a, **kw)

    rt.trace = spy2
    try:
        E_off = la.apply_real_lens(
            E, prescription=pres, wavelength=wl, dx=dx,
            seidel_correction=False)
    finally:
        rt.trace = real_trace
    out['spy_trace_calls_correction_off'] = len(seen2)
    out['digest_correction_off'] = digest(E_off)

    # ---- 4. on the tip only: can the caller get the old arithmetic? ---
    # There is no keyword, so the ONLY way back is to monkeypatch the
    # tracer.  Record what the old arithmetic would have produced, by
    # wrapping trace with the pre-WP-C2 keywords forced.
    def forced(*a, **kw):
        kw.setdefault('sphere_normal', 'generic')
        kw.setdefault('renormalize', 'surface')
        return real_trace(*a, **kw)

    try:
        rt.trace = forced
        E_forced = la.apply_real_lens(E, **kwargs)
        out['digest_forced_old'] = digest(E_forced)
        out['forced_differs_from_default'] = (
            out['digest_forced_old'] != out['digest_default'])
        out['forced_max_abs_delta'] = float(
            np.max(np.abs(E_forced - E_out)))
    except TypeError as exc:
        out['digest_forced_old'] = 'TypeError: %s' % exc
    finally:
        rt.trace = real_trace

    pathlib.Path(out_path).write_text(json.dumps(out, indent=1),
                                      encoding='utf-8')
    for k, v in out.items():
        if k != 'spy_kwargs':
            print('%-36s %s' % (k, v))
    print('spy_kwargs', out['spy_kwargs'])


if __name__ == '__main__':
    tag = 'post'
    args = [a for a in sys.argv[1:]]
    if '--tag' in args:
        tag = args[args.index('--tag') + 1]
        args = args[:args.index('--tag')]
    main(args[0] if args else 'vr2_alias.json', tag)

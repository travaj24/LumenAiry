"""Warmed-tracemalloc probe for the D121 audit S11 items 1(c) and 6.

Item 1(c): the ``sag_dtype`` memory term for the runner preflight -- MEASURE
           the float32 peak, do not divide the float64 peak by two.
Item 6:    the ``sag_chunk_rows`` lever on the analytic tangent-facet path.

Protocol is the one the shipped anchors used (runner docstring ANCHOR
2026-08-16 and tests/unit/test_obl_banded_halo.py::
test_banded_peak_is_smaller_than_whole_grid): WARMED -- one throwaway call of
the identical arm first, because the FIRST apply_real_lens of a process also
pays FFT-plan / lazy-import allocations that would otherwise land in the peak
and flatter whichever arm ran second.

Units: float64 grids of 8*N*N bytes, so the readings are directly comparable
to the anchors already in the preflight docstring.
"""
import gc
import json
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements._lens_real import apply_real_lens

_EXPECT = os.environ.get('PROBE_EXPECT_LIB', '')
print('lumenairy.__file__ =', la.__file__, flush=True)
print('lumenairy.__version__ =', la.__version__, flush=True)
if _EXPECT:
    assert os.path.normcase(os.path.abspath(la.__file__)).startswith(
        os.path.normcase(os.path.abspath(_EXPECT))), (
        f'wrong lumenairy: {la.__file__} not under {_EXPECT}')

LAM = 1.31e-6


def _surf(radius, gb, ga, **extra):
    d = {'radius': radius, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': gb, 'glass_after': ga}
    d.update(extra)
    return d


def _biconvex():
    """The fixture the shipped tangent-facet anchor used: the fast biconvex
    singlet standing in for design 121's binding facet (R = +19.6 / -27.4 mm,
    N-SSK2, 3 mm aperture, 4 mm centre thickness)."""
    return {'name': 'biconvex', 'aperture_diameter': 3e-3,
            'thicknesses': [4e-3],
            'surfaces': [_surf(19.6e-3, 'air', 'N-SSK2'),
                         _surf(-27.4e-3, 'N-SSK2', 'air')]}


SPHERE = la.TiltedCarrier(0.25, 0.05, 0.0)   # finite radius: the conservative
                                             # pricing (see the preflight doc)


def _field(N, dx):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (0.8e-3) ** 2)
    return (env * np.exp(1j * 2 * np.pi / LAM * 0.01 * X)).astype(np.complex128)


def peak(E, **kw):
    """Warmed tracemalloc peak of ONE apply_real_lens call, in bytes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        apply_real_lens(E.copy(), **kw)          # warm-up, discarded
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        out = apply_real_lens(E.copy(), **kw)
        wall = time.perf_counter() - t0
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    gc.collect()
    return pk, wall, out


def run(N, do_chunk=False):
    dx = 4.0e-3 / N
    grid = 8.0 * N * N
    E = _field(N, dx)
    presc = _biconvex()
    base = dict(prescription=presc, wavelength=LAM, dx=dx)
    la.set_fft_auto_promote(False)

    rows = {}

    def rec(tag, **kw):
        pk, wall, out = peak(E, **base, **kw)
        rows[tag] = {'peak_bytes': pk, 'grids': pk / grid, 'wall_s': wall}
        print(f'  N={N:<6d} {tag:<38s} {pk/grid:8.3f} grids '
              f'({pk/1e9:7.3f} GB)  {wall:7.2f} s', flush=True)
        return out

    # --- the anchors' BASELINE: paraxial, no carrier, no surface model ------
    rec('baseline_paraxial_nocarrier')
    rec('baseline_paraxial_nocarrier_f32', sag_dtype=np.float32)

    # --- item 1(c): the sag_dtype term on the tangent-facet family ----------
    rec('tf_carrier_f64', surface_model='tangent_facet', carrier=SPHERE)
    rec('tf_carrier_f32', surface_model='tangent_facet', carrier=SPHERE,
        sag_dtype=np.float32)
    rec('tf_nocarrier_f64', surface_model='tangent_facet')
    rec('tf_nocarrier_f32', surface_model='tangent_facet',
        sag_dtype=np.float32)
    # control: does sag_dtype move the VERTEX-PLANE screen too?
    rec('vertex_carrier_f64', carrier=SPHERE, screen_obliquity=True,
        on_screen_obliquity='silent')
    rec('vertex_carrier_f32', carrier=SPHERE, screen_obliquity=True,
        on_screen_obliquity='silent', sag_dtype=np.float32)

    # --- item 6: the sag_chunk_rows lever, tangent_facet + carrier ----------
    if do_chunk:
        ref = None
        auto_rows = max(256, N // 16)
        for tag, ch in (('chunk_AUTO', None), ('chunk_1024', 1024),
                        ('chunk_512', 512), ('chunk_256', 256)):
            kw = dict(surface_model='tangent_facet', carrier=SPHERE)
            if ch is not None:
                kw['sag_chunk_rows'] = ch
            out = rec(tag, **kw)
            if ref is None:
                ref = out
                print(f'    (AUTO resolves to {auto_rows} rows)', flush=True)
            else:
                same = bool(np.array_equal(out, ref))
                rows[tag]['byte_identical_to_AUTO'] = same
                print(f'    byte-identical to AUTO: {same}', flush=True)
                del out
            gc.collect()
        del ref
        gc.collect()

    return rows


if __name__ == '__main__':
    Ns = [int(t) for t in sys.argv[1].split(',')]
    chunk_at = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    out = {'lumenairy': la.__version__, 'file': la.__file__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'rows': {}}
    for N in Ns:
        out['rows'][str(N)] = run(N, do_chunk=(N == chunk_at))
    dest = sys.argv[3] if len(sys.argv) > 3 else 'probe_sag_out.json'
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', dest, flush=True)

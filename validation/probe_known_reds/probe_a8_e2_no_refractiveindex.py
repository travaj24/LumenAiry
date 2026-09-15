"""A8/E2: measure ``get_glass_index*`` with and without ``refractiveindex``.

Reproduces the six A8/E2 reds of CI run 34914295323 on a box where the package
IS installed, by importing ``block_refractiveindex`` first (``--block``).

Emits JSON on stdout and to ``--out``.  Arm label goes in ``arm``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ap = argparse.ArgumentParser()
_ap.add_argument('--block', action='store_true',
                 help='make refractiveindex unimportable before importing lumenairy')
_ap.add_argument('--out', default=None)
_args = _ap.parse_args()

if _args.block:
    import block_refractiveindex
    block_refractiveindex.install()

import numpy as np                                    # noqa: E402
import lumenairy.glass as G                           # noqa: E402

# Pin the tree: the pip -e install points at a DIFFERENT branch, and this
# probe is also run against a ``git archive HEAD`` copy for the
# before/after comparison, so assert against PYTHONPATH's first entry
# rather than hard-coding one worktree name.
_want = os.path.abspath(os.environ['PYTHONPATH'].split(os.pathsep)[0])
assert os.path.abspath(G.__file__).startswith(_want), (G.__file__, _want)


WAVELENGTHS = (1.31e-6, 1.55e-6)


def tuple_names():
    return sorted(n for n, e in G.GLASS_REGISTRY.items()
                  if isinstance(e, tuple) and e[0] != '__user__')


def bundled(name):
    return (name in G.SELLMEIER_COEFFICIENTS
            or name in G.POLYNOMIAL_COEFFICIENTS)


res = {
    'arm': 'blocked' if _args.block else 'present',
    'python': sys.version.split()[0],
    'lumenairy_file': G.__file__,
    'refractiveindex_available': bool(G._REFRACTIVEINDEX_AVAILABLE),
    'n_tuple_registered': len(tuple_names()),
    'tuple_without_bundled_row': [n for n in tuple_names() if not bundled(n)],
}

# --- E2 arm 1: does get_glass_index_complex raise for a catalogue glass? ----
raises = {}
values = {}
for wl in WAVELENGTHS:
    r, v = [], {}
    for name in tuple_names():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                nc = G.get_glass_index_complex(name, wl)
                v[name] = [nc.real, nc.imag]
            except Exception as exc:                  # noqa: BLE001
                r.append([name, type(exc).__name__, str(exc)[:120]])
    raises[f'{wl:g}'] = r
    values[f'{wl:g}'] = v
res['complex_raises'] = raises
res['n_nonfinite_or_negative_kappa'] = {
    k: [[n, val] for n, val in v.items()
        if not (np.isfinite(val[0]) and np.isfinite(val[1]) and val[1] >= 0.0)]
    for k, v in values.items()}
res['n_with_nonzero_kappa'] = {
    k: sum(1 for val in v.values() if val[1] != 0.0) for k, v in values.items()}

# --- E2 arm 2: real-index entry point on the same names --------------------
real_raises = []
for name in tuple_names():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            G.get_glass_index(name, 1.31e-6)
        except Exception as exc:                      # noqa: BLE001
            real_raises.append([name, type(exc).__name__])
res['real_index_raises_1.31um'] = real_raises

# --- warn-once, two calls, for the four names the a8 test parametrises -----
warn_once = {}
for name in ('CaF2', 'FUSED_SILICA', 'MgF2', 'SILICON', 'N-BK7'):
    G._kappa_warned.clear()
    G._clear_glass_caches()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            a = G.get_glass_index_complex(name, 1.31e-6)
            b = G.get_glass_index_complex(name, 1.31e-6)
            hits = [str(w.message)[:60] for w in rec
                    if issubclass(w.category, RuntimeWarning)
                    and 'extinction' in str(w.message)]
            warn_once[name] = {'n_warnings': len(hits),
                               'kappa': a.imag, 'identical': a == b,
                               'n_real': a.real}
        except Exception as exc:                      # noqa: BLE001
            warn_once[name] = {'raised': type(exc).__name__}
res['warn_once'] = warn_once

# --- N-BK7 exact complex value (the byte-identity anchor) ------------------
G._kappa_warned.clear()
G._clear_glass_caches()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nbk7 = G.get_glass_index_complex('N-BK7', 1.55e-6)
res['N-BK7_1.55um'] = {'repr': repr(nbk7),
                       'real_hex': float(nbk7.real).hex(),
                       'imag_hex': float(nbk7.imag).hex()}

# --- the out-of-range refusal the verify test pins -------------------------
oor = {}
for wl in (0.633e-6, 1.064e-6, 20e-6):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fn in ('get_glass_index', 'get_glass_index_complex'):
            try:
                out = getattr(G, fn)('SILICON', wl)
                oor[f'{fn}@{wl:g}'] = ['returned', repr(out)]
            except Exception as exc:                  # noqa: BLE001
                oor[f'{fn}@{wl:g}'] = [type(exc).__name__, str(exc)[:90]]
res['silicon_out_of_range'] = oor

# --- every catalogue glass's complex value, for the byte-identity compare ---
G._kappa_warned.clear()
G._clear_glass_caches()
allvals = {}
for wl in WAVELENGTHS:
    for name in tuple_names():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                nc = G.get_glass_index_complex(name, wl)
                allvals[f'{name}@{wl:g}'] = [float(nc.real).hex(),
                                             float(nc.imag).hex()]
            except Exception as exc:                  # noqa: BLE001
                allvals[f'{name}@{wl:g}'] = ['RAISED', type(exc).__name__]
res['complex_hex_all'] = allvals

out = json.dumps(res, indent=1, sort_keys=True)
if _args.out:
    with open(_args.out, 'w', encoding='cp1252') as fh:
        fh.write(out)
print(out)

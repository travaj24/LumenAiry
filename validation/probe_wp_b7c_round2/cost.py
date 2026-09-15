"""The arbiter's cost, measured on the tree this is run against.

Times ``apply_real_lens_traced_uniform`` (the only caller that asks for the
reading) and ``apply_real_lens_traced_multibranch`` (which never does) at a
spread of planes including the near-identity exit vertex, where the map is
uncompressed and the half-pitch render is at its most expensive.  Run against
the round-1 tree and against HEAD and divide.

Usage::

    python cost.py <tree> <out.json>
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)

CASES = [('V', 1758.0), ('V', 0.0), ('S', 3214.78), ('F_alt', 1063.0),
         ('G', 950.0), ('A', 2800.0), ('K', 2000.0), ('Q', 5680.0)]


def _time(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fn()
        except RuntimeError:
            pass
        ts.append(time.perf_counter() - t)
    return min(ts)


def main(argv):
    tree = os.path.abspath(argv[1])
    assert os.path.abspath(lumenairy.__file__).startswith(tree), (
        f'lumenairy.__file__ = {lumenairy.__file__} is not under {tree}')
    res = {'tree': tree, 'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    print(f'tree {lumenairy.__file__}')
    for name, zum in CASES:
        fx = FX.FIXTURES[name]
        E = FX.input_field(fx)
        kw = dict(prescription=fx['prescription'],
                  wavelength=fx['wavelength'], dx=fx['dx'],
                  output_plane_distance=zum * 1e-6, return_diagnostics=True)
        t_uni = _time(lambda: apply_real_lens_traced_uniform(E, **kw))
        t_mb = _time(lambda: apply_real_lens_traced_multibranch(E, **kw))
        r = {'fixture': name, 'z_um': zum, 'N': fx['N'],
             't_uniform_s': t_uni, 't_multibranch_s': t_mb}
        res['rows'].append(r)
        print(f"  {name:6s} z={zum:9.2f} N={fx['N']:4d}  uniform "
              f"{t_uni:6.3f}s  multibranch {t_mb:6.3f}s", flush=True)
        with open(argv[2], 'w') as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)

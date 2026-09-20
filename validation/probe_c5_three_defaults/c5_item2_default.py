"""WP-C5 item 2, the DEFAULT-PATH reading.

``c5_item2_budget.py`` names the accounting mode on every key, which is what
makes "``'legacy'`` is byte-identical to the parent commit" checkable -- and
what makes it blind to the thing the item actually changes.  This probe digests
calls that touch NO switch at all, so the comparison against the parent archive
reads the DEFAULT move and nothing else.

Small grids on purpose: the question here is which calls move, not how large
the transient is.

Usage (BLAS and the env budget pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_c5_three_defaults/c5_item2_default.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                             # noqa: E402

from _digest import dig, write                                 # noqa: E402

from lumenairy.propagators import gbd as G                     # noqa: E402

DX, WL = 2.0e-6, 1.0e-6


def bundle(n, seed=0):
    rng = np.random.default_rng(seed)
    return G.BeamletBundle(
        positions=rng.normal(0.0, 2.0e-4, size=(n, 3)),
        directions=np.zeros((n, 3)),
        Q=np.full(n, 1.0 / (1.0e-3 - 0.02j), dtype=np.complex128),
        amplitude=(rng.normal(size=n)
                   + 1j * rng.normal(size=n)).astype(np.complex128),
        waist0=np.full(n, 1.0e-3))


def main(out):
    res = {'shipped_default': G.DENSE_MEM_BUDGET_ACCOUNTING,
           'floor_helper': hasattr(G, '_dense_budget_floor_bytes')}
    digests = {}
    notices = {}
    b = bundle(512)

    for N in (64, 128, 192, 256):
        for mb in (512.0, 64.0, 8.0, 1.0):
            key = 'default/N%d/%gMB' % (N, mb)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                f = G.reconstruct_field_from_beamlets(
                    b, Ny=N, Nx=N, dx=DX, wavelength=WL,
                    chunk_beamlets=4096, mem_budget_mb=mb)
            digests[key] = dig(f)
            notices[key] = [m for m in (str(w.message) for w in rec)
                            if 'mem_budget_mb' in m and 'floor' in m]

    # the shipped default budget of every caller that reaches this loop
    for N in (64, 128):
        f = G.reconstruct_field_from_beamlets(
            b, Ny=N, Nx=N, dx=DX, wavelength=WL)
        digests['default/N%d/no-budget-arg' % N] = dig(f)
        digests['windowed/N%d/default' % N] = dig(
            G.reconstruct_field_from_beamlets(
                b, Ny=N, Nx=N, dx=DX, wavelength=WL, window=5.0))
        E = np.exp(-((np.arange(N) - N / 2) * DX) ** 2 / (2.0e-4) ** 2)
        digests['frame_completeness/N%d' % N] = dig(
            G.frame_completeness(b, np.outer(E, E).astype(np.complex128),
                                 DX, wavelength=WL))

    res['digests'] = digests
    res['floor_notices'] = {k: v for k, v in notices.items() if v}
    res['n_keys_with_notice'] = sum(1 for v in notices.values() if v)
    write(out, res)


if __name__ == '__main__':
    main(sys.argv[1])

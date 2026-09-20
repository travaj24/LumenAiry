"""VERIFY-WP-C4 -- does the Migration Guide's process-wide remedy actually
restore the base bytes at the entry points that expose no route keyword?

Runs the six drivable no-keyword public entry points twice in one process --
once as shipped and once with ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`` --
and prints both digest maps.  Run it against the branch and against a
``git archive`` of ``49ddf4bd``, then compare: the base's default digests must
equal the branch's ``never`` digests (the remedy works) and differ from the
branch's default ones (the exposure is real).

    PYTHONPATH=<tree> python v4_never_entry.py <tree>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

WL = 633e-9


def gauss(np, ny, nx, dx, w, seed=3):
    rng = np.random.default_rng(seed)
    y = (np.arange(ny) - ny / 2.0) * dx
    x = (np.arange(nx) - nx / 2.0) * dx
    return (np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / w ** 2
                   ).astype(np.complex128)
            * np.exp(1j * 0.3 * rng.standard_normal((ny, nx))))


def dig(np, r):
    while isinstance(r, tuple):
        r = r[0]
    for attr in ('envelope', 'field'):
        if not isinstance(r, np.ndarray) and hasattr(r, attr):
            r = getattr(r, attr)
            break
    a = np.ascontiguousarray(np.asarray(r))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def cases(np):
    from lumenairy.analysis.psf_mtf_otf import compute_psf
    from lumenairy.propagators.carrier import (
        carrier_referenced_exact_focus_readout,
        carrier_referenced_focus_readout)
    from lumenairy.propagators.carrier_field import (CarrierField, CarrierSpec,
                                                     FieldGrid, re_reference)
    from lumenairy.propagators.dispatch import propagate
    from lumenairy.propagators.mft import resample_field

    E = gauss(np, 512, 512, 4e-6, 40e-6)
    pup = gauss(np, 512, 512, 4e-6, 400e-6)
    dxo = WL * 0.05 / (512 * 4e-6)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out['compute_psf'] = dig(np, compute_psf(
            pup, WL, 0.05, 4e-6, 16, method='mft'))
        out['resample_field'] = dig(np, resample_field(
            E, 4e-6, 4e-6 * 32, 16, method='chirpz'))
        out['propagate_asm'] = dig(np, propagate(
            E, z=0.05, wavelength=WL, dx=4e-6, method='asm',
            output_grid=(16, dxo), return_result=False))
        out['carrier_focus'] = dig(np, carrier_referenced_focus_readout(
            E, 2e-2, 2e-2, WL, 4e-6, dx_out=4e-7, N_out=16,
            on_replica='ignore'))
        out['carrier_exact'] = dig(np, carrier_referenced_exact_focus_readout(
            E, 2e-2, 2e-2, WL, 4e-6, dx_out=4e-7, N_out=16, N_fine=2048,
            on_replica='ignore', on_readout_window='ignore'))
        f = CarrierField(E.copy(), FieldGrid((512, 512), 4e-6),
                         CarrierSpec(R=-2e-2), WL)
        out['re_reference'] = dig(np, re_reference(
            f, CarrierSpec(R=-2.4e-2), FieldGrid((16, 16), 4e-6 * 32),
            on_nyquist='ignore', on_window='ignore'))
    return out


def main(tree):
    v4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    res = {'tree': tree, 'build': v4lib.build_tag(), 'default': cases(np)}
    if hasattr(B, '_MFT_DIRECT_MAX_RATIO'):
        saved = B._MFT_DIRECT_MAX_RATIO
        B._MFT_DIRECT_MAX_RATIO = B._MFT_DIRECT_NEVER
        try:
            res['never'] = cases(np)
        finally:
            B._MFT_DIRECT_MAX_RATIO = saved
    print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1])

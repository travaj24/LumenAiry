"""VERIFY-WP-C4 claim 5 -- byte identity archive-to-archive against
``49ddf4bd``, on MY OWN fixtures.

Three groups, the same three the branch claims, driven on fixtures built
independently of its probe: 13 shapes of which only two coincide with its 12,
a different field (a windowed random phase rather than a tapered Gaussian),
a different second-axis ``alpha`` (``0.75*a`` where it uses ``1.5*a``), a
different off-centre convention, and anisotropic cases chosen so the two axes
disagree about the boundary:

``bl/*``      the previous route named explicitly, ``method='bluestein'``
``sep/*``     the previous route named explicitly, ``method='separable'``
``never/*``   no keyword, with ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER``
``nokw/*``    no keyword at all -- the SPLIT, which must agree with the rule

``rule_says`` is recorded beside each key so the split is CHECKED and not
counted, and ``v4_bitid_compare.py`` asserts the agreement.

    PYTHONPATH=<tree> python v4_bitid.py <tree> OUT.json
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

#: ``(tag, Ny_in, Nx_in, N_out_y, N_out_x)``.  Ratios straddle 1/32; the last
#: four are anisotropic with the two axes on opposite sides of it, which is
#: what the ``max`` conjunction decides.
SHAPES = [
    ('s1o128', 256, 256, 2, 2),
    ('s1o64', 384, 384, 6, 6),
    ('s1o32', 288, 288, 9, 9),
    ('s1o32b', 512, 512, 16, 16),
    ('s1o30', 240, 240, 8, 8),
    ('s1o16', 192, 192, 12, 12),
    ('s1o8', 128, 128, 16, 16),
    ('s1o2', 40, 40, 20, 20),
    ('s1o1', 26, 18, 13, 9),
    ('a_both_under', 512, 128, 16, 4),
    ('a_y_under', 512, 128, 16, 16),
    ('a_x_under', 128, 512, 16, 16),
    ('a_thin', 1024, 32, 32, 1),
]


def _field(np, ny, nx, seed=606):
    rng = np.random.default_rng(seed)
    y = (np.arange(ny) - ny / 2.0) / max(ny, 1)
    x = (np.arange(nx) - nx / 2.0) / max(nx, 1)
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return (np.exp(-6.0 * r2) * (1.0 + 0.2 * rng.standard_normal((ny, nx)))
            ).astype(np.complex128) * np.exp(
                2j * np.pi * rng.random((ny, nx)))


class Rec:
    def __init__(self):
        self.keys = {}

    def record(self, key, fn, *a, **kw):
        import numpy as np
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                r = fn(*a, **kw)
            except Exception as exc:                      # noqa: BLE001
                self.keys[key] = f"RAISED:{type(exc).__name__}"
                return
            arr = np.asarray(r)
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        for ww in w:
            h.update(b'\x00W\x00')
            h.update(str(ww.message).encode('utf-8', 'replace'))
        self.keys[key] = h.hexdigest()


def drive(rec, np, B, prefix, kw_route):
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    for tag, ny, nx, my, mx in SHAPES:
        E = _field(np, ny, nx)
        # a budget of 1e3, so nothing here measures the warning machinery
        a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
        for sgn in (-1, +1):
            for sep in (False, True):
                base = dict(sign=sgn, xp=np, fft2=_fft2, ifft2=_ifft2,
                            separable=sep)
                rec.record(f"{prefix}/plain/{tag}/s{sgn}/sep{int(sep)}",
                           B._bluestein_2d, E, a, a * 0.75, my, mx,
                           **base, **kw_route)
                rec.record(f"{prefix}/centred/{tag}/s{sgn}/sep{int(sep)}",
                           B._bluestein_centred_2d, E, a, a * 0.75, my, mx,
                           **base, **kw_route)
        rec.record(f"{prefix}/offcentre/{tag}",
                   B._bluestein_centred_2d, E, a, a * 0.75, my, mx,
                   n_centre_in_x=nx / 2.0, n_centre_in_y=ny / 2.0 - 0.25,
                   k_centre_out_x=mx / 3.0, k_centre_out_y=my / 2.0,
                   sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, **kw_route)


def main(tree, out_path):
    lum = v4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    rec = Rec()
    drive(rec, np, B, 'nokw', {})
    drive(rec, np, B, 'bl', {'method': 'bluestein'})
    drive(rec, np, B, 'sep', {'method': 'separable'})
    saved = getattr(B, '_MFT_DIRECT_MAX_RATIO', None)
    if saved is not None:
        B._MFT_DIRECT_MAX_RATIO = B._MFT_DIRECT_NEVER
    try:
        drive(rec, np, B, 'never', {})
    finally:
        if saved is not None:
            B._MFT_DIRECT_MAX_RATIO = saved
    rule = {}
    if hasattr(B, '_auto_selects_direct'):
        for tag, ny, nx, my, mx in SHAPES:
            rule[tag] = bool(B._auto_selects_direct(ny, nx, my, mx))
    v4lib.write_json({'build': v4lib.build_tag(), 'tree': tree,
                      'lumenairy_file': lum.__file__,
                      'version': lum.__version__,
                      'has_rule': hasattr(B, '_MFT_DIRECT_MAX_RATIO'),
                      'rule_says': rule, 'keys': rec.keys}, out_path)
    print(f"{len(rec.keys)} keys ({v4lib.build_tag()}, {lum.__file__})")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

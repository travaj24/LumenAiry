"""WP-C4 tasks 2 and 5 -- the digest map the byte-identity claims are read
from, and the rule's purity.

Run against BOTH trees (the branch worktree and a read-only ``git archive``
extraction of the base commit) on BOTH builds; ``c4_compare.py`` compares the
two maps key by key.

WHAT THE KEYS COVER
-------------------
``live/wayback_bl/*`` and ``live/wayback_sep/*``
    Every fixture with the PREVIOUS route's keyword passed explicitly
    (``method='bluestein'`` / ``'separable'``, and for the public propagators
    the same).  These must be byte-identical base-to-branch at EVERY shape --
    that is the "one keyword away" claim.
``live/nokw/*``
    Every fixture with NO keyword at all.  These are byte-identical base to
    branch exactly at the shapes the rule sends to the previous route, and
    must DIFFER at the shapes it sends to the dense one.  The comparison
    script counts both sides and checks the split against the rule.
``never/*``
    The same no-keyword fixtures with ``_MFT_DIRECT_MAX_RATIO`` set to
    ``_MFT_DIRECT_NEVER`` for the whole process.  Byte-identical to base at
    EVERY shape -- the process-wide way back.  (On the base tree the constant
    does not exist, so these keys are produced by simply running the
    no-keyword fixtures; the constant is set only where it is present.)
``rule_says``
    Not a digest: what :func:`_auto_selects_direct` answers for each fixture's
    shape, recorded beside the digests so the ``live/nokw`` split can be
    CHECKED against the rule rather than merely counted.  (The claim that
    ``'auto'`` is one of the two routes bit for bit, and never a third
    arithmetic, is ``c4_rule.py``'s -- it is a within-tree claim.)

The propagator fixtures are driven at shapes on BOTH sides of the boundary so
the split is populated, including the anisotropic and off-axis cases the rule's
``max`` over the two axes exists for.

    PYTHONPATH=<tree> python c4_bitid.py <tree> OUT.json
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

WL = 633e-9


def _gauss(np, ny, nx, dx, w, seed=3):
    rng = np.random.default_rng(seed)
    y = (np.arange(ny) - ny / 2.0) * dx
    x = (np.arange(nx) - nx / 2.0) * dx
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    E = np.exp(-r2 / w ** 2).astype(np.complex128)
    return E * np.exp(1j * 0.3 * rng.standard_normal((ny, nx)))


def _rand(np, ny, nx, seed=20260920):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


#: ``(tag, Ny_in, Nx_in, N_out_y, N_out_x)``.  Ratios straddle 1/32 on purpose,
#: and the last three are anisotropic: one axis under the boundary and one over,
#: which is the case the ``max`` over the two axes decides.
PRIM_SHAPES = [
    ('r1o128', 256, 256, 2, 2),
    ('r1o64', 512, 512, 8, 8),
    ('r1o32', 512, 512, 16, 16),
    ('r1o32b', 1024, 1024, 32, 32),
    ('r1o24', 768, 768, 32, 32),
    ('r1o16', 512, 512, 32, 32),
    ('r1o8', 256, 256, 32, 32),
    ('r1o2', 48, 48, 24, 24),
    ('r1o1', 28, 22, 15, 13),
    ('aniso_lo', 512, 1024, 16, 32),
    ('aniso_split', 512, 1024, 16, 64),
    ('aniso_hi', 512, 1024, 64, 32),
]

#: ``(tag, N_in, dx, z, N_out, zoom)`` for the three public propagators.
PROP_SHAPES = [
    ('p_small_out', 512, 4e-6, 0.05, 16, 1.0),
    ('p_tiny_out', 1024, 2e-6, 0.02, 32, 1.0),
    ('p_mid_out', 256, 8e-6, 0.05, 32, 1.0),
    ('p_same', 64, 8e-6, 0.02, 64, 1.0),
    ('p_zoom', 128, 8e-6, 0.05, 128, 10.0),
]


def _drive(rec, np, lum, prefix, force_never=False):
    from lumenairy.propagators import _bluestein as B
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    from lumenairy.propagators.mft import (angular_spectrum_propagate_mft,
                                           fraunhofer_propagate_mft,
                                           fresnel_propagate_mft,
                                           resample_field)

    saved = getattr(B, '_MFT_DIRECT_MAX_RATIO', None)
    if force_never and saved is not None:
        B._MFT_DIRECT_MAX_RATIO = B._MFT_DIRECT_NEVER
    try:
        kwsets = {'nokw': {}, 'wayback_bl': {'method': 'bluestein'},
                  'wayback_sep': {'method': 'separable'}}
        if prefix == 'never':
            kwsets = {'nokw': {}}
        for tag, ny, nx, my, mx in PRIM_SHAPES:
            E = _rand(np, ny, nx)
            alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
            for kwtag, kw in kwsets.items():
                for sgn in (-1, +1):
                    for sep in (False, True):
                        base = dict(sign=sgn, xp=np, fft2=_fft2,
                                    ifft2=_ifft2, separable=sep)
                        rec.record(
                            f"{prefix}/{kwtag}/plain/{tag}/s{sgn}/sep{int(sep)}",
                            B._bluestein_2d, E, alpha, alpha * 1.5, my, mx,
                            **base, **kw)
                        rec.record(
                            f"{prefix}/{kwtag}/centred/{tag}/s{sgn}/sep{int(sep)}",
                            B._bluestein_centred_2d, E, alpha, alpha * 1.5,
                            my, mx, **base, **kw)
                # one off-centre convention per shape, forward only
                rec.record(
                    f"{prefix}/{kwtag}/offcentre/{tag}",
                    B._bluestein_centred_2d, E, alpha, alpha * 1.5, my, mx,
                    n_centre_in_x=0.0, n_centre_in_y=ny / 3.0,
                    k_centre_out_x=mx / 2.0 - 0.37,
                    k_centre_out_y=my / 2.0,
                    sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, **kw)
            del E

        for tag, n_in, dx, z, n_out, zoom in PROP_SHAPES:
            E = _gauss(np, n_in, n_in, dx, 12.0 * dx)
            dx_out = (WL * z / (n_in * dx)) / zoom
            for kwtag, kw in kwsets.items():
                for fname, fn in (('fresnel', fresnel_propagate_mft),
                                  ('fraunhofer', fraunhofer_propagate_mft),
                                  ('asm', angular_spectrum_propagate_mft)):
                    rec.record(f"{prefix}/{kwtag}/{fname}/{tag}",
                               fn, E, z, WL, dx, dx_out, n_out, **kw)
                    rec.record(f"{prefix}/{kwtag}/{fname}/{tag}/offaxis",
                               fn, E, z, WL, dx, dx_out, n_out,
                               centre_out=(3.0 * dx_out, -2.0 * dx_out), **kw)
            del E

        # resample_field's chirp-Z leg takes no ``method`` of this vocabulary
        for tag, n_in, n_out in (('rs_down', 512, 16), ('rs_same', 64, 64),
                                 ('rs_up', 64, 256)):
            E = _gauss(np, n_in, n_in, 4e-6, 40e-6)
            rec.record(f"{prefix}/resample/chirpz/{tag}",
                       resample_field, E, 4e-6, 4e-6 * n_in / n_out,
                       n_out, method='chirpz')  # returns (E, dx)
            del E

        # the carrier readouts that drive _collins_transport
        from lumenairy.propagators.carrier import (
            carrier_referenced_exact_focus_readout)
        for tag, n_in, n_out, nf in (('cr_tiny', 256, 16, 2048),
                                     ('cr_wide', 128, 64, 512)):
            E = _gauss(np, n_in, n_in, 4e-6, 40e-6)
            rec.record(f"{prefix}/carrier/exact_focus/{tag}",
                       carrier_referenced_exact_focus_readout,
                       E, 2e-2, 2e-2, WL, 4e-6,
                       dx_out=4e-7, N_out=n_out, N_fine=nf,
                       on_replica='ignore', on_readout_window='ignore')
            del E
    finally:
        if force_never and saved is not None:
            B._MFT_DIRECT_MAX_RATIO = saved


def main(tree, out_path):
    lum = c4lib.anchor(tree)
    import numpy as np
    rec = c4lib.Recorder()
    _drive(rec, np, lum, 'live')
    from lumenairy.propagators import _bluestein as B
    if hasattr(B, '_MFT_DIRECT_MAX_RATIO'):
        _drive(rec, np, lum, 'never', force_never=True)
    # what the rule says about each fixture's shape, recorded beside the
    # digests so the ``live/nokw`` split can be CHECKED and not just counted
    rule_says = {}
    if hasattr(B, '_auto_selects_direct'):
        for tag, ny, nx, my, mx in PRIM_SHAPES:
            rule_says[f'plain/{tag}'] = bool(
                B._auto_selects_direct(ny, nx, my, mx))
        for tag, n_in, dx, z, n_out, zoom in PROP_SHAPES:
            rule_says[f'prop/{tag}'] = bool(
                B._auto_selects_direct(n_in, n_in, n_out, n_out))
    out = {'build': c4lib.build_tag(), 'tree': tree,
           'lumenairy_file': lum.__file__, 'version': lum.__version__,
           'python': sys.version.split()[0],
           'has_rule': hasattr(B, '_MFT_DIRECT_MAX_RATIO'),
           'ratio': float(getattr(B, '_MFT_DIRECT_MAX_RATIO', float('nan'))),
           'rule_says': rule_says,
           'keys': rec.keys, 'warned': rec.notes}
    c4lib.write_json(out, out_path)
    print(f"{len(rec.keys)} keys  ({c4lib.build_tag()}, {lum.__file__})")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

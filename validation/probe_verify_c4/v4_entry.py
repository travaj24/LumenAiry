"""VERIFY-WP-C4 claim 5b -- every public entry point that reaches the MFT
without exposing ``method=``, and whether its callers have a ONE-KEYWORD way
back to the previous bytes.

The campaign rule is that every public entry point a default flip MOVES has a
one-keyword way back.  A module-level constant (``_MFT_DIRECT_MAX_RATIO =
_MFT_DIRECT_NEVER``) is NOT one: it is private, it is process-wide, and a
caller who wants the previous bytes for ONE call cannot reach it without
monkey-patching a private name.

The entry points here come from an AST sweep of ``lumenairy/`` for calls to
``_bluestein_2d`` / ``_bluestein_centred_2d`` / ``_direct_matrix_2d`` and to
the three public MFT propagators, NOT from the branch's report.  Each is
driven at a shape the rule CAPTURES and (where the API allows) one it refuses,
on the branch and on a ``git archive`` of the base commit, and the digests are
compared by :mod:`v4_entry_compare`.

For each entry point the probe records THREE things:

``moves``
    does the no-keyword call differ base-to-branch at the captured shape?
``way_back_keyword``
    a keyword the CALLER can pass, on the entry point's OWN signature, that
    reproduces the base bytes.  ``None`` means there is none, which is the
    defect shape.
``stays``
    does the no-keyword call stay byte-identical at the refused shape?

    PYTHONPATH=<tree> python v4_entry.py <tree> OUT.json
"""
from __future__ import annotations

import hashlib
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
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return (np.exp(-r2 / w ** 2).astype(np.complex128)
            * np.exp(1j * 0.3 * rng.standard_normal((ny, nx))))


class Rec:
    def __init__(self):
        self.keys = {}
        self.notes = {}

    def record(self, key, fn, *a, **kw):
        import numpy as np
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                r = fn(*a, **kw)
            except Exception as exc:                     # noqa: BLE001
                self.keys[key] = f"RAISED:{type(exc).__name__}"
                self.notes[key] = str(exc)[:300]
                return
            arr = r
            while isinstance(arr, tuple):
                arr = arr[0]
            # CarrierField and friends are not arrays: np.asarray() on one
            # yields a 0-d OBJECT array whose bytes are a pointer, which
            # digests as noise.  Unwrap the field explicitly.
            for attr in ('envelope', 'field'):
                if not isinstance(arr, np.ndarray) and hasattr(arr, attr):
                    arr = getattr(arr, attr)
                    break
            arr = np.asarray(arr)
            if arr.dtype == object or arr.ndim == 0:
                raise AssertionError(
                    f"{key}: unwrapped to a {arr.dtype} / {arr.ndim}-d "
                    f"object -- the digest would be a pointer")
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        for ww in w:
            h.update(b'\x00W\x00')
            h.update(str(ww.message).encode('utf-8', 'replace'))
        self.keys[key] = h.hexdigest()
        if w:
            self.notes[key] = f"{len(w)} warning(s): " \
                              f"{str(w[0].message)[:160]}"


def drive(rec, np):
    from lumenairy.analysis.psf_mtf_otf import compute_psf
    from lumenairy.propagators.carrier import (
        carrier_referenced_exact_focus_readout,
        carrier_referenced_focus_readout)
    from lumenairy.propagators.dispatch import asm_propagate, propagate
    from lumenairy.propagators.mft import (angular_spectrum_propagate_mft,
                                           fraunhofer_propagate_mft,
                                           fresnel_propagate_mft,
                                           resample_field)
    from lumenairy.propagators.system import propagate_through_system

    E512 = gauss(np, 512, 512, 4e-6, 40e-6)
    pup = gauss(np, 512, 512, 4e-6, 400e-6)
    dxo = WL * 0.05 / (512 * 4e-6)

    # ---- the three public MFT propagators: method= IS on the signature ----
    for fname, fn in (('asm_mft', angular_spectrum_propagate_mft),
                      ('fresnel_mft', fresnel_propagate_mft),
                      ('fraunhofer_mft', fraunhofer_propagate_mft)):
        for shp, nout in (('captured', 16), ('refused', 128)):
            rec.record(f"{fname}/{shp}/nokw", fn, E512, 0.05, WL, 4e-6,
                       dxo, nout)
            rec.record(f"{fname}/{shp}/kw_bluestein", fn, E512, 0.05, WL,
                       4e-6, dxo, nout, method='bluestein')
            rec.record(f"{fname}/{shp}/kw_separable", fn, E512, 0.05, WL,
                       4e-6, dxo, nout, method='separable')

    # ---- compute_psf(method='mft'): `method` names the SAMPLER, not the
    # MFT route, so there is no route keyword on this signature -----------
    for shp, npsf in (('captured', 16), ('refused', 128)):
        rec.record(f"compute_psf/{shp}/nokw", compute_psf, pup, WL, 0.05,
                   4e-6, npsf, method='mft')

    # ---- resample_field(method='chirpz'): same shape of problem ----------
    for shp, nout in (('captured', 16), ('refused', 128)):
        rec.record(f"resample_field/{shp}/nokw", resample_field, E512, 4e-6,
                   4e-6 * 512 / nout, nout, method='chirpz')

    # ---- propagate(): `method` names the propagator FAMILY ---------------
    for shp, nout in (('captured', 16), ('refused', 128)):
        rec.record(f"propagate_asm/{shp}/nokw", propagate, E512, z=0.05,
                   wavelength=WL, dx=4e-6, method='asm',
                   output_grid=(nout, dxo), return_result=False)
        rec.record(f"propagate_fresnel/{shp}/nokw", propagate, E512, z=0.05,
                   wavelength=WL, dx=4e-6, method='fresnel',
                   output_grid=(nout, dxo), return_result=False)

    # ---- asm_propagate(): no `method` of its own, so **method_kwargs
    # carries one straight through -- a real one-keyword way back ---------
    for shp, nout in (('captured', 16), ('refused', 128)):
        rec.record(f"asm_propagate/{shp}/nokw", asm_propagate, E512, 0.05,
                   WL, 4e-6, output_dx=dxo, output_N=nout)
        rec.record(f"asm_propagate/{shp}/kw_separable", asm_propagate, E512,
                   0.05, WL, 4e-6, output_dx=dxo, output_N=nout,
                   method='separable')
        rec.record(f"asm_propagate/{shp}/kw_bluestein", asm_propagate, E512,
                   0.05, WL, 4e-6, output_dx=dxo, output_N=nout,
                   method='bluestein')

    # ---- the carrier readouts ------------------------------------------
    for shp, nout in (('captured', 16), ('refused', 128)):
        rec.record(f"carrier_focus_readout/{shp}/nokw",
                   carrier_referenced_focus_readout, E512, 2e-2, 2e-2, WL,
                   4e-6, dx_out=4e-7, N_out=nout, on_replica='ignore')
        rec.record(f"carrier_exact_focus_readout/{shp}/nokw",
                   carrier_referenced_exact_focus_readout, E512, 2e-2, 2e-2,
                   WL, 4e-6, dx_out=4e-7, N_out=nout, N_fine=2048,
                   on_replica='ignore', on_readout_window='ignore')

    # ---- re_reference (carrier_field.py): a top-level verb, and its
    # `_separable` flag is NOT a route keyword ----------------------------
    from lumenairy.propagators.carrier_field import (CarrierField,
                                                     CarrierSpec, FieldGrid,
                                                     re_reference)
    for shp, nout in (('captured', 16), ('refused', 128)):
        g_in = FieldGrid((512, 512), 4e-6)
        g_out = FieldGrid((nout, nout), 4e-6 * 512 / nout)
        s1 = CarrierSpec(R=-2e-2)
        s2 = CarrierSpec(R=-2.4e-2)
        f = CarrierField(E512.copy(), g_in, s1, WL)
        rec.record(f"re_reference/{shp}/nokw", re_reference, f, s2, g_out,
                   on_nyquist='ignore', on_window='ignore')

    # ---- propagate_through_system's fresnel leg: N_out is pinned to the
    # chain's own sample count, so the ratio is 1 and the rule cannot fire.
    # Recorded so "cannot fire" is a MEASUREMENT and not an assumption. ---
    E128 = gauss(np, 128, 128, 4e-6, 40e-6)
    rec.record("propagate_through_system/fresnel_leg",
               propagate_through_system, E128,
               [{'type': 'propagate', 'z': 0.02}], WL, 4e-6,
               method='fresnel', return_result=False)


def main(tree, out_path):
    lum = v4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    rec = Rec()
    drive(rec, np)
    rule = {}
    if hasattr(B, '_auto_selects_direct'):
        for tag, ny, nx, my, mx in (('captured_512_16', 512, 512, 16, 16),
                                    ('refused_512_128', 512, 512, 128, 128),
                                    ('psf_512_16', 512, 512, 16, 16),
                                    ('system_128_128', 128, 128, 128, 128)):
            rule[tag] = bool(B._auto_selects_direct(ny, nx, my, mx))
    out = {'build': v4lib.build_tag(), 'tree': tree,
           'lumenairy_file': lum.__file__, 'version': lum.__version__,
           'has_rule': hasattr(B, '_MFT_DIRECT_MAX_RATIO'),
           'ratio': float(getattr(B, '_MFT_DIRECT_MAX_RATIO', float('nan'))),
           'rule_says': rule, 'keys': rec.keys, 'notes': rec.notes}
    v4lib.write_json(out, out_path)
    print(f"{len(rec.keys)} keys ({v4lib.build_tag()}, {lum.__file__})")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

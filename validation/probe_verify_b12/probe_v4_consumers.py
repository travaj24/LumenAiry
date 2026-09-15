"""VERIFY-WP-B12 probe V4 -- the convention decision, re-measured.

**C1. Is GBD double-corrected?**  ``apply_real_lens_gbd(per_surface=True)`` on
a CURVED-last-surface prescription, run twice in one process -- as shipped,
and with both differential primitives wrapped so that ``reference`` is
DISCARDED.  The two fields must be BIT-identical (same SHA-256), because
``gbd.py`` never passes the keyword and the default did not move.  A
source-level check accompanies it: the ``_jac(...)`` call in
``apply_prescription_persurface_to_beamlets`` carries no ``reference``.

**C2. How wrong is ``gbd.py``'s own in-line sag copy?**  Its ``_Rl`` / ``_kl``
block reads ``radius`` and ``conic`` and nothing else.  Transcribed here and
differenced against the package's own shared ``_surface_sag_xy`` on four
last-surface classes: conic, even asphere, biconic, field-frame decentred.

**C3. What does that cost in a FIELD?**  Model-free: the in-line block's
error is a pure phase screen ``k * (sag_inline - sag_true) * sec`` on the exit
pupil, so it is applied to MY OWN exit-vertex boundary field and both are
propagated to the focus with MY OWN angular spectrum.  The fidelity between
them is what a consumer of that block loses -- no GBD internals, no
monkeypatching of library code.  GBD's shipped field is ALSO scored against
the same oracle on an aspheric-last and on a conic-last fixture, so the
in-line error can be placed against GBD's own floor.
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402
from probe_v3_field import _install_forced_surface, _remove_forced_surface  # noqa: E402


def gbd_inline_sag(surface, x, y):
    """``gbd.apply_prescription_persurface_to_beamlets``'s in-line copy,
    transcribed from the source (the ``_Rl`` / ``_kl`` block)."""
    _Rl = float(getattr(surface, 'radius', np.inf))
    _kl = float(getattr(surface, 'conic', 0.0) or 0.0)
    if np.isfinite(_Rl) and _Rl != 0.0:
        _cl = 1.0 / _Rl
        _r2 = x ** 2 + y ** 2
        return _cl * _r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
    return np.zeros_like(x)


def c1_source_check():
    import lumenairy.propagators.gbd as GB
    src = open(GB.__file__, encoding='utf-8', errors='replace').read()
    call = re.search(r'dt = _jac\((.*?)\)\n', src, re.S)
    body = call.group(1) if call else ''
    rec = dict(jac_call=' '.join(body.split()),
               passes_reference=('reference' in body),
               inline_block_reads=sorted(set(
                   re.findall(r"getattr\(surfs\[-1\], '(\w+)'", src))))
    print(f"  C1s gbd _jac call: {rec['jac_call']}")
    print(f"  C1s passes reference=: {rec['passes_reference']}; "
          f"in-line block reads {rec['inline_block_reads']}")
    return rec


def c1_gbd_bit_identity():
    from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
    out = {}
    for key in ('asph', 'menisc'):
        fx = C.fixture(key).coarse()
        kw = dict(prescription=fx.prescription(), wavelength=fx.lam,
                  dx=fx.dx, per_surface=True,
                  output_plane_distance=fx.best_focus())
        _remove_forced_surface()
        a = apply_real_lens_gbd(fx.E_in(), **kw)
        _install_forced_surface()
        b = apply_real_lens_gbd(fx.E_in(), **kw)
        _remove_forced_surface()
        out[key] = dict(sha_native=C.sha(a), sha_forced=C.sha(b),
                        bit_identical=bool(np.array_equal(a, b)),
                        max_abs_diff=float(np.max(np.abs(a - b))))
        print(f"  C1 GBD [{key}] curved last surface, field bit-identical "
              f"native vs forced-'surface': {out[key]['bit_identical']} "
              f"({out[key]['sha_native'][:16]}...)", flush=True)
    return out


def c2_sag_classes():
    from lumenairy.raytrace import Surface
    from lumenairy.raytrace.surface import _surface_sag_xy
    lam = 780e-9
    h = 128e-6
    x = np.linspace(-h, h, 401)
    y = np.zeros_like(x)
    ydiag = np.linspace(-h, h, 401) * 0.6
    cases = {
        'conic': (Surface(radius=-0.911e-3, conic=-0.60), x, y),
        'even_asphere': (Surface(radius=-0.911e-3, conic=-0.60,
                                 aspheric_coeffs={4: 2.0e9, 6: -4.0e16}),
                         x, y),
        'biconic': (Surface(radius=-0.858e-3, radius_y=-1.17e-3, conic=0.0,
                            conic_y=0.0), x, ydiag),
        'field_decentred': (Surface(radius=-0.911e-3, conic=0.0,
                                    field_decenter=(20e-6, 0.0)), x, y),
    }
    out = {}
    for nm, (surf, xx, yy) in cases.items():
        true = np.asarray(_surface_sag_xy(xx, yy, surf), float)
        inline = gbd_inline_sag(surf, xx, yy)
        err = np.abs(inline - true)
        i = int(np.nanargmax(err))
        rec = dict(max_abs_err_m=float(err[i]),
                   max_err_waves=float(err[i] / lam),
                   true_sag_at_worst=float(true[i]),
                   frac_of_sag=float(err[i] / max(abs(true[i]), 1e-300)))
        if nm in ('conic', 'even_asphere'):
            mine = C.sag(xx, yy, dict(zv=0.0, Rx=surf.radius, Ry=None,
                                      kx=surf.conic,
                                      ax=surf.aspheric_coeffs, ay=None))
            rec['my_kernel_vs_library'] = float(np.nanmax(np.abs(mine - true)))
        out[nm] = rec
        print(f'  C2 {nm:16s} in-line vs shared sag: '
              f"{rec['max_abs_err_m']:.4e} m = {rec['max_err_waves']:.3f} "
              f"waves @780nm ({rec['frac_of_sag'] * 100:.1f} % of the sag)")
    return out


def c3_field_cost():
    """The in-line block's sag error as a pure exit-pupil phase screen, on MY
    exit field, propagated by MY angular spectrum."""
    from lumenairy.raytrace import surfaces_from_prescription
    out = {}
    for key in ('asph', 'bicon', 'menisc'):
        fx = C.fixture(key)
        zf = fx.best_focus()
        E, info = fx.exit_field(refine=4)
        X, Y = fx.grid(4)
        last_lib = surfaces_from_prescription(fx.prescription())[-1]
        last_mine = fx.oracle_surfaces()[-1]
        true = C.sag(X, Y, last_mine)
        inline = gbd_inline_sag(last_lib, X, Y)
        # the ray leaves at slope ~ r/f, so sec is close to 1; use the exact
        # local sec from the exit map's own direction cosines (r / f).
        sec = np.sqrt(1.0 + (np.sqrt(X ** 2 + Y ** 2) / zf) ** 2)
        d = (inline - true) * sec
        k = 2.0 * np.pi / fx.lam
        Ep = E * np.exp(1j * k * np.nan_to_num(d))
        a = C.asm_propagate(E, fx.dx / 4, fx.dx / 4, fx.lam, zf)[::4, ::4]
        b = C.asm_propagate(Ep, fx.dx / 4, fx.dx / 4, fx.lam, zf)[::4, ::4]
        m = np.isfinite(d) & (np.abs(E) > 0)
        out[key] = dict(
            max_phase_err_waves=float(np.nanmax(np.abs(d[m])) / fx.lam),
            rms_phase_err_waves=float(np.sqrt(np.nanmean(d[m] ** 2))
                                      / fx.lam),
            fidelity_true_vs_inline=C.fidelity(b, a),
            oracle_mode=info['mode'])
        print(f"  C3 [{key:8s}] in-line sag screen: "
              f"{out[key]['max_phase_err_waves']:.3f} waves peak / "
              f"{out[key]['rms_phase_err_waves']:.3f} rms  ->  focal-plane "
              f"fidelity {out[key]['fidelity_true_vs_inline']:.6f}")
    return out


def c3b_gbd_vs_oracle():
    """GBD's own shipped fidelity against MY oracle -- aspheric-last (where
    the in-line copy is wrong) against conic-last (where it is exact)."""
    from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
    out = {}
    for key in ('asph', 'menisc', 'doublet'):
        fx = C.fixture(key).coarse()
        zf = fx.best_focus()
        O, _ = fx.oracle_field(zf, refine=4)
        t0 = time.time()
        E = apply_real_lens_gbd(fx.E_in(), prescription=fx.prescription(),
                                wavelength=fx.lam, dx=fx.dx,
                                per_surface=True, output_plane_distance=zf)
        out[key] = dict(fidelity=C.fidelity(E, O), power=C.power_ratio(E, O),
                        seconds=time.time() - t0,
                        last_surface_class=('even_asphere' if key == 'asph'
                                            else 'conic'))
        print(f"  C3b GBD [{key:8s}] vs my oracle at focus: "
              f"{out[key]['fidelity']:.6f} (P {out[key]['power']:.3f}, "
              f"{out[key]['last_surface_class']} last surface, "
              f"N={fx.N} dx={fx.dx:.2e})", flush=True)
    return out


def main():
    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__), flush=True)
    out = {'env': C.env_block()}
    out['c1_source'] = c1_source_check()
    out['c2_sag_classes'] = c2_sag_classes()
    out['c3_field_cost'] = c3_field_cost()
    out['c1_gbd_bit_identity'] = c1_gbd_bit_identity()
    out['c3b_gbd_vs_oracle'] = c3b_gbd_vs_oracle()
    C.dump(out, 'probe_v4_consumers')


if __name__ == '__main__':
    main()

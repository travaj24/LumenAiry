# ABLATION driver for design 121's LAST post-DOE group
# (docs/audits/ABLATE_LAST_GROUP_2026_07_30.md).
#
# The scope doc (SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30) localises the
# whole per-order loss to ONE ``apply_real_lens_traced`` call -- the chain's
# sixth (last) element pass -- and leaves three constructions untested: the
# Newton seed/bracket, the entrance-pullback DOMAIN, and
# ``preserve_input_phase='remap'``'s resampling.  All three are properties of
# the element's NUMERICAL FRAME, which is built symmetrically about the OPTICAL
# AXIS while the beam sits 1.079 amplitude radii off it.
#
# METHOD.  ``lumenairy.elements.apply_real_lens_traced`` is monkeypatched (the
# chain resolves it from the module at call time) and the call whose
# prescription is the LAST group is intercepted.  The scoring path is
# ``hybrid_localize_121.main()`` at ``NMIN=NMAX=6`` unchanged, so every EE3 here
# is directly comparable with the scope doc's knockout table (its own baseline
# at NOUT=61 DXO=0.4 NL=121 is EE3 66.24 % at order (-4,-2)).
#
# ---------------------------------------------------------------------------
# MODE=recentre[:f] -- the Option-A CHEAP PROBE (experiment 3).
#
# A lateral translation of BOTH the beam and the group is an EXACT symmetry of
# the sequential trace, so it changes NO physics -- only which part of the
# element's axis-centred numerical frame the beam occupies.  It is applied as
#
#   * the input field is shifted by an INTEGER number of pixels (a pure array
#     copy: no interpolation, no resampling of the phase, no aliasing -- this
#     is why the shift is integer and not the sub-pixel Fourier shift the chain
#     itself uses).  The shift is NON-periodic (zero-filled), so nothing wraps;
#     the discarded power is measured and printed.
#   * every surface of the group gets the matching field-frame ``decenter``
#     (validated in ``ablate_recon.py``: exit positions translate by t to
#     1.3e-18 m and the OPL is invariant to 8e-12 waves).
#   * the ``TiltedCarrier``'s ``(x0, y0)`` moves with them.
#   * the returned exit field is shifted back by the same integer vector.
#
# ``f`` (default 1.0) is the FRACTION of the chief-ray decentre removed, so
# f = 0 reproduces the shipped call and f = 1 puts the beam on the grid centre.
# The residual sub-pixel decentre is printed.
#
# Other MODEs
#   base           no patch (shipped)
#   hardmask       off-centre fit uses the CONCENTRIC hard-NaN mask instead of
#                  the D1 weights (isolates the fit branch from the frame)
#   seed=...       Newton initial-guess ablations (needs the temporary
#                  _lens_traced.py edit; see ablate_newton_seed_121.py)
#
# Env: ORD, MODE, GRP (group index to transform, default last), plus every
# hybrid_localize_121 knob (NOUT, DXO, NL, RN, RS, UP, TKW, NMIN, NMAX).
import io
import os
import re
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C                                        # noqa: E402
import lumenairy.elements as _EL                                # noqa: E402
import lumenairy.elements._lens_traced as _LT                   # noqa: E402

_REAL = _EL.apply_real_lens_traced


def _int_shift(a, sy, sx):
    """``out[i, j] = a[i - sy, j - sx]`` with ZERO fill (non-periodic).

    Integer pixel shift == exact array copy: the sample VALUES are untouched,
    so no phase is resampled and nothing can alias.  Returns the shifted array
    and the fraction of |a|^2 that fell off the grid."""
    a = np.asarray(a)
    out = np.zeros_like(a)
    n0, n1 = a.shape[-2], a.shape[-1]
    d0s, d0e = max(sy, 0), min(n0 + sy, n0)
    d1s, d1e = max(sx, 0), min(n1 + sx, n1)
    s0s, s1s = d0s - sy, d1s - sx
    out[d0s:d0e, d1s:d1e] = a[s0s:s0s + (d0e - d0s), s1s:s1s + (d1e - d1s)]
    p_all = float(np.sum(np.abs(a) ** 2))
    p_kep = float(np.sum(np.abs(out) ** 2))
    return out, (1.0 - p_kep / p_all if p_all else 0.0)


def make_patch(target_name, mode, log):
    frac = 1.0
    if mode.startswith('recentre'):
        if ':' in mode:
            frac = float(mode.split(':', 1)[1])
        mode = 'recentre'

    def patched(E_in, *, prescription, wavelength, dx, **kw):
        if prescription.get('name') != target_name:
            return _REAL(E_in, prescription=prescription,
                         wavelength=wavelength, dx=dx, **kw)
        car = kw.get('carrier')
        x0 = float(getattr(car, 'x0', 0.0))
        y0 = float(getattr(car, 'y0', 0.0))
        if '_exit_na_out' not in kw:
            _na = {}
            kw['_exit_na_out'] = _na
        else:
            _na = kw['_exit_na_out']
        log.append(f"[patch] hand-off |E_in| second-moment radius about the "
                   f"carrier centre: "
                   f"{_LT._input_beam_amp_radius(E_in, dx, dx, centre=(x0, y0)) * 1e3:.4f}"
                   f" mm; about the ORIGIN "
                   f"{_LT._input_beam_amp_radius(E_in, dx, dx, centre=None) * 1e3:.4f}"
                   f" mm")
        if mode == 'base':
            log.append(f"[patch] {target_name}: PASS-THROUGH, carrier centre "
                       f"({x0 * 1e3:+.4f}, {y0 * 1e3:+.4f}) mm")
            _r = _REAL(E_in, prescription=prescription,
                       wavelength=wavelength, dx=dx, **kw)
            log.append(f"[patch] exit-NA diag {_na}")
            return _r
        if mode == 'hardmask':
            _old = _LT._FIT_DISC_OUTSIDE_WEIGHT_REL
            _LT._FIT_DISC_OUTSIDE_WEIGHT_REL = 0.0
            log.append(f"[patch] {target_name}: off-centre fit forced to the "
                       f"CONCENTRIC hard-NaN mask")
            try:
                return _REAL(E_in, prescription=prescription,
                             wavelength=wavelength, dx=dx, **kw)
            finally:
                _LT._FIT_DISC_OUTSIDE_WEIGHT_REL = _old
        if mode != 'recentre':
            raise ValueError(f"unknown MODE {mode!r}")
        # ---- rigid translation of (beam + group) by t = -frac * (x0, y0) ----
        npx = int(round(-frac * x0 / dx))
        npy = int(round(-frac * y0 / dx))
        tx, ty = npx * dx, npy * dx
        E2, lost = _int_shift(E_in, npy, npx)
        presc2 = dict(prescription)
        presc2['surfaces'] = [dict(s, decenter=(tx, ty))
                              for s in prescription['surfaces']]
        kw2 = dict(kw)
        if car is not None:
            kw2['carrier'] = _LT.TiltedCarrier(car.R, car.L, car.M,
                                               x0 + tx, y0 + ty)
        log.append(
            f"[patch] {target_name}: RIGID TRANSLATION frac={frac} "
            f"t = ({tx * 1e3:+.4f}, {ty * 1e3:+.4f}) mm = ({npx:+d}, {npy:+d}) "
            f"px;  carrier centre ({x0 * 1e3:+.4f}, {y0 * 1e3:+.4f}) -> "
            f"({(x0 + tx) * 1e3:+.4f}, {(y0 + ty) * 1e3:+.4f}) mm;  "
            f"power discarded by the non-periodic shift {lost:.3e}")
        out = _REAL(E2, prescription=presc2, wavelength=wavelength, dx=dx,
                    **kw2)
        back, lost2 = _int_shift(np.asarray(out), -npy, -npx)
        log.append(f"[patch] exit field shifted back; power discarded "
                   f"{lost2:.3e}; exit-NA diag {_na}")
        return back

    return patched


def main():
    mode = os.environ.get('MODE', 'base')
    os.environ.setdefault('NMIN', '6')
    os.environ.setdefault('NMAX', '6')
    os.environ.setdefault('NOUT', '61')
    os.environ.setdefault('DXO', '0.4')
    os.environ.setdefault('NL', '121')
    _pre, post, _g, _p = C.geometry()
    gi = int(os.environ.get('GRP', str(len(post) - 1)))
    target = post[gi]['prescription'].get('name')
    log = []
    if mode != 'base':
        _EL.apply_real_lens_traced = make_patch(target, mode, log)
    else:
        _EL.apply_real_lens_traced = make_patch(target, 'base', log)
    import hybrid_localize_121 as H
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            H.main()
    finally:
        _EL.apply_real_lens_traced = _REAL
    txt = buf.getvalue()
    print(txt)
    for line in log:
        print(line)
    ee = re.findall(r"EE3\s+([0-9.]+)\s+EE6\s+([0-9.]+)\s+EE12\s+([0-9.]+)",
                    txt)
    fw = re.findall(r"FWHM\s+([0-9.]+)\s+um", txt)
    print(f"RESULT  ORD={os.environ.get('ORD', '-4,-2')}  MODE={mode}  "
          f"GRP={gi} ({target})  "
          + "  ".join(f"EE3={a} EE6={b} EE12={c} FWHM={f}"
                      for (a, b, c), f in zip(ee, fw)))


if __name__ == '__main__':
    sys.exit(main())

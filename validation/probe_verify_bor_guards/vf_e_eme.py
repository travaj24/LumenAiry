"""TASK F / probe E -- RE-MEASURE every asserted quantity in
tests/unit/test_fix_eme_branch_cut.py, and probe the two premises the file
rests on (the narrowed window is still DECISIVE; the fixture finds >= 10
modes).

Usage:  python vf_e_eme.py <pre|post> <tag> [full]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
FULL = len(sys.argv) > 3 and sys.argv[3] == "full"
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.eme import eme_2d  # noqa: E402

POST = BUILD == "post"
_eb = None
if POST:
    from lumenairy.elements.eme import _branch as _eb  # noqa: E402

_NX, _LX, _LY, _K0, _KY0 = 96, 1.0, 1.0, 20.0 * np.pi, 0.37
_WINDOW, _NSCAN = (26055.8, 28500.0), 60
_WIN_FULL, _NSCAN_FULL = (26055.8, 35530.6), 300


def _strips(imag):
    hi = 12.0 + 1j * imag
    e1 = np.full(_NX, 2.25 + 0j)
    e1[_NX // 2:] = hi
    e2 = np.full(_NX, 2.25 + 0j)
    e2[_NX // 4:3 * _NX // 4] = hi
    return [(e1, 0.5 * _LY), (e2, 0.5 * _LY)]


def _modes(imag, window=_WINDOW, nscan=_NSCAN):
    return np.asarray(eme_2d.layer_modes(
        _strips(imag), _LX, _NX, _LY, _K0, window,
        kx0=0.0, ky0=_KY0, n_scan=nscan))


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG, window=_WINDOW, nscan=_NSCAN)

# --- gate: test_no_strip_mode_flips_under_an_infinitesimal_loss -----------
def _fwd(imag):
    lam = np.asarray(eme_2d.strip_x_modes(
        _strips(imag)[0][0], _LX, _NX, _K0)[0], dtype=complex)
    lam = lam[np.lexsort((lam.imag, lam.real))]
    return eme_2d._ky_forward(lam, 0.0)


real, tiny = _fwd(0.0), _fwd(1e-30)
dif = np.abs(real - tiny)
tolv = 1e-6 * np.maximum(np.abs(real), 1.0)
flipped = int(np.sum(dif > tolv))
# how far is the WORST non-flipped mode from the 1e-6 relative tolerance?
ratio = dif / tolv
out["strip_flip"] = dict(
    n=int(real.size), flipped=flipped,
    worst_ratio_to_tol=float(np.max(ratio)) if real.size else None,
    worst_abs_diff=float(np.max(dif)) if real.size else None,
    all_imag_nonneg=bool(np.all(tiny.imag >= 0.0)),
    min_imag=float(np.min(tiny.imag)), max_imag=float(np.max(tiny.imag)),
    hash_real=_vh.hash_arrays(real), hash_tiny=_vh.hash_arrays(tiny))

# --- gate: test_mode_count_is_build_independent_under_an_infinitesimal_loss
with _vh.timed("layer_modes narrowed x2"):
    mr, mt = _modes(0.0), _modes(1e-30)
same = mr.size == mt.size
pos = (float(np.max(np.abs(np.sort(mr) - np.sort(mt)))) if same and mr.size
       else float("nan"))
out["layer_modes_narrowed"] = dict(
    n_real=int(mr.size), n_tiny=int(mt.size), same_count=bool(same),
    worst_dqz2=pos, atol_bar=1e-6, bar_n_modes=10,
    scan_resolution=(_WINDOW[1] - _WINDOW[0]) / _NSCAN,
    real=[float(x) for x in np.sort(mr)][:80],
    tiny=[float(x) for x in np.sort(mt)][:80])

# --- the file's DECISIVENESS premise: restore the pre-5.45.1 exact-zero pin
#     and re-run the narrowed window.  Done WITHOUT editing the library, by
#     monkeypatching eme_2d._ky_forward to the shipped-pre body.
def _ky_forward_prefix(lam, qz2):
    ky = np.sqrt(np.asarray(lam, dtype=complex) - qz2)
    return np.where(ky.imag < 0.0, -ky, ky)


if POST:
    orig = eme_2d._ky_forward
    eme_2d._ky_forward = _ky_forward_prefix
    try:
        with _vh.timed("layer_modes narrowed x2 with the PRE-FIX pin"):
            pr, pt = _modes(0.0), _modes(1e-30)
        lam = np.asarray(eme_2d.strip_x_modes(
            _strips(1e-30)[0][0], _LX, _NX, _K0)[0], dtype=complex)
        lam = lam[np.lexsort((lam.imag, lam.real))]
        pre_fwd = _ky_forward_prefix(lam, 0.0)
        lam0 = np.asarray(eme_2d.strip_x_modes(
            _strips(0.0)[0][0], _LX, _NX, _K0)[0], dtype=complex)
        lam0 = lam0[np.lexsort((lam0.imag, lam0.real))]
        pre_fwd0 = _ky_forward_prefix(lam0, 0.0)
        pre_flip = int(np.sum(np.abs(pre_fwd - pre_fwd0)
                              > 1e-6 * np.maximum(np.abs(pre_fwd0), 1.0)))
    finally:
        eme_2d._ky_forward = orig
    out["prefix_pin_narrowed"] = dict(
        n_real=int(pr.size), n_tiny=int(pt.size),
        decisive=bool(pr.size != pt.size), strip_flipped=pre_flip,
        note="pre-5.45.1 exact-zero pin restored by monkeypatch; the file's "
             "docstring claims 16 vs 18 here")

# --- gate: test_no_forward_root_of_a_propagating_strip_mode_is_negated ----
lam = np.array([100.0 - 1e-13j, 400.0 + 3e-14j, 2500.0 - 5e-12j])
ky = eme_2d._ky_forward(lam, 0.0)
out["synthetic_roots"] = dict(
    ky_real=[float(x) for x in ky.real], ky_imag=[float(x) for x in ky.imag],
    all_re_pos=bool(np.all(ky.real > 0.0)),
    all_im_nonneg=bool(np.all(ky.imag >= 0.0)),
    band=(float(_eb.cut_band(np.sqrt(lam - 0.0), xp=np)) if POST else None),
    worst_abs_imag=float(np.max(np.abs(np.sqrt(lam).imag))))

# --- gate: test_the_band_is_relative_to_the_spectrum_and_floored_at_one ---
if POST:
    out["cut_band"] = dict(
        at_1e_minus_6=float(_eb.cut_band(np.array([1e-6 + 0j]), xp=np)),
        at_1e4=float(_eb.cut_band(np.array([1e4 + 0j]), xp=np)),
        const=_eb._EME_CUT_BAND_REL,
        live_low=float(_eb.forward_decaying_root(
            np.array([5.0 - 1e-3j]), xp=np, band=1e-9)[0].real),
        live_high=float(_eb.forward_decaying_root(
            np.array([5.0 - 1e-3j]), xp=np, band=1e-2)[0].real))
    z = np.array([2.0 - 0.5j, 3.0 + 0.5j])
    o = _eb.forward_decaying_root(z, xp=np)
    out["lossy_root"] = dict(out_real=[float(x) for x in o.real],
                             out_imag=[float(x) for x in o.imag])

# --- the SCOPING's full window, only when asked (167 s) -------------------
if FULL:
    with _vh.timed("layer_modes FULL x2"):
        fr = _modes(0.0, _WIN_FULL, _NSCAN_FULL)
        ft = _modes(1e-30, _WIN_FULL, _NSCAN_FULL)
    out["layer_modes_full"] = dict(n_real=int(fr.size), n_tiny=int(ft.size),
                                   first_real=float(np.min(fr)) if fr.size else None,
                                   first_tiny=float(np.min(ft)) if ft.size else None)

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_e_eme_%s_%s.json" % (BUILD, TAG)), out)
print("strip flip:", out["strip_flip"]["flipped"], "of",
      out["strip_flip"]["n"], " worst/tol=%.3e"
      % (out["strip_flip"]["worst_ratio_to_tol"],))
print("layer_modes narrowed: %d vs %d  worst dqz2=%.4e (atol 1e-6)"
      % (out["layer_modes_narrowed"]["n_real"],
         out["layer_modes_narrowed"]["n_tiny"],
         out["layer_modes_narrowed"]["worst_dqz2"]))
if POST:
    print("PRE-FIX pin restored on the narrowed window: %d vs %d (decisive=%s)"
          % (out["prefix_pin_narrowed"]["n_real"],
             out["prefix_pin_narrowed"]["n_tiny"],
             out["prefix_pin_narrowed"]["decisive"]))

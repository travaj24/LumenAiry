"""VERIFY-WP-C2 item 8 -- the two restated d3 guard bars, asked the two
questions the restatement does not answer itself:

1. Is the FLOOR each bar is measured against stable, or is it itself a
   draw?  Both restatements measure the floor from ONE perturbation --
   ``np.nextafter(..., +inf)`` on the input envelope.  A floor that moves
   by more than the bar's own multiplier under a DIFFERENT one-ULP
   perturbation (down instead of up; the real part only; a single
   element) is not a floor, and a multiplier of 3 sitting on it is inside
   its own noise.
2. Is the claim two-sided?  Both arms assert only "the effect exceeds the
   floor".  Neither asserts the converse -- that with the degree HELD
   FIXED the same comparison lands AT the floor -- which is what would
   make the comparator a decision rather than a threshold.

Everything is imported from the shipped test module, so the quantities are
the ones the tests read, not a re-implementation.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_d3_bars.py OUT.json``
"""
import importlib.util
import json
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _load_tests():
    path = os.path.join(_ROOT, 'tests', 'unit', 'test_niche_d3_guards.py')
    spec = importlib.util.spec_from_file_location('_d3', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_d3'] = mod
    spec.loader.exec_module(mod)
    return mod


def _nudge(G, kind):
    """One-ULP perturbations of the input envelope, several directions."""
    G = np.asarray(G)
    cx = np.iscomplexobj(G)
    if kind == 'up':                       # what the shipped floor uses
        return (np.nextafter(G.real, np.inf)
                + 1j * np.nextafter(G.imag, np.inf)) if cx \
            else np.nextafter(G, np.inf)
    if kind == 'down':
        return (np.nextafter(G.real, -np.inf)
                + 1j * np.nextafter(G.imag, -np.inf)) if cx \
            else np.nextafter(G, -np.inf)
    if kind == 'real_only':
        return (np.nextafter(G.real, np.inf) + 1j * G.imag) if cx \
            else np.nextafter(G, np.inf)
    if kind == 'one_element':
        H = G.copy()
        flat = H.reshape(-1)
        i = int(np.argmax(np.abs(flat)))
        flat[i] = (np.nextafter(flat[i].real, np.inf)
                   + 1j * flat[i].imag) if cx \
            else np.nextafter(flat[i], np.inf)
        return H
    raise ValueError(kind)


def main(out_path):
    d3 = _load_tests()
    lt = d3._lens_traced
    res = dict(meta=dict(python=sys.version.split()[0],
                         numpy=np.__version__, lumenairy=la.__version__,
                         file=la.__file__))

    TILT = 0.023

    # ---------- arm 2: the multiplexed `moved` bar -----------------
    def _mux(field, degree, launch):
        deg0 = lt._REMAP_RESID_EIKONAL_DEGREE
        lch0 = lt.REMAP_STATIONARY_PHASE_LAUNCH
        lt._REMAP_RESID_EIKONAL_DEGREE = degree
        lt.REMAP_STATIONARY_PHASE_LAUNCH = launch
        try:
            return d3._chain(field, quiet=True, focus_readout=None,
                             on_multi_congruence='ignore').field
        finally:
            lt._REMAP_RESID_EIKONAL_DEGREE = deg0
            lt.REMAP_STATIONARY_PHASE_LAUNCH = lch0

    X, Y = d3._grid(d3._CN, d3._CDX)
    G0 = np.asarray(d3._gauss(d3._CN, d3._CDX, d3._CW))

    def _fan(G):
        return sum(G * np.exp(1j * d3._K0 * TILT * (sx * X + sy * Y))
                   for sx in (-1, 1) for sy in (-1, 1))

    on6 = _mux(_fan(G0), 6, True)
    on4 = _mux(_fan(G0), 4, True)
    ref = float(np.linalg.norm(on6))
    moved = float(np.linalg.norm(on6 - on4)) / ref

    floors = {}
    for kind in ('up', 'down', 'real_only', 'one_element'):
        f = _mux(_fan(_nudge(G0, kind)), 6, True)
        floors[kind] = float(np.linalg.norm(f - on6)) / ref
    fv = [v for v in floors.values() if v > 0]
    arm2 = dict(moved=moved, floors=floors,
                shipped_floor=floors['up'],
                floor_min=min(fv), floor_max=max(fv),
                floor_spread_ratio=max(fv) / min(fv),
                margin_at_shipped_floor=moved / floors['up'],
                margin_at_worst_floor=moved / max(fv),
                bar=3.0,
                passes_at_shipped_floor=moved > 3.0 * floors['up'],
                passes_at_worst_floor=moved > 3.0 * max(fv),
                headroom_over_the_bar=(moved / floors['up']) / 3.0,
                # TWO-SIDED control: degree held FIXED, two independent
                # runs -- the comparison that ought to land AT the floor
                same_degree_twice=float(
                    np.linalg.norm(_mux(_fan(G0), 6, True) - on6)) / ref)
    res['arm2_moved'] = arm2

    # ---------- arm 1: the linearity-error `degree_effect` bar -----
    bad6 = d3._linearity_error(TILT)
    lt._REMAP_RESID_EIKONAL_DEGREE = 4
    try:
        bad4 = d3._linearity_error(TILT)
    finally:
        lt._REMAP_RESID_EIKONAL_DEGREE = 6
    good6 = None
    degree_effect = abs(bad4 - bad6) / max(bad6, 1e-300)
    shipped_noise = abs(d3._linearity_error(TILT, nudge=True) - bad6) \
        / max(bad6, 1e-300)
    arm1 = dict(bad6=bad6, bad4=bad4, good6=good6,
                bad4_over_bad6=bad4 / bad6,
                degree_effect=degree_effect,
                shipped_floor=shipped_noise,
                margin=degree_effect / max(shipped_noise, 1e-300),
                bar=10.0,
                passes=degree_effect > 10.0 * shipped_noise,
                headroom_over_the_bar=(
                    degree_effect / max(shipped_noise, 1e-300)) / 10.0,
                same_degree_twice=abs(d3._linearity_error(TILT) - bad6)
                / max(bad6, 1e-300))
    res['arm1_degree_effect'] = arm1

    res['verdict'] = dict(
        floors_are_measured_in_process=True,
        multipliers_are_chosen_not_derived=True,
        arm2_floor_spread_ratio=arm2['floor_spread_ratio'],
        arm2_bar=3.0,
        arm2_bar_inside_its_own_floor_spread=(
            arm2['floor_spread_ratio'] >= 3.0),
        arm2_headroom=arm2['headroom_over_the_bar'],
        arm1_headroom=arm1['headroom_over_the_bar'],
        two_sided=False)

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])

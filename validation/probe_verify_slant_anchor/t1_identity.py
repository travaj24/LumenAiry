"""TASK 1 -- the BIT-IDENTITY census, on this verification's own fixtures.

Runs every fixture in :mod:`fixtures` plus seven 2-D / single-layer entries and
writes one sha256 digest per SURFACE.  ``summarize.py identity`` then diffs the
digests arm against arm on the SAME build; a digest that moves is a byte that
moved, and the claim under test is that the ONLY movers are TRANSMISSION-side
surfaces on SHEARED fixtures.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.environ["LUM_ARM_TREE"]) if os.environ.get(
    "LUM_ARM_TREE") else None
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402
from fixtures import FIXTURES_1D, run_fixture  # noqa: E402

UM = 1e-6


def _cell(sx=5, sy=4, lo=1.25, hi=3.15, const=False):
    """An x-ASYMMETRIC, y-varying ``(sx, sy, 3, 3)`` in-plane tensor cell."""
    c = np.zeros((sx, sy, 3, 3), dtype=complex)
    for i in range(sx):
        for j in range(sy):
            if const:
                v = 2.20
            else:
                v = lo + (hi - lo) * ((i * 3 + j * 2) % 7) / 6.0
            c[i, j] = np.eye(3) * v
            if not const:
                c[i, j, 0, 1] = c[i, j, 1, 0] = 0.06 * ((i + j) % 3)
    return c


def twod_surfaces():
    """The 2-D / single-layer entries, as ``name -> {surface: digest}``."""
    from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure, pmm_jones_2d
    out = {}

    def _rec(name, fn):
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                d = fn()
            except Exception as exc:                            # noqa: BLE001
                d = {"all": f"raise:{type(exc).__name__}",
                     "_err": str(exc)[:200]}
        d["_secs"] = round(time.time() - t, 3)
        out[name] = d

    kw = dict(period_x=0.85 * UM, period_y=0.85 * UM,
              n_substrate=1.5, n_superstrate=1.0, depth=0.42 * UM,
              wavelength=0.58 * UM, n_orders=3, degree=5)

    def _j2d(slant, const=False, theta=np.deg2rad(25.0), phi=0.0):
        def go():
            o, R, T, J = pmm_jones_2d(eps_tensor_cell=_cell(const=const),
                                      theta=theta, phi=phi, slant=slant, **kw)
            return {"orders": _lib.sha(np.asarray(o)),
                    "R": _lib.sha(np.asarray(R)), "T": _lib.sha(np.asarray(T)),
                    "Jrefl": _lib.sha(np.asarray(J))}
        return go

    _rec("j2d_vert_ob25", _j2d(None))
    _rec("j2d_slant_ob25", _j2d((0.4, 0.0)))
    _rec("j2d_slant_conical", _j2d((0.4, 0.0), phi=np.deg2rad(31.0)))
    _rec("j2d_slant_const_ob25", _j2d((0.4, 0.0), const=True))
    _rec("j2d_vert_const_ob25", _j2d(None, const=True))

    def _hyb(slant):
        def go():
            s = PMM2DStackHybrid(0.85 * UM, n_substrate=1.5, degree=5,
                                 n_orders=3)
            s.add_layer(0.42 * UM, eps_tensor_cell=_cell(), slant=slant)
            s.set_source(0.58 * UM, theta=np.deg2rad(25.0))
            o, R, T, J = s.solve()
            d = {"orders": _lib.sha(np.asarray(o)),
                 "R": _lib.sha(np.asarray(R)), "T": _lib.sha(np.asarray(T)),
                 "Jrefl": _lib.sha(np.asarray(J))}
            try:
                d["Jtrans"] = _lib.sha(np.asarray(s.jones_transmission()))
                a = s.per_order_amplitudes("transmission")
                d["perT_Ex"] = _lib.sha(np.asarray(a["Ex"]))
            except Exception as exc:                            # noqa: BLE001
                d["Jtrans"] = d["perT_Ex"] = f"raise:{type(exc).__name__}"
            return d
        return go

    _rec("hyb_vert_ob25", _hyb(None))
    _rec("hyb_slant_ob25", _hyb((0.4, 0.0)))

    def _pure(slant):
        def go():
            s = PMM2DStackPure(0.85 * UM, n_substrate=1.5, n_modes=3,
                               n_orders=2)
            e = np.full((4, 4), 1.30)
            e[1:3, :] = 3.05
            s.add_layer(0.42 * UM, eps_cell=e, slant=slant)
            s.set_source(0.58 * UM, theta=np.deg2rad(25.0))
            o, R, T, J = s.solve()
            d = {"orders": _lib.sha(np.asarray(o)),
                 "R": _lib.sha(np.asarray(R)), "T": _lib.sha(np.asarray(T)),
                 "Jrefl": _lib.sha(np.asarray(J))}
            try:
                d["Jtrans"] = _lib.sha(np.asarray(s.jones_transmission()))
            except Exception as exc:                            # noqa: BLE001
                d["Jtrans"] = f"raise:{type(exc).__name__}"
            return d
        return go

    _rec("pure_vert_ob25", _pure(None))
    _rec("pure_slant_ob25", _pure((0.4, 0.0)))
    return out


def main():
    rows = {}
    t0 = time.time()
    for name in FIXTURES_1D:
        t = time.time()
        rows[name] = run_fixture(name)
        rows[name]["_secs"] = round(time.time() - t, 3)
        print(f"  {name:38s} {rows[name]['_secs']:7.2f}s")
    two = twod_surfaces()
    for k, v in two.items():
        print(f"  {k:38s} {v['_secs']:7.2f}s")
    _lib.save("t1_identity", dict(oned=rows, twod=two,
                                  total_secs=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()

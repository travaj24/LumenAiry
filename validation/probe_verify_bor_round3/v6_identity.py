"""BIT IDENTITY against 1ac6de7e on THIS verification's own battery.

The round-3 report records that the round-2 verification's own 65-fixture
battery is "honest and NOT probative", because every one of its legacy-nodal
rows puts its loss on the MIDDLE layer and both half-spaces are exactly
lossless -- so it contains no row of the population round 3 changed.  This
battery is built the other way round: **32 of its 48 BOR rows carry a loss on
the INCIDENCE half-space or on both half-spaces**, which is the population, and
the remaining 16 are the controls (lossless, or loss on the middle layer only).

Every fixture is reduced to the SHA-256 of the exact IEEE-754 bytes of what a
caller receives, with the guards ARMED -- a raising fixture records its
exception class instead.  Run twice, ``LUM_PROBE_TAG=BASE`` against the
1ac6de7e tree and untagged against ``fix/bor-guards-round3``, then diffed by
``v6_identity_diff.py``.

LEGAL vs ILLEGAL.  The only legal move is ``HASH -> BORNodalPassivityError``:
round 3 adds refusals and changes no number.  A changed hash, a refusal that
became an answer, or a different exception class is a defect.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

K0 = _vb3.K0


def _hash_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def _guarded(fn):
    """Run a fixture with the guards ARMED; record the answer's bytes or the
    exception class the caller receives."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return dict(kind="hash", value=fn())
    except Exception as exc:                         # noqa: BLE001 - recorded
        return dict(kind="raise", value=type(exc).__name__)


# ------------------------------------------------------------------ BOR rows #
def _bor_nodal(family, m, N, rbl, where, im):
    def run():
        import lumenairy.elements.bor.bor_solve as bs
        lay = _vb3.stack("nodal", family, m, N, rbl, im, where)
        res = bs.solve(lay, K0)
        return _hash_arrays(res["R"], res["T"], res["energy"], res["inc"],
                            res["out"], res["q_inc"], *res["S"])
    return run


def _bor_staggered(family, m, N, rbl, where, im):
    def run():
        import lumenairy.elements.bor.bor_solve as bs
        lay = _vb3.stack("staggered", family, m, N, rbl, im, where)
        res = bs.solve(lay, K0)
        return _hash_arrays(res["R"], res["T"], res["energy"], res["inc"],
                            res["out"], res["q_inc"], *res["S"])
    return run


def _bor_stack(basis, m, N, Rbig, nsup, nsub, k0, degree=None):
    def run():
        from lumenairy.elements.bor.bor_stack import BORStack
        kw = dict(Rbig=Rbig, m=m, N=N, n_superstrate=nsup,
                  n_substrate=nsub, basis=basis)
        if degree is not None:
            kw["degree"] = degree
        s = BORStack(**kw)
        s.add_layer(0.4, eps=2.9)
        s.add_layer(0.55, rings=(2.4, 0.4, 2.2, 1.5))
        s.add_layer(0.4, eps=2.9)
        s.set_source(k0=k0)
        res = s.solve()
        return _hash_arrays(res["R"], res["T"], res["energy"], res["q"],
                            res["gamma"], *res["S"])
    return run


def bor_fixtures():
    out = {}
    #: 32 rows of the population round 3 changed -- the loss is on the
    #: INCIDENCE half-space or on BOTH
    for family in ("para", "grate", "core", "bilayer"):
        for m in (0, 1, 2, 3):
            for where, im in (("inc", 1e-9), ("both", 1e-6)):
                out["nodal_%s_m%d_%s_%g" % (family, m, where, im)] = (
                    _bor_nodal(family, m, 120, 2.0, where, im))
    #: 8 controls: lossless, and loss on the middle layer only
    for family in ("para", "grate", "core", "bilayer"):
        out["nodal_%s_lossless" % family] = _bor_nodal(family, 1, 120, 2.0,
                                                       "none", 0.0)
        out["nodal_%s_mid1e-6" % family] = _bor_nodal(family, 1, 120, 2.0,
                                                      "mid", 1e-6)
    #: 8 div-conforming twins, which the guard never touches
    for family in ("para", "grate"):
        for where, im in (("inc", 1e-9), ("both", 1e-6), ("mid", 1e-6),
                          ("none", 0.0)):
            out["stag_%s_%s_%g" % (family, where, im)] = (
                _bor_staggered(family, 1, 120, 2.0, where, im))
    #: 6 BORStack rows across the three production bases
    for basis, degree in (("staggered", None), ("fd", None), ("sem", 6)):
        for m in (0, 2):
            out["stack_%s_m%d" % (basis, m)] = _bor_stack(
                basis, m, 96, 14.0, 1.45, 1.7, 3.0, degree)
    return out


# ------------------------------------------------------------------ EME rows #
def _eme_layer_modes(strips, Lx, Nx, Ly, k0, rng, ky0):
    def run():
        from lumenairy.elements.eme import layer_modes
        q = np.asarray(layer_modes(strips, Lx, Nx, Ly, k0, rng, ky0=ky0))
        return _hash_arrays(q)
    return run


def _eme_strip_x(eps_x, Lx, Nx, k0, kx0):
    def run():
        from lumenairy.elements.eme import strip_x_modes
        lam, Phi = strip_x_modes(np.asarray(eps_x, float), Lx, Nx, k0, kx0)
        return _hash_arrays(np.asarray(lam), np.asarray(Phi))
    return run


def _eme_ref2d(eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0):
    def run():
        from lumenairy.elements.eme import ref_2d_modes
        q = np.asarray(ref_2d_modes(eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0))
        return _hash_arrays(q)
    return run


def eme_fixtures():
    Lx = Ly = 1.0
    Nx = 24

    def grating(e_lo, e_hi, duty=0.5):
        xg = (np.arange(Nx) + 0.5) / Nx
        return np.where(xg < duty, e_hi, e_lo).astype(float)

    out = {}
    uni = np.full(Nx, 4.0)
    g14 = grating(1.0, 4.0)
    g23 = grating(2.0, 3.0, 0.35)
    for i, (strips, k0, ky0, rng) in enumerate((
            ([(uni, 0.5), (uni, 0.5)], 8.0, np.pi, (150, 256)),
            ([(g14, 0.5), (uni, 0.5)], 8.0, np.pi, (120, 256)),
            ([(g14, 0.3), (g23, 0.7)], 8.0, np.pi, (120, 256)),
            ([(g23, 0.5), (uni, 0.5)], 6.5, 0.7 * np.pi, (60, 190)),
            ([(g14, 0.25), (uni, 0.5), (g23, 0.25)], 8.0, np.pi, (120, 256)),
            ([(uni, 1.0)], 5.0, 0.3 * np.pi, (40, 130)))):
        out["eme_layer_modes_%d" % i] = _eme_layer_modes(
            strips, Lx, Nx, Ly, k0, rng, ky0)
    for i, (eps_x, k0, kx0) in enumerate(((uni, 8.0, 0.0), (g14, 8.0, 0.0),
                                          (g23, 6.5, 0.4 * np.pi),
                                          (g14, 5.0, np.pi))):
        out["eme_strip_x_%d" % i] = _eme_strip_x(eps_x, Lx, Nx, k0, kx0)
    from lumenairy.elements.eme import strips_to_eps_xy
    for i, (strips, Ny, k0) in enumerate(
            ((((g14, 0.5), (uni, 0.5)), 24, 8.0),
             (((g23, 0.5), (uni, 0.5)), 32, 6.5))):
        eps_xy = strips_to_eps_xy(list(strips), Lx, Nx, Ly, Ny)
        out["eme_ref2d_%d" % i] = _eme_ref2d(eps_xy, Lx, Ly, Nx, Ny, k0,
                                             0.0, np.pi)
    return out


def main():
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    t0 = time.time()
    res = {}
    for name, fn in sorted(bor_fixtures().items()):
        res[name] = _guarded(fn)
        print("  BOR  %-34s %s %s" % (name, res[name]["kind"],
                                      res[name]["value"][:16]), flush=True)
    n_bor = len(res)
    for name, fn in sorted(eme_fixtures().items()):
        res[name] = _guarded(fn)
        print("  EME  %-34s %s %s" % (name, res[name]["kind"],
                                      res[name]["value"][:16]), flush=True)
    summary = dict(n_bor=n_bor, n_eme=len(res) - n_bor, n=len(res),
                   raising=sum(v["kind"] == "raise" for v in res.values()),
                   seconds=time.time() - t0)
    print(" SUMMARY", summary)
    _vb3.dump("v6_identity", dict(fixtures=res, summary=summary), a)


if __name__ == "__main__":
    main()

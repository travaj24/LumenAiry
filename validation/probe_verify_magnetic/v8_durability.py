"""V8 -- DURABILITY audit of ``tests/unit/test_pmm2d_staggered_magnetic.py``.

Every numeric bar in that file, RE-MEASURED here by driving the file's own
helpers with the file's own fixtures -- so the number printed is exactly the
number the assertion sees, not a number read out of a comment.

Prints, per bar: the assertion, the bar, the re-measured value, the ratio to
the bar, and (where the file states one) the smallest real signal on the other
side.  ``docs/TESTING_STANDARDS.md`` asks for DECADES of gap on both sides, so
any ratio under 10x is flagged SUB-DECADE.
"""
import importlib.util
import pathlib
import time

import numpy as np

import lumenairy

assert lumenairy.__file__.startswith("C:\\tmp\\lum_vmag"), lumenairy.__file__

_ROOT = pathlib.Path("C:/tmp/lum_vmag")
_SPEC = importlib.util.spec_from_file_location(
    "magtests", _ROOT / "tests/unit/test_pmm2d_staggered_magnetic.py")
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)

_ROWS = []


def row(gate, assertion, bar, measured, sense="<", note=""):
    ratio = (bar / measured) if sense == "<" else (measured / bar)
    flag = "SUB-DECADE" if ratio < 10.0 else ""
    _ROWS.append((gate, assertion, f"{bar:.3g}", f"{measured:.4e}",
                  f"{ratio:.3g}x", flag, note))
    print(f"  {gate:5s} {assertion:46s} bar {sense}{bar:<9.3g} measured "
          f"{measured:.4e}  {ratio:8.3g}x  {flag} {note}")


class _MP:
    """The two monkeypatch calls the fail-before helpers make."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, val, raising=True):
        if isinstance(obj, str):
            mod, _, attr = obj.rpartition(".")
            import importlib  # noqa: PLC0415
            obj, name = importlib.import_module(mod), attr
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo.clear()


def main():
    t0 = time.perf_counter()
    print("--- G1 ---")
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    kw = dict(alpha0x=0.31, alpha0y=-0.17, k0=2 * np.pi / 0.62)
    a = T.Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw)
    b = T.Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw,
                              mu_cell=np.ones((2, 2), dtype=complex))
    worst = 0.0
    for name in ("Lmat", "Rmat", "Stt", "Schur"):
        aa, bb = getattr(a, name), getattr(b, name)
        worst = max(worst, float(np.max(np.abs(aa - bb))
                                 / max(float(np.max(np.abs(aa))), 1e-30)))
    g2a, g2b = T._region_modes(a)[3], T._region_modes(b)[3]
    near = float(np.max(np.min(np.abs(g2a[:, None] - g2b[None, :]), axis=1))
                 / np.max(np.abs(g2a)))
    row("G1", "mu=1 forced: worst retained-operator rel", 1e-12, worst)
    row("G1", "mu=1 forced: eigenvalue set (rel)", 1e-12, near)
    ec = T._uni(4.0 * T._EYE)
    p = T.pmm_jones_2d_staggered(T._P, T._P, ec, 1.0, 1.0, T._DEP, T._WL,
                                 degree=5, n_orders=2)
    q = T.pmm_jones_2d_staggered(T._P, T._P, ec, 1.0, 1.0, T._DEP, T._WL,
                                 mu_cell=np.ones((2, 2), dtype=complex),
                                 degree=5, n_orders=2)
    row("G1", "mu=1 forced: public dR / dT / dJones", 1e-12,
        max(float(np.max(np.abs(x - y))) for x, y in zip(p[1:], q[1:])))

    print("--- G2 ---")
    worst = max(T._slab_residual(4.0, 1.0, th, 8, magnetic=m)
                for th in (0.0, 0.35) for m in (False, True))
    row("G2", "nonmagnetic pin (eps=4, mu=1, M=8)", 1e-12, worst)
    mags = {}
    for eps, mu in ((4.0, 2.0), (1.0, 4.0), (4.0, 2.0 + 0.3j),
                    (4.0 + 0.2j, 2.0)):
        mags[(eps, mu)] = max(T._slab_residual(eps, mu, th, 8)
                              for th in (0.0, 0.35))
    row("G2", "magnetic slabs, worst of 4 pairs x 2 theta", 1e-12,
        max(mags.values()),
        note="smallest real signal above: the M=5 oblique residual "
             f"{T._slab_residual(4.0, 2.0, 0.35, 5):.3e}")
    lad = [T._slab_residual(4.0, 2.0, 0.35, m) for m in (5, 8)]
    row("G2", "spectral ladder M=5 -> M=8 (drop)", 1e3, lad[0] / lad[1], ">")

    print("--- G3 ---")
    ue, um = T._uni(T._LC), T._uni(T._LC2)
    d, dj, ctrl = T._duality(ue, um, 0.30, 0.70, 8)
    row("G3", "uniform tensor pair: R/T duality", 1e-11, d)
    row("G3", "uniform tensor pair: Jones duality", 1e-11, dj)
    row("G3", "uniform pair: no-rotation control", 1e-2, ctrl, ">")
    ecp = T._cell(T._LC, 4.0 * T._EYE)
    d5 = max(T._duality(ecp, None, th, ph, 5)[0]
             for th, ph in ((0.0, 0.0), (0.30, 0.70)))
    d8s = [T._duality(ecp, None, th, ph, 8)
           for th, ph in ((0.0, 0.0), (0.30, 0.70))]
    row("G3", "patterned ladder: R/T at M=8", 7.6e-4, max(x[0] for x in d8s))
    row("G3", "patterned ladder: Jones at M=8", 7.6e-4,
        max(x[1] for x in d8s))
    row("G3", "patterned ladder drop M=5 -> M=8", 5.0,
        d5 / max(x[0] for x in d8s), ">")
    row("G3", "patterned: no-rotation control", 1e-2,
        min(x[2] for x in d8s), ">")

    print("--- G4 ---")
    ors = _g4_oracles()
    for name, oracle in zip(("pmm_jones_1d", "rcwa_jones_1d"), ors):
        drt, dj, forb, unsw = T._g4_residual(8, oracle)
        row("G4", f"M=8 R/T vs {name}", 3e-5, drt)
        row("G4", f"M=8 Jones vs {name}", 7e-5, dj)
        row("G4", f"unswapped control vs {name}", 1e-3, unsw, ">")
    d5 = T._g4_residual(5, ors[0])[0]
    d8, _dj, forb, _u = T._g4_residual(8, ors[0])
    row("G4", "ladder drop M=5 -> M=8", 20.0, d5 / d8, ">")
    row("G4", "y-forbidden (n != 0) leak", 1e-20, forb)

    print("--- G5 ---")
    tot = T._closure(T._uni(T._LC), T._uni(T._MU_GYRO), 8)
    row("G5", "uniform Hermitian mu closure", 1e-11,
        max(abs(v - 1.0) for v in tot))
    ecp = T._cell(T._LC, 4.0 * T._EYE)
    mcp = T._cell(T._MU_GYRO, 1.2 * T._EYE)
    totp8 = T._closure(ecp, mcp, 8, 0.25, 0.60)
    totp6 = T._closure(ecp, mcp, 6, 0.25, 0.60)
    row("G5", "patterned Hermitian mu closure (M=8)", 1e-4,
        max(abs(v - 1.0) for v in totp8))
    row("G5", "patterned closure drop M=6 -> M=8", 10.0,
        max(abs(v - 1.0) for v in totp6)
        / max(abs(v - 1.0) for v in totp8), ">")
    # the test's OWN call: normal incidence, record=True
    lossy, fired = T._closure(ecp, T._cell(T._MU_LOSSY, 1.2 * T._EYE), 8,
                              record=True)
    row("G5", "lossy mu: sum R+T stays below 0.99", 0.99, max(lossy),
        note=f"rows {lossy[0]:.6f} / {lossy[1]:.6f}, tripwire fired {fired}")
    tot3 = T._closure(ecp, mcp, 3, 0.25, 0.60)
    row("G5", "tripwire: under-resolved defect", 0.3,
        max(abs(v - 1.0) for v in tot3), ">",
        note="shipped window _STAG_CLOSURE_TOL = 5e-2")

    print("--- G9 / G6 / fail-before / API ---")
    a5, a8 = T._absorption_closure(5), T._absorption_closure(8)
    row("G9", "layer_absorption closure at M=8", 1e-8, a8)
    row("G9", "absorption ladder drop M=5 -> M=8", 100.0, a5 / a8, ">")
    ea, ma = T._cell(T._LC, 4.0 * T._EYE), T._cell(T._MU_GYRO, 1.2 * T._EYE)
    dev, djj = T._transpose_residual(ea, ma, T._transpose_cell(ea),
                                     T._transpose_cell(ma))
    row("G6", "transpose, correct placement (R/T)", 1e-11, dev)
    row("G6", "transpose, correct placement (Jones)", 1e-11, djj)
    # the test's OWN quantities: the FIRST element (R/T) of each pair, and the
    # SWAP applied to arm A, not arm B
    ebt, mbt = T._transpose_cell(ea), T._transpose_cell(ma)
    maw = ma.copy()
    maw[..., 0, 1], maw[..., 1, 0] = ma[..., 1, 0].copy(), ma[..., 0, 1].copy()
    bad, badj = T._transpose_residual(ea, maw, ebt, mbt)
    row("G6", "m12/m21 swapped (must BREAK)", 1e-7, bad, ">",
        note=f"its Jones residual {badj:.4e}")
    eaw = ea.copy()
    eaw[..., 0, 1], eaw[..., 1, 0] = ea[..., 1, 0].copy(), ea[..., 0, 1].copy()
    noop, noopj = T._transpose_residual(eaw, ma, ebt, mbt)
    row("G6", "e12/e21 swapped (must be a NO-OP)", 1e-11, noop,
        note=f"its Jones residual {noopj:.4e}")
    mp = _MP()
    for kind in ("gram", "chi33", "chi_t"):
        try:
            v = T._knockout_residual(mp, kind)
        finally:
            mp.undo()
        row("FB", f"knockout {kind} (must BREAK)", 1e-4, v, ">")
    row("FB", "intact analytic residual", 1e-12,
        T._slab_residual(4.0, 2.0, 0.35, 8))
    # the MIXED blocks need an anisotropic mu -- gated by duality, not Airy
    row("FB", "mixed blocks: INTACT duality (M=7)", 1e-11,
        T._duality(ue, um, 0.30, 0.70, 7)[0])
    orig_chi = T.Granet2DTransverseE._chi_maps

    def _zero_mixed(self):
        c = orig_chi(self)
        if c is None:
            return None
        c11, c12, c21, c22, c33 = c
        return c11, np.zeros_like(c12), np.zeros_like(c21), c22, c33
    mp.setattr(T.Granet2DTransverseE, "_chi_maps", _zero_mixed)
    try:
        row("FB", "mixed chi12/chi21 zeroed (must BREAK)", 1e-4,
            T._duality(ue, um, 0.30, 0.70, 7)[0], ">")
    finally:
        mp.undo()
    (d4, _f4), (d7, f7) = _cascade(4), _cascade(7)
    row("API", "magnetic + out-of-plane cascade at M=7", 1e-4, d7,
        note=f"tripwire fired {f7} times (must be 0)")
    row("API", "cascade drop M=4 -> M=7", 50.0, d4 / d7, ">")

    print("--- G10 (this verification's follow-up) ---")
    c = T._wood_stack(T._WL_CUT, mu=1.0)
    d = T._wood_stack(T._wood_nudged_wl(), mu=1.0)
    row("G10", "mu=1: the two wavelengths DIFFER", 1e-10,
        float(np.max(np.abs(c[0] - d[0]))), ">")
    e = T._wood_stack(T._WL_CUT)
    row("G10", "mu=1 magnetic vs nonmagnetic (re-summation)", 1e-12,
        max(float(np.max(np.abs(x - y))) for x, y in zip(c, e)))

    print(f"\n  probe wall time {time.perf_counter() - t0:.1f} s")
    sub = [r for r in _ROWS if r[5]]
    print(f"  {len(_ROWS)} bars re-measured, {len(sub)} SUB-DECADE:")
    for r in sub:
        print(f"    {r[0]} {r[1]}  bar {r[2]} vs {r[3]} = {r[4]}")


def _g4_oracles():
    """The file's OWN module-scope G4 fixture, rebuilt verbatim."""
    kw = dict(angle=T._G4_TH)
    return (T.pmm_jones_1d(T._G4_P, T._G4_RIDGE, T._G4_GROOVE, 1.0, 1.0,
                           T._G4_DEP, 0.5, T._WL, degree=16, stabilize=False,
                           **kw),
            T.rcwa_jones_1d(T._G4_P, T._G4_RIDGE, T._G4_GROOVE, 1.0, 1.0,
                            T._G4_DEP, 0.5, T._WL, n_orders=40, **kw))


def _cascade(m):
    """The file's own generalized-cascade fixture, rebuilt verbatim."""
    import warnings  # noqa: PLC0415
    oop = T.uniaxial_tensor(1.5, 1.8, 0.6, phi=0.3)
    ec = T._uni(oop)
    ec[0, 0] = 4.0 * T._EYE
    st = T.PMM2DStackPure(T._P, T._P, n_modes=m, n_orders=3)
    st.add_layer(0.20e-6, eps_cell=ec)
    st.add_layer(0.20e-6, eps=4.0, mu_cell=T._uni(1.6 * T._EYE))
    st.set_source(T._WL, theta=0.2, phi=0.4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _o, r, t, _j = st.solve()
    fired = len([x for x in w if "energy closure" in str(x.message)])
    return (max(abs(float(r[i].sum() + t[i].sum()) - 1.0) for i in (0, 1)),
            fired)


if __name__ == "__main__":
    main()

"""ROUND 3: how far does the answer MOVE between the round-2 body (`conj(r)`)
and the round-3 body (`-r`), over the RCWA / PMM / Berreman fixture sets?

The repair brief anticipated a BIT-IDENTITY census -- "the flipped-mode values
move by exactly 2 Re(r) ~ 1e-16".  The premise is wrong: the band admits
``Re(r)`` out to ``1e-8 * scale``, so the two bodies differ by up to ~1e-8
RELATIVE in the flipped modes, and the answer moves correspondingly.  This
probe measures the motion instead of assuming it.

Reported per fixture: max |post - pre| over every returned efficiency, and the
relative motion against the answer's own scale.
"""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib
import numpy as np
import threadpoolctl
import lumenairy
import lumenairy.elements.rcwa._core as _rc

_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")


def _round2_body(x, xp=None, band=_rc._CUT_BAND_REL):
    """The SHIPPED 59105d6 body: the same band, the ``conj(r)`` flip."""
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    on_cut = xp.abs(r.real) <= band * scale
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)


class _arm:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        self._saved = []
        if self.body is not None:
            for n in _BOUND:
                m = importlib.import_module(n)
                if hasattr(m, "_sqrt_decay"):
                    self._saved.append((m, m._sqrt_decay))
                    m._sqrt_decay = self.body
        return self

    def __exit__(self, *a):
        for m, f in self._saved:
            m._sqrt_decay = f
        return False


from lumenairy.elements.pmm import pmm_efficiency_2d, PMM2DStackHybrid
from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_jones_1d_segments
from lumenairy.elements import berreman_jones_1d

P, WL, DEP = 0.6e-6, 0.55e-6, 0.25e-6
XB = (0.2 * P, 0.6 * P)
HOST, WEAK, NSUB = 2.25, 2.26, 1.63


def _vec(out):
    return np.concatenate([np.asarray(a, float).ravel()
                           for a in out if np.asarray(a).dtype.kind == "f"])


def FIX():
    rows = []
    for th in (0.0, 1e-7, 0.3, 0.6):
        for pol in ("te", "tm"):
            rows.append((f"pmm2d th={th} {pol}",
                         lambda th=th, pol=pol: _vec(pmm_efficiency_2d(
                             P, P, 6.0 + 0j, 1.0, XB, XB, 1.5, 1.0, DEP, WL,
                             theta=th, degree=5, n_orders=2,
                             polarization=pol)[1:])))
    rows.append(("pmm2d conical .4/.7",
                 lambda: _vec(pmm_efficiency_2d(
                     P, P, 6.0 + 0j, 1.0, XB, XB, 1.5, 1.0, DEP, WL,
                     theta=0.4, phi=0.7, degree=5, n_orders=2,
                     polarization="te")[1:])))
    rows.append(("pmm2d LOSSY",
                 lambda: _vec(pmm_efficiency_2d(
                     P, P, 6.0 + 0.4j, 1.0, XB, XB, 1.5, 1.0, DEP, WL,
                     theta=0.0, degree=5, n_orders=2,
                     polarization="te")[1:])))

    def _stack(M, spacer=HOST):
        e = np.full((6, 6), HOST + 0j)
        e[2:4, 2:4] = WEAK
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=1.0,
                              degree=7, n_orders=M, symmetry=False)
        st.add_layer(0.1e-6, eps=spacer)
        st.add_layer(DEP, eps_cell=e)
        st.add_layer(0.1e-6, eps=spacer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.set_source(WL, theta=0.0).solve()
        return np.concatenate([np.asarray(R, float).ravel(),
                               np.asarray(T, float).ravel()])
    for M in (3, 4, 5):
        rows.append((f"pmm2d SPACER stack M={M}", lambda M=M: _stack(M)))
    for th in (0.0, 0.2, 0.5):
        for pol in ("te", "tm"):
            rows.append((f"rcwa1d th={th} {pol}",
                         lambda th=th, pol=pol: _vec(rcwa_efficiency_1d(
                             P, 6.0 + 0j, 1.0, 0.5, 1.5, 1.0, DEP, WL,
                             theta=th, n_orders=6, polarization=pol)[1:])))
    rows.append(("rcwa1d THIN M=19 te",
                 lambda: _vec(rcwa_efficiency_1d(
                     10e-6, 1.55, 1.5, 1.5, 1.5, 0.5e-6, 0.5, 700e-9,
                     angle=0.0, n_orders=19, polarization="te",
                     stabilize=False)[1:])))
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    er = rot @ np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex) @ rot.T
    eg = np.diag([1.5 ** 2] * 3).astype(complex)
    for n in (11, 19, 31):
        rows.append((f"rcwa jones COINCIDENCE n={n}",
                     lambda n=n: np.concatenate([
                         np.asarray(a, float).ravel() for a in
                         rcwa_jones_1d_segments(0.7e-6, [(0.5, er), (0.5, eg)],
                                                1.5, 1.0, 0.5e-6, 1.0e-6,
                                                angle=0.0, n_orders=n)[1:3]])))
    for ang in (0.0, 0.4):
        rows.append((f"berreman ang={ang}",
                     lambda ang=ang: np.abs(np.concatenate([
                         np.asarray(a).ravel() for a in berreman_jones_1d(
                             [(np.diag([2.25, 2.9, 2.25]).astype(complex),
                               0.2e-6)], 1.5, 1.0, WL, angle=ang)]))))
    return rows


if __name__ == "__main__":
    print(f"# lumenairy {lumenairy.__file__}")
    print(f"# py{sys.version.split()[0]} np{np.__version__} "
          f"arch={threadpoolctl.threadpool_info()[0].get('architecture','?')} "
          f"CORETYPE={os.environ.get('OPENBLAS_CORETYPE','-')}")
    worst_abs, worst_rel, worst_name, n_moved, n_tot = 0.0, 0.0, "-", 0, 0
    for name, fn in FIX():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _arm(_round2_body):
                pre = fn()
            with _arm(None):
                post = fn()
        n_tot += 1
        d = np.max(np.abs(post - pre)) if pre.shape == post.shape else np.nan
        sc = max(float(np.max(np.abs(pre))), 1e-300)
        rel = d / sc
        if d > 0:
            n_moved += 1
        if rel > worst_rel:
            worst_rel, worst_abs, worst_name = rel, d, name
        print(f"  {name:30s} max|post-pre| {d:.6e}   rel {rel:.6e}"
              + ("   IDENTICAL" if d == 0.0 else ""))
    print(f"\n{n_moved} of {n_tot} fixtures MOVED at all")
    print(f"WORST motion {worst_abs:.6e} (rel {worst_rel:.6e}) on {worst_name}")

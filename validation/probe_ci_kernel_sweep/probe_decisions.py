"""CI KERNEL SWEEP (2026-09-11) -- the library's guard DECISIONS, per arm.

WHY THIS PROBE EXISTS.  The 5.45.0 release matrix went RED on CI (AMD EPYC
runners) while the same commit was green on both local builds.  Every failing
assertion in that matrix was a *decision* -- a guard that refuses on one build
and returns on another, or a test whose bar is pinned on a build-dependent
reading.  ``docs/TESTING_STANDARDS.md`` ("flakiness is bad math") forbids
re-running such a thing to green: the decision itself has to be made
build-independent, and that requires MEASURING it on more than one BLAS kernel.

WHAT AN "ARM" IS.  One (build, kernel) pair.  ``build`` is the interpreter +
numpy + OS (``WIN`` = Windows/CPython 3.14, ``WSL`` = Ubuntu/CPython 3.12);
``kernel`` is the OpenBLAS micro-kernel family, selected on the command line
with ``OPENBLAS_CORETYPE``.  The bundled scipy-openblas is DYNAMIC_ARCH, so
the variable re-dispatches the whole BLAS/LAPACK kernel set without rebuilding
anything.  Thread caps are pinned to 1 on every arm so the ONLY axis that
moves is the kernel.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_CORETYPE=NEHALEM python probe_decisions.py --out arm.json

MEASURED CAVEAT (2026-09-11, this host = AMD Ryzen 9 5950X, Zen 3):

  * ``OPENBLAS_CORETYPE=ZEN`` is NOT a distinct kernel in these wheels.  It
    is not in the DYNAMIC_ARCH table, so it falls back to auto-detection --
    and auto-detection on a Zen CPU reports ``Haswell``.  An unrecognised
    name (``BOGUSCORE``) resolves to exactly the same thing.  The practical
    consequence is GOOD news: the local default arm ALREADY IS the CI Zen
    arm at the BLAS-kernel level, so a decision that differs between here
    and CI does not differ because of a Zen kernel.
  * ``OPENBLAS_CORETYPE=SKYLAKEX`` sets the corename but the kernels are
    AVX-512, which this Zen 3 host cannot execute: the first BLAS call dies
    with SIGILL (exit 132) on BOTH builds.  It is therefore unreachable here
    and is recorded as such rather than silently skipped.

The usable ladder is consequently HASWELL (= the default = Zen) / SANDYBRIDGE
(AVX, no FMA) / NEHALEM (SSE4.2) / PRESCOTT (SSE2, reported as ``Katmai``) --
four different reduction orders and blockings, which is what the decisions
have to survive.

WHAT IS RECORDED.  Two blocks per arm:

  ``decisions``     the guard OUTCOMES the library actually takes, as short
                    strings ("refuse"/"return", "warn"/"silent",
                    "accept"/"refuse", a count).  These are what
                    ``tests/unit/test_ci_kernel_consistency.py`` asserts
                    AGREE across every committed arm.  A decision must be a
                    DISCRETE outcome, never a float -- comparing floats
                    across kernels is the mistake this probe exists to catch.
  ``hypothetical``  the verdict a bar the library does NOT ship would return
                    at a site the library deliberately leaves UNGUARDED.
                    These are recorded to be shown NON-unanimous: they are
                    the standing evidence that the site is undecidable, and
                    the consistency gate asserts they stay that way rather
                    than asserting they agree.  Promoting one of these to a
                    real guard is exactly the mistake the audit forbids.
  ``readings``      the underlying floats, for the audit document only.  They
                    are NOT asserted anywhere; they move with the kernel by
                    design and that is the whole point.

Every fixture below is cheap (the whole probe is a few seconds) so the
consistency gate can re-run it on the current arm inside its 30 s budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import stack as _st1d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C  # noqa: E402
from lumenairy.elements.rcwa import _core as _rc  # noqa: E402

# --------------------------------------------------------------- fixtures
_P = 1.0e-6
_WL = 0.62e-6
_TH, _PH = 0.09, 0.3
_EPS_H, _EPS_P, _EPS_B, _EPS_SPACER = 2.25, 6.0, 6.0, 2.1
_E_OOP = np.array([[4.0, 0.0, 0.8], [0.0, 3.4, 0.0], [0.75, 0.0, 3.2]],
                  dtype=_C)
_WA = (0.2371, 0.6183)
_WC = (0.2371, 0.7402)


def _sc(walls):
    return [w * _P for w in walls]


def _mid(walls):
    b = (0.0,) + tuple(walls) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def _tensor_cell(walls, lo, hi, eps_in=None, eps_host=_EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    e = _E_OOP if eps_in is None else eps_in
    for i in range(n):
        for j in range(n):
            c[i, j] = e if (lo < m[i] < hi and lo < m[j] < hi) \
                else np.eye(3) * eps_host
    return c


def _scalar_cell(walls, lo, hi, eps_in=_EPS_B, eps_host=_EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.full((n, n), _C(eps_host))
    for i in range(n):
        for j in range(n):
            if lo < m[i] < hi and lo < m[j] < hi:
                c[i, j] = _C(eps_in)
    return c


def _decade(x):
    """The DECADE INDEX of a positive reading -- ``floor(log10(x))``.

    A decade index is the coarsest honest discretisation of a float: it is
    the statement "this reading is between 1e-11 and 1e-10", which is a
    DECISION, where the reading itself is not.  Non-finite and non-positive
    readings get their own labels rather than a number, because a bar test
    treats them specially too.
    """
    x = float(x)
    if not math.isfinite(x):
        return "nonfinite"
    if x <= 0.0:
        return "zero_or_negative"
    return "1e%+03d" % math.floor(math.log10(x))


# ======================================================================
# A -- the plain 1-D ``_interface_smatrix`` site (the mortar round-2
#      rationale test's subject)
# ======================================================================
#: the 1-D section runs in the ROUND-2 GATE FILE'S OWN UNITS (period 1.2,
#: wavelength 0.85), verbatim -- the point of this section is to reproduce
#: exactly the population that file's rationale test reads, and a stack driven
#: with a different period/wavelength pair is a different device.
_1D_P, _1D_WL, _1D_TH = 1.2, 0.85, 0.15
_1D_EPS_P, _1D_EPS_H = 9.0, 2.25


def _pmm1d_two_layer(delta):
    a0, a1 = 0.27865, 0.62505
    st = PMMStack(_1D_P, degree=12, far_field_orders=5)
    st.add_layer(0.08, segments=[(a0, _1D_EPS_H), (a1 - a0, _1D_EPS_P),
                                 (1 - a1, _1D_EPS_H)])
    b0, b1 = a0 - delta, a1 + delta
    st.add_layer(0.08, segments=[(b0, _1D_EPS_H), (b1 - b0, _1D_EPS_P),
                                 (1 - b1, _1D_EPS_H)])
    st.set_source(_1D_WL, theta=_1D_TH)
    return st


def _interface_site_population(dec, hyp, rea):
    """The ``_interface_smatrix`` rcond population, WITH THE SLIVER GUARD
    DISARMED.

    Disarming is not a convenience: the 1-D sliver guard's own refusal is
    itself a kernel-dependent decision (recorded separately in section B and
    handed to the sliver-guard owner), so leaving it armed would make this
    section measure the SLIVER guard rather than the MORTAR site.  Disarmed,
    the site is reached at both wall separations on every build and the
    population is the same object everywhere.
    """
    real = _st1d._interface_smatrix
    prev_guard = _st1d.PMM_SLIVER_GUARD
    seen = []

    def _patched(Wa, Va, Wb, Vb):
        for A in (np.asarray(Wb), np.asarray(Vb)):
            lu, piv = sla.lu_factor(A)
            gecon = sla.get_lapack_funcs("gecon", (A,))
            rc, _i = gecon(lu, float(np.max(np.sum(np.abs(A), axis=0))))
            seen.append(float(rc))
        return real(Wa, Va, Wb, Vb)

    _st1d._interface_smatrix = _patched
    _st1d.PMM_SLIVER_GUARD = False
    try:
        for delta in (1e-4, 1e-5):
            seen.clear()
            st = _pmm1d_two_layer(delta)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _o, R, T = st.solve()[:3]
            tot = float(np.max(np.atleast_2d(R).sum(1)
                               + np.atleast_2d(T).sum(1)))
            rc = min(seen)
            tag = "%.0e" % delta
            rea["pmm1d_interface/rcond@%s" % tag] = rc
            rea["pmm1d_interface/R+T@%s" % tag] = tot
            dec["pmm1d_interface/answer@%s" % tag] = (
                "closes" if abs(tot - 1.0) < 1e-5 else "open")
            dec["pmm1d_interface/warned@%s" % tag] = (
                "warn" if len(w) else "silent")
            dec["pmm1d_interface/returns@%s" % tag] = "return"
            # HYPOTHETICAL, not shipped: what a 1e-12 rcond bar -- the value
            # the 2-D IN-PLANE mortar sites use -- would decide here.  It is
            # in the ``hypothetical`` block precisely because its verdict is
            # NOT the same on every kernel.
            hyp["pmm1d_interface/rcond_decade@%s" % tag] = _decade(rc)
            hyp["pmm1d_interface/bar_1e-12_would@%s" % tag] = (
                "refuse" if rc < _pc._MORTAR_RCOND_REFUSE else "accept")
    finally:
        _st1d._interface_smatrix = real
        _st1d.PMM_SLIVER_GUARD = prev_guard


# ======================================================================
# B -- the 1-D SLIVER guard's refusal (NOT owned here: handed off)
# ======================================================================
def _sliver_decisions(dec, rea):
    for delta in (1e-4, 1e-5):
        tag = "%.0e" % delta
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _o, R, T = _pmm1d_two_layer(delta).solve()[:3]
            tot = float(np.max(np.atleast_2d(R).sum(1)
                               + np.atleast_2d(T).sum(1)))
            rea["sliver/R+T@%s" % tag] = tot
            dec["sliver/pmm1d@%s" % tag] = (
                "warn" if any("SLIVER" in str(x.message) for x in w)
                else "return")
        except ValueError as exc:
            dec["sliver/pmm1d@%s" % tag] = (
                "refuse" if "SLIVER" in str(exc) else "raise_other")
            rea["sliver/msg@%s" % tag] = str(exc)[:80]


# ======================================================================
# C -- the 2-D per-layer MORTAR sites (rcond screen + residual screen)
# ======================================================================
def _mixed(kind, M, n_orders=2):
    st = PMM2DStackPure(_P, n_modes=M, n_orders=n_orders, n_substrate=1.5,
                        layer_grids="per-layer")
    if kind == "spacer":
        st.add_layer(0.13e-6, eps_cell=_tensor_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA))
        st.add_layer(0.10e-6, eps=_EPS_SPACER)
    elif kind == "pattern":
        st.add_layer(0.13e-6, eps_cell=_tensor_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA))
        st.add_layer(0.10e-6, eps_cell=_scalar_cell(_WC, *_WC),
                     x_walls=_sc(_WC), y_walls=_sc(_WC))
    elif kind == "slant_both":
        st.add_layer(0.13e-6, eps_cell=_scalar_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA), slant=(0.08, 0.03))
        st.add_layer(0.10e-6, eps_cell=_scalar_cell(_WC, *_WC, eps_in=4.0),
                     x_walls=_sc(_WC), y_walls=_sc(_WC), slant=(0.08, 0.03))
    else:
        raise ValueError(kind)
    st.set_source(_WL, theta=_TH, phi=_PH)
    return st


def _mortar_decisions(dec, rea):
    for kind, M in (("spacer", 5), ("pattern", 4), ("slant_both", 4)):
        key = "mortar/%s@M%d" % (kind, M)
        census = []
        _pc._MORTAR_SOLVE_CENSUS = census
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _mixed(kind, M).solve(jones=False)
            outcome = "accept"
        except Exception as exc:                       # noqa: BLE001
            outcome = ("refuse" if type(exc).__name__ == "_ConditioningError"
                       else "raise_" + type(exc).__name__)
            rea[key + "/msg"] = str(exc)[:100]
        finally:
            _pc._MORTAR_SOLVE_CENSUS = None
        dec[key] = outcome
        # per-SCREEN detail: how many solves each screen refused, and the
        # worst reading each screen saw.  ``n`` and the refusal COUNT are
        # decisions; the readings are not.
        refused = sum(1 for r in census if r[3])
        dec[key + "/refused_solves"] = str(refused)
        dec[key + "/n_solves"] = str(len(census))
        rc_all = [r[2] for r in census if r[2] is not None
                  and math.isfinite(r[2])]
        res_all = [r[4] for r in census if r[4] is not None]
        if rc_all:
            rea[key + "/worst_rcond"] = min(rc_all)
            dec[key + "/worst_rcond_decade"] = _decade(min(rc_all))
        if res_all:
            rea[key + "/worst_resid"] = max(res_all)
            dec[key + "/worst_resid_decade"] = _decade(max(res_all))


# ======================================================================
# D -- the per-layer WIDTH band: warn / silent / refuse
# ======================================================================
def _band_stack(frac, M=4, centre=0.44):
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    sw = (centre, centre + frac)
    tile = np.full((3, 3), _C(_EPS_H))
    tile[1, 1] = _C(_EPS_B)
    st.add_layer(0.10e-6, eps_cell=tile, x_walls=_sc(_WA), y_walls=_sc(_WA))
    st.add_layer(0.10e-6, eps_cell=tile, x_walls=_sc(sw), y_walls=_sc(sw))
    st.set_source(_WL, theta=_TH, phi=_PH)
    return st


def _band_decisions(dec, rea):
    for frac in (2.0e-2, 6.0e-2, 1.2e-3, 9.0e-4):
        key = "band/%.1e" % frac
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                _band_stack(frac).solve(jones=False)
            hit = [w for w in ws if issubclass(w.category, UserWarning)
                   and "degradation band" in str(w.message)]
            dec[key] = "warn" if hit else "silent"
        except ValueError as exc:
            dec[key] = ("refuse_min_seg" if "below the minimum" in str(exc)
                        else "raise_other")
            rea[key + "/msg"] = str(exc)[:80]


# ======================================================================
# E -- the RCWA generalized interface's T22 refusal
# ======================================================================
def _t22_decisions(dec, rea):
    """The ``T22`` guard, exercised through the PURE staggered engine's
    SLANTED and OUT-OF-PLANE routes (both take
    ``_interface_smatrix_general``) plus the ordinary in-plane control.
    """
    for kind, M in (("slant_both", 4), ("spacer", 5), ("pattern", 4)):
        key = "t22/%s@M%d" % (kind, M)
        census = []
        _rc._INV_CENSUS = census
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _mixed(kind, M).solve(jones=False)
            dec[key] = "accept"
        except Exception as exc:                       # noqa: BLE001
            dec[key] = ("refuse"
                        if "T22" in str(exc) or "generalized interface"
                        in str(exc) else "raise_" + type(exc).__name__)
        finally:
            _rc._INV_CENSUS = None
        t22 = [r for r in census if "T22" in r[0]]
        dec[key + "/n_t22"] = str(len(t22))
        dec[key + "/refused_t22"] = str(sum(1 for r in t22 if r[4]))
        rcs = [r[2] for r in t22 if r[2] is not None and math.isfinite(r[2])]
        if rcs:
            rea[key + "/worst_rcond_eq"] = min(rcs)
            dec[key + "/worst_rcond_eq_decade"] = _decade(min(rcs))
            dec[key + "/bar_would_refuse"] = (
                "refuse" if min(rcs) < _rc._INV_T22_RCOND_REFUSE else "accept")


# ======================================================================
# F -- the modal BRANCH-CUT orientation census
# ======================================================================
def _branch_cut_decisions(dec, rea):
    """``_sqrt_decay``'s ON-CUT decision, as a census rather than a float.

    For each fixture spectrum: how many roots land inside the relative
    on-cut band, and how many of those the selector CONJUGATES (i.e. how
    many modes would otherwise have carried the INCOMING root).  Both are
    integers -- the decision -- while the ``|Re(r)|/scale`` ratios that
    drive them are rounding-level noise and are recorded only as readings.
    """
    rng = np.random.default_rng(20260911)
    cases = {
        # a LOSSLESS propagating spectrum: lam^2 real-negative up to the
        # eigensolver's backward error, which is exactly the population the
        # band exists for
        "lossless_propagating": (-np.array([4.0, 9.0, 16.0, 25.0, 36.0])
                                 + 1j * rng.normal(0, 3e-16, 5)),
        # a genuinely DECAYING spectrum: real positive, must NOT be touched
        "evanescent": np.array([4.0, 9.0, 30.0, 120.0, 400.0]) + 0j,
        # a LOSSY spectrum: a real decay rate on the signal side of the band
        "lossy": (-np.array([4.0, 9.0, 16.0]) * (1.0 - 0.02j)),
        # a spectrum AT a layer cutoff -- the binding population of _CUT_BAND_REL
        "at_cutoff": np.array([-3.2e-16, -1.0, -4.0, -9.0]) + 1j * np.array(
            [1e-20, -2e-16, 5e-16, -9e-16]),
    }
    for name, lam2 in cases.items():
        r = np.sqrt(np.asarray(lam2, dtype=_C))
        scale = max(float(np.max(np.abs(r))), 1.0)
        ratio = np.abs(r.real) / scale
        on_cut = ratio <= _rc._CUT_BAND_REL * scale / scale
        out = _rc._sqrt_decay(np.asarray(lam2, dtype=_C))
        flipped = int(np.sum(out != r))
        dec["branch_cut/%s/n_on_cut" % name] = str(int(np.sum(on_cut)))
        dec["branch_cut/%s/n_conjugated" % name] = str(flipped)
        # the ORIENTATION itself: after the selector, does every mode carry
        # a non-negative imaginary part where the real part is on the cut?
        oriented = bool(np.all(out.imag[on_cut] >= 0.0)) if on_cut.any() \
            else True
        dec["branch_cut/%s/outgoing_everywhere" % name] = (
            "yes" if oriented else "no")
        rea["branch_cut/%s/max_ratio" % name] = float(np.max(ratio))
        rea["branch_cut/%s/min_ratio" % name] = float(np.min(ratio))


# ======================================================================
def _arm_id():
    """``BUILD-KERNEL`` -- the arm's identity, both halves MEASURED.

    The kernel half is read back from the loaded OpenBLAS (via
    ``threadpoolctl``) rather than from ``OPENBLAS_CORETYPE``, because the
    request and the result are NOT the same thing: ``ZEN`` and ``BOGUSCORE``
    both resolve to ``Haswell`` in these wheels (see the module docstring).
    Recording the REQUEST would let two arms that ran identical code look
    like independent evidence.
    """
    build = "WSL" if sys.platform.startswith("linux") else "WIN"
    arch = "unknown"
    try:
        import threadpoolctl  # noqa: I001, PLC0415
        for d in threadpoolctl.threadpool_info():
            if d.get("internal_api") in ("openblas", "mkl"):
                arch = str(d.get("architecture") or "unknown")
                break
    except Exception:                                   # noqa: BLE001
        pass
    return "%s-%s" % (build, arch), build, arch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the arm JSON here")
    args = ap.parse_args(argv)

    dec, hyp, rea = {}, {}, {}
    _interface_site_population(dec, hyp, rea)
    _sliver_decisions(dec, rea)
    _mortar_decisions(dec, rea)
    _band_decisions(dec, rea)
    _t22_decisions(dec, rea)
    _branch_cut_decisions(dec, rea)

    arm, build, arch = _arm_id()
    doc = {
        "arm": arm,
        "build": build,
        "kernel": arch,
        "coretype_requested": os.environ.get("OPENBLAS_CORETYPE", ""),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "threads": {v: os.environ.get(v, "")
                    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS")},
        "decisions": dec,
        "hypothetical": hyp,
        "readings": rea,
    }
    text = json.dumps(doc, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="cp1252", errors="replace") as fh:
            fh.write(text + "\n")
        print("wrote %s (%d decisions)" % (args.out, len(dec)))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixtures and helpers shared by the ROUND-2 branch-cut probes.

Round 1 (``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md``) pinned the
OUTGOING root in ``rcwa/_core.py::_sqrt_decay`` with a RELATIVE band and a
``conj`` flip.  Its independent verification
(``docs/audits/VERIFY_RCWA_EVEN_SECTOR_2026_09_11.md``, defect D1) found FIVE
copies of the same function inside ``lumenairy/elements/pmm/`` that kept the
EXACT-ZERO pin ``r.real == 0``.  These probes measure what those copies cost
before and after the round-2 change.

ARM DETECTION.  Every probe runs UNCHANGED on both arms; which arm it measured
is read off the LIVE source of the PMM copies, never off a flag, so no reading
can be mis-attributed.  PRE = the exact-zero pin still present in
``pmm/twod.py``; POST = the relative band.

The RCWA side is already POST on both arms of this round (round 1 shipped it),
which is exactly why the RCWA path can serve as this round's INDEPENDENT
REFERENCE for a PMM answer.
"""
from __future__ import annotations

import inspect
import json
import os
import platform
import sys

import numpy as np

_C = complex

WL = 0.6e-6
PX = 0.5e-6
DEPTH = 0.2e-6


# --------------------------------------------------------------------- stamps
def arm_stamp():
    """Which arm, which build, which thread pinning -- in every JSON."""
    import lumenairy
    from lumenairy.elements.pmm import twod as ptw
    from lumenairy.elements.rcwa import _core as rc
    src_pmm = _src(ptw._sqrt_decay)
    return dict(
        lumenairy_file=lumenairy.__file__,
        arm=("pre" if "r.real == 0" in src_pmm else "post"),
        pmm_twod_has_exact_pin=("r.real == 0" in src_pmm),
        pmm_twod_has_conj=("conj" in src_pmm),
        pmm_cut_band_rel=getattr(ptw, "_CUT_BAND_REL", None),
        rcwa_cut_band_rel=getattr(rc, "_CUT_BAND_REL", None),
        jax_copies=_jax_copy_state(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.system(),
        node=platform.node(),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
    )


def _src(fn):
    try:
        return inspect.getsource(fn)
    except Exception:                                    # pragma: no cover
        return ""


def _jax_copy_state():
    """The three JAX twins define ``_sqrt_decay`` in a NESTED scope, so the
    only way to read their branch test is the enclosing module's source."""
    import importlib
    out = {}
    for mod in ("_jax_twod", "_jax_stack2d", "_jax_twod_jones"):
        try:
            m = importlib.import_module("lumenairy.elements.pmm." + mod)
            src = inspect.getsource(m)
            out[mod] = dict(exact_pin=("on_cut = r.real == 0" in src),
                            band=("_CUT_BAND_REL" in src),
                            conj=("jnp.conj(r)" in src))
        except Exception as exc:                          # pragma: no cover
            out[mod] = dict(error=repr(exc))
    return out


def require_local_tree():
    """HARD GUARD (inherited from the round-1 verification, which measured one
    census against an unrelated checkout on another drive before it existed):
    ``python some/dir/probe.py`` puts the SCRIPT's directory on ``sys.path``,
    never the working directory."""
    import lumenairy
    got = os.path.abspath(os.path.dirname(os.path.dirname(lumenairy.__file__)))
    want = os.path.abspath(os.getcwd())
    if got != want:
        raise SystemExit(
            "REFUSED: imported lumenairy from %s but the working directory is "
            "%s.  Re-run with PYTHONPATH=. from the tree you mean to measure."
            % (got, want))
    return got


def dump(path, payload):
    payload = dict(payload)
    payload["_stamp"] = arm_stamp()
    with open(path, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_default)
    print("wrote %s" % path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ------------------------------------------------------------------ geometry
def pillar_cell(S=32, host=2.25, pillar=6.0, half=0.25, eps_im=0.0):
    """Scalar pillar-in-host pixel grid for the HYBRID entry points.

    ``host = n_substrate**2`` makes the layer background coincide with the
    substrate EXACTLY -- the state round 1 showed puts a layer mode on top of a
    region mode on the RCWA side.
    """
    e = np.full((S, S), host + 1j * eps_im, dtype=complex)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < half) & (np.abs(x[None, :]) < half)
    e[m] = pillar + 1j * eps_im
    return e


def stag_cell(host=2.25, pillar=6.0, eps_im=0.0):
    """Square 4 x 4 per-SEGMENT grid for the PURE STAGGERED entries (the grid
    IS the wall layout; it must be square)."""
    e = np.full((4, 4), host + 1j * eps_im, dtype=complex)
    e[1:3, 1:3] = pillar + 1j * eps_im
    return e


def stag_tensor_cell(host=2.25, no=1.5, ne=1.7, twist=0.7, eps_im=0.0,
                     oop=0.0):
    """(4, 4, 3, 3) block-form tensor cell for ``pmm_jones_2d_staggered``.

    ``oop != 0`` adds the xz/zx entries that route the solve to the 4q^2
    OUT-OF-PLANE generator, whose forward set is chosen by
    ``_select_forward_flux`` and NOT by any square-root branch test.
    """
    tc = np.zeros((4, 4, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = host + 1j * eps_im
    no2, ne2 = no ** 2, ne ** 2
    c0, s0 = np.cos(twist), np.sin(twist)
    m = np.zeros((4, 4), bool)
    m[1:3, 1:3] = True
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0 + 1j * eps_im
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0 + 1j * eps_im
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2 + 1j * eps_im
    if oop:
        tc[m, 0, 2] = tc[m, 2, 0] = oop
        tc[m, 1, 2] = tc[m, 2, 1] = 0.5 * oop
    return tc


def tensor_cell(S=32, host=2.25, no=1.5, ne=1.7, twist=0.7, eps_im=0.0):
    """(S, S, 3, 3) in-plane tensor pixel grid for ``pmm_jones_2d`` / the
    hybrid stack's ``eps_tensor_cell``."""
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = host + 1j * eps_im
    no2, ne2 = no ** 2, ne ** 2
    c0, s0 = np.cos(twist), np.sin(twist)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0 + 1j * eps_im
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0 + 1j * eps_im
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2 + 1j * eps_im
    return tc


# --------------------------------------------------------------------- oracle
def closure_eff(res):
    """``sum R + sum T - 1`` -- the independent oracle for a lossless cell
    under one incident polarization.  Needs no reference solve."""
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 1.0)


def closure_jones(res):
    """``sum R + sum T - 2`` -- the two-polarization Jones return."""
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 2.0)


def rt_vec(res):
    """Flat per-order (R, T) vector, for a bit-identity / motion comparison."""
    return np.concatenate([np.asarray(res[1], dtype=float).ravel(),
                           np.asarray(res[2], dtype=float).ravel()])


def motion(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ------------------------------------------------- PMM eigenvalue collection
class PMMEigSpy:
    """Collect every eigenvalue array the PMM copies of ``_sqrt_decay`` are
    handed, together with the band ratio the ROUND-2 change computes.

    Patches the module-level ``_sqrt_decay`` of ``pmm.twod`` (the live NumPy
    copy) so the arrays are seen whichever arm is installed.  The wrapper
    DELEGATES, so nothing about the solve changes.
    """

    def __init__(self, module="lumenairy.elements.pmm.twod"):
        self.calls = []
        self.module = module
        self._saved = None

    def __enter__(self):
        import importlib
        mod = importlib.import_module(self.module)
        orig = mod._sqrt_decay
        calls = self.calls

        def wrapped(x):
            xa = np.asarray(x, dtype=complex)
            r = np.sqrt(xa)
            scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
            ratio = np.abs(r.real) / scale
            own = np.abs(r.real) / np.maximum(np.abs(r), 1e-300)
            neg = r.imag < 0
            oncut = (xa.real < 0) & (np.abs(xa.imag) <= 1e-6 * np.abs(xa.real))
            inband = neg & (ratio <= 1e-8)
            calls.append(dict(
                n=int(r.size),
                n_neg_imag=int(np.sum(neg)),
                n_on_cut=int(np.sum(oncut)),
                n_exact_zero_real=int(np.sum(r.real == 0)),
                n_incoming_after_exact_pin=int(
                    np.sum(neg & (r.real != 0) & oncut)),
                n_in_band=int(np.sum(inband)),
                max_band_ratio_neg=(float(np.max(ratio[neg])) if np.any(neg)
                                    else None),
                max_own_ratio_in_band=(float(np.max(own[inband]))
                                       if np.any(inband) else None),
                max_abs_r=float(np.max(np.abs(r))) if r.size else 0.0,
                min_abs_r=float(np.min(np.abs(r))) if r.size else 0.0,
            ))
            return orig(x)

        self._saved = (mod, orig)
        mod._sqrt_decay = wrapped
        return self

    def __exit__(self, *a):
        mod, fn = self._saved
        mod._sqrt_decay = fn
        return False

    def summary(self):
        if not self.calls:
            return dict(n_calls=0)
        rat = [c["max_band_ratio_neg"] for c in self.calls
               if c["max_band_ratio_neg"] is not None]
        return dict(
            n_calls=len(self.calls),
            modes=int(sum(c["n"] for c in self.calls)),
            on_cut=int(sum(c["n_on_cut"] for c in self.calls)),
            exact_zero_real=int(sum(c["n_exact_zero_real"]
                                    for c in self.calls)),
            incoming_after_exact_pin=int(
                sum(c["n_incoming_after_exact_pin"] for c in self.calls)),
            in_band=int(sum(c["n_in_band"] for c in self.calls)),
            max_band_ratio_neg=(max(rat) if rat else None),
        )


class PMMEigSpyRaw:
    """Keep the RAW eigenvalue arrays the PMM copy is handed, unscored.

    ``PMMEigSpy`` summarises per call; the band-scale census needs the arrays
    themselves so both candidate shapes can be scored from one collection.
    """

    def __init__(self, module="lumenairy.elements.pmm.twod"):
        self.arrays = []
        self.module = module
        self._saved = None

    def __enter__(self):
        import importlib
        mod = importlib.import_module(self.module)
        orig = mod._sqrt_decay
        arrays = self.arrays

        def wrapped(x, *a, **kw):
            arrays.append(np.asarray(x, dtype=complex).copy())
            return orig(x, *a, **kw)

        self._saved = (mod, orig)
        mod._sqrt_decay = wrapped
        return self

    def __exit__(self, *a):
        mod, fn = self._saved
        mod._sqrt_decay = fn
        return False


# ----------------------------------------------------------- the PRE/POST A/B
class PreSqrtDecayPMM:
    """Reinstate the PRE (exact-zero-pin) body in the NumPy PMM copies, inside
    whichever tree is installed.  Used for the WITHIN-ONE-INTERPRETER A/B so a
    thread ladder can be run on both arms in one process.

    The body below is the shipped pre-round-2 source, re-typed; every probe
    that uses it also runs the plain cross-tree A/B, and the two agree.
    """

    _MODULES = ("lumenairy.elements.pmm.twod",
                "lumenairy.elements.pmm.twod_staggered")

    @staticmethod
    def body(x):
        r = np.sqrt(np.asarray(x, dtype=complex))
        on_cut = r.real == 0
        return np.where(on_cut & (r.imag < 0), -r, r)

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib
        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = PreSqrtDecayPMM.body
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


class PostSqrtDecayPMM(PreSqrtDecayPMM):
    """Install the ROUND-2 (relative band + ``conj`` flip) body in the NumPy
    PMM copies, inside whichever tree is installed.

    The mirror image of :class:`PreSqrtDecayPMM`: together they make the A/B
    runnable on EITHER arm in one interpreter, which is what lets a thread
    ladder be taken on both arms at one thread setting.
    """

    @staticmethod
    def body(x, xp=None, band=1e-8):
        r = np.sqrt(np.asarray(x, dtype=complex))
        scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
        on_cut = np.abs(r.real) <= band * scale
        return np.where(on_cut & (r.imag < 0), np.conj(r), r)

    def __enter__(self):
        import importlib
        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = PostSqrtDecayPMM.body
        return self


class PreSqrtDecayRCWA:
    """Reinstate the PRE-ROUND-1 (exact-zero-pin) body in every module that
    binds the shared ``_sqrt_decay``, so the RCWA-side arms (X-1, the band's
    RCWA population) can be A/B-ed inside one interpreter.

    The within-interpreter A/B is used deliberately: a cross-tree run of an
    RCWA entry point would also carry every unrelated change between the two
    trees, which is exactly the mis-attribution the round-1 verification warns
    about in its section 8.
    """

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.stack",
                "lumenairy.elements.berreman",
                "lumenairy.elements.pmm.twod")

    @staticmethod
    def body(x, xp=None, band=1e-8):
        from lumenairy.backend.array import array_namespace
        if xp is None:
            xp = array_namespace(x)
        x = xp.asarray(x).astype(complex)
        r = xp.sqrt(x)
        on_cut = r.real == 0
        return xp.where(on_cut & (r.imag < 0), -r, r)

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib
        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = PreSqrtDecayRCWA.body
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False

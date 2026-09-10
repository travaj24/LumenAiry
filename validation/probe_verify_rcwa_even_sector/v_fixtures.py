"""Fixtures and helpers shared by the independent verification probes for the
RCWA modal branch-cut fix (``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md``).

Every probe in this directory runs UNCHANGED against the POST tree
(``C:/tmp/lum_vrcwa``, the wave2/pmm2d tip carrying the fix) and against the
PRE tree (``C:/tmp/lum_vrcwa_pre``, the fix's branch point ``48c8747``), so the
arm is decided by which ``lumenairy`` is importable and never by a flag in the
probe.  ``arm_stamp()`` records which one actually got imported, so no reading
can be mis-attributed.

The fixtures are BUILT HERE and not copied from the fix's own probes: the point
of the exercise is to re-measure, not to re-read.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np

_C = complex


# --------------------------------------------------------------------- stamps
def arm_stamp():
    """Which tree, which build, which thread pinning -- recorded in every JSON."""
    import lumenairy
    from lumenairy.elements.rcwa import _core as rc
    band = getattr(rc, "_CUT_BAND_REL", None)
    src = ""
    try:
        import inspect
        src = inspect.getsource(rc._sqrt_decay)
    except Exception:                                    # pragma: no cover
        pass
    return dict(
        lumenairy_file=lumenairy.__file__,
        arm=("post" if band is not None else "pre"),
        cut_band_rel=band,
        sqrt_decay_has_exact_pin=("r.real == 0" in src),
        sqrt_decay_has_conj=("conj" in src),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.system(),
        node=platform.node(),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
    )


def require_local_tree():
    """HARD GUARD.  ``python some/dir/probe.py`` puts the SCRIPT's directory on
    ``sys.path``, never the working directory, so a probe run without an
    explicit ``PYTHONPATH=.`` silently imports whatever ``lumenairy`` is
    installed -- measured once during this verification: a census attributed to
    the POST tree had in fact been taken against an unrelated development
    checkout on another drive.  Refuse to produce a number under that
    ambiguity.
    """
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


# ------------------------------------------------------------------- geometry
_P = 0.5e-6
_WL = 0.6e-6
_DEPTH = 0.2e-6


def uniaxial_cell(S=48, twist=0.7, no=1.5, ne=1.7, bg=2.25, half=0.25,
                  eps_im=0.0):
    """The anisotropic block-in-background cell.

    ``bg`` is the BACKGROUND permittivity; the block is a uniaxial material
    whose optic axis lies in the plane, rotated by ``twist``.  ``eps_im`` adds
    a uniform imaginary part to every tensor entry's diagonal (the loss
    ladder).  With ``bg = no**2`` and ``n_substrate = no`` the layer background,
    the block's ``zz`` entry and the substrate permittivity coincide EXACTLY --
    that is the state the fix is about.
    """
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = bg + 1j * eps_im
    no2, ne2 = no ** 2, ne ** 2
    c0, s0 = np.cos(twist), np.sin(twist)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < half) & (np.abs(x[None, :]) < half)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0 + 1j * eps_im
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0 + 1j * eps_im
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2 + 1j * eps_im
    return tc


def scalar_cell(S=48, bg=2.25, blk=4.0, half=0.25, eps_im=0.0):
    """Isotropic scalar block-in-background cell (the ``rcwa_efficiency_2d``
    path, which never builds a tensor)."""
    e = np.full((S, S), bg + 1j * eps_im, dtype=complex)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < half) & (np.abs(x[None, :]) < half)
    e[m] = blk + 1j * eps_im
    return e


def closure_defect_jones(res):
    """``sum R + sum T - 2`` for a two-polarization Jones return.  A provably
    lossless cell conserves energy EXACTLY at any truncation under the Laurent
    rule, so this is an oracle whose error floor is the arithmetic."""
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 2.0)


def closure_defect_eff(res):
    """``sum R + sum T - 1`` for a single-polarization efficiency return.

    Both ``rcwa_efficiency_1d`` and ``rcwa_efficiency_2d`` return
    ``(orders, R, T)`` -- entry 0 is the ORDER INDEX, not a power.
    """
    return float(np.sum(np.asarray(res[1])) + np.sum(np.asarray(res[2])) - 1.0)


# --------------------------------------------------------------- band scoring
def band_ratio(r):
    """The exact quantity ``_sqrt_decay`` thresholds: ``|Re(r)| / max(max|r|,1)``.

    ``r`` is the PRINCIPAL square root of the modal eigenvalue array.
    """
    r = np.asarray(r)
    scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    return np.abs(r.real) / scale, scale


def principal_root(lam2):
    return np.sqrt(np.asarray(lam2, dtype=complex))


# ------------------------------------------------------- eigenvalue collection
class EigSpy:
    """Collect every layer eigenvalue array the RCWA solve produces.

    ``_eig_for`` is imported BY NAME into ``rcwa.oned`` as well as living in
    ``_core``, so BOTH bindings are patched -- patching only ``_core`` makes the
    probe silently blind to the whole 1-D fast path (measured: 0 eigenvalue
    arrays collected for ``rcwa_efficiency_1d``).
    """

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned")

    def __init__(self):
        self.seen = []
        self._saved = []

    def __enter__(self):
        import importlib

        from lumenairy.elements.rcwa import _core as rc
        orig = rc._eig_for
        seen = self.seen

        def factory(xp):
            base = orig(xp)

            def wrapped(A):
                w, v = base(A)
                seen.append(np.asarray(w).astype(complex).copy())
                return w, v
            return wrapped

        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_eig_for"):
                self._saved.append((mod, mod._eig_for))
                mod._eig_for = factory
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._eig_for = fn
        return False


class InterfaceSpy:
    """Collect ``cond(a+b)``, the smallest singular direction and its inverse
    participation ratio at every RCWA interface mode-match the solve performs.

    ``_interface_smatrix`` is imported BY NAME into ``rcwa.oned``, ``rcwa.twod``
    and ``rcwa.stack``, so all four module bindings are patched -- patching only
    ``_core`` would silently see nothing on the 2-D path.
    """

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.twod",
                "lumenairy.elements.rcwa.stack")

    def __init__(self):
        self.rows = []
        self._saved = []

    def __enter__(self):
        import importlib

        from lumenairy.elements.rcwa import _core as rc
        orig = rc._interface_smatrix
        rows = self.rows

        def wrapped(Wa, Va, Wb, Vb):
            try:
                a = np.linalg.solve(np.asarray(Wb), np.asarray(Wa))
                b = np.linalg.solve(np.asarray(Vb), np.asarray(Va))
                apb = a + b
                sv = np.linalg.svd(apb, compute_uv=True)
                U, s, Vh = sv
                smax = float(s[0])
                smin = float(s[-1])
                cond = smax / smin if smin > 0 else float("inf")
                nv = np.conj(Vh[-1])
                w = np.abs(nv) ** 2
                w = w / np.sum(w)
                ipr = float(1.0 / np.sum(w ** 2))
                tiny = int(np.sum(s < 1e-10 * smax))
                rows.append(dict(n=int(apb.shape[0]), cond=cond, smin=smin,
                                 smax=smax, ipr=ipr, n_tiny=tiny,
                                 top_mode=int(np.argmax(w)),
                                 top_weight=float(np.max(w))))
            except Exception as exc:                      # pragma: no cover
                rows.append(dict(error=repr(exc)))
            return orig(Wa, Va, Wb, Vb)

        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_interface_smatrix"):
                self._saved.append((mod, mod._interface_smatrix))
                mod._interface_smatrix = wrapped
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._interface_smatrix = fn
        return False


class PreSqrtDecay:
    """Run the PRE-FIX ``_sqrt_decay`` body inside the POST tree.

    Used only for the WITHIN-ONE-INTERPRETER A/B, where the cross-tree run
    would otherwise also carry the unrelated ``pmm/stack.py`` round-3 change
    that sits between ``48c8747`` and the merge.  The body below is the
    ``48c8747`` source, re-typed from ``git show 48c8747:lumenairy/elements/
    rcwa/_core.py``; ``verify_matches_pre_tree`` in ``v4_scope.py`` checks it
    against the real PRE tree rather than trusting the transcription.
    """

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.twod",
                "lumenairy.elements.rcwa.stack",
                "lumenairy.elements.berreman")

    @staticmethod
    def body(x):
        from lumenairy.backend.array import array_namespace
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
                mod._sqrt_decay = PreSqrtDecay.body
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False

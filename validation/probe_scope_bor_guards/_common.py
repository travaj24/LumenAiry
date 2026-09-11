"""Shared harness for the BOR multilayer-guard SCOPING probes (2026-09-12).

MEASUREMENT ONLY -- nothing here edits ``lumenairy/``.  Two instruments:

* :func:`inv_census` -- a context manager that monkeypatches
  ``numpy.linalg.inv`` / ``numpy.linalg.solve`` for the duration of a solve
  and records, for every call made from inside ``lumenairy/elements/bor/``,
  the SITE (caller file:line), the block size, the equilibrated reciprocal
  1-condition and the equilibrated inverse residual.  It uses the library's
  OWN instruments (``rcwa/_core._rcond_1_equilibrated`` and
  ``_equilibrated_inverse_residual``) so the populations are directly
  comparable with the Cartesian censuses.
* :func:`orientation_audit` -- per-mode forward-orientation diagnostics of one
  layer's modal basis: the oriented ``q``, the post-normalization z-flux (its
  SIGN is the orientation outcome and its MAGNITUDE says whether the flux
  decision was made on signal or on noise), the propagating classifier's
  ratio ``|Im q| / |Re q|`` against the shipped ``1e-9`` bar, and the
  fallback-branch population.

Every probe pins the tree from ``lumenairy.__file__``.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback

TREE = r"C:\tmp\lum_borscope"
# The worktree MUST win over any installed lumenairy (there is one on D:).
if TREE not in sys.path:
    sys.path.insert(0, TREE)
if os.path.isdir("/mnt/c/tmp/lum_borscope") and "/mnt/c/tmp/lum_borscope" not in sys.path:
    sys.path.insert(0, "/mnt/c/tmp/lum_borscope")

import numpy as np  # noqa: E402


def pin_tree():
    import lumenairy
    p = os.path.abspath(lumenairy.__file__)
    norm = p.replace("\\", "/").lower()
    if "/tmp/lum_borscope/lumenairy/" not in norm:
        raise SystemExit(f"WRONG TREE: {p}")
    return p


def build_tag():
    import platform

    import numpy as _np
    return dict(
        python=sys.version.split()[0],
        platform=platform.system(),
        numpy=_np.__version__,
        threads_env={k: os.environ.get(k) for k in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS")},
    )


def outdir():
    # the probe directory is THIS file's own directory -- the same physical
    # folder from Windows (C:\tmp\lum_borscope) and from WSL
    # (/mnt/c/tmp/lum_borscope), which a hardcoded Windows path is not.
    return os.path.dirname(os.path.abspath(__file__))


def dump(name, payload):
    payload = dict(payload)
    payload["_build"] = build_tag()
    payload["_tree"] = pin_tree()
    p = os.path.join(outdir(), name)
    with open(p, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, default=_jsonable)
    print("WROTE", p)
    return p


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------- #
#  Instrument 1: explicit-inverse census inside lumenairy/elements/bor/        #
# --------------------------------------------------------------------------- #
def _instruments():
    from lumenairy.elements.rcwa._core import (
        _equilibrated_inverse_residual,
        _rcond_1_equilibrated,
    )
    return _rcond_1_equilibrated, _equilibrated_inverse_residual


def _site_of(depth_skip=2):
    st = traceback.extract_stack()
    for fr in reversed(st[:-depth_skip]):
        fn = fr.filename.replace("\\", "/")
        if "_common.py" in fn:
            continue
        short = fn.split("/lumenairy/")[-1] if "/lumenairy/" in fn else fn
        return f"{short}:{fr.lineno}:{fr.name}"
    return "?"


@contextlib.contextmanager
def inv_census(records, *, residual=True, only_bor=True):
    """Record every ``np.linalg.inv`` / ``np.linalg.solve`` call made from
    inside ``lumenairy/elements/bor/``."""
    rcond_f, resid_f = _instruments()
    real_inv = np.linalg.inv
    real_solve = np.linalg.solve

    def _rec(kind, A, X):
        site = _site_of()
        if only_bor and "elements/bor/" not in site.replace("\\", "/"):
            return
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return
        try:
            Xi = X if X is not None else real_inv(A)
            rc = float(rcond_f(A, Xi))
        except Exception:                          # noqa: BLE001
            rc = float("nan")
        rs = None
        if residual:
            try:
                rs = float(resid_f(A))
            except Exception:                      # noqa: BLE001
                rs = float("nan")
        records.append(dict(site=site, kind=kind, n=int(A.shape[0]),
                            rcond=rc, resid=rs))

    def inv(A):
        X = real_inv(A)
        try:
            _rec("inv", A, X)
        except Exception:                          # noqa: BLE001
            pass
        return X

    def solve(A, B):
        X = real_solve(A, B)
        try:
            _rec("solve", A, None)
        except Exception:                          # noqa: BLE001
            pass
        return X

    np.linalg.inv = inv
    np.linalg.solve = solve
    try:
        yield records
    finally:
        np.linalg.inv = real_inv
        np.linalg.solve = real_solve


# --------------------------------------------------------------------------- #
#  Instrument 2: per-mode forward-orientation audit                            #
# --------------------------------------------------------------------------- #
_PROP_BAR = 1e-9        # the shipped classifier bar |Im q| < 1e-9 |Re q|
_FALLBACK_BAR = 1e-10   # the shipped flux-normalizer fallback |P| <= 1e-10 fnrm


def flux_and_norm(L, *, sem):
    """z-Poynting flux and r-dr field norm per mode, from the RETURNED
    (normalized) W, V with the layer's own two-grid weights.  The RATIO
    ``|flux| / fnrm`` is scale-invariant and therefore equals the RAW
    ``|P| / fnrm`` the normalizer screened on."""
    W, V = L["W"], L["V"]
    wq_f = np.real(np.asarray(L["wq_face"]))
    wq_n = np.real(np.asarray(L["wq_node"]))
    if sem:
        n1 = L["n1"]
        n0p = W.shape[0] - n1
        flux = np.real(np.sum(W[:n1] * np.conj(V[n0p:]) * wq_f[:, None], axis=0)
                       - np.sum(W[n1:] * np.conj(V[:n0p]) * wq_n[:, None],
                                axis=0))
        fnrm = (np.sum(np.abs(W[:n1]) ** 2 * wq_f[:, None], axis=0)
                + np.sum(np.abs(W[n1:]) ** 2 * wq_n[:, None], axis=0))
    else:
        N = len(wq_f)
        flux = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                       - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
        fnrm = (np.sum(np.abs(W[:N]) ** 2 * wq_f[:, None], axis=0)
                + np.sum(np.abs(W[N:]) ** 2 * wq_n[:, None], axis=0))
    return flux, fnrm


def mode_table(L, *, sem):
    """(q, flux, relflux, rho) for one layer dict."""
    q = np.asarray(L["q"])
    flux, fnrm = flux_and_norm(L, sem=sem)
    rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
    rho = np.abs(q.imag) / np.maximum(np.abs(q.real), 1e-300)
    return q, flux, rel, rho


def orientation_audit(L, *, sem=False):
    q, flux, rel, rho = mode_table(L, sem=sem)
    prop = rho < _PROP_BAR
    noise_oriented = prop & (rel <= _FALLBACK_BAR)
    backward_prop = prop & (flux < 0.0) & (~noise_oriented)
    backward_evan = (~prop) & (q.imag < 0.0)
    return dict(
        n_modes=int(q.size), n_prop=int(prop.sum()), n_evan=int((~prop).sum()),
        n_backward_prop=int(backward_prop.sum()),
        n_backward_evan=int(backward_evan.sum()),
        n_noise_oriented=int(noise_oriented.sum()),
        prop_ratio_max=float(rho[prop].max()) if prop.any() else None,
        evan_ratio_min=float(rho[~prop].min()) if (~prop).any() else None,
        relflux_prop_min=float(rel[prop].min()) if prop.any() else None,
        relflux_prop_max=float(rel[prop].max()) if prop.any() else None,
        q_sign_key=_sign_key(q), q_hash=_arr_hash(q),
    )


def _sign_key(q):
    s = np.where(q.real != 0, np.sign(q.real), np.sign(q.imag))
    return "".join("+" if v >= 0 else "-" for v in s)


def _arr_hash(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, dtype=complex)).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
#  Fixture builders                                                            #
# --------------------------------------------------------------------------- #
def make_stack(spec):
    from lumenairy import BORStack
    kw = dict(n_substrate=spec.get("n_sub", 1.41),
              n_superstrate=spec.get("n_sup", 1.41),
              N=spec.get("N", 120), basis=spec.get("basis", "fd"),
              degree=spec.get("degree", 8))
    if spec.get("eps_seg"):
        kw["elements_per_segment"] = spec["eps_seg"]
    s = BORStack(spec["Rbig"], spec["m"], **kw)
    for lay in spec["layers"]:
        thk = lay["t"]
        if "eps" in lay:
            s.add_layer(thk, eps=complex(lay["eps"]))
        elif "rings" in lay:
            s.add_layer(thk, rings=tuple(lay["rings"]))
        elif "segments" in lay:
            s.add_layer(thk, segments=[(float(r), complex(e))
                                       for r, e in lay["segments"]])
        else:
            raise ValueError(f"bad layer spec {lay}")
    s.set_source(k0=spec["k0"])
    return s


def solve_quiet(s, **kw):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = s.solve(**kw)
    return res, [str(x.message)[:200] for x in w]


def closure(res):
    """max |R + T - 1| over incident orders (lossless stacks only)."""
    e = np.asarray(res["energy"])
    return float(np.max(np.abs(e - 1.0))) if e.size else float("nan")

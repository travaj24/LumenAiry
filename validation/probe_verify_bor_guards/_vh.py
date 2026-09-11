"""Shared harness for the INDEPENDENT VERIFICATION of the 5.45.1 BOR
multilayer guards (branch ``verify/bor-multilayer-guards``).

Every probe in this directory imports this module.  It does three things and
nothing else:

1. **Pins the tree.**  ``tree()`` returns which clone ``lumenairy`` was
   actually imported from and ``require_tree()`` refuses to measure anything
   if it is not the one the caller asked for on the command line.  A ``pip
   -e`` install points at a different clone and would silently make every
   "post" number a "pre" number (or vice versa).
2. **Reads the arithmetic back.**  ``arm()`` reports the OpenBLAS kernel that
   actually loaded (from ``threadpoolctl``, never inferred from
   ``OPENBLAS_CORETYPE``: ``ZEN`` silently aliases Haswell on this host) and
   the thread count, so every JSON row carries the arm it was taken on.
3. **Hashes answers.**  ``hash_arrays()`` is the SHA-256 of the exact
   IEEE-754 bytes of an answer, which is the only bit-identity claim that is
   not an eyeball.

The fixture builders live here too, so the pre-build tree and the post-build
tree are driven by ONE description of each fixture.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# tree / arm provenance
# ---------------------------------------------------------------------------


def tree():
    import lumenairy
    return str(pathlib.Path(lumenairy.__file__).resolve().parents[1])


def require_tree(want):
    """``want`` is 'pre' or 'post'; the tree must be the matching clone."""
    t = tree().replace("\\", "/").lower()
    if want == "pre":
        ok = t.endswith("lum_vbor_pre")
    elif want == "post":
        ok = t.endswith("lum_vbor")
    else:
        raise SystemExit("build must be 'pre' or 'post', got %r" % (want,))
    if not ok:
        raise SystemExit(
            "TREE MISMATCH: asked for build=%r but lumenairy came from %s.  "
            "Run with PYTHONPATH pointing at the right clone and a cwd that "
            "does not contain another one." % (want, tree()))
    return tree()


def kernel():
    try:
        import threadpoolctl
        for d in threadpoolctl.threadpool_info():
            if d.get("internal_api") == "openblas":
                return str(d.get("architecture")), int(d.get("num_threads"))
    except Exception:
        pass
    return "unknown", -1


def arm():
    arch, nthr = kernel()
    import numpy
    import scipy
    return dict(
        platform=("wsl" if sys.platform.startswith("linux") else "win"),
        python=sys.version.split()[0], numpy=numpy.__version__,
        scipy=scipy.__version__,
        requested_coretype=os.environ.get("OPENBLAS_CORETYPE", "(unset)"),
        loaded_kernel=arch, blas_threads=nthr,
        env_threads=os.environ.get("OPENBLAS_NUM_THREADS", "(unset)"),
        tree=tree())


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def hash_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dump(path, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="cp1252", errors="replace") as f:
        json.dump(obj, f, indent=1, default=_jdefault)
    print("wrote", p)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.complexfloating,)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ---------------------------------------------------------------------------
# THE FIXTURE BATTERY -- built HERE, not copied from the build's probes.
# ---------------------------------------------------------------------------

def _prof_step(edges_eps):
    """A callable radial step profile from ((r_out, eps), ...)."""
    def f(r):
        out = np.full_like(np.asarray(r, dtype=float), edges_eps[-1][1],
                           dtype=complex)
        prev = 0.0
        for r_out, e in edges_eps:
            out[(np.asarray(r) > prev) & (np.asarray(r) <= r_out)] = e
            prev = r_out
        return out
    return f


def bor_fixtures():
    """>= 40 BORStack fixtures spanning both bases, m = 0/1/2/5, lossless and
    lossy, ring gratings, explicit segment layers, the two uniform
    coincidences (a layer whose eps equals the region it sits in, and a layer
    whose eps equals its neighbour's), thin rings, tapers at 4..64 slices, and
    the anisotropic path.  Each entry is (name, builder) where builder returns
    a configured, unsolved BORStack.

    Deliberately contains NO near-cutoff fixture: those are the population the
    band change is entitled to move and they are measured separately.
    """
    from lumenairy.elements.bor.bor_stack import BORStack
    F = []

    def add(name, fn):
        F.append((name, fn))

    # --- ring-grating pair, both bases x m x k0 -------------------------
    for basis in ("fd", "sem"):
        for m in (0, 1, 2, 5):
            for k0 in (0.8, 2.0, 3.5):
                def mk(basis=basis, m=m, k0=k0):
                    s = BORStack(Rbig=6.0, m=m, N=120, n_superstrate=1.4,
                                 n_substrate=1.4, basis=basis, degree=8)
                    s.add_layer(0.5, rings=(1.2, 0.5, 1.8, 1.3))
                    s.add_layer(0.35, eps=2.10)
                    s.set_source(k0=k0)
                    return s
                add("ringpair_%s_m%d_k%g" % (basis, m, k0), mk)

    # --- three-layer with a wall-free spacer ----------------------------
    for basis in ("fd", "sem"):
        for m in (0, 1):
            def mk(basis=basis, m=m):
                s = BORStack(Rbig=8.0, m=m, N=140, n_superstrate=1.0,
                             n_substrate=1.5, basis=basis, degree=8)
                s.add_layer(0.4, segments=[(2.0, 2.25), (8.0, 1.0)])
                s.add_layer(0.7, eps=1.21)                    # wall-free
                s.add_layer(0.4, segments=[(3.0, 2.25), (8.0, 1.0)])
                s.set_source(k0=2.0)
                return s
            add("spacer3_%s_m%d" % (basis, m), mk)

    # --- LOSSY ring layer ----------------------------------------------
    for basis in ("fd", "sem"):
        for imn in (1e-1, 1e-3, 1e-6):
            def mk(basis=basis, imn=imn):
                e = complex(1.8, imn) ** 2
                s = BORStack(Rbig=6.0, m=1, N=120, n_superstrate=1.4,
                             n_substrate=1.4, basis=basis, degree=8)
                s.add_layer(0.5, segments=[(2.4, e), (6.0, 1.69)])
                s.set_source(k0=2.0)
                return s
            add("lossy_%s_im%g" % (basis, imn), mk)

    # --- uniform-equals-REGION coincidence (layer eps == superstrate) ----
    for basis in ("fd", "sem"):
        def mk(basis=basis):
            s = BORStack(Rbig=6.0, m=1, N=120, n_superstrate=1.4,
                         n_substrate=1.4, basis=basis, degree=8)
            s.add_layer(0.6, eps=1.4 ** 2)          # == the superstrate
            s.add_layer(0.4, segments=[(2.0, 2.6), (6.0, 1.96)])
            s.set_source(k0=2.0)
            return s
        add("uni_eq_region_%s" % basis, mk)

    # --- uniform-equals-NEIGHBOUR coincidence ---------------------------
    for basis in ("fd", "sem"):
        def mk(basis=basis):
            s = BORStack(Rbig=6.0, m=1, N=120, n_superstrate=1.0,
                         n_substrate=1.0, basis=basis, degree=8)
            s.add_layer(0.3, eps=2.25)
            s.add_layer(0.3, eps=2.25)              # == its neighbour
            s.add_layer(0.3, segments=[(2.5, 2.25), (6.0, 1.0)])
            s.set_source(k0=2.0)
            return s
        add("uni_eq_neighbour_%s" % basis, mk)

    # --- THIN RING (a narrow annulus the CALLER asked for) --------------
    for basis in ("fd", "sem"):
        for w in (1e-2, 1e-3):
            def mk(basis=basis, w=w):
                s = BORStack(Rbig=6.0, m=1, N=200, n_superstrate=1.0,
                             n_substrate=1.0, basis=basis, degree=8)
                s.add_layer(0.4, segments=[(2.0, 1.0), (2.0 + w * 6.0, 4.0),
                                           (6.0, 1.0)])
                s.set_source(k0=2.0)
                return s
            add("thinring_%s_w%g" % (basis, w), mk)

    # --- TAPER staircases (ordinary geometry, cross-layer walls) --------
    for basis in ("fd", "sem"):
        for ns in (4, 8, 16, 32, 64):
            def mk(basis=basis, ns=ns):
                s = BORStack(Rbig=24.0, m=1, N=240, n_superstrate=1.0,
                             n_substrate=1.0, basis=basis, degree=8)
                for j in range(ns):
                    r = 8.0 + (2.0 - 8.0) * (j + 0.5) / ns
                    s.add_layer(1.2 / ns, segments=[(r, 4.0), (24.0, 1.0)])
                s.set_source(k0=2.0)
                return s
            add("taper_%s_n%d" % (basis, ns), mk)

    # --- ANISOTROPIC (diagonal cylindrical tensor) ----------------------
    for basis in ("fd", "sem"):
        def mk(basis=basis):
            s = BORStack(Rbig=6.0, m=1, N=120, n_superstrate=1.0,
                         n_substrate=1.0, basis=basis, degree=8)
            s.add_layer(0.5, segments=[(2.5, (2.5, 2.1, 2.3)),
                                       (6.0, (1.0, 1.0, 1.0))])
            s.set_source(k0=2.0)
            return s
        add("aniso_%s" % basis, mk)

    # --- hp-refined / graded mesh ---------------------------------------
    for eps_seg, grade in ((4, True), (4, False)):
        def mk(eps_seg=eps_seg, grade=grade):
            s = BORStack(Rbig=6.0, m=1, N=120, n_superstrate=1.0,
                         n_substrate=1.0, basis="sem", degree=8,
                         elements_per_segment=eps_seg, grade=grade)
            s.add_layer(0.5, rings=(1.5, 0.5, 1.9, 1.2))
            s.add_layer(0.3, eps=1.44)
            s.set_source(k0=2.0)
            return s
        add("hp_sem_eps%d_grade%d" % (eps_seg, int(grade)), mk)

    # --- a nm-unit fixture (k0 and Rbig six orders of magnitude away) ---
    for basis in ("fd", "sem"):
        def mk(basis=basis):
            Rb = 6.0e-6
            s = BORStack(Rbig=Rb, m=1, N=120, n_superstrate=1.4,
                         n_substrate=1.4, basis=basis, degree=8)
            s.add_layer(0.5e-6, segments=[(2.4e-6, 3.24), (Rb, 1.69)])
            s.set_source(k0=2.0e6)
            return s
        add("nmunits_%s" % basis, mk)

    return F


def solve_fixture(fn):
    """Solve one fixture; return the answer dict plus derived scalars, or the
    exception class/name if it raises (a REFUSAL is an answer too)."""
    import warnings
    s = fn()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = s.solve()
        except BaseException as e:          # noqa: BLE001 - a refusal is data
            return dict(raised=type(e).__name__, msg=str(e)[:400],
                        warnings=[str(x.message)[:200] for x in w])
        R = np.asarray(res["R"]);  T = np.asarray(res["T"])
        return dict(raised=None,
                    n_channels=int(R.size),
                    hash=hash_arrays(R, T),
                    hash_R=hash_arrays(R), hash_T=hash_arrays(T),
                    closure=float(np.max(np.abs(R + T - 1.0)))
                    if R.size else float("nan"),
                    sumR=float(np.sum(R)), sumT=float(np.sum(T)),
                    warnings=[str(x.message)[:200] for x in w])


def timed(label):
    t0 = time.time()

    class _T:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            print("[%s] %.1f s" % (label, time.time() - t0), flush=True)
    return _T()

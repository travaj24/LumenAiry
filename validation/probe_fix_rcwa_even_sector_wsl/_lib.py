"""Shared arm-stamping for the RCWA even-sector WSL investigation
(``docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md``).

Every probe records WHICH tree answered the import (from
``lumenairy.__file__``, never from a flag) and WHICH build ran it
(interpreter, platform, numpy/scipy versions, the BLAS numpy reports and the
thread caps), so a "both builds" table can never be assembled out of one
build's numbers.

Three trees are legitimate:

* ``C:/tmp/lum_wslfix``   -- this FIX branch's worktree (tip / post-fix).
* ``C:/tmp/lum_wsl_main`` -- a detached worktree at ``9af9376`` = published
  v5.44.0, the tree whose release CI is green.  Read-only reference arm.
* ``C:/tmp/lum_wslfix_pre`` -- a read-only ``git archive`` of ``lumenairy/``
  at the branch point, so the fail-before arm can run without flipping the
  working tree under a concurrent suite run.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np

import lumenairy

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")

_TREES = (
    ("/lum_wslfix_pre/", "prefix(48c8747)"),
    ("/lum_wsl_main/", "main(9af9376)"),
    ("/lum_wslfix/", "branch(fix tip)"),
)


def _blas():
    """The BLAS/LAPACK numpy is actually linked against, as numpy reports it."""
    try:
        cfg = np.show_config(mode="dicts")
    except TypeError:                       # numpy < 1.25 has no dict mode
        return {"note": "numpy.show_config has no dict mode"}
    except Exception as exc:                # pragma: no cover - defensive
        return {"note": "unavailable: %s" % exc}
    out = {}
    for key in ("Build Dependencies",):
        dep = cfg.get(key, {})
        for name in ("blas", "lapack"):
            d = dep.get(name, {})
            if d:
                out[name] = {k: d.get(k) for k in
                             ("name", "version", "detection method",
                              "openblas configuration")}
    return out


def arm():
    p = os.path.abspath(lumenairy.__file__)
    low = p.replace("\\", "/").lower()
    tree = None
    for frag, label in _TREES:
        if frag in low:
            tree = label
            break
    if tree is None:
        raise RuntimeError(
            "probe_fix_rcwa_even_sector_wsl: refusing to run against an "
            "unexpected tree -- lumenairy.__file__ = %s" % p)
    import scipy as _sp
    build = "win" if sys.platform.startswith("win") else "wsl"
    return dict(build=build, tree=tree,
                tag="%s.%s" % (build, tree.split("(")[0]),
                lumenairy=p, version=lumenairy.__version__,
                python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, scipy=_sp.__version__, blas=_blas(),
                threads={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS")})


def dump(name, payload, suffix=""):
    os.makedirs(_RESULTS, exist_ok=True)
    a = arm()
    payload = dict(_arm=a, **payload)
    fn = os.path.join(_RESULTS, "%s%s.%s.json" % (name, suffix, a["tag"]))
    with open(fn, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print("[wrote] " + fn)
    return fn


# ------------------------------------------------------------------ fixture
def even_sector_cell(S=48, twist=0.7, no=1.5, ne=1.7, halfwidth=0.25,
                     eps_bg=2.25):
    """The exact cell ``test_jones_2d_even_sector_matches_full`` builds: an
    ``eps_bg`` background with a centred square of uniaxial material whose
    optic axis is rotated ``twist`` radians in the xy plane.  Centro-symmetric
    about (0, 0), which sits HALF A SAMPLE off grid point 0."""
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = eps_bg
    no2, ne2 = no ** 2, ne ** 2
    c0, s0 = np.cos(twist), np.sin(twist)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < halfwidth) & (np.abs(x[None, :]) < halfwidth)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2
    return tc


P_DEFAULT, WL_DEFAULT = 0.5e-6, 0.6e-6

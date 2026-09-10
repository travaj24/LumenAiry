"""Shared fixtures / arm stamping for the V1 / V2 / O2 FIX branch
(``docs/audits/FIX_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md``).

Every probe asserts which tree it imported from ``lumenairy.__file__`` and
stamps the interpreter, numpy, scipy and the thread caps into its JSON, so a
"both builds" table can never be assembled from one build's numbers.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys

import numpy as np

import lumenairy

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")


def arm():
    """Which LIBRARY is imported -- decided from ``lumenairy.__file__``, never
    from a flag.

    Two trees are legitimate and no others.  ``C:/tmp/lum_slfix`` is the branch
    tip (POST-FIX).  ``C:/tmp/lum_slfix_pre`` is a READ-ONLY ``git archive`` of
    ``lumenairy/`` at the branch point ``4a987e3`` (PRE-FIX), extracted so the
    fail-before arm can run WITHOUT flipping the working tree under a
    concurrent suite run; it is never edited and never imported by anything but
    these probes (``PYTHONPATH=C:/tmp/lum_slfix_pre`` from this directory).
    Both stamp ``arm = 'slfix'`` so the file naming stays uniform -- the
    ``_prefix`` / ``_postfix`` SUFFIX is what separates the arms -- and the
    ``tree`` field records which one actually answered."""
    p = os.path.abspath(lumenairy.__file__)
    low = p.replace("\\", "/").lower()
    if "/lum_slfix_pre/" in low:
        name, tree = "slfix", "prefix(4a987e3)"
    elif "/lum_slfix/" in low:
        name, tree = "slfix", "postfix(branch tip)"
    else:
        raise RuntimeError(
            "probe_fix_slant_anchor_v1v2o2: refusing to run against an "
            "unexpected tree -- lumenairy.__file__ = %s" % p)
    import numpy as _np
    import scipy as _sp
    build = "win" if sys.platform.startswith("win") else "wsl"
    return dict(arm=name, build=build, tag="%s.%s" % (name, build),
                tree=tree, lumenairy=p, version=lumenairy.__version__,
                python=sys.version.split()[0], platform=platform.platform(),
                numpy=_np.__version__, scipy=_sp.__version__,
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


# ------------------------------------------------------------------ metrics
def resid(A, B):
    """Relative Frobenius residual over two same-shaped complex arrays."""
    A = np.asarray(A)
    B = np.asarray(B)
    d = float(np.linalg.norm(A - B))
    s = float(np.linalg.norm(B))
    return d / s if s > 0 else d


def dmax(A, B):
    return float(np.max(np.abs(np.asarray(A) - np.asarray(B))))


def sha(a):
    import hashlib
    x = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(x.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ 1-D fixture
# Deliberately NOT the verification's own numbers where a choice exists: the
# V2 anchor must be derived, not transcribed.
P1D = 0.80e-6            # period            (verify probe used 0.90 um)
WL1D = 0.55e-6           # wavelength        (verify probe used 0.62 um)
D1D = 0.40e-6            # thickness         (verify probe used 0.45 um)
ER, EG = 4.20, 1.45      # ridge / groove    (verify probe used 3.05 / 1.30)
DUTY = 0.45
SHEAR = 0.30             # walk = 0.30 P -- neither a quarter nor a half
NSUP1D, NSUB1D = 1.0, 1.6
DEG1D, NORD1D = 12, 7
TH25 = math.radians(25.0)


def oned(kind, *, ns=1, fac="convection", shear=SHEAR, theta=TH25,
         degree=DEG1D, n_orders=NORD1D, centre=0.5, duty=DUTY,
         thickness=D1D, layer_grids="shared", extra_film=None):
    """A 1-D PMMStack holding either the SHEARED parallelogram (`kind='shear'`)
    or its z-STAIRCASE of `ns` vertical rungs (`kind='stair'`), which is
    LAB-referenced by construction."""
    import warnings

    from lumenairy.elements.pmm.stack import PMMStack
    st = PMMStack(P1D, n_superstrate=NSUP1D, n_substrate=NSUB1D,
                  degree=degree, n_orders=n_orders, factorization=fac,
                  layer_grids=layer_grids)
    if kind == "shear":
        st.add_sheared_grating(thickness, eps_ridge=ER, eps_groove=EG,
                               duty=duty, shear=shear, centre=centre)
    elif kind == "stair":
        st.add_tapered_grating(thickness, eps_ridge=ER, eps_groove=EG,
                               duty_bottom=duty, duty_top=duty, shear=shear,
                               n_slices=ns)
    else:
        raise ValueError(kind)
    if extra_film is not None:
        st.add_layer(extra_film[0], eps=extra_film[1])
    st.set_source(WL1D, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


def jt(st):
    try:
        return np.asarray(st.jones_transmission())
    except Exception:                                    # noqa: BLE001
        return None


def P_of(alpha, walk, wl=WL1D):
    """The anchor factor ``exp(+i k0 alpha W)`` -- the quantity under test."""
    k0 = 2.0 * np.pi / wl
    return np.exp(1j * k0 * np.asarray(alpha) * walk)

"""R7 -- the PUREST D1 measurement: the mortar's OWN error, against analysis.

``python r7_allhost.py [pure nomortar taper]``

Three layers, ALL HOST, each on a DIFFERENT non-uniform element grid.  The
device is a HOMOGENEOUS SLAB, so:

* the exact answer is the ANALYTIC Fresnel/Airy result for ``n_sup | n_h |
  n_sub`` -- no oracle engine, no self-gap, no convergence question;
* a homogeneous layer is EXACTLY representable on ANY element grid (the basis
  contains the constants), so the region solve contributes NO error at any
  ``M``;
* therefore every digit of the deviation is the MORTAR's cross-grid
  projection, and it must be round-off at every wall separation.

That makes this the one fixture on which "the mortar is wrong" is a statement
about the mortar alone.  ``nomortar`` is the attribution control: the same
three all-host layers on ONE shared grid (no mortar at all).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import twod_staggered as _ts
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
T0 = time.time()
PER, WL, TH, PH = 1.2, 0.85, 0.15, 0.0
EPSH, TT = 2.25, 0.06
W0, W2 = (0.21, 0.68), (0.33, 0.79)
YW = (0.27, 0.61)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _airy():
    """Analytic TE/TM reflectance of a homogeneous slab (1 | n_h | 1).

    The SYMMETRIC-slab Airy form ``r = r01 (1 - e^{2 i beta}) /
    (1 - r01^2 e^{2 i beta})`` is used because it needs only the interface
    REFLECTION coefficient (``r12 = -r01`` for equal outer media), so it is
    free of the transmission-coefficient convention that makes ``t = 1 + r``
    true for TE and false for TM.  Lossless with equal outer media, so
    ``T = 1 - R`` exactly."""
    k0 = 2 * np.pi / WL
    n0, nh = 1.0, np.sqrt(EPSH)
    kx = k0 * n0 * np.sin(TH)
    kz0 = np.sqrt((k0 * n0) ** 2 - kx ** 2 + 0j)
    kzh = np.sqrt((k0 * nh) ** 2 - kx ** 2 + 0j)
    d = 3.0 * TT
    ph = np.exp(2j * kzh * d)
    out = {}
    for pol in ("TE", "TM"):
        if pol == "TE":
            r01 = (kz0 - kzh) / (kz0 + kzh)
        else:
            r01 = (nh ** 2 * kz0 - n0 ** 2 * kzh) / (nh ** 2 * kz0
                                                     + n0 ** 2 * kzh)
        r = r01 * (1.0 - ph) / (1.0 - r01 * r01 * ph)
        R = float(abs(r) ** 2)
        out[pol] = (R, 1.0 - R)
    return out


def _stack(delta, M, *, mortar=True):
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    yw = [YW[0] * PER, YW[1] * PER]
    sw = [(0.5 - delta / 2) * PER, (0.5 + delta / 2) * PER]
    walls = ([[W0[0] * PER, W0[1] * PER], sw, [W2[0] * PER, W2[1] * PER]]
             if mortar else [sw, sw, sw])
    for xw in walls:
        st.add_layer(TT, eps=EPSH, x_walls=xw, y_walls=yw)
    st.set_source(WL, theta=TH, phi=PH)
    return st


def _solve(st):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return (float(R[0, p0]), float(R[1, p0]), float(T[0, p0]),
            float(T[1, p0]),
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
            [str(x.message)[:70] for x in w])


DELTAS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6,
          1e-6)


def _run(mortar, key):
    an = _airy()
    _log(f"analytic slab: TE R={an['TE'][0]:.9f} T={an['TE'][1]:.9f}   "
         f"TM R={an['TM'][0]:.9f} T={an['TM'][1]:.9f}")
    out = {"analytic": an}
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False        # this probe MAPS the band
    try:
        for delta in DELTAS:
            rec = {}
            for M in (4, 5, 6, 7):
                try:
                    r0, r1, t0, t1, clo, w = _solve(_stack(delta, M,
                                                           mortar=mortar))
                    # row 0 = incident Ex (TM-ish), row 1 = incident Ey (TE)
                    err = max(abs(r1 - an["TE"][0]), abs(t1 - an["TE"][1]),
                              abs(r0 - an["TM"][0]), abs(t0 - an["TM"][1]))
                    rec[str(M)] = {"err": err, "closure": clo, "R_TE": r1,
                                   "R_TM": r0, "warnings": w}
                except Exception as exc:                # noqa: BLE001
                    rec[str(M)] = {"REFUSED":
                                   f"{type(exc).__name__}: {str(exc)[:100]}"}
            out[f"{delta:g}"] = rec
            _log(f"{key} delta={delta:.0e}: err(M=4..7) = "
                 + " ".join(("REFUSED" if "REFUSED" in rec[str(M)]
                             else f"{rec[str(M)]['err']:.3e}")
                            for M in (4, 5, 6, 7))
                 + "   clo "
                 + " ".join(f"{rec[str(M)].get('closure', float('nan')):.1e}"
                            for M in (4, 5, 6, 7)))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    RES[key] = out


def sec_pure():
    _run(True, "pure")


def sec_nomortar():
    _run(False, "nomortar")


def sec_taper():
    """How narrow a segment does ``add_tapered_pillar`` actually build?"""
    out = {}
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        for n_slices in (2, 4, 8, 16, 32, 64, 128):
            for wtop in (0.5, 0.1, 1e-3, 0.0):
                st = PMM2DStackPure(PER, n_modes=4, n_orders=1,
                                    layer_grids="per-layer")
                try:
                    st.add_tapered_pillar(
                        0.24, eps_pillar=9.0, eps_host=EPSH,
                        x_bounds_bottom=[0.25 * PER, 0.75 * PER],
                        y_bounds_bottom=[0.25 * PER, 0.75 * PER],
                        x_bounds_top=[(0.5 - wtop / 2) * PER,
                                      (0.5 + wtop / 2) * PER],
                        y_bounds_top=[(0.5 - wtop / 2) * PER,
                                      (0.5 + wtop / 2) * PER],
                        n_slices=n_slices)
                    nar = min(float(np.min(np.diff(np.asarray(L["wx"]))))
                              for L in st._layers) / PER
                    out[f"n{n_slices}_w{wtop:g}"] = nar
                except Exception as exc:                # noqa: BLE001
                    out[f"n{n_slices}_w{wtop:g}"] = \
                        f"{type(exc).__name__}: {str(exc)[:60]}"
            _log(f"taper n_slices={n_slices}: narrowest segment / period = "
                 + "  ".join(f"w_top={w:g}: "
                             + (v if isinstance(v, str) else f"{v:.3e}")
                             for w in (0.5, 0.1, 1e-3, 0.0)
                             for v in [out[f"n{n_slices}_w{w:g}"]]))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    RES["taper_widths"] = out


SECTIONS = {"pure": sec_pure, "nomortar": sec_nomortar, "taper": sec_taper}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R_TAG", "")
    path = os.path.join(HERE, f"r7_allhost{('_' + tag) if tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()

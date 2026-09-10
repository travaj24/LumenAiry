"""V6 -- the MECHANISM behind the fff_nv stripe fixture's per-build closure
(Task H.b), re-derived from the library internals rather than from the
builder's probes.

Rebuilds exactly what `rcwa_jones_1d_segments` builds for the stripe fixture
(period 0.7 um, wl 1.0 um, normal incidence, n_sup 1.0 / n_sub 1.5,
segments [(0.5, rot(35 deg, 1.5, 2.3)), (0.5, eps_g I)]) and measures, per
truncation and per groove value:

  * Hermiticity of the Li in-plane operator ``[[Cxx, Cxy], [Cyx, Cyy]]``
    -- max|C - C^H| / max|C|.  If it is Hermitian an energy theorem EXISTS,
    so "no finite-truncation energy theorem" cannot explain the defect.
  * cond(W) and cond([W; V]) of the LAYER eigenproblem -- the amplification
    the modal basis could contribute.
  * cond(a + b) at BOTH interfaces -- the matrix `_interface_smatrix`
    inverts explicitly, i.e. the actual candidate amplifier.
  * the modal DEGENERACY itself: the smallest |lam_layer - kz_region| pair
    gap, and how many layer modes sit within 1e-12 of a region mode.

Then the detuning law: worst closure defect over the ladder vs the relative
detune of the groove, of ``no``, and of ``n_sub``.

    python v6_mech.py <lumenairy-root> [tag]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.rcwa._core import (  # noqa: E402
    _homogeneous_eigenmodes,
    _layer_eigenmodes_tensor,
    _sqrt_decay,
    _tensor_convolutions,
)
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
LADDER = list(range(11, 42, 2))


def rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def internals(M, eps_g, no=1.5, n_sub=1.5, ne=2.3):
    """Everything the 1-D in-plane path builds, at truncation M."""
    N = 2 * M + 1
    orders = np.arange(-M, M + 1)
    er = np.conj(rot(np.deg2rad(35.0), no, ne).astype(complex))
    eg = np.conj(np.diag([eps_g] * 3).astype(complex))
    eps_sup = complex(np.conj(complex(1.0) ** 2))
    eps_sub = complex(np.conj(complex(n_sub) ** 2))
    kx0 = 0.0
    k0 = 2.0 * np.pi / WL
    kx = kx0 + orders * (WL / PX)
    Kx = np.diag(kx.astype(complex))
    Ky = np.zeros((N, N), dtype=complex)
    comp = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1),
            "zz": (2, 2)}
    profiles = {k: np.stack([er[i, j], eg[i, j]]) for k, (i, j) in comp.items()}
    Cxx, Cxy, Cyx, Cyy, EZZ = _tensor_convolutions(profiles, M, "li",
                                                   (0.0, 0.5, 1.0))
    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    C = np.block([[Cxx, Cxy], [Cyx, Cyy]])
    # the REGION modal eigenvalues, formed exactly as _homogeneous_eigenmodes
    # forms them internally, so they are comparable with the layer's `lam`
    lam_ref = _sqrt_decay(-np.concatenate([kz_ref, kz_ref]) ** 2)
    lam_trn = _sqrt_decay(-np.concatenate([kz_trn, kz_trn]) ** 2)
    return dict(C=C, Wl=Wl, Vl=Vl, lam=lam, Wref=Wref, Vref=Vref,
                kz_ref=kz_ref, Wtrn=Wtrn, Vtrn=Vtrn, kz_trn=kz_trn, k0=k0,
                lam_ref=lam_ref, lam_trn=lam_trn)


def cond_apb(Wa, Va, Wb, Vb):
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    return float(np.linalg.cond(a + b))


print("A. operator Hermiticity, layer conditioning, interface conditioning, "
      "modal degeneracy")
print(f"{'eps_g':>8s} {'M':>3s} {'|C-C^H|/|C|':>12s} {'cond(W)':>10s} "
      f"{'cond[W;V]':>11s} {'cond(a+b)sup':>13s} {'cond(a+b)sub':>13s} "
      f"{'min|dlam|':>11s} {'n<1e-12':>8s}")
ROWS = []
for eps_g in (2.25, 2.10):
    for M in (5, 11, 21, 31, 41, 61):
        d = internals(M, eps_g)
        C = d["C"]
        herm = float(np.max(np.abs(C - C.conj().T)) / np.max(np.abs(C)))
        cw = float(np.linalg.cond(d["Wl"]))
        cwv = float(np.linalg.cond(np.vstack([d["Wl"], d["Vl"]])))
        cs = cond_apb(d["Wref"], d["Vref"], d["Wl"], d["Vl"])
        ct = cond_apb(d["Wl"], d["Vl"], d["Wtrn"], d["Vtrn"])
        # layer modal eigenvalues vs the two regions' modal eigenvalues
        reg = np.concatenate([np.asarray(d["lam_ref"]).ravel(),
                              np.asarray(d["lam_trn"]).ravel()])
        gaps = np.abs(np.asarray(d["lam"]).ravel()[:, None] - reg[None, :])
        mn = float(np.min(gaps))
        near = int(np.sum(np.min(gaps, axis=1) < 1e-12))
        ROWS.append(dict(eps_g=eps_g, M=M, herm=herm, cw=cw, cwv=cwv,
                         cs=cs, ct=ct, minlam=mn, near=near))
        print(f"{eps_g:8.4f} {M:3d} {herm:12.3e} {cw:10.3f} {cwv:11.1f} "
              f"{cs:13.4e} {ct:13.4e} {mn:11.3e} {near:8d}")

# ------------------------------------------------------------------ detune
print()
print("B. the detune law: worst |sum R + sum T - 2| over n_orders 11..41")


def worst(eps_g, no=1.5, n_sub=1.5):
    er = rot(np.deg2rad(35.0), no, 2.3)
    eg = np.diag([eps_g] * 3).astype(complex)
    out = 0.0
    for n in LADDER:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], n_sub, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
    return out


DET = {}
print(f"{'r':>10s} {'groove':>12s} {'no':>12s} {'n_sub':>12s} "
      f"{'eps/r':>12s}")
EPSM = float(np.finfo(float).eps)
for r in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
    g = worst(2.25 * (1.0 + r))
    a = worst(2.25, no=1.5 * (1.0 + r))
    s = worst(2.25, n_sub=1.5 * (1.0 + r))
    DET[str(r)] = dict(groove=g, no=a, n_sub=s)
    ref = EPSM / r if r else float("nan")
    print(f"{r:10.0e} {g:12.3e} {a:12.3e} {s:12.3e} {ref:12.3e}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"_out_v6_{TAG}.json"), "w", encoding="cp1252") as fh:
    json.dump({"rows": ROWS, "detune": DET}, fh, indent=1)

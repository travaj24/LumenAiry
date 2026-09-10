"""ROUND 2, probe 7 -- the ANISOTROPIC (liquid-crystal) class, V-5.

The verification's S4.4 blind spot 1: ``_stack_provably_passive`` accepted only
DIAGONAL tensors, so the guard was structurally silent on every off-diagonal or
out-of-plane permittivity -- the entire birefringent / liquid-crystal device
class.  Measured there: the O-11 sliver on an ``eps_xy`` layer read ``R+T`` =
2.183 and only WARNED.

Round 2 accepts a tensor that is EXACTLY HERMITIAN (``eps == eps^H``), which is
lossless and therefore passive with no tolerance and no eigen-solve.  This
probe scores four classes on the same sliver, two-sided:

  1. rotated uniaxial LC director, in-plane  (real symmetric -> Hermitian)
  2. rotated uniaxial LC director, OUT-OF-PLANE (eps_xz/eps_zx -> Hermitian)
  3. GYROTROPIC, eps_xy = -eps_yx = i g       (Hermitian, lossless)
  4. NON-Hermitian off-diagonal (eps_xy = 0.2, eps_yx = 0) -- not provably
     passive by any exact argument, so it must STAY silent

and, as the negative control, the same tensors on a SLIVER-FREE stack.

    python validation/probe_pmmstack_sliver_round2/r7_anisotropic.py [out.json]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

HERE = os.path.dirname(os.path.abspath(__file__))
P, WL, TH = 1.2e-6, 0.85e-6, 0.15
DZ = 0.32e-6 / 4
EH = 2.25
A0, B0 = 0.27865, 0.62505
NO, NE = 1.50, 1.72


def uniaxial(theta, axis="xy"):
    """A rotated uniaxial director: real SYMMETRIC, hence exactly Hermitian."""
    d = np.diag([NE ** 2, NO ** 2, NO ** 2]).astype(complex)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "xy":
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:                                   # xz -> an OUT-OF-PLANE director
        R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return R @ d @ R.T


def gyrotropic(eps=4.0, g=0.35):
    M = np.eye(3, dtype=complex) * eps
    M[0, 1] = 1j * g
    M[1, 0] = -1j * g
    return M


def non_hermitian(eps=4.0, off=0.2):
    M = np.eye(3, dtype=complex) * eps
    M[0, 1] = off
    return M


CLASSES = {
    "lc_in_plane_45deg": uniaxial(np.pi / 4.0, "xy"),
    "lc_out_of_plane_30deg": uniaxial(np.pi / 6.0, "xz"),
    "gyrotropic": gyrotropic(),
    "non_hermitian_epsxy": non_hermitian(),
}


def build(M, delta, degree=14, mf=P * 1e-12):
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=degree,
                  min_feature=mf)
    for (a, b) in [(A0, B0), (A0 - delta, B0 + delta)]:
        st.add_layer(DZ, segments=[(a, EH), (b - a, M), (1.0 - b, EH)])
    st.set_source(WL, theta=TH)
    return st


def run(st, guard=True):
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = st.solve()
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return dict(ok=True, m=o[i], R=np.asarray(R)[1][i],
                    T=np.asarray(T)[1][i], worst=float(np.max(tot)),
                    warned=[str(w.message)[:60] for w in rec])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(ok=False, sliver="NEAR-COINCIDENT-WALL SLIVER" in str(exc),
                    msg=str(exc)[:150])
    finally:
        ps.PMM_SLIVER_GUARD = was


def gap(a, b):
    c = np.intersect1d(a["m"], b["m"])
    ia, ib = np.searchsorted(a["m"], c), np.searchsorted(b["m"], c)
    return float(max(np.abs(a["R"][ia] - b["R"][ib]).max(),
                     np.abs(a["T"][ia] - b["T"][ib]).max()))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r7_anisotropic.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()
    rows = []
    for name, M in CLASSES.items():
        herm = bool(np.array_equal(M, np.conjugate(M).T))
        passive = bool(ps._stack_provably_passive(build(M, 1e-4)))
        row = dict(cls=name, hermitian=herm, provably_passive=passive,
                   sliver=dict(), clean=dict())
        # the SLIVER arm
        ref = run(build(M, 0.0), guard=False)
        for d in (1e-4, 3e-5):
            cur = run(build(M, d), guard=False)
            g = run(build(M, d), guard=True)
            e = gap(cur, ref) if (cur["ok"] and ref["ok"]) else None
            row["sliver"][f"{d:g}"] = dict(
                unguarded_RplusT=cur.get("worst"),
                err=e, err_over_delta=None if e is None else e / d,
                refused=not g["ok"], names_sliver=g.get("sliver", False),
                guarded_RplusT=g.get("worst"),
                warned=g.get("warned", [])[:1])
            print(f"  {name:24s} hermitian={herm!s:5s} passive={passive!s:5s} "
                  f"delta={d:.0e}: unguarded R+T {cur.get('worst')} err "
                  f"{'n/a' if e is None else format(e, '.3e')} "
                  f"({'n/a' if e is None else format(e / d, '.0f')}x) -> "
                  f"{'REFUSED' if not g['ok'] else 'returned'}")
        # the NEGATIVE control: no sliver at all
        clean = run(build(M, 3e-3), guard=True)
        row["clean"] = dict(ok=clean["ok"], RplusT=clean.get("worst"),
                            refused=not clean["ok"])
        print(f"  {name:24s} sliver-FREE control (delta 3e-3): "
              f"{'REFUSED' if not clean['ok'] else 'returned'} R+T "
              f"{clean.get('worst')}")
        rows.append(row)
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       classes=rows, wall_s=time.time() - t0), f, indent=1,
                  default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

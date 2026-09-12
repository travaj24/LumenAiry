"""PROBE 10d: an optimized _sem_modes_tensor that
  (1) exploits the EXACTLY-DIAGONAL nodal masses (inv / products O(n) not O(n^3)),
  (2) computes ``Q @ W2`` ONCE (the shipped code builds it TWICE -- V0 and V2 --
      and each ``... @ np.diag(v)`` is a full (2n)^3 gemm),
  (3) uses column scaling instead of ``@ np.diag(v)``.
A/B end-to-end, min of N, plus the deviation from the shipped answer.
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc
from lumenairy.elements.pmm import PMMStack

_C = pc._C
_orig_modes = pc._sem_modes_tensor


def _dinv(A):
    d = np.diag(A)
    return np.diag(1.0 / d)


def _fast_modes(mats, k0, kx0=0.0, robust=False, ky0=0.0):
    if ky0 != 0.0:
        return _orig_modes(mats, k0, kx0, robust, ky0)
    n = mats["n_glob"]
    k02 = k0 * k0
    S0 = mats["S0"]
    s0d = np.diag(S0).astype(_C)
    is0d = 1.0 / s0d                                   # diagonal of iS0
    mass, stiff, conv = mats["mass"], mats["stiff"], mats["conv"]
    # every mass operator is EXACTLY diagonal -> keep only the diagonals
    cinv_xx = is0d * np.diag(mass["inv_xx"])
    cxx = 1.0 / cinv_xx                                # _safe_inv of a diagonal
    exy_xx = is0d * np.diag(mass["exy_xx"])
    eyx_xx = is0d * np.diag(mass["eyx_xx"])
    schur = is0d * np.diag(mass["schur"])
    cxy = cxx * exy_xx
    cyx = eyx_xx * cxx
    cyy = schur + eyx_xx * cxx * exy_xx

    def _kxop(skey, ckey, mkey):
        op = stiff[skey]
        if kx0:
            Cw = conv[ckey]
            op = op - 1j * kx0 * (Cw - Cw.T) + (kx0 * kx0) * mass[mkey]
        return (1.0 / k02) * (is0d[:, None] * op)      # diag @ dense = row scale
    KxEzziKx = _kxop("inv_ezz", "inv_ezz", "inv_ezz")
    Kx2 = _kxop("one", "one", "one")
    G = np.eye(n, dtype=_C) - KxEzziKx
    CyyK = np.diag(cyy) - Kx2
    Mbig = np.block([[G * cxx[None, :], G * cxy[None, :]],
                     [np.diag(cyx), CyyK]])
    Q = np.block([[np.diag(cyx), CyyK],
                  [np.diag(-cxx), np.diag(-cxy)]])
    q2, W2 = np.linalg.eig(Mbig)
    q = np.sqrt(q2)
    QW = Q @ W2                                        # ONCE
    lam0 = -1j * q
    safe0 = np.where(np.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = QW * (1.0 / safe0)[None, :]
    SVt = s0d[:, None] * np.conj(V0[:n])
    SVb = s0d[:, None] * np.conj(V0[n:])
    flux = np.imag(np.einsum("in,in->n", W2[:n], SVb)
                   - np.einsum("in,in->n", W2[n:], SVt))
    thr = pc._mass_flux_threshold(flux, W2, SVt, SVb, n)
    prop = np.abs(flux) > thr
    passive = pc._grid_is_passive(mats)
    flip = pc._forward_growth_flip(flux, q, thr, prop, np, passive)
    q = np.where(flip, -q, q)
    lam = -1j * q
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    V2 = QW * (1.0 / safe)[None, :]                    # reuse QW
    return W2, V2, lam, q


def build(degree, nlay, nseg):
    st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=degree,
                  far_field_orders=21)
    for i in range(nlay):
        w = np.diff(np.linspace(0, 1, nseg + 1))
        segs = [(float(w[j]), (3.48 ** 2 if (i + j) % 2 else 1.444 ** 2))
                for j in range(nseg)]
        segs[-1] = (1.0 - sum(x for x, _ in segs[:-1]), segs[-1][1])
        st.add_layer(0.05e-6, segments=segs)
    st.set_source(1.55e-6, angle=np.deg2rad(13.0))
    return st


print("=== A/B end-to-end, min of 7 ===")
for degree, nlay, nseg in ((16, 6, 4), (24, 8, 6), (32, 8, 6)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = build(degree, nlay, nseg).solve()
    ta, tb, dev = [], [], 0.0
    for _ in range(7):
        for tag in ("A", "B"):
            if tag == "B":
                pc._sem_modes_tensor = _fast_modes
            try:
                st = build(degree, nlay, nseg)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t = time.perf_counter()
                    out = st.solve()
                    dt = time.perf_counter() - t
            finally:
                pc._sem_modes_tensor = _orig_modes
            (ta if tag == "A" else tb).append(dt)
            if tag == "B":
                dev = max(np.max(np.abs(np.asarray(out[i])
                                        - np.asarray(ref[i])))
                          for i in (1, 2, 3))
    a, b = min(ta), min(tb)
    print(f"  degree={degree} nlay={nlay} nseg={nseg}: shipped {a:.3f}s  "
          f"optimized {b:.3f}s  speed-up {a/b:.2f}x   max|d| = {dev:.2e}")

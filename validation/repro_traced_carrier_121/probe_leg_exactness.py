# DIRECT measurement of the chain's TILTED COARSE-LEG model error.
#
# No ray trace, no diffraction integral, no unwrap, no FFT derivative -- so
# none of the instrument traps this study has collected can fire here.
#
# CONSTRUCTION.  Put an EXACT point-source congruence at design 121's leg-5
# entrance plane: a smooth real Gaussian amplitude times the exact eikonal of a
# point source placed so the chief ray at (x_c, y_c) carries direction cosines
# (L, M),
#
#     W_true(du; R) = sgn(R)( sqrt(|du + R n|^2 + R^2 (1-|n|^2)) - |R| ) .
#
# Free-space transport of such a field over an axial gap z is EXACT and
# CLOSED-FORM in the geometric limit: the congruence dilates about the source
# projection by m = R1/R0 with R1 = R0 + z/N (the radius is measured ALONG the
# chief ray), the chief ray advances by z n / N, and the amplitude carries 1/m.
# Diffraction is negligible by 4 orders of magnitude here (Rayleigh range of a
# 3.6 mm 1.31 um Gaussian is 31 m against a 3.3 mm gap); the DIFF sweep below
# proves it empirically by changing the amplitude profile.
#
# The chain's leg is then run on exactly the same field --
# ``propagate_carrier_referenced`` plus the chain's own tilt bookkeeping -- and
# the two exit fields are compared POINTWISE.  Both are known analytically at
# every node, so the comparison needs no interpolation and no derivative.
#
# Env: RUN=leg (default) | screen ; W (beam radius mm), NG (grid), GAPF.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumenairy.propagators.carrier import (  # noqa: E402
    _radial_carrier_phase,
    _sphere_parab_conversion,
    _tilt_ramp,
    propagate_carrier_referenced,
)

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM

# design 121, leg 5 (group 4 exit -> group 5 front vertex), order (-4,-2).
# Every number below is READ OFF the chain's own stage table / probe log.
R0 = -24.4625e-3
GAP = 3.3233e-3
DX0 = 38.4324e-6
NG = 1024
LT, MT = None, None       # filled from the order below
W0 = 3.6253e-3
XC0, YC0 = -3.0162e-3 + 0.0, -1.5081e-3 + 0.0     # exit chief ray (approx)


def w_true(u, v, R, L, M):
    """Exact displaced-point-source eikonal with R the ALONG-RAY radius."""
    sgn = 1.0 if R > 0 else -1.0
    uu = u + R * L
    vv = v + R * M
    return sgn * (np.sqrt(uu * uu + vv * vv + R * R * (1.0 - L * L - M * M))
                  - abs(R))


def w_true_ax(u, v, Z, L, M):
    """The SAME eikonal with Z the AXIAL radius (Z = R * N).  This is the
    convention the chain's R is already in: chain A is untilted (so its R is
    unambiguous), the free legs advance it by the AXIAL gap, and
    ``_paraxial_group_r_out``'s Moebius law is the paraxial AXIAL image
    distance."""
    sgn = 1.0 if Z > 0 else -1.0
    N = np.sqrt(1.0 - L * L - M * M)
    uu = u + Z * L / N
    vv = v + Z * M / N
    return sgn * (np.sqrt(uu * uu + vv * vv + Z * Z) - abs(Z) / N)


def w_chain(u, v, R, L, M):
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(u * u + v * v + R * R) - abs(R)) + L * u + M * v


def stats(ph, wgt, name):
    wn = wgt / max(wgt.sum(), 1e-300)
    pm = float(np.sum(wn * ph))
    d = ph - pm
    # remove best tilt too (a tilt is a pointing offset, not a spot defect)
    return float(np.sqrt(max(np.sum(wn * d * d), 0.0)))


def main():
    global LT, MT
    # design 121, order (-4,-2), the tilt the chain ACTUALLY carries into
    # leg 5 (group 4's L_out/M_out -- it has flipped sign through group 4).
    LT = float(os.environ.get('L', '0.0490735'))
    MT = float(os.environ.get('M', '0.0245367'))
    m_ord = n_ord = 0
    w = float(os.environ.get('W', str(W0 * 1e3))) * 1e-3
    ng = int(os.environ.get('NG', str(NG)))
    dx = DX0 * NG / ng
    z = GAP * float(os.environ.get('GAPF', '1.0'))
    tap = os.environ.get('TAPER', '1') == '1'

    s = LT * LT + MT * MT
    N = np.sqrt(1.0 - s)
    R1c = R0 + z                    # what the chain does
    R1t = R0 + z / N                # radius measured ALONG the chief ray
    mc = R1c / R0
    mt = R1t / R0
    print(f"order ({m_ord},{n_ord})  L,M = {LT * 1e3:+.4f},{MT * 1e3:+.4f} "
          f"mrad  |n| = {np.sqrt(s):.6f}  1/N-1 = {1 / N - 1:.3e}")
    print(f"grid {ng}^2 @ dx {dx * 1e6:.4f} um   w {w * 1e3:.4f} mm   "
          f"z {z * 1e3:.4f} mm")
    print(f"R0 {R0 * 1e3:.5f}  R1(chain) {R1c * 1e3:.5f}  R1(exact) "
          f"{R1t * 1e3:.5f} mm   m(chain) {mc:.8f}  m(exact) {mt:.8f}")

    # ---- tracking-frame coordinates at the ENTRANCE -----------------------
    t = (np.arange(ng, dtype=np.float64) - ng / 2) * dx
    U = t[None, :]
    V = t[:, None]
    A = np.exp(-(U * U + V * V) / (w * w))
    dW0 = w_true(U, V, R0, LT, MT) - w_chain(U, V, R0, LT, MT)
    env_in = (A * np.exp(1j * K0 * dW0)).astype(np.complex128)

    # ---- the chain's leg --------------------------------------------------
    cr = propagate_carrier_referenced(env_in, R0, z, LAM, dx)
    env_out, R_out, dx_out = cr.env, float(cr.R), float(cr.dx)
    assert abs(R_out - R1c) < 1e-15 and abs(dx_out - mc * dx) < 1e-18
    # the chain's own analytic tilt bookkeeping (piston only -- the chief-ray
    # advance is a frame move, and we compare IN the tracking frame)
    env_out = env_out * np.exp(1j * K0 * z * (1.0 / N - 1.0))

    # ---- reconstruct the chain's exit FULL field, exactly as the chain does
    # (parabola about the chief ray, band-limited sphere conversion, ramp) --
    sh = (ng, ng)
    ph = _radial_carrier_phase(sh, dx_out, dx_out, LAM, R_out, +1,
                               centre=(0.0, 0.0))
    E_chain = np.asarray(env_out) * ph
    if tap:
        cf = _sphere_parab_conversion(sh, dx_out, LAM, R_out, +1)
        if cf is not None:
            E_chain = E_chain * cf
    else:
        t1 = (np.arange(ng, dtype=np.float64) - ng / 2) * dx_out
        U1_ = t1[None, :]
        V1_ = t1[:, None]
        r2_ = U1_ * U1_ + V1_ * V1_
        sgn0 = 1.0 if R_out > 0 else -1.0
        E_chain = E_chain * np.exp(1j * K0 * (
            sgn0 * (np.sqrt(r2_ + R_out * R_out) - abs(R_out))
            - r2_ / (2.0 * R_out)))
    rp = _tilt_ramp(sh, dx_out, LAM, LT, MT, 0.0, 0.0, +1)
    if rp is not None:
        E_chain = E_chain * rp

    # ---- the EXACT exit field on the SAME (tracking-frame) grid -----------
    t1 = (np.arange(ng, dtype=np.float64) - ng / 2) * dx_out
    U1 = t1[None, :]
    V1 = t1[:, None]
    A1 = np.exp(-((U1 / mt) ** 2 + (V1 / mt) ** 2) / (w * w)) / mt
    E_true = A1 * np.exp(1j * K0 * w_true(U1, V1, R1t, LT, MT))

    # ---- compare ----------------------------------------------------------
    d = np.angle(E_chain * np.conj(E_true))
    wgt = np.abs(E_true) ** 2
    # nearest-neighbour step of the WRAPPED difference, amplitude-weighted:
    # the licence to treat ``d`` as an unwrapped field at all.
    aw = np.abs(E_true) / np.abs(E_true).max()
    st = np.concatenate([
        (np.abs(np.diff(d, axis=0)) * np.minimum(aw[1:], aw[:-1])).ravel(),
        (np.abs(np.diff(d, axis=1)) * np.minimum(aw[:, 1:], aw[:, :-1])
         ).ravel()])
    print(f"  wrapped nn-step of the difference: p50 "
          f"{np.percentile(st, 50):.2e}  p99.9 {np.percentile(st, 99.9):.4f}"
          f"  max {st.max():.4f} rad   (pi = 3.1416)")
    rms = stats(d, wgt, 'total')
    core = np.abs(E_true) >= np.exp(-2.0) * np.abs(E_true).max()
    print(f"  MEASURED leg error: rms {rms / (2 * np.pi):.5f} waves, "
          f"|max| over the exp(-2) core "
          f"{np.abs(d - np.sum(wgt * d) / wgt.sum())[core].max() / (2 * np.pi):.5f} waves")

    # ---- the PROPOSED FIX in the AXIAL convention: eikonal only, the leg
    # distance and the obliquity piston UNCHANGED from the shipped chain.
    envA = (A * np.exp(1j * K0 * 0.0)).astype(np.complex128)
    crA = propagate_carrier_referenced(envA, R0, z, LAM, dx)
    RA, dxA = float(crA.R), float(crA.dx)
    tA = (np.arange(ng, dtype=np.float64) - ng / 2) * dxA
    EA = np.asarray(crA.env) * np.exp(1j * K0 * (
        w_true_ax(tA[None, :], tA[:, None], RA, LT, MT)
        + z * (1.0 / N - 1.0)))
    mA = RA / R0
    AA = np.exp(-((tA[None, :] / mA) ** 2 + (tA[:, None] / mA) ** 2)
                / (w * w)) / mA
    ETA = AA * np.exp(1j * K0 * w_true_ax(tA[None, :], tA[:, None],
                                          RA, LT, MT))
    dA = np.angle(EA * np.conj(ETA))
    wA = np.abs(ETA) ** 2
    print(f"  AXIAL-convention fix (eikonal only, leg distance = z): "
          f"R1 {RA * 1e3:.5f} mm dx {dxA * 1e6:.5f} um   error rms "
          f"{stats(dA, wA, 'a') / (2 * np.pi):.6f} waves")

    # ---- the PROPOSED FIX, emulated on the same leg ------------------------
    # carrier eikonal = W_true, leg distance = z/N (the radius is measured
    # ALONG the chief ray).  The stored envelope of a perfect congruence is
    # then the bare amplitude -- no coma riding on it.
    crf = propagate_carrier_referenced(A.astype(np.complex128), R0, z / N,
                                       LAM, dx)
    R_f, dx_f = float(crf.R), float(crf.dx)
    E_fix = np.asarray(crf.env) * np.exp(
        1j * K0 * w_true((np.arange(ng) - ng / 2)[None, :] * dx_f,
                         (np.arange(ng) - ng / 2)[:, None] * dx_f,
                         R_f, LT, MT))
    t1f = (np.arange(ng, dtype=np.float64) - ng / 2) * dx_f
    A1f = np.exp(-((t1f[None, :] / mt) ** 2 + (t1f[:, None] / mt) ** 2)
                 / (w * w)) / mt
    E_tf = A1f * np.exp(1j * K0 * w_true(t1f[None, :], t1f[:, None],
                                         R_f, LT, MT))
    df = np.angle(E_fix * np.conj(E_tf))
    wf = np.abs(E_tf) ** 2
    print(f"  FIXED leg (W_true carrier, z/N distance): R1 {R_f * 1e3:.5f} mm "
          f"dx {dx_f * 1e6:.5f} um   error rms "
          f"{stats(df, wf, 'f') / (2 * np.pi):.6f} waves")

    # ---- modal decomposition (weighted least squares on a monomial basis
    # in (u/w1, v/w1); no unwrap, no derivative -- ``d`` is already the
    # wrapped difference and its nn-step is << pi over the weighted support)
    w1 = w * mt
    P = U1 / w1 + 0.0 * V1
    Q = V1 / w1 + 0.0 * U1
    names, cols = [], []
    for deg in range(0, 6):
        for i in range(deg + 1):
            names.append(f"u^{deg - i} v^{i}")
            cols.append((P ** (deg - i) * Q ** i).ravel())
    Bm = np.stack(cols, axis=1)
    wq = np.sqrt((np.abs(E_true) ** 2).ravel())

    def fit(f):
        c, *_ = np.linalg.lstsq(Bm * wq[:, None], f.ravel() * wq, rcond=None)
        return c

    cd = fit(d)
    print("  modal fit (waves), |c| > 0.004 -- MEASURED vs PREDICTED(full):")

    # ---- the PREDICTED screen (probe_tilted_eikonal's C, both modes) ------
    for lbl, Rex in (('eik  (R1 = R0 + z)  ', R1c), ('full (R1 = R0 + z/N)', R1t)):
        d0 = (w_true(U1 / mc, V1 / mc, R0, LT, MT)
              - w_chain(U1 / mc, V1 / mc, R0, LT, MT))
        d1 = (w_true(U1, V1, Rex, LT, MT) - w_chain(U1, V1, R1c, LT, MT))
        pred = -K0 * (d1 - d0)      # E_chain/E_true = exp(-i k0 C)
        r = stats(pred, wgt, lbl)
        res = stats(d - pred, wgt, lbl)
        print(f"  PREDICTED {lbl}: rms {r / (2 * np.pi):.5f} waves   "
              f"residual (measured - predicted) rms "
              f"{res / (2 * np.pi):.5f} waves   "
              f"ratio measured/predicted "
              f"{(float(np.sum(wgt * (d - np.sum(wgt * d) / wgt.sum()) * (pred - np.sum(wgt * pred) / wgt.sum()))) / max(float(np.sum(wgt * (pred - np.sum(wgt * pred) / wgt.sum()) ** 2)), 1e-300)):.4f}")
        if 'full' in lbl:
            cp = fit(pred)
            for nm_, a, b in zip(names, cd, cp):
                if max(abs(a), abs(b)) / (2 * np.pi) > 0.004:
                    print(f"      {nm_:10s} meas {a / (2 * np.pi):+9.5f}   "
                          f"pred {b / (2 * np.pi):+9.5f}   diff "
                          f"{(a - b) / (2 * np.pi):+9.5f}")


if __name__ == '__main__':
    sys.exit(main())

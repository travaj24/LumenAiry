"""V3 -- THE FRAME-ANCHOR PHASE, derived here and measured three ways.

DERIVATION (independent of the build's).  A slanted region is solved in the
frame ``u = x - t_int w``, ``v = y - t_int_y w``, ``w = z``, anchored at the
layer's TOP face, so the layer occupies ``0 <= w <= d``.  Every modal field in
the region is a function of ``(u, v)`` and the region's Rayleigh expansion at
the BOTTOM interface ``w = d`` is ``sum_m A_m exp(i alpha_m . (u, v))``.  In the
lab that plane is ``z = d`` with ``(u, v) = (x - t_int d, y - t_int_y d)``, so
against the substrate's OWN lab basis ``exp(i alpha_m . (x, y))``::

    A_lab(m) = exp(-i alpha_m . t_int d) A_frame(m)

``alpha_m`` is real for every order (propagating or evanescent), so the factor
is UNIMODULAR: it cannot move an efficiency, and the superstrate side is
unshifted (the frame is anchored there), so the reflection Jones needs nothing.
Only the TRANSMITTED amplitudes carry it.  In a stack the offsets ADD, giving
``sum_j t_j d_j``.  The shipped internal shear is ``t_int = -slant`` (the
congruence is taken with ``-slant`` and the phase is written with
``_shx = -sum slant_x d``), so in the PUBLIC vector the factor is::

    A_lab(m) = exp(+i k0 (alpha_m . slant) d) A_frame(m)

THE THREE ARMS, built WITHOUT touching the library.  The factor is a scalar on
the zeroth order, so the "no correction" and "conjugate" arms are recovered
EXACTLY from the shipped answer by dividing it out once / twice::

    P0 = exp(-i k0 (kx0 * Shx + ky0 * Shy)),  Shx = -sum_j slant_x,j d_j
    J_none = J_shipped * conj(P0)       (P0 divided out)
    J_conj = J_shipped * conj(P0)**2    (P0 replaced by its conjugate)

THREE ORACLES, in increasing strength:

  A. a UNIFORM slab (isotropic and out-of-plane uniaxial), slanted vs vertical
     -- a shear of a homogeneous medium is a coordinate change, so the two
     transmission Jones must agree EXACTLY;
  B. a PATTERNED layer against a z-STAIRCASE of VERTICAL layers of the same
     solid.  A staircase has no frame at all, so its transmitted amplitudes are
     LAB-referenced by construction -- this is the one oracle that tests the
     anchor on a real pattern, and it is same-engine and independent of any
     other library's conventions;
  C. a 2-D CHIRAL slanted cell, where no 1-D oracle exists, against
     ``PMM2DStackHybrid``'s transmission Jones on an ``n_orders`` ladder.
"""
import numpy as np

from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 0.68e-6
NSUP, NSUB = 1.0, 1.5
K0 = 2.0 * np.pi / WL

T10 = float(np.tan(np.deg2rad(10.0)))
T35 = float(np.tan(np.deg2rad(35.0)))
DIAG35 = (T35 / np.sqrt(2.0), T35 / np.sqrt(2.0))
MOUNTS = {"oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))


def p0_factor(slants_thick, theta, phi):
    """``P0`` for the (0, 0) order, from the DERIVATION above -- built from the
    incident transverse k and the accumulated ``sum_j slant_j d_j``."""
    shx = -sum(s[0] * d for s, d in slants_thick)
    shy = -sum(s[1] * d for s, d in slants_thick)
    kx0 = float(np.real(NSUP)) * np.sin(theta) * np.cos(phi)
    ky0 = float(np.real(NSUP)) * np.sin(theta) * np.sin(phi)
    return complex(np.exp(-1j * K0 * (kx0 * shx + ky0 * shy)))


def arms(J, P0):
    return dict(shipped=J, none=J * np.conj(P0), conj=J * np.conj(P0) ** 2)


# ------------------------------------------------------------------ oracle A
def uniform_stack(tensor, sl, depth, theta, phi, M=5, nord=3, grid=2):
    st = PMM2DStackPure(1.10e-6, 1.10e-6, n_superstrate=NSUP,
                        n_substrate=NSUB, n_modes=M, n_orders=nord)
    if tensor is None:
        st.add_layer(depth, eps=2.25 + 0j, slant=sl)
    else:
        st.add_layer(depth, eps=np.asarray(tensor, dtype=complex), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve(jones=True)
    return st.jones_transmission()


# ------------------------------------------------------------------ oracle B
PXB = PYB = 1.20e-6
DEPB = 1.20e-6
NXB = 6
TB = 1.0                     # a whole period over the layer: exact on the grid
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])


def cellB(two_d=True):
    c = np.ones((NXB, NXB), dtype=complex)
    if two_d:
        c[:, 0:3] = XPROF[:, None]
    else:
        c[:, :] = XPROF[:, None]
    return c


def pureB_slant(cell, sl, theta, phi, M=3, nord=3):
    st = PMM2DStackPure(PXB, PYB, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    st.add_layer(DEPB, eps_cell=cell, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve(jones=True)
    return st.jones_transmission()


def pureB_stair(cell, K, theta, phi, M=3, nord=3, sign=+1):
    st = PMM2DStackPure(PXB, PYB, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    d = DEPB / K
    for k in range(K):
        st.add_layer(d, eps_cell=np.roll(cell, sign * (NXB * k // K), axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    st.solve(jones=True)
    return st.jones_transmission()


# ------------------------------------------------------------------ oracle C
PXC = PYC = 1.20e-6
DEPC = 0.80e-6
TC = 0.75
CHIRAL = np.array([[2.60, 0.35 + 0.0j, 0.0],
                   [0.35, 2.10, 0.0],
                   [0.0, 0.0, 2.30]], dtype=complex)


def cellC():
    c = np.zeros((3, 3, 3, 3), dtype=complex)
    c[:, :] = np.eye(3)
    c[1, 1] = CHIRAL
    c[0, 1] = np.diag([2.0, 2.0, 2.0])
    return c


def pureC(sl, theta, phi, M=5, nord=3):
    st = PMM2DStackPure(PXC, PYC, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    st.add_layer(DEPC, eps_cell=cellC(), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve(jones=True)
    return st.jones_transmission()


def hybC(sl, theta, phi, nord):
    st = PMM2DStackHybrid(PXC, PYC, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    st.add_layer(DEPC, eps_tensor_cell=cellC(), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    st.solve()
    return st.jones_transmission()


def main():
    out = {"A_uniform": {}, "B_staircase": {}, "C_chiral_vs_hybrid": {},
           "phase_size": {}}

    # ---------------------------------------------------------------- A
    dep = 0.34e-6
    for tname, tens in (("isotropic", None), ("oop_uniaxial", TIL)):
        ref = {}
        for mname, (th, ph) in MOUNTS.items():
            ref[mname] = uniform_stack(tens, None, dep, th, ph)
        for sname, sv in (("x10", (T10, 0.0)), ("x35", (T35, 0.0)),
                          ("diag35", DIAG35)):
            for mname, (th, ph) in MOUNTS.items():
                J = uniform_stack(tens, sv, dep, th, ph)
                P0 = p0_factor([(sv, dep)], th, ph)
                a = arms(J, P0)
                key = f"{tname}/{sname}/{mname}"
                out["A_uniform"][key] = {k: mx(v, ref[mname])
                                         for k, v in a.items()}
                out["phase_size"][key] = float(abs(np.angle(P0)))
                print(f"[A] {key}: shipped "
                      f"{out['A_uniform'][key]['shipped']:.3e}  none "
                      f"{out['A_uniform'][key]['none']:.3e}  conj "
                      f"{out['A_uniform'][key]['conj']:.3e}  "
                      f"|arg P0| {abs(np.angle(P0)):.3f}")

    # ---------------------------------------------------------------- B
    for cname, two_d in (("pillar2d", True), ("stripe", False)):
        cell = cellB(two_d)
        for mname, (th, ph) in MOUNTS.items():
            J = pureB_slant(cell, (TB, 0.0), th, ph)
            P0 = p0_factor([((TB, 0.0), DEPB)], th, ph)
            a = arms(J, P0)
            row = {"abs_arg_P0": float(abs(np.angle(P0)))}
            for K in (2, 3, 6):
                st = pureB_stair(cell, K, th, ph)
                for k, v in a.items():
                    row[f"K{K}_{k}"] = mx(v, st)
                # the staircase's own convergence: rung-to-rung move
                if K > 2:
                    row[f"K{K}_stair_own_step"] = mx(
                        st, pureB_stair(cell, K - 1 if K == 3 else 3, th, ph))
            out["B_staircase"][f"{cname}/{mname}"] = row
            print(f"[B] {cname}/{mname}: K6 shipped {row['K6_shipped']:.3e} "
                  f"none {row['K6_none']:.3e} conj {row['K6_conj']:.3e} "
                  f"(stair own step {row.get('K6_stair_own_step', 0):.3e}, "
                  f"|arg P0| {row['abs_arg_P0']:.3f})")

    # ---------------------------------------------------------------- C
    for mname, (th, ph) in MOUNTS.items():
        J = pureC((TC, 0.0), th, ph)
        P0 = p0_factor([((TC, 0.0), DEPC)], th, ph)
        a = arms(J, P0)
        row = {"abs_arg_P0": float(abs(np.angle(P0)))}
        prev = None
        for nord in (5, 7, 9, 11):
            H = hybC((TC, 0.0), th, ph, nord)
            for k, v in a.items():
                row[f"n{nord}_{k}"] = mx(v, H)
            if prev is not None:
                row[f"n{nord}_hybrid_own_step"] = mx(H, prev)
            prev = H
        out["C_chiral_vs_hybrid"][mname] = row
        print(f"[C] {mname}: n11 shipped {row['n11_shipped']:.3e} none "
              f"{row['n11_none']:.3e} conj {row['n11_conj']:.3e} "
              f"(hybrid own step {row.get('n11_hybrid_own_step', 0):.3e}, "
              f"|arg P0| {row['abs_arg_P0']:.3f})")

    dump("v3_frame_anchor_phase", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

"""M0b -- two diagnostics that separate 'the mortar is wrong' from 'the coarse
layer is under-resolved':

(a) SINGLE-LAYER h-refinement control: the same physical stripe drawn on N=2
    and on its exact N=4 refinement, through the SHIPPED engine only (no
    mortar anywhere).  Whatever this shows is the resolution difference the
    M2 comparison necessarily inherits.
(b) TRANSPARENT-INTERFACE probe: one uniform slab split into two sub-layers on
    DIFFERENT grids.  The interface is physically absent, so the exact answer
    is the single slab (analytic Fresnel) -- any deviation is the mortar's own
    error, with no resolution confound at normal incidence (the excited mode
    is the constant, which lies exactly in every grid's basis)."""
import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

PX = PY = 0.9e-6
WL = 0.60e-6
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
TA = 0.20e-6

print("\n(a) single-layer h-refinement control (shipped engine only), th=0.20")
for M in (4, 5, 6, 7):
    o1, R1, T1 = pmm_efficiency_2d_staggered(PX, PY, A2, 1.0, 1.0, TA, WL,
                                             degree=M, n_orders=2,
                                             polarization="te", theta=0.20)
    o2, R2, T2 = pmm_efficiency_2d_staggered(PX, PY, refine_cell(A2, 2), 1.0,
                                             1.0, TA, WL, degree=M, n_orders=2,
                                             polarization="te", theta=0.20)
    sc = max(R1.max(), T1.max())
    print(f"  M={M}  N2 vs N4(refined) rel dR {np.abs(R1-R2).max()/sc:.3e} "
          f"dT {np.abs(T1-T2).max()/sc:.3e}   sumR N2 {R1.sum():.6f} "
          f"N4 {R2.sum():.6f}", flush=True)


def fresnel_slab_te(n0, n1, n2, d, wl, theta):
    k0 = 2 * np.pi / wl
    s = n0 * np.sin(theta)
    kz = [k0 * np.sqrt(complex(n ** 2 - s ** 2)) for n in (n0, n1, n2)]
    r01 = (kz[0] - kz[1]) / (kz[0] + kz[1])
    r12 = (kz[1] - kz[2]) / (kz[1] + kz[2])
    t01 = 2 * kz[0] / (kz[0] + kz[1])
    t12 = 2 * kz[1] / (kz[1] + kz[2])
    ph = np.exp(2j * kz[1] * d)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    t = t01 * t12 * np.exp(1j * kz[1] * d) / (1 + r01 * r12 * ph)
    return float(abs(r) ** 2), float(abs(t) ** 2 * (kz[2] / kz[0]).real)

print("\n(b) transparent interface: uniform eps=4 slab split in two halves")
NSLAB, D = 2.0, 0.30e-6
for theta in (0.0, 0.20):
    Rex, Tex = fresnel_slab_te(1.0, NSLAB, 1.0, D, WL, theta)
    for M in (5, 7):
        for (Na, Nb) in ((2, 2), (2, 4), (2, 3), (3, 4)):
            mor = MortarStack2D(PX, PY, n_modes=M, n_orders=2)
            mor.add_layer(D / 2, eps=NSLAB ** 2, grid=Na)
            mor.add_layer(D / 2, eps=NSLAB ** 2, grid=Nb)
            mor.set_source(WL, theta=theta, phi=0.0)
            o, R, T = mor.solve(jones=False)
            R0 = float(R[1].sum())
            T0 = float(T[1].sum())
            print(f"  th={theta:.2f} M={M} grids({Na},{Nb})  R {R0:.10f} "
                  f"(exact {Rex:.10f}, dR {abs(R0-Rex):.2e})  "
                  f"R+T-1 {R0+T0-1:+.2e}", flush=True)

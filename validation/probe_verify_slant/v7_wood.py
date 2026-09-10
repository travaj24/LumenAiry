"""V7 -- THE WOOD-LIST DECISION: a slanted layer contributes its LAB
permittivities, not the covariant diagonal ``eps (1 + t^2)``.

The nudge exists to move a wavelength off an EXACT Rayleigh coincidence,
because a layer mode with ``q -> 0`` is what makes the interface S-matrix
singular.  The decision is therefore an EMPIRICAL question: at which candidate
coincidence does a SLANTED layer actually go near a null mode?

The probe walks a slanted scalar layer (``eps = 4``, ``t = (0.6, 0)``) onto
each candidate by choosing the period, at normal incidence on a SQUARE cell, so
that all four first orders sit at ``|alpha| = sqrt(candidate)`` together:

  * LAB          ``eps        = 4.00``   -> ``wl/px = 2.0000``
  * COVARIANT    ``eps(1+t^2) = 5.44``   -> ``wl/px = 2.3324``
  * PER-ORDER    ``|a|^2 + (t.a)^2``     -> ``wl/px = 1.1662``

and measures the layer's own ``min |q|`` over the assembled ``4 q^2`` spectrum
at each.  The ANALYTIC expectation, stated before the run: at the LAB
candidate the ``(0, +/-1)`` orders have ``t . alpha = 0`` and
``q = +/- sqrt(eps - |alpha|^2) = 0`` -- a genuine null mode; at the COVARIANT
candidate the same orders give ``q = +/- i sqrt(5.44 - 4) = +/- 1.2 i``, i.e.
``|q| = 1.2``, two decades away and not a cut-off at all.

Also measured: whether ``_grazing_safe_wavelength`` (the shipped helper, called
with the list ``PMM2DStackPure.solve`` actually builds) FIRES on each candidate
under the lab list and under a hypothetical covariant list, and whether a
slanted stack sitting exactly on its lab cut-off comes out clean.
"""
import numpy as np
import scipy.linalg as sla
from _lib import arm, dump  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _wood_eps_reals,
)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

WL = 0.68e-6
EPS = 4.0
TX = 0.6
NG = 2
M = 6
NORD = 3
NSUP, NSUB = 1.0, 1.5

CANDIDATES = {
    "lab_eps": EPS,
    "covariant_eps_1_plus_t2": EPS * (1.0 + TX ** 2),
    "per_order_alpha2_plus_ta2": 1.0 + TX ** 2,
}


def min_abs_q(px, slant):
    cell = np.full((NG, NG), complex(EPS))
    k0 = 2.0 * np.pi / WL
    sol = Granet2DTransverseE(px, px, NG, NG, M, cell, alpha0x=0.0,
                              alpha0y=0.0, k0=k0, slant=slant)
    if not sol.offplane:
        # vertical scalar: the 2 q^2 in-plane pencil; its eigenvalue is q^2
        from lumenairy.elements.pmm.twod_staggered import _region_modes
        res = _region_modes(sol)
        lam = res[2]
        return float(np.min(np.abs(lam))), "inplane_lam"
    qv = sla.eig(sol.Agen, sol.Bgen, right=False)
    return float(np.min(np.abs(qv))), "oop_q"


def nudge_fires(px, eps_list):
    mo = np.arange(-NORD, NORD + 1)
    mx = np.tile(mo, len(mo))
    my = np.repeat(mo, len(mo))
    wl2 = _grazing_safe_wavelength(WL, 0.0, 0.0, mx, my, px, px,
                                   _wood_eps_reals(*eps_list))
    return bool(wl2 != WL), float(wl2 / WL - 1.0)


def main():
    out = {"config": dict(eps=EPS, t=(TX, 0.0), M=M, grid=NG, n_orders=NORD),
           "candidates": {}, "nudge": {}, "stack": {}}
    lab_list = [complex(NSUP) ** 2, complex(NSUB) ** 2, complex(EPS)]
    cov_list = lab_list + [complex(EPS * (1.0 + TX ** 2))]
    for cname, val in CANDIDATES.items():
        px = WL / np.sqrt(val)
        mq_s, kind = min_abs_q(px, (TX, 0.0))
        mq_v, _ = min_abs_q(px, None)
        fires_lab, rel_lab = nudge_fires(px, lab_list)
        fires_cov, rel_cov = nudge_fires(px, cov_list)
        out["candidates"][cname] = dict(
            candidate_value=val, wl_over_px=float(WL / px),
            min_abs_q_slanted=mq_s, min_abs_q_vertical=mq_v, kind=kind,
            nudge_fires_lab_list=fires_lab, nudge_rel_lab=rel_lab,
            nudge_fires_covariant_list=fires_cov, nudge_rel_cov=rel_cov)
        print(f"{cname:28s} value {val:6.4f}  wl/px {WL / px:7.4f}  "
              f"slanted min|q| {mq_s:.4e}  vertical min|lam| {mq_v:.4e}  "
              f"nudge lab {fires_lab} cov {fires_cov}")

    # ---- a slanted stack sitting EXACTLY on its lab cut-off: does the shipped
    # path nudge, and does the solve come out clean?
    px = WL / np.sqrt(EPS)
    for sl, lab in (((TX, 0.0), "slanted"), (None, "vertical")):
        st = PMM2DStackPure(px, px, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=M, n_orders=NORD)
        cell = np.full((NG, NG), complex(EPS))
        cell[0, 0] = 1.0                    # patterned, so the layer is real
        st.add_layer(0.34e-6, eps_cell=cell, slant=sl)
        st.set_source(WL, theta=0.0, phi=0.0)
        o, R, T, J = st.solve(jones=True)
        clo = float(np.max(R.sum(1) + T.sum(1)))
        # the same structure a hair OFF the coincidence
        st2 = PMM2DStackPure(px, px, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=M, n_orders=NORD)
        st2.add_layer(0.34e-6, eps_cell=cell, slant=sl)
        st2.set_source(WL * (1.0 + 1e-7), theta=0.0, phi=0.0)
        o2, R2, T2, J2 = st2.solve(jones=True)
        out["stack"][lab] = dict(
            closure_on_cutoff=clo,
            closure_detuned=float(np.max(R2.sum(1) + T2.sum(1))),
            dJones_on_vs_detuned=float(np.max(np.abs(J - J2))),
            dR=float(np.max(np.abs(R - R2))))
        print(f"[stack {lab}] closure on-cutoff {clo:.12f}  detuned "
              f"{out['stack'][lab]['closure_detuned']:.12f}  dJones "
              f"{out['stack'][lab]['dJones_on_vs_detuned']:.3e}")
    dump("v7_wood", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

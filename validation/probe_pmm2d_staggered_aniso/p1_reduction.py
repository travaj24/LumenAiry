"""Probe 1 -- G1 reduction: a tensor ``e*I`` cell must reproduce the shipped
isotropic assembly BIT-FOR-BIT (same build, two arms), and the two public
entries must agree.

Run:  PYTHONPATH=/c/tmp/lum_aniso python validation/probe_pmm2d_staggered_aniso/p1_reduction.py
"""
import hashlib

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    cell = np.array([[6.25, 1.0], [1.0, 2.25]], dtype=complex)
    tens = np.zeros(cell.shape + (3, 3), dtype=complex)
    for i in range(3):
        tens[..., i, i] = cell
    kw = dict(alpha0x=0.31, alpha0y=-0.17, k0=2 * np.pi / 0.62)
    a = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, cell, **kw)
    b = Granet2DTransverseE(1.1, 1.1, 2, 2, 6, tens, **kw)
    print("operator hashes (scalar arm | tensor e*I arm):")
    for name in ("Lmat", "Rmat", "Stt", "Schur"):
        ha, hb = _h(getattr(a, name)), _h(getattr(b, name))
        print(f"  {name:6s} {ha}  {hb}  {'IDENTICAL' if ha == hb else 'DIFFER'}"
              f"   maxdiff={np.max(np.abs(getattr(a, name) - getattr(b, name))):.3e}")
    for k in (0, 1):
        ha, hb = _h(a.Et_blocks[k]), _h(b.Et_blocks[k])
        print(f"  Et[{k}]  {ha}  {hb}  {'IDENTICAL' if ha == hb else 'DIFFER'}")
    print("  Et_offdiag scalar:", a.Et_offdiag,
          "| tensor max|Et12|:",
          float(np.max(np.abs(b.Et_offdiag[0]))),
          "max|Et21|:", float(np.max(np.abs(b.Et_offdiag[1]))))

    # public entries
    g = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6,
             wavelength=0.633e-6, n_substrate=1.5, n_superstrate=1.0)
    o1, R1, T1 = pmm_efficiency_2d_staggered(
        eps_cell=cell, degree=6, n_orders=3, polarization="tm", **g)
    o2, R2, T2, J2 = pmm_jones_2d_staggered(
        eps_cell=tens, degree=6, n_orders=3, **g)
    assert np.array_equal(o1, o2)
    # scalar 'tm' at normal incidence == incident E_x (row 0)
    print("entry agreement (scalar 'tm' vs tensor row 0):",
          f"maxdiff R {np.max(np.abs(R1 - R2[0])):.3e}",
          f"T {np.max(np.abs(T1 - T2[0])):.3e}")
    print("closure tensor rows:", float(R2[0].sum() + T2[0].sum()),
          float(R2[1].sum() + T2[1].sum()))
    print("jones:\n", J2)


if __name__ == "__main__":
    main()

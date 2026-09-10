"""V5 -- THE PARITY QUESTION.  Is the explicit refusal of the block-eig
accelerator on a slanted cell covering a wrong answer, or is it conservatism?

The build reports that ``_stag_block_eig``'s structural residual does NOT catch
a shear (2.3e-15 against a 1e-10 bar) and that the reduction, FORCED onto the
slanted pencil, is correct.  This probe re-measures that on its own fixtures
and pushes it: bigger slants, y and diagonal slants, a lossy cell, high
contrast, both grids, and -- the arm the build did not run -- the reduction
forced END TO END through a stack solve, on R, T and the Jones.

WHY the shear does not break the structure, derived (and it matters for the
verdict, because a coincidence and a symmetry warrant different decisions).
``R`` is a 180-degree ROTATION about z, and the pencil relation is
``R A R = -A`` with ``R B R = +B`` -- i.e. ``(q, x)`` pairs with ``(-q, R x)``,
which is the algebraic form of invariance under the FULL INVERSION
``(x, y, z) -> (-x, -y, -z)``.  A sheared solid ``x = u + t z`` maps under that
inversion to ``-x = u + t(-z)``, i.e. ``x = (-u) + t z`` -- THE SAME SHEAR, on
the inverted cross-section.  So a centro-symmetric cross-section sheared by any
``t`` is inversion-symmetric, exactly as the vertical one is.  The structure is
a SYMMETRY of the sheared solid, not a numerical coincidence.

FORCING, two ways, neither of them ``_slant_is_zero``: the build records that
monkeypatching that name silently builds the VERTICAL pencil, because
``Granet2DTransverseE.__init__`` reads it to decide whether to shear at all.

  * ``_Shim`` -- a proxy around an ALREADY-BUILT slanted solver whose ``slant``
    attribute reads ``None``.  The pencil it exposes is the genuine slanted
    one; only the gate's view of it changes.
  * for the end-to-end arm, ``_region_modes_oop`` is wrapped (in the stack
    module only) so it receives the shim.  The ASSEMBLY is untouched -- the
    wrapper runs after ``__init__`` has already sheared the cell.

Both are verified non-vacuous: the forced solve must differ from a genuinely
VERTICAL solve by the full slant effect, and the shim's pencil is compared byte
for byte against the un-shimmed solver's.
"""
import numpy as np
import scipy.linalg as sla
from _lib import arm, dump, mx, sha  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm import stack2d_pure as _sp
from lumenairy.elements.pmm.twod_staggered import (
    _STAG_BLOCK_TOL,
    Granet2DTransverseE,
    _stag_block_eig,
    _stag_parity_gauge,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 0.68e-6
PX = PY = 1.10e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5
K0 = 2.0 * np.pi / WL

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
LOSSY = uniaxial_tensor(1.5 + 0.05j, 1.7 + 0.02j, np.deg2rad(35.0),
                        phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
T20 = float(np.tan(np.deg2rad(20.0)))
T35 = float(np.tan(np.deg2rad(35.0)))
T60 = float(np.tan(np.deg2rad(60.0)))
SLANTS = {"vertical": None, "x20": (T20, 0.0), "x35": (T35, 0.0),
          "y35": (0.0, T35), "diag35": (T35 / np.sqrt(2), T35 / np.sqrt(2)),
          "x60": (T60, 0.0)}


def centro_tensor(n=2, t=TIL):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = AIR
    if n == 2:
        c[0, 0] = c[1, 1] = t
    else:
        c[0, 0] = c[0, 1] = c[0, 2] = t      # a full row: its own R-image
        c[2, 0] = c[2, 1] = c[2, 2] = t
    return c


def centro_scalar(n=2, e=4.0):
    c = np.ones((n, n), dtype=complex)
    if n == 2:
        c[0, 0] = c[1, 1] = e
    else:
        c[1, :] = e
    return c


def offcentre_tensor():
    c = np.zeros((3, 3, 3, 3), dtype=complex)
    c[:, :] = AIR
    c[0, 1] = TIL                              # NOT its own parity image
    return c


class _Shim:
    """Proxy over a built slanted solver whose ``slant`` reads ``None``."""
    slant = None

    def __init__(self, sol):
        object.__setattr__(self, "_s", sol)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_s"), k)


def struct_residual(A, B, parity):
    perm, r = parity
    _n4 = A.shape[0]
    rr = r[:, None] * r[None, :]
    ra = float(np.max(np.abs(rr * A[np.ix_(perm, perm)] + A)))
    rb = float(np.max(np.abs(rr * B[np.ix_(perm, perm)] - B)))
    return ra / float(np.max(np.abs(A))), rb / float(np.max(np.abs(B)))


def dense_spectrum(A, B):
    qq2 = A.shape[0]
    Lc = np.linalg.cholesky(B)
    Ah = sla.solve_triangular(Lc, A, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
    _ = qq2
    return qv, X


def spec_gap(a, b):
    """HAUSDORFF distance between two spectra -- NOT a sorted elementwise
    difference.  A lexicographic sort scrambles the many purely-imaginary
    (evanescent) eigenvalues, whose real parts are +/- round-off, and reports a
    gap of O(10) on two IDENTICAL sets; that mistake was made here first and
    caught by the vertical control, where the two branches are the shipped
    code and must agree."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        return float("inf")
    d1 = float(np.max(np.min(np.abs(a[:, None] - b[None, :]), axis=1)))
    d2 = float(np.max(np.min(np.abs(a[:, None] - b[None, :]), axis=0)))
    return max(d1, d2)


def pencil_resid(A, B, qv, X):
    r = A @ X - (B @ X) * qv[None, :]
    sc = float(np.max(np.abs(A))) * np.linalg.norm(X, axis=0)
    return float(np.max(np.linalg.norm(r, axis=0) / np.where(sc == 0, 1, sc)))


def solver(cell, sl, M=6, theta=0.0):
    n = np.shape(cell)[0]
    return Granet2DTransverseE(PX, PY, n, n, M, cell, alpha0x=0.0,
                               alpha0y=0.0, k0=K0, slant=sl)


def stack_solve(cell, sl, M=6, forced=False, nord=3):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord, symmetry=True)
    st.add_layer(DEP, eps_cell=cell, slant=sl)
    st.set_source(WL, theta=0.0, phi=0.0)
    if not forced:
        return st.solve(jones=True)
    orig = _sp._region_modes_oop

    def wrapped(sol, *, symmetry=False):
        return orig(_Shim(sol), symmetry=symmetry)
    _sp._region_modes_oop = wrapped
    try:
        return st.solve(jones=True)
    finally:
        _sp._region_modes_oop = orig


def main():
    out = {"structural": {}, "forced": {}, "end_to_end": {}, "refusing": {},
           "tol": _STAG_BLOCK_TOL}
    fixtures = {
        "centro_oop_2x2": centro_tensor(2),
        "centro_oop_3x3": centro_tensor(3),
        "centro_scalar_2x2": centro_scalar(2),
        "centro_lossy_2x2": centro_tensor(2, LOSSY),
        "centro_highcontrast_2x2": centro_scalar(2, 12.0),
    }
    for fname, cell in fixtures.items():
        for sname, sl in SLANTS.items():
            if sl is None and np.ndim(cell) == 2:
                continue            # a vertical scalar cell has no 4q^2 pencil
            sol = solver(cell, sl)
            if not sol.offplane:
                continue
            shim = _Shim(sol)
            g_ship = _stag_parity_gauge(sol)
            g_force = _stag_parity_gauge(shim)
            key = f"{fname}/{sname}"
            row = dict(shipped_gauge_none=g_ship is None,
                       forced_gauge_none=g_force is None)
            if g_force is not None:
                ra, rb = struct_residual(sol.Agen, sol.Bgen, g_force)
                row.update(resid_A=ra, resid_B=rb,
                           passes_bar=bool(ra <= _STAG_BLOCK_TOL
                                           and rb <= _STAG_BLOCK_TOL))
                fac = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q,
                                      g_force)
                row["reduction_runs"] = fac is not None
                if fac is not None:
                    qf, Xf = fac
                    qd, Xd = dense_spectrum(sol.Agen, sol.Bgen)
                    row["spectrum_gap"] = spec_gap(qf, qd)
                    row["forced_pencil_residual"] = pencil_resid(
                        sol.Agen, sol.Bgen, qf, Xf)
                    row["dense_pencil_residual"] = pencil_resid(
                        sol.Agen, sol.Bgen, qd, Xd)
            out["structural"][key] = row
            print(f"[S] {key:34s} gauge shipped={'REFUSE' if row['shipped_gauge_none'] else 'ok'}"
                  f" forced={'REFUSE' if row['forced_gauge_none'] else 'ok'}"
                  f"  residA {row.get('resid_A', float('nan')):.2e}"
                  f"  specgap {row.get('spectrum_gap', float('nan')):.2e}"
                  f"  pencil {row.get('forced_pencil_residual', float('nan')):.2e}"
                  f" (dense {row.get('dense_pencil_residual', float('nan')):.2e})")

    # ---- the REFUSING arm: geometries that must NOT pass the structural bar
    for rname, cell, sl in (("offcentre_pillar_slanted", offcentre_tensor(),
                             (T35, 0.0)),
                            ("offcentre_pillar_vertical", offcentre_tensor(),
                             None)):
        sol = solver(cell, sl)
        g = _stag_parity_gauge(_Shim(sol))
        ra, rb = struct_residual(sol.Agen, sol.Bgen, g)
        out["refusing"][rname] = dict(
            resid_A=ra, resid_B=rb,
            reduction_runs=_stag_block_eig(sol.Agen, sol.Bgen,
                                           sol.q * sol.q, g) is not None)
        print(f"[X] {rname}: residA {ra:.3e} residB {rb:.3e} runs "
              f"{out['refusing'][rname]['reduction_runs']}")

    # ---- END TO END, the arm the build did not run
    for fname, cell in (("centro_oop_2x2", centro_tensor(2)),
                        ("centro_scalar_2x2", centro_scalar(2)),
                        ("centro_lossy_2x2", centro_tensor(2, LOSSY))):
        for sname in ("x35", "diag35", "x60"):
            sl = SLANTS[sname]
            dense = stack_solve(cell, sl, forced=False)
            forced = stack_solve(cell, sl, forced=True)
            vert = stack_solve(cell, None, forced=False) \
                if np.ndim(cell) == 4 else None
            row = dict(dR=mx(dense[1], forced[1]), dT=mx(dense[2], forced[2]),
                       dJones=mx(dense[3], forced[3]),
                       identical=(sha(dense[1], dense[2], dense[3])
                                  == sha(forced[1], forced[2], forced[3])))
            if vert is not None:
                row["slant_effect_dJones"] = mx(dense[3], vert[3])
            out["end_to_end"][f"{fname}/{sname}"] = row
            print(f"[E] {fname}/{sname}: dR {row['dR']:.3e} dT "
                  f"{row['dT']:.3e} dJones {row['dJones']:.3e} "
                  f"(slant effect {row.get('slant_effect_dJones', 0):.3e})")

    dump("v5_parity", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

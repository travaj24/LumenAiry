"""B11a -- the PARITY ACCELERATOR must REFUSE a slanted cell.

Two-sided, on the SAME centro-symmetric cell at NORMAL incidence:

  * VERTICAL  -- ``_stag_parity_gauge`` returns a gauge, the structural
    residual is below ``_STAG_BLOCK_TOL`` and the reduction ENGAGES;
  * SLANTED   -- the gauge is refused OUTRIGHT, and the structural residual is
    MEASURED (with the gauge forced onto the real slanted pencil through a
    geometry shim) to sit FIVE DECADES BELOW the bar: the structure test does
    NOT catch a shear, so the explicit refusal is the only gate there is;
  * and ``symmetry='auto'`` on the slanted cell is sha256-IDENTICAL to
    ``symmetry=False`` while DIFFERING from the vertical answer.

B11b -- the REFUSALS: mixed slants between patterned layers, a mix of
vertical and slanted layers above a pattern, magnetic + slant,
``retain_internal`` + slant, ``pmm_efficiency_2d_staggered`` + slant.

WOOD -- what a slanted SCALAR layer contributes to the Rayleigh nudge list.
"""
import hashlib
import time
import warnings

import numpy as np
from _lib import tile, uniaxial, write

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    _STAG_BLOCK_TOL,
    Granet2DTransverseE,
    _region_modes_oop,
    _slant_congruence,
    _stag_block_eig,
    _stag_parity_gauge,
    _wood_eps_reals,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5
K0 = 2.0 * np.pi / WL

TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
CENTRO = tile(AIR, 2)
CENTRO[0, 0] = TIL
CENTRO[1, 1] = TIL
T35 = float(np.tan(np.deg2rad(35.0)))

res = {}
t00 = time.time()


def h(*a):
    m = hashlib.sha256()
    for x in a:
        m.update(np.ascontiguousarray(x).tobytes())
    return m.hexdigest()


def struct_dA(sol, gauge):
    perm, r = gauge
    A = sol.Agen
    rr = r[:, None] * r[None, :]
    return (float(np.max(np.abs(rr * A[np.ix_(perm, perm)] + A)))
            / float(np.max(np.abs(A))))


class _Shim:
    """The real solver's GEOMETRY with the slant hidden -- the honest way to
    force the gauge onto a slanted pencil.

    Monkeypatching ``twod_staggered._slant_is_zero`` does NOT work here: that
    name is also what ``Granet2DTransverseE.__init__`` reads to decide whether
    to apply the shear at all, so patching it silently builds the VERTICAL
    pencil and the "forced" arm measures nothing.  (Found by measurement --
    the "forced" answer reproduced the vertical answer to the digit.)
    """

    def __init__(self, sol):
        self.alpha0x = sol.alpha0x
        self.alpha0y = sol.alpha0y
        self.bx, self.by, self.q = sol.bx, sol.by, sol.q
        self.slant = (0.0, 0.0)


def _dense(A, B):
    import scipy.linalg as sla
    Lc = np.linalg.cholesky(B)
    Ah = sla.solve_triangular(Lc, A, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    return qv, sla.solve_triangular(Lc.conj().T, Y, lower=False)


print("B11a  PARITY ACCELERATOR vs a SLANT (centro cell, NORMAL incidence)")
rows = []
for sn, sv in (("vertical", None), ("x35", (T35, 0.0)),
               ("diag35", (T35 / np.sqrt(2), T35 / np.sqrt(2)))):
    sol = Granet2DTransverseE(PX, PY, 2, 2, 6, CENTRO, alpha0x=0.0,
                              alpha0y=0.0, k0=K0, slant=sv)
    g = _stag_parity_gauge(sol)
    g_forced = _stag_parity_gauge(_Shim(sol))
    dA = struct_dA(sol, g_forced) if g_forced is not None else None
    qq = sol.q * sol.q
    fac = (_stag_block_eig(sol.Agen, sol.Bgen, qq, g_forced)
           if g_forced is not None else None)
    spec = resid = None
    if fac is not None:
        qr, Xr = fac
        qd, _Xd = _dense(sol.Agen, sol.Bgen)
        spec = max(max(float(np.min(np.abs(qd - z))) for z in qr),
                   max(float(np.min(np.abs(qr - z))) for z in qd))
        resid = float(np.max(np.abs(sol.Agen @ Xr
                                    - sol.Bgen @ Xr * qr[None, :]))
                      / np.max(np.abs(sol.Agen @ Xr)))
    rows.append({"slant": sn, "gauge_refused": g is None,
                 "struct_dA_forced": dA,
                 "forced_reduction_would_run": fac is not None,
                 "forced_spectrum_gap": spec, "forced_pencil_resid": resid,
                 "tol": _STAG_BLOCK_TOL})
    print("  %-9s  gauge refused=%-5s  forced struct |RAR+A|/|A| = %s "
          "(tol %.0e)  would run=%-5s  spec gap %s  pencil resid %s"
          % (sn, g is None, "None" if dA is None else "%.2e" % dA,
             _STAG_BLOCK_TOL, fac is not None,
             "-" if spec is None else "%.1e" % spec,
             "-" if resid is None else "%.1e" % resid))
res["B11a_gauge"] = rows

print("")
print("B11a2  symmetry='auto' vs False on a SLANTED cell (sha256, e2e)")
sha = {}
J_slant = J_vert = None
for sn, sv in (("vertical", None), ("x35", (T35, 0.0))):
    for sym in ("auto", False):
        o, R, T, J = pmm_jones_2d_staggered(
            PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5, n_orders=3,
            theta=0.0, phi=0.0, symmetry=sym, slant=sv)
        sha["%s_%s" % (sn, sym)] = h(R, T, J)
        if sn == "x35" and sym == "auto":
            J_slant = J.copy()
        if sn == "vertical" and sym == "auto":
            J_vert = J.copy()
res["B11a2"] = {
    "slant_auto_eq_false": sha["x35_auto"] == sha["x35_False"],
    "vert_auto_eq_false": sha["vertical_auto"] == sha["vertical_False"],
    "slant_eq_vertical": sha["x35_auto"] == sha["vertical_auto"],
    "dJ_slant_vs_vertical": float(np.max(np.abs(J_slant - J_vert))),
    "sha": {k: v[:16] for k, v in sha.items()}}
print("  slanted: auto == False -> %s ; slanted == vertical -> %s "
      "(dJones %.2e)"
      % (res["B11a2"]["slant_auto_eq_false"],
         res["B11a2"]["slant_eq_vertical"],
         res["B11a2"]["dJ_slant_vs_vertical"]))
print("  vertical control: auto == False -> %s (the reduction ENGAGES there, "
      "so equality is the accelerator's own bit-identity claim)"
      % res["B11a2"]["vert_auto_eq_false"])

# does the reduction actually engage on the vertical control?  (a decision,
# asserted before any agreement claim)
solv = Granet2DTransverseE(PX, PY, 2, 2, 6, CENTRO, k0=K0)
gv = _stag_parity_gauge(solv)
res["B11a2"]["vertical_gauge_present"] = gv is not None
res["B11a2"]["vertical_reduction_runs"] = (
    gv is not None
    and _stag_block_eig(solv.Agen, solv.Bgen, solv.q * solv.q, gv) is not None)
print("  vertical control: gauge present=%s  reduction runs=%s"
      % (res["B11a2"]["vertical_gauge_present"],
         res["B11a2"]["vertical_reduction_runs"]))

# ------------------------------------------------------------------ B11b
print("")
print("B11b  REFUSALS")
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
ref = {}


def expect(name, fn):
    try:
        fn()
        ref[name] = "NO RAISE"
    except NotImplementedError as e:
        ref[name] = "NotImplementedError: " + str(e)[:120]
    except Exception as e:                                   # noqa: BLE001
        ref[name] = type(e).__name__ + ": " + str(e)[:120]
    print("  %-28s -> %s" % (name, ref[name][:110]))


def mixed_patterned():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.add_layer(DEP, eps_cell=SCA)
    st.set_source(WL)
    st.solve()


def slanted_above_pattern():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps=2.1, slant=(T35, 0.0))
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.add_layer(DEP, eps=2.1)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.set_source(WL)
    st.solve()


def magnetic_slant():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA, mu=1.4, slant=(T35, 0.0))


def retain_internal_slant():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.set_source(WL)
    st.solve(retain_internal=True)


def eff_slant():
    pmm_efficiency_2d_staggered(PX, PY, np.real(SCA), NSUB, NSUP, DEP, WL,
                                degree=4, n_orders=3, slant=(T35, 0.0))


def jones_mu_slant():
    pmm_jones_2d_staggered(PX, PY, SCA, NSUB, NSUP, DEP, WL, degree=4,
                           n_orders=3, mu_cell=np.full((2, 2), 1.4),
                           slant=(T35, 0.0))


def bad_slant_shape():
    PMM2DStackPure(PX, PY, n_modes=4).add_layer(DEP, eps=2.1,
                                                slant=(0.1, 0.2, 0.3))


expect("mixed patterned slants", mixed_patterned)
expect("slanted above a pattern", slanted_above_pattern)
expect("magnetic + slant", magnetic_slant)
expect("retain_internal + slant", retain_internal_slant)
expect("efficiency entry + slant", eff_slant)
expect("jones mu_cell + slant", jones_mu_slant)
expect("bad slant shape", bad_slant_shape)

# and the ACCEPTED shapes, which must NOT raise
print("  -- accepted --")


def accepted(name, fn):
    try:
        fn()
        ref["ok_" + name] = "ok"
        print("  %-28s -> ok" % name)
    except Exception as e:                                   # noqa: BLE001
        ref["ok_" + name] = type(e).__name__ + ": " + str(e)[:120]
        print("  %-28s -> %s" % (name, ref["ok_" + name][:110]))


def one_pattern_uniform_films():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps=2.1)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.add_layer(DEP, eps=1.8, slant=(T35, 0.0))
    st.set_source(WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()


def two_same_slant_patterns():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    st.add_layer(DEP, eps_cell=SCA * 0.9, slant=(T35, 0.0))
    st.set_source(WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()


def uniform_only_mixed_slants():
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=4, n_orders=3)
    st.add_layer(DEP, eps=2.1, slant=(T35, 0.0))
    st.add_layer(DEP, eps=1.8, slant=(0.0, 0.2))
    st.set_source(WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()


accepted("1 pattern + uniform films", one_pattern_uniform_films)
accepted("2 patterns, same slant", two_same_slant_patterns)
accepted("uniform-only mixed slants", uniform_only_mixed_slants)
res["B11b"] = ref

# ---------------------------------------------------------------- WOOD
print("")
print("WOOD  what a slanted SCALAR layer contributes to the nudge list")

EPSL = 4.0
tw = 0.6
cov = _slant_congruence(EPSL * np.eye(3, dtype=complex), -tw, 0.0)
res["WOOD_lab_diag"] = [float(np.real(EPSL))] * 3
res["WOOD_cov_diag"] = [float(np.real(cov[i, i])) for i in range(3)]
print("  lab diag %s   covariant diag %s"
      % (res["WOOD_lab_diag"], ["%.4f" % v for v in res["WOOD_cov_diag"]]))

# walk an order EXACTLY onto each candidate cut-off and see which nudge fires
# and which reading corresponds to a REAL layer cut-off.
mo = np.arange(-3, 4)
mx = np.tile(mo, len(mo))
my = np.repeat(mo, len(mo))
wood_rows = []
for lab, epsv in (("lab eps", EPSL),
                  ("covariant e11 = eps(1+t^2)", res["WOOD_cov_diag"][0])):
    # choose wl so that order (+1, 0) sits exactly on |kt|^2 = epsv at normal
    wl_on = PX * float(np.sqrt(epsv))
    for listname, lst in (("lab", _wood_eps_reals(1.0, NSUB ** 2, np.real(
            np.array([[EPSL, 1.0], [1.0, 1.0]])))),
            ("covariant", _wood_eps_reals(1.0, NSUB ** 2, np.array(
                res["WOOD_cov_diag"] + [1.0, 1.0, 1.0])))):
        wl_new = _grazing_safe_wavelength(wl_on, 0.0, 0.0, mx, my, PX, PY, lst)
        wood_rows.append({"cutoff_of": lab, "list": listname,
                          "wl_on": wl_on, "wl_nudged": wl_new,
                          "moved": bool(wl_new != wl_on),
                          "rel_move": abs(wl_new - wl_on) / wl_on})
        print("  order lands on %-26s | %-9s list -> nudge fires=%-5s "
              "(rel %.2e)" % (lab, listname, wood_rows[-1]["moved"],
                              wood_rows[-1]["rel_move"]))
res["WOOD_nudge"] = wood_rows

# Is the COVARIANT diagonal a real cut-off of the SLANTED layer?  The layer's
# null-mode condition is q = 0, i.e. kz_root(eps; alpha) = -t.alpha, i.e.
# eps = |alpha|^2 + (t.alpha)^2 -- ORDER-DEPENDENT, so no single eps in a flat
# list expresses it.  Measure the layer's actual min |q| at each candidate.
minq = []
for lab, epsv in (("lab eps", EPSL),
                  ("covariant e11", res["WOOD_cov_diag"][0]),
                  ("true null eps = |a|^2 + (t.a)^2", None)):
    if epsv is None:
        a = 1.0 / 1.0            # order (+1,0) at normal: alpha = wl/px = 1
        epsv = 1.0 + (tw * 1.0) ** 2
        wl_on = PX * 1.0
    else:
        wl_on = PX * float(np.sqrt(epsv))
    k0l = 2.0 * np.pi / wl_on
    cellw = np.full((2, 2), _C := complex(EPSL))
    sol = Granet2DTransverseE(PX, PY, 2, 2, 5, cellw, k0=k0l, slant=(tw, 0.0))
    _Wf, _Vf, lam_f, _Wb, _Vb, _lb = _region_modes_oop(sol)
    qs = np.abs(-1j * np.concatenate([lam_f, _lb]))
    minq.append({"candidate": lab, "eps": float(np.real(epsv)),
                 "wl": wl_on, "min_abs_q": float(np.min(qs))})
    print("  candidate %-33s eps %7.4f -> layer min|q| %.3e"
          % (lab, float(np.real(epsv)), minq[-1]["min_abs_q"]))
res["WOOD_minq"] = minq

res["wall_s"] = time.time() - t00
write("g5_parity_refusals_wood", res)

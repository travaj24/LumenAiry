"""V1 -- THE WITHOUT-ARM.  ``slant`` absent / zero must leave the whole library
BIT-IDENTICAL to the pre-slant tree.

Two arms, two CHECKOUTS:

  * ``base`` = ``D:/.../Lumenairy`` @ 2efc7a2 -- everything on ``wave2/pmm2d``
    EXCEPT the slant.  It has no ``slant=`` keyword at all, so it runs the
    fixtures bare;
  * ``tip``  = ``C:/tmp/lum_vslant`` @ 8b9af801 -- the same fixtures bare AND
    once per "zero" spelling (``None``, ``0.0``, ``(0, 0)``, ``[0, 0]``,
    ``np.zeros(2)``).

Every fixture is hashed by sha256 over the RAW BYTES of R, T and (where the
entry returns one) the Jones matrix.  The gate is 0 differing bytes both
WITHIN the tip (bare vs every zero spelling) and ACROSS the two checkouts.

16 fixtures: scalar / in-plane tensor / out-of-plane tensor / magnetic /
uniform-tensor / two-layer and three-layer stacks, at normal, oblique and
conical incidence, through three entry points (``pmm_jones_2d_staggered``,
``pmm_efficiency_2d_staggered``, ``PMM2DStackPure.solve``), including the
``symmetry='auto'`` reduction path (which the slant build touched).

Run:
    python validation/probe_verify_slant/v1_without_identity.py
"""
import numpy as np
from _lib import arm, dump, sha  # noqa: I001  (probe-local)

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5

OB = (np.deg2rad(25.0), 0.0)
CON = (np.deg2rad(25.0), np.deg2rad(40.0))
NORM = (0.0, 0.0)

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
INP = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)


def _tile(t, n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t
    return c


SCA2 = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
SCA3 = np.array([[4.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 4.0]],
                dtype=complex)
INPC = _tile(AIR)
INPC[0, 0] = INP
OOPC = _tile(AIR)
OOPC[0, 0] = TIL
# a CENTRO-symmetric out-of-plane cell -- this one is eligible for the
# symmetry='auto' parity reduction at normal incidence, so it exercises the
# code path _stag_parity_gauge's new slant refusal sits in.
CENTRO = _tile(AIR)
CENTRO[0, 0] = CENTRO[1, 1] = TIL
MUC = np.zeros((2, 2, 3, 3), dtype=complex)
MUC[:, :] = np.eye(3)
MUC[0, 0] = np.diag([1.4, 1.2, 1.1])


def _jones(cell, theta, phi, M=5, sym="auto"):
    return pmm_jones_2d_staggered(
        PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M, n_orders=3,
        theta=theta, phi=phi, symmetry=sym)


def _stack_two(theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(0.21e-6, eps=2.10 + 0j)
    st.add_layer(DEP, eps_cell=SCA2)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def _stack_three(theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(0.15e-6, eps=np.diag([2.1, 2.4, 2.2]).astype(complex))
    st.add_layer(DEP, eps_cell=OOPC)
    st.add_layer(0.11e-6, eps_cell=SCA2)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def _stack_mu(theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA2, mu_cell=MUC)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def _hash_res(res):
    """sha256 over the raw bytes of every array the entry returns."""
    if len(res) == 4:
        o, R, T, J = res
        return sha(R, T, J)
    o, R, T = res[0], res[1], res[2]
    return sha(R, T)


# name -> (callable taking a slant-kwarg dict, supports_slant_kw)
FIXTURES = {}


def _add(name, fn, slant_kw=True):
    FIXTURES[name] = (fn, slant_kw)


for mname, (th, ph) in (("normal", NORM), ("oblique", OB), ("conical", CON)):
    _add(f"jones_scalar_{mname}",
         lambda kw, th=th, ph=ph: _hash_res(
             pmm_jones_2d_staggered(PX, PY, SCA2, NSUB, NSUP, DEP, WL,
                                    degree=5, n_orders=3, theta=th, phi=ph,
                                    **kw)))
    _add(f"jones_inplane_{mname}",
         lambda kw, th=th, ph=ph: _hash_res(
             pmm_jones_2d_staggered(PX, PY, INPC, NSUB, NSUP, DEP, WL,
                                    degree=5, n_orders=3, theta=th, phi=ph,
                                    **kw)))
    _add(f"jones_oop_{mname}",
         lambda kw, th=th, ph=ph: _hash_res(
             pmm_jones_2d_staggered(PX, PY, OOPC, NSUB, NSUP, DEP, WL,
                                    degree=5, n_orders=3, theta=th, phi=ph,
                                    **kw)))

# the parity-reduction path: a centro-symmetric OOP cell at NORMAL incidence,
# symmetry='auto' (the accelerator engages) and symmetry=False (dense).
_add("jones_centro_auto_normal",
     lambda kw: _hash_res(pmm_jones_2d_staggered(
         PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5, n_orders=3,
         symmetry="auto", **kw)))
_add("jones_centro_false_normal",
     lambda kw: _hash_res(pmm_jones_2d_staggered(
         PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5, n_orders=3,
         symmetry=False, **kw)))
# a 3x3 grid, the other admissible union grid
_add("jones_scalar3_conical",
     lambda kw: _hash_res(pmm_jones_2d_staggered(
         PX, PY, SCA3, NSUB, NSUP, DEP, WL, degree=4, n_orders=3,
         theta=CON[0], phi=CON[1], **kw)))
# the scalar EFFICIENCY entry (both polarizations), which on the tip accepts
# slant= only to raise -- so the zero spellings must all pass through it.
_add("eff_te_oblique",
     lambda kw: _hash_res(pmm_efficiency_2d_staggered(
         PX, PY, SCA2, NSUB, NSUP, DEP, WL, degree=5, n_orders=3,
         polarization="te", theta=OB[0], phi=OB[1], **kw)))
_add("eff_tm_conical",
     lambda kw: _hash_res(pmm_efficiency_2d_staggered(
         PX, PY, SCA2, NSUB, NSUP, DEP, WL, degree=5, n_orders=3,
         polarization="tm", theta=CON[0], phi=CON[1], **kw)))

# stacks (slant is a per-LAYER keyword there, so these arms are run with the
# keyword on EVERY layer)
_add("stack2_normal", lambda kw: _hash_res(_stack_two(*NORM)), slant_kw=False)
_add("stack2_conical", lambda kw: _hash_res(_stack_two(*CON)), slant_kw=False)
_add("stack3_oblique", lambda kw: _hash_res(_stack_three(*OB)),
     slant_kw=False)
_add("stack_mu_conical", lambda kw: _hash_res(_stack_mu(*CON)),
     slant_kw=False)


def _stack_two_sl(sl, theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(0.21e-6, eps=2.10 + 0j, slant=sl)
    st.add_layer(DEP, eps_cell=SCA2, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def _stack_three_sl(sl, theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(0.15e-6, eps=np.diag([2.1, 2.4, 2.2]).astype(complex),
                 slant=sl)
    st.add_layer(DEP, eps_cell=OOPC, slant=sl)
    st.add_layer(0.11e-6, eps_cell=SCA2, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


def _stack_mu_sl(sl, theta, phi, M=5):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(DEP, eps_cell=SCA2, mu_cell=MUC, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


STACK_SL = {
    "stack2_normal": lambda sl: _hash_res(_stack_two_sl(sl, *NORM)),
    "stack2_conical": lambda sl: _hash_res(_stack_two_sl(sl, *CON)),
    "stack3_oblique": lambda sl: _hash_res(_stack_three_sl(sl, *OB)),
    "stack_mu_conical": lambda sl: _hash_res(_stack_mu_sl(sl, *CON)),
}

ZEROS = {
    "none": None,
    "float0": 0.0,
    "tuple00": (0, 0),
    "list00": [0, 0],
    "npzeros2": np.zeros(2),
    "int0": 0,
    "npfloat0": np.float64(0.0),
}


def main():
    a = arm()
    out = {"bare": {}, "spellings": {}}
    for name, (fn, _sl) in FIXTURES.items():
        out["bare"][name] = fn({})
    if a == "tip":
        for zname, zval in ZEROS.items():
            row = {}
            for name, (fn, sl_kw) in FIXTURES.items():
                if sl_kw:
                    row[name] = fn({"slant": zval})
                else:
                    row[name] = STACK_SL[name](zval)
            out["spellings"][zname] = row
    dump("v1_without_identity", out)

    # same-arm summary
    if a == "tip":
        bad = []
        for zname, row in out["spellings"].items():
            for name, h in row.items():
                if h != out["bare"][name]:
                    bad.append((zname, name))
        print(f"tip: {len(FIXTURES)} fixtures x {len(ZEROS)} zero spellings; "
              f"mismatches vs bare: {len(bad)} {bad}")
    print(f"arm={a} fixtures={len(FIXTURES)}")


if __name__ == "__main__":
    main()

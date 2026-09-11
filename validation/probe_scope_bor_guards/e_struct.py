"""E-STRUCT -- structural census of the NON-Cartesian / other multilayer engines.

Reads only.  Answers, per engine:
  * class A  : which sqrt/branch site it uses, and whether that site is the
               SHARED round-2 ``rcwa/_core._sqrt_decay`` (relative band + conj)
               or a private rule with an EXACT-zero / exact-sign pin;
  * class B  : every explicit inverse / solve / lstsq on its cascade, and
               whether it routes through ``_guarded_inverse`` /
               ``_guarded_mortar_solve`` / ``_guarded_lstsq``;
  * class C  : whether it builds a union grid, a per-layer grid or a mortar,
               and whether a minimum-width contract exists there.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_struct.py out.json
"""
from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

RAW_SITES = re.compile(r"(np|jnp)\.linalg\.(inv|solve|lstsq|pinv)\s*\(")
GUARDED = re.compile(
    r"_guarded_(inverse|mortar_solve|lstsq)\s*\(|_safe_(inv|solve)\s*\(")


def scan(modname):
    import importlib
    m = importlib.import_module(modname)
    path = os.path.abspath(m.__file__)
    lines = open(path, "r", encoding="utf-8", errors="replace").read().split("\n")
    raw, guarded, sqrt_sites = [], [], []
    for i, ln in enumerate(lines, 1):
        s = ln.split("#")[0]
        if RAW_SITES.search(s):
            raw.append((i, ln.strip()[:110]))
        if GUARDED.search(s):
            guarded.append((i, ln.strip()[:110]))
        if re.search(r"(np|jnp)\.sqrt\(|_sqrt_decay\(|_sqrt_forward\(", s):
            sqrt_sites.append((i, ln.strip()[:110]))
    return dict(path=path, raw_linalg=raw, guarded=guarded, sqrt=sqrt_sites)


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_struct.json"

    from lumenairy.elements import _berreman_jax as BJ
    from lumenairy.elements import berreman as B
    from lumenairy.elements import coatings as CO
    from lumenairy.elements import thin_grating as TG
    from lumenairy.elements.eme import eme_2d as E2
    from lumenairy.elements.eme import eme_2d_vector as E2V
    from lumenairy.elements.eme import eme_diffraction as ED
    from lumenairy.elements.pmm import _core as pcore
    from lumenairy.elements.pmm import stack as pstack
    from lumenairy.elements.pmm import stack2d as p2h
    from lumenairy.elements.pmm import stack2d_pure as p2p
    from lumenairy.elements.pmm import twod as ptw
    from lumenairy.elements.pmm import twod_staggered as pstag
    from lumenairy.elements.rcwa import _core as rc
    from lumenairy.elements.rcwa import stack as rstack

    res = {}
    res["shared_sqrt_decay"] = dict(
        berreman_is_shared=(B._sqrt_decay is rc._sqrt_decay),
        pmm_twod_is_shared=(ptw._sqrt_decay is rc._sqrt_decay),
        cut_band_rel=rc._CUT_BAND_REL,
        berreman_has_private_def=("def _sqrt_decay" in E.src(B)),
        berreman_jax_has_private_def=("def _sqrt_decay" in E.src(BJ)),
    )
    res["guard_constants"] = dict(
        INV_RESID_REFUSE=rc._INV_RESID_REFUSE,
        INV_T22_RCOND_REFUSE=rc._INV_T22_RCOND_REFUSE,
        INTERFACE_CONDITIONING_GUARD=rc.INTERFACE_CONDITIONING_GUARD,
        MORTAR_RCOND_REFUSE=pcore._MORTAR_RCOND_REFUSE,
        MORTAR_RESID_REFUSE=pcore._MORTAR_RESID_REFUSE,
        LSTSQ_RESID_BAR=pcore._LSTSQ_RESID_BAR,
        PMM_SLIVER_GUARD=pstack.PMM_SLIVER_GUARD,
        STACK_SUPERUNITY_BAR=pstack._STACK_SUPERUNITY_BAR,
        SLIVER_TRIGGER_BAR=pstack._SLIVER_TRIGGER_BAR,
        SLIVER_OWN_SCALE_RATIO=pstack._SLIVER_OWN_SCALE_RATIO,
        SLIVER_Q_EXCESS=pstack._SLIVER_Q_EXCESS,
        STAG_MIN_SEG_FRAC=pstag._STAG_MIN_SEG_FRAC,
        STAG_SLIVER_BAND_FRAC=pstag._STAG_SLIVER_BAND_FRAC,
        STAG_CLOSURE_TOL=getattr(p2p, "_STAG_CLOSURE_TOL", None),
    )
    res["branch_rules"] = {
        "eme_2d._ky_forward": dict(
            exact_pin=("ky.imag < 0.0" in E.src(E2._ky_forward)),
            line=inspect.getsourcelines(E2._ky_forward)[1]),
        "eme_2d_vector._strip_split_forward": dict(
            relative_band=("1e-9 * max" in E.src(E2V._strip_split_forward)),
            line=inspect.getsourcelines(E2V._strip_split_forward)[1]),
        "eme_diffraction.mode_match(qz)": dict(
            exact_pin=("qz.imag < 0.0" in E.src(ED.mode_match)),
            line=inspect.getsourcelines(ED.mode_match)[1]),
        "berreman._split_fwd_bwd": dict(
            relative_band=("1e-9 * max" in E.src(B._split_fwd_bwd)),
            line=inspect.getsourcelines(B._split_fwd_bwd)[1]),
        "pmm.twod._kz_forward2": dict(
            exact_pin=("val.imag < 0.0" in E.src(ptw._kz_forward2)),
            line=inspect.getsourcelines(ptw._kz_forward2)[1]),
        "coatings._cos_theta": dict(
            exact_pin=(".imag < 0.0" in E.src(CO.coating_reflectance)),
            abs_floor=("abs(ct) < 1e-12" in E.src(CO.coating_reflectance)),
            line=inspect.getsourcelines(CO.coating_reflectance)[1]),
    }
    mods = ["lumenairy.elements.eme.eme_2d",
            "lumenairy.elements.eme.eme_2d_vector",
            "lumenairy.elements.eme.eme_diffraction",
            "lumenairy.elements.berreman",
            "lumenairy.elements._berreman_jax",
            "lumenairy.elements.rcwa.stack",
            "lumenairy.elements.rcwa._core",
            "lumenairy.elements.pmm.stack",
            "lumenairy.elements.pmm.stack2d",
            "lumenairy.elements.pmm.stack2d_pure",
            "lumenairy.elements.pmm.twod",
            "lumenairy.elements.pmm.twod_staggered",
            "lumenairy.elements.thin_grating",
            "lumenairy.elements.coatings"]
    res["modules"] = {m: scan(m) for m in mods}
    res["berreman_jax_sqrt_call"] = [
        ln.strip() for ln in E.src(BJ).split("\n") if "_sqrt_decay(" in ln]
    res["thin_grating"] = dict(
        functions=[n for n, o in vars(TG).items() if inspect.isfunction(o)
                   and o.__module__ == TG.__name__],
        has_cascade=any(k in E.src(TG) for k in
                        ("redheffer", "_star(", "smatrix", "linalg")),
        reflection_is_zero=("R_eff = np.zeros" in E.src(TG)))
    res["coatings"] = dict(
        functions=[n for n, o in vars(CO).items() if inspect.isfunction(o)
                   and o.__module__ == CO.__name__],
        cascade_kind=("2x2 characteristic (Abeles) transfer-matrix product"
                      if "characteristic-matrix" in E.src(CO) else "unknown"),
        has_explicit_inverse=bool(RAW_SITES.search(E.src(CO))))
    rcsrc = E.src(rc)
    res["rcwa_armed"] = dict(
        guarded_inverse_sites=[ln.strip() for ln in rcsrc.split("\n")
                               if "_guarded_inverse(" in ln and "def " not in ln],
        t22_armed=("rcond_refuse=_INV_T22_RCOND_REFUSE" in rcsrc))
    h = E.src(p2h)
    res["hybrid_grid"] = dict(
        uses_union_grid=("_pmm_union_grid" in h),
        uses_mortar=("mortar" in h.lower()),
        builds_per_layer_axis=("_build_axis(" in h),
        min_width_contract=("MIN_SEG" in h or "min_feature" in h))
    ptwsrc = E.src(ptw)
    res["build_axis"] = dict(
        min_width_contract=("MIN_SEG" in ptwsrc or "min_feature" in ptwsrc
                            or "too narrow" in ptwsrc),
        has_mass_inverse=("Minv = np.linalg.inv(M)" in ptwsrc),
        has_epn_solve=("np.linalg.solve(P_inv, M)" in ptwsrc),
        inv_lam_abs_floor=("np.abs(lam) < 1e-12" in ptwsrc))
    res["pure_grid"] = dict(
        uses_mortar=("mortar" in E.src(p2p).lower()),
        guarded_lstsq=("_guarded_lstsq(" in E.src(p2p)),
        interface_general=("_interface_smatrix_general(" in E.src(p2p)),
        mortar_2d=("_interface_smatrix_general_mortar_2d" in E.src(p2p)))
    res["pmmstack_grid"] = dict(
        uses_union_grid=("_pmm_union_grid" in E.src(pstack)),
        sliver_guard=("PMM_SLIVER_GUARD" in E.src(pstack)),
        guarded_lstsq=("_guarded_lstsq" in E.src(pstack)))
    res["rcwastack"] = dict(
        sqrt_decay=("_sqrt_decay" in E.src(rstack)),
        raw_inverses=scan("lumenairy.elements.rcwa.stack")["raw_linalg"])
    E.dump(out, res)


if __name__ == "__main__":
    main()

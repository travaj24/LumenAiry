"""E-CENSUS -- merge the per-probe ``e_*_<build>_<threads>.json`` files into the
single deliverable ``e_census_<build>.json``, one file per build, plus the
per-engine verdict row.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_census.py win
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

VERDICTS = {
 "eme (lumenairy/elements/eme/)": {
  "A": {"reachable": True,
        "sites": ["lumenairy/elements/eme/eme_2d.py:115 _ky_forward, pin at eme_2d.py:130 -- EXACT np.where(ky.imag < 0.0, -ky, ky)",
                  "lumenairy/elements/eme/eme_diffraction.py:129 mode_match, pin at eme_diffraction.py:176 -- EXACT np.where(qz.imag < 0.0, -qz, qz)"],
        "guard_today": "NONE.  The in-module VECTOR sibling lumenairy/elements/eme/eme_2d_vector.py:251 _strip_split_forward already uses the correct RELATIVE band (eme_2d_vector.py:255, tol = 1e-9 * max(1, max|ky|)); the scalar path does not.",
        "verdict": "NO-GO -- build-dependent forward/backward mode assignment on PROPAGATING modes, and a mode set that changes at Im(eps) = 0+."},
  "B": {"reachable": True,
        "sites": ["lumenairy/elements/eme/eme_2d.py:145 ia = np.linalg.inv(a + b)   (peer of the guarded rcwa interface mode-match)",
                  "lumenairy/elements/eme/eme_2d.py:159/160 np.linalg.inv(I - B11 A22) / (I - A22 B11)   (peer of the guarded rcwa Redheffer star)",
                  "lumenairy/elements/eme/eme_2d.py:142/143 np.linalg.solve(Wb, Wa) / np.linalg.solve(Vb, Va)",
                  "lumenairy/elements/eme/eme_diffraction.py:192 np.linalg.lstsq(A, rhs, rcond=None)"],
        "guard_today": "NONE at any of them; no census hook exists either.",
        "verdict": "GO -- port the guard.  Healthy away from a band edge, degrades unwatched next to one."},
  "C": {"reachable": False,
        "why": "no union grid and no mortar: every strip shares ONE uniform Nx x-grid and the y direction is ANALYTIC (exp(i ky h)).  The structural analogue, a vanishing strip height, converges monotonically.",
        "verdict": "NOT-APPLICABLE"}},

 "berreman (lumenairy/elements/berreman.py)": {
  "A": {"reachable": True,
        "sites": ["lumenairy/elements/berreman.py:594 lam = _sqrt_decay(...) -- the SHARED round-2 implementation (identity-checked: berreman._sqrt_decay IS rcwa._core._sqrt_decay; no private def in berreman.py or _berreman_jax.py)",
                  "lumenairy/elements/berreman.py:170 _split_fwd_bwd -- already a RELATIVE band (berreman.py:188, tol = 1e-9 * max(1, max|gam|)) plus a stable argsort"],
        "guard_today": "_CUT_BAND_REL = 1e-8 (shared) + the relative _split_fwd_bwd band.",
        "verdict": "GO -- already guarded and verified."},
  "B": {"reachable": True,
        "sites": ["lumenairy/elements/rcwa/_core.py:2875 _interface_smatrix_general T22 -- ARMED (_INV_T22_RCOND_REFUSE = 1e-10); the Berreman planar AND off-plane cascades both route through it",
                  "lumenairy/elements/rcwa/_core.py:2703/2704 _redheffer_star -- census only",
                  "lumenairy/elements/berreman.py:1126 np.linalg.inv(eye(2) - A22 Sb11)  UNGUARDED (2x2, post-cascade amplitude recovery)",
                  "lumenairy/elements/berreman.py:508/1055/1081 np.linalg.solve(Wf_s, Einc)  UNGUARDED (2x2 half-space mode matrix)"],
        "guard_today": "T22 ARMED; the two 2x2 sites unguarded.",
        "verdict": "GO -- already guarded where it matters."},
  "C": {"reachable": False,
        "why": "planar 1-D: a single Rayleigh order, no lateral grid, no mortar, no union grid.",
        "verdict": "NOT-APPLICABLE"}},

 "RCWAStack (lumenairy/elements/rcwa/stack.py + _core.py)": {
  "A": {"reachable": True,
        "sites": ["lumenairy/elements/rcwa/_core.py:1284 _sqrt_decay -- THE definition (relative band + conj flip)",
                  "called at _core.py:2360/2462/2480/2553/2570/2618/3870/3882 and stack.py:2608"],
        "guard_today": "_CUT_BAND_REL = 1e-8.",
        "verdict": "GO -- reference implementation, already guarded."},
  "B": {"reachable": True,
        "sites": ["_core.py:2875 generalized interface T22 -- ARMED at 1e-10",
                  "_core.py:2741 interface mode-match (a+b) -- guarded fn, census only",
                  "_core.py:2703/2704 Redheffer star -- guarded fn, census only",
                  "stack.py:942 np.linalg.inv(eye(2N) - Sa22 Sb11) UNGUARDED (post-cascade amplitude recovery)",
                  "stack.py:1086 np.linalg.solve(EPS, rhs) UNGUARDED (internal-field Ez)"],
        "guard_today": "_guarded_inverse at 4 cascade sites, refusal ARMED only at T22; INTERFACE_CONDITIONING_GUARD = True; _INV_RESID_REFUSE = 1e-8.",
        "verdict": "GO -- already guarded and verified."},
  "C": {"reachable": False,
        "why": "pure Fourier (Rayleigh) basis: no spectral-element grid, no union grid, no mortar.  A narrow feature is a truncation question, not a sliver.",
        "verdict": "NOT-APPLICABLE"}},

 "PMMStack 1-D (lumenairy/elements/pmm/stack.py)": {
  "A": {"reachable": True,
        "sites": ["shared _sqrt_decay via lumenairy/elements/pmm/twod.py:92 import (identity-checked)",
                  "lumenairy/elements/pmm/_core.py _forward_branch_flip (relative band, the pattern _sqrt_decay was aligned to)"],
        "guard_today": "_CUT_BAND_REL = 1e-8.", "verdict": "GO -- already guarded."},
  "B": {"reachable": True,
        "sites": ["pmm/_core.py 'pmm interface mode-match (a+b)' guarded inverse",
                  "rcwa/_core.py:2875 T22 ARMED",
                  "pmm/stack.py:2721 and :2934 _guarded_lstsq (_LSTSQ_RESID_BAR = 1e-9, rank AND residual)",
                  "pmm/stack.py:3289 np.linalg.solve(M, A21 @ cinc) UNGUARDED (post-cascade amplitude recovery)"],
        "guard_today": "3 armed families.", "verdict": "GO -- already guarded and verified."},
  "C": {"reachable": True,
        "sites": ["pmm/_core.py:4467 _pmm_union_grid (walls unioned ACROSS layers)"],
        "guard_today": "PMM_SLIVER_GUARD = True with _STACK_SUPERUNITY_BAR 1e-2, _SLIVER_TRIGGER_BAR 1e-3, _SLIVER_OWN_SCALE_RATIO 100, _SLIVER_ATTRIB_CLOSURE 1e-5, _SLIVER_CLOSURE_FRACTION 1e-2, _SLIVER_MOVE_FACTOR 100, _SLIVER_Q_EXCESS 1e6, plus the min_feature wall SNAP + UserWarning.",
        "verdict": "GO -- already guarded and verified (measured two-sided)."}},

 "PMM2DStackHybrid (lumenairy/elements/pmm/stack2d.py)": {
  "A": {"reachable": True,
        "sites": ["shared _sqrt_decay via pmm/twod.py:668/755/778 (round 2 deleted the private copy; identity-checked here)",
                  "pmm/twod.py:421 _kz_forward2 -- EXACT val.imag < 0.0, but its argument (eps - kx^2 - ky^2) is EXACTLY real for a real eps, so the test is decided in exact arithmetic"],
        "guard_today": "_CUT_BAND_REL = 1e-8.", "verdict": "GO -- already guarded (round 2)."},
  "B": {"reachable": True,
        "sites": ["pmm/_core.py 'pmm interface mode-match (a+b)' guarded inverse",
                  "rcwa/_core.py:2875 T22 ARMED",
                  "pmm/stack2d.py:839 EPS_inv = np.linalg.inv(EpsF) UNGUARDED (Laurent formulation only)",
                  "pmm/twod.py:393 Minv = np.linalg.inv(M) and :408 np.linalg.solve(P_inv, M) UNGUARDED (SEM assembly)",
                  "pmm/stack2d.py:1794/1828 np.linalg.solve UNGUARDED (post-cascade)"],
        "guard_today": "cascade guarded; the SEM-assembly and Laurent-EPS inverses are not.",
        "verdict": "GO -- cascade already guarded; the assembly inverses were measured benign to cond 1.06e15."},
  "C": {"reachable": False,
        "why": "NO union grid and NO mortar -- every layer is Fourier-Galerkin PROJECTED into the shared Rayleigh basis, so the LAYER MODES live in ORDER space (measured modal width 162 = 2*(2*4+1)^2 at n_orders 4).  A narrow spectral element therefore cannot inject a spurious modal wavenumber the way the 1-D union grid (C1) and the 2-D L2 mortar (C2) can; it only supplies projected operator entries.  pmm/twod.py:203 _build_axis carries NO minimum-width contract and none was needed at any width measured.",
        "verdict": "NOT-APPLICABLE (measured, not assumed)"}},

 "PMM2DStackPure (lumenairy/elements/pmm/stack2d_pure.py)": {
  "A": {"reachable": True,
        "sites": ["pmm/_core.py _forward_branch_flip / _select_forward_flux (relative band by construction)",
                  "shared _sqrt_decay where the staggered path reaches it"],
        "guard_today": "relative bands throughout.", "verdict": "GO -- already guarded."},
  "B": {"reachable": True,
        "sites": ["pmm/_core.py:5315 _guarded_mortar_solve (_MORTAR_RCOND_REFUSE 1e-12 at the two IN-PLANE sites; _MORTAR_RESID_REFUSE 1e-6 at _interface_smatrix_general_mortar_2d), census hook _MORTAR_SOLVE_CENSUS",
                  "stack2d_pure.py:1560 and :1928 _guarded_lstsq",
                  "rcwa/_core.py:2875 T22 ARMED, reached via stack2d_pure.py:1458-1461",
                  "stack2d_pure.py:1995 np.linalg.solve(M, A21 @ cinc) UNGUARDED (post-cascade)"],
        "guard_today": "4 armed families.", "verdict": "GO -- already guarded and verified."},
  "C": {"reachable": True,
        "sites": ["pmm/twod_staggered.py per-layer L2 mortar; Basis1D non-uniform element grid"],
        "guard_today": "_STAG_MIN_SEG_FRAC = 1e-3 (REFUSE) + _STAG_SLIVER_BAND_FRAC = 3e-2 (degradation-band UserWarning) + _guarded_mortar_solve.",
        "verdict": "GO -- already guarded and verified (measured two-sided: silent >= 3e-2, warn on [1e-3, 3e-2), refuse < 1e-3)."}},

 "thin_grating (lumenairy/elements/thin_grating.py)": {
  "A": {"reachable": False, "why": "no sqrt branch anywhere; the order amplitudes are ANALYTIC Fourier coefficients of a phase screen.", "verdict": "NOT-APPLICABLE"},
  "B": {"reachable": False, "why": "no cascade and no linear algebra at all: zero np.linalg call sites in the module.  Reflection is identically zero by construction (R_eff = np.zeros).", "verdict": "NOT-APPLICABLE"},
  "C": {"reachable": False, "why": "no grid, no layers, no mortar.", "verdict": "NOT-APPLICABLE"}},

 "coatings (lumenairy/elements/coatings.py)": {
  "A": {"reachable": True,
        "sites": ["lumenairy/elements/coatings.py:177 ct = np.sqrt(1 - (n0 sin0 / n)^2 + 0j), pinned by the EXACT `if (n_layer * ct).imag < 0.0: ct = -ct` and then floored by the ABSOLUTE `if abs(ct) < 1e-12: ct = 1e-12`"],
        "guard_today": "the exact sign test plus the absolute 1e-12 floor.  For a REAL n the argument is exactly real, so the test is decided in exact arithmetic; for a complex n the imaginary part is physical.",
        "verdict": "GO -- measured exact at, and either side of, the critical angle."},
  "B": {"reachable": False,
        "why": "the cascade is a 2x2 CHARACTERISTIC (Abeles) TRANSFER-matrix product with a closed-form scalar r/t at the end -- zero np.linalg.inv / solve / lstsq call sites in the module.",
        "verdict": "NOT-APPLICABLE"},
  "C": {"reachable": False, "why": "planar; no lateral grid, no mortar.", "verdict": "NOT-APPLICABLE"}},
}


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="cp1252", errors="replace") as fh:
        return json.load(fh)


def main():
    E.pin_tree()
    build = (sys.argv[1] if len(sys.argv) > 1 else "win").lower()
    out = os.path.join(HERE, "e_census_%s.json" % build)
    payload = {
        "verdicts": VERDICTS,
        "struct": load("e_struct_%s.json" % build),
        "eme": {t: load("e_eme_%s_t%s.json" % (build, t)) for t in ("1", "4", "8")},
        "eme_flip": {t: load("e_eme_flip_%s_t%s.json" % (build, t))
                     for t in ("1", "4", "8")},
        "cascade": {t: load("e_cascade_%s_t%s.json" % (build, t))
                    for t in ("1", "4", "8")},
        "sliver": {t: load("e_sliver_%s_t%s.json" % (build, t))
                   for t in ("1", "4", "8")},
    }
    for k in ("eme", "eme_flip", "cascade", "sliver"):
        payload[k] = {t: v for t, v in payload[k].items() if v is not None}
    E.dump(out, payload)


if __name__ == "__main__":
    main()

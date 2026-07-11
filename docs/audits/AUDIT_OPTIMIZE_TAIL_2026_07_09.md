# Optimize Tail Audit (multiconfig / multi-objective / JIT) — 2026-07-09

Scope: the remainder of `optimize/` — `multiconfig.py` (440, afocal /
zoom / beam-expander / telescope builders), `multi_objective.py` (396,
the pymoo NSGA-II Pareto driver), and `_merit_jit.py` (263, the Numba
tilt-phasor kernel behind `MultiFieldMerit`).  With this tranche the
**entire `optimize/` subsystem is audited**.  Read-only; the afocal
optics and the constraint translation re-derived.

---

## 1. Verdict

**Clean — no findings.**  Verified this pass:

* **`_merit_jit._multi_field_tilt_phasor_masked`** — the fused Numba
  kernel computes exactly `where(mask, exp(1j·(sin_tx·k_X +
  sin_ty·k_Y)), 0)` (`out = cos ph + i·sin ph`), matching the NumPy
  expression documented in `MultiFieldMerit`; the c64/c128
  specialisation (Numba can't erase the output dtype), the pixel-count
  threshold dispatch, the C-contiguity coercions, the lazy numba probe
  (P2-D cold-start), and the pure-NumPy fallback are all correct.  The
  complex64 path casts real/imag separately to avoid a silent upcast.
* **`multiconfig` afocal physics** — the key derivations:
  - `afocal_angular_magnification` uses the correct **C = 0**
    condition (collimated in → collimated out), `M_angular = D`, with
    the dimensional `|C|·aperture_radius < 1e-6` afocal test (a residual
    output angle in radians) — not the B = 0 that would be a 1:1 imager.
  - `_zero_C_air_gap` exploits that ABCD's `C` is exactly linear in a
    single air gap, so two samples + one linear solve nails the afocal
    gap to machine precision; the P1-NEW-G degenerate-slope guard
    raises (instead of the old silent `return g1`) when `C` is
    gap-independent.
  - `beam_expander_prescription` (Galilean, `f_eye = −f_obj/M`) and
    `keplerian_telescope` build equi-shaped singlets via the lensmaker
    `R = 2f(n−1)` (I verified `1/f = (n−1)(1/R − 1/(−R)) = (n−1)·2/R`),
    use the glass+wavelength-aware index (the P1-MC fix replacing the
    hardcoded `n=1.5`, ~17% radius error on N-LASF9), and the
    equi-concave eyepiece fix (the prior plano-concave halved the
    eyepiece power).  Both solve the true afocal gap with a thin-lens
    fallback on solver failure.
* **`multi_objective.design_optimize_multi_objective`** — the pymoo
  NSGA-II wrapper validates ≥2 merits and finite `lb < ub` bounds;
  translates each `Constraint` to the correct pymoo `g(x) ≤ 0` form
  (`lb − f(x) ≤ 0` for a lower bound, `f(x) − ub ≤ 0` for an upper) —
  directions verified; and captures `_f`/`_v` as lambda default args,
  the correct idiom against the late-binding-closure loop bug.  Merits
  map one-to-one to minimised objectives (with the documented
  negation-for-maximisation pattern).

## 2. Findings

**None.**  (Minor observation, not a defect: `_zero_B_air_gap` is kept
as a backwards-compatible alias for the renamed `_zero_C_air_gap` — the
old name was a misnomer, the alias is documented as such.)

## 3. Coverage statement

Deep-read: `_merit_jit.py` in full (both kernels + dispatch +
fallback); `multiconfig.py`'s `afocal_angular_magnification`,
`_zero_C_air_gap`, `beam_expander_prescription`, `keplerian_telescope`,
`multi_config_merit`, `create_zoom_configs`; `multi_objective.py`'s
input validation + constraint translation + problem setup.
Structurally covered: the `multiconfig` `Configuration` dataclass +
`_resolve_lens_glass_index`, and `multi_objective`'s pymoo
`ElementwiseProblem._evaluate` body + `ParetoResult` assembly (pymoo
plumbing).  `core.py` (419) is the re-export + `_json`/`_os` alias
shell (the mock-patch indirection point referenced throughout the
driver/wrapper audits) — verified by role, not line-audited.

**With this tranche the `optimize/` subsystem is complete**: merit_terms
(OPT-1), jax_merits (OPT-1), driver, parameterizations, wrapper_merits
(OPT-2), context, multi_objective, multiconfig, _merit_jit, core.  The
only remaining unaudited library ground is `io/storage.py` (1,604) and
`io/codegen.py` (1,009) — serialisation / code-emission infrastructure.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_OPTIMIZE_DRIVER_2026_07_09.md`,
`AUDIT_OPTIMIZE_MERITS_2026_07_09.md` (OPT-1),
`AUDIT_OPTIMIZE_WRAPPERS_2026_07_09.md` (OPT-2).*

# Lumenairy v4.11.2 — Round-4 Pre-PyPI Audit
Date: 2026-05-16
Codebase: ~74,517 LOC across ~80 Python files
Method: 13 parallel audit agents, mixing (a) verification of every v4.11.2 fix against the
round-3 findings, (b) fresh-eyes audit of modules that weren't touched in v4.11.2, (c)
release-readiness concerns specific to a PyPI deployment.

---

## TL;DR — DO NOT SHIP TO PYPI YET

v4.11.2 successfully closed ~70 of the round-3 findings, including all the headline physics
work (mirror Seidel parity, EVENASPH α₄ Zemax loader, Richards–Wolf `1/f²` + sign,
HFPI Kirchhoff, dtype hygiene, C-LR-1 revert with a real ground-truth pin against
`apply_real_lens_traced`). 243 unit tests + 314 validation tests pass cleanly.

**However, this audit identified ~15 PyPI-release blockers**, ranging from broken README
cookbook examples (the literal first thing a `pip install` user runs) to silently wrong JAX
backend behaviour. Several are unfixed round-3 findings explicitly claimed as fixed in the
v4.11.2 release notes; others are new findings the fix wave introduced or didn't catch.

The fundamental physics work in v4.11.2 is solid. The remaining issues fall into three
patterns: (1) **fixes that didn't reach the sibling code paths** (NumPy fix, JAX twin missed);
(2) **fixes claimed in CHANGELOG but never landed in code**; (3) **modules that weren't
touched but contain known correctness issues** (`ghost.py`, `propagation.py`, `dispatch.py`,
`system.py` JAX path, `_apply_doe_kick_jax`).

---

## PyPI release blockers (must fix before tagging)

### Tier 0 — User-facing first impressions (showstoppers)

**B0-1 README cookbook examples are broken.** README lines 2460-2606 — the "Three minimal
end-to-end examples" block, literally the first thing a new user runs after `pip install
lumenairy` — fails on multiple cookbook lines:

- `la.create_gaussian_beam(N=512, dx=2e-6, sigma=50e-6)` — missing required `wavelength` → `TypeError`.
- `la.apply_real_lens(E, presc, wavelength=..., dx=...)` — positional `presc` → `TypeError` (kw-only since v4.7).
- `la.load_zmx_prescription(...)` — function renamed to `load_zemax_zmx` in v4.7, no back-compat alias → `AttributeError`.

This is a brand-damaging first impression for a brand-new PyPI listing. Single highest-priority fix.

**B0-2 `_deprecation.py` is dead code.** Module exists but no code path imports any of its
helpers (`warn_deprecated_kwarg`, `deprecated_alias`, `warn_renamed_function`). Renamed
functions like `load_zmx_prescription` should have `deprecated_alias` shims at the top level — they don't. Users with pre-v4.7 code hit cold `AttributeError`.

### Tier 1 — Silently-wrong physics in default code paths

**B1-1 JAX/NumPy aperture schemas STILL incompatible.** `system.py:556` reads
`params.get('radius')` while NumPy `apply_aperture` uses `params.get('diameter')`. A working
NumPy element list ported to `propagate_through_system_jax` has every aperture silently
skipped. v4.11.2 CHANGELOG claims this was fixed in v4.10 — **false**. Round-3 flagged it;
v4.11.2 didn't touch it.

**B1-2 `propagate_through_system_jax` is not actually JAX-end-to-end traceable.** Fallback
at `system.py:577-595` calls `np.asarray(E)` on a (potentially traced) JAX array →
`TracerArrayConversionError` under `jax.jit`/`jax.grad`. Any element list containing
`spherical_lens`, `aspheric_lens`, `real_lens`, `real_lens_traced`, `mirror`,
`cylindrical_lens`, `axicon`, `grin_lens`, `propagate_tilted`, `turbulence`, `zernike`, or
`gaussian_aperture` is non-traceable.

**B1-3 RS back-propagation kernel produces wrong physics.** `propagation.py:2828`. For
`z < 0`, amplitude flips but `exp(1j·k·r)` retains forward-propagating phase. No `z ≤ 0`
guard. ASM correctly back-propagates; RS does not. Round-3 CRIT, unfixed.

**B1-4 ASM-MFT band-limit boundary differs between backends.** `propagation.py:1890` (JAX
uses `<`) vs `:1917` (NumPy uses `≤`). Comment claims "fixed in 4.10 to use `<`" — but only
the JAX branch was edited. Same propagator, two backends, one-bin different output. "Fixed
in name only."

**B1-5 SAS asymmetric padding `as1 = (N+1)//2` only correct for `pad=2`.** For
`pad=4, N=512`: input centered at 512 but `N_new/2 = 1024` — off by 512 pixels. Global
linear-phase tilt in output. Docstring invites `pad > 2`.

**B1-6 Dispatcher routes negative `z` to forward-only propagators.** `dispatch.py:212-230`
uses `abs(z)` to pick regime, returns `'fraunhofer'`/`'sas'` for some negative-z calls, then
the kernel hard-raises. User gets a confusing back-trace from a function they didn't call by
name.

**B1-7 `propagate(return_result=True)` returns `field=None` for tuple-returning kernels.**
`dispatch.py:113-123`. For `method='fresnel'/'fraunhofer'/'sas'` the underlying propagator
returns `(E, dx_out, dy_out)`; `_coerce_field(tuple)` fails (silently caught), `field=None`,
`dx` is the INPUT dx not output. The documented `return_result=True` example is outright
broken for half the methods. NEW finding.

**B1-8 Dispatcher silently ignores `output_grid`/`output_dx` for ASM family.** GBD/HFPI/HF
paths forward them; ASM/Fresnel/Fraunhofer/RS/SAS drop them. User passes `output_dx=2e-6`
expecting MFT-style output sampling, gets plain ASM at input pitch. NEW finding.

**B1-9 `_apply_doe_kick_jax` blocks gradients.** `float(period_x)` strips the JAX trace;
`np.isfinite(period_x)` raises on a traced array. Any user differentiating w.r.t. grating
period gets silent zero gradient (or crash). Round-3 CRIT, unfixed.

**B1-10 Half-pixel grid drift across propagator families.** ASM/Fresnel/RS/sources use
`(arange(N) - N/2)·dx` (pixel-centred); GBD/HF/subaperture/MHS/`optimize/core.py:188` use
`(arange(N) - N/2 + 0.5)·dx` (cell-centred). Silent half-pixel shift produces wrong-physics
phase error proportional to `k₀·dx/2·(off-axis distance)`, larger for high NA / large field
angles. Round-3 flagged; v4.11.2 didn't reconcile.

**B1-11 `makedammann2d` mutates global `np.random` state.** `elements/doe.py:608` calls
`np.random.seed(seed)`. Library function mutating user's global RNG is a high-severity
anti-pattern. NEW finding.

### Tier 2 — Silently wrong physics in non-default paths (HIGH but narrower scope)

**B2-1 `ghost.py` intensity formula ignores transmission losses.** `R_i · R_j` reported for
ghost intensity, omitting `Π(1−R_k)` over transmitted-through surfaces. ~3× over-estimate
for 10-surface systems. Round-3 HIGH, untouched.

**B2-2 `ghost.py` `focus_z_estimate` is dimensionally arbitrary.** Harmonic mean of
`|R_i|, |R_j|`; reported as a focus position, used by stray-light budgets. Round-3 HIGH,
untouched.

**B2-3 `image_plane_wfe` reference-sphere radius missing `1/N_chief` factor.** For off-axis
chief with `N_chief < 1`, sphere doesn't pass through the chief; quadratic shape error
absorbed as phantom defocus by `best_rms`. Round-3 HIGH, partially fixed (chief-image
landing got `1/N_chief` but the sphere radius itself didn't).

**B2-4 `distortion_grid` constructs unphysical N=0 rays.** No `L²+M² ≤ 1` guard. For
`tx = ty = 45°`, `N = 0`; the image-plane transfer `t = z/N` blows up. Trace failure
swallowed by `except: pass`. Round-3 CRIT C-AB-2, unfixed.

**B2-5 `apply_real_lens_traced` no explicit mirror guard.** Relies on inner `apply_real_lens`
to raise — error message names the wrong function, and the ray-traced OPL leg doesn't fire
the guard. NEW finding from cross-cutting agent.

**B2-6 `gerchberg_saxton(backend='jax')` dispatch silently drops `seed`/`dtype`/
`initial_phase`.** Function-level kwargs are wired through `gerchberg_saxton_jax` correctly,
but `phase_retrieval.py:127-128` dispatches without forwarding the kwargs. One-line edit.

---

## What v4.11.2 successfully closed (~70 findings)

Mirror Seidel parity (Cassegrain, hand-derived S4=-16 pinned to 1e-9). `system_abcd` ↔
`seidel_coefficients` mirror sign agreement. `seidel_wfe` `(1/4)·S₃·ρ²` DC term.
`bundles.py` rewrite. EP-aiming siblings (`ray_fan_data`, `opd_fan_data`,
`field_aberration_sweep`, `relative_illumination`). `RAY_MISSED_SURFACE` stamping. Airy
radius with `|f_eff|`. JAX-trace round-3 closures (sign(R) param twin, double-where in
intersect twins, `~isfinite(t)` masking, `_refract_jax` alive-clearing). EVENASPH PARM
indexing for Zemax round-trip with non-trivial α₄ at rtol=1e-6. Quadoa aspheric serializer
writes coefficient values not keys. `normalize_prescription` mirror filter. Zemax STOP marker
on folded designs. Mirror/coord-break DISZ round-trip. Richards–Wolf `1/f²` + `exp(+ikf)`
(test pins peak ratio = f² to ±20%; on-axis phase advance to 0.05 rad). Coating `'avg'` mode
per-pol admittances. `apply_waveplate` docstring sync. `apply_real_lens_traced` /
`apply_real_lens_maslov` `stop_index` warns. `_lens_thin.py` clean. C-LR-1 SIGN REVERT with
real ground-truth pin against `apply_real_lens_traced` (the round-3 meta-finding). GBD
`axial_opl` `getattr` fix with `RuntimeWarning` on failure. S-LAH64/79 wrong coefficients
removed. `propagate_hfpi_through_prescription` finite-conjugate dead-path. `init_paths_
stratified` cartesian product. HFPI Kirchhoff `1/(iλ)·dΩ` (scalar paths). HFPI per-aperture
child seeds. `propagate_huygens_fresnel_with_opl_callable` `-1j` prefactor.
`propagate_huygens_fresnel_through_prescription` honouring E_in via `decompose_lg`. HF
Chebyshev real-`E_in` dtype-promote. Subaperture per-patch fit centring. Asymptotic Maslov
shared helper. Phase retrieval `seed=`/`dtype=` plumbing (function-level). `find_best_focus`
NaN guard. `monte_carlo_tolerancing_linearized` `a_k ≥ 0` clamp (Maréchal). `compute_psf`
non-square pupil error. `t_strehl_perfect` strengthened to `abs(peak - 1) < 1e-9`.
`apply_detector` non-integer scale correction. `Source.*` `**factory_kwargs`. AO rim FD all
four quadrants. `polychromatic_strehl`/`_psf` dtype hygiene. `petzval_radius` docstring sign.
`image_plane_wfe` chief from alive. `ChromaticFocalShiftMerit` self-contained.

---

## Test-suite quality

**243 unit tests + 314 validation tests pass cleanly in ~13 s.** ~55 new pinning tests across
6 v4.11.2 test files, most STRONG ground-truth pins. The Cassegrain Seidel test does
hand-computed S4 = -16 pinned to 1e-9 — exactly the kind of independent analytic ground truth
that prior rounds lacked. The C-LR-1 ground-truth pin against `apply_real_lens_traced` is
particularly impressive.

**2 weak tests** (down from round-3's 3-of-9):
- `test_real_E_in_yields_complex_out_via_dtype_introspection` uses `inspect.getsource` to
  scan for source-string match — pins tokens, not behavior. Acknowledged in test as a
  fallback.
- `test_axial_opl_path_does_not_emit_failure_warning` checks warning absence, not actual
  `axial_opl != 0`. Future silent-zero-init regression would slip past.

**14 coverage gaps** — fixes claimed in CHANGELOG without pinning tests:
1. `compute_psf` non-square pupil error
2. `apply_detector` non-integer pixel ratio
3. `find_best_focus` NaN injection
4. `monte_carlo_tolerancing_linearized` `a_k ≥ 0` (no negative-a_k input test)
5. `load_material` RuntimeWarning on dispersion drop
6. `Source.*` `**factory_kwargs` propagation
7. `apply_real_lens_traced` M_x/M_y transpose
8. NaN sentinel mask in `apply_real_lens`
9. `stop_index != 0` warn in `_traced` / `_maslov`
10. Freeform-terms `RuntimeWarning` in thin-element
11. Zemax coord-break-only STOP marker (only mirror tested)
12. **JAX↔NumPy phase-retrieval cross-parity** (only same-backend reproducibility tested)
13. Cassegrain S1/S2/S3/S5 hand-derivation (only S4 hand-pinned)
14. Richards-Wolf vs paraxial Airy at low NA

**Validation harness scoped filter is well-reasoned.** Only `DeprecationWarning`,
`PendingDeprecationWarning`, `ResourceWarning`, `ImportWarning` suppressed.
`RuntimeWarning`/`UserWarning`/`FutureWarning`/numerical overflow now propagate. 3 of 4
bare-`except: return True, 'skipped'` patterns in validation files fixed; one residual at
`validation/io/test_io.py:196`.

---

## Process-level findings (audit-of-the-audit)

**One round-3 finding was a FALSE POSITIVE:** the M_x/M_y axis-swap bug claimed in
`_lens_jax.py:476-479`. The JAX twin has had the correct stencil since v4.7.0; only the
NumPy `_lens_traced.py:1789-1792` had the transposed indexing. v4.11.2 fixed NumPy
correctly; no JAX fix was needed. This is the first known false positive in the audit
series — useful calibration on round-3 agent accuracy.

**Recurring patterns:**
1. **Fixes that don't reach sibling code paths.** When bug A is fixed in function X, the
   same bug in sibling Y is often missed (5+ examples across rounds 2-4).
2. **Fixes claimed in CHANGELOG but never landed in code.** v4.11.2 has at least 3 (JAX
   aperture schema, ASM-MFT NumPy branch, dispatcher overrides).
3. **`__del__`/bare-except patterns hide failures.** `_dtype_restore_guard.__del__` and
   `_quadoa_deserialize_aspheric` legacy-list fallback both silently produce wrong answers.
4. **Test infrastructure improvements pay off.** v4.11.2's ground-truth pins (Cassegrain S4,
   C-LR-1 against `apply_real_lens_traced`, RS-vs-ASM phase agreement) caught real
   regressions that the round-3 "field is finite" pattern would have missed.

---

## Recommended path forward

### Option A — Hold release, patch and re-test (1-2 days)

**Tier 0 (mandatory, ~1 hour):**
1. Fix README cookbook examples (sweep for positional `apply_real_lens` and renamed
   functions).
2. Wire `_deprecation.py` for `load_zmx_prescription` → `load_zemax_zmx`,
   `load_zemax_prescription_txt` → `load_zemax_prescription_data_txt`.
3. Fix `gerchberg_saxton(backend='jax')` dispatch to forward `seed`/`dtype`/`initial_phase`.

**Tier 1 (high-leverage physics, ~1 day):**
4. JAX aperture schema unification (`radius` → `diameter`).
5. RS back-prop `z ≤ 0` guard (raise) — match Fresnel/Fraunhofer/SAS.
6. SAS pad>2 centring (`as1 = (N_new - N) // 2`).
7. Dispatcher negative-z routing (restrict regime check to ASM/RS).
8. `propagate(return_result=True)` tuple-unpacking.
9. ASM-MFT NumPy `≤` → `<` to match JAX.
10. `makedammann2d` use `default_rng(seed)`, not global seed.

**Tier 2 (documentation / known-limitations callouts in README+CHANGELOG):**
- `propagate_through_system_jax` non-traceable element list — document and raise
  `NotImplementedError` at trace-build time on unsupported element types.
- Half-pixel grid drift between propagator families — document or fix by sweeping
  `+0.5`-offset usages.
- `_apply_doe_kick_jax` differentiability gap — document.
- `ghost.py` magnitudes are upper bounds (no transmission losses) — document.

### Option B — Ship v4.11.2 now, hot-patch in v4.11.3 within a week

Acceptable IF:
- Tier 0 items (README, deprecation shims, gerchberg_saxton dispatch) are fixed first via a
  v4.11.2.post1 or v4.11.3 within a day.
- A "Known limitations" section is added to CHANGELOG listing every Tier 1 / Tier 2 item.
- A CI workflow is added to run README code blocks (so future regressions don't reach PyPI
  again).

### Option C — Ship without further changes (NOT RECOMMENDED)

The fundamental physics is solid, but the first-impression failure (README cookbook) plus the
silently-wrong JAX backend (aperture skipped, gradient drop, traceability) will generate
support requests within the first week. The dispatcher / RS / SAS issues will surface for any
user doing back-propagation or non-trivial output-pitch sampling.

---

## Summary numbers by audit round

| Round | Total findings | CRIT | HIGH | MED | LOW | Fixed in subsequent release |
|---|---|---|---|---|---|---|
| 1 (external, 8 agents) | ~100 | ~22 | ~35 | ~30 | ~13 | ~95 across v4.10–v4.11.2 |
| 2 (verification) | 6 dead-on-arrival + 5 new + 3 overclaims | — | — | — | — | all in v4.11.1 |
| 3 (11 agents) | ~120 | ~25 | ~50 | ~30 | ~15 | ~70 in v4.11.2 |
| **4 (13 agents, this round)** | **~75** | **~15 release-blockers** | **~25** | **~20** | **~15** | TBD in v4.11.3 |

The fix rate has been monotonically improving: round 2 found 6 dead-on-arrival fixes from
v4.10; round 3 found 5 fewer; round 4 found 0 dead-on-arrival in v4.11.2's claimed fixes
(though 3+ items claimed-fixed weren't actually touched). The audit-of-the-audit process is
working — but the Tier-0 items (README, deprecation, dispatcher) need to be closed before
v4.11.2 goes to PyPI to avoid a brand-damaging first impression.

**Bottom line: HOLD release, complete Tier 0 + Tier 1 patches, then ship.** The physics
correctness work in v4.11.2 is genuinely strong; don't undermine it with an unworkable
README and a JAX backend that silently produces wrong answers.

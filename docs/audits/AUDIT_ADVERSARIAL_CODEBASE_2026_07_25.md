# Adversarial codebase audit — physics, bugs, conventions, dead code (2026-07-25)

**Scope:** the full library (171k lines) at `main @ 77ccdc6` (v5.29.0 + lint-clean, CI
fully green), audited by five parallel Opus subagents with disjoint territories,
audit-only (no code changed).  House evidence rules: every finding tagged
**MEASURED** has a runnable repro with numbers (scratch dirs `audit_p/e/m/r/a`);
**INSPECTION** findings are explicitly unverified code-reading claims.  Verified-clean
maps and honest coverage statements are part of the record — what was checked and
cleared matters as much as what was found.  Severity: **CRITICAL-physics** (wrong
answers from public API), **HIGH-bug**, **MEDIUM-convention/bug**, **LOW-hygiene**.

**Status: TIER 1 FIXED (2026-07-25).**  All six CRITICALs plus the two
silent-data-corruption HIGHs are fixed and pinned: P1+P2 `e29a8db`, R-1+R-2
`1fc8b1f` (also fixing two further measured defects found during the fix: R-1b
`ep_z` distance-vs-signed-coordinate, R-1c post-stop leg walked in air), A-1+A-2
`720f689`, M1 `a240c15`, E-C1+E-C2 `2be264c` (docstring-only, AST-verified).  236
new pins, each set verified to fail on pre-fix code; CHANGELOG [Unreleased] has
the summary.  Two corrections to this report discovered during fixing: (a) R-1's
"exact ep_z −16.9860 mm" is the object DISTANCE — the signed coordinate is
**+16.9860 mm** (the audit oracle printed the + value; the − transcription was
itself the R-1b convention slip); (b) M1's repro geometry is centro-symmetric, so
the default `symmetry='auto'` fold MASKS the collision there — it reproduces on
the default path with fold-ineligible centres (pinned that way).

**Status: TIER 2 BATCH B FIXED (2026-07-25)** — the infra/contract cluster
`P3, P4, P6, P7, E-H6, E-H11, A-3`, 55 pins in
`tests/unit/test_niche_audit_p2b_infra_contracts.py` (40 verified failing
pre-fix), with 22/22 captured arrays proven bit-identical across the pre/post
trees for everything the fixes claim not to change; CHANGELOG [Unreleased] has
the numbers.  Two corrections to this report found while fixing: (a) **E-H6 is
worse than "parameter never read"** — the fixed n_H=2.3/n_L=1.38 stack
satisfies the quarter-wave admittance match only at n_substrate = 2.778, so
measured against this module's own TMM it is *worse than bare glass* over the
whole common range (R = 0.0856 vs 0.0426 uncoated on N-BK7); it is an AR
function that doubled the reflectance, not merely a design frozen to one
substrate.  (b) **P6's "the NumPy twin validates and honours it" is only half
true** — the twin honours `'fresnel'`/`'sas'` and rejects `'rs'`, but junk
method names fall through to ASM there too (measured 0.0 relative difference
from `method='asm'`); that sibling gap is NOT closed by this batch (see the
declined list in the fix report).

**Status: TIER 2 BATCH A FIXED (2026-07-25)** — the remaining HIGH territories:
R-3/R-4/R-5/R-6/R-7 `bcb568d`, E-H1/E-H5 `03fa01e`, E-H2/E-H3/E-H4/E-H7 +
E-M2..M5 `0a443d6`, E-H8/E-H9/E-H10 + E-L12..L17 + E-M13 `5f9d82b` (incl. the
CONVENTIONS.md IEEE relabel), M2/M3/M4/M5/M6/M9 `4602b7f`.  238 pins, each set
verified failing pre-fix; the 121 acceptance battery gated every traced-engine
edit.  Corrections to this report found while fixing: (a) **M3's suggested
one-liner is insufficient AND mis-sited** — `_require_propagating_incidence`
lives in `rcwa/_core.py:563`, `pmm/_core.py:786` is `_assemble_jones_farfield`,
and the `kz_inc` comparison change alone misses NaN substrates and
non-propagating metallic superstrates (three guard layers were needed); (b)
M4's single "A = 0.208" figure was not reproduced — the same instability
manifests as hard `_EnergyError`s at 3 of 5 truncations and survivors wandering
A = 0.080-0.113 vs converged li 0.0787 (configuration-dependent, hence a scope
gate rather than a numeric fix).

**Every CRITICAL and HIGH finding in this audit is now fixed and pinned.**
Still open: the MEDIUM/LOW inventory (incl. P5's dispatch return-shape design
question — an API-contract decision for the owner), the flagged world.py
coord-break transpose (needs a pure-tilt oracle), mirror-parity signing of
stop-adjacent pupil legs (needs a fold-pupil oracle), and four NEW measured
findings recorded during the fix waves: `aberration_tensor` output_mode
degeneracy (LG non-piston channels indistinguishable from piston); NumPy
`propagate_through_system` junk-method fall-through (P6's sibling); uniform
cell + fff_nv `AssertionError`; `orientation=NaN`/`retardance=NaN` unguarded
in polarization.  Fixes should follow the same measure-first / gate / pin
discipline.

**Status: WAVE 3 COMPLETE (2026-07-26) — THE AUDIT IS FULLY IMPLEMENTED.**
Every finding in this report is now fixed, refuted-with-evidence, or
explicitly deferred-with-record: infra `d045980`, UI/deprecation `54a2dcf`,
propagators `3f22778`, oracle wave `1523d8e`+`86cadbe`, raytrace/sources
`e843f6f`, RCWA/PMM `3ead8cb`, elements `bba1bc4` (+ CI cross-platform pin
fix `0f63efe`).  ~470 wave-3 pins, every set verified failing pre-fix.
Audit-correction ledger from wave 3: **REFUTED** — R-18's
`_invalidate_glass_name`/`_GLASS_CACHE`/`_POLYNOMIAL_STUB_NAMES`/
`TraceResult.rays_at` dead-code claims (live caller; reflection-based
walker dependency; documented invariant; documented API), E-L18
`coronagraph.py` unreferenced (public namespace), `FirstOrderData.summary`
and `DifferentialTransfer`/`ParetoResult` dead-claims (public API).
**CORRECTED** — P10's offset formula (half the surplus coverage, not
w·overlap/2 in general); P11's replica period (kernel-dependent); M3's
mask drift attribution (PMM-1-D vs everything, not PMM vs RCWA).
**DECLINED WITH MEASUREMENT** — R-14 dead-row clip parity, R-15 fan-axis
convention (both contract-locked), E-H5's power/peak ROI normalization
(unevaluable by construction, loud warning instead), P5 return-type
unification (deferred to roadmap Part F/F1, owner decision).
**DEFERRED WITH RECORD** — input_kind 67-site rollout; A-4's three h5
sibling writers; the immersed-conjugate pupil 1/n finding (W3-2, needs
its own oracle); the 1-D-RCWAStack y-averaging trap; `aberration_tensor`
default w_o dimensional question + JAX-twin scale split; the GBD paraxial
z_image default warning.

**Status: W3 ORACLE WAVE (2026-07-25)** — the two oracle-needing physics
questions above are RESOLVED, both by building the oracle first:
**(W3-1)** the flagged `world.py` coord-break rotation is a CONFIRMED
disagreement and **`world.py` is the correct side** — Zemax defines
`Tilt About X/Y/Z` by the LOCAL-TO-WORLD matrix (right-hand `R_math(+θ)`,
intrinsic X→Y→Z for `PARM 6 = 0`, `r_global = R·r_local + offset`;
OpticStudio KB KA-01638), so `intersection._apply_coord_break` (the 3.7.1
flip), its op-for-op twin `differential._adrt_coordbreak`, and
`ui.model.recompute_element_frames` (3.7.4) were the three inverted sites and
were fixed to the transpose convention.  A pure-tilt oracle was required
because a balanced mirror fold is sign-degenerate in the local frame — the
reason the 408b8c3 revert mistook RT-4 for a phantom (that revert's argument
is retracted in `AUDIT_RAYTRACE_CORE_2026_07_08.md` §RT-4).  Measured
pre-fix: 8.243032° of angular disagreement on a single `tilt_x = +12°` break.
Also closed the `world_trace.py:169` DOE zero/non-finite period guard (R-13's
sibling: `ZeroDivisionError` / silent NaN-alive ray pre-fix).  See the
Territory R "Flagged, not claimed" entry for the full numbers.

---

## Executive summary

**6 CRITICAL-physics, 26 HIGH, ~40 MEDIUM, ~50 LOW findings; extensive verified-clean
map.**  The core numerics the v5.29 campaign hardened (traced chain, sag/trace/glass
agreement, propagator energy/cross-agreement at even N, polarization algebra, RCWA/PMM
vs analytic oracles) measure clean, several at machine precision.  The defects live at
the edges the campaign never touched: odd-N spectral grids, non-front-stop pupil
computation, documentation asserting physics the code doesn't implement, silent
acceptance of junk inputs, parameters that are never read, and objects that freeze
stale state.  None of the six CRITICALs is caught by CI — in every case the existing
pins are invariant to the bug (isfinite-only, dtype-only, or spectral quantities).

### Master table — CRITICAL

| # | ID | where | one-line |
|---|---|---|---|
| 1 | P1 | all ASM-family propagators | Odd-N grids: DC sits at −0.5 bin → silently shifted, phase-wrong fields (26% err, −3.9 px at N=257); reachable via Shack-Hartmann odd subapertures; RS and Fresnel-MFT immune (discriminators). |
| 2 | R-1 | `raytrace/seidel.py:669` | `compute_pupils` drops the last pre-stop transfer → wrong EP position/radius/f-number for every non-front-stop system; feeds chief/marginal ray aiming in `analysis/field.py`. |
| 3 | R-2 | `raytrace/seidel_analysis.py:353` | `seidel_wfe` returns −W (sign-composition across modules); all sign-carrying uses inverted. |
| 4 | A-1 | `analysis/psf_mtf_otf.py:1275,981` | Default radial FWHM/Rayleigh resolution biased −8% to −21% by integer-pixel binning; the unbiased subpixel profile exists in the same module. |
| 5 | E-C1 | `_lens_thin.py:553` | Docstring's SA-nulling conic (`k1=−n²` on the curved FIRST surface) is on the wrong surface: following it is 2.6× worse than a plain sphere; flat-first k2=−n² is the exact null. |
| 6 | E-C2 | `_lens_thin.py:360,508` | "Exact OPD incl. all higher-order aberrations" claim on an orientation-blind thin screen that provably never reads the thickness (up to 21.7 waves PV error at f/2). |

### Master table — HIGH (grouped)

| IDs | theme |
|---|---|
| P2, E-H4, M9 | Ownership bugs: live FFT buffer returned (corrupts already-returned fields, live via carrier.py); prepared-lens prescription held by reference (silent stale-OPL hybrid); writable cache returns. |
| E-H3/H4, A-3 | Frozen-defaults bugs: `prepare_real_lens*` and top-level `DEFAULT_*` constants snapshot settings that setters silently don't move. |
| E-H8/H9/H10, M2, M3 | Silent-wrong-input physics: polarization handedness typo → LEFT circular; ER<1 swaps PBS ports; wrong Jones layout accepted; PMM segment widths unvalidated (clip, not error); classical PMMStack accepts gain/NaN media. |
| R-5, R-6, M4, M5 | Optimization/solver traps: LG merit minimizes Strehl to 0; vignetting NaN zero-filled into Zernike fits (sign-flipped descent); fff_nv wrong on metal squares inside documented scope; fff_nv+stabilize always hard-raises lossless. |
| R-3, R-4, P3, P4 | Guard gaps: grazing-ray phantoms in 2 of 4 trace paths (negative OPL; immortal ray); Van-Vleck FD step all-roundoff (9% amplitude); Richards-Wolf pupil silently clipped to a square at array NA (5.5× PSF error). |
| E-H1, M1 | Lattice/cache correctness: Maslov upsampler is the exact sibling of the fixed `ii·Ns/N` bug (+3.7 px at sub=8); RCWA shapes-layer eigenmode cache collides distinct layers (32% silent, energy-conserving error). |
| A-2, E-H2, E-H5, E-H6, E-H7, E-H11, R-7 | Contract violations: EE radius drifts +6% with zero-padding; `newton_max_iters` inert on the pool path; `roi=` discards normalization (8 orders); `n_substrate` never read; `local_only` does the opposite of its docs; Dammann default silently ×1e-6 SI inputs; analytic Jacobian raises on degenerate bundles. |

Fix-priority recommendation: (1) the six CRITICALs + P2 (data corruption) + M1
(silent 32%); (2) the silent-wrong-input cluster (cheap validation, high yield);
(3) frozen-defaults/ownership; (4) guard gaps; (5) conventions/docs/dead code as a
batch hygiene pass.  Every CRITICAL/HIGH needs a pin that would have caught it —
the audit's repro scripts are the seeds.

---

## Territory R — raytrace / algebra / sources / glass / _math / optimize

### Findings

| ID | site | severity | claim (evidence) |
|---|---|---|---|
| R-1 | `raytrace/seidel.py:669` `compute_pupils` | **CRITICAL-physics** | Drops the last pre-stop→stop transfer (`system_abcd(surfaces[:stop])` never applies the final element's thickness; `seidel_coefficients:900` explicitly prepends `T_last` for the same subsystem).  EP position/radius and f/# wrong for EVERY non-front-stop system: measured ep_z −3.41 vs exact −16.99 mm, ep_radius −21%, f/# 11.88 vs 9.38 (GAP=10 mm); error exactly 0 at GAP=0 (mechanism discriminator); sweep to −84% ep_radius at GAP=40 mm.  Consequence amplifier: `analysis/field.py` aims chief/marginal rays from these values, so every field-analysis trace of a non-front-stop system launches through the wrong pupil.  No numeric pin exists (`isfinite`-only), hence green CI.  (MEASURED) |
| R-2 | `raytrace/seidel_analysis.py:353` `seidel_wfe` | **CRITICAL-physics** | Returns the NEGATIVE of the physical wavefront: composes the `(1/8)S1ρ⁴+…` expansion with S-values documented as `code = −S_Welford`.  Exact-trace oracle on 3 designs: ratio −0.9975…−0.9998 over ρ∈[0.3,1].  Magnitude consumers unaffected; sign-carrying uses (adding to a pupil, Zernike fits, coma/distortion asymmetry) inverted.  Public leaf, no internal consumers; tests feed hand-written dicts so the composition is never exercised.  (MEASURED) |
| R-3 | `raytrace/jax_trace.py:266` | HIGH-bug | Flat-with-aspherics branch lacks the grazing-miss guard its pure-flat branch and param twin both have; under default float32 a grazing ray survives, refracts backwards, and returns NEGATIVE OPL (−8.05e-3 m measured).  Masked under x64.  (MEASURED) |
| R-4 | `raytrace/intersection.py:140` | HIGH-bug | NumPy flat fast path gives a grazing ray t=0, alive, RAY_OK → immortal phantom (opd 0.0 through a 4-flat stack, counted in centroid/summary as alive).  Reachable via the P3-58 DOE-order case (L²+M²==1, N=0 kept alive by design).  JAX twins kill it.  (MEASURED) |
| R-5 | `optimize/merit_terms.py:1160` `LGAberrationMerit` | HIGH-bug | Sums |L(p,ℓ)|² over targets INCLUDING (0,0) ("piston/Strehl") — minimising drives Strehl→0.  The JAX twin carries the documented OPT-1 fix (`1−|res|²`); never applied here.  (INSPECTION — the fix text is verbatim in the sibling) |
| R-6 | `optimize/merit_terms.py:327,897,950` | HIGH-bug | All three OPD merits zero-fill non-finite OPD before `zernike_decompose` (which handles NaN itself): an in-pupil vignetted annulus (ρ>0.8 NaN) changes the fitted spherical coefficient from +0.100 to −0.017 waves (SIGN FLIP), defocus 0.300→0.126 — wrong magnitude AND wrong descent direction whenever any pupil pixel is vignetted.  (MEASURED) |
| R-7 | `raytrace/differential.py` `ray_transfer_jacobian_analytic` | HIGH-bug | Public twin raises bare ZeroDivisionError on a degenerate bundle its documented sibling handles.  (MEASURED) |
| R-8 | `surface.py:536` / `_conic_core.py:211` / `trace.py:383` | MEDIUM | Odd aspheric powers sag/normal-inconsistent, differently per backend; the guard lives only in `validate_prescription`, not on the public `Surface` dataclass → silent wrong trace.  (MEASURED) |
| R-9 | `sources/core.py:537,700` | MEDIUM | `create_hermite_gauss`/`laguerre_gauss` take `w0` positionally in the slot every other factory uses for `wavelength`; the swapped call is silently accepted.  (MEASURED) |
| R-10 | `sources/core.py:1112 vs 2510` | MEDIUM | Radius-vs-diameter kwarg sprawl between annular siblings.  (INSPECTION) |
| R-11 | `raytrace/paraxial.py:187` | MEDIUM | `f_number` returns SIGNED f/# (−3.85 measured) while all three siblings use abs(); different pupil definition too.  (MEASURED) |
| R-12 | `glass.py:649` | MEDIUM | `_sellmeier_index` scalar-only; array input dies with numpy's opaque ambiguity error; sibling docstring claims parity.  (MEASURED) |
| R-13 | `trace.py:179` / `world_trace.py:169` | MEDIUM | NumPy DOE kick has no zero/non-finite period guard (ZeroDivisionError); JAX twin returns zero kick.  (MEASURED) |
| R-14—R-16 | various | LOW | Aperture-clip order differs numpy-vs-jax on dead rows; `make_fan(axis='x')` puts field angle in L where every sibling uses M; numba/numpy `_dsqrtq` NaN-radicand divergence.  |
| R-17 | `optimize/driver.py:460` etc. | LOW | `wave_traced=`, `use_traced_lens=`, `focus_search=`, `match=` — documented public flags with live branches and ZERO callers anywhere (4 penalty helpers consequently unexercised).  (grep-verified) |
| R-18 | (dead-code list below) | LOW | Dead symbols/fields/aliases; overdue deprecation shims (removal "v5.0" still shipping at v5.29).  |

### Flagged, not claimed
- `world.py`/`world_trace.py` coordinate-break rotation composes the TRANSPOSE of the
  rotation used (exactly, and in agreement) by `intersection.py:569` and
  `differential.py:356`; a prior audit's fix was reverted as a phantom on a
  mirror-fold oracle — which cannot constrain a pure tilt sign.  Needs a dedicated
  pure-`tilt_x` oracle.  (INSPECTION)
  → **RESOLVED W3-1 (2026-07-25): CONFIRMED defect, and `world.py` is the CORRECT
  side.**  Pure-tilt oracle (one `tilt_x` coord break in front of a flat
  air→N-BK7 interface, axial ray, ground truth = exact vector Snell in the world
  basis against the normal `Q·ẑ`): at `tilt_x = +12°` `trace()` deviated the ray
  **+4.121516° toward world +y** and `trace_world()` **−4.121516° toward world
  −y** — 8.243032° apart, local `M = ∓0.137072578`, max|Δ| = 2.741452e-01 (2.301e-02
  at 1°, 6.593e-01 at 30°); each was exact for its OWN frame convention (≤1.2e-16),
  so neither had a Snell bug.  A coordinate break is a passive frame change, so
  `T == Qᵀ` is linear algebra, not convention; Zemax fixes the sign by defining
  `Tilt About X/Y/Z` through the LOCAL-TO-WORLD matrix with right-hand
  `R_math(+θ)` in intrinsic X→Y→Z order for `PARM 6 = 0` (`r_global = R·r_local +
  offset`, OpticStudio KB KA-01638). So `world.py` (`Rx(+tx)@Ry(+ty)@Rz(+tz)` as
  `world_R`) is right and three sites were inverted and were fixed:
  `intersection._apply_coord_break` (the 3.7.1 flip),
  `differential._adrt_coordbreak` (its op-for-op twin) and
  `ui.model.recompute_element_frames` (3.7.4, `elem.R` — feeds both GUI layouts
  AND `world_trace_surfaces`, so the GUI had the inverted frame on both of its own
  trace paths).  Corrections to the flag's wording: for MULTI-axis tilts the two
  sites are **not** transposes but the same intrinsic X→Y→Z order with every angle
  negated (the transpose statement is exact only for a single axis), and the
  decenter halves already AGREED (measured max|Δpos| = 0.0 pre-fix), refuting
  RT-4's "the decenter half does not flip" claim.  Zero effect whenever every
  coord-break tilt is 0 (bit-identical control pinned); a balanced mirror fold is
  sign-degenerate in the local frame, which is exactly why the 408b8c3 revert
  looked like a phantom — that revert's own argument compared `Qᵀ·ẑ` (new LOCAL
  frame) with `world_R[:,2] = Q·ẑ` (WORLD) and is retracted in
  `AUDIT_RAYTRACE_CORE_2026_07_08.md` §RT-4.  20 pins in
  `tests/unit/test_niche_audit_w3_oracles.py` (15 verified failing pre-fix); the
  `test_v5_21_2_subsystem_audits.py` RT-4 pin, which asserted the category error,
  was rewritten to pin the passive-frame identity `Q @ local_dir == world_dir`.
- `world_trace.py:169` NumPy DOE kick had no zero/non-finite period guard (the
  R-13 sibling site).  MEASURED pre-fix: `period=0.0` → `ZeroDivisionError`
  mid-trace; `period=nan` → `(L, M, N, opd) = (nan, 0.0, nan, nan)` with
  `alive=True`, a silently NaN-poisoned LIVE ray.  Fixed to the JAX twin's
  contract (zero kick); `inf` and real periods bit-identical.  Pinned.

### Verified clean (highlights; method in the agent record)
Glass dispersion exact vs published catalog values (|Δn_d| ≤ 1e-6, Abbe ≤ 0.003; no
µm/m unit error); `system_abcd` EFL/BFL vs thick-lens analytic AND exact trace (≤5e-10);
the flat-fold parity fix present in all three paraxial implementations with no missed
sibling (JAX rejects mirrors loudly at both entry points); numpy↔jax↔jax-params trace
triangle at machine precision (≤4.4e-16 cosines) incl. degenerate probes; all 14
sag/derivative twin flavours consistent (≤1.6e-9) except the odd-power case (R-8);
every sources/core.py width convention correct (1/e² radius exact for 6 factories;
no waist/FWHM confusion — the campaign's earlier confusion was in an audit DOC, not
code); optimizer gradients vs FD (≤2.7e-10); Zernike orthonormal quadrature model
correct; grid anchors consistent (N/2 convention) across sources/zernike/merits;
`_math` chebyshev/levin exact with bit-identical JAX twins; `refract_snell` textbook
vector form, single-sourced, signed-t OPL telescoping verified end-to-end.

### Dead code (repo-wide grep-verified)
`glass.py` `_invalidate_glass_name`, `_GLASS_CACHE` alias, empty `_POLYNOMIAL_STUB_NAMES`
(makes an error arm statically unreachable); `TraceResult.rays_at` (public, documented,
zero callers); `FirstOrderData.summary` + 3 write-only fields; jax `surface_diffraction`
params + `jp_aux`; `DesignResult.scipy_result` write-only; `ParetoResult` fields;
sundry aliases/sentinels; exported-but-unused `DifferentialTransfer`, `ParetoResult`;
overdue shims (Source.gaussian/plane_wave legacy signatures, Schell return_kind,
LED positional shim); HG/LG factories accept a documented-"currently unused"
wavelength.

### Coverage
Numeric verification across glass, ABCD/Seidel/pupils, the trace triangle, sag twins,
sources conventions, _math, optimizer gradients/Zernike chain.  Inspection-only:
world coord-breaks, R-5, R-10, R-14.  Not covered: pymoo multi-objective, multiconfig,
wrapper-merit numerics, partial-coherence modal accuracy, chromatic merit paths.

---

## Territory A — analysis / io / core infra / backend / public API / ui (light)

### Findings

| ID | site | severity | claim (evidence) |
|---|---|---|---|
| A-1 | `analysis/psf_mtf_otf.py:1275,981` | **CRITICAL-physics** | `fwhm_resolution` and `rayleigh_resolution` default `axis='radial'` route through integer-pixel radial binning whose bias the same module's `_radial_profile_subpixel` exists to avoid: FWHM −7.98% at 4.9 samples/first-zero, −21.0% at 2.4 (Airy oracle); the subpixel sibling reads the same case at −0.4%.  `rayleigh_resolution` returns NaN + a warning blaming the PSF ("Gaussian-like… no true first zero") on a perfect Airy at coarse sampling.  No analytic-accuracy pin exists.  (MEASURED) |
| A-2 | `analysis/psf_mtf_otf.py:600` `encircled_energy_radius` | HIGH-bug | Hard-coded 256 radii spanning to the CORNER: the returned EE radius drifts +6.05% with zero-padding alone (N=64→2048 sweep, fixed physical beam); direct curve inversion is −0.8% at every N.  Docstring claims sub-percent; the pin tolerates 20% ("generous slop").  (MEASURED) |
| A-3 | `__init__.py:192-201` | HIGH-bug | `DEFAULT_COMPLEX_DTYPE/DY/REAL_DTYPE/WAVE_PROPAGATOR` are import-time snapshots; the setters move the live values (PEP-562-forwarded in propagation.py) but NOT the top-level constants — reading `la.DEFAULT_COMPLEX_DTYPE` after `set_default_complex_dtype('complex64')` returns complex128.  (MEASURED) |
| A-4 | `io/storage.py:424` | MEDIUM | `append_plane_h5(metadata=)` hands raw values to h5py attrs: raises on nested/heterogeneous/empty containers, silently DROPS None — while the module's own `_meta_dumps` round-trips all 19 probe types and is wired only to `write_sim_metadata`.  (MEASURED) |
| A-5 | `cache.py:200` `deep_nbytes` | MEDIUM | Counts a numpy VIEW at its slice size (8 B for a view of a 4 MB base) and double-counts repeated arrays — the byte budget can retain far more than the cap when views are cached.  (MEASURED) |
| A-6 | `memory.py:661` `estimate_asm_memory` | MEDIUM | est/measured first-call peak 0.53 (N=512), 0.96 (N=1024), 1.22 (N=2048) — neither a first-call bound nor steady-state.  (MEASURED, fresh-interpreter tracemalloc) |
| A-7 | `psf_mtf_otf.py:220,264` | MEDIUM | OTF/MTF docstrings claim `otf[0,0]` is DC=1; output is fftshifted (DC at `[N//2,N//2]`, `otf[0,0]`≈−1e-16).  (MEASURED) |
| A-8 | `user_library.py:893,903` | MEDIUM | Corrupted saved materials skipped by `except: pass` TWICE with no warning — user glass silently vanishes.  (INSPECTION) |
| A-9—A-14 | various | LOW | `input_kind` validation wired at 2/67 call sites (+stale TODO); dead `_jnp_or_none`; memory.py type-guard/round-trip-claim gaps; negative cost accepted; Zernike cache returns mutable cached array; io/ lacks `__all__` while analysis/ always has it.  |

### UI breadth pass (measured via importlib + signature-bind; PySide6 absent)
Six user-reachable dead actions, all swallowed by `except Exception` (92 empty-body
handlers across 22 files): all four whole-prescription propagator choices in
`waveoptics_dock.py:775` import names `propagators.propagation` no longer exports
(GBD/HFPI/Huygens/Subaperture menu items dead); `waveoptics_dock.py:994` imports a
nonexistent `..detector` (detector option silently no-ops, latent unpack bug behind
it); `coherence_dock.py:42`, `shack_hartmann_dock.py:128`, `lg_aberration_dock.py:104`
pass kwargs that no longer exist; `optimizer_dock.py:1088` `ToleranceAwareMerit(
inner_merit=…)` vs actual `sub_merit` aborts the optimizer run.  Plus
`ui/surface_table.py` (370 lines) entirely unreferenced.

### Deprecation registry rot
10 of 12 deprecations past their stated removal version (eight say v5.0; shipping
v5.29); the removed-in banner now emits "will be removed in v5.27" FROM v5.29;
`sources/core.py:2100` Schell shim can never fire from production (tests call it
directly — green CI, dead user path); `elements/doe.py:789` legacy values its own
comment advertises hard-raise before the deprecation branch.  Cross-territory:
`_lens_traced.py:1731`/`_lens_jax.py:310` document `lens_prescription`; the real
param is `prescription`.

### Verified clean (highlights)
MTF/OTF numerics vs analytic circular-aperture MTF (2.1e-3 max, MTF50 0.10%, axes
correct); Strehl trio consistent to 5-6 digits with correct amplitude weighting and
Maréchal convention; EE curve itself exact (+0.02% at the Airy dark ring);
`sparrow_resolution` 0.1-0.4%; the whole Zernike module (orthonormality, conditioning
to 95% obstruction, WLS equivalence, bit-identical `weighting=ones`); **the Zemax
.zmx importer**: thickness/glass off-by-one CORRECT, unit rows exact (MM/CM/M/IN…),
EVENASPH power law exact over 7 terms in two unit systems, all 14 unsupported surface
types warn loudly (nothing silently approximated beyond the known DGRATING family
behaviour), round-trip preserved to 2e-9 (CURV format), storage round-trip
bit-identical incl. NaN/±inf/−0.0/subnormals; `ByteBudgetedLRU` 8-thread stress =
zero drift, correct global LRU victim; export integrity 882/882 `__all__` entries
resolve; 0 mutable defaults / bare excepts in scoped core files; unit-vocabulary
audit 710 claims, 0 contradictions.

### Coverage
Deep: psf_mtf_otf, strehl, zernike, cache, _validation, memory API, zmx importer,
storage h5 paths, backend conversion.  NOT audited (real gaps): plotting.py (2345
lines), most of through_focus/field/ao/phase_retrieval/ghost/image_plane_wfe,
polychromatic, aberration, coronagraph, coherence, interferometry, io/codegen, the
Zemax .txt prescription-data importer, Zarr storage half, multi-process filelock
append.  One self-caught vacuous scan (shelled-out `rg` absent → false "0 dead")
was redone in pure Python.

---

## Territory E — elements/ (excl. rcwa/, pmm/)

### Findings (condensed; full repro scripts in scratch `audit_e/`)

| ID | site | severity | claim (evidence) |
|---|---|---|---|
| E-C1 | `_lens_thin.py:553` | **CRITICAL-physics** | Docstring prescribes `k1=−n²` on the CURVED first surface to null third-order SA; the −n² hyperboloid belongs on the EXIT surface of a flat-first lens.  Exact-trace: following the advice gives PV 10.38 waves vs 3.94 for a plain sphere (2.6× WORSE); flat-first k2=−n² is exactly stigmatic (0.00000); and the screen model this function computes has its null at −1−(n−1)², not −n².  (MEASURED) |
| E-C2 | `_lens_thin.py:360,508` | **CRITICAL-physics** | `apply_spherical/aspheric_lens` claim "exact OPD … all higher-order monochromatic aberrations": it is the orientation-blind paraxial screen and provably never reads the thickness `d` (d=1e-9 vs d=1.0 bit-identical).  Exact-vs-screen error up to 21.7 waves PV at f/2; the screen reads two orientations 1.8% apart where the truth differs 4.0×.  `apply_real_lens` documents the identical formula honestly.  (MEASURED) |
| E-H1 | `lenses_maslov.py:2102` | HIGH-bug | `output_subsample>1` upsamples with edge-anchored `scipy.zoom` against a stride-subsampled lattice — the EXACT sibling of the `ii·Ns/N` bug fixed at 0a743a6: measured centroid walk +0.50/+1.47/+3.71 fine px at sub=2/4/8 on a symmetric on-axis element (closed-form prediction matches); existing pins are invariant to it (spectral centroid, dtype).  (MEASURED) |
| E-H2 | `_lens_traced.py:247` | HIGH-bug | The ProcessPool Newton worker hardcodes 12 iterations — `newton_max_iters` is INERT on the pool path (≥200k pts + spline fit) and the pool never emits the unconverged warning whose own advice is "increase newton_max_iters"; bit-identical proof at N=512 for 1-vs-12 iters.  Mitigated: shipped default fit is polynomial (serial).  (MEASURED) |
| E-H3/H4 | `_lens_real.py:3505`, `_lens_traced.py:5294` | HIGH-bug | `prepare_real_lens*` freeze import-time/unresolved defaults (flipping `set_default_wave_propagator` afterwards desynchronizes prepared vs direct by 49.6-53.3) and hold `prescription` BY REFERENCE (in-place mutation yields a silent stale-OPL × new-amplitude hybrid — measured 0.71 from a correct rebuild) — in exactly the optimizer loops the class advertises.  (MEASURED) |
| E-H5 | `lenses_maslov.py:1839` | HIGH-bug | `roi=` silently discards `normalize_output` (returned patch matches the 'none' scale, ~8 orders below the power-normalized field, no warning).  (MEASURED) |
| E-H6 | `coatings.py:441` | HIGH-bug | `broadband_ar_v_coat(n_substrate, …)` never reads `n_substrate` (identical output for 1.45→4.00).  (MEASURED) |
| E-H7 | `_lens_thin.py:141` | HIGH-bug | `lens_model='local_only'` does the OPPOSITE of its docstring (steers sub-beam onto axis; bit-identical to `paraxial(xc=0)` up to piston) — an inverted-doc dead duplicate, 0 callers.  (MEASURED) |
| E-H8-H10 | `polarization.py:999,888,1149` | HIGH-bug | Handedness parsing: anything not starting with 'r' (incl. 'cw', 'clockwise', typos) silently returns LEFT circular; PBS `extinction_ratio<1` silently SWAPS ports (power conserved, nothing flags); `jones_pupil_to_stokes_unpolarized` accepts the module's own "canonical" (2,2,Ny,Nx) layout and returns wrong-shape wrong-value Stokes.  (MEASURED) |
| E-H11 | `doe.py:518` | HIGH-bug | `makedammann2d(_legacy_units='auto')` default silently ×1e-6 any period/wavelength > 1e-3 — an SI THz design gets 5e-10 m cells; only a suppressed DeprecationWarning fires; the shim's own removal version is 5.0 (shipping 5.29).  (MEASURED) |
| E-M1-M15 | various | MEDIUM | Highlights: `on_noncollimated='delegate'` forwards RESOLVED `sag_chunk_rows` (re-enabling banding against an explicit opt-out) and silently discards 8 physics kwargs at the model swap; `on_noncollimated`/`inversion_method` accept any junk value as 'warn'/Newton; ray_density loses 2.1% power at sub=8 with no diagnostic; periodic phase-mask lattice closed with `clip` instead of `%` (spurious orders); spherical-lens out-of-domain clamp finite while every sibling NaNs; S3 sign labelled Born-Wolf but implements IEEE (doc-only, self-consistent); prescription `radius` key documents no sign convention; docstrings name a parameter removed in 4.7.  |
| E-L1-L22 | various | LOW | Highlights: worker-pool atexit leak + broken double-checked locking + overbroad pool-fallback catch; dead numexpr scaffold + dead constants with misleading comments; odd aspheric powers silently evaluate as next-lower even (111× error); DOP absolute floor reports fully-polarized weak fields as 0.0; grazing `kz→1.0` substitution (7e12×); `fold_split` drops the requested observation plane; 10 dead args across Maslov integrators (NumPy/CuPy twin drift); `coronagraph.py` unreferenced.  |

### Verified clean (highlights)
Cross-implementation sag agreement exact (≤6.5e-18) across 3 independent implementations
and 7 surface flavours; every element sign convention mutually consistent and as
documented; Maslov KMAH signature + SPA prefactor complete (no missing λ), NumPy/CuPy
twins identical; DOE efficiencies analytic to 6 digits (kinoform sinc², FZP 1/π² and
4/π², exact binary complement); microlens Voronoi/no-steer exact; energy conservation
0.999996-1.0; the ENTIRE polarization element algebra textbook-exact (Malus 1e-16,
unitarity, realizability over 400 random Stokes, cross-family retardance agreement
9e-15); all chunked/banded/parallel variants inherit every accuracy option of their
whole-grid paths (the one divergence is E-H2) and the S11-era row-band fix is
confirmed complete; cache keys complete, no lru_cache in scope; every resample anchor
in 20 files on the repo-standard lattice; zero mutable defaults / bare excepts /
constant-if / dirty-global exception paths (AST sweeps); every live feature flag
exercised both ways by tests.

### Coverage
Deep: _lens_thin, polarization, doe creates, lenses sag helpers, _lens_traced
(Newton/pool/prepared/validation), lenses_maslov (upsample/integrator/roi),
_lens_real (kwarg surface, prepare), coatings, apply_mirror.  Skimmed: berreman,
bsdf, emt, freeform, thin_grating, _lens_jax, bor/, eme/.  Not reached: those
modules' physics, all CuPy/JAX paths (inspection only), multibranch/uniform caustic
physics, displaced-model numeric envelopes.

---

## Territory P — propagators/

### Findings

| ID | site | severity | claim (evidence) |
|---|---|---|---|
| P1 | `asm.py:234,337,1021`, `fft_infra.py:1395,1428`, `fresnel.py:95,228`, `mft.py:208` | **CRITICAL-physics** | The `(arange(N)−N/2)/(N·dx)` frequency grid + `ifftshift` puts DC at −0.5 bin for ODD N: every ASM-family propagator silently returns a laterally shifted, phase-wrong field.  Measured (N=257, Gaussian vs ABCD oracle): `angular_spectrum_propagate` max rel err 2.59e-1 with a −3.8916 px centroid walk — matching the closed form Δx=−λz/(2N·dx) to 4 digits — vs 5.7e-5 / 0.0 px at N=256; also fresnel_tf, asm_mft, fresnel (single-FFT).  Discriminators: RS (even 2N pad) and Fresnel-MFT (explicit n−N/2 Bluestein) are CORRECT at odd N.  Reachable in-library: Shack-Hartmann subaperture ASM at odd `sa_pixels` → focal-spot centroid −8.09 px (Np=65) vs −0.19 (Np=64).  No odd-N pin exists anywhere.  (MEASURED) |
| P2 | `fresnel.py:102` | HIGH-bug | `fresnel_tf_propagate` returns the LIVE pyFFTW inverse ping-pong buffer (no copy): the 2nd subsequent same-shape call silently overwrites the previously returned field — measured max|Δ|=0.497 on a peak-1 field, the array becoming byte-identical to a LATER leg's result; live in-library via `carrier.py:475` (`propagate_carrier_referenced` stores it into the returned env).  ASM/RS are safe (fftshift copies).  The `.copy()` fix is already documented at `rs.py:335` for exactly this class (audit F-3).  (MEASURED) |
| P3 | `hf.py:201` | HIGH-bug | Van-Vleck cross-Hessian default `finite_diff_step=1e-9` is ~all roundoff: density amplitude 9.05% low at origin, spatially-varying end-to-end amplitude error up to 1.56e-2 vs exact Fresnel quadrature (h=1e-6 → 8.3e-9).  (MEASURED) |
| P4 | `vector_diffraction.py:164` | HIGH-bug | No guard that the pupil array spans the f·NA rim: when it doesn't, the exit pupil silently becomes a SQUARE at the array-limited NA — measured 5.5× PSF-width error at NA_eff=0.16 vs requested 0.9, zero warnings.  (MEASURED) |
| P5 | `dispatch.py:126,304` | MEDIUM | `propagate(method='auto')` returns a bare ndarray OR a 3-tuple, at a different output pitch, depending on z — caller cannot know without re-running the selector.  (MEASURED) |
| P6 | `system.py:1206` | MEDIUM | `propagate_through_system_jax(method=…)` never reads `method` — ASM always used; the NumPy twin validates and honours it.  (MEASURED) |
| P7 | `hf.py:197` | MEDIUM | `…with_opl_callable(wavelength=)` required-keyword and never read; the callable must return Φ in WAVES, stated nowhere — metres+wavelength is silently ~1e6 wrong.  (MEASURED) |
| P8-P11 | `gbd.py:529`, `fga.py:1686`, `subaperture.py:53`, `mft.py:49` | MEDIUM | `recommend_gbd_sampling(wavelength)` inert (identical output λ=0.4→10 µm); `fga_memory_estimate(nsig)` inert while the real path reads it 6×; `patches_for_box(centred=)` inert AND the tiling is asymmetric (offset w·overlap/2, measured); MFT output grid unvalidated + silent periodic replicas when the output window exceeds the input cell.  (MEASURED/INSPECTION) |
| P12 | `asm.py:249`, `rs.py:286`, `fft_infra.py:1426`, `mft.py:224` | LOW | Band-limit uses the z→∞ asymptote L/(2λz) of Matsushima's exact cutoff while citing the paper (one-sided, never over-filters).  (MEASURED) |
| P13-P16 | various | LOW | Dead z≠0 guards below enforced-positive z; `beam_d4sigma` fallback broadcasts output-shape arange against input-shape field (uncaught ValueError); public `backend.fft.fft2/ifft2` return the raw ping-pong buffer with no ownership contract in the docstring; `PropagationResult.__iter__` yields 2 items where unwrapped kernels unpack 3.  |

### Verified clean (highlights)
Parseval P/P₀=1.000000000 across all 8 kernels (bandlimit on/off); cross-implementation
agreement even-N (ASM↔ASM-MFT ≤1.4e-13, Fresnel↔Fresnel-MFT 4.9e-14, Gaussian-ABCD
oracle down to 2.8e-16); tilted-ASM demod/remod verified with a carrier-bearing input
(centroid z·tanθ to 4 digits; collapses to plain ASM exactly for carrier-free input —
docstring accurate); SAS vs analytic Gaussian ≤5.6e-4 at z=50 mm; dtype-follows-input
honoured by all 12 entry points; Richards-Wolf weights/prefactor/defocus RE-DERIVED
independently and confirmed term-for-term (Novotny & Hecht 3.66); Maslov branch-parity
logic re-derived correct; H-cache keys complete for all five tags (SAS `pad` correctly
absent); fft_infra locking sound, no in-place mutation of cached vectors; zero mutable
defaults, zero bare excepts; NO dead functions in ~600 defs (dead parameters are the
real signal — list above).

### Coverage
Deep: asm, fresnel, rs, sas, mft, _bluestein, fft_infra, hf, vector_diffraction,
subaperture, result, propagation.  Spot: dispatch, mhs, asymptotic_maslov, ensemble.
Skimmed for new pattern classes only (prior sweep's territory): gbd, fga, carrier,
system, hfpi.  Not reached: asymptotic family interior (~4400 lines), GBD beamlet-ABCD
algebra, FGA Gabor normalisation, HFPI Monte-Carlo variance.

---

## Territory M — rcwa/ + pmm/

### Findings

| ID | site | severity | claim (evidence) |
|---|---|---|---|
| M1 | `rcwa/stack.py:1810` | HIGH-bug | `_layer_eig_key` flattens all shapes' (key, repr) pairs into ONE sorted multiset — structurally different shape LISTS collide and a layer silently reuses another layer's eigenmodes.  Measured: two disk layers with centres exchanged return bit-identically the (A,A) answer; correct rasterised oracle R₀=0.03705 vs 0.02805 — 32% relative error, silent, and fully energy-conserving (closure cannot catch it).  (MEASURED) |
| M2 | `pmm/stack.py:204` | HIGH-bug | `add_layer(segments=)` never validates width fractions; `cw[-1]=1.0` silently CLIPS: sum-1.4 input solved bit-identically to a clipped structure (6.8e-2 Jones from the normalising reading); metre-valued widths solve as an unpatterned slab, energy-clean.  The 1-D sibling raises.  Energy conservation provably cannot detect this class.  (MEASURED) |
| M3 | `pmm/stack.py:50`, `pmm/_core.py:786` | HIGH-bug | Classical (φ=0) `PMMStack` lacks the gain/non-propagating incidence guard wired at 26 other PMM sites: n=1−1e-9j → tot=[−0.960,−0.960] SILENT; NaN substrate → tot=[nan,nan] silent; `_warn_stack_energy` is one-sided and NaN-blind where RCWA's `_check_energy` raises on all three (and RCWA's docstring documents these exact past failures).  (MEASURED) |
| M4 | `rcwa/twod.py:484` | HIGH-physics | `fff_nv` unstable/wrong on an axis-aligned METAL square pillar — inside its documented scope: R+T up to 2.15 (raises) or A=0.208 vs converged 0.088 (2.4×) at the surviving point.  The guard gates on CURVATURE (0.006-0.012 for the square → admitted) but the documented-unvalidated Cxy term is driven by max|Nx·Ny| (0.500 for the square, 0.000 for the validated stripe).  (MEASURED) |
| M5 | `rcwa/_core.py:348` | HIGH-bug | `fff_nv` + `stabilize=True` ALWAYS hard-raises on a lossless cell: fff_nv's closure error is inherent (non-Hermitian operator — no finite-truncation energy theorem), so every ladder rung logs an `_EnergyWarning` = "failed attempt"; measured raises at nord=3,4 after 1162 s/922 s (7 full solves each), on the MOST accurate of the three formulations; the tripwire's own printed advice is "Pass stabilize=True".  (MEASURED) |
| M6 | `rcwa/_core.py:117` | MEDIUM | `set_blas_threads` silently inert without `threadpoolctl` (absent here) while `_get_blas_threads()` still reports the cap; two false `# pragma` comments.  (MEASURED) |
| M7 | cross-family | MEDIUM | Order-count kwarg spelled 3 ways; default `formulation` differs across siblings for identical physics; `rcwa_efficiency_2d_shapes` missing formulation/stabilize/symmetry; PMM vs RCWA propagating-order masks use different thresholds.  (MEASURED signatures) |
| M8 | `rcwa/_core.py:650` | MEDIUM | `n_orders_y >= 1` over-broad: N_y=0 is legitimate on a y-invariant cell (closure 1e-16 when allowed) and the forced minimum costs 27× the eigensolve.  (MEASURED) |
| M9 | `pmm/_core.py:1320` | MEDIUM | Geo-eig cache values returned writable BY IDENTITY — latent cache poisoning; the module's own `_readonly()` guard is applied elsewhere but not here.  (MEASURED, no active mutation) |
| M10 | various | LOW | Discarded 2nd returns (3 sites); inert `_sem_modes(robust=)` with contradictory docstring; unreachable `_nv_field_2d(method='xy_wedge')`; stale docstrings (dual-Laurent; "a warning fires" x64).  |

### Verified clean (highlights)
1-D slab/Fresnel analytic limit: worst |RCWA−TMM|=1.3e-15, PMM 3.2e-14 across 50
probes against TWO from-scratch oracles (Airy + Abelès, mutual 6.7e-16); the
lossy-exit-substrate forward flux EXACTLY correct (the one apparent 2.4e-3
discrepancy was the auditor's own oracle bug — re-derived S_z ∝ Re(ε/k_z)|Eₓ|²,
agreement 1e-16, retraction recorded); Li's-rule factorization placement verified
against Li 1997 with a same-limit Richardson check (no wrong-rule convergence);
RCWA↔PMM cross-solver 18 configs ≤1.9e-5 per order; triple-oracle Jones agreement
9.5e-6; deep-subwavelength EMT limit (residual is the physical O((P/λ)²) term);
2-D lossless closure ≤1.2e-13 off-normal AND conical over 10 configs; cache
determinism bit-exact cold/warm/interleaved/post-eviction; even-parity symmetry
fold refutation-tested (hypothesized degenerate-eig routing DISPROVEN, ≤1.4e-15);
no float32 anywhere; `_require_jax_x64` raises at all 12 JAX entry points; NumPy
vs JAX forward-flux selectors equivalent by construction.

Flaky "energy-closure" CI test NOT reproduced (machine saturated by sibling
auditors); two candidate mechanisms ruled out by measurement (cache
non-determinism; BLAS reassociation — 7900× margin at 8 threads); best remaining
candidate (INSPECTION): warnings-filter-ordering interaction between M5's
systematic `_EnergyWarning` and a test that suppresses it.

### Coverage
Measured: rcwa oned/twod/stack/_core (all three formulations, conical, symmetry
fold, caches, tripwires), pmm oned/stack/_core.  Not covered: JAX twins (1427
lines — no gradient/parity checks), CuPy, twod_staggered, stack2d_pure, ASR,
circular truncation, slanted/covariant metrics, internal-field reconstruction,
dispersive/threaded sweeps.

---

## Cross-cutting patterns (what to sweep for next)

1. **Odd-N frequency-lattice convention** (P1) — the `(arange(N)−N/2)` spectral
   grid is only DC-correct for even N.  Any module building its own freq grid +
   `ifftshift` needs the even-N guard or the `N//2` convention.  Sibling of the
   spatial-lattice class fixed in S9-S11 — spectral this time.
2. **Silent-junk enum/kwarg acceptance** — the single largest class this audit:
   E-M3/M4 (`on_noncollimated`, `inversion_method`), E-H8 (handedness), P6
   (`method` unread), M2 (segments unvalidated).  The house rule from the S-sweeps
   (§3: unknown values raise) is enforced in the campaign-touched files and
   widely violated outside them.
3. **Inert parameters** — 20+ documented, sometimes REQUIRED, parameters that are
   never read (E-H6 `n_substrate`, P7 `wavelength`, P8-P10, R-17, E-L9).  These
   are worse than dead code: they are silent contract violations.
4. **Buffer/reference ownership** — P2 (live FFT buffer returned), E-H4
   (prescription held by reference), M9/A-14 (mutable cache returns).
5. **Prepared/frozen objects vs process-wide defaults** — E-H3/H4 and A-3 are the
   same bug in three places: import-time or prepare-time snapshots that setters
   silently don't move.
6. **Guards not inherited by siblings** — R-3/R-4 (grazing kill missing in 2 of 4
   paths), M3 (guard at 26 of 27 sites), E-M8 (NaN convention), R-13 (DOE zero
   period).  When a guard is added, grep for the twins.
7. **Docstring physics claims** — E-C1/E-C2/A-7/E-M13/P12: the five worst
   "physics" findings in elements are documentation asserting physics the code
   does not implement.  Docstrings claiming exactness/conventions should carry a
   pin that measures the claim.

## Consolidated dead-code inventory

Territory sections carry the detail.  Roll-up: `ui/surface_table.py` (370 lines)
and `elements/coronagraph.py` unreferenced; dead numexpr scaffold in
`_lens_traced.py`; `backend/fft.py:55 _jnp_or_none`; glass/TraceResult/
FirstOrderData/DesignResult/ParetoResult dead members; ~20 inert parameters
(pattern 3 above); 10/12 deprecation shims past their stated removal version
(banner emits "removed in v5.27" FROM v5.29); 4 optimizer penalty functions
unreachable (R-17); 6 dead UI actions swallowed by 92 empty `except Exception`
handlers.  Notably: propagators/ has ZERO dead functions in ~600 defs.

## Global coverage statement

Five territories, ~1.35M subagent tokens, every finding above tagged MEASURED has
a self-contained repro script under the session scratchpad (`audit_a/ e/ m/ p/ r/`).
Honest gaps (union): plotting.py, most of analysis/{through_focus,field,ao,
phase_retrieval,ghost}, polychromatic/aberration/coronagraph/coherence/
interferometry, io/codegen + .txt importer, ALL CuPy and JAX numeric paths
(inspection only, no twin-agreement measured), asymptotic-family interior,
GBD/FGA/HFPI numeric interiors, eme//bor//berreman/emt/bsdf physics, RCWA/PMM JAX
twins, pymoo multi-objective, partial-coherence modal accuracy.  One flagged
unresolved question: the world.py coordinate-break rotation transpose (needs a
pure-tilt oracle).  Three auditor self-errors were caught and retracted in-run
(vacuous `rg` scan; carrier-free tilted-ASM probe; PMM segment-units misuse) —
the last became finding M2.

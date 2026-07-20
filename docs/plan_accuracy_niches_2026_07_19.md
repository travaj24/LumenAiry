# Accuracy-Niches Remediation Plan (2026-07-19)

Status: PLAN -- implementation campaign in progress on `feat/accuracy-niches`.
Base: main @ 7c79f44 (v5.25.0 + H6 + PR #20 capability/generality batches).

## Purpose

Close every documented accuracy niche left open by the hammer campaign
(`docs/audit_real_lens_hammer_2026_07_19.md`), the displaced-screen audit
(`docs/audit_real_lens_displaced_2026_07_19.md`), and the capability/generality
batches (PR #20). Each item below records: the measured current envelope, the
root cause, the implementation approach, the adversarial validation plan, an
acceptance gate, and the honest risk that the niche is a genuine model limit
(in which case the deliverable is a measured envelope + automatic routing away
from the weak model, never a silent wrong answer).

Method rules (binding for every phase):

- Fail-before / pass-after oracle-backed tests for every behavior change.
- Defaults stay byte-identical unless the change is a validated accuracy
  improvement; anything else is opt-in.
- Bidirectional adversarial validation: an independent verifier attempts to
  REFUTE each claimed fix (and to prove out each claimed limitation) before
  the phase is accepted. Kills are handed back to a fixer; max two rounds,
  then the finding is documented as open rather than papered over.
- Oracles must be independent of the code under test: the Debye/Huygens
  exact-raytrace scripts (lumenairy-free), ZOS-API POP / Huygens PSF
  (Zemax 2023 R1 via zospy), ABCD q-traces, and energy conservation.
- Every new cache: measured footprint, bounded, registry-enrolled, releasable.
- ASCII-only source; single pytest runs < 10 min, < 16 GB.

---

## N0. Oracle infrastructure (prerequisite phase)

The campaign's verdicts are only as good as the oracles. Three gaps found in
prior rounds must be fixed FIRST, because N1/N2/N6 depend on them.

### N0.1 Ring-Huygens Debye oracle: non-collimated congruence support

- **Current state:** `debye_oracle2.py` (conic/aspheric extension) is BROKEN
  for non-collimated input congruences -- G2 traced it to exit-pupil measure
  mis-weighting (ring weights assume the uniform pupil measure of a collimated
  fan). G2 fell back to a geometric-spot oracle, which ignores diffraction.
- **Approach:** weight each Huygens ring by the actual annular measure the
  traced fan carries to the reference sphere: input amplitude at the launch
  height times the energy-conserving fan Jacobian |h_in dh_in / (r_p dr_p)|,
  with r_p the exit-pupil ray height. Keep the oracle lumenairy-free.
- **Adversarial validation:** (a) collimated limit must reproduce
  `debye_oracle.py` to <0.1% on the f/5 golden numbers; (b) aberration-free
  finite-conjugate lens vs ABCD Gaussian image size/position; (c) f/5 with
  R_in=+150 mm vs ZOS POP; (d) energy closure: image-plane integral equals
  launched power to <0.5%; (e) verifier probes a caustic-adjacent case where
  the Jacobian changes sign (fold) -- oracle must either handle |.| correctly
  or refuse loudly.
- **Acceptance:** all five checks green; the geometric-spot fallback retired
  for congruence cases.

### N0.2 ZOS Huygens-PSF point-source oracle mode

- **Current state:** `zos_oracle.py` runs POP only (Gaussian pilot). No
  diffraction-faithful point-source PSF cross-check exists.
- **Approach:** add a `HuygensPsf` analysis job type (point source at finite /
  infinite conjugate, pupil-limited) returning the PSF grid + centroid/EE
  metrics, alongside the existing POP path.
- **Validation:** unaberrated lens PSF vs Airy analytic (first-zero radius,
  Strehl=1); aberrated f/2 case vs the Debye oracle (two independent
  implementations of the same integral -- must agree <1% on EE radii).

### N0.3 Multi-valued caustic ground-truth case

- **Current state:** FGA's specialty regime (fold/cusp caustics with
  multi-valued ray maps) has never been oracle-validated.
- **Approach:** dense DIRECT Huygens integral (no stationary phase, no ray
  branching assumptions) on a strongly aberrated singlet at a through-focus
  plane exhibiting a fold caustic; modest N is fine (the case is chosen
  compact). This is the ground truth for N6.
- **Validation:** grid-convergence of the direct integral itself (halving dx
  changes r2m <0.5%); energy closure.

---

## N1. Analytic displaced screen: extreme finite-conjugate aberration

- **Current envelope (measured, G2):** congruence-fan displaced screen reaches
  0.83-0.89x oracle r2m on moderate conjugates but ~0.50x on the negative-lens
  real-focus case (M5) and ~0.58x on virtual-image back-propagation.
  Collimated is oracle-accurate (0.99x).
- **Root cause:** a single-plane phase screen cannot represent transverse ray
  displacement THROUGH the element (beam walk between entrance and exit
  surfaces). For extreme conjugates the walk is comparable to the aberration
  scale, so half the aberration is simply not representable at one plane.
- **Approach -- two candidates, decided by measurement, winner ships:**
  1. **Exit-plane remap:** use the congruence fan to build the exit ray map
     h_out(h_in) and apply an energy-conserving coordinate/amplitude remap
     (Jacobian |h_in dh_in / (h_out dh_out)|) plus the OPD screen referenced
     to the exit pupil. The element becomes geometric-transfer x phase.
  2. **Split screen:** factor the element into entrance screen + internal
     homogeneous propagation (t/n per gap) + exit screen, with per-surface
     congruence cosines. Costs one short ASM step per interface gap.
- **Adversarial validation:** M5 real-focus and virtual-image cases plus the
  M1 doublet at R_in=+150 mm, each vs the N0.1-fixed congruence oracle AND ZOS
  POP; phase-continuity probe on the remap (coherent fringe test against
  traced); collimated + moderate matrix numbers regression-pinned
  byte-identical on defaults.
- **Acceptance:** extreme cases from ~0.5x to within 15% of oracle r2m; no
  regression elsewhere. **Risk (honest):** virtual-image back-prop may be a
  genuine single-element-model limit; if <15% is not reachable, document the
  measured envelope and make the N8 gate route those conjugates to traced
  automatically.

## N2. Analytic: decenter / tilt / freeform (rotational-symmetry limit)

- **Current state:** displaced screen derives cosines from a MERIDIONAL fan,
  assuming rotational symmetry. Decentered/tilted/freeform elements have no
  analytic model at all.
- **Approach:** pointwise 2-D obliquity -- evaluate sag and the local surface
  normal on the 2-D grid (analytic sag gradient for conic+polynomial;
  callable hook for freeform), compute cos_ti/cos_tt per point via vector
  Snell against the input congruence direction field (gradient of the carrier
  eikonal W). The meridional LUT stays as the fast path for symmetric
  elements. Surface decenter (dx, dy) enters as sag(x-dx, y-dy); tilt as a
  rotated normal frame (small-angle linear term first, full rotation if the
  verifier kills the small-angle version).
- **Adversarial validation:** (a) symmetric limit must reproduce the LUT path
  to <0.1% (kills interpolation/convention bugs); (b) decentered singlet vs
  ZOS (surface decenter parameters) -- centroid shift AND induced-coma EE
  ratios; (c) tilted element vs ZOS; (d) verifier probes sign conventions by
  mirroring the decenter (+d vs -d must mirror the PSF exactly).
- **Acceptance:** decenter/tilt cases within 10% of ZOS EE radii; symmetric
  regression <0.1%. **Risk:** the traced model does not currently take
  decenter, so ZOS is the only external oracle here; mitigate with the
  N0.2 Huygens-PSF mode as a second, independent Zemax analysis.

## N3. GBD strong-reconvergence frame limit (~0.94 power)

- **Current envelope (measured, G1):** converging input through a negative
  element with strong reconvergence saturates at ~0.94 of input power; the
  missing ~6% is frame incompleteness (documented), `normalize_output='power'`
  recovers the total but not the spatial structure.
- **Root cause:** a single global beamlet frame chosen at the input plane
  under-spans the output phase space after strong congruence reshaping.
- **Approach:** mid-chain re-expansion. Publish a frame-completeness metric
  (power captured by the frame vs input). When it degrades past a threshold,
  propagate to an intermediate plane where the field is compact, reconstruct
  on a grid, re-decompose with a fresh frame matched to the LOCAL congruence
  (carrier-referenced, reusing the H7 machinery), continue. Opt-in
  `reexpand='auto'` first; default flips only if the verifier confirms zero
  regression on the whole existing GBD matrix.
- **Adversarial validation:** the H7 extreme case target power >0.99 AND
  spatial agreement (windowed r2m within 5% of traced); Parseval audit on the
  re-decomposition (no double-counted power); grid-convergence of the
  intermediate-plane reconstruction; regression-pin every existing GBD
  number; runtime measured (expect <=2x on re-expanded chains, document).
- **Acceptance:** >0.99 power + <5% spatial on the reconvergence matrix;
  byte-identical defaults unless flipped with evidence.

## N4. GBD scope gap: `propagate_gbd_through_prescription` carrier extension

- **Current state:** the H7 carrier-referenced decomposition fix landed at the
  `apply_real_lens_gbd` entry only; the prescription-chain entry still runs
  the old frame on diverging input.
- **Approach:** port the same carrier decomposition into the chain entry with
  per-group carrier updates (ABCD-advanced conjugate between groups).
- **Adversarial validation:** chain-vs-sequential equivalence (the chain entry
  must match back-to-back `apply_real_lens_gbd` calls to numerical precision
  on a two-group system); M1 doublet diverging input power >0.99; fail-before
  test demonstrating the old path's power loss.

## N5. Traced `tilt_aware_rays`: entrance-eikonal omission (H6 class)

- **Current state:** the H6 fix restored the entrance-plane eikonal
  k0*W(x_in) on the carrier path; the `tilt_aware_rays` path has the SAME
  omission class (flagged by the H6 agent, docs steer users to `carrier=`).
- **Approach:** apply the identical fix -- add the entrance eikonal to the
  per-ray OPL in the tilt-aware path, sharing `_compute_carrier`'s `w_fn`.
- **Adversarial validation:** fail-before test reproducing the collapse
  (diverging input + tilt_aware focuses at the collimated plane); after the
  fix, tilt_aware and carrier paths must agree on the H6 R_in scan (EE100
  >0.99 at the ABCD focus); collimated tilt_aware byte-identical; verifier
  probes the interaction when BOTH tilt_aware and an explicit carrier are
  requested (no double-count of W).

## N6. FGA: caustic-specialty validation + fidelity cost-cap

- **6a Caustic validation (untested specialty):** run FGA on the N0.3 fold
  caustic vs the direct-Huygens ground truth; run GBD and traced on the same
  case (bidirectional -- their expected degradation at the fold must actually
  be observed, else the "FGA niche" claim is wrong and the docs change).
  Acceptance: FGA within 10% on windowed r2m/EE at the fold plane; the
  comparison table goes into the hammer doc.
- **6b Fidelity cap (0.97 at 0.23-NA, n_p cost-capped):** implement
  content-adaptive momentum sampling -- concentrate p-samples where the local
  spectrum lives (importance sampling from the input Wigner/local-frequency
  estimate) instead of uniform p-grids. Measure the fidelity-vs-cost curve.
  Acceptance: >=0.99 fidelity at unchanged cost on the 0.23-NA diverging
  transport, OR a documented curve + knob if the trade is fundamental.
  Adversarial: verifier checks power normalization under non-uniform p-measure
  (quadrature weights must follow the sampling density) and re-runs the H5
  content-override contracts.

## N7. Carrier-referenced ASM: astigmatic carriers + aperture hardening

- **Astigmatic R_x / R_y:** separable Sziklas-Siegman -- per-axis
  magnification m_x(z), m_y(z), per-axis focus-crossing logic (the two
  crossings happen at DIFFERENT z -- the split/bridge machinery from G2 must
  trigger per axis). API: `carrier=(R_x, R_y)`.
  Validation: cylindrical-lens astigmatic Gaussian vs exact fine-grid ASM at
  both line foci and mid-astigmatic plane (<1% windowed r2m); isotropic input
  through the astigmatic path must reproduce the isotropic path byte-identical.
- **Apertures mid-chain:** a hard aperture on the envelope grid invalidates
  the fitted carrier downstream (post-aperture R differs). Re-fit the carrier
  from the apertured envelope (or the user passes the new conjugate).
  Validation: apertured converging leg vs exact ASM; energy accounting.
- **Non-goal here:** CuPy/JAX backends (performance, not accuracy) -- deferred
  unless trivial.

## N8. Dispatcher gate: Seidel-based true-SA estimator

- **Current state:** the gate uses a conservative per-surface c4 sag-coefficient
  bound with a paraxial height trace -- a MODEL-ERROR proxy, not a system-SA
  estimate. An asphere that nulls system SA still trips it (false positive =
  safe but slow).
- **Approach:** compute the actual Seidel S1 sum along the gate's existing
  paraxial trace (marginal ray, per-surface refraction invariants), including
  conic/aspheric contributions, at the given conjugate; gate on the implied
  W_rms. Keep the c4 bound as the fallback for surfaces the Seidel walk
  cannot classify. The S4/S5 sign fixes (v5.24.4) make the machinery
  trustworthy, but the verifier must INDEPENDENTLY re-check S1 against Debye
  W(h) polynomial fits on M1-M6 (the audit's concern: internal agreement is
  not truth).
- **Adversarial validation:** gate-decision matrix M1-M6 x {collimated,
  diverging, converging}: (a) SAFETY -- no case whose analytic error exceeds
  threshold may route to analytic (checked against oracle errors, not
  internal estimates); (b) measured false-positive reduction vs the c4 bound;
  (c) an SA-nulled asphere case must now route fast.

## N9. Capstone: composed end-to-end + full-matrix adversarial regression

- **E2E:** a production-class composed chain -- real prescription per-group
  traced (H6/N5-fixed) + carrier-referenced legs (N7) + displaced-analytic
  fast path (N1/N2) -- vs ZOS POP end-to-end, at two wavelengths. This is the
  one integration never demonstrated; every piece is oracle-validated but the
  COMPOSITION is not. Uses a generic fast doublet+relay design, NOT tuned to
  any single production design.
- **Full-aperture fast case:** the f/2 M4 at w0=9 mm full aperture, feasible
  now via the pilot-beam machinery (was budget-infeasible at N>20000 exact).
- **Final adversarial pass:** one independent verifier sweeps ALL phase
  claims: re-runs each acceptance gate from scratch, energy audits, grid
  halving/doubling on every headline number, sign-mirror probes, and the
  full cumulative test suite + ruff. Only then does the branch go to PR.

---

## N10. Decentered/tilted elements: transverse ray walk-off (Run 3)

Added at user request after Run 2. P3 delivered the analytic decenter/tilt
*phase* (correct centroid + coma DIRECTION) but the single-plane screen cannot
represent the transverse ray walk between a thick element's surfaces, so the
induced-coma SPOT is wrong: model EE80 NARROWS ~0.906x where the geometric
oracle and ZOS both BROADEN ~1.02-1.03x (decentered EE80 -19% vs ZOS,
grid-robust to N=6144 -- a structural limit, not a sampling artefact). P3 also
established that `traced` AND `gbd` currently IGNORE the decenter/tilt keys
entirely (they return a centered spot), so there was no accurate model to route
to. N10 makes the two ray-based models honor decenter/tilt so they become the
accurate reference, THEN N11 attempts to lift the analytic screen to match.

### N10a Traced decenter/tilt (the accurate reference -- do first)

- **Current state:** `apply_real_lens_traced` traces a FULL 2-D ray congruence
  (Xs_in, Ys_in meshgrid) but the surface intersection + normal ignore the
  `decenter`/`tilt` keys.
- **Approach:** thread per-surface `decenter=(dx,dy)` and `tilt=(tx,ty)` into the
  Newton sag intersection and the surface-normal (vector-Snell) evaluation --
  sag evaluated at the shifted/rotated coordinate `sag(R^T(x-dx, y-dy))`, normal
  = the correspondingly transformed gradient. Because traced already carries
  each ray through the glass gap, the transverse walk-off (and hence true coma
  broadening) emerges naturally. Mirror the field-frame vs surface-frame
  convention already defined for the analytic path (`surface_frame` kwarg) so
  the two models agree on what a decenter MEANS.
- **Adversarial validation:** (a) zero-decenter reproduces the current traced
  result byte-identical (pin); (b) decentered singlet EE80 BROADENS and lands
  within 10% of ZOS (POP + the P8 Huygens-PSF mode) AND within 10% of the
  geometric-spot oracle -- direction AND magnitude correct, killing the P3
  shrink; (c) sign-mirror: +d and -d produce mirror-image PSFs to <1%;
  (d) tilt case vs ZOS; (e) two wavelengths.

### N10b GBD decenter/tilt

- **Current state:** the GBD real-lens path ignores decenter/tilt.
- **Approach:** thread decenter/tilt into each beamlet's base-ray intersection +
  refraction and pick up the LOCAL surface curvature at the (decentered)
  intersection for the differential (ABCD) matrix. Reuse N10a's transformed
  sag/normal helper so the two models share one geometry definition.
- **Adversarial validation:** decentered singlet power >0.99 + EE80 broadening
  within 10% of ZOS and within 10% of the N10a-fixed traced result (GBD vs
  traced cross-check); zero-decenter byte-identical; sign-mirror.

### N11 Analytic 2-D transverse-walk remap (fix the shrink)

- **Approach:** generalize P2's exit-plane remap to the full 2-D off-axis case
  -- launch a 2-D (non-meridional) congruence fan against the decentered
  surface, build the 2-D exit map (x_out,y_out)(x_in,y_in) carrying the
  transverse walk, and apply an energy-conserving scattered->grid remap with the
  2-D Jacobian |d(x_in,y_in)/d(x_out,y_out)| plus the exit-pupil-referenced OPD.
  This restores the walk-off the single-plane screen drops.
- **Adversarial validation:** decentered EE80 broadens to within ~15% of ZOS and
  of the N10a traced reference; symmetric/on-axis limit reproduces the P2
  numbers byte-identical; phase-continuity fringe probe vs traced; sign-mirror.
- **Risk (honest, pre-registered):** P2 measured remap == screen on the on-axis
  conjugate cases (there the walk is symmetric = a radial rescale the screen
  already captures), so the 2-D remap is NOT guaranteed to close the 19%
  directional coma gap. If it cannot reach ~15%, the honest outcome is: keep the
  analytic screen for centroid/pointing, DOCUMENT the residual, and have the N8
  gate route strong-decenter-coma cases to the now-decenter-capable `traced`
  (N10a) -- which is why N10 lands first and gives routing a correct target
  either way.

---

## Execution

| Phase | Items | Depends on |
|---|---|---|
| P0 | N0.1, N0.2, N0.3 oracle infra | -- |
| P1 | N5 traced tilt_aware + N4 GBD chain scope | -- |
| P2 | N1 displaced extreme conjugates | P0 |
| P3 | N2 decenter/tilt/freeform | P0, P2 |
| P4 | N3 GBD re-expansion | P1 |
| P5 | N6 FGA caustic + sampling | P0 |
| P6 | N7 carrier astigmatic + apertures | -- |
| P7 | N8 Seidel gate | P0, P2 |
| P8 | N9 capstone E2E + adversarial regression | all |
| P9 | N10a/N10b traced + GBD decenter/tilt (Run 3) | P3, P8 (ZOS Huygens PSF) |
| P10 | N11 analytic 2-D transverse-walk remap (Run 3) | P9 |

Sequential single-writer Opus agents; each phase = implementer -> adversarial
verifier -> (on kills) fixer -> re-verify, max two rounds; unresolved kills are
documented open findings, never silently accepted. Checkpoint commits after
P0-P4, P5-P8, and P9-P10. Run 3 (P9-P10) launches ONLY after Run 2 (P5-P8) has
landed + committed -- never two repo-writing workflows concurrently. Release
only on explicit user approval.

---

## RESULTS (Runs 1-2, N0-N9; 2026-07-20)

Final per-niche status after the P8 capstone (the last gate before PR for
Runs 1-2).  Every measured envelope below is oracle-backed and pinned by a test.
N10/N11 are Run 3 (not started here).

### Per-niche status + measured envelope

| N | phase | status | measured envelope (oracle) |
|---|---|---|---|
| N0.1 | P0 | SHIPPED | `debye_oracle_v3` congruence diffraction oracle: collimated f/5 EE80 55.4 vs dual-oracle 55.2 um (0.4%); ABCD Gaussian small-beam limit 0.4-1%; virtual-image `asm_backprop` auto-route. Geometric-spot fallback retired for congruence cases. |
| N0.2 | P8 | SHIPPED | ZOS `huygens_psf` oracle mode: unaberrated f/25 first-zero 40.32 vs Airy 40.15 um (**0.4%**), Strehl 1.000; aberrated f/4 uniform pupil vs `debye_oracle_v3` at a matched 110-um window **EE50 1.9% / EE80 0.72% / EE95 0.17%**. |
| N0.3 | P5 | SHIPPED | `caustic_fold_truth.py` fold ground truth: grid convergence 1.3e-5, energy closure 0.999, 2-branch fold; r2m 11.22 / EE50 10.05 / EE80 13.50 um. |
| N1 | P2 | SHIPPED (premise refuted) | The "0.50x floor" was an ORACLE artefact (geometric ray-density spot over-estimates the wave spot ~2x near reconvergence). Default `displaced_mode='screen'` is within **4-8%** of the diffraction-faithful oracle across M5-real / M5-virtual / M1 / M6 (0.916-0.998); `remap`/`split` are documented experimental peers. Defaults byte-identical. |
| N2 | P3 | SHIPPED + 1 OPEN FINDING | Pointwise 2-D obliquity: symmetric limit rel L2 6.8e-6; decenter CENTROID 2.5% vs ZOS / 0.1% vs geom; tilt deflection 0.2% vs rigid-rotation; coma flare DIRECTION mirror-exact. OPEN: induced-coma **EE growth** is directionally wrong (model narrows 0.906x where ZOS/geom broaden ~1.02-1.03x; decentered EE80 -19% vs ZOS) -- a genuine single-plane walk-off limit, PINNED by a regression test; ZOS is the reference for decentered-spot EE. (N10/N11, Run 3, address this.) |
| N3 | P4 | SHIPPED (opt-in) | `apply_real_lens_gbd(reexpand='auto')` closes the ~0.94 strong-reconvergence power cap to **>0.99 power + windowed r2m within 0.3% of traced**; collimated/well-conditioned inputs are not re-expanded (byte-identical to `'off'`); overhead ~1.5-1.9x when it fires. |
| N4 | P1 | SHIPPED | Carrier decomposition ported to `propagate_gbd_through_prescription`: chain == back-to-back `apply_real_lens_gbd` to precision on a two-group system; diverging-input power >0.99; fail-before pins the old path's loss. |
| N5 | P1 | SHIPPED | `apply_real_lens_traced(tilt_aware_rays=True)` entrance-eikonal restored: tilt-aware and carrier paths agree on the H6 R_in scan (EE100 >0.99 at the ABCD focus); collimated byte-identical; no double-count of W when both are set. |
| N6a | P5 | SHIPPED (claim corrected) | FGA at a genuine single fold: r2m 1.5% / EE80 4.9% / fidelity 0.9956 (PASSES the 10% gate) -- but GBD (0.3% / 3.0% / 0.9997) and traced+ASM (0.8% / 0.0% / 0.9991) do NOT degrade there, so the "FGA is uniquely needed at a fold" claim is refuted; only `traced_multibranch` (geometric) degrades (14.2% / 9.9%). |
| N6b | P5 | SHIPPED (opt-in, measured non-improvement) | `momentum_sampling='adaptive'` is a valid non-uniform quadrature (unit-tested; naive uniform weights over-count >200% on concentrated nodes) but a MEASURED non-improvement -- the FGA integrand is beamlet-broadened, so uniform already reaches fidelity 1.000 at n_p=21. Documented fidelity-vs-cost curve; default `'uniform'` byte-identical. |
| N7 | P6 | SHIPPED | Astigmatic `carrier=(R_x,R_y)` separable Sziklas-Siegman: per-axis focus-crossing matches fine-grid ASM **<1%** at both line foci + the mid-astigmatic plane; isotropic `(R,R)` byte-identical to the scalar path; `carrier_referenced_aperture` removes clipped power with exact accounting (no renormalization). |
| N8 | P7 | SHIPPED | `_seidel_sa_wfe_rad` system-S1 gate: an SA-nulled conic now routes to the fast displaced screen (the c4 bound false-positived it); genuinely-aberrated M1-M6 route to a ray member; SAFETY checked against oracle errors, not internal estimates. |
| N9 | P8 | SHIPPED | Capstone composition (STEP A/B/C below). |

### STEP A -- ZOS Huygens-PSF oracle mode (N0.2 prerequisite)

Added a `huygens_psf` job type to `zos_oracle.py` (point source, finite/infinite
conjugate, pupil-limited) returning the PSF grid + Strehl + centroid/EE/first-zero,
alongside POP.  Invocation + full caveats: `validation/oracles/README.md`.

- **Airy (unaberrated).**  Slow equiconvex singlet (f~100 mm, EPD 4 mm, f/25),
  infinite conjugate: Strehl **1.000**, first-dark-ring **40.32 um** vs the analytic
  Airy `1.22*lambda*F/# = 40.15 um` (**0.4%**).
- **Aberrated vs `debye_oracle_v3`.**  Two independent Huygens integrals.  The f/2
  equiconvex singlet stopped to f/4 (aperture 12 mm), UNIFORM pupil, same
  paraxial-focus plane, MATCHED 110-um metric window: **EE50 1.9% / EE80 0.72% /
  EE95 0.17%**.  Two hard-won lessons (both measured): (1) the EE-about-centroid
  metric renormalizes to captured energy, so the metric window MUST match between
  tools -- an un-matched window spuriously halves the Zemax EE (the "0.6x" red
  herring); (2) a heavily-aberrated PSF needs BOTH an adequate image window (>= EE95
  radius) AND high ZOS pupil sampling (full f/2 aperture-24 mm is still
  pupil-sampling-limited at 512x512: EE80 203->306->376 um across 256/512/1024 pupil
  samples, and the Debye ring-Huygens J0 kernel itself carries ~0.7-rad edge phase
  at f/2 -- so f/4 is the clean cross-check regime).

### STEP B -- composed doublet + relay end-to-end (N9)

Generic (not design-tuned) weak cemented doublet (f1~173 mm) + relay singlet
(f2~40 mm), collimated Gaussian w0=1.5 mm, aperture 6 mm.  Propagated end-to-end
through the NEW stack: **per-group `traced`** (H6/N5) for the doublet -> a
**carrier-referenced gap leg** (N7) -> the **`apply_real_lens_universal`-gated
relay** (routes to the displaced phase-screen -- the N8 gate selects it at the
low relay NA) -> a **carrier-referenced FINAL-FOCUS leg** landing at the paraxial
image.  Compared to `debye_oracle_v3` (independent, lumenairy-free) over the whole
5-surface chain at two wavelengths:

| wl | composed EE50/EE80 um | Debye EE50/EE80 um | EE80 ratio | EE50 ratio |
|---|---|---|---|---|
| 1.31 um | 7.286 / 11.014 | 7.354 / 11.110 | **0.991** | 0.991 |
| 0.633 um | 3.523 / 5.338 | 3.642 / 5.495 | **0.972** | 0.967 |

The composition matches the independent oracle to **<1% (1.31 um) / <3% (0.633
um)** -- the integration nobody had demonstrated.  LESSON: the comparison plane is
load-bearing -- landing at the fitted best-focus rather than the oracle's paraxial
image inflates the discrepancy to ~14%; both must measure at the SAME plane.
ZOS POP corroboration recorded in the report (the debye oracle is primary; POP's
Gaussian pilot is the secondary cross-check).

### STEP C -- full-aperture f/2 M4 fast case (N9)

The f/2 M4 biconvex (R=+-51.68 mm, aperture 22 mm, f/2.31) at collimated Gaussian
w0=9 mm ("full aperture" = truncated at the f/2 stop), reconstructed AT FOCUS via
the GBD pilot beam on a MODEST N=3072 grid (dx=8 um), 4.3 s, 3217 beamlets, frame
completeness **0.979**:

| model | EE50/EE80/EE95 um | EE80 vs Debye |
|---|---|---|
| GBD pilot beam @ focus | 60.71 / 182.78 / 322.09 | **0.974** |
| `debye_oracle_v3` (Gaussian w0=9 mm) | 63.13 / 187.63 / 325.40 | 1.000 |
| fixed-grid thin+ASM (same N) | 7.92 / 22.40 / 39.74 | **0.12 (aliased)** |

The exit NA is ~0.64, so a fixed grid holding the full aperture at the exit
Nyquist would need **N ~ 21500** (budget-infeasible for the per-pixel exact
route); the pilot beam propagates each beamlet analytically and needs no fine grid
to sample the exit fringe.  A fixed-grid ASM at the same modest N reads 0.12x
(aliased garbage) -- the fail-before anchor that shows why the pilot beam is
required.  GBD lands within **2.6%** of the diffraction oracle.

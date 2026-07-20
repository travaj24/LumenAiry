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

Sequential single-writer Opus agents; each phase = implementer -> adversarial
verifier -> (on kills) fixer -> re-verify, max two rounds; unresolved kills are
documented open findings, never silently accepted. Checkpoint commits after
P0-P4 and after P5-P8. Release only on explicit user approval after P8.

# Real-lens hammer campaign — findings + resolutions (2026-07-19)

**Scope:** adversarial validation of the real-lens propagator family
(`apply_real_lens` analytic, `_traced`, `_gbd`, `_fga`) against external
oracles, following the 2026-07-18 thin-lens audit's item 5.

**Oracles (two, fully independent):**

1. **Zemax OpticStudio POP via ZOS-API** (zospy 2.1.5, py3.13 venv) —
   prescriptions built with *dispersionless model glass* (vd=0) and the
   exact per-surface indices read back via `INDX` operands, EFL
   cross-checked via `EFFL` (agreed with hand thick-lens calculation to
   6 digits). No index/geometry ambiguity by construction.
2. **Exact-raytrace Debye/Huygens integral** (`debye_oracle.py`,
   session scratchpad) — a self-contained meridional exact raytrace +
   ring-Huygens sum. No FFT, no lumenairy, no Zemax.

Oracle cross-agreement: **0.5–8 % on r2m across every case** (plano-convex
good orientation: 43.22 vs 43.00 µm — 0.5 %). The POP pilot-beam caveat
was explicitly checked and cleared by the second oracle.

**Test cases:** biconvex singlet R = ±51.68 mm, t = 5 mm, n = 1.5168 at
λ = 1.31 µm (benign w0 = 0.5 mm and f/5 w0 = 5 mm illumination);
plano-convex f ≈ 51.7 mm in both orientations (textbook ~4× spherical
penalty reproduced: oracle 43/128 µm); through-focus scans.

## Final verdicts

| model | verdict |
|---|---|
| `traced` | **Validated: 99.7 %** of dual-oracle r2m (64.77 vs 64.98 µm) at rule-compliant sampling; exact ray OPL to λ/44.  Diverging-input carrier path fixed (H6, PR #19): focuses the diverging singlet at the ABCD image, EE(100µm) = 0.999 |
| `gbd` | **Validated: 98.4 %**, grid-insensitive (63.96/63.97 µm at dx = 3/6 µm).  Diverging-input carrier launch fixed (H7): `direction_sampling='auto'` default focuses the diverging singlet at the ABCD image, power 0.997–0.998 across the R_in = 300/150/100 mm scan, NaN warnings gone |
| `fga` | Feasible + sane by default after H4/H5; r2m within ~2 % with matched sampling.  **N6a (2026-07-20): the "FGA niche" over GBD/traced at a genuine MULTI-VALUED fold caustic is REFUTED** -- at a validated fold (f/3.65 singlet, direct-Huygens ground truth) FGA passes the 10 % r2m/EE gate (fidelity 0.9956) but a fine-frame GBD (0.9997, peak-I error 0.0 %) and `traced`(exit)+ASM (0.9991) MATCH or BEAT it; FGA's default is in fact the WORST at peak-I (17 %).  GBD/traced remain preferable for smooth single-valued AND single-fold caustics (cost); FGA's narrow value is a one-pass grid-insensitive caustic-plane field.  See N6a below |
| `analytic` | Honest model plateau: converged 40.5 µm vs oracle 65 (H2); ~50 µm with the H1-fixed slant screen. Cannot represent orientation-dependent aberration (60.4/60.9 µm where truth is 43/128) |
| benign regime | all models exact to 4 digits vs Zemax (29.981 vs 29.979 µm) |

## Findings

### H1 — FIXED: `slant_correction` had inverted cosines (physics-derivation error)

The slant OPD computed `n·sag/cosθ` (ray path-length through a slab)
where the wavefront OPD of a locally tilted refracting facet is
`(n₂cosθ_t − n₁cosθ_i)·sag` — cosines in the **numerator**. The inversion
sign-flips the leading obliquity/spherical term; on a symmetric biconvex
the two wrong-signed corrections cancelled the pupil SA → an impossible
3.6 µm near-diffraction-limited spot vs the 65 µm truth. Fixed in both
byte-identical copies (`_lens_real.py` banded + whole-grid) + docstrings
+ UI tooltip. Corrected slant measures 50.3 µm — the correct-signed term
moves the analytic model *toward* the oracle. `analysis/through_focus.py`
used `slant_correction=True` internally as its "ideal pupil" (it looked
sharp *because* of the bug); goldens derived from it shift.
Tests: `test_hammer_h1_slant_obliquity.py`.

### H2 — QUANTIFIED: analytic sag-screen validity envelope

Per-surface phase screens structurally cannot represent the transverse
ray-displacement physics of orientation-dependent aberration (same
conclusion as the 2026-07-18 audit, now dual-oracle-quantified). Use
`traced`/`gbd` for absolute spot fidelity on aberrated systems; analytic
remains exact in the benign regime.

### H3 — FIXED: traced's sampling rule now enforced (model itself exact)

`traced`'s documented critical-sampling rule (`dx ≤ λ·f/aperture`) was
never enforced; violating it silently aliases the exit converging
wavefront beyond grid Nyquist (folds 8.5 % of energy → r2m reads 37 %
low while EE50/EE80 stay plausible). At compliant dx the model is
**99.7 %** of the oracle. Added an amplitude-aware exit-NA Nyquist
warning (rays ≥ e⁻⁴ of peak input amplitude define NA_exit; warns when
`dx > λ/(2·NA_exit)`; `on_undersample='silent'` suppresses; never
raises). Tests: `test_hammer_h3_traced_nyquist_guard.py`.

### H4 — FIXED: FGA auto-path memory wall

Peak memory ≈ Nq·Np·(80–130 B) with the (Nq, Np) coefficient matrix
allocated unchunked (29.2 GiB at N=8192) because (a) the existing
byte-identical chunk loop was gated on `mem_budget_mb=None`, (b) the
cost model ignored the 9-ray FD Jacobian bundle (~6× under-count),
(c) the FD bundle itself is unnecessary for all-conic prescriptions.
Fixed: default RAM-fraction budget, FD-bundle-aware cost model,
`exact_jacobian` auto-default for conic-only prescriptions.

### H5 — FIXED: FGA auto-sampler halo on near-collimated strong-focusing input

`coarse_stride` was **exonerated** (byte-faithful, rel L2 2.4e-4). The
5×-wrong result came from `_resolve_sampling` flooring `p_max` at the
S2-2 beamlet-completeness width `3/(k·w0)` — 32–130× the field's real
angular content for near-collimated input — whose excess-momentum
beamlets spray a halo ~`efl·p_max` through a strong lens. Fixed: when
floor/content > 10, p_max is content-sized (×3 tail) with the
completeness energy restored via output power normalization + a
RuntimeWarning. S2-2 identity-power and diverging-beam contracts
unchanged and re-verified. Tests: `test_fga_h4_h5.py`.

## Method lessons (for future campaigns)

- **Windowed r2m is domain-sensitive; EE curves + through-focus r2m
  curves are the robust cross-tool metrics.** A full-grid second moment
  is dominated by r²-weighted background at 1e-6 relative intensity.
- **Grid-coverage trap:** the input beam must fit the grid
  (±>2.4 w0); `P_out/P_in` computed post-grid hides input truncation.
- **Beamlet methods (GBD/FGA) are output-grid-insensitive** (phase rides
  analytically); phase-screen models are not — sampling rules differ
  fundamentally between the two families.
- **Best-focus curves discriminate mechanisms:** a shifted curve means a
  reference offset; a deeper minimum means lost aberration.
- **Platform notes:** Windows `np.longdouble` IS float64 (oracle
  pitfalls); Linux glibc's complex exp can beat an f64 mod-2π fold —
  assert platform-independent *bounds*, not per-platform orderings.

## Production-scale addendum (user's 121 scoreboard, 2026-07-19)

The thin-lens audit doc's model scoreboard (production 1x1 no-DOE
control, Zemax reference 2.74 µm, **strongly DIVERGING input**) extends
this campaign's singlet-class coverage and surfaced three findings the
near-collimated hammer cases could not reach — the original
**"NO current lumenairy model is valid for the 121's REAL surfaces"**
verdict, **since superseded by the H6 fix** (traced is now valid for the
diverging-input real-surface class — see below):

- **H6 — RESOLVED (v5.25.0, PR #19) — traced on strongly-diverging input**:
  root cause found — the carrier path (`carrier='auto'` / scalar conjugate /
  ndarray) omitted the carrier's ENTRANCE-plane eikonal `k0·W(x_in)` from the
  per-ray OPL, while the `preserve_input_phase` reference leg (`exp(i·k0·W)`
  through `apply_real_lens`) included it.  The mismatched `−k0·W` imprinted on
  the field cancelled the input divergence the wave model correctly carried, so
  EVERY diverging-input trace collapsed to the COLLIMATED focal plane `f` and the
  true image at `z_img` smeared by `NA_exit·(z_img − f)` (exp22: energy over
  ±1.8 mm, EE(100µm) = 0.9% — reproduced to the digit).  Cruel twist: the
  `on_noncollimated` guard measures the POST-carrier residual (~0), so the broken
  path was SILENT while `carrier=None` warned users into it.  **Fix:** add the
  entrance eikonal to `final.opd`.  Result: EE(100µm) **0.009 → 0.999** across the
  R_in = 300/150/100 mm scan AND per-group relay chains (every intermediate field
  carries finite curvature, so the omission bit at every hand-off — exp22's
  6-group smear).  Collimated input unchanged (W ~ 0, a no-op).  Independent
  oracle: ABCD Gaussian q-trace through the dual-oracle singlet — fully
  independent of the traced implementation.  Tests:
  `test_hammer_h6_traced_carrier_eikonal.py`.
- **H7 — RESOLVED — GBD diverging-beam energy collapse**:
  root cause found — `apply_real_lens_gbd`'s position-only (axial) beamlet
  decomposition carried the input's wavefront **curvature in the beamlet
  AMPLITUDE only, not the LAUNCH DIRECTION**, so the axial base rays refracted
  as if collimated (focused at `f`) and the diverging beam's angular content
  had nowhere to live — the beamlet frame shed it.  Reproduced on the
  dual-oracle singlet (w_L = 1 mm diverging Gaussian, R_in = 300/150/100 mm):
  **power collapsed to 0.55/0.36/0.19** (worse — down to 1e-4 + NaN warnings —
  at the 121's NA~0.23), best focus pinned near `f` not `z_img`.  This is the
  GBD twin of H6 (traced): the input carrier must ride the beamlet launch
  directions.  GBD already had the machinery — the Husimi
  (`direction_sampling=True`) decomposition launches each beamlet along the
  field's local wavevector (`k_local = Im(grad E / E)` = the carrier normal),
  and `apply_prescription_persurface_to_beamlets` already threads those launch
  slopes through the differential trace; the beamlet AMPLITUDE already carries
  `E_in` including `exp(i·k0·W)`, so GBD needs **no separate entrance-eikonal
  term** (unlike traced's H6 fix — the reference is intrinsic to the amplitude
  here).  **Fix:** `direction_sampling='auto'` is now the `apply_real_lens_gbd`
  default — it measures the input's RMS local-tilt spread and launches Husimi
  when the wavefront is curved/tilted, axial otherwise.  A flat-wavefront
  (collimated) input measures spread **exactly 0** (real / globally-phased
  field), so it takes the byte-identical axial path — the 98.4 % collimated
  baseline is preserved to the bit.  Result: **power 0.997–0.998** across the
  R_in = 300/150/100 mm scan, best focus at the ABCD `z_img` to within the
  scan resolution (< 5 %, and beating `f`), **NaN warnings gone**.  Independent
  oracle: ABCD Gaussian q-trace through the dual-oracle singlet.  Tests:
  `test_hammer_h7_gbd_diverging.py`.  *Envelope:* validated for a
  single-congruence (single-beam) diverging/converging or tilted input; the
  per-pixel Husimi launch, like traced's `tilt_aware_rays`, is not for a
  multi-emitter interference field (noisy per-pixel tilt) — decompose those
  per source.  At extreme finite conjugate (R_in → f, e.g. 60 mm) the frame
  completeness costs a few tenths of a % of power at a coarse `bpa`; it
  recovers with a finer beamlet frame.  **G1 CORRECTION (2026-07-19):** that
  "recovers with a finer frame" claim holds for a POSITIVE element only.  A
  converging input RECONVERGED by a NEGATIVE element to a *near* real focus
  (G1 M5: biconcave, R_in = -35 mm → real image ~108 mm) sheds ~6 % of power
  that a finer frame does NOT recover -- power saturates at ~0.94 at the MAX
  frame density (0.88 at bpa 128 → 0.94 at bpa ≥ 256 = step 1) and is
  non-monotonic in `waist_factor` (overshoots to 1.16 at wf 2), i.e. a
  frame-completeness/normalization floor for the negative-lens reconvergence
  geometry, NOT a density knob.  The LAUNCH is still correct (focuses at the
  ABCD image, EE ~ 0.999) and `normalize_output='power'` restores absolute
  power exactly.  Power stays > 0.99 when the negative element's output is
  gently diverging / virtual (M5 at the task's literal R_in = -60 mm: 0.9956)
  and for a converging input through a POSITIVE element (0.998).  The doublet
  M1 with a diverging input (R_in = +150 mm) conserves 0.998 and focuses at the
  ABCD image.  A genuine frame-completeness fix for the negative-reconvergence
  geometry is a G2 item.  The world-frame machinery
  (`world_output_plane`) was checked and is **orthogonal** — it re-references
  the OUTPUT plane for FOLDED systems and its focus finder itself assumes a
  collimated input, so it does not address the INPUT-carrier class.
- **H8 — RE-TESTED (v5.25.0, H4/H5 fixes landed) — FGA at production scale**:
  the 345 GB `_gabor_coeff` wall is **GONE** — H4-FIXED, no OOM.  The default
  RAM-fraction chunk budget + FD-bundle-aware cost model bound the chunked peak
  (**16 GB observed at an 18 GB budget, N = 4096**), and the auto sampler is
  correct for the diverging input: content-sized `p_max`, the H5 near-collimated
  override correctly **NOT** triggered, analytic Jacobian auto-selected (conic
  prescription).  FGA itself is **validated correct** — 0.2% vs the ASM focusing
  control, 0.97 field fidelity at 0.23-NA diverging transport (n_p cost-cap
  limited).  BUT it is now **grid-floor-bound, not coefficient-bound**: the raw
  grid arrays are ~40 GB (~63 GB total) at production N = 28672, so it needs a
  ≥ 64 GB box + an explicit `mem_budget_mb` on shared machines; and it is
  **compute-infeasible** at the production swarm size (Nq ~ 161 M).  **KEY
  INSIGHT:** the diverging input's OWN phase fringes set the grid — the fringe
  pitch `λ·R/r ~ 12.6 µm` at the beam edge mandates the production N
  **propagator-independently** (no model can render this field on a coarser grid).

With the H6 fix, **`traced` (carrier-referenced) is now the reference model for
the diverging-input real-surface class** — it focuses the diverging singlet at the
ABCD image (EE(100µm) = 0.999) and threads per-group relay chains, so it is the
practical stack choice for real-surface residual aberration on this beam.  With
the H7 fix **`gbd` (carrier-normal launch, `direction_sampling='auto'`) is now
valid for the same diverging-input class** — power 0.997–0.998 and focus at the
ABCD image — so the beamlet model is again a peer choice (its per-surface tensor-Q
carries astigmatism the traced per-pixel OPL renders differently).  The
conjugate-matched `stigmatic` thin chain remains the fast aberration-free
design-intent surrogate (2.97 µm / EE6 = 100% vs Zemax 2.74 µm; the real design's
residual WFE is only ~26–35 mλ), a higher-order sag-screen projection correction
in the analytic model is still open, and H8 (FGA at production scale) is
memory/compute-bound rather than incorrect.

## N6a -- FGA fold-caustic validation: GBD/traced do NOT degrade at a genuine fold (P5, 2026-07-20)

**Ground truth (independent, lumenairy-free):** `validation/oracles/caustic_fold_truth.py`
builds the reference wave field at a THROUGH-FOCUS plane of a strongly-aberrated
plano-convex singlet (convex-first, R1 = 2.7 mm, t = 1.0 mm, n = 1.5168,
f ~ 5.2 mm, **f/3.65**, Fresnel number N_F ~ 82; observation plane 4.3704 mm past
the exit vertex, 1.0 LSA short of the marginal focus) where the collimated ray map
genuinely **FOLDS** -- two ray branches reach each radius inside a caustic RING at
14.26 um, with Airy-type fringes on the bright side (on-axis I/I_peak = 0.32, peak
at the 6 um fringe) and an exponential dark tail. The reference is a DENSE, DIRECT
Rayleigh-Sommerfeld ring integral of the exact exit field (energy-conserving
amplitude + eikonal phase) -- **no stationary phase, no ray-branch assumptions** in
the propagation, so the fold interference emerges on its own. Self-verified (all
lens-model-independent): **grid convergence** (windowed r2m changes 1.3e-5 << 0.5 %
under fan/rho doubling), **energy closure 0.999**, and the **fold** (2 branches).
Axis-centred reference metrics on the N = 1280, dx = 2 um grid:
**r2m 11.22, EE50 10.05, EE80 13.50 um** (reproduces the grid-free radial metric to
<1 %; the peak-centred `_wave_metrics` convention inflates r2m ~12 % on a
rotationally-symmetric RING caustic -- the peak sits off-axis -- so axis-centred
metrics are used for every model).

**Comparison** (each model -> field at the fold plane vs the SAME reference, same
grid, axis-centred; N = 1280, dx = 2 um):

| model | r2m um (err) | EE80 um (err) | field fidelity | peak-I err | verdict |
|---|---|---|---|---|---|
| direct-Huygens reference | 11.22 | 13.50 | 1.0000 | 0.0 % | ground truth |
| `apply_real_lens_fga` (auto default) | 11.39 (**1.5 %**) | 14.16 (**4.9 %**) | **0.9956** | **17.0 %** | within 10 % r2m/EE -- PASSES the gate |
| `apply_real_lens_gbd` (b80) | 11.19 (**0.3 %**) | 13.90 (3.0 %) | **0.9997** | **0.0 %** | most accurate -- NO degradation |
| `traced`(exit) + exact ASM | 11.13 (0.8 %) | 13.50 (0.0 %) | **0.9991** | 0.8 % | accurate -- NO degradation |
| `traced_multibranch` (geometric) | 9.63 (14.2 %) | 12.17 (9.9 %) | 0.9003 | 1.1 % | degrades (SHAPE) |

**FINDING (bidirectional, honest -- the naive niche is REFUTED).** The plan's N6a
required FGA to land within 10 % (it does: r2m 1.5 %, EE80 4.9 %, fidelity 0.9956)
**AND** GBD/traced to be SHOWN to DEGRADE at the fold. **They do not.** On the plan's
own windowed r2m / EE / fidelity metric, a properly-framed GBD (fidelity 0.9997,
peak-intensity error **0.0 %** -- the most accurate model here) and `traced`(exit) +
exact ASM (0.9991) MATCH or BEAT FGA at this genuine multi-valued fold. So the
blanket claim that FGA is *uniquely needed* at fold/cusp caustics while GBD/traced
degrade is **false for a single-fold caustic**, and the FGA verdict is corrected
(scoreboard row updated). In fact FGA's DEFAULT auto config has the WORST
peak-intensity rendering here (17 % -- the wider default frozen beamlet
`w0_factor=5` + the H5 content-sized / power-normalized swarm blur the sharp caustic
peak).  A finer frozen beamlet (`w0_factor=3`, the FGA convergence knob) is expected
to sharpen the peak, but that is a hand-tuned config; the DEFAULT is what ships and
what a user gets, and its 17 % peak-I -- the WORST of the wave models here -- is the
honest measured number.

Why GBD/traced hold up at a fold:

- **`traced` returns the field at the EXIT VERTEX** (single-valued -- the fold is
  DOWNSTREAM); the caustic is then formed by an EXACT wave-propagation step (ASM),
  which renders the fold accurately. `traced` never inverts the multi-valued map, so
  it structurally cannot "degrade" at the fold -- it defers the caustic to wave
  optics. (Cost: two steps + a grid that Nyquist-resolves the converging exit
  wavefront and the fold fringes; FGA renders it in one beamlet pass,
  grid-insensitively.)
- **GBD** with an adequately fine beamlet frame (`beamlets_per_aperture=80`) resolves
  the fold interference to fidelity 0.9997 / peak 0.0 %; its thawed per-beamlet phase
  mis-renders a caustic only with a COARSE / few-beamlet frame, not here.
- The ONLY model that degrades in SHAPE is `traced_multibranch` (the pure GEOMETRIC
  multi-arrival / Maslov construction: fidelity 0.9003, r2m 14.2 %; its peak-I error
  is only 1.1 %). FGA (full wave) correctly beats the geometric construction, but
  that is a specialty geometric tool, not the primary GBD / `traced` family.

**Corrected FGA verdict.** FGA is caustic-ACCURATE and passes the fold gate, but at a
properly-sampled fold it holds **no accuracy advantage** over a well-framed GBD or
`traced` + wave propagation, and its DEFAULT is actually the WORST at peak intensity.
Its genuine, narrower value is: (i) rendering the multi-valued caustic-plane field in
ONE beamlet pass, grid-insensitively (no separate wave-propagation step, no
exit-Nyquist grid a `traced`+ASM needs); (ii) the documented coarse-frame
peak-intensity edge over GBD from the original SA-caustic test (0.01-0.07 vs
0.03-0.34) -- which is NOT reproduced here against a fine-frame GBD (whose peak-I
error is itself 0.0 %). Reach for FGA when you need the caustic-plane field directly
and cannot afford the exit-Nyquist grid; otherwise GBD / `traced` are equal or better
and cheaper.

Tests: the fold-caustic comparison is a scratch script (heavy -- N = 1280, four
propagators); the reference oracle's self-checks and the sampling-weight correctness
are unit tests (`tests/unit/test_niche_p5_sampling.py`). Reference case + field are
installed at `validation/oracles/caustic_fold_case.json` /
`caustic_fold_ref.npz`.

## N6b -- FGA content-adaptive momentum sampling: implemented + opt-in, but a MEASURED non-improvement (P5, 2026-07-20)

The `n_p`-capped ~0.97 fidelity on a 0.23-NA diverging transport (H8) motivated
CONTENT-ADAPTIVE momentum sampling: importance-sample the `n_p` momentum nodes by
the field's angular spectrum instead of a uniform grid.  Shipped as opt-in
`apply_real_lens_fga(momentum_sampling='adaptive')` (default `'uniform'`,
**byte-identical** to prior releases), with the MATCHING per-node midpoint-cell
quadrature weights (which reduce to the uniform `dp**2` on a uniform grid) and
auto power-normalization (the non-uniform quadrature carries the correct RELATIVE
weights = the SHAPE, but not the calibrated uniform absolute scale).

**MEASURED fidelity-vs-cost (fidelity vs exact ASM, free-space transports, dq_step 2):**

0.23-NA diverging beam (content 0.23, p_max 0.345, z = 60 um, N = 256):

| n_p | dp | uniform fid | adaptive fid |
|---|---|---|---|
| 11 | 0.069 | 0.9305 | **0.9452** |
| 21 | 0.0345 | **1.00000** | 0.9942 |
| 31 | 0.0230 | **1.00000** | 0.9993 |
| 61 | 0.0115 | **1.00000** | 0.99998 |

Concentrated-spectrum WIDE cone (content 0.093, p_max 0.30 = 3.2x content, z = 200 um):

| n_p | uniform fid | adaptive fid |
|---|---|---|
| 15 | **0.9895** | 0.9812 |
| 31 | **0.9997** | 0.9993 |
| 61 | 0.9997 | 0.9996 |

**FINDING (honest, adversarial).** Content-adaptive momentum sampling does NOT
robustly improve FGA fidelity at fixed `n_p`:

1. The "0.97 cap" does NOT reproduce for a smooth 0.23-NA diverging FREE-SPACE
   transport -- UNIFORM already reaches fidelity **1.000 at n_p = 21**.  The
   conservative `dp` target (0.008) over-estimates the sampling a smooth diverging
   beam needs; the historical 0.97 was a STRUCTURED-field (fine-fringe) /
   through-lens phenomenon.
2. Adaptive is neutral-to-slightly-WORSE at n_p >= 21; it wins only in the
   very-coarse n_p = 11 regime (0.945 vs 0.931).
3. Even at p_max/content = 3.2 (a wide cone), uniform beats adaptive (0.989 vs
   0.981 at n_p = 15).

**Root cause -- the trade is FUNDAMENTAL.** The FGA reconstruction integrand is not
the raw field spectrum but the spectrum CONVOLVED with each frozen beamlet's own
momentum Gaussian (`sigma_p ~ 1/(k w0)`), so it is intrinsically broad -- never
concentrated enough for importance sampling to beat uniform, and concentrating by
the raw content UNDER-samples the completeness skirt.  Where the genuine 0.97 cap
lives -- a STRUCTURED field whose fine fringes fill p-space with a BROAD spectrum --
there is nothing to concentrate, so adaptive cannot help there either.  Importance
sampling would only win for a SPARSE / multi-peak spectrum (discrete emitter tilts
with spectral gaps), a niche the auto sampler does not target.

**Acceptance (plan N6b OR-clause: ">= 0.99 fidelity at unchanged cost, OR a
documented curve + knob if the trade is fundamental").**  The
">= 0.99-at-fixed-cost" branch is not delivered by adaptive (uniform already meets
it; adaptive does not beat it).  The **OR branch IS met**: the fidelity-vs-cost
curve is documented above, the opt-in knob is shipped (default byte-identical), and
the trade is shown fundamental.  The **adversarial weight-correctness** check is a
unit test -- the matching midpoint-cell weights are a valid quadrature (< 30% error
on a test integral, converging in `n_p`) while forcing the naive uniform `dp` weight
on the SAME concentrated nodes over-counts by **> 200%** -- and the H5
content-override contracts stay green.  Tests: `tests/unit/test_niche_p5_sampling.py`.

## Follow-ups (tracked, not blocking)

- **H6 — DONE** (traced diverging-input carrier entrance eikonal, PR #19) and
  **H8 — RE-TESTED** (FGA on the 121 is now H4-bounded, but grid-floor- and
  compute-bound at production N, not incorrect) — both closed above.
- **Dispatcher aberration gate — DONE**: `apply_real_lens_universal` now estimates
  the sag-screen spherical-aberration error at routing time
  (`_sag_screen_aberration_rad`: per-surface `k·r⁴/(2|R|³)`) and steers a
  heavily-aberrated prescription AWAY from the analytic `phase_screen` even at low
  NA (calibrated: the f/5 case ~21.7 rad trips, the benign small-beam ~0.002 rad
  does not), warning when `phase_screen` is explicitly forced out of envelope.
  Tests: `test_gate_h2_*` in `tests/unit/test_fga.py`.
- Extended oracle matrix: 4f relay at finite conjugates (121-chain
  class), cemented doublet, point-source (ZOS Huygens PSF oracle),
  FGA's true multi-valued-caustic specialty.
- Re-baseline any goldens derived from `through_focus`'s internal
  slant-corrected reference fields (H1 blast radius).
- The traced JAX×OpenBLAS lstsq deadlock library-side mitigation
  (threadpool_limits(1) when jax is co-imported) — from the v5.24.4
  cycle, still open.

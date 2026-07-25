# Root-cause audit — the traced element's FROZEN INTRA-GROUP AMPLITUDE (2026-07-24)

**Status: SUPERSEDED IN PART by §6 (same-day adversarial review).  The frozen-amplitude
mechanism (§4) is CONFIRMED as a real defect, but it is a ~3-EE3-point effect, its
chain-level narrative had the WRONG SIGN for the 121, and the DOMINANT defect is a
different, first-order carrier-convention mismatch (§6.2) that closes the ideal chain
EE6 63.4% → 99.7% by itself.  §5's fix design is superseded by §6.5 — implementing §5
first would REGRESS the headline metric (measured, §6.3).  Read §6 before acting on
anything above it.**

Continues `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` (the P0 closure plan) on the
64-GB-class box it called for (this box: 136 GB / 20 cores).  All numbers below were
measured this session on this box; the prior box's axis-B rows reproduce here **to the
digit** (EE6 50.0 / 59.8, window 58.2 / 68.7 at N=2048, wf=2.0, nfc=2048/4096), so
cross-box parity is exact and every comparison below is like-for-like.

Artifacts: `validation/repro_traced_carrier_121/stigmatic_control_121.py` (new),
`validation/repro_traced_carrier_121/traced_group_dx_probe.py` (new),
`repro_dx_scaling.py` (gained `NFC`/`WF`/`RNF` env knobs, defaults unchanged).

---

## 1. P0.1 stigmatic control — the MACHINERY IS CLEAN (both axes)

`propagate_traced_carrier_chain` was run on the real 121 geometry with
`apply_real_lens_traced` monkeypatched to an IDEAL STIGMATIC element (exact
sphere(R_in)→sphere(R_out) phase map from the same ABCD the orchestrator uses, hard
aperture, |T|=1).  Same gaps, same envelope/reconstruct hand-offs, same
`_fine_trace_group_exit` resample+retrace, same exact Bluestein readout.

| axis | settings | result |
|---|---|---|
| B (n_fine_cap) | N=2048, wf=2.0, nfc=2048/4096/8192 | **IDENTICAL to the digit**: FWHM 4.15 µm, EE6 65.2%, EE12 79.8%, window 84.0% at all three |
| (traced, same axis) | same | EE6 50.0 → 59.8 → **28.1% (collapse)** |

Per-stage power in the stigmatic chain is constant to 7 digits; the peak lands exactly
on-axis (the traced chain lands 21 µm off-axis, diagonally — a further element-side
symptom).  Supporting unit-energy audit (self-contained, no .zmx):
`_fourier_upsample_crop` Parseval = 1.000000 up & down, smooth and hard-edged (loss ≤
0.17% on hard-edge spectral truncation, never gain); `propagate_carrier_referenced`
conserves power to 8 digits and reproduces the analytic Gaussian w exactly at every
N ∈ {1024…16384}; `angular_spectrum_propagate_mft` window power = 100.000% with
analytic-matching focus, converged from N=1024.

**Conclusion: the S1(B) divergence/collapse axis is entirely the traced element re-run
on a finer grid inside `_fine_trace_group_exit`.  Nothing in the carrier / resample /
readout machinery moves.**

## 2. Per-group dx probe — the element itself diverges at fixed physics

`traced_group_dx_probe.py`: single 121 group, chain-representative carrier Gaussian
input, window fixed at 8.4 w, PHYSICAL ray pitch fixed (rs ∝ N), grid pitch alone swept.
Exit compared pointwise to the exact meridional-ray oracle.

**Lens S3-S4** (first group; aperture 24.65 mm = 2.47× the 10 mm beam):

| N | dx µm | rms_res rad | P_out/P_in | w_out/w_in |
|---|---|---|---|---|
| 512 | 81.9 | 0.004 | 1.000 | 0.9999 |
| 1024 | 41.0 | 0.007 | 1.000 | 1.0002 |
| 2048 | 20.5 | 0.015 | 1.000 | 1.0005 |
| 4096 | 10.2 | 0.041 | 1.000 | 1.0026 |
| 8192 | 5.1 | **0.243** | 1.000 | **1.0578** |

S5-S7 shows the same w-inflation pattern (1.000 → 1.054) with a clean core phase.
Power is conserved everywhere — this is pure redistribution, the chain-level F-B
signature (stage-exit envelope +3.4% at S3-S4, +11% by S16-S17, window-total decline
87.6→75.2% across the N sweep) reproduced in ONE group with zero chain machinery.

## 3. Ingredient discriminator — two defects, cleanly split

S3-S4, N ∈ {2048, 8192}:

| variant | rms_res @2048 | rms_res @8192 | w_out/w_in @8192 |
|---|---|---|---|
| V1 traced default (`preserve_input_phase=True`) | 0.015 | **0.243** | 1.0578 |
| V2 traced `preserve_input_phase=False` | **0.001** | **0.012** | 1.0578 |
| V3 bare `apply_real_lens` (amplitude only) | – | – | **1.0579** |

- **Defect A (phase):** V1−V2 = the wave-phase pair `angle(E_analytic) −
  angle(E_analytic_pw)` contributes 0.015 rad at the chain's own pitch and 0.24 rad at
  fine pitch, per group.  The pure `k0*opl_map` phase (V2, carried by the H6 carrier
  entrance eikonal) is 20× cleaner at fine dx and 15× cleaner at the chain pitch.
- **Defect B (amplitude):** the exit-width inflation is byte-identically present in the
  bare `apply_real_lens` amplitude — the traced element inherits it via
  `amp = |E_analytic|`.

**Taper/rim hypothesis REFUTED (V4):** clamping the aperture to 4w (radius 2w) and,
independently, soft-tapering the input tail at 1.75–2.25w leave BOTH defects unchanged
at N=8192 (rms 0.243, w ratio 1.056).  The divergence does not come from
marginal/rim-zone content; it comes from inside the beam.

## 4. The root cause — the amplitude leg runs on a carrier-ALIASED field and
##    freezes the intra-group beam expansion

The chain (and any carrier-referenced call) hands `apply_real_lens_traced` a field whose
spherical carrier is far beyond the grid Nyquist inside the beam itself: at N=2048
(dx = 20.5 µm), the R_in = +47.9 mm entrance sphere aliases beyond r = R·λ/(2dx) ≈
1.5 mm, while the beam is w = 5 mm.  The carrier ARCHITECTURE handles this for the
PHASE (pointwise conjugation is exact at sample points; the traced OPL is referenced to
the analytic carrier — H6/R7).  But **Step 1's amplitude pass — `apply_real_lens(E_in)`
— receives the raw aliased field**.  An aliased sphere cannot diffract/diverge on the
grid, so the wave pass transports the amplitude through the group essentially
UNCHANGED.

The physics says otherwise.  Exact q-trace through S3-S4 (n(N-SK2, 1.31 µm) =
1.5917 from the registered Sellmeier; exit R reproduces the repro's ABCD R_out
+143.37 mm to 4 digits):

- true exit beam: **w = 5.6265 mm = 1.1267× the entrance** (7 mm of glass at
  R ≈ +48 mm),
- traced element delivers **1.0005×** at N=2048 (frozen at the entrance beam),
  **1.058×** at N=8192 (the aliasing radius has moved out to ~6 mm, so the pass begins
  to resolve the real divergence — the model **crawls toward the truth as dx → 0**, and
  the "F-B divergence" is precisely this crawl, superposed with defect A's growing
  phase corruption).

Chain-level consequence, verified against the stage prints: S3-S4's exit re-envelopes
with the CORRECT R_out but the WRONG (frozen) w, so S5-S7 receives w = 5.40 mm where
the design q-trace says 6.02 mm — and so on down the chain, compounding.  A corrected
relay's aberration balance and its final NA assume the design beam sizes; the chain
therefore runs off-design at EVERY grid, with dx only changing HOW wrong:

- coarse dx → fully frozen amplitudes → under-sized beam → under-filled final NA →
  wide focus (FWHM 3.95–4.05 µm vs Zemax POP 2.74 µm) with clean-ish per-group phase;
- fine dx → partially restored expansion but 0.24 rad/group of defect-A phase
  corruption → focus collapse (EE6 46.5% at N=8192; nfc=8192 collapse row).

The per-group oracle gates (R7: < 0.023 rad) never caught this because they validate
PHASE against the oracle and launch each group with its own TRUE q-trace beam; the
amplitude expansion was never gated, and per-group stage w "looks plausible" in the
chain because the gap transport grows the beam at the correct RATE from the wrong size.

This also subsumes/illuminates:
- **the R9-addendum's "~1.68 rad accumulated wavefront"** entering the tail — the
  accumulated inter-group inconsistency of a chain run off-design (NOT a launch defect
  — Part E's negative result is consistent);
- **the 21 µm diagonal peak offset** (aliased-carrier junk breaking symmetry — the
  stigmatic control is exactly on-axis);
- **the axis-B collapse** (`_fine_trace_group_exit` re-runs the LAST group at
  dx_fine ≈ 1 µm at nfc=8192 — deep into defect A's fine-dx regime);
- part of the **aperture:beam cliff**'s sharpness on fast singlets (E4) remains a
  separate (fit-domain) finding — V4a shows the 121 divergence is NOT the cliff.

## 5. Fix design (pending adversarial review — NOT yet implemented)

Principle: when an explicit carrier R_in is engaged, no ingredient of the assembly may
consume the raw (carrier-aliased) field.  The phase side already obeys this (traced OPL
+ H6 carrier eikonal).  The amplitude side must be rebuilt in the carrier frame:

1. **Carrier-referenced amplitude (paraxial remap), opt-in kwarg on
   `apply_real_lens_traced`:** de-chirp `env = E_in · exp(−i k0 W(R_in))` (pointwise,
   exact); compute the intra-group magnification for the R_in congruence from the
   group's own air-to-air ABCD, `m = A + B/R_in` (equals the q-trace w ratio; 1.1267
   for S3-S4 — same ABCD source as `_paraxial_group_r_out`); amplitude_exit(r) =
   |env|(r/m)/|m| (band-limited envelope interpolation), then the existing
   valid/aperture masks.  Intra-group envelope DIFFRACTION is neglected (Fresnel number
   over ~t/n at mm beams ≫ 1; quantify in the gate).
2. **Exit phase (defect A):** replace the wave-pair with `k0·opl_map` (V2 form) PLUS
   the input envelope's residual phase transported on the same remap,
   `angle(env)(r/m)` — the geometric transport of the non-carrier residual (this is
   what `preserve_input_phase` was for; the remapped-residual form carries it without
   ever touching the aliased total phase).
3. **Gating:** engages ONLY for an explicit engaged carrier (chain case); `carrier=None
   / 'auto'-flat` paths byte-identical.  Chain opt-in first
   (`traced_kwargs={'carrier_amplitude': 'remap'}` or equivalent), promoted to the
   chain default once gates pass.
4. **Gates:** (a) per-group probe: w_out/w_in = q-trace ratio ± <1% and rms_res ≤
   ~0.02 rad, FLAT across N = 512…8192; (b) chain axis A+B: EE6 plateau (<2 pt drift),
   window ≤ stop losses, monotone approach; (c) byte-identical default suite; (d) 121
   vs Zemax POP (2.74 µm) — the P1 measurement, expected to close most of the 50→99
   gap if this diagnosis is complete.

## 6. ADVERSARIAL REVIEW OUTCOME (2026-07-24, same day) — the dominant defect is a
##    CARRIER-CONVENTION MISMATCH; §4's causal claim corrected; §5 superseded

An independent adversarial review (Opus subagent; scratch scripts
`scratchpad/review/{a_alias,stig_variants,real_traced,d_closure*}.py`, key runs
re-verifiable via the env knobs documented in them) attacked §1–§5.  Verdicts:

### 6.1 §4's mechanism CONFIRMED — as a real but ~3-point defect with the wrong sign

The carrier-aliasing → frozen-transport mechanism was reproduced in isolation (bare
band-limited ASM, no lens: Gaussian × exact sphere, sweep dx so the alias radius
`r_al = |R|·λ/(2·dx)` crosses the beam).  The freeze is coherent (power conserved to
1e-6, exit stays Gaussian) and is pure *angular clamping* (the grid cannot transport
rays beyond λ/2dx).  Transition at `r_al/w ≈ 1`, full recovery by `≈ 2`; reproduces
the §2 probe numbers quantitatively.  **Compression freezes identically** — and that
flips §4's sign for the 121: the relay is net DEMAGNIFYING
(Π|m_k| = 1.1267 · 1.0549 · 1.0000 · 1.0000 · 0.9419 · 0.9179 · 0.8744 · 0.3776 =
**0.339**), so the frozen chain runs its final leg at an artifactual NA ≈ 0.45 where
the TRUE design exit NA is **0.152** — every audit's "NA ~ 0.46 final leg" is itself
the frozen-w artifact (`na_exit = w_in/|R_out|` at `carrier.py:2176` with the
unexpanded w).  Direct cost of frozen amplitude in an otherwise-perfect chain:
**3.2 EE3 points / 0.3 EE6 points** (§6.3 factorial).  §4's "under-sized beam →
under-filled NA → wide focus" narrative is REFUTED (backwards for this design).

### 6.2 The DOMINANT defect (new): the chain hands the element a PARABOLIC carrier;
the element consumes an EXACT-SPHERE carrier

`carrier_referenced_reconstruct`/`_envelope` build `exp(±i·k·r²/2R)`
(`_radial_carrier_phase`, carrier.py:318); `apply_real_lens_traced`'s carrier
machinery — `_compute_carrier` scalar branch, the H6 entrance eikonal, and
`_reference_input()` — all use the exact sphere `W = sign(R)(√(r²+R²) − |R|)`
(`_lens_traced.py:1207-1230`, whose own comment warns the parabola "leaves several
radians of spurious r⁴" on a steep conjugate).  At `carrier.py:2206→2207` and
`1948→1983` the field is therefore built with one convention and consumed as the
other; the injected wavefront error is `+k·r⁴/(8R³)` per group: **+3.4 rad** at
S3-S4 (r=w), −1.3 rad at S23-S24, **−5.9 rad** at S25-S27.  The R7 per-group oracle
gates are blind to it by construction (the probe launches `E_in` with the SPHERE —
`traced_group_dx_probe.py:157` — so element-only tests never see the chain's
parabola).  This resolves the "individually clean groups / broken chain" paradox and
is the best-supported home for the R9-addendum's "~1.68 rad accumulated wavefront".
Measured on the real traced chain (N=2048, nfc=8192, wf=4.0): exit wavefront vs
exact sphere(R_out) = **1.333 rad rms (PV 6.42 rad)**.

### 6.3 The deciding 2×2 factorial (ideal-element chain, N=2048, NFC=8192, WF=4.0)

| chain carrier convention | element amplitude | exit rms vs sphere(R_out) | FWHM | EE3 | EE6 | window |
|---|---|---|---|---|---|---|
| parabola (library today) | frozen (= §1 control) | 0.659 rad | 3.65 µm | 41.0% | 63.4% | 94.8% |
| parabola | remapped by m (= §5 fix) | 0.598 rad | 7.85 µm | 20.7% | **49.5%** | 87.7% |
| **exact sphere** | frozen | **0.000 rad** | 3.55 µm | 83.7% | **99.7%** | 100.0% |
| **exact sphere** | remapped by m | **0.000 rad** | 3.55 µm | 86.9% | **100.0%** | 100.0% |

(Last row's final-stage w = 1.1743 mm vs exact q-trace 1.1749 mm.)  Consequences:
the convention fix alone is worth ~36 EE6 points; the §5 amplitude fix applied FIRST
**regresses** EE6 63.4 → 49.5 (it correctly deflates the artifactual NA 0.45 → 0.152,
un-masking the convention error the over-filled NA was hiding).  Sequencing is
load-bearing: convention first, amplitude second, and expect the amplitude fix to
look harmful if ever benchmarked in isolation.

### 6.4 Further corrections adopted from the review

- **§1 over-claim**: the stigmatic control established dx-STABILITY, not accuracy.
  Its axis-B absolute numbers (65.2% / 84.0%) are additionally confounded by
  `window_factor=2.0` — a self-inflicted 1·w crop applied twice (retrace + readout;
  the readout's own docstring requires wf=7 for <1e-6 truncation).  At WF=4/NFC=8192
  the same control reads FWHM 3.65 µm / EE6 63.4% / window 94.8%.  The §1(B) audit
  matrix (incl. the nfc=8192 "collapse" row) carries the same 1·w-crop confound.
- **Target metric unit error (all prior audits)**: Zemax POP's 2.736 µm is the waist
  RADIUS (1/e²).  The correct comparison targets at the readout plane are
  **FWHM 3.223 µm, EE3 ≈ 91%, EE6 ≈ 100%** (true exit NA 0.152; Gaussian waist
  λ|R|/(πw) = 2.7373 µm matches POP to 4 digits).  The real miss is 3.950/3.223 =
  1.23×, not 1.44×.
- **§5's amplitude design superseded**: the library already ships the general,
  carrier-alias-immune amplitude — `amplitude_model='ray_density'`
  (`_lens_traced.py:1930-1963`, N12/P11: `|E_in(x_in)|/√|det J|` from the traced
  entrance→exit Jacobian; non-paraxial, handles decenter/astigmatism, caustic-aware).
  The proposed scalar `|env|(r/m)/|m|` remap is a weaker re-invention (singular at
  internal conjugate images, scalar-symmetric only, uniform-magnification
  approximation) and additionally double-counts expansion on the fine retrace leg,
  where `r_al/w ≈ 2.3` means the wave pass ALREADY delivers ~82% of the true
  compression (measured: 1.581 mm vs 3.51 frozen / 1.175 true).  Test `ray_density`
  before writing any new amplitude code.
- V2's 0.001 rad support for the "remapped residual phase transport" was vacuous
  (the test input had `angle(env) ≡ 0`); in the chain the residual carries ~1.3 rad
  rms and the transport term is untested.

### 6.5 Revised P0.3 plan (supersedes §5)

1. **Carrier-convention reconciliation at the element boundary** (the P0.3 core):
   convert parabola↔sphere explicitly where the chain reconstructs for / re-envelopes
   after `apply_real_lens_traced` (`carrier.py:2206/2212` and the fine-leg `1948`),
   i.e. multiply by `exp(±i·k·(S(R) − r²/2R))` **band-limited to
   r < r_safe = (R³λ/dx)^{1/3}** with a smooth taper (the difference term itself
   aliases beyond r_safe; a whole-grid swap measurably breaks the real chain —
   window 77.5% → 7.1% — because the guard band's junk phase scatters; r_safe/w =
   10.5 (launch), 3.6 (S3-S4), 5.1 (S23-S24), 2.6 (S25-S27 coarse), 5.5 (fine), so
   every group's beam is covered).  The envelope handed to Sziklas-Siegman transport
   stays parabola-referenced (that approximation is what the transport is built on).
   Acceptance: real-chain exit residual 1.333 rad rms → ≪0.1; EE6 up from 68.9%.
2. **Amplitude second**: `traced_kwargs={'amplitude_model': 'ray_density'}` (existing
   feature) on the convention-fixed chain; acceptance: final-stage w → ~1.175 mm,
   FWHM → 3.223 µm, EE3 → ~91%.  Expect it to look harmful before step 1 lands
   (§6.3) — that is the predicted signature, not a regression.
3. **Then** re-run both convergence axes (the §1 sweeps) for the plateau gate, with
   WF ≥ 4 (avoid the 1·w-crop confound) and the corrected targets (§6.4).
4. Defect A (`preserve_input_phase` pair, 0.015 rad/group at chain pitch) and the
   ~1-coarse-pixel diagonal focus walk: re-measure AFTER 1+2; only chase if they
   survive.

### 6.6 EXECUTION RESULT (same day): §6.5 step 1 REFUTED on the real chain;
###     step 2 (`ray_density`) CONFIRMED as the working fix

All §6.5 candidates were executed on the real 121 chain (N=2048, NFC=8192, WF=4.0,
wide ±51.2 µm readout so the diagonal focus walk cannot escape the window):

| config | peak offset | FWHM | EE3 | EE6 | window |
|---|---|---|---|---|---|
| library baseline | (−13.7, −13.7) µm | 3.95 µm | 54.8% | 68.9% | 85.6% |
| **`amplitude_model='ray_density'` alone** | (−16.0, −16.0) | 4.55 µm | 56.0% | **85.8%** | **96.8%** |
| band-limited sphere conversion alone | (−31.5, −31.5) | 12.65 µm | 7.1% | **22.3%** | 74.0% |
| conversion + ray_density | (−27.4, −27.4) | 10.75 µm | 11.2% | 33.4% | 90.4% |

(The taper worked as designed — stage traces identical to the whole-grid swap to 4
digits, i.e. the guard band truly carries nothing — so the breakage is in-band and
intrinsic, not an aliasing artifact.  The conversion runs' spot is walked AND
genuinely blurred; the narrow-window 7.1% was the walk clipping the readout corner.)

**Resolution — the real element is (approximately) INPUT-HONEST, the stub is not.**
`apply_real_lens_traced`'s exit is `E_analytic·exp(i(k·opl − φ_pw))`: the
input-dependent phase rides through `E_analytic = apply_real_lens(E_in)` (honest
transport of whatever phase the input carries), while the input-independent pair
replaces only the lens OPD.  Since `reconstruct∘envelope` is a pointwise identity,
an input-honest element sees NO net boundary-convention error across a hand-off —
§6.2's per-group r⁴ injection applies to the sphere-ASSUMING ideal stub (which
imposes `exp(ik(S_out−S_in))`), i.e. to the CONTROLS that measured it, not to the
real element.  Applying the boundary conversion to an honest element instead
INJECTS `(S−parab)(R_in) − (S−parab)(R_out)` ≈ −3.3 rad at r=w per group (S3-S4)
— quantitatively matching the observed collapse.  The §6.3 factorial remains valid
for what it is (a stub-chain result); its 36-point transfer to the real chain is
hereby REFUTED by direct measurement.

**What stands after this round:**
- `amplitude_model='ray_density'` (existing, opt-in, reachable via
  `traced_kwargs`) is the verified fix for the §4 frozen-amplitude defect:
  +16.9 EE6 points, window 96.8%, design beam trace restored at N=2048 (final
  group w = 1.193 mm vs design 1.175 mm) — dx-independent by construction.
- The real chain's remaining gap (EE6 85.8 vs ~100% target; FWHM 4.55 vs 3.223 µm;
  the ~16 µm diagonal walk) is now the open P0/P1 question — candidates: the
  element's internal phase-side dx effects (§3 defect A), the walk artifact, and
  the exit residual composition (§6.2's 1.333 rad measurement needs re-taking
  under `ray_density` before further attribution).
- The dx-convergence question moves to: is the `ray_density` chain FLAT across the
  axis-A N-sweep?  (Running at write time.)

### 6.7 CONVERGENCE CLOSED BY CONFIGURATION; the remaining gap is a converged
###     fidelity plateau (same day, continued)

Axis-A sweeps of the candidate configurations (extent-preserving N refinement,
pitch-preserving rays, NFC=8192, WF=4.0, wide ±51.2 µm readout):

| config | N=1024 | N=2048 | N=4096 | N=8192 |
|---|---|---|---|---|
| `ray_density` alone (EE6) | 89.1% | 85.8% | 60.8% | 56.7% |
| `ray_density` + `preserve_input_phase=False` (EE6) | 78.6% | **76.7%** | **76.7%** | **76.7%** |

- **The `rd+pip0` configuration is dx-flat TO THE DIGIT for N ≥ 2048** (FWHM 5.65 µm,
  EE3 46.6, EE12 85.4, window 97.8 — all three rows byte-identical), meeting the
  production-readiness P0 acceptance (EE6 stable < 2%, window ≈ stop losses,
  genuine plateau).  With both dx-dependent ingredients removed (amplitude from the
  ray Jacobian, phase from the ray fit alone) the traced exit is fully determined
  by the physical ray pitch, and the chain inherits the machinery's grid
  independence.  `ray_density` WITHOUT `pip0` remains non-convergent (the
  preserve-pair phase corruption grows with fine dx and its coarse-N contribution
  crosses BELOW the flat plateau by N=4096 — its +9 points at N=2048 are not real
  fidelity).
- **Residual-carry prototype (defect-A "proper fix" candidate): NULL RESULT.**
  Multiplying the exit by the de-chirped entrance residual phasor
  `exp(i·angle(E_in·e^{−ik·S(R_in)}))` (pointwise, no gradient extraction;
  identity-coordinate transport) changes nothing (EE6 76.2 vs 76.7) despite the
  carried content measuring ~1.8 rad rms/group (dominated by the legitimate
  sphere-vs-parabola skirt term).  So the plateau is NOT limited by dropped input
  phase at first order, and the preserve-pair's coarse-N advantage is confirmed
  artifact.
- **Open (the new P1 gap): the converged plateau sits at FWHM 5.65 µm / EE3 46.6 /
  EE6 76.7 vs the corrected targets 3.223 µm / 91% / ~100%, with a constant
  (−16.0, −16.0) µm diagonal peak walk** (dx-independent in the flat
  configuration; the stigmatic control has none — element-linked, odd-order).
  The clean-sphere per-group oracle shows only 0.001–0.012 rad rms in exactly this
  configuration, which cannot explain a 1.75× FWHM — so the error either (a)
  arises only under CHAIN conditions (parabola-referenced input + accumulated
  content + chain windows), (b) lives in the amplitude/Jacobian leg (the oracle
  gate is phase-only), or (c) is odd-order/2-D content the radial-cut metrics miss
  (the diagonal walk points this way).  Next diagnosis: per-group ORACLE AT CHAIN
  CONDITIONS with a complex-field overlap (Strehl-like) metric, groupwise walk/tilt
  measurement, and a ray-pitch sensitivity check of the ray-density Jacobian.

## 7. Session artifacts / open items (updated post-review)

- Axis-A sweep COMPLETE.  Traced (full final-leg settings, readout N_fine pinned
  16384): N=1024 → EE6 81.5 / window 87.0; N=2048 → 68.2 / 75.3; N=4096 → row
  INVALID (the traced chain's ~1-coarse-pixel diagonal focus walk reached the
  readout window corner: peak offset (−25.60, −25.60) µm — the window saw only the
  spot's skirt, 2.0%); N=8192 → 49.7 / 61.7.  The un-freezing crawl is visible
  directly in the stages (S3-S4 exit w = 4.996/5.002/5.167 mm at N=2048/4096/8192
  vs design 5.627 mm).  Stigmatic control on the SAME axis: N=2048/4096/8192
  IDENTICAL to the digit (FWHM 3.65 µm, EE6 63.2%, window 92.8%) — dx-stable, with
  the absolute level explained by §6.2's convention error (63.4% in the §6.3
  factorial at the same settings).
- **N=28672 extent-preserving traced run (post-F-A-fix) COMPLETE — the energy-gain
  bug is CLOSED and the frozen-amplitude mechanism is confirmed at chain level.**
  (dx0=0.0714 µm, rs=4, nfc=16384, wf=7.0, readout auto; 2.6 h chain time.)
  Result: FWHM 7.75 µm / EE3 21.6 / EE6 51.1 / EE12 70.2 / **window-total 88.9%**
  — vs the PRE-fix audit row's impossible 102.3/128.2/130.8%.  Energy is conserved
  at the F-A trigger condition (n_crop = 20924 > n_fine_cap); the production-
  readiness audit's hypothesized "second, distinct energy-gain mechanism" is
  REFUTED — F-A was the whole energy bug.  At this fully-carrier-resolved pitch
  (alias radius beyond the beam at every stage) the stage beam trace lands on the
  TRUE design values — S3-S4 exit w = 5.6277 mm (design 5.6265), S5-S7 6.357
  (design 6.35), final group 1.1820 mm (design 1.1749) — and the diagonal focus
  walk collapses to (−1.35, −1.35) µm from ~−21 µm at coarse N.  What remains at
  the resolved grid (EE6 51%, FWHM 7.75 vs target 3.223 µm) is the §6.2
  carrier-convention error class, exactly as the review's plan predicts.
- T4/T5 (readout end-to-end energy/convergence at NA 0.45) OOMed against the sweep;
  rerun pinned when the box is free.  Superseded in urgency by §6 (the readout was
  cleared by T3 + the stigmatic flatness; T4/T5 remain worth one clean pinned run).

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

### 6.8 The upsample LATTICE bug (walk root cause, FIXED) + through-focus
###     methodology + the remaining plateau's exclusion list

- **Diagonal-walk root cause FOUND and FIXED (commit `0a743a6`).**  The
  walk-hunter instrumentation localized the walk to the G8 fine-retrace call
  (exit centroid jumps to (−16, −16) µm in one step, pointing stays 0 — an
  amplitude-map SHIFT, not a tilt).  Cause: the coarse→fine upsample of the
  OPL / ray-density / valid maps used `coords = ii·Ns/N` (`Ns = ceil(N/sub)`),
  exact only when `ray_subsample | N`; otherwise a corner-anchored scale error
  displaces every map diagonally by `(N/2)·(Ns·sub−N)/N` pixels.  Verified by
  prediction (−6.100/−12.187/−14.467 µm measured vs −6.11/−12.22/−14.51
  predicted at sub=50/48/51, exactly 0 at divisor subs), fixed at all four
  sites (`ii/sub`), bit-identical for divisor subs (49/49 pinned tests),
  pinned by `tests/unit/test_niche_upsample_lattice_fix.py`.  The F-C
  fine-retrace rescale routinely produces non-divisor subs, so every
  exact-final-leg run since F-C carried this walk.
- **Through-focus scans are now mandatory methodology**: at NA 0.152 the
  focused Rayleigh range is ~18 µm and the chain's best focus sits +60-75 µm
  past the fixed MSoP readout plane, so at-plane numbers confound focus
  position with quality (the pre-fix lattice scale error shifted best focus
  TOWARD the plane — at-plane EE6 "dropped" 76.7→58.7 on fixing it while
  best-focus quality stayed).  All P1 comparisons from here use best-focus +
  at-plane pairs (`focus_scan` pattern).
- **New opt-in mode `preserve_input_phase='remap'`** (this session): the
  carrier-de-chirped input residual phase transported geometrically to the
  exit, sampled at each exit pixel's Newton-inverted entrance point (the
  ray-density pullback).  dx-independent by construction; no-double-count and
  carried-residual pins green (8/8 in the lattice-fix test file).  On the 121
  it is fidelity-neutral (best-focus EE6 74.6 vs 79.9 for `False`) — the
  discarded-pre-correction hypothesis for the plateau is REFUTED.
- **The remaining converged plateau** (best-focus FWHM ~5.15-5.35 µm / EE3
  ~50-56 / EE6 ~75-80 / defocus +60-75 µm vs targets 3.223 µm / 91 / ~100 at
  +6 µm) is now measured INDEPENDENT of: grid pitch (flat N=2048-8192), ray
  fit density (rs=1/2/4 identical), input-residual handling (False / 'remap'
  / entrance-coordinate carry all within ~5 EE6 pts), amplitude model
  (ray_density restores the design beam trace), the walk (fixed), and the
  machinery (stigmatic flat; unit-energy audits exact).  Remaining suspects,
  in order: (a) traced-OPL accuracy under CHAIN conditions (the clean-sphere
  per-group oracle reads 0.001-0.012 rad but the chain-level exit residual has
  not been re-measured post-fix in the converged config); (b) the paraxial
  inter-group envelope transport of the legitimate (S−parab) r⁴ content (±7
  rad on the last gap at w=3.6 mm, R=−24 mm); (c) the exact-readout's
  fine-leg re-trace conditions (n_fine_cap/window interplay at the retrace).
  Next: MEASURE-hook exit-wavefront residual of the converged config, then
  bisect element-vs-transport with a fixed-fine-grid tail run.

  **MEASURED (exit_residual probe, converged config, post-fix):** the field
  entering the exact readout carries, vs the exact sphere(R_out=−7.7124 mm):
  defocus d(1/R) = +0.975 m⁻¹ (≡ the observed +59 µm focus shift — the
  readout leg is exonerated; the shift is created at/before the last exit),
  **r⁴ = +3.113 rad at r = w** (r⁶ ≈ 0.000), defocus-removed rms 0.347 rad
  (PV 1.47), and the +x/+y cuts are IDENTICAL (fully symmetric — the walk
  fix holds).  The entire remaining P1 gap is this one symmetric
  (defocus + spherical) pair.  Next leg: a per-stage r⁴/defocus tracker on
  the converged chain (envelope-stage best-r²-removed radial fit per
  hand-off) to find which group/transport injects it; then compare that
  group's traced OPL against the meridional oracle AT CHAIN CONDITIONS.

- **FULL-TRAIN MERIDIONAL ORACLE (design truth, noDOE) — the reference
  stands and the mechanism is pinned.**  Tracing rays through ALL 23
  surfaces (DGRATING surfaces 9/11 flattened, exactly as the chain runs —
  note: the .zmx import DROPS the grating parameters with a warning, so the
  chain has always been the noDOE configuration) from the emitter waist to
  the last exit vertex: the TRUE noDOE exit wavefront vs the ABCD sphere is
  **defocus −2.9 µm, r⁴ −0.000 rad, rms 0.016 rad** — the glass train is
  corrected WITHOUT the DOE, so the POP-class focus target stands for this
  chain and the measured +3.11 rad / +59 µm is numerical error.  Combined
  with the clean per-group probes (every group ≤0.04 rad r⁴ vs its own
  oracle except S25-S27 at −0.277, at BOTH probe and exact retrace
  conditions, rs=16/50 identical), the composition arithmetic pins the
  mechanism: each group's `k·opl` exit phase carries only that group's OWN
  design contribution; the design corrects via the CARRIED inter-group
  content (individually large, transported sum ≈ 0), and
  `preserve_input_phase=False` discards the carried part at every hand-off
  — the exit then shows the negative of a large partial sum (+3.11 rad).
  The per-group oracles are blind to this by construction (they test own
  contributions only).  **Discriminator result:** under `'remap'` the exit
  residual is WORSE (+4.302 rad r⁴, +82 µm defocus) — the carry actively
  degrades.  Analysis: after every `k·opl`-form group exit, the parab-
  referenced ENVELOPE legitimately carries `(S_exact−parab)(R_out)` + the
  design content; that term is slow in the core but approaches the grid
  Nyquist at the beam skirt on the co-moving pitch (~1.2 rad/px at r≈2w at
  the last-group entrance), so BOTH the inter-group envelope transport of
  it and remap's phasor pullback (which samples it, junk included, wherever
  the entrance amplitude is non-negligible) degrade at exactly this order.
  **The remaining P1 work is therefore a robust carried-content
  representation across hand-offs**, with three candidate paths for the
  next session: (1) skirt-gate/roll-off the remap phasor at a
  beam-relative radius (carry the core content, identity beyond — the
  carried correction lives in the core); (2) reference the hand-off
  envelope to the EXACT sphere for the traced exits (making the carried
  content core-slow by construction) with the band-limited (S−parab)
  conversion applied ONLY between the exit and the next entrance (NOT the
  stub-style whole-chain swap §6.6 refuted — the pip=False exit really IS
  sphere-referenced, unlike the input-honest pip=True exit); or (3)
  carrier-fit upgrade: let the per-group carrier include the r⁴ term
  (an aspheric carrier eikonal), absorbing the dominant carried component
  into the reference that both the OPL bookkeeping and the launch already
  handle analytically.  Candidate (2) is the most principled: in the
  converged configuration each exit's phase is k·opl = S_exact(R_out) +
  own-design content EXACTLY, so an exact-sphere envelope reference at the
  exit leaves only design content in the envelope — removing the
  near-Nyquist (S−parab) skirt from the transport entirely.

  **Candidate (2) prototype result (same day, script-level):** the full
  stack (band-limited sphere hand-offs + ray_density + 'remap') flips the
  defocus sign (best focus ≤ −40 µm vs +60 µm for pip=False; at-plane EE6
  67.4, on-axis) but the best-focus EE6 stays in the same 75-80 band
  (77.7 at the scan edge).  Across ALL configurations tried (pip=False /
  remap / candidate-2), the defocus swings ±60 µm while the
  defocus-removed exit rms stays 0.35-0.48 rad and best-focus EE6 stays
  75-80 — the COMMON cap is that shared ~0.35 rad non-defocus residual,
  not the defocus bookkeeping.  The naive composition of candidate (2)
  over-corrects (likely double-counting the (S−parab) term between the
  conversion and the remap de-chirp — both reference S_exact at the same
  boundary).  NEXT SESSION: (i) fix the candidate-2 composition so the
  conversion and the remap share ONE reference hand-off (the conversion
  factor belongs on the TRANSPORT side only, or the remap's de-chirp
  should use the post-conversion field's actual reference); (ii) decompose
  the shared 0.35 rad residual (Zernike-order it: if it is coma-free
  r⁴-balanced content it is still carried-content bookkeeping; if it is
  higher-order/azimuthal it is a different mechanism); (iii) gate with the
  full-train oracle (rms 0.016 rad is the design floor).
- **Default-flip decision (recorded 2026-07-24):** the element-level defaults
  stay as they are — `amplitude_model='screen'` and
  `preserve_input_phase=True` are genuinely correct for non-carrier /
  well-sampled inputs (diffraction amplitude, tilt/DOE phase), and
  `ray_density` genuinely degrades at caustics/folds — but the CHAIN
  (`propagate_traced_carrier_chain`) always operates in the
  beyond-Nyquist-carrier regime where the wave-pair corrupts and the screen
  amplitude freezes, so the convergent configuration is not optional there.
  Once the plateau leg concludes and the P0/P1 gates pass, the P0.3
  packaging commit flips the CHAIN-level defaults to the validated
  configuration (R8 chain pins updated deliberately and documented;
  a legacy escape hatch retained for reproducing pre-flip runs).

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

## 8. THE PLATEAU IS CLOSED (2026-07-25, adversarial review round 2) — the gap
##    was the DISCARDED CARRIED CONTENT plus the PARABOLIC CARRIER CONVENTION,
##    and the two had to be fixed TOGETHER

Scripts: `scratchpad/review2/{common121,run_chain,dump_exit,amp_factorial,
train_oracle,pupil_strehl}.py` with logs beside them (env knobs
`AM/PIP/TAPER/NATIVE/PRECOMP/RN/RS/NFC/WF/NOUT/SCAN/DZ*/DUMP/TAG`).  All runs
N=2048 unless stated, NFC=8192, WF=4.0, readout ±51.2 µm, **through-focus
scanned** (−120…+100 µm, 10 µm steps; refined where needed).

### 8.1 The priority hypothesis (truncated-disc exit AMPLITUDE) is REFUTED

Measured on the field entering the exact readout in the converged `rd+pip0`
configuration (`dump_exit.py` → 8192² dump → `amp_factorial.py`):

| r/w | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.45 | 1.75 | 2.00 |
|---|---|---|---|---|---|---|---|---|
| measured abs(E)/abs(E0) | 0.9315 | 0.7644 | 0.5544 | 0.3573 | 0.2061 | 0.1251 | 0.0511 | 0.0224 |
| Gaussian(w) | 0.9340 | 0.7720 | 0.5644 | 0.3651 | 0.2090 | 0.1244 | 0.0474 | 0.0188 |
| ratio | 0.997 | 0.990 | 0.982 | 0.978 | 0.986 | 1.006 | 1.077 | 1.194 |

Super-Gaussian exponent **n = 1.971** (Gaussian = 2; a truncated disc / flat-top
would be much greater than 2).  Encircled power P(r<0.5w/1.0w/1.5w/2.0w) =
0.3985/0.8645/0.9875/0.9996 vs the Gaussian 0.3935/0.8647/0.9889/0.9997.  The
x-cut and the diagonal agree to 4 digits.  **The exit amplitude is the design
Gaussian to within 2 % inside 1.45w** — no clipping, no flat-topping, no
vignetting signature.  (The mild >1 ratios at 1.75–2w are a skirt 7–19 % ABOVE
Gaussian, i.e. the opposite of truncation, and carry 0.04 % of the power.)

The decisive amplitude/phase 2×2 factorial (same dump, same exact readout, each
row through-focus scanned):

| variant | at-plane FWHM / EE3 / EE6 | best-focus dz / FWHM / EE3 / EE6 |
|---|---|---|
| V1 A_meas × phi_meas (= the chain) | 7.150 / 26.5 / 58.7 | +60 / 5.150 / 55.6 / 79.7 |
| V2 A_meas × exact sphere | 3.450 / 87.6 / 99.8 | +2 / 3.450 / 89.2 / 99.8 |
| V3 A_gauss × phi_meas | 7.150 / 26.3 / 58.3 | +60 / 5.050 / 55.7 / 79.6 |
| V4 A_gauss × exact sphere | 3.550 / 87.6 / 99.8 | +4 / 3.450 / 90.3 / 99.8 |

**100 % of the plateau is PHASE.**  Swapping the measured amplitude for a perfect
Gaussian changes nothing (V1→V3: 0.1 µm FWHM, 0.1 EE6 point); swapping the
measured phase for a perfect sphere recovers the target (V2).  V4 additionally
fixes the **achievable ceiling through this readout**: FWHM 3.45–3.55 µm,
EE3 90.3 %, EE6 99.8 % — so the paraxial-Gaussian target FWHM 3.223 µm is ~7 %
optimistic for the exact-sphere readout of a 4w-windowed Gaussian, while the
EE3 ≈ 91 % target is right.  Re-running V4 at `window_factor=10` (no crop at
all) recovers part of it — best-focus FWHM **3.350 µm**, EE3 90.7, EE6 99.8 — so
~0.1 µm of the offset is the wf=4 crop at 2w, and the remaining ~0.13 µm is the
exit w (1.187 vs the design 1.175 mm), the non-paraxial sphere focus, and the
azimuthal-profile FWHM estimator on a 0.05 µm grid.  **The ideal-field ceiling,
not 3.223 µm, is the correct gate for a chain metric taken this way.**

### 8.2 Why 0.347 rad DID explain a 1.6× FWHM — the rms figure was a cut artifact

`pupil_strehl.py` computes the wrap-free, amplitude-weighted pupil metric
`S = max_c |sum A e^{i(phi − c r^2)}|^2 / (sum A)^2` (the best-defocus coherent
Strehl).  For `rd+pip0`: **Strehl 0.421**, rms_eff 0.931 rad, amplitude-weighted
rms **1.032 rad** — 3× the 0.347 rad the campaign had been quoting, because 0.347
came from an *unweighted* fit along the +x *cut* over r<1.1w, whereas the pupil
integral weights by amplitude over AREA (and an r⁴ term grows as r⁴).  The
defocus-removed, amplitude-weighted mean residual per annulus is
+0.740 / +0.299 / −0.299 / −0.493 / +0.536 rad at r/w = 0–0.25 / 0.25–0.5 /
0.5–0.75 / 0.75–1.0 / 1.0–1.25 (97 % of the power) — textbook balanced spherical
aberration of ≈1 rad.  Strehl 0.42 is fully consistent with FWHM 5.15 µm and
EE3 55.6 %.  **No unexplained mechanism remained; the paradox was a metric
error.**

### 8.3 Localization — the ELEMENT IS EXACT; the CHAIN discards the design's
###   carried content, and the exit shows minus what it dropped

`dump_exit.py` prints a per-hand-off wavefront table (every
`carrier_referenced_reconstruct` = group entrance, every
`carrier_referenced_envelope` = traced exit, each fitted against the EXACT sphere
of that boundary's carrier).  `train_oracle.py` traces the design (all 23
surfaces, DGRATING faces flat = the as-run noDOE configuration) from the emitter
waist as a spherical fan and reports the TRUE cumulative residual `C` at exactly
those planes, plus each group's own contribution.

| group | chain exit r⁴@w (rd+pip0) | ORACLE own contribution | oracle TRUE cumulative C at that exit |
|---|---|---|---|
| S3-S4 | −1.305 (rms 0.146) | −1.306 (0.146) | −1.306 |
| S5-S7 | +1.233 (0.138) | +1.236 (0.138) | −0.072 |
| S14-S15 | −0.000 (0.000) | +0.000 (0.000) | −0.072 |
| S16-S17 | +0.000 (0.000) | +0.000 (0.000) | −0.072 |
| S18-S20 | +4.702 (0.527) | +4.731 (0.528) | +4.660 |
| S21-S22 | −0.901 (0.101) | −0.889 (0.099) | +3.747 |
| S23-S24 | −7.020 (0.786) | −6.998 (0.778) | −3.272 |
| S25-S27 (readout entrance) | **+3.113** (0.347) | — | **−0.000 (rms 0.018 = design floor)** |

Three measured facts:

1. **Every group's own traced contribution matches the ray oracle to ≤0.03 rad**
   — at CHAIN conditions, including the fine-retrace final leg.  The traced
   element, the ray-density amplitude, the retrace and the exact readout are all
   exonerated.  Nothing "injects" the +3.1 rad.
2. **`preserve_input_phase=False` wipes the incoming wavefront at every
   hand-off.**  Direct proof: S14-S15 / S16-S17 receive an entrance residual of
   −0.071 rad r⁴ and emit an EXACTLY spherical exit (r⁴ −0.000, rms 0.000); and
   forcing the entrance field to be a perfect sphere (the band-limited
   conversion, §8.4 cell A) moves the exits by ≤0.03 rad.  The element launches
   rays normal to the exact sphere of `carrier=R_in` and writes `k·opl`, so its
   exit is a function of the prescription and `R_in` ONLY.
3. Consequently the chain's exit carries **only the last group's own
   contribution** — and because the design distributes its correction, what the
   last hand-off dropped is exactly what the exit shows with the opposite sign:
   the discarded content at the S25-S27 entrance is `C = −3.273 rad r⁴,
   d(1/R) = −0.145 1/m, rms 0.363` at w = 3.128 mm, which maps to the exit pupil
   (m = 2.63) as +3.273 rad r⁴ and +1.007 1/m — measured exit **+3.113 rad /
   +0.975 1/m / rms 0.347**, agreeing to ~5 %.

**So the +3.1 rad r⁴ and the +59 µm defocus enter NOWHERE.  They are the design's
own correction, appearing with the wrong sign because the chain throws the
accumulated wavefront away at each element boundary.**

A second, independent defect sits at the other end of the chain: the carrier is
the paraxial PARABOLA, so at the first group vertex the reconstructed field
carries **+3.362 rad of r⁴** relative to the physical (spherical) diverging wave
— analytically `+k w⁴/(8R³)` = +3.39 rad at w = 4.994 mm, R = 47.907 mm, while
the oracle's `C` there is 0.000.  This is the Fresnel error of the first carrier
leg (emitter NA 0.104 over 45.9 mm).  It is **invisible under `pip=False`** (the
element re-imposes the sphere) but is **carried straight to the exit the moment
the carry is switched on**.  That is why every single-variable attempt plateaued,
and it is the correct explanation of §6.8's `'remap'` discriminator (+4.302 rad):
remap works — and it faithfully carried the parabola's artifact.

### 8.4 The composition matrix, completed — and the candidate-2 FALSE NEGATIVE

| # | hand-off reference | amplitude | preserve_input_phase | exit r⁴@w / cut rms | pupil Strehl | at-plane FWHM/EE3/EE6 | best-focus dz / FWHM/EE3/EE6 |
|---|---|---|---|---|---|---|---|
| A | sphere (band-limited) | ray_density | False | +3.086 / 0.345 | — | 7.150 / 26.5 / 58.7 | +60 / 5.150 / 55.7 / 79.8 |
| B | parabola (library) | ray_density | False | +3.113 / 0.347 | 0.421 | 7.150 / 26.5 / 58.7 | +60 / 5.150 / 55.6 / 79.7 |
| C | parabola + crude launch pre-comp | ray_density | False | +0.000 / 0.206 | — | 7.750 / 27.8 / 64.3 | +60 / 4.750 / 65.4 / 88.7 |
| D | **sphere (band-limited)** | **ray_density** | **remap** | **−0.128 / 0.015** | **0.911** | **3.650 / 87.1 / 99.3** | **+10 / 3.550 / 88.4 / 99.3** |
| E | parabola (library) | ray_density | remap | +4.302 / 0.479 | 0.349 | 7.850 / 20.7 / 49.7 | +70 / 5.450 / 47.8 / 74.5 |
| F | sphere | screen | remap | rejected by validation (`remap` requires `ray_density`) | | | |
| — | ideal reference V4 (Gaussian × exact sphere, same readout) | | | 0 / 0 | 1.000 | 3.550 / 87.6 / 99.8 | +4 / 3.450 / 90.3 / 99.8 |

* **Cell A is a measured NO-OP** (the missing cell this review was asked to
  fill): with `pip=False` the conversion makes the entrance field a perfect
  sphere — a 3.36 rad change to the input — and the exits move by ≤0.03 rad, the
  final metrics not at all.  Every "conversion alone" experiment in this campaign
  was therefore *bound* to read null; §6.6's stronger inference that the
  conversion is *harmful* came from the `pip=True` default, i.e. a different
  mechanism (the input-honest wave pair).
* **Cell C (crude launch pre-compensation) is a REFUTED red herring.**  It looks
  like a win (+9 EE6 points, exit r⁴ → 0.000) but it works by *shrinking the
  beam*: the pre-compensation phase is beyond Nyquist in the skirt, so the
  second-moment w at the first vertex drops 4.994 → 4.400 mm and the exit w
  1.189 → 1.041 mm; the r⁴ aberration over the illuminated pupil then falls as
  w⁴ while the diffraction spot grows.  An apodization artifact, not a
  correction.  Recorded so nobody re-chases it.
* **The §6.8 candidate-2 refutation was a HARNESS BUG.**
  `scratchpad/p0_sweep/convention_taper_proto.py` only honours `PIP=0`
  (`preserve_input_phase=False`) — there is no `'remap'` branch — so the
  "conversion + ray_density + remap" run actually executed with the library
  DEFAULT `preserve_input_phase=True`, the wave-pair path §6.6 had already shown
  collapses under the conversion.  Candidate 2 was never tested.  Re-run properly
  (cell D) it is the fix.  (Its "best-focus EE6 77.7 at the scan edge" was also
  taken at the edge of a −40 µm scan, as §6.8 itself flagged.)

### 8.5 THE FIX (library, opt-in, gated) — `carrier_reference='sphere'`

`propagate_traced_carrier_chain(carrier_reference='sphere')` band-limits the
paraxial parabola out of every element hand-off: `_sphere_parab_conversion()`
multiplies each reconstruction by `exp(+i k (S(R) − r²/2R) · T(r))` before the
element sees it, and by the inverse after the traced exit, with a cos² taper
ending at `r_safe = (|R|³ λ / dx)^{1/3}` (the radius beyond which the difference
term itself aliases).  The stored envelope is then the physical wavefront
RESIDUAL vs the exact sphere — the carried content — and
`preserve_input_phase='remap'` transports it geometrically through each group.
The three options are only meaningful together.

Gate results (design-121, all measured this session):

| gate | requirement | measured | verdict |
|---|---|---|---|
| (a) exit residual | approach the 0.016–0.018 rad full-train-oracle design floor | **rms 0.015 rad** (cut), r⁴ **−0.128**, defocus d(1/R) = −0.022 1/m ⇒ best focus **+1.3 µm** from the plane (oracle −2.9 µm); the per-hand-off table matches the oracle's TRUE cumulative residual at ALL 8 boundaries (−1.305/−0.071/−0.071/−0.071/+4.598/+3.703/−3.279/−0.128 vs oracle −1.306/−0.072/−0.072/−0.072/+4.660/+3.747/−3.272/−0.000) | **PASS** |
| (b) focal metrics | best-focus FWHM ≲ 3.5 µm, EE3 ≳ 85 %, best focus within ±10 µm | **FWHM 3.550 µm** (ideal-field ceiling through this readout 3.45–3.55), **EE3 88.4 %** (peak 89.6 % at +5 µm; ceiling 90.3), **EE6 99.3 %** (ceiling 99.8), best focus **+5…+10 µm**, on-axis (walk 0.00 µm), window-total 99.4 % | **PASS** — FWHM equals the achievable ceiling (the 3.223 µm target is the paraxial-Gaussian estimate) |
| (c) dx-flatness | flat at N=2048 vs 4096 | exit rms 0.016/0.015/0.015 and r⁴ −0.128/−0.128/−0.129 at **N = 1024/2048/4096** (pitch-preserving rays rs = 2/4/8); best-focus FWHM 3.650/3.550/3.550, EE3 87.8/88.4/88.4, EE6 **99.1/99.3/99.3** | **PASS** |
| (d) byte-identical defaults | the pinned suites pass | `test_niche_{r6,r7,r8,r9_highna,r9_dx_scaling,e4,upsample_lattice}` **57 passed**; `test_carrier_referenced` + h3/h6/k2/p1 **46 passed, 3 skipped (no cupy)**; new `test_niche_s8_sphere_carrier_reference.py` **11 passed**.  Default `carrier_reference='parabola'` is byte-identical by construction (every new path is gated) and pinned by an explicit `array_equal` test | **PASS** |

Equivalence check: the library implementation reproduces the monkeypatched
prototype exactly (`NATIVE=1`: exit r⁴ −0.128 / rms 0.015 / dz +1.3 µm; at-plane
3.650 / 87.1 / 99.3; best focus +10 µm → 3.550 / 88.4 / 99.3 — identical to
cell D).

Working-tree diff (NOT committed): `lumenairy/propagators/carrier.py`
(`_sphere_parab_conversion`, `_fine_trace_group_exit(sphere_reference=)`,
`propagate_traced_carrier_chain(carrier_reference=)` + docs),
`tests/unit/test_niche_s8_sphere_carrier_reference.py` (new, 11 pins),
`CHANGELOG.md`, and this section.

**Recommended chain configuration for a carrier-regime traced chain** (and the
candidate for the §6.8 default flip):

    res = propagate_traced_carrier_chain(
        env0, groups, wavelength, dx, r_in=R1,
        carrier_reference='sphere',
        traced_kwargs={'amplitude_model': 'ray_density',
                       'preserve_input_phase': 'remap'},
        final_distance=z, focus_readout=fr)

### 8.6 What remains (P2-class; none of it blocks design-121)

* **The residual 9 % of Strehl sits beyond r > 1.5w** (1.2 % of the power), where
  the annulus rms is 0.965 rad against ≤0.16 rad everywhere inside 1.25w: that is
  the band-limit taper's mixed-convention skirt plus the ray-fit domain edge.  A
  `RuntimeWarning` now fires when `r_safe < 2w`; refining dx is the lever.
* The inter-group Sziklas-Siegman leg is still **paraxial**, so under `'sphere'`
  the `(S − parabola)` term rides inside the transported envelope (~7 rad at r=w
  on the final 121 gap).  Verified against the oracle hand-off by hand-off
  (≤0.01 rad), but it is an approximation with a validity envelope; an exact
  sphere-referenced gap transport (the existing exact-readout machinery applied
  to a gap) is the principled generalization, and the natural home for a
  high-NA-gap guard.
* The **default flip** (§6.8) is now unblocked: the validated triple is measured,
  dx-flat and pinned.  Flipping the CHAIN defaults changes the R8 chain pins
  deliberately and needs its own commit.
* Untouched by this session: the aperture:beam cliff guard, the memory-bounded
  mode, and the P2 design battery (production-readiness audit §4/§5).

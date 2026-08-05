# Exact sphere-referenced gap transport — specification

Consumer-authored spec, written against **v5.32.1 + niches D8/D9**. Requested
because the inter-group transport is the last *structurally* paraxial step in
the carrier chain, and because its validation is **single-design**.

Companion to `APPROXIMATION_AUDIT_TRACED_2026_07_30.md` (which priced the
approximation), `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`, and
`ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md`.

---

## 1. The ask, in one sentence

Replace the paraxial Sziklas-Siegman envelope transport between groups with a
transport that is exact with respect to the **reference sphere**, WITHOUT
requiring the carrier to be sampled — i.e. keep the chain's small grids and
remove its last frame-level approximation.

---

## 2. What the gap does today, and which part is paraxial

`_carrier_step_fast` (`propagators/carrier.py`) implements the Sziklas-Siegman
step: for a non-crossing leg (`m = R_out/R > 0`) the converging/diverging beam
is mapped onto a collimated-equivalent problem by a **co-moving coordinate
scaling**, and the envelope is then propagated by a **Fresnel** transfer
function over a reduced distance.

Two distinct approximations are stacked here, and they are usually conflated:

| # | approximation | status |
|---|---|---|
| A | the **kernel** is Fresnel, not `exp(ikz sqrt(1-(lambda f)^2))` | MEASURED NULL — see §3 |
| B | the **frame** (Sziklas-Siegman scaling) is derived from the PARAXIAL WAVE EQUATION | **structural, unpriced in general** |
| C | the anisotropic tilt stretch `z(1-M^2)/N^3` is **unimplemented** (transport uses `z`) | priced on 121 only |

**The important point: fixing A does not fix B.** Substituting the exact
kernel gives an exact propagator riding an approximate frame. Any spec that
stops at "use the exact transfer function" has not addressed the actual
approximation.

Note also what is ALREADY exact and must not be re-litigated: under
`carrier_reference='sphere'` (niche C9) the sphere<->parabola conversion at the
HAND-OFF is exact, and the dropped quartic cancels analytically (verified to
2.2e-16; 0.000 EE points across NA 0.35-0.45, against -20 to -33 points under
the legacy `'parabola'` path). The residual is in the transport, not the
hand-off.

---

## 3. Why "just use the exact kernel" is not the answer

Already measured (`approx_leg_budget_121.py`, S2): substituting
`exp(ikz sqrt(1-(lambda f)^2))` for the Fresnel kernel on the same reduced leg
gives **relL2 3.56e-07 end to end, +0.0000 EE3 points**, and the closed-form
dropped term is **9.4e-07 waves rms** at the worst coarse leg (5.8e-05 summed
as band maxima). The unimplemented anisotropic stretch (C) prices at
**7.1e-05 waves rms** worst leg, 2.2e-03 summed.

The reason it is null is structural and worth stating, because it is also the
reason the result does **not** generalise: once the carrier is divided out, the
envelope's angular content is small BY CONSTRUCTION, so the Fresnel and exact
kernels agree to ~7 digits. The paraxial kernel is not accurate at NA 0.4 — it
is accurate on *what is left after the NA has been removed analytically*.

That argument holds only while the ENVELOPE stays low-angle. §4 is about when
it does not.

---

## 4. The generality problem (the reason this spec exists)

Every number in §3 is design 121, one wavelength, six coarse legs. The chain's
validation base is narrow in ways that matter here:

* **Single design, single wavelength.** The C10 residual-eikonal degree and the
  C9 taper removal are both recorded as measured on ONE design at ONE
  wavelength, with no second design run. The gap budget inherits that.
* **121's coarse legs are benign.** Mid-chain the carrier is near-collimated
  (`R ~ 7e5 mm`), so the envelope's own angular content on those legs is tiny.
  A relay with SHORT conjugates, a fast intermediate image, or a large
  intermediate NA puts real angular content on the envelope, which is exactly
  the regime where the paraxial FRAME (B) — not the kernel (A) — degrades.
* **The existing guard is sag-based, not frame-based.** `on_gap_paraxial`
  (niche C3, `gap_sag_tol` default 0.30 rad, gap-NA 0.60) trips on the DROPPED
  QUARTIC. Design 121 is silent on every shipping leg with 4.08x margin. But
  the quartic is the *hand-off* term, and under `'sphere'` it cancels — so the
  guard is watching the axis that C9 already fixed, not the Sziklas-Siegman
  frame error. **A design can be well inside `gap_sag_tol` and still have a
  frame that is invalid.** No guard covers B.
* **The roadmap already flagged the hole**: "there is **no high-NA-gap
  guard**" and "an exact sphere-referenced gap transport is the principled
  generalisation and is not implemented."

**Consumer position:** the chain is currently trustworthy on 121-like designs
(long, near-collimated intermediate legs) and *unquantified* elsewhere. That is
a generality gap, not a correctness claim against 121.

---

## 5. What "exact with respect to the sphere" should mean

The field is carried as `E = env * exp(i k S(r; R))` with `S` the EXACT sphere
`sign(R)(sqrt(r^2+R^2)-|R|)`. Exact transport over `z` must map
`(env, R) -> (env', R+z)` such that the reconstructed full field equals the
exact free-space propagation of the reconstructed input field — **without ever
forming or sampling `exp(i k S)` on the grid.**

That last clause is the whole constraint. Forming the full field forces
`lambda/(2 NA)` sampling across the aperture (the ~28k-class fixed grid), at
which point the carrier chain has no purpose — that IS the fixed-grid ASM run.
So the transport must act on the envelope while being exact about the sphere.

### Candidate mechanisms

**(M1) Direction-cosine (Debye/Richards-Wolf) transport — RECOMMENDED.**
Between two concentric reference spheres, free-space propagation is DIAGONAL in
direction cosines `(s_x, s_y)`: each plane-wave component acquires
`exp(i k z sqrt(1-s_x^2-s_y^2))`, exactly, with no paraxial step anywhere. The
algorithm is: envelope on the input sphere -> direction-cosine space -> exact
diagonal phase -> back onto the output sphere. The sphere-to-plane projection
carries the obliquity factor `1/sqrt(1-s_x^2-s_y^2)` (and its Jacobian), which
is precisely the term the paraxial frame drops.
*Cost:* the projection is non-uniform, so it needs an NUFFT or a resample; the
diagonal phase is free. *Why recommended:* it is exact by construction, it is
the same framework the library already points at for the rigorous final leg
(Richards-Wolf), and its error sources are sampling ones that can be bounded.

**(M2) Exact-eikonal scaled frame.** Keep the co-moving scaling but derive it
from the exact eikonal rather than the paraxial wave equation, i.e. the
generalisation of Sziklas-Siegman with `sqrt` retained. *Attraction:* minimal
disturbance to the existing code path. *Risk:* the closed form may not exist in
a separable/FFT-able shape; needs derivation before it can be costed. **Do not
schedule M2 without first showing the transform closes.**

**(M3) Full-field exact ASM on the gap.** Rejected — see the constraint above;
it is the fixed-grid method, not a chain transport.

---

## 6. Implementation plan (staged, each stage independently verifiable)

**Stage 0 — instrument the existing error (no behaviour change).**
Add a diagnostic that reports, per coarse leg, the ENVELOPE's angular spread
after carrier removal and the implied frame error. This is the missing
observable: today nothing measures B. Ship it warn-only. *Deliverable:* the
frame error on 121's six legs, plus at least two contrasting designs.

**Stage 1 — a frame-validity guard (`on_gap_frame`).**
Refuse/warn when the envelope's residual angular spread exceeds the
Sziklas-Siegman frame's validity envelope, independently of `gap_sag_tol`.
This closes the "no high-NA-gap guard" hole even before M1 exists, and it is
the item with the best risk/benefit ratio in this document.

**Stage 2 — M1 behind a flag, default OFF.**
`gap_transport='sziklas'` (default, bit-identical) | `'exact_sphere'`.
Byte-identity on the default path is non-negotiable and must be pinned with
`np.array_equal`, exactly as niches D8/D9 were.

**Stage 3 — the equivalence oracle (the acceptance gate).**
On a geometry where BOTH are valid (a long, near-collimated leg), M1 must
reproduce the Sziklas-Siegman answer to a stated tolerance. Then on a
geometry where only M1 is valid (short conjugate / fast intermediate),
M1 must reproduce an INDEPENDENT reference — a full-field exact-ASM run on a
Nyquist-sampled grid — while Sziklas-Siegman visibly departs. **The second
half is the point: a test only M1 passes is what proves the frame error is
real rather than asserted.**

**Stage 4 — default flip**, only after Stage 3 passes on >= 3 designs.

---

## 7. Acceptance criteria — explicitly NOT 121-only

This is the part that must not be relaxed. The current chain's weakness is
that its evidence base is one design; repeating that here would reproduce the
problem this spec exists to fix.

1. **>= 3 designs**, spanning: (a) 121-like long near-collimated legs;
   (b) a short-conjugate relay with a fast intermediate image; (c) a
   high-intermediate-NA gap. At least one must be a design where
   Sziklas-Siegman is expected to FAIL — a spec that cannot name such a design
   has not established that the frame error exists.
2. **>= 2 wavelengths**, since every existing traced acceptance is monochromatic.
3. **An absolute reference, not a self-comparison.** Full-field exact ASM on a
   Nyquist grid, or the existing skew-ray + Debye oracle. Chain-vs-chain
   agreement proves nothing about the frame.
4. **A negative control**: a configuration where the two transports must
   visibly disagree, with the disagreement measured. Without it the tolerance
   is vacuous (this is the discipline niche D9's oracle used, and it is why
   that result is credible).
5. **Cost stated honestly** — wall time and peak RAM per leg, both transports,
   at the same accuracy.

---

## 8. Cost note, and the comparison that motivates this

The consumer-side observation that prompted this spec: **the traced chain is
currently slower than the fixed-grid ASM run it replaced, while being less
mathematically exact.**

* Fixed-grid ASM at N=28672 propagates the FULL multiplexed field — all 32 DOE
  orders at once — in a single pass (~2.5 h historically for the 8-group chain).
* The traced chain cannot represent a multi-valued field at all, so the same
  study needs **32 independent congruence runs**; at N=8192 with 2 workers that
  is ~9-10 h, and each run carries approximations A/B/C that ASM does not.

The traced chain's justification is physical, not numerical: split-step BPM
broadens at F/1.6 (measured ~1.7-2x too wide, focus ~170 um past geometric),
which is a *model* error that finer grids do not fix. But that justification is
about the LENS INTERIORS. It does not extend to the gaps, where ASM is exact
and cheap. **An honest hybrid would use exact transport in the gaps and reserve
the ray machinery for the glass** — which is precisely what M1 delivers.

---

## 9. Open questions for the implementer

1. Does M2 close in a separable/FFT-able form? If yes it may beat M1 on cost;
   if not, say so and drop it rather than approximating the generalisation.
2. M1's resampling: NUFFT vs oversampled-uniform + interpolation — which is
   cheaper at the chain's envelope bandwidths, and what does each cost in
   accuracy? Sampling error must not be traded silently for frame error.
3. Should the tilted (DOE-order) case share the transport, or does the tilted
   congruence need its own direction-cosine origin? Niche D9 moved the grid
   origin; the analogous question here is whether the direction-cosine origin
   should follow the chief ray.
4. Interaction with the exact final leg: if the gaps become exact, is the
   separate `_fine_trace_group_exit` retrace still required, or does the last
   gap subsume it? A merge would remove the single largest memory term in the
   chain (the 12288^2 fine grid).
5. Does `C` (the anisotropic tilt stretch) become moot under M1, or does it
   reappear as a direction-cosine-origin question?

# Wave Lens-Propagation Models Audit — 2026-07-02 (v5.18.1)

Scope: the three real-lens wave-propagation models in `lumenairy/elements/` —
`apply_real_lens_traced` (`_lens_traced.py`), `apply_real_lens`
(`_lens_real.py`, the "analytic" split-step model), and
`apply_real_lens_maslov` (`lenses_maslov.py`) — plus the thin-lens ABCD
reference. Field-driven audit: the findings below all surfaced while
diagnosing a real production regression (the Reverse_Symmetric_ASM TX Design
Study 119/120 "imaged spots are blurry, not sharp" report), and every claim is
backed by a measured end-to-end run on design 119 at N=16384, dx=1.35 µm,
complex64, focus offset 0, against `poc1-19.zmx`. Read-only audit — no library
code was changed by this pass; the fixes below are recommendations.

Trigger question from the sponsor: *the ray-traced lens "should give the
correct answer — it just determines the phase mask via ray tracing"; why is
that a large error, and can the traced model be generalized into one universal
model that beats the analytic one?*

---

## 1. Executive Summary

The three models are individually sound within their documented regimes, but
the suite has **no guard rails around those regimes** and **no working
general-purpose model** for the common "image a divergent source through a
thick relay" case. Concretely:

* **F1 (P1 — silent wrong answer).** `apply_real_lens_traced` launches every
  pixel ray **parallel to z** (`_lens_traced.py:1211-1213`), i.e. it assumes
  collimated input. For a *converging/diverging* field it computes the OPL of
  the wrong ray and returns a **plausible-but-wrong blurred result with no
  error and no reliable warning**. Measured: a single **on-axis** emitter
  images to **17.6 × 9.4 µm and astigmatic** — physically impossible for a
  rotationally-symmetric relay (the ideal is ~6.5 µm, symmetric). The docstring
  documents the limitation; nothing enforces it.

* **F2 (P1 — unusable at scale).** `apply_real_lens_maslov` — the phase-space
  model that *is* the correct generalization of the traced OPL — materialises an
  `(N_out² × M)` design matrix (`lenses_maslov.py:498`, M ≈ 210 poly terms).
  At N=16384 full output resolution that is **451 GB**. The only mitigation,
  `output_subsample`, coarsens the output grid, and for a high-NA system there
  is **no setting that both fits memory and satisfies Nyquist** (see §4). It is
  therefore not usable as a drop-in for production-grid, high-NA systems today.

* **F3 (P2 — API break).** `apply_real_lens_maslov`'s progress callback
  (`lenses_maslov.py:209-215`) is invoked as `progress(phase=…, fraction=…)`
  and `progress(phase, frac, dt)` — incompatible with the `(label, frac[,
  msg])` convention the traced/analytic models use. Passing a standard progress
  callback **crashes the call** with `TypeError` mid-lens.

* **F4 (P2 — misleading escape hatch).** `tilt_aware_rays=True` is advertised by
  the traced model's own runtime warning as the fix for tilted input, but for a
  **multi-source / multi-angle-per-pixel** field it produces
  **catastrophically wrong output** (measured: ~50× dimmer, structure
  destroyed) because a single ray direction cannot represent a superposition of
  plane waves. No guard distinguishes the single-beam case (where it helps)
  from the multi-beam case (where it must not be used).

* **F5 (P3 — no model-selection contract).** There is no universal lens model
  and no machine-checkable selection rule; the correct choice is
  regime-dependent and currently lives only in prose scattered across three
  docstrings. Users pick `traced` for "highest fidelity" and silently get the
  worst answer for divergent-source imaging.

**Headline recommendations, in order:** (1) add a **collimation guard** to
`apply_real_lens_traced` that measures the input field's angular spread and
warns/raises (or auto-delegates to `apply_real_lens`) when the collimated
assumption is violated — this alone would have turned a multi-day silent
regression into a one-line diagnostic (F1). (2) Give `apply_real_lens_maslov`
an **output-banded** evaluation path so it runs at full output resolution with
bounded memory (F2) — this is the concrete route to the "one universal model"
the sponsor asked for. (3) Fix the Maslov progress-callback signature (F3).
(4) Guard `tilt_aware_rays` against multi-source fields (F4). (5) Publish a
single **model-selection decision table** (F5).

---

## 2. The sponsor's question, answered precisely

**"The traced model just determines the lens phase mask by ray tracing — why is
that a big error?"**

It is not the *concept* that is wrong; it is one specific assumption in the
implementation. `apply_real_lens_traced` accumulates the OPL of a ray launched
**parallel to the z-axis** from each entrance pixel (`_lens_traced.py:1211`),
then applies `k0·OPL(x,y)` as an exit-plane phase screen. That phase screen is
the correct lens transmission phase **only for the ray that is actually
travelling parallel to z at that pixel** — i.e. collimated, normally-incident
light.

For the no-MLA TX designs the bare grating couplers diverge hard and strike each
relay lens at large angles (measured RMS wavefront tilt **0.11–0.21 rad** across
the seven lens groups). A straight-through ray and the actual tilted ray
traverse a thick lens along *different physical paths*; the OPL difference
scales as roughly `thickness · tan θ · (n−1)`, which for ~10 mm groups at
0.1–0.2 rad is **many waves** across the pupil. The single-screen model has no
way to represent that — hence the blur, and (because the error grows with pupil
radius) the spurious on-axis astigmatism.

This is exactly the case the traced docstring flags:

> *Limitations: Assumes the input field is approximately a collimated plane
> wave (each pixel ray launched parallel to z). For converging or tilted input,
> fall back to `apply_real_lens`.* (`_lens_traced.py:1209-1213`)

So the fix the field investigation converged on — switch to `apply_real_lens`
(analytic) — is the library's own documented guidance. The problem is that
nothing **enforces** it (F1).

**"Can the traced model be generalized into one universal model that beats
analytic?"**

Yes in principle — and lumenairy already ships that generalization as
`apply_real_lens_maslov`. The per-pixel traced OPL is the single-ray
stationary-phase limit of the Maslov canonical integral

    E(s₂) = ∫ E_in(s₁(s₂,v)) · exp(i·k·OPD(s₂,v)) · |det ∂s₁/∂v| dv ,

which integrates over pupil momentum `v` instead of assuming one ray per point.
It keeps the exact geometric OPD (the traced model's genuine advantage over the
analytic split-step's per-surface thin-element phase), handles arbitrary input
fields, and is caustic-safe. **The method is excellent** — measured canonical-map
fit RMS OPD residual **7×10⁻¹¹ waves** on the design-119 lenses. But the current
*implementation* cannot run it at the required grid size (F2/§4), so today it
does **not** beat analytic in practice for this workload.

---

## 3. Measured model comparison (design 119, N=16384, focus 0, v5.18.1)

Same source, DOE, prescription, grid, and focus in every row — only the lens
model changed. "Core FWHM" is the imaged-spot full-width at half-max in the
central DOE frame; the diffraction-limited target is 6.46 µm (|A|·MFD).

| Lens model | Spot core FWHM | Frames resolved? | Notes |
|---|---|---|---|
| **thin** (paraxial ABCD, aberration-free control) | 8.1 µm, symmetric | ✅ crisp | proves geometry/DOE/conjugate/source are all correct |
| **traced** (`apply_real_lens_traced`, tilt_aware=False) | ~17 µm, **astigmatic** even on-axis | ❌ fuzzy blobs | the regression; collimated-input assumption violated (F1) |
| **traced**, tilt_aware_rays=True | — (structure destroyed, ~50× dimmer) | ❌ chaos | multi-source field has no single local tilt (F4) |
| **analytic** (`apply_real_lens`, split-step) | **5.4 µm, symmetric** | ✅ crisp, 19.7 dB P/V | correct + fast (~1 min/lens); the delivered fix |
| **maslov** (`apply_real_lens_maslov`, output_subsample=8) | 15–16 µm, frames **absent** | ❌ speckle, 12 % power loss | output-coarsening destroyed the field (F2/§4) |

The thin-lens control is the load-bearing result: perfect paraxial optics with
the *identical* coherent 8×8 source and DOE produce sharp, well-separated
frames. That isolates the softness to the lens **model**, not the geometry, the
Dammann DOE, the imaging conjugate, or the (coherent, in-phase) source model
— the coherent source contributes only a modest pedestal, not the blur.

---

## 4. F2 detail — why Maslov cannot currently run at production scale

`apply_real_lens_maslov` evaluates its canonical integral on an
`N_out_coarse = N // output_subsample` grid and builds the Chebyshev design
matrix `G = np.empty((N_out_coarse², M))` (`lenses_maslov.py:498`), M ≈ 210 for
`poly_order=6`. `output_subsample` is the only lever on that allocation, and it
directly sets the output sampling `dx_coarse = output_subsample · dx`.

For this system the exit field carries angular content up to ~0.2 rad, so
Nyquist requires `dx_coarse < λ / (2 sin θ_max) ≈ 3.3 µm`:

| `output_subsample` | coarse dx | G-matrix | fits 137 GB box? | samples the field? |
|---|---|---|---|---|
| 1 | 1.4 µm | **451 GB** | ✗ | ✓ |
| 2 | 2.7 µm | **113 GB** | ✗ (+ ~27 GB baseline → OOM) | ✓ |
| 4 | 5.4 µm | 28 GB | ✓ | ✗ undersampled |
| 8 | 10.8 µm | 7 GB | ✓ | ✗ badly undersampled → the §3 speckle |

**There is no setting in the feasible region that also satisfies Nyquist.** The
configurations that sample correctly (≤2) do not fit; the ones that fit (≥4)
throw away the angular content that forms the spots and the DOE frames. Empirical
cost was also **~13.6 min/lens** at N=16384 (vs ~1 min/lens for analytic) — ~10×
slower even at the coarse settings.

**Recommended fix (the route to a universal model):** band the output grid.
Evaluate the canonical integral in output **row-bands** — build and consume `G`
one band of `N_out` rows at a time inside the existing `chunk_v2` loop
(`lenses_maslov.py:498-542`), accumulating `E_out_flat` per band — exactly the
pattern used for the row-band memory mode in `apply_real_lens_traced`
(`sag_chunk_rows`). Peak memory then drops from `O(N² · M)` to
`O(N_band · N · M)`, which makes `output_subsample=1` (full resolution,
Nyquist-correct) feasible at N=16384–32768. Combined with `chunk_v2`, that turns
Maslov from "unusable at scale" into the exact-OPD, arbitrary-field, caustic-safe
model that outclasses analytic — the sponsor's goal.

---

## 5. Findings ledger

### F1 (P1) — `apply_real_lens_traced` has no collimation guard; silently wrong on non-collimated input
* **Where:** `_lens_traced.py:1211-1213` (documented assumption); the model is
  applied unconditionally with no runtime check on input collimation.
* **Symptom:** blurred, on-axis-astigmatic spots with no error (§3). The
  existing tilt advisory (`_lens_traced.py:1936`, RMS of `grad(angle(E))/k0`)
  only fires on a net linear tilt and does **not** reliably flag a diverging
  (curved-wavefront) or multi-angle field; it also (mis)recommends
  `tilt_aware_rays=True`, which is wrong for multi-source fields (F4).
* **Fix:** before applying the model, estimate the input field's local angular
  spread (e.g. RMS of the local phase-gradient magnitude over the aperture, or
  the second moment of |FFT(E)|²) and (a) warn loudly, and/or (b) when a new
  `on_noncollimated='delegate'` option is set, transparently call
  `apply_real_lens`. Cheap, and converts a silent multi-day regression into a
  one-line diagnostic.

### F2 (P1) — `apply_real_lens_maslov` O(N²·M) design matrix; not scale-usable
* **Where:** `lenses_maslov.py:498`. See §4.
* **Fix:** output-row-banded evaluation (§4).

### F3 (P2) — Maslov progress-callback signature incompatible with the suite convention
* **Where:** `lenses_maslov.py:209-215` — calls `progress(phase=…, fraction=…)`
  then `progress(phase, float(frac), dt)`. The traced/analytic models call
  `progress(label, frac)` / `progress(label, frac, msg)`.
* **Symptom:** a caller wiring the standard progress callback into all three
  models gets `TypeError: _cb() got an unexpected keyword argument 'phase'` /
  `takes from 2 to 3 positional arguments but 4 were given`, mid-lens.
* **Fix:** normalise Maslov's progress protocol to the suite's
  `(label: str, frac: float, msg: str = '')` signature (or route it through a
  shared adapter).

### F4 (P2) — `tilt_aware_rays=True` unguarded against multi-source fields
* **Where:** `_lens_traced.py` tilt-aware branch; recommended by the warning at
  `:1936`.
* **Symptom:** for a coherent multi-emitter field the per-pixel local-tilt
  extraction returns a meaningless dominant direction; output structure is
  destroyed (§3). The warning actively steers users toward this.
* **Fix:** detect multi-modal local angular content (e.g. >1 significant peak in
  the windowed local spectrum) and refuse/warn; soften the `:1936` advisory so
  it does not recommend `tilt_aware_rays` for non-single-beam inputs — point at
  `apply_real_lens` / `apply_real_lens_maslov` instead.

### F5 (P3) — No unified model-selection contract
* **Symptom:** "use traced for highest fidelity" is a natural but wrong default
  for divergent-source imaging; the correct rule lives only in prose.
* **Fix:** publish one decision table (below) in the module docstring and the
  user guide, and — ideally — have each model self-check its validity domain
  (F1/F4) so the wrong choice is caught at runtime rather than in the output.

---

## 6. Recommended model-selection table (until a universal model lands)

| Input / situation | Use | Why |
|---|---|---|
| Collimated / MLA-relayed (single local angle per pixel), thick/cemented optics, sub-nm OPD wanted | `apply_real_lens_traced` | exact geometric OPD; its collimated assumption holds |
| Divergent / converging / tilted source, multi-element relay (e.g. no-MLA direct imaging) | **`apply_real_lens`** (analytic) | handles all angles via split-step; sharp + ~10× faster; **the correct default for these TX studies** |
| Autodiff design loop, or output at/near a caustic — **and** it fits memory | `apply_real_lens_maslov` | phase-space, exact OPD, differentiable, caustic-safe (blocked at scale by F2 today) |
| Aberration-free reference / isolating whether softness is optics vs model | `thin` (ABCD) | paraxial-perfect control |

---

## 7. Verification notes

* All §3 numbers are from end-to-end sim runs (`Reverse_Symmetric_ASM`,
  `run_poc_119_120_v518.py`, lumenairy 5.18.1), measured on the "Before
  Metasurface" field with a common spot metric (core FWHM + 2nd-moment MFD +
  central-row P/V).
* The thin-lens control required a small fix to the consumer (zero-power afocal
  group passthrough) and is a *reference*, not a physical model — it strips real
  aberrations by construction.
* The Maslov canonical-map fit quality (7×10⁻¹¹ waves RMS OPD) is from the
  library's own `verbose` fit report; the 451 GB / Nyquist figures in §4 are
  analytic (grid size × M × 8 bytes; `λ/(2 sin θ)`), cross-checked against the
  observed `G matrix (16777216, 210) = 28185.7 MB` log line at
  `output_subsample=4`.
* No physics error was found in any of the three models *within its documented
  regime*; every finding here is a **guard-rail / usability / scaling** gap, not
  a formula defect.

---

*Audit performed single-context against lumenairy 5.18.1, driven by the
Reverse_Symmetric_ASM TX 119/120 field investigation, 2026-07-02.*

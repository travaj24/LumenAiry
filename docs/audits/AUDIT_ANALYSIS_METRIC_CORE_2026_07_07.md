# analysis/ Metric-Core Audit — 2026-07-07

Scope (tranche 1): full line-by-line reads of the four metric modules every
other analysis result quotes — `analysis/psf_mtf_otf.py` (1,275),
`analysis/strehl.py` (537), `analysis/zernike.py` (787),
`analysis/beam_stats.py` (482).  **Tranche 2 (same day)**: `analysis/opd.py`
(686, full read) and `analysis/through_focus.py` (~1,430 of 1,776 lines read:
scan machinery, perturbation application, deterministic sweep, full MC,
linearized MC, JAX kernel builder; the JAX-scan body detail and
`tolerancing_report` formatting covered structurally).  Chosen as the
continuation of the under-covered-subsystem sweep (after `sources/core.py`):
`analysis/` is 16k LOC with only 5 findings from the 90-agent 2026-07-01
deep audit.  Read-only single-context pass; independent re-derivation of the
physics formulas.

---

## 1. Verdict

**All four modules are clean.**  No P1–P3 findings; the formulas were
independently re-derived and verified, and the prior-audit hardening
(Parseval normalisation, mandatory vector-Strehl reference, first-zero ring
confirmation, the Noll sign fix) is correct.  Only P4 nits below.

Independently verified this pass:

* **`compute_psf` Parseval `'power'` normalisation** — the area-element form
  `sum(|E_pupil|²)dx_p² == sum(psf)dx_psf²` (4.10 fix) is physically correct;
  the Fraunhofer grid relation `dx_psf = λf/(N dx_pupil)` is consistent
  between the normaliser and the returned value.
* **`rayleigh_resolution` first-zero search** — the ring-confirmation
  lookahead (require a post-minimum rise ≥ 0.5 % of peak) correctly rejects
  Gaussian monotone-to-underflow profiles (audit P1-F1-4) while accepting the
  ~1.75 %-of-peak Airy first ring; parabolic sub-pixel refinement is
  standard-form correct.
* **`sparrow_resolution`** — the reduction of the dip-vanishing condition to
  `I″(d/2) = 0` for even-symmetric profiles is correct, and the sub-pixel
  polar resample → natural cubic spline → brentq-on-`I″` pipeline is the
  right construction (accuracy pinned at <1 % on the Airy fixture).
* **`strehl_phase_integral`** — matches Born & Wolf 9.1.10 exactly
  (`|Σ pupil|² / (Σ|pupil|)²`, area elements cancel on a uniform grid).
* **`strehl_marechal`** — the extended Maréchal `exp(−(2πσ)²)` with the
  1/14-wave ≈ 0.82 doctest anchor.
* **Zernike P2-02 fix (Noll sign) — mathematically confirmed**: Noll 1976's
  sine modes are `+√(2(n+1)) R sin(mθ)` with m > 0, identical to OSA's
  m < 0 modes, so the removed −1 factor indeed matched no published
  convention.  OSA index maps, the radial closed-form sum, the orthonormal
  normalisation `√((2−δ_{m0})(n+1))`, the WLS `√w` row transformation, and
  the gelsy→numpy lstsq fallback chain are all correct.
* **`M2` (ISO 11146 single-plane)** — `M² = 2√(σ_x²σ_kx² − σ_x,kx²)`
  recovers exactly 1 on an analytic Gaussian (σ_x² = w0²/4, σ_kx² = 1/w0²).
  The Wigner cross-term `Σ (x−c_x)·Im(E*∂E)/P` omits the angular-centroid
  subtraction, but the omitted term is `c_k·⟨x−c_x⟩_I ≡ 0` by construction —
  the code is exactly right, and the 4.10 comment shows this was deliberate.
* Zernike basis cache: lock discipline (build outside the lock,
  double-build benign), LRU bound, registry enrolment — all sound.

---

## 1b. Tranche 2 — `opd.py` + `through_focus.py`

Verified correct: the OPD-unwrap Nyquist rule (`dx ≤ λf/aperture` from the
pupil-edge phase gradient — re-derived, consistent between
`check_opd_sampling` and the `wave_opd_1d/2d` warnings); the
reference-sphere conditioning sign conventions (subtract
`exp(−ik r²/2f_ref)`, re-add `−r²/2f_ref` so the returned OPD stays
absolute — self-consistent with the library's forward `+k·OPL` phase);
the through-focus H-hoist (algebraically identical to per-z ASM, input FFT
buffer defensively copied against pyFFTW plan recycling); the
nominal-pupil-fixed Strehl denominator applied consistently across
`tolerancing_sweep` (F-12) and `monte_carlo_tolerancing` (P3-F1-1) — the
Strehl > 1 pathology is closed everywhere; and the linearized MC's
**quadratic Maréchal superposition** (`S_pred = S_nom − Σ a_k ξ_k²` with
one-sided-probe calibration and the physical `a_k ≥ 0` clip) — the right
model, honestly scoped to small perturbations.

### AN-1 (P3) — `depth_of_focus` returns 2× the classical Rayleigh quarter-wave half-range
The defocus OPD at the marginal ray for axial shift δz is
`W = δz·NA²/2 = δz/(8 f#²)`, so the λ/4 Rayleigh criterion gives a
**one-sided** DOF of `2λf#²` (= `λ/(2NA²)`).  The function returns
`4λf#²` while documenting it as the ± half-range ("total axial tolerance is
2 × depth_of_focus(...)") — i.e. a total of `8λf#²`, twice the classical
Rayleigh tolerance.  The `'marechal'` branch (`λ/NA²`) evaluates to the
same number, and its docstring simultaneously claims it is "a tighter bound
than Rayleigh" and (in Notes) "mathematically equivalent" — while the
inline comment claims "a factor of 1 instead of 4", contradicting both.
(For reference, the true Maréchal S>0.8 defocus bound is ≈ the Rayleigh
λ/4 value — the classic λ/4 PV ↔ λ/14 RMS coincidence — so the two names
*should* return nearly the same number, but that number is `2λf#²`, not
`4λf#²`.)  **Fix**: return `2λf#²` (both formulas), or re-document the
return as the full range; either way reconcile the three contradictory
statements and re-pin the doctest.

### AN-2 (P4) — `wave_opd_1d` unwraps through interior invalid samples
The docstring says out-of-aperture zero-amplitude samples "are excluded
from unwrapping", but the valid mask is only used to crop the *ends* of the
cut — interior zero-amplitude samples (an annular / obscured pupil) stay in
the unwrap chain with `angle(0) = 0` garbage phase, which can inject 2π
slips into the far side of the pupil.  Fix the docstring or unwrap the two
sides of an interior gap independently.

### AN-3 (P4) — linearized-MC probe failures silently zero the sensitivity
`monte_carlo_tolerancing_linearized` maps a failed FD probe (broad except)
to `S_p = S_nom`, i.e. zero sensitivity for that knob, without any warning
or count — an optimistic bias in the tolerance budget.  Emit a warning
naming the knob and report a `failed_probes` count in the result dict.

---

## 1c. Tranche 3 — `field.py` (full read, 1,270 lines)

Verified correct: the Welford mirror-parity Petzval handling (4.13.2 —
mirror as `n₂ = −n₁` refraction + parity flip, matching the Seidel
module's convention; the pre-fix silent drop of every mirror contribution
is properly closed); the Born & Wolf Petzval sign (4.10); the 4.10 sag/tan
fan convention fix (chief tilted in +y, sagittal fan spread along x — the
pre-fix version compared two unrelated tangential fields); the
`relative_illumination` EP-aimed bundle translation (whole-bundle shift is
correct for infinity conjugates); the `distortion_grid` L²+M² ≥ 1 guard
with cross-platform ULP tolerance (v5.0.1).

### AN-4 (P3) — EP-aiming fix applied to only 2 of the 6 field-swept analyses
The 4.11.2 H-AB-3 fix (aim the chief ray through the entrance-pupil centre
at `ep_z` from `first_order_data`, instead of launching at the
first-surface vertex) was applied to `relative_illumination` and
`field_aberration_sweep` — but **not** to `distortion_vs_field`,
`distortion_grid`, `spot_diagram_vs_field`, or `footprint_per_surface` in
the same file.  Distortion is *defined* on the chief ray, so for mid- or
rear-stop systems the vertex-launched ray samples the pupil off-centre and
the reported distortion carries a pupil-aberration bias (exactly the error
mode the 4.11.2 comment describes for `relative_illumination`: "the chief
walks across the aperture as field angle grows"); the spot/footprint
bundles are similarly mis-centred and vignette asymmetrically.  **Fix**:
hoist the EP-aiming block (already written twice in this file) into a
shared helper and apply it to all six entries; a stop-at-front system is a
no-op, so the change is regression-safe.

### Tranche-3 nits (P4)
* `sensitivity_ranking`'s failure branch comment says "leave deriv[i] at 0
  (sentinel)" but the array is initialised to NaN — the NaN is the better
  sentinel; fix the comment.
* `field_aberration_sweep`'s focus search is a plain grid argmin (no
  parabolic refinement), quantising focus shifts to
  `2·dz_search/(n_z−1)` ≈ 4 % of the search window — fine for screening,
  worth one line in the docstring.

---

## 1d. Tranche 4 — `image_plane_wfe.py`, `coherence.py`, `polychromatic.py` (full reads)

Verified correct: the exact ray–sphere quadratic (unit-direction form,
smallest-|t| root for continuity through the chief), the 4.12.0 B2-3
arc-length sphere radius (`d/N_chief`, consistently inverted when
back-solving the best-RMS image distance), the 4.10 EP aiming and
alive-chief selection, the best-RMS defocus-to-curvature identity
`Δ(1/R) = 2c₁λ/r²` (re-derived), the piston-removed RMS (F-11), the
honestly-documented unweighted-RMS caveat on Chebyshev grids, the
Zemax/CGL pupil-grid presets, the Koehler cos-θ radiance weighting
(P3-12), and the whole polychromatic stack (common-image-plane
chromatic-defocus-aware accumulation, both Strehl conventions
cross-documented, anamorphic-`dy` threading).

**Cross-reference strengthening SRC-1** (`AUDIT_SOURCES_CORE_2026_07_07`):
`coherence.mutual_coherence` had the *identical* ensemble-MCF conjugation
bug and was fixed in 4.10 — its fix comment reads "`rows.T.conj() @ rows`
... produces the complex conjugate of the documented quantity ... any
phase-sensitive consumer saw the off-diagonals with flipped sign".
`PartialCoherenceMCF.from_ensemble`'s dense branch
(`E_mat.conj().T @ E_mat`) is exactly that pre-4.10 pattern, unfixed —
in-codebase precedent that SRC-1 is real and worth the one-line fix.

### AN-5 (P4) — best-RMS closed form uses the entrance-pupil radius
`eval_image_plane_wfe(image_plane='best_rms')` converts the fitted defocus
coefficient to a sphere-radius shift with `r_pup = semi` (the
entrance-pupil semi-aperture), but the correct scale is the exit-beam
radius at the sphere's tangent plane (the coordinate in which the
normalised `px` maps to physical sag).  Exact for pupil magnification ≈ 1
(the cross-check singlets); for strongly telephoto/retrofocus systems the
one-shot shift leaves residual defocus ∝ `(1 − (semi/a_exit)²)·c₁` and
the reported plane is not the true RMS optimum.  Iterating the closed
form once, or using the traced marginal-ray height at the tangent plane,
closes it.

### Tranche-4 nits (P4)
* `img_d_m` is documented as measured "from the last lens vertex" but is
  implemented as the axial offset from the *chief ray's last-surface
  intersection* (`cz = z_chief + img_d_m`) — internally consistent, but
  off-axis the two differ by the last-surface sag at the chief height.
  One docstring line.

---

## 1e. Tranche 5a — `aberration.py`, `detector.py`, `interferometry.py`, `coronagraph.py` (full reads)

All four clean.  Verified: `caustic_diagnostic`'s **per-eigenvalue KMAH
counting** (point caustic = 2 Maslov increments via separate
eigenvalue-crossing detection, near-coincident crossings merged for
display but both counted) — correct physics, and *notable*: this is
exactly the per-leg caustic-counting machinery that delta-audit finding
D3 (`AUDIT_V5_21_DELTA_2026_07_07`) recommends for closing the
multibranch KMAH mod-2 gap — it already exists in-tree.  Also verified:
the NaN-propagation of Seidel failures (4.10 — no more zero-filled
"diffraction-limited" lies), the flux-conserving detector binning (F-10,
both integer block-sum and sample-centre scatter branches), Poissonian
dark current, the quantized-pitch SH wavefront integration
(P1-DEEP-2-1), the general phase-shifting LSQ (F-13; both `'hardware'`
and `'library'` sign conventions expand correctly), the Michelson
visibility fix (4.10), the two-beam fringe spacing `λ/(2 sin(θ/2))`, and
the coronagraph λ·f/D pixel-scale fix (4.9) with the honest legacy-N
fallback warning.

### Tranche-5a nits (P4)
* `apply_detector` line ~160: a leftover expression
  (`float(pixel_pitch / (Ny / n_pixels * dx_field)) ...`) computes and
  discards its value; `shack_hartmann` similarly has two discarded
  `np.where(valid_mask, cx_ref, 0.0)` statements — dead code from
  refactors.
* `shack_hartmann`'s reference-centroid calibration pass uses a plain
  Fraunhofer FFT while the measurement pass uses batched ASM to the
  lenslet focal plane — the calibration should ride the identical
  propagation (a flat field through the same batched-ASM path) so
  grid-parity biases cancel exactly.
* `phase_shift_extract` documents "at least 3" frames but never
  validates it — `n < 3` silently returns a rank-deficient pinv fit.

---

## 1f. Tranche 5b — `ghost.py`, `phase_retrieval.py`, `ao.py` (full/near-full reads)

All three clean at the physics level.  Verified: the ghost retrace's
direction-aware surface intersection (v5.4.1 promotion to the canonical
`_intersect_surface`), correct forward/backward medium bookkeeping
(n₁/n₂ swapped by propagation direction), the reflect→×R / transmit→×(1−R)
Fresnel product (closing the B2-1 upper-bound gap for retraced paths), the
cos-weighted TIS Monte-Carlo estimator (`mean(f)·π` under the cosθ/π PDF —
re-derived, the 4.10 fix is right); GS/ER/HIO (power-matched target,
careful `F/|F|` zero-amplitude epsilon semantics, the exact Fienup HIO
support-feedback rule, dtype-keyed LRU kernel caches with per-cache locks);
and the AO stack (σ_IF coupling formula, consistent row-major actuator
indexing across `set_command`/`_influence_function_kth`/the cached-view
gemm, the P2-01 streamed normal-equations remediation implemented as
specified, the four-quadrant rim-FD in `zernike_modal_basis` (4.11.2),
LeakyIntegrator + `ao_closed_loop` gain/leak semantics).

### Tranche-5b findings (P4)
* `ghost_analysis` accepts `n_rays` but never traces a ray — the
  docstring's "Trace all 2-bounce ghost paths ... ray fan" overclaims
  (the honest upper-bound caveats elsewhere are correct); drop the dead
  parameter or route it to `retrace_ghost_path`.
* `retrace_ghost_path`'s FWHM code is *correct* (FWHM = 2·r₅₀ for a
  Gaussian) but the docstring and the inline comment give two different,
  garbled derivations ("25 % and 75 % radii ... times 1.1774" — neither
  is what the code does).
* `gerchberg_saxton_jax` returns the error of the *previous* iterate's
  far field (the `F` carried in loop state), while the NumPy path
  re-transforms the final constrained field — the two backends report
  final errors one iteration apart.
* Dead code from refactors: `gerchberg_saxton`/`error_reduction` each
  keep a discarded `*.shape[0]` statement; `zernike_modal_basis` keeps
  four discarded `rho.copy()`/`np.clip(...)` lines.
* Perf notes: `ao_closed_loop`'s scratch DM re-derives (and, under
  eager caching, duplicates) the parent's IF basis; and `fit_phase`
  re-solves the same design matrix every iteration — pre-factoring
  (pinv or Cholesky of AᵀA) would give an `n_iterations×` speedup of
  the loop's dominant cost.

---

## 2. Nits (P4)

* `mtf_cutoff` documents that `freq` "must be strictly increasing" but never
  validates it — a decreasing/unsorted axis silently mis-interpolates.
* `_zernike_radial`'s docstring claims "returns zero outside the unit disk";
  it doesn't — the zeroing lives in `zernike_polynomial`.  Harmless today
  (single caller), misleading for future direct callers.
* The Zernike basis-cache fingerprint keys on shape + dtype + first/last
  entries only: two grids with equal corners but different interiors (e.g.
  a warped vs uniform grid passed via the public `X, Y` arguments) would
  silently share a basis.  The documented staleness trade-off covers
  in-place mutation but not this case; adding one mid-point sample to the
  key closes it for free.
* `zernike_decompose` docstring: "21 [modes] covers up through 5th-order
  spherical" — 21 modes covers radial order n ≤ 5; primary spherical is
  j = 12 (n = 4) and *secondary* spherical (6, 0) is j = 24, outside the
  default.  Loose wording only.
* `beam_diameter` thresholds the *radially averaged* profile against the
  *2-D* peak — slightly biased low for asymmetric beams (the azimuthal
  averaging is documented, the peak-reference subtlety is not).

---

## 3. Coverage statement

**The `analysis/` subsystem line-audit is complete** except for
presentation glue.  Fully read: `psf_mtf_otf.py`, `strehl.py`,
`zernike.py`, `beam_stats.py`, `opd.py`, `field.py`,
`image_plane_wfe.py`, `coherence.py`, `polychromatic.py`,
`aberration.py`, `detector.py`, `interferometry.py`, `coronagraph.py`,
`ghost.py` (~12.5k lines).  Near-full: `through_focus.py` (~80 % — the
JAX scan/MC bodies at 1091–1441 are structural mirrors of their audited
NumPy siblings; `tolerancing_report` is formatting), `phase_retrieval.py`
(~70 % — NumPy GS/ER/HIO bodies + dispatchers + GS JAX kernel + cache
machinery read; the ER/HIO JAX kernel bodies at ~695–1022 mirror the
audited GS kernel), `ao.py` (~80 % — the `make_shack_hartmann_wfs`
closure tail is wiring per its docstring).  Deferred: `plotting.py`
(2,334, matplotlib presentation glue, low physics risk) and
`analysis/core.py`/`__init__.py` (re-export shells).  Everything
deferred retains the structural/probe coverage of the 2026-06-10
full-library audit and the 2026-07-01 deep sweep.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-07.
Companion docs from the same sweep: `AUDIT_V5_21_DELTA_2026_07_07.md`,
`AUDIT_SOURCES_CORE_2026_07_07.md`.*

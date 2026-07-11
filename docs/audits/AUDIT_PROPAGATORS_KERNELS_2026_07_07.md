# Propagator Kernel-Core Audit — 2026-07-07

Scope: full line-by-line reads of the free-space kernel family —
`propagators/asm.py` (894), `propagators/rs.py` (345),
`propagators/fresnel.py` (316), `propagators/sas.py` (338).  Continuation
of the under-covered-subsystem sweep (after `sources/core.py` and the
complete `analysis/` pass): these four kernels are the propagation engine
every simulation rides on.  `gbd.py` was fully audited in the v5.21 delta
round.  Read-only single-context pass; kernel physics independently
re-derived.

---

## 1. Verdict

**All four kernel modules are physics-correct.**  Independently
re-derived and verified this pass:

* **ASM Matsushima band-limit** — `f_max = L/(2λ|z|)` follows from the
  phase-gradient-per-frequency-sample ≤ π criterion on the paraxial
  kernel chirp; the tilted variant's 4.10 fix (mask evaluated at the
  *shifted* frequencies `|F + f₀| < f_max`) is the correct
  carrier-centred form of the same criterion.
* **The v5.5.3/P2-28 two-shift natural-layout fold** — `ifftshift`
  distributes over elementwise products and `ifftshift∘fftshift = id`,
  so the 4→2 shift fold is algebraically exact for any N (even or odd),
  as claimed.
* **Tilted ASM** — the demodulate → shifted-kz → remodulate pipeline is
  *exactly* equivalent to direct ASM of the tilted field (the shifted H
  applies `exp(i·kz(f+f₀)·z)` to the component that carries original
  frequency `f+f₀`); its genuine value is spectral recentring (sampling
  headroom for large tilts) plus the carrier-centred band-limit.
* **RS-I impulse response** — `h = (z/(2πr²))·(1/r − ik)·e^{ikr}` matches
  `−(1/2π)∂/∂z[e^{ikr}/r]` term-for-term (the 4.10 sign fix is correct);
  the centred-kernel `ifftshift` + centred-pad + same-window extraction
  implements linear convolution correctly, and the F-3 buffer-detach
  `.copy()` respects the pyFFTW double-buffer contract.
* **Fresnel/Fraunhofer** — textbook single-FFT forms (input/output
  quadratic phases, `e^{ikz}/(iλz)` prefactor, `dx·dy` integral
  discretisation) consistent with the library's forward `exp(+ikz)`
  convention; forward-only guards; the P2-29 float64-carrier-then-cast
  discipline applied to every quadratic-phase build.
* **SAS (Heintzmann/Loetgering/Wechsler 2023)** — precompensation
  `W·exp(ikz(h_AS − h_Fr))` with the paper's eq.-12 validity mask,
  natural-order Fresnel chirp matching the reference convention, the
  B1-5 pad-centring fix (`(N_new−N)//2`), and the deliberately-added
  physical normalisation `dx²/(iλz)` (absent from the reference
  notebook, consistent with `fresnel_propagate`).
* Cache hygiene throughout: per-family key tags (`'ASM'`, `'ASM_TILTED'`,
  `'RS'`, `'SAS'`) keep entries disjoint; SAS bundles its three kernels
  under one key; returned transfer functions are re-centred **copies**
  (the 4.10 mutation guard); complex64 paths fold phase mod 2π in
  float64 before casting everywhere.

---

## 2. Findings

### PK-1 (P4) — SAS `z_limit` exceedance is silent unless `verbose=True`
`scalable_angular_spectrum_propagate` computes the paper's closed-form
validity bound `z_limit` but reports exceeding it only inside
`if z > z_limit > 0 and verbose:` — a plain `print`, off by default.
Every sibling accuracy guard in the library (`check_opd_sampling`
warnings, the traced-lens `on_noncollimated`, the fiber-NA warning)
surfaces as a `RuntimeWarning` regardless of verbosity.  A user running
SAS past its validity envelope gets silently degraded phase.  **Fix**:
`warnings.warn(...)` unconditionally (keep the verbose print for the
diagnostics block).

### PK-2 (P4) — tilted-ASM docstring physics
Two statements in `angular_spectrum_propagate_tilted`'s docstring
mislead: (a) "`z` — propagation distance along the tilted axis" — `z` is
the **axial separation between parallel planes** (the shifted-kz math
transforms plane-to-plane at axial distance z, exactly like plain ASM);
(b) "keeps the field well-centred on the grid ... avoiding grid
walk-off" — the demod/remod construction avoids *spectral* aliasing of
the carrier, but the envelope still physically translates by `z·tanθ`
across the output grid (the linear term of `kz(f+f₀)` is retained, as it
must be).  Docstring-only; the math is exact.

---

## 2b. `fft_infra.py` (2,060 lines, full read — same day)

**Clean.**  The FFT dispatch / plan-cache / buffer layer is the most
disciplined infrastructure module audited in this sweep.  Verified:

* **The v4.12 double-buffer ownership contract is structurally
  satisfied library-wide**: the returned buffer is stable across exactly
  one subsequent same-key call, and every audited consumer either wraps
  the result in `fftshift`/elementwise arithmetic (materialising a copy
  and detaching immediately) or takes an explicit `.copy()` (the RS
  cached kernel, the through-focus hoisted input FFT); forward and
  inverse directions use separate cache keys, so chained `_ifft2(_fft2(
  ...)·H)` pairs never collide.  Slot advancement happens under the
  cache lock; per-key execution locks hold across copy-in + execute.
* Lock discipline: the P3-55 fix (pyFFTW structures cleared under
  `_PYFFTW_PLAN_LOCK`, ASM caches under `_ASM_CACHE_LOCK`, acquired
  sequentially, never nested — no lock order established) and the
  P3-14 in-place blacklist clear are correct; the auto-promote-under-
  lock stall is documented as a known limitation (P3-13).
* The P2-26 complex-only gate on all four dispatchers (real input →
  scipy fallback, no blacklist poisoning), the enumerated pyFFTW
  failure spectrum with one-shot warnings, `casting='no'` copy-in
  consistency with the dtype-keyed plans, and the NumPy 2.x
  `_is_cupy_array` isinstance fix.
* Byte-budgeted H-cache (`_entry_bytes` handles the SAS tuple bundles;
  per-entry 2 GB skip + 8 GB total eviction), the F-32 warmup
  thread-key fix, the `snapshot_fft_state`/`restore_fft_state`
  spawn-boundary carrier with setter-routed keys (P3-54), and the
  honestly-documented fork-safety limitations of the plain-global
  config knobs.
* `_validate_propagator_inputs`: metre-unit sniffing (wavelength
  > 1 mm raise; `dx` > 100 mm raise / > 1 mm warn — the 4.9
  telescope-pupil loosening) with actionable messages.

No findings.  (One design note, not a defect: benign double-build races
on a cold plan cache are accepted by design — the second build
overwrites the first while the first caller's references stay valid.)

---

## 2c. Chirp-Z family — `_bluestein.py` + `mft.py` (full reads, same day)

**Both clean.**  Independently verified:

* **The Bluestein construction** (`_bluestein_2d`): the chirp identity
  signs (pre/post chirps `+σ`, convolution kernel `−σ`), the
  wrapped-kernel circular-convolution folding (negative lags at
  `[L−(N_in−1), L)`, don't-care middle region covered by the zero-padded
  support — checked against `CONV[k] = Σ g[n]·h[(k−n) mod L]` for the
  full index range), the `next_fast_len` padding bound
  `L ≥ N_in + N_out − 1`, the float64-chirp-then-cast precision
  discipline, and the `~1e15`-radian phase-budget warning.
* **The centred wrapper** (`_bluestein_centred_2d`): all four factors of
  the `(n−cI)(k−cO)` decomposition (n-chirp, k-chirp, constant, core)
  carry the correct signs.
* **Double-buffer boundary case**: `G_FFT = fft2(g_pad)` followed by
  `H_FFT = fft2(h_2d)` at the *same* pyFFTW key sits exactly at the
  ownership contract's limit (stable across one subsequent same-key
  call) and consumes both immediately in the product — correct, but
  worth knowing it has zero slack if a third same-key FFT were ever
  inserted between them.
* **The three MFT propagators**: `α = dx_out/(N_in·dx_in)` (ASM-MFT
  inverse, sign +1, `1/(N_x·N_y)` DFT-consistent normalisation) and
  `α = dx_in·dx_out/(λz)` (Fresnel/Fraunhofer forward, sign −1) both
  re-derived; the `centre_out` offset is folded into the output-centre
  index `kc = N_out/2 − xc/dx_out` **and** the output quadratic phase is
  evaluated on the true physical coordinates including `xc` (the
  easy-to-miss part — correct here); the B1-4 strict-`<` band-limit
  boundary now agrees across NumPy/JAX/plain-ASM; `'ASM_MFT'`-tagged H
  cache keyed on input geometry only (output grid enters only in the
  Bluestein step) is the right cache boundary; forward-only guards on
  the Fresnel/Fraunhofer variants with actionable messages.

---

## 2d. `dispatch.py` (901 lines, full read — same day)

The routing layer is clean in itself: the B1-6 (forward-only methods
raise a dispatcher-level error on z < 0, and the auto-selector never
routes z < 0 anywhere but ASM), B1-7 (tuple-returning kernels unpacked
into `PropagationResult` with the kernel's true output pitch, anamorphic
`dy` threaded per L3), and B1-8 (output-grid requests auto-promote to
the MFT family or raise with the right pointer; the v5.2.3/v5.2.5
`output_shape` forwarding closures for gbd/hfpi/hf both branches) fixes
are all implemented as described.  The auto-selector's regime logic
(DOE→hfpi, aspherics→gbd, hard-aperture+accurate→hf, else maslov;
free-space N_F/Q thresholds) matches its documentation.

### DS-1 (P3) — `Source.propagate` mishandles pitch-changing propagators
`sources/core.Source.propagate` calls the dispatcher **without**
`return_result=True` and wraps the raw return:
`Source(E=E_out, dx=kwargs.get('output_dx', self.dx) or self.dx, ...)`.
For the tuple-returning kernels (`method='fresnel'`, `'fraunhofer'`,
`'sas'`) the raw return is `(E, dx_out, dy_out)` — the new `Source`
then carries a **3-tuple as its field** (its `.shape` property raises on
first access) and the **stale input pitch** as `dx` even though these
kernels change the pitch to `λz/(N·dx)`.  Reachable without naming a
method: free-space `method='auto'` selects `'fraunhofer'` at N_F < 0.1
and `'sas'` at Q > 1, so `Source.propagate(z=<far>)` silently produces
the broken Source.  The dispatcher's B1-7 `return_result=True` wrapping
exists precisely for this and is unused here.  **Fix**: have
`Source.propagate` call `propagate(..., return_result=True)` and build
the new Source from `result.field` / `result.dx` / `result.dy`.
(Supersedes the `output_dx=0`-falsiness nit in
`AUDIT_SOURCES_CORE_2026_07_07` §SRC-3.)

### Dispatch nits (P4)
* `_dispatch_to_method(method='asm', z=None)` returns `E_in` itself (no
  copy) — a caller mutating the "propagated" output mutates the source
  field; return a copy or document the aliasing.
* `_select_asm_variant` accepts `aperture_radius` and its docstring's
  decision item 4 says "z and aperture given" — but the body never
  consults it (only `which_propagator` uses it, for the reported
  Fresnel number).  Dead parameter in the selector.

---

## 2e. `system.py` (1,426 lines, full read — 2026-07-08)

The chain propagator gets the DS-1 theme **right**: Fresnel/SAS steps
resample back to the working pitch before the next element (with the
correct rationale — element phases must stay on the physical
coordinates), element handlers thread `current_dx`/`current_dy`
(C-P1-3), `evaluate()` routes through `return_result=True` (the pattern
`Source.propagate` should copy), the mutually-exclusive prescription
shapes raise (P1-F1-6), and the JAX fast path (dtype-keyed jit-kernel
cache, expensive trace outside the lock, mask data as positional args so
new mask *values* don't retrace, B1-2 fail-fast with the traceable-set
constant exposed) is correctly engineered.

### SY-1 (P3) — `evaluate()` silently drops DOE surfaces
`_prescription_to_elements` skips `'doe_placeholder'` decomposition
steps with `continue` and a code comment ("skip silently ... future
versions can plug a DOE handler").  A Zemax prescription containing an
air-to-air DOE/aspheric surface therefore propagates through the
ergonomic top-level `evaluate()` **without that element's phase** — a
physically wrong result with no diagnostic.  **Fix**: `warnings.warn`
naming the skipped surface(s) (one line), or refuse with a pointer at
`propagate_through_system` + a hand-built `'mask'` element until a DOE
handler exists.

### SY-2 (P4) — anamorphic gaps on the pitch-changing branches
The C-P1-3 anamorphic threading stops at the `fresnel`/`sas` propagate
branches: `fresnel_propagate` returns a distinct `dy_new`
(`= λz/(N·dy) ≠ dx_new` when `dy ≠ dx`) that is discarded, and the
resample-back call applies the *x*-ratio to both axes;
`scalable_angular_spectrum_propagate` takes no `dy` at all (assumes
square pitch) and the branch passes `current_dx` only, with no
`dy == dx` guard; the `turbulence` element builds its screen from
`E.shape[0]` + `current_dx` only.  Square-pitch chains (the common
case) are unaffected.  Guard or thread `dy` on all three.

### System nits (P4)
* `_make_system_jax_kernel` line ~1122: a computed-and-discarded list
  comprehension (`[i for i, sig ... if sig[0] == 'mask']`) — the same
  refactor-residue class as PK-3.
* `propagate_through_system_jax`'s docstring still says the pre-v4.12
  aperture schema "is still accepted with a one-shot
  DeprecationWarning" — v5.0 removed it and
  `_resolve_aperture_params` now raises `ValueError`; the docstring is
  stale against the module's own v5.0 migration notes.

---

## 2f. Huygens–Fresnel family — `hf.py` + `hfpi.py` (full reads, 2026-07-08)

Verified correct: the Van Vleck machinery in `hf.py` (4-point
mixed-partial FD stencils for the cross-Hessian, `√|det|` density, the
4.11.2 `−1j` Van Vleck–Morette prefactor reconciling the OPL-callable
variant with `1/(iλz)`); the B1-10 pixel-centred grids; the 4.11.2 LG
projection of the actual input field (structured sources no longer
discarded) with the honest warn-on-failure fallback; and in `hfpi.py`
the uniform-per-solid-angle cone sampling, the symmetric Kirchhoff
obliquity `(cosθ_in + cosθ_out)/2`, the per-stream RNG decorrelation
(`_spawn_rng`), the P1-NEW-H grazing-ray step-zeroing, the P2-32
complex-dtype promotion, the correctly-fixed 4-D stratification
(4.11.2 cartesian product), and — importantly — the **P2-31 honest
normalisation warning verified in place** (HFPI documented as a
phase-structure diagnostic: the missing per-path `1/r` and
output-binning Jacobian are enumerated with the measured ~14×
spatial-bias number).

### HF-1 (P4) — dead primary waist estimate, correct by accident
`propagate_huygens_fresnel_through_prescription` estimates the source
waist as `w_s = float(beam_d4sigma(E_in, dx=dx)) / 4.0` — but
`beam_d4sigma` returns a **tuple** `(d4x, d4y)`, so `float(tuple)`
raises `TypeError` on every call and control always falls into the
except-branch second-moment fallback (which happens to compute the same
σ_x, so results are right).  The primary path is dead, and the
exception-driven flow would also mask any genuine `beam_d4sigma`
failure.  Fix: `d4x, _ = beam_d4sigma(...); w_s = 0.25 * float(d4x)`.

### HFPI-1 (P4) — left-edge output binning (half-pixel shift)
`accumulate_to_grid` bins with `ix = floor(x/dx + Nx/2)`: pixel `i`
collects paths in `[x_i, x_i + dx)` instead of the cell-centred
`[x_i − dx/2, x_i + dx/2)` every other grid consumer in the library
uses — a systematic half-pixel image shift in HFPI outputs relative to
ASM/Fresnel results on the same geometry.  HFPI is already documented
as spatially non-quantitative (P2-31), but registration against other
propagators is exactly its advertised use ("re-normalise against ASM").
Fix: `floor(x/dx + Nx/2 + 0.5)`.

### HFPI-2 (P4) — the `sampling` kwarg is dead; stratified never runs
`propagate_hfpi_through_prescription(sampling='stratified')` documents
stratified sampling as the **default** ("partitions the
source-direction cone into equal-solid-angle cells ... reducing
variance") — but the body calls `init_paths_from_field` unconditionally
and never dispatches to `init_paths_stratified`.  Every prescription
HFPI run uses uniform sampling regardless of the kwarg.  One `if`
statement to wire, or drop the parameter.

### HF-family nits (P4)
Three more discarded statements (`array_namespace(E_in)`,
`list(range(len(surfaces)))`, `int(paths.positions.shape[0])`) in
`hfpi.py` — the recurring refactor-residue class.

---

## 2g. `mhs.py`, `vector_diffraction.py`, `ensemble.py` (full reads, 2026-07-08)

`ensemble.py` is **clean** (bail-loud shape-mismatch guard, backend
preservation, dtype-mapped accumulator, empty-ensemble guard, correct
Wolf `⟨|E_k|²⟩` averaging).  `mhs.py`'s framework is clean (the 4.10
`asm_subdomain` pitch-mismatch raise; the v5.2.3 maslov square-guards +
resample-with-Parseval-renorm).  `vector_diffraction.py`'s
Richards–Wolf physics was **re-derived and verified**: the Leutenegger
polarisation-rotation matrix entries are exact; the
Cartesian-to-solid-angle Jacobian `dΩ = dx·dy/(f²cosθ)` confirms the
4.10 `cos^(−1/2)θ` effective apodisation; and the 4.11.2 prefactor
`(−ik/(2πf))·e^{+ikf}·dx²` (with its 1/f amplitude scaling and the
`e^{+ikf}` sign matching the library's forward convention) follows
exactly from Richards & Wolf Eq. 3.7 under that substitution.

### MHS-1 (P4) — `gbd_freespace_subdomain` uses the deprecated kwarg
It forwards `output_grid=(Ny, Nx)` to `propagate_gbd_freespace` — the
kwarg v5.2 renamed to `output_shape` with a DeprecationWarning shim.
Every GBD MHS subdomain call fires the warning, and the factory breaks
outright when the shim is removed.  Rename to `output_shape=`.

### MHS-2 (P4) — `prescription_subdomain` mixes the two output-grid contracts
Its non-maslov branch calls `propagate(..., output_grid=(out_s.Ny,
out_s.Nx), output_dx=out_s.dx)` — but the dispatcher parses
`output_grid` as the `(N_out, dx_out)` form, so `out_s.Nx` is
mis-parsed as a *pitch*.  Correctness survives only because
`output_dx` is always passed and overrides the mis-parsed value; the
residue is that the resolved shape becomes the square `(Ny, Ny)`, so a
non-square `out_surface` is silently squared — despite this function's
own maslov-guard error messages advertising "methods that support
anamorphic outputs (gbd/hfpi/hf)".  Pass
`output_grid={'N': out_s.Ny, 'dx': out_s.dx}` or thread
`output_shape` through `method_kwargs`.

### VD-1 (P4) — immersion NA silently clamped
`richards_wolf_focus` computes `theta_max = arcsin(min(NA, 0.9999))` —
an oil/water-immersion NA (e.g. 1.4) is silently clamped to an
air-objective 89.2° cone with no error and no immersion-index
parameter.  Raise for `NA >= 1` (pointing out the missing
`n_immersion` support) instead of silently computing wrong physics.

---

## 2h. `subaperture.py` + `vectorial_hfpi.py` (full reads, 2026-07-08) — subsystem complete

`subaperture.py`'s machinery is correct: the cosine-taper windows with
weight-normalised recombination, the v5.2.3 image-plane window mapping
(the ABCD conjugate-solve algebra verified — `d_img = −B_pre/D_pre`,
`M = A_l + d_img·C_l`; only the code comment writes the operator order
backwards), the per-patch `source_centre` fits (4.11.2) and actual-field
LG projections (4.13.2).  `vectorial_hfpi.py` mirrors the scalar HFPI
fixes faithfully (Kirchhoff weighting, symmetric obliquity, `_spawn_rng`
decorrelation, grazing step-zeroing), and the optional P3-24 transverse
Jones projection `E_t = E − (E·ρ̂)ρ̂` is algebraically correct with the
dropped-`Ez′` limitation documented.

### HF-1 UPGRADE (P3 at the second site) — `float(tuple)` waist bug ×2
The `w_s = float(beam_d4sigma(E_in, dx=dx)) / 4.0` always-TypeError
pattern (HF-1) occurs **again** in
`propagate_subaperture_asymptotic` — and here the except-branch
fallback is `w_s = source_box_half / 2`, a pure geometry guess **not**
equivalent to the intended data-driven estimate.  The per-patch LG
projections therefore always use a basis waist unrelated to the actual
beam; with the default (p≤3, |ℓ|≤3) truncation, a beam whose waist
differs substantially from `source_box_half/2` is silently
mis-represented across every patch — degraded wide-field PSFs with no
diagnostic.  Same one-line fix at both sites
(`d4x, _ = beam_d4sigma(...)`).

### VHFPI-1 (P4) — P2-32 fix not mirrored to the vector accumulator
`propagate_vector_hfpi_freespace_aperture` passes
`output_dtype=Ex_in.dtype` raw into `accumulate_vector_to_grid` —
the scalar siblings promote via `_complex_output_dtype` (the v5.17
P2-32 fix) precisely because a real-dtype input otherwise allocates a
real accumulator and `np.add.at` silently discards the imaginary half
of every path weight (~40 % intensity loss measured in the scalar
case).  A real-dtype `(Ex_in, Ey_in)` hits the identical bug here.
One-line fix: `output_dtype=_complex_output_dtype(Ex_in.dtype)`.
`accumulate_vector_to_grid` also shares HFPI-1's left-edge binning
(same half-pixel shift), and this module kept the pre-v5.2
`output_grid=(Ny, Nx)` kwarg name without the rename/shim its siblings
got — a naming inconsistency to fold into the next API pass.

### PK-3 (P4) — dead code + an uncached kernel
`fraunhofer_propagate_mft` keeps two discarded `np.arange(N_in, ...)`
statements (leftovers from removing the input-quad grids — the same
refactor-residue class found in `detector.py`, `ao.py`,
`phase_retrieval.py`, `sources/core.py`).  Perf note: `_bluestein_2d`
re-FFTs its chirp kernel `h_2d` on every call although it depends only
on `(α, N_in, N_out, sign, dtype)` — an H-cache-style memo of `H_FFT`
would save one padded-grid FFT per MFT call in zoom loops
(coronagraph contrast curves hit this path repeatedly at fixed
geometry).

---

## 3. Coverage statement

**The propagators subsystem line-audit is complete** (non-asymptotic
modules).  Fully read: `asm.py`, `rs.py`, `fresnel.py`, `sas.py`,
`fft_infra.py`, `_bluestein.py`, `mft.py`, `dispatch.py`, `system.py`,
`hf.py`, `hfpi.py`, `mhs.py`, `vector_diffraction.py`, `ensemble.py`,
`subaperture.py`, `vectorial_hfpi.py` (~11.4k lines); `gbd.py` (3,188)
fully audited in `AUDIT_V5_21_DELTA_2026_07_07.md`.  Deferred:
`result.py`/`propagation.py` (thin facades/re-export shells) and the
asymptotic family (~4.9k across 6 files — `asymptotic.py`,
`asymptotic_canonical_fit.py`, `asymptotic_modes.py`,
`asymptotic_maslov.py`, `asymptotic_aberration_tensor.py`,
`asymptotic_jax_twin.py` — partially covered by the 2026-07-02
wave-lens audits and the 07-01 deep sweep; the natural next target).
All deferred modules retain the 2026-06-10 probe coverage
(ASM/RS/Fresnel energy + waist probes PASS) and the 2026-07-01
deep-sweep findings/remediations (P2-26/27/28/29/31/32 verified
in-code this pass).

---

*Audit performed single-context against lumenairy v5.21, 2026-07-07.
Companion docs from the same sweep: `AUDIT_V5_21_DELTA_2026_07_07.md`,
`AUDIT_SOURCES_CORE_2026_07_07.md`,
`AUDIT_ANALYSIS_METRIC_CORE_2026_07_07.md`.*

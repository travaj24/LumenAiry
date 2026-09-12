# TR-MAIN-1 audit — `apply_real_lens_traced` first half (validation / delegation / amplitude legs / launch grid / trace / fits / Newton call)

Repo: `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`
All repro scripts under `…/scratchpad/TR-MAIN-1/`. Nothing in the repo was modified.

## Scope read (files + line ranges actually read; what I did NOT get to)

* `lumenairy/elements/_lens_traced.py`
  * `6731–7767` signature + full docstring
  * `7768–9900` the audited body: validation gates, origin gate, dy/mirror/fold guards,
    multibranch dispatch, stop-index warnings, grid axes, chunk-assembly gate, carrier
    resolution + engage test, `_pip_residual_ri` / `_pip_sample_residual`, F1 collimation
    guard + `delegate`, both amplitude legs (`parallel_amp`, `fast_analytic_phase`),
    launch radius / fit-radius / beam-centre / decentred-order resolution, `_imap` domain
    gate, `min_coarse_samples_per_aperture`, launch grid + tilt/carrier launch, `trace`,
    exit-vertex transfer, H6 entrance eikonal, exit-NA Nyquist guard, reshape + piston
    reference, `_TracedExitSupport`, fit-domain restriction.
  * `9900–10700` (block continuation past the 9700 boundary, as instructed): C11/C12
    arbiter, the fit application, `inversion_method='fit'`, `_Cheb2DEvaluator` /
    `RectBivariateSpline` construction, paraxial-magnification stencil, `_spline_data`,
    `_invert_newton`, `_invert_newton_parallel` head.
  * Read for context only (TR-MAIN-2's range): `10991–11006` (`_build_newton_mask`),
    `11170–11290`, `11314–11460` (coarse Newton + OPL upsample), `11660–11780`,
    `12355–12450` (final assembly), `13057–13120` (`prepare_real_lens_traced` head).
  * Helper context (TR-INFRA's range): `100–160`, `3291–3402` (`_geometric_lens_phase`),
    `4476–4530` + the float/ndarray/'auto' branches of `_compute_carrier`.
* `lumenairy/elements/_lens_real.py:6823-6825` (grid-axis convention only),
  `lumenairy/raytrace/surface.py:391-448`, `lumenairy/raytrace/trace.py:461-620`,
  `lumenairy/ui/lens_options_dialog.py:125-145`,
  `tests/unit/test_v5_1_0_agent_a.py:180-200, 610-690`.
* `REAL_LENS_CHANGES.md` §2, §2b, §2c, §3; `docs/audits/APPROXIMATION_AUDIT_TRACED_2026_07_30.md`
  (§0), `AUDIT_TRACED_MEMORY_2026_08_09.md` §5.7–5.8, `AUDIT_TRACED_SPEED_2026_08_09.md`
  (ranked table), `ARCH_TRACED_ENCAPSULATION_2026_08_03.md` (skimmed).

**NOT covered**: the GPU (`use_gpu`/`amp_use_gpu`) code paths beyond desk-checking (no cupy);
the process-pool dispatch body below `10700`; `caustic='multibranch'/'uniform'` (delegated
modules); `_lens_imap` internals; `_fit_residual_eikonal` / niche-C6 internals;
`inversion_method='backward_trace'`.

---

## Findings

### [P1] `fast_analytic_phase=True` raises `AttributeError` on every refracting prescription — the path is completely dead
`lumenairy/elements/_lens_traced.py:3359` (reached from `:8878` and `:8928`)

`_geometric_lens_phase` does `from .. import raytrace as _rt` then `_rt._surface_sag_xy(X, Y, surf)`.
`_surface_sag_xy` lives in `lumenairy/raytrace/surface.py:391` and **is not re-exported by
`lumenairy/raytrace/__init__.py`** (that file imports from `bundles, core, differential,
from_field, jax_trace, paraxial, seidel_analysis, world` only).

Measured:

```
>>> import lumenairy.raytrace as rt; hasattr(rt, '_surface_sag_xy')
False
apply_real_lens_traced(..., fast_analytic_phase=True, parallel_amp=True ) ->
    AttributeError: module 'lumenairy.raytrace' has no attribute '_surface_sag_xy'
apply_real_lens_traced(..., fast_analytic_phase=True, parallel_amp=False) -> same
```

(repro: `TR-MAIN-1/p4_fastphase.py`, and the 8-line confirmation in the transcript.)
The loop only survives when `abs(n2 - n1) < 1e-15` for **every** surface, i.e. an all-air
prescription.

Why nobody noticed: the only tests that touch the function
(`tests/unit/test_v5_1_0_agent_a.py:649,671`) deliberately use
`_trivial_no_refraction_prescription`, whose own docstring says it exists for
"avoiding the latent `_rt._surface_sag_xy` lookup that's outside this agent's scope to fix"
(`:184-190`). The third test is a *source-text* assertion (`'get_default_real_dtype' in src`).
No test anywhere passes `fast_analytic_phase=True`.

Impact: the knob is documented in `REAL_LENS_CHANGES.md` §2c with measured claims
("0.000 % intensity change … 9.1 nm RMS phase difference … 24 % wall-time savings",
"53 %" in `docs/changelogs/v4.md:5783`) and is a **GUI checkbox**
(`lumenairy/ui/lens_options_dialog.py:133`, tooltip "~25% speedup with <10 nm OPL error").
Any user who ticks it gets a traceback. It is also the only cheap alternative to the
17.2 GB / ~10 s (N=32768) plane-wave ASM pass, so the crash costs the documented speed-up too.

Fix: `from ..raytrace.surface import _surface_sag_xy` (or `_rt.surface._surface_sag_xy`),
plus one behavioural test with a refracting prescription.

Latent sibling in the same function: `phase` is accumulated at
`get_default_real_dtype()` and includes the bulk piston `k0·n·t` (`:3367-3369`), which is
2.91e4 rad for 4 mm of n=1.5168 glass at 1.31 µm. Under `set_default_real_dtype(np.float32)`
the float32 ulp there is **1.95e-3 rad (λ/3220)** and grows linearly with total thickness —
a genuine float32-phase-wrap defect, currently unreachable because of the crash above.

---

### [P1] `newton_fit='spline'` returns an **identically-zero field**, with no message that says so, for any prescription whose surfaces carry a semi-diameter
`_lens_traced.py:9719-9725` (NaN fill) + `:10262-10264` (`RectBivariateSpline(...)`)

The launch grid is a **square** of half-width `launch_radius = 0.75·aperture` (`:9016`), so its
corners sit at `√2·0.75·aperture = 1.06·aperture`, i.e. **2.12× a typical
`semi_diameter = aperture/2`**. Those rays are vignetted, `final.alive.all()` is False, and
`x_out_grid / y_out_grid / opl_grid` get `np.nan` at the dead nodes. `RectBivariateSpline`
propagates a single NaN through its banded solve, so `So.ev` is NaN everywhere,
`valid = np.isfinite(opl_map)` is all-False and `E_out` is all zero.

Measured (`TR-MAIN-1/p17_spline_zero.py`, biconvex R=±60 mm, t=4 mm, ap=8 mm, N=384, sub=4):

| per-surface `semi_diameter` | dead launch rays | nonzero pixels, polynomial | nonzero pixels, **spline** |
|---|---|---|---|
| 4 mm (= aperture/2) | 8116 / 12321 | 68525 | **0** |
| 7 mm (corners still clipped) | 896 / 12321 | 68525 | **0** |
| none | 0 / 12321 | 68525 | 68525 |

Against the independent oracle the "error" saturates at exactly λ/2 (655.0000 nm max,
282.8 nm rms — the `np.angle(±0.0)` artefact of an all-zero field);
`TR-MAIN-1/p16_spline.py`, `p16b_spline_diag.py`.

The in-code comment at `:9710-9715` states the opposite: *"we guard against it by filling
dead entries with NaN and extrapolating with the spline's natural extrapolation (OK inside
the entrance disc of interest)"* — NaN does not extrapolate, it annihilates the fit. The
comment also says *"vignetting is rare for normal lenses"*; the launch-square geometry makes
it the norm. The D5 note at `:9120-9128` knows about the all-zero symptom but attributes it
to the aperture:beam cliff, which is a different and much rarer trigger.

Diagnostics actually emitted on the all-zero call: a 100 %-unconverged `RuntimeWarning`
(suppressed by `on_undersample='silent'`) and the `_lens_imap` "guard G8 … the incumbent
returned only 0 finite answers" notice. Nothing states that the returned field is zero.
This contradicts the header claim at `:7773-7789` that "the two fits are indistinguishable in
accuracy — differences sit in the 4th-5th significant figure"; and `newton_fit='spline'` was
briefly the resolved `'auto'` default in v5.30.2.

Fix (cheapest correct): when `newton_fit == 'spline'` and `not final.alive.all()`, raise with
a message pointing at `'polynomial'`; or fill the dead nodes by extrapolation from live
neighbours **before** building the spline, which is what the comment already claims happens.

---

### [P1] A real-dtype `E_in` is accepted, runs the whole pipeline, then crashes on the last assembly line
`_lens_traced.py:11697` (`target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128`)
consumed at `:12392` and `:12399`

```
apply_real_lens(float64 field, ...)          -> complex128, fine
apply_real_lens_traced(float64 field, ...)   ->
  File "_lens_traced.py", line 12392, in apply_real_lens_traced
    E_out = np.where(valid, E_out, target_cdtype.type(0))
AttributeError: type object 'numpy.complex128' has no attribute 'type'
```

`np.complex128` is a scalar **type**, not a `np.dtype`, so `.type` does not exist. The whole
point of the `else np.complex128` branch is to support a real input, and
`_check_2d_scalar_field` lets one through; the ray trace, the three fits and the Newton
inversion all run before the failure. Reproduced at `sub=1` and `sub=8`
(`TR-MAIN-1/p14_realdtype.py` and the traceback run).
Fix: `target_cdtype = np.dtype(E_in.dtype if np.iscomplexobj(E_in) else np.complex128)`.

*(The failing statements are inside TR-MAIN-2's line range; the root assignment and the
accepting validator are shared. Flagged here because the validator at `:7772` is in my range.)*

Related, currently unreachable because of the above: `_reference_input()` at `:7765-7768`
does `np.exp(1j*_k0*_carrier_W).astype(E_in.dtype)`. For a real `E_in.dtype` that silently
discards the imaginary part (numpy emits only a `ComplexWarning`) — measured: the cast turns a
unit-modulus phasor into `|cast| ∈ [1.8e-4, 1.0]`. Fix the dtype and this becomes live.

---

### [P2] `carrier=<ndarray>` quantises the H6 **entrance eikonal** to the wave grid by nearest neighbour — 54× worse than the analytically identical `carrier=<float>`
`_compute_carrier` ndarray branch (`w_fn` returns `W_full[fy, fx]`, `grad_fn` returns
`gWx[fy, fx]`), consumed at `_lens_traced.py:9465` (launch cosines) and `:9606`
(`final.opd += _carrier_W_fn(h_x, h_y)`).

Diverging point source at s = +200 mm through an f/32 plano-convex singlet, N=768,
dx = 10.5 µm, w = 1.0 mm; oracle = independent exact-sphere trace launched along grad W with
W(x_in) added (`TR-MAIN-1/p6_carrier.py`):

| carrier | rms residual vs oracle (piston+tilt removed) | raw max |
|---|---|---|
| `None` | 6.97e-3 rad (1.45 nm) | 2.37e-2 rad |
| `200e-3` (float, exact sphere) | **1.845e-3 rad (0.385 nm)** | 6.58e-3 rad |
| `'auto'` | 1.845e-3 rad (0.385 nm) | 6.57e-3 rad |
| `TiltedCarrier(200e-3,0,0,0,0)` | 1.845e-3 rad (0.385 nm) | 6.58e-3 rad |
| `ndarray` holding the **same** W | **9.914e-2 rad (20.67 nm)** | 5.70e-1 rad (λ/11) |

Mechanism isolated at the source (`TR-MAIN-1/p6b_ndarray.py`), querying both branches at the
launch-lattice nodes (which are *not* wave-grid centres — `xs_in` is a `linspace` over
±0.75·aperture with odd `n_launch`):

| N | dx | eikonal `w_fn` error rms / max | predicted `|∇W|·dx/2` | cosine `L` error rms |
|---|---|---|---|---|
| 256 | 31.6 µm | 301.4 / 1030.4 nm (4.94 rad) | 237.3 nm | 4.5e-5 |
| 512 | 15.8 µm | 150.9 / 589.2 nm (2.83 rad) | 118.6 nm | 2.3e-5 |
| 1024 | 7.9 µm | 69.0 / 248.8 nm (1.19 rad) | 59.3 nm | 9.9e-6 |

Exactly linear in `dx` — a half-pixel nearest-neighbour error, not a gradient error. The
in-code note (`:7045-7047`) warns only about the **direction cosines**, which are the *smaller*
of the two errors here; the eikonal lookup is the dominant one and is unmentioned.
Impact: `carrier=ndarray` is the documented way to hand the element a measured or externally
computed wavefront, and it degrades the traced OPL by ~50× at the same cost.
Fix: sample `W_full` and its gradient with `scipy.ndimage.map_coordinates(order=1 or 3)`
(or fit a low-order model once), which is the same cost class as the current fancy-index.

---

### [P2] The exit-NA Nyquist guard measures NA over rays **outside the physical aperture** — 3.14× overstated on a plane-wave input
`_lens_traced.py:9662-9666` (`_sig = (_amp >= e^-4·max).ravel() & final.alive`)

The trace runs on `pres_no_ap` (`aperture_diameter` popped, `:9006-9008`), so
`surfaces_from_prescription` gives `semi_diameter = inf` unless the *surfaces* carry their own.
Rays are launched out to `0.75·aperture`. `_sig` gates only on input **amplitude**; there is no
gate on launch height, even though the returned field is masked to
`X²+Y² ≤ (aperture/2)²` at `:12397-12399`. The amplitude gate does nothing for a flat /
top-hat / wide-Gaussian input.

Measured (`TR-MAIN-1/p10_nyquist.py`, the file's own f/5 fixture R=±51.68 mm, ap=24 mm,
`E_in = ones`, N=512):

| prescription | reported `na_exit` | true marginal NA at the 12 mm aperture edge | overstatement |
|---|---|---|---|
| with `semi_diameter = 12 mm` | 0.24992 | 0.24964 | 1.001× |
| **no per-surface `semi_diameter`** | **0.78487** | 0.24964 | **3.144×** |

Consequences: (a) the RuntimeWarning demands `dx ≤ 0.83 µm` instead of the correct 2.62 µm —
a 3× finer grid, i.e. 9× the memory; (b) `_exit_na_out['na_exit']` is the number
`propagate_traced_carrier_chain` feeds to `on_tilt_exact_grid`, whose **default action is
`'error'`** (`lumenairy/propagators/carrier.py:6414, 6535, 6550`) — so an overstated NA can
hard-refuse a legitimate chain leg. The C4 note at `:9650-9661` fixed a transpose in exactly
this statistic, so it is known to be load-bearing.
Fix: intersect `_sig` with `h_x² + h_y² ≤ (aperture/2)²` when `aperture is not None` (and with
the per-surface clear aperture otherwise) — i.e. gate on the same disc the output mask uses.

---

### [P2] The `ray_subsample` accuracy contract is wrong, and the returned field's accuracy swings ~4 decades on an internal guard with no user-visible signal
docstring `:6887-6894`; upsample `:11404-11428`; gate `:9181-9182` + `_IMAP.imap_enabled`

The docstring promises, grid-independently, that "the default `8` (and `ray_subsample=4`)
typically loses < 1 nm of fidelity". The real cost of the order-1 coarse→fine OPL upsample is
`(sub·dx)²·f''/8`. Measured against my independent oracle on a biconvex f/7.5 singlet
(R=±60 mm, t=4 mm, ap=8 mm, N=768, dx=13.54 µm; `TR-MAIN-1/p5c_imap.py`, `p15_imap_guard.py`):

| configuration | rms vs oracle | max |
|---|---|---|
| `inverse_map=True` (shipped default), sub = 4 / 8 / 16 | **0.000 nm** (<5e-17 m) | 0.000 nm |
| `inverse_map=False`, sub = 4 | 3.40 nm | 12.90 nm |
| `inverse_map=False`, sub = 8 | **11.63 nm** | **51.58 nm** |
| `inverse_map=False`, sub = 16 | 44.18 nm | 206.3 nm |
| `inversion_method='fit'`, sub = 8 | 11.63 nm | 51.58 nm |
| sub = 1 (no coarse lattice at all) | 0.00035 nm | 0.00088 nm |

Exact `sub²` scaling confirms the mechanism. So the "<1 nm" claim is off by ~12× in rms and
~50× in peak at a perfectly ordinary grid, and is only rescued by the v5.35
inverse-characteristic evaluator. The evaluator is **off** for `use_gpu=True`
(`_imap_domain_gate` excludes it), for `inversion_method != 'newton'`, for
`inverse_map=False`, and whenever an internal guard refuses — and the refusal is reported
**only** through the private `_imap_out` dict, plus (since v5.44) a RuntimeWarning naming the
guard letter. A caller who flips `use_gpu=True` for speed silently trades 0 nm for 11.6 nm rms
with no message at all.

Fix: (a) restate the `ray_subsample` bound as `(sub·dx)²/(8·f_exit)` with a worked number;
(b) make the non-imap fallback emit the same one-line notice the imap refusal does;
(c) consider raising `_opl_up_order` to 3 unconditionally (it is already the carrier default) —
see P3 below for its boundary caveat.

---

### [P2] `parallel_amp_min_free_gb = 48.0` is a fixed threshold independent of N, so the measured 1.35× is unavailable on any machine under 48 GB free
`_lens_traced.py:6756, 8826-8846`

Measured at N=1024, sub=4, plano-convex, warm caches, median of 4 runs
(`TR-MAIN-1/p12b.py`, `p12_perf.py`):

```
parallel_amp=False : 5.195 s   tracemalloc peak 151.0 MB = 18.0 x (8 N^2)
parallel_amp=True  : 3.839 s   tracemalloc peak 218.2 MB = 26.0 x (8 N^2)
                    -> 1.35x wall, +67 MB peak
```

The guard demands 48 GB free to spend 67 MB — 700× over-conservative at this size. The
docstring itself says the number is "tuned for the N=32768 complex128 case". On a 32 GB
workstation or a 12 GB CI runner `parallel_amp` never engages at **any** N. (The GUI tooltip
claims "~1.7x speedup"; measured 1.35× here.)
Fix: scale the requirement with the field, e.g. `max(2.0, 6 * E_in.nbytes/1e9)` GB, keeping
`parallel_amp_min_free_gb` as an override.

---

### [P2] `amp = np.abs(E_analytic)` allocates a full-grid float64 that the default path never reads
`_lens_traced.py:8877 / 8901 / 8912`, consumed only at `:11188` (`del amp`), `:11291`
(`amp[::sub, ::sub]`), `:11467` (sub == 1) and `:12382` (non-preserve branch)

On the DEFAULT configuration (`preserve_input_phase=True`, `ray_subsample=8`), `amp` is either
deleted unread (inverse-map + band path, `:11187-11188`) or read exactly once as
`amp[::sub, ::sub]` to build the Newton mask (`:11291`, then `del amp_coarse, amp`).

Measured cost of the full reduction (`TR-MAIN-1/p12_perf.py` microbenchmarks):

| N | `np.abs(E_analytic)` | `np.abs(E[::8, ::8])` | result size |
|---|---|---|---|
| 1024 | 14.17 ms | 0.114 ms | 8.4 MB |
| 2048 | 37.29 ms | 0.853 ms | 33.6 MB |
| 32768 (extrapolated) | **~9.6 s / 8.59 GB** | ~0.15 s / 134 MB | — |

Fix: compute `np.abs(E_analytic[::sub, ::sub])` on the `sub > 1` preserve path and keep the
full `np.abs` only for `sub == 1` and the `preserve_input_phase=False` branch. Saves one
full-grid float64 (8.59 GB at N=32768) and one full-grid pass.

Same class, smaller: `_reference_input()` materialises a full complex128 `np.ones_like(E_in)`
(17.18 GB / ~10 s at N=32768, measured 15.4 ms at N=1024) purely so `apply_real_lens` can be
run on it and its phase read — and the code itself notes at `:7760-7764` that this reference is
**input-independent**. A module-level cache keyed on
`(prescription-hash, wavelength, dx, N, carrier-key)` for `phase_analytic_lens` alone would
remove the entire second ASM pass from every chain/sweep call after the first
(`_carrier_reuse_key` at `:12452` already exists for precisely this key).

---

### [P2] `_pip_residual_ri`'s carrier de-chirp is NOT row-banded, contradicting the memory audit's own claim
`_lens_traced.py:8566` (`_r = np.asarray(E_in) * np.exp(-1j * _k * _pip_remap_W)`)

`AUDIT_TRACED_MEMORY_2026_08_09` §5.7 asserts: *"the sibling `_pip_residual_ri` in the same
closure ALREADY row-bands its own exponential (`_bd = 4194304 // N`)"*. Reading the code, only
the **`_resid_eik`** exponential (`:8589-8593`) is banded. The **carrier** de-chirp on `:8566`
is whole-grid and builds four full-size arrays: `_k*_pip_remap_W` (float64),
`-1j*(…)` (complex128), `np.exp(…)` (complex128) and the product `_r` (complex128) —
≈ 2.15 + 4.30 + 4.30 + 4.30 = **15.0 GB of transient at n_fine = 16384** for a 4.30 GB answer.
`_a = np.abs(_r)`, `np.maximum(_a, 1e-300)` and the `np.where` add three more full grids, and
`np.real(_r).astype(np.float64)` / `np.imag(_r).astype(...)` each copy a strided view into a
fresh contiguous grid. The band idiom three lines below needs 67 MB.
This is on the chain's default configuration (`preserve_input_phase='remap'`).
Fix: extend the existing `for _b0 in range(0, N, _bd)` loop upward to cover the carrier
de-chirp and the normalisation, writing the two float64 outputs band by band.

---

### [P3] The `on_noncollimated='delegate'` dropped-kwarg report omits four physics-affecting knobs, and can be completely silent
`_lens_traced.py:8675-8725`

Comparing the signature against what is forwarded to `apply_real_lens` (`:8732-8739`) and what
the `_dropped` list enumerates (`:8676-8698`), these are **neither forwarded nor reported**:

```
dy, n_workers, min_coarse_samples_per_aperture, on_undersample, on_noncollimated,
on_aperture_beam, on_fit_domain_basis, on_pool_memory, beam_centre, parallel_amp,
parallel_amp_min_free_gb, newton_amp_mask_rel, newton_mask_dilate_coarse_px,
fast_analytic_phase, use_gpu, caustic_ray_subsample, caustic_band,
caustic_min_area_ratio, origin
```

Of these, `newton_amp_mask_rel`, `newton_mask_dilate_coarse_px`, `beam_centre` and
`fast_analytic_phase` change the returned field, which is exactly what the list's stated
purpose is (`:8641-8647`: "Report the non-default ones instead of discarding them silently").

Measured (`TR-MAIN-1/p7_delegate.py`): a delegating call passing
`newton_amp_mask_rel=0.0, beam_centre=(1e-4, 0), newton_mask_dilate_coarse_px=7,
fast_analytic_phase=True` emits **zero warnings** — the model swap itself is unannounced,
because the emitter is gated on `if _dropped or carrier is not None` (`:8699`).
Delegate dtype/shape parity is correct (complex128 → complex128, complex64 → complex64,
bitwise equal to a direct `apply_real_lens` call).

### [P3] `return_screen=True` + `on_noncollimated='delegate'` returns an input-dependent field where a reusable screen was requested
`_lens_traced.py:8732-8739`

Measured: the returned array is bitwise equal to `apply_real_lens(E_in)`, i.e. it contains
`E_in`. Only a RuntimeWarning (which does name `return_screen`) distinguishes it. A caller who
caches it as a screen and multiplies later fields by it gets `E_in` baked in.
`prepare_real_lens_traced` protects itself only for `carrier is not None` (`:13180`).
Fix: raise when `return_screen and on_noncollimated == 'delegate'`, as `origin` already does
at `:8075-8082`.

### [P3] `on_noncollimated='off'` does not save the cost it claims
docstring `:7079` ("'off' disables the check (and its one-FFT-free cost)"); code `:9507-9509`

When the F1 guard is skipped, the `else:` launch branch recomputes the identical
`_input_tilt_stats(E_in, wavelength, dx)` for the tilt warning. Measured at N=1024, sub=4,
median of 4: `on_noncollimated='warn'` 4.013 s vs `'off'` 4.209 s — no saving.
cProfile puts `_input_tilt_stats` at 0.310 s of a 5.760 s call (**5.4 %**), plus
`_input_beam_amp_radius` at 0.092 s (1.6 %), both for warnings only.
Fix: gate the tilt-warning block on the same policy, or document that `'off'` is a
suppression knob rather than a cost knob.

### [P3] The coarse→fine OPL upsample extrapolates the last `sub−1` rows/columns with `mode='nearest'` (constant)
`_lens_traced.py:11404-11428`

`coords = arange(N)/sub` reaches `(N-1)/sub`, but the coarse lattice only has
`Ns = ceil(N/sub)` samples covering fine indices `0 … (Ns-1)·sub`. The trailing
`N-1-(Ns-1)·sub` rows and columns are clamped to the edge coarse value.

Measured (`TR-MAIN-1/p1_registration.py`, f = 100 mm defocus, dx = 2 µm):

| N | sub | interior max err | **max err incl. trailing band** | trailing rows/cols | % of pixels |
|---|---|---|---|---|---|
| 256 | 8 | 0.64 nm | 69.2 nm | 7 | 5.39 % |
| 512 | 8 | 0.64 nm | **140.8 nm** | 7 | 2.72 % |
| 1024 | 8 | 0.64 nm | — | 7 | 1.36 % |

In practice the aperture mask (`:12397`) and the Newton out-of-domain NaN
(`xe²+ye² > (0.99·launch_radius)²`) usually kill those pixels, but nothing *enforces* that —
an apertureless prescription whose beam reaches the grid edge keeps them.
Fix: clamp the coordinate to `Ns-1-ε`, or pad the coarse array by one linearly-extrapolated
row/column, or simply NaN the trailing band so it is visibly excluded.

### [P3] The R7 order-3 upsample has a spline-prefilter boundary transient, and its comment describes the opposite of the code
`_lens_traced.py:11409-11428`

Comment: *"Cubic needs a prefilter so the NaN 0-fill cannot bleed across the ray-domain
boundary; `map_coordinates` with `prefilter=False` on a 0-filled array … keeps the boundary
crisp."* The code passes `prefilter=(_opl_up_order > 1)` — i.e. **`True`** for the cubic path.
The prefilter is an IIR filter (pole `2−√3 ≈ 0.268`), so it is precisely what makes the
0-fill and the `mode='nearest'` constant extension bleed, with a transient decaying 3.73× per
coarse cell.

Measured (`TR-MAIN-1/p1b_cubic_edge.py`, N=256, sub=8, dx=2 µm), max error inset by *m* coarse
cells from the lattice edge:

| inset (coarse cells) | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| **pure 1 mrad tilt**, order 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 nm |
| **pure 1 mrad tilt**, order 3 | 1.3603 | 0.3645 | 0.0977 | 0.0070 | 0.0000 nm |
| **f=100 mm defocus**, order 1 | 0.6400 | 0.6400 | 0.6400 | 0.6400 | 0.6400 nm |
| **f=100 mm defocus**, order 3 | 7.0903 | 1.8998 | 0.5091 | 0.0365 | 0.0002 nm |

So order 3 is the right choice deep inside (0.0002 vs 0.64 nm) and **worse than order 1 within
the outermost ~2 coarse cells**, including on a function order 1 reproduces exactly.
Fix: `mode='mirror'` (or `spline_filter(..., mode='mirror')` then `prefilter=False`), or pad
the coarse lattice by ≥4 cells of smooth extrapolation before prefiltering. Reference:
M. Unser, "Splines: A Perfect Fit for Signal and Image Processing", *IEEE Signal Processing
Magazine* **16**(6):22–38 (1999), §IV on the anti-causal boundary initialisation.

### [P3] Newton amp-mask rim pixels carry an arbitrarily wrong phase (quantified); and the mask knobs are inert on the default path
`_lens_traced.py:10991-11006` (`_build_newton_mask`), `:11425-11456` (0-fill + `>0.5` NaN threshold)

`opl_coarse` is 0-filled where masked, interpolated, and then only the pixels with
`nan_full > 0.5` are re-NaN'd — so a fine pixel whose NaN weight is ≤ 0.5 keeps
`(1-w)·opl_valid`, i.e. an error of up to 50 % of the *absolute* on-axis-referenced OPL.

Measured as the difference against the unmasked (`newton_amp_mask_rel=0.0`) field
(`TR-MAIN-1/p8b_mask.py`, plano-convex R=200 mm, ap=8 mm, N=512, sub=8, `inverse_map=False`):

| input | dilate | pixels zeroed by the mask | pixels with >1 mrad phase error | their share of total power | max |Δφ| |
|---|---|---|---|---|---|
| top-hat r=2.4 mm | 0 | 13828 (1.2e-8 of power) | 7488 | **2.24e-8** | 3.140 rad |
| top-hat r=2.4 mm | **2 (default)** | 1860 | 1904 | **3.74e-10** | 3.120 rad |
| top-hat r=2.4 mm | 4 | 0 | 332 | 6.30e-14 | 3.062 rad |
| gaussian w=1.2 mm | 2 (default) | 1280 | 4020 | 6.83e-10 | 3.132 rad |

So the documented dilation **does** work — the wrong-phase ring sits at |E| ≤ 1.5e-4 of peak —
but the errors there are up to π, not small, so this is a hard 1e-9-of-power floor for
far-halo / stray-light work. The phase is never NaN and never inf in the returned field.

Separately: with the inverse-characteristic evaluator engaged (**the shipped default** at
`sub > 1`), `newton_amp_mask_rel` and `newton_mask_dilate_coarse_px` are completely inert —
fields for `mask_rel ∈ {0, 1e-4}` × `dilate ∈ {0, 2, 4}` are bitwise identical (max Δφ
5.5e-17 rad). That inertness is undocumented, and the file otherwise has a whole policy knob
(`on_fit_domain_basis`) devoted to announcing exactly this class of dead-knob.

### [P3] Dead / misleading code in the audited range
* `:8266` — `float(sum(thicknesses))`: a statement whose value is discarded. If it is meant as
  a validation-by-exception, say so; otherwise delete.
* `:8175-8181` — `_warn_if_aperture_exceeds_grid` wrapped in a bare
  `except (KeyError, ValueError, TypeError, AttributeError): pass`. A pre-flight guard that
  silently disables itself on any malformed prescription is the class of thing the rest of
  this file refuses.
* `:8869` — `if fast_analytic_phase and preserve_input_phase:` inside a branch already gated on
  `_use_parallel_amp = (preserve_input_phase and parallel_amp)`; the second conjunct is dead.
* `:8826-8851` — in the `fast_analytic_phase` sub-branch no thread pool is ever created, yet
  the psutil/RAM gate has already run and the progress budget has been switched to
  `hi=0.50`. Cosmetic but it means "parallel_amp engaged" is reported for a serial call.
* The function is ~5700 lines (`6731–12449`) with a 1037-line docstring; the body carries at
  least nine independently-gated feature families (`caustic`, `origin`, `remap`, `ray_density`,
  `inverse_map`, C6/C11/C12, banding, pool). Several blocks are pure bookkeeping for one
  another. The `TRACED_LAYER_MAP.md` split is the right idea; the argument-validation prologue
  (`7768–8200`, ~430 lines) in particular is mechanically extractable into one
  `_resolve_traced_args(...)` returning a frozen options object, which would also make the
  delegate-drop list derivable instead of hand-maintained.

---

## Performance opportunities (estimated gain; how measured/estimated)

| # | change | gain | basis |
|---|---|---|---|
| 1 | Scale `parallel_amp_min_free_gb` with `E_in.nbytes` instead of a flat 48 GB | **1.35× wall** at N=1024 on any box with < 48 GB free (currently 1.00×) | measured, median of 4, `p12b.py` |
| 2 | `np.abs(E_analytic[::sub, ::sub])` instead of the full-grid `np.abs` on the `sub>1` preserve path | −8.59 GB peak and −9.6 s at N=32768; −14 ms / −8.4 MB at N=1024 | measured microbench + code path trace |
| 3 | Cache `phase_analytic_lens` (input-independent by the file's own `:7760-7764`) keyed on `(rx, λ, dx, N, carrier)` | removes the **entire** second ASM pass — measured 1.0 s of a 5.76 s call at N=1024 (cProfile: `_apply_real_lens_impl` 1.235 s tottime for 2 calls), plus the 17.18 GB `ones_like` at N=32768 | cProfile + microbench |
| 4 | Row-band the carrier de-chirp in `_pip_residual_ri` (`:8566`) | −15.0 GB transient at n_fine=16384 | array-size arithmetic; the band idiom already exists 3 lines below |
| 5 | Fix `fast_analytic_phase` (P1) | the documented 24–53 % on the amp stage becomes reachable | REAL_LENS_CHANGES §2c / v4 changelog |
| 6 | Skip `_input_tilt_stats` when `on_noncollimated='off'` **and** the tilt warning is not wanted | 5.4 % of the call (0.310 s of 5.760 s at N=1024) | cProfile |

cProfile top sites, N=1024, `ray_subsample=4`, `parallel_amp=False`, warm caches
(5.760 s total, `p12_perf.py`):

```
1.301 s   5 calls  _lens_traced.py:3181  _Cheb2DEvaluator.ev_value_and_grad   (numba backend confirmed)
1.235 s   2 calls  _lens_real.py:4524    _apply_real_lens_impl                (the two amp legs)
0.610 s   1 call   _lens_traced.py:6731  apply_real_lens_traced               (own body)
0.458 s   3 calls  _lens_traced.py:2964  _Cheb2DEvaluator.__init__            (the three forward fits)
0.310 s   1 call   _lens_traced.py:4260  _input_tilt_stats                    (diagnostic only)
0.274 s  2832 calls c_einsum                                                  (ray trace)
0.238 s   4 calls  lenses.py:180         surface_sag_general
0.221 s   1 call   _lens_imap.py:744     eval_into
0.168 s  36 calls  ndarray.copy
0.110 s   4 calls  numpy roll
0.099 s   3 calls  np.angle
0.092 s   1 call   _input_beam_amp_radius
0.087 s   2 calls  _ifft2
0.083 s   2 calls  _fft2
```

Memory census at N=1024, `ray_subsample=4` (tracemalloc peak / `8·N²` = 8.39 MB):
traced serial **18.0 grids**, traced `parallel_amp=True` **26.0 grids**,
`apply_real_lens` alone 13.0 grids.

Final-assembly temporaries (`:12362-12371`) materialise three full grids in sequence —
`delta_phase` (float64), `phase_exp` (complex128), `E_out` (complex128) — each freed at its
consumer. `|E_analytic|·exp(iΔφ)` is **not** fused; the row-band path
(`sag_chunk_rows`, AUTO at N ≥ 4096) is the existing mitigation and it is correct.

---

## Alternative algorithms / methods

1. **The plane-wave analytic pass is not needed for the phase reference at all.**
   It supplies only `φ_analytic_lens`, which the assembly immediately subtracts; `E_analytic`
   already supplies the amplitude *and* the transported input phase. Three cheaper routes, in
   increasing order of change: (a) cache it (it is input-independent — the file says so);
   (b) fix `fast_analytic_phase` and let the sag-screen sum supply it (P1 above);
   (c) note that `preserve_input_phase=False`/`'remap'` already skips it entirely — the
   chain's default configuration therefore never pays it, and the *default* configuration
   pays it on every call.
2. **Amplitude from the ray Jacobian** (`|E_in|/√|det J|`) blended with a diffraction envelope
   is already implemented as `amplitude_model='ray_density'`; it removes the need for a
   *complex* analytic result, leaving only a real envelope. The obstruction to making it the
   default is caustics, which `caustic='multibranch'/'uniform'` already handle (Chester–
   Friedman–Ursell / Maslov). Ref: Kravtsov & Orlov, *Caustics, Catastrophes and Wave Fields*,
   Springer 1999, §3; Ludwig, "Uniform asymptotic expansions at a caustic", *Comm. Pure Appl.
   Math.* **19** (1966) 215.
3. **Do not replace Newton with a scatter-to-grid deposition.** The two already-shipped
   alternatives measure no better: `inversion_method='fit'` (scattered exit-coordinate
   Chebyshev) gives **11.63 nm rms** on my f/7.5 fixture against the inverse-characteristic
   evaluator's **0.000 nm**, and an inverse-distance / NUFFT deposition would add hull and
   density artefacts that a global fit does not have. The productive direction is the
   opposite: make `_lens_imap` the only `sub > 1` path and delete the coarse-Newton +
   `map_coordinates` chain (which is also the file's #2 profile bucket at production N).
4. **Zernike instead of total-degree Chebyshev for the forward fit.** The whole
   `fit_radius_beam_factor` / `_CARRIER_FIT_RADIUS_FRAC` / C11-arbiter / C12-predictor
   apparatus exists because a total-degree Chebyshev fit on a square couples out-of-beam
   marginal rays into the low-order (defocus) coefficients. Zernike circle polynomials are
   orthogonal on the disc, so restricting the domain to the beam disc changes only the
   normalisation, not the coefficients of the retained modes — the coupling the guard exists
   to suppress is absent by construction. Refs: Born & Wolf, *Principles of Optics*, 7th ed.
   §9.2; V. N. Mahajan, "Zernike annular polynomials for imaging systems with annular pupils",
   *JOSA* **71** (1981) 75. Cost: identical (one least-squares solve); the analytic gradients
   needed by Newton are standard (Janssen & Dirksen recursions).
5. **`NA_exit` estimator.** Instead of `max |(L,M)|` over launch rays, use the
   amplitude-weighted 99.9th percentile of `|∇(k₀·opl_map)|/k₀` evaluated on the **exit**
   grid inside the output mask. That is the quantity the Nyquist statement is actually about
   (`|sin θ| > λ/2dx` of the *returned* field), it cannot be contaminated by rays the output
   mask deletes, and it costs one gradient of an array the function already has.
6. **`_geometric_lens_phase` piston in float32** — if the real-dtype wiring is kept, reduce
   `k0·n·t` modulo 2π *before* the accumulation (in float64) rather than at the end. Standard
   practice; it removes the 1.95e-3 rad ulp entirely.

---

## Code organization observations

* `apply_real_lens_traced` is 5722 lines including a 1037-line docstring. The docstring is
  itself a design document (it contains measured tables, audit IDs and refutation notes);
  much of it belongs in `REAL_LENS_CHANGES.md`, and the parameter list should be a reference
  table with the derivations linked out.
* The prologue `7768–8200` is pure argument resolution across ten independent knobs, several
  of which cross-validate (`caustic`↔`amplitude_model`↔`use_gpu`↔`output_plane_distance`,
  `origin`↔`preserve_input_phase`↔`on_noncollimated`↔`caustic`). Extracting it into one
  `_resolve_traced_args()` would let the delegate branch derive its dropped-kwarg list from the
  resolved options rather than from a hand-maintained tuple (the source of the P3 above), and
  would make the "which knob is inert on which path" question answerable in one place — the
  file currently answers it with three different mechanisms (`on_fit_domain_basis`, the
  `_dropped` list, and silence).
* Comment-to-code ratio in the audited range is roughly 3:1, and at least two comments are now
  factually wrong about the code beneath them (`:11418-11422` prefilter, `:9710-9715` NaN
  "extrapolation"). Long audit-trail comments have real value here, but they need the same
  regression discipline as the code — a stale one is worse than none, because the next reader
  trusts it instead of the two lines below it.
* `dx`/`dy`: the function requires `dy == dx` (`:8195-8199`) yet threads `_dy_eff = dy if dy is
  not None else dx` through the NA guard (`:9646-9648`). Dead generality.
* `_TracedExitSupport` (C14) is a genuinely good consolidation — three previously duplicated
  hull/finiteness computations behind one object with per-view rules. More of the file should
  look like that.

---

## Unverified suspicions

* **`use_gpu=True` accuracy.** By code reading, `_imap_domain_gate` excludes `use_gpu`, so the
  GPU path always takes the order-1 coarse upsample. If the CPU numbers transfer, a GPU call at
  the default `ray_subsample=8` carries ~11.6 nm rms where the CPU call carries 0. Not testable
  here (no cupy). Confirm by running the P2 fixture with `use_gpu=True` on a CUDA box.
* **`min_coarse_samples_per_aperture` default action is `'error'`** and its stated rationale
  ("the cubic-spline interpolation of the wavefront will alias") describes a step the default
  path no longer performs. The guard still bounds the *ray* sampling, which is the real
  constraint, so I could not construct a false refusal — but the rationale and the threshold
  were calibrated against the upsample, so the 32-sample floor may now be mis-tuned in either
  direction. Confirm with a sweep of `n_coarse_across` against the oracle with the imap on.
* **Pool bit-identity.** `_spline_data['cheb_fit']`/`cheb_backend` pinning claims pool == serial
  by construction. I only exercised `n_workers=1`; a genuine pool run (≥200 000 Newton points)
  was outside the time budget.
* **`origin != 0`** paths (niche D9) were desk-checked only; the ray-density + remap combination
  they require needs a chain fixture I did not build.
* `np.real(_r).astype(np.float64)` at `:8598-8599` copies a strided view; whether numpy elides
  the copy for an already-float64 non-contiguous view was not measured.

---

## Checked and found correct (brief)

* **Coarse-lattice registration is exact.** For N ∈ {255, 256}, sub ∈ {1,2,4,8}, and
  `origin ≠ 0`, `X[::sub, ::sub]` samples coincide with wave-grid pixel centres to **0.0 m**,
  and `coords = arange(N)/sub` maps coarse index *u* to fine index *u·sub* exactly (the
  v5.x fix for non-divisor `sub` is right). `p1_registration.py`.
* **No half-pixel mismatch between the amplitude leg and the ray leg**: `_lens_real.py:6823`
  and `_lens_traced.py:8312` both use `(arange(N) - N/2)*dx`.
* The **ray launch grid** `linspace(-launch_radius, launch_radius, n_launch)` with `n_launch`
  forced odd is exactly symmetric about the axis with an on-axis sample — so the `i_axis`
  piston reference is an exact ray, as claimed.
* **Traced OPD vs an independent exact-sphere sequential-Snell oracle** (written from scratch
  in `common.py`, not `lumenairy.raytrace`): plano-convex f/32 singlet, plane wave,
  `|residual| ≤ 2.4e-10 rad ≈ 5e-17 m` at N=512 and N=1024, `ray_subsample` 1 and 8.
  RMS after piston+tilt removal: **0.00000 nm**. `p2b.py`.
* **Absolute piston is right and the two models are coherently combinable.** Traced field
  phase at the axis vs `k0·OPL_oracle(0)`: **1.04e-12 waves** (sub=8), 6.41e-7 waves (sub=1).
  `piston(traced) − piston(apply_real_lens)` = **+4.4e-4 waves** (N=512) / **−4.0e-5 waves**
  (N=1024) — a Mach-Zehnder combination of one arm through each model is meaningful.
  `p2_opd_oracle.py`.
* **Exit-vertex signed transfer (`t = −z/N`) is correct for both signs.** Negative meniscus
  (R₂ = −25 mm, 260 µm ≈ 200 waves of OPL at stake): residual **0.0042 nm** max, 0.0015 nm rms
  at sub=1, < 1e-5 nm at sub=8. Biconvex (concave exit): 0.00088 nm / 0.00035 nm.
  `p5_meniscus.py`.
* **`preserve_input_phase=True` does not double-count the input phase.** Through a zero-power
  4 mm plate, the traced-minus-analytic correction is *identical* for a collimated,
  R=−50 mm converging, R=−20 mm converging and 5 mrad tilted input
  (mean −8.093e-6 rad, std 1.980e-3 rad, max dev 2.89e-2 rad in every case;
  `|E|` relative difference 4e-17). The correction factor is input-independent, as intended.
  `p3_pip.py`.
* **Carrier forms are mutually consistent.** Against the oracle on a 200 mm diverging source:
  `carrier=200e-3`, `'auto'` and `TiltedCarrier(200e-3,0,0,0,0)` all give 1.845e-3 rad rms
  (0.385 nm); the exact-sphere `W = sign(s)(√(r²+s²) − |s|)` and its analytic gradient are
  used consistently by the reference leg, the launch cosines and the H6 eikonal.
  (`ndarray` is the exception — see P2.) `p6_carrier.py`.
* **`parallel_amp` is numerically safe**: `parallel_amp=True` vs `False` is **bitwise identical**
  (max |Δ| = 0.0, relative L2 = 0.0) and repeatable run-to-run. No shared-mutable-state
  interference observed between the two `apply_real_lens` calls. `p9_11_parallel_dtype.py`.
* **complex64 end-to-end is clean.** Output dtype is preserved; phase error vs complex128 is
  **1.93e-6 rad rms / 1.52e-4 rad max** (0.0004 / 0.032 nm), amplitude 5.5e-8 rms relative.
  The absolute OPL (2.91e4 rad) never enters float32: `opl_map` is float64, the piston is
  applied as a complex128 unit phasor (`:11718-11720`), and the only float32 quantity is the
  wrapped `phase_analytic_lens` (ulp 3.7e-7 rad). `p9_11_parallel_dtype.py`.
* **`newton_poly_order=6` carries A8/A10 aspheric content.** Conic −0.6 plus A4/A6/A8/A10 with
  an 11.0 µm departure at r = 3.24 mm, measured at `sub=1` against an asphere oracle with
  Newton surface intersection: order 4 → 2.99 nm rms, **order 6 → 0.103 nm rms / 0.33 nm max**,
  order 8 → 0.0025 nm, order 10 → 0.00004 nm. At a 1.14 µm departure order 6 gives 0.002 nm.
  `p13_asphere.py`, `p13b.py`.
* **Enum membership guards all raise as documented** for `on_noncollimated` (with the
  `'silent'`/`'ignore'` → `'off'` aliases), `inversion_method`, `on_fit_domain_basis` (with
  `'ignore'`/`'off'` → `'silent'`), `on_pool_memory`, `amplitude_model`, `caustic`,
  `preserve_input_phase`, `remap_sampling`, `inverse_map`, `origin`, `beam_centre`,
  `decentred_fit_poly_order`, `fit_radius_beam_factor`, `on_aperture_beam`.
* The **mirror-in-`surfaces`** guard, the **fold** guard, the **square-grid / `dy == dx`** guard
  and the `stop_index != 0` / decentred-stop warnings all fire on construction as written.
* The **delegate** return is dtype- and shape-correct and bitwise equal to a direct
  `apply_real_lens` call for both complex128 and complex64 inputs.

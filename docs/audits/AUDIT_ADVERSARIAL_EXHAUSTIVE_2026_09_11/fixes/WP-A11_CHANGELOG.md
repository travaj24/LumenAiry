# WP-A11 changelog text (polarization / coatings / sources / algebra / infrastructure)

Assembled from findings Z1-Z4 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` Section 10 plus the
ORCHESTRATOR `coating_reflectance(polarization='te')` row.  Every number below was re-measured on
this branch; the "before" readings come from running the audit's own repro scripts
(`repro/POLAR-SOURCES-INFRA/p2_coatings.py`, `p8_memory.py`, `p9_infra.py`, `p10_algebra.py`) on
HEAD before the change, and the "after" readings from re-running them.

---

### Fixed -- coatings: `polarization='te'` applied the **TM** coefficient to a TE beam, and every unrecognised spelling did the same, silently (Z1, P1)

`coating_reflectance` and its JAX twin never normalised or validated the polarization argument.
Both branched on the literal `pol == 's'` and sent *everything else* down the p branch --
`'te'`, `'S'`, `'tm'`, `'P'`, `'banana'`, `''`.  `CONVENTIONS.md` Section 7 states the opposite
and names this file: *"`s` == `te` ... both aliases are accepted everywhere (case-insensitive) |
`rcwa.py::_normalize_pol`, `coatings.py`"*.  `rcwa._core._normalize_pol` does implement that
contract and raises on junk; `coatings.py` did neither.

Measured on the audit's fixture (50 nm MgF2 on n = 1.52 at 60 deg, 550 nm), `R` by spelling:

| `polarization=` | before | after |
|---|---|---|
| `'s'` | 0.15427715 | 0.15427715 |
| `'te'` / `'S'` / `'TE'` | **0.00322899** (p branch) | **0.15427715** (s branch) |
| `'p'` / `'tm'` / `'P'` / `'TM'` | 0.00322899 | 0.00322899 |
| `'avg'` | 0.07875307 | 0.07875307 |
| `'banana'`, `''`, `'ss'`, `0`, `None` | **0.00322899** (p branch, silent) | **`ValueError`** naming `{'s','te','p','tm','avg'}` |

Reachable from the public wave path: a `{'type': 'coating', 'polarization': 'te'}` element in
`propagate_through_system` returned `|t| = 0.9969635` (the TM coefficient) for a TE beam where the
TE answer is `0.94840576` -- 5 % in amplitude, 10 % in power at this angle, and order-unity near
Brewster (`R_s = 0.154` vs `R_p = 0.003`).  `p9_infra.py` section I now reports
`te == s ? True   te == p ? False` (was `False` / `True`).

The JAX twin carried the identical defect, so a gradient-based AR/HR design optimising
`polarization='te'` optimised the TM stack: measured `dR/d(thickness)` under `'te'` is now exactly
the `'s'` gradient (-1052475.33 1/m) and differs from the `'tm'` one.

Both entry points now route through a shared `_normalize_coating_pol` helper that delegates the
te/tm/s/p alias table to the RCWA `_normalize_pol` (single source of truth -- a regression test
asserts the two accept exactly the same set, plus the coatings-only `'avg'`) and raises a
`ValueError` with the Section 2 `f"{fn_name}: ..."` prefix on anything else.

**Migration.**  A call that passed a typo, an empty string or any spelling outside
`{'s','te','p','tm','avg'}` used to return the p-polarised result and now raises.  Callers that
meant `'p'` should say so; callers that meant `'te'` were getting the wrong physics and now get
the right one.  `'s'`, `'p'` and `'avg'` are bit-identical to before (the TMM itself is untouched:
the audit's Airy / Rouard oracles still agree to `<= 6.7e-16` on R and `<= 7.8e-16` on T,
including absorbing exit media and past-critical TIR).

Files: `lumenairy/elements/coatings.py`.
Tests: `tests/unit/test_audit2609_a11_polar_sources_infra.py::test_z1_*` (24 cases: ten spellings
against an in-test analytic Airy oracle, nine junk rejections, the RCWA-equivalence pin, the
`propagate_through_system` path, and four JAX-twin pins including the gradient).

---

### Fixed -- sources: Gaussian-Schell / Schell ensembles realised the grid-PERIODISED coherence kernel, so opposite edges of the grid came out ~99 % coherent (Z2, P1)

`_schell_phase_realizations` filtered complex white noise with `H(k) = exp(-|k|^2 sigma_g^2 / 4)`
on the FFT grid and inverse-FFT'd it.  An FFT filter is a **circular** convolution, so by Poisson
summation the two-point correlation it realises is the periodised Gaussian
`sum_m exp(-|d + mL|^2 / (2 sigma_g^2))` with `L = N dx`, not the `exp(-|d|^2/(2 sigma_g^2))` the
module header, both factory docstrings and the Schell-model contract all promise.  Nothing in the
code or docs mentioned the window and no guard fired.

Measured (N = 64, dx = 1 um so L = 64 um; pair-averaged linear correlation over 20 000
realisations; the escape hatch `pad_sigma=0.0` reproduces the pre-fix path bit-for-bit, so both
arms come from the same run):

| sigma_g / L | max abs(mu - Gaussian), before | after | mu at the largest separation (N-1)dx, before | after | true value |
|---|---|---|---|---|---|
| 0.313 | **0.9949** | 0.0081 | **1.0019** | 0.0151 | 0.0070 |
| 0.125 | **0.9922** | 0.0036 | **0.9922** | -0.0036 | 3.4e-14 |
| 0.031 | **0.8836** | 0.0007 | **0.8836** | 0.0007 | 3.4e-216 |

The wrap manufactured *long-range* coherence between opposite edges of the grid -- the failure mode
that most looks like a real physical result.  With the audit's own single-reference estimator
(40 000 realisations, `p6_gsm_grid.py`) the departure from the documented Gaussian at
`sigma_g = L/3` was **0.2692** of peak and is now **0.0041** (Monte-Carlo floor).

Downstream, the coherent-mode spectrum of the `return_kind='mcf'` form (N = 20, dx = 2.5 um,
w0 = 20 um, sigma_g = 10 um, 4000 realisations) against the Starikov & Wolf (1982) analytic
eigen-spectrum `[0.38202, 0.14592, 0.14592, 0.05574, 0.05574, 0.05574]`:

| | leading six eigenvalues | n=2 shell mass | degeneracy spread within the shell |
|---|---|---|---|
| before | `[0.40081, 0.14175, 0.13891, 0.07445, 0.06990, 0.05077]` | 0.19512 (**+16.7 %**) | **36.4 %** |
| after | `[0.39205, 0.15411, 0.14527, 0.05830, 0.05597, 0.05424]` | 0.16851 (+0.8 %) | 7.2 % |

(the n = 2 shell is exactly 3-fold degenerate in the analytic answer).

**What ships.**  The noise is drawn and filtered on a grid padded by `4 * sigma_g` per side
(rounded up to an FFT-fast length) and the central window cropped.  The nearest wrapped image then
sits at least `8 sigma_g` away, leaving `exp(-32) = 1.3e-14` of residual -- below the float64 noise
floor of the kernel itself.  The noise is drawn over the *whole* padded grid rather than zeroed
outside the window, so `phi` stays statistically stationary (zero-padding the noise itself would
taper the variance within `sigma_g` of the crop edge).  The deterministic unit-mean-intensity
normalisation (v5.4.6 P3-10) is computed on the padded grid, which is the right constant for the
cropped window because that grid is homogeneous: measured `E[<|phi|^2>]` = 0.9967-0.99999.

### Changed -- sources: Gaussian-Schell / Schell ensembles are no longer bit-identical for a given `rng` seed, and warn when the grid is too short for the coherence length (Z2)

Fixing the kernel changes the realisations a seed produces (the RNG is now drawn on the padded
grid).  This is a deliberate change of a WRONG default; `_schell_phase_realizations(pad_sigma=0.0)`
reproduces the pre-v5.46 path bit-for-bit for reproducing an archived ensemble, and is pinned as
such.  The public factories have no new kwarg.

`create_gaussian_schell_source` / `create_schell_model_source` now emit a `UserWarning` when
`sigma_g > N*dx/6`: the kernel is correct there, but fewer than six coherence cells fit across the
grid, so the ensemble estimate of any two-point quantity is aperture-dominated (the degeneracy
breaking above was measured at `sigma_g = L/5`), and the anti-wrap pad costs `~((L + 8 sigma_g)/L)^2`
the FFT work.  A second `UserWarning` fires -- naming the residual periodisation it actually leaves
-- if the pad has to be capped at 4x the grid per axis (only reachable for `sigma_g > ~L/2`).

The pad is not free.  Interleaved medians of 5 runs, 64 realisations, `OPENBLAS_NUM_THREADS=1`:

| N | sigma_g/L | padded N | before | after | slowdown | peak before | peak after |
|---|---|---|---|---|---|---|---|
| 64 | 0.031 | 80 | 22.6 ms | 29.0 ms | 1.28x | 4.6 MB | 4.9 MB |
| 64 | 0.125 | 128 | 23.2 ms | 71.6 ms | 3.09x | 4.6 MB | 5.9 MB |
| 128 | 0.031 | 160 | 74.6 ms | 123.1 ms | 1.65x | 18.5 MB | 19.5 MB |
| 128 | 0.125 | 256 | 80.2 ms | 464.7 ms | 5.79x | 18.5 MB | 23.6 MB |
| 256 | 0.125 | 512 | 529 ms | 2339 ms | 4.42x | 73.9 MB | 94.4 MB |

The FFT work scales as `(1 + 8 sigma_g/L)^2`; the resident ensemble dominates the peak, so memory
grows only ~28 %.  Gori's pseudo-mode representation would remove both the window and the FFT cost
-- see the deferred item in `WP-A11_REPORT.md`.

**Docstring corrections.**  `sigma_g`'s "``sigma_g >> w0`` approaches the coherent limit" steered
users straight into the bad regime, since a typical grid is `L ~ 4 w0` (so `sigma_g >> w0` means
`sigma_g > L/6`).  It now states the global degree of coherence `q = sigma_g/w0` as the physics and
`dx << sigma_g <= N dx/6` as the grid constraint, and says to enlarge `N` rather than shrink `w0`.

Files: `lumenairy/sources/core.py`.
Tests: `tests/unit/test_audit2609_a11_polar_sources_infra.py::test_z2_*` (five: the kernel against
both hypotheses with the pre-fix arm measured in-process, unit mean intensity, the `pad_sigma=0.0`
bit-identity against a local restatement of the old body, the `L/6` warning and its silent
counter-arm, and both public factories);
`tests/unit/test_s3_7_broadcast_grid.py::test_schell_phase_realizations_kxky_bit_identical_{unpadded,padded}`
(the S3-7 broadcast-view orientation pin, split so it covers both paths).

---

### Fixed -- memory: `estimate_lens_memory(lens_model='real')` under-predicted `apply_real_lens`'s peak by up to 2.8x (Z3, P2)

The `'real'` branch reused the *traced* calibration and then scaled the float64 core DOWN by
`5 / _LENS_F64_ARRAYS`, on the documented reasoning that the bare entry point "omits the traced
final-assembly float64 arrays".  Measured, it does not.  A pre-flight budget computed with the
model the docstring names for that entry point under-reserved -- the exact failure
`check_sim_memory` exists to prevent.

Re-derived from `tracemalloc` on `apply_real_lens` itself (N-BK7 biconvex singlet R = +-50 mm,
d = 5 mm, 25 mm aperture, 633 nm, caches warmed, whole-grid mode).  The peak is pure `N^2` -- no
fixed term -- so bytes/pixel is the whole calibration, and two dtypes give an exact two-unknown
solve on the `N >= 1024` asymptote: **7.00 complex full-grid arrays + 8.00 float64 ones**, of which
2 float64-equivalents are the `phase_exp` complex128-first transient the shared term already
models.  The shipped constants carry ~7 % margin so the estimate BOUNDS the measurement.

`est / measured` (the audit's own `p8_memory.py`, plus N = 512 and complex64):

| N | dtype | measured peak | before (`parallel_amp=False`) | before (`=True`, default) | after |
|---|---|---|---|---|---|
| 512 | c128 | 46.8 MB | 0.39 | 0.68 | **1.06** |
| 1024 | c128 | 184.6 MB | 0.36 | 0.63 | **1.07** |
| 2048 | c128 | 738.3 MB | 0.36 | 0.63 | **1.07** |
| 512 | c64 | 31.5 MB | 0.52 | 0.91 | **1.07** |
| 1024 | c64 | 125.9 MB | 0.52 | 0.91 | **1.07** |
| 2048 | c64 | 503.4 MB | 0.52 | 0.91 | **1.07** |

`estimate_asm_memory` is untouched (the audit verified it over-predicts, i.e. is already fail-safe).

### Changed -- memory: `parallel_amp` is inert for `estimate_lens_memory(lens_model='real')` (Z3)

`apply_real_lens` has no `parallel_amp` argument, so there is nothing to switch off; the `'real'`
constants are measured on its shipped behaviour and doubling them would double-count.  The kwarg
still applies to `lens_model='traced'` and to the row-band (`sag_chunk_rows`) branch.  Before,
`parallel_amp=False` halved the `'real'` estimate to `0.36x` the measured peak.

Files: `lumenairy/memory.py`.
Test: `...::test_z3_estimate_lens_memory_real_bounds_apply_real_lens` (tracemalloc vs the estimator
at N = 512, two-sided: `1.0 <= est/measured <= 1.6`).

---

### Performance -- sources: `create_gaussian_beam` peak memory 3.00x -> 1.50x the output (complex128) and 5.00x -> 2.00x (complex64), byte-identical (Z3, P2)

It was the last factory still building a dense `np.meshgrid` -- two float64 `N^2` arrays whose rows
and columns are all identical -- and normalising out of place.  Every sibling carries the S3-7
`# broadcast views, not a dense N x N grid` comment.

Measured at N = 4096 (`tracemalloc` peak / `time.perf_counter`):

| dtype | peak before | peak after | output | ratio before -> after | time before -> after |
|---|---|---|---|---|---|
| complex128 | 805.4 MB | 402.7 MB | 268.4 MB | 3.00x -> **1.50x** | 0.705 s -> **0.326 s** |
| complex64 | 671.2 MB | 268.5 MB | 134.2 MB | 5.00x -> **2.00x** | 0.682 s -> **0.337 s** |

At N = 8192 / complex128 that is ~1.6 GB of avoidable transient.  The exponent is now built through
one real `N^2` buffer with `np.negative` / `np.exp` `out=`, and both `normalize` divides are in
place (matching `_apply_field_normalization`).  Output is **bit-identical** over 36 configurations
(N in {17, 64, 257} x three `normalize` modes x complex128/complex64 x on- and off-axis centres):
the arithmetic is elementwise and `(-S)/c == -(S/c)` exactly in IEEE.  `create_fiber_mode` and
`Source.gaussian` route through this function and inherit it.

Files: `lumenairy/sources/core.py`.
Tests: `...::test_z3_gaussian_beam_broadcast_is_bit_identical` (12 configs),
`...::test_z3_gaussian_beam_peak_over_output` (both dtypes, pre-fix arm measured in the same test),
`tests/unit/test_s3_7_broadcast_grid.py::test_gaussian_beam_broadcast_bit_identical` (adds the one
missing factory to the S3-7 census).

### Performance -- polarization: `stokes_parameters` 7.00 -> 6.00 and `degree_of_polarization` 8.25 -> 6.00 full-grid arrays, bit-identical (Z3, P2)

`stokes_parameters` recomputed `abs(Ex)**2` for S0 and again for S1, and `Ex*conj(Ey)` for S2 and
again for S3; `degree_of_polarization` then built `safe`, three ratios, three squares and two
`np.where` results on top of it.  Each quantity is now computed once and consumed, and the DOP
accumulates in place over the Stokes arrays it owns.

Measured at N = 2048 (`tracemalloc` peak, in full-grid REAL arrays of `N*N*8` B; times are medians
of the same runs):

| | peak before | peak after | time before | time after |
|---|---|---|---|---|
| `stokes_parameters` | 7.00 (234.9 MB) | **6.00** (201.3 MB) | 0.258 s | **0.110 s** |
| `degree_of_polarization` | 8.25 (276.8 MB) | **6.00** (201.3 MB) | 0.396 s | **0.198 s** |

6.00 is the floor for a bit-identical implementation (four outputs plus the one complex cross-term
the exact S2/S3 need).  Output is bit-identical over eight configurations including dark, NaN, inf
and 1e-160-underflow pixels -- the regime the E-L13/E-L17 guards exist for.  One subtlety is
recorded in the source: NumPy's vectorised complex multiply is **not** bitwise commutative here
(measured 1.8e-15 on S3 at N = 64), so `Ex` stays the left operand.

Files: `lumenairy/elements/polarization.py`.
Tests: `...::test_z3_stokes_and_dop_are_bit_identical`, `...::test_z3_stokes_and_dop_peak_arrays`.

---

### Changed -- algebra: `FreeSpace`'s default propagator is now `'asm'`, so the delivered grid agrees with the ABCD the operator reports (Z3, P2)

With the previous `method='auto'` default the dispatcher selected SAS on every far-field segment,
which RESAMPLES.  On the canonical 4f chain
`FreeSpace(f) ThinLens(f) FreeSpace(2f) ThinLens(f) FreeSpace(f)` (f = 200 mm, N = 256, dx = 8 um,
633 nm) the operator reported `abcd = [[-1, 0], [0, -1]]` -- magnification -1, so the field returns
at the input pitch -- while delivering `dx_out = 15.45 um`, **1.93x** the input; and `propagate`
raised its own `UserWarning` three times per evaluation ("a caller that unpacks this return has no
stable contract ... drop the argument"), advice the caller could not act on because the argument it
names is the algebra layer's own.

| | before | after |
|---|---|---|
| `UserWarning`s per 4f evaluation | **3** | **0** |
| `dx_out / dx_in` (ABCD says 1.0000) | **1.9318** | **1.0000** |
| power ratio | 1.0000 | 0.9995 |
| centroid +304 um -> | -304.0 um | -303.9 um |

`'auto'` remains fully supported and, when named, no longer emits the un-actionable warning either
(0 per evaluation) while still reporting its resampled pitch honestly (1.9318x) -- the algebra layer
now consumes the shape-stable `PropagationResult` on the square-grid branch.  The anamorphic branch
keeps the legacy tuple and its v5.30 `dy` contract unchanged (verified: `dx=2 um, dy=3 um` in ->
`2 um / 3 um` out, for both `'asm'` and `'auto'`).  `FourierTransform` deliberately pins `'auto'`
for its two legs (the optical FT *is* a re-gridding operation) and is unchanged.

The PITCH CONTRACT is now documented on `Operator.abcd`: a ray matrix describes `(height, angle)`,
not sampling; read the delivered pitch off the returned Source/tuple, never off `|A|`, unless every
`FreeSpace` in the chain is pitch-preserving.

**Migration.**  `FreeSpace(d)` now propagates by ASM.  Pass `method='auto'` to restore the
dispatcher-selected kernel and its resampling.  `algebra.from_prescription` keeps `method='auto'`
(measured pitch-preserving on singlet/meniscus/doublet at N = 256, dx = 8 um).

Files: `lumenairy/algebra/primitives.py`, `lumenairy/algebra/base.py`,
`lumenairy/algebra/from_prescription.py`.
Tests: `...::test_z3_freespace_default_preserves_the_pitch_its_abcd_reports`,
`...::test_z3_freespace_auto_still_available_and_no_longer_spams`,
`...::test_z3_freespace_anamorphic_dy_still_threaded`; existing pins updated in
`tests/unit/test_niche_audit_w4_p5_return_contract.py`, `tests/unit/test_v4_15_2_agent_b.py`,
`tests/unit/test_v4_15_1_agent_g_application.py`.

---

### Fixed -- deprecation warnings named lumenairy instead of the caller (Z4, P3)

`_deprecation._emit` used `stacklevel=3`, which from inside `_emit` resolves to the *public
function's own body*, and from inside `deprecated_alias._shim` to `_deprecation.py` itself.  The
`_emit` docstring's claim ("lets the warning point at the caller of the public function") was wrong
by one frame in both cases, and the v4.15.3 sweep had papered over it by passing `stacklevel=4`
explicitly at seven call sites.

`la.load_zmx_prescription(...)` called from a user script, before: `filename=_deprecation.py
line=551`.  After: the user's own file and line.  Same for the direct helpers (before: the public
function's body; after: its caller).  The default is now `_DEFAULT_STACKLEVEL = 4` with the frame
arithmetic written out; `warn_renamed_function` takes 5 because it inserts a frame of its own.  The
library's two live aliases (`load_zmx_prescription`, `load_zemax_prescription_txt`) are the only
production consumers.

Files: `lumenairy/_deprecation.py`.
Tests: `...::test_z4_deprecated_alias_warning_names_the_caller`,
`...::test_z4_live_alias_points_at_user_code`.

### Fixed -- `_check_2d_scalar_field` promised "2-D complex" but accepted object dtype and `np.matrix` (Z4, P3)

The guard tested `ndim == 2` and nothing else while every message it emits says *"expected 2-D
complex ..."*.  `np.matrix` is the dangerous one: its `*` is matrix multiplication and its `**` is
matrix power, so any kernel multiplying a field by an aperture mask, a phase screen or a transfer
function silently computed a matrix product and returned a plausible finite array.  Object dtype
reached the kernels too and ran NumPy's Python-object slow path.

| input | before | after |
|---|---|---|
| 2-D complex / float / int / bool, non-contiguous | accepted | accepted (unchanged -- a real-dtype amplitude mask is legitimate) |
| **2-D object dtype** | **accepted** | `TypeError` |
| **`np.matrix`** | **accepted** | `TypeError`, pointing at `np.asarray(M)` |
| 1-D / 3-D / 0-D / nested list | `ValueError` | `ValueError` (unchanged) |

The dtype gate is a `dtype.kind` membership test (~60 ns) because this helper runs at the entry
point of every propagator and lens call.  The module's absolute `from lumenairy.sources.core
import ...` also became the relative form the rest of the package uses (the audit's secondary nit).

Files: `lumenairy/_validation.py`.
Tests: `...::test_z4_check_2d_scalar_field_dtype_gate`,
`...::test_z4_check_2d_scalar_field_rejects_np_matrix`;
`tests/unit/test_v4_15_4_agent_b.py::test_validation_helper_lazy_import_removed` broadened to
accept either import spelling (its contract -- module scope, not lazy -- is unchanged).

### Fixed -- cache registry: a colliding name silently dropped the second clearer, and a failing clearer was invisible (Z4, P3)

`register_cache_clearer` returned early on an existing name -- intended as reload-idempotence, but
it equally dropped a *different* function registered under a colliding name, leaving that cache
permanently unclearable, in the module whose entire purpose is to retire the "fix N, miss N+1"
pattern.  `clear_all_registered_caches` then swallowed `ImportError`/`RuntimeError`/`AttributeError`
per clearer, so a caller who ran `clear_asm_caches()` to free RAM before a large allocation got no
signal that a cache had stayed full.

A collision now raises a `RuntimeWarning` naming both call sites; the first registration still wins,
so behaviour is otherwise unchanged.  Reload-idempotence is preserved by deciding "same call site"
on `(module, qualname, source file, first line)` -- which `importlib.reload` preserves and two
distinct functions (including two lambdas on different lines) do not.  The walk is still
best-effort and does not raise, but now emits one `RuntimeWarning` at the end naming every clearer
that failed and why.

Files: `lumenairy/_cache_registry.py`.
Tests: `...::test_z4_cache_registry_warns_on_a_colliding_name`,
`...::test_z4_cache_registry_is_still_reload_idempotent`,
`...::test_z4_clear_all_reports_a_failing_clearer`.

### Fixed -- `lumenairy.algebra.from_prescription` the MODULE shadowed the function of the same name (Z4, P3)

`from lumenairy.algebra import from_prescription; from_prescription(rx, 633e-9)` -- the call the
submodule's own `__all__` and docstring advertise -- raised `TypeError: 'module' object is not
callable`, because importing the submodule bound it as an attribute of the package.  Only
`Operator.from_prescription(rx, wl)` and the fully-qualified
`lumenairy.algebra.from_prescription.from_prescription` worked.  The package attribute is now
rebound to the function after the import.  The submodule stays in `sys.modules` under its dotted
name, so `from lumenairy.algebra.from_prescription import from_prescription` and
`importlib.import_module(...)` keep working; the chained
`...from_prescription.from_prescription` workaround no longer resolves -- drop the duplicated tail.

Files: `lumenairy/algebra/__init__.py`, `lumenairy/algebra/from_prescription.py`.
Test: `...::test_z4_algebra_from_prescription_is_callable` (all four access paths).

### Fixed -- `_plane_wave_carrier` centred its grid on `nx // 2` while the package uses `N / 2` (Z4, P3)

Identical for even `N`, half a pixel apart for odd `N`, so a `JonesField` built by
`jones_field_from_orders` on an odd grid sat `dx/2` off every element subsequently applied to it
(apertures, spatially-varying Jones callables).  Measured at nx = 7, dx = 1 um: the package grid is
`[-3.5 ... +2.5] um`, the carrier grid was `[-3 ... +3] um`.  The site also moved to broadcast views
(dropping two dense float64 `(ny, nx)` arrays).

Files: `lumenairy/elements/polarization.py`.
Test: `...::test_z4_plane_wave_carrier_uses_the_package_grid_centring` (nx in {7, 8, 9, 16},
cross-checked against the grid `apply_jones_matrix` hands to a callable).

### Fixed -- `user_library`: `**` and `<<` on integer literals could exhaust memory from an untrusted phase-mask string (Z4, P3)

The allowlist AST interpreter in `_safe_eval_expression` is a genuine sandbox -- the auditor could
not escape it -- but CPython's `int` is arbitrary precision, so `2 ** (10 ** 9)` asks for a 125 MB
integer (and `1 << (10 ** 9)` the same) inside a routine whose input is an untrusted user-library
string.  Nothing is compromised; the process stalls or dies.  The evaluator now bounds the RESULT's
bit length at 4096 bits (~1234 decimal digits) for the pure-Python-`int` path only, so it refuses
before allocating: measured 0.02 ms per rejection.  Float, complex and every ndarray operand are
untouched, and `X ** 2`, `2 ** 10`, `10 ** 9`, `1 << 32`, `2 ** -3`, `(-1) ** 1000000000` all still
evaluate.  `<<` is guarded alongside `**` (the audit named only `Pow`; the shift is the identical
bignum path and was one `_BIN_OPS` entry away).

Files: `lumenairy/user_library.py`.
Tests: `...::test_z4_user_library_bounds_integer_growth`,
`...::test_z4_user_library_pow_guard_passes_real_expressions`,
`...::test_z4_user_library_pow_guard_ignores_array_operands`.

### Documentation -- `JonesField.propagate*` mutate in place; two docstrings said otherwise (Z4, P3)

`propagate_fresnel` and `propagate_fraunhofer` were documented "Returns new grid spacings"; both
return `self` and write the new spacings onto `self.dx` / `self.dy`.  `propagate` documented neither
the mutation nor the return.  All four now state the in-place-and-return-`self` convention that
`sas_propagate` already did, note that the caller's original arrays are rebound rather than written,
and say which methods change the pitch.  No behaviour change.

Also documented: `coating_reflectance`'s p-polarisation reflection phase is pi from the textbook
Fresnel convention (the Macleod tilted admittance `eta_p = n/cos(theta)`), which is observable
through `propagate_through_system`'s `'reflection'` port -- the audit verified the sign is
self-consistent with `berreman_jones_1d` to 1.8e-15, but nothing said so at the call site.

Files: `lumenairy/elements/polarization.py`, `lumenairy/elements/coatings.py`.
Test: `...::test_z4_jones_propagate_methods_are_in_place_and_return_self` (pins the behaviour, not
the prose).

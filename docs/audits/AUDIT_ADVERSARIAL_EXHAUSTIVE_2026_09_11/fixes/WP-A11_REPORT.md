# WP-A11 — Polarization, coatings, Berreman, sources, algebra, infrastructure

Branch `audit-fixes-2026-09`.  Findings Z1–Z4 of
`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` §10, the ORCHESTRATOR
`coating_reflectance(polarization='te')` row, and the
`POLAR-SOURCES-INFRA.md` partition report.  All measurements taken on this
workstation with `OPENBLAS_NUM_THREADS=1`, 2026-09-12.

---

## 1. Summary

| ID | item | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|---|
| **Z1** | `coating_reflectance` routes `'te'`/`'S'`/junk down the **p** branch | **fixed** | `elements/coatings.py:29–74` (new `_normalize_coating_pol`), `:185–186`, `:352` | `test_audit2609_a11_polar_sources_infra.py::test_z1_*` (24) | analytic single-layer Airy (in-test), + `rcwa._core._normalize_pol` set-equality | `R('te')` 0.00322899 (p) → **0.15427715** (s); junk silently 0.00322899 → **`ValueError`**; via `propagate_through_system` `te==s? False→True`, `te==p? True→False` |
| **Z1** | same defect in `coating_reflectance_jax` | **fixed** | `elements/coatings.py:434, :463` | `::test_z1_jax_*` (4) | NumPy twin, `jax.grad` | `R_jax('te')` 0.00322899 → 0.15427715 (‖NumPy‖ ≤ 1.1e-16); `dR('te')/dd` = `dR('s')/dd` = −1 052 475.33 exactly |
| **Z2** | Gaussian–Schell sources realise the **periodised** kernel | **fixed** | `sources/core.py:2085–2120` (new helpers/constants), `:2123–2298` (`_schell_phase_realizations`) | `::test_z2_*` (5); `test_s3_7_broadcast_grid.py::…_{unpadded,padded}` | pair-averaged linear (non-circular) FFT correlation vs both the Gaussian and the wrapped-Gaussian hypotheses; Starikov–Wolf eigen-spectrum | max\|μ−Gaussian\| **0.9949 → 0.0081** (σ=L/3), **0.8836 → 0.0007** (σ=L/32); edge-to-edge μ **1.0019 → 0.0151**; audit's own p6 estimator **0.2692 → 0.0041** |
| **Z2** | docstring "σ_g ≫ w0 approaches the coherent limit" steers into the bad regime | **fixed** | `sources/core.py:1733–1740` (module header), `create_gaussian_schell_source` / `create_schell_model_source` docstrings | `::test_z2_warns_…`, `::test_z2_public_factories_…` | — | now states `q = σ_g/w0` + `dx ≪ σ_g ≤ N·dx/6`, and warns above `L/6` |
| **Z3** | `estimate_lens_memory(lens_model='real')` under-predicts 1.6–2.8× | **fixed** | `memory.py:510–543` (new constants + calibration note), `:645–668` (docstring), `:745–775` | `::test_z3_estimate_lens_memory_real_bounds_apply_real_lens` | `tracemalloc` peak of one `apply_real_lens` call, 3 grids × 2 dtypes | est/meas **0.36 / 0.63 → 1.06–1.07** (six points) |
| **Z3** | `create_gaussian_beam` dense meshgrid + out-of-place normalise | **fixed** | `sources/core.py:503–556` | `::test_z3_gaussian_beam_*` (14); `test_s3_7_broadcast_grid.py::test_gaussian_beam_broadcast_bit_identical` | in-test restatement of the pre-fix body (bit-identity) + `tracemalloc` | peak/output **3.00× → 1.50×** (c128), **5.00× → 2.00×** (c64); 0.705 s → 0.326 s at N=4096; **bit-identical** over 36 configs |
| **Z3** | algebra `FreeSpace` warning spam + pitch ↔ ABCD contradiction | **fixed** | `algebra/primitives.py:69–100`, `:176–205`; `algebra/base.py:169–191`; `algebra/from_prescription.py:72–81` | `::test_z3_freespace_*` (3) | 4f chain: ABCD vs delivered pitch, centroid, power | **3 → 0** `UserWarning`s per 4f evaluation; `dx_out/dx_in` **1.9318 → 1.0000** (ABCD \|A\| = 1) |
| **Z3** | `stokes_parameters` / `degree_of_polarization` temporaries | **fixed** | `elements/polarization.py:1340–1364`, `:1416–1454` | `::test_z3_stokes_and_dop_*` (3) | in-test restatement of the pre-fix expressions + `tracemalloc` | peak **7.00 → 6.00** and **8.25 → 6.00** full-grid real arrays; 0.258 s → 0.110 s, 0.396 s → 0.198 s; **bit-identical** incl. NaN/inf/dark/underflow pixels |
| **Z4** | `deprecated_alias` stacklevel off by one | **fixed** | `_deprecation.py:470–500` (`_DEFAULT_STACKLEVEL` + `_emit`), and the five helper defaults | `::test_z4_deprecated_alias_warning_names_the_caller`, `::test_z4_live_alias_points_at_user_code` | `warnings.catch_warnings(record=True).filename` | `_deprecation.py:551` → **the caller's own file and line** |
| **Z4** | `algebra.from_prescription` module shadows the function | **fixed** | `algebra/__init__.py:96–120`; `algebra/from_prescription.py:1–24` | `::test_z4_algebra_from_prescription_is_callable` | four access paths + ABCD equality with `Operator.from_prescription` | `TypeError: 'module' object is not callable` → **callable**, dotted-module import preserved |
| **Z4** | `_check_2d_scalar_field` accepts object dtype / `np.matrix` | **fixed** | `_validation.py:23–29`, `:62–78`, `:139–158`, `:249–281` | `::test_z4_check_2d_scalar_field_*` (8) | the 11-case accept/reject table from `p9_infra.py` §D | object dtype and `np.matrix` **ACCEPTED → `TypeError`**; real/int/bool still accepted |
| **Z4** | cache-registry name collisions + swallowed clearer failures | **fixed** | `_cache_registry.py:36–39`, `:54–87`, `:120–175`, `:220–246` | `::test_z4_cache_registry_*`, `::test_z4_clear_all_reports_a_failing_clearer` (3) | `p9_infra.py` §G | collision **silent → `RuntimeWarning`** naming both sites (reload still silent); failing clearer **silent → one `RuntimeWarning`** naming it |
| **Z4** | `_plane_wave_carrier` centres on `nx//2` | **fixed** | `elements/polarization.py:1651–1674` | `::test_z4_plane_wave_carrier_uses_the_package_grid_centring` (4) | the package grid `(arange(N)−N/2)·d` + `apply_jones_matrix`'s callable grid | odd-N grid `[−3…+3] µm` → **`[−3.5…+2.5] µm`**; even N unchanged |
| **Z4** | `JonesField.propagate*` docs vs in-place mutation | **fixed (docs)** | `elements/polarization.py` — all four `propagate*` docstrings | `::test_z4_jones_propagate_methods_are_in_place_and_return_self` | behaviour, not prose | no behaviour change; all four now state the convention |
| **Z4** | `user_library` unbounded `Pow` | **fixed** | `user_library.py:187–232`, `:278–292` | `::test_z4_user_library_*` (9) | timing + value round-trip of legitimate expressions | `2**(10**9)` allocated a 125 MB int → **`ValueError` in 0.02 ms**; `<<` guarded too |
| **Z4** | `import lumenairy` pulls 128 scipy submodules | **not mine / already in flight** | `lumenairy/__init__.py`, `backend/scipy.py` | — | `python -X importtime` | see §5 — `lumenairy/__init__.py` is explicitly out of my ownership and another WP has already started the lazy-`backend.scipy` work |

Nothing in the audit's "Checked and found correct" list was changed: re-ran
`p12_final.py` — `dy` threading passes in **all 12** factories, and coatings
p-transmittance against an absorbing exit medium still agrees to ≤ 7.8e-16
over 12 `(n_sub, angle, pol)` combinations.  The TMM itself is untouched.

---

## 2. Per finding

### Z1 (P1) — `coating_reflectance` sent every non-`'s'` spelling down the p branch

**What was wrong.**  `pols = ['s','p'] if polarization == 'avg' else [polarization]`,
then `if pol == 's': eta = n·cosθ else: eta = n/cosθ`.  Anything not the literal
lowercase `'s'` — `'te'`, `'S'`, `'TE'`, `'tm'`, `'banana'`, `''`, `0`, `None` —
took the p branch.  `CONVENTIONS.md` §7 names this very file as accepting
`te`/`tm` case-insensitively; `rcwa._core._normalize_pol` implements that
contract and raises on junk, `coatings.py` did neither.  Reachable from the
public wave path (`propagate_through_system` forwards the string verbatim), and
present identically in the JAX twin, so a gradient-based AR/HR design on
`'te'` optimised the TM stack.

**What I changed.**  A module-level `_normalize_coating_pol(fn_name, polarization)`
that (a) handles the coatings-only `'avg'`, (b) delegates te/tm/s/p to
`rcwa._core._normalize_pol` via a lazily-resolved, cached reference (CONVENTIONS
§10 shape — importing `coatings` does not pull the RCWA stack), and (c) re-raises
with the §2 `f"{fn_name}: ..."` prefix naming `{'s','te','p','tm','avg'}`.  Both
`coating_reflectance` and `coating_reflectance_jax` normalise **first**, and
every branch below keys off the canonical token (including the `'avg'`
aggregation at the end, which used to re-test the raw string).  The delegation
is deliberate: a duplicated alias table is exactly the "divergent copies" shape
§15.4 describes, so a test asserts the two helpers accept the identical set.

**How I verified.**  Ran the audit's own `repro/POLAR-SOURCES-INFRA/p2_coatings.py`
before and after.  §D before: `te`/`tm`/`S`/`P`/`banana`/`''` all 0.00322899
(the p value); after: `te`/`S` → 0.15427715 (the s value), `tm`/`P` → 0.00322899,
`banana`/`''` → `ValueError`.  §A/§B/§C/§E/§F (Airy oracle, Au film, normal-incidence
sign, TIR, absorbing substrate) unchanged to the printed digits.  `p9_infra.py`
§I now reports `|t|` = `{'s': 0.94840576, 'p': 0.9969635, 'te': 0.94840576,
'tm': 0.9969635}` — was `'te': 0.9969635`.  JAX: parity 1.1e-16 across six
spellings plus `'avg'`, and `jax.grad` w.r.t. thickness under `'te'` is bit-equal
to `'s'` (−1052475.3257100915) and differs from `'tm'`.

**Residual risk.**  A caller who was passing a *typo* and had (unknowingly)
calibrated around the p result now gets an exception.  That is the point of the
fix, and it is stated as a migration note in the changelog.  Whitespace is not
stripped — `' s '` raises — deliberately, so the accepted set is provably
identical to `_normalize_pol`'s; if the house later decides to strip, both
helpers must change together and the equivalence test will hold them to it.

### Z2 (P1) — the Gaussian–Schell coherence kernel was the grid-periodised one

**What was wrong.**  White noise filtered by `H(k)=exp(−|k|²σ_g²/4)` on the FFT
grid and inverse-FFT'd: the realised two-point correlation is the inverse **DFT**
of `|H|²`, i.e. (Poisson summation) `Σ_m exp(−|Δ+mL|²/(2σ_g²))` with `L=N·dx`.
The wrap manufactures spurious *long-range* coherence: at the largest separation
the grid can form, `(N−1)dx`, the nearest wrapped image sits one pixel away, so
opposite edges of the grid came out ~99 % coherent where the model says ~0.

**What I changed.**  `_schell_phase_realizations` gained `pad_sigma=4.0`
(`_SCHELL_PAD_SIGMA`).  The noise is drawn **and filtered** on a grid
`N + 2·ceil(pad_sigma·σ_g/d)` per axis, rounded up to `scipy.fft.next_fast_len`,
and the central `(Ny, Nx)` window is cropped.  Three deliberate details:

* the noise covers the *whole* padded grid rather than being zeroed outside the
  window — zero-padding the noise itself would taper the variance within `σ_g`
  of the crop edge, i.e. trade a correlation error for a non-stationary
  intensity one;
* the deterministic unit-mean-intensity constant (v5.4.6 P3-10) is computed on
  the padded grid, which is the right constant for the crop because that grid is
  statistically homogeneous;
* the pad is capped at `4×` the grid per axis (`_SCHELL_MAX_PAD_GROWTH`) so
  `σ_g ≫ L` cannot turn into an out-of-memory; when the cap binds, a
  `UserWarning` quotes the residual periodisation it actually leaves, computed
  by `_periodised_gaussian_error`.

`pad_sigma=0.0` reproduces the pre-fix path bit-for-bit (same RNG draws, same
FFT, same normalisation) — an escape hatch for reproducing an archived ensemble
and, more importantly, the in-process "before" arm of the regression test.

A `UserWarning` also fires for `σ_g > L/6` — not about the kernel (which is now
right at any `σ_g`) but about what *is* still true there: fewer than six
coherence cells across the grid, so any ensemble estimate is aperture-dominated,
and the pad costs `~((L+8σ_g)/L)²` the FFT work.

**How I verified.**  Two independent estimators, both streamed so no run exceeds
~130 MB:

* the audit's own single-reference estimator (`p6_gsm_grid.py`'s arithmetic,
  chunked): reproduces the audit's table exactly on HEAD — 0.2692 / 0.0023 /
  (0.7719, 0.7261, 0.7734) — and reads **0.0041** vs the Gaussian after;
* a pair-averaged **linear** (zero-padded FFT) correlation, which counts no
  wrapped pair and so cannot itself introduce periodicity.  At 20 000
  realisations: max\|μ−Gaussian\| 0.9949 → 0.0081 (σ=L/3), 0.9922 → 0.0036
  (σ=L/8), 0.8836 → 0.0007 (σ=L/32); edge-to-edge μ 1.0019/0.9922/0.8836 →
  0.0151/−0.0036/0.0007 against true values 0.0070/3.4e-14/3.4e-216.

`E[⟨|φ|²⟩]` stays 0.9967–0.99999, so the normalisation contract survives.
Downstream (`return_kind='mcf'`, the audit's §3 fixture) the coherent-mode
spectrum moves from `[0.40081, 0.14175, 0.13891, 0.07445, 0.06990, 0.05077]` to
`[0.39205, 0.15411, 0.14527, 0.05830, 0.05597, 0.05424]` against Starikov &
Wolf's `[0.38202, 0.14592, 0.14592, 0.05574, 0.05574, 0.05574]`: the exactly
3-fold-degenerate n=2 shell's internal spread falls **36.4 % → 7.2 %** and its
mass error **+16.7 % → +0.8 %**.

**Residual risk.**  (a) Ensembles are no longer bit-identical for a given seed —
a deliberate change of a wrong default, with `pad_sigma=0.0` as the documented
reproduction path.  (b) The pad costs 1.3–5.8× the FFT time (table in the
changelog); the ensemble array still dominates the peak so memory grows ~28 %.
(c) The Gaussian filter is still *sampled* at the Nyquist band, so for
`σ_g ≲ dx` the kernel is a band-limited Gaussian rather than the continuum one;
that is inherent to any grid representation and the docstring now states the
`dx ≪ σ_g` requirement.  Gori's pseudo-mode form removes both the window and the
FFT cost — deferred, designed in §6.

### Z3 (P2) — `estimate_lens_memory(lens_model='real')`

**What was wrong.**  The `'real'` branch reused the traced calibration and scaled
the float64 core down by `5/_LENS_F64_ARRAYS`, documented as "omits the traced
final-assembly float64 arrays".  Measured, `apply_real_lens` does not.

**What I changed.**  Dedicated `_LENS_REAL_F64_ARRAYS` / `_LENS_REAL_COMPLEX_ARRAYS`,
derived from `tracemalloc` on `apply_real_lens` itself.  The measured peak is
pure `N²` (bytes/pixel spread 1.4 % over N = 512/1024/2048), so two dtypes give
an exact two-unknown solve: `8F + 16C = 176.02` and `8F + 8C = 120.01` →
**C = 7.00 complex** and **F = 8.00 float64** full-grid arrays, of which 2
float64-equivalents are the `phase_exp` complex128-first transient the shared
term already models.  Shipped with ~7 % margin so the estimate *bounds* the
measurement (the fail-safe direction).  `parallel_amp` is now inert for
`'real'`: `apply_real_lens` has no such argument, so doubling would double-count.

**How I verified.**  The audit's `p8_memory.py`: est/meas **0.36 (parallel_amp=False)
/ 0.63 (True) → 1.07 / 1.07** at N = 1024 and 2048.  My own sweep adds N = 512
and complex64: 1.06–1.07 at all six points.  **Measured against
`elements/_lens_real.py` sha256 `737e62954a4de29c` (412 308 bytes)** — i.e. WP-A2's
reductions as of 2026-09-12 04:35; those had already cut the peak from the
audit's 203.0/809.6 MB to 185.2/738.3 MB, and I re-measured after they landed
(the earlier revision `58249f1c…` gave identical bytes/pixel, so the constants
are stable across that change).  A further WP-A2 reduction moves the ratio *up*,
which stays fail-safe and is what the test's upper bar (1.6) is there to surface.

**Residual risk.**  The row-band (`sag_chunk_rows`, the auto default at N ≥ 4096)
branch is untouched and keeps its own calibration; Z3 did not cover it and I did
not re-measure it (a 16384² anchor is well over the 2 GB budget for this shared
machine).

### Z3 (P2) — `create_gaussian_beam`, algebra `FreeSpace`, `stokes_parameters` / `degree_of_polarization`

All three are described with their measured before/after in the changelog.
Three implementation notes worth keeping:

* **`create_gaussian_beam`** — the sign is folded into a separate in-place
  `np.negative` so the exponent needs one real `N²` buffer instead of four:
  `(−S)/c == −(S/c)` exactly in IEEE (division is sign-symmetric, negation
  exact), which is why the output is bit-identical over all 36 configurations.
  I deliberately did **not** build the exponent in float32 for a `complex64`
  request (the audit's optional suggestion): it would change the returned values
  and COMMON requires bit-identity or a documented tolerance for performance
  items.  Deferred with a design in §6.
* **`FreeSpace`** — the default moved `'auto' → 'asm'`.  The audit offered that
  or "keep 'auto' but re-sample to the ABCD-implied pitch"; the latter *also*
  changes the delivered field, and silently, so I took the honest one and
  documented the migration.  Separately, the square-grid branch now consumes the
  shape-stable `PropagationResult`, which removes the un-actionable warning even
  when the user names `'auto'`.  The anamorphic branch keeps the legacy tuple
  because the v5.30 comment's reason is real there (the wrapper reports
  `dy == dx` for the pitch-preserving kernel that branch forces) — verified by
  `::test_z3_freespace_anamorphic_dy_still_threaded` and by the existing
  `test_freespace_operator_preserves_the_anamorphic_pitch`.
* **`stokes_parameters`** — NumPy's vectorised complex multiply is **not**
  bitwise commutative on this build (measured 1.8e-15 on S3 at N = 64 when the
  operands swap), so the `out=` multiply keeps `Ex` as the left operand.  I
  caught this because the first version used `conj(Ey) *= Ex` and the
  bit-identity test failed; it is recorded in the source so the next reader does
  not "simplify" it back.

### Z4 (P3) — the seven infrastructure items

Each is described with its before/after in the changelog.  Notes:

* **stacklevel** — the arithmetic is now written out next to
  `_DEFAULT_STACKLEVEL = 4`: `user → public fn → warn_deprecated_* → _emit →
  warn` is four frames, and `warn_renamed_function` takes five because it
  inserts one.  No production site relied on the old default (the v5.30
  shim-removal wave cleared them; the seven sites the v4.15.3 sweep fixed passed
  `stacklevel=4` explicitly, so they are unaffected and now redundant).  The
  audit also measured the alias shim at 7.66 µs/call; on this machine it reads
  0.87–1.86 µs and the cost is `warnings.warn`'s stack walk.  I did **not** add
  a "already warned" flag: it would break `warnings.simplefilter('always')` /
  `catch_warnings` semantics that the suite relies on.  Recorded in §6.
* **`_check_2d_scalar_field`** — the dtype gate is a `dtype.kind` membership test
  (~60 ns) rather than `np.issubdtype` (~20× that), because the helper runs at
  the entry point of every propagator and lens call.  Real/int/bool stay
  accepted: an amplitude mask is a legitimate field and every kernel promotes it.
* **cache registry** — "same call site" is decided by
  `(module, qualname, co_filename, co_firstlineno)`.  `importlib.reload`
  preserves all four; two distinct lambdas in the same module differ in
  `co_firstlineno`, which is exactly the audit's evidence case.  Callables with
  no code object fall back to a coarser key that errs towards warning.
* **`from_prescription`** — I rebound the package attribute rather than renaming
  the module to `_from_prescription.py`.  The rename is cleaner in the file tree
  but breaks `from lumenairy.algebra.from_prescription import from_prescription`
  (which the audit's own `p10_algebra.py` uses) and would stale the dotted-name
  exemption in `test_v4_16_0_walker_all_symmetry.py`, a meta-test I do not own.
  The rebind fixes the reported symptom and keeps every other spelling working;
  only the chained `…from_prescription.from_prescription` workaround goes away.
  I did **not** add the name to `lumenairy.algebra.__all__` — see §5.
* **`user_library`** — `<<` is guarded alongside `**`.  The audit named only
  `Pow`, but `LShift` is the identical CPython bignum path and sits one
  `_BIN_OPS` entry away; leaving it is the sibling-gap shape §15.4 describes.

---

## 3. Files touched

**Library (all within the WP-A11 ownership list):**

| file | what |
|---|---|
| `lumenairy/elements/coatings.py` | Z1: `_normalize_coating_pol` + routing in both twins; p-phase convention doc note |
| `lumenairy/sources/core.py` | Z2: anti-wrap pad, guards, docstrings; Z3: `create_gaussian_beam` broadcast views + in-place normalise |
| `lumenairy/memory.py` | Z3: `'real'` lens-model constants, branch, docstring |
| `lumenairy/algebra/primitives.py` | Z3: `FreeSpace` default `'asm'`, `PropagationResult` on the square-grid branch |
| `lumenairy/algebra/base.py` | Z3: PITCH CONTRACT note on `Operator.abcd` |
| `lumenairy/algebra/__init__.py` | Z4: rebind `from_prescription` to the function |
| `lumenairy/algebra/from_prescription.py` | Z4: name-resolution note; Z3: `method` doc note |
| `lumenairy/elements/polarization.py` | Z3: lean `stokes_parameters` / `degree_of_polarization`; Z4: `_plane_wave_carrier` centring + broadcast views, four `propagate*` docstrings |
| `lumenairy/_validation.py` | Z4: `np.matrix` + dtype-kind gate, relative import |
| `lumenairy/_deprecation.py` | Z4: `_DEFAULT_STACKLEVEL`, five helper defaults, docstrings |
| `lumenairy/_cache_registry.py` | Z4: collision warning + identity helper, failure reporting |
| `lumenairy/user_library.py` | Z4: `_guard_int_growth` for `**` / `<<` |

Not touched: `lumenairy/elements/berreman.py`, `_berreman_jax.py` (no finding
needed them; the audit verified both), `lumenairy/cache.py`, `lumenairy/_context.py`
(both verified correct by the audit, nothing to change).

**New file:** `tests/unit/test_audit2609_a11_polar_sources_infra.py` (85 tests).

**Existing tests changed** (all covering my modules; each documented in-file with
the finding ID and what the pin's contract still is):

| file | why |
|---|---|
| `tests/unit/test_s3_7_broadcast_grid.py` | the Schell bit-identity pin reproduced the *periodised* implementation — split into `_unpadded` (`pad_sigma=0.0`) and `_padded` arms so it still pins the KX/KY orientation on both paths; added `test_gaussian_beam_broadcast_bit_identical`, the one factory the S3-7 census was missing |
| `tests/unit/test_v4_15_2_agent_a.py` | `test_default_path_emits_no_warnings_at_all` used a 16×16 / dx=2 µm fixture with `σ_g = L/4`, which the new grid-constraint warning targets; the fixture now satisfies `N·dx ≥ 6σ_g` (sibling tests in the class filter to `DeprecationWarning` and were unaffected) |
| `tests/unit/test_v4_15_4_agent_b.py` | `test_validation_helper_lazy_import_removed` matched the absolute import spelling literally; broadened to accept the relative one (its contract — module scope, not lazy — is unchanged and still enforced on both spellings) |
| `tests/unit/test_niche_audit_w4_p5_return_contract.py` | two FreeSpace-specific assertions: the AST pin now requires the call to *name* `return_result` (value `wrap`) rather than the literal `False`; the grid-changing pin names `method='auto'` and additionally pins that the default does **not** re-grid.  **Only the FreeSpace arms were touched** — the mhs and `Source.propagate` assertions in the same file belong to other WPs and are untouched |
| `tests/unit/test_v4_15_2_agent_b.py`, `tests/unit/test_v4_15_1_agent_g_application.py` | the `FourierTransform` ↔ 3-stage-chain equivalence pins built the comparison chain with the default `FreeSpace`; `FourierTransform` deliberately pins `method='auto'` for its legs (the optical FT *is* a re-gridding operation), so the comparison chain now names the same kernel |

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result | time |
|---|---|---|
| `pytest tests/unit/test_audit2609_a11_polar_sources_infra.py` | **85 passed** | 6.9–12.7 s |
| `pytest tests/unit/test_v5_5_3_coatings_jax.py test_v5_4_coating_materials.py test_v5_4_1_coating_sellmeier_nan_fix.py test_v5_4_5_coating_edge_cases.py test_v5_4_6_wave3_coatings_glass.py test_s3_7_broadcast_grid.py test_g08_s3_16_sources.py test_v5_4_6_wave7_sources.py` | **90 passed** | 9.3 s |
| `pytest` × 9 algebra files (`test_v4_15_1_agent_g_*`, `test_v4_15_2_agent_b`, `test_v4_15_3_agent_b`, `test_niche_s11_sibling_deferred`, `test_niche_audit_w4_p5_return_contract`) | **274 passed** | 11.9 s |
| `pytest` × 12 infra files (`test_memory_guardrail`, `test_niche_audit_w4_input_kind`, `test_v4_15_3_dispatcher_pin_2d_scalar_field`, `test_v4_16_0_agent_d_cache_registry`, `test_v4_16_1_dispatcher_pin_cache_registry_enrollment`, `test_g08_s4_15_cache_hygiene`, `test_c2_s4_19_user_library_safe_eval`, `test_niche_audit_w3_infra`, `test_niche_audit_w3_ui_deprecation`, `test_v5_4_6_wave4_polarization`, `test_audit_polarization`, `test_niche_audit_e_polarization_inputs`) | 584 passed, **3 failed (none mine — §4.1)** | 34.4 s |
| `pytest` × 14 sources files (`test_v4_15_*`, `test_v4_16_*`, `test_audit_v5_24_2_b1_source_conventions`, `test_niche_s12_source_shape_wiring`, `test_audit_sources`) | 447 passed, 4 failed (**all the flaky subprocess shape — §4.2**) | 20.4 s |
| `pytest` × 11 polarization / user_library / walker files | 469 passed, **3 failed (none mine — §4.1)** | 147 s |
| `python validation/run_all.py test_coherence test_polarization` | **2/2 PASS** | 7.8 s |
| `python validation/run_all.py test_sources test_elements` | **2/2 PASS** | 22.7 s |

Repro scripts re-run before and after: `repro/POLAR-SOURCES-INFRA/p2_coatings.py`,
`p8_memory.py`, `p9_infra.py`, `p10_algebra.py` (§F rewritten — see §4.3),
`p12_final.py`; `p6_gsm_grid.py`'s estimator reproduced in a streamed form
(the shipped script allocates a 2.6 GB ensemble, over this machine's budget).

### 4.1 Pre-existing failures caused by other WPs (measured, not assumed)

| test | reading | judgement |
|---|---|---|
| `test_niche_audit_w4_input_kind.py::test_all_sixty_eight_sites_are_wired` and `::…[beam_stats.py::beam_d4sigma(E)->field]` | "expected 69 `_check_2d_scalar_field` calls in lumenairy/; found 70" | `lumenairy/analysis/beam_stats.py` has +50/−9 lines from another WP, which added a 70th call site without updating the meta-pin's count/registry.  I added **no** call sites (only the helper's body). |
| `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512-complex128]` | `est/measured = 1.3824 > 1.35` (Windows fence) | `propagators/asm.py` (+39), `propagators/fft_infra.py` (+96) and `backend/scipy.py` (+77) are all modified by another WP; the measured *cold* ASM peak fell to 61.7 MB (deterministic over 3 runs), which is the lazy-scipy import reduction §15.8 recommends.  The A-6 contract (`est ≥ measured`) still holds; only the "absurd looseness" fence trips.  `estimate_asm_memory` and `_ASM_FIRST_CALL_FIXED_BYTES` live in `memory.py` (mine) but the audit verified the estimator is conservative and Z3 did not touch it — re-deriving the 56 MiB fixed term *downward* against a mid-flight change would make it less conservative on a moving measurement.  See §5 for the request. |
| `test_niche_audit_w5_shim_removals.py::TestPropagatorInertKwargRemovals::test_hf_chunk_output_is_KEPT` | "DID NOT WARN" | `propagators/hf.py` has +305 lines from another WP implementing K22 — `chunk_output` is now a working parameter (it chunks at `hf.py:510–515`), so its deprecation warning is gone.  The test asserts the old no-op behaviour. |
| `test_v5_4_7_walker_v20_cross_backend_parity.py::test_every_jax_twin_is_registered`, `::test_jax_intersect_direction_aware_root_pick_present` | new `raytrace.jax_trace:exit_vertex_transfer_jax` unregistered | the §15.1 exit-vertex helper from another WP. |
| `test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn::test_traced_emits_warning_for_stop_index_2` | `apply_real_lens` now *raises* on an out-of-range `stop_index` where the test expects a warning | `elements/_lens_real.py` has +1024 lines from another WP. |
| `test_v5_4_6_wave4_polarization.py::test_vector_aperture_diffraction_has_projection_kwarg` | `vector_projection` default is now `True`, the pin asserts `False` | `propagators/vectorial_hfpi.py` gained +351 lines while I was running this batch (file mtime 05:05, the failure appeared at 05:09) — the K17 fix from another WP.  It passed in my earlier run of the same file. |

### 4.2 Environmental flakiness on this shared machine

Five `subprocess.run`-based tests fail intermittently with
`OSError: [WinError 6] The handle is invalid` / `[WinError 50] The request is not
supported` when run in a large batch, and pass in isolation:
`test_v4_15_4_agent_b.py::{test_validation_helper_no_circular_import_on_module_load,
test_no_deprecation_warning_errors_in_v4_15_1_agent_b_module,
test_no_deprecation_warning_errors_in_v4_15_3_agent_a_module,
test_representative_v4_15_x_modules_pass_under_deprecation_error}` and
`test_v4_15_2_agent_a.py::TestChangelogDriftFixes::test_changelog_test_count_arithmetic_reconciles`.
Measured: `test_v4_15_4_agent_b.py` alone → **11/11 passed** on two consecutive
runs; the changelog-drift test alone → passed 2 of 3 runs, the failure being the
same `OSError` at `subprocess.py:1431`.  Several other agents are spawning
Python processes on this host; none of these tests reads any module I changed
except through the same import path that succeeds in isolation.

### 4.3 A defect in one shipped repro script

`repro/POLAR-SOURCES-INFRA/p10_algebra.py` §F calls
`S(E, dx=dx, wavelength=wl)`, which is not an `Operator.__call__` signature
(the tuple form is), so §F has always raised `TypeError` — before my change as
well.  §D likewise calls `CylindricalLens(f=…)` where the parameter is not `f`.
I reproduced §F's claims with the corrected call form in a scratch script and
report those numbers (3 warnings, `dx_out/dx_in = 1.9318`, power 1.0000,
centroid +304 → −304 µm — matching the audit's text).  `p12_final.py` §3
unpacks `mcf.coherent_modes()` as `(lam, modes)` where the method returns
`(modes, eigenvalues)`; the audit's quoted spectrum is right, the script's print
is not.  Left the scripts untouched (they are audit evidence, not library code).

---

## 5. Requested changes outside my ownership

1. **`lumenairy/__init__.py`** — export `from_prescription` at top level.
   Exact change: add `from .algebra import from_prescription` to the algebra
   import block (~line 931) and `'from_prescription'` to `__all__` (~line 1291).
   Why: `la.from_prescription` does not exist, and `lumenairy.algebra.__all__`
   cannot list the name until it does — `test_v4_16_0_walker_all_symmetry.py`
   requires every submodule `__all__` entry to be top-level-re-exported or
   exempted.  I fixed the *shadowing* (the name now resolves to the callable) but
   left `algebra.__all__` alone so I would not trip a meta-test I do not own.
   If the project prefers to keep `Operator.from_prescription` as the only
   public spelling (the walker's documented rationale), then nothing is needed —
   the shadowing fix stands on its own.

2. **`tests/unit/test_v4_16_0_walker_all_symmetry.py`** — if (1) is taken, delete
   the `('lumenairy.algebra.from_prescription', 'from_prescription')` exemption
   and its docstring paragraph.

3. **`tests/unit/test_niche_audit_w4_input_kind.py`** — belongs to whoever added
   the 70th `_check_2d_scalar_field` site in `lumenairy/analysis/beam_stats.py`:
   wire it into `_WIRED_SITES` and bump the literal 69 → 70 in
   `test_all_sixty_eight_sites_are_wired` and the module docstring.

4. **`memory.py::_ASM_FIRST_CALL_FIXED_BYTES` (mine) vs `propagators/asm.py` +
   `backend/scipy.py` (not mine)** — once the lazy-`backend.scipy` work settles,
   the 56 MiB first-call fixed term should be re-derived (the cold ASM peak at
   N = 512 / complex128 fell from ~80 MB to 61.7 MB, pushing est/measured to
   1.3824 against `test_niche_audit_w3_infra.py`'s 1.35 Windows fence).  I did
   not re-derive it because Z3 did not cover `estimate_asm_memory`, the audit
   verified it is conservative, and calibrating downward against a mid-flight
   change would reduce a safety margin on a moving target.  Either the constant
   comes down or that test's Windows fence goes up, with the new measurement
   dated in the comment — one decision, one place.

5. **`tests/unit/test_niche_audit_w5_shim_removals.py::test_hf_chunk_output_is_KEPT`**
   — belongs to the WP that made `chunk_output` functional; it now asserts a
   deprecation that no longer exists.

---

## 6. Deferred, with designs

1. **Gori pseudo-mode generation for Schell sources** (removes the FFT and the
   window entirely).  A stationary complex field with Gaussian correlation
   `exp(−|Δ|²/(2σ²))` is exactly `φ(r) = M^(−1/2) Σ_j exp(i(k_j·r + ψ_j))` with
   `k_j ~ N(0, σ^(−2))` per component and `ψ_j` uniform, because
   `⟨exp(i k·Δ)⟩ = exp(−|Δ|²/(2σ²))` — no grid, no periodisation, exact by
   construction, and it reproduces the Starikov–Wolf spectrum by construction
   rather than by sampling.  Each realisation is a rank-`M` update computable as
   one `zgemm`: `φ = A @ B` with `A = exp(i(y ⊗ k_y + ψ))` `(Ny, M)` and
   `B = exp(i k_x ⊗ x)` `(M, Nx)`.  Cost `O(M·N²)` vs the FFT's `O(N² log N)`;
   at `M = 256, N = 512` that is ~5e8 flops per realisation, ~100× the padded
   FFT here, so it is the right default only once `M` can be chosen adaptively
   (Gaussian statistics need `M ≳ 100`; the CLT error falls as `M^(−1/2)`).
   Effort: ~1 day including a `generator='fft'|'modes'` kwarg, an `M` heuristic
   from `(L/σ_g)²`, and a χ² test that the marginal is circular-Gaussian.
2. **`create_gaussian_beam` in the target real dtype for `complex64`** — would
   take peak/output from 2.00× to ~1.50× and roughly halve the exp cost, but
   changes the returned values (float32 `exp` vs float64-then-cast), so it needs
   a `geometry_dtype=` opt-in plus a documented tolerance rather than a silent
   default change.  Effort: ~2 h.
3. **`apply_jones_matrix` `out=` accumulation** — the audit measured peak 4.00
   full-grid complex arrays (`J00*Ex + J01*Ey` builds two temporaries per
   component); `np.multiply(..., out=)` + `+=` halves it.  Not in Z3's list, and
   it needs the same bit-identity care the complex-multiply non-commutativity
   above just demonstrated (the operand order must be preserved exactly).
   Effort: ~2 h including the bit-identity matrix.
4. **`deprecated_alias` per-call cost** (0.87–1.86 µs here, 7.66 µs on the
   auditor's loaded machine) — a "already warned" flag would skip
   `warnings.warn`'s stack walk but breaks `simplefilter('always')` /
   `catch_warnings`, which the suite depends on.  If it ever matters, the right
   shape is a module-level flag consulted only when
   `warnings.filters` is unchanged since the last call (cheap to detect via
   `warnings._filters_mutated`), which is an internal API — hence deferred.
5. **Berreman `_split_fwd_bwd` Poynting criterion** (the partition report's
   unverified suspicion) — the NumPy cascade partitions by decay/phase while the
   JAX generalized path uses `Sz > 0`.  The auditor could not construct a case
   where they differ and neither could I short of a genuinely
   negative-group-velocity medium.  No change made; recorded so the next audit
   does not re-derive it.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A11_CHANGELOG.md`

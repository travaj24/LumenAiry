# VERIFY Wave 5 hygiene 2 -- independent adversarial re-measurement

Subject: `refactor/wave5-hygiene-2` (`2796d551` H2-1 direct-matrix MFT opt-in,
`76d013c3` H2-2 `_collins_transport` on JAX, `2d6eef9c` H2-3 near-focus table,
`a985ac2c` H2-4 loud stale-patch refusal, `b86fe31f` report + CHANGELOG) against
base `f4f18851`.  Author's report:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WAVE5_HYGIENE2_REPORT.md`.

Verification branch `verify/wave5-hygiene-2`, worktree `C:/tmp/lum_vhyg2`.
Probes and JSON: `validation/probe_verify_wave5_hyg2/`.  Decision tests:
`tests/unit/test_verify_wave5_hyg2.py`.

## How this verification was gated

* **Re-measured, never read.**  Every number below was produced by a probe
  written for this verification.  Where a claim is reproduced, the reproduction
  is on a DIFFERENT fixture from the author's wherever the claim is about a
  property rather than about a specific reading.
* **Archive-to-archive.**  Bit-identity runs extract `f4f18851` and `b86fe31f`
  with `git archive` into `C:/tmp/vhyg2_arch/{base,branch}` and run a child
  process whose `cwd` and `PYTHONPATH` name one tree.  Every probe prints
  `lumenairy.__file__` and refuses to continue outside its tree.
* **Both builds, every time.**  Windows py3.14 (numpy 2.4.4, scipy 1.17.1, jax
  0.11.0, CuPy 14.0.1 with a device and a broken cuFFT DLL) and WSL py3.12
  (numpy 2.4.6 on scipy-openblas SkylakeX, scipy 1.17.1, jax 0.10.2, no CuPy).
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and
  `LUMENAIRY_MEM_BUDGET_MB=8192 PYTHONHASHSEED=0` on the COMMAND LINE.
* **Nothing under `lumenairy/` was edited.**  Every mutation arm ran on a tree
  copy under the session scratchpad; defects carry the exact requested edit.

**A trap worth recording before anything else.**  A dev-install of lumenairy
exists on this box at `D:\Metacept\...\Lumenairy\lumenairy`, and it is what
`import lumenairy` resolves to when `PYTHONPATH` is not pinned.  Two of this
verification's first probe runs bound it.  The author's `hlib.anchor` and this
verification's `vlib.anchor` both caught it -- but see **V-D20**: the refusal
arrives as the wrong exception.

---

## Verdict table

### H2-1 -- the direct-matrix MFT branch (`2796d551`)

| # | claim | verdict | measured (WIN / WSL) |
|---|---|---|---|
| 1 | ONE `xp`-parametrised implementation serving both index conventions | **CONFIRMED-WITH-CAVEAT** | `_bluestein_2d(method='direct')` calls it with no centres, `_bluestein_centred_2d(method='direct')` with all four (instrumented by monkeypatching the callee); centred-at-zero vs non-centred **bit-identical, max&#124;a-b&#124; = 0.0** at three shapes on both builds; no second dense implementation in `lumenairy/`. **Caveat:** the kernel BUILDER is NumPy-only (`np.arange` / `np.rint` / `np.exp`, then `xp.asarray`) -- `_bluestein.py:377-387`. It is `xp`-parametrised for the two `matmul`s only. |
| 2 | default byte-identical 179/179 archive-to-archive | **CONFIRMED** | independent 91-key set (no fixture shared with the author's): **70/70 legacy keys identical**, both builds; the 21 `NEW::` keys differ only because base raises `TypeError: unexpected keyword argument 'method'`. Cross-key check base-keyword-free vs branch-`method='auto'`: identical on all three propagators, both builds. |
| 3 | derived tolerance `(g_a+g_b)*eps*sum&#124;E&#124;`; every route 1.47-2.59 decades inside | **CONFIRMED at the shapes measured, REFUTED as a general statement** | see **V-D8**. Reproduced in kind: dense 1.84/2.26/2.62 decades, chirp 1.81/1.64/1.44 at the author's three cases. Extended: chirp margin falls **1.93 -> 1.06 -> 0.86 -> 0.73 decades** at N = 16/96/192/256; dense margin RISES **1.84 -> 3.60**. Bar never crossed up to N = 256 on either build. |
| 3a | is `sqrt(n)` the right growth for a float64 BLAS matmul? | **CONFIRMED, right number / wrong stated reason** | Higham's deterministic bound for a length-`m` inner product is `m*u/(1-m*u)`; with the code's `eps = 2u` convention two chained products of lengths `Nx`, `Ny` give `g_rigorous = (Nx+Ny)/2`. `sqrt(Nx*Ny)` is the GEOMETRIC mean where the rigorous bound is the ARITHMETIC one, so by AM-GM it is always at or below it -- **never above**, i.e. never a false pass. For a SQUARE input the two coincide exactly. At 1024x4 the author's bar is 8x tighter than rigorous. The stochastic-heuristic worry does not bite; the *justification text* is what needs correcting. |
| 4 | memory ordering `dense < separable < chirp-Z` at all 29 shapes | **CONFIRMED** | 29/29 on BOTH builds, zero exceptions, `tracemalloc` peak with all 22 registered cache clearers run cold and no wall clock read in the same pass. Worst margin `separable/dense = 1.2517` at N=M=1024 (67.11 / 84.00 / 553.89 MB). Ordering still holds outside the requested box at M = 16N and M = 8N. |
| 5 | the time crossover differs ~4x between builds (WIN `M <~ N/4`, WSL `M ~ N/16`) | **per-build CONFIRMED, the factor REFUTED (it is 2x, not 4x)** | at N = 1024 the Windows crossover brackets `M/N` in (1/8, 1/4) and the WSL one in (1/16, 1/8) -- adjacent octaves sharing the endpoint `M = N/8`. WSL ladder taken on a quiet box (3 Python processes of 12); Windows with ~20 resident. The decision the claim carries -- per-build, therefore no threshold may ship -- stands. One sentence does not survive: see the timing section. |
| 6 | CuPy dense route 3.79e-16 vs NumPy | **CONFIRMED-WITH-CAVEAT** | reproduced in kind but **fixture-dependent**: 2.803e-16 (16x16->8x8), 4.451e-16 (64->32), 5.309e-16 (128x96->48x64), 3.556e-16 (centred 32->16). The single quoted number needs its fixture. The capability is real: on this box `cupy.fft.fft2` and `_bluestein_2d(xp=cupy)` both die with `ImportError: DLL load failed while importing cufft` while `_direct_matrix_2d(xp=cupy)` runs. JAX also works, eager and under `jit`, at 3.72e-16 / 3.61e-16. |
| 7 | the phase-budget warning fires three decades late | **CONFIRMED -- and it is worse: 7.5 decades** | see **V-D5**. Against a `math.fsum` correctly-rounded reference at two geometries on both builds: the error is LINEAR in the budget, `rel ~ eps * alpha*N_max^2` -- **1.5e-08 already at a budget of 1e8**, 1.9e-04 at 1e12, 2.5e-01 at 1e15, first warning at 3.16e15. Dense route 2.8e-16 .. 4.5e-16 at every budget, never warns. The author's three quoted numbers reproduce exactly. |
| 8 | vocabulary CLOSED and checked | **CONFIRMED-WITH-CAVEAT** | 14 bad values x 2 primitives = 28/28 `ValueError`, both builds; 18/18 on the public entry points; no non-raising input found on any path (the `_bluestein_centred_2d` call is unconditional in all three). Caveats in **V-D17**, **V-D18**. |
| 9 | the decision tests must not pin BLAS bits | **CONFIRMED-WITH-CAVEAT** | `OPENBLAS_CORETYPE` IS honoured on both builds (`threadpoolctl` confirms the architecture changes). The dense route's bytes move: **2-4 distinct digests** over {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x threads {1,4}; `OPENBLAS_NUM_THREADS` alone moves them once the matrices are blocked; WIN and WSL are never bit-equal. Worst relative spread **1.25e-15**. `pytest tests/unit/test_wave5_h2_mft_direct.py` = **29 passed on all 8 rungs x 2 builds** (16 runs). Tightest bar in the file has 1.83 decades while the BLAS spread is <= 0.31 % of a bar -- ~22 000x of headroom. One assertion does pin BLAS bits (`:379`) and survives only because both sides run in the same process on the same kernel. |

### H2-2 -- `_collins_transport` on the field's backend (`76d013c3`)

| # | claim | verdict | measured (WIN / WSL) |
|---|---|---|---|
| 10 | NumPy path 84/84 identical | **CONFIRMED** | independent 93-key set: **93/93 identical on both builds**. Sensitivity control (so the instrument is not degenerate): the same 93 keys differ on **57** keys between WIN and WSL. |
| 11 | no `_jax` twin (structural inventory) | **CONFIRMED-WITH-CAVEAT** | AST: 139 functions in `carrier.py`, **0** backend-suffixed; the only `xp`-suffixed names are `_exact_tf_2d_xp` / `_fresnel_tf_2d_xp`, which are parametrised, not per-flavour. Package-wide (4523 functions): **0** backend-suffixed siblings of any `_collins*` helper. Context: 79 backend-suffixed functions DO exist elsewhere, including three genuine flavour pairs -- H2-2 is cleaner than the repo baseline. **Caveat:** no repo-wide census gate exists; the author's own gate inspects `dir(CA)` + a substring search of one file, so a twin in a SIBLING module would not be caught. |
| 12 | three of six helpers needed nothing because they route through `to_numpy`; giving them `xp` would have moved the NumPy summation order | **decision CONFIRMED, stated REASON REFUTED** | `_collins_power_marginals` bands at `_PHASOR_BAND_BYTES = 32e6` -> **62 500 rows per band at nx = 64**, so every Collins fixture is ONE band and the banded accumulation IS the whole-array reduction. Measured: `Px` and `Py` **bit-identical (0 ULP)** to a single `np.sum` on the Gaussian, on its spectrum and on a 10^+-8 dynamic-range random array, both builds. Bits only move above ~4 Mpixel (37 ULP at 4096x2935), which no Collins fixture reaches; `Py` is bit-identical in EVERY regime. `_collins_sampling_stats` really does take only Python floats (16 scalar parameters, no `np.*` in the body, raises on an array). |
| 13 | JAX parity 8.27e-16 against a bar of 1.61e-15 derived from the two backends' FFT spread | **CONFIRMED** | on an independent fixture (N=96, dx=5.5 um, w=41 um, R=-28 mm, z=2.3 mm, lambda=1.064 um, off-axis tilt) with an independently counted chain depth (4 transform-equivalents on `'fresnel'`, 5 on `'exact'`): spread **3.2498e-16 / 2.7681e-16**, bar **1.2999e-15 / 1.1072e-15**, fresnel **8.3014e-16 / 8.4036e-16**, exact **1.0543e-15 / 9.7747e-16**, smallest real signal **1.18733e-05** (~10 decades above the bar). The two JAX versions give spreads 17 % apart, which is why the bar must be measured per build. Re-running the author's own fixture reproduces **2.6853056e-16 / 1.6111834e-15 / 8.271056e-16 / 9.658315e-16** -- his digits exactly. **But see V-D16f:** the bar the test actually asserts against is the `32*eps` FLOOR (7.105e-15), which dominates the measured term by 4.4x; the report tabulates the pre-floor number. |
| 14 | the transport REFUSES under a trace, naming the two ways out; eager JAX/CuPy measure normally | **CONFIRMED-WITH-CAVEAT (four holes)** | 3 `gap_kernel` x 3 `on_collins_sampling` x 2 `stats_out` x {grad, jit} = 36 cells: **exactly one runs**, identical on both builds; also refuses correctly under `vmap`, `linearize`, `vjp`. Message names both ways out and counts the blocked decisions. Eager JAX measures normally and DOES raise the Kelly warning. Eager CuPy cannot run here (broken cuFFT). Holes: **V-D3**, **V-D11**, **V-D12**, **V-D13**. |
| 14a | is the gradient correct THROUGH the refusal path -- clean raise, no partial state? | **CONFIRMED** | after a refusing `jax.grad`: plain `ValueError`, cause chain `['ValueError']` only (**no** `JaxStackTraceBeforeTransformation` wrapper); the caller's `stats_out` dict byte-for-byte untouched; `_H_FFT_CACHE` len/hits/bytes unchanged (0/0/0); `warnings.filters` repr unchanged; a subsequent `jax.grad` on the allowed spelling works. Both builds. Independently re-confirmed by this verification's own probe. |
| 15 | `jax.grad` vs central FD 6.47e-13 against an `eps^(2/3)` floor of 3.67e-11 over a 7-step ladder | **CONFIRMED as a gradient check, REFUTED as a floor model** | see **V-D7**. On an independent merit and an 11-step ladder: best **3.9191e-15 / 4.9932e-14**, i.e. 93 564x / 7 344x inside the bar. The gradient is right -- checked additionally against a hand-derived analytic Fresnel-chirp gradient at **rel L2 1.80e-14**, correlation 1.0. But `P''' == 0` for both merits (the transport is LINEAR in the envelope, so the merit is an exact quadratic form): the ladder has no truncation branch and never brackets a minimum, so `eps^(2/3)` is the wrong floor and the bar is ~4-4.7 decades loose. Adding a genuinely cubic merit restores a textbook `h^2` arm and a real U minimum at h = 1e-4. |
| 16 | `_fft2_pair` returns the dispatch BY IDENTITY, and a wrapper would turn the chirp-kernel cache off for every Collins leg | **identity CONFIRMED, stated CONSEQUENCE REFUTED** | see **V-D9**. `_fft2_pair(np, False) is fft_infra._fft2` -> True/True; the cache key is at `_bluestein.py:642-646`. A wrapper does turn the cache off and costs 3.3x wall time on WSL. But `_collins_transport` passes `separable=bool(_EXACT_READOUT_SEPARABLE_BLUESTEIN)` (True), and `_bluestein_2d:561` then takes the NumPy-only separable route, which never reaches the cache at all: **0 entries, 0 hits over three legs**, both builds. The cache is off for every Collins leg TODAY. |
| 17 | `_as_c_order` because `jax.numpy` has no `ascontiguousarray` | **CONFIRMED** | absent on BOTH jax versions (0.11.0 and 0.10.2); CuPy accepts `dtype=` and its half runs on the DEVICE (no cuFFT needed); the NumPy path byte-identical in 16 layout/dtype cases on both builds. |

### H2-3 -- the near-focus exact-kernel table (`2d6eef9c`)

| # | claim | verdict | measured (WIN / WSL) |
|---|---|---|---|
| 18 | the bookkeeping validation (5.572e-12 / 1.838e-04 / 4.801e-13 both spellings / 1.759e-11 / 1.35) | **CONFIRMED-WITH-CAVEAT** | every row to the printed digits on both builds (5.5715211e-12 / 1.8380364e-04 / 4.8009649e-13 / 4.8009009e-13 / 1.7587095e-11 / 1.349875890). The three bars re-derived independently from `10*(eps*k*|z| + exp(-(N dx_out/2)^2/w_out^2))` come out **1.5553e-10 / 1.2341e-03 / 2.7107e-10** -- the author's, independently. **Caveat V-D16a:** "the two Collins spellings agree with each other exactly" is false; they differ by 1.334e-05 relative (6.4e-18 absolute). |
| 19 | the table: collins+fresnel at the oracle floor 5 mm -> 1 um; Sziklas 7.3e-4 within 100 um because the auto-split bridge engages; `'exact'` == `'auto'` | **CONFIRMED (c stronger than claimed)** | all 54 cells reproduce; worst build-to-build spread 1.47e-05 relative on the smallest cell, 242 of 300 recorded numbers bit-identical. (b) instrumented with counters: at d <= 100 um `_near_focus_needs_bridge` is True and the split runs; at 1 um the leg returns dx = 1.9098e-06 m against a plain co-moving `m*dx` = 1.3044e-08 m (146.4x). (c) `'exact'` is `==` in float64 to `'auto'` at all 9 rungs on BOTH transports. **Caveat V-D16b:** the Sziklas column has TWO mechanisms and the prose names one; its worst rung is 300 um (1.1565e-02) with the bridge OFF. |
| 20 | the derived law `1.2248 * k|z_eff| theta_env^4/8`, slopes 0.9999985 and 3.99970, same constant on a collimated leg through the other transport | **CONFIRMED, and the constant is explained** | on an INDEPENDENT fixture (lambda = 1.55 um, w0 = 22.0 um, f = 35.0 mm, N=512 at 16 um, each angle rung's window sized from its OWN beam): slope in `z_eff` **0.9999986** (worst deviation 1.87e-06 over a 205.8x span), slope in `theta_env` **3.99990** (worst 1.39e-04 over a 5.60x span), **C = 1.22475** (14 readings, spread 3.0e-04) against the author's 1.2248. **C = sqrt(3/2) = 1.2247449.** It is universal for a 2-D circular Gaussian envelope and NOT a universal number: the relative L2 of a field perturbed by a phase is the intensity-weighted RMS of that phase, and `<u^8> = 3/2` exactly under the spectral weight `exp(-2u^2)` (checked by quadrature: 1.5000000002). The same integral in 1-D gives `sqrt(105/256) = 0.6404` (measured 0.6404344). Corroborated by a numpy-only oracle that never calls the transport (Parseval), agreeing to 11 digits on the collimated leg. Independently re-confirmed by this verification's own sign/magnitude gate (below). |
| 20a | this verification's own corroboration of the law | **new measurement** | on the H2-2 fixture (633 nm, N=64, dx=8 um, w=60 um, R=-50 mm, z=5 mm), `z_eff = 5.5556e-03 m`: departure **1.07407e-06** against `sqrt(3/2) k z_eff theta^4/8` = **1.07367e-06** (0.04 %), and the MEAN correction phase **-4.38331e-07** against the `<u^4> = 1/2` moment `-k z_eff theta^4/16` = **-4.38323e-07** (0.002 %). Both moments of the same quartic, a third fixture, both builds. |
| 21 | VERIFY-B4 F3's beam-angle quartic is 4.0e5 too large because the ENVELOPE angle is what matters | **CONFIRMED-WITH-CAVEAT** | F3's fixture rebuilt exactly (lambda = 1.064 um, N = 1024, dx = 4 um, w = 0.30 mm, R = -40 mm, dx_out = 5.6447e-06, N_out = 128): `'fresnel'` **1.7042e-14**, `'auto'` **2.3497e-03**, `k4` **9.1137e-03**, `z_eff` **1600.04 m**, resolved `exact`, at dz = 1 um -- F3's published numbers to three digits, WSL identical. The two reports measure the SAME quantity: instrumented against a clipped window (N_out 512 and 4x coarser dx_out move `'auto'` only in the 6th digit), a wrapped chirp-Z (`K3 = 0.068`), the grid (identical at N = 512/1024/2048) and the reference. The ratio is the law: predicted 498.2x from `k`, `z_eff` and the analytic envelope angle, measured **498.1x**. **Caveat V-D16c:** the 4.0e5 arithmetic is right for WP-B11 sec. 2.20 (which did use the beam's 20 mrad), but **VERIFY-B4 F3 itself used the envelope's MEASURED containment angle**, which over-predicts by **33.5x** -- not five decades. The report's decision paragraph attributes the beam-angle framing to F3. |
| 22 | the `k4` gate table and "the gate is not near firing" | **CONFIRMED** | author's fixture: `k4` = 2.2309e-05 / 5.5069e-06 / 1.0861e-07, `z_eff` = 12.2658 / 3.0277 / 0.0597 m, `K1` = 0.0325 / 0.0364 / 0.2928, envelope angle 1.9530643e-03 measured / 7.9512089e-04 analytic -- every digit, both builds. Peak 4.65 decades below the bar of 1. On F3's fixture `k4` is still 2 decades below the bar at the rung where `'auto'` costs 2.3e-03: the gate bounds REPRESENTABILITY, not accuracy. |
| 23 | recommendation "no change" | **REFUTED** | see the maintainer answer below and **V-D16d**. |

### H2-4 -- the loud stale-patch refusal (`a985ac2c`)

| # | claim | verdict | measured (WIN / WSL) |
|---|---|---|---|
| 24 | refuse-the-write chosen because live forwards drop `CUPY_AVAILABLE` from `import *` and lay a `LOAD_GLOBAL -> NameError` trap | **CONFIRMED-WITH-CAVEAT** | rebuilt on independent synthetic modules with `from ... import *` executed as a real statement into an empty namespace: `import *` **True / False**, `vars()` **True / False**, bare-name read **works / `NameError`**, write reaches the leaf **False / True** -- identical on both builds, no py3.12/3.14 difference. **Caveat:** the `dir()` row is True for option A only BECAUSE of the `__dir__` override; without it, False. AST walk of `lenses.py`: **0** bare-name loads, **0** bare calls, **0** attribute writes, **0** `globals()[...]` subscripts -- the author's zero confirmed. Control: `_lens_kernels.py` itself has 6 bare calls / 7 bare loads of the eight, which is exactly why a facade write cannot reach them. |
| 25 | 99/131 identical + exactly the 32 write/delete keys moved + 0 unexpected | **CONFIRMED (agrees in kind)** | independent 115-key set with module state saved and restored by direct `__dict__` mutation around every attempt: **83 identical / 32 expected-differ-and-did / 0 expected-differ-but-identical / 0 unexpected / 0 only-base / 0 only-branch**, both builds. The 32 are exactly the eight x {write, read-after-write, delete, read-after-delete}. |
| 26 | `__setattr__`/`__delattr__` raise `AttributeError` naming `_lens_kernels` while reads, `dir()` and `import *` are untouched | **CONFIRMED** | 16/16 raise, all name the leaf, verb correct. Nothing moved on either module (`shell_unmoved` and `leaf_unmoved` True for all eight). On BASE the same attempts DO write and DO delete -- the defect reproduced. Behavioural proof of the redirect: patch `_lens_kernels._is_cupy_array` with a counting fake, call `lenses.surface_sag_general` -> fake called **1** time on both trees; patch the FACADE instead -> base succeeds and the fake is called **0** times (the silent no-op), branch raises. `dir(lenses)` **74 entries identical**, `import *` **38 names identical**, `vars(lenses)` **66 keys identical**, type and MRO identical, base vs branch, both builds. |
| 27 | does ANY existing test or library site write to one of the eight on the facade? | **CONFIRMED: none. No blocker.** | AST walk of EVERY `.py` in each tree with four finders (attribute `Assign`/`AugAssign`/`AnnAssign`/`Delete`; `setattr`/`delattr` with a literal name; `monkeypatch.setattr` / `mock.patch` / `patch.object` in both the object+string and the single dotted-string forms) plus a regex net: **0 facade write sites**. Two writes to `_lens_kernels` DIRECTLY (`tests/unit/test_verify_b11c_structure.py:645,663`) still work. `tests/unit/test_audit2609_a16_verify_config_and_arch.py:620` parametrises over `(_lens_real, _lens_traced, _lens_kernels)` and explicitly excludes `lenses`. `conftest.py` files: 0 mentions. |
| 28 | `importlib.reload(lumenairy.elements.lenses)` still works | **CONFIRMED, mechanism proved** | reload OK on both trees and both builds; reload of `_lens_kernels` OK; a second reload OK; a fresh subprocess import OK. The mechanism was confirmed by COUNTING, not deduced: wrapping the live facade class's `__setattr__` records exactly **8** calls during a reload -- `__spec__, __name__, __loader__, __package__, __spec__, __file__, __cached__, __class__` -- and **none** is one of the eight. The module body's `from ._lens_kernels import ...` compiles to `STORE_NAME`, which writes `__dict__` directly. |
| 28a | idempotent self-assignment through the facade | **measured, not a blocker** | `setattr(lenses, n, getattr(lenses, n))` RAISES on the branch for all eight. No library or test code does this today (see 27), so nothing breaks; it remains a shape a future author could reach for, and the refusal message redirects them. |

### Durability of the four new/extended test files

| # | claim | verdict | measured |
|---|---|---|---|
| 29 | every new test's bar is derived, two-sided, premise-gated, and FAILS (not skips) on a silent regression | **CONFIRMED-WITH-CAVEAT** | `test_wave5_h2_near_focus_table.py`: 141 numeric literals AST-enumerated, **no expected measurement value pinned anywhere**; every comparison bar derived at runtime; 5 premise-gated ids; 5 two-sided asserts. `test_wave5_h2_mft_direct.py`: bars derived from the summation. Caveats: **V-D6** (a pinned `1e-6` with 0.50 decades), **V-D7** (a floor model that does not apply), **V-D8** (a bar claimed outside its range of validity), and the weakest pin `0.5 < C_conv < 3.0`, a factor-6 window that the 1-D constant 0.6404 would pass. |
| 30 | mutation matrix, >= 10 mutations across the four files, both builds | **16 arms run; 13 caught, 3 NOT** | table below. |

**Mutation matrix** (every arm applied to a tree copy under the scratchpad;
`pytest -q --capture=sys -p no:randomly`; both builds agree arm for arm):

| file | arm | WIN | WSL | caught |
|---|---|---|---|---|
| `_bluestein.py` | m1 dense route returns the separable chirp-Z answer | 3 failed / 26 | 3 failed / 26 | **yes** |
| | m2 drop the `t - rint(t)` phase reduction | 2 failed / 27 | 2 failed / 27 | **yes** |
| | m3 always associate y-first | 2 failed / 27 | 2 failed / 27 | **yes** |
| | m3b same rule change, source shape preserved | 1 failed / 28 | 1 failed / 28 | **yes** (by exactly one assertion) |
| | m4 centred primitive drops the four centres | 4 failed / 25 | 4 failed / 25 | **yes** |
| `carrier.py` (H2-2) | m1 `bld` threading bypassed in `_collins_axis_chirp` | 21 passed | 21 passed | **NO** |
| | m2 traced refusal removed | 7 failed / 14 | 7 failed / 14 | **yes** |
| | m3 `_fft2_pair` returns wrappers (breaks `is`) | 1 failed / 20 | 1 failed / 20 | **yes** |
| | m4 `_as_c_order` always `xp.asarray` | 21 passed | 21 passed | **NO** |
| `carrier.py` (H2-3) | m1 `'auto'` -> `'fresnel'` unconditionally | 2 failed / 12 | 2 failed / 12 | **yes** |
| | m2 `k4` gate always fires | 6 failed / 8 | 6 failed / 8 | **yes** |
| | m3 **sign of the exact kernel's `z_eff`** | **14 passed** | **14 passed** | **NO** |
| | m4 carrier applied twice on the output | 3 failed / 11 | 3 failed / 11 | **yes** |
| `lenses.py` | m1 facade refusal removed | 17 failed / 40 | 17 failed / 40 | **yes** |
| | m2 refuse on `__setattr__` only | 9 failed / 48 | 9 failed / 48 | **yes** |
| | m3 drop `_lens_kernels` from the message | 16 failed / 41 | 16 failed / 41 | **yes** |
| | m4 union `_LEAF_OWNED_NAMES` into `__dir__` | 57 passed | 57 passed | no -- **and correctly so**: measured a true no-op (`dir(lenses)` 74 entries, byte-identical with and without) |

All three genuinely undetected arms are closed by `tests/unit/test_verify_wave5_hyg2.py`
(see **V-D4**, **V-D14**).

### Citations

| # | claim | verdict |
|---|---|---|
| 31 | the three re-anchored 5.47.0-block citations point at the right content | **2 PASS, 1 WRONG-CONTENT -- and the block as a whole is REFUTED**: see **V-D2** |

---

## Defects, with reproducers and exact requested edits

### V-D1 -- SHIPPING BLOCKER: the branch breaks the repo's own `.test_durations` freshness gate

`tests/unit/test_audit2609_a15a_durations_staleness.py::test_durations_covers_at_least_98_percent_of_collected_ids`
is load-bearing CI infrastructure (`unit-tests.yml` and `publish.yml` both shard
with `pytest-split --splitting-algorithm least_duration`).

MEASURED, same box, same flags:

| tree | result |
|---|---|
| base `f4f18851` | **4 passed** in 142.32 s |
| branch `b86fe31f` | **1 failed, 3 passed** in 177.40 s -- `393 of 16370 collected ids (2.40%) carry no timing, above the 2% bar` |

The worst-files list names exactly the branch's own additions:
`test_verify_b11c_structure.py` (57), `test_wave5_h2_mft_direct.py` (29),
`test_wave5_h2_collins_jax.py` (21), `test_wave5_h2_near_focus_table.py` (14)
= **121 ids**, which is 0.74 of the 2.40 percentage points.  Without them the
branch reads 1.67 %, under the bar -- i.e. this is a regression the branch
introduces, not pre-existing staleness.

Reproducer:
```
cd /c/tmp/lum_vhyg2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LUMENAIRY_MEM_BUDGET_MB=8192 python -m pytest \
  tests/unit/test_audit2609_a15a_durations_staleness.py -q --capture=sys
```

**Requested edit:** splice the 121 new ids into `.test_durations` (the file's own
docstring carries the serial regeneration procedure), or regenerate both lanes.
This verification has spliced its own 12 ids; the branch's 121 remain.

### V-D2 -- SHIPPING BLOCKER: 15 further stale source-line citations in the same CHANGELOG block

Three citations were re-anchored.  Verified by `git show` of both blobs, both
builds:

| # | citing site | now cites | should be | verdict |
|---|---|---|---|---|
| 1 | `CHANGELOG.md:1709` | `mft.py:588` | 588 | PASS |
| 2 | `CHANGELOG.md:2105` | `mft.py:645-654` | **`mft.py:649-658`** | **WRONG-CONTENT** -- the reading the sentence points at is at 657-658, outside the cited range; `+34` was applied where the content moved `+38` |
| 3 | `CHANGELOG.md:2256` | `carrier.py:1598` | 1598 | PASS |

The 5.47.0 block (`CHANGELOG.md:565-4669`) contains **19** `path.py:N` citations
into the four files this branch moved lines in.  **Fourteen more are stale**,
plus one bare sibling:

`mft.py`: `:489`->527 (two lines below the one citation that WAS fixed, in the
same sentence), `:769`->807, `:972`->1050, `:605-630`->643-668.
`carrier.py`: `:1966`->2041, `:1564`->1621, `:1626`->1683 (`_collins_axis_chirp`,
signature changed by H2-2), `:2266`->2418, `:1830`->1895, `:1700`->1765 and its
bare sibling `:1736`->1801 (both at `CHANGELOG.md:2240`), `:2127`->2279,
`:1906`->1971 (`_collins_exact_kernel_correction`, signature changed by H2-2),
`:1868`->1933, `propagators/carrier.py:1068`->**1125**.

Root cause, mechanical: `validation/probe_wp_b11c/reanchor_citations.py`'s
`OWNED` map (lines 38-46) covers only `_lens_real.py`, `lenses.py`,
`lenses_maslov.py` and `rcwa/_core.py`, and sources its "before" numbers from
`B11C_BASE=96cb2096`.  It can see neither `mft.py`/`carrier.py` nor this
branch's base.  Run on the branch it prints `0 re-anchored` -- a green that means
nothing here.

**V18 passes all fifteen.**  `scripts/check_source_line_citations.py` reads
`ok=107 drift=0 total=107`, rc=0 on both builds, because every stale citation
lands on a non-trivial line.  That is exactly the blind spot
`reanchor_citations.py`'s own docstring names.  **The branch's green walker tail
is not evidence its citations are right.**

**Requested edit:** extend `OWNED` with
`'mft.py': ('lumenairy/propagators/mft.py', [])`,
`'carrier.py': ('lumenairy/propagators/carrier.py', [])`,
`'_bluestein.py': ('lumenairy/propagators/_bluestein.py', [])`
and re-run with `B11C_BASE=f4f18851` (the tool is idempotent and sources from
`git show`), or apply the 16 numbers by hand.  Either way citation 2 must become
`lumenairy/propagators/mft.py:649-658`.

### V-D3 -- MAJOR: H2-2 reaches the private transport, not the public `transport='collins'` leg

`lumenairy/propagators/carrier.py:2337`, in `_collins_carrier_leg`:
```python
    env_a = np.asarray(env)
```
unchanged by H2-2.  MEASURED, same probe on both trees:

| input | `_collins_transport` base | `_collins_transport` branch | public `transport='collins'` (base AND branch) | public default (sziklas) |
|---|---|---|---|---|
| numpy | ndarray | ndarray | ndarray | ndarray |
| eager JAX | **numpy.ndarray** | **jaxlib ArrayImpl** | **numpy.ndarray** (silent host demotion, bitwise equal to the NumPy arm) | ArrayImpl |
| CuPy | `TypeError: Implicit conversion ...` | reaches cuFFT (stayed on CuPy) | **`TypeError: Implicit conversion to a NumPy array is not allowed`** at `carrier.py:2337`, no mention of the transport | reaches cuFFT |
| traced | n/a | the DESIGNED `ValueError` naming both ways out | **raw `TracerArrayConversionError`** -- even when the caller passes BOTH documented ways out | works |

So the port is real at `_collins_transport` and the CHANGELOG's user-facing
sentence over-claims:

> "It now runs on NumPy, CuPy and JAX, selected by the array the caller passes,
> exactly as every other leg in the module is."

It does not, at the surface a caller reaches.  Consequence for the gate: the
author's `tests/unit/test_wave5_h2_collins_jax.py:208
test_the_public_entry_agrees_across_backends[collins]` compares NumPy to NumPy
and would pass even if `_collins_transport`'s JAX arm were deleted.

Reproducer:
```
cd /c/tmp/lum_vhyg2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LUMENAIRY_MEM_BUDGET_MB=8192 PYTHONHASHSEED=0 PYTHONPATH=C:/tmp/lum_vhyg2 \
  python validation/probe_verify_wave5_hyg2/v0_backend_reach.py
```

**Requested edit -- either (a) thread the leg**, `carrier.py:2337`:
```python
    xp, is_jax, _bld = _backend_of(env)
    env_a = xp.asarray(env)
```
(and `_collins_input_box`'s own transform at `carrier.py:1757-1759` must then use
`_fft2_pair` / `_as_c_order` instead of
`_fft2(np.ascontiguousarray(env, dtype=np.complex128))`);

**or (b) say so and gate it.**  Narrow the CHANGELOG sentence to
`_collins_transport` and add: "Not yet: `propagate_carrier_referenced(
transport='collins')` still converts to host NumPy at `_collins_carrier_leg`
(`carrier.py:2337`), so at the public surface a JAX array is demoted and a CuPy
array raises."  Then change
`test_the_public_entry_agrees_across_backends[collins]` to assert the conversion
rather than a parity it cannot observe.  This verification's
`test_the_public_collins_leg_demotes_the_backend_but_never_the_answer` bounds
the defect meanwhile (the demotion must stay a backend downgrade, never a change
of answer).

### V-D4 -- the sign of the exact-kernel correction is invisible to the file that publishes its law (the LIBRARY is guarded elsewhere)

`|exp(i phi) - 1|` is EVEN in `phi`, so every magnitude
`tests/unit/test_wave5_h2_near_focus_table.py` reads -- the relative L2 against
the oracle, the exact-minus-fresnel departure, the monotonicity, both power-law
slopes and `C` -- is invariant under `phi -> -phi`.  MEASURED: flipping
`phase *= z_eff` to `phase *= -z_eff` in `_collins_exact_kernel_correction`
(`carrier.py:2029`) leaves **all 14 ids green on both builds**, while the
intensity-weighted mean of `arg(exact/fresnel)` flips
`-1.926583e-06 -> +1.926587e-06`.  The file's sign-blindness is therefore
proved live, not argued.

**The package-wide check completed and it CORRECTS the first reading of this
defect.**  Against the same mutation,
`tests/unit/test_audit2609_b4_collins_transport.py` reads **2 failed / 124
passed** (base: 126 passed), both failures in
`TestSameTheorem::test_collins_on_the_co_moving_lattice_is_the_sziklas_step` --
`[auto-0.012]` at 8.1876e-08 against a 1e-9 bar and `[auto-0.02]` at 1.8760e-07;
the `[fresnel-...]` cells correctly pass.  The mechanism is exactly the
distinction that matters: that test compares the Collins `'auto'` output against
the SZIKLAS `'auto'` output, and the mutation touches only the Collins copy of
the kernel, so a cross-IMPLEMENTATION agreement test IS sign-sensitive, while
every bar in `test_wave5_h2_near_focus_table.py` is a magnitude against a
paraxial oracle and is even in the phase sign.

So the verdict is narrower than "a conjugated kernel would ship": **the library
is not unguarded, and the file that publishes the law cannot see the sign.**  The
edit below is DEFENCE IN DEPTH in the file that states the physics, not the only
guard -- and it is worth having precisely because the existing guard is an
agreement test between two copies (see **V-D22**), which would evaporate the day
those copies are consolidated.

The sign is not a convention: `sqrt(k^2-q^2) < k - q^2/(2k)` for every real `q`,
so the exact kernel RETARDS and on a leg with `z_eff > 0` the mean correction
phase is negative, with size `-k z_eff theta_env^4 / 16` (the `<u^4> = 1/2`
moment, beside the `sqrt(<u^8>) = sqrt(3/2)` moment the magnitude law uses).
Independently re-measured on a third fixture: **-4.38331e-07 against a predicted
-4.38323e-07 (0.002 %)**.

**Closed here** by
`tests/unit/test_verify_wave5_hyg2.py::test_the_exact_kernel_retards_the_envelope_and_the_sign_is_gated`,
which gates BOTH moments of the same quartic.
**Requested edit** in the author's file, after `:545` (`C_conv = ...`), where
`a`, `b` and `quartic` already exist:
```python
    # THE SIGN.  |e^{i phi} - 1| is EVEN in phi, so every magnitude in this
    # file is invariant under phi -> -phi: a sign-flipped z_eff in
    # _collins_exact_kernel_correction passes all 14 ids untouched (MEASURED
    # 2026-09-19, both builds).  The exact kernel RETARDS relative to the
    # paraxial one -- sqrt(k^2-q^2) < k - q^2/(2k) -- so on a leg with
    # z_eff > 0 the intensity-weighted mean of arg(exact/fresnel) is NEGATIVE,
    # with the RMS factor sqrt(3/2) replaced by the MEAN factor 1/2.
    wgt = np.abs(b.env) ** 2
    mean_phi = float((wgt * np.angle(a.env / b.env)).sum() / wgt.sum())
    assert mean_phi < 0.0, (
        f"the exact-kernel correction ADVANCES the envelope (mean phase "
        f"{mean_phi:+.4e}); sqrt(k^2-q^2) - k + q^2/(2k) is negative, so the "
        f"refinement's z_eff carries the wrong sign")
    assert abs(mean_phi) == pytest.approx(quartic / 2.0, rel=1e-2), (
        f"the mean correction phase is {mean_phi:.4e}, not the derived "
        f"-k z_eff theta^4/16 = {-quartic / 2.0:.4e}")
```
Verified, not proposed: with this edit the file reads **14 passed** on base and
**1 failed, 13 passed** on the sign-flipped tree, on both builds.

### V-D5 -- MAJOR: the chirp phase-budget warning is 7.5 decades late, and the report says three

`lumenairy/propagators/_bluestein.py:542-555` (guard) and `:505-510` (its Notes).
Against a `math.fsum` correctly-rounded reference at N=24 M=12 AND N=48 M=24, 23
budgets from 1e6 to 1e17, both builds to the digit:

| budget | 1e8 | 1e10 | 1e12 | 1e14 | 1e15 | 1e17 |
|---|---|---|---|---|---|---|
| chirp-Z rel L2 | **1.5e-08** | 2.0e-06 | **1.9e-04** | 1.3e-02 | **2.5e-01** | 1.6e+00 |
| warned? | no | no | no | no | no | yes |
| dense rel L2 | 3.3e-16 | 3.3e-16 | 3.5e-16 | 3.1e-16 | 3.1e-16 | 2.8e-16 |

The error is LINEAR in the budget -- `rel ~ eps * alpha * N_max^2` -- verified
over 11 decades at two geometries, so there is no cliff to sit just below.
First budget that warns: **3.16e15**.  The author's three quoted numbers
reproduce exactly; the characterisation "three decades late" understates it.

**Requested edit** (threshold derived from the measured law, not from taste --
`1e-6/eps = 4.5e9` is the budget at which six significant figures remain; shipped
callers stay silent because at the natural MFT grids `alpha = zoom/N`, so
`budget = zoom*N ~ 1e4` at N=1024 with 10x zoom):
```diff
-    if phase_budget > 1e15:
+    _EPS64 = float(np.finfo(np.float64).eps)
+    _PHASE_BUDGET_MAX = 1e-6 / _EPS64                 # 4.5e9
+    if phase_budget > _PHASE_BUDGET_MAX:
         import warnings
         warnings.warn(
-            f"Bluestein chirp phase argument ~{phase_budget:.1e} approaches "
-            f"float64 precision limit (1e15-1e16).  Reduce N or alpha, "
-            f"or fall back to a regular FFT propagator.",
+            f"Bluestein chirp phase argument ~{phase_budget:.1e} exceeds the "
+            f"float64 chirp budget {_PHASE_BUDGET_MAX:.1e}; the chirp-Z "
+            f"routes' relative error at this budget is "
+            f"~{phase_budget * _EPS64:.1e}.  Reduce N or alpha, pass "
+            f"method='direct' (the dense route reduces its phase modulo one "
+            f"turn and measured 3e-16 at every budget tested), or fall back "
+            f"to a regular FFT propagator.",
             RuntimeWarning, stacklevel=2)
```
with the Notes at `:505-510` restated to name the linear law and the measured
readings.  This IS a behaviour change on existing quiet callers, which is why it
is a maintainer decision; the measurement says 1e15 is indefensible on any
criterion and the author's own suggested 1e12 is still four decades late by a
1e-8 criterion.

**Not pinned here.**  `tests/unit/test_verify_wave5_hyg2.py::test_the_chirp_phase_error_is_linear_in_the_budget_and_dense_is_immune`
gates the LAW and the dense route's immunity, and deliberately does not pin the
threshold, so it stays true whatever the threshold becomes.

### V-D6 -- MAJOR: the `corr > 1.0 - 1e-6` bar is pinned, undressed and 0.50 decades from firing

`tests/unit/test_wave5_h2_collins_jax.py:441-445`.  MEASURED on BOTH builds:
`1 - corr = 3.168867e-07` against a bar of `1e-6` -- **3.16x, i.e. 0.50 decades**.
The constant has no stated origin, which the testing standards call a defect on
its own.  Worse, the quantity is a property of the FIXTURE, not of the port:

| envelope width `w` | 30 um | 40 um | 50 um | **60 um (shipped)** | 70 um | 80 um | 100 um |
|---|---|---|---|---|---|---|---|
| `1 - corr` | 6.08e-11 | 1.15e-10 | 6.30e-09 | **3.17e-07** | **6.93e-06** | 6.44e-05 | 1.13e-03 |

A 17 % change of the fixture's width puts it **7x over the bar**; the quantity
moves 4.3 decades over a 3.3x width range, because a wider envelope on a fixed
64x64 grid at 8 um clips harder and the leg departs further from a scaled
isometry.

**Requested edit:** derive the bar from that departure, which the running build
can measure.  The merit is `P(a) = a^T (L^H L) a`, so `grad P = 2 (L^H L) a` is
proportional to `a` exactly when `a` is an eigenvector of `L^H L`, and Pearson's
shortfall is second order in the residual:
```python
    c = float(np.dot(g[m], a[m]) / np.dot(a[m], a[m]))
    r = float(np.linalg.norm(g[m] - c * a[m]) / np.linalg.norm(c * a[m]))
    assert 0.0 < r < 1e-1, ("PREMISE: ...")
    assert (1.0 - corr) < 10.0 * r * r, (...)
    assert (1.0 - corr) > 0.01 * r * r, (...)
```
MEASURED on the shipped fixture, both builds: `r = 4.998e-04`, `r^2/2 =
1.2490e-07`, `(1-corr)/(r^2/2) = 2.54`.  Gated here by
`test_the_gradient_shape_bar_is_the_legs_departure_from_a_scaled_isometry`.

### V-D7 -- MAJOR: the gradient bar's `eps^(2/3)` floor model does not apply to its own merit

`_collins_transport` is LINEAR in the envelope, so `P(a) = sum|L a|^2` is an
exact quadratic form and `P''' == 0`.  A central difference of a quadratic has
NO truncation error, so the ladder has no U-curve minimum: the disagreement
falls MONOTONICALLY with `h` over the whole ladder.  MEASURED (WIN, the author's
own merit and fixture, ladder extended upward):

| h | 1e-1 | 3e-2 | 1e-2 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|---|
| rel disagreement | **7.55e-15** | 1.73e-13 | 6.47e-13 | 2.07e-12 | 1.21e-11 | 7.23e-10 |

The author's ladder stops at `h = 1e-2`, entirely on the round-off branch, and
reads 6.47e-13; the true best is 7.55e-15 at the top.  `eps^(2/3)` is the
minimum of a genuine U-curve, so the bar is **4.7 decades looser than the
measurement** and the test would pass with a gradient wrong by three decades.
(The gradient is nonetheless right: independently checked against a hand-derived
analytic Fresnel-chirp gradient at rel L2 1.80e-14, correlation 1.0.)

**Requested edit -- either** derive the floor from the cancellation branch alone,
`eps*|P|/(h*|dP/da|)` at the ladder's best `h`, **or** use a merit with a real
`P'''` so the stated model applies.  MEASURED: `M3 = (|E_out(0,0)|^2)^3` gives a
textbook `h^2` truncation arm over three decades (3.67e-06 at h=1e-1 falling
exactly 10x per decade) and a real minimum at h = 1e-4 (5.16e-12), i.e. a
full U on both builds.  Either way the docstring's "the U-curve's upturn below
h = 1e-4 is visible" should say that the upturn is the cancellation branch and
that the truncation branch is absent by construction.  Gated here by
`test_the_central_difference_through_this_merit_has_no_truncation_branch`.

### V-D8 -- MAJOR: the summation tolerance is claimed outside its range of validity, and the stated reason is backwards

The report says the routes sit "1.47 to 2.59 decades inside it -- the margin
narrows with `n`, as it should, because the growth factors are upper bounds and
the actual errors grow more slowly."  A bar that grows faster than its quantity
gives a WIDENING margin.  The two routes do opposite things:

| N -> M | 16->8 | 48->24 | 96->48 | 128->64 | 192->96 | 256->128 |
|---|---|---|---|---|---|---|
| chirp-Z decades inside | 1.93 | 1.34 | 1.06 | 1.10 | **0.86** | **0.73** |
| dense decades inside | 1.84 | 2.65 | 2.89 | 3.05 | 3.27 | **3.60** |

So the dense route's margin WIDENS (its error grows ~`sqrt(n)` while its bar
grows ~`n`) and the chirp-Z routes' margin NARROWS (their error grows FASTER
than `g_chirp * eps * sum|E|`).  The bar is never crossed up to N = 256 on
either build, and the shapes the shipped test file uses (max 32x24) are 1.6-2.4
decades inside -- so **the shipped assertions are sound**; what is wrong is the
generality of the claim and the sentence explaining it.  Extrapolating the chirp
trend, the chirp bar would first be crossed near N ~ 4000-5000, outside anything
the file exercises but inside nothing about the formula that says so.

**Requested edit**, `WAVE5_HYGIENE2_REPORT.md` (the "Stated tolerance"
paragraph) and the module docstring of
`tests/unit/test_wave5_h2_mft_direct.py`: replace "the margin narrows with `n`,
as it should, because the growth factors are upper bounds and the actual errors
grow more slowly" with "the DENSE route's margin WIDENS with `n` (1.84 -> 3.60
decades from 16x16->8x8 to 256x256->128x128, measured) because its error grows
like `sqrt(n)` against a bar that grows like `n`; the chirp-Z routes' margin
NARROWS (1.93 -> 0.73 over the same ladder) because theirs grows faster than
`3*log2(L^2) * eps * sum|E|`.  The bar is a bound for both at every shape
measured; the chirp-Z half of it is the one with a finite lifetime."  Gated here
by `test_the_two_reductions_margins_move_in_opposite_directions_with_n`.

### V-D9 -- the `_fft2_pair` cache consequence is false as shipped

MEASURED, three identical Collins legs, cache cleared first, both builds:

| arm | entries | hits | engaged |
|---|---|---|---|
| Collins leg, shipped (`_EXACT_READOUT_SEPARABLE_BLUESTEIN = True`) | **0** | **0** | **no** |
| Collins leg with the separable flag forced False | 1 | 2 | yes |
| primitive, `separable=False`, raw `fft_infra._fft2` | 1 | 2 | yes |
| primitive, `separable=False`, through a wrapper | 0 | 0 | no |

The MECHANISM is exactly as described (a wrapper does turn the cache off, worth
3.3x wall time on WSL), but its premise fails: `_collins_transport` passes
`separable=True` (`carrier.py:2249`, flag at `:6091`) and `_bluestein_2d:561`
then takes the NumPy-only separable route, which never reaches the cache.  The
identity is still worth keeping -- it protects the `separable=False` route and
every other caller of the 2-D arm.

**Requested edit:** `WAVE5_HYGIENE2_REPORT.md:323-325` and the identical
sentence in `lumenairy/propagators/carrier.py:528-530`: after "would make that
test false", replace "and silently turn the cache off for every Collins leg"
with "MEASURED: with the shipped `_EXACT_READOUT_SEPARABLE_BLUESTEIN = True` a
Collins leg takes the separable route (`_bluestein.py:561`), which never reaches
that cache at all -- 0 entries and 0 hits over three legs -- so the cache is off
for every Collins leg today, and the identity is what keeps it available to the
`separable=False` route and to every other caller of the 2-D arm."

### V-D10 -- `CUPY_AVAILABLE` is cited as False; it reads True

`WAVE5_HYGIENE2_REPORT.md:438-439` and `:792-793`.
Reproducer: `python -c "from lumenairy.propagators import fft_infra as FI;
print(FI.CUPY_AVAILABLE)"` -> `True`; the failure is in the first transform
(`ImportError: DLL load failed while importing cufft`).  The CONCLUSION (CuPy not
exercised on hardware) stands; the evidence cited for it does not.
**Requested edit:** "reads False on this Windows build" -> "reads **True** on
this Windows build (cupy 14.0.1, one visible device), but the first transform
raises `ImportError: DLL load failed while importing cufft`".  (Note that
`_as_c_order`'s CuPy half DOES run on the device, since it needs no FFT.)

### V-D11 -- the traced-refusal rule is stated more strictly than the code enforces

An ASTIGMATIC carrier with `gap_kernel='auto'` and
`on_collins_sampling='ignore'` **RUNS** under `jax.grad` (max|g| =
1.9999999996), because the kernel clause at `carrier.py:2144` is guarded by
`and Ax == Ay`.  Numerically benign -- the eager path resolves that same
configuration to `'fresnel'` too -- but the docstring at `carrier.py:2085-2090`
and the report say "refuses unless `gap_kernel='fresnel'`".
**Requested edit:** append to that paragraph -- "An ASTIGMATIC carrier has no
exact-kernel arm at all (`Ax != Ay`), so `gap_kernel='auto'` takes no measured
decision there and IS accepted under a trace; explicit `'exact'` is refused
earlier by the astigmatic gate."

### V-D12 -- a closed-over JAX constant under `jax.jit` fails with an undesigned error

`_is_traced(env_a)` is False for a concrete `jnp` array closed over by a jitted
function, so the measuring branch runs -- but inside a jit trace every `jnp`
operation is staged, so `fft2(...)`'s output IS a Tracer and
`_collins_power_marginals`'s `to_numpy` (`carrier.py:1651-1652`) raises
`TracerArrayConversionError`.  MEASURED for all four spellings including the
allowed `fresnel`/`ignore`, both builds.  A closed-over NumPy constant runs fine.
**Requested edit**, after `carrier.py:2181`:
```python
        if _is_traced(S):
            raise ValueError(
                f"{fn}: the envelope is a concrete array, but an enclosing "
                f"jax.jit / jax.grad trace STAGES every jax.numpy operation, "
                f"so this measurement transform's output is a Tracer and the "
                f"two measured decisions cannot be taken on it.  Pass the "
                f"envelope as an ARGUMENT of the traced function -- then "
                f"gap_kernel='fresnel' with on_collins_sampling='ignore' runs "
                f"and every other spelling is refused by name -- or move the "
                f"call outside the trace.")
```
(`jax.core.trace_state_clean` is not available on either jax version here, so the
value-level check is the portable form.)

### V-D13 -- a traced SCALAR argument blows up with an unhelpful error

`jax.grad` w.r.t. `z` or `R_in` with a concrete envelope raises
`ConcretizationTypeError ... The problem arose with the 'float' function` (from
`float(z)` in `_collins_envelope_abcd`, `carrier.py:1638`); w.r.t. `dx` raises
`TracerArrayConversionError`.  Neither mentions Collins, the transport, or a
remedy.  Independently reproduced by this verification
(`v0_public_entry_trace.py`) and by the H2-2 stream.  The docstring's "Every
other stage of the transport is trace-safe" over-claims.
**Requested edit**, before `carrier.py:2098`:
```python
    for _nm, _v in (('R_in', R_in), ('z', z), ('wavelength', wavelength),
                    ('dx', dx), ('dy', dy), ('R_ref', R_ref)):
        if _is_traced(_v):
            raise ValueError(
                f"{fn}: {_nm} is a JAX Tracer.  The leg's ABCD entries, its "
                f"output lattice and its chirp screens are built from these "
                f"as PYTHON floats, so only the ENVELOPE may be traced here.  "
                f"Differentiate with respect to the field, or rebuild the "
                f"call at each concrete {_nm}.")
```

### V-D14 -- two H2-2 threadings have no executing test on any available arm

`bld` threading in `_collins_axis_chirp` (m1) and `_as_c_order`'s contiguity
(m4) are both invisible to every value test: `_backend_of` returns `bld = np`
for NumPy AND for JAX, so only CuPy would differ; and `np.asarray` /
`np.ascontiguousarray` agree on VALUES for every input.  MEASURED: both
mutations leave all 21 ids and all 93 bit-identity keys unchanged.  The report's
"the CuPy arm ... is exercised structurally" overstates what is gated.
**Closed here** by `test_the_axis_chirp_builds_on_the_bld_it_is_handed`
(a recording namespace) and `test_as_c_order_makes_the_numpy_path_contiguous`
(a layout assertion) -- both backend-free, both sub-0.1 s.

### V-D15 -- `5.48.0` named in shipped docstrings while `__version__` is 5.47.0

Seven sites in shipped library source: `_bluestein.py:43`, `:486`;
`carrier.py:2068`; `mft.py:242`, `:910`, `:957`, `:1180`.  `lumenairy/__init__.py:1115`
reads `5.47.0`; the four CHANGELOG entries sit under `## [Unreleased]`.

Measured against the repo's own practice: for each of the last five releases,
mentions of the released version inside `lumenairy/**/*.py` at the release
commit's PARENT were **0, 0, 0, 1 (a false positive: a numeric table cell), 0**,
and the release commit (`4bf26c5e` for 5.47.0) touches exactly one library file
-- `lumenairy/__init__.py`.  The only forward-version references in the library
at base are deprecation HORIZONS (`_CARRIER_FIELD_FROZEN_IN = '5.48'`,
`version_removed='5.48'`), which are a scheduled future, not a claimed shipping
history.  No test enforces version labelling, so nothing catches it if the next
release is numbered differently -- and
`PLAN_WAVE5_LEFTOVERS_2026_09_14.md:38` already shows numbering in motion
(`NEXT_REMOVAL_VERSION` slipping 5.48 -> 5.50).
**Requested edit:** drop the bare version from the seven shipped docstrings and
let the CHANGELOG carry it (`"shipped since 5.48.0 as the opt-in"` -> `"shipped
as the opt-in"`; `"BACKENDS (5.48.0)."` -> `"BACKENDS."`; drop the trailing
`"(5.48.0)"` at the four `mft.py` sites), or stamp them at the release commit
the way `__init__.py` is stamped and re-record the three history fingerprints
then.

### V-D16 -- report prose that does not survive re-measurement

| # | site | says | measured |
|---|---|---|---|
| a | `WAVE5_HYGIENE2_REPORT.md:518` | "the two Collins spellings agree with each other exactly" | they differ by **1.334e-05 relative** (6.4e-18 absolute on a 4.8e-13 quantity). "Identical to all printed digits" (`:516`) is true; "exactly" is not. The test's own bar (`:238`, `<= 1e-3 * max`) is already the honest one. |
| b | `:541-547` | the Sziklas column's 7.3e-04 is the bridge | the column has TWO mechanisms and its WORST rung is the one the prose skips: **300 um reads 1.1565e-02 with the bridge OFF**, because the co-moving window there holds only 1.99 beam radii (edge truncation `exp(-3.97) = 1.879e-02`). At 1/3/5 mm the window opens to 4.1/5.0/5.1 radii and the reading tracks the truncation to within 3x. |
| c | `:660-663` | "A rule keyed on the BEAM's angle would fire five decades too eagerly ... VERIFY-B4 F3's own reading" | the 4.0e5 arithmetic is right for **WP-B11 sec. 2.20**, which did use the beam's 20 mrad. **VERIFY-B4 F3 used the envelope's MEASURED containment angle** (2.8574e-03 rad), which over-predicts by **33.5x** -- 1.5 decades, not five. |
| d | `:621-624`, `:672` | "the cost of not doing so is 4.7e-06 ... falling monotonically to 2.3e-08" / "Recommendation: no change" | that is a bound on the ladder's closest approach to `A = 0`, not on the fixture. The ladder measures `d` from the WAIST, and the carrier's geometric focus is 31.66 um further on, so `z_eff` caps at 12.27 m. Walked to its own carrier focus the SAME fixture pays **1.5431e-04 at 1 um and 1.5431e-03 at 0.1 um**, 327x the published worst case. See the maintainer answer. |
| e | H2-1 "3.79e-16" under CuPy | one number | fixture-dependent: **2.803e-16 .. 6.95e-16** over four fixtures. |
| f | H2-2 "bar (x6 chain depth) 1.6112e-15" | the bar | the test's `_fft_spread_bar` takes `max(6*spread, 32*eps)`, and on this fixture the FLOOR dominates by 4.4x: the bar actually asserted against is **7.105e-15**. The measured 8.2711e-16 is 0.94 decades inside the real bar, not 0.29 decades inside the tabulated one. (The floor is well reasoned; `32` has no stated origin either.) |

### V-D17 -- `_bluestein_centred_2d` validates `method` after reading `E.shape`

`_bluestein_centred_2d([[1+0j, 2+0j]], ..., method='bogus')` raises
`AttributeError: 'list' object has no attribute 'shape'` because `:743` runs
before `:744`; `_bluestein_2d` with the same arguments gives the correct
`ValueError`.  The two primitives also disagree on precedence with
`sign=0, method='bogus'` (`_bluestein_2d` reports `sign`,
`_bluestein_centred_2d` reports `method`).
**Requested edit:** move `Ny_in, Nx_in = E.shape` below the `method` check in
`_bluestein_centred_2d`, and validate `sign` there too so both primitives report
the same first error.

### V-D18 -- the public MFT docstrings advertise a narrower vocabulary than the code accepts

`mft.py:241`, `:909`, `:1179` all document `method : {'auto', 'direct'}`, but all
three entry points accept `'bluestein'` and `'separable'` by pass-through
(measured: all four return a field).  The author's "the vocabulary the error
names is the vocabulary that works" test covers only the primitives.
**Requested edit:** widen the three docstrings to the full `_SUM_METHODS`, or
narrow the public surface with its own check.  Gated here (subset direction, so
it stays true either way) by
`test_the_public_mft_method_vocabulary_is_closed_and_documented`.

Two further readings, recorded not as defects but as facts a caller should know:
the refusal arrives AFTER 0.29 % (fraunhofer), 16.6-19.7 % (fresnel) and 67.5 %
(ASM) of the peak work, and the ASM entry emits its replica `UserWarning` before
raising.  No FFT is ever executed.

### V-D19 -- a NON-REPRODUCING failure in the sweep, on an assertion whose shape invites one

`tests/unit/test_audit2609_b4_collins_transport.py::TestGateCTwoGroupChain::test_the_two_transports_agree_on_this_chain`
asserts `np.array_equal(sziklas_field, collins_field)` on a 2048x2048 chain.

* **In isolation on the branch: 1 passed** in 728.64 s; **in isolation on the
  base: 1 passed** in 696.41 s.  So the assertion is true on both trees when the
  test is the only thing in the session.
* **In the 48-file sweep on the branch: FAILED**, with the two arms differing in
  the sixth significant figure (5.01246221e-06 vs 5.01241961e-06, ~8.5e-06
  relative) -- far beyond round-off, so the two arms took DIFFERENT quadratures.
  The test's own docstring explains that it is bit-equal only while every leg
  stays in the transfer-function half of the split (`K1 = 1.92 > 1`), so an
  earlier test in the session moved something that moved that measurement.

**The contaminating prefix was hunted and NOT found.**  In the 48-file sweep the
b4 file runs at position 15.  Both prefixes ending at it pass:

| run | files | result |
|---|---|---|
| files 1-14 (INCLUDING all four new ones) + `b4::TestGateCTwoGroupChain` | 15 | **299 passed, 2 skipped** in 1283.15 s |
| files 5-14 (the four new ones REMOVED) + `b4::TestGateCTwoGroupChain` | 11 | **178 passed, 2 skipped** in 1264.95 s |

So nothing that RUNS before it explains the failure -- including the branch's own
new files, which therefore are not the mechanism.  Everything left runs AFTER
it, which leaves only collection-time import side effects (pytest imports every
selected module before running any) or the larger session's memory and cache
state.

**It did not reproduce.**  A second full 48-file branch sweep (with this
verification's own file appended, which runs after it and cannot reach it) reads
**1350 passed, 13 skipped in 1446.41 s** -- zero failures.  Four sweep-scale runs
and two isolated runs, and the assertion failed in exactly one: the FIRST, which
was also the most heavily loaded (a WSL sweep and four measurement streams were
resident).  That points at the session's memory state rather than at any single
earlier test, and `LUMENAIRY_MEM_BUDGET_MB` is the one budget-gated axis in this
set.

**So this is not attributable to the branch, and it is not a flake to rerun
away either.**  An 8.5e-06 difference is a QUADRATURE switch, not round-off, and
by the repository's own standard ("flaky = bad math") a green rerun is not the
answer.  An exact-equality assertion whose truth depends on a measured `K1`
straddling 1 is testing-standards shape **S1/S5**.
**Requested edit** to `tests/unit/test_audit2609_b4_collins_transport.py:685`:
split the claim in two -- assert that the leg is on the transfer-function side
(`K1 > 1`, read from `stats_out`, as its own premise) and that the two
transports then agree to a derived bar -- instead of one `np.array_equal` whose
truth is conditional on a measurement the assertion does not make.

### V-D20 -- the probe harness's tree guard raises the wrong exception across drives

`validation/probe_wave5_hyg2/hlib.py:49` (and this verification's copy) uses
`os.path.commonpath([got, want])`, which raises
`ValueError: Paths don't have the same drive` instead of the intended
`SystemExit("WRONG TREE: ...")` when the resolved install is on another drive --
which it is on this box (`D:\Metacept\...\Lumenairy`).  It still refuses, loudly,
but with a message that does not say what went wrong.
**Requested edit:**
```python
    try:
        same = os.path.commonpath([got, want]) == want
    except ValueError:                      # different drives on Windows
        same = False
    if not same:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
```

### V-D21 -- the refusal message's last sentence is misleading in its own context

`lumenairy/elements/lenses.py:160-162` ends "Reading
`lumenairy.elements.lenses.<name>` is unaffected and returns the same object."
Measured on both trees: while `_lens_kernels.X` holds a substitute, `lenses.X`
still returns the ORIGINAL (`facade_read_during_patch_is_real: true`).  That
aliasing is pre-existing and is the stated reason forwarding was rejected, but in
a message whose advice is "patch the leaf instead" the sentence reads as if the
facade read would follow the patch.
**Requested edit:** "...is unaffected by this refusal; note that it returns the
object bound at import time, so read it from `_lens_kernels` while a substitute
is in place."

### V-D22 -- THREE copies of the exact-dispersion phase live in `carrier.py`, and H2-2 parametrised one of them

The campaign's standing rule is ONE `xp`-parametrised implementation per
kernel.  An AST walk of `lumenairy/propagators/carrier.py` for functions that
build `root0` finds **three**:

| function | line | namespace | what it builds |
|---|---|---|---|
| `_exact_tf_2d_xp` | 600 (radical at 631-637) | `xp` / `bld` parametrised | `arg = k*z + z*(root - root0 + lin)` -- the exact TF, piston included |
| `_exact_envelope_tf_step` | 1357 (radical at 1433-1453) | **NumPy only**, and written twice inside itself (an in-place untilted arm and a tilted arm) | `phase = k*z_eff + z_eff*(root - root0 + lin)` -- the same TF |
| `_collins_exact_kernel_correction` | 1971 (radical at 2019-2029) | `(xp, is_jax, bld)` parametrised **by H2-2** | `phase = z_eff*(root - root0 + lin + q^2/(2k))` -- the exact/Fresnel RATIO |

All three compute `sqrt(k^2 - |k s + q|^2) - root0 (+ (s.q)/N_z)` from the same
tilt algebra, with the same `s2 < 1` guard written three times under three
different error prefixes.  They differ only in what is added afterwards: the
first two add the piston `k*z`, the third subtracts the paraxial `q^2/(2k)`
instead.  That is one kernel and two uses, not two kernels.

This is not a defect H2-2 introduced -- the three copies predate it -- but H2-2
made one of them `(xp, is_jax, bld)`-parametrised and left the other two as they
were, so the module now carries an `xp` copy, a `bld` copy and a NumPy-only
copy of one radical.  It is also load-bearing in an unexpected way: **the only
thing that caught the sign mutation in V-D4 was an agreement test between two of
these copies.**  Consolidating them would remove that guard at the same time as
it removes the defect the guard exists for, which is exactly why the
consolidation and the sign gate should land together.

**Requested edit:** factor the shared part into one helper beside
`_tf_phase_to_H`, e.g.
```python
def _exact_dispersion_phase(qx, qy, k, tilt, bld, fn):
    """``sqrt(k^2 - |k s + q|^2) - k N_z + (s.q)/N_z`` on ``bld``.

    The ONE place the non-paraxial dispersion is written.  Callers add their
    own piston (``k*z``) or paraxial subtraction (``q^2/(2k)``) and multiply
    by their own reduced distance.
    """
```
and have all three call it; keep the two in-place NumPy fast paths as an
`xp is np` arm INSIDE that helper rather than as separate transcriptions.  Pin
it with a census test in the shape the repo already uses for single-definition
claims:
```python
def test_the_exact_dispersion_is_written_once():
    """One kernel, one implementation.  MEASURED 2026-09-19: three copies of
    ``sqrt(k*k - (ax*ax + ay*ay))`` and of the ``s2 < 1`` tilt guard lived in
    carrier.py, at lines 631, 1449 and 2022, and a sign mutation of one of
    them was caught only because another copy disagreed with it."""
    import ast, inspect
    import lumenairy.propagators.carrier as CA
    src = inspect.getsource(CA)
    owners = [n.name for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and 'root0' in ast.get_source_segment(src, n)]
    assert owners == ['_exact_dispersion_phase'], owners
```
(This verification does not ship that test, because it would fail today; it is
the gate that should accompany the consolidation.)

---

## The time crossover, re-measured (H2-1 claim 5)

`validation/probe_verify_wave5_hyg2/v0_timing_crossover.py`, best-of-five per
route per shape, cold (every registered cache cleared and `gc.collect()` before
each repeat), timing in a pass with no `tracemalloc` in it.  **The load is
recorded in the JSON**: WIN `python_processes` 24 -> 20 of 478 total processes
with a single-thread reference loop at 3.01 / 3.16 ms; WSL `python_processes` 3
of 12, reference loop 4.01 / 6.28 ms.  The WSL ladder ran on a genuinely quiet
box; the Windows one ran with ~20 other interpreters resident but no other
Bash-launched job of this verification, and its reference loop moved by 5 %
between the two ends, so the readings are usable and the CROSSOVER (a ratio
taken under the same conditions for all three routes) is the robust part.

| N | crossover (WIN) | crossover (WSL) |
|---|---|---|
| 128 | between M = 32 and 64 | between M = 64 and 128 |
| 256 | dense wins to M = 128; separable at 256 | between M = 64 and 128 |
| 512 | **dense wins at every M up to 512** | between M = 64 and 128 |
| 1024 | between M = **128 and 256** (dense 0.0447 vs 0.0990 at 128; separable 0.0885 vs 0.1007 at 256) | between M = **64 and 128** (dense 0.0185 vs 0.0336 at 64; separable 0.0204 vs 0.0296 at 128) |
| 2048 | dense at M = 64; separable at M = 1024 | dense at M = 64; separable at M = 1024 |

**The crossover IS per-build, and the two builds differ by a factor of TWO, not
four.**  At N = 1024 the Windows crossover brackets `M/N` in (1/8, 1/4) and the
WSL one in (1/16, 1/8) -- adjacent octaves on the `M` ladder that share the
endpoint `M = N/8`.  The report's "Windows `M <~ N/4`, WSL `M ~ N/16`" quotes
the OUTER edge of each bracket, which is what makes it read as 4x.  The
qualitative claim -- per-build, therefore no threshold may ship -- is
CONFIRMED and is the one the decision rests on.

**One sentence does not survive re-measurement.**  The report says "The 2-D
chirp-Z route is the slowest of the three at every shape on both builds."  On
WSL that holds at 29/29.  On Windows it FAILS at three shapes -- (128, 512),
(2048, 1024) and (1448, 1448) -- where the DENSE route is the slowest
(1448x1448: dense 1.4120 s, chirp-Z 0.8540 s, separable 0.6822 s; 2048x1024:
dense 1.4411, chirp-Z 1.1533, separable 0.6331).  The author's own Windows table
reads the other way at 1448x1448 (chirp-Z 1.5587 vs dense 1.4957), which is
itself the point: that ordering is inside the run-to-run spread of a contended
box and is not a build-free statement the way the MEMORY ordering is.
**Requested edit**, `WAVE5_HYGIENE2_REPORT.md`: replace "The 2-D chirp-Z route
is the slowest of the three at every shape on both builds" with "The 2-D chirp-Z
route is the slowest of the three at every shape on WSL and at most shapes on
Windows; at `M >= N/2` with `N >= 1448` the Windows ordering between it and the
dense route is inside the run-to-run spread (measured both ways on two runs) and
is not build-free."

---

## Runs, tails and ruff

Every tail below was produced with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LUMENAIRY_MEM_BUDGET_MB=8192` on the command line and `pytest -q --capture=sys
-p no:randomly`.

| run | tree | tail |
|---|---|---|
| the 48-file sweep (the four new files + the mft / carrier / collins / b4 / b11 / a6 / a25 groups + the census, walker, dispatcher-pin, public-API and doc-consistency sweep + `test_audit_except_budget.py` + `test_ci_kernel_consistency.py`) | branch, WIN | **1 failed, 1337 passed, 13 skipped, 29 warnings in 1902.09 s** -- the one failure is V-D19 |
| the same 48 files **+ `test_verify_wave5_hyg2.py`**, re-run | branch, WIN | **1350 passed, 13 skipped, 29 warnings in 1446.41 s** -- V-D19 did NOT reproduce |
| the same set minus the b4 file, + `test_verify_wave5_hyg2.py` | branch, WSL | **2 failed, 1221 passed, 14 skipped, 29 warnings in 1304.91 s** -- both failures ENVIRONMENTAL, see below |
| prefix: files 1-14 + `b4::TestGateCTwoGroupChain` | branch, WIN | **299 passed, 2 skipped in 1283.15 s** |
| prefix: files 5-14 (new files removed) + `b4::TestGateCTwoGroupChain` | branch, WIN | **178 passed, 2 skipped in 1264.95 s** |
| `b4::TestGateCTwoGroupChain` alone | branch, WIN | **1 passed in 728.64 s** |
| `b4::TestGateCTwoGroupChain` alone | base, WIN | **1 passed in 696.41 s** |
| `test_audit2609_a15a_durations_staleness.py` | branch, WIN | **1 failed, 3 passed in 177.40 s** (V-D1) |
| `test_audit2609_a15a_durations_staleness.py` | base, WIN | **4 passed in 142.32 s** (V-D1) |
| `test_verify_wave5_hyg2.py` (this verification's own) | branch, WIN | **12 passed in 11.95 s** |
| `test_verify_wave5_hyg2.py` | branch, WSL | **12 passed in 16.87 s** |
| `test_verify_wave5_hyg2.py` | **base**, WIN | **6 failed, 1 passed in 12.24 s** -- the ids FAIL, they do not skip |
| `scripts/record_history_fingerprints.py --check` | branch, WIN | `OK: every history document matches its module` |
| `ruff check .` (ruff 0.15.16) | branch, WSL | **All checks passed!** |

**The two WSL failures are environmental and reproduce on the BASE tree or off
the branch entirely.**

* `test_public_api.py::test_installed_metadata_version_matches_source_version`
  -- MEASURED on the BASE archive in the same WSL venv: the same failure,
  `Installed distribution metadata says lumenairy==5.11.0 but the source says
  __version__==5.47.0`.  A stale editable install in `~/lumvenv`, not a branch
  defect.
* `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught`
  -- the walker needs `git diff v5.1.0..v5.1.1`, and from WSL this WINDOWS
  worktree's gitdir pointer does not resolve: `fatal: not a git repository:
  /mnt/c/tmp/lum_vhyg2/D:/Metacept/.../Lumenairy/.git/worktrees/lum_vhyg2`.  It
  gets rc=2 where it wants rc=1.  Green on Windows in both full sweeps.

`.test_durations`: this verification's 12 ids were measured serially with BLAS
pinned to one thread and spliced in (16253 -> 16265 entries, JSON re-parsed,
diff = 13 insertions / 1 deletion).  The branch's own 121 ids are still missing
-- that is V-D1.

---

## What could not be measured

* **Whether V-D19 is a branch regression.**  The failure appeared once in four
  sweep-scale runs and did not reproduce.  Both prefixes ending at the failing
  test pass, in isolation both trees pass, and the full branch sweep passes on a
  second run -- so the contaminator, if there is one, acts through
  collection-time imports of files that RUN after it, or through the session's
  memory state (this is the one budget-gated code path in the set, and the
  failing run was the more heavily loaded of the two).  A base-vs-branch
  comparison of the full sweep is not even well posed, because the branch adds
  four files to the session.  The experiment that would settle it is a bisect
  over the 33 files that run after the b4 file, with each candidate added to the
  passing 15-file prefix; at ~21 minutes a run that is ~12 hours of machine time
  and was not attempted.
* **CuPy on the device for the Collins chain.**  Confirmed independently:
  `cupy.fft.fft2` raises `ImportError: DLL load failed while importing cufft`
  on this box and WSL has no CuPy, so the Collins CuPy arm is structural only.
  The dense MFT kernel and `_as_c_order`'s CuPy half DO run on the device,
  because neither needs a transform.
* **An uncontended WINDOWS timing ladder.**  The WSL ladder ran at 3 Python
  processes; the Windows one at ~20 (none of them this verification's).  The
  crossover, being a ratio taken under the same conditions for all three routes,
  is reported; the absolute times are not claimed to be a quiet-box reading.
* **A non-paraxial oracle for H2-3.**  Unchanged from the author's statement,
  and re-confirmed: the analytic Gaussian is paraxial, so it can measure how far
  the exact kernel departs from the paraxial truth and cannot say which kernel
  is more physical.
* **The `_H_FFT_CACHE` behaviour of the dense route under CuPy at scale.**  The
  dense kernels are built on the HOST (`np.arange` / `np.exp`, then
  `xp.asarray`), so the measured memory table is a NumPy-path table and does not
  describe the CuPy path's total footprint.  Not measured on a device.

---

## The maintainer question: should `gap_kernel='auto'` fall back near a focus?

**Yes -- but keyed on the distance to `A = 0`, not to the waist, and with the
threshold written in the envelope's ANALYTIC angle.**  On the hygiene-2 fixture
(`f` = 20 mm, `w0` = 15.915 um, `lambda` = 1.0 um, N = 512 at 8 um) `'auto'`
resolves to `'exact'` at every rung and costs 4.7171e-06 relative one micron
short of the waist against `'fresnel'` at 9.7868e-13 -- small enough to ratify,
which is what the report concludes.  But that ladder's carrier focus sits 31.66
um BEYOND its waist, so `z_eff = B/A` never exceeds 12.27 m; walked to its own
carrier focus the same fixture pays **1.5431e-04 at 1 um and 1.5431e-03 at 0.1
um**, 327x the published worst case, and VERIFY-B4 F3's fixture (`w` = 0.3 mm,
N = 1024, `dx` = 4 um, `lambda` = 1.064 um, `R` = -40 mm), whose ladder does
approach `A = 0`, pays **2.3497e-03 at `z_eff` = 1600 m** against `'fresnel'` at
**1.7042e-14** -- eleven decades.  That reading is not an artefact: it is
identical at N = 512, 1024 and 2048, moves only in the sixth digit when the
output window is quadrupled or the pitch coarsened 4x, and sits at `K3` = 0.068,
so it is the refinement and not the grid.  All four readings, on three fixtures
and two builds, are ONE law -- `departure = sqrt(3/2) * k |z_eff| theta_env^4 /
8` with `theta_env = lambda/(pi w_env)` the envelope's analytic `1/e^2`
half-angle, the constant being the RMS-vs-peak moment of a quartic phase over a
2-D circular Gaussian (`sqrt(<u^8>) = sqrt(3/2) = 1.22474`; it predicts F3's
2.3497e-03 to a ratio of 1.00002 and the 498x between the two fixtures to four
digits).  The existing `k4` wrap gate cannot serve: it is 4.65 decades below its
bar where the departure is 4.7e-06 and still 2 decades below it where the
departure is 2.3e-03, and it fires only at `A = 0` exactly (where `'auto'` does
correctly fall back and `'exact'` is refused).  So the report's "no change" is a
defensible reading of ITS ladder and is not defensible in general; add the
second, accuracy-keyed condition VERIFY-B4 F3 sketched -- fall back to
`'fresnel'` when `sqrt(3/2) k |z_eff| theta_env^4 / 8 > tau` -- noting that the
chain's own `_GAP_ENV_PHI_TOL_DEFAULT = 0.3` as `tau` would never fire on either
fixture, so `tau` has to be set from the accuracy actually wanted (`tau = 1e-4`
leaves the hygiene-2 ladder inert and makes F3's 1 um and 10 um rungs fall
back).  Two honest caveats for the other side: the oracle is PARAXIAL, so it can
only say how far the exact kernel departs from the paraxial truth and not which
kernel is more physical; and the departure is a phase refinement, so on a leg
where the exact kernel is the better physics the fallback would be trading
accuracy for agreement with an oracle.

---

## Ship recommendation

**Do not merge as-is.  Two blockers, both mechanical; the physics and the code
are sound.**

* **V-D1** (`.test_durations` regression) and **V-D2** (15 stale citations, one
  re-anchored wrong) are release-gate failures by the repository's own rules and
  both are fixable without touching a line of library code.
* **V-D3** (the port does not reach the public `transport='collins'` leg, and
  the CHANGELOG says it does) should be resolved before the entry ships -- by
  threading the leg or by narrowing the sentence and fixing the now-vacuous
  cross-backend test.
* **V-D4** (the file that publishes the near-focus law cannot see the sign of
  the kernel it publishes) is a six-line test edit with a verified before/after
  and should land with the package.  The library itself is guarded --
  `test_audit2609_b4_collins_transport.py::TestSameTheorem` catches a conjugated
  kernel at 2 failed / 124 passed -- but only because two independent copies of
  that kernel exist to disagree (**V-D22**).
* Everything else -- **V-D5** through **V-D22** -- is a docstring, a report
  sentence, a bar derivation, a consolidation or a maintainer decision, and can
  follow.

**V-D19 is explicitly NOT counted as a blocker.**  It failed once in four
sweep-scale runs, did not reproduce on a second full sweep, passes in both
prefixes that precede it and passes in isolation on both trees.  It is not
attributable to this branch.  It is still a real fragility -- an
`np.array_equal` conditioned on a measured `K1` straddling 1 -- and the
restatement is requested there.

What the branch gets RIGHT, re-measured and confirmed: the default is
byte-identical on independent fixtures on both builds for all three packages
(70/70, 93/93, 83+32/115); `_direct_matrix_2d` really is one implementation
serving both index conventions, bit-identical at zero centres; the memory
ordering holds at 29/29 shapes on both builds; the JAX port of the private
transport is real (it FAILS on base) and its traced refusal is clean, complete
over a 36-cell matrix, and leaves no partial state; the near-focus law
reproduces on three independent fixtures with a constant that turns out to be
`sqrt(3/2)` exactly; and the facade refusal breaks nothing (0 write sites found
by an exhaustive AST walk of every `.py` in the tree), survives `importlib.reload`
for a mechanically proved reason, and leaves `dir()`, `vars()` and `import *`
byte-identical.

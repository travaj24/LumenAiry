# WP-A21 — final hygiene pass (audit 2026-09-11)

Branch `audit-fixes-2026-09`.  No git write command was issued.  Inputs: the
"Requested changes outside my ownership" sections of `WP-A15a_REPORT.md` (§5),
`WP-A15b_REPORT.md` (§5.6 / §6.4), `WP-A18_REPORT.md` (§8, §9) and the CI
warning-filter note in `VERIFY_WP-A13.md`, plus one mid-pass addition from the
orchestrator (`propagation._LIVE_FORWARD_NAMES`).

Everything below was measured on this box: Windows 11 / CPython 3.14.6 /
numpy 2.4.4 / scipy 1.17.1 (scipy-openblas), `OMP/OPENBLAS/MKL_NUM_THREADS=1`.

---

## 1. Summary

| # | item (source) | status | files:lines | tests | oracle | before → after |
|---|---|---|---|---|---|---|
| 1 | RCWA lint, 2 F401 + 3 I001, delete the TODO ignores (A15a §5.4) | **fixed** | `rcwa/oned.py:26`, `rcwa/twod.py:24`, `oned.py:53`, `twod.py:50`, `stack.py:58`; `pyproject.toml:424-433` deleted | `test_rcwa.py`, `test_audit2609_a14_*.py` unchanged | `ruff check` with the ignores cleared | 5 findings, 3 TODO ignores → **0 findings, 0 ignores**; `ruff check lumenairy tests scripts` clean |
| 2 | Wood-anomaly diagnostic names a non-existent class (A18 §8.2) | **fixed** | `rcwa/twod.py:1300`, `:1302` | `test_audit2609_a21_rcwa_and_ui.py::test_the_wood_anomaly_diagnostic_names_a_class_that_exists`, `::test_no_dead_class_name_survives_in_the_rcwa_2d_diagnostics` | `getattr(rcwa.twod, <name in the warning>)` | `RCWA2DPrepared.solve` (`AttributeError`) → **`PreparedRCWA2D.solve`** (resolves) |
| 3 | `memory.py` broad excepts (A15a §5.6, §2.9) | **fixed** | `memory.py:1242-1283`, `:1306-1316`; `test_audit_except_budget.py:114-126,134-139,229` | `test_audit_except_budget.py` (4 ids) | AST census of `lumenairy/**` | `memory.py` 2 broad → **0**; census **53 → 51**, tree 51, slack 0 |
| 4 | `fast_analytic_phase` tooltip (A3 §5.8 via A18 §8.3) | **fixed** | `ui/lens_options_dialog.py:135-148` | `test_audit2609_a21_rcwa_and_ui.py::test_the_fast_analytic_phase_tooltip_states_the_per_mm_error` (+ registry guard) | `_lens_traced._geometric_lens_phase.__doc__` + `docs/subsystems/real_lens.md` §4; a fresh speed measurement | "~25 % speedup with <10 nm OPL error" → **~7 nm rms per mm of glass**, speed 0.87-1.00x (amp ON) / 1.06-1.17x (OFF) |
| 5a | 3 wall-clock/speedup assertions (A15a §5.8, §2.7) | **fixed** | `test_pmm2d_staggered_slant.py:1308-1397`, `test_v4_15_agent_a.py:289-393` | the two converted tests | eigendecomposition census; batched-vs-scalar kernel counts | `min(ratio) < 2.0` + `ti > 0.0` + `speedup >= 5.0` → **exact integer counts**; A15a's other 3 sites already landed by VERIFY-A4 (re-measured) |
| 5a | 5 missing `slow` markers (A15a §2.6) | **fixed** (4 of 5; 1 already landed) | `test_niche_d2_chain_multi.py:99`, `test_pmm2d_staggered_mortar.py:75`, `test_niche_d6_exact_tilted_leg.py:117`, `test_audit_misc.py:1386` | collection under `-m slow` / `-m "not slow"` | committed `.test_durations` | **1 001.4 s** moved out of the fast lane |
| 5b | 3 subprocess spawns lacking `stdin=` (A15a §5.9) | **fixed** | `test_niche_audit_w3_infra.py:760`, `test_niche_d14_deterministic_carrier_fit.py:223,:422`; `test_audit2609_a15a_packaging.py:262-271` | `test_audit2609_a15a_packaging.py` (11 ids) | AST gate over `tests/` | 3 exempt sites → **`_STDIO_EXEMPT` empty** |
| 5c | 1-ULP bit-equality drift (A15a) | **fixed** | `test_audit2609_a7_misc.py:163-256` | `::test_transfer_function_recurrence_engages_only_on_a_uniform_scan` | FFT rounding model + an in-process wrong-grid counter-arm | RED at 1 ULP → **derived two-sided 52.7-ULP band**, measured 2.0 |
| 6 | mypy whitelist +9 (A15a §5.10 / A15b §5.6) | **partially fixed** (8 of 9) | `pyproject.toml:527-534`; `test_audit2609_a15a_packaging.py:115-136` | `test_audit2609_a15a_packaging.py::test_the_mypy_whitelist_only_grows` | `mypy --strict` | 17 paths / 22 files → **25 paths / 30 files**, 0 new errors.  Root `lumenairy/__init__.py` held back: 33 errors, all WP-A16's in-flight (§5.1) |
| 7 | PMM 2-D warning attribution (VERIFY-A13) | **fixed (decided: message filter)** | `pmm/twod_staggered.py:493, :598-624, :637` | `test_audit2609_a21_pmm_warning_filter.py` (6 ids) | live `-W` filter behaviour on both entry points | module filter caught nothing on either path → **documented message filter, measured to catch both** |
| 8 | committed identifier resolver (A18 §9) | **fixed** | new `scripts/check_doc_identifiers.py` (585 lines) | new `test_audit2609_a21_doc_identifiers.py` (5 ids) | the importable package + a static AST index | A18's scratch script → **committed, 593/593 resolve, 0 unresolved, 4.5 s, no network, exit 0/1** |
| + | `propagation._LIVE_FORWARD_NAMES` (orchestrator, mid-pass) | **fixed** | `propagators/propagation.py:280-288` | `test_v5_2_walker_pep562_forwarding.py` (4 ids) | the walker's own AST census of `fft_infra` globals | 1 failed → **4 passed**; forwarding demonstrated live |

**Not mine, found on the way** (§5): two mypy errors and 33 more, three broad
excepts in a brand-new file, and one call-site kwarg.  None was introduced by
this work package; each is measured and attributed.

---

## 2. Per item

### 2.1 RCWA lint (deliverable 1)

**What was wrong.**  `ruff check lumenairy/elements/rcwa --config 'lint.per-file-ignores={}'`
measured exactly the five findings A15a recorded:

```
oned.py:3:1  I001   oned.py:26:5  F401 `._core._grazing_safe_wavelength` imported but unused
twod.py:3:1  I001   twod.py:24:5  F401 (same)
stack.py:3:1 I001
```

The three I001s are ONE two-line swap each — ruff's isort wants `_wood_symmetric`
before `_WoodAnomaly` (case-insensitive ordering) — not the
import-with-commentary shape the `(a)` block of the ignore list exists for, so
nothing is lost by sorting them.

**What I changed.**  Dropped the two dead imports and re-sorted the three
`from ._core import (...)` blocks; deleted the three `# TODO(audit-2609)`
entries from `[tool.ruff.lint.per-file-ignores]`, leaving a note in their place
saying the files are gated again.

**Why it is safe.**  `_grazing_safe_wavelength` is still exported from
`rcwa/_core.__all__:4927`, and every live caller (13 sites across
`elements/pmm/*`, plus `rcwa/stack.py:2995`, which still uses it and is
therefore not flagged) imports it from `_core`.  It is in neither
`oned.__all__` nor `twod.__all__`, so no re-export is dropped; grep confirms no
test or probe reads `oned._grazing_safe_wavelength` / `twod._grazing_safe_wavelength`
(the monkeypatching probes in `validation/` all patch `_core`).

**Ownership note.**  My brief's file list names `rcwa/_core.py`, but `_core.py`
is clean — the third I001 is in **`rcwa/stack.py`**, which is what A15a §5.4
actually asks the RCWA work package to re-sort.  I made that edit (a two-line
import swap, no semantic change, `git status` showed the file untouched by
anyone) because deliverable 1 requires all three I001s fixed and all three
ignores deleted; flagged here so it is visible in review.

**Verified.**  `ruff check lumenairy tests scripts --no-cache` → *All checks
passed!*  `python -c "import lumenairy"` → 5.45.1.  RCWA bit-identity: see §4.

### 2.2 The Wood-anomaly diagnostic (deliverable 2)

**What was wrong.**  `rcwa/twod.py:1300` and `:1302` passed
`fn_name="RCWA2DPrepared.solve"` to `_grazing_safe_wavelength_pair` and to the
`_WoodAnomaly` control-flow signal.  The class is `PreparedRCWA2D`
(`twod.py:1258`, in `twod.__all__`, built by `prepare_rcwa_2d`); the
`fn_name` also reaches `_wood_mean`'s error text through
`_wood_symmetric`, so both literals are user-visible.

**Verified by measurement.**  A 2-D Moharam mount (`period_x = period_y =
lambda = 1 um`, air/air, 16x16 cell, `n_orders 3x3`) puts the `(±1,0)` and
`(0,±1)` orders exactly at cut-off:

```
n wood = 1   CATEGORY WoodNudgeWarning
HEAD: PreparedRCWA2D.solve: a diffracted order sits EXACTLY at cut-off ...
wl_eff: (9.99999999e-07, 1.000000001e-06)
```

**The pin is not a string compare.**  `test_the_wood_anomaly_diagnostic_names_a_class_that_exists`
splits the emitted message at the first `:`, then asserts
`getattr(rcwa.twod, <class>)` exists, that `<method>` on it is callable, and
that the class is in `twod.__all__`.  A rename that updated the literal and
forgot the class — the defect being fixed — still fails it.  A second test
statically walks every `fn_name=` literal handed to
`_grazing_safe_wavelength_pair` / `_WoodAnomaly` in the module (4 of them) and
resolves each.

**Fail-before, in process** (literal reverted, test run, restored):

```
1) Wood fn_name reverted to RCWA2DPrepared: rc=1  2 failed, 2 passed
   message present ("does not exist on lumenairy.elements.rcwa.twod"): True
   after restore: rc=0  4 passed
```

**Residual.**  `Migration-Guide.md:1289` still spells `RCWA2DPrepared.solve`
(deliberately — A18 documented both spellings so the guide was right either
way).  Now that only one spelling ships, that line could be simplified; it is
WP-A18's file, so it is a request (§5.4) and the resolver's exclusion list
carries the reason.

### 2.3 `memory.py` (deliverable 3)

**The two sites and what they can actually raise.**  Both wrapped a
first-party import plus other work in `except Exception: pass`.

| site | statements inside the old `try` | what each can raise |
|---|---|---|
| `set_low_memory` enable arm (`:1242`) | `from .propagators.fft_infra import get_default_complex_dtype, set_default_complex_dtype` | **`ImportError`** |
| | `prior['complex_dtype'] = get_default_complex_dtype()` | nothing — a bare `return DEFAULT_COMPLEX_DTYPE` |
| | `set_default_complex_dtype(np.complex64)` | nothing reachable — it raises `ValueError` only outside `{complex64, complex128}`, and `np.complex64` is a member |
| | `warnings.warn(..., RuntimeWarning)` | **`RuntimeWarning`**, but only when a filter promotes it |
| `set_low_memory` disable arm (`:1306`) | `from .propagators.fft_infra import set_default_complex_dtype` | **`ImportError`** |
| | `set_default_complex_dtype(stash['complex_dtype'])` | nothing — the value came out of the getter at capture time, so it is already one of the two accepted dtypes |

So `ImportError` is the only thing the guard exists for, and it is now the only
thing it catches: the `try` wraps the import alone, the rest moves to `else`.
The reachable failure is narrower still — the module is imported *unguarded* at
`memory.py:1210` in the same function, so the case left is a vendored
`fft_infra` missing the two dtype names.

**One behaviour improves, and it is stated as a change.**  The aggressive-mode
`RuntimeWarning` used to sit inside the `except Exception`, so under
`-W error::RuntimeWarning` the promoted warning was *swallowed* — no error, no
warning, and a flipped dtype.  Narrowing to `ImportError` would have let it
propagate **before** `_LOW_MEMORY_PRIOR` was written, leaving the other four
knobs flipped with no restore record.  The warning is therefore emitted after
the snapshot is stashed, guarded by a local `flipped_dtype` flag.  Net: `-W
error` now sees the warning and `set_low_memory(False)` still restores.

**Census.**  `test_audit_except_budget.py`: `memory.py: 2` deleted from
`_NARROWING_REQUESTS`, group-(1) header 34-of-53 → 34-of-51, scalar comment and
the counter-pin's measured line updated.  Measured after: tree 51, census 51,
slack 0.

**Fail-before** (one `except ImportError` re-broadened, test run, restored):

```
3) memory.py except re-broadened: rc=1  3 failed, 1 passed
   message present ("more broad ``except Exception:`` clauses than the justified census"): True
```

**This file's gate is RED for a reason that is not mine** — see §5.2.

### 2.4 The `fast_analytic_phase` tooltip (deliverable 4)

**What was wrong.**  Both halves of "~25 % speedup with <10 nm OPL error on
typical refractive prescriptions".

*The error.*  WP-A3's T4 measurement (carried in
`_lens_traced._geometric_lens_phase.__doc__` and in
`docs/subsystems/real_lens.md` §4): the omitted term is the in-glass ASM leg,
so it is **linear in centre thickness, not in f-number** — 0.007 nm rms at
1 um, 0.7 nm at 100 um, **14.1 nm rms / 41.6 nm PV at 2 mm** on an N-BK7
100/-100 biconvex at f/12.1, i.e. **~7 nm rms per mm of glass**.  The flat
10 nm holds only below ~1.5 mm of glass.

*The speedup, re-measured here* (the brief did not ask, but shipping one
corrected claim beside an uncorrected one is not a fix).  `apply_real_lens_traced`,
`preserve_input_phase=True`, medians of 9 interleaved pairs:

| prescription | N=256 | N=512 | N=1024 |
|---|---|---|---|
| 2 mm N-BK7 singlet | 1.00x | 0.87x | 0.91x |
| 8 mm N-BK7 singlet | 1.07x | 0.98x | 0.93x |

and with the mechanism isolated (2 mm singlet):

| | N=512 | N=1024 |
|---|---|---|
| `parallel_amp=True` (the default) | 0.90x | 0.88x |
| `parallel_amp=False` | 1.06x | **1.17x** |

The default already runs `apply_real_lens(input)` and `apply_real_lens(plane
wave)` concurrently, so skipping the reference leg saves no wall clock — and
the analytic sag evaluation adds serial work in the main thread, which is why
the "fast" path is *slower* at the default.  There is no ~25 %.

**What I changed.**  The tooltip now leads with the per-mm error and its
measurement, states the thickness scaling, gives both speed regimes, and ends
"Take it for thin-element accuracy control, not for speed."

**Verified under the auditor's Qt stub.**  PySide6 is absent; the new test
installs the stub from
`docs/audits/.../repro/UI/stub` with permissive `QtWidgets`/`QtGui` shims and
**parks** it after importing `lumenairy.ui.lens_options_dialog`, exactly as
`test_audit2609_a9_ui.py` does, so sibling UI files that guard on
`import PySide6` still see the interpreter they had.  The pin reads the tooltip
out of `LENS_KWARG_REGISTRY` (the module docstring's stated single source of
truth), asserts the per-mm phrase and the 7 nm figure are present, asserts the
two retired claims are **absent**, and cross-checks that
`_geometric_lens_phase.__doc__` still says the same thing so the two cannot
drift apart again.  `test_audit2609_a9_ui.py` re-run: 33 passed.

**Fail-before**: re-injecting the old sentence makes the test fail on the
retired `25%` claim (`1 failed, 3 passed`); restoring returns `4 passed`.

### 2.5 Test hygiene (deliverable 5)

#### (a) wall-clock / speedup assertions

I re-ran A15a's AST sweep (a function that reads a clock, and an `assert` whose
test mentions a name derived from that read) over `tests/`.  **The three sites
A15a listed are already converted** — `test_audit_propagation.py:2678` and
`:3787` and `test_audit_optimize.py:905` all now carry
"v5.46 (VERIFY-A4, request from WP-A15a section 5 item 8)" blocks with
operation-count pins.  The sweep finds three DIFFERENT live sites, and two of
them are exactly the VERIFY-A6 / VERIFY-A13 files my brief names:

| site | assertion | disposition |
|---|---|---|
| `test_pmm2d_staggered_slant.py:1361, :1372` | `min(ratios) < 2.0` on `perf_counter` medians, then `assert ti > 0.0` | **converted** |
| `test_v4_15_agent_a.py:351` | `speedup >= 5.0` | **converted** |
| `test_ci_kernel_consistency.py:492` | `elapsed < 20.0` | **left, deliberately** — A15a §6 item 3 defers it to the census's owner, and my brief scopes me to the A6/A13 files |

**Site 1 — `test_cost_slant_is_free_against_the_out_of_plane_solve_it_must_use`
→ `test_cost_slant_adds_no_eigenwork_to_the_solve_it_must_use`.**

The old test timed three region solves and asserted `min(slant/vertical) < 2.0`,
then closed with `ti = perf_counter()-t0; assert ti > 0.0` — an assertion that
cannot fail.  Its own docstring recorded the problem: the ratio measured
0.869..1.167 across WIN/WSL, idle and loaded, i.e. the asserted quantity is a
property of the runner.

The claim underneath is structural and exact: a sheared cell has out-of-plane
tensor entries, so it must run the `4 q^2` first-order generator anyway, and
the congruence plus the six extra Kronecker blocks must add no eigenwork.  The
new test censuses every dense eigendecomposition by **kind and dimension** —
the only two quantities that set the `O(n^3)` cost — by counting `np.linalg.eig`
and `scipy.linalg.eig`:

| arm | M = 5 (q = 8) | M = 6 (q = 10) |
|---|---|---|
| vertical out-of-plane | 1 standard, n = 256 | 1 standard, n = 400 |
| **slanted** | 1 standard, n = 256 | 1 standard, n = 400 |
| vertical in-plane (the cheaper path it does not take) | 1 generalized, n = 128 | 1 generalized, n = 200 |

`4 q^2` and `2 q^2` exactly; slanted census **equal** to vertical.  No
tolerance exists to derive and none is needed.  Runs in 2.1 s.

**Site 2 — `test_modal_asymptotic_perf_win` →
`test_modal_asymptotic_is_batched_not_per_pixel`.**

v4.15's win is a vectorisation, so it is pinned as one.  Counting calls on the
four kernels (patched on `propagators.asymptotic` *and* in the test module's
globals, because the preserved pre-v4.15 reference resolves the scalar names
there):

| N | pixels | public path | pre-v4.15 reference | measured speedup |
|---|---|---|---|---|
| 16 | 256 | 1 batch + 1 batch, 0 scalar | 256 scalar solves | — |
| 32 | 1 024 | 1 batch + 1 batch, 0 scalar | 1 024 scalar solves | 14.9x |
| 64 | 4 096 | 1 batch + 1 batch, 0 scalar | — | 16.1x |
| 128 | 16 384 | 1 batch + 1 batch, 0 scalar | — | 14.9x |

The public count is **flat across a 16x change in pixel count**; the
reference's is exactly `N*N`.  The ratio is reported in the docstring, not
asserted.  The test also drops from ~35 s (2.25 s batched + 33.45 s reference at
N = 128) to **4.3 s**.

#### (a) slow markers

A15a §2.6 listed five files it could not mark.  One (`test_audit_lens_models_2026_07.py`,
641.0 s) has since been marked by VERIFY-A4.  The remaining four:

| file | `.test_durations` | marker | why this shape |
|---|---|---|---|
| `test_niche_d2_chain_multi.py` | 338.8 s / 38 ids (heaviest 59.3 s) | module-level | not jax-guarded; weight is spread, no single test to mark |
| `test_pmm2d_staggered_mortar.py` | 229.3 s / 31 ids (heaviest 32.9 s) | module-level | same |
| `test_niche_d6_exact_tilted_leg.py` | 124.7 s / 38 ids (heaviest 30.1 s) | module-level | same |
| `test_audit_misc.py` | 320.7 s / 228 ids | **one test** | see below |

**`test_audit_misc.py` must NOT be marked module-level**, and A15a's list did
not catch this.  The file carries `pytest.importorskip('jax')` guards, so the
`jax-unit` CI job selects it (its selection grep is
`importorskip.{0,4}jax|skipif\(.{0,40}jax`) and runs it with
`-m "not integration and not slow"` — a module marker would have deleted the
file's jax ids from the only leg in CI that installs jax, which is precisely
the coverage trap A15a designed around for the other four files.  It is also
unnecessary: **308.4 s of the 320.7 s is one id**,
`ReadmeCookbookExamples::test_real_lens_from_zemax_cookbook`; every other id in
the file is ≤ 2.6 s.  The marker is on that test, which reaches no jax path.

Measured selection over the four files: 336 ids collected, **108 selected by
`-m slow`**, 228 by `-m "not slow"`.  Total moved out of the fast lane:
338.8 + 229.3 + 124.7 + 308.4 = **1 001.4 s**.

#### (b) subprocess stdin

`stdin=subprocess.DEVNULL` added beside `capture_output=True` at
`test_niche_audit_w3_infra.py:760` and `test_niche_d14_deterministic_carrier_fit.py:223`
and `:422`, each with a one-line why-comment pointing at A15a §2.10's
measurement (CPython's `Popen._get_handles` resolves an unnamed stream through
a `GetStdHandle()` that pytest's fd-capture has left stale).  None of the three
children reads stdin.  `_STDIO_EXEMPT` in
`test_audit2609_a15a_packaging.py` is now **empty**, so the AST gate covers
`tests/` with no hole, and its counter-pin
(`test_the_stdio_exemption_list_is_still_accurate`) is vacuously true rather
than pointing at live flakes.

**Fail-before**: reverting one spawn → `1 failed, 10 passed` with
"leave some -- but not all -- stdio streams inherited"; restored → `11 passed`.

#### (c) the 1-ULP bit-equality

`test_transfer_function_recurrence_engages_only_on_a_uniform_scan` asserted
`scan.peak_I[i] == float((np.abs(Ez)**2).max())` and was RED on this build:
`0.2570220974913172` vs `...71` at plane 1.

**Diagnosed, not papered over.**  The brief's hypothesis (recurrence rounding)
is not the cause: for this `z = [15, 17, 20, 24, 25] mm` the uniformity gate
computes `dz = 2.5 mm`, `max|z - z_model| = 1.5 mm`, and `k * 1.5e-3 >> 1e-12`,
so `use_recurrence` is **False** — the direct `exp` path is taken, as the test
demands.  Measured instead:

* the two transfer functions are **bit-identical**.  Building `H` the
  `through_focus_scan` way (centred layout, `k*k - kx2 - ky2`,
  `sqrt(where(prop, ., 0))`) and the `angular_spectrum_propagate` way (natural
  layout, `k**2 - ...`, `sqrt(maximum(., 0))`) gives
  `max|H_scan - fftshift(H_asm)| = 0.0` at N = 96, z = 17 mm;
* what differs is where the `fftshift` pair sits.  `through_focus_scan`
  transforms the SHIFTED field and un-shifts the product;
  `angular_spectrum_propagate` uses the identity documented at
  `propagators/asm.py:105-117` — for even N the two shifts are the same
  `(-1)^(kx+ky)` pattern applied twice, so they cancel — and runs
  `ifft2(fft2(E) * H_natural)` with no shifts at all.  Exact in exact
  arithmetic; two different floating-point computations of it.

Measured drift of the scan against the per-plane propagator, worst |ULP| over
the five planes:

| N | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|
| max \|ULP\| | **0** | **2** | **0** | **4** | **0** |

Zero at every power-of-two N — there the sign pattern commutes through every
radix-2 butterfly exactly — and a few ULP at the mixed-radix 96 = 2^5·3 and
192 = 2^6·3.  The bit-equality was pinning the FFT plan, not the recurrence
gate.

**The band, derived.**  A 2-D FFT of an N×N grid runs `2 log2(N)` butterfly
stages, each contributing at most `eps` of relative rounding, so the two routes'
fields differ by at most `2 log2(N) eps`; `peak_I = |E|^2` doubles that to
`4 log2(N) eps`; 1 ULP of a float64 is at least `eps/2` relative, so the band
in ULPs is `8 log2(N)` — **52.7 at N = 96**, against a measured 2.0 (1.4
decades inside).

**The other side, computed in-process** so the band cannot pass vacuously: if
the recurrence *had* engaged it would have evaluated the uniform MODEL grid
15 / 17.5 / 20 / 22.5 / 25 mm.  Measured, planes 1 and 3 move by **5.6e-02 and
1.0e-01 relative — 2.6e14 and 5.0e14 ULP**, thirteen decades outside the band.
The test asserts that separation (`min(moved) > 1e4 * band`), so a fixture that
stopped discriminating fails rather than passing quietly.

`test_audit2609_a7_misc.py`: 24 passed (was 23 passed, 1 failed).

### 2.6 mypy whitelist (deliverable 6)

Measured `mypy --strict --python-version 3.12 --follow-imports silent
--ignore-missing-imports` on each of the nine candidates:

| module | errors |
|---|---|
| `_math/__init__.py`, `sources/__init__.py`, `io/__init__.py`, `analysis/__init__.py`, `elements/{pmm,rcwa,eme,bor}/__init__.py` | **0 each** |
| `lumenairy/__init__.py` | **33** |

The eight clean ones are in.  `mypy` (whole whitelist) now checks **30 source
files, up from 22**, and reports the same 2 errors it reported before my change
— both in `lumenairy/backend/__init__.py`, which was already whitelisted and
whose uncommitted PEP 562 `__getattr__`/`__dir__` pair is unannotated (§5.1).

The root `lumenairy/__init__.py` is deliberately held back, with the reason
written into the `[tool.mypy]` comment: its 33 errors are 31
`Module "lumenairy.elements.lenses" does not explicitly export attribute ...`
rows plus an unannotated `__getattr__`/`__dir__` at `:2192`/`:2220` — all from
the lens-family work in flight in the same tree.  Adding it now would have made
a second file red for someone else's reason.

`test_the_mypy_whitelist_only_grows`'s floor raised 17 → 25 in the same change,
as A15a asked.

### 2.7 PMM 2-D warning attribution (deliverable 7)

**The decision: document the MESSAGE filter.**  It is not a stacklevel that can
be tuned — measured, a module-scoped filter on `twod_staggered` cannot work at
any *correct* stacklevel.  Python matches a filter's `module` against the
`__name__` of the frame `stacklevel` selects, and the purpose of a non-1
stacklevel is to select the CALLER; a warning that correctly points at the
user's code is by construction not attributed to the library module that
raises it.

Measured on a 4x4 SEGMENT grid whose walls sit on the 2x2 lattice (M = 3,
`n_orders = 1`; pencil `2*(4*2)^2 = 128`, so the advice fires and the QZ is
trivial — the obvious 12x12 fixture is the 498 s / 8.4 GB case the warning is
*about*):

| path | reported at | module filter that catches it |
|---|---|---|
| `pmm_efficiency_2d_staggered` (direct) | the caller's own line | `__main__` only |
| `PMM2DStackPure.solve` (deferred, shared grid) | `stack2d_pure.py:1441` | `stack2d_pure` only |
| either | — | `...twod_staggered`: **NOT caught** |
| either | — | message filter: **CAUGHT** |

**Implemented.**  A block beside the warning gives CI the exact spelling
(`-W "error:.*eps_cell is a SEGMENT grid:UserWarning"`, and the pytest
`filterwarnings` form) with the table above and the reason a module filter
cannot work.

**Also fixed, because the measurement turned it up:** the deferred path's
stacklevel is one frame SHORT — it points at `stack2d_pure.py`'s own
`self._warn_stag_shared_redundancy()` line, not at the user's `solve()`.
`_validate_stag_cost` gains `stacklevel=3` as a keyword (default unchanged, so
every existing caller is bit-identical) so that call site can pass 4 in a
one-word change.  Requested in §5.3; the half that is mine is done and pinned.

**Pinned** by `test_audit2609_a21_pmm_warning_filter.py` (6 ids, 1.1 s): the
documented filter raises on both entry points; the module filter does **not**
catch it on either (a deliberately two-edged counter-pin — if it ever started
catching, the warning would have stopped pointing at the caller); the direct
path reports at the calling file; and `stacklevel=1`/`2` are honoured, so the
requested one-word change lands on a keyword that demonstrably works.

### 2.8 The committed identifier resolver (deliverable 8)

A18's scratch resolver was present at `<scratch>/wpa18/resolve_ids.py` and is
the basis of `scripts/check_doc_identifiers.py`.  Changes made in committing
it: the repo root comes from `__file__` (it was hard-coded); the index is built
in a class so importing the module is cheap and a test can call `check()`
in-process; `--docroot` / `--no-exclusions` replace two `WPA18_*` environment
hooks; `main()` returns 0/1 so CI can gate on it; the module docstring carries
the rules from A18 §2.1 and the "no network" claim; and the ~25-line resolver
docstring explains why the second exclusion layer is a list and not a rule.

Two list changes, both measured: `wavelength_nm` removed (it resolves honestly
as a member of `ui.model.SourceDefinition` through the static index — its
"prescription dict key" reason was wrong), and the three lens config objects
re-worded, because `LensGeometry` / `LensNumerics` / `LensResources` have since
landed as real top-level exports (`elements/lens_config.py`) and their
"proposed future name (ROADMAP)" reason is no longer true.  They are kept as
belt-and-braces so this committed gate does not depend on an uncommitted
landing; the comment says so and says they can simply be deleted.  Net: 52 → 51 triaged entries, and the
denominator grows 592 → 593.

**Measured on the current tree** (identical to A18's numbers):

```
backticked tokens scanned (occurrences): 1952
distinct identifier-shaped tokens:       1055
API-claiming denominator (distinct):      593
  resolve:                                593
  DO NOT resolve:                           0
unresolved per file: README 0 / ROADMAP 0 / Migration-Guide 0 / CONVENTIONS 0
rc 0, elapsed 4.5 s
```

**The gate** (`tests/unit/test_audit2609_a21_doc_identifiers.py`, 5 ids, 6.6 s,
one module-scoped index shared by all five) asserts:

1. zero unresolved, with the offending tokens and their citation sites in the
   failure message;
2. the denominator has not collapsed (floor 450 against a measured 593) — a
   resolver can always reach zero by excluding everything, so the zero is only
   worth what it is measured over;
3. the gate FAILS on a fabricated name, demonstrated in-process against a
   temporary document carrying both shapes the audit found
   (`propagate_through_a_wormhole()` and
   `lumenairy.propagators.definitely_not_here`) plus a real control, and
   `main()` returns 1;
4. every triaged exclusion is still CITED in one of the four documents
   (measured 51/51 — no dead entries), with a "may shrink, never grow" ceiling;
5. **no network**, asserted structurally by walking the script's own AST for an
   import of any networking module rather than by trusting the docstring.

Bounded runtime, stated: the resolver imports the package and AST-walks
`lumenairy/**/*.py` once; 4.5 s standalone, 6.6 s for the whole test file.  No
wall clock is asserted (TESTING_STANDARDS S1).

### 2.9 `propagation._LIVE_FORWARD_NAMES` (orchestrator addition)

`fft_infra._PYFFTW_FIRST_FFT_THREAD` and `_PYFFTW_SHARED_BUFFERS_UNSAFE` — the
K4 ping-pong-buffer race latch VERIFY-A5 added, rebound by `_note_fft_thread()`
the moment a second thread issues an FFT — were absent from
`propagation._LIVE_FORWARD_NAMES`, so a consumer reading
`propagation._PYFFTW_SHARED_BUFFERS_UNSAFE` saw the import-time snapshot
forever.  Added to the frozenset (not to the walker's exemption list: they are
live state, not a retired latch), with a why-comment saying what reads them and
why a snapshot is wrong.

Verified behaviourally as well as by the walker:

```
before: False None        # propagation.*
(set fft_infra._PYFFTW_SHARED_BUFFERS_UNSAFE = True, _PYFFTW_FIRST_FFT_THREAD = 12345)
after : True 12345        # propagation.*
```

`test_v5_2_walker_pep562_forwarding.py`: 1 failed, 3 passed → **4 passed**.
Fail-before: removing the two names reproduces the original failure
("...NOT listed in propagation._LIVE_FORWARD_NAMES").

---

## 3. Files touched

**Source (7)**

* `lumenairy/elements/rcwa/oned.py` — dropped the dead `_grazing_safe_wavelength`
  import (`:26`), re-sorted the `_core` block (`:53`).
* `lumenairy/elements/rcwa/twod.py` — same (`:24`, `:50`); `fn_name`
  `RCWA2DPrepared.solve` → `PreparedRCWA2D.solve` at `:1300` and `:1302`.
* `lumenairy/elements/rcwa/stack.py` — re-sorted the `_core` block (`:58`).
  *(Ownership note, §2.1.)*
* `lumenairy/memory.py` — `set_low_memory`'s two guards narrowed to
  `except ImportError:` with the unguardable work moved out of the `try`, and
  the aggressive-mode `RuntimeWarning` moved after the restore snapshot
  (`:1242-1283`, `:1306-1316`).
* `lumenairy/ui/lens_options_dialog.py` — the `fast_analytic_phase` tooltip
  (`:135-148`).
* `lumenairy/elements/pmm/twod_staggered.py` — `_validate_stag_cost` gains
  `stacklevel=3` (`:493`, used at `:637`); the CI-filter block (`:598-624`).
* `lumenairy/propagators/propagation.py` — two names added to
  `_LIVE_FORWARD_NAMES` (`:280-288`).

**Configuration (1)**

* `pyproject.toml` — deleted the three RCWA `TODO(audit-2609)` per-file ignores
  (was `:424-433`); `[tool.mypy] files` 17 → 25 paths (`:527-534`).

**Tests, existing (10)**

* `tests/unit/test_audit_except_budget.py` — census 53 → 51.
* `tests/unit/test_audit2609_a15a_packaging.py` — `_STDIO_EXEMPT` emptied;
  mypy-whitelist floor 17 → 25.
* `tests/unit/test_audit2609_a7_misc.py` — the ULP band.
* `tests/unit/test_pmm2d_staggered_slant.py` — the eigenwork census.
* `tests/unit/test_v4_15_agent_a.py` — the batched-vs-per-pixel census.
* `tests/unit/test_niche_d2_chain_multi.py`,
  `tests/unit/test_pmm2d_staggered_mortar.py`,
  `tests/unit/test_niche_d6_exact_tilted_leg.py` — `pytestmark`.
* `tests/unit/test_audit_misc.py` — one `@pytest.mark.slow`.
* `tests/unit/test_niche_audit_w3_infra.py`,
  `tests/unit/test_niche_d14_deterministic_carrier_fit.py` — `stdin=`.

**New (4)**

* `scripts/check_doc_identifiers.py`
* `tests/unit/test_audit2609_a21_doc_identifiers.py`
* `tests/unit/test_audit2609_a21_pmm_warning_filter.py`
* `tests/unit/test_audit2609_a21_rcwa_and_ui.py`

Plus this report and `fixes/WP-A21_CHANGELOG.md`.

---

## 4. Tests run

All commands `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest <files> -q --no-header -p no:cacheprovider`.

| batch | files | result | duration |
|---|---|---|---|
| fast | `test_memory_guardrail.py`, `test_audit2609_a9_ui.py`, `test_verify_perf_fixes_2026_08_10.py`, `test_audit2609_a7_misc.py`, `test_v5_2_walker_pep562_forwarding.py`, `test_audit2609_a15a_packaging.py`, `test_audit2609_a21_pmm_warning_filter.py`, `test_audit2609_a21_rcwa_and_ui.py`, `test_audit2609_a21_doc_identifiers.py` | **137 passed, 3 skipped** | 19.0 s |
| census | `test_audit_except_budget.py` | **3 failed, 1 passed** — not mine, §5.2 | 0.7 s |
| A | `test_audit_misc.py`, `test_niche_d6_exact_tilted_leg.py` | **1 failed, 263 passed, 3 skipped** — not mine, §5.6 | 272.3 s |
| B | `test_niche_d2_chain_multi.py`, `test_pmm2d_staggered_mortar.py` | **69 passed** | 538.1 s |
| C | `test_rcwa.py`, `test_audit2609_a14_rcwa_eme_bor.py`, `test_audit2609_a14_verify.py` | **151 passed** | 272.4 s |
| D | `test_pmm2d_staggered_slant.py`, `test_v4_15_agent_a.py`, `test_niche_audit_w3_infra.py`, `test_niche_d14_deterministic_carrier_fit.py`, `test_niche_audit_w9_traced_determinism.py` | **184 passed** | 235.4 s |

Batch C is the RCWA **bit-identity** requirement of deliverable 1: 151 passed
across `test_rcwa.py` and both `test_audit2609_a14_*.py` files, unchanged, with
the dead imports dropped and the three import blocks re-sorted.

The 3 skips in the fast batch are `test_verify_perf_fixes_2026_08_10.py`
declining a numexpr retention check (numexpr is not installed); the 3 in batch
A are `test_audit_misc.py` skipping cupy/h5py-guarded ids.

**Two failures, neither introduced here** — §5.2 (the except census, WP-A16's
new `lens_config.py`) and §5.6 (the d6 carrier envelope).  Both are attributed
by measurement, not by assertion.

Other gates:

```
ruff check lumenairy tests scripts --no-cache   ->  All checks passed!
python -m mypy                                  ->  2 errors, both pre-existing (§5.1); 30 source files
python -c "import lumenairy"                    ->  lumenairy 5.45.1
python scripts/check_doc_identifiers.py         ->  rc 0, 593/593 resolve, 4.5 s
```

---

## 5. Requested changes outside my ownership

### 5.1 `lumenairy/backend/__init__.py` (WP-A16) — annotate the PEP 562 pair

`mypy` is RED on the whitelist and was before I touched it:

```
lumenairy\backend\__init__.py:56: error: Function is missing a type annotation  [no-untyped-def]
lumenairy\backend\__init__.py:79: error: Function is missing a return type annotation  [no-untyped-def]
```

`lumenairy/backend` has been in `[tool.mypy] files` since v5.0; the lazy
`scipy` forward that landed there (uncommitted; A15b §5.1's request) is
unannotated.  Two lines:

```python
def __getattr__(name: str) -> Any: ...
def __dir__() -> list[str]: ...
```

(with `from typing import Any`).  A15a §5.10 anticipated exactly this — "add
them to `[tool.mypy] files` once that lands **with the `__getattr__`
annotated**".

The same change unblocks the ninth whitelist entry.  `lumenairy/__init__.py`
measures 33 errors: the same unannotated pair at `:2192`/`:2220`, plus 31
`Module "lumenairy.elements.lenses" does not explicitly export attribute X`
rows (`PreparedAnalyticLens`, `PreparedTracedLens`, `TiltedCarrier`,
`prepare_real_lens`, `set_lens_parallel_amp`, `get_lens_sag_dtype`, ... and
`raytrace.trace_world`).  Under `--strict`, a re-export must be in the
defining module's `__all__` or spelled `from X import Y as Y`.  Once those
land, add `"lumenairy/__init__.py"` to the list and raise the ratchet 25 → 26.

### 5.2 `lumenairy/elements/lens_config.py` (WP-A16) — three broad excepts

`tests/unit/test_audit_except_budget.py` is RED, and **not because of my
change**.  Measured:

```
over = {'elements/lens_config.py': (3, 0)}
tree total 53, census 51
```

The file is new and untracked (`git status` → `?? lumenairy/elements/lens_config.py`)
and carries three `except Exception:` clauses at `:124`, `:128`, `:134`, all in
one `_same(a, b)` value-comparison helper.  With my memory.py narrowing reverted
the over-set is identical (tree 55 vs census 53), so the failure is entirely
attributable to that file.

I did **not** add a census entry for it: the gate's own message calls an
unjustified bump "the defect this gate exists to catch", and the file's owner
should be the one to say whether these are kept or narrowed (the first,
`np.array_equal`, raises only `TypeError`/`ValueError`).  They are almost
certainly justified — the census already carries the identical case, and the
file's own comment states it — so if the owner keeps them, this is the entry,
to be added to group (3) `_PROBE_GUARDS`:

```python
    'elements/lens_config.py': 3,     # _same compares two config values of
                                      # arbitrary type (ndarray == ndarray is
                                      # not a bool); a non-comparison is
                                      # reported as "not equal".  Same case as
                                      # _knobs.py, which this file deliberately
                                      # does not share (that module is a leaf
                                      # with no NumPy import).
```

and the counter-pin's measured line becomes "census 54, tree 54, slack 0".

### 5.3 `lumenairy/elements/pmm/stack2d_pure.py:1418` (WP-A13 / VERIFY-A13) — one word

The deferred shared-grid warning is reported at `stack2d_pure.py:1441`, i.e. at
the library's own `self._warn_stag_shared_redundancy()` line, because that path
reaches `_validate_stag_cost` through one extra helper.  `_validate_stag_cost`
now takes `stacklevel=`; the call at `:1418` should pass `stacklevel=4`:

```python
        _validate_stag_cost("PMM2DStackPure.solve", int(self.M), *cells,
                            check=("warn",), stacklevel=4)
```

Measured today: at 3 the warning reports in `stack2d_pure.py`; at 1 it reports
in `twod_staggered.py` and at 2 in the immediate caller, so the keyword is
demonstrably honoured (pinned by
`test_audit2609_a21_pmm_warning_filter.py::test_the_stacklevel_is_a_parameter_so_the_deferred_path_can_correct_it`).
Nothing breaks until it lands: the default is unchanged.

### 5.4 `Migration-Guide.md:1289` (WP-A18) — optional simplification

The guide names both `RCWA2DPrepared.solve` and the real class because the
shipped warning used the wrong spelling.  Only `PreparedRCWA2D.solve` ships
now, so the sentence could be reduced to the real name plus a note that the old
spelling appeared in warnings before v5.46.  Until then the old spelling stays
on `scripts/check_doc_identifiers.py`'s triaged list with that reason written
out; if the guide drops it, the entry should be deleted in the same change (the
gate's `test_every_hand_triaged_exclusion_is_still_cited` will say so).

### 5.5 Orchestrator — the slow lane grew again

This pass moves **1 001.4 s** into the slow lane (three module markers and one
test marker).  A15a measured the slow lane at 32.9 min on 3 shards / 19.7 min
on 5 and raised it to 5; this adds ~16.7 min of work to it.  The shard-count
review belongs with the `.test_durations` regeneration A15a §5.1 asks for at
campaign exit, not now.

### 5.6 `tests/unit/test_niche_d6_exact_tilted_leg.py::test_decentred_carrier_decentre_penalty_envelope` — RED, owner WP-A16 / VERIFY-A6

```
E   AssertionError: the ON-AXIS path regressed: EE2 ratio 0.9698
E   assert 0.9697869227555169 > 0.97
```

A 0.02 % miss on a per-build bar.  Attribution, measured rather than asserted:

* **my change to that file cannot cause it** — `git diff` shows exactly one
  added statement, `pytestmark = pytest.mark.slow`, plus its comment;
* **WP-A17's carrier work cannot cause it** — `git diff -U0` on
  `propagators/carrier.py`, `carrier_field.py` and `fft_infra.py`, filtered to
  non-comment non-blank changed lines, returns **0 lines**.  That work is the
  history relocation and is comment-only;
* **WP-A16's lens-family work can** — the same filter returns 101 lines on
  `elements/_lens_traced.py`, 91 on `_lens_real.py`, 64 on
  `_lens_traced_multibranch.py`, 50 on `propagators/fga.py`, 40 on
  `lenses.py`, 31 / 29 / 19 on the rest.  This test's chain runs straight
  through `_lens_traced`: `la.TiltedCarrier.__module__` is
  `lumenairy.elements._lens_traced`.

So the failure belongs with whoever holds the lens family this round.  Worth
saying separately: `r_on > 0.97` against a measured 0.9698 is the per-build-bar
shape TESTING_STANDARDS S5 warns about — the bar carries no derivation and no
error floor in the test, so it cannot distinguish "the on-axis path regressed"
from "the lens family moved by 0.02 %".  Whoever owns it should either derive
the bar or replace it with a two-arm comparison.

### 5.7 Orchestrator — wire the new gate into CI, if wanted

`scripts/check_doc_identifiers.py` exits 0/1, needs no network and takes 4.5 s.
It is already covered by `tests/unit/test_audit2609_a21_doc_identifiers.py`, so
a separate CI step is optional; a `docs` job would give a clearer signal than a
unit-test failure.  Same shape as the existing
`scripts/check_source_line_citations.py` wiring.

---

## 6. Deferred, with designs

1. **`test_ci_kernel_consistency.py:492` (`assert elapsed < 20.0`).**  The last
   live wall-clock assertion in `tests/`.  A15a §6 item 3 already designed the
   conversion (assert on the NUMBER OF DECISIONS probed, not on seconds) and
   deferred it to whoever owns the cross-arm census, because the honest
   conversion needs that owner to say what the decision count should be.  I did
   not touch it: my brief scopes item 5(a) to the VERIFY-A6 / VERIFY-A13 files.
   Effort ~1 h with the number in hand.
2. **A message-filter CI step for the PMM staggered advice.**  §2.7 gives the
   exact spelling and pins that it works; arming it is a `unit-tests.yml`
   change (a `-W` on the PMM job, or a `filterwarnings` entry scoped to
   `tests/unit/test_pmm2d_staggered*.py`).  Not done: I own no workflow file,
   and arming it now would fail the two staggered files until every
   `max_pencil_dof=` acknowledgement in them is audited.  Effort ~1 h plus one
   CI run.
3. **Delete the three lens-config entries from the resolver's triaged list.**
   `LensGeometry` / `LensNumerics` / `LensResources` resolve on their own now;
   the entries are kept only so this gate does not depend on an uncommitted
   landing.  Once WP-A16 is committed, delete them and the denominator goes
   593 → 596.  Effort ~2 min.
4. **The remaining ~20 `.md`-prose tests** (A15a §6 item 4) and the
   `lumenairy/ui/` ruff cleanup (A15a §6 item 6) are untouched; both need one
   owner in one pass, and the UI one needs a GUI runner this box does not have.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A21_CHANGELOG.md`

# WP-A15b changelog text (for assembly into `CHANGELOG.md`)

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` section 14 (TESTS-ARCH) finding **V6** /
sub-findings **P2-5** (process-global knobs), **P2-7** (import-time cost), **P2-8** (layering),
**P2-9** (duplicated backend-detection scaffolds), plus the re-export requests left by
WP-A1 / A5 / A7 / A10 / A11 / A13 and the `subharmonics` passthrough requested by WP-A8.

---

### Added -- config: `lumenairy.override(...)`, a scoped context manager for every process-global knob

`with lumenairy.override(fft_threads=1, pyfftw_planner='FFTW_ESTIMATE'): ...` now scopes any
registered knob and restores the previous values in reverse order on every exit path -- normal
return, `break`, or exception -- with a roll-back if a setter raises partway through entry.
Unknown names raise `ValueError` **before** the first setter runs, so a typo cannot leave half a
block applied.

The audit measured **53 `set_` verbs, 0 context-manager forms and 0 resets** across 227 modules,
in a suite that is deliberately run serially: a knob a test forgot to put back changed a later
test's physics and did not reproduce under `-k`. New `lumenairy/_knobs.py` (357 lines) carries the
registry -- `register_knob(name, *, getter, setter, doc)`, `override(**knobs)`, `snapshot()`,
`restore(snapshot)`, `knobs()`, `knob_doc(name)` -- and **17 knobs** are registered one line beside
their setter:

* `lumenairy/propagators/fft_infra.py` (12): `asm_cache_size`, `default_complex_dtype`,
  `default_dy`, `default_real_dtype`, `default_wave_propagator`, `fft_auto_promote`,
  `fft_double_buffer`, `fft_fallback`, `fft_plan_cache_size`, `fft_plan_max_bytes_per_buffer`,
  `fft_threads`, `pyfftw_planner`;
* `lumenairy/memory.py` `max_ram`; `lumenairy/cache.py` `cache_budget`;
  `lumenairy/io/storage.py` `storage_backend`; `lumenairy/user_library.py` `library_path`;
  `lumenairy/elements/rcwa/_core.py` `blas_threads`.

Registration happens at import time of the module that OWNS the knob, so a knob becomes visible
only once that module is imported -- documented, and the reason `knobs()` grows as a session
proceeds. `override` is process-global (not thread-local) and is documented as such.
`lumenairy_context` (which scopes a fixed set of five knobs) is unchanged and still works.

`tests/conftest.py` gains an autouse per-test fixture (`_knob_leak_guard`) that snapshots every
registered knob and restores it, alongside the existing flag and glass guards; set
`LUMEN_TEST_KNOB_LEAK_STRICT=1` to fail the leaking test instead of restoring silently.
Measured cost: snapshot 2.98 us, no-change restore 5.67 us, i.e. **11.6 us per test** -- 0.155 s
across the 13 319-test suite (1.2e-5 of its 3.65 h). A setter is called only for a knob whose
value actually changed, so a clean test does not pay the cache invalidations several setters carry.

Tests: `tests/unit/test_audit2609_a15b_optional_and_knobs.py` (33 tests -- restore ORDER,
roll-back on a raising setter, nesting, unknown-name validation before any apply, worker-thread
use, the "every audited process global is registered" census, the `setter(getter())` round-trip
for all 17).

### Added -- backend: `lumenairy/backend/_optional.py`, one place for the optional-accelerator probes

`ensure_cupy()`, `cupy_module()`, `is_cupy_array(x)`, `load_numba()`, `numba_handles()` plus the
`CUPY_AVAILABLE` / `NUMBA_AVAILABLE` flags. The audit found `_ensure_cupy_loaded`,
`_is_cupy_array` and `_load_numba` hand-copied **five times each** across eight modules, so the
accelerator-absent path had five implementations and no single place to test it. The three
NON-lens sites now delegate -- `propagators/fft_infra.py:46-48`, `sources/core.py:26-27`,
`optimize/_merit_jit.py:44-45` -- while keeping their module-level `cp` alias and their
`_ensure_cupy_loaded` / `_is_cupy_array` / `_NUMBA_AVAILABLE` names, all of which are load-bearing
(`fft_infra.__all__` exports two of them, `propagation.py` re-exports them, and
`test_v5_3_multi_field_merit_jit.py` monkeypatches `_merit_jit._NUMBA_AVAILABLE`). Behaviour is
unchanged, including CONVENTIONS section 10 absence semantics (absence is a value, never an
exception) and the first-use memoisation. The five lens-family copies are WP-A16's to switch.

### Performance -- import: `import lumenairy` no longer builds the rigorous-solver stack

`lumenairy/elements/__init__.py` and the solver blocks of `lumenairy/__init__.py` resolve
`berreman` / `bor` / `eme` / `pmm` / `rcwa` and their public names through a PEP 562
`__getattr__` (63 names at the package root, 38 of them also on `lumenairy.elements`), caching
each resolved object into module globals so the second access is an ordinary dict hit. `from lumenairy.elements import rcwa`, `import lumenairy.elements.rcwa as r`,
`lumenairy.elements.rcwa.X`, `dir()`, `from lumenairy.elements import *`, pickling of objects
defined in those modules, and every name in both `__all__` lists behave exactly as before; an
unknown name raises `AttributeError` (never `ImportError`).

MEASURED (`python -X importtime -c "import lumenairy"`, medians of 5 before / 7 after on a
shared workstation): `lumenairy.elements` cumulative **144.6 ms -> 121.7 ms**, and the four solver
subpackages (16.0 + 5.3 + 6.1 + 5.3 ms cumulative) disappear from the trace entirely. The
noise-free interleaved measurement -- the same build, `import lumenairy` against
`import lumenairy` plus an explicit import of exactly what used to be eager, 9 pairs -- gives
**749.0 ms -> 711.7 ms, i.e. -37.3 ms (-5.0 %)**.

That is much less than the audit's "~65 % of import time", and the reason is a measurement the
audit's chain no longer matches: on this HEAD `scipy.linalg` + `scipy.special` are reached through
`analysis.beam_stats -> lumenairy.backend -> backend.scipy` (then a module-level `from . import scipy` in
`backend/__init__.py`; WP-A16 replaced it with the PEP 562 `__getattr__` at
`lumenairy/backend/__init__.py:58`), not through `elements -> berreman -> rcwa`, and `lumenairy.backend` has
**41 module-level importers** including `propagators.rs`, `propagators.hf` and
`elements.elements`. `backend.scipy` costs 566 ms of the 955 ms total. Deferring it is a ~12-line
PEP 562 change in `lumenairy/backend/__init__.py` -- outside this work package -- and it is worth
~47 ms on its own, or ~540 ms (≈70 %) together with the module-level
`from scipy.special import airy` in `elements/_lens_traced_multibranch.py` (WP-A16 binds it on
first use in `_airy`), which was the only other module-level `scipy.special` importer left. Both requests are in `WP-A15b_REPORT.md` section 5 with the measured numbers.

### Fixed -- architecture: the library's last import-time layering violation

`lumenairy/raytrace/surface.py` imported `elements.lenses` at module level -- the ONE import-time
edge among the audit's 12 layering violations (the other 11 are already in-function). The two sag
kernels move into `_base_surface_sag_xy`, which already imports `elements.freeform` the same way
three lines below. Bit-identical: the sag dispatch is pinned against `elements.lenses`'s own
kernels with `np.array_equal` on rotationally-symmetric, biconic and flat surfaces.
Tests: `tests/unit/test_audit2609_a15b_lazy_and_layering.py` (23 tests, including an AST walk of
the whole package that separates module-level from in-function imports and excludes
`if TYPE_CHECKING:` bodies).

### Added -- public API: 15 names promoted to the package root

`lumenairy.__all__` 708 -> 723. Every one was already public in a submodule `__all__` and was
requested by the work package named beside it; the `__all__`-symmetry walker was RED on 19 entries
and is now green with **no new exemptions**:

* `unwrap_phase_2d`, `clear_meshgrid_cache`, `meshgrid_cache_bytes`, `zernike_basis_cache_bytes`
  (WP-A7);
* `MinEdgeThicknessMerit`, `edge_thickness` (WP-A10);
* `pmm_2d_order_drift` (WP-A13 / VERIFY-A13; VERIFY-A12 had added it to
  `elements/pmm/__init__.py::__all__`);
* `exit_vertex_transfer`, `EXIT_VERTEX_GRAZING_TOL`, `resolve_exit_index`,
  `vertex_plane_transfer_t`, `exit_vertex_transfer_jax` -- the shared exit-vertex transfer of
  audit section 15.1 and its pieces (WP-A1);
* `seed_entrance_eikonal` (WP-A7 section 5.3; also added to `lumenairy.raytrace.__all__`
  alongside `resolve_exit_index` / `vertex_plane_transfer_t`);
* `aberration_free_reference_fit` (WP-A4's Y2 follow-up);
* `override` (this work package).

`algebra.from_prescription` stays submodule-only by decision: `Operator.from_prescription` is the
canonical spelling and a free function would put two names with identical semantics on the surface.
The walker's existing exemption and its cited rationale stand, and
`tests/unit/test_audit2609_a15b_reexports.py` records the decision so it is not silently
re-litigated.

### Fixed -- propagators: `propagate_through_system` forwards `subharmonics` to the turbulence screen

`system.py:1007` now passes `subharmonics=elem.get('subharmonics', 0)` to
`generate_turbulence_screen`, and the element-dict documentation lists the key. WP-A8 added the
Lane subharmonic levels (audit section 5, E3) but the element chain was the one caller that could
not reach them, so `{'type': 'turbulence', ..., 'subharmonics': 3}` was silently discarded -- the
"documented option that does not work" class of audit section 15.3. The default is the generator's
own `0`, so a chain without the key is **bit-identical** (`np.array_equal` against the 0-level
screen applied by hand). MEASURED effect of the discarded knob on the regression fixture:
`subharmonics=3` moves the screen by max |dphi| = 2.6701 rad, rms 1.3526 rad (1 level: 0.8413 /
0.3906 rad). Tests: `tests/unit/test_audit2609_a15b_system_subharmonics.py` (6 tests).

### Changed -- memory: `_ASM_FIRST_CALL_FIXED_BYTES` re-calibrated 56 -> 40 MiB, and its test bar is now derived

`estimate_asm_memory`'s one-time first-call term was calibrated at 56 MiB on 2026-08-01 against a
then-measured 52.5-53.0 MiB backend import. RE-MEASURED 2026-09-12 by the same method (fresh
interpreter + `tracemalloc`, 12 points N = 64..2048 x {complex64, complex128}, fitting
`cold = slope * N^2 + fixed`): the import cost is now **36.71 MiB** -- the three N >= 256
complex128 pair fits agree to 0.01 MiB and the cold peak reproduces to 0.003 % over 5 fresh
interpreters. At 56 MiB the estimate read **est/measured = 1.341 at N = 512** against a docstring
promising 1.06-1.09: the published accuracy had stopped being true. The constant is back to 40 MiB
(6.3 % headroom over the worst pair fit, the same convention the 2026-08-01 calibration used), and
the measured band is **1.061-1.112** over all eight N >= 256 points, a bound at every one.

`tests/unit/test_niche_audit_w3_infra.py`'s flat `<= 1.35` Windows fence -- sized in 2026-08-01 to
admit a band then measured at 1.06-1.09, i.e. 25 % of unexplained slack -- is replaced by a
two-sided DERIVED bar: `est/cold` is a weighted mediant of `F_est/F_meas` and `s_est/s_meas` and
therefore lies between them for every N, giving `1.0 <= ratio <= 1.12` with no per-build number.
The measured cold peak is additionally pinned against its dated two-term model (2 % band on a
quantity that reproduces to 0.003 %), and a new test bars the CONSTANT itself at
`[F_meas, 1.10 * F_meas]` so the next drift has to be re-measured rather than absorbed. Both new
bars FAIL on the 56 MiB constant (1.341 / 1.188 against 1.12; 1.526x against 1.10).
`tests/unit/test_verify_perf_fixes_2026_08_10.py`'s two absolute GB pins move by exactly the
16 MiB re-calibration (19.762 -> 19.745 GB; 22.648 -> 22.631 GB) and now derive that term from the
constant so a future re-calibration moves them with it. Closes WP-A11 section 5 item 4
("either the constant comes down or that test's Windows fence goes up -- one decision, one place").

### Changed -- tests: the `fft_infra` leak-guard comment now matches the code

`tests/conftest.py` cited `ui/waveoptics_dock.py` clearing `USE_PYFFTW` unconditionally as the live
reason `fft_infra` is in `_LEAK_GUARD_MODULES`. WP-A9 fixed the dock (U4 -- every exit path
restores the FFT and RAM overrides), so the sentence now says the leak is fixed and the guard is
there to catch a recurrence. No code change. (WP-A9 report section 5.3.)

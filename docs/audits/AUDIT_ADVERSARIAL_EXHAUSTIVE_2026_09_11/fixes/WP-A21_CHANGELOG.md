# WP-A21 -- changelog text (final hygiene pass, audit 2026-09-11)

Assembled by the release step into `CHANGELOG.md`.  Every number below was
measured on branch `audit-fixes-2026-09`, Windows 11 / CPython 3.14.6 /
numpy 2.4.4 / scipy 1.17.1, with `OMP/OPENBLAS/MKL_NUM_THREADS=1`.

---

### Fixed -- RCWA 2-D: the Wood-anomaly diagnostic named a class that does not exist

`rcwa_efficiency_2d`'s prepared sweep path passed
`fn_name="RCWA2DPrepared.solve"` into the Rayleigh-anomaly nudge and into the
`_WoodAnomaly` control-flow signal (`lumenairy/elements/rcwa/twod.py:1300`,
`:1302`).  There is no `RCWA2DPrepared`: the class is **`PreparedRCWA2D`**
(`lumenairy/elements/rcwa/twod.py:1258`, built by `prepare_rcwa_2d`), so a user who grepped the
`WoodNudgeWarning` text for the class found nothing, and `Migration-Guide.md`
had to document both spellings.  Both literals now name the real class.
Measured on the 2-D Moharam mount (`Lambda_x = Lambda_y = lambda = 1 um`, air
both sides, `n_orders 3x3`): the emitted warning now leads with
`PreparedRCWA2D.solve:` and `getattr(rcwa.twod, 'PreparedRCWA2D').solve`
resolves.  No numerics change -- the nudge, the symmetric bracket and the
returned `wl_eff` are untouched.  (Audit WP-A18 section 8.2.)

### Fixed -- designer: the `fast_analytic_phase` tooltip promised an error and a speedup that were not measured

`lumenairy/ui/lens_options_dialog.py`'s `LENS_KWARG_REGISTRY` entry claimed
"~25 % speedup with <10 nm OPL error on typical refractive prescriptions".
Both halves were wrong:

* the OPL error is not a flat 10 nm.  The omitted term is the **in-glass ASM
  leg**, so it is LINEAR IN CENTRE THICKNESS, not in f-number: measured
  0.007 nm rms at 1 um of centre thickness, 0.7 nm at 100 um and
  **14.1 nm rms / 41.6 nm PV at 2 mm** on an N-BK7 100/-100 biconvex at
  f/12.1 -- i.e. **~7 nm rms per mm of glass**, the form
  `docs/subsystems/real_lens.md` section 4 and
  `_lens_traced._geometric_lens_phase`'s docstring already carried;
* there is no ~25 % speedup on this build.  Measured on a 2 mm N-BK7 singlet,
  medians of 9 interleaved pairs: **0.87x-1.00x** (i.e. no gain, and a small
  loss) with the "Parallel amp pass" default ON -- that option already
  overlaps the analytic reference leg with the amplitude leg, so skipping the
  reference saves nothing -- and **1.06x-1.17x** with it off (N = 512 / 1024).

The tooltip now states the per-mm error with its measurement and says the
option is an accuracy control, not a speed one.  Behaviour unchanged.
(Audit WP-A3 section 5.8, re-raised by WP-A18 section 8.3.)

### Fixed -- `memory.py`: two broad `except Exception:` narrowed, and a promoted warning no longer swallowed

`set_low_memory`'s two `try: from .propagators.fft_infra import ... except
Exception: pass` guards (`lumenairy/memory.py`) are now `except ImportError:`
with only the import inside the `try`.  The guarded work cannot raise anything
else and now says so: `get_default_complex_dtype()` is a bare global read, and
`set_default_complex_dtype(np.complex64)` validates against the
`{complex64, complex128}` pair it is being handed a member of.  The module is
imported unguarded higher in the same function, so the reachable failure is
narrower still -- a vendored `fft_infra` without the two dtype names.

One behaviour improves as a consequence.  The aggressive-mode
`RuntimeWarning` ("set the default field dtype to complex64") used to sit
INSIDE the `except Exception`, so under `-W error::RuntimeWarning` the promoted
warning was caught and discarded -- the caller got no error, no warning, and a
flipped dtype.  It is now emitted **after** the restore snapshot is written, so
`-W error` sees it and `set_low_memory(False)` still has the prior values to
put back.

`tests/unit/test_audit_except_budget.py`'s per-file census drops the
`memory.py: 2` narrowing request: **53 -> 51 justified sites across 26
modules** (tree measured 51, slack 0).  (Audit WP-A15a section 2.9 / section 5
item 6.)

### Fixed -- `propagators/propagation.py`: the pyFFTW buffer-race latch was not live-forwarded

`fft_infra._PYFFTW_FIRST_FFT_THREAD` and `_PYFFTW_SHARED_BUFFERS_UNSAFE` -- the
K4 ping-pong-buffer race latch, rebound by `_note_fft_thread()` the moment a
second thread issues an FFT -- were missing from
`propagation._LIVE_FORWARD_NAMES`, so `propagation._PYFFTW_SHARED_BUFFERS_UNSAFE`
returned the import-time snapshot (`False`) forever.  Both are now forwarded;
measured, flipping the `fft_infra` globals is now visible through
`propagation` immediately.  `tests/unit/test_v5_2_walker_pep562_forwarding.py::test_v14_every_mutable_fft_infra_global_is_in_live_forward_names`
goes red -> green.

### Changed -- RCWA: three dead imports and three import blocks cleaned; the ruff TODO ignores are gone

`_grazing_safe_wavelength` was imported and no longer called in
`lumenairy/elements/rcwa/oned.py` and `twod.py` (WP-A14 routed both entry
points through `_grazing_safe_wavelength_pair`); the import is dropped, and the
`_core` import block in `oned.py`, `twod.py` and `stack.py` is re-sorted.  The
three `# TODO(audit-2609)` entries that covered them in
`[tool.ruff.lint.per-file-ignores]` are **deleted**, so those files are gated
like the rest of the package.  `ruff check lumenairy tests scripts` is clean.
No behaviour change: the name is still exported from `rcwa/_core.__all__` and
every caller imports it from there.  (Audit WP-A15a section 5 item 4.)

### Changed -- `mypy --strict` whitelist: 17 -> 25 declared paths

The eight subpackage `__init__.py` re-export surfaces (`_math`, `sources`,
`io`, `analysis`, `elements/pmm`, `elements/rcwa`, `elements/eme`,
`elements/bor`) join `[tool.mypy] files`, each measured strict-clean **after**
the lazy-loading rewrite.  They are pure re-export bookkeeping, which is
exactly where a mis-spelled or dropped name ships as an `ImportError` at a
user's first call.  `mypy` now checks 30 source files, up from 22.  The ratchet
in `test_audit2609_a15a_packaging.py::test_the_mypy_whitelist_only_grows` moves
17 -> 25.  (Audit WP-A15a section 5 item 10 / WP-A15b section 5.6.)

### Changed -- PMM 2-D staggered: how to catch the SEGMENT-grid redundancy advice in CI

VERIFY-A13 recorded that `-W error::UserWarning:lumenairy.elements.pmm.twod_staggered`
"does not reliably catch" the shared-grid redundancy warning and had to count
it by grepping the message text.  Measured and settled: a module-scoped filter
on that module **cannot** work at any correct `stacklevel`, because Python
matches a filter's `module` field against the `__name__` of the frame
`stacklevel` selects and the point of a non-1 `stacklevel` is to select the
CALLER.  Measured on a 4x4 segment grid reducible to 2x2 (M = 3): the direct
`pmm_efficiency_2d_staggered` entry is attributed to the caller's own module
and the deferred `PMM2DStackPure.solve` entry to `stack2d_pure`; neither to
`twod_staggered`.

The filter CI must use is therefore a MESSAGE filter, now documented beside the
warning and pinned:

```
-W "error:.*eps_cell is a SEGMENT grid:UserWarning"
filterwarnings = ["error:.*eps_cell is a SEGMENT grid:UserWarning"]
```

`_validate_stag_cost` also gains a `stacklevel=` keyword (default 3 --
unchanged for every existing caller) so the deferred path, which reaches the
warning through one extra helper and currently lands on `stack2d_pure.py`'s own
line instead of the user's `solve()` call, can pass 4 in a one-word change.

### Added -- `scripts/check_doc_identifiers.py`: the doc-identifier resolver is now a gate

The audit's "78 % of backticked identifiers in the top-level docs do not
resolve" was measured on a 60-token sample with no clean denominator; WP-A18
built the denominator and drove the genuinely unresolved count to 0 but could
commit no `.py`.  The resolver now ships beside
`scripts/check_source_line_citations.py`.  It extracts every backticked
identifier-shaped token from `README.md`, `ROADMAP.md`, `Migration-Guide.md`
and `CONVENTIONS.md`, removes two auditable layers of exclusions (mechanical:
builtins, live parameter names, test ids, file names, third-party prefixes,
enum-like option strings, file-format record tokens; plus 51 hand-triaged
tokens each carrying the reason read at its citation site) and resolves the
rest against the importable package plus a static AST index of
`lumenairy/**/*.py` -- the AST index is what makes `lumenairy/ui/` resolvable
where PySide6 is absent.

Measured on this tree: 1 952 backticked occurrences, 1 055 distinct
identifier-shaped tokens, a **593-token API-claiming denominator, 593
resolving, 0 unresolved**, 4.5 s end to end, no network.  Exit status 0/1 so CI
can gate on it.  `tests/unit/test_audit2609_a21_doc_identifiers.py` pins the
zero, pins that the denominator has not collapsed, demonstrates the gate
failing on a fabricated name, and keeps the triaged list from going stale or
growing.

### Changed -- test hygiene: three wall-clock bars converted, four slow markers, three subprocess spawns, one bit-equality

Following TESTING_STANDARDS S1 ("no wall-clock or speedup assertions"), the
last three live timing bars outside `test_ci_kernel_consistency.py` are
operation counts:

* `test_pmm2d_staggered_slant.py::test_cost_slant_is_free_against_the_out_of_plane_solve_it_must_use`
  -> `::test_cost_slant_adds_no_eigenwork_to_the_solve_it_must_use`.  Was
  `min(slant/vertical) < 2.0` over `perf_counter` medians, closed by
  `assert ti > 0.0` -- an assertion that cannot fail -- on a ratio its own
  docstring measured at 0.869..1.167 across WIN/WSL, idle and loaded.  Now a
  census of every dense eigendecomposition by kind and dimension: the vertical
  out-of-plane and the SLANTED region solves are each **one standard eig at
  4q^2** (256 at M = 5, 400 at M = 6) and identical, while the in-plane path
  the slant does not take is **one generalized eig at 2q^2** (128, 200).
* `test_v4_15_agent_a.py::test_modal_asymptotic_perf_win`
  -> `::test_modal_asymptotic_is_batched_not_per_pixel`.  Was
  `t_ref/t_new >= 5.0` at N = 128.  Now: the public path issues **exactly one**
  `_solve_envelope_stationary_batch` and **one** `_compute_M_b_batch` at
  N = 16/32/64 (flat across a 16x change in pixel count) and **zero** scalar
  per-pixel calls, while the preserved pre-v4.15 reference issues exactly
  `N*N` scalar solves.  The speedup is reported, not asserted (measured 14.9x /
  16.1x / 14.9x at N = 32/64/128).  The test also drops from ~35 s to 4.3 s.
* `test_audit2609_a7_misc.py::test_transfer_function_recurrence_engages_only_on_a_uniform_scan`.
  Was a bit-equality that was RED on this build by 1 ULP.  Diagnosed: both
  routes build a **bit-identical** transfer function (measured
  `max|H_scan - fftshift(H_asm)| = 0.0`); what differs is where the `fftshift`
  pair sits -- `through_focus_scan` transforms the shifted field,
  `angular_spectrum_propagate` uses the even-N identity at `asm.py:105-117` and
  runs unshifted.  Measured drift: **0 ULP at N = 64/128/256 and 2 / 4 ULP at
  the mixed-radix 96 / 192**.  The bar is now a derived two-sided band,
  `8 log2(N)` = 52.7 ULP at N = 96 (2-D FFT: `2 log2(N)` stages x `eps`,
  doubled by `|E|^2`, over a >= `eps/2` ULP), with an in-process counter-arm
  showing that a wrongly-engaged recurrence moves the answer by 2.6e14-5.0e14
  ULP.

`pytestmark = pytest.mark.slow` added to `test_niche_d2_chain_multi.py`
(338.8 s), `test_pmm2d_staggered_mortar.py` (229.3 s) and
`test_niche_d6_exact_tilted_leg.py` (124.7 s); `test_audit_misc.py` gets the
marker on ONE test instead (`ReadmeCookbookExamples::test_real_lens_from_zemax_cookbook`,
308.4 s of the file's 320.7 s) because the file is jax-guarded and a
module-level marker would have deleted its jax ids from the only CI leg that
installs jax.  **1 001.4 s moved out of the fast lane.**

`stdin=subprocess.DEVNULL` added to the last three partial subprocess spawns
(`test_niche_audit_w3_infra.py`, `test_niche_d14_deterministic_carrier_fit.py`
x2), the shape that raises `OSError [WinError 6]` under pytest's fd-capture on
Windows; `test_audit2609_a15a_packaging.py::_STDIO_EXEMPT` is now **empty**, so
that gate covers `tests/` with no hole.

---

**Migration notes.** None.  No public API, default, signature or numeric
result changes in this work package.  Two operational notes for anyone reading
CI configuration:

* the slow lane gains three files and one test (~1 001 s), so a shard-count
  review belongs with the `.test_durations` regeneration at campaign exit;
* the PMM 2-D staggered redundancy advice must be filtered **by message**, not
  by module -- see the block above.

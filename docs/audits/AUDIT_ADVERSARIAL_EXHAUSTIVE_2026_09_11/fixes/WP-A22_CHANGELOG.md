# WP-A22 changelog text

Voice and headings follow the repository's CHANGELOG conventions.  The
orchestrator assembles the real entry from this file.

---

### Performance -- `import lumenairy` is **2.1x** faster: `scipy.fft` is no longer imported eagerly

`lumenairy/propagators/fft_infra.py` imported `scipy.fft` at module scope.
`import lumenairy` reaches that module unconditionally
(`analysis.coherence` -> `elements.lenses` -> `propagators.fft_infra`), and on
this SciPy build `scipy/fft/_fftlog_backend.py` does
`from ..special import loggamma, poch`, so `scipy.special` and its shared
prefix (`numpy.f2py`, `numpy.testing`, `charset_normalizer`) were pulled in
too -- for every caller, including the ones that never take an FFT.

`scipy.fft` is now probed with `importlib.util.find_spec` at import and bound
on first use by `_ensure_scipy_fft_loaded()`, exactly as pyFFTW already was in
the same module.  `fft_infra._scipy_fft` still resolves (through a PEP 562
module `__getattr__`, which performs the same first-use import), so
`propagators/_bluestein.py` and any external reader are unaffected.

MEASURED, WP-A15b's interleaved same-build method, medians of 9 fresh-
interpreter pairs, `OPENBLAS_NUM_THREADS=1`:

| tree | arm | median | the 9 samples (ms) |
|---|---|---|---|
| BEFORE | `import lumenairy` | **633.3 ms** | 616.8 617.3 620.3 623.0 **633.3** 636.7 648.1 651.1 691.5 |
| BEFORE | + explicit `scipy.fft` | 628.3 ms | 614.6 621.7 623.3 623.8 **628.3** 635.6 652.1 668.9 669.2 |
| | | delta **-5.0 ms** | as expected: both arms were already eager |
| AFTER | `import lumenairy` | **298.1 ms** | 279.6 290.3 295.5 297.2 **298.1** 301.8 308.7 309.0 310.7 |
| AFTER | + explicit `scipy.fft` | 622.2 ms | 600.4 602.4 606.2 610.9 **622.2** 629.6 636.3 644.7 673.9 |
| | | delta **+324.1 ms** | the cost this change defers |

Two independent numbers that agree: the plain before/after median difference
is **633.3 -> 298.1 ms, -335.2 ms (-52.9 %)**, and the interleaved A/B on the
AFTER tree is **+324.1 ms**.  Named cumulative rows (`python -X importtime`,
medians of 5):

| module | BEFORE cum | AFTER cum |
|---|---|---|
| `lumenairy` | 622.7 ms | **280.3 ms** |
| `lumenairy.propagators.fft_infra` | 415.5 ms | **50.2 ms** |
| `scipy.fft` | 396.0 ms | *(not imported)* |
| `scipy` | 27.6 ms | 27.6 ms |
| `scipy.special` | 49.0 ms | *(not imported)* |
| `numpy.f2py` | 108.3 ms | *(not imported)* |
| `numpy.testing` | 96.6 ms | *(not imported)* |
| `charset_normalizer` | 77.2 ms | *(not imported)* |

The `find_spec` probe costs the plain `scipy` package import, 27.6 ms with
NumPy already loaded, against the 396.0 ms it defers.

**No behaviour change.**  `SCIPY_FFT_AVAILABLE`, `USE_SCIPY_FFT`,
`SCIPY_FFT_WORKERS`, every dispatcher and `fft_infra._scipy_fft` behave as
before; the first FFT that routes to SciPy imports it.  `_fft2` / `_ifft2` /
`_fft2_nd` / `_ifft2_nd` / `_scipy_or_numpy_fft2` / `_scipy_or_numpy_ifft2`
return bit-identical arrays.  Gate: the seven FFT/ASM/propagation test files
pass unchanged (248 passed).

### Added -- `scripts/record_history_fingerprints.py`, the maintenance path for the history pin

`tests/unit/test_audit2609_a17_history_relocation.py` pins every module with a
document under `docs/history/` to the AST and token fingerprints recorded when
its version-history narrative moved out of the source.  That pin has no expiry,
so the first *deliberate* code change to any of those modules turns it red --
correct, and unworkable if the only way to clear it were hand-editing a hash.

`python scripts/record_history_fingerprints.py <module> --reason "..."`
recomputes both fingerprints **with the checker's own helpers** (a second
implementation would be two definitions of "unchanged") and rewrites the two
header lines in place, appending a `re_recorded: <date> -- <reason>` line so
the header carries every baseline move and its justification.  `--check`
reports drift across every document and exits 1 without writing; a write
without `--reason` is refused.

**The rule, now stated in `CONTRIBUTING.md`, in the checker's module docstring
and in both of its failure messages: a commit that intentionally changes code
in a module with a history document re-records that document in the same
commit, and says why.**

Gated by `tests/unit/test_audit2609_a22_history_fingerprint_tool.py` (11 ids,
2.5 s), which drifts a temp copy of a two-file tree and re-records it.  It
asserts on the recorder's own AST that it defines neither fingerprint itself;
that `--check` is byte-for-byte read-only; that a value-preserving literal
re-spelling (`5` -> `0x5`, invisible to the AST) is still caught; that the
re-recorded hashes equal the *checker's* reading of the drifted source; that
every other header field survives; that two re-records leave two trail lines;
and that one malformed document does not abort the sweep.

### Changed -- the last wall-clock assertion in `tests/` is now an operation count

`tests/unit/test_ci_kernel_consistency.py::test_this_arm_agrees_with_the_committed_census`
ended with `assert elapsed < 20.0`.  That was a proxy for the real promise --
the current-arm re-take runs the four CHEAP census sections and leaves the two
expensive ones (mortar 4.7 s, T22 5.3 s) to the committed table -- and it is
the shape TESTING_STANDARDS S1 rules out: it goes red when the box is loaded,
and green on the regression it exists to catch, because a fast enough machine
can run both expensive sections inside 20 s.

It is replaced by the count WP-A15a section 6 item 3 designed, derived from
the committed census rather than transcribed: the re-take must cover exactly
the census's cheap rows, produce none from the expensive prefixes, and probe
the recorded number of hypothetical bars.  MEASURED: 24 decisions (interface
6, sliver 2, band 4, branch_cut 12) against 24 cheap census rows, 4
hypotheticals, 0 expensive rows, against the 30 expensive rows every measured
arm carries.  Both sides exact, gap 0.  The wall clock (1.7 s) is PRINTED for
triage and asserted nowhere.

### Fixed -- `tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py` passed only by collection order

Twelve of its 35 ids -- every `apply_real_lens_traced_jax` /
`apply_real_lens_maslov_jax` parametrisation -- failed when the file was run
ALONE, with `RuntimeError: ... requires double precision, but jax_enable_x64
is disabled`.  `jax.config` is process-global and the module never enabled
x64, so the file was passing on whatever another module (about twenty of them
do) had switched on first.

The flag is now set in the same `try:` block that decides whether JAX exists,
and a module-scoped autouse fixture re-asserts it at run time (two modules in
this suite deliberately toggle it off to exercise refusal paths, and imports
all happen before any test runs) and RESTORES the previous value on teardown,
so the file does not export its own precondition to whatever runs next.

MEASURED: alone, **12 failed / 23 passed -> 0 failed / 35 passed**.  Run
alongside `test_v5_12_0_audit_fixes.py`, which toggles the flag, 40 passed in
both orders.

### Changed -- documentation

* `Migration-Guide.md` 5.46.0 gained the three late packages' notes: the
  `LensGeometry` / `LensNumerics` / `LensResources` / `LensConfig` objects
  (**new API, no migration required** -- every keyword still works, with a
  runnable snippet verified bit-identical to the loose-keyword form, and a
  pointer to `docs/lens_configuration.md`); the `docs/history/` relocation
  with the re-record rule; and three diagnostic corrections (the committed
  `scripts/check_doc_identifiers.py` gate, the staggered-PMM **message**
  filter spelling `-W "error:.*eps_cell is a SEGMENT grid:UserWarning"` and
  why a module filter cannot work, and the `fast_analytic_phase` tooltip's
  real figure of ~7 nm rms **per mm of glass**).
* `Migration-Guide.md` no longer quotes the dead `RCWA2DPrepared.solve`
  spelling; it states that pre-v5.46.0 logs name a class that does not exist.
  The matching hand-triaged exclusion was deleted from
  `scripts/check_doc_identifiers.py` in the same change.
* `README.md`'s "Start here" table gained rows for `docs/lens_configuration.md`
  and `docs/history/`, and the real-lens section gained a pointer to the
  configuration objects.
* `CONTRIBUTING.md` gained a "Modules with a history document" section
  carrying the re-record rule and the commands, and a bullet against
  version-history narrative in the source.

### Changed -- gates tightened as their debts were paid

* `pyproject.toml`: the `lumenairy/elements/_lens_real.py = ["F541"]`
  per-file ignore is **deleted** -- the two placeholder-free `f"..."`
  continuation lines are gone (MEASURED: `ruff check --isolated --select F541`
  reports 0 on that file), so it is F541-gated like the rest.  The TODO-ignore
  group is now empty.  `ruff check lumenairy tests scripts` passes.
* `tests/unit/test_audit_except_budget.py`: the `elements/_lens_imap.py`
  census entry is deleted -- its clause is now `except ImportError:`.  Census
  **51 -> 50**, tree 50, slack 0.  The NARROWING REQUESTS group is empty:
  every narrowing the census asked for has landed.
* `scripts/check_doc_identifiers.py`: the `LensGeometry` / `LensNumerics` /
  `LensResources` triaged exclusions are deleted -- they resolve on their own
  now that the config objects ship.  API-claiming denominator **593 -> 597**,
  unresolved 0.

---

## Follow-up group (same work package, after the element sweeps landed)

### Fixed -- JAX asymptotic twin: two `where` fills silently made a real phase complex

`lumenairy/propagators/asymptotic_jax_twin.py`'s `_modal_field_lg00_pixel_jax`
filled two guarded `jnp.where` branches with the Python literal `0.0 + 0.0j`.
A Python complex is WEAKLY typed in JAX, so the fill promotes the whole `where`
to complex whenever the other branch is REAL -- and `phi_star` is real
(measured float64 on the x64 path).  Both fills are now dtype-matched
(`jnp.zeros((), x.dtype)`).

MEASURED under `jax.jit`, x64 enabled:

| operand | before (`0.0 + 0.0j`) | after (`zeros((), dtype)`) |
|---|---|---|
| `b_quad` complex64 / complex128 | unchanged | unchanged |
| `phi_star` float32 | **complex64** | **float32** |
| `phi_star` float64 | **complex128** | **float64** |

Values are preserved on every branch, and the end-to-end float64 output of
`propagate_modal_asymptotic_lg00_jax` is **bit-identical** (`np.array_equal`
True, `max|diff| = 0.0` on the Y3 9x9 fixture) -- so this is a dtype-contract
fix with no numerical change.

Corrects the characterisation in the finding as reported: the literal does not
force complex128 on a complex64 iterate (weak typing prevents that); it forces
real -> complex.  The site the existing pin caught (`safe_bquad`) was inert on
this build because `b_quad` is already complex; the site it MISSED (`safe_phi`)
was the live one.

### Fixed -- the P1-NEW-4 pin could not see a wrapped call

`tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` scanned line by
line with a regex requiring the `where` call and its fill on the SAME line, so

```python
safe_phi = jnp.where(jnp.isfinite(jnp.abs(phi_star)), phi_star,
                     0.0 + 0.0j)
```

was invisible to it -- and stayed invisible from v4.14 to v5.45, two lines
below a site the pin did catch, on nothing but where the line broke.

A structural (AST) pass now runs beside the regex and the two are UNIONED, so
the regex's behaviour is unchanged and the walk only adds sites it was blind
to.  It matches any `*.where(cond, value, <literal complex zero>)` regardless
of line breaks, treats `0j` / `0.0j` / `0.0 + 0.0j` as one rule via
`ast.literal_eval`, and -- being structural -- cannot match inside a comment or
a string at all.  Two stated exemptions: a NON-ZERO sentinel such as
`1.0 + 0.0j`, whose value is load-bearing and for which `zeros((), dtype)` is
not the migration; and a call whose value branch is itself a literal complex
constant, where the array is complex by construction and there is no operand
dtype to preserve.

`test_the_structural_walk_has_teeth` pins the behaviour on eight synthetic
sources, asserting each row's premise (what the regex sees) as well as its
verdict.  The file goes **239 -> 247 ids**.

It found one further site, `elements/doe.py:539`
(`T = np.where(inside, T, 0.0 + 0j)`), rated **P3 by measurement**: on that
branch `T` is complex128 unconditionally (the phase is built from Python float
literals), so nothing is promoted on any reachable call.  It is recorded in the
pin's own `_P3_ALLOWLIST` with that measurement and the one-line migration
written at the entry.

### Fixed -- architecture: the fourth lens import cycle is gone

`lumenairy/elements/_lens_thin.py` took `CUPY_AVAILABLE` and its CuPy helpers
from `.lenses`, which imports `_lens_thin` back.  It now takes them from
`lumenairy/backend/_optional.py`, a leaf, keeping the PEP 562 `cp` forward
pointed at the same shared lazy slot.  MEASURED with a module-level-only AST
walk over the lens family: **4 -> 3** module-level 2-cycles, and the six
thin-element entry points plus a complex64 case are bit-identical.  No public
API changed -- `lenses` still re-exports every name.

### Fixed -- the staggered PMM 2-D shared-grid advisory points at the caller

`PMM2DStackPure.solve`'s deferred advisory reached the warning through one
extra helper, so at the default `stacklevel=3` it reported at
`stack2d_pure.py`'s own line instead of at the user's `solve()` call.  It now
passes `stacklevel=4`.  The advice is about a ~1000x cost cliff, so a caller
who cannot see WHICH call is expensive cannot act on it.  Pinned by a new
`test_the_deferred_path_reports_at_the_callers_line` (that file: 6 -> 7 ids),
which fails on the pre-fix spelling.

### Changed -- three history blocks that were CODE, not comments

The WP-A17 element sweep could not take these, because changing any of them
moves both of a module's fingerprints:

* `elements/eme/eme_diffraction.py` -- the zero-norm refusal's user-facing
  message carried "this used to surface as an opaque 'SVD did not converge'
  LinAlgError".  It now states, in the present tense, what the guard prevents.
* `elements/pmm/stack.py` -- the per-layer `stabilize='slices'` refusal quoted
  and retracted its own earlier wording; it now states the corrected fact
  positively.  Every operative instruction is unchanged in both.
* `elements/pmm/_core.py` -- `_ARCHIVE_SLANT_FOLD`, a ~50-line module-level
  raw-string constant archiving a superseded slant treatment, was parsed and
  bound on every `import lumenairy` and read by nothing (verified: no code,
  test or `__all__` entry referenced it).  Its text moved verbatim into
  `docs/history/lumenairy.elements.pmm._core.md` and the two comments that
  named it now name the document.  `__all__` is byte-identical and exactly one
  module-level name was removed.

Each retired wording is reproduced verbatim in the module's history document
under its pre-relocation line number, and every affected module's fingerprints
were re-recorded in the same change with
`scripts/record_history_fingerprints.py --reason ...`.

### Fixed -- the history checker's falsifiability arm could run out of targets

`tests/unit/test_audit2609_a17_history_relocation.py`'s third mutation
re-spells an integer literal to prove the token fingerprint sees what the AST
folds away.  `lumenairy/_context.py` contains no integer constant at all
(measured), so the arm failed with "no small integer literal found to
re-spell" -- reporting the module as suspect when the catalogue had simply run
out of targets.  It now falls back to flipping a string's quote style,
skipping docstrings (which both fingerprints ignore by design) and any string
where the flip would not be value-preserving.  That file is now fully green:
**708 passed**.

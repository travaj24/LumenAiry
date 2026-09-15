# WP-B11c (Wave 5.3, part 1) -- the structural half of the WP-B11 hygiene items not reached

Handoff item 4.5, the structural half: the rest of the `rcwa/_core.py` split
(WP-B11 part a section 2.2's per-block hazard list), the two remaining lens
import cycles, and the `lenses <-> lenses_maslov` back-edge.  Part a section
2.3's whole-grid surface body is **deferred, deliberately**, and section 5
below says what it would change.

Branch `refactor/wp-b11c-structure`, base `96cb2096` (the Wave-5 plan commit).
Three items, three commits, three bit-identity censuses, plus this write-up.
Not pushed.

---

## How every claim here was gated

**Bit identity, archive-to-archive.**  `git archive 96cb2096 lumenairy`
extracted into a scratch tree.  Each probe runs TWICE in a CHILD process --
once with `cwd` and `PYTHONPATH` bound to that archive, once bound to the
working tree -- asserting `lumenairy.__file__` lives under the expected root
BEFORE it computes anything, and the two JSON digest maps are compared key by
key.  Never through pytest (pytest puts the repository root ahead of
`PYTHONPATH`, so both arms would import the same tree), and never against the
pip `-e` install, which points at a different checkout on a different branch.
Every digest is SHA-256 over the exact IEEE-754 bytes plus dtype and shape;
`probelib.caught` folds a call's exception type and message, and every warning
it emitted with its category, into the same digest, so a refactor that moves a
message or a warning's category is caught as loudly as one that moves a float.
Every python run carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`, and every census ran on BOTH builds (Windows py3.14 and the
WSL venv).

Harness: `validation/probe_wp_b11c/` -- `bi.py` (driver), `probelib.py` (bind +
hash), `probe_item1_rcwa.py`, `probe_item2_lenses.py`, plus the two measurement
tools `core_blocks.py` (the per-block dependency map of `rcwa/_core.py`) and
`import_graph.py` (the executed module-level import graph).  Hash JSONs are
`hashes_item<N>_{win,wsl}.json`.

**Import graph, measured twice.**  Structurally by the module-level-only AST
walk WP-B11 part a used (what the source SAYS), and dynamically by
`validation/probe_wp_b11c/import_graph.py`, an `__import__` hook in a child
process that keeps only calls whose frame is a MODULE BODY (`f_locals is
f_globals` -- true in no function, comprehension or class body).  The dynamic
instrument is not redundant: it is the only one that sees the edge a cycle is
actually made of.  A module-level `from .lenses import x` whose target is
already in `sys.modules` half-initialised executes no loader, so it leaves no
trace in `-X importtime` and none in `sys.modules` afterwards.  The recorder
reproduces the AST walk exactly on every tree it was run against -- two
two-cycles at this base, one after item 2, none after item 3 -- which is the
cross-check that it is reading the right thing rather than agreeing by
construction.

**The DECISION tests** are `tests/unit/test_audit2609_b11c_structure.py`, 16
ids, every one a structural fact with no build-dependent component: nothing in
the file reads a float, a mode count or a timing.  The whole file was run on an
isolated `git archive` of the base: **15 red, 1 green**, against 16 green here
(section 6 names each one and says why the sixteenth is green on both).

---

## 1. Item 1 -- the rest of the `rcwa/_core.py` split

### 1.1 What moved

`lumenairy/elements/rcwa/_blas.py` is a new leaf holding the whole opt-in
BLAS-thread cap, moved verbatim: the constants and state
`_BLAS_STATE`, `_BLAS_WARNED_UNCONTROLLABLE`, `_BLAS_CONTROLLER`,
`_BLAS_CONTROLLER_UNAVAILABLE`, `_BLAS_CONTROLLER_LOCK`, and the nine
functions `_get_blas_threads`, `_threadpoolctl_available`,
`_warn_blas_uncontrollable`, `set_blas_threads`, `rcwa_blas_threads`,
`_blas_threads_quiet`, `_get_blas_controller`, `_blas_limit`,
`_with_blas_limit`, plus the `blas_threads` knob registration.  It imports the
standard library and `lumenairy._knobs` (itself a stdlib-only leaf) and nothing
else from the package.

`rcwa/_core.py` drops **4967 -> 4764 lines** and loses three now-unused imports
(`contextlib`, `typing.Optional`, `_knobs.register_knob`).

Five of the module-level mutable names part a's deferral named are in that
block: `_BLAS_STATE` and `_BLAS_CONTROLLER` from its list, plus
`_BLAS_CONTROLLER_UNAVAILABLE`, `_BLAS_CONTROLLER_LOCK` and
`_BLAS_WARNED_UNCONTROLLABLE`.

### 1.2 The re-export table

| name | in `_core.__all__`? | re-exported into `_core`? | why |
|---|---|---|---|
| `_BLAS_STATE` | yes | **yes**, same object | a `threading.local()` OBJECT every reader mutates through and nothing rebinds, so one object seen through two paths is one piece of state |
| `_get_blas_threads` | yes | yes, same object | pure function |
| `set_blas_threads` | yes | yes, same object | public entry point |
| `rcwa_blas_threads` | yes | yes, same object | public context manager |
| `_blas_threads_quiet` | yes | yes, same object | read by `rcwa/stack.py`, `pmm/stack.py`, `pmm/stack2d.py` |
| `_blas_limit` | yes | yes, same object | read by `rcwa/oned.py`, `rcwa/twod.py`, `rcwa/stack.py`, both PMM stacks |
| `_with_blas_limit` | yes | yes, same object | the decorator on every public RCWA entry point |
| `_BLAS_CONTROLLER` | no | **no** | state, read through `_blas`'s globals |
| `_BLAS_CONTROLLER_UNAVAILABLE` | no | **no** | state, read through `_blas`'s globals |
| `_BLAS_CONTROLLER_LOCK` | no | **no** | state, read through `_blas`'s globals |
| `_BLAS_WARNED_UNCONTROLLABLE` | no | **no** | state, read through `_blas`'s globals |
| `_threadpoolctl_available` | no | **no** | read at call time by `_warn_blas_uncontrollable` |
| `_warn_blas_uncontrollable` | no | **no** | read at call time by `set_blas_threads` |
| `_get_blas_controller` | no | **no** | read at call time by `_blas_limit` |

Nothing was dropped from `__all__`: all 95 names still resolve on `_core`, and
the seven the cap contributes are the SAME objects on `_core`, on
`lumenairy.elements.rcwa` and on `_blas` (asserted by identity, not by name).

### 1.3 The monkeypatch proof

The item's instruction was to measure which site the code actually reads, not
to assume.  The measurement is `_defining_globals(fn) is vars(_blas)` for the
eight cap functions that read state (the ninth, `_with_blas_limit`, is a
decorator whose closure calls `_blas_limit`, and the hashed answer of a
decorated call covers it) -- a function's global namespace IS its defining module's
`__dict__`, so this is exact rather than a heuristic (the two
`@contextlib.contextmanager` functions are asked of `__wrapped__`, since the
decorator replaces the function with a helper defined in `contextlib`).  The
answer is the same for all eight: **the definition site**.  The four test files
therefore patch `rcwa._blas`, while still CALLING through `_core`'s re-export,
which is how a user reaches the cap.

| file | names patched | where it patches now | its own proof of patch |
|---|---|---|---|
| `tests/unit/test_audit_s5_8_perf_noloss.py` | `_BLAS_CONTROLLER`, `_BLAS_CONTROLLER_UNAVAILABLE` | `_blas` | `build_count` reads 1 (a fake controller that is never constructed reads 0); the unavailable arm asserts the latch flipped |
| `tests/unit/test_niche_audit_m4_m5_m6_rcwa.py` | `_threadpoolctl_available`, `_BLAS_WARNED_UNCONTROLLABLE` | `_blas` | ADDED: a counting stand-in on the one arm where the code calls it, plus `_assert_patch_site_is_live()` on all six |
| `tests/unit/test_pmm_m3_efficiency.py` | `_get_blas_controller` | `_blas` | `len(calls) == 1` (an unpatched run reads 0) |
| `tests/unit/test_v5_20_8_rcwa_threaded_sweep.py` | `_get_blas_controller` | `_blas` | `len(calls) == 1`, and `calls[0]['limits'] == 1` |

The gap that needed closing was in the M6 block.  Four of its six arms assert
that NO warning is emitted, and this box HAS `threadpoolctl` -- so if the patch
stopped landing, all four would stay green while measuring the unpatched code.
`_assert_patch_site_is_live()` states, unconditionally, that
`_warn_blas_uncontrollable.__globals__ is vars(_rblas)` and that
`_core.set_blas_threads is _blas.set_blas_threads`; the one arm whose code path
does call the stand-in (`test_m6_no_warning_when_a_controller_is_available`)
now counts the calls and asserts the count is non-zero.

**Why no PEP 562 forward here, deliberately.**  A forward for the state names
would make a stale `setattr(_core, '_BLAS_CONTROLLER', None)` SUCCEED -- binding
a shadow attribute in `_core.__dict__` that nothing reads, leaving the test
green and the code unpatched.  Absent from `_core`, the same line raises
`AttributeError` at the patch site.  A loud failure is the correct behaviour for
a patch target that moved; `test_the_state_names_are_not_re_exported_so_a_stale_patch_fails_loudly`
pins it, and `test_the_four_monkeypatching_files_patch_the_definition_site` is
a fail-closed inventory over the whole `tests/unit` directory, so a fifth file
that starts patching one of these names on `_core` is named by id and line.

### 1.4 The per-block hazard analysis for what did NOT move

Part a named four blocks that "could follow": `_blas.py`; the Wood-anomaly /
grazing guards; the Redheffer algebra; the tensor convolutions.  Measured with
`validation/probe_wp_b11c/core_blocks.py`, which reports for each
banner-delimited section the module-level names it defines, the names defined
elsewhere in `rcwa/_core.py` that it reads, and who reads it:

| block | lines | reads from elsewhere in `rcwa/_core.py` | verdict |
|---|---|---|---|
| BLAS controls | 223 | **none** | **moved** |
| JAX gauge-stable eig (`_JAX_EIG_STABLE`) | 157 | **none** | leaf by dependency, but see below -- **not moved** |
| conditioning guard (`_INV_CENSUS`, `INTERFACE_CONDITIONING_GUARD`) | 675 | `_EnergyError`, `_EnergyWarning` (both read ONLY from inside this block, so moving the two exception classes with it closes it) | **the largest remaining leaf** -- not moved, see below |
| branch cuts / flux | 347 | `_C` | not a leaf |
| Wood-anomaly / grazing guards | 788 | `Efficiency2D`, `_MAX_HARMONICS` | not a leaf, and worse -- see below |
| Fourier factorization | 35 | `_C` | not a leaf |
| even-parity symmetry | 635 | 12 names | not a leaf |
| Redheffer algebra | 295 | `_C`, `_concrete`, `_guarded_inverse` | not a leaf (depends on the conditioning block) |
| generalized S-matrix | 290 | `_INV_T22_RCOND_REFUSE`, `_block`, `_guarded_inverse` | not a leaf |
| tensor convolutions / 1-D anisotropic | 940 | 10 names | not a leaf |
| homogeneous-mode cache (`_HOMOG_CACHE`, `_HOMOG_LOCK`) | 180 | `_homogeneous_eigenmodes` | not a leaf -- the cached function sits inside the 635-line symmetry block |

So, measured: **of part a's four named candidate blocks only the BLAS one is a
leaf.**  Three specific hazards are worth recording because they are not
visible from the deferral text:

1. **The Wood / grazing block would re-break the W9 overlap test.**
   `_validate_geometry`, `_validate_cell_sampling` and `_validate_shapes` live
   inside it.  Part a deliberately kept `_validate_shapes` in `rcwa/_core.py`
   because `tests/unit/test_niche_audit_w9_overlap_exact.py` substitutes a
   counting wrapper at `_core._shapes_overlap` to prove the exact predicate is
   reached, and a `_validate_shapes` living elsewhere resolves that name in ITS
   globals.  Moving the section wholesale undoes that; moving it minus the
   validators means splitting an interleaved section.

2. **The JAX eig block is a clean leaf by dependency and still must not move
   as a plain re-export.**  `_JAX_EIG_STABLE` is BOTH in `_core.__all__` and
   monkeypatched (`tests/unit/test_niche_audit_w9_eig_vjp.py`).  Unlike
   `_BLAS_STATE` it is REBOUND, not mutated in place, so a re-export would be a
   stale duplicate -- and unlike the cap's state it cannot simply be left off
   `_core`, because `__all__` is an import-path contract.  A read-only PEP 562
   forward does not help: module attribute lookup can be intercepted, but a
   `setattr` cannot be, without giving the module a type (which is what item 2
   had to do for `lenses`).  Doing that to `rcwa/_core.py` for one cached
   builder buys 157 lines and adds a mechanism; it is recorded here rather than
   done.

3. **The conditioning guard is the biggest genuine leaf left** -- 675 lines,
   closed once `_EnergyError` and `_EnergyWarning` travel with it, and it
   carries the most-patched mutable name in the file (`_INV_CENSUS`, patched
   from seven test files).  Its patch surface splits three ways, which is the
   analysis a future cut needs: `_INV_CENSUS` and
   `INTERFACE_CONDITIONING_GUARD` are read by `_guarded_inverse` and would have
   to be patched at the new module; `_guarded_inverse` itself is CALLED by
   `_interface_smatrix`, `_interface_smatrix_general`, `_redheffer_star` and
   `_redheffer_star_rt`, which after a re-export would still resolve it in
   `rcwa/_core.py`'s globals -- so that one name would have to keep being
   patched on `_core`.  Seven test files with roughly forty patch sites is more
   than this item's brief (the handoff names "the four monkeypatching test
   files", which is exactly the BLAS set), so it is written up rather than done.

### 1.5 Bit identity

`validation/probe_wp_b11c/probe_item1_rcwa.py`, **43 keys, 43/43 identical on
both builds** (`hashes_item1_win.json`, `hashes_item1_wsl.json`):

* 13 keys on the re-exported surface itself: `__all__` as an ordered list and
  as a set, every name resolving, identity through `_core` / the package facade
  / `lumenairy`, the `blas_threads` knob round trip, the scoped and quiet
  context managers restoring the prior request, `functools.wraps` metadata on
  the decorator, and the `max(1, int(n))` floor over `n in (0, -3, 2.9, 7)`;
* 23 keys on the RCWA engines that run under the decorator: 1-D TE / TM /
  oblique / metallic-TM-with-Li / laurent, 1-D Jones by both spellings, 2-D
  crossed at normal and conical incidence, 2-D analytic shapes, 2-D Jones on a
  full tensor cell, a wavelength sweep, `RCWAStack.solve` with its efficiencies,
  both Jones matrices and the absorptance, `solve_vs_wavelength` serial AND
  threaded (plus their equality), the same 1-D solve and stack solve INSIDE an
  explicit `rcwa_blas_threads(1)` scope and inside `_blas_threads_quiet(2)` --
  the only arithmetic a BLAS cap can reach -- and the uncapped solve again
  afterwards, and `rcwa_extrapolate`;
* 7 keys on the PMM consumers of `_blas_threads_quiet` / `_blas_limit`:
  `pmm_efficiency_1d` TE and TM, `pmm_jones_1d` on a tensor grating,
  `PMMStack.solve_vs_wavelength` serial and threaded,
  `PMM2DStackHybrid.solve_vs_wavelength`, and `pmm_efficiency_2d_cell`.

---

## 2. Item 2 -- the `_lens_real <-> lenses` cycle

### 2.1 What moved

Into `lumenairy/elements/_lens_kernels.py`, verbatim: the optional-backend
plumbing (`CUPY_AVAILABLE`, `cp`, `_ensure_cupy_loaded`, `_is_cupy_array`;
`NUMEXPR_AVAILABLE`, `_ne`, `_ensure_numexpr_loaded`; `_NUMBA_AVAILABLE`,
`_numba`, `_njit`, `_prange`, `_NUMBA_KERNELS`, `_load_numba`,
`_get_aspheric_sag_accum_numba`) and the two surface-sag builders
`surface_sag_general` (with its `_surface_sag_general` alias) and
`surface_sag_biconic`.  `lumenairy/elements/lenses.py` drops **746 -> 368
lines** at this item (338 after item 3); `_lens_real` reads the leaf.

The sag builders could not move alone: they read the plumbing from module
scope.  That is why the plumbing came with them, and it is why the leaf's own
"imports nothing from `lumenairy`" wording had to change (section 2.3).

### 2.2 The two-way forward, and why PEP 562 alone was not enough

`docs/lens_configuration.md` recorded, before this item, that "the mechanical
answer to both is a PEP 562 `__getattr__` on `lenses.py` ... so
`lenses._NUMBA_AVAILABLE = False` reaches the leaf and `lenses.cp` stays live."

**Measured: that is the answer to the second half only.**  PEP 562's
`__getattr__` is consulted on a failed attribute LOOKUP.  It has no say over
`setattr`, and the monkeypatch in question is a write.  With `__getattr__`
alone, `monkeypatch.setattr(lenses, '_NUMBA_AVAILABLE', False)` writes into
`lenses.__dict__`, the leaf's global is untouched, `_load_numba()` returns
`True`, the test goes green while exercising the numba arm -- and monkeypatch's
undo restores the value it read through the forward, leaving the shadow
attribute in `lenses.__dict__` for every later test in the process.

`lenses.py` therefore installs a module TYPE.  `_LIVE_FORWARD_NAMES` is the
eight-name whitelist (`cp`, `_ne`, `NUMEXPR_AVAILABLE`, `_NUMBA_AVAILABLE`,
`_numba`, `_njit`, `_prange`, `_NUMBA_KERNELS`; the last is mutated in place
rather than rebound, so it does not strictly need the write half, but it is
forwarded anyway so `dir(lenses)` and the whitelist agree);
`_LensesFacade(ModuleType)` defines `__getattr__`, `__setattr__`, `__delattr__`
and `__dir__`, each forwarding those names to `_lens_kernels`; and
`sys.modules[__name__].__class__ =
_LensesFacade`, which is the documented way to give a module a full attribute
protocol (PEP 562 names it as the alternative it does not replace).  The facade
keeps no copy of its own -- `test_the_facade_still_surfaces_the_moved_gates_live`
asserts `name not in vars(lenses)` for all eight whitelisted names, which is
what makes the read live rather than accidentally-equal -- and a name outside the whitelist
still raises `AttributeError`, so typos and `hasattr` probes behave normally.

`_lens_thin.py` keeps its own one-name `__getattr__` for `cp`: it only ever
reads.

### 2.3 The re-export table, and three ratchets that legitimately relocate

Every moved name is re-exported from `lenses` with the `X as X` spelling the
module uses for deliberate re-exports, so
`from lumenairy.elements.lenses import surface_sag_general` resolves unchanged
and to the SAME object.  `lumenairy/elements/elements.py` (`_surface_sag_general`),
`lumenairy/elements/freeform.py` (both builders) and every test that imports
either from `lenses` were left untouched and are covered by the census.

Three existing gates name the module that OWNS a property, and that module
changed.  None was relaxed:

1. `tests/unit/test_audit2609_b11_hygiene.py::TestTheLensKernelsLeaf` asserted
   the leaf imports nothing from `lumenairy` at all.  It now imports
   `lumenairy.backend._optional` -- the ONE place the library probes and lazily
   loads CuPy and numba since WP-A16, where before there were five hand-copied
   pairs.  Re-inlining a sixth copy to keep the old wording would undo A16, so
   the test is restated as the property that was always the point and is
   strictly stronger: it walks the transitive closure of the leaf's `lumenairy`
   imports and asserts **nothing reachable from it, at any depth, imports
   `lumenairy.elements`**.  (The old check would have passed a one-hop detour
   through any other subpackage.)  Measured closure:
   `{lumenairy.backend._optional}`, itself stdlib-only.

2. `tests/unit/test_audit2609_a16_lens_arch.py`'s `_CUPY_CONSUMERS` /
   `_NUMBA_CONSUMERS` and the twin parametrisation in
   `tests/unit/test_audit2609_a16_verify_config_and_arch.py` named `lenses` as
   one of the modules that must keep its own `cp` alias and numba gate.  That
   copy is now the leaf's, so the tuples name `_lens_kernels`.  Leaving
   `lenses` in them would have been worse than a failure: the verify test
   substitutes `_optional_is_cupy_array` and `_ensure_cupy`, which are NOT in
   the forward whitelist, so on the facade it would have bound attributes the
   function never reads and gone green against the unpatched code.  What
   `lenses` owes instead is a NEW test,
   `test_the_facade_still_surfaces_the_moved_gates_live`, which covers the read
   AND the write AND the absence of a shadow copy -- strictly more than the old
   parametrisation asserted.

3. `test_audit2609_b11_hygiene.py`'s family-cycle ratchet was an upper bound
   (`cycles <= {the two pairs}`).  It is now an EQUALITY, so a cycle that
   closes forces the line to be revisited in the same commit: one open after
   item 2, zero after item 3.

### 2.4 Bit identity

`validation/probe_wp_b11c/probe_item2_lenses.py`, **40 keys, 40/40 identical on
both builds** (`hashes_item2_win.json`, `hashes_item2_wsl.json`).  Everything is
addressed through the STABLE paths (`lumenairy`, `lenses`, `lenses_maslov`),
because the probe has to run unchanged on both trees and because "a caller's
spelling did not change, so its answer must not either" is the honest question
to ask of a re-export.  The structural claims live in the DECISION tests, not
in a hash.

* 12 sag keys: `surface_sag_general` over conic / conic+aspheric / flat /
  negative-R / hyperbolic / beyond-aperture (its refusal) / float32 input, and
  `surface_sag_biconic` symmetric, asymmetric, aspheric, per-axis-aspheric and
  flat-in-y;
* 5 keys on the numba gate as a LIVE property: the fast-path answer, the gate
  flipped to `False` THROUGH the facade, the pure-NumPy answer under it (equal
  to the fast path, bit for bit, which is what proves the flip reached the
  kernel), `_load_numba()` returning `False`, and the gate read back and
  restored;
* 5 keys on the facade surface: the numexpr loader, the `AttributeError` for a
  name nobody defines, the gate values, and which names resolve on `lenses` and
  on `lumenairy`;
* 18 consumer keys: `apply_thin_lens`, `apply_spherical_lens`,
  `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply_real_lens` plain /
  slant+fresnel / on an aspheric singlet / in complex64,
  `apply_real_lens_maslov` on two prescriptions, `apply_real_lens_traced`,
  `apply_real_lens_gbd`, `surface_sag_xy_polynomial`, `Surface` sag rotational
  and biconic, and the three grid-versus-aperture helpers with every warning
  they emit.

---

## 3. Item 3 -- the `lenses <-> lenses_maslov` back-edge

### 3.1 What moved

The four names the back-edge still carried after WP-B11 part b took it from
five to four.  Two of them -- `NUMEXPR_AVAILABLE` and
`_ensure_numexpr_loaded` -- reached the leaf with item 2, as part of the sag
builders' plumbing; until this item `lenses_maslov` still read all four through
the facade, which is what kept the edge alive.  This item moves the remaining
two, `_fit_normaliser` and `_multi_indices_total_degree`, and repoints
`lenses_maslov`'s module-scope import at the leaf.

Part a recommended "`_lens_kernels` or a new leaf"; `_lens_kernels` is the
right home, because two of the four were already there and because the
`lenses` re-export keeps every existing spelling working either way.

### 3.2 The re-export table

| name | defined in | re-exported by `lenses` | read by `lenses_maslov` from |
|---|---|---|---|
| `NUMEXPR_AVAILABLE` | `_lens_kernels` | yes (LIVE forward -- it is a gate) | `_lens_kernels` |
| `_ensure_numexpr_loaded` | `_lens_kernels` | yes (`X as X`) | `_lens_kernels` |
| `_fit_normaliser` | `_lens_kernels` | yes (`X as X`) | `_lens_kernels` |
| `_multi_indices_total_degree` | `_lens_kernels` | yes (`X as X`) | `_lens_kernels` |

`NUMEXPR_AVAILABLE` is imported BY VALUE in `lenses_maslov` on purpose: that
module reads it only to decide whether to call `_ensure_numexpr_loaded()`, and
the loader re-reads the leaf's live flag, so a flipped gate is honoured where it
matters.  The tests that flip it set it on `_lens_kernels`, or on `lenses`,
which forwards the write.

Six existing test files import `_multi_indices_total_degree` or
`_fit_normaliser` from `lumenairy.elements.lenses`
(`test_audit2609_a4_asymptotic.py`, `test_audit2609_a4_maslov_gbd.py`,
`test_audit2609_a4_verify_maslov_asymptotic.py`,
`test_audit2609_b1_maslov_input_wavevector.py`,
`test_niche_audit_w6_asymptotic.py`, `test_v5_21_lens_accuracy_extensions.py`).
None was edited: the re-export is the point.

### 3.3 The graph, before and after

Measured by both instruments, on `import lumenairy`:

| | at `96cb2096` | after item 2 | after item 3 |
|---|---|---|---|
| module-level 2-cycles in the lens family | `('_lens_real','lenses')`, `('lenses','lenses_maslov')` | `('lenses','lenses_maslov')` | **none** |
| `_lens_real ->` | `_lens_kernels`, `lens_config`, `lenses` | `_lens_kernels`, `lens_config` | `_lens_kernels`, `lens_config` |
| `lenses_maslov ->` | `_lens_kernels`, `_lens_real`, `lens_config`, `lenses` | `_lens_kernels`, `_lens_real`, `lens_config`, `lenses` | `_lens_kernels`, `_lens_real`, `lens_config` |
| `_lens_kernels ->` | (nothing in the family) | (nothing in the family) | (nothing in the family) |
| `rcwa._core ->` | `_geometry` | `_blas`, `_geometry` | `_blas`, `_geometry` |
| `rcwa._blas ->` | (module does not exist) | (nothing in the package) | (nothing in the package) |

The columns are CUMULATIVE, so the two `rcwa` rows change in the "after item 2"
column only because item 1 landed before it; nothing in items 2 or 3 touches the
rcwa package.

The AST walk and the `__import__` recorder agree on every row.  The family's
cycle count is now **zero**, closing the arc the audit's TESTS-ARCH section
opened at four: `_lens_thin <-> lenses` and `_lens_traced <-> lenses` in part a,
these two here.

### 3.4 Bit identity

The full `probe_item2_lenses.py`, **48 keys, 48/48 identical on both builds**
(`hashes_item23_win.json`, `hashes_item23_wsl.json`) -- item 2's 40 plus six,
plus the two added while running down the one order-dependent red of section 4
(the vector Maslov wrapper's `P_x/P_y` under all three normalisations, and its
raw field, on WP-A4-VERIFY's own fixture).  The six are:
`_multi_indices_total_degree` over a (n_vars, order) grid of 20 combinations;
`_fit_normaliser` on five inputs, three of them degenerate (all-zero, constant,
and a pair straddling zero at 1e-9), and over three `pad` values; the
`lenses_maslov` spelling of all four moved names answering identically to the
`lenses` one; and `dir(lenses)` still listing the eight forwarded names.  That
last one is the reason the facade's module type
also defines `__dir__`: a module's default lists `__dict__` only, so without it
the forwarded names would have VANISHED from introspection -- an observable
surface change in a refactor whose contract is that nothing observable moves.

---

## 4. Test tails

All runs carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
on the command line and `--capture=sys` (handoff 4.1b), one pytest process at a
time, with `PYTHONPATH` pinned to this worktree.  "win" is Windows py3.14,
"wsl" the WSL venv.

| run | build | tail |
|---|---|---|
| item 1: `test_rcwa` + the four m4/m5/m6, w3, w7, w9-overlap, m1-conditioning, b5, b11, b11c files | win | `9 failed, 399 passed, 2 skipped` -- the nine are the items 2 and 3 DECISION tests, red by design at that commit (section 6) |
| item 1: `test_pmm_m3_efficiency.py` | win | `46 passed` |
| item 1: b11c + s5_8 + m4/m5/m6 + v5_20_8 + `test_rcwa` | wsl | `135 passed, 1 skipped` |
| census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget` + the `__all__`-symmetry walker (11 files) | win | `1 failed, 899 passed, 5 skipped` -- the one is the pre-existing order-dependent `test_public_api.py::test_installed_metadata_version_matches_source_version` (below) |
| item 2: the lens consumers (17 files: `test_lens*`, a2 analytic + displaced + verify, w3 elements, w4 gaps, v4_15 agent D x2, glass, s4_3 biconic, wave5-delegated, the PEP-562 and shell-vs-canonical walkers, a15b lazy + optional) | win | `616 passed` |
| items 2 + 3: `test_audit2609_a16_lens_arch` + `a16_verify_config_and_arch` + `b11_hygiene` + `b11c_structure` | win | `155 passed` |
| item 3: `b11c_structure` + `b11_hygiene` | win | `99 passed` |
| the FINAL sweep on the completed tree: census / walker / dispatcher-pin / public-API / doc-consistency + `test_audit_except_budget` + the `__all__`-symmetry walker + the history-fingerprint tool (12 files) | win | **`911 passed, 5 skipped`** -- and `test_public_api` green here, which is the other half of the evidence that its earlier red was an ORDER effect |
| the CHANGELOG walkers on the written block (V18 self-citation, V12 content + changeset, V17 self-citation, doc consistency) | win | `22 passed, 6 skipped` -- the six are the clean "nothing to verify" skips the handoff records for the 5.47.0 block |
| item 3: the Maslov / asymptotic slice (10 files: a4 asymptotic + maslov-gbd + verify, b1 input-wavevector, w6 asymptotic, v5_21 lens-accuracy + gbd-maslov-perf + maslov-jax-caustic, eh1 upsample, mhs resampling) | win | run 1 `1 failed, 245 passed, 2 deselected`; **run 2 `246 passed`**; the SAME ten files on an isolated `git archive` of the base, same command: `246 passed`.  The one red is a 4-ULP bar with 1 ULP of margin, dissected below |
| the same slice's verify file among a 10-file lens/Maslov set (b11c, b11, a16 x2, chunked sag, a2 analytic, w4 gaps, the two walkers, a4 verify) | wsl | `440 passed` |
| the PMM-2D slice from the RUNS list (12 `test_pmm2d*` files + the M2 window contract + per-layer grids) | win | `366 passed` -- including `test_pmm_m2_window_contract.py`, whose T3-1 classification the handoff lists as a known red; it is green on this box today |

**The second red: a bar with one ULP of margin, and the measurement that settles
whose it is.**
`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py::test_s10_vector_normalisation_is_one_joint_scale_for_the_pair`
failed once in the ten-file slice.

It did not reproduce -- but that is deliberately NOT the argument here.
"It passed on the re-run" is rerun-to-green, and the house rule is that a flake
is bad math until measured.  The re-runs are recorded as corroboration; the
argument is the measurement:

* The bar is `abs(ratios[mode] - r0) <= 4 * np.spacing(r0)` -- FOUR ULP on
  `P_x/P_y` -- and its docstring records the measurement it was set from
  ("0 ULP for 'power', 1 ULP for 'peak'", WP-A4-VERIFY, 2026-09-13).
* Re-measured in a clean process on this box TODAY, on this branch: **3 ULP for
  'power', 1 ULP for 'peak'**.  One ULP of margin on a 4-ULP bar is exactly the
  S4 shape `docs/TESTING_STANDARDS.md` names -- a bar whose pass/fail boundary
  sits inside the spread of the quantity it reads.
* Loading `jax` and setting `jax_enable_x64` in the same process (what the two
  files that run before it do, without restoring) does NOT move it: still 3 and
  1.  So the trigger is somewhere else in the run's history and was not traced
  further.
* The same clean-process measurement on the BASE tree, extracted independently,
  reads the same two numbers: **3 and 1**.  Base and branch are not merely
  close; they are the same values.
* **The decisive measurement is that the quantity itself does not move.**
  `E13_maslov_vector_ratios` and `E14_maslov_vector_field` were added to the
  probe for this: `apply_real_lens_maslov_vector` on WP-A4-VERIFY's own
  fixture, all three `normalize_output` modes, the two powers and the raw
  field, hashed archive-to-archive.  **Bit-identical on both builds.**  If
  `r0`, `ratios['power']` and `ratios['peak']` are the same bytes on the base
  tree and on this branch, then the ULP deltas the bar reads are the same on
  both, and the test fails or passes identically on both for any given process
  history.  The red is therefore not this package's, by measurement rather than
  by argument.

The run tallies corroborate that and are recorded for completeness, not relied
on: run 1 on the branch `1 failed / 245 passed` (on a box carrying 18 concurrent
python processes, several of them this package's own probes), run 2 on the
branch `246 passed`, the identical ten files on an isolated `git archive` of the
base `246 passed`, and the same verify file inside a 440-test WSL run green.
A bar with one ULP of margin behaves exactly like this; the one thing that would
be evidence -- the quantity moving -- is measured and does not.

Recorded, not masked, and NOT re-derived here: re-deriving that bar is
`test_audit2609_a4_verify_maslov_asymptotic.py`'s owner's work, and it needs
the two-sided envelope TESTING_STANDARDS asks for, measured across the run
orders that move it.

**The first red, and why it is not this package's.**
`tests/unit/test_public_api.py::test_installed_metadata_version_matches_source_version`
fails in multi-file runs with `installed == '3.7.8'` against
`la.__version__ == '5.47.0'`, and PASSES on its own (`1 passed in 0.29s`); read
directly, `importlib.metadata.version('lumenairy')` on this interpreter returns
`5.47.0` from
`.../Python314/Lib/site-packages/lumenairy-5.47.0.dist-info`.  So it is an
ORDER effect: something earlier in the run leaves a stale distribution finder
visible.  `tests/unit/test_public_api.py` is not in this branch's diff
(`git diff --stat 96cb2096..HEAD` lists 31 files, and it is not one of them),
the same id fails the same way in this session's older run logs from other
trees, and it is the same shape as the handoff's known order-dependent
glass-validity one-shot pin.  Recorded, not masked; NOT root-caused here
because it is outside this package's ownership.

---

## 5. Item 4 -- the whole-grid surface body, DEFERRED and why

Part a section 2.3 already recorded that this is not a bit-identical refactor,
and this item confirms it rather than attempting it.  The brief's shape is "the
three surface bodies in `lumenairy/elements/_lens_real.py` -- the two banded
ones and the WHOLE-GRID one -- collapsed into one `for band in bands(...)`".
Part a shared what IS shareable: the integer schedule, now `_row_bands` and
`_band_in_halo`, read by all four sites, gated at 56/56 bit-identical hashes.

Expressing the whole-grid pass as a one-band iteration would move three things,
each of which moves numbers:

1. **The numexpr gate.**  The whole-grid path takes its
   fused-multiply decision on `E.size`, the size of the FULL field.  A band
   loop asks it per band, so a grid that crosses the threshold as a whole but
   not per band (or the reverse) changes which of the two evaluation orders
   runs, and numexpr's chunked multi-threaded reduction is not the same
   floating-point expression as NumPy's.
2. **`_ensure_full_grids`.**  The whole-grid path allocates the full coordinate
   grids; the banded paths deliberately never reach that allocation, working
   from per-band slices instead.  Folding them together either pays that
   allocation on the banded paths (a memory regression the banding exists to
   avoid) or removes it from the whole-grid path (which changes the arrays the
   arithmetic reads).
3. **The Fresnel dtype promotion.**  It happens at a different pipeline step in
   the two paths; unifying them promotes earlier or later, and a promotion that
   moves changes the last bits of everything downstream of it.

A refactor that moves a bit is a defect unless the report says why it must, and
here there is no "must": what remains after part a is a set of
`if <banded> / else` branches around the schedule part a already shared, not
three copies of the arithmetic.  The correct shape for this
work is a measured behaviour change with a Migration note (pick ONE numexpr
threshold policy, ONE allocation policy and ONE promotion point, and re-pin the
fixtures that move), which is a different kind of work package from a hygiene
pass.  Recommended for a wave that is already moving lens defaults.

---

## 6. Fail-before

The whole DECISION file was run on the pre-refactor tree, in an ISOLATED
`git archive 96cb2096` checkout (not the working tree, which other agents write
in), with only two files copied in: the final
`tests/unit/test_audit2609_b11c_structure.py` and the recorder
`validation/probe_wp_b11c/import_graph.py` it calls.  Command:

```
git archive 96cb2096 | tar -x -C <scratch>
cp tests/unit/test_audit2609_b11c_structure.py <scratch>/tests/unit/
cp validation/probe_wp_b11c/import_graph.py <scratch>/validation/probe_wp_b11c/
cd <scratch> && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   PYTHONPATH=<scratch> python -m pytest tests/unit/test_audit2609_b11c_structure.py   -q --capture=sys -p no:randomly
```

**`15 failed, 1 passed in 9.63s`** -- against `16 passed` on this branch.

```
FAILED test_the_blas_cap_is_a_leaf_module
FAILED test_every_promised_core_name_still_resolves_and_is_the_same_object
FAILED test_the_cap_state_is_read_out_of_the_leaf_so_the_patch_site_is_the_leaf
FAILED test_the_state_names_are_not_re_exported_so_a_stale_patch_fails_loudly
FAILED test_the_four_monkeypatching_files_patch_the_definition_site
FAILED test_the_executed_rcwa_graph_has_the_leaf_edge_and_no_back_edge
FAILED test_the_lens_family_carries_only_the_cycles_still_open
FAILED test_lens_real_does_not_import_the_hub_at_module_scope
FAILED test_the_executed_lens_graph_agrees_with_the_source
FAILED test_the_numba_gate_monkeypatch_still_reaches_the_kernel
FAILED test_the_lazy_backend_slots_forward_live_and_not_a_stale_none
FAILED test_lenses_maslov_does_not_import_the_hub_at_module_scope
FAILED test_the_executed_graph_carries_no_maslov_back_edge
FAILED test_the_four_moved_names_resolve_from_both_ends_and_are_identical
FAILED test_the_numexpr_gate_is_live_through_the_facade_too
```

The one that passes on BOTH trees is
`test_the_module_getattr_still_raises_for_a_name_nobody_defines`, and it is
declared as such rather than counted as evidence: it is the guard on the
forward (a forward that answers everything hides typos and breaks `hasattr`
probes), and a module with no `__getattr__` at all raises `AttributeError` too.
Its job is to stay green after the forward exists.

Two of the reds are worth naming because they are the hazard, not the plumbing.
`test_the_four_monkeypatching_files_patch_the_definition_site` is red on the
base because the four files patch `rcwa/_core.py`'s module there -- which is
correct there and would be a silent no-op here.
`test_the_numba_gate_monkeypatch_still_reaches_the_kernel` is red because the
leaf has no gate to reach; the version of this test that would be red on the
*intermediate* tree -- a `lenses` with a read-only PEP 562 forward -- is the one
that produced the measurement in section 2.2.

## 7. Runs

### The gates

| gate | result |
|---|---|
| `ruff check lumenairy/ tests/` (WSL) | **All checks passed** after each item |
| `scripts/check_source_line_citations.py` (V18) | drift **0**, 107 of 107 -- after re-anchoring 28 citations, see below |
| `scripts/verify_changelog_closures.py` (V16) | exit 2, the clean skip (no audit-closure bullets, no tag to diff against) |
| `scripts/record_history_fingerprints.py --check` | OK; `lumenairy.elements._lens_real`, `lumenairy.elements.lenses` and `lumenairy.elements.lenses_maslov` re-recorded in the SAME commits that moved them, with the reason.  `rcwa/_core.py` carries no history document, so item 1 moved none |
| `scripts/check_doc_identifiers.py` | OK -- 621 API-claiming backticked identifiers, 621 resolve, 0 do not |
| `test_audit_except_budget.py` | in the census sweep, green |
| the `__all__`-symmetry walker (`test_v4_16_0_walker_all_symmetry.py`) | green |
| the dispatcher pins (cache locks, cache-registry enrollment, doc consistency) | green -- `_BLAS_CONTROLLER_LOCK` stays on the cache-lock exemption list and is still found there after the move |
| the PEP-562 forwarding walker (`test_v5_2_walker_pep562_forwarding.py`) | green -- it walks `propagators/fft_infra.py` against `propagators/propagation.py`'s whitelist and is unaffected by this package; `lenses._LIVE_FORWARD_NAMES` is deliberately spelled the same way so a future generalisation of that walker finds it |
| `.test_durations` | topped up with `--store-durations` on each landing and validated as JSON: 16 186 -> 16 207 entries, all 16 `b11c` ids present, and the four tests renamed by this package re-stored under their new ids |
| `durations_gap.py` | 7 tracked ids without an entry -- 4 in `test_v5_14_5_viewer_polarization.py`, 2 in the cache-locks pin, 1 in `test_niche_k3_perf.py`.  The handoff recorded 3 at the 5.47.0 close; the 4 that appeared are in a file this package never touched and are not in its diff.  Not root-caused here |

### The citation re-anchor

Moving code moves line numbers, and `CHANGELOG.md` cites source lines.  Four of
the files this package touched carry citations in the 5.47.0 block, and all of
them shifted: `_lens_real.py` by -1 (one import statement fewer),
`lenses.py` by -400, `lenses_maslov.py` by +8, `rcwa/_core.py` by +12 above the
split and -203 below it.

**28 citations were re-anchored by CONTENT, not by arithmetic.**
`validation/probe_wp_b11c/reanchor_citations.py`'s method: take the number
the citation had at `96cb2096`,
read the line it named in that commit's version of the file, find that exact
line in the tree as it is now, and break ties with a +-2-line context check.
Every one resolved uniquely except one, which the context check settled 4/4.
One citation followed its line out of the file it named: the inert-cap
warning's "REQUIRES ``threadpoolctl``" paragraph, cited as `rcwa/_core.py` line
249 (as it then was), now lives in `rcwa/_blas.py`, and the tool rewrote the
PATH as well as the number.

This matters beyond tidiness.  The V18 walker refuses a citation that lands on
a TRIVIAL line (blank, a lone brace, a docstring delimiter), so it caught two
of the twelve `_lens_real.py` shifts and would have passed the other ten while
they pointed one line off -- the "right conclusion, wrong numbers" shape
`docs/TESTING_STANDARDS.md` calls the most dangerous, because it reads as
authoritative and it passes.  Content-matching is what finds those.  V18 reads
drift 0 afterwards.

The tool is idempotent, and it had to be made so on purpose: the first cut read
the numbers out of the WORKING CHANGELOG, so a second run treated the numbers it
had just written as base numbers and shifted everything again.  It now sources
its "before" numbers from `git show <base>:CHANGELOG.md`, so a second run
recomputes the same mapping and finds nothing left to replace -- `0
re-anchored`, which doubles as the check that the first run converged.  It takes
`--check` to report without writing.

### What could not be measured

* **CuPy.**  Not installed on this box, so the GPU arms of `_is_cupy_array` /
  `_ensure_cupy_loaded` are exercised only through a substituted module
  (`test_a_true_cupy_answer_really_binds_the_module_cp`, which fakes a True
  answer and checks the alias binds) and not against a real device.  That was
  already true before this package; the move does not change what is reachable.
* **The pre-refactor `_JAX_EIG_STABLE` hazard, end to end.**  Section 1.4 item 2
  argues from the language (a module `__getattr__` cannot intercept a
  `setattr`) plus the measurement made for `lenses` in item 2, where the same
  shape WAS built and its failure mode observed.  It was not separately
  demonstrated on `rcwa/_core.py`, because doing so would mean building the
  mechanism the item declines to build.
* **The full two-lane release gate.**  Out of scope for a package branch; the
  slices above are the RUNS the brief names, plus every file that imports a
  moved name.  Whoever merges this wave should run the two-lane gate on the
  merge, as handoff section 6 step 7 describes.
* **Whether the order effect behind the `test_public_api` red is the same one
  the handoff's glass-validity pin sees.**  Both are order-dependent and both
  pass alone; no shared cause was traced, and tracing it is outside this
  package's files.

# CHANGELOG text -- WP-A17 SWEEP-3 (version-history relocation, library-wide)

Assemble into the release block in the repository's CHANGELOG voice.  Nothing
here changes behaviour: every module in this sweep has a byte-identical
docstring-free AST and comment-free token stream before and after.

---

### Changed -- source comments: version history moved to `docs/history/`

- **`lumenairy/*.py` (except `__init__.py`, `memory.py`), `io/`, `optimize/`,
  `raytrace/`, `ui/` (except `lens_options_dialog.py`), `backend/` (except
  `backend/__init__.py`), `_math/`, `algebra/` — version-history narrative
  relocated out of the source** (audit 2026-09-11, finding **P2-4** /
  consolidated report **§14 V6**).  94 modules changed, **68 900 → 68 577
  lines**; the audit's own history classifier reads **9 350 (13.6 %) → 5 799
  (8.5 %)**, and a per-line count of the shapes the finding actually names
  (`vN.N (`, `pre-fix`, `used to`, `formerly`, `previously`, `the old`,
  `was wrong`, `no longer`, `historically`) reads **843 → 158 (−81 %)**.
  444 blocks are reproduced **verbatim** in 45 new documents under
  `docs/history/`, each under the source line it came from, each with a
  *Left in the source:* note saying which half of the rationale stayed and
  why.  A further 287 one-line release TAGS carrying no narrative (111 of
  them the same three lines of `v5.4.3 (audit GUI-resize)` /
  `v5.4.4 (audit GUI-resize round 2)` boilerplate repeated across the UI dock
  family) were stripped in place; the complete verbatim before/after list is
  in the WP report.  **No executable change:** the docstring-free AST and the
  comment-free, docstring-free token stream are byte-identical on all 111
  modules in the partition, enforced by
  `tests/unit/test_audit2609_a17_history_relocation.py`.

- **`lumenairy/_deprecation.py` 688 → 619 lines** — the largest single
  relocation, and the clearest instance of the pattern §14 V6 names: three
  running logs had accumulated inside the module (a five-entry
  horizon-slip log on `NEXT_REMOVAL_VERSION`, a shim-removal execution log,
  and two registry tombstones), each release appending an entry and none
  updating the one above.  All three now live in
  `docs/history/lumenairy._deprecation.md`.  `NEXT_REMOVAL_VERSION` is
  unchanged at `'5.48'`, `REMOVAL_SCHEDULE` and `API_TRANSITION_VERSION`
  unchanged, and `check_removal_schedule()` still returns no violations.

### Fixed -- comments that stated the opposite of the code (audit §15.7)

- **`glass.py`** — the `get_glass_index` polynomial-fallback comment claimed
  `POLYNOMIAL_COEFFICIENTS` was *"Empty at v4.16.2 ship; populating the 24
  catalogue entries is staged for v5.2.1"*.  All 24 entries have been bundled
  since v5.2.3 and `_POLYNOMIAL_STUB_NAMES` is an empty frozenset, so that arm
  covers every registered formula-3 glass on a minimal install.  Comment
  corrected.
- **`glass.py`** — `_glass_value_cache`'s comment carried a parenthetical
  correcting its own earlier *"femtometre"* wording.  The source now states
  the key once, in picometres, with the expression that produces it.
- **`raytrace/trace.py`** — `apply_doe_phase_traced`'s Notes corrected an
  earlier version of the same docstring (*"the pre-R5 docstring's claim that
  it 'neglects the cosine factor …' was itself inaccurate"*).  The source now
  states the settled fact once: the direction-cosine form is exact for
  in-plane diffraction, and the only approximation is the missing `1 / n2`.

### Unchanged on purpose

- Every **measured derivation of a live value** stayed in the source, as
  `docs/TESTING_STANDARDS.md` S5 requires — the `_GLASS_VALUE_CACHE_SIZE`
  65536 sizing argument, the wrapper-merit grid cache's 7.6 GB-at-N=2048
  retention figure, the Zemax EVENASPH power derivation with its +2.8 mm
  poc1-19/20 defocus, the Seidel sign conventions with their exact-trace
  oracles, the grating `1 / n2` ratio 1.503583 = n(N-BK7), `deep_nbytes`'s
  64× over-budget and 8000 B double-count, and the HDF5 compression
  measurements.
- Every **live migration statement** stayed — the `.. versionchanged:: 5.30`
  blocks on `design_optimize` / `MatchIdealSystem`, `io/storage.py`'s
  `compression='auto'` / `pre-v5.46` note, `optimize/context.py`'s one-cycle
  `DeprecationWarning`, `_context.py`'s `install_atexit_restore` alias.
- **`io/prescriptions_code_v.py`** was left entirely alone: its `pre-v5.46`
  references describe a file dialect the reader still has to detect on disk,
  which is a format-migration contract, not narrative.
- **`optimize/core.py`** and **`raytrace/core.py`** were left entirely alone:
  their comment blocks are a source-grep contract that five separate tests
  depend on.

### Migration

None.  No public API, default, signature or numeric result changes in this
sweep.  A reader looking for the rationale a comment used to carry will find
it, verbatim and under its original source line, in
`docs/history/<dotted.module.path>.md` — for example
`docs/history/lumenairy.io.storage.md` for `lumenairy/io/storage.py`.

### Note -- the `Tombstone, v5.30` label is back on the two executed deprecation-registry slots

Sweep 3 moved the two tombstone passages of `lumenairy/_deprecation.py` into `docs/history/lumenairy._deprecation.md` and left the slots saying only that their entries had been EXECUTED.  `tests/unit/test_niche_audit_w4_p5_return_contract.py::TestTransitionMachineryIsRetired::test_the_executed_entry_is_tombstoned_in_the_registry` pins the literal label, so each slot now names its tombstone in the present tense -- `REMOVAL_SCHEDULE`: the `'5.27' -> '5.32'` source-factory kwarg entry, removed with the kwargs; `API_TRANSITION_VERSION`: `propagate()` returns a `PropagationResult` by default -- without re-importing the narrative (comment-only; the module's fingerprints are unchanged by construction).

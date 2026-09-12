# WP-A17 SWEEP-4 -- changelog text

### Changed -- documentation: version-history relocated out of the lens family, the RCWA/PMM hygiene files and the package roots (audit 2026-09-11, finding P2-4 / sec. 14 V6, sec. 15.7)

The fourth and final version-history relocation sweep clears the partition the
audit's census found densest: `elements/_lens_traced.py` (14 898 lines, 32.6 %
classified history -- the largest module in the library), `_lens_real.py`,
`lenses_maslov.py`, the rest of the lens family, `propagators/gbd.py` /
`fga.py`, the RCWA/PMM hygiene files and the three package roots.
**22 modules, 60 926 -> 60 707 lines; comments and docstrings only, zero
executable change.**

* **Strict version-history lines 263 -> 10 (-96 %)** across the partition,
  where "strict" is the per-line count of the shapes the finding names
  (`vN.N (audit …)`, `pre-fix`, `pre-4.10` / `pre-v5.29`, `used to say/claim/
  read`, `formerly`, `was wrong`, `superseded`, `re-scheduled`, `retract`).
  The residual 10 are all the English word *prefix* -- `CONVENTIONS.md`
  Section 2 requires every error message to carry an `fn_name:` prefix -- which
  `\bpre-?fix\b` matches by construction.
* **88 blocks (863 source lines) moved VERBATIM** into **7 new documents**
  under `docs/history/` (`lumenairy.elements._lens_traced.md`,
  `._lens_real.md`, `.lenses_maslov.md`, `.lenses.md`, `._lens_jax.md`,
  `lumenairy.propagators.gbd.md`, `lumenairy.memory.md` -- 1 895 document
  lines), each under the source line it came from in the pre-relocation file,
  with a line-ordered table of contents and a *Left in the source* note per
  block.  Where the old rationale still explains present behaviour the source
  keeps a condensed present-tense why-comment plus a pointer to its document.
* **192 release tags stripped or rewritten in place** on otherwise-live why-comments, or
  rewritten from "pre-fix this did X" into the present-tense hazard they
  describe.  A `pre-fix` guard note that describes a hazard still reachable
  today was rewritten, not deleted, with the original wording recorded in the
  module's history document.
* **What stayed**: every measured derivation of a live constant
  (`docs/TESTING_STANDARDS.md` S5) -- the `_RD_HALO_*` 180-call calibration and
  its 123x separation table, the `_FIT_DISC_OUTSIDE_WEIGHT_REL` sweep and its
  niche-C2 envelope, the `_DECENTRE_GATE_*` table, the niche-C11 arbiter's
  42-point validation, the Newton pool's 267.2 B/point commit sweep, the
  `_GRAM_COND_MAX` conditioning numbers, the `lens_model='real'`
  bytes-per-pixel solve -- plus every live migration statement
  (`.. versionchanged::`, the CODE V / RCWA `formulation='li'` correctness
  advisories, the `output_grid` -> `output_shape` deprecation) and the
  fail-before switches.
* **Behaviour-free, proved**: every one of the 22 modules is byte-identical to
  its pre-relocation self under BOTH the AST fingerprint (docstrings and
  positions removed) and the token fingerprint (comments and docstrings
  dropped).  The splicer recomputes both and refuses to write on any drift, so
  a mistake could not reach disk.

### Added -- tests: a ratchet that stops the version-history backlog re-accumulating

* `tests/unit/test_audit2609_a17_history_lint.py` (+ its committed baseline
  `test_audit2609_a17_history_lint_baseline.json`, 122 modules / 688 lines,
  measured on the finished tree).  The relocation checker proves each MOVE was
  behaviour-free; it says nothing about the next comment somebody writes.  This
  one counts the finding's own shapes per module over `lumenairy/**/*.py` and
  **fails when any module's count grows**, listing the offending lines; a count
  that shrinks passes and prints the re-baseline hint.  A module with no
  baseline entry is held to 0, so narrative cannot enter through a new file.
  Re-baseline with `python tests/unit/test_audit2609_a17_history_lint.py
  --write` or `LUMENAIRY_HISTORY_LINT_WRITE=1 python -m pytest <file>`.
  Falsifiability: `test_the_counter_actually_counts` (11 synthetic modules,
  including the "executable string is behaviour, not prose" and "unparseable
  file falls back to a whole-file count" arms), `test_the_ratchet_direction_is_
  enforced` and `test_the_write_path_round_trips`.

### Fixed -- tests: the V20 JAX root-pick parity pin is structural, not a grep

* `tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py::
  test_jax_intersect_direction_aware_root_pick_present` matched the forbidden
  direction-blind selector as TEXT (`'t1 if R' not in src`), which a comment
  quoting that selector trips -- the test already carried a hand-written
  `str.replace` to hide one such comment from itself -- while a real regression
  spelled `t1 if radius > 0` or `jnp.where(R_safe > 0, t1, t2)` passed
  untouched.  The pin is now an AST match on the root-pick EXPRESSION: the
  min-modulus `where(|t1| <= |t2|, t1, t2)` or the Spencer-Murty stable-
  quadratic quotient must be present, and the direction-blind
  radius-sign selector (in both its `IfExp` and `jnp.where` forms) must not be.
  Two new falsifiability tests:
  `test_the_root_pick_matcher_rejects_a_direction_blind_kernel` mutates each
  REAL kernel in memory into exactly that regression (2 sites in
  `_intersect_jax`, 1 in `_intersect_jax_param`) and requires the verdict to
  flip on both counts, and `test_the_root_pick_matcher_is_not_a_text_search`
  asserts both failure directions the text form had.

### Migration

None.  Comments and docstrings only; no public API, default, or numerical
result changes.  Seven `docs/history/` documents are new
(`docs/history/lumenairy.elements._lens_traced.md` and siblings), the source
points at each one, and `scripts/record_history_fingerprints.py --check`
covers all 123 documents green.

### Not in this change, but found by it

`tests/unit/test_niche_audit_w4_p5_return_contract.py::TestTransitionMachineryIsRetired::test_the_executed_entry_is_tombstoned_in_the_registry`
is RED on the current tree: it pins the literal `Tombstone, v5.30` in
`lumenairy/_deprecation.py`, which WP-A17 SWEEP 3 moved wholesale into
`docs/history/lumenairy._deprecation.md` (0 occurrences in the source, 2 in
the document; the test's other four phrases all survive).  `_deprecation.py`
is not in this sweep's partition and is unmodified here.  Restoring the
heading `Tombstone, v5.30 (EXECUTED)` on the surviving registry-slot comment
fixes it without re-importing narrative -- the word is a present-tense
description of what that entry is.

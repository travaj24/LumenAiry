# WP-A17 SWEEP-1 -- changelog text

Assembled by the orchestrator into `CHANGELOG.md`.  Finding **P2-4**
(`TESTS-ARCH.md:394`) / consolidated report **sec. 14 V6** and **sec. 15.7**.

---

### Changed -- documentation: version-history narrative moved out of
`propagators/` (non-lens), `analysis/` and `sources/`

The "vN.N (audit X): pre-fix this did A, which was wrong because B; now it
does C" passages were relocated **verbatim** from 29 modules into per-module
documents under `docs/history/`, leaving a pointer in the source only where
the old rationale still explains present behaviour (audit P2-4 / sec. 14 V6).
This is the library-wide continuation of the WP-A17 part-1 relocation
(`propagators/carrier.py`, `carrier_field.py`, `fft_infra.py`).

**Zero executable change.**  Every one of the 29 modules is byte-for-byte
identical to its pre-relocation self under both fingerprints the checker
records -- the docstring-free AST and the comment-free/docstring-free token
stream -- verified by
`tests/unit/test_audit2609_a17_history_relocation.py`.

Measured over the 29 modules: 33 820 -> 33 180 source lines; "history" lines
by the audit's own loose classifier 11 034 -> 9 738 (32.6 % -> 29.3 % of the
partition); by the strict "pre-fix this did A" classifier the finding actually
names, 8 065 -> 5 618 (-30 %).  178 blocks, 1 754 prose lines, moved into
4 526 lines of `docs/history/*.md`; 1 058 lines of condensed rationale left at
the source sites.

Measured derivations of live constants stayed in the source, as
`docs/TESTING_STANDARDS.md` S5 requires: the `_SIGMA_GRID_N_MAX_DEFAULT` 7-rung
accuracy/cost table and the W4-T1 chirp-Nyquist argument
(`asymptotic_aberration_tensor.py`), the K18 source-area derivation and the
W9-14 sampling-adequacy occupancy figures (`hfpi.py`), the W9-12
`ray_subsample` two-arm measurement (`system.py`), the 2-shift-fold and
audit-P1 DC-anchor derivations (`asm.py`), the W6-A2 scale-relative
convergence argument (`asymptotic_maslov.py`), the W6-A11 grid-corner
fingerprint proof (`asymptotic_modes.py`) and the `bandlimit`-costs-five-decades
measurement (`rs.py`).

### Fixed -- documentation: comments that stated the OPPOSITE of the code
(sec. 15.7)

* `analysis/detector.py` -- the banner above the detector-integration block had
  been appended to three times without being edited and described **three
  different algorithms in sequence**, two of which name a `scipy.ndimage.zoom`
  path this module has not used since 4.10 (`scipy.ndimage` is not imported;
  `zoom` appeared nowhere outside that comment).  It now states what the two
  live branches do, and keeps the dimensional argument against point-sampling
  interpolation as a do-not-do-this.
* `propagators/asymptotic_modes.py` -- `_lg_mode_conj_stack`'s docstring opened
  by naming the grid ORIGIN as its cache key, two paragraphs above the
  paragraph explaining that the origin is not enough.  The key is now stated
  once, with the `indexing='xy'` / `indexing='ij'` collision that forced the
  three-corner fingerprint.
* `sources/core.py` -- `PartialCoherenceMCF`'s "deferred to v4.16+",
  `Source.gaussian_schell`'s "MCF-aware downstream propagators are not in
  v4.15.x scope", and the promise of "a future `Source.realizations()`
  per-realization iterator ... in scope for v4.16+".  The library is at v5.46;
  none shipped, and `lumenairy/_validation.py` still refuses a
  `PartialCoherenceMCF` at every propagator entry point.  All three now state
  the live limitation.

### Changed -- tests

* `tests/unit/test_niche_audit_w4_input_kind.py` --
  `test_shack_hartmann_declares_field_not_pupil`'s `assert 'superseding' in
  src` required `analysis/detector.py` to keep the v5.32 retraction of its own
  v4.15.5 `Input kind: 'pupil'` gloss.  With the gloss itself relocated there
  is nothing left to contradict, so the assertion tightens to the stronger
  pair: the superseded claim must be ABSENT and the live declaration
  (`input_kind='field'`, not 'pupil') must be stated.

### Added

* 29 documents under `docs/history/` (one per relocated module, named by the
  dotted module path because basenames collide across packages), each carrying
  the module's two pre-relocation fingerprints, a line-ordered table of
  contents, and every moved block verbatim under the source line it came from
  with a *Left in the source:* note.

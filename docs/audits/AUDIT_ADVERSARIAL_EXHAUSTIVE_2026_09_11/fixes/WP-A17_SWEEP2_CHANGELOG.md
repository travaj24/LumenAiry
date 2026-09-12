# WP-A17 SWEEP-2 -- changelog text

(Assemble into `CHANGELOG.md` alongside the other WP-A17 entries.  Finding
**P2-4** / consolidated report **sec. 14 V6** and **sec. 15.7**.)

### Changed -- elements: version-history narrative moved out of the source into `docs/history/`

Thirty modules under `lumenairy/elements/` (the PMM, RCWA-stack, BOR, EME,
Berreman, polarization, thin-element, DOE, coating, BSDF and geometry families)
carried `vN.NN (audit X): pre-fix this did A, which was wrong because B; now it
does C` narrative, multi-round `ROUND 2 / ROUND 3 / UPDATE 2026-08-xx`
chronologies, and comments that quote and retract an earlier revision of
themselves.  All of it moves **verbatim** into one `docs/history/<dotted module
path>.md` document per module (182 blocks, 4 078 lines of documentation), under
the source line it came from, with a line-ordered table of contents and a
*Left in the source* note per block.  Where the rationale is load-bearing for
what the code does now, the source keeps a condensed why-comment plus a pointer
to the document.

Measured over the thirty modules: source 42 100 -> 41 765 lines; the audit's
loose history classifier 8 091 -> 7 627 lines; its strict
"pre-fix this did A" classifier 4 614 -> **2 042** (-56 %); and narrative
trigger lines (`pre-fix`, `used to`, `formerly`, `previously`, `the old`,
`was wrong`, `historical`, `retracted`, `superseded`, excluding the bare
`vN.NN (audit X)` provenance tag) 231 -> **81** (-65 %).

**Zero executable change**: every module's docstring-free AST fingerprint and
its comment-free / docstring-free token fingerprint are byte-identical to the
pre-relocation file, pinned by
`tests/unit/test_audit2609_a17_history_relocation.py`, which discovers the new
documents automatically.  The 31 test files that assert on this partition's
source or docstrings run green unchanged (1 267 passed, 6 skipped, 0 failed);
no prose assertion was retired and no test file was modified.

### Fixed -- elements: comments that stated the opposite of the code or of the repo

Seven sites where a comment contradicted the code beneath it, the truth
elsewhere in the repo, or a correction printed a few lines below it:

* `pmm/stack.py` `PMM_SLIVER_GUARD` -- "the guard changes nothing on any solve
  that does not trip BOTH conjuncts" described round 1.  The trigger has been
  the geometric screen alone since round 4.
* `polarization.py` module docstring -- "CONVENTIONS.md section 7 still calls
  this row `Born-Wolf` and should be relabelled".  `CONVENTIONS.md` line 159
  already reads "IEEE / right-hand-rule".
* `bor/bor_solve.py` `_BOR_NODAL_SUPERUNITY_WARN` -- the bar claimed
  "3.71 decades above the healthy ceiling"; the binding two-sided margin,
  measured ten lines below, is **0.57 decades**.
* `bor/_sem_contract.py` `_BOR_Q_EXCESS` -- a two-sided margin ("0.95 decades
  above the worst ordinary geometry") that the conjunction can never reach.
  The honest one-sided margin is 1.20 decades below the mildest damaging rung.
* `bor/_sem_contract.py` `_BOR_SLIVER_BAND_FRAC` -- the ordinary margin is
  **1.003x** (graded hp refinement), not the 9.77x stated above it.
* `bor/_orient.py` `orient_band_scale` -- the two-sided population table prints
  `0.98 dec` for the thin SIGNAL end; the union minimum is **0.38 decades**.
* `pmm/_core.py` `_MORTAR_RESID_REFUSE` -- "rank-deficient by construction
  whenever ONE side is promoted", which reads as "either side"; the mechanism
  needs an ASYMMETRIC interface (EXACTLY ONE promoted side), and with BOTH
  promoted the operand is healthy by five decades.

No behaviour change: all seven are comment/docstring text and are covered by
the identity gate above.

### Added

* `docs/history/lumenairy.elements.*.md` -- 30 documents (one per relocated
  module), each carrying the module's two pre-relocation fingerprints, its
  pre-relocation line count, a line-ordered table of contents, and every moved
  block verbatim.

**Migration note:** none.  No public API, default, message or numeric result
changes.  A reader looking for the rationale behind a guard finds a condensed
why-comment plus the `docs/history/` pointer at the site; the full
round-by-round record is in that document, and `git log -S` on any phrase in it
still lands on the commit that wrote it.

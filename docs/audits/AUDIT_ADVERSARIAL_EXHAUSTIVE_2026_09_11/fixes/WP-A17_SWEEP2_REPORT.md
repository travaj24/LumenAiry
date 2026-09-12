# WP-A17 SWEEP-2 -- version-history relocation, `lumenairy/elements/`

Partition: every `.py` under `lumenairy/elements/` EXCEPT the traced/analytic
lens family (`_lens_traced*.py`, `_lens_real.py`, `_lens_imap.py`,
`_lens_jax.py`, `lenses*.py`, `lens_config.py`, `elements/__init__.py`) and
EXCEPT `rcwa/twod.py`, `rcwa/_core.py`, `rcwa/oned.py`,
`pmm/twod_staggered.py` (held by the hygiene agent).  **52 modules in scope;
30 relocated, 22 deliberately left alone** (see sec. 6).

Finding: **P2-4** (`TESTS-ARCH.md:394`) / consolidated report **sec. 14 V6** and
**sec. 15.7**.  Comments and docstrings only -- **zero executable change**, and
that is proved rather than asserted.

---

## 1. Summary

| finding | status | files | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| P2-4 / V6 -- version-history narrative in `lumenairy/elements/` | **fixed** | 30 modules, 42 100 -> 41 765 lines | `tests/unit/test_audit2609_a17_history_relocation.py` (30 new parametrised cases, auto-discovered) | SHA-256 of the docstring-free AST and of the comment-free/docstring-free token stream, recorded from each PRE-relocation file | strict "pre-fix this did A" lines 4 614 -> **2 042** (-56 %); narrative trigger lines 231 -> **81** (-65 %); loose classifier 8 091 -> 7 627 |
| sec. 15.7 -- comments that state the OPPOSITE of the code | **fixed, 7 sites** | see sec. 3 | the same identity gate + the existing behaviour tests | the code beneath each comment, `CONVENTIONS.md`, and the correction printed below each bar | 7 stale claims corrected, 0 behaviour change |

**Both fingerprints are byte-for-byte identical to the pre-relocation file on
all 30 modules.**  Not "equivalent" -- identical.  The applier recomputes both
from the spliced text and refuses to write anything on a mismatch; its guards
fired three times during this sweep (sec. 4) and every one was a real defect.

`ruff check lumenairy/elements/` -- *All checks passed*.
`python -c "import lumenairy"` -- OK.
Line endings preserved per file (7 of the 30 are LF, 23 CRLF; verified
byte-wise against the baseline copies -- no file changed its convention and
none acquired mixed endings).

---

## 2. What moved, and where it went

Thirty new documents, `docs/history/<dotted.module.path>.md` -- dotted because
basenames collide across packages (`pmm/stack.py` vs `rcwa/stack.py` vs
`pmm/stack2d.py`).  Each holds its blocks **verbatim** under the source line
they came from in the pre-relocation file, with a line-ordered table of
contents and a *Left in the source:* note per block, exactly as part 1 did.

| module | lines b -> a | loose b -> a | strict b -> a | narrative b -> a | blocks | doc lines |
|---|---|---|---|---|---|---|
| `_lens_thin.py` | 1431 -> 1355 | 805 -> 713 | 604 -> 16 | 25 -> 2 | 19 | 423 |
| `pmm/_core.py` | 7690 -> 7648 | 1155 -> 1118 | 510 -> 203 | 36 -> 17 | 21 | 370 |
| `rcwa/stack.py` | 3372 -> 3355 | 716 -> 699 | 298 -> 31 | 12 -> 0 | 10 | 192 |
| `pmm/stack.py` | 5360 -> 5256 | 1378 -> 1138 | 363 -> 170 | 21 -> 9 | 19 | 446 |
| `pmm/oned.py` | 1864 -> 1864 | 500 -> 500 | 172 -> 0 | 6 -> 1 | 5 | 94 |
| `polarization.py` | 1838 -> 1813 | 774 -> 711 | 527 -> 400 | 21 -> 0 | 22 | 394 |
| `eme/eme_2d_vector.py` | 1376 -> 1371 | 121 -> 121 | 140 -> 23 | 9 -> 2 | 9 | 149 |
| `berreman.py` | 1256 -> 1253 | 131 -> 130 | 162 -> 61 | 14 -> 5 | 8 | 166 |
| `doe.py` | 1258 -> 1222 | 281 -> 247 | 261 -> 168 | 7 -> 1 | 10 | 219 |
| `coatings.py` | 904 -> 902 | 350 -> 348 | 123 -> 37 | 5 -> 3 | 4 | 88 |
| `bor/bor_solve.py` | 826 -> 816 | 102 -> 102 | 106 -> 34 | 6 -> 4 | 7 | 167 |
| `elements.py` | 1402 -> 1402 | 415 -> 415 | 151 -> 83 | 3 -> 1 | 2 | 61 |
| `pmm/twod.py` | 2029 -> 2030 | 313 -> 314 | 198 -> 131 | 11 -> 6 | 5 | 91 |
| `pmm/conical.py` | 693 -> 693 | 37 -> 37 | 53 -> 0 | 4 -> 1 | 3 | 74 |
| `eme/eme_diffraction.py` | 299 -> 301 | 22 -> 24 | 51 -> 0 | 2 -> 0 | 2 | 59 |
| `bor/_sem_contract.py` | 569 -> 551 | 53 -> 53 | 94 -> 45 | 5 -> 4 | 10 | 205 |
| `bor/zcascade.py` | 279 -> 281 | 39 -> 41 | 39 -> 0 | 2 -> 0 | 2 | 64 |
| `bor/bor_stack.py` | 1031 -> 1030 | 173 -> 172 | 68 -> 39 | 3 -> 0 | 3 | 76 |
| `_berreman_jax.py` | 592 -> 592 | 34 -> 34 | 16 -> 0 | 2 -> 0 | 2 | 60 |
| `pmm/stack2d.py` | 2261 -> 2261 | 257 -> 274 | 190 -> 175 | 11 -> 7 | 3 | 84 |
| `bor/radial_eigensolver.py` | 178 -> 179 | 10 -> 10 | 14 -> 0 | 2 -> 1 | 1 | 52 |
| `bor/farfield.py` | 136 -> 137 | 0 -> 0 | 12 -> 0 | 1 -> 0 | 1 | 52 |
| `bor/_jax_bor.py` | 206 -> 206 | 11 -> 11 | 11 -> 0 | 1 -> 0 | 1 | 48 |
| `bsdf.py` | 810 -> 811 | 96 -> 96 | 58 -> 49 | 3 -> 2 | 1 | 46 |
| `segment_geometry.py` | 574 -> 574 | 17 -> 17 | 29 -> 22 | 3 -> 1 | 2 | 61 |
| `bor/_orient.py` | 296 -> 292 | 14 -> 14 | 175 -> 171 | 2 -> 2 | 2 | 85 |
| `bor/coupled_radial_eigensolver.py` | 716 -> 717 | 109 -> 109 | 86 -> 82 | 7 -> 6 | 1 | 47 |
| `eme/_branch.py` | 161 -> 160 | 0 -> 0 | 77 -> 76 | 1 -> 1 | 3 | 79 |
| `eme/eme_2d.py` | 503 -> 504 | 27 -> 28 | 26 -> 26 | 4 -> 3 | 2 | 59 |
| `pmm/stack2d_pure.py` | 2190 -> 2189 | 151 -> 151 | 0 -> 0 | 2 -> 2 | 2 | 67 |
| **total (30)** | **42 100 -> 41 765** | **8 091 -> 7 627** | **4 614 -> 2 042** | **231 -> 81** | **182** | **4 078** |

Each document's header is the machine-readable block the checker parses:

```
<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/stack.py
ast_sha256: ...
token_sha256: ...
pre_relocation_lines: 5360
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->
```

### 2.1 The three patterns that actually carried the history

**(a) A multi-ROUND campaign narrated in the source.**  The single largest
source in this partition.  `pmm/stack.py`'s O-11 sliver section narrated four
rounds of counter-measurement in 89 lines of header comment -- round 1's
super-unity conjunction, its 110-of-648 false positives, round 2's arbiter,
round 3's relative closure, round 4's removal of the super-unity precondition --
before any constant was declared.  `pmm/_core.py`'s T3-4 guard carried three
dated `UPDATE 2026-08-0x` sections, the third of which opens
*"The first bullet above was WRONG, and ubuntu CI proved it"*.
`bor/bor_solve.py` and `bor/_sem_contract.py` narrate rounds 2 and 3 in
docstrings; `pmm/stack2d_pure.py` and `eme/_branch.py` the same.  The source
now states what the guard IS -- screen, arbiter, verdict, and the bars each
carries -- and the rounds are in the document.

**(b) A comment correcting an earlier COMMENT.**  Part 1's pattern (b), and it
is everywhere here: *"This block used to say ... It is NOT"* (`pmm/stack.py`),
*"this docstring used to say `min_feature` never enters, which was wrong"*
(`pmm/stack.py`), *"HISTORY / CORRECTION ... That claim is REFUTED by
measurement"* (`berreman.py`), *"and its pre-v5.29.1 docstring described the
OPPOSITE of what it does"* (`_lens_thin.py`), *"the comment that used to sit
here ... was backwards"* (`_lens_thin.py`), *"this docstring used to conflate
them ... That guidance is RETRACTED"* (`_lens_thin.py`), *"(Historical note:
the docstring here previously reported a few-percent OOP residual ...)"*
(`pmm/conical.py`), *"the earlier claim here that `cut_band` computes the
identical quantity this line always did was also false"*
(`eme/eme_2d_vector.py`), *"audit M10 corrected this docstring, which used to
describe it as selecting the branch"* (`pmm/_core.py`), *"this docstring used
to read 'Returns new grid spacings', which it never did"* (`polarization.py`,
twice).  In every case the source now states the corrected rule once, plainly.

**(c) A guard whose justification was written as its release history.**
`_lens_thin.py` is the pure case: 55 lines of measured `f = nan` / `f = inf` /
`f = 0` behaviour across five `lens_model` branches, written as "v5.31 (audit
W4-1) ... v5.32 (audit W5-2) ...", plus the same tables again in the public
`Raises` section.  The source now states the hazard in present tense -- *left
to the models they disagree completely* -- and the measured tables are in the
document.  That module's strict count falls 604 -> 16.

### 2.2 What deliberately did NOT move

* **Derivations of live numeric bars.**  `docs/TESTING_STANDARDS.md` S5
  requires every numeric bar to carry its oracle, error floor and measured
  values.  Every sliver bar in `pmm/stack.py`, `_MORTAR_RESID_REFUSE` /
  `_MODE_CUT_*` in `pmm/_core.py`, `_BOR_Q_EXCESS` /
  `_BOR_SLIVER_BAND_FRAC` / `_BOR_NODAL_SUPERUNITY_*`, `_CENSUS_BAND` /
  `_STRUCTURAL_SAT`, the `raster` convergence ladders, the PIXEL CELL
  CONTRACT, the thin-element validity boundaries -- all stayed.  Three of them
  are read straight out of the source by tests (sec. 4).
* **Live migration and `versionchanged` statements.**  `doe.makedammann2d`'s
  4.14.2 SI-unit recipe and its 5.30 `_legacy_units` statement,
  `pmm.twod`'s v5.11 -> v5.12 API-change note, `coatings.v_coat_ar`'s 5.30
  `n_substrate` change, `apply_waveplate`'s retarder-sign change (rewritten as
  a `.. versionchanged::` directive with the consequences a caller must act
  on), `PMMStack.internal_field`'s H-scale change (condensed, not removed),
  `min_feature`'s MIGRATION paragraph.
* **Fail-before switch descriptions.**  `PMM_SLIVER_GUARD`,
  `PMM_MODE_CUT_GUARD`, `PMM_FORWARD_GROWTH_REPAIR`, `BOR_SEM_MESH_GUARD`,
  `BOR_NODAL_PASSIVITY_GUARD`, `PMM_JAX_MINNORM_PROJECTION` and the rest say "``False`` restores the pre-fix behaviour bit for bit".  That is
  the switch's live contract, not history.
* **Provenance tags.**  `v5.29 (audit E-L15): coerce FIRST` is attribution on
  a live why-comment.  Those stay, which is why the audit's *strict* counter
  (whose first alternative is the bare `vN.N (` token) falls only to 2 042
  while the narrative counter falls to 81 -- see sec. 2.3.

### 2.3 Two counters, and why the report quotes both

The audit's strict pattern is `vN.N (` OR `pre-fix` OR `used to` OR `formerly`
OR `previously` OR `the old` OR `was wrong`, scored over whole blocks.  Its
first alternative fires on the repo's ordinary attribution style, so a 55-line
live docstring scores 55 strict lines for one `v5.29 (audit E-H9)` tag.  The
second counter in the table is the same pattern with that tag dropped and
`historical`, `retracted`, `superseded`, `pre-guard` added, scored per LINE:
it is what this WP actually removes.  Reported both ways:

| counter | before | after | change |
|---|---|---|---|
| audit loose ("mentions a version, an audit or a date") | 8 091 | 7 627 | -6 % |
| audit strict (whole blocks, `vN.N (` included) | 4 614 | 2 042 | **-56 %** |
| narrative lines (`pre-fix` / `used to` / `formerly` / `previously` / `the old` / `was wrong` / `historical` / `retracted` / `superseded`) | 231 | **81** | **-65 %** |

Across the whole 52-module partition 85 narrative lines remain (the 81 above,
plus 4 in `freeform.py` and `_traced_flags.py`, which were not relocated).
Line by line they are:

| what the line is | count |
|---|---|
| the forwarding pointers this sweep ADDED ("the measured pre-guard tables are in `docs/history/...`") | 6 |
| fail-before switch contracts ("``False`` restores the pre-fix behaviour bit for bit") -- the switch's live meaning | 5 |
| live migration notes and `.. note:: API change (v5.11 -> v5.12)` directives | 3 |
| false positives ("the recurrence used TO EVALUATE `Q_m`", "any re-solve invalidates PREVIOUSLY RETAINED internals", "the historical 1-D PMM spelling" naming a still-accepted alias) | 11 |
| one clause inside a live why-comment naming the form the comment exists to warn against re-introducing ("the old real-Snell cap", "the historical `eig(solve(Mk, Ak))`", "the old strict 3-point grid") | 60 |

The last row is a deliberate judgement: those clauses have no narrative arc --
they name the alternative the current code rejects, which is what makes the
comment load-bearing.  Deleting the subject would leave the warning without
one.

Separately, narrative that lives inside **string literals** (user-facing
warning and error messages) and inside the `_ARCHIVE_SLANT_FOLD` module
constant is NOT counted above and was NOT touched: both are executable and
outside a documentation-only sweep.  See sec. 7 items 2 and 3.

---

## 3. Stale comments corrected (audit sec. 15.7, "worse than none")

Seven comments contradicted the code beneath them, the repo elsewhere, or a
correction printed a few lines below them.  All seven are comment/docstring
text and are covered by the identity gate.

1. **`pmm/stack.py` `PMM_SLIVER_GUARD`** -- "the guard changes nothing on any
   solve that does not trip BOTH conjuncts".  Round 4 removed the super-unity
   conjunct from the trigger: `_sliver_screen` is now a pure geometric test
   that runs on every solve.  Corrected to name the screen.
2. **`polarization.py` module docstring** -- "CONVENTIONS.md section 7 still
   calls this row `Born-Wolf` and should be relabelled `IEEE /
   right-hand-rule`".  `CONVENTIONS.md` line 159 already reads "IEEE /
   right-hand-rule ... Born & Wolf sec. 1.4.2 uses the exact NEGATIVE".  The
   sentence now says the two agree.
3. **`bor/bor_solve.py` `_BOR_NODAL_SUPERUNITY_WARN`** -- "3.71 decades above
   the healthy ceiling measured above", with a `ROUND 2 RESTATEMENT` ten lines
   below measuring the binding two-sided margin at **0.57 decades**.  The
   binding number is now stated first and the one-sided 3.71 is named as such.
4. **`bor/_sem_contract.py` `_BOR_Q_EXCESS`** -- the build quoted a two-sided
   margin ("0.95 decades above the worst ordinary geometry"), which the
   conjunction structurally cannot reach; its own restatement says so.  The
   source now carries only the one-sided margin (1.20 decades below the mildest
   damaging rung) and the reason the other side is vacuous.
5. **`bor/_sem_contract.py` `_BOR_SLIVER_BAND_FRAC`** -- "9.77x (0.99
   decades)", with a restatement below measuring **1.003x** on graded hp
   refinement.  The binding margin is now stated where the claim is.
6. **`bor/_orient.py` `orient_band_scale`** -- the two-sided population table
   prints `0.98 dec` for the thin SIGNAL end; the union minimum measured in
   the restatement is **0.38 decades**.  The source now says so immediately
   under the table instead of ten lines later.
7. **`pmm/_core.py` `_MORTAR_RESID_REFUSE`** -- "RANK-DEFICIENT BY
   CONSTRUCTION whenever ONE side ... is promoted", which a reader takes as
   "either side"; a `ROUND 4 CORRECTION` parenthetical inside the same
   paragraph said so.  The paragraph now states the ASYMMETRIC (EXACTLY ONE
   promoted side) scoping directly, with the both-promoted measurement folded
   in below rather than appended as a retraction -- and the three tokens
   `test_the_promoted_side_bar_is_scoped_to_the_asymmetric_interface` reads
   (`EXACTLY ONE side`, `ASYMMETRIC`, `BOTH sides promoted`) are all still
   present, verified.

---

## 4. Method, and the gate on it

Per module, in size order:

1. Record BOTH fingerprints from an untouched baseline copy taken before any
   edit (`ast_fingerprint` / `token_fingerprint` imported from the part-1
   checker itself, so there is no second implementation to drift).
2. Enumerate the flagged blocks with the audit's own classifier plus a
   chronology-marker grep (`versionchanged`, `UPDATE 20`, `ROUND N`, `RETIRED`,
   `no longer`, `pre-fix`, `post-fix`, `superseded`, ...), read each one, and
   decide: move, condense-and-point, or keep.
3. Write a plan of `(line range -> replacement text, site, what it records,
   what is left in the source)` and apply it with a splicer that works from the
   BASELINE copy every time -- so a plan can be extended and re-applied without
   compounding line-number drift.
4. The splicer recomputes both fingerprints from the spliced text and
   **refuses to write anything** unless both match, and refuses unless the new
   source names its own history document.
5. `ruff check` the module; `python -c "import lumenairy"`.

Step 4's guards fired three times, and each was a real defect:

* `eme/eme_diffraction.py` -- an edit landed in a `raise ValueError(...)`
  message, which is a string the interpreter executes.  Both fingerprints
  moved; the edit was dropped and recorded as a requested change instead
  (sec. 7 item 2).
* `pmm/_core.py`, an intermediate pass -- a replacement whose line numbers had
  been read off the ALREADY-EDITED file rather than the baseline, which
  truncated a docstring and made the file unparseable.  Caught at `ast.parse`,
  nothing written.  All such spans were then re-derived from the baseline by
  content.
* `eme/_branch.py` -- the new source did not name its own history document.
  Refused until a pointer was added.

A fourth class the fingerprints cannot see -- a splice that duplicates or
orphans a line of prose at its seam -- was caught by a separate pass that
re-derives every edit's four boundary lines in the final file and flags a
first/last line identical to its neighbour.  It found three: a duplicated
`# covariant GENERATOR itself is exact` line in `pmm/oned.py`, a duplicated
`dedupes that ~2x cost ...` clause in `bor/zcascade.py`, and a duplicated
`` `n_substrate` is READ `` line in `coatings.py`.  All three were corrected
and the pass is clean.  A fifth check reads every `.. versionchanged::`
directive's preceding line, because a dropped block can leave two directives
adjacent with no blank line between them (RST needs one); it found one, in
`doe.makedammann2d`, now fixed.

Tests that read this partition's source or docstrings were enumerated two
ways: `grep -rn "getsource\|getdoc\|__doc__" tests/` filtered to the module
names (28 files), and a second sweep for tests that read the module file
directly (`Path(...).read_text()`).  The load-bearing prose pins found, and
verified still satisfied after the sweep:

| test | what it pins on my modules | outcome |
|---|---|---|
| `test_fix_pmmstack_sliver_round4.py` | `833.78`, `906.56`, `dR/dx` within 4 000 chars BEFORE `_SLIVER_WALL_RATIO =` | kept; verified True/True/True |
| `test_fix_pmm2d_mortar_round4.py` | `EXACTLY ONE side`, `ASYMMETRIC`, `BOTH sides promoted` in `_MORTAR_RESID_REFUSE`'s derivation; `EXACTLY ONE promoted in-plane side` in `_guarded_mortar_solve.__doc__`; `EXACTLY ONE side is an in-plane region promoted` in the site source | kept; all verified True |
| `test_niche_audit_w9_raster_harmonic.py` | `raster='harmonic'`, `eps_cell_normal`, `Farjadpour`, `companion`, `harmonic` in `rcwa/stack.py`; `'harmonic'` / `RECOMMENDATION` / `REJECTED` in `add_tapered_grating.__doc__`; `eps_cell_normal` in `add_layer.__doc__` | kept; all verified present |
| `test_niche_audit_e_polarization_inputs.py` | `IEEE`, `right-hand-rule`, `Born & Wolf` in `polarization.__doc__`; `IEEE` in `stokes_parameters.__doc__` | kept; verified |
| `test_audit2609_a8_thin_elements.py` | `different reference planes`, `back vertex`, `503 um`, `0.272 waves` in `apply_spherical_lens`; `t_0 = f * exp(i*phi_r) + ...` in `thin_grating.__doc__`; OSA/Noll on `elements.zernike` | kept; verified |
| `test_niche_audit_w3_elements.py` | `NaN` + `aperture_diameter` on `apply_spherical_lens`; `not the whole library` + five function names on `apply_thin_lens`; absence of `Requires ``sigma``` | kept; verified |
| `test_audit_lens.py` | no bare `1.0 + 0.0j` in `_lens_thin` executable code | verified |
| `test_audit2609_a12_pmm1d.py` | absence of `Converges SPECTRALLY in the polynomial ``degree`` with no`; `TM` + `wall corner` in `pmm_jones_1d`; `_farfield_order_set` structure | untouched |
| `test_audit_s5_4_standalone_jones_transmission.py` | `CROSS-ENGINE SEAM` in `pmm_jones_1d.__doc__` | untouched |
| `test_audit2609_a13_staggered_cost.py` | `SEGMENT grid` / `PIXEL grid` on `pmm/twod._cell_to_walls_tile` and `stack2d_pure.add_layer` | untouched |
| `test_audit_v5_24_2_b2_bor_exports.py` | `azimuthal order`, `mode`, `order`, `S5-6` on `BORStack.__doc__` | the "previously mixed" clause moved, the terminology block kept |
| `test_g10_s5_9_layerspec_tracked.py` | `S5-9` + `LayerSpec` in `elements.__doc__` | untouched |
| `test_fix_pmm2d_mortar_round2.py` | `_interface_smatrix` source must NOT contain `gecon` / `_guarded_mortar_solve` / `rcond` (comments included) | no tokens introduced |
| `test_fix_bor_multilayer_guards.py`, `test_fix_pmmstack_sliver_round3.py` | source with comments (and docstrings) STRIPPED | structurally unaffected |
| `test_v4_15_1_agent_d.py` | the corrected Q-bfs coefficient form in `freeform.py`'s module comment | `freeform.py` not edited |

**No test file was modified.**  Every prose pin found was a pin on a statement
that is still true of the current behaviour, so keeping the statement was both
cheaper and more honest than retiring the test.

---

## 5. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

### 5.1 The A17 checker

```
python -m pytest tests/unit/test_audit2609_a17_history_relocation.py -q --no-header -p no:cacheprovider
```

```
79 failed, 402 passed in 21.55s
```

**All 79 failures are `test_the_header_names_a_real_module_and_two_fingerprints`
(77 of them) plus two `test_the_fingerprints_are_actually_sensitive` cases that
belong to another partition** (`lumenairy.analysis.coronagraph`,
`lumenairy.optimize.multi_objective`).  The 77 are the orchestrator's
dotted-name bug (sec. 7 item 1) and hit every dotted document from all three
sweeps, not only mine.  Deselecting that one assertion:

```
python -m pytest tests/unit/test_audit2609_a17_history_relocation.py -q --no-header     -p no:cacheprovider -k "not test_the_header_names_a_real_module"
2 failed, 399 passed, 80 deselected in 21.17s
```

with the same two other-partition failures and **zero** failures on any of my
30 documents.  Re-run at the end of the sweep, after the other partitions had
added more documents (the registry is shared and grows under all three of us):

```
1 failed, 540 passed, 108 deselected in 23.13s
```

-- the one failure being `test_the_fingerprints_are_actually_sensitive[lumenairy._context]`,
again another partition's module.

Independently of pytest, all 30 of my documents were re-verified directly
against their own recorded headers (AST fingerprint, token fingerprint,
source-points-at-the-document, TOC ascending, every TOC row has a matching
`### L<n>` anchor, no TOC line past `pre_relocation_lines`) as the last action
of the sweep:

```
30 documents fully verified, 0 failures
```

### 5.2 The test files that assert on this partition's source and docstrings

```
python -m pytest     tests/unit/test_audit2609_a12_pmm1d.py \n    tests/unit/test_audit2609_a12_verify_pmm1d.py \n    tests/unit/test_audit2609_a13_staggered_cost.py \n    tests/unit/test_audit2609_a14_rcwa_eme_bor.py \n    tests/unit/test_audit2609_a21_pmm_warning_filter.py \n    tests/unit/test_audit2609_a21_rcwa_and_ui.py \n    tests/unit/test_audit2609_a8_thin_elements.py \n    tests/unit/test_audit_lens.py \n    tests/unit/test_audit_polarization.py \n    tests/unit/test_audit_s5_4_standalone_jones_transmission.py \n    tests/unit/test_audit_v5_24_2_b2_bor_exports.py \n    tests/unit/test_audit_w3_pmm_jax_guards.py \n    tests/unit/test_audit_w5_elements_misc.py \n    tests/unit/test_audit_w6_analysis_elements.py \n    tests/unit/test_fix_bor_guards_round2.py \n    tests/unit/test_fix_bor_multilayer_guards.py \n    tests/unit/test_fix_pmm2d_mortar_round2.py \n    tests/unit/test_fix_pmm2d_mortar_round4.py \n    tests/unit/test_fix_pmmstack_sliver_round3.py \n    tests/unit/test_fix_pmmstack_sliver_round4.py \n    tests/unit/test_fix_pmmstack_sliver_walls.py \n    tests/unit/test_fix_pmmstack_sliver_walls_round2.py \n    tests/unit/test_g10_s5_9_layerspec_tracked.py \n    tests/unit/test_niche_audit_e_polarization_inputs.py \n    tests/unit/test_niche_audit_w3_elements.py \n    tests/unit/test_niche_audit_w3_rcwa_pmm.py \n    tests/unit/test_niche_audit_w6_berreman.py \n    tests/unit/test_niche_audit_w8_shapes.py \n    tests/unit/test_niche_audit_w9_raster_harmonic.py \n    tests/unit/test_v4_15_1_agent_d.py \n    tests/unit/test_verify_pmm2d_mortar_round3.py     -q --no-header -p no:cacheprovider --tb=line -rf

1267 passed, 6 skipped, 19 warnings in 911.89s (0:15:11)
```

**Zero failures.**  The 6 skips and 19 warnings are the files' own
(optional-dependency skips and the library's deliberate diagnostics -- the
sliver-band `UserWarning`, `RCWAYAverageWarning`, the union-grid snap notice).
These 31 files are the complete set that reads this partition's source or
docstrings, assembled by the two sweeps described in sec. 4, and they include
every test the brief named:
`test_audit2609_a12_pmm1d.py::test_g4_the_1d_jones_is_the_lab_cartesian_basis_and_jxx_is_minus_rp`
and `::test_g4_there_is_one_definition_of_the_1d_far_field_order_budget`,
`test_fix_pmmstack_sliver_round3/round4`, `test_fix_pmm2d_mortar_round2/round4`,
`test_niche_audit_w9_raster_harmonic`, `test_niche_audit_e_polarization_inputs`,
`test_audit2609_a8_thin_elements`, `test_niche_audit_w3_elements`,
`test_fix_bor_multilayer_guards`, `test_niche_audit_w6_berreman`,
`test_audit2609_a13_staggered_cost`, `test_audit_lens`,
`test_v4_15_1_agent_d`, `test_g10_s5_9_layerspec_tracked`,
`test_audit_v5_24_2_b2_bor_exports`,
`test_audit_s5_4_standalone_jones_transmission`.

**Retired prose assertions: NONE.**  Every prose pin found was a pin on a
statement that is still true of the current behaviour, so the statement was
kept rather than the test changed.  **No test file was modified by this WP.**

Because both fingerprints are byte-identical on all 30 modules, no behaviour
test in the repository can move: the interpreter sees the same program.  The
31 files above are the ones that can move, and they do not.

### 5.3 Static gates

```
ruff check lumenairy/elements/            -> All checks passed
python -c "import lumenairy"              -> OK
git diff --stat -- lumenairy/elements/    -> 30 files changed, 871 insertions(+), 1205 deletions(-)
```

Only the 30 files of this sweep appear in that diff: no lens-family file and
none of the hygiene agent's four files was touched.  Line endings were verified
byte-wise against the baseline copies -- no file changed its convention and no
file acquired mixed endings.

---

## 6. Modules deliberately left alone (partition complete)

22 of the 52 modules in scope carry no version-history narrative and got no
document.  Listed so the orchestrator can see the partition is complete:

| module | lines | audit loose | audit strict | narrative | why it was left |
|---|---|---|---|---|---|
| `pmm/twod_jones.py` | 1087 | 59 | 0 | 0 | no narrative |
| `freeform.py` | 739 | 259 | 125 | 2 | measured Q-bfs / Q-con derivations; both narrative hits are "the recurrence used TO EVALUATE `Q_m`", and the strict count is one `v5.2 (ROADMAP ...)` provenance tag.  `test_v4_15_1_agent_d.py` reads the corrected Q-bfs coefficient form out of this module's comment block |
| `pmm/_jax_stack.py` | 722 | 18 | 0 | 0 | no narrative |
| `bor/sem_radial.py` | 566 | 98 | 0 | 0 | derivations only |
| `pmm/_jax_twod.py` | 492 | 71 | 0 | 0 | no narrative |
| `_traced_flags.py` | 476 | 95 | 41 | 2 | an ERA / fail-before table; every entry is a live contract and the "pre-fix" wording is what the flag exists to force |
| `bor/_jax_sem.py` | 401 | 1 | 0 | 0 | no narrative |
| `pmm/_jax_stack2d.py` | 322 | 4 | 0 | 0 | no narrative |
| `emt.py` | 316 | 44 | 35 | 0 | one `v5.17 (audit P3-24)` provenance tag on a live warning |
| `thin_grating.py` | 294 | 81 | 0 | 0 | derivations only; its module docstring formula is pinned by `test_audit2609_a8_thin_elements.py` |
| `pmm/_jax_twod_jones.py` | 267 | 79 | 0 | 0 | no narrative |
| `eme/_jax_modes.py` | 259 | 11 | 0 | 0 | no narrative |
| `pmm/_stack2d_cache.py` | 252 | 42 | 0 | 0 | no narrative |
| `bor/_inv_census.py` | 109 | 0 | 0 | 0 | no narrative |
| `materials.py` | 104 | 0 | 0 | 0 | no narrative |
| `bor/stepindex_oracle.py` | 104 | 0 | 0 | 0 | no narrative |
| `bor/__init__.py` | 100 | 8 | 0 | 0 | no narrative |
| `eme/__init__.py` | 85 | 0 | 0 | 0 | no narrative |
| `bor/fiber_oracle.py` | 67 | 0 | 0 | 0 | no narrative |
| `coronagraph.py` | 54 | 36 | 0 | 0 | a re-export namespace |
| `rcwa/__init__.py` | 42 | 0 | 0 | 0 | no narrative |
| `pmm/__init__.py` | 41 | 0 | 0 | 0 | no narrative |

30 relocated + 22 left = **52**, the whole partition.

---

## 7. Requested changes outside my ownership

1. **`tests/unit/test_audit2609_a17_history_relocation.py`** -- the dotted-name
   generalisation has a bug and currently fails on **all 69** dotted documents
   across all three sweeps, mine included.  It builds the dotted name from the
   ABSOLUTE path:

   ```python
   src_path = REPO_ROOT / header["module"]
   dotted = ".".join(src_path.with_suffix("").parts)      # <- D:\.Metacept.Neurophos...
   assert name in (src_path.stem, dotted)
   ```

   so the comparison value is
   `D:\.Metacept.Neurophos.Python_Test_Scripts.Free_Space_Optics.Lumenairy.lumenairy.elements.pmm.stack`.
   The fix is to derive it from the RELATIVE path in the header:

   ```python
   dotted = ".".join(pathlib.PurePosixPath(header["module"]).with_suffix("").parts)
   ```

   With `test_the_header_names_a_real_module_and_two_fingerprints` deselected,
   every other case on my 30 documents passes -- AST identity, token identity,
   TOC ordering and anchors, the source-points-at-the-document check, and the
   falsifiability check.  Per the brief I did **not** rename the documents.

2. **`lumenairy/elements/eme/eme_diffraction.py` lines 168-169** -- the
   zero-norm refusal's user-facing message carries version-history narrative
   inside a string literal:
   `"... (this used to surface as an opaque 'SVD did not converge' LinAlgError
   from the least-squares solve)."`  Rewriting it to the present tense changes
   a STRING the interpreter executes, so both fingerprints move.  It needs a
   deliberate one-line change plus a re-record of that document's two hashes in
   the same commit.  The same applies to
   `lumenairy/elements/pmm/stack.py` lines ~3113-3119, where a warning message
   quotes and retracts its own earlier wording
   (*"this message used to say per-layer grids have 'no cross-layer walls to
   perturb', which is WRONG"*), and to a handful of
   `"... invalidates previously retained internals"` phrasings.

3. **`lumenairy/elements/pmm/_core.py` lines 7472-7648, `_ARCHIVE_SLANT_FOLD`**
   -- an explicit in-source ARCHIVE of a superseded method ("SUPERSEDED: the
   ezz*tan^2 STATIC METRIC FOLD ... WHY SUPERSEDED: ...") held as a ~180-line
   module-level raw-string CONSTANT.  This is exactly what `docs/history/`
   exists for, and its own header says the record is kept "in the code, not
   only in volatile notes" -- which `docs/history/` now supersedes.  Moving it
   deletes a module-level assignment, i.e. an AST change, so it is outside a
   documentation-only sweep.  Suggested follow-up: move the string body into
   `docs/history/lumenairy.elements.pmm._core.md`, delete the constant, leave
   the `_build_generator_metric` comment pointing at the document instead of at
   `_ARCHIVE_SLANT_FOLD`, and re-record that document's two hashes in the same
   commit.  ~0.25 d.  It is the single largest remaining block of history in
   the partition.

4. **`CONVENTIONS.md`** -- no change needed; item 2 of sec. 3 was a stale claim
   ABOUT `CONVENTIONS.md`, and `CONVENTIONS.md` itself is already correct.

---

## 8. Deferred

* **The `vN.NN (audit X)` provenance tag.**  Left alone deliberately: it is the
  repo's attribution style on live why-comments and `CONTRIBUTING.md` welcomes
  why-comments.  It is also the single reason the audit's *strict* counter
  stays at 2 042.  If the orchestrator wants that counter driven down, the
  mechanical change is to drop the release number and keep the audit id
  (`v5.29 (audit E-L15)` -> `(audit E-L15)`) across ~250 sites; that is pure
  churn and I did not do it unasked.
* **A CI lint on new history.**  Part 1 proposed it (~0.5 d) and this sweep is
  more evidence for it: every one of the multi-ROUND blocks here grew by
  APPENDING a round rather than by updating the statement, which is the process
  root cause the finding names.  A lint that refuses a new comment block
  matching the narrative pattern in a module that already has a
  `docs/history/` document would stop the backlog re-forming.  `CONTRIBUTING.md`
  and the CI config are not in my ownership.

---

## 9. Files touched

**Modified (comments and docstrings only -- AST and token stream identical):**
the 30 modules in the sec. 2 table.

**New:**

* `docs/history/lumenairy.elements.*.md` -- 30 documents (4 078 lines).
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP2_REPORT.md` (this file)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP2_CHANGELOG.md`

**Not touched:** every test file; the lens family; `rcwa/twod.py`,
`rcwa/_core.py`, `rcwa/oned.py`, `pmm/twod_staggered.py`; `CHANGELOG.md`,
`README.md`, `CONVENTIONS.md`, `pyproject.toml`, `lumenairy/__init__.py`.

## 10. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP2_CHANGELOG.md`

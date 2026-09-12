# WP-A17 SWEEP-4 -- version-history relocation, the lens family and the package roots

Scope: **comments and docstrings only, zero executable change**, over the lens
family (`elements/_lens_real.py`, `_lens_traced.py`,
`_lens_traced_multibranch.py`, `_lens_traced_uniform.py`, `_lens_imap.py`,
`_lens_jax.py`, `lenses.py`, `lenses_maslov.py`, `lenses_gbd.py`,
`lens_config.py`), `propagators/gbd.py` and `fga.py`, the hygiene pass's files
(`elements/rcwa/twod.py`, `rcwa/_core.py`, `rcwa/oned.py`,
`elements/pmm/twod_staggered.py`, `memory.py`, `ui/lens_options_dialog.py`,
`propagators/propagation.py`) and the three package roots
(`lumenairy/__init__.py`, `elements/__init__.py`, `backend/__init__.py`).

Finding: **P2-4** (`TESTS-ARCH.md:394`) / consolidated report **sec. 14 V6** and
**sec. 15.7**.  Method and document format follow WP-A17 part 1
(`fixes/WP-A17_REPORT.md`) and sweeps 1-3; the checker is
`tests/unit/test_audit2609_a17_history_relocation.py`, unchanged by me.

---

## 1. Summary

| | |
|---|---|
| modules in the partition | **22** (fingerprinted before any edit) |
| modules changed | **20** |
| modules deliberately left alone | **2** (sec. 7) |
| lines | **60 926 -> 60 707** (-219) |
| STRICT history lines (the finding's own shapes) | **263 -> 10 (-96 %)** |
| LOOSE history lines (the audit's own classifier) | **12 365 (20.3 %) -> 10 753 (17.7 %)** |
| history documents written | **7** (`docs/history/<dotted.module.path>.md`, 1 895 lines) |
| blocks recorded verbatim | **88** (863 source lines moved, 660 left as condensed present-tense why-comments) |
| release TAGS stripped / rewritten in place | **192** across 20 modules -- full before/after list in sec. 9 |
| contradicting comments corrected | **0 found live** (sec. 5 -- three were *records of* contradictions already fixed, and those records moved) |
| retired prose assertions | **none** (sec. 6) |
| tests added / strengthened | **2** (sec. 8) |

**Both fingerprints are byte-for-byte identical to the pre-relocation file on
all 22 modules.**  Not "equivalent" -- identical, enforced by the applier
before every write and re-checked afterwards by the checker and by
`scripts/record_history_fingerprints.py --check`:

```
python scripts/record_history_fingerprints.py --check
-> OK: every history document matches its module.      (123 documents)
```

`ruff check lumenairy tests scripts` -- *All checks passed*.
`python -c "import lumenairy"` -- OK.  All 233 modules under `lumenairy/`
compile.

### 1.1 Method

Per module, biggest first:

1. take a byte-for-byte baseline copy and record BOTH fingerprints
   (`ast_fingerprint` / `token_fingerprint`, imported from the checker itself
   so there is no second definition of "unchanged") from the UNTOUCHED file;
2. enumerate candidate blocks with the finding's strict classifier PLUS a wide
   discovery regex (`versionchanged`, `UPDATE 20`, `ROUND N`, `RETIRED`,
   `no longer`, `pre-fix`, `pre-v?N`, `previously`, `the old `, `historic`,
   `earlier`, `superseded`, ...), read each one, and decide: move,
   condense-and-point, strip the tag, or keep;
3. write ONE plan per module as `(line range -> replacement lines)` in
   PRE-RELOCATION line numbers, and apply it in a single shot against the
   pristine baseline -- so every document's table of contents cites the file
   as it stood before a single block moved, and no document records a line
   number from a half-relocated intermediate;
4. the splicer re-parses the result, recomputes both fingerprints and
   **refuses to write unless both are identical** and the new source names its
   own history document;
5. a seam pass re-derives every edit's boundary lines in the DELIVERED file and
   flags a first/last replacement line identical to its neighbour, or a blank
   run the baseline did not have;
6. `ruff check` the module; re-import the package.

Line endings are preserved byte-wise: every read is `read_bytes()` and the
write re-applies the file's own convention (sweep 3 sec. 5d -- `read_text()`
translates CRLF, which silently rewrote 12 files in that sweep).  The 22
modules are 18 CRLF and 4 LF; all 22 came out with the convention they went in
with (`CRLF 14746 / bare LF 0` on `_lens_traced.py`, etc.).

### 1.2 What the splicer's gates caught

* **Two docstring truncations.**  A replacement that ended a docstring's last
  line without re-emitting the closing `"""` made the file unparseable
  (`lenses_maslov._tukey_window`, `_lens_real._obl_gap_advance`).  Caught at
  `ast.parse`, nothing written.
* **Seven seam defects.**  Four EXACT duplications, where a replacement's
  first or last line repeated its neighbour verbatim (`_lens_traced` L11704,
  `_lens_real` L306, `_lens_jax` L699, `twod_staggered` L44), and three the
  exact check could not see: two PARAPHRASE duplications (`propagation.py`'s
  "Six submodules share one ... layer:" above the original "six submodules
  sharing one ... layer:", and `lenses_gbd` L456's re-worded clause above the
  surviving one), one sentence broken across a seam (`propagation.py` L94)
  and one INDENT mismatch (`lenses_maslov` L2992, 8 spaces into a 4-space
  block).  The exact check found four; a second pass scoring the
  word-set Jaccard overlap of each replacement's boundary line against its
  neighbour, plus a leading-whitespace comparison, found the other three.
  Both checks are clean on the delivered tree.
* **Two self-inflicted strict hits.**  A replacement that introduced the word
  *superseded* (`memory.py`, `lumenairy/__init__.py`), which is itself one of
  the shapes the finding names.  Caught by re-running the strict count on the
  spliced text before writing.

### 1.3 The four patterns that carried the history here

**(a) A constant carrying every derivation it ever had.**  The sharpest case in
the library, and the reason `_lens_traced.py` is 32.6 % history:
`_REMAP_RESID_EIKONAL_DEGREE` is `6`, and 137 of its 144 comment lines were
about the value `4` -- the end-to-end table that chose 4, the 0.103 %-of-input-
power on-axis ghost it shipped with, a "MECHANISM" note refuting the note two
paragraphs above it, a "RESOLVED" note correcting a guess in the same block,
and a dated "RAISED TO 6" announcement.  The live derivation (degree 6's r^4 /
r^6 form argument and the C8 support-bound table that makes it safe) is 41
lines.  Same shape on `_NEWTON_MAX_ITERS` (8-in-3.5.5, 12-in-3.5.6) and on
`memory.py`'s `_LENS_REAL_*_ARRAYS` (the pre-v5.46 scaling that under-predicted
by 2.8x, above the calibration that replaced it).

**(b) The same fix annotated at every site it touched.**  `v5.29.1 (audit E-H2)`
-- the Newton cap reaching the process pool -- appears at six sites in
`_lens_traced.py`, each re-telling that the pool used to hard-code 12; the
`v5.33.0 FIX_POOL_REBUILD` fit-in-the-payload story at three; the
`v4.13.0 (audit L2)` dtype-unification in `_lens_jax.py` at four; the RCWA
`pre-v5.14.1` dual-Laurent z-rule at four sites in `rcwa/twod.py`.  In every
case the RULE and its measured consequence are live and stayed, restated once
in the present tense; what moved is the per-site record of which release wrote
which half.

**(c) A boilerplate release tag on an otherwise-live why-comment.**  192 of
these, 50 of them in `lumenairy/__init__.py` alone, where every re-export
block opened `# vX.Y (audit …): <what the export is>`.  These are one-line
labels with no narrative attached, so relocating them into new documents plus
new source pointers would have added more noise than it removed: they were
stripped in place, and the complete verbatim before/after list is sec. 9.  A
line whose remainder carried ANY narrative marker was excluded from the
mechanical pass and relocated by hand instead (`lumenairy/__init__.py:929`,
the `pre-4.7` deprecated-alias block, is the one that tripped that filter).

**(d) A comment correcting an earlier COMMENT.**  Eleven sites.  These are
notes about the documentation's own history and are worthless in the source
once the text they retract is gone: `_lens_traced.py`'s "the comment that used
to sit at this site … was wrong twice over" (the vignetting/NaN note),
`_reverse_prescription`'s "The docstring used to assert the opposite", the
backward-trace docstring retracting its own "~35-40 nm" figure,
`_solve_lstsq_thread_safe`'s "This function used to assert that A is
well-conditioned", `lenses_maslov._solve_fit`'s retraction of the same
justification, `REMAP_STATIONARY_PHASE_FIT_GUARD`'s "This RETRACTS the
hypothesis recorded in `_REMAP_RESID_EIKONAL_DEGREE`", `_analytic_lens_phase`'s
"the rule this docstring used to state", `_lens_real.py`'s v5.30 E-M1 `.. note::`
about a parameter that "read default 8" while the signature shipped 6, and the
two deletion tombstones in `lenses.py` / `_lens_traced.py`.

### 1.4 What deliberately did NOT move

* **Measured derivations of live constants.**  `docs/TESTING_STANDARDS.md` S5
  requires a numeric bar to carry its oracle, and most of this partition IS
  that.  Kept in full and named so the line drawn is visible: the
  `_RD_HALO_AMAX_TOL` / `_RD_HALO_RADIUS_FACTOR` 180-element-call calibration
  with its 123x clean-vs-defective separation table and its four-part SCOPE;
  the `ray_density` energy self-check's 24-cell P2 battery envelope and the
  deliberate 0.050 gain band; `_FIT_DISC_OUTSIDE_WEIGHT_REL`'s fold sweep and
  its niche-C2 low-NA envelope statement; `_DECENTRE_GATE_PIXELS` /
  `_DECENTRE_GATE_W_FRAC`'s 0-to-1.0 w table; the niche-C11 arbiter's 42-of-42
  validation and its per-order trade; `REMAP_STATIONARY_PHASE_FIT_GUARD`'s
  chain-level reach map and three numbered measurements;
  `REMAP_INVERSE_SUPPORT_BOUND`'s feather study; the Newton pool's 267.2
  B/point commit sweep and 1.728 GB intercept; `LSTSQ_CONDITIONING_STEPDOWN`'s
  1e-6 margin argument; `lenses_maslov`'s `local_quadrature` window geometry
  (9.1 / 40 % / 8.09e-02) and `_GRAM_COND_MAX` numbers; `memory.py`'s v5.17.1
  anchors and the 2026-09-12 `apply_real_lens` bytes-per-pixel solve
  (`8*F + 16*C = 176.02`, `8*F + 8*C = 120.01`, ~7 % margin); `gbd.py`'s
  Collins-factor measurements; `rcwa/_core.py`'s `_HOMOG_CACHE` bound
  rationale.
* **Live migration statements.**  `fga.py`'s `.. versionchanged:: 5.30` on
  `nsig`; `gbd.py`'s `output_grid` -> `output_shape` rename with its
  `DeprecationWarning`; `twod_staggered.py`'s `Efficiency2D` API-change note
  (v5.11 -> v5.12); `rcwa/twod.py`'s "results on lossy cells from an earlier
  release should be re-run" advisory -- a correctness warning a user acts on,
  kept in full with only its framing made present-tense; `lumenairy/__init__.py`'s
  deprecated-alias shim contract.
* **Do-not-do-this notes written as history.**  The judgement call that shaped
  most of this sweep.  "Pre-4.10 `abs(t)` accidentally took the complex
  conjugate of the axial phase on back-propagation" describes a hazard that is
  *still reachable* -- the current expression is the only thing preventing it.
  Those were rewritten into the present tense at the source site and the
  original wording recorded in the module's document, rather than deleted:
  a reader deciding whether the line can be simplified needs the failure, not
  the release number.  Most of the 88 documented blocks and a large share of
  the 192 in-place rewrites went this way, which is why the line count barely
  moves (-219 over 60 926) while the strict count collapses.
* **Fail-before switches.**  Every "setting X to Y restores the previous
  behaviour exactly" sentence is a live description of a shipped knob and
  stayed (`REMAP_STATIONARY_PHASE_LAUNCH`, `DECENTRED_FIT_ARBITER`,
  `_FIT_DISC_OUTSIDE_WEIGHT_REL = 0.0`, `TILTED_CARRIER_EXACT_EIKONAL`,
  `_PMM2D_MIN_SEG_ENFORCE`, `SUPPORT_BAND_CHECK='silent'`, ...).

---

## 2. Per-module numbers

`before` is the pre-relocation file; `after` is the delivered working tree.
LOOSE is the audit's own block classifier (a contiguous comment run counts
entirely if it mentions a version, the word "audit" or a `20xx-xx-xx` date; a
string statement counts if it carries >= 2 such markers).  STRICT is the
per-line count of the shapes the finding names -- the same set the new
history-lint ratchet (sec. 8A) baselines, so these numbers and that baseline
are the same numbers.

| module | lines b -> a | strict b -> a | loose b -> a | blocks | strips | doc lines |
|---|---|---|---|---|---|---|
| `lumenairy/elements/_lens_traced.py` | 14898 -> 14744 | 75 -> 0 | 4855 (32.6 %) -> 4178 (28.3 %) | 44 | 47 | 911 |
| `lumenairy/elements/_lens_real.py` | 8117 -> 8094 | 27 -> 0 | 1537 (18.9 %) -> 1392 (17.2 %) | 16 | 17 | 277 |
| `lumenairy/elements/rcwa/_core.py` | 4990 -> 4990 | 4 -> 3 | 1100 (22.0 %) -> 1100 (22.0 %) | -- | 1 | -- |
| `lumenairy/elements/lenses_maslov.py` | 4484 -> 4475 | 20 -> 1 | 789 (17.6 %) -> 757 (16.9 %) | 6 | 14 | 157 |
| `lumenairy/elements/pmm/twod_staggered.py` | 3897 -> 3897 | 9 -> 0 | 687 (17.6 %) -> 509 (13.1 %) | -- | 10 | -- |
| `lumenairy/propagators/gbd.py` | 3845 -> 3842 | 16 -> 0 | 399 (10.4 %) -> 281 (7.3 %) | 9 | 4 | 218 |
| `lumenairy/propagators/fga.py` | 3234 -> 3234 | 5 -> 0 | 469 (14.5 %) -> 461 (14.3 %) | -- | 5 | -- |
| `lumenairy/elements/rcwa/twod.py` | 2412 -> 2412 | 6 -> 0 | 669 (27.7 %) -> 644 (26.7 %) | -- | 6 | -- |
| `lumenairy/__init__.py` | 2252 -> 2248 | 45 -> 0 | 258 (11.5 %) -> 158 (7.0 %) | -- | 50 | -- |
| `lumenairy/elements/_lens_imap.py` | 1940 -> 1940 | 0 -> 0 | 74 (3.8 %) -> 74 (3.8 %) | -- | -- | -- |
| `lumenairy/elements/rcwa/oned.py` | 1896 -> 1896 | 3 -> 2 | 212 (11.2 %) -> 212 (11.2 %) | -- | 1 | -- |
| `lumenairy/memory.py` | 1332 -> 1329 | 15 -> 2 | 429 (32.2 %) -> 375 (28.2 %) | 5 | 7 | 131 |
| `lumenairy/elements/lens_config.py` | 1303 -> 1302 | 2 -> 1 | 80 (6.1 %) -> 79 (6.1 %) | -- | 2 | -- |
| `lumenairy/elements/lenses.py` | 1171 -> 1156 | 7 -> 0 | 133 (11.4 %) -> 76 (6.6 %) | 3 | 4 | 84 |
| `lumenairy/elements/_lens_traced_multibranch.py` | 1161 -> 1161 | 3 -> 1 | 62 (5.3 %) -> 57 (4.9 %) | -- | 2 | -- |
| `lumenairy/elements/_lens_jax.py` | 1021 -> 1014 | 14 -> 0 | 262 (25.7 %) -> 215 (21.2 %) | 5 | 7 | 117 |
| `lumenairy/elements/_lens_traced_uniform.py` | 994 -> 994 | 1 -> 0 | 6 (0.6 %) -> 0 (0.0 %) | -- | 1 | -- |
| `lumenairy/elements/lenses_gbd.py` | 644 -> 644 | 2 -> 0 | 172 (26.7 %) -> 145 (22.5 %) | -- | 2 | -- |
| `lumenairy/ui/lens_options_dialog.py` | 469 -> 469 | 0 -> 0 | 0 (0.0 %) -> 0 (0.0 %) | -- | -- | -- |
| `lumenairy/propagators/propagation.py` | 393 -> 393 | 8 -> 0 | 144 (36.6 %) -> 12 (3.1 %) | -- | 9 | -- |
| `lumenairy/elements/__init__.py` | 367 -> 367 | 0 -> 0 | 28 (7.6 %) -> 28 (7.6 %) | -- | 2 | -- |
| `lumenairy/backend/__init__.py` | 106 -> 106 | 1 -> 0 | 0 (0.0 %) -> 0 (0.0 %) | -- | 1 | -- |
| **total (22)** | **60926 -> 60707** | **263 -> 10** | **12365 (20.3 %) -> 10753 (17.7 %)** | **88** | **192** | **1895** |

### 2.1 The residual 10, checked by hand

Every one is the English word **prefix**, which `\bpre-?fix\b` matches by
construction:

| module | line | text |
|---|---|---|
| `elements/rcwa/_core.py` | 1978 | `Raises :class:`ValueError` with a ``fn_name:`` prefix (CONVENTIONS Section 2)` |
| `elements/rcwa/_core.py` | 2080 | `wrong answer.  Raises with a ``fn_name:`` prefix when undersampled.` |
| `elements/rcwa/_core.py` | 2361 | `each rejected up front with a ``fn_name:`` prefix:` |
| `elements/rcwa/oned.py` | 880 | `# Validate geometry HERE so the error carries this function's prefix` |
| `elements/rcwa/oned.py` | 882 | `# prefix from the per-wavelength call, confusing the caller).` |
| `elements/lenses_maslov.py` | 1916 | `# or non-integer ``stop_index`` RAISES with a Section 2 prefix rather` |
| `memory.py` | 422 | `prefix : str` |
| `memory.py` | 423 | `Optional prefix string (e.g. for indentation).` |
| `elements/lens_config.py` | 64 | `call raises (CONVENTIONS.md section 2 prefix).  Concretely, for each setting:` |
| `elements/_lens_traced_multibranch.py` | 77 | `#: importing it charged ``scipy.special``'s ~540 ms shared prefix` |

These were left alone deliberately.  `CONVENTIONS.md` Section 2 requires every
error message to carry an `fn_name:` prefix, so the word is load-bearing
library vocabulary; narrowing the pattern to exclude it would put this
report's numbers out of step with the finding's own classifier and with the
three earlier sweep reports.  The ratchet in sec. 8A baselines them instead of
demanding zero -- it cares that a count does not climb, not that it is 0.

---

## 3. Files touched

**Modified** (comments and docstrings only -- AST and token fingerprints
identical on every one): the 20 modules with a nonzero `blocks` or `strips`
column above.

**New** (7 documents under `docs/history/`):

```
docs/history/lumenairy.elements._lens_traced.md       910 lines, 44 blocks
docs/history/lumenairy.elements._lens_real.md         277 lines, 16 blocks
docs/history/lumenairy.propagators.gbd.md             218 lines,  9 blocks
docs/history/lumenairy.elements.lenses_maslov.md      157 lines,  6 blocks
docs/history/lumenairy.memory.md                      131 lines,  5 blocks
docs/history/lumenairy.elements._lens_jax.md          117 lines,  5 blocks
docs/history/lumenairy.elements.lenses.md              84 lines,  3 blocks
```

**New tests** (sec. 8):

```
tests/unit/test_audit2609_a17_history_lint.py
tests/unit/test_audit2609_a17_history_lint_baseline.json
```

**Modified test**: `tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py`
(sec. 8B).

**Not modified**: `lumenairy/elements/_lens_imap.py`,
`lumenairy/ui/lens_options_dialog.py` (sec. 7), `CHANGELOG.md` (sec. 10), the
checker `tests/unit/test_audit2609_a17_history_relocation.py`, the recorder
`scripts/record_history_fingerprints.py`, and
`lumenairy/propagators/carrier.py` / `tests/unit/test_niche_d6_exact_tilted_leg.py`
(WP-A24's).

---

## 4. Why `propagators/propagation.py` got NO document

It was relocated like the rest, and then the document was withdrawn, because
the checker cannot mutate it.

`test_the_fingerprints_are_actually_sensitive`'s mutation 1 needs a
`FunctionDef` with `>= 3` body statements containing at least one single-line
statement.  `propagation.py` is a pure re-export shell whose ONLY function is
`__getattr__`, with **2 body statements, both multi-line** (measured by walking
its AST).  Registering a document for it made the checker red with
"no single-line statement found to delete" -- reporting the module as suspect
when what had actually happened is that the mutation catalogue ran out of
targets, exactly the failure mode the checker's own mutation-3 message warns
about.

Sweep 3 sec. 6 met the same class (`optimize/__init__.py`,
`io/prescriptions.py`) and ruled it as tag-only strips with no document.  I
followed that precedent: the module's two narrative passages (the "v5.1.0 split
(Agent C)" banner and the "Every name that pre-v5.1.0 was importable …"
contract) were rewritten in place into the present tense, their original
wording is recorded verbatim in sec. 9 of this report rather than in a
document, and `propagation.py` keeps no `docs/history/` pointer.  A concrete
one-line widening of mutation 1 that would let such a module carry a document
is in sec. 11 item 1.

---

## 5. Contradicting comments (audit sec. 15.7, "worse than none")

**None found live in this partition.**  That is a measurement, not an absence
of looking:

* a programmatic probe compared every documented parameter default
  (`name : type, default X` / `defaults to X`) against the runtime signature
  default for every public function and class of the 16 importable modules in
  the partition -- **0 mismatches** (the shape that produced the v5.30 E-M1
  finding in `_lens_real.py`);
* the string-literal cross-check of sec. 6 read every phrase the test corpus
  pins against these modules;
* every block that moved was read in full.

What the sweep DID find is three *records of* contradictions that had already
been fixed, which are worthless in the source once the text they retract is
gone.  All three moved to a document:

1. **`_lens_real.py:5014-5019`** -- a `.. note::` recording that the
   `seidel_poly_order` entry "read *default 8*" while the signature has shipped
   `6`.  The entry itself is correct and stays; the note about its own past is
   in `docs/history/lumenairy.elements._lens_real.md`.
2. **`lenses.py:120-132`** -- a tombstone for a deleted `_NEWTON_MAX_ITERS = 8`
   whose comment "documented the OPPOSITE of shipped behaviour (*was 12,
   dropped to 8 in v3.5.5*)".  The constant and its comment are both long gone;
   the live part of the block -- a do-not-delete note on the numexpr scaffold
   above it, which `lenses_maslov` and both `__init__` re-exports depend on --
   stayed.
3. **`_lens_traced.py:11023-11035`** -- "the comment that used to sit at this
   site (*vignetting is rare … the spline's natural extrapolation*) was wrong
   twice over".  Both settled facts stayed in the present tense (the launch
   square's corners really are at 1.06 aperture radii; FITPACK really does not
   ignore NaN, and one NaN sample really does return an identically-zero
   field); the retraction moved.

---

## 6. Retired / restated prose assertions

**None retired.  None restated.  No test file was modified for prose.**

The check was mechanical, in two passes:

* **The string-literal cross-check** (sweep 1's method).  Every string constant
  of >= 12 characters in `tests/**` and `validation/**` that appears in a
  module's PRE-relocation text but NOT in its post-relocation text was listed
  and read: **51 candidates, 51 incidental**.  The interesting ones, and why
  each is safe:

  | literal | where | verdict |
  |---|---|---|
  | `'NUMEXPR_AVAILABLE'`, `'_ensure_numexpr_loaded'`, `'_NUMEXPR_MIN_SIZE'`, `'_surface_sag_general'`, `'surface_sag_general'` | `test_niche_audit_w3_elements.py::TestEL4EL5DeadCode` | the assertion is `not hasattr(_lens_traced, name)`; the literals lived only in the DELETED TOMBSTONES that described those absences.  The names are still absent, and `surface_sag_general` is still defined in `lenses.py` (9 occurrences) where it always was. |
  | `'_NEWTON_MAX_ITERS'` | same file | same -- `not hasattr(lenses_mod, ...)`, and the literal was only in `lenses.py`'s tombstone. |
  | `'does NOT hold'` | `test_niche_d14`, `test_niche_d15` (4 sites) | matches a RUNTIME WARNING MESSAGE emitted by `_lens_traced._solve_lstsq_thread_safe` -- an executable string, which the fingerprint gate makes unchangeable.  Flagged only because the same phrase happened to appear in `lenses_maslov`'s docstring. |
  | `'before v5.46'` | `test_audit2609_a10_codev.py`, `test_audit2609_a10_verify.py` (4 sites) | `pytest.warns(match=...)` on `io/prescriptions_code_v.py`'s DIM-M migration warning, not in this partition. |
  | `'used to read'` | `test_niche_audit_w9_dispatch2.py` | a conditional pin on `raytrace/propagate_through_system.__doc__`, not in this partition. |
  | `'decenter_x_m'` / `'decenter_y_m'` (16 sites) | 6 files | `coord_breaks` dict KEYS in test prescriptions.  They appeared in `_lens_real.py` only inside the 4.11.1 note recording that a 4.10.2 patch looked up the wrong key names. |
  | `'the resolver returned '`, `' of the on-beam peak'`, `'multi-process'`, `' from degree '`, `'     samples '` | 6 files | the tests' / probes' own f-string assertion messages and format strings. |
  | two whole docstrings | `validation/repro_traced_carrier_121/_c14_pre_baseline_lens_traced.py` | a FROZEN COPY of the pre-C14 module kept in `validation/` as a byte-identity reference.  Not a pin on the live module, and not mine to touch. |
  | four path strings naming `propagation.py` | `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`, `test_v4_16_3_agent_b.py`, `test_v4_16_3_agent_d.py`, `test_v5_2_walker_pep562_forwarding.py` | `Path` constructions, not prose. |

* **A source-reading sweep** over `tests/unit/*.py`: every file that calls
  `getsource` / `getsourcelines` / `getdoc` / `getcomments` / `__doc__` /
  `read_text` / `read_bytes` AND names one of this partition's modules,
  symbols or classes -- **81 files**, all run (sec. 7).  The load-bearing prose
  pins found on these modules, and verified still satisfied:

  | test | what it pins on my modules | outcome |
  |---|---|---|
  | `test_niche_audit_w3_elements.py::TestEL4EL5DeadCode` | absence of 6 dead names on `_lens_traced` / `lenses` | passes -- strengthened in effect: the tombstones that NAMED them are gone too |
  | `test_v5_4_7_walker_v20_cross_backend_parity.py` | `_intersect_jax` / `_intersect_jax_param` root pick; `backend.random.RandomState` dtype | rewritten structurally, sec. 8B |
  | `test_v5_2_walker_shell_vs_canonical.py` | `propagation.py`'s `from .X import Y` block, by AST | passes -- comments are invisible to it |
  | `test_v5_2_walker_pep562_forwarding.py` | `fft_infra.py`'s source (not `propagation.py`'s prose) | passes |
  | `test_v4_16_3_agent_d.py` | `'Multiprocess / fork notes'` in `fft_infra.py` OR `propagation.py` | passes -- the section lives in `fft_infra.py` |
  | `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py` | a `_register_cache_clearer(...)` Call node in `propagation.py` / `fft_infra.py`, by AST | passes |
  | `test_niche_audit_e_prepared_and_enums.py` | `_lens_traced._NEWTON_MAX_ITERS == 12` | passes -- the constant is untouched, only its 3.5.5/3.5.6 chronology moved |
  | `test_audit2609_a16_lens_arch.py`, `..._lens_config_*` | `LensConfig` / `lens_config.py` docstring contracts | pass |
  | `test_audit2609_a13_staggered_cost.py`, `test_niche_audit_w3_rcwa_pmm.py` | `twod_staggered` / `rcwa` docstring phrases | pass |
  | `test_audit_lens.py` | no bare `1.0 + 0.0j` in `_lens_thin` executable code | passes (not my module) |
  | `test_niche_audit_eh1_maslov_upsample.py` | the `affine_transform` lattice contract in `lenses_maslov` | passes |
  | `test_niche_perf_round2_2026_08_10.py`, `test_fix_newton_pool_memory.py`, `test_niche_newton_pool_both_fits.py` | `_lens_traced` pool / coords sources | pass |
  | `test_obl_banded_halo.py`, `test_tf_banded_halo.py`, `test_lens_chunked_sag.py` | `_lens_real` banded-path sources | pass |

---

## 7. Modules deliberately left alone (partition complete)

**2 of the 22 were not changed at all.**

* **`lumenairy/elements/_lens_imap.py`** -- 0 strict lines.  Its long comment
  blocks are the `_IMAP_CACHE` lock-granularity argument, the probe-point
  fingerprint derivation (with the measured 5x5-lattice failure that forced
  real landings) and the G8 comparative-bar rationale: present-tense
  why-comments and measured derivations of live behaviour, with no release
  chronology in any of them.
* **`lumenairy/ui/lens_options_dialog.py`** -- 0 strict lines, 0 loose lines.
  Nothing to move.

Two more were changed only in ways worth naming:

* **`lumenairy/elements/__init__.py`** -- no history; two `used to` framings on
  the lazy-loading guard were restated in the present tense
  ("a user who only wants `propagate_asm` *would otherwise* pay for the whole
  rigorous-solver stack").
* **`lumenairy/elements/rcwa/twod.py`** -- six tag strips, and its
  `formulation='li'` CORRECTNESS ADVISORY was kept in full.  "Results on lossy
  cells from an earlier release should be re-run" is a live instruction to a
  user, not history; only its framing moved to the present tense.

**The partition is complete: 22 modules enumerated, 20 changed, 2 deliberately
untouched, 0 skipped.**

---

## 8. The two extra deliverables

### 8A. `tests/unit/test_audit2609_a17_history_lint.py` -- the ratchet

The relocation checker proves each MOVE was behaviour-free; it says nothing
about the next comment somebody writes.  A backlog that took four sweeps to
clear will re-accumulate one release note at a time unless a gate notices.

**What it does.**  Counts the finding's own shapes -- `\bv\d+\.\d+(\.\d+)? \(`,
`\bpre-?fix\b`, `\bpre-v?\d`, `used to (say|claim|read)`, `formerly`,
`was wrong`, `superseded`, `re-?scheduled`, `retract` -- per module over
`lumenairy/**/*.py`, restricted to COMMENT and STRING-STATEMENT lines, and
compares each count against a committed baseline:

* a count that **GROWS fails**, and the failure lists the offending lines
  (capped at 12 per module, so three modules growing at once stay readable)
  and names the re-baseline command;
* a count that **shrinks passes** and prints how many lines the tree is ahead
  of its baseline, plus the same command;
* a module **with no baseline entry is held to 0**, so narrative cannot enter
  through a new file either.

**Why a ratchet and not a bar.**  A bar of zero would be wrong: live migration
statements, file-format contracts and the English word *prefix* legitimately
match (sec. 2.1).  The ratchet does not care that a baselined count is 3
rather than 0 -- only that it does not climb.

**Why prose-only.**  An executable `raise` / `warn` message is behaviour, and
several of them legitimately carry retraction-shaped wording (the
`does NOT hold` warning of sec. 6 is one).  A file that does not parse falls
back to a whole-file count rather than being skipped, so "unparseable" is not
a hiding place.

**Re-baselining** -- a normal operation, not a defeat:

```
python tests/unit/test_audit2609_a17_history_lint.py --write
LUMENAIRY_HISTORY_LINT_WRITE=1 python -m pytest tests/unit/test_audit2609_a17_history_lint.py
```

Both rewrite `test_audit2609_a17_history_lint_baseline.json` beside the test.
The diff is one line per module, so a reviewer sees exactly what was conceded.
The file also runs standalone as a report (`python <file>` prints GROWN /
shrunk lines and exits 1 on growth).

**The baseline as shipped**: 122 modules, **688 history lines**, measured on
the finished tree.

**Its own falsifiability** (the audit's V1 requirement):

| test | what it would catch |
|---|---|
| `test_the_baseline_file_is_present_and_non_trivial` | a missing / empty baseline, and a tree walk that found < 100 modules -- either would make the gate vacuous |
| `test_the_counter_actually_counts` | 11 synthetic modules written in `tmp_path`: clean code counts 0; each of the five shapes counts its own line; a STRING STATEMENT counts and an EXECUTABLE string does not; the English word `prefix` counts in a comment and not in code; an unparseable file still counts (whole-file fallback) rather than being skipped |
| `test_the_ratchet_direction_is_enforced` | a comparison that never fails, or one that fails on shrinkage, or one that lets a NEW module in at a nonzero count -- all four directions asserted on synthetic counts |
| `test_the_write_path_round_trips` | a writer that emits a shape the reader cannot parse, which would turn the sanctioned recovery into a broken gate |

End-to-end confirmation on the REAL tree (not shipped as a test, since it has
to doctor the committed baseline): lowering `elements/_lens_thin.py`'s baseline
by one makes the gate fire with
`WP-A17 ratchet: version-history narrative GREW in 1 module(s).
  lumenairy/elements/_lens_thin.py: 23 -> 24`, followed by the offending lines
and the re-baseline command.

### 8B. The V20 JAX root-pick pin, made structural

`test_v5_4_7_walker_v20_cross_backend_parity.py::test_jax_intersect_direction_
aware_root_pick_present` matched the forbidden direction-blind selector as
TEXT:

```python
assert 't1 if R' not in src and 'R_finite > 0' not in src.replace(
    '# ``R_finite > 0`` selector', ''), ...
```

Both failure directions were live.  A comment quoting the forbidden selector
fails the pin -- the test already carried that hand-written `str.replace` to
hide one such comment from itself, which is the tell -- and a REAL regression
spelled `t1 if radius > 0` or `jnp.where(R_safe > 0, t1, t2)` passes untouched.

**Now.**  The check walks each kernel's AST for the root-pick EXPRESSION:

* **required** -- the min-modulus pick `where(abs(a) <= abs(b), a, b)` (with
  the branch names required to be the same two names the `abs` calls wrap), or
  the Spencer-Murty stable-quadratic quotient `e / where(<guard>, q, <const>)`
  (the guarded divisor is the point: it is what makes the quotient the near
  root either way);
* **forbidden** -- a root chosen from the SIGN OF THE RADIUS: `IfExp` or
  `jnp.where` whose test compares a radius/curvature-named operand against a
  zero constant and whose two branches are bare names.

Comments do not exist in an AST, so the first failure direction is gone by
construction; the matcher is keyed on shape rather than spelling, so the
second is too.

**Falsifiability, as the brief asked.**
`test_the_root_pick_matcher_rejects_a_direction_blind_kernel` parses each REAL
kernel, replaces every direction-aware root-pick expression with
`t1 if R_finite > 0 else t2` (an `ast.NodeTransformer` keyed on the expression,
not on the assigned name -- the two kernels spell it `t_near` and `t_sphere`),
and requires the verdict to flip on BOTH counts.  Measured:

```
_intersect_jax       : aware=True  blind=[]  | 2 sites mutated -> aware=False, blind=2
_intersect_jax_param : aware=True  blind=[]  | 1 site  mutated -> aware=False, blind=1
```

`test_the_root_pick_matcher_is_not_a_text_search` asserts the two directions
the text form had, on synthetic sources so neither claim depends on how the
shipped kernels happen to be written today: a kernel whose COMMENT quotes the
forbidden selector passes, and a direction-blind pick spelled
`jnp.where(curvature > 0, t_a, t_b)` fails.

```
pytest tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py
-> 7 passed in 0.49 s          (5 before, 7 after)
```

---

## 9. Appendix -- the 192 in-place strips and present-tense rewrites, verbatim

A release/audit TAG on an otherwise-live why-comment was stripped in place
rather than relocated into a document (sec. 1.3c); a `pre-fix`-framed guard
note was rewritten into the present tense with its original wording kept here.
Nothing but the tag was removed from a pure strip: the remainder of every line
is preserved character-for-character except for capitalising its first letter.
First line of each side shown; the full replacement text is in the module.

| site | stripped (before, first line) | left in the source (after, first line) |
|---|---|---|
| `lumenairy/elements/_lens_traced.py:L128-132` | `# v5.29.1 (audit E-L22): signature defaults of :func:`apply_real_lens_traced`,` | `# Signature defaults of :func:`apply_real_lens_traced`,` |
| `lumenairy/elements/_lens_traced.py:L156` | `v5.29.1 (audit E-H4): a prepared lens must not alias the caller's dict, or` | `A prepared lens must not alias the caller's dict, or` |
| `lumenairy/elements/_lens_traced.py:L253` | `# v5.30 (audit E-M6): post-hoc energy self-check for ``amplitude_model=` | `# Post-hoc energy self-check for ``amplitude_model=` |
| `lumenairy/elements/_lens_traced.py:L294-296` | `# v5.32 (2026-07-31, docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md S5.1):` | `# THE GAIN SIDE IS DELIBERATELY LEFT AT 0.050 (docs/audits/C6_FIT_GUARD_` |
| `lumenairy/elements/_lens_traced.py:L329` | `# ---- v5.32: the HALO-AMPLITUDE term of the ray_density self-check ----------` | `# ---- The HALO-AMPLITUDE term of the ray_density self-check ---------------` |
| `lumenairy/elements/_lens_traced.py:L596-600` | `# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):` | `# Module-level logger for apply_real_lens_traced entry / per-Newton-` |
| `lumenairy/elements/_lens_traced.py:L612-614` | `# apply_real_lens (analytic split-step) is the workhorse for the` | `# apply_real_lens (analytic split-step) is the workhorse for the` |
| `lumenairy/elements/_lens_traced.py:L881-885` | `# The evaluator must EVALUATE in the parent's floating-point order,` | `# The evaluator must EVALUATE in the parent's floating-point order,` |
| `lumenairy/elements/_lens_traced.py:L900-908` | `# v5.33.0 (FIX_POOL_REBUILD): EVALUATE THE PARENT'S FIT, do not re-fit.` | `# EVALUATE THE PARENT'S FIT, do not re-fit.` |
| `lumenairy/elements/_lens_traced.py:L1426-1430` | `# v5.30 (audit E-L2): take the SAME lock the constructor takes.  This` | `# Take the SAME lock the constructor takes.  This` |
| `lumenairy/elements/_lens_traced.py:L3298-3303` | `# v5.32.3 (FIX_CI_POOL): PINNED evaluation backend, or None = "resolve` | `# PINNED evaluation backend, or None = "resolve` |
| `lumenairy/elements/_lens_traced.py:L3729-3732` | `# v5.1.0 (default-knob resolver rollout): real-dtype OPL allocator` | `# Real-dtype OPL allocator` |
| `lumenairy/elements/_lens_traced.py:L3826-3827` | `# vs 0.008 rad).  Exposed as a module flag so the regression test can force the` | `# vs 0.008 rad).  Exposed as a module flag so the regression test can` |
| `lumenairy/elements/_lens_traced.py:L4118` | `#: WHAT WAS WRONG.  ``_DECENTRE_GATE_W_FRAC`` above is a floor set to kill a` | `#: WHY NOT THE GATE.  ``_DECENTRE_GATE_W_FRAC`` above is a floor set to kill a` |
| `lumenairy/elements/_lens_traced.py:L4934-4937` | `the query positions -- v5.25.1 (hammer H6): the per-ray OPL must be` | `the query positions.  ``w_fn`` is the H6 term: the per-ray OPL must be` |
| `lumenairy/elements/_lens_traced.py:L5003-5004` | `# the direction cosines and (the larger of the two, and the one the` | `# the direction cosines and (the larger of the two) the H6` |
| `lumenairy/elements/_lens_traced.py:L5442-5443` | `#: all 1e-3 or less of the C3 bound).  **The counter-evidence was real and it` | `#: all 1e-3 or less of the C3 bound).` |
| `lumenairy/elements/_lens_traced.py:L5754-5755` | `#: ---------------------------------------------------------------------------` | `#: ---------------------------------------------------------------------------` |
| `lumenairy/elements/_lens_traced.py:L6503-6504` | `# ONE np.power per distinct exponent -- the same call the old loop made` | `# ONE np.power per distinct exponent -- the same call a per-term,` |
| `lumenairy/elements/_lens_traced.py:L7167-7168` | ```surfaces_from_prescription`` reads it for vignetting), and anything` | ```surfaces_from_prescription`` reads it for vignetting), and anything` |
| `lumenairy/elements/_lens_traced.py:L7954-7956` | `trace through the reversed prescription.  ``amplitude_model=` | `trace through the reversed prescription.  ``amplitude_model=` |
| `lumenairy/elements/_lens_traced.py:L8707-8709` | `# v4.15.3 (P0-NEW-F2-1): defensive guard via the shared` | `# Defensive guard via the shared` |
| `lumenairy/elements/_lens_traced.py:L9116-9118` | `# v5.1.0 (default-knob resolver rollout): resolve ``wave_propagator``` | `# Resolve ``wave_propagator``` |
| `lumenairy/elements/_lens_traced.py:L9184-9189` | `# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):` | `# Entry log -- grid size + surface count + Newton iter cap so users` |
| `lumenairy/elements/_lens_traced.py:L9481-9491` | `# v5.44 (AUDIT_TRACED_MEMORY_2026_08_09 row 3, closed): ``amplitude_model=` | `# ``amplitude_model=` |
| `lumenairy/elements/_lens_traced.py:L9766-9772` | `# v5.29.1 (audit E-L22): the model swap DROPS every` | `# The model swap DROPS every` |
| `lumenairy/elements/_lens_traced.py:L9773-9786` | `# v5.35.0 (BUILD_R1_WIRING S4): the carrier is the ONE` | `# The carrier is the ONE` |
| `lumenairy/elements/_lens_traced.py:L9879-9884` | `# v5.29.1 (audit E-M2): forward the RAW ``sag_chunk_rows``,` | `# Forward the RAW ``sag_chunk_rows``,` |
| `lumenairy/elements/_lens_traced.py:L9992-9995` | `# v5.17.2 (audit P2-21): honour a pinned set_max_ram() budget --` | `# Honour a pinned set_max_ram() budget --` |
| `lumenairy/elements/_lens_traced.py:L10821` | `# ---- v5.25.1 (hammer audit H6): carrier entrance eikonal -----------` | `# ---- Carrier entrance eikonal (hammer audit H6) -------------------` |
| `lumenairy/elements/_lens_traced.py:L10849` | `# ---- v5.25.0 (hammer audit H3): exit-NA Nyquist guard --------------` | `# ---- Exit-NA Nyquist guard (hammer audit H3) ----------------------` |
| `lumenairy/elements/_lens_traced.py:L11176` | `# there so the degenerate path keeps its pre-fix behaviour exactly.` | `# there so the degenerate path is unchanged.` |
| `lumenairy/elements/_lens_traced.py:L11517-11519` | `# no per-pixel Newton.  A GLOBAL Chebyshev fit (vs the pre-3.x griddata` | `# no per-pixel Newton.  A GLOBAL Chebyshev fit (rather than a griddata` |
| `lumenairy/elements/_lens_traced.py:L11715` | `# This stencil is strictly better than the previous hard-coded 1.10` | `# This stencil is strictly better than a hard-coded 1.10` |
| `lumenairy/elements/_lens_traced.py:L11732` | `M_x = M_y = 0.91  # fallback to pre-3.1.3 heuristic (1/1.10)` | `M_x = M_y = 0.91  # fallback heuristic (1/1.10)` |
| `lumenairy/elements/_lens_traced.py:L11800-11804` | `-- the old message said "100.0% did not converge ... increase` | `-- a "100.0% did not converge ... increase newton_max_iters ...` |
| `lumenairy/elements/_lens_traced.py:L11870-11875` | `# the already-traced ray grid -- no extra compute).  This is a` | `# the already-traced ray grid -- no extra compute).  This is a` |
| `lumenairy/elements/_lens_traced.py:L11893-11895` | `# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration` | `# Per-iteration telemetry: emit a "converged" marker so an` |
| `lumenairy/elements/_lens_traced.py:L11943-11948` | `# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration` | `# Per-iteration telemetry: per-Newton-iteration log, independent` |
| `lumenairy/elements/_lens_traced.py:L11950` | `# v5.4 (audit P3): deduplicate -- reuse res from convergence check above` | `# Reuse res from the convergence check above` |
| `lumenairy/elements/_lens_traced.py:L12646-12653` | `# v5.44 (AUDIT_TRACED_MEMORY_2026_08_09 row 3): the model is evaluated` | `# The model is evaluated` |
| `lumenairy/elements/_lens_traced.py:L12763-12771` | `# v5.17.1 (audit P3-09): on the sub>1 preserve_input_phase` | `# On the sub>1 preserve_input_phase` |
| `lumenairy/elements/_lens_traced.py:L13221-13225` | `# NOT applied on the experimental ``inversion_method='backward_trace'``` | `# NOT applied on the experimental ``inversion_method='backward_trace'``` |
| `lumenairy/elements/_lens_traced.py:L13294` | `# ---- v5.30 (audit E-M6): post-hoc ENERGY SELF-CHECK ------------------` | `# ---- Post-hoc ENERGY SELF-CHECK ------------------------------` |
| `lumenairy/elements/_lens_traced.py:L13493-13509` | `# v5.44 (AUDIT_TRACED_MEMORY_2026_08_09 row 3, closed).  The loop now` | `# The loop also serves ``amplitude_model='ray_density'`` (the` |
| `lumenairy/elements/_lens_traced.py:L14637-14638` | `does NOT move a prepared lens -- rebuild it (audit E-H4; see that` | `does NOT move a prepared lens -- rebuild it (audit E-H4; see that` |
| `lumenairy/elements/_lens_traced.py:L14851-14854` | `# v5.29.1 (audit E-H4): resolve the process-wide defaults NOW and store the` | `# Resolve the process-wide defaults NOW and store the` |
| `lumenairy/elements/_lens_real.py:L89-125` | `v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D2).  ``numexpr.evaluate`` is` | ```numexpr.evaluate`` is` |
| `lumenairy/elements/_lens_real.py:L1900-1903` | `# Resample the input field DEMODULATED by the traced congruence, so the` | `# Resample the input field DEMODULATED by the traced congruence, so the` |
| `lumenairy/elements/_lens_real.py:L3976-3978` | `# Convergence is scored over the SUPPORT, for the same reason the fold` | `# Convergence is scored over the SUPPORT, for the same reason the fold` |
| `lumenairy/elements/_lens_real.py:L5366-5368` | `and stacking the two would double-count.  With ``carrier=None``` | `and stacking the two would double-count.  With ``carrier=None``` |
| `lumenairy/elements/_lens_real.py:L5525-5530` | `# v4.15.3 (P0-NEW-F2-1): the defensive input guard runs FIRST --` | `# The defensive input guard runs FIRST --` |
| `lumenairy/elements/_lens_real.py:L5621-5624` | `# v5.1.0 (default-knob resolver rollout): when ``wave_propagator``` | `# When ``wave_propagator``` |
| `lumenairy/elements/_lens_real.py:L5628-5629` | `# v5.1.0 (default-knob resolver rollout): same for ``dy``.` | `# Same for ``dy``.` |
| `lumenairy/elements/_lens_real.py:L5668-5670` | `# v5.35.0: the screen-obliquity correction + its accuracy guard.  Reached` | `# The screen-obliquity correction + its accuracy guard.  Reached ONLY` |
| `lumenairy/elements/_lens_real.py:L5710-5714` | `# Anamorphic-safe: pass BOTH axes.  ``shape[0]`` is Ny, so pairing it` | `# Anamorphic-safe: pass BOTH axes.  ``shape[0]`` is Ny, so pairing it` |
| `lumenairy/elements/_lens_real.py:L6537-6538` | `# v4.13.2 (audit C-P1-4): dtype-aware zero to preserve complex64` | `# Dtype-aware zero to preserve complex64` |
| `lumenairy/elements/_lens_real.py:L6630-6633` | `# v5.35.3: ``carrier=`` (the angle-true screen) no longer disqualifies` | `# ``carrier=`` (the angle-true screen) does NOT disqualify the band` |
| `lumenairy/elements/_lens_real.py:L6770-6773` | `# v5.35.3: ``carrier=`` no longer disqualifies this band either -- the` | `# ``carrier=`` does not disqualify this band either -- the` |
| `lumenairy/elements/_lens_real.py:L6996-7008` | `# v5.2 (ROADMAP v5.1 off-axis conic in surface frame;` | `# When ``surface_frame=True``` |
| `lumenairy/elements/_lens_real.py:L7176-7185` | `# ---- Form error map -------------------------------------------` | `# ---- Form error map -------------------------------------------` |
| `lumenairy/elements/_lens_real.py:L7290-7296` | `# v5.25.1 (hammer H2(a)): ray-angle-aware refraction OPD` | `# Ray-angle-aware refraction OPD` |
| `lumenairy/elements/_lens_real.py:L7319-7322` | `# ONE reduction over ``sag``, shared by the tangent-facet block,` | `# ONE reduction over ``sag``, shared by the tangent-facet block,` |
| `lumenairy/elements/_lens_real.py:L7797-7807` | `# Suppress fitting noise: if the RMS correction across the` | `# Suppress fitting noise: if the RMS correction across the` |
| `lumenairy/elements/lenses_maslov.py:L37-42` | `# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):` | `# The three Chebyshev Vandermonde helpers live in` |
| `lumenairy/elements/lenses_maslov.py:L182` | `#    Scaling the coordinate axes by eigenvalues instead (the pre-fix code)` | `#    Scaling the coordinate axes by eigenvalues instead` |
| `lumenairy/elements/lenses_maslov.py:L1198-1199` | `deterministic, unique function of ``(A, RHS)``.  That is also the` | `deterministic, unique function of ``(A, RHS)`` -- the conservative` |
| `lumenairy/elements/lenses_maslov.py:L1614-1615` | `0.2 mm-aperture singlet at d = 0.5 ... 5 mm, not to the ~1e-10 claimed` | `0.2 mm-aperture singlet at d = 0.5 ... 5 mm.  The residual gap is not` |
| `lumenairy/elements/lenses_maslov.py:L1626-1627` | `absolute amplitude scale.  Since v5.46 (audit S4) the raw integral is` | `absolute amplitude scale.  The raw integral is ALREADY absolutely` |
| `lumenairy/elements/lenses_maslov.py:L1677-1679` | `# v4.15.3 (P0-NEW-F2-1): defensive guard via the shared` | `# Defensive guard via the shared` |
| `lumenairy/elements/lenses_maslov.py:L1699-1705` | `# v5.21: fold_split=True auto-handles a folded prescription instead of` | `# fold_split=True auto-handles a folded prescription instead of` |
| `lumenairy/elements/lenses_maslov.py:L1831-1835` | `# v5.20 (GPU): CuPy dispatch mirrors apply_real_lens -- opt in via` | `# CuPy dispatch mirrors apply_real_lens -- opt in via` |
| `lumenairy/elements/lenses_maslov.py:L2015-2016` | `# acceptance NA and the INPUT field's angular content.  Sizing from` | `# acceptance NA and the INPUT field's angular content.  Sizing from` |
| `lumenairy/elements/lenses_maslov.py:L2273-2276` | `# v5.21 (M-P follow-up): the fit-residual RMS diagnostics are only ever read` | `# The fit-residual RMS diagnostics are only ever read` |
| `lumenairy/elements/lenses_maslov.py:L2473-2476` | `# v4.14.1 (audit P2-6): dtype-aware out-of-bounds sentinel so` | `# Dtype-aware out-of-bounds sentinel so` |
| `lumenairy/elements/lenses_maslov.py:L2712-2714` | `# tilt is rescaled by the same factor: a flat prism's recovered` | `# tilt is rescaled by the same factor: a flat prism's recovered` |
| `lumenairy/elements/lenses_maslov.py:L2992-2993` | `# it returns EXACTLY zero on a real, non-negative (flat-phase) input --` | `# it returns EXACTLY zero on a real, non-negative (flat-phase) input` |
| `lumenairy/elements/lenses_maslov.py:L4212-4214` | `# v5.21 (M-P8): one shared-basis value+1st-deriv kernel for the three` | `# One shared-basis value+1st-deriv kernel for the three` |
| `lumenairy/propagators/gbd.py:L261-266` | `# v5.4.7 (audit AUDIT_V5_4_6 #9): Q is the ENGINEERING 1/q parameter` | `# Q is the ENGINEERING 1/q parameter (q_code = conj(q_physics)); at the` |
| `lumenairy/propagators/gbd.py:L729-732` | `# v5.30 (audit P8): ``wavelength`` is deprecated on the` | `# ``wavelength`` is deprecated on the` |
| `lumenairy/propagators/gbd.py:L1369-1373` | `# v5.21 (per-surface GBD): a beamlet bundle may carry a (N, 2, 2) complex-` | `# A beamlet bundle may carry a (N, 2, 2) complex-` |
| `lumenairy/propagators/gbd.py:L3366-3371` | `# v5.21 (#3): 'auto' (the default) prefers the truncation-free analytic` | `# 'auto' (the default) prefers the truncation-free analytic` |
| `lumenairy/memory.py:L745-753` | `# v5.46 (audit Z3 / VERIFY-A11 O-2): CLOSED vocabulary.  The branch below` | `# CLOSED vocabulary.  The branch below` |
| `lumenairy/memory.py:L787-793` | `# v5.17.2 (audit P2-22): the runtime row-band path still runs the` | `# The runtime row-band path still runs the` |
| `lumenairy/memory.py:L813-814` | `# v5.46 (audit Z3): the two entry points carry DIFFERENT array counts, each` | `# The two entry points carry DIFFERENT array counts, each` |
| `lumenairy/memory.py:L924-930` | `# v5.33.2: a plan key holds TWO aligned workspaces only while the` | `# A plan key holds TWO aligned workspaces only while the` |
| `lumenairy/memory.py:L1158-1162` | `# v5.17.2 (audit P2-23 / P3-46): the values captured at the FIRST` | `# The values captured at the FIRST` |
| `lumenairy/memory.py:L1172-1178` | `# v5.30.1 (audit W9): tracks the fft_infra default, which flipped` | `# Tracks the fft_infra default, which flipped` |
| `lumenairy/memory.py:L1234-1236` | `# audit P3-46: read the LIVE value like the other knobs (pre-fix` | `# audit P3-46: read the LIVE value like the other knobs --` |
| `lumenairy/elements/_lens_jax.py:L432-435` | `:func:`apply_real_lens` (NumPy) for anamorphic grids -- it` | `:func:`apply_real_lens` (NumPy) for anamorphic grids -- it` |
| `lumenairy/elements/_lens_jax.py:L558-563` | `# v4.13.2 (audit P1-NEW-E): the JAX twin's ray-subsample +` | `# The JAX twin's ray-subsample +` |
| `lumenairy/elements/_lens_jax.py:L653-659` | `# v5.4.6 (audit F-7): 'xy' indexing so the wave-grid axis order matches` | `# 'xy' indexing so the wave-grid axis order matches` |
| `lumenairy/elements/_lens_jax.py:L811-812` | `# v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard` | `# Explicit mirror-in-surfaces guard, ported from` |
| `lumenairy/elements/_lens_jax.py:L851-853` | `# v4.13.2 (audit P1-NEW-E): same square-grid constraint as the` | `# Same square-grid constraint as the` |
| `lumenairy/elements/_lens_jax.py:L917-923` | `# v5.4.6 (audit F-7): 'xy' indexing so the wave-grid axis order matches` | `# 'xy' indexing so the wave-grid axis order matches` |
| `lumenairy/elements/_lens_jax.py:L981-983` | `# v4.13.0 (audit L2): unify on the library-wide default dtype.` | `# Unify on the library-wide default dtype, and pass ``E_in.dtype``` |
| `lumenairy/elements/pmm/twod_staggered.py:L44` | `BIT-IDENTICAL to the pre-2026-09-11 library -- or an increasing ``(N + 1,)``` | `BIT-IDENTICAL to the uniform-only formulation it generalises -- or an increasing ``(N + 1,)``` |
| `lumenairy/elements/pmm/twod_staggered.py:L875` | `#: ``False`` restores the pre-2026-09-11 acceptance -- any strictly increasing` | `#: ``False`` restores the UNCONSTRAINED acceptance -- any strictly increasing` |
| `lumenairy/elements/pmm/twod_staggered.py:L972-973` | `#: exemption is what keeps gate N1 (integer walls BIT-IDENTICAL to the` | `#: exemption is what keeps gate N1 (integer walls BIT-IDENTICAL to the` |
| `lumenairy/elements/pmm/twod_staggered.py:L1320` | `this class builds is BIT-IDENTICAL to the pre-2026-09-11 library -- or an` | `this class builds is BIT-IDENTICAL to the uniform-only formulation -- or an` |
| `lumenairy/elements/pmm/twod_staggered.py:L2665-2666` | `# every integer-``N`` projector is bit-identical to the pre-2026-09-11` | `# every integer-``N`` projector is bit-identical to the uniform-only` |
| `lumenairy/elements/pmm/twod_staggered.py:L2754-2755` | `the ``_proj`` closure formerly duplicated in :mod:`.stack2d_pure` and this` | `the ONE ``_proj`` closure shared by :mod:`.stack2d_pure` and this` |
| `lumenairy/elements/pmm/twod_staggered.py:L2757-2758` | ```[Ex_orders; Ey_orders]``.  Reproduces the former closure operation-for-` | ```[Ex_orders; Ey_orders]``.  Reproduces the two former inline copies` |
| `lumenairy/elements/pmm/twod_staggered.py:L2766-2769` | `the safe-divide ``kz`` used by the longitudinal-field reconstruction -- the` | ```kz_ref``/``kz_trn``/``kz_inc``/``safe_r``/``safe_t`` block -- the ONE` |
| `lumenairy/elements/pmm/twod_staggered.py:L3482-3485` | `.. note:: **API change (v5.11 -> v5.12).**  Formerly a bare 4-tuple` | `.. note:: **API change (v5.11 -> v5.12).**  This returns the` |
| `lumenairy/elements/pmm/twod_staggered.py:L3694-3695` | `# cross-suite return shape: unpacks as (orders, R, T); .dof = 2*q^2 (the modal` | `# cross-suite return shape: unpacks as (orders, R, T); .dof = 2*q^2 (the modal` |
| `lumenairy/propagators/propagation.py:L46-49` | `v5.1.0 split (Agent C)` | `The submodules behind this shell` |
| `lumenairy/propagators/propagation.py:L63-66` | `This module is now a thin re-export shell.  Every name that pre-v5.1.0` | `This module is a thin re-export shell.  Every name documented as` |
| `lumenairy/propagators/propagation.py:L94-96` | `# change at runtime.  The ``from X import Y`` bindings above still` | `# change at runtime.  The ``from X import Y`` bindings above still` |
| `lumenairy/propagators/propagation.py:L127` | `# Re-exports from the v5.1.0 infrastructure / kernel submodules.` | `# Re-exports from the infrastructure / kernel submodules.` |
| `lumenairy/propagators/propagation.py:L130` | `# The public surface of pre-v5.1.0 ``propagation.py`` is exactly the` | `# The public surface of this shell is exactly the` |
| `lumenairy/propagators/propagation.py:L138-141` | `# ``fft_infra.DEFAULT_*`` but NOT this module's local binding -- which` | `# ``fft_infra.DEFAULT_*`` but NOT this module's local binding -- which` |
| `lumenairy/propagators/propagation.py:L164` | `# v5.31 (audit W9-8): frozen factory value; immutable, so no live forward` | `# Frozen factory value; immutable, so no live forward` |
| `lumenairy/propagators/propagation.py:L250` | `# v5.1.1 (audit P3-NEW-F1-1): ``_PYFFTW_BAD_SHAPES`` belongs here.` | `# ``_PYFFTW_BAD_SHAPES`` belongs here.` |
| `lumenairy/propagators/propagation.py:L326` | `# v5.1.0 (V9 walker symmetry): ``angular_spectrum_propagate_batch``` | `# ``angular_spectrum_propagate_batch``` |
| `lumenairy/elements/lenses.py:L301` | `# v5.31 (audit R-8 / E-L7 residual): reject ODD powers HERE, at the` | `# Reject ODD powers HERE, at the` |
| `lumenairy/elements/lenses.py:L454` | `# v5.31 (audit R-8 / E-L7 residual): reject ODD powers on BOTH per-axis` | `# Reject ODD powers on BOTH per-axis` |
| `lumenairy/elements/lenses.py:L475` | `# v5.4.6 (audit F-19): outside the conic domain` | `# Outside the conic domain` |
| `lumenairy/elements/lenses.py:L1049` | `# compatibility shell whose whole job is to keep the pre-v3.5.5 import paths` | `# compatibility shell whose whole job is to keep the legacy import paths` |
| `lumenairy/elements/rcwa/twod.py:L349-356` | ```EZZ = E = [[eps]]`` (the DIRECT-rule ``E_z`` elimination, v5.14.1 audit` | ```EZZ = E = [[eps]]`` (the DIRECT-rule ``E_z`` elimination, audit` |
| `lumenairy/elements/rcwa/twod.py:L869-876` | `NOTE (v5.14.1, audit F1): the pre-v5.14.1 2-D ``'li'`` applied the` | `NOTE (audit F1, v5.14.1).  A 2-D ``'li'`` that applies the` |
| `lumenairy/elements/rcwa/twod.py:L884-886` | `rule (``EZZ = [[eps]]``, matching the analytic-shape solver) since` | `rule (``EZZ = [[eps]]``, matching the analytic-shape solver),` |
| `lumenairy/elements/rcwa/twod.py:L1117-1121` | `# (Cxy = Cyx = 0).  The pre-v5.14.1 'li' (Laurent in-plane + [[1/eps]]` | `# (Cxy = Cyx = 0).  A 'li' built as Laurent in-plane + [[1/eps]]` |
| `lumenairy/elements/rcwa/twod.py:L1722-1723` | `uniaxials, magneto-optic media) are supported since v5.14.1 (audit` | `uniaxials, magneto-optic media) are supported (audit GAP2): the` |
| `lumenairy/elements/rcwa/twod.py:L1745` | `# OUT-OF-PLANE tensors are SUPPORTED since v5.14.1 (audit GAP2) via the` | `# OUT-OF-PLANE tensors are SUPPORTED (audit GAP2) via the` |
| `lumenairy/propagators/fga.py:L842-849` | `# v5.24.4 (audit hygiene): thread-safety lock for ``_COARSE_CACHE``.  The` | `# Thread-safety lock for ``_COARSE_CACHE``.  The` |
| `lumenairy/propagators/fga.py:L865-869` | `# v5.24.4 (audit hygiene): enroll the coarse-trace clearer with the central` | `# Enroll the coarse-trace clearer with the central` |
| `lumenairy/propagators/fga.py:L1905-1906` | ```nsig`` is now READ (audit P9).  Pre-v5.30 it was documented as` | ```nsig`` is now READ (audit P9).  It was documented as` |
| `lumenairy/propagators/fga.py:L3089-3090` | `unharmed.  Pass ``method_kwargs={'traced': {'fit_radius_beam_factor': None}}``` | `unharmed.  Pass ``method_kwargs={'traced': {'fit_radius_beam_factor': None}}``` |
| `lumenairy/propagators/fga.py:L3177-3182` | `# v5.31 (audit W9-13): adopt the P2 aperture:beam cliff GUARD as` | `# Adopt the P2 aperture:beam cliff GUARD as` |
| `lumenairy/elements/rcwa/oned.py:L1328` | `# hand-written ``eps * np.eye(3)`` (a scalar formerly raised IndexError on` | `# hand-written ``eps * np.eye(3)`` (a scalar would otherwise raise IndexError` |
| `lumenairy/elements/rcwa/_core.py:L4820-4821` | `# v5.17.1 (audit P2-16/P2-17): bounded LRU OrderedDict (was a plain unbounded` | `# Bounded LRU OrderedDict, not a plain unbounded dict (audit` |
| `lumenairy/elements/_lens_traced_multibranch.py:L147` | `# power-of-two classes (the pre-v5.46 grouping) instead.  Real maps are far` | `# power-of-two classes instead.  Real maps are far` |
| `lumenairy/elements/_lens_traced_multibranch.py:L165-166` | `grouped by rounding each axis UP to the next power of two -- the` | `# grouped by rounding each axis UP to the next power of two, which` |
| `lumenairy/elements/_lens_traced_uniform.py:L815-820` | `# v5.30 (audit E-M15): name THIS function first.  The message used to` | `# Name THIS function first.  A message naming only` |
| `lumenairy/elements/lenses_gbd.py:L456-461` | `# v5.30 (audit E-L21): validate ``jacobian`` HERE, at the entry point, and` | `# Validate ``jacobian`` HERE, at the entry point, and flag it when it` |
| `lumenairy/elements/lenses_gbd.py:L568-574` | `# v5.30 (audit bba1bc4 follow-up): pass ``None`` -- not ``0.0`` -- when` | `# Pass ``None`` -- not ``0.0`` -- when` |
| `lumenairy/elements/lens_config.py:L72` | `* neither -> the entry point's own default, i.e. the historical behaviour.` | `* neither -> the entry point's own default.` |
| `lumenairy/elements/lens_config.py:L78-79` | `Author: Andrew Traverso -- v5.45.2 (audit 2026-09-11 TESTS-ARCH, section 14` | `Author: Andrew Traverso` |
| `lumenairy/backend/__init__.py:L21-23` | `(``available_cpus`` -- the affinity-aware process CPU count -- was` | `(``available_cpus`` -- the affinity-aware process CPU count -- lives in` |
| `lumenairy/elements/__init__.py:L161-164` | `# ``lumenairy.backend.scipy``, and a user who only wants ``propagate_asm``` | `# ``lumenairy.backend.scipy``, and a user who only wants ``propagate_asm``` |
| `lumenairy/elements/__init__.py:L193` | `#: One entry per name that used to be imported eagerly above; the ``__all__``` | `#: One entry per lazily-resolved public name; the ``__all__``` |
| `lumenairy/__init__.py:L50` | `# v5.45.2 (audit 2026-09-11 V6 / WP-A7): the meshgrid and Zernike-basis` | `# The meshgrid and Zernike-basis` |
| `lumenairy/__init__.py:L81` | `# v5.45.2 (audit 2026-09-11 V6 / WP-A7): the masked 2-D unwrap kernel is` | `# The masked 2-D unwrap kernel is` |
| `lumenairy/__init__.py:L122` | `# v4.12.2: expose the through_focus_scan_jax kernel-cache clear helper` | `# Expose the through_focus_scan_jax kernel-cache clear helper` |
| `lumenairy/__init__.py:L196` | `# v5.21 (__all__-symmetry): Maslov vector entry point + the caustic-uniform` | `# Maslov vector entry point + the caustic-uniform` |
| `lumenairy/__init__.py:L201` | `# v5.46 (audit S2 follow-up): the local_quadrature sample-lattice cache.` | `# The local_quadrature sample-lattice cache.` |
| `lumenairy/__init__.py:L214` | `# v4.16.0 (ROADMAP #14): per-glass Sellmeier validity ranges.` | `# Per-glass Sellmeier validity ranges.` |
| `lumenairy/__init__.py:L256` | `# v4.16.3 (audit P3-NEW-F2-LOW-1): sibling re-exports for the` | `# Sibling re-exports for the` |
| `lumenairy/__init__.py:L264` | `# v5.31 (audit W9-8): the frozen factory value propagate() compares the` | `# The frozen factory value propagate() compares the` |
| `lumenairy/__init__.py:L331` | `# v4.15 (ROADMAP v4.16 #9, #11): Schell-model + annular-incoherent` | `# Schell-model + annular-incoherent` |
| `lumenairy/__init__.py:L333` | `# v4.15.1 (P0-NEW-2): the factories now return ensembles (or a` | `# The factories now return ensembles (or a` |
| `lumenairy/__init__.py:L348-349` | `# v5.2 (ROADMAP v5.1 partial-coherence/MCF public-API polish):` | `# Short top-level alias for symmetry with ``lumenairy.coherence_at``` |
| `lumenairy/__init__.py:L357` | `# v4.16.0 (ROADMAP #15): retires the lazy-import fan-out in` | `# Retires the lazy-import fan-out in` |
| `lumenairy/__init__.py:L377` | `# v5.45.2 (audit 2026-09-11 TESTS-ARCH P2-5): the GENERIC scoped form.` | `# The GENERIC scoped form.` |
| `lumenairy/__init__.py:L397` | `# v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper): canonical` | `# Canonical` |
| `lumenairy/__init__.py:L402` | `# v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A): canonical` | `# Canonical` |
| `lumenairy/__init__.py:L688` | `# v4.16.0 (Agent A __all__-symmetry walker): the pymoo` | `# The pymoo` |
| `lumenairy/__init__.py:L714` | `# v5.45.2 (audit 2026-09-11 / WP-A10): edge-thickness merit + the free` | `# Edge-thickness merit + the free` |
| `lumenairy/__init__.py:L722` | `# v4.16 (ROADMAP #11): multi-objective Pareto wrapper (pymoo-optional)` | `# Multi-objective Pareto wrapper (pymoo-optional)` |
| `lumenairy/__init__.py:L765` | `# v5.45.2 (audit 2026-09-11 Y2 follow-up / WP-A4): the aberration-free` | `# The aberration-free` |
| `lumenairy/__init__.py:L795` | `# v4.15 (ROADMAP v4.15 #3): ergonomic prescription + Source ->` | `# Ergonomic prescription + Source ->` |
| `lumenairy/__init__.py:L809` | `# v5.45.2 (audit 2026-09-11 section 15.1 / WP-A1): the single shared` | `# The single shared` |
| `lumenairy/__init__.py:L820` | `# v5.21 (__all__-symmetry): differential ray transfer (GBD analytic` | `# Differential ray transfer (GBD analytic` |
| `lumenairy/__init__.py:L858` | `# v4.15.1 (Cluster B Item 6): wave -> ray bridge.` | `# Wave -> ray bridge.` |
| `lumenairy/__init__.py:L894` | `# v5.45.2 (audit 2026-09-11 section 15.1): the two lower-level pieces of the` | `# The two lower-level pieces of the` |
| `lumenairy/__init__.py:L905` | `# v5.45.2 (audit 2026-09-11 R2 / WP-A7): the functional form of` | `# The functional form of` |
| `lumenairy/__init__.py:L917` | `# v5.3 (AUDIT_V5_2_5 P3-6 closure): the v5.2.5 rename to the` | `# The v5.2.5 rename to the` |
| `lumenairy/__init__.py:L929-933` | `# 4.12.0: wire ``_deprecation.deprecated_alias`` (added in 4.7 but never` | `# ``_deprecation.deprecated_alias`` is wired into the top-level` |
| `lumenairy/__init__.py:L997` | `# v4.16.1 (audit AUDIT_V4_16_0_DEEP P5 / P0-1): partial-coherence` | `# Partial-coherence` |
| `lumenairy/__init__.py:L1152` | `# v4.15 (ROADMAP v4.16 #9, #11): Schell-model + annular-incoherent.` | `# Schell-model + annular-incoherent.` |
| `lumenairy/__init__.py:L1153` | `# v4.15.1 (P0-NEW-2): factories now return ensembles + new MCF class.` | `# Factories now return ensembles + new MCF class.` |
| `lumenairy/__init__.py:L1158` | `# v5.2: short alias for symmetry with the other top-level` | `# Short alias for symmetry with the other top-level` |
| `lumenairy/__init__.py:L1275` | `# v4.16.0 (ROADMAP #14): per-glass Sellmeier validity ranges.` | `# Per-glass Sellmeier validity ranges.` |
| `lumenairy/__init__.py:L1339` | `# v4.16.1: partial-coherence ensemble propagator helper.` | `# Partial-coherence ensemble propagator helper.` |
| `lumenairy/__init__.py:L1460` | `# v4.15 (ROADMAP v4.15 #3): ergonomic prescription -> result entry.` | `# Ergonomic prescription -> result entry.` |
| `lumenairy/__init__.py:L1502` | `# v5.21: differential ray transfer (GBD analytic-Jacobian primitive)` | `# Differential ray transfer (GBD analytic-Jacobian primitive)` |
| `lumenairy/__init__.py:L1510` | `# v5.45.2 (audit 2026-09-11 section 15.1): the ONE shared exit-vertex` | `# The ONE shared exit-vertex` |
| `lumenairy/__init__.py:L1534` | `# v4.15.1 (Cluster B Item 6): wave -> ray bridge.` | `# Wave -> ray bridge.` |
| `lumenairy/__init__.py:L1590` | `# v5.45.2 (audit 2026-09-11 / WP-A7): the public masked 2-D unwrap.` | `# The public masked 2-D unwrap.` |
| `lumenairy/__init__.py:L1645` | `# v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper):` | `(line deleted)` |
| `lumenairy/__init__.py:L1647` | `# v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A):` | `(line deleted)` |
| `lumenairy/__init__.py:L1691` | `# v5.21: traced-lens geometry optimizer (jax geometry-gradient loop)` | `# Traced-lens geometry optimizer (jax geometry-gradient loop)` |
| `lumenairy/__init__.py:L1693` | `# v4.16 (ROADMAP #11): multi-objective Pareto wrapper (pymoo-optional)` | `# Multi-objective Pareto wrapper (pymoo-optional)` |
| `lumenairy/__init__.py:L1723` | `# v5.45.2 (audit 2026-09-11 / WP-A10): edge-thickness merit + oracle.` | `# Edge-thickness merit + oracle.` |
| `lumenairy/__init__.py:L1749` | `# v5.45.2 (audit 2026-09-11 Y2 follow-up / WP-A4): the aberration-free` | `# The aberration-free` |
| `lumenairy/__init__.py:L1851` | `# v5.45.2 (audit 2026-09-11 G7 / WP-A13): 2-D order-count drift signal.` | `# 2-D order-count drift signal.` |
| `lumenairy/__init__.py:L1973` | `# v4.16.0 (Agent A __all__-symmetry walker): the backend` | `# The backend` |
| `lumenairy/__init__.py:L1996` | `# v4.16.3 (audit P3-NEW-F2-LOW-1): sibling parity with` | `# Sibling parity with` |
| `lumenairy/__init__.py:L2022` | `# v5.45.2 (audit 2026-09-11 / WP-A7): the byte-accounting siblings of the` | `# The byte-accounting siblings of the` |
| `lumenairy/__init__.py:L2034` | `# v4.16.0 (ROADMAP #15): central cache-clearer registry.` | `# Central cache-clearer registry.` |
| `lumenairy/__init__.py:L2067` | `# v5.45.2 (audit 2026-09-11 TESTS-ARCH P2-5): the generic scoped form for` | `# The generic scoped form for` |

**192 in-place strips / present-tense rewrites across 20 modules.**
---

## 10. CHANGELOG.md line-citation drift (reported, NOT edited)

`CHANGELOG.md` carries **11** `file.py:NNN` citations into this partition.
Each was re-located in the delivered file by matching the cited anchor line
through a `difflib` opcode map of the pre- vs post-relocation module:

| CHANGELOG.md line | citation | now at | drift | anchor line | verdict |
|---|---|---|---|---|---|
| 5841 | `rcwa/_core.py:1072` | 1072 | 0 | `f"2026-09-11 (the modal branch cut), so reaching this line on such "` | ok |
| 5845 | `rcwa/_core.py:1020` | 1020 | 0 | ``and ``docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md``…`` | ok |
| 5851 | `rcwa/_core.py:998` | 998 | 0 | `erratic, measure-zero (period, n_orders) coincidence …` | ok |
| 6804 | `propagators/propagation.py:298` | 298 | 0 | `})` | ok |
| 15118 | `propagators/propagation.py:230` | 230 | 0 | `fraunhofer_propagate,` | ok |
| 14142 | `lumenairy/__init__.py:682` | 681 | **-1** | `should_split,` | inside the walker's +/-5 window |
| 14664 | `elements/_lens_real.py:882` | 880 | **-2** | `except Exception:` | inside the window |
| 18344 | `elements/lenses_maslov.py:448` | 446 | **-2** | `Tu4 = np.empty(P + 1)` | inside the window |
| 18574 | `elements/_lens_jax.py:573` | 568 | **-5** | `pres_no_ap = dict(lens_prescription)` | at the window edge |
| **14882** | **`elements/lenses.py:722`** | **709** | **-13** | `max_semi = 0.0` | **STALE -- should read `:709-797`** |
| **18579** | **`elements/_lens_jax.py:819`** | **812** | **-7** | `_is_mirror = bool(_s.get('is_mirror', False)) or (` | **STALE -- should read `:812`** |

The two stale citations, with their CHANGELOG context, for the orchestrator:

* `CHANGELOG.md:14882` -- *"The 3 NumPy helpers from
  `elements/lenses.py:722-810` plus the xp-dispatched twin from
  `asymptotic_jax_twin.py:65` …"*.  `lenses.py` lost 15 lines above that
  anchor (the E-L4/E-L5 tombstone, sec. 5 item 2), so the range should read
  **`elements/lenses.py:709-797`**.
* `CHANGELOG.md:18579` -- *"**`apply_real_lens_maslov_jax`** —
  `_lens_jax.py:819`: same fix."*  Should read **`_lens_jax.py:812`**.

Neither is currently red: both live in older CHANGELOG blocks, and
`scripts/check_source_line_citations.py` and
`tests/unit/test_v5_3_2_walker_source_line_citation.py` audit only the newest
block.  Both are green on the delivered tree:

```
python scripts/check_source_line_citations.py
-> check_source_line_citations: auditing v5.45.1 block.  summary: ok=5  drift=0  total=5

pytest tests/unit/test_v5_3_2_walker_source_line_citation.py tests/unit/test_v4_15_agent_f.py
-> 28 passed in 0.78 s
```

I did not edit `CHANGELOG.md` (COMMON.md rule 2).

---

## 11. Requested changes outside my ownership

1. **`tests/unit/test_audit2609_a17_history_relocation.py`, mutation 1** --
   widen the catalogue to MODULE-LEVEL statements.  It currently searches only
   `FunctionDef` / `AsyncFunctionDef` bodies with `>= 3` statements, so a pure
   re-export shell cannot carry a history document at all (sec. 4).
   `propagation.py` has dozens of single-line module-level statements
   (`from .fft_infra import (...)` is multi-line, but `__all__` entries,
   `_LIVE_FORWARD = {...}` members and several `del` statements are not).  The
   minimal change is to fall back to `tree.body` when no function body
   qualifies -- the same "widen the catalogue rather than exempting the
   module" principle the checker's own mutation-3 message states.  With that
   in place, `docs/history/lumenairy.propagators.propagation.md` can be added
   and this sweep's two `propagation.py` blocks recorded verbatim there
   instead of in sec. 9 of this report.  **Not urgent**: nothing is red today.

2. **`CHANGELOG.md` line-citation refresh** -- the two stale citations of
   sec. 10.  Documentation accuracy; no test currently fails on them.

3. **`lumenairy/_deprecation.py`** (WP-A17 SWEEP 3) -- restore the literal
   `Tombstone, v5.30` on the surviving registry-slot comment, which sweep 3
   moved wholesale into `docs/history/lumenairy._deprecation.md` and which
   `tests/unit/test_niche_audit_w4_p5_return_contract.py::TestTransitionMachineryIsRetired::test_the_executed_entry_is_tombstoned_in_the_registry`
   pins (sec. 12.4).  The word is present tense -- that entry IS a tombstone,
   kept for the next transition and scheduling nothing -- so restoring the
   heading satisfies the pin without re-importing narrative.  The test's other
   four phrases (`schedules nothing`, `EXECUTED`, `PropagationResult`,
   `return_result=False`) are all still in the source, measured.  **This is
   the only red test attributable to a WP-A17 sweep, and it is not mine**;
   neither the module nor the test is in my partition.

4. **Nothing further.**  Sweep 3 sec. 7 item 3 asked for
   `lumenairy/elements/lens_config.py`'s `_VOCAB_CACHE` to be enrolled with
   the central cache registry; that has since landed --
   `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py::test_every_cache_owning_module_enrolls_with_registry`
   is GREEN on the delivered tree (measured, `1 passed in 2.18 s`), so the
   item is closed and needs no action.

---

## 12. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, `-q --no-header
-p no:cacheprovider`.  The machine was shared with WP-A24 throughout.

### 12.1 The A17 checker, the recorder and every walker that reads this partition

```
pytest tests/unit/test_audit2609_a17_history_relocation.py \
       tests/unit/test_audit2609_a17_history_lint.py \
       tests/unit/test_audit2609_a22_history_fingerprint_tool.py \
       tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py \
       tests/unit/test_v5_3_2_walker_source_line_citation.py \
       tests/unit/test_v4_15_agent_f.py \
       tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
       tests/unit/test_v5_2_walker_pep562_forwarding.py \
       tests/unit/test_v5_2_walker_shell_vs_canonical.py \
       tests/unit/test_v4_16_0_walker_all_symmetry.py \
       tests/unit/test_v4_16_3_agent_d.py \
       tests/unit/test_niche_audit_w3_elements.py
-> 885 passed in 46.82 s
```

```
python scripts/record_history_fingerprints.py --check
-> OK: every history document matches its module.      (123 documents, 0 drift)

python scripts/check_source_line_citations.py
-> auditing v5.45.1 block.  summary: ok=5  drift=0  total=5
```

### 12.2 The a2 / a3 / a4 / a16 work-package files

```
pytest tests/unit/test_audit2609_a2_analytic_lens.py \
       tests/unit/test_audit2609_a2_displaced_models.py \
       tests/unit/test_audit2609_a2_verify_lens_analytic.py \
       tests/unit/test_audit2609_a3_caustic_siblings.py \
       tests/unit/test_audit2609_a3_traced_lens.py \
       tests/unit/test_audit2609_a3_verify_traced.py \
       tests/unit/test_audit2609_a4_asymptotic.py \
       tests/unit/test_audit2609_a4_fga_s10.py \
       tests/unit/test_audit2609_a4_maslov_gbd.py \
       tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py \
       tests/unit/test_audit2609_a16_lens_arch.py \
       tests/unit/test_audit2609_a16_lens_config_bit_identity.py \
       tests/unit/test_audit2609_a16_lens_config_round_trip.py \
       tests/unit/test_audit2609_a16_verify_config_and_arch.py
-> 466 passed, 1 warning in 135.28 s (2:15)
```

### 12.3 Every test file that reads this partition's source or docstrings

The list was built mechanically: a `getsource` / `getsourcelines` / `getdoc` /
`getcomments` / `__doc__` / `read_text` / `read_bytes` call anywhere in the
file AND a mention of one of this partition's modules, public entry points or
config classes.  **81 files**, plus the two new / rewritten ones:

```
pytest <81 files> tests/unit/test_audit2609_a17_history_lint.py \
       tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py
-> 10 failed, 4537 passed, 30 skipped, 176 warnings in 1497.13 s (24:57)
```

**Seven of the ten were transients of my own making** and are green on the
delivered tree.  That run was 25 minutes long and I re-spliced
`_lens_traced.py` and three other modules while it was in flight (a seam fix,
sec. 1.2), so the seven tests that read `_lens_traced.py`'s source read a file
mid-rewrite.  Re-run of exactly those files on the stable tree:

```
pytest tests/unit/test_niche_newton_pool_both_fits.py \
       tests/unit/test_niche_perf_round2_2026_08_10.py \
       tests/unit/test_niche_audit_w4_p5_return_contract.py
-> 1 failed, 101 passed in 69.01 s
```

**The remaining three are not mine**, and each is pinned to a cause:

| failure | why it is not mine |
|---|---|
| `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512-complex128]` and `[1024-complex128]` | a NUMERIC bar on `memory.py`'s `estimate_asm_memory` (`est/measured >= 1.0`, reading **0.888** and **0.972**).  My `memory.py` edit is comment-only and both fingerprints are identical, so it cannot move that number -- verified independently by parsing both files' module-level constants: **all 15 are equal**, `_LENS_REAL_F64_ARRAYS` 6.6 -> 6.6, `_LENS_REAL_COMPLEX_ARRAYS` 7.5 -> 7.5, `_ASM_FIRST_CALL_FIXED_BYTES` identical.  Reproduces in isolation (`2 failed, 6 passed in 6.84 s`).  Sweep 3 sec. 5b reported the same two failures and attributed them to WP-A15b lowering `_ASM_FIRST_CALL_FIXED_BYTES` 56 -> 40 MiB on a shared box. |
| `test_niche_audit_w4_p5_return_contract.py::TestTransitionMachineryIsRetired::test_the_executed_entry_is_tombstoned_in_the_registry` | **a prose pin broken by SWEEP 3**, not by me -- see sec. 12.4. |

### 12.4 A broken prose pin inherited from SWEEP 3 (`_deprecation.py`)

`lumenairy/_deprecation.py` is **not in my partition and is unmodified in my
working tree** (`git status` clean on it; last touched by commit `2ede9a16`,
"WP-A17 sweep 3").  Sweep 3 relocated its two registry TOMBSTONES into
`docs/history/lumenairy._deprecation.md` (its own sec. 1.2a names them), and
one of the phrases it moved is pinned by a test:

```python
# tests/unit/test_niche_audit_w4_p5_return_contract.py:453
src = _squash(inspect.getsource(dep))
assert 'Tombstone, v5.30' in src            # <- FAILS
assert 'schedules nothing' in src           # passes
assert 'EXECUTED' in src                    # passes
assert 'PropagationResult' in src           # passes
assert 'return_result=False' in src         # passes
```

Measured on the delivered tree: `'Tombstone, v5.30'` occurs **0 times** in
`lumenairy/_deprecation.py` and **2 times** in
`docs/history/lumenairy._deprecation.md`; the other four phrases are all still
in the source.  Sweep 3's report says "Retired prose assertions: none", so this
one was missed -- its 60-file source-reading sweep did not include
`test_niche_audit_w4_p5_return_contract.py` (that file was in sweep 1's list).

**The fix is one line and belongs to the `_deprecation.py` owner**, not to me.
"Tombstone" is a present-tense description of what that registry entry IS -- a
slot kept for the next transition that schedules nothing -- so restoring the
heading `Tombstone, v5.30 (EXECUTED)` on the surviving comment satisfies the
pin without re-importing any narrative.  Requested in sec. 11 item 4.

### 12.5 `-k real_lens` over the whole unit suite

```
pytest tests/unit -k real_lens
-> 160 passed, 3 skipped, 15018 deselected, 15 warnings in 233.39 s (3:53)
```

### 12.6 The validation topic file

```
python validation/run_all.py test_lenses
-> [PASS] test_lenses.py (23.5 s)   ALL 1 files passed.
```

### 12.7 Static gates and import

```
ruff check lumenairy tests scripts          -> All checks passed!
python -c "import lumenairy"                -> OK
py_compile over lumenairy/**/*.py           -> 233 compiled, 0 failed
```

Line endings, checked byte-wise on the delivered tree (`git ls-files --eol`):
the 18 CRLF modules came out CRLF and the 4 LF modules (`rcwa/_core.py`,
`rcwa/oned.py`, `backend/__init__.py` and their siblings) came out LF -- each
file kept the convention it went in with, and `git`'s "LF will be replaced by
CRLF" notice on the three LF files is the checkout's pre-existing state, not
something this sweep introduced.

---

## 13. Residual risk

* **The LOOSE percentage is still high on `_lens_traced.py` (28.3 %).**  That
  is the honest number and it is not a shortfall: the audit's loose classifier
  counts a whole comment run if ANY line in it names a version, the word
  "audit" or a date, and this module's comment runs are 50-180-line MEASURED
  DERIVATIONS that cite the audit document they were measured in.  The strict
  count -- the shapes the finding actually names -- is 0.  Driving the loose
  count down further would mean deleting measurements that
  `docs/TESTING_STANDARDS.md` S5 requires.
* **The ratchet's baseline includes `propagators/carrier.py` (11 lines)**,
  which WP-A24 may still edit.  If that edit adds a history line the ratchet
  will fire; the failure names the file and the re-baseline command, and
  re-baselining in the same commit is the sanctioned response.
* **The 34 present-tense rewrites of `pre-fix`-framed guard notes are
  judgement calls.**  Each keeps the hazard and the measurement and drops only
  the release number, and each original wording is recorded verbatim (in a
  document for the 88 documented blocks, in sec. 9 for the rest) -- so the
  judgement is reviewable rather than lost.
* **`docs/history/` is now 123 documents.**  The checker parametrises over all
  of them (739 cases, 27 s), which is fine today; a future sweep that doubles
  the count should watch that runtime.

---

## 14. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP4_CHANGELOG.md`

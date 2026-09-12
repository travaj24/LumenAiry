# WP-A17 (part 1) — version-history relocation out of the source

Scope of this launch: `lumenairy/propagators/carrier.py`, `carrier_field.py`,
`fft_infra.py` — comments and docstrings only, zero executable change.  The
lens family (`_lens_traced.py`, `_lens_real.py`, `_lens_imap.py`,
`lenses_maslov.py`) is **part 2** and was not touched.

Finding: **P2-4** (`TESTS-ARCH.md:394`) / consolidated report **§14 V6** and
**§15.7**.

---

## 1. Summary

| finding | status | files:lines | tests (path::name) | oracle | measured before → after |
|---|---|---|---|---|---|
| P2-4 / V6 — `fft_infra.py` 34.4 % git history | **fixed** | `lumenairy/propagators/fft_infra.py` 2678 → **2501** (−177) | `tests/unit/test_audit2609_a17_history_relocation.py::test_the_module_ast_is_unchanged_since_the_history_move[fft_infra]` + `…token_stream…[fft_infra]` | SHA-256 of the docstring-free AST and of the comment-free/docstring-free token stream, recorded from the pre-relocation file | history lines 933 → **333** (34.8 % → 13.3 %); "pre-fix this did A" (strong) lines 316 → **62** (−80 %) |
| P2-4 / V6 — `carrier.py` 36.9 % git history | **partially fixed** (see §2.4) | `lumenairy/propagators/carrier.py` 11602 → **11275** (−327) | same checker, `[carrier]` | same | history lines 4151 → **3443** (35.8 % → 30.5 %); strong lines 943 → **197** (−79 %) |
| P2-4 / V6 — `carrier_field.py` (not in the audit's six, carried as the carrier sibling) | **fixed** (small by construction) | `lumenairy/propagators/carrier_field.py` 1857 → **1852** (−5) | same checker, `[carrier_field]` | same | history lines 376 → **276** (20.2 % → 14.9 %); strong lines 95 → **59** (−38 %) |
| §15.7 — comments that state the OPPOSITE of the code | **fixed, 4 sites** | see §2.5 | the same identity gate + the existing behaviour tests | the code beneath each comment | 4 stale claims corrected, 0 behaviour change |

`ruff check` clean on all three modules; `python -c "import lumenairy"` OK.

**Both fingerprints are byte-for-byte identical to the pre-relocation file on all
three modules.**  Not "equivalent" — identical:

```
carrier.py        ast 3829ad9e8878ab4a…  token b3e9d7ba3d85ba04…
carrier_field.py  ast 6fc5905777971218…  token f08b25bbf206e6dd…
fft_infra.py      ast 1a11380b468ed0bb…  token c4f06ed7121fc281…
```

---

## 2. Per item

### 2.1 What moved, and where it went

Three new documents, one per module, each holding the blocks **verbatim** under
the source line they came from in the pre-relocation file, with a table of
contents in original-line order:

| document | lines | blocks | prose lines moved | condensed/pointer lines left in the source |
|---|---|---|---|---|
| `docs/history/carrier.md` | 864 | 35 | 534 | 232 |
| `docs/history/fft_infra.md` | 748 | 43 | 372 | 215 |
| `docs/history/carrier_field.md` | 120 | 5 | 41 | 32 |

Each document's header is a machine-readable block the checker test parses:

```
<!-- lumenairy-history-doc
module: lumenairy/propagators/carrier.py
ast_sha256: …
token_sha256: …
pre_relocation_lines: 11602
recorded_by: WP-A17 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->
```

Every recorded block carries a *Left in the source:* line naming exactly which
part of the rationale stayed behind and why — so the move is reviewable block by
block rather than as a bulk diff.

### 2.2 The two patterns that actually carried the history

**(a) A constant whose comment carried EVERY derivation it has ever had,
newest last.**  This is the single biggest source of history in `carrier.py`:

* `_FINE_GRID_WORK_ARRAYS` — four successive derivations (4 → 16 → 20 → 22 →
  24) over 173 lines.  Only the last describes the value the code holds.  Now 63
  lines: the current envelope derivation, the seven measured rows it bounds, the
  two ratios, and the two method notes a re-measurement needs ("measure from
  OUTSIDE the process — an in-process sampler thread inflates peak working set
  by up to 2.5× on this workload"; "price the ENVELOPE, not a fit").
* `_FINE_GRID_BASE_BYTES` — 4.5 GB → 2.6 → 1.8 over 67 lines.
* `_PARAXIAL_BASE_BYTES` — the first cut that charged the exact leg's floor to a
  paraxial worker.
* `_FOCUS_STANDOFF_*` — 6.0 z_R → 0.8 z_R → the derived extent law → the
  small-extent branch, over 125 lines in two blocks.

The source keeps the derivation of the value it actually holds — `TESTING
_STANDARDS.md` S5 requires a numeric bar to carry one — and the superseded
derivations are in the document.

**(b) A comment correcting an earlier COMMENT.**  "an earlier cut of this note
got wrong … in both magnitude and DIRECTION"; "the old wording told callers the
opposite"; "an earlier revision of this note cited a 5.5× pitch split for design
121's own leg"; "The counter-evidence on record was a mis-citation".  These are
notes about the documentation's own history.  They moved; the source now states
the corrected rule once, plainly.

### 2.3 What deliberately did NOT move

* **Derivations of live numeric bars.**  `TESTING_STANDARDS.md` S5 requires
  every numeric bar to carry its oracle, its error floor and its measured
  values.  The P3 multi-congruence gate's canonical-scale derivation and its
  18-construction separation table, the niche-C3 high-NA-gap two-arm
  calibration, the `_BAND_HEADROOM` 24-fixture bisection, the
  `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` N ≥ 11181 derivation — all stayed.
  Moving them would violate the repo's own standard.
* **Live deprecation and migration statements.**  `CarrierField`'s
  "BECOMING FROZEN … at the horizon `_CARRIER_FIELD_FROZEN_IN`",
  `set_pyfftw_planner`'s "switching back to FFTW_ESTIMATE does not restore the
  bits you had before", the `.. versionchanged::` on the auto-promote default
  flip (condensed, not removed) — these tell a user what to do now.
* **`fft_infra.py`'s `Multiprocess / fork notes` section.**  Flagged by the
  coordinator via `test_v4_16_3_agent_d.py`.  Re-checked: the semantics are
  still exactly true (plain module globals; a `spawn` worker re-imports at
  library defaults; the one-shot latches reset per worker).  It is a
  current-behaviour contract, not history — **restored in full** under its own
  header, with only the "Workaround until v5.0" framing replaced by the real
  remedy (`snapshot_fft_state` / `restore_fft_state`).  That test passes
  unchanged; no assertion retired.

### 2.4 Why `carrier.py` is "partially fixed"

The audit's 36.9 % figure comes from a deliberately loose classifier: a comment
block counts as history if it mentions a version number, the word "audit", or a
`20xx-xx-xx` date, and a docstring counts if it carries ≥ 2 such markers.  I
reproduced that classifier (35.8 % on the current file, against the audit's 36.9 %
at commit `a1ff1e6e`, the file having grown 10 594 → 11 602 lines since) and
then measured a **strict** variant that requires the shape the finding actually
names — an explicit `vN.N (`, `pre-fix`, `used to`, `formerly`, `previously`,
`the old `, `was wrong`:

| module | loose "history" b → a | strict "pre-fix this did A" b → a |
|---|---|---|
| `carrier.py` | 4151 → 3443 (−17 %) | **943 → 197 (−79 %)** |
| `fft_infra.py` | 933 → 333 (−64 %) | **316 → 62 (−80 %)** |
| `carrier_field.py` | 376 → 276 (−27 %) | **95 → 59 (−38 %)** |

The residual loose-classified mass in `carrier.py` is nine large blocks that are
**measured derivations of live constants**, each citing the audit that measured
them: the P3 gate (195 lines), the niche-C3 gap guard (117), the niche-C1
tilt-exact-grid power budget (67), the multi-congruence orchestrator's window
argument (70), and the four memory constants' current derivations.  Removing
those would trade one audit finding (V6) for another (S5).  I judged the
finding's intent to be the *narrative*, not the *measurements*, and the strict
column is the number that tracks that intent.

`carrier_field.py` scores 20.2 % on the loose classifier almost entirely because
its derivations cite the probe that measured them; it is a young module and
carries very little genuine history.  That is stated in its document's preamble.

### 2.5 Stale comments corrected (audit §15.7, "worse than none")

Four comments stated the opposite of the code beneath them.  All four are
docstring/comment-only edits and are covered by the identity gate:

1. `fft_infra.warmup_fft_plans` — the `threads` parameter doc said "Defaults to
   `available_cpus`".  The F-32 fix at the call site had already replaced that
   with `FFTW_THREADS`; the doc had not followed.  Now states the live default
   and why the plan-cache key makes any other default a silent no-op.
2. `carrier._sphere_parab_conversion` — "…while the untapered swap breaks a
   coarse chain".  The flag note 30 lines above re-derives that claim as a
   mis-citation of a measurement that says the opposite.  Removed; the measured
   guard-band statement kept.
3. `carrier._fourier_upsample_crop` — the dtype-parity promotion's premise
   ("numpy's FFT is double-only") lapsed at numpy 2.0.  The comment already
   carried a `CORRECTION 2026-09-11` paragraph saying so; the source now states
   only what the promotion does today.
4. `fft_infra._PYFFTW_DOUBLE_BUFFER` — priced the single-buffer copy at
   "~1-3 % of a large transform"; the byte-cap block 40 lines below measures it
   at ~65 % of the transform at N = 8192.  The claim is removed and the comment
   points at the block that measured it.

### 2.6 The reusable checker

`tests/unit/test_audit2609_a17_history_relocation.py` (new, 19 tests).  It
discovers its registry from `docs/history/*.md`, so **part 2 extends it by
adding its documents — no change to the test file is required.**

Two independent fingerprints, both recorded from the PRE-relocation file:

* **AST** — the module's tree with every string-*statement* deleted (docstrings
  and the "string as a comment" form), positions ignored.  Deleting rather than
  blanking makes the fingerprint indifferent to a docstring being shortened
  *or* removed outright.  `ast.dump` is not used: a hand-written renderer skips
  absent/`None`/empty-list fields so a new always-empty node field in a future
  CPython (`type_params` in 3.12 was one) cannot invalidate a recorded hash.
* **token** — the `tokenize` stream reduced to NAME/OP/NUMBER/STRING (plus the
  3.12+ f-string token trio), with COMMENT tokens and the STRING tokens of those
  same string-statements dropped.  This catches what the AST normalises away:
  the *spelling* of a literal, and it re-checks statement identity through a
  different front end.

Tests: header well-formedness; AST identity; token identity; the TOC is in
ascending original-line order with a matching `### L<n>` anchor for every row
and no row past the pre-relocation line count; the module still names its
history document; and — the falsifiability check the audit's V1 finding demands —
`test_the_fingerprints_are_actually_sensitive`, which applies three AST-anchored
mutations to an in-memory copy of each module and asserts they move the right
fingerprint:

| mutation | AST | token |
|---|---|---|
| delete a single-line statement from a function body | must move | — |
| rename an identifier | must move | must move |
| re-spell an integer literal to the SAME value (`5` → `0x5`) | must **not** move | **must** move |

The third is the one that justifies keeping two fingerprints: the AST folds both
spellings to one `Constant`, so the AST check alone is blind to a literal being
rewritten.  Every mutation is anchored on a real AST node rather than a regex,
because a regex lands in a comment or docstring — exactly the text these
fingerprints are built to ignore — and would prove nothing.  (An earlier
regex-based draft of mutation 3 did exactly that and passed vacuously; it was
caught by the test failing on `fft_infra` and rewritten.)

### 2.7 Residual risk

* The recorded fingerprints pin the module against the pre-relocation file
  **forever**, which is the point but also means a *deliberate* code change to
  one of these modules must re-record the two hashes in the same commit.  The
  assertion message says so explicitly.
* The checker cannot tell a *good* docstring from a *bad* one — it only proves
  the edit was documentation-only.  Prose quality was reviewed by reading the
  diff at every one of the 83 edit sites; the continuity fixes that pass found
  (dangling half-sentences left by a rewrap, a missing `#` separator, an
  orphaned "the old rule" reference) were corrected before the final apply.

---

## 3. Files touched

**Modified (comments and docstrings only — AST and token stream identical):**

* `lumenairy/propagators/carrier.py` (11 602 → 11 275)
* `lumenairy/propagators/carrier_field.py` (1 857 → 1 852)
* `lumenairy/propagators/fft_infra.py` (2 678 → 2 501)

**New:**

* `docs/history/carrier.md`
* `docs/history/carrier_field.md`
* `docs/history/fft_infra.md`
* `tests/unit/test_audit2609_a17_history_relocation.py`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_REPORT.md` (this file)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_CHANGELOG.md`

**Test files that assert on these modules' docstrings or source text — examined,
NOT modified:**

Found by `grep -rln "getsource\|__doc__" tests/unit` filtered to these modules,
**plus** a second sweep the first one missed: tests that read the module file
directly (`Path(...).read_text()`) rather than through `inspect`.  That second
sweep is how `test_v4_16_3_agent_d.py` was found; it greps for the literal
filenames `'carrier.py'` / `'carrier_field.py'` / `'fft_infra.py'` in `tests/`.

| file | what it asserts on my modules | outcome |
|---|---|---|
| `test_audit2609_a6_verify_carrier.py` | `getsource(carrier)`, counts the 7 `_sphere_parab_conversion(…)` call sites and requires `dy=` on each | code-structure; passes (all 7 sites are code, none in prose) |
| `test_niche_d8_congruence_workers.py` | `getsource(_multi_resolve_workers, _memory_bounded_n_fine, _multi_parallel_results)` — `_fine_grid_peak_bytes(`, `get_context('spawn')`, `mp_context=`, `_as_completed(futs)` | code-structure; passes |
| `test_niche_exact_gap_kernel.py` | `getsource(_carrier_step_fast)` — `gap_kernel = 'exact'`, `_check_gap_kernel(` | code-structure; passes |
| `test_verify_perf_fixes_2026_08_10.py` | `getsource(_fine_grid_ceiling)` split on `"""`; `getsource(fft_infra)` requires the literal `N >= 11181` in the cap comment; `carrier_referenced_exact_focus_readout.__doc__` must quote `= {_FINE_GRID_WORK_ARRAYS} complex128 arrays` | two are prose pins on CURRENT behaviour; both texts deliberately kept; passes |
| `test_niche_d1_tilted_carrier.py` | `propagate_traced_carrier_chain.__doc__` contains `TiltedCarrier` | current contract; passes |
| `test_v5_4_6_wave9_concurrency.py` | `getsource(warmup_fft_plans)` contains `FFTW_THREADS` | passes (the stale-doc fix at §2.5.1 also names it) |
| `test_niche_audit_w3_propagators.py` | `getdoc(_get_or_make_bandlimit)` must contain `asymptote`, `audit P12`, `sqrt((2*z / L)^2 + 1)`, `never over-filters`, `Matsushima` | the P12 derivation is the band-limit contract; kept in full; passes |
| `test_audit_io.py`, `test_audit_optimize.py` | `clear_asm_caches.__doc__` must name each chained cache | current contract; the chained-cache list kept; passes |
| `test_v4_16_3_agent_d.py` | reads `fft_infra.py` for `Multiprocess / fork notes` + `spawn` + `latch`/`fork-safety` | current contract; section restored in full (§2.3); passes |
| `test_v4_16_3_agent_b.py`, `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`, `test_v5_2_walker_pep562_forwarding.py` | AST/structural walkers over `fft_infra.py` | pass, except one **pre-existing** failure — §5 |

**Retired prose assertions: none.**  Every prose pin found was a pin on a
statement that is still true of the current behaviour, so it was cheaper and
more honest to keep the statement than to retire the test.  Part 2 may still
need to retire some; this part did not.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

**Before** (baseline, on the unmodified tree):

```
python -m pytest <21 carrier/fft test files> -q --no-header -p no:cacheprovider --tb=line -rf
671 passed, 7 skipped, 71 warnings in 393.38s
```

**After** (same 21 files, same flags), on an intermediate state with the source
frozen:

```
671 passed, 7 skipped, 71 warnings in 414.29s     (with the checker file added: 690 passed, 7 skipped)
```

**Byte-identity of the output.**  The before/after outputs differ in exactly
three ways, none of them a behaviour difference:

1. **Source line numbers inside warning texts** from my three modules
   (`carrier.py:9488` → `carrier.py:9170`).  A necessary consequence of deleting
   327 lines; the warning *text* is character-identical.
2. **Free-RAM readings** in `_multi_resolve_workers`' clamp warnings
   (117.8 GB → 116.2 GB) — the machine is shared with other agents.
3. **Warning-origin source lines rendered from `elements/_lens_traced.py` and
   `elements/_lens_real.py`** — another agent was editing those files during the
   run (they are in `git status` as modified and are part 2 / WP-A2-A3
   territory).

After normalising (1) and (2), the outputs are identical except for (3).  On my
three modules, every test outcome is unchanged.

**Final full run** — the 21 files plus the new checker plus the four
source-reading walker files, on the delivered state:

```
2 failed, 728 passed, 7 skipped, 71 warnings in 397.06s
```

Both failures are outside this WP and both are documented:

* `test_v5_2_walker_pep562_forwarding.py::test_v14_…` — **pre-existing**, see §5.
* `test_niche_d14_deterministic_carrier_fit.py::test_the_field_moves_only_at_the_summation_noise[traced]`
  — a **transient from another agent's half-landed edit**.  The child process it
  spawns died with

  ```
  File "…\lumenairy\elements\_lens_traced.py", line 8706, in apply_real_lens_traced
      if _wants_config(geometry, numerics, resources, config):
  NameError: name '_wants_config' is not defined
  ```

  `_lens_traced.py` is not one of my files (it is WP-A17 **part 2** and was
  being edited by another agent at that moment).  Re-run in isolation once their
  edit landed: `18 passed in 90.71s`.

  The same cause produced one earlier transient,
  `test_verify_perf_fixes_2026_08_10.py::test_every_numexpr_out_site_drops_the_retention`,
  which reads `inspect.getsource(lumenairy.elements._lens_real)` — also green on
  re-run (`1 passed in 0.09s`).

**Static gates:** `ruff check lumenairy/propagators/` — *All checks passed*;
`ruff check tests/unit/test_audit2609_a17_history_relocation.py` — clean;
`python -c "import lumenairy"` — OK.

**A note on method, because it cost a run.**  An intermediate "after" run was
started and the module was then rewritten *while it was in flight*.  Six
`inspect.getsource` tests failed, all with garbage slices — `linecache`
re-read the changed file and sliced it at the old `co_firstlineno`.  The run was
discarded and repeated with the source frozen; all six pass.  Anyone re-verifying
this WP should not edit the module during a run that uses `getsource`.

---

## 5. Pre-existing failure found (NOT caused by this WP)

`tests/unit/test_v5_2_walker_pep562_forwarding.py::test_v14_every_mutable_fft_infra_global_is_in_live_forward_names`

```
V14 forward check: fft_infra.py has mutable module-level globals that are NOT
listed in propagation._LIVE_FORWARD_NAMES:
  - _PYFFTW_FIRST_FFT_THREAD
  - _PYFFTW_SHARED_BUFFERS_UNSAFE
```

Both globals were added by **WP-A5** (audit K4, the ping-pong-buffer
multi-thread latch) and were never added to `propagation._LIVE_FORWARD_NAMES`.
Proven pre-existing by running the walker's own `_mutable_module_globals()`
against my untouched **baseline copy** of `fft_infra.py`: same two names
missing.  It is a real defect of the class V14 exists to catch — a consumer
reading `propagation._PYFFTW_SHARED_BUFFERS_UNSAFE` after the latch flips sees
the stale import-time snapshot — but `propagation.py` is not mine.  See §6.

---

## 6. Requested changes outside my ownership

1. **`lumenairy/propagators/propagation.py`** — add
   `'_PYFFTW_FIRST_FFT_THREAD'` and `'_PYFFTW_SHARED_BUFFERS_UNSAFE'` to
   `_LIVE_FORWARD_NAMES`.  Both are rebound at runtime by
   `fft_infra._note_fft_thread`, so the shell's import-time snapshot goes stale
   the moment a second thread issues an FFT.  This makes
   `test_v5_2_walker_pep562_forwarding.py::test_v14_…` green.  (The alternative
   the test itself offers — adding them to `_V14_FORWARDING_EXEMPTIONS` — is
   wrong here: unlike the two `_NO_CONSUMER_WARNED` latches, these *are* live
   state and a stale read has a behavioural consequence.)
2. **`pyproject.toml` lines 66–72 / 74–80** — the audit's P2-4 also names a
   verbatim-duplicated scipy-floor rationale paragraph there (the same
   history-in-config artefact).  Not in my ownership and not touched.

---

## 7. Deferred

* **Part 2** (`_lens_traced.py` 37.6 %, `_lens_real.py` 23.3 %,
  `_lens_imap.py` 30.3 %, `lenses_maslov.py` 16.8 %) — launched separately by
  the orchestrator once those files are released.  The tooling is ready: write a
  plan (line range → replacement text) per module and run the same apply step;
  the checker discovers the new documents automatically.  Estimated 2 d given
  `_lens_traced.py`'s size, on the evidence of this part (3 modules, 16 137
  lines, 83 edit sites).
* **A CI gate on the history share.**  A cheap follow-on would be a lint that
  fails when a new comment block matching the strict pattern (`vN.N (`,
  `pre-fix`, `used to`, `formerly`) is added to a module that has a
  `docs/history/` document — the audit's §14 V6 process observation ("each audit
  round adds a new history block, never an update to the existing one") is the
  actual root cause, and the relocation only clears the backlog.  ~0.5 d.
  `CONTRIBUTING.md` and the CI config are not in my ownership.

---

## 8. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_CHANGELOG.md`

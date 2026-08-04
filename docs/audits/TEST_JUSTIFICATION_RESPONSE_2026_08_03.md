# Response to AUDIT_TEST_JUSTIFICATION_2026_08_03 (2026-08-03, for the 5.32.1 train)

Answers `docs/audits/AUDIT_TEST_JUSTIFICATION_2026_08_03.md` ("does every test in
`tests/unit/` still earn its place?"): 10 parallel read-only agents, no pytest execution,
no repository files modified.

Method here is the inverse of the audit's: **nothing was accepted on static reading
alone.** Every claim that could be settled by running something was run. Every claim
this response rejects carries a counter-measurement, not an opinion -- the bar set by
the previous two audit responses.

Standing rules applied (each has cost real CI rounds to learn):

* comparative-envelope assertions only -- no absolute bars on BLAS-dependent magnitudes;
* both-BLAS-builds evidence for any behavior claim;
* **adjudicate against an oracle before re-pinning any test** -- this rule already caught
  a 17,000x predictor regression this session that a blanket re-pin would have cemented;
* era-pins keep their original assertions verbatim;
* fail-before switches for behavior changes.

And one rule specific to this audit: the audit predates C13/C14, so **every claim was
re-checked against the current working tree** before acting. Several prior audits in this
repo attacked already-fixed code or mis-cited their own evidence. This one does both, in
places -- including in its single highest-confidence item.

---

## 0. Bottom line

The audit's *framing* is right and worth saying plainly: the suite is well-disciplined,
and the audit-citation convention does work. Its clean-bill findings in section 9 (zero
orphaned imports, supersession handled almost everywhere) match what this response found
independently.

Its *specific actionable claims* are a different story.

| Audit section | Claim | Verdict |
|---|---|---|
| S2 | "A concrete, actionable bug": `test_niche_r4_fga_dual_vectorize.py` has **no** numba guard; all 10 tests **fail outright** on a base install | **REJECTED -- measured false.** All 10 tests carry `pytest.importorskip("numba")`. Simulated base install: **10 skipped, 0 failed** |
| S1.1 | EVENASPH round-trip test is stale; **recommend deletion** | **REJECTED (deletion) / ACCEPTED (prose).** The test **passes** -- the skip the audit assumed fires does not. Assertions are convention-agnostic and valid. Docstring repaired; a real green-mask closed |
| S1.2 | `..._is_SUPERSEDED` tombstone duplicates a sibling; **safe to delete** | **REJECTED -- measured false.** Different input (2.0e7 vs 61.0); it pins **guard order**. Same input on the `'SI'` path raises a *different* error. Prose repaired |
| S3 | ~30 confirmed duplicates across ~19 files | **DEFERRED**, with an adjudicated ledger. Independent re-check: **~50-60% accurate**; several rows are false, one is backwards |
| S4 | ~14 docstring/comment drifts | **ACCEPTED and IMPLEMENTED** (13 files). Highest-value part of the audit |
| S5 | Untraceable citation labels | **DEFERRED** -- cosmetic, its own cycle |
| S6 | 9 always-pass diagnostic tests in 6 files | **PARTIALLY ACCEPTED.** Count/file-list wrong (8 tests, 7 files); one named file is a **false accusation**. Two *other* vacuous tests the audit missed are now disclosed |
| S7.1 | C12 predictor: live keep-vs-delete decision | **ACCEPTED as already-resolved.** Keep opt-in. The C6/C8 precedent the audit says is missing does exist for C12 |
| S7.3 | `DECENTRED_FIT_ARBITER` contradicts the C11 doc | **ACCEPTED -- best finding in the audit.** Verified and escalated. The doc contradicts **itself**; pins re-confirmed green (96 passed) |

Net: **1 of the audit's 3 headline actionable items survived** (S7.3). The bulk of its
value is section 4, which it filed as the lowest-priority "purely a documentation fix".

**Three real defects the audit did not find** were turned up while checking its claims --
one of them a test that cannot fail for the regression it names (section 6 below).

---

## 1. S2 -- "a concrete, actionable bug" -- REJECTED, measured

The audit's most confident item:

> `_assert_dual_numba_match` does `assert nb is not None` with **no** `importorskip`/`skipif`
> guard anywhere in the file ... All 10 tests in the file that route through this helper
> will **fail outright** (not skip) on a base install without numba.

**The file contains 10 tests and 10 `pytest.importorskip("numba")` calls** -- one as the
first statement of every test:

```
$ grep -c "^def test_" tests/unit/test_niche_r4_fga_dual_vectorize.py   -> 10
$ grep -c 'importorskip("numba")' tests/unit/test_niche_r4_fga_dual_vectorize.py -> 10
```

Measured, not inferred. A faithful base-install simulation was built, because lumenairy
probes for numba **two** different ways and a naive blocker is not representative:

* `importlib.util.find_spec("numba") is not None` -- the `_NUMBA_AVAILABLE` probes in
  `elements/lenses.py:80`, `elements/_lens_traced.py:61`, `elements/lenses_maslov.py:119`,
  `optimize/_merit_jit.py:53`. Must return `None`.
* a bare `import numba` -- `raytrace/differential.py:843`, `propagators/fga.py:103,731`.
  Must raise `ModuleNotFoundError`.

(A first attempt raising from `find_spec` was rejected as unfaithful: it broke
`lenses.py:80`, which a real base install does not.) Simulation validated first:

```
find_spec numba -> None
import numba -> ModuleNotFoundError No module named 'numba'
_NUMBA_AVAILABLE (lenses) = False
_adrt_numba_kernel() -> None
```

Then the file, under that simulation:

```
ssssssssss                                                               [100%]
SKIPPED [1] tests\unit\test_niche_r4_fga_dual_vectorize.py:113: could not import 'numba'
... (10 identical SKIPPED lines) ...
10 skipped in 0.14s
```

**10 skipped, 0 failed.** The claimed defect does not exist. No change made.

Residual worth recording (a *different* claim from the audit's, and not a bug): once numba
imports, `_adrt_numba` returns `None` only if the njit kernel fails to **build**
(`differential.py:856-864`). The bare assert then fails rather than skips. That is correct
as designed -- numba present but its kernel unbuildable is a regression worth failing on,
not an environment quirk to skip past.

## 2. S1.1 -- EVENASPH -- deletion REJECTED, prose ACCEPTED

Audit: "Recommend deletion", on three sub-claims. Each checked separately.

**(a) "The class docstring prescribes the discredited convention" -- TRUE.** The docstring
told the reader to fix the loader to `power = 2 + 2*parm_num`. That rule was **reversed**;
the live loader ships `power = 2*parm_num` (`lumenairy/io/prescriptions_zemax.py`, with the
reversal rationale in-comment). Genuinely misleading. **Fixed.**

**(b) "Green-mask pattern; the skip references a file that no longer exists" -- FALSE as
stated.** The audit assumed the skip fires. Measured:

```
$ python -m pytest "tests/unit/test_audit_misc.py::TestAuditFixesV4_11_1_EvenAsphParmRoundTrip" -q
1 passed in 0.68s
```

The test **runs and asserts**. It is not masking anything today. Also: `lumenairy/io/prescriptions.py`
**does still exist** -- the *line range* rotted and the EVENASPH loader moved out of it, which is
a stale citation, not a missing file.

**(c) "Already covered by `test_audit_io.py`" -- PARTIAL.** The sibling round-trips a
**single** coefficient (`{4: alpha4}`). This test round-trips **two** (`{4: ..., 6: ...}`)
through a different construction path (`lm.make_singlet` + the full exporter argument set).
A single-coefficient round trip cannot see a mis-**ordering** between adjacent slots. Not
redundant.

**Implemented:** docstring rewritten to state the convention-agnostic claim, record the
v5.16.1 reversal, and drop the rotted citations. Separately, a **genuine green-mask was
closed** -- the `if found is None: pytest.skip(...)` was converted to a hard assertion.
That branch is not taken today, so this cannot change the current outcome; it only stops a
future loader regression retiring the pin behind a skip. This is the fail-before-safe
direction: strictly more failure surface, none of it new-behavior-dependent.

## 3. S1.2 -- the "self-admitted tombstone" -- REJECTED, measured

Audit: "Its only assertion ... duplicates a sibling in the same file (line 153). **Safe to
delete.**"

The two tests pass **different inputs**. Sibling: `periodx=61.0`. Tombstone:
`periodx=2.0e7` -- deliberate metre-scale nonsense. Measured:

```
A sibling(153) auto/61.0  -> makedammann2d: _legacy_units='auto' was REMOVED in v5.30. ...
B tombstone   auto/2.0e7  -> makedammann2d: _legacy_units='auto' was REMOVED in v5.30. ...
C SI default       2.0e7  -> makedammann2d: periodx=20000000.0 m exceeds 1 m; ...
```

Line C is the point. The **same value** on the `'SI'` path raises the **1 m bound**, a
different error. So the tombstone pins **guard order**: mode rejection must fire *before*
value inspection. Reorder the guards and B raises "exceeds 1 m" and fails its
`match='REMOVED in v5.30'` -- while the sibling, whose 61.0 is in-bounds, still passes and
notices nothing.

The audit read the docstring's "there is no post-rescale bound to test" as "this test
guards nothing". **The wording was the defect, not the test.** Docstring rewritten to state
the ordering claim and why it is not redundant. Assertion untouched.

Its `_is_SUPERSEDED` name is now misleading too; renaming churns the node id and
`.test_durations`, so it stays, noted in-file.

## 4. S3 -- duplicates -- DEFERRED, with an adjudicated ledger

Two independent adjudications, reading both sides of every claimed pair in the current
tree. **Of 17 rows checked, roughly half hold.**

Confirmed TRUE duplicates (safe whenever the owner runs a dedup cycle):
`test_v4_15_1_agent_f.py` (8 -- but see section 6; "byte-for-byte" is wrong for 3 of them,
which are *weaker* variants); `test_v4_15_3_agent_a.py` (10 migration-sanity);
`test_v4_15_3_agent_c.py` (**the 2 PerturbedABCD tests only**);
`test_v4_15_1_agent_g_application.py` (1); `test_through_focus_bucket_boundary.py` (1,
exact 2-of-6-metric subset -- claim precisely right); `test_v4_15_4_agent_d.py` (1);
`test_v5_4_5_coating_edge_cases.py` (1); `test_audit_except_budget.py` (1);
`test_v4_16_1_agent_c.py` (1); `test_v5_2_physics_fixes.py` (**1 of the 3 claimed**);
plus one the audit **missed**, `test_v4_16_2_agent_d.py::test_migration_guide_md_exists_at_repo_root`.

Claims that do **not** hold -- do not act on these:

* **`test_v4_15_3_agent_c.py::TestSentinelPickleRoundTrip` -- strictly STRONGER**, not a
  duplicate. The counterpart asserts only `restored is inst`; this adds post-round-trip
  subclass identity and `float(recovered) == approx(1e9)`. A `__reduce__` returning the
  singleton but breaking `__float__` passes the counterpart and fails this.
* **`test_v4_16_2_agent_d.py` -- 3 of 4 wrong.** `test_roadmap_claims_correct_meta_pin_count`
  asserts an absolute floor (`n_claimed >= 11`); the "superset" asserts internal
  consistency (`claimed == listed`). A ROADMAP that regressed to "ALL 9" while listing
  exactly V1-V9 passes the walker and fails this one. Neither subsumes the other. Two
  README/requirements rows likewise catch failure modes the walker misses (structural-skip
  vs assert; substring vs bullet-regex; VCS requirement lines).
* **`test_niche_audit_p2b_infra_contracts.py` -- "subsumed" is FALSE.** It uniquely pins the
  migration target named in the message, the absence of the abandoned 5.32 horizon, that the
  raise *replaces* the old warning, and non-vacuity in the SI THz/MMW regime.
* **`test_niche_audit_w4_p5_return_contract.py` -- backwards.** Assertions are textually
  identical but w4 never passes `method`, exercising the `_NO_DEFAULT` sentinel branch that
  w3 (`method='auto'`) cannot reach; a leaked `set_default_wave_propagator()` fails w4 and
  passes w3. Where these two files *do* truly overlap, **w4 is the fuller side** -- if
  anything is cut it should come out of w3.
* **`test_audit_g06_perf.py` -- distinct, and the criticism is already fixed.** It is a
  black-box probe of the live thread table (catching a `pyfftw` daemon from *any* path);
  the "more robust pair" are white-box call-spies that can only see calls inside the two
  functions they patch. The `pytest-timeout` false-fail the audit cites was **already
  addressed in this session's uncommitted work**. The audit is describing a version of the
  file that no longer exists.
* **`test_v5_2_physics_fixes.py` -- merge rationale does not hold, but neither does whole-class
  deletion.** Every Maslov test there calls only the *constructor*; the counterpart always
  runs the propagator -- different entry point, different failure moment. The "preserve the
  option-a -> option-b history" argument fails (that history is triplicated, and the class
  docstring itself defers to the standalone file), but the class also holds a `method='gbd'`
  negative control with no counterpart anywhere. Correct action is **deleting one test**, not
  merging a class.
* Of the two "mild overlap" spot-checks: `r0` vs `r1` -- audit accurate, no action.
  `r9` vs `s10` -- audit **understated** it; same entry point, same contract, `s10` looser,
  distinguished only by two boundary params.

**Why deferred rather than executed.** Deleting ~30 tests across ~19 files is a dedup cycle,
not a patch-release change: it is a large diff in a release train, it churns
`.test_durations` (whose staleness has already cost this repo a 30-min shard timeout at a
release tag), and by the audit's own section 8.5 it is a "does the owner want this done"
question with no open investigation behind it. Nothing here is urgent -- duplicates cost
runtime, not correctness. The ledger above is the executable part; it is now adjudicated,
so the cycle can be run without re-deriving it.

## 5. S4 -- docstring/comment drift -- ACCEPTED, IMPLEMENTED

The audit's lowest-priority category is its most valuable, because several of these do not
merely describe an old state -- they instruct a reader to do the **wrong** thing.
13 files fixed. Assertions were not touched in any of them.

| File | What was wrong | Verified against |
|---|---|---|
| `test_audit_misc.py` | prescribed the reversed EVENASPH rule as the fix to land | live loader; test passes |
| `test_niche_audit_w3_elements.py` | "no post-rescale bound to test" -> read as "guards nothing" | 3-way error-message probe |
| `test_fga.py` | cited `xdist --dist loadfile` (abandoned) and "~7 min" | workflow has **no** xdist; `.test_durations` sums **1791.7 s (29.9 min)** |
| `test_niche_s8_sphere_carrier_reference.py` | described the `cos^2` taper C9 removed (5 spots) | `SPHERE_PARAB_CONVERSION_EXACT = True` ships |
| `test_v5_4_1_coating_sellmeier_nan_fix.py` | said ranges were **relaxed** to 0.4-5 um; v5.4.6 **tightened** them | registry: TiO2 `(430e-9, 1530e-9)`, Ta2O5 `(500e-9, 1000e-9)` |
| `test_v5_2_glass_formula3.py` | described the "0 of 24 ingested" ship state | measured: 24 coefficients, `_POLYNOMIAL_STUB_NAMES` **empty** -> 4 tests vacuous, now disclosed |
| `test_audit_v5_24_2_g05_seams.py` | header bullet described the abandoned "lockstep" resolution | file's own S1-16 section |
| `test_audit_v5_24_2_b1_source_conventions.py` | described the v5.25 deprecation phase | tests assert v5.30 `TypeError` |
| `test_v4_15_3_agent_b.py` | "preserved for back-compat" (2 spots) | tests assert v5.30 removal |
| `test_v4_16_3_agent_c.py` | "Total: 8 tests" + a live P2-NEW-F1-1 contract | actual count **3**; 5 deleted in `155141b7` |
| `test_v5_11_0_rcwa_fff_nv_2d.py` | "lands closer than `li`" | the test asserts `err_nv <= err_li + 6e-3` |
| `test_v5_14_5_{emt_and_berreman_jax,viewer_polarization}.py` | docstrings say v5.14.5 | CHANGELOG puts all three features in **`[5.14.4]`**; `[5.14.5]` is only `_fast_geig` |
| `test_v5_2_chebyshev_extraction.py` | `ROADMAP.md` lines 221-225 | now the V13/V14 walker entries; item is at :340. Re-cited **by title** so it cannot rot again |
| `test_niche_audit_w3_oracles.py` | "Four oracles" | file also holds W3-3b + W4 (~1/3 of it) |

## 6. Three defects the audit did not find

**(a) A test that cannot fail for the regression it names.**
`test_v4_15_1_agent_f.py::test_threshold_boundary_inclusive_all_modes` sets the peak at
`E[N//2, N//2]` and then lists `(8, 8)` -- the same cell on a 16x16 grid -- among its
"boundary" pixels, overwriting the peak. Measured:

```
N//2 = 8 -> peak cell (8,8) is IN boundary_pixels: True
max|E|^2 = 0.010000000000000002 (docstring/comment claim: 1.0)
  normalised I at (2,2) = 1.0   (intended: 0.01)   [... all five identical ...]
strict >  keeps: 5 pixels
inclusive >= keeps: 5 pixels
=> strict and inclusive AGREE; the test cannot distinguish the convention it names.
```

The audit filed this as merely a duplicate. It is a duplicate **and** degenerate. The claim
is pinned correctly on a non-degenerate fixture (peak at the corner `(0,0)`, plus an
explicit precondition guard asserting the normalised intensity **is** the threshold) by
`test_v4_15_3_agent_d.py::test_rays_from_field_threshold_inclusive_consistent_across_modes`.
**Deleted**, with a tombstone recording the measurement. This is the one deletion taken from
section 3's list, and it is taken for degeneracy, not for duplication.

**(b) Unreachable assertions.** `test_v4_16_3_agent_b.py::TestSetDefaultDyNoConsumerWarning::test_first_call_emits_no_userwarning`
carried a `return  # short-circuit` followed by 8 lines of v4.16.3-era expectations that
**invert** the assertion above them (`assert len(uw) == 1`). Dead, and reads like a live
contract. **Deleted.**

**(c) Two vacuous tests the audit's own section-6 sweep missed.** Both
`test_validation_failure_no_warning` methods assert `not _prop._DEFAULT_*_NO_CONSUMER_WARNED`.
Nothing in `lumenairy` writes those globals at runtime -- they are bound `True` once at import
(`fft_infra.py:340-341`) and never flipped; the assertion only re-reads what the fixture set.
Kept as forward guards, now **disclosed in-docstring** so a green run is not miscounted as
coverage.

## 7. S6 -- always-pass diagnostics -- PARTIALLY ACCEPTED

The category is real but the accounting is wrong: **8 tests across 7 files**, not "9 across
6" (the audit's own prose says 6 files then lists 8). It also **missed**
`test_v4_16_0_walker_xp_of_dispatch.py::test_discovered_dispatch_candidates_for_diagnostics`.

**`test_v4_16_1_agent_d.py` is a false accusation** -- it contains no always-pass test. The
likely mis-attribution, `test_factory_verb_naming_contract`, does say "INTENTIONALLY
informational" in its docstring, but it walks the live `lumenairy` namespace and ends with a
real, failable `assert not violations` that fires the moment a new `make_*` export gains an
`N` parameter without an exemption.

One nuance the audit's framing loses: these tests cannot *fail*, but they can *error* -- each
drives the real AST walker, so a walker crash surfaces here. Weak, non-zero signal. No change
made; converting them to a `--collect`-time report is a reasonable idea for its own cycle.

## 8. S7 -- the two live decisions

**S7.1 C12 predictor -- already resolved; keep opt-in.** The audit asks whether to retire it
like Part E's `wavefront_aware` or keep it opt-in like the C6 fit guard, and says "no
equivalent argument is recorded for C12". It is recorded -- in the flag's own note, which
the audit read past. `DECENTRED_FIT_PREDICTOR = False` ships, and its note states the
measurement that rejected it *and* the fall-back ladder tying it to `DECENTRED_FIT_ARBITER`.
That is the C6-shaped argument: a genuinely different object (a closed-form per-call
predictor) from its replacement (a build-and-compare arbiter). The tests correctly assert it
stays off. **No action.** This is also the item the adjudicate-against-an-oracle rule already
protected once: the predictor's 17,000x regression was caught by oracle comparison rather
than blanket re-pinning.

**S7.3 `DECENTRED_FIT_ARBITER` -- the audit's best finding. ACCEPTED, verified, escalated.**

The audit reports the working tree contradicting the C11 doc. It is worse than that: **the
C11 doc contradicts itself.**

* `C11_...md:33` -- "**`DECENTRED_FIT_ARBITER` ships `False` -- OPT-IN**"
* `C11_...md:562` (S6.3, the decision section, with the measurement table) -- "**Resolved by
  shipping the arbiter OPT-IN** (`DECENTRED_FIT_ARBITER = False`)"
* `C11_...md:795` ("The library change is:") -- "**`DECENTRED_FIT_ARBITER = True`**"
* live code -- `_lens_traced.py:1862: DECENTRED_FIT_ARBITER = True`

The flip is **uncommitted and deliberate**: `HEAD` has `False`, the working tree has `True`,
and a sibling flag's note added in the same diff reads "``DECENTRED_FIT_ARBITER`` did ship
``True``". So the code is the later decision and S6.3 is stale prose.

This matters for 5.32.1 specifically, because S6.3's stated reason for shipping OFF was a
release-safety argument: flag-off is a genuine no-op, so "the whole C11 layer ... can ride a
patch release without a physics re-verification cycle." Shipping ON gives that up, and S6.3
also records that the arbiter **fails a per-order "improve or hold" bar at (-1,0)** by
0.0262 points against a 0.003-0.015 floor -- calling the default flip "a judgement about
design 121 rather than a library fact" that "belongs to an explicit decision with this table
in front of it."

**Not changed here.** Flipping a physics default is the owner's call, not an audit
response's, and the tree is another agent's verified in-flight work. **Escalated as a
must-reconcile-before-ship item:** either the code returns to `False` for 5.32.1, or C11
S6.3 and line 33 are rewritten to record the explicit decision that overrode them. Today the
release record says one thing and the release says the other.

The audit's actionable half **was** executed -- re-confirm the D6/D7 branch-selection pins
against whichever value actually ships:

```
$ python -m pytest tests/unit/test_niche_c11_decentred_fit_arbiter.py \
      tests/unit/test_niche_d7_decentred_fit.py tests/unit/test_niche_d6_exact_tilted_leg.py -q
96 passed, 23 warnings in 531.55s (0:08:51)
$ python -m pytest tests/unit/test_niche_d7_decentred_fit.py tests/unit/test_niche_d6_exact_tilted_leg.py -q
75 passed, 23 warnings in 518.88s (0:08:38)
```

The pins describe the shipped path correctly at `True`. The inconsistency is documentary,
not behavioural.

## 9. Deferred, with reasons

* **S3 bulk dedup** -- adjudicated ledger in section 4; its own cycle (see rationale there).
* **S5 citation labels** -- cosmetic by the audit's own account, and it confirmed the
  underlying findings are real. A mapping pass is a docs cycle, not a patch-release change.
* **S6 -> `--collect`-time report** -- reasonable, but it is a pytest-plugin change touching
  collection for 8 tests' worth of benefit. Not in a release train.
* **S8.2 D7 double-era-pinned witnesses** -- the audit is right that they use raw
  `monkeypatch.setattr` rather than the sanctioned era-pin registry
  (`lumenairy/elements/_traced_flags.py`). Migrating them is a real cleanup, but
  `_traced_flags.py` is **new uncommitted work this session** and `test_niche_d7_decentred_fit.py`
  is in-flight; touching both mid-verification is exactly the "augment, don't restart" failure
  mode. Next cycle, once C13/C14 land.
* **S8.4 `test_v4_16_3_agent_c.py` docstring-vs-CHANGELOG** -- the docstring half is fixed
  (section 5). Whether `CHANGELOG.md:1598` should also drop the retired `DeprecationWarning`
  is a CHANGELOG edit, out of scope for this branch by instruction.
* **S8.8 weak gates (e.g. the 50-wave RMS gate, S5-12)** -- tightening a gate is a behavior
  change needing a fail-before and both-BLAS evidence. Not a docs-response item.

## 10. Verification

Ruff clean across all 17 touched files:

```
$ python -m ruff check <17 files>
All checks passed!
```

Suites (all green; every file whose prose changed was executed, not just collected):

```
429 passed, 3 skipped in 124.86s      # the 12-file fast set, incl. test_audit_misc,
                                      # test_niche_audit_w3_elements, test_v4_15_1_agent_f,
                                      # test_v4_16_3_agent_b/c, glass_formula3, b1, agent_b,
                                      # coating nan-fix, g05, chebyshev, viewer_polarization
202 passed in 678.31s                 # w3_oracles + rcwa_fff_nv_2d + emt_and_berreman_jax
 13 passed in 16.43s                  # test_niche_s8_sphere_carrier_reference
 20 passed in 1.45s                   # test_v4_15_1_agent_f (post-deletion)
 21 passed in 15.99s                  # test_niche_c11_decentred_fit_arbiter
 96 passed in 531.55s                 # C11 + D6 + D7 against the shipped flag
 75 passed in 518.88s                 # D6 + D7 re-run
 27 collected                         # test_fga.py -- comment-only change
 10 skipped, 0 failed                 # r4 under the simulated base install (S2 counter-measurement)
```

The 3 skips are pre-existing and self-documenting; two of them
(`test_v5_2_glass_formula3.py:318,339`) independently corroborate the empty-stub-manifest
finding in section 5.

Both-BLAS evidence was **not** required: no assertion, tolerance, or numeric bar was
changed anywhere in this response. The two behavior-adjacent edits are the EVENASPH
`skip -> assert` (strictly more failure surface on a branch that is not taken today) and
two deletions of code that provably could not fail (a degenerate fixture and an unreachable
block). No absolute magnitude bars were introduced.

## 11. Coordination

Another agent was finishing verification runs in `tests/` during this work.
`tests/unit/test_niche_d3_guards.py` (17:17) and `docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md`
(18:10) were modified within the hour and were **read only, never edited**. None of the 17
files changed here appears in that agent's working set;
`tests/unit/test_niche_c12_physics_fit_selection.py` and `test_niche_d7_decentred_fit.py`
were **executed but not edited**. `C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md` was deliberately
left alone despite its self-contradiction (section 8) -- it is the other agent's decision
record. No `pmm/**` file, no `CHANGELOG.md`, and no git/gh operation was touched. Everything
is uncommitted.

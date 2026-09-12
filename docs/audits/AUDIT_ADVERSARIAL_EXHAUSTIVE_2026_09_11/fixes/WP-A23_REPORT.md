# WP-A23 — re-record the CI kernel-consistency census

Branch `audit-fixes-2026-09`, HEAD `2622449f`.  Finding: WP-A22 §4(a) / §5.5 —
`tests/unit/test_ci_kernel_consistency.py::test_this_arm_agrees_with_the_committed_census`
RED at its RULE check, the committed census recording `sliver/pmm1d@1e-05` as
`wrong` on every measured arm while this tree read all eight 1-D rows `correct`.

Every number below was MEASURED on this workstation with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.  No wall-clock
number is used as evidence.  **Host, and it matters for everything that
follows:** Intel Xeon w3-2535 (Sapphire Rapids, 10 cores / 20 threads),
Windows 11, CPython 3.14.6, numpy 2.4.6, scipy 1.17.1 — plus a reachable WSL2
build (Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1).

---

## 1. Summary

| item | status | files | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| **Mechanism** — what invalidated the census | **established by measurement, not inference** | — | `test_audit2609_a23_census_mechanism.py` (5) | the census probe's own fixture, one constant flipped in process | at `min_feature = period*1e-5`: `rcond` 9.7297e-13, `R+T` **3.6124215325**, guard **refuses** — the committed census's own readings.  At the shipped `period*1e-3`: `rcond` **5.2104e-04** (nine decades), `R+T` **1.0000000000000628**, guard **returns**.  Reversible in process, twice |
| **Census re-recorded** | **fixed** | `validation/probe_ci_kernel_sweep/{probe_decisions,merge_arms,make_ci_arm}.py`, `arms/*.json`, `decisions.json` | `test_ci_kernel_consistency.py` (8) | the probe re-run per arm | **12 arms (11 measured + 1 synthetic)** → **24 arms: 12 LIVE (2026-09-12, `2622449f`), 11 HISTORICAL (2026-09-11, `da377e0b`), 1 synthetic**; 2 builds × 5 kernels × 2 widths live; rule violations **0**, same-class splits **0** |
| **The rule test** | **fixed (was RED)** | `tests/unit/test_ci_kernel_consistency.py` | itself | the census + the live re-take | `test_this_arm_agrees_with_the_committed_census` **FAILED** (2 rows `warn` at class `correct`) → **7 tests → 8 tests, all passing**, 2.45 s |
| **The 2026-09-11 OPEN item** (why the CI arm differs) | **CLOSED, by measurement** | — | `test_the_census_carries_an_arm_that_answers_these_fixtures_correctly` | bit-identity against the transcribed CI arm | `WSL-SkylakeX-t4` reads `R+T` = **1.0000010471871335** (1e-05) and **1.0000003658397656** (1e-04) — the CI runner's two readings **BIT FOR BIT**, twice, independently.  `WIN-SkylakeX-t1`, same silicon, reads 3.6124215325 |
| **Premise gates, six families** | **re-measured; all still hold** | 8 files, docstring notes only | 20 files, 270 ids | pytest `-rs` on the running arm | **268 passed, 2 skipped, 0 failed**; neither skip is the census divergence (one is the documented numpy-vs-scipy LAPACK skip, one is a dense-grid population at 9 rows against a bar of 10) |
| `warned@` read `warn` on a CORRECT answer | **fixed in the instrument, not in the rule** | `probe_decisions.py` | `test_the_interface_voices_are_split_so_a_notice_is_not_a_false_alarm` | the two warning sites in `lumenairy/elements/pmm/` | rule violation on 2 rows → **0**; the rule table is byte-identical, the geometry notice has its own censused row |

**Nothing under `lumenairy/` was touched.**  No bar was relaxed anywhere; the
one fixture change makes the census's fixture ill-conditioned again, which is
strictly harder than what it had become.

---

## 2. Deliverable 1 — the mechanism, by measurement

### 2.1 The experiment

The census probe's 1-D fixture (`probe_decisions._pmm1d_two_layer`) is a
two-layer `PMMStack` whose cross-layer walls differ by `delta` = 1e-04 / 1e-05
of the period.  It took no `min_feature`, so it inherited the library default.
Run in ONE process, three passes: shipped default → old default
(`_MIN_FEATURE_DEFAULT_FRAC` = 1e-5, flipped in process) → shipped default
again as a reversibility control.

```
SHIPPED default (min_feature = period*1e-3)
  answer@1e-04   closes  correct    answer@1e-05   closes  correct
  warned@1e-04   warn    correct    warned@1e-05   warn    correct   <-- RULE VIOLATION
  sliver@1e-04   return  correct    sliver@1e-05   return  correct
  rcond  4.2942220016832993e-04 / 5.210363751156515e-04
  R+T    1.0000000000000562     / 1.0000000000000628
  hypothetical 1e-12 bar:  accept / accept        rcond decade: 1e-04 / 1e-04

OLD default (min_feature = period*1e-05)
  answer@1e-04   closes  correct    answer@1e-05   open    wrong
  warned@1e-04   silent  correct    warned@1e-05   warn    wrong
  sliver@1e-04   return  correct    sliver@1e-05   refuse  wrong
  rcond  9.693991669420469e-11  / 9.729688297808238e-13
  R+T    1.0000004784011818     / 3.6124215324602997
  hypothetical 1e-12 bar:  accept / refuse        rcond decade: 1e-11 / 1e-13

SHIPPED default AGAIN  -- identical to the first pass, field for field
```

The committed 2026-09-11 census records, over its ten measured arms, `rcond`
9.6940e-11 … 9.6969e-11 at 1e-04 and 9.7297e-13 … 1.0538e-12 at 1e-05, `R+T`
1.17 … 3.6116 at 1e-05, class `wrong`, decision `refuse`.  **The old default
reproduces that population; the shipped default does not, by nine decades of
`rcond`.**  So the census was invalidated by G2 and by nothing else: the only
thing that changed between the two passes was one float.

The band and branch-cut sections are byte-identical across the two passes
(16 rows), which is the control: `min_feature` moves the 1-D section and
nothing else.

### 2.2 Why, physically

`min_feature` is the threshold `_pmm_union_grid` uses to SNAP near-coincident
cross-layer walls.  G2 (WP-A12, `56a76f22`) raised the default two decades, to
`period*1e-3` — above BOTH of this fixture's separations.  Both wall pairs are
therefore snapped to coincidence, the two layers become geometrically
identical, and the near-singular interface every row in sections A and B is
about ceases to exist.  The census fixture stopped constructing its own
subject.

Measured, at the raising site (`lumenairy/elements/pmm/_core.py:4763`):

```
_pmm_union_grid: snapped 2 pair(s) of NEAR-COINCIDENT cross-layer walls
closer than min_feature=0.001 (period fractions): ...
```

at BOTH separations, including the 1e-04 one the census records as `correct`
and `silent`.

### 2.3 The second half of the finding: a notice is not a false alarm

The census's `pmm1d_interface/warned@` row carried an answer-following RULE —
a `correct` answer must be SILENT — but was DECIDED as "did any warning come
out".  That was the same thing while the energy tripwire
(`pmm/stack.py:1435`) was the only voice at the site.  G2 added a second,
deliberate, answer-INDEPENDENT voice (the snap notice above), and the row then
read `warn` at class `correct`: the census reporting a guard crying wolf when
no guard had spoken.  **That is the rule violation WP-A22 §4(a) saw**, on both
tags.

Fixed in the INSTRUMENT: `warned@` now measures the energy tripwire — the
voice its own comment always named — and the answer-independent voices get
their own row, `pmm1d_interface/notices@` (`none` / `snap` / `other` /
`snap+other`), with no answer class, compared across arms for plain equality.
**The rule table is unchanged**: a correct answer still may not be warned
about.  Nothing was loosened; the instrument was made as specific as the
sibling `sliver/pmm1d@` row already was (it has always filtered on `SLIVER`).

A third voice class, `env`, was added and EXCLUDED from the decision, and it
earned its place on first use: the WSL build reachable here is a venv without
`psutil`, and `lumenairy/memory.py:167` advises "psutil not installed —
assuming 4 GB available memory" **three times per solve** there and zero times
on Windows.  Without the split the two builds would have disagreed on a row
with no answer class — i.e. a reported P1 — for a reason with nothing to do
with the library's guards.  The count is kept as a READING
(`env_advisories@…` = 3.0 on every WSL arm), so an arm missing dependencies
still says so.

---

## 3. Deliverable 2 — the census, re-recorded

### 3.1 Which arms this box can produce, and how that was established

`threadpoolctl` — a **declared core dependency** of the library
(`pyproject.toml:97`) — is **not installed** in either interpreter here, and
has not been all campaign (WP-A14 §512 records it, WP-A15a §H5 made it a
declared dependency, nobody installed it).  The probe read the kernel back
through `threadpoolctl.threadpool_info()` only, so every arm on this box
reported `kernel = "unknown"` — which is why WP-A22 saw the arm as
`WIN-unknown-t1`, and why four different kernels would all have wanted the
same arm key.  It is also exactly CI's situation (`CI_PREMISE_GATES` §2.1:
the runner says so itself, which is why the transcribed arm is
`CI-unknown-t1`).

I did **not** install it: several other work packages have measured and
reported behaviour that depends on its absence, and installing it mid-campaign
would silently change their results.  Instead `probe_decisions._arm_id` gained
a `threadpoolctl`-free fallback that performs the SAME read-back with
`ctypes` — `openblas_get_corename()`, the runtime-dispatched micro-kernel,
which is what `threadpoolctl` itself calls.  The census's "measured, not
requested" invariant is preserved exactly; the request is still never
recorded as the kernel.

Wheel symbol spellings, MEASURED (the plain names are absent; the
scipy-openblas wheels rename them so numpy's ILP64 and scipy's LP64 builds can
coexist):

| library | export |
|---|---|
| `libscipy_openblas-<hash>` (scipy 1.17.1) | `scipy_openblas_get_corename` |
| `libscipy_openblas64_-<hash>` (numpy 2.4.6) | `scipy_openblas_get_corename64_` |

Loaded-image enumeration is `/proc/self/maps` first (dependency-free, and the
only route in the WSL venv, which has no `psutil`) then `psutil`.

**The kernel ladder on this host**, corename read back per request, both
libraries agreeing on every rung:

| `OPENBLAS_CORETYPE` | dispatched |
|---|---|
| *(unset)* | **SkylakeX** |
| `SKYLAKEX` | SkylakeX |
| `HASWELL` | Haswell |
| `SANDYBRIDGE` | Sandybridge |
| `NEHALEM` | Nehalem |
| `KATMAI` / `PRESCOTT` | Katmai |
| `ZEN` | Haswell |
| `BOGUSCORE` | SkylakeX |

Both 2026-09-11 caveats survive the change of host — `ZEN` is not a distinct
kernel, an unrecognised name falls back to auto-detection — and the second one
**inverts**: `SKYLAKEX` was unreachable there (SIGILL on a non-AVX-512 host)
and is the AUTO-DETECTED default here.  So this box reaches **five** kernels
where the 2026-09-11 host reached four, and the extra one is the interesting
one (§3.3).

### 3.2 What was recorded

**12 LIVE arms**, one process each (`OPENBLAS_CORETYPE` is read at BLAS load
time, so the kernel cannot be switched inside a process):

```
WIN-SkylakeX-t1  WIN-Haswell-t1  WIN-Sandybridge-t1  WIN-Nehalem-t1  WIN-Katmai-t1  WIN-SkylakeX-t4
WSL-SkylakeX-t1  WSL-Haswell-t1  WSL-Sandybridge-t1  WSL-Nehalem-t1  WSL-Katmai-t1  WSL-SkylakeX-t4
```

2 builds × 5 kernels × 2 thread widths, 61 decisions and 6 hypotheticals each,
every one carrying `recorded` = 2026-09-12, `tree` = `audit-fixes-2026-09
2622449f +dirty`, `lumenairy` = 5.45.1, `min_feature_default_frac` = 0.001,
`kernel_source` = `ctypes(openblas_get_corename)` and the per-library corename
/ thread-width read-back.

**The other arms are MARKED, not deleted** — which is what the brief asked
for, and there are two reasons beyond the instruction.  They are the only
evidence of a host that could not execute AVX-512, and they are the
before-side of the change this round is about.  The eleven 2026-09-11 arms
carry `"historical": true`, `recorded` = 2026-09-11 and `tree` = "commit
`da377e0b` … BEFORE WP-A12's `56a76f22` raised `_MIN_FEATURE_DEFAULT_FRAC`",
and are re-keyed `<arm>@2026-09-11` by `merge_arms.py`.  **Not one measured
value in them was touched** — `git diff --numstat` reads +5 / −1 per file (the
deletion is the reflowed closing brace), and every `decisions` / `classes` /
`hypothetical` / `readings` block plus every identity field compares equal to
the `HEAD` blob.  The re-key is necessary rather than cosmetic: `WIN-Haswell-t1`
now exists twice, once per tree, and that pair is the finding.

The synthetic CI arm is unchanged apart from gaining the same `recorded` /
`tree` provenance fields.

**Census totals: 24 arms — 12 live, 11 historical, 1 synthetic.**

### 3.3 The design decision, and where it is documented

The gate's coverage requirements had to be split, because a requirement that
an archived arm can satisfy is satisfied forever — which is exactly how the
2026-09-11 table kept passing its span check while its rows described a
library that no longer existed.  Documented in
`test_ci_kernel_consistency.py`'s module docstring (THE THREE PROVENANCE
KINDS) and in each affected test's own docstring:

| requirement | asserted of | why |
|---|---|---|
| ≥ 6 arms, ≥ 2 builds, ≥ 2 kernels, ≥ 2 widths incl. `t1` and a multi-thread arm | **measured** arms (live + historical) | unchanged in substance; a historical arm was a real run and is real coverage |
| ≥ 6 **live** arms, ≥ 2 kernels, ≥ 2 widths, all carrying this file's `recorded` date and `tree` prefix | **live** arms only | the anti-staleness requirement that did not exist before, and the direct lesson of this finding.  Kernels and widths, not builds: a second build is a whole second interpreter and wheel set and may genuinely be unavailable, while a second kernel is one environment variable away on any DYNAMIC_ARCH build |
| every arm answered every row ("no holes") | **live** arms only | an archived arm covers only the rows that existed when it was taken |
| every row an arm carries is a row the live probe still produces ("no orphans") | **archived** arms | this is what bounds the exemption above: a renamed fixture would otherwise leave an archived row comparing against nothing |
| every historical arm carries a date, a tree, and the `@date` key suffix | historical arms | an undated old reading is folklore, not evidence.  `merge_arms.py` refuses to merge a historical arm with no `recorded`, and refuses an arm marked both historical and synthetic |
| the 1e-12 hypothetical bar is non-unanimous | the whole census **and, new, the live arms alone** | otherwise the argument for leaving the plain-1-D site unguarded could come to rest entirely on history |

### 3.4 The result

```
rule violations across all 24 arms ................ NONE
guard decisions differing at the SAME class ....... NONE
rows whose CLASS differs between arms (reported) .. 4
    pmm1d_interface/{answer,returns,warned}@1e-05, sliver/pmm1d@1e-05
live rows per arm 61 | historical 54 | synthetic 24 | cheap union 31
live holes {} | archived orphans {}
```

### 3.5 The 2026-09-11 OPEN item, closed

`CI_PREMISE_GATES_2026_09_11.md` §2 recorded as **OPEN** why the CI runner
answers these fixtures correctly, having ruled out the thread width, the numpy
version, the OS and — it believed — the BLAS micro-kernel.  It could not rule
the kernel IN, because the 2026-09-11 host could not execute the AVX-512
kernels at all.  This host can:

| arm | `R+T` @ 1e-04 | `R+T` @ 1e-05 | class | sliver |
|---|---|---|---|---|
| **CI-unknown-t1** (transcribed) | 1.0000003658397656 | 1.0000010471871335 | correct | return |
| **WSL-SkylakeX-t4** (measured here) | **1.0000003658397656** | **1.0000010471871335** | correct | return |
| WSL-SkylakeX-t1 | 1.00000037426296 | 1.0000010472218661 | correct | return |
| WIN-SkylakeX-t1 | 1.0000004784011818 | **3.6124215324602997** | wrong | refuse |
| WIN/WSL Haswell, Sandybridge, Nehalem, Katmai (8 arms) | ~1.0000004 | 2.1702 … 3.6134 | wrong | refuse |

`WSL-SkylakeX-t4` reproduces the transcribed CI runner **bit for bit on both
readings**, confirmed over two independent runs.  `rcond` matches too
(9.7296703051673e-13).  So the CI arm is a **Linux + AVX-512** arm, the
divergence IS a micro-kernel effect, and the campaign now has a local
reproduction of it instead of a transcription.  The `WIN-SkylakeX-t1` row is
what makes it a statement about build-and-kernel rather than about either
alone: same silicon, same kernel name, same numpy and scipy, opposite class.

Two consequences worth the orchestrator's attention:

* the premise gates the campaign wrote are not defensive over-engineering —
  the arm they were written for is now one `wsl` invocation away, and
  `CI_PREMISE_GATES` §7's table of "skips expected on CI" is testable locally;
* `CI_PREMISE_GATES` §2.3's row "a Zen-specific BLAS micro-kernel → **not the
  cause**" and its premise that the runner is AMD EPYC 7763 are refuted as an
  explanation.  That premise was never measured (§2.2 says so: the logs print
  no CPU model).  §8 item 1's proposal — print `numpy.show_config()` and
  `/proc/cpuinfo` in the matrix — is now worth doing to confirm it from the
  runner's side, and is recorded below as a request.

---

## 4. Deliverable 3 — the premise-gated families

### 4.1 Re-measurement

The six families (18 files per `CI_PREMISE_GATES` §6) plus the two WP-A12
files, run on the default arm with `-rs`:

```
268 passed, 2 skipped, 0 failed in 382.33s
```

Per file, the premise-gated assertions and what they did on this tree:

| file | premise-gated claim(s) citing the divergence | result here | action |
|---|---|---|---|
| `test_fix_pmm2d_mortar_round2.py` | "the 1e-05 row is WRONG at all" | **premise HELD** — the fixture pins `min_feature=_P*1e-5` (VERIFY-A12 §8.1), so G2 never reached it | note only |
| `test_fix_pmmstack_sliver_walls.py` | `wrong >= 10` on the 60-row dense grid | **SKIPPED**: 9 wrong / 51 right / 0 grey.  `_solve` pins `min_feature=_MF`, so this is **not** G2 — and it is a KERNEL effect, isolated by re-running the same file under `OPENBLAS_CORETYPE=HASWELL`, where it **runs** (§5.3).  SkylakeX is this box's default and no census before this round had measured it | note only; §7 item 5 |
| `test_fix_pmmstack_sliver_walls_round2.py` | "the guard says something here"; "at least one LC class is WRONG" | **premise HELD** | note only |
| `test_fix_pmmstack_sliver_round3.py` | "a truncation-noted row exists" | **premise HELD** | note only |
| `test_fix_pmmstack_sliver_round4.py` | `len(wrong) >= 4`, `len(attributed) >= 4`, `small >= 1` | **premise HELD** | note only |
| `test_verify_pmmstack_sliver_round2.py` | both population readings | **premise HELD** | note only |
| `test_verify_pmmstack_sliver_walls.py` | remedy slope; thin-feature spread | **premise HELD** | note only |
| `test_verify_pmmstack_sliver_round3.py` | R3-A; false refusal | **premise HELD** | — |
| `test_m1_conditioning_guard.py` | M1 X-1 at ≥ 10× | **premise HELD** | — |
| `test_fix_branch_cut_round2.py`, `test_verify_branch_cut_round2.py` | pre-arm spread | **premise HELD** | — |
| `test_audit2609_a12_pmm1d.py` (3 splits) | tripwire rows; old-default scatter; truncation fork | **premise HELD** on all three | note only |
| `test_audit2609_a12_verify_pmm1d.py` | cites the divergence in prose only | n/a | note only |

The second skip is the documented numpy-vs-scipy LAPACK bit-identity gate,
which `CI_PREMISE_GATES` §6 records as skipping **on every arm including both
local builds**; it reported worst relative residual 1.712e-15 against a
1.023e-12 bar and worst `|guarded − numpy|` exactly 0.  Unrelated to this
round.

**So no gated half is skipping for a stale reason, and none needed
restating on a different fixture.**  The brief's contingency ("where the
premise no longer holds … restate them on a fixture that still exhibits the
build property, pinning `min_feature=period*1e-5` if that is what the premise
needs, or fold them into their unconditional siblings") did not arise,
because WP-A12 and VERIFY-A12 had already pinned `min_feature` on every
fixture that needed it.  The census fixture was the ONE that had been missed —
and it is now pinned, by the same remedy, in `probe_decisions._1D_MF_PINNED`.

### 4.2 What was updated, and why it was not nothing

All eight files that cite the `3.61 vs 1.000115` kernel readings say, in
prose, that the correct reading belongs to "the CI runner's kernel".  After
§3.5 that framing is misleading: the reading is reproducible here.  Each of
the eight gained ONE dated block in its module docstring recording the
re-measurement, the bit-identity, and the instruction to read "the CI runner's
kernel" as "a Linux AVX-512 (SkylakeX) arm".  **Docstrings only — not one
assertion, bar, fixture or import was touched in any of the eight** (`git
diff`: +12 lines each, all inside the module docstring).

---

## 5. Deliverable 4 — tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`-q --no-header -p no:cacheprovider`.

| command | result |
|---|---|
| `test_ci_kernel_consistency.py` (the RED one) | **8 passed**, 2.45 s (7 ids before this round, one of them failing) |
| `test_audit2609_a23_census_mechanism.py` (new) | **5 passed**, 1.46 s |
| the 20-file premise ladder (six families + both A12 files), **default arm = SkylakeX** | **268 passed, 2 skipped, 0 failed**, 382.3 s |
| the same 20 files under **`OPENBLAS_CORETYPE=HASWELL`** | **269 passed, 1 skipped, 0 failed**, 411.1 s |
| **final consolidated run** — every file this WP touched, after every edit: the two above + `test_fix_pmm2d_mortar_round2.py` + the eight noted files | **156 passed, 2 skipped, 0 failed**, 181.0 s |
| `ruff check validation/probe_ci_kernel_sweep/` + the three test files + the eight noted files | **All checks passed** |
| `probe_decisions.py` on 12 arms + `make_ci_arm.py` + `merge_arms.py` | 61 decisions per live arm; merged 24 arms |

### 5.3 The two arms, side by side — and a kernel effect isolated

Running the SAME 20 files on two kernels of this box is the cheapest
independent check that nothing in this round is arm-specific, and it isolated
one of the two skips:

| arm | result | skips |
|---|---|---|
| default (**SkylakeX**, AVX-512) | 268 passed, 2 skipped | numpy-vs-scipy LAPACK; **dense grid at 9 WRONG rows against a bar of 10** |
| `OPENBLAS_CORETYPE=HASWELL` | **269 passed, 1 skipped** | numpy-vs-scipy LAPACK only |

So `test_fix_pmmstack_sliver_walls.py::test_the_guards_DECISION_is_right_on_a_
dense_grid_not_just_this_ladder` skips on SkylakeX and runs on Haswell — its
fixture pins `min_feature`, so this is a **micro-kernel** property, not a
consequence of G2, and the gate is doing exactly what it was built to do.
Reported in §7 item 5 because the bar has one row of margin on what is now
this host's default kernel.

### 5.4 Every new bar, exercised against a mutated table

A gate nobody has broken on purpose is a gate nobody has tested.  Each new
assertion was run against a copy of the census with one thing changed
(the real table is never written):

| mutation | caught by |
|---|---|
| a live arm loses its `recorded` date | freshness check |
| a live arm claims a different `tree` | freshness check |
| every live arm demoted to historical (the table goes archival) | "census carries 0 LIVE arm(s)" |
| the live arms narrowed to one kernel | live-kernel span |
| a live arm loses a row | census-hole check |
| an archived arm carries a row the live probe no longer produces | orphan check (verified with a key carrying NO rule prefix, so the rule check could not catch it first) |
| the 1e-12 bar goes unanimous **on the live arms only** | the new live non-unanimity assertion |
| no measured arm reads the 1e-05 row correct any more | "the only arm reading it CORRECT is the transcription" |
| the CI bit-identity drifts by **one ULP** | the bit-identity assertion |
| a guard decision flips at the same class | the pre-existing same-class check (still has teeth) |
| an arm marked both `historical` and `synthetic` | `merge_arms.py` exit **3** |
| a historical arm with no `recorded` | `merge_arms.py` exit **4** |
| two live arms resolving to the same kernel | `merge_arms.py` exit **2** (pre-existing, still discriminates) |

**13 of 13 caught**, and the unmutated table passes all four tests.

### 5.1 Fail-before, stated as what was actually confirmed

* **The rule test.**  WP-A22 §4(a) measured the HEAD blob of
  `test_ci_kernel_consistency.py` failing identically against this tree, with
  the two `warned@*` rows named.  I reproduced the same two violations in
  process (§2.1, SHIPPED-default pass) before changing anything, and they are
  gone in the same tree after the instrument fix.
* **The premise gate in the new file.**  Verified to SKIP rather than fail on
  a CI-class arm, by repointing the test module's `_MF_OLD_FRAC` in process
  with a collection-time plugin:

  ```
  SKIPPED [1] test_audit2609_a23_census_mechanism.py:285: premise absent on
  this arm: the PINNED census fixture (min_feature = period*1e-5) already
  answers the 1e-05 wall separation correct here -- max(R+T) =
  1.000000000000063, |R+T - 1| = 6.284e-14 ... The UNCONDITIONAL claims
  passed ...
  ```

  The plugin is deliberately coarse — it also trips the two STRUCTURAL pins
  that assert the probe's pin really is 1e-5, which is those tests working —
  so the claim verified here is narrow and exact: the gated test skips with
  its reading and names its unconditional sibling, and does not fail.
* **The new census bars.**  Thirteen mutations, thirteen caught — the table
  is in §5.4.

### 5.2 Pre-existing failures found

None, on either arm.  The two skips are analysed in §4.1 and §5.3.

---

## 6. Files touched

**Modified — the census (`validation/probe_ci_kernel_sweep/`)**

* `probe_decisions.py` — the `ctypes` kernel read-back (`_blas_read_back_ctypes`,
  `_loaded_blas_paths`, `_arm_id` now returns the source and the per-library
  detail); `min_feature` PINNED as a fixture parameter (`_1D_MF_PINNED`,
  `_1D_CASES`, `_pmm1d_two_layer(delta, mf_frac)`); the voice classifier
  (`_voices`, `_is_env_advisory`, `_notice_label`, `_VOICE_*`); the
  `notices@` decision row; `recorded` / `tree` / `lumenairy` /
  `min_feature_default_frac` in every arm document (`_tree_id`); docstring
  corrections for both measured caveats.
* `merge_arms.py` — the three provenance kinds, the `@date` key for historical
  arms, the refusals (both-kinds, undated-historical), `_mechanism` and
  `_provenance` blocks in the merged table, the new per-arm fields.
* `make_ci_arm.py` — `recorded` / `tree` on the synthetic arm.
* `arms/win_*.json`, `arms/wsl_*.json` (11 files) — **provenance fields only**;
  no measured value changed.
* `arms/win_live_*.json`, `arms/wsl_live_*.json` (**12 new**) — the live arms.
* `arms/ci_RUNNER_t1.json` — regenerated by `make_ci_arm.py`.
* `decisions.json` — regenerated, 24 arms.

**Modified — tests**

* `tests/unit/test_ci_kernel_consistency.py` — the module docstring
  (re-record, provenance kinds, the closed OPEN item, the corrected caveats,
  the regeneration procedure); `_live` / `_archived`; the freshness and
  provenance requirements; holes over live arms + the orphan check; the live
  non-unanimity assertion; `_RETAKE_DECISIONS` 24 → 31, `_RETAKE_HYPOTHETICALS`
  4 → 6, `_LIVE_RECORDED` / `_LIVE_TREE_PREFIX`; `_arm_id`'s new arity;
  `test_the_census_carries_the_ci_arm_…` → `test_the_census_carries_an_arm_…`,
  restated around the class-divergence and the bit-identity; one new test
  (`test_the_1d_section_still_pins_the_min_feature_that_makes_it_a_fixture`).
* `tests/unit/test_audit2609_a12_pmm1d.py`, `test_audit2609_a12_verify_pmm1d.py`,
  `test_fix_pmmstack_sliver_round3.py`, `test_fix_pmmstack_sliver_round4.py`,
  `test_fix_pmmstack_sliver_walls.py`, `test_fix_pmmstack_sliver_walls_round2.py`,
  `test_verify_pmmstack_sliver_round2.py`, `test_verify_pmmstack_sliver_walls.py`
  — **one dated docstring block each**, nothing else.

**New**

* `tests/unit/test_audit2609_a23_census_mechanism.py` (5 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A23_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A23_CHANGELOG.md`

**NOT touched**: anything under `lumenairy/`; `CHANGELOG.md`, `README.md`,
`Migration-Guide.md`, `CONVENTIONS.md`, `pyproject.toml`.

---

## 7. Requested changes outside my ownership

1. **`pyproject.toml` / the environment owner — install `threadpoolctl`.**  It
   is a declared CORE dependency (`pyproject.toml:97`, with a measured 400×
   justification in its own comment) and is absent from BOTH interpreters on
   this workstation, which makes `set_blas_threads` inert on a 20-thread box.
   I did not install it: other work packages have measured and reported
   behaviour that depends on its absence, and changing that mid-campaign would
   invalidate their results.  Worth a deliberate, announced install at
   campaign close, followed by one re-run of this census (the arm names will
   then come from `threadpoolctl` instead of the `ctypes` fallback; the
   fallback stays, because CI does not have it either).
2. **The WSL environment owner** — the reachable WSL build is a bare venv
   (`~/qwp-gpu`) without `psutil`, `threadpoolctl` or `pytest`.  It can run the
   census probe but not the test ladder, so `CI_PREMISE_GATES` §7's predicted
   CI skips still cannot be confirmed locally end to end.  A lumenairy venv on
   the WSL side would let the campaign run the whole 18-file ladder on the arm
   that answers these fixtures correctly — which, after §3.5, is the most
   valuable arm there is.
3. **`docs/audits/CI_PREMISE_GATES_2026_09_11.md` and
   `docs/audits/CI_KERNEL_SWEEP_2026_09_11.md`** (not mine) — §2.3's "a
   Zen-specific BLAS micro-kernel → not the cause" and the AMD EPYC 7763 /
   "local default arm IS the CI kernel" premise are refuted by §3.5 above, and
   §2's OPEN item is answered.  The corrections are carried in
   `test_ci_kernel_consistency.py`'s docstring so the code does not mislead;
   the audit documents should follow.
4. **`docs/audits/CI_PREMISE_GATES_2026_09_11.md` §8 item 1** — printing
   `numpy.show_config()`, `scipy.show_config()` and `/proc/cpuinfo` in the
   matrix log is now worth the CI run: it would confirm from the runner's side
   what §3.5 establishes from this one.
5. **`test_fix_pmmstack_sliver_walls.py`'s owner** — the dense-grid premise
   reads 9 WRONG rows against a bar of 10 on this box's default kernel
   (SkylakeX) and skips; under `OPENBLAS_CORETYPE=HASWELL` the same file on
   the same tree **runs** (§5.3), so the population is kernel-sensitive at
   exactly the bar.  The fixture pins `min_feature`, so this is not G2; the
   skip carries the full reading and the unconditional half still runs, so
   the gate is working as designed.  Flagged because it is a NEW skip since
   2026-09-11 (where all ten arms reported "221 passed, 1 skipped"), and
   because `wrong >= 10` against a measured 9 has no margin left on what is
   now this host's default kernel.  A population bar with one row of slack is
   the shape TESTING_STANDARDS S5 warns about; re-deriving it over the two
   kernels now measurable here would be a half-hour of work and would make
   the gate say something on both.

---

## 8. Deferred, with designs

1. **Record the per-section row counts in `decisions.json` at merge time** so
   `_RETAKE_DECISIONS` need not be hand-edited when the table is regenerated.
   This is WP-A22 §6 item 2, still open, and this round made it more valuable
   by moving the number (24 → 31).  `merge_arms.py` change, ~1 h.
2. **A `tauto` live arm.**  The census carries a historical unpinned arm
   (`WSL-Haswell-tauto@2026-09-11`, 24 threads) but no live one; CI's fast
   lane runs unpinned, so it is the faithful arm.  It was skipped here for
   budget (the probe is a few hundred small solves and thread spawn dominates
   on 20 threads).  ~20 min plus whatever the run costs.
3. **A second `env` advisory class member.**  `_is_env_advisory` matches the
   library's advisory idiom (a `RuntimeWarning` saying a package is "not
   installed" and asking the user to install it).  Exactly one such advisory
   exists today.  If a second appears with different wording it will land in
   `other` and fail a `notices@` comparison — visibly, which is the right
   failure, but the matcher should then be widened deliberately rather than
   in a hurry.
4. **The two skips in §4.1** are the census owner's to watch, not to fix now.

---

## 9. Path to the changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A23_CHANGELOG.md`

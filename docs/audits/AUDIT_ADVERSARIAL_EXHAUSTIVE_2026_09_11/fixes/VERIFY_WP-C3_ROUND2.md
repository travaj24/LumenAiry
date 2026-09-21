# VERIFY-WP-C3 ROUND 2 — narrow independent re-verification of `feat/c3-collins-round2`

Branch under test: `feat/c3-collins-round2` @ `30a95553` (14 commits on
`feat/c3-collins-default` @ `4d87ff48`).
Chain: `4d87ff48` → `verify/c3-collins-default` @ `50ec3ea4`
(`fixes/VERIFY_WP-C3.md`) → round 2.
Verification branch/worktree: `verify/c3-collins-round2`, `C:/tmp/lum_vc3b`.
PRE trees: `git archive 49ddf4bd` → `C:/tmp/vc3b_base49`,
`git archive 4d87ff48` → `C:/tmp/vc3b_pre4d`,
`git archive aa198ad0` → `C:/tmp/vc3b_one` (the one-line fix commit),
`git archive 50ec3ea4` → `C:/tmp/vc3b_ver50`.
Builds: **WIN-py3.14** (3.14.6, numpy 2.4.4) and **WSL-py3.12** (3.12.3,
numpy 2.4.6, `~/lumvenv`).  Every probe pins `PYTHONPATH` and asserts
`lumenairy.__file__` under the tree; every run carries
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line and every pytest run carries `--capture=sys`.
Probes and JSON: `validation/probe_verify_c3_round2/`.

**The scope was deliberately narrow — five items, named by the maintainer.**
Nothing else on the branch was re-verified, and nothing in `lumenairy/` was
edited.

---

## 0. VERDICT TABLE

| item | verdict | the number that decides it |
|---|---|---|
| **1** D5/D6 — one line closed both | **CONFIRMED**, both builds | 192-cell census reads **192 / 0 / 0**, 0 Kelly warnings; 12-config reproducer **12 / 12**; 103 archive keys **0 ok→raised**; D6 ladder 1295.3594 / 1278.4007 / 1274.1529 um, bit-identical to `'sziklas'`.  One documentation defect (**V1**): the PRE-fix split is 119 / 51 / 22 here and 117 / 53 / 22 in the prior verification's own WSL data, not 118 / 52 / 22 |
| **1b** is the flip byte-identical on ordinary relays? | **YES, and the flat-reference class is the ENTIRE difference** | `git archive aa198ad0` — whose only executable change in `lumenairy/` is deleting `and not flat` — already reads 192 / 0 / 0 on both builds, and the round-2 tip is bit-identical to it on all 192 cells.  **0 of 192 (0.0 %)** ordinary chains move, against the prior verification's 61.5 % bit-identical |
| **2** D8 — the standoff keyword | **CONFIRMED**, both builds | Collins standoff leg relL2 **4.750712e-05** against my own oracle (report: 4.7340e-05); pinned leg **2.4049894840869106** (report: 2.4049894966524197); scale 0.99999006, peak ratio 0.99997617 exactly as published; flipping the default turns **exactly 8** a6 ids red.  One documentation defect (**V2**): `carrier.py:4739` says the opposite of the shipped default |
| **3** D13 — the C3 x C5 merge, rebuilt | **CONFIRMED, GREEN**, against C5's real final tip `575020c4` | `carrier.py` merges with ZERO conflicts at `_GAP_KERNEL_ACCURACY_TAU = 1e-4` beside `transport: str = 'collins'`; **106 passed** (h2 + both C3 files + `fix_v1_v8` + the nine C5-listed ids), **130 passed** (b4 whole), **18 passed** (h2 alone, 17 on the branch).  The five d3 bars were **RE-RECORDED, not re-derived — and re-derivation was not required**: their three quantities are bit-identical on the two trees and the ids reach **zero** Collins legs |
| **4** the p8 open item | **SETTLED — the FIELD is right; only the returned LATTICE is coarse.  NOT a P1** | on one common lattice the two transports read EE80 **11.0575 um** and **11.0568 um** (0.006 % apart) and their intensity maps agree to **1.29e-04** relative L2; the same Collins leg driven at `dx_out` = 0.25/0.5/1/2/3 um reads 11.0090 / 10.9921 / 11.0708 / 10.9417 / 10.9083 um against the oracle's 11.1102 um, and reads 12.3588 um only on its own floored 6.3126 um pitch |
| **5** the 54-file blast set | **GREEN**, both builds | Windows (b4 included) **`1888 passed, 4 skipped, 273 warnings in 3440.39s (0:57:20)`**; WSL (b4 excluded) **`1757 passed, 5 skipped, 273 warnings in 3874.08s (1:04:34)`** |

**SHIP RECOMMENDATION: SHIP.**  Every blocker the prior verification raised is
closed on the measurements it asked for, the merge with WP-C5's final tip is
green, and the one item round 2 left open resolves in the default's favour.
Three documentation defects below (V1, V2, V3) should be applied first; none
of them changes code and none of them blocks.

---

## 1. D5 / D6 — the one line, re-measured

### 1.1 The censuses

`validation/probe_verify_c3/probe_vc3_blastwidth.py` run unchanged on four
trees, classified with `validation/probe_c3_round2/r2_blast_compare.py`
against `git archive 49ddf4bd`:

| tree | default | IDENTICAL / MOVED / OK→RAISED | Kelly warnings / cells |
|---|---|---|---|
| 49ddf4bd (base) | `'sziklas'` | — | 0 / 0 |
| 4d87ff48 (pre round 2) | `'collins'` | **119 / 51 / 22** | 74 / 51 |
| **aa198ad0 (the one-line fix alone)** | `'collins'` | **192 / 0 / 0** | **0 / 0** |
| **30a95553 (round-2 tip)** | `'collins'` | **192 / 0 / 0** | **0 / 0** |

Identical on WIN-py3.14 and WSL-py3.12.  `aa198ad0` vs the round-2 tip is
itself 192 / 192 IDENTICAL, so the other thirteen round-2 commits move
nothing on this space.

`validation/probe_verify_c3/probe_vc3_newraise_indep.py` (the 12-configuration
reproducer), both builds:

| tree | returns | raises |
|---|---|---|
| 49ddf4bd | 12 / 12 | 0 |
| 4d87ff48 | 1 / 12 | **11** |
| **30a95553** | **12 / 12** | **0** |

and the round-2 tip's peak on all 12 rows is bit-equal to the base tree's,
on both builds.

The 103-key archive-to-archive set
(`validation/probe_verify_c3/run_wayback.py`, three arms, both builds):

| comparison | shared | identical | differ | transitions |
|---|---|---|---|---|
| base default vs round-2 default | 103 | 54 | 49 | 47 ok→ok, **2 raised→ok**, **0 ok→raised** |
| base default vs round-2 `transport='sziklas'` | 103 | **103** | 0 | — |

### 1.2 The other direction, which is the question worth asking

**"192 identical" IS the statement that the Collins default is byte-identical
to 5.48.1 on ordinary relays** — the census hashes the chain's returned field
with SHA-256, so an identical cell is an identical array.  Stated plainly:

* **the flip changes 0 of 192 (0.0 %) ordinary relay chains.**  The prior
  verification measured 61.5 % bit-identical / 27.1 % moved / 11.5 % raising;
  it is now 100 % / 0 % / 0 %;
* **the flat-reference class is the ENTIRE difference.**  `aa198ad0`'s only
  executable change in `lumenairy/` is deleting `and not flat` from
  `tf_available` (everything else in that commit is comment and test), and it
  alone takes the census from 119 / 51 / 22 to 192 / 0 / 0 and the Kelly count
  from 74 to 0.  Every moved and every raising cell was that one leg;
* **this does NOT mean the flip is inert.**  On the package's own 103-key set —
  which deliberately includes the focus crossing, past-focus, astigmatic and
  fine-grid geometries an ordinary relay does not visit — **47 of 103 (45.6 %)
  answers still move**, and 2 calls that RAISED at 5.48.1 now return.  The
  honest sentence is that the flip is bit-identical where the chirp-Z is not
  representable (which is what an ordinary relay leg is) and buys its accuracy
  where it is.

### 1.3 The D6 exit radius

`validation/probe_verify_c3/probe_gridladder.py`, window `N·dx` = 10.24 mm
fixed, plus `probe_vc3_moment_oracle.py`.  Identical to every printed digit on
both builds:

| N | pre round 2 (default) | **round 2 (default)** | round 2 `'sziklas'` | 49ddf4bd default | second-moment oracle | ratio |
|---|---|---|---|---|---|---|
| 256 | 3823.7543 | **1295.3594013252132** | 1295.3594013252132 | 1295.3594013252132 | 1413.3041 | **0.9165** |
| 512 | 3847.8052 | **1278.4007207813852** | 1278.4007207813852 | 1278.4007207813852 | 1281.0221 | **0.9980** |
| 1024 | 4099.0717 | **1274.1528507772896** | 1274.1528507772896 | 1274.1528507772896 | 1157.1133 | **1.1011** |

The default's record is bit-identical to `'sziklas'` on the same tree AND to
the base tree's default, peak included, on every rung; the Kelly warning the
leg used to emit is gone (1 → 0 per rung).  **CONFIRMED.**

### 1.4 Defect V1 — documentation. The PRE-fix 118 / 52 split is a per-run reading, not a fact

The "before" column is quoted as `118 / 52 / 22`.  Three independent
measurements of the same two commits disagree about it:

| measurement | IDENTICAL / MOVED / OK→RAISED |
|---|---|
| this verification, WIN-py3.14 (fresh archives) | **119 / 51 / 22** |
| this verification, WSL-py3.12 (fresh archives) | **119 / 51 / 22** |
| `validation/probe_verify_c3/blastwidth_{base,branch}_win.json` (prior verification) | 118 / 52 / 22 |
| `validation/probe_verify_c3/blastwidth_{base,branch}_wsl.json` (prior verification) | **117 / 53 / 22** |

The cause is one cell, `f60|g2|N512|w1.5|r0.06|fd5.0|ro0`.  Every Windows run
of it anywhere — my base tree, my base tree re-run, my pre-round-2 branch, my
one-line tree, my round-2 tip, and the prior verification's own BRANCH run —
reads peak `0.8201103961582621`; the prior verification's Windows BASE run
alone reads `0.8201103961906113`, 3.9e-11 relative away, which classified that
cell MOVED in its table and IDENTICAL in mine.  My own base run is
bit-reproducible (two full 192-cell runs, 0 differing cells), so this is a
cross-SESSION difference on the base commit, not a property of either tree.

Only **OK→RAISED = 22** and the Kelly counts **74 warnings over 51 cells** are
stable across all five runs and both builds.  The IDENTICAL/MOVED split is not:
its boundary sits inside the cross-session spread of one cell, which is
`docs/TESTING_STANDARDS.md` shape S5 applied to a census.

**Requested edits** (documentation only; the round-2 conclusion is unaffected,
because the 22 raising cells and the 74→0 Kelly count are the load-bearing
numbers and both reproduce exactly):

1. `docs/audits/.../WP-C3_COLLINS_DEFAULT_REPORT.md:919`.
   OLD: `| 192 ordinary chain cells: IDENTICAL / MOVED / OK->RAISED | — | 118 / 52 / **22** | **192 / 0 / 0** |`
   NEW: `| 192 ordinary chain cells: IDENTICAL / MOVED / OK->RAISED | — | 119 / 51 / **22** (the IDENTICAL/MOVED split moves one or two cells between SESSIONS -- 118/52 and 117/53 were also measured, on one cell whose base-tree peak differs by 3.9e-11; the 22 and the Kelly counts are stable) | **192 / 0 / 0** |`

2. `docs/audits/.../WP-C3_COLLINS_DEFAULT_REPORT.md:928-929` — "All 52 MOVED
   cells of the 192 were the same aliased flat-reference leg" → "Every MOVED
   and every RAISING cell of the 192 was the same aliased flat-reference leg
   (51 and 22 as measured here; the MOVED count is a per-session reading, see
   the table)".

3. `docs/history/carrier.md:20` — `192-cell ordinary-chain census 118/52/22 ->
   192/0/0` → `192-cell ordinary-chain census 119/51/22 -> 192/0/0 (the
   IDENTICAL/MOVED split is a per-session reading; 22 -> 0 and 74 -> 0 Kelly
   warnings are the stable halves)`.

4. `Migration-Guide.md:1869` — "(An earlier cut of this release moved 52 of
   those 192 cells and raised on 22; ..." → "(An earlier cut of this release
   moved about 51 of those 192 cells and raised on 22; ...".

---

## 2. D8 — the focus readout's `transport` keyword

`carrier_referenced_focus_readout` now takes `transport`, keyword-only,
defaulting to `'sziklas'` (confirmed by `inspect.signature`).

### 2.1 The headline numbers, on my own oracle

`validation/probe_verify_c3_round2/vr2_d8_standoff.py` drives the SHIPPED
keyword rather than monkey-patching `propagate_carrier_referenced`, and grades
against two oracles of its own: a composite-Simpson dense separable Fresnel
quadrature at 512x the input pitch (self-convergence 256x → 512x =
**4.35e-16**), cross-checked against the untruncated analytic ABCD Gaussian
(normalised amplitude agreement **8.10e-05**, i.e. the 1.05 % window
truncation is a tail effect).  Fixture as published: `N = 128`, `dx = 4 um`,
`lambda = 633 nm`, `w = 120 um`, `R = -30 mm`, `z = 30 mm`,
`standoff = 1 mm`, `dx_out = 0.2 um`, `N_out = 32`.

| standoff leg | relL2 vs my oracle | best global scale | peak ratio | published |
|---|---|---|---|---|
| **`transport='collins'`** | **4.750711683025854e-05** | 0.9999900594377032 | 0.999976170447058 | 4.7340e-05 / 0.99999006 / 0.99997617 |
| `transport='sziklas'`, guard → `'warn'` | **2.4049894840869106** | 2.2738346686107413 | 5.143274054224108 | 2.4049894966524197 / 2.2738 / 5.1433 |

The scale and peak ratio reproduce to every published digit; the two relL2
figures agree to 12 digits (Sziklas) and 0.35 % (Collins, and 0.35 % is the
difference between two quadratures, mine being 9 decades better converged).
Identical on both builds.  **CONFIRMED.**

Worth stating separately, because it is the substance of the decision and it
is still live: **on this fixture the SHIPPED DEFAULT raises.**
`transport='sziklas'` gives
`RuntimeError: ... the beam does not fit the co-moving grid at the stop
plane.  The grid half-width there is 8.5333 um against a measured amplitude
radius of 4.6391 um`, while `transport='collins'` returns an answer good to
4.75e-05.

### 2.2 The 30-cell sweep, and what I could not reproduce

My own five geometries x six standoffs (mine, not the package's — the
published geometry list is not in the tree):

| leg | raises | relL2 max | ≤ 5.3e-04 |
|---|---|---|---|
| `'sziklas'` | **13 / 30** | 0.2541 (of the 17 that return) | 5 / 17 |
| `'collins'` | **0 / 30** | **1.502e-04** | **30 / 30** |

Direction, sign and magnitude all confirm the published claim; if anything the
pinned leg is worse on my set.  **The exact published count "7 of 30" I could
not reproduce and could not refute**, because the five geometries and six
standoffs it was measured on are not recorded anywhere in the tree.

### 2.3 The default's cost, re-measured

Flipping the keyword's default in place (`__kwdefaults__`, no source edit;
plugin `validation/probe_verify_c3_round2/vr2_flip_readout_default.py`) and
running the two a6 files:

```
8 failed, 158 passed in 63.55s (0:01:03)
```

**exactly the 8 the round-2 report names**, and all eight are C1 claims about
the co-moving stop plane
(`TestC1FocusReadoutContainment` x2, `TestVerifyC1Quadratic` x1,
`TestVerifyC1AgainstTheAnalyticFocus` x5).  The reason given for keeping the
default is measured and correct.

### 2.4 The documentation

Present, with the measured reason, in all three places:
`lumenairy/propagators/carrier.py:4541-4595` (the `transport` parameter's own
docstring, carrying the oracle, the 4.7340e-05 / 2.4049 pair, the 7-of-30
sweep and both reasons the default stays put), `CHANGELOG.md:245-...` (the
"Added — carrier (WP-C3)" entry) and `Migration-Guide.md:1878-1918` ("The
focus readout's standoff leg, and why it takes a keyword", with the two-line
recipe).  **CONFIRMED.**

### 2.5 Defect V2 — documentation. The in-code comment says the opposite of the shipped default

`lumenairy/propagators/carrier.py:4735-4739`, the comment block immediately
above the standoff leg, ends:

```
    # ``transport`` keyword, so moving it
    # would have moved a public answer with no one-keyword way back (5 of this
    # package's 103 archive keys, all of them this readout's).  Round 2 gives
    # it the keyword instead of the pin, so the campaign's rule is satisfied
    # and the better quadrature is the default here as it is everywhere else.
```

The last clause is false: the default at `carrier.py:4382` is `'sziklas'`, and
the docstring 150 lines above it spends two paragraphs explaining why.  A
reader of the code is told the opposite of what the code does — the same shape
as D1 in the prior verification.

**Requested edit**, `lumenairy/propagators/carrier.py:4739`.

OLD:

```
    # it the keyword instead of the pin, so the campaign's rule is satisfied
    # and the better quadrature is the default here as it is everywhere else.
```

NEW:

```
    # it the keyword instead of the pin, so the campaign's rule is satisfied
    # and the better quadrature is ONE KEYWORD away.  The DEFAULT stays
    # ``'sziklas'`` -- see this function's ``transport`` docstring for the two
    # reasons, the second of which is measured: with the default flipped, 8 ids
    # across ``test_audit2609_a6_carrier.py`` and
    # ``test_audit2609_a6_verify_carrier.py`` go red, and every one is a C1
    # claim about the co-moving stop plane this apparatus is derived about
    # (re-measured 2026-09-20, VERIFY-WP-C3 ROUND 2: 8 failed, 158 passed).
```

---

## 3. D13 — the C3 x C5 merged tree, rebuilt against C5's real final tip

The round-2 report records the merge against `feat/c5-three-defaults-round2`
@ `f1402bee`.  **That is not C5's final tip**; `575020c4` is, four commits
later.  Rebuilt here against `575020c4`: `git archive feat/c3-collins-round2`
into `C:/tmp/vc3b_merge`, then a per-path 3-way `git merge-file` over base
`49ddf4bd` for every path C5 touches.  (`f1402bee..575020c4` changes no
`lumenairy/` file, so the merged library is the same one the report measured;
what the four commits move is `.test_durations`, two C5-only test files and
prose.)

**`lumenairy/propagators/carrier.py` merges with ZERO conflicts**, and the
merged file reads `_GAP_KERNEL_ACCURACY_TAU = 1e-4` (line 1781) beside
`transport: str = 'collins'` at 1156 / 9899 / 12711 and `'sziklas'` at 4435
(the focus readout's new keyword).  Outside it, exactly the conflicts the
report predicts — and **`test_audit2609_b4_collins_transport.py` merges
cleanly**, so commit `7a2cf19f` holds against the real tip:

| file | conflict | resolution here |
|---|---|---|
| `CHANGELOG.md` | yes | OURS (C3); prose only |
| `Migration-Guide.md` | yes | OURS (C3); prose only |
| `docs/history/carrier.md` | yes | OURS (C3); prose only |
| `tests/unit/test_fix_v1_v8_readout_guard_and_standoff.py` | yes | **union** — C5's `_chain(..., replica_fill=None)` signature with C3's docstring above the body |
| `tests/unit/test_audit2609_b4_collins_transport.py` | **none** | — |
| `.test_durations` | (JSON) | 3-way merged key-by-key: base 16596, C3 16642, C5 16644 → **16690** |

No conflict marker survives anywhere in the merged tree, and every
`tests/unit/*.py` plus `carrier.py` parses.

### 3.1 The tails (WIN-py3.14, merged tree)

| run | tail |
|---|---|
| `test_wave5_h2_near_focus_table.py` + `test_c3_collins_default.py` + `test_verify_c3_collins_default.py` + `test_fix_v1_v8_readout_guard_and_standoff.py` + the nine VERIFY-C5-listed ids | **`106 passed, 1 warning in 232.36s (0:03:52)`** |
| `test_audit2609_b4_collins_transport.py`, whole | **`130 passed in 287.37s (0:04:47)`** |
| `test_wave5_h2_near_focus_table.py` alone, merged | **`18 passed in 8.96s`** |
| the same file alone on the branch (tau `None`) | **`17 passed in 9.55s`** |
| `test_fix_v1_v8_readout_guard_and_standoff.py` alone, merged | **`37 passed, 1 warning in 16.49s`** |

The nine ids are the five `test_niche_d3_guards.py`, the three
`test_niche_gap_frame_observable.py` and
`test_niche_d2_chain_multi.py::test_memory_budget_is_honoured`.  The one
warning is `angular_spectrum_propagate_mft`'s faithful-zone notice inside
`TestV3ReplicaGuardSeesCentreOut::test_ignore_and_warn_still_escape`, which
that id provokes on purpose.

### 3.2 The five d3 ids — re-derived, or re-recorded?

**RE-RECORDED — and re-derivation was not required, which is a measurement
rather than an opinion.**

`validation/probe_verify_c3_round2/vr2_count_collins.py` wraps
`_collins_transport` and `_collins_exact_kernel_departure` and counts per test
id.  Self-test on `test_wave5_h2_near_focus_table.py` (merged tree): **84
Collins legs, 57 tau evaluations, 10 tau firings** — the instrument works.
The same instrument on the five d3 ids, merged tree, `tau = 1e-4`:

```
5 passed in 174.28s   |   Collins legs: 0   tau evaluated: 0   tau fired: 0
```

**Not one of the five reaches the Collins transport**, so WP-C5's rule — which
lives inside `_collins_transport` — is never consulted.  Four of the five name
`transport='sziklas'` in their chain helpers (`_linearity_error`,
`_mux_chain_field`); the fifth,
`test_the_verdict_is_identical_through_a_slow_and_a_fast_chain`, takes the
default and still reaches zero, because every leg it runs falls back to the
transfer-function form.

And the three quantities the bars are placed on are bit-identical on the two
trees (`vr2_d3_bars.py`, WIN-py3.14):

| | branch (`tau = None`) | merged (`tau = 1e-4`) |
|---|---|---|
| `bad6` | 1.6462265362767432 | **1.6462265362767432** |
| `good6` | 0.00831245978054869 | **0.00831245978054869** |
| `bad4` | 115.24895826678058 | **115.24895826678058** |
| separation `bad6/good6` vs the **5.0** bar | **198.04x** | **198.04x** |
| `bad4/bad6` vs the **2.0** bar | **70.01x** | **70.01x** |

So the merged run re-RECORDS a green; nothing was re-derived, and nothing
needed to be.  Two consequences worth putting on the record:

* VERIFY-WP-C5's "0 → 25 fallbacks" table for these ids was measured by
  FORCING `'collins'` on the PRE-round-2 semantics.  With round 2's
  flat-reference fallback in place the legs never reach the transport at all,
  so **the "4x blast radius" premise does not describe the merged tree** for
  these ids;
* the separation bars themselves (5.0x against a measured 198x, 2.0x against
  70x) have two decades of headroom, so even if a future change did route
  these legs onto the chirp-Z the bars would not be the thing that broke
  first.

---

## 4. THE OPEN ITEM — `test_niche_p8_capstone.py::test_stepB_composed_doublet_relay_matches_debye`

### 4.1 The answer

**The Collins FIELD is right.  The 11.2 % is the metric quantising on the
LATTICE the leg returns.  This is NOT a P1 against the default.**

`validation/probe_verify_c3_round2/vr2_p8_settle.py`, WIN-py3.14 and
WSL-py3.12 agreeing to every printed digit.

**(a) The two arms differ only in the final leg.**  The chain prefix — traced
doublet, gap leg, universal-gated relay, carrier fit and envelope — is
**bit-identical** between `transport='sziklas'` and `transport='collins'`
(`max|env3_collins - env3_sziklas| = 0.0`; gap pitch
1.4103792870687537 um both).  The final leg lands at
`A = 0.006663133864514448`, reproducing the published 0.006663.

**(b) As shipped, read on each arm's own returned lattice** (the test's own
`_ee_metrics`, 200 um window), against the test's own ring-Huygens Debye
oracle (`EE50 = 7.3539 um`, `EE80 = 11.1102 um`):

| final leg | returned pitch | samples inside EE80 | EE50 | EE80 | vs the oracle |
|---|---|---|---|---|---|
| `'sziklas'` (co-moving + bridge) | 0.1052 um | **104.7** | 7.2890 | **11.0139** | **−0.87 %** |
| `'collins'` (chirp-Z, flat ref, pitch floored) | 6.3126 um | **1.96** | 6.2930 | **12.3588** | **+11.24 %** |

reproducing the report's numbers exactly.  The shipped Collins leg emits, by
itself, `propagate_carrier_referenced: the Collins chirp-Z stage is
under-sampled -- K2 (output) 5.1055: the requested output pitch does not
resolve ...` — **the library already says what is wrong, and it is the pitch.**

**(c) The decisive measurement — the same transport on a lattice that can
resolve the spot.**  Driving the identical leg with an explicit `dx_out`
(no interpolation of anything, no oracle of mine, just the shipped transport):

| `dx_out` | 0.25 um | 0.5 um | 1 um | 2 um | 3 um | 6.3126 um (its own floor) |
|---|---|---|---|---|---|---|
| EE80 (um) | **11.0090** | **10.9921** | **11.0708** | **10.9417** | **10.9083** | **12.3588** |
| K2 warning | none | none | none | 1.x | 2.x | 5.1055 |

Five pitches spanning 12x all read 10.91–11.07 um against the oracle's 11.1102
and the co-moving arm's 11.0139.  Only the floored pitch reads 12.36.

**(d) Corroboration — one common lattice, exact band-limited resampling.**
Both final-leg envelopes resampled onto ONE 0.5 um lattice by a full-N,
separable non-uniform inverse DFT (every DFT coefficient, no cropping, no
apodisation), metric and window identical for both arms (168.1 um, set by the
co-moving array's own 431 um extent):

| arm | EE50 | EE80 |
|---|---|---|
| `'sziklas'` | 7.2691 | **11.0575** |
| `'collins'` | 7.2686 | **11.0568** |

**0.006 % apart on EE80**, and the two intensity maps agree to **1.29e-04** in
relative L2 (1.26e-04 scale-free).  Peaks 14495.7 vs 14498.1 (1.7e-04);
windowed powers 3.53299e-06 vs 3.53308e-06 (2.5e-05).

So: the disagreement is 11.2 % when each arm is read on its own lattice and
0.006 % when both are read on one.  The field is the same field.

### 4.2 What follows for the shipped code and the shipped test

* **Nothing in `lumenairy/` is wrong here** — no P1, no P2.  The transport
  returns a correct field and warns that the lattice it had to choose does not
  resolve it.  The remedy is the caller's `dx_out`, and it works.
* **The report's and the test's "cannot say whether the field is wrong or only
  its returned sampling is" is now answered** and should be replaced rather
  than left standing (defect **V3** below).
* **The id's second arm stays green either way** — it asserts
  `gap80 > 0.05`, and the gap it measures (the two arms on their own lattices)
  is still 11.2 %.  But its docstring now describes a settled question as an
  open one, and its failure message tells a future reader to "retire this arm
  with the measurement" when in fact the measurement exists.
* **The floor is not obviously wrong either.**  `2 r_out / N` is what holds the
  ABCD image of the input box in `N` samples; on this leg that box is 25.9 mm
  wide and the spot is 11 um, so no single `N = 4096` lattice holds both.  The
  real follow-up is the one the report already owes: let a near-focus readout
  ask for the pitch it needs.

### 4.3 Defect V3 — documentation. The open question is closed; say so

**Requested edit 1**, `tests/unit/test_niche_p8_capstone.py`, the
`_run_composed_chain` docstring, replacing the paragraph beginning "The
Collins reading is not obviously the FIELD being wrong":

NEW:

```
    SETTLED 2026-09-20 (VERIFY-WP-C3 ROUND 2): it is the LATTICE, not the
    field.  Resampled onto one common 0.5 um lattice by exact full-N
    band-limited interpolation the two arms read EE80 11.0575 um and
    11.0568 um -- 0.006 % apart -- and their intensity maps agree to
    1.29e-04 in relative L2; and the SAME Collins leg driven with an
    explicit ``dx_out`` reads EE80 11.0090 / 10.9921 / 11.0708 / 10.9417 /
    10.9083 um at 0.25 / 0.5 / 1 / 2 / 3 um against the oracle's 11.1102 um,
    reading 12.3588 um only on its own floored 6.3126 um pitch.  The leg
    says so itself: it emits ``the Collins chirp-Z stage is under-sampled --
    K2 (output) 5.1055``.  This id keeps ``transport='sziklas'`` because
    that is the lattice its 6 % metric was calibrated on, and the second arm
    keeps measuring the 11.2 % because that gap is the READOUT PITCH and
    will close when a near-focus readout can ask for the pitch it needs --
    not because the two transports disagree about the field.
```

and the same in that id's `gap80` assertion message: "That is the READOUT
PITCH, not a physics disagreement (VERIFY-WP-C3 ROUND 2 §4)" in place of
"That is the OPEN near-focus question in the WP-C3 report closing".

**Requested edit 2**, `docs/audits/.../WP-C3_COLLINS_DEFAULT_REPORT.md` §R2.3b
("What this round can and cannot say") and the matching §R2.4 bullet
("THE NEAR-FOCUS ACCURACY QUESTION OF R2.3b"): replace both with the
measurement above and move the open item from "is the Collins field wrong on a
real aberrated composition" to "let a near-focus readout name its own output
pitch, so the floor `2 r_out / N` is not the only lattice on offer".  The
Migration guide's near-focus caveat should likewise say that the way to a
resolved near-focus spot is `dx_out`, not `transport='sziklas'`.

### 4.4 The decision test

`tests/unit/test_verify_c3_round2.py::test_the_near_focus_ee80_gap_is_the_returned_lattice_not_the_field`
pins the settlement on a 256-grid stand-in for the same leg (flat-resolving,
representable chirp-Z, pitch set by the floor), in under a second.  Measured
on both builds
(`validation/probe_verify_c3_round2/vr2_p8_decision_{win,wsl}.json`):

| quantity | bar | WIN | WSL |
|---|---|---|---|
| samples inside EE80, Collins lattice | `< 3` | 1.4148 | 1.4148 |
| samples inside EE80, co-moving lattice | `> 30` | 102.47 | 102.00 |
| EE80 on its own lattice against EE80 on the fine one, Collins vs itself (relative) | `> 0.08` | **0.17055** | **0.17055** |
| EE80 of Collins-on-the-fine-lattice against the co-moving arm (relative) | `< 0.03` | 0.00099 | 0.00361 |

The third row compares one transport with itself and is bit-identical on the
two builds; the whole cross-build spread of the fixture lives in the fourth
row's co-moving reference (46.1129 um WIN against 45.9016 um WSL, 0.46 %),
which is why that bar sits 8x above the worse reading rather than tighter.

---

## 5. THE BLAST SET

`validation/probe_c3_collins_default/blast_files.txt` (54 files) plus
`test_c3_collins_default.py` and `test_verify_c3_collins_default.py`, on the
round-2 tip:

| build | scope | tail |
|---|---|---|
| WIN-py3.14 | all 54 + the two C3 files, `test_audit2609_b4_collins_transport.py` included whole | **`1888 passed, 4 skipped, 273 warnings in 3440.39s (0:57:20)`** |
| WSL-py3.12 | the same minus b4 (its `TestGateCTwoGroupChain` stalls on WSL, pre-existing) | **`1757 passed, 5 skipped, 273 warnings in 3874.08s (1:04:34)`** |

**No red on either build, so no pre-existing-red comparison against `git archive
49ddf4bd` was needed.**  The 273 warnings are the same count on both builds and
the 4 / 5 skips are the platform ones the suite always carries.

---

## 6. WHAT THIS RE-VERIFICATION COULD NOT MEASURE

* **The device (CuPy) arm.**  `cupy.fft` on this box still raises
  `ImportError: DLL load failed while importing cufft`; not attempted, and it
  was not in scope.
* **The published "7 of 30" standoff sweep.**  The five geometries and six
  standoffs are not recorded in the tree, so my 30-cell sweep is my own and
  confirms the direction (13/30 vs 0/30) without confirming the count.
* **The MERGED tree's full 54-file blast set.**  The brief asked for the blast
  set on the branch; the merge was measured on the files the two branches touch
  and on the nine C5-listed ids, as round 2 did.  A full merged run remains the
  pre-tag gate.
* **`test_audit2609_b4_collins_transport.py` on WSL.**  Excluded by the brief
  (its per-class run is the documented staller); it is green whole on Windows
  on both the branch and the merge.
* **A non-paraxial truth for §4.**  The p8 oracle is the test's own ring-Huygens
  Debye integral and both transports are paraxial, so §4 establishes that the
  two agree and which one the oracle backs — not the true physical field below
  ~1e-3.
* **Everything on the branch outside the five items**, including the D1/D2/D3/
  D4/D7/D9/D10/D11/D12 closures, the CuPy censuses, the mutation matrices and
  the design-121 rows.  The scope was narrowed on purpose.

---

## 7. GATES

Every row run with the three thread caps on the command line and
`--capture=sys`.  WIN = py3.14.6 on this machine, WSL = py3.12.3 in
`~/lumvenv`.

| gate | build | result |
|---|---|---|
| `test_c3_collins_default.py` + `test_verify_c3_collins_default.py` + `test_verify_c3_round2.py` + `test_niche_p8_capstone.py` + `test_public_api.py` + `test_audit_except_budget.py` + `test_audit2609_a15a_durations_staleness.py` + `test_audit2609_a21_doc_identifiers.py` | WIN | **`71 passed, 5 warnings in 188.13s (0:03:08)`** |
| `test_v4_16_2_dispatcher_pin_doc_consistency.py` | WIN | **`8 passed in 0.29s`** |
| `test_verify_c3_round2.py` alone | WIN | **`1 passed in 1.29s`** |
| the same | WSL | **`1 passed in 0.99s`** |
| the 54-file blast set (b4 whole) | WIN | **`1888 passed, 4 skipped, 273 warnings in 3440.39s (0:57:20)`** |
| the same minus b4 | WSL | **`1757 passed, 5 skipped, 273 warnings in 3874.08s (1:04:34)`** |
| the C3 x C5 merged tree, twelve-id set | WIN | **`106 passed, 1 warning in 232.36s (0:03:52)`** |
| the merged tree, `test_audit2609_b4_collins_transport.py` whole | WIN | **`130 passed in 287.37s (0:04:47)`** |
| `ruff check lumenairy/ tests/ scripts/`, then the whole repo | WSL | **All checks passed!** twice |
| `python -m mypy` (no arguments) | WIN | **Success: no issues found in 33 source files** |
| `scripts/record_history_fingerprints.py --check` | WIN | **OK: every history document matches its module** |
| `.test_durations` | -- | 16642 -> **16643** keys, valid JSON (the one new id spliced with the branch's own `r2_splice_durations.py`) |

Nothing in `lumenairy/` was edited, so no history document needed
re-recording and the citation gate's block is untouched.

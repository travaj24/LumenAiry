# Response to AUDIT_CI_TEST_TIME_2026_08_03 -- assessment + 5.32.1 implementation

**Scope.** Every recommendation in `docs/audits/AUDIT_CI_TEST_TIME_2026_08_03.md` is assessed here
against its own evidence class (measured vs asserted), against what this session already shipped
(regenerated `.test_durations`, 5-way `publish.yml` verify sharding, the 91-flag order-independence
conftest guard), and against this session's standing rules (no absolute numeric bars, no
build-singular evidence, no silent behavior changes, fail-before switches where behavior moves).
Accepted items are implemented; everything else is deferred or rejected with a reason.

**Everything below is uncommitted.**  No `CHANGELOG.md` and no `pmm/**` file was touched.
`tests/unit/test_niche_d6_exact_tilted_leg.py` and `tests/unit/test_niche_d7_decentred_fit.py` are
owned by another agent this session and were not read into any change; no audit item required them.

---

## 0. Verification environment (read before trusting any number I add)

Everything was measured on the same 24-core Windows desktop the audit used, **while another agent
was running its own pytest processes** -- so my absolute seconds are contended upper bounds, not
idle figures.  Where the audit supplies an idle `[M]` number I checked one of them end-to-end
before reusing the rest:

| File | Audit `[M]`, idle + pinned | Mine, pinned + contended | Agreement |
|---|---|---|---|
| `test_v5_6_rcwa_convergence.py` | 317.2 s | **324.9 s** | 2.4% |

That is close enough that I treat the audit's other `[M]` per-file/per-test seconds as usable, and
its `[E]` estimates as unverified.  Two audit figures I re-measured did **not** hold up:
`test_fga.py` is **189.4 s** pinned here, not the ~7 min its own comment claims (that figure is
`[E]`, from a stale in-code note); and the whole-tree collection cost (§2 below).

I also reproduced the audit's §7 platform hazard directly: stopping the backgrounded pytest task
left the underlying `python -m pytest` alive and running to completion (PID observed holding the
box for another ~5 min).  Worth keeping in the runbook.

---

## 1. Does anything conflict with what this session already shipped?

Four possible conflicts were checked; **one is real and it is a criticism of this session's own
artifact, not a stale finding.**

| Session artifact | Audit's position | Conflict? |
|---|---|---|
| `.test_durations` regenerated this session (11,223 entries, 316,585 units) | §2 calls it "not usable as a time reference" | **YES -- and correct.** The audit's numbers are *this* file (identical entry count and unit total), i.e. it audited the regenerated artifact, not a stale one. See §2 below; I confirmed the defect independently and acted on the part I can act on safely. |
| `publish.yml` verify moved to 5-way sharding after two 30-min-cap timeouts | Audit never examines `publish.yml` (its §0 counts only `unit-tests.yml` + `validate.yml`) | No conflict. The two timeouts are *evidence for* §2: shards balanced "to <0.1%" on inconsistent weights still blew the cap. I fixed the stale `/3` labels the reshard left behind. |
| 91-flag order-independence conftest guard (module-scoped) | Audit does not mention it | No conflict. Nothing I accepted touches flag-restore semantics. The one conftest change is a comment block. |
| Envelope-assertion discipline / no absolute numeric bars | §4 proposes ~8 "shrink this parameter" items | No direct conflict, but it is why those are deferred rather than batched -- each needs its own measured margin, which is also the audit's own §6.4 advice. |

---

## 2. §2 -- `.test_durations` and sharding

### 2.1 The "inconsistent scale" defect: CONFIRMED, and worse than a balance nuisance

The audit says the file's implied "units per real second" varies ~26-74x file to file.  I measured
the spread independently and it is at least that wide:

| File | `.test_durations` units | Real seconds (measured) | Implied units/s |
|---|---|---|---|
| `test_v5_6_rcwa_convergence.py` | 23,074 | 324.9 (pinned, mine) | **71** |
| `test_fga.py` | 1,792 | 189.4 (pinned, mine) | **9.5** |
| `test_v5_3_2_stamp_changelog.py` | 268 | ~120 (the 2-subprocess form the entry was recorded from) | **2.2** |
| `test_v5_6_1_rcwa_symmetry.py` | 4,252 | 8.6 (audit, idle+pinned) | ~490 |

So the balancer's weights are internally inconsistent by more than two orders of magnitude, and the
inconsistency is **structured, not random**: it tracks how BLAS-thread-hungry a test is.  The
fast-gate half of the file was captured with multi-threaded BLAS unpinned on a many-core box (and
under contention), which inflates eig-heavy tests enormously and subprocess-bound tests barely at
all.  A 2-4 vCPU GitHub runner cannot reproduce that inflation, so the committed weights
systematically **over**-weight eig-heavy tests and **under**-weight subprocess/IO-bound ones
relative to the machine that has to run them.

This is the mechanism behind the two publish-verify timeouts: `--splits` divided the *units*
evenly (the workflow comment's "<0.1%" claim is true and irrelevant) while dividing the *seconds*
unevenly.

**Verdict: ACCEPT.**  Regeneration is the right fix and it is the top follow-up item.

**Implemented now**
* The *protocol* is written into both workflows next to the `--durations-path` flag
  (`unit-tests.yml` fast-gate step, `publish.yml` verify step): regenerate **serially, on an idle
  machine, with `OMP`/`OPENBLAS`/`MKL_NUM_THREADS=1`, for both gates**, so every entry is on one
  scale.  Without that line the next regeneration reproduces the same artifact.
* Stale entries for the 19 items I moved between gates were **deleted** rather than carried over
  (§4.1).  Rationale: a fast-capture weight is not comparable to the slow gate's entries, and
  pytest-split falls back to the *relevant gate's own average* for a missing nodeid
  (`algorithms._get_avg_duration_per_test`) -- an average-weight guess is far better than a value
  that is wrong by 10-70x in a gate whose total is small.  I did **not** invent replacement numbers.

**Deferred (with reason)**
* The regeneration itself.  It needs an idle box for the full serial suite (10,834 fast + 227 slow
  items) and it must happen **after** gate membership settles, which is exactly the audit's §6.5
  ordering.  Doing it today, on a contended box, would bake in a third inconsistent scale.

### 2.2 The "collection-time blind spot": REJECTED as stated, real observation underneath

The audit claims 6-11 files `rglob` + `read_text` + `ast.parse` the `lumenairy/` tree **at
module-import (collection) time**, costing "+47.8 s of real collection time" invisible to the
balancer, and proposes a shared `tests/unit/_source_index.py`.

Checked directly:

* An AST scan of all 448 unit files for tree-walking work in **module-level** statements (including
  one-level-deep helper calls) finds **2 files**, not 6-11:
  `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` and
  `test_v4_16_0_walker_sentinel_propagation.py`.  Both materialise eagerly **on purpose**, so the
  discovered files appear as parametrised test IDs at collection -- moving that work into a test
  would delete the per-file IDs, which is a real coverage-legibility loss.
* **Whole-tree collection of `tests/unit` costs 19.7-21.4 s** (pytest-reported; ~51 s wall
  including interpreter start and every import).  The claimed "+47.8 s from 9 files" is larger than
  the entire measured collection phase, so it cannot be right as stated.
* The genuine observation underneath: **7 files** both walk the tree and `ast.parse` it (4 with
  their own private `lru_cache`).  That duplication is real, but it happens at **call** time in 5
  of the 7, where pytest-split *does* see it.

**Verdict: REJECT the framing and the 47.8 s number; the shared-index idea is DEFERRED**, to be
justified by measuring the 7 files' actual parse time first.  It touches up to 7 walker files for a
payoff that is currently bounded above by ~20 s of total collection, most of which is plain module
import.

---

## 3. §1 -- BLAS pinning

### 3.1 "The 265-file per-file guard is dead code": ACCEPT (mechanism confirmed)

282 files (my count) carry `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")`-style blocks that
cannot fire under pytest, because `tests/conftest.py` imports numpy and lumenairy at module scope
and pytest loads conftest before any test module.  The audit reproduced this with
`threadpool_info()`; the mechanism is not in doubt.

### 3.2 "Do NOT blanket-pin the fast gate": ACCEPT -- and the audit is right to refuse its own headline

§1.2 is the most disciplined part of the audit: three chunks measured large speedups from pinning,
chunk 6 measured **2.76x slower** on `test_v5_6_rcwa_convergence.py` and a **net +27% worse** across
its own 75-file range, and chunk 1's 2- and 4-thread control run showed the 33x desktop figure is a
24-core artifact.  That is exactly the "no build-singular evidence" standard.  A blanket pin is a
trade, not a win.

**Implemented now:** a comment block at the top of `tests/conftest.py` recording (a) that the
per-file guards are inert under pytest, (b) which lanes actually pin and where (`slow-tests` and
`jax-unit` `env:` blocks), (c) that the fast `unit` matrix deliberately does not, with the
two-sided measurement, and (d) "a no-op guard is not a pin -- check the workflow."  A matching note
is in the `unit-tests.yml` fast-gate step.  This is documentation, not behavior: it exists because
282 inert guards currently read as evidence that the fast lane is pinned, and someone will act on
that.

**Deferred:** deleting the 282 inert blocks (a 282-file no-op diff, better done in its own pass
than mixed into a behavior change), and the CI-hardware A/B for a 2-4 thread middle ground.  The
A/B is the audit's §6.6 and belongs last, on real runners, exactly as it says.

### 3.3 A pin/marker interaction the audit does not flag

The slow gate **pins BLAS**; the fast gate does not.  So "move a heavy test from fast to slow" is
not cost-neutral for a *pin-hostile* file: `test_v5_6_rcwa_convergence.py` costs 112 s unpinned and
309-325 s pinned, so moving it to the slow gate would land ~2.8x its fast-gate cost on a single
slow shard (less than 2.8x on a 2-vCPU runner, which has fewer threads to lose, but the sign is the
same).  This is why the marker move for that specific file is deferred below rather than taken --
it is the audit's own measurement, applied to the audit's own recommendation.

---

## 4. §3 + §4 -- itemized recommendations

### 4.1 IMPLEMENTED

| # | Audit item | Evidence | What I did | Verified |
|---|---|---|---|---|
| I1 | §3 -- `test_audit_g06_perf.py::test_s5_8f_pyfftw_interfaces_cache_daemon_disabled` cannot pass under `pytest-timeout` (its own node ID contains `pyfftw`, and the watchdog thread is named after the node ID) | measured, reproduced by the audit | Probe now excludes threads whose name contains `pytest_timeout`/`pytest-timeout`; the real pyFFTW daemon never carries a node ID, so the filter cannot mask it.  Docstring records the mechanism. | **fail-before proven**: the old assertion form FAILS under `--timeout=200 --timeout-method=thread` (`['mainthread', 'pytest_timeout ::test_s5_8f_pyfftw_...']`), the new form passes; the real test passes with `--timeout` active |
| I2 | §4/chunk 6 -- jax tests guarded by `skipif(not _jax_ok())` are selected by no CI leg and skipped by the fast matrix, i.e. they **run nowhere** | measured (guard-idiom scan) | Widened the `jax-unit` selection grep to both guard idioms (`grep -rlEi "importorskip.{0,4}jax\|skipif\(.{0,40}jax"`).  Selection stays by GUARD, never by name.  77 -> 86 files. | The 9 newly-selected files are exactly the hole: `maslov_jax_caustic`, `gbd_feature_complete`, `analytic_ray_transfer`, `lens_accuracy_extensions`, `gbd_maslov_perf`, `delta_audit`, `audit_w6_propagators`, `wave2_p1_parity`, plus one the audit missed -- `test_v5_2_glass_formula3.py` (`_JAX_AVAILABLE`) |
| I3 | (pin for I2) | -- | `test_v5_24_3_jax_ci_coverage.py`: guard regex mirrors the workflow, 4 new blind-spot files, and a **new** test that lifts the literal grep pattern + flags out of the YAML and re-runs it against the tree, so narrowing the pattern reds the pin with the file it stopped selecting | passes |

**I2 cost, and a hole that does *not* exist.**  The jax leg is unsharded with a 45-min step cap, so
widening its selection has to be sized.  Measured on the same capture (so the comparison is
internally consistent even though the absolute scale is not trustworthy -- §2.1): the jax job's
selection goes 77 -> 86 files and 2,727 -> 2,854 executed items, **+3.4%** of its own weight, after
the I7 marker moves take three of the newly-added files' heavy NumPy tests off it.  Separately, I
checked for the dual of this hole -- a test marked `slow` *and* jax-guarded would skip on the
no-jax slow gate and be excluded from the jax gate, running nowhere.  There are **0** such tests
today, and none of the 14 marks in I7 creates one (all 14 are NumPy tests).
| I4 | §4/chunk 6 R2 -- `test_v5_3_2_stamp_changelog.py::test_dry_run_against_current_changelog` runs the same ~20 s subprocess twice and snapshots "before" *after* the first one | measured | Snapshot moved before the (now single) subprocess.  Strictly **better** coverage -- it now checks the invocation that could actually have modified the file -- at half the cost. | passes; file's dominant test 59.3 s (was 2x that) |
| I5 | §4/chunk 4 -- `test_v4_15_2_agent_a.py::test_changelog_test_count_arithmetic_reconciles` spawns **10** `pytest --collect-only` subprocesses (60.4 s of the file's 61.2 s) | measured | One subprocess over all 10 files; per-file counts from the `-q` node-id lines.  Two guards keep the cheaper form honest: it asserts nothing was deselected (the old form read the pre-deselection summary) and that per-file counts reconstruct pytest's own total. | passes; that test now **5.5 s** (contended), 18/18 green |
| I6 | §4/chunk 4 -- `test_v4_15_4_agent_c.py::_pytest_collect_only_count` is dead since the v4.16.1 rewrite | measured (no caller) | Removed, with a note on why not to revive a whole-tree collection subprocess inside a unit test.  Unused `subprocess`/`sys` imports removed. | passes, ruff clean |
| I7 | §4/chunks 3+6 -- marker inversions: heavy tests unmarked while cheap siblings carry `slow` | measured `[M]` per test | 14 nodes marked `slow` across 8 files (list below), each with an in-file note giving the measurement and why the claim is not Python-version-sensitive | collection verified: all 14 now select under `-m "slow and not integration"` and vanish from the fast selection |

**I7 detail** -- marked `slow`, with the audit's idle `[M]` cost:

| File | Node(s) | `[M]` |
|---|---|---|
| `test_v5_21_maslov_jax_caustic.py` | `..._integration_method_auto_matches_and_is_fast`, `..._fold_split_matches_manual_chain`, `..._fold_split_noop_on_unfolded` | 174.6 s of the file's 190 s (its only marked-worthy tests were unmarked; the jax tests it is named for cost 6.9+3.5 s) |
| `test_niche_audit_w6_bor.py` | `test_w6_b1_default_path_is_bit_identical_to_explicit_rbig` | 130.9 s of 138.6 s -- the only heavy test in the file *without* a mark, next to five lighter ones that have it |
| `test_niche_audit_w6_eme.py` | `test_w6_1_layer_modes_recall_vs_fd_oracle` (6 cells), `test_clean_structured_diffraction_still_warns_and_does_not_converge` | file measured 197 s pinned with **zero** `slow` marks |
| `test_v5_21_pmm2d_staggered_oblique.py` | `test_staggered_per_order_orientation_vs_1d` (2 cells) | 53.8 s of 56.1 s; the file's existing mark is on a test that was already excluded |
| `test_v5_21_gbd_maslov_perf.py` | `test_fft_reconstruct_matches_dense_uniform_Q`, `..._anamorphic_diagonal_Q` | 49.8 s of 64.7 s (N=128 dense O(N^4) reference) |
| `test_v5_21_gbd_windowed_adaptive.py` | `test_soft_edge_improves_hard_aperture_focus` | 27.6 s of 40.7 s |
| `test_v5_21_lens_accuracy_extensions.py` | `test_traced_multibranch_matches_exact_diffraction_oracle` | 25.1 s (2.56M-ray launch grid) |
| `test_v5_2_3_subaperture_image_plane.py` | `TestV5_2_3_MagnificationSpatialExtent::test_2x_telephoto_image_extent_larger_than_source`, `TestV5_2_3_ClosedFormStrehl::test_2x_singlet_waist_scales_with_M` | 16.6 s of 26.5 s; both classes' own docstrings call their bars deliberately weak |

**Coverage cost, stated plainly (not a silent change).**  A `slow` mark moves a test from 4 Pythons
to 1 (3.12).  Every node above asserts a NumPy numerics identity, a bit-identity, or an accuracy
envelope -- the same class the slow gate was created for ("the numerics are not version-sensitive",
`unit-tests.yml` v5.15.0).  None is an API/dispatch/typing contract, which is where version
sensitivity actually lives.  No test was deleted, disabled, or weakened.

**Gate-load effect, measured rather than asserted:**

* Fast gate: **-~630 s per Python** (sum of the `[M]` figures above), i.e. ~-42 min of runner time
  per push across the 4x3 matrix, ~-3.5 min off each fast shard's critical path.
* Slow gate: **+~630 s total**, ~+3.5 min per shard across its 3 shards.  In the balancer's own
  units the slow gate goes 5,373 -> 5,921 (**+10.2%**), simulated as a perfectly even 1,973.5 u
  per shard.  Against the documented ~15 min/shard and the 30-min step cap that is comfortable, but
  it is the change to look at first if a slow shard ever reports near-cap.
* Plus I4/I5: pure subprocess deletion, ~-74 s per Python (I4 one ~20 s `[M]` subprocess; I5
  60.4 s `[M]` -> 5.5 s measured), i.e. ~-4.9 min of runner time per push.  I6 saves nothing
  (the helper was dead) -- see §4.3.

**Files changed (16):** `.github/workflows/unit-tests.yml`, `.github/workflows/publish.yml`,
`tests/conftest.py`, `.test_durations`, and 13 files under `tests/unit/` --
`test_audit_g06_perf.py`, `test_v5_3_2_stamp_changelog.py`, `test_v4_15_2_agent_a.py`,
`test_v4_15_4_agent_c.py`, `test_v5_24_3_jax_ci_coverage.py`, plus the 8 marker files in the I7
table.  Plus this document.

### 4.2 DEFERRED (accepted diagnosis, not landed in 5.32.1)

| Audit item | Why deferred |
|---|---|
| §4/chunk 2 -- `test_fga.py` whole-file `slow` (its own comment's ~7 min, `[E]`) | Measured **189.4 s** pinned here, not ~7 min; still the single largest fast-gate file I looked at (-12.6 min runner time if moved).  Held because it would be the largest single addition to a gate whose *real* headroom cannot be established until `.test_durations` is regenerated on one scale.  Land it together with the regeneration.  (Its stale in-code justification -- "the fast gate is xdist-parallelised `--dist loadfile`" -- is indeed dead; the workflow uses job sharding.  It is the only file in the suite citing xdist.) |
| §4/chunk 6 -- `test_v5_6_rcwa_convergence.py` 3 tests `slow` | The audit's own §1.2 measures this file as the most pin-hostile in the suite (2.76x slower pinned).  The slow gate pins BLAS, so this move lands ~2.8x the cost on one slow shard -- see §3.3.  Right answer is probably to reduce its `M` sweep in place (chunk 4 makes the analogous argument for `test_rcwa.py`), which is a parameter shrink and needs its own margin check. |
| §4 -- all ~8 "needs 1 confirming run" parameter shrinks (chunk 5 R1 `M=17->13`, chunk 4 `test_rcwa.py` M-sweep, chunk 1 `N=96->48`, `elements_per_region 6->3`, `n_slabs 200->64`, `n_orders 120->30`, `n_orders_y 16->2`, ...) | Each changes what a physics assertion is computed from.  The audit itself says do them one at a time against each test's measured margin (§6.4).  That is a measurement cycle, not a CI-config cycle, and it is the one place where "no absolute numeric bars" bites hardest -- several of these bars were set by earlier campaigns for reasons not visible in the diff. |
| §4/chunks 2,3,4 -- byte-identical duplicate-solve fixture/`lru_cache` sharing (`test_s5_5_jones_field_bridge.py` 305 of 379 s, `test_eme_2d_vector.py`, `test_coupled_eigensolver.py`, `test_gate4.py`, `test_audit_p1_bor_flux.py`, ...) | Mechanically sound and the second-biggest lever after the marker moves.  Deferred because caching a solve across tests shares a mutable NumPy result between them -- safe only with a per-site check that no consumer mutates it, plus a before/after byte-identity diff.  That is a reviewable batch of its own; done carelessly it manufactures exactly the cross-test coupling the C11 guard exists to kill. |
| §4/chunk 2 -- O(N) encircled-energy loop rewrite (189x, 3 files) | Changes an *oracle's* numerics (agrees to 1.2e-7, not bitwise).  Needs each test's margin re-measured against the new oracle.  High value, own cycle. |
| §3 -- split `test_audit_misc.py`'s jax-guarded tests into their own module so the hard-coded `--deselect` can go | Real cleanup, but it is test-file surgery on a large shared file whose only current symptom is one workaround line that works.  No CI-time payoff. |
| §4/chunk 3 -- `test_niche_k3_perf.py::test_remap_2d_shared_delaunay_is_faster` (flaky wall-clock assertion) | Agree it should not gate PRs on a shared runner's timing.  But the proposed remedy does not work as stated: the `bench` marker is declared in `pyproject.toml` and carried by **zero** tests, and no gate selects against it -- the fast gate's `-m "not integration and not slow"` would still run it.  Moving it to `bench` is therefore a no-op for CI time; it needs `slow` too, or a `bench` exclusion added to the gates.  That is a marker-policy decision (what does `bench` mean, and which gate honours it) worth making once, deliberately, rather than as a side effect of one flaky test. |
| §4/chunk 2,3 -- the `[E]`-only marker moves (`test_g2_displaced_congruence`, `hammer_h1`, `hammer_h7`, `niche_e4`, `niche_p2`, `niche_d5`, `w8_shapes` parametrize) | Estimated, not measured.  Marker moves are cheap but they are still coverage moves; measure first, then land them in one batch with the `test_fga.py` move and the regeneration. |
| §4/chunk 4 -- delete 2 subsumed subprocess tests in `test_v4_15_4_agent_b.py` (197.6 s) | Deleting tests is the one class the audit otherwise avoids everywhere.  "Strict subset" needs to be shown, not asserted, and the review for that is not a CI-config review. |
| §0/§6 -- reshard the fast `unit` gate 3 -> 5 to match `publish.yml` | My own idea, deliberately **not** taken: the fast gate's 45-min step cap gives ~1.6x headroom at the documented 22-28 min/shard, whereas the publish verify hit its **30**-min cap -- the two are consistent, not contradictory.  Resharding adds 8 more runner-jobs per push for headroom that is not currently needed, and the marker moves plus the regeneration should be measured first. |

### 4.3 REJECTED / corrected

| Audit claim | Correction |
|---|---|
| §2.1 -- "6-11 files walk/parse the `lumenairy/` tree at collection time, +47.8 s of collection" | **2** files do so at module level (both deliberately, for parametrised test IDs), and total collection of `tests/unit` measures 19.7-21.4 s -- less than the claimed saving.  See §2.2. |
| §4/chunk 4 -- removing dead `_pytest_collect_only_count` saves "full cost of an unused helper" | It has no caller, so its runtime cost is exactly **zero**.  Removed as hygiene (it invites re-adoption of a whole-tree collection subprocess), not as a saving.  The audit's own confidence column says "high" for a saving that cannot exist. |
| §4/chunk 2 -- `test_fga.py` "~7 min, author's own figure" | Measured 189.4 s pinned.  The 7-min figure is `[E]`, inherited from an in-code comment that also cites a since-removed xdist configuration. |
| §5 -- "roughly 35-50 minutes of CI runner-time per push/PR" | Directionally plausible but it sums `[M]` and `[E]` items and multiplies some by the Python matrix and not others.  What I landed is **~47 min/push** of runner time built only from `[M]` figures (~42 min marker moves + ~4.9 min subprocess deletions), which happens to sit inside that range -- but the range itself should not be quoted as a measured total. |
| §0 -- "265 files carry the dead guard" | 282 by my count (`grep -rl OPENBLAS_NUM_THREADS tests/unit`).  Does not change the conclusion. |

---

## 5. Verification performed

* **Affected suites green.**  All 13 edited/affected test files run together, BLAS-pinned:
  `test_v5_21_maslov_jax_caustic`, `test_v5_21_gbd_maslov_perf`, `test_v5_21_pmm2d_staggered_oblique`,
  `test_v5_21_gbd_windowed_adaptive`, `test_v5_21_lens_accuracy_extensions`, `test_niche_audit_w6_bor`,
  `test_niche_audit_w6_eme`, `test_v5_2_3_subaperture_image_plane`, `test_audit_g06_perf`,
  `test_v4_15_4_agent_c`, `test_v4_15_2_agent_a`, `test_v5_24_3_jax_ci_coverage`,
  `test_v5_3_2_stamp_changelog`.  Result recorded in §5.1.
* **Fail-before for the one behavior move.**  I1's old assertion form demonstrably FAILS under an
  active `pytest-timeout` and the new form passes (both run side by side).  I4 and I5 change what a
  test measures, so both carry explicit in-test guards (I4: snapshot ordering documented; I5: the
  no-deselection and count-reconstruction assertions) rather than a silent swap.
* **ruff clean**: `ruff check lumenairy/ tests/unit/` -> `All checks passed!`, and
  `ruff check tests/conftest.py` likewise.
* **YAML + shard math**: all four workflows parse with `yaml.safe_load`.  `publish.yml` verify is
  5 shards / `--splits 5` / 30-min step / 35-min job -- labels and math now agree (they said "/3").
  `unit-tests.yml` is unchanged in shard count, caps, and selection; the only functional edit is the
  `jax-unit` selection grep, which was re-extracted from the YAML and re-run against the tree.
* **Split simulation** (pytest-split's own `least_duration` greedy, replayed offline against the
  real collected item lists):
  * fast gate: 10,834 items, 88 without a durations entry, 3 shards at 101,759 u each;
  * slow gate: 227 items, 21 without an entry (my 19 deletions + 2 pre-existing), 3 shards at
    1,973.5 u each -- i.e. the deletions do **not** unbalance the gate, they are absorbed at the
    gate's own average of 26.1 u.
* **Collection health**: `10834/11063 tests collected (229 deselected)` with no collection errors
  after all edits.

**Shard math at the measured suite size** (10,834 fast + 227 slow items; the per-version serial
figure is the one this session established when it resharded the release verify):

| Lane | Shards | Step cap | Per-shard load | Headroom | Change from this work |
|---|---|---|---|---|---|
| `unit-tests.yml` fast `unit` | 3 (unchanged) | 45 min | ~25-30 min (75-90 min/version) | ~1.5-1.8x | **-~3.5 min/shard** (I7) plus -~25 s/shard (I4/I5) |
| `unit-tests.yml` `slow-tests` | 3 (unchanged) | 30 min | last *documented* ~15 min (v5.23.0 comment -- stale, the campaign added slow tests since, and the durations scale cannot convert) | unknown, >=1x today since the gate is green | **+~3.5 min/shard**; balancer units 5,373 -> 5,921 (+10.2%), simulated even at 1,973.5 u/shard |
| `unit-tests.yml` `jax-unit` | 1, unsharded | 45 min | -- | -- | **+3.4%** of its own selection weight (77 -> 86 files, 2,727 -> 2,854 items) |
| `publish.yml` `verify` | 5 (unchanged) | 30 min | ~15-18 min | ~1.7-2x | labels/comment corrected `/3` -> `/5`; no functional change |

No cap was raised and no shard count was changed.  The only lane that gains work is the slow gate,
by ~3.5 min per shard.  Its true headroom is the one number this work could **not** establish --
the durations scale cannot convert its 5,921 units into runner minutes (§2.1) -- which is precisely
why the two largest candidate moves (`test_fga.py`, `test_v5_6_rcwa_convergence.py`) are held back
until the regeneration lands.  If a slow shard ever reports near its 30-min cap, this +10.2% is the
first change to look at, and reverting any subset of the 14 marks is a one-line-per-test undo.

### 5.1 What was actually run, and what was not

Every **behavior-changing** edit was verified directly and green:

| Run | Result |
|---|---|
| `test_v5_24_3_jax_ci_coverage.py` + `test_v5_3_2_stamp_changelog.py` + `test_v4_15_4_agent_c.py` (pinned) | **17 passed in 61.0 s**; stamp_changelog's dry-run test 59.3 s for its now-single subprocess |
| `test_v4_15_2_agent_a.py` | **18 passed in 5.8 s**; the consolidated count test 5.5 s (was 60.4 s `[M]`) |
| `test_audit_g06_perf.py -k "pyfftw or fftw" --timeout=200 --timeout-method=thread` | **2 passed** -- the fix holds with the watchdog active |
| fail-before pair (old vs new probe form, same run, `--timeout` active) | **1 failed, 1 passed** -- old form fails exactly as the audit describes |
| `test_v5_21_pmm2d_staggered_oblique.py` + `test_v5_2_3_subaperture_image_plane.py -m "slow and not integration"` | **5 passed** -- the newly-marked nodes execute green *under the slow-gate selection*, which is the selection that now owns them |
| `ruff check lumenairy/ tests/unit/ tests/conftest.py` | All checks passed |
| Whole-tree collection after all edits | `10834/11063 tests collected (229 deselected)`, no errors |

**Not completed here:** a single combined run of all 13 touched files.  It was launched
(BLAS-pinned) but the desktop went to five concurrent pytest processes from the other agent's C13
work, and contention inflated it past a useful window -- the two-file slow-selection spot check
above took 484 s for what the audit measures at ~71 s idle, a ~7x contention factor.  The residual
risk this leaves is small and specific: the six marker files not spot-run are **decorator-only**
edits (no test body, helper, or assertion changed), and every one of their nodes was verified to
land in the intended gate by collection.  Re-run the combined set on an idle box before the 5.32.1
tag; it is also the natural moment to take the `.test_durations` regeneration (§6).

---

## 6. Audit §6 (order of operations) and §7 (methodology): ACCEPTED

§6's ordering is right and I followed it: correctness fixes first (I1), then marker moves (I7),
then the mechanical dedupes (deferred as a batch), then parameter shrinks one at a time (deferred),
then regeneration, then the BLAS A/B last.  The one place I deviate is that §6.2 puts *all* marker
moves before regeneration; I split them, landing only the `[M]`-backed ones now and holding the
`[E]` ones and `test_fga.py` for the regeneration change, because gate membership and the durations
scale have to move together for the slow gate's headroom to stay knowable.

§7's two platform hazards are both confirmed useful and one was re-confirmed live (§0).  Its
integrity claim -- that no historical regression assertion is recommended for deletion -- holds for
every item I accepted; nothing here deletes or weakens an assertion.

### Recommended order for the next cycle

1. Measure the `[E]`-only marker candidates (chunk 2/3 list above) and land them **with**
   `test_fga.py`.
2. **Regenerate `.test_durations`** -- serial, idle box, BLAS pinned, both gates -- once gate
   membership has settled.  This is the single highest-value item left and it unblocks any honest
   statement about slow-gate headroom or fast-gate shard counts.
3. The duplicate-solve fixture/cache batch, with byte-identity diffs per site.
4. The parameter shrinks, one at a time, each against its own measured margin.
5. The BLAS A/B on real CI hardware (2-4 threads, not 1), last -- as the audit says.
6. Only then revisit shard counts, and revisit `--timeout` on the main gate: I1 removed the one
   test that made `--timeout` impossible, but adopting it still needs the slowest *single test* on
   a 2-vCPU runner measured first, so the timeout is a hang-guard and not a new absolute bar.

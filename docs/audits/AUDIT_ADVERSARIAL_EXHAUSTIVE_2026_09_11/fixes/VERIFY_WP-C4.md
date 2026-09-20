# VERIFY-WP-C4 -- `method='auto'` selects the direct-matrix MFT route

Independent adversarial verification of `feat/c4-mft-direct-default`
(`eb245b45`..`57f71923`, eight commits on `49ddf4bd`).  Author's report:
`WP-C4_MFT_DIRECT_DEFAULT_REPORT.md` beside this file.

Verification branch `verify/c4-mft-direct`, worktree `C:/tmp/lum_vc4`.
Probes and JSON: `validation/probe_verify_c4/`.  Decision tests:
`tests/unit/test_verify_c4_mft_direct.py` (17 ids).

Builds: Windows py3.14 (numpy 2.4.4, scipy 1.17.1, jax 0.11.0, CuPy present
with a broken cuFFT DLL) and WSL py3.12 (numpy 2.4.6 on scipy-openblas,
scipy 1.17.1, jax 0.10.2, no CuPy).  Every command carried
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line; every pytest run used `--capture=sys -p no:randomly`.  Every probe binds
ONE tree by `PYTHONPATH` and refuses to run unless `lumenairy.__file__`
resolves under it; the PRE tree is this verification's own
`git archive 49ddf4bd`, extracted read-only into the session scratchpad.
Nothing under `lumenairy/` was edited: every mutation ran on a tree copy in
the scratchpad, and the defects below carry the exact requested edit.

---

## The probes, and what each one is the evidence for

| probe | JSON | the claim it measures |
|---|---|---|
| `v4lib.py` | -- | the shared harness: tree anchor, load census, cold-cache helper, digest |
| `v4_ladder.py` | `v4_ladder_time_r1_{win,wsl}.json`, `v4_ladder_mem_r1_{win,wsl}.json` | the 36-shape TIME ladder (interleaved instrument) and the `tracemalloc` MEMORY ladder with byte counts derived from the code |
| `v4_boundary.py` | `v4_boundary_all_wsl.json`, `v4_boundary_thin_{win,wsl}.json`, `v4_boundary_thin_blocked_{win,wsl}.json`, `v4_boundary_thin64_win.json`, `v4_boundary_minarm_win.json` | the deciding shapes on their own: the branch's family, the THIN family under both instruments, the same anisotropy at 1/64, and the shapes a `min` conjunction would take |
| `v4_rss.py` | `v4_rss_{win,wsl}.json` | peak RSS from the OS, one child process per (shape, route), beside the derived byte counts |
| `v4_purity.py` | `v4_purity_{win,wsl}.json` | the selection under perturbations that are not a shape, the DISPATCH under seven input variants, no third arithmetic, the warning counts, and the JAX arms |
| `v4_bitid.py` + `v4_bitid_compare.py` | `v4_bitid_{base,branch}_{win,wsl}.json`, `v4_bitid_compare_{win,wsl}.json` | byte identity archive-to-archive and the SPLIT checked against the rule |
| `v4_entry.py` + `v4_entry_compare.py` | `v4_entry_{base,branch}_{win,wsl}.json`, `v4_entry_compare_{win,wsl}.json` | every public entry point that reaches the MFT, what moves, and whether a keyword brings it back |
| `v4_accuracy.py` | `v4_accuracy_{win,wsl}.json`, `v4_accuracy_g1_{win,wsl}.json` | the exactly-reduced reference, the re-derived bars, the degeneracy controls and the impostor control |
| `v4_blas.py` | `v4_blas_{win,wsl}.json` | four `OPENBLAS_CORETYPE`s x two thread counts |
| `v4_mutations.py` | `v4_mutations_{win,wsl}.json` | eight source-level mutations against the whole shipped file |
| `v4_census_plugin.py` | `v4_census_{win,wsl}.json` | where the rule fires in the shipped suite, and through which caller |
| `v4_c3_interaction.py` | `v4_c3_interaction_{win,wsl}.json` | the shapes WP-C3's transport hands the rule |
| `v4_never_entry.py` | `v4_never_entry_{base,branch}_win.json` | whether the Migration Guide's process-wide remedy restores the base bytes at the six no-keyword entry points |
| `v4_proposed_fix.py` | `v4_proposed_fix_win.json` | the V-C4-D1 edit, applied to a tree copy and measured |

---

## Verdict table

| # | claim | verdict | measured (WIN / WSL) |
|---|---|---|---|
| 1 | the boundary `1/32` -- "the dense route is never slower on EITHER build" over the region the rule captures | **REFUTED in the anisotropic region; CONFIRMED for square grids** | 36-shape ladder, best-of-7, routes INTERLEAVED: 21 captured shapes, **20 never slower, 1 slower**. Then a 12-shape THIN family (both ratios exactly 1/32, one input axis short), three rounds of best-of-nine, worst round, under BOTH instruments and on BOTH builds: **6 of 12 captured shapes are SLOWER on each build (7 of 12 in the union), by 1.09x to 9.69x**, than `min(chirp-Z, separable)`. Worst: `4096x64 -> 128x2` at **2.80 (WIN) / 9.69 (WSL)**. See **V-C4-D1**. Square and non-dyadic captured shapes at 1/32 are safe on both builds (WIN worst 0.391, WSL worst 0.683 on the same 36-shape ladder, 0.686 on the three-round boundary family). |
| 1a | is `max` the right conjunction? | **CONFIRMED as the SAFE conjunction, REFUTED as a tight one** | the brief's `(1/64, 1/8)` shape `1024x1024 -> 16x128`: rule says chirp, dense is **0.208 (WIN) / 0.615 (WSL, same ladder; 0.496 on the three-round family)** -- 1.6x to 4.8x FASTER. 12 (WIN) and 11 (WSL) of the 15 refused shapes on my ladder would have been faster on the dense route. `min` would take them all and be faster at every one of them -- but `min` also takes `1024x1024 -> 1024x8` (ratios `(1, 1/128)`), where dense reads **0.995 / 1.188** over two rounds, i.e. SLOWER. So `max` is the conjunction the safety claim needs, and the cost is coverage, not correctness. Five `min`-would-take shapes measured, `validation/probe_verify_c4/v4_boundary_minarm_win.json`. |
| 1b | "1/16 FAILS on WSL at N=1024, M=64 (dense 1.38-1.45x slower, three rounds)" | **INSTRUMENT-DEPENDENT** | the branch's OWN probe, re-run by me on the same WSL build, reproduces its verdict (`1/16 worst 1.411 NOT SAFE`). My instrument -- the same three routes, best-of-nine, but the REPEAT loop outermost and the route order rotating -- reads **0.709 / 0.994 / 0.765** at that shape, i.e. never slower. Both instruments agree that **1/32 is safe**, so the shipped value is the conservative one either way; what is not reproducible is the *reason given for excluding 1/16*. See **V-C4-N1**. |
| 2 | memory: dense cheapest at 42/42 shapes, 6.4x-334.9x | **CONFIRMED for square grids, REFUTED as "42 of 42" / "never argues against the rule anywhere"** | my 36-shape ladder, `tracemalloc` peak, cold: **35 of 36 dense-cheapest on BOTH builds**, 0.84x to 73.5x. The exception is a CAPTURED shape, `2048x64 -> 64x2`: dense **5.264 MB** against separable **4.399 MB** -- and it is the same shape family as V-C4-D1. Peak RSS from the OS, one child process per (shape, route), 7 shapes: dense cheapest at 7/7 on both builds. Byte counts re-derived from the code predict both the rule and the exception. |
| 2a | "the two builds' readings are IDENTICAL TO THE BYTE at every shape (`yes` at 42 of 42)" | **REFUTED -- and the branch's own table and JSON already say so** | the report's §3 table prints `NO` in that column at **all 42 rows**, and its own `c4_ladder_mem_{win,wsl}.json` agree byte for byte at **0 of 42** shapes (e.g. `N=64, M=16` chirp-Z: 8,001,186 bytes WIN against 801,515 WSL, a factor of 10). My ladder: **0 of 36**. The ORDERING is build-free; the readings are not. See **V-C4-D3**. |
| 3 | purity: one answer under perturbations that are not a shape | **CONFIRMED, and hardened** | 20 shapes x {python int, `np.int64`, `np.int32`, exact float} -> **one answer at 20 of 20**, both builds. DISPATCH driven through 7 input variants (`complex128` C-order, `complex64`, Fortran order, transposed, strided, real float64, real float32) x 2 `separable` settings x 12 shapes = **168 cases, 0 third arithmetic, 168/168 agree with the rule**, both builds. JAX `jit` agrees with eager at 3/3 and dispatches as the rule says. `jax.export` SYMBOLIC shapes raise -- on the BRANCH *and* on `49ddf4bd`, so that limitation is pre-existing (see **V-C4-N2**). |
| 4 | no third arithmetic: 88/88 match the named route, 0 match neither | **CONFIRMED and extended to 288 cases** | both primitives x both signs x both `separable` x centred/off-centre x 2 budgets (one under the guard, one past it) x 12 shapes = **288 cases**: matched neither **0**, agrees with the rule **288/288**, on BOTH builds. |
| 5 | byte identity archive-to-archive vs `49ddf4bd` | **CONFIRMED** | my own 468-key set (13 shapes, of which only 2 coincide with the branch's 12; a different field, a different `alpha` law and different centre conventions, so no FIXTURE is shared): `method='bluestein'` **117/117 identical**, `method='separable'` **117/117**, `_MFT_DIRECT_NEVER` **117/117**; no keyword **63 identical / 54 moved**, and the split **agrees with the rule at 117/117, 0 disagreements**. Identical on both builds. |
| 5a | every moved public entry point has a ONE-KEYWORD way back | **REFUTED -- 7 public entry points have none** | archive-to-archive at a captured shape and at a refused one, both builds: `compute_psf`, `resample_field`, `propagate` (asm and fresnel legs), `carrier_referenced_focus_readout`, `carrier_referenced_exact_focus_readout`, `re_reference` all MOVE and expose no route keyword; `propagate_traced_carrier_chain` reaches the same primitive through `_collins_transport`. The four that DO have one (`fresnel/fraunhofer/angular_spectrum_propagate_mft`, `asm_propagate`) reproduce the base bytes exactly. See **V-C4-D2** and the entry-point table. |
| 6 | accuracy: all inside derived two-sided bars, min room 1.54 decades; dense 31x-485x closer in the captured region | **CONFIRMED** | my own `fractions.Fraction` + `math.fsum` reference, bars re-derived from the code: 17 shapes x 2 budgets = **34 rows, all inside their own bars on BOTH builds, minimum room 1.606 (WIN) / 1.605 (WSL) decades**. `'auto'` matched neither route **0 times**; matched the route the rule names **34/34**. At the higher budget the dense-side gap is **21.5x to 91.8x** with a non-dyadic `alpha` at every shape; with the branch's own `alpha` it reads **28.2 .. 34.3** where the dense phase is genuinely rounded and **202 .. 397** where `alpha` makes it exact -- see **V-C4-N5**, which is why the quoted 31x-485x is two populations. At a budget of 1 the gap is **0.62x to 1.45x**: the branch's own statement that the advantage is about the PHASE and not the summation is confirmed. |
| 6a | the degenerate `alpha = 1e15/4096` control | **CONFIRMED** | `1e15/4096 = 244140625000.0` is an exact integer in float64 (`is_integer()` True), and at a DYADIC `alpha` the exact reduction and the route's `t - rint(t)` agree with **max &#124;frac difference&#124; exactly 0.0** -- so a power-of-two fixture with a dyadic `alpha` measures the instrument. At a generic `alpha` they part (4.77e-07 at my fixture). Both builds. |
| 6b | is the bar `R/4` with `R = N_max^2/((N-1)(M-1))` DERIVED, and is it two-sided? | **two-sided CONFIRMED; the derivation carries a factor-2 slip that the `/4` absorbs** | an IMPOSTOR dense arm (the separable answer returned in its place) reads a gap of **0.995 .. 1.003** against bars of 10.1 .. 16.3, i.e. **10x to 16x UNDER** -- the bar really is two-sided. But `R` is about TWICE the ratio the kernels imply: the chirp kernel is `exp(i*pi*alpha*m^2)` over `&#124;m&#124; <= L - N_out`, which is `alpha*(L-N_out)^2/2` TURNS, against the dense route's `alpha*(N-1)*(M-1)`. The `/4` is a CHOSEN safety factor, not a derived one. See **V-C4-D6**. |
| 7 | BLAS sweep: dense digests take 2-3 values, the route takes one, everything inside the bar | **CONFIRMED on BOTH builds** | 8 cells per build (`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, SANDYBRIDGE, KATMAI} x `OPENBLAS_NUM_THREADS` in {1,4}), `threadpoolctl` confirming the architecture actually changes in every cell: distinct dense digests per shape **3 / 2 / 2** (both builds), distinct chirp digests **1 / 1 / 1**, distinct routes **1** everywhere, `all_inside_bars` **true**, `route_follows_rule_everywhere` **true**.  The threads axis alone moves nothing at these sizes; the CORETYPE does. |
| 8 | the phase-budget guard runs before `'auto'` chooses; `on_dense=False` reproduces the old message; explicit `'direct'` stays silent | **CONFIRMED, and the counting and the message's truthfulness gated** | over 288 driven cases: past the guard, `'auto'` warns **exactly once at 144/144**, `method='direct'` is silent at **144/144**, `method='bluestein'`/`'separable'` warn once at **144/144**; under the guard `'auto'` warns **0 times**. The message names the route actually taken at **144/144**. A mutation that makes the message lie (`on_dense=False` forced) is caught by exactly one shipped id, `test_the_default_flip_does_not_take_a_warning_away_from_a_caller`. Both builds. |
| 9 | the census: 35 direct calls in 16 ids across 5 files, smallest ratio 1/512 | **CONFIRMED** (see the census section) | my own pytest plugin, wrapping `_auto_selects_direct` for a whole session and also recording each call's CALLER; the five firing files, Windows. |
| 10 | the version-narrative fix: no forward token remains, and the lines still say when | **CONFIRMED** | the only `5.49` in `lumenairy/**/*.py` is the unrelated numeric `5.49e-03` in `elements/rcwa/stack.py:1925`. The rephrasing is exactly the practice the gate's own docstring MEASURES as the repository's ("describe the change and let the CHANGELOG carry the version"); the CHANGELOG and the Migration Guide both name 5.49.0. |
| 11 | the 21 new tests and the three-entry mutation matrix | **CONFIRMED** | 8 mutations applied at SOURCE level to a scratchpad tree copy, the whole shipped file run against each, identical results on BOTH builds. The branch's three reproduce, and its own five-claim table reproduces exactly at two of three rows: run through `_run_every_claim()`, `rule_inverted` -> `['dispatch_dense_side', 'dispatch_chirp_side', 'way_back', 'accuracy']` (the report says the same four), `dense_answer_is_really_separable` -> `['accuracy']` (the report says "accuracy only" -- correct), control -> `[]` (correct), but `constant_silently_zero` -> **`['boundary', 'accuracy']`**, not "boundary only" -- see **V-C4-N4**. Of my four: `conjunction_max_to_min` caught (boundary only), `boundary_lt_not_le` caught (6 ids), `constant_1_over_31` NOT caught (correct, and the control: the claims are relation-based, so a retune is legal), `swapped_axis_arguments` caught by `test_auto_returns_the_route_the_rule_names_on_the_dense_side` (my first run of it said otherwise and was wrong -- see **V-C4-D4**, retracted). |
| 12 | the four "pre-existing on base" UI failures | **CONFIRMED** | all four reproduce on MY `git archive 49ddf4bd`. The citation walker's two V18.5 ids also fail on the archive -- for the environmental reason the report names (an archive is not a git repository) -- and pass on the branch worktree. |
| 13 | the WSL `test_installed_metadata_version_matches_source_version` failure is environmental | **CONFIRMED** | reproduces on the branch (`- 5.48.1 / + 5.11.0`); the WSL venv's installed distribution metadata is stale. Remedy in the environment section. |
| 14 | the WP-C3 interaction | **MEASURED on the C3 branch's own code** | see the C3 x C4 section: the Collins TRANSPORT leg can never be captured (ratio exactly 1 by construction); the Collins FOCUS READOUT is captured whenever `N_out <= N_fine/32`, which is 4 of the 6 configurations I drove. All of them are square and work-dense, so V-C4-D1 does not bite them -- but V-C4-D2 does, because `propagate_traced_carrier_chain` has no route keyword. |

---

## Defects

### V-C4-D1 (P1) -- the "never slower" premise is false in the anisotropic part of the captured region, and no value of `_MFT_DIRECT_MAX_RATIO` fixes it

**What.**  The constant was derived from a ladder of SQUARE shapes (`N x N ->
M x M`, 42 of them).  The rule it feeds is a statement about two RATIOS, and
two shapes with the same ratio pair can differ by more than two decades in the
quantity that actually decides the race.

**The mechanism, from the code.**  `_direct_matrix_2d` builds two kernels of
`My*Ny` and `Mx*Nx` entries -- each entry a `numpy` complex `exp`, tens of
times the cost of a multiply-add -- and then spends
`min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)` multiply-adds using them (the
two costs the function itself compares to pick its association order).  The
ratio of the two, `work per kernel entry`, is what says whether the BUILD or
the products dominate.  For a square `N -> M` it is `(N+M)/2`, so it is large
for every square shape the rule captures.  For a thin input it collapses:
`2048x2048 -> 64x64` reads **1056**, `2048x64 -> 64x2` reads **4.0**, and both
sit at ratio exactly `(1/32, 1/32)`.

**Measured.**  Three independent rounds of best-of-nine each, the verdict on
the WORST round, both builds, and under BOTH instruments (mine, which
interleaves the routes; and the branch's, which runs them in blocks) --
`dense / min(chirp-Z 2-D, separable)`:

| shape (captured) | work/entry | WIN interleaved | WIN blocked | WSL interleaved | WSL blocked |
|---|---|---|---|---|---|
| `2048x256 -> 64x8` | 39.4 | 0.529 | 0.549 | 0.850 | 0.789 |
| `2048x128 -> 64x4` | 12.0 | 0.849 | 0.770 | **1.138** | **1.250** |
| `2048x64 -> 64x2` | 4.0 | **1.432** | **1.287** | **1.810** | **1.308** |
| `2048x32 -> 64x1` | 2.0 | **2.654** | **2.357** | **2.293** | **1.936** |
| `1024x128 -> 32x4` | 19.7 | 0.505 | 0.489 | 0.588 | 0.810 |
| `1024x64 -> 32x2` | 6.0 | 0.696 | 0.854 | 0.751 | 0.768 |
| `1024x32 -> 32x1` | 2.0 | **1.317** | **1.581** | 0.965 | 0.956 |
| `512x64 -> 16x2` | 9.8 | 0.471 | 0.581 | 0.445 | 0.525 |
| `512x32 -> 16x1` | 3.0 | 0.642 | 0.595 | 0.486 | 0.390 |
| `64x2048 -> 2x64` | 4.0 | **1.182** | **1.092** | **1.378** | **1.615** |
| `32x2048 -> 1x64` | 1.5 | **2.040** | **2.180** | **2.040** | **1.623** |
| `4096x64 -> 128x2` | 3.0 | **2.802** | **2.619** | **4.040** | **9.685** |

Six of twelve captured shapes are slower on EACH build (seven in the union of
the two), in every round, under both instruments.  The 36-shape main ladder
finds the same thing independently at `2048x64 -> 64x2` (1.278).

**The work ratio is a one-sided SCREEN, not a crossover.**  It is not monotone
in the readings: `512x32 -> 16x1` reads 3.0 and is safe on both builds, and
`1024x64 -> 32x2` reads 6.0 and is safe, because at small absolute sizes the
chirp-Z route's fixed costs (planning, padding to `next_fast_len`) dominate
whatever the asymptotic count says.  So a threshold refuses some shapes that
would have been fine.  That is the correct direction for a rule whose premise
is "never slower": the cost of refusing a safe shape is a few per cent of
time; the cost of capturing an unsafe one is up to 9.7x.

**Tightening the ratio does not help.**  At `_MFT_DIRECT_MAX_RATIO = 1/64` --
the value the report offers as "the ratio with a two-fold margin at every
shape" -- the same family is still captured and still slower: `4096x64 ->
64x1` reads **1.487 / 1.073** in two rounds and `64x4096 -> 1x64` reads
**1.079 / 1.102**.  The ratio is simply the wrong axis for this failure.

**The memory half does not rescue it either** (V-C4-D3): at `2048x64 -> 64x2`
the dense route's `tracemalloc` peak is 5.264 MB against the separable route's
4.399 MB, on both builds.  So at that shape the rule is taking a route that is
both slower and larger, and the only remaining argument for it is accuracy.

**Requested edit** (`lumenairy/propagators/_bluestein.py`).  Add a second,
equally build-free condition to `_auto_selects_direct`, and a constant beside
`_MFT_DIRECT_MAX_RATIO` for it:

```python
#: The dense route builds ``My*Ny + Mx*Nx`` transcendental kernel entries and
#: then spends ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds
#: using them.  Below this many multiply-adds PER kernel entry the build
#: dominates and the chirp-Z route wins even at a ratio the square ladder
#: measured safe.  DERIVED, 2026-09-20 (VERIFY-WP-C4): over a 12-shape thin
#: ladder, three rounds of best-of-nine, both builds, both instruments, the
#: largest ratio at which the dense route was measured SLOWER is 12.0
#: (``2048x128 -> 64x4``) and the smallest at which it was measured safe above
#: that is 19.7 (``1024x128 -> 32x4``); 16 sits in that gap.  For a SQUARE
#: ``N -> M`` the quantity is ``(N + M)/2``, so every square shape with
#: ``N >= 32`` is unaffected -- including every shape the shipped suite drives
#: (smallest 264).  It is a one-sided SCREEN and not a crossover: the readings
#: are not monotone in it (``512x32 -> 16x1`` reads 3.0 and is safe, because
#: at that size the chirp-Z route's fixed costs dominate), so a threshold
#: refuses some shapes that would have been fine.  That is the direction a
#: "never slower" premise needs.
_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = 16.0
```

and, at the end of `_auto_selects_direct`, replace

```python
    return max(my / ny, mx / nx) <= r
```

with

```python
    if not (max(my / ny, mx / nx) <= r):
        return False
    # A ratio cannot see a THIN input: at ratio (1/32, 1/32) the dense route
    # is 1.1x to 9.7x slower once the other axis is short, because it is then
    # paying more transcendentals than multiply-adds.  MEASURED 2026-09-20
    # (VERIFY-WP-C4, ``validation/probe_verify_c4/v4_boundary_thin_*.json``).
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops >= float(_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries
```

**The edit was APPLIED and MEASURED, not merely suggested.**
`validation/probe_verify_c4/v4_proposed_fix.py` copies `lumenairy/` into the
session scratchpad, applies exactly the constant and the three lines above,
and asks the patched rule about four groups of shapes:

| group | before | after |
|---|---|---|
| the 9 shapes this verification measured SLOWER | 9 of 9 captured | **0 of 9** |
| the 11 square / near-square ladder shapes | 11 of 11 | **11 of 11** |
| the 6 shapes the whole shipped suite drives | 6 of 6 | **6 of 6** |
| the 7 anisotropic shapes measured safe | 7 of 7 | 4 of 7 (the screen is one-sided) |

and then runs both test files on the patched tree:
`test_c4_mft_direct_default.py` **21 passed**,
`test_verify_c4_mft_direct.py` **17 passed**.  So the edit closes the defect,
costs three anisotropic shape families nobody drives, and breaks nothing.

**Nothing shipped moves.**  `test_c4_mft_direct_default.py`'s `_DENSE_SIDE`
reads 33.0, 49.5, 66.0, 132.0 and 217.6 multiply-adds per kernel entry, and
the six shapes the rule fires at in the whole suite read 264, 516, 520, 1028,
1044 and 2052 -- all far above 16, so the proposed guard captures every one of
them exactly as today.  `tests/unit/test_verify_c4_mft_direct.py` is written
so that this edit does not falsify it either.  The report's §2 and the CHANGELOG's "why 1/32" block
need a sentence saying the ladder was square and what the second condition is
for; the constant's own comment block needs the same.

### V-C4-D2 (P1) -- seven public entry points move with no one-keyword way back

**What.**  The campaign rule is that every public entry point a default flip
moves has a ONE-KEYWORD way back.  An AST sweep of `lumenairy/` for calls to
`_bluestein_2d` / `_bluestein_centred_2d` / `_direct_matrix_2d` and to the
three public MFT propagators finds twelve callers.  Driven
archive-to-archive at a captured shape (`512x512 -> 16x16`) and at a refused
one (`512x512 -> 128x128`), on both builds:

| public entry point | reaches the rule via | captured shape MOVES | refused shape identical | one-keyword way back |
|---|---|---|---|---|
| `angular_spectrum_propagate_mft` | `_bluestein_centred_2d` | yes | yes | **`method='bluestein'` / `'separable'`** -- verified byte-identical to base |
| `fresnel_propagate_mft` | `_bluestein_centred_2d` | yes | yes | **`method=`** -- verified |
| `fraunhofer_propagate_mft` | `_bluestein_centred_2d` | yes | yes | **`method=`** -- verified |
| `asm_propagate` | the three above, `**method_kwargs` | yes | yes | **`method='separable'`** (no `method` of its own, so it forwards) -- verified |
| `compute_psf(method='mft')` | `fraunhofer_propagate_mft` | **yes** | yes | **NONE** -- its `method` names the sampler (`'fft'`/`'mft'`) |
| `resample_field(method='chirpz')` | `_bluestein_centred_2d` | **yes** | yes | **NONE** -- its `method` names the resampler |
| `propagate(method='asm', output_grid=...)` | `angular_spectrum_propagate_mft` | **yes** | yes | **NONE** -- its `method` names the propagator family, so the keyword is taken |
| `propagate(method='fresnel', output_grid=...)` | `fresnel_propagate_mft` | **yes** | yes | **NONE** -- same |
| `carrier_referenced_focus_readout` | `angular_spectrum_propagate_mft` | **yes** | yes | **NONE** |
| `carrier_referenced_exact_focus_readout` | `angular_spectrum_propagate_mft` | **yes** | yes | **NONE** (it has `_bluestein_separable`, which selects a chirp-Z ARM and is powerless once the rule fires) |
| `re_reference` (`CarrierField` verb) | `angular_spectrum_propagate_mft` | **yes** | yes | **NONE** (it has `_separable`, same objection) |
| `propagate_traced_carrier_chain` | `_collins_focus_readout` -> `_collins_transport` -> `_bluestein_centred_2d` | yes (census: `_collins_transport` answers `'direct'` 3 times in `test_audit2609_b4`) | -- | **NONE** (`transport='sziklas'` changes the PHYSICS route, not the MFT one) |
| `propagate_through_system` (fresnel leg) | `fresnel_propagate_mft` | **no** -- `N_out` is pinned to the chain's own sample count, so the ratio is exactly 1 | -- | not needed |

**The process-wide remedy DOES work -- it is just not a keyword.**  Measured
archive-to-archive at the captured shape, on both trees: with
`_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER` set for the process, all six
drivable no-keyword entry points (`compute_psf`, `resample_field`,
`propagate(method='asm')`, both carrier readouts and `re_reference`) reproduce
the `49ddf4bd` bytes EXACTLY, and all six move without it.  So the Migration
Guide's remedy is true; the objection is that it is private, process-wide, and
unavailable to a caller who wants the previous bytes for ONE call inside a
program that also wants the new ones.

**What "verified" means in that table.**  Two separate checks.  (a) The
keyword call is byte-identical base-to-branch at the captured shape, so naming
the route reaches the same arithmetic on both trees.  (b) In-tree,
`method='bluestein'` reproduces the bytes a no-keyword call gives with
`_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER` -- which the shipped file proves is
the pre-flip dispatch -- and those bytes DIFFER from what the caller gets by
default, so the check is not vacuous.  Gated by
`test_verify_c4_mft_direct.py::test_the_keyword_way_back_reproduces_the_previous_dispatch_exactly`.
Which of `'bluestein'` and `'separable'` is the right one for a given caller
depends on the `separable` flag that caller passed, exactly as before.

All twelve are exported at the top level of `lumenairy`.  The Migration Guide
acknowledges two of them (`resample_field`, "the carrier readouts") and offers
the private module constant `_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`.  That
is process-wide and private; it is not a keyword, and it is exactly what the
campaign rule was written to exclude.  `compute_psf`, `re_reference` and both
`propagate` legs are not mentioned at all, and the guide's line
"`propagate(method='asm' | 'fresnel' | 'fraunhofer', ...)`, which forwards to
the three above" reads as though the keyword reaches them; it cannot, because
`propagate`'s own `method` parameter already has that name.

**Requested edits.**

1.  `lumenairy/analysis/psf_mtf_otf.py` -- add a pass-through keyword to
    `compute_psf` and forward it, keeping `method` for the sampler:

    ```python
    def compute_psf(pupil, wavelength, f, dx_pupil, N_psf=None, oversample=1,
                    normalize='power', *, method='fft', dx_psf=None,
                    mft_method='auto'):
    ```
    and at `psf_mtf_otf.py:513`
    ```python
        E_focal = fraunhofer_propagate_mft(
            pupil, float(f), float(wavelength), float(dx_pupil),
            dx_psf, int(N_psf), method=mft_method)
    ```
    with `_compute_psf_mft` gaining the same parameter and the docstring
    saying that `mft_method='bluestein'` restores the pre-5.49.0 bytes.

2.  `lumenairy/propagators/mft.py` -- add `chirpz_method='auto'` to
    `resample_field`, forward it into `_resample_field_chirpz` and on to
    `_bluestein_centred_2d` at `mft.py:618`.

3.  `lumenairy/propagators/dispatch.py` -- add `mft_method='auto'` to
    `propagate` and to `_dispatch_bare_grid_with_output`, and pass it as
    `method=mft_method` at `dispatch.py:940 / 944 / 948`.  (`asm_propagate`
    needs nothing; `**method_kwargs` already carries `method=`.)

4.  `lumenairy/propagators/carrier.py` -- add `mft_method='auto'` to
    `carrier_referenced_focus_readout` (`:4479`),
    `carrier_referenced_exact_focus_readout` (`:7197`),
    `_collins_transport` (`:2498`), `_collins_focus_readout`,
    `_collins_carrier_leg` and `propagate_traced_carrier_chain`, forwarding it
    to the MFT call in each.

5.  `lumenairy/propagators/carrier_field.py` -- add `mft_method='auto'` to
    `re_reference` and forward it at `:1460`.

6.  `Migration-Guide.md` -- list all twelve entry points, say which have a
    keyword, and delete the implication that `propagate`'s `method=` reaches
    the MFT route.

If the maintainer prefers ONE keyword name across the library, `mft_method=`
is the only one free on every signature above (`method` is taken on four of
them).

### V-C4-D3 (P2) -- the memory section's cross-build claim contradicts its own table and its own data

The report's §3 says "the two builds' readings are IDENTICAL TO THE BYTE at
every shape (the `identical WIN/WSL` column is `yes` at 42 of 42)".  The
column printed immediately above reads `NO` at all 42 rows, and the branch's
own `validation/probe_c4_mft_direct/c4_ladder_mem_win.json` and
`..._wsl.json` agree at **0 of 42** shapes.  My own 36-shape ladder agrees at
0 of 36.  What IS build-free is the ORDERING, which holds at 35 of 36 on both
builds and fails at the same shape on both.

**Requested edit** -- `WP-C4_MFT_DIRECT_DEFAULT_REPORT.md` §3 and the
CHANGELOG's memory paragraph: replace "the readings are identical to the byte
across builds" with "the ORDERING is identical across builds at every shape;
the readings are not (the two builds' `tracemalloc` peaks differ at 42 of 42
shapes, by up to a factor of 10 at `N=64, M=16`)", and change "the memory half
never argues against the rule anywhere" to "... anywhere on the SQUARE ladder;
at `2048x64 -> 64x2` the dense route is the larger of the two (5.264 MB
against 4.399 MB), which is the same anisotropic family as V-C4-D1".

### V-C4-D4 -- RETRACTED: a y/x transposition IS caught, and the first measurement of it was my instrument's fault

**What I first measured, and why it was wrong.**  Mutating the two call sites
to `_auto_selects_direct(Ny_in, Nx_in, N_out_x, N_out_y)` appeared to leave
`test_c4_mft_direct_default.py` at 21 passed.  It did not: my source-level
mutation matched the string `_auto_selects_direct(Ny_in, Nx_in, N_out_y,
N_out_x)` and the function's own `def` line is spelled exactly that way, so
the patch renamed the PARAMETERS in the same order as the call sites and the
whole mutation was a no-op.  The matrix then reported "not caught" about a
tree that had not been mutated.

**Re-measured** with the anchor narrowed to `and _auto_selects_direct(...)`,
which matches only the two call sites: the shipped file **does** catch it, at
`test_auto_returns_the_route_the_rule_names_on_the_dense_side` -- all four
parametrizations, both primitives.  The shape that sees it is
`(512, 1024, 16, 32)` in `_DENSE_SIDE`: both its ratios are 1/32, and
exchanging the two OUTPUT sizes takes it to `(1/16, 1/64)`, which is past the
boundary.  So the shipped file is not transposition-blind after all; it has
exactly one shape that sees a transposition, and that shape is enough.

**Kept anyway, as a widening rather than a closure.**
`test_verify_c4_mft_direct.py::test_auto_dispatches_as_the_rule_says_at_transpose_sensitive_shapes`
drives four transposition-sensitive shapes instead of one, on both primitives,
and
`::test_the_rule_is_sensitive_to_a_y_x_transposition_at_all` asserts the
PREMISE those shapes rest on -- that a partial swap changes the answer there --
which nothing in the shipped file states, so a future edit to `_DENSE_SIDE`
cannot quietly remove the only shape that carries the property.

**The trap, recorded.**  A source-level mutation whose anchor also matches the
definition is a mutation that measures nothing and reads as a coverage gap.
Anchor on the CALL, not on the name.

### V-C4-D5 (P3) -- the shipped warning claim counts "at least one", not "exactly one"

The guard now has TWO call sites (`_bluestein_2d` and
`_bluestein_centred_2d`'s dense arm).  A centred `'auto'` call at a refused
shape reaches the inner `_bluestein_2d`, and a centred `'auto'` call at a
captured shape returns before it -- so the code is right, and I measured it
right (exactly one warning at 144/144 cases).  But nothing in the shipped file
asserts the COUNT, which is precisely the failure a two-call-site guard
invites.  Closed here by
`test_verify_c4_mft_direct.py::test_the_phase_guard_warns_exactly_once_per_call`.

### V-C4-D6 (P3) -- `_phase_term_ratio`'s derivation is off by a factor of two

`_phase_term_ratio` returns `N_max^2 / ((N-1)(M-1))` and its docstring calls
the chirp-Z route's phase argument `alpha*N_max^2`.  The kernel the code
builds is `exp(i*sign*pi*alpha*m^2)` over the PADDED index `|m| <= L - N_out`
with `L = next_fast_len(N_in + N_out - 1)`, i.e. `alpha*(L - N_out)^2 / 2`
TURNS -- the `pi` is half a turn.  The dense route's is
`alpha*(N-1)*(M-1)` turns exactly.  So the true ratio is about `R/2`, and the
bar `R/4` is `R_turns/2`, which the measured constant ratio
(`C_chirp/C_dense` in [1.48, 4.04]) clears by 3x to 8x.  The bar is therefore
safe; what is wrong is the word DERIVED in its docstring -- the `/4` is a
chosen safety factor, and the reader is told it comes from the kernels.

**Requested edit** -- in `tests/unit/test_c4_mft_direct_default.py`'s
`_phase_term_ratio` and `_derived_bar` docstrings, and in the report's §4.2
and §10: say `pi*alpha*N_max^2` radians = `alpha*N_max^2/2` turns (and that
`m` runs to `L - N_out`, which is at or above `N-1`), so `R` as returned is
twice the turn ratio and the `/4` is a chosen margin on top of a 2x slack.
Gated here by
`test_verify_c4_mft_direct.py::test_the_shipped_phase_ratio_bar_is_conservative_against_the_kernels`.

---

## Notes that are not defects

**V-C4-N1 -- the 1/16 verdict is instrument-dependent, and the instrument
asymmetry is real.**  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` pins the DENSE route (two BLAS products) to one thread and
does NOT constrain scipy's pocketfft, which `fft_infra.SCIPY_FFT_WORKERS = -1`
gives every core.  The shipped comparison therefore races a one-thread dense
route against an all-core chirp-Z route, and its outcome moves with how many
cores are spare.  That is the correct comparison for a DECISION (it is what a
caller sees), and it is conservative (it can only make the dense route look
worse), but it makes the boundary reading load-dependent, which is why my
contended WSL box reads 0.99 at `1024 -> 64` where the branch's quiet one read
1.45.  Both instruments put 1/32 inside the safe region, so the shipped
constant is not in question -- only the sentence "this is reproducible and not
a contention artefact" about the 1/16 failure is.  `--workers 1` is wired into
`validation/probe_verify_c4/v4_boundary.py` for whoever wants the symmetric
arm.

**V-C4-N2 -- JAX symbolic shapes.**  `jax.export` with a polymorphic shape
raises `InconclusiveDimensionOperation` on the BRANCH (`int()` in
`_auto_selects_direct`) and on `49ddf4bd` (`'a' > 'b'` inside the chirp-Z
path).  Shape-polymorphic export of `_bluestein_2d` has never worked; the rule
makes it fail one line earlier, with a clearer message.  Pre-existing.

**V-C4-N3 -- the `max` conjunction leaves speed on the table, and it is still
the right one.**  At 12 of the 15 shapes my Windows ladder refuses (11 of 15
on WSL), the dense route is the faster one, by 1.3x to 17x -- including every mixed-ratio shape
(`1024x1024 -> 16x128`, ratios `(1/64, 1/8)`: dense 0.0120 s against separable
0.0575 s, `d/fb = 0.208` on Windows and 0.615 on WSL; `1024x64 -> 16x1024`
reads 0.059 on Windows).  The two-sided arm, which the branch does not have: of five
shapes a `min` conjunction would take and `max` refuses, four are faster on
the dense route and one -- `1024x1024 -> 1024x8`, ratios `(1, 1/128)` -- is
SLOWER, 0.995 and 1.188 over two rounds.  So `min` is measurably unsafe and
`max` is measurably conservative; the report should simply say what the
conservatism costs.

**V-C4-N5 -- the headline "31x to 485x" is two populations, and only the
smaller number is a comparison of the routes.**  The branch's accuracy ladder
holds `alpha = budget / N_max^2`, and at `budget = 1e3` that value is EXACTLY
representable whenever `N_max^2`'s odd part divides 125 -- which is true at
`64 -> 2`, `128 -> 4`, `160 -> 5`, `256 -> 8`, `128 -> 2` and (in my ladder)
`320 -> 10`.  At those fixtures `t = alpha*n*k` is exact, `t - rint(t)` is
exact, and the dense route's phase carries NO error at all: it agrees with an
exactly-reduced reference by construction, which is the same degeneracy the
report itself identifies at `1e15` and keeps as a labelled control.

Re-measured with the branch's own `alpha` (my probe, `--generic 1.0
--budgets 1000`), the dense-side gap splits cleanly:

| `alpha*n*k` exact? | shapes | gap |
|---|---|---|
| NO (a real comparison) | `96 -> 3`, `192 -> 6`, `224 -> 7` | **28.2 .. 34.3** (WIN) / **27.9 .. 34.0** (WSL); the branch reads 43.4 / 31.6 / 30.9 at the same three |
| YES (dense exact by construction) | `64 -> 2`, `128 -> 4`, `160 -> 5`, `256 -> 8`, `128 -> 2`, `320 -> 10` | **202 .. 397** (WIN) / **209 .. 408** (WSL); the branch reads 146 .. 485 |

With a non-dyadic `alpha` at EVERY shape (my default, `GENERIC = 1/3`) the
whole dense-side ladder reads **21.5 .. 91.8**, and it tracks `R` as the
derivation says it should.  So the mechanism is confirmed and the ordering is
confirmed; it is the upper end of the quoted range that is a property of the
fixture.  Suggested edit: quote "about 30x at fixtures where the dense route's
phase is genuinely rounded, and larger where `alpha` happens to make it exact",
and mark the five exact-`alpha` rows of §4.3 the way `64 -> 2` is marked at
`1e15`.  Evidence: `validation/probe_verify_c4/v4_accuracy_g1_{win,wsl}.json`
(`alpha_makes_t_exact` is recorded per row).

**V-C4-N4 -- one row of the shipped mutation table is narrower than the
measurement.**  `WP-C4_MFT_DIRECT_DEFAULT_REPORT.md` §10 records
`the constant is silently 0` as refused by "**boundary only**".  Re-run
through the file's own `_run_every_claim()`, it is refused by **boundary AND
accuracy**: the accuracy claim ends with
`assert _bits(auto) == _bits(dense)`, and with the constant at zero `'auto'`
is not the dense answer.  The paragraph under the table -- "the rule and the
dispatch agree with each other perfectly, so no dispatch comparison can see
it" -- is true and is the point; it is only the word "only" that overreaches.
The shipped id is unaffected (it asserts `expected in caught`, not
`caught == [expected]`).  Suggested edit: change that cell to
"boundary (and accuracy, on its bit-identity tail)".

---

## The boundary, my ladder

36 shapes, best of 7, COLD (every registered cache cleared and `gc.collect()`
before each repeat), routes INTERLEAVED with the order rotating by repeat, the
reference loop read before and after every shape, all three routes' answers
digested in the same pass (all three differ at 36 of 36, so no row is fast
because a route did not run).  Load: 12 -> 24 Python processes of 518 -> 534,
reference loop 0.06 ms at both ends, worst per-row drift 2.11x.  Windows
py3.14, seconds.

| name | in -> out | max ratio | rule | dense | sep | chirp | d/fb |
|---|---|---|---|---|---|---|---|
| `sq_2048_16` | 2048x2048 -> 16x16 | 0.007812 | **direct** | 0.0171 | 0.1455 | 0.2029 | 0.117 |
| `sq_2048_32` | 2048x2048 -> 32x32 | 0.015625 | **direct** | 0.0263 | 0.1426 | 0.1946 | 0.185 |
| `sq_2048_64` | 2048x2048 -> 64x64 | 0.031250 | **direct** | 0.0535 | 0.1385 | 0.3186 | 0.386 |
| `sq_1024_16` | 1024x1024 -> 16x16 | 0.015625 | **direct** | 0.0041 | 0.0324 | 0.0517 | 0.128 |
| `sq_1024_32` | 1024x1024 -> 32x32 | 0.031250 | **direct** | 0.0079 | 0.0306 | 0.0678 | 0.257 |
| `sq_512_16` | 512x512 -> 16x16 | 0.031250 | **direct** | 0.0015 | 0.0071 | 0.0213 | 0.207 |
| `sq_512_8` | 512x512 -> 8x8 | 0.015625 | **direct** | 0.0010 | 0.0081 | 0.0138 | 0.126 |
| `sq_256_8` | 256x256 -> 8x8 | 0.031250 | **direct** | 0.0004 | 0.0020 | 0.0103 | 0.226 |
| `sq_128_4` | 128x128 -> 4x4 | 0.031250 | **direct** | 0.0002 | 0.0006 | 0.0012 | 0.391 |
| `sq_64_2` | 64x64 -> 2x2 | 0.031250 | **direct** | 0.0001 | 0.0003 | 0.0006 | 0.350 |
| `sq_1024_33` | 1024x1024 -> 33x33 | 0.032227 | chirp | 0.0078 | 0.0302 | 0.0696 | 0.258 |
| `sq_512_17` | 512x512 -> 17x17 | 0.033203 | chirp | 0.0015 | 0.0089 | 0.0225 | 0.167 |
| `sq_2048_128` | 2048x2048 -> 128x128 | 0.062500 | chirp | 0.0985 | 0.1574 | 0.2168 | 0.626 |
| `sq_1024_64` | 1024x1024 -> 64x64 | 0.062500 | chirp | 0.0147 | 0.0401 | 0.0541 | 0.368 |
| `sq_512_32` | 512x512 -> 32x32 | 0.062500 | chirp | 0.0023 | 0.0090 | 0.0134 | 0.260 |
| `sq_2048_256` | 2048x2048 -> 256x256 | 0.125000 | chirp | 0.2285 | 0.2068 | 0.4618 | 1.105 |
| `sq_1024_128` | 1024x1024 -> 128x128 | 0.125000 | chirp | 0.0311 | 0.0415 | 0.1078 | 0.750 |
| `sq_512_64` | 512x512 -> 64x64 | 0.125000 | chirp | 0.0055 | 0.0098 | 0.0199 | 0.563 |
| `sq_1024_256` | 1024x1024 -> 256x256 | 0.250000 | chirp | 0.0683 | 0.0559 | 0.1287 | 1.222 |
| `sq_1024_512` | 1024x1024 -> 512x512 | 0.500000 | chirp | 0.1675 | 0.0929 | 0.1845 | 1.802 |
| `an_2048x512` | 2048x512 -> 64x16 | 0.031250 | **direct** | 0.0139 | 0.0445 | 0.0734 | 0.312 |
| `an_512x2048` | 512x2048 -> 16x64 | 0.031250 | **direct** | 0.0091 | 0.0351 | 0.0721 | 0.260 |
| `an_2048x256` | 2048x256 -> 64x8 | 0.031250 | **direct** | 0.0069 | 0.0135 | 0.0335 | 0.514 |
| `an_1024x128` | 1024x128 -> 32x4 | 0.031250 | **direct** | 0.0016 | 0.0040 | 0.0114 | 0.397 |
| `an_2048x64` | 2048x64 -> 64x2 | 0.031250 | **direct** | 0.0054 | 0.0042 | 0.0111 | **1.278 SLOWER** |
| `mx_in_1024_8x32` | 1024x1024 -> 8x32 | 0.031250 | **direct** | 0.0050 | 0.0431 | 0.0610 | 0.116 |
| `mx_in_2048_16x64` | 2048x2048 -> 16x64 | 0.031250 | **direct** | 0.0218 | 0.1907 | 0.3705 | 0.114 |
| `mx_in_an` | 2048x1024 -> 64x16 | 0.031250 | **direct** | 0.0196 | 0.1008 | 0.2129 | 0.195 |
| `mx_out_1024_16x128` | 1024x1024 -> 16x128 | 0.125000 | chirp | 0.0120 | 0.0575 | 0.0857 | 0.208 |
| `mx_out_1024_128x16` | 1024x1024 -> 128x16 | 0.125000 | chirp | 0.0106 | 0.0450 | 0.1095 | 0.236 |
| `mx_out_2048_32x256` | 2048x2048 -> 32x256 | 0.125000 | chirp | 0.0454 | 0.1937 | 0.3673 | 0.235 |
| `mx_out_1024_8x256` | 1024x1024 -> 8x256 | 0.250000 | chirp | 0.0148 | 0.0619 | 0.1399 | 0.239 |
| `mx_out_2048_64x128` | 2048x2048 -> 64x128 | 0.062500 | chirp | 0.0574 | 0.2022 | 0.2554 | 0.284 |
| `nd_1000_25` | 1000x1000 -> 25x25 | 0.025000 | **direct** | 0.0065 | 0.0277 | 0.0781 | 0.235 |
| `nd_768_24` | 768x768 -> 24x24 | 0.031250 | **direct** | 0.0036 | 0.0202 | 0.0675 | 0.178 |
| `nd_1536_48` | 1536x1536 -> 48x48 | 0.031250 | **direct** | 0.0236 | 0.0865 | 0.2351 | 0.273 |

The same 36-shape ladder on WSL py3.12 (14 Python processes of 23, reference
loop 0.041 -> 0.044 ms, all three routes' digests distinct at 36 of 36): the
same 21 shapes are captured, **20 never slower and 1 slower** -- the slower one
is `an_2048x64` again, at **1.802**.  The square and non-dyadic captured
shapes read 0.083 .. 0.683.

WSL, the boundary family (three rounds, worst round):
`1/32` reads 0.083 .. 0.686, `1/16` reads 0.249 .. 0.994, `1/8` reads
0.552 .. 1.752.  The branch's own probe, re-run by me on the same build,
reads `1/64 0.460 SAFE`, `1/32 0.721 SAFE`, `1/16 1.411 NOT SAFE`,
`1/8 1.901 NOT SAFE`.

---

## Accuracy, re-derived

The reference: `alpha` is a float64 and therefore an exact rational and the
indices are integers, so `t = alpha*n*k` and its fractional turn are EXACT in
`fractions.Fraction`; the single float64 rounding lands on a number already
inside `[-1/2, 1/2)`.  Products are formed in float64 (one rounding each) and
the double sum is accumulated with `math.fsum`, so the reference's summation
growth factor is 1.  `alpha` carries a generic non-dyadic multiplier (1/3),
because at a power-of-two `N` with a dyadic `alpha` both the reference and the
route's `t - rint(t)` are exact and agree by construction -- the degeneracy the
branch caught at `64 -> 2` and which this probe therefore cannot fall into at
any shape.

The bars are re-derived from the two kernels:
`bar = (g + 1 + 2*pi*max|t|) * eps * sum|E|`, with `g = 3*log2(L^2)` for the
chirp-Z routes and `g = sqrt(N) + sqrt(M)` for the dense route's two products,
`max|t|_chirp = alpha*(L - M)^2 / 2` turns and
`max|t|_dense = alpha*(N-1)*(M-1)` turns.

| budget | shape | side | chirp rel | dense rel | gap | `R/4` | `R_turns/4` | inside bars |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 -> 2 | dense | 3.724e-16 | 3.099e-16 | 1.20 | 16.25 | 8.13 | yes / yes |
| 1 | 96 -> 3 | dense | 3.279e-16 | 3.748e-16 | 0.87 | 12.13 | 5.94 | yes / yes |
| 1 | 256 -> 8 | dense | 3.123e-16 | 3.049e-16 | 1.02 | 9.18 | 4.59 | yes / yes |
| 1 | 24 -> 12 | chirp | -- | -- | -- | 0.57 | 0.26 | yes / yes |
| 1000 | 64 -> 2 | dense | 4.711e-14 | 5.131e-16 | **91.8** | 16.25 | 8.13 | yes / yes |
| 1000 | 96 -> 3 | dense | 5.983e-14 | 1.638e-15 | **36.5** | 12.13 | 5.94 | yes / yes |
| 1000 | 128 -> 4 | dense | 3.653e-14 | 1.199e-15 | **30.5** | 10.75 | 5.38 | yes / yes |
| 1000 | 192 -> 6 | dense | 7.807e-14 | 1.417e-15 | **55.1** | 9.65 | 4.83 | yes / yes |
| 1000 | 256 -> 8 | dense | 5.313e-14 | 2.475e-15 | **21.5** | 9.18 | 4.59 | yes / yes |
| 1000 | 320 -> 10 | dense | 5.216e-14 | 2.135e-15 | **24.4** | 8.92 | 4.46 | yes / yes |
| 1000 | 24 -> 12 | chirp | 4.430e-14 | 3.386e-14 | 1.31 | 0.57 | 0.26 | yes / yes |
| 1000 | 48 -> 24 | chirp | 4.940e-14 | 3.255e-14 | 1.52 | 0.53 | 0.27 | yes / yes |

34 rows, all inside their own bars on both builds; minimum room **1.606
(WIN) / 1.605 (WSL) decades**; `'auto'` matched the route the rule names at
34/34 and matched neither at 0.  The impostor control (the separable answer
handed back as the dense one) reads a gap of **0.995 .. 1.003** at four
dense-side fixtures whose bars are 10.1 .. 16.3, i.e. the bar refuses it with
10x to 16x to spare.

---

## The census, re-run with my own plugin

`validation/probe_verify_c4/v4_census_plugin.py` wraps `_auto_selects_direct`
for a whole pytest session, changes no answer, and additionally records the
immediate CALLER of every rule call -- so "which entry point reached the rule"
is a measurement rather than a grep.

Over the five firing files (276 ids, Windows, 940.36 s), with my plugin:

| quantity | branch's reading | mine |
|---|---|---|
| calls answering `'direct'` | **35** | **35** |
| ids in which the rule fires | **16** | **16** |
| files | **5** | **5** |
| smallest ratio seen | **1/512** | **1/512** (0.001953125) |
| ratios at or under the boundary | 1/512, 1/256, 1/128, 1/64, 1/51.2, 1/32 | the same six |
| distinct shapes the rule is asked about | 83 (over 33 files) | 35 (over 5 files) |
| rule calls | 1572 (over 33 files) | 615 (over these 5) |

| file | ids where the rule fires | direct calls | shapes it fires at |
|---|---|---|---|
| `test_audit2609_a25_carrier_focus_readout.py` | 2 | 2 | `512x512 -> 16x16` |
| `test_audit2609_b4_collins_transport.py` | 2 | 5 | `1024x1024 -> 16x16` |
| `test_niche_audit_w9_dispatch2.py` | 3 | 9 | `1024 -> 8`, `2048 -> 8`, `4096 -> 8` |
| `test_niche_d2_chain_multi.py` | 7 | 17 | `1024 -> 8`, `1024 -> 16` |
| `test_niche_tight_focus_readout.py` | 2 | 2 | `2048x2048 -> 40x40` |

All **276 ids passed**.  Every one of the six shapes the rule captures is
SQUARE and work-dense (`work/entry` 264 .. 2048), so V-C4-D1 touches none of
them -- the exposure it names is real and is simply not driven by the shipped
suite.

**The extra column my plugin adds.**  The CALLER of every one of the 35
`'direct'` answers is one of exactly two places:

* `mft.py:558` `angular_spectrum_propagate_mft` (32 of the 35), and
* `carrier.py:2498` `_collins_transport` (3 of the 35).

`angular_spectrum_propagate_mft` does expose `method=`; `_collins_transport`
does not.  But the recorded caller is the frame immediately outside
`_bluestein.py`, not the caller who chose the geometry: in all five of these
files the ASM-MFT calls come from the carrier readouts and the chain, none of
which forwards a route keyword.  That is V-C4-D2 from the other end -- where
the rule actually fires in the shipped suite, the caller who picked the shape
has no way back.

---

## The WP-C3 x WP-C4 interaction

WP-C3 (`feat/c3-collins-default`) flips `transport` from `'sziklas'` to
`'collins'`.  The Collins transport calls `_bluestein_centred_2d` with no
`method=` (C3 tip `carrier.py:2507`), so on the merged tree every Collins call
is decided by WP-C4's rule.  I extracted that branch's tip with `git archive`
into the session scratchpad (never into this worktree, never edited), ran its
OWN transport code with a spy on `_bluestein_centred_2d`, and put the shapes
it was handed to THIS tree's `_auto_selects_direct`.  Identical on both
builds, as a decision that reads four integers must be:

| C3 call site | configuration | shape handed to the MFT | max ratio | rule | work/entry |
|---|---|---|---|---|---|
| `_collins_focus_readout` | `N_in=1024, N_out=16` | `1024x1024 -> 16x16` | 0.01562 | **direct** | 520 |
| `_collins_focus_readout` | `N_in=1024, N_out=32` | `1024x1024 -> 32x32` | 0.03125 | **direct** | 528 |
| `_collins_focus_readout` | `N_in=1024, N_out=64` | `1024x1024 -> 64x64` | 0.06250 | chirp | 544 |
| `_collins_focus_readout` | `N_in=512, N_out=16` | `512x512 -> 16x16` | 0.03125 | **direct** | 264 |
| `_collins_focus_readout` | `N_in=2048, N_out=40` | `2048x2048 -> 40x40` | 0.01953 | **direct** | 1044 |
| `_collins_focus_readout` | `N_in=256, N_out=128` | `256x256 -> 128x128` | 0.50000 | chirp | 192 |
| `_collins_carrier_leg` | every configuration that reaches the MFT | `N x N -> N x N` | 1.00000 | chirp | N |

**Readings.**

* The Collins TRANSPORT leg can never be captured: `_collins_carrier_leg`
  passes `N_out_x, N_out_y = env_a.shape` (C3 tip `carrier.py:2641`), so both
  ratios are exactly 1 by construction.  That is a structural statement, not a
  sampling one.
* The Collins FOCUS READOUT is captured exactly when `N_out <= N_fine/32`,
  which is **4 of the 6** configurations I drove -- and which is the workflow
  the readout exists for.  C3 therefore ENLARGES the population that reaches
  the C4 rule: chains that used to end on the Sziklas readout now end on the
  Collins one and meet the dense route by default.
* Every captured Collins shape is SQUARE (`N_out_x = N_out_y`, `dx_out =
  dy_out`) and work-dense (264 .. 1044), so **V-C4-D1 does not bite them**.
  It could only bite a chain whose own grid is strongly anisotropic, which
  nothing in the library produces today.
* **V-C4-D2 does bite them.**  `propagate_traced_carrier_chain` has no route
  keyword, and after C3 its default path reaches the rule.  `transport=
  'sziklas'` is a way back to a different PHYSICS route, not to the previous
  MFT arithmetic.  If both branches land, the `mft_method=` pass-through asked
  for in V-C4-D2 item 4 becomes more valuable, not less.

---

## Ship recommendation

**Ship after V-C4-D1 and V-C4-D2 are addressed.**  Everything the branch
claims about correctness holds under re-measurement: the selection is a pure
function of the four grid sizes (168 + 288 driven cases, two builds, no third
arithmetic anywhere), the byte-identity split is exactly the rule's (468 keys,
0 disagreements, two builds), the accuracy argument reproduces with an
independent exact reference and a two-sided bar, the BLAS sweep behaves, and
the phase-budget guard moved without losing or inventing a diagnostic.

What must not ship as it stands is the CONSTANT's justification.  The
"never slower on either build" premise is false at 6 of 12 anisotropic shapes
the rule captures, reproducibly, on both builds and under two independent
timing instruments, and by up to 9.7x; the memory argument fails at the same
shapes; and no retune of `_MFT_DIRECT_MAX_RATIO` reaches them.  The fix is one
constant and three lines (V-C4-D1); APPLIED to a tree copy and measured, it
refuses 9 of 9 shapes measured slower, keeps 11 of 11 square ladder shapes and
6 of 6 shapes the shipped suite drives, and leaves both C4 test files green.

The way-back gap (V-C4-D2) is the campaign rule, not a numerical finding, and
it is the third time this campaign has found it (VERIFY-WP-C1 D4,
VERIFY-WP-C2).  Seven public entry points is the largest count so far, and one
of them -- `compute_psf` -- is not mentioned in the Migration Guide at all.

D3 and D6 are documentation corrections; D5 is a test-coverage gap this
verification closes in its own file.  D4 is RETRACTED -- the shipped file does
catch a y/x transposition, and my first measurement of it was an instrument
error I have recorded rather than deleted.

---

## Runs

| run | build | tail |
|---|---|---|
| the 8 core MFT files (`test_c4_mft_direct_default`, `test_verify_c4_mft_direct`, `test_wave5_h2_mft_direct`, `test_verify_wave5_hyg2`, `test_verify_hyg2_round2`, `test_fix_v3_mft_centre_window`, `test_wave5_h2_near_focus_table`, `test_wave5_h2_collins_jax`) | WIN | **141 passed, 4 warnings in 70.60 s** |
| `tests/unit/test_verify_c4_mft_direct.py` alone | WIN | **17 passed in 3.64 s** (slowest id 0.91 s; every id far inside the 60 s cap) |
| both C4 test files on the PATCHED tree carrying the V-C4-D1 edit | WIN | **21 passed** and **17 passed** -- the requested edit breaks neither file |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget.py` + `test_ci_kernel_consistency.py` | WIN | **634 passed, 14 skipped, 7 warnings in 174.80 s** (28 files) |
| 12 more MFT-touching files (propagator kernels, resample call sites, dispatch, carrier verification, JAX c64 precision, odd-N grids) | WIN | **615 passed, 5 skipped, 14 warnings in 164.86 s** |
| the durations-staleness, history-lint and test-hygiene gates (4 files) | WIN | **97 passed in 96.71 s** |
| the five firing files, with my rule census | WIN | **276 passed, 4 warnings in 940.36 s** -- 615 rule calls, 35 answering `'direct'`, in 16 ids across 5 files |
| the four "pre-existing" UI ids | WIN branch | **4 failed in 9.17 s** |
| the same four, on my `git archive 49ddf4bd` | WIN base | **4 failed** (+ the two V18.5 citation ids, which fail on an archive because it is not a git repository) |
| `tests/unit/test_public_api.py` | WSL | **1 failed, 8 passed in 2.35 s** -- `test_installed_metadata_version_matches_source_version`, environmental |
| the mutation matrix (8 mutations x the whole shipped file) | WIN and WSL | identical on both builds |
| `ruff check .` | **WSL** | `All checks passed!` |
| `python -m mypy` (no args) | WIN | `Success: no issues found in 33 source files` |
| `scripts/record_history_fingerprints.py --check` | WIN | `OK: every history document matches its module.` |

### The environment fix the WSL failure needs

`test_public_api.py::test_installed_metadata_version_matches_source_version`
reads `importlib.metadata.version('lumenairy')` = **5.11.0** against a source
`__version__` of **5.48.1**.  The WSL venv's `.dist-info` is from an editable
install made at 5.11.0 and never refreshed; the source tree it points at has
moved 37 minor versions since.  Remedy, once, outside any worktree:

```bash
wsl -e bash -lc 'cd /path/to/the/canonical/lumenairy && \
                 ~/lumvenv/bin/pip install -e . --no-deps'
```

It reproduces identically on `49ddf4bd`, so it is not WP-C4's and it is not a
library defect -- but it does mean the WSL arm of `test_public_api.py` has been
reporting a stale number for every campaign that ran on this box.

---

## What I could not measure

1.  **A quiet box.**  Two other verification sessions were running heavy WSL
    and Windows pytest work throughout.  Every timing here is a BOUND, the
    load census is in each probe's JSON, and the quantity the defect rests on
    is a RATIO taken under the same conditions for all three routes -- which
    is why the thin-shape finding survives across two instruments, two builds
    and three rounds while the 1/16 reading does not.
2.  **A second Linux build or a different BLAS wheel on Linux.**  One WSL
    build, as the branch had.
3.  **CuPy on WSL** (not installed) and **a working cuFFT** (this box's DLL is
    broken).  I did not re-run the branch's CuPy contrast; its conclusion --
    that the dense route runs where the chirp-Z route cannot -- follows from
    the dense route using no FFT, which I confirmed by reading it.
4.  **The CI matrix.**  Two local builds only.
5.  **A traced carrier chain end to end on the merged C3 x C4 tree.**  The two
    branches are not merged anywhere; I drove the C3 branch's transport
    functions directly from an archive of its tip and applied this tree's
    rule to the shapes they produced, which answers "which shapes get
    captured" but not "what a full chain's answer moves by".
6.  **The `separable=True` arm off NumPy.**  `'separable'` falls back to the
    2-D chirp-Z arm on CuPy and JAX, so its TIME is a NumPy-only reading, as
    it was for the branch.

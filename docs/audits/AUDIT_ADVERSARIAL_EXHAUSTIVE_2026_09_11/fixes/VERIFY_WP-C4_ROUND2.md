# VERIFY WP-C4 ROUND 2 -- independent re-verification of `feat/c4-mft-direct-round2`

Date: 2026-09-20.  Chain: `feat/c4-mft-direct-default` (`57f71923`) ->
`verify/c4-mft-direct` (`bedcabe8`, `VERIFY_WP-C4.md`) ->
`feat/c4-mft-direct-round2` (tip `ea83d5c4`, 9 commits).  Worktree
`C:/tmp/lum_vc4b`, branch `verify/c4-mft-direct-round2`.

Both builds for every measurement: **Windows py3.14** (numpy 2.4.4, scipy
1.17.1) and **WSL py3.12** (numpy 2.4.6 on scipy-openblas, scipy 1.17.1), with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCIPY_FFT_WORKERS=1` on the command line, `fft_infra.SCIPY_FFT_WORKERS = 1`
forced in every timing probe (the three thread variables do not reach scipy's
pocketfft), and every probe pinned to ONE tree by `PYTHONPATH` with
`lumenairy.__file__` asserted under it and printed.  PRE trees are this
round's own `git archive 49ddf4bd` (pre-C4) and `git archive bedcabe8`
(pre-round-2), extracted to `C:/tmp/vc4b_trees/`.

Probes and JSON: `validation/probe_verify_c4_round2/`.  Decision tests:
`tests/unit/test_verify_c4_round2.py` (17 ids).  The harness
(`vc4blib.py`) shares no code with the branch's `probe_c4_round2/` or with
round 1's `probe_verify_c4/v4lib.py`: this is a third instrument.

---

## Verdict table

| # | claim under test | verdict | this round's numbers |
|---|---|---|---|
| 1a | the work/entry formula is the arithmetic the dense route does | **CONFIRMED, exactly** | `min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)` equals the multiply-adds implied by the operand shapes `xp.matmul` actually receives at **18 of 18** shapes; `My*Ny + Mx*Nx` equals the elements `np.exp` is actually handed at **18 of 18**; the association order taken equals the one the two costs predict at **18 of 18**, including `x_first` on every thin-x shape.  Both builds. |
| 1b | the SAFETY claim: no captured shape is slower | **CONFIRMED on an independent 35-shape ladder, both builds** | 15 captured shapes, worst dense-over-fallback **0.938 (WIN) / 0.656 (WSL)**; **captured-and-slower = [] on both builds**.  All three routes produced distinct digests at 35 of 35 shapes on both builds. |
| 1c | the DERIVATION: "16.0 is the largest value in the admissible interval, clearing the slower region by 1.34x" | **NOT REPRODUCED (V-R2-1, P3)** | my union interval is **(7.99, 8.73]**, not (11.95, 16.00].  The 11.95 reading the derivation rests on is one build, one round, and straddles unity in the branch's OWN data (WSL worst 1.110, best 0.968; WIN 0.924 = safe).  16.0 is therefore MORE conservative than my data requires -- it clears my slower region by **2.00x** -- so the constant is safe and the sentence that derives it is not. |
| 1d | the conjunction is pinned | **CONFIRMED** | second constant silently 0 -> caught by `work_screen`; first condition dropped -> caught by `boundary`; conjunction made OR -> caught by `dispatch_chirp_side`.  Identical on both builds. |
| 2 | the one-sided-screen cost is stated | **CONFIRMED in all three places** | the constant's own block ("11 of the 34 thin shapes here were safe on both builds and are refused anyway"), CHANGELOG ("11 of the 34 thin shapes here"), and `Migration-Guide.md` ("it also refuses some thin shapes that would have been fine (11 of the 34 on the round-2 ladder)"), the Guide with an escape (`mft_method='direct'`).  On MY denser ladder the cost is **16 of the 20 refused shapes**. |
| 3 | memory: 42/51 and an empty captured-and-not-cheapest set | **CONFIRMED, re-counted from the committed JSON and re-measured** | recount of `r2_memcensus_all_{win,wsl}.json`: 51 shapes, dense cheapest **42** on both builds, **captured-and-not-cheapest = [] on both**, identical readings across builds **0 of 51**, identical cheapest route **51 of 51**.  My own 35-shape census: dense cheapest 31 of 35, captured-and-not-cheapest **[] on both builds**.  The byte counts ARE derived: my independently written formulas predict the measured cheapest route at **34 of 35**. |
| 4 | the way back at 11 entry points, archive-to-archive, my own shapes | **CONFIRMED, row for row, both builds** | at `512 -> 16` captured / `512 -> 32` refused (not the branch's `256 -> 8` / `256 -> 64`): **11 of 11** move at the captured shape, **12 of 12** are identical at the refused one, **11 of 11** are reproduced EXACTLY by one `mft_method` spelling.  A twelfth entry point the branch does not drive, `propagate_carrier_referenced(transport='collins')`, does NOT move -- the report's signature argument is now a measurement. |
| 4b | the spelling table | **CONFIRMED** | `'separable'` at exactly `carrier_referenced_exact_focus_readout`, `re_reference`, `propagate_traced_carrier_chain(collins)` (and the twelfth entry point), `'bluestein'` everywhere else; identical on both builds.  At those three, `'bluestein'` does NOT reproduce the base bytes, so the per-caller spelling is load-bearing. |
| 4c | the AST census: 13 exported entry points | **CONFIRMED by a walk taken the other way round** | my FORWARD, module-qualified walk finds **13** exported entry points, the SAME thirteen.  The "22 reachable functions" is 21 callers plus the `_bluestein_2d` seed -- the two sets differ by exactly that one node and by nothing else.  Adding a route-less exported caller to a scratch tree: the shipped census names it, by name and module. |
| 5 | the refusal of `mft_method=` where no transform is reached | **CONFIRMED two-sided at the three named doors; NOT UNIFORM (V-R2-3, P3)** | accepted where a transform is reached and refused by a `ValueError` naming the keyword where it is not, at all three, on both builds.  But `resample_field(method='spline')` and `propagate_traced_carrier_chain` with no `focus_readout` accept the keyword and drop it.  No shipped caller breaks: `mft_method` appears nowhere in `lumenairy/` or `tests/` outside the round-2 work. |
| 6 | `None` stamps nothing | **CONFIRMED, spy and bytes** | the spy reproduces (first recorded call carries `method='<absent>'` at 5 of 5 driven entry points); and at a captured shape the bytes of `mft_method=None` are identical to omitting the keyword at **11 of 11** entry points on **both** builds. |
| 7 | D6: `R/4 = R_turns/2`, sitting 3.0x-8.1x below the gap | **HALF CONFIRMED (V-R2-4, P3)** | `R/4` and `R_turns/2` agree to within **3.05 %** over 12 shapes -- confirmed.  But `C_chirp/C_dense` at `N=24 -> M=12` over eleven budgets spanning ten decades reads **[0.942, 3.086]**, not the report's [1.481, 4.035]; the bar therefore sits **1.88x to 6.17x** below the implied gap, not 3.0x to 8.1x.  The bar itself is not violated (0 of 12) and the impostor reads 0.996-1.005 at 12 of 12 -- but the measured clearance falls to **1.30x** once the non-exact population is widened past the shipped fixtures. |
| 8 | N5: two populations | **CONFIRMED qualitatively, numbers wider** | exactness measured (float64 product vs exact `Fraction`), agreeing with the odd-part-of `N_max^2` criterion at 12 of 12.  Six rows each.  Rounded population **12.2 .. 50.8 (WIN) / 12.3 .. 48.8 (WSL)**; exact population **85.4 .. 854.4 / 106.8 .. 671.2**.  The report's "~30x (28.2 .. 34.3)" for the rounded population is a reading of three shapes, not a law. |
| 9 | the census after the fix: 615 / 35 / 16 / 5 | **CONFIRMED, exactly, with my own plugin** | 276 passed in 1050.06 s; **615** rule calls, **35** answering direct, **16** ids, **5** files, 35 distinct shapes, smallest ratio **1/512**, the same six firing shapes, the same per-file split 2/2/3/7/2. |
| 10 | the carrier-only commits | **CONFIRMED** | `4912e229` touches `lumenairy/propagators/carrier.py` only (61+/13-); `023b717c` touches that file (7+/5-) and `docs/history/carrier.md` (3+/2-) only.  Every hunk is keyword threading or comment wording.  Full statement below. |
| 11 | round-2 neutrality over round 1's suite | **CONFIRMED, both builds** | **24 of 24** keys byte-identical `bedcabe8` vs the tip, on Windows and on WSL. |
| 12 | `.test_durations`, staleness gate, no forward version token | **CONFIRMED** | 61 ids added / 38 removed for the four C4 files at the tip; JSON valid, 16,657 ids.  No `5.49.x` / `5.50.x` / `5.51.x` token anywhere in `lumenairy/`; the staleness and relocation gates pass (756 passed). |
| 13 | the mutation matrix, plus four of mine | **8 of 9 caught, identically on both builds** | the shipped five each caught by the claim named beside it.  Of my four: `mft_method` ignored at one entry point -> caught; `ratio_condition_dropped` -> caught; `work_formula_max_instead_of_min` -> caught; **`work_comparison_strict` (`>` for `>=`) -> caught by NOTHING (V-R2-2, P3)**. |

**Ship recommendation for the whole C4 chain: SHIP.**  Every safety claim the
default flip rests on reproduced on an independent instrument on both builds:
the rule captures no shape measured slower, the dense route is the cheapest of
the three at every shape it captures, every moved entry point has a
one-keyword way back that reproduces the previous bytes exactly, and round 2
moves nothing round 1 measured.  The five defects below are all P3 -- three
wording, one uncovered mutant, one contract asymmetry -- and none of them
changes a byte or weakens the rule.  They should be fixed in the same branch
before the tag, because four of them are statements a future maintainer would
re-derive from.

---

## Defects

### V-R2-1 (P3) -- the derivation of `16.0` is instrument-dependent, in the same way `VERIFY_WP-C4` N1 recorded for `1/16`

**The claim.**  `_bluestein.py`'s constant block, the CHANGELOG and the report
all say: the largest work/entry measured SLOWER on either build is **11.95**
(`2048x128 -> 64x4`), the smallest measured safe above it is **16.00**, no
shape lies strictly between, and therefore "16.0 is the LARGEST value in the
admissible interval... it clears the slower region by **1.34x**".

**What I measure.**  An independent ladder of 35 shapes -- 26 of them with
work/entry in [6, 30], in MATCHED PAIRS of both orientations at three absolute
sizes, because in that band the branch's ladder carries 14 shapes of which 13
are the same orientation -- two rounds of best-of-nine, routes interleaved
with the order rotating per repeat, cold before every single timed call, all
three routes digested in the same pass, verdict on the worst round:

| | shapes measured SLOWER | largest work/entry among them | smallest safe above it |
|---|---|---|---|
| Windows py3.14 | 4 of 35 (1.22 .. 5.89) | **7.99** | **8.73** |
| WSL py3.12 | 4 of 35 (1.08 .. 7.92) | **7.99** | **8.73** |
| union | 4 of 35 | **7.99** | **8.73** |

The four are the same four on both builds: `4096x32 -> 128x1` and
`32x4096 -> 1x128` (work/entry 1.25) and `4096x128 -> 128x4` /
`128x4096 -> 4x128` (7.99).  The 7.99 pair reproduces the branch's Windows
finding (1.659 there, 1.590/1.220 here) and extends it to WSL, which the
branch's ladder did not see in that orientation.

**The 11.95 reading does not reproduce, and it is thin in the branch's own
data.**  `2048x128 -> 64x4` reads, on the branch's instrument, WIN worst 0.924
(safe) and WSL worst **1.110** / best **0.968** -- the two rounds straddle
unity by 14 %.  On mine it reads 0.952 (WIN) / 0.969 (WSL), safe in every
round, and its mirror `128x2048 -> 4x64` reads 0.624 / 0.894.  So the number
the constant's admissible interval is derived from is one build, one round,
and within noise of 1.0.  That is the S1 shape `docs/TESTING_STANDARDS.md`
names.

**Why this is P3 and not P1.**  It moves the derivation, not the safety.  On
my data 16.0 refuses everything measured slower with **2.00x** of margin
instead of 1.34x; what is false is only "16.0 is the LARGEST admissible value"
-- on my ladder that value is 8.73, and 16.0 refuses seven further work/entry
values (8.73, 8.96, 9.85, 9.92, 11.95, 11.98, 15.11) that were safe on both
builds.  The constant errs in the conservative direction, which is the
direction the rule's premise wants.

**Requested edit**, in `lumenairy/propagators/_bluestein.py`, inside the
`_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY` block, replacing the sentence
"16.0 is therefore the LARGEST value that still captures every shape measured
safe above the slower region, and it clears the slower region by **1.34x**.":

```
#: 16.0 is the largest value this ladder's two boundary readings admit, and it
#: clears the slower region by 1.34x on them.  BOTH of those boundary readings
#: are instrument-dependent and only one of them is thin: ``2048x128 -> 64x4``
#: (11.95) is SAFE on Windows here (0.924) and straddles unity on WSL (worst
#: 1.110, best 0.968), and an independent re-measurement
#: (``VERIFY_WP-C4_ROUND2.md``, 35 shapes, both orientations matched, two
#: builds) reads it safe in every round on both builds and puts the largest
#: work/entry measured slower at 7.99 instead -- which would make the
#: admissible interval (7.99, 8.73] and 16.0 a value with 2.00x of margin
#: rather than the tightest one.  What BOTH instruments agree on is the only
#: thing the rule needs: no shape at or above 16.0 was measured slower on
#: either build, on either ladder.  Read "the largest admissible value" as
#: this ladder's reading and not as a settled fact.
```

and the matching sentence in the CHANGELOG (line 86, "16 is the LARGEST value
that still captures every shape measured safe above the slower region, and it
clears that region by **1.34x**.") and in `Migration-Guide.md` (line ~2009,
"the largest value that still captures every anisotropic shape measured safe
above the slower region, and it clears that region by 1.34x (the largest work
ratio measured slower on either build is 11.95)"), each gaining "on the
round-2 ladder; an independent ladder reads that boundary at 7.99 and 16.0
therefore with more margin, not less".

**Reproducer:**
```
PYTHONPATH=<tree> python validation/probe_verify_c4_round2/vc4b_ladder.py \
    <tree> --rounds 2 --reps 9
```
JSON: `vc4b_ladder_{win,wsl}.json`, keys `SLOWER`, `CAPTURED_AND_SLOWER`,
`largest_work_per_entry_measured_slower`.

---

### V-R2-2 (P3) -- the work condition's inclusive comparison is not gated

**The claim.**  `_auto_selects_direct`'s Notes: "The comparison is ``<=`` and
the constant is the largest ratio MEASURED safe, rather than ``<`` against the
first ratio measured unsafe, so the boundary ratio itself is inside the dense
region and the constant names a shape that was actually timed."  For the FIRST
condition that is gated -- `_claim_the_boundary_comes_from_the_constant` puts
a square shape at exactly the boundary ratio on the dense side.  For the
SECOND condition nothing does.

**Measured.**  A mutant that changes `flops >= w * entries` to
`flops > w * entries` passed **all eight** shipped claims of
`test_c4_mft_direct_default.py` and `test_c4_round2_mft_method.py`, on both
builds.  It is the only one of nine mutations that nothing refused.  Its
effect is not hypothetical: the constant is set to exactly the work ratio of
`256x64 -> 4x1`, which reads **16.00**, so a strict comparison refuses the one
shape the whole derivation names as the smallest measured safe.

**Closed by** `tests/unit/test_verify_c4_round2.py::
test_the_work_screens_comparison_is_inclusive_at_the_constant`, which states
the claim about the CONSTANT and not about one shape: set the constant to a
scanned thin shape's own work ratio and that shape must be inside; move the
constant one ULP up and it must be outside.

**Requested edit** (optional, the id above already closes it): add
`work_comparison_strict` to `_MUTATIONS` in
`tests/unit/test_c4_mft_direct_default.py`, named for the new id.

**Reproducer:**
```
PYTHONPATH=<tree> python validation/probe_verify_c4_round2/vc4b_mutations.py <tree>
```
-> `UNCAUGHT: ['work_comparison_strict']`, identical on both builds.

---

### V-R2-3 (P3) -- the `mft_method=` contract refuses at three doors and swallows at two

**Measured**, both builds
(`vc4b_refusal_{win,wsl}.json`, identical row for row):

| call | reaches a transform? | outcome |
|---|---|---|
| `compute_psf(method='mft', mft_method=...)` | yes | accepted |
| `compute_psf(method='fft', mft_method=...)` | no | **ValueError naming the keyword** |
| `propagate(method='asm', output_grid=..., mft_method=...)` | yes | accepted |
| `propagate(method='asm', no output grid, mft_method=...)` | no | **ValueError naming the keyword** |
| `propagate_carrier_referenced(transport='collins', mft_method=...)` | yes | accepted |
| `propagate_carrier_referenced(transport='sziklas', mft_method=...)` | no | **ValueError naming the keyword** |
| `resample_field(method='chirpz', mft_method=...)` | yes | accepted |
| `resample_field(method='spline', mft_method=...)` | no | **accepted and ignored** |
| `propagate_traced_carrier_chain(focus_readout=..., mft_method=...)` | yes | accepted |
| `propagate_traced_carrier_chain(no focus_readout, mft_method=...)` | no | **accepted and ignored** |

The contract is RIGHT -- the stated reason ("accepting it there would silently
ignore a caller who was asking for the pre-shape-rule bytes") is exactly the
campaign rule -- and it is two-sided at each of the three doors that have it.
It is simply not applied at two further doors, one of which
(`resample_field`) documents the silent ignore in its own docstring and one of
which (`propagate_traced_carrier_chain`) does not document it anywhere.  No
harm results today, because at those two states no MFT is reached and the old
bytes ARE the current bytes; what is lost is the diagnostic.

A second, smaller imprecision: `propagate(method='rs', output_grid=...,
mft_method=...)` is admitted by the `mft_method` screen (`'rs'` is in
`_BARE_GRID_METHODS`) and then refused two lines later by the `rs`/output-grid
`ValueError`, so the caller never sees the keyword message.  Harmless, since
the call fails either way.

**Requested edit**, in `lumenairy/propagators/mft.py`, in `resample_field`
after the `method not in ('spline', 'chirpz')` check:

```python
    # WP-C4 round 2 (V-C4-D2), completed VERIFY-WP-C4 round 2 (V-R2-3): the
    # same refusal ``compute_psf``, ``propagate`` and
    # ``propagate_carrier_referenced`` carry.  The spline leg reaches no
    # matrix Fourier transform, so accepting the keyword there would silently
    # drop a caller asking for the pre-shape-rule bytes.
    if mft_method is not None and method != 'chirpz':
        raise ValueError(
            f"resample_field: mft_method= is only meaningful with "
            f"method='chirpz' (got method={method!r}).  It names the route "
            f"through the matrix Fourier transform ('auto' / 'bluestein' / "
            f"'separable' / 'direct'); the spline resampler interpolates with "
            f"map_coordinates and reaches no such transform.")
```

with the docstring's "Ignored by ``method='spline'``, which reaches no
transform." replaced by "A ``ValueError`` on ``method='spline'``, which
reaches no transform.", and the same refusal added to
`propagate_traced_carrier_chain` / `..._multi` when no `focus_readout` /
`output_grid` is requested.  **This is a new `ValueError` on signatures that
shipped accepting the keyword within this same branch**, so it belongs in the
5.49.0 Migration note rather than in a later release; if the maintainer would
rather keep the accept, the edit is instead to document the ignore in the two
chain docstrings and to say in the Guide that the keyword is inert wherever
no readout is requested.  `tests/unit/test_verify_c4_round2.py::
test_the_entry_points_that_accept_and_ignore_the_keyword_are_the_known_two`
pins whichever choice is made and says which way it moved.

---

### V-R2-4 (P3) -- D6's `C_chirp/C_dense` envelope does not reproduce, and the bar's headroom is thinner than stated

**The claim.**  "`R/4` is `R_turns/2`, and it sits **3.0x to 8.1x** below the
gap the measured `C_chirp/C_dense` in [1.481, 4.035] implies."

**Measured**, against an oracle written from scratch -- the phase reduced
exactly as a `Fraction`, `exp` and both matrix products accumulated in
`mpmath` at 40 digits, so the oracle's floor is ~1e-40 and the dense route's
error is RESOLVED rather than compared against another float64 sum:

* `R/4` and `R_turns/2` agree to within **3.05 %** over 12 shapes (the
  `(N/(N-1))^2` correction the docstring names, and below 2 wherever
  `next_fast_len` pads further, i.e. in the conservative direction).
  **CONFIRMED.**
* `C_chirp/C_dense` at `N = 24 -> M = 12` over eleven budgets from 1e5 to
  1e15: **[0.942, 3.086]** on WIN and on WSL, against the report's
  [1.481, 4.035].  The bar therefore sits **1.88x to 6.17x** below the implied
  gap, not 3.0x to 8.1x.
* The bar is two-sided and holds: **0 bar violations** in 12 shapes on both
  builds, and the impostor (the separable chirp-Z arm answering in the dense
  arm's place) reads **0.996 .. 1.005** at 12 of 12 -- 8.6x to 52.6x under the
  bar.
* But the ABOVE side is thinner than the report's "smallest 4.2x": on a
  non-exact population widened past the shipped fixtures the clearance falls
  to **1.30x (WIN) / 1.31x (WSL)** at `224 -> 7`.  The SHIPPED claim is not at
  risk -- its thinnest shape is `96 -> 3` at 4.19x / 4.03x -- but the margin
  the derivation states is the margin of four fixtures, not of the bar.

A related observation the report should carry: at the exact-`alpha` shapes the
dense route's error falls to about **2e-14**, which is the same order as a
float64 `math.fsum` reference's own floor.  The report's "202 .. 397" for that
population is therefore bounded by its reference, not by the route: the
mpmath oracle reads **85 .. 854** at the same fixtures (`64 -> 2` reads 854
here against the report's 233).

**Requested edit**, in `tests/unit/test_c4_mft_direct_default.py`'s
`_claim_the_dense_side_is_the_more_accurate_side` docstring, replacing
"with ``C_chirp/C_dense`` MEASURED in [1.481, 4.035] over ten decades of budget
at the shipped N=24 -> M=12 geometry (hygiene-2 round 3, reproduced 2026-09-20
on both builds).  The bar asserted is ``R / 4``, which IS ``R_turns / 2`` -- so
it sits 3.0x to 8.1x below the implied gap":

```
    with ``C_chirp/C_dense`` MEASURED in [0.94, 3.09] over eleven budgets
    spanning ten decades at the shipped N=24 -> M=12 geometry, against an
    mpmath-40-digit reference on both builds (VERIFY-WP-C4 round 2; an
    earlier wording read [1.481, 4.035] from hygiene-2 round 3 and has not
    reproduced).  The bar asserted is ``R / 4``, which IS ``R_turns / 2`` to
    within 3.1 % -- so it sits 1.9x to 6.2x below the implied gap.  That
    factor is a chosen margin and not a derivation.  The SHAPES here clear it
    by 4.2x at worst; a wider non-exact ladder (224 -> 7) clears it by only
    1.30x, so the margin belongs to these fixtures and not to the bar.
```

and the matching paragraph in the report's section 4.2 and in the round-2
addendum's D6 row.

**Reproducer:**
```
PYTHONPATH=<tree> python validation/probe_verify_c4_round2/vc4b_accuracy.py <tree>
```
JSON: `vc4b_accuracy_{win,wsl}.json`.

---

### V-R2-5 (P3) -- two stale documentation references

1.  **A gate reference that names no test.**  `lumenairy/propagators/
    _bluestein.py:1268`, in `_mft_route_kwargs`, says "Gated by
    ``tests/unit/test_c4_round2_mft_method.py::
    test_mft_method_none_stamps_nothing_on_the_primitive``".  No such id
    exists; the id is `test_mft_method_none_stamps_nothing_on_the_call_it_
    makes`.  **Requested edit:** change the name to the real one.
2.  **Six of nine signatures carrying `mft_method=` do not document it.**
    `propagate`, `propagate_carrier_referenced`,
    `propagate_traced_carrier_chain`, `propagate_traced_carrier_chain_multi`,
    `carrier_referenced_focus_readout` and
    `carrier_referenced_exact_focus_readout` accept the keyword and say
    nothing about it in `help()`; only `compute_psf`, `resample_field` and
    `re_reference` carry a `Parameters` entry.  Since `propagate` is the
    library's main public door and the Guide's table is the only place the
    per-caller spelling is written down, this is the difference between a
    discoverable way back and a documented one.  **Requested edit:** add to
    each of the six, in its `Parameters` block:

    ```
    mft_method : {None, 'auto', 'bluestein', 'separable', 'direct'}, optional
        Which route through the matrix Fourier transform this call's readout
        takes, forwarded to the primitive as its ``method=``.  ``None`` (the
        default) names nothing -- the keyword is left off, so the library's
        own default governs.  The one-call way back to the dispatch this
        entry point had before the MFT shape rule is ``'<spelling>'`` (see
        the 5.49.0 section of ``Migration-Guide.md``).
    ```

    with `<spelling>` = `'separable'` for
    `carrier_referenced_exact_focus_readout`, `'bluestein'` elsewhere, as
    measured.  `propagate_carrier_referenced`'s entry should instead say what
    was measured there: the rule cannot fire on its Collins leg (the output
    lattice keeps the input's `N`), so no spelling is needed today and the
    keyword exists so that stays true by construction.

**Also noted, no edit requested:** `test_c4_round2_memory_claim.py::
test_the_prose_does_not_carry_the_refuted_cross_build_sentence` cannot fail.
It refuses the literal `'readings are identical to the byte across builds'`,
which is absent from the report and the CHANGELOG at `57f71923` (before the
correction), at `bedcabe8` and at the tip -- the sentence it was written
against reads "the two builds' readings are IDENTICAL TO THE BYTE at every
shape" and wrapped across a line break.  Its companion assertion,
`'ordering' in text.lower()`, was already satisfied at `57f71923`.  A guard
with no fail-before is closed here by
`tests/unit/test_verify_c4_round2.py::
test_the_cross_build_memory_guard_would_have_refused_the_old_sentence`, whose
predicate is whitespace-insensitive, skips occurrences that a correction
marker introduces, and is exercised against the real sentence as a literal
before it is applied to the current documents.  Measured: the new predicate
fires on the report AND the CHANGELOG at `57f71923` and at `bedcabe8`, and
does not fire at the tip.

---

## The boundary on this round's data

Ladder: 35 shapes -- 26 with work/entry in [6, 30] in matched orientation
pairs at three absolute sizes, four anchors below the band, three square
controls above it.  Two rounds of best-of-nine, interleaved, rotating order,
cold before each timed call, worst round, against `min(chirp-Z 2-D,
separable)`, `SCIPY_FFT_WORKERS = 1`.

| work/entry | shape (T = thin x, W = thin y) | WIN worst d/fb | WSL worst d/fb | rule |
|---|---|---|---|---|
| 1.25 | `4096x32 -> 128x1` / `32x4096 -> 1x128` | 5.887 / 4.090 | 7.924 / 7.115 | refused |
| 2.99 | `512x32 -> 16x1` | 0.633 | 0.609 | refused |
| 3.00 | `64x2048 -> 1x32` | 0.771 | 0.664 | refused |
| **7.99** | `4096x128 -> 128x4` / `128x4096 -> 4x128` | **1.590** / **1.220** | **1.084** / **1.265** | refused |
| 8.73 | `256x64 -> 8x1` / `64x256 -> 1x8` | 0.451 / 0.399 | 0.397 / 0.383 | refused |
| 8.96 | `2048x128 -> 16x1` / `128x2048 -> 1x16` | 0.241 / 0.142 | 0.302 / 0.219 | refused |
| 9.85 | `512x64 -> 16x2` / `64x512 -> 2x16` | 0.465 / 0.601 | 0.440 / 0.478 | refused |
| 9.92 | `1024x64 -> 16x2` / `64x1024 -> 2x16` | 0.409 / 0.307 | 0.498 / 0.432 | refused |
| 11.95 | `2048x128 -> 64x4` / `128x2048 -> 4x64` | 0.952 / 0.624 | 0.969 / 0.894 | refused |
| 11.98 | `4096x256 -> 128x4` / `256x4096 -> 4x128` | 0.947 / 0.570 | 0.801 / 0.645 | refused |
| 15.11 | `128x64 -> 4x1` / `64x128 -> 1x4` | 0.373 / 0.348 | 0.396 / 0.307 | refused |
| **16.00** | `256x64 -> 4x1` / `64x256 -> 1x4` | 0.275 / 0.321 | 0.356 / 0.414 | **captured** |
| 16.87 | `2048x256 -> 16x1` / `256x2048 -> 1x16` | 0.139 / 0.104 | 0.172 / 0.106 | captured |
| 17.93 | `4096x1024 -> 128x2` / `1024x4096 -> 2x128` | 0.204 / 0.166 | 0.059 / 0.058 | captured |
| 19.69 | `1024x128 -> 32x4` / `128x1024 -> 4x32` | 0.519 / 0.342 | 0.608 / 0.516 | captured |
| 23.91 | `4096x256 -> 128x8` / `256x4096 -> 8x128` | **0.938** / 0.565 | 0.587 / **0.656** | captured |
| 29.33 | `256x128 -> 4x1` / `128x256 -> 1x4` | 0.235 / 0.263 | 0.307 / 0.275 | captured |
| 264 / 528 / 1056 | the three square controls | 0.179 / 0.238 / 0.362 | 0.360 / 0.363 / 0.150 | captured |

**The boundary on this data: the slower region ends at 7.99 and the safe
region begins at 8.73, on BOTH builds.**  16.0 refuses everything measured
slower with 2.00x of margin and captures nothing measured slower on either
build.  The cost is 16 of the 20 refused shapes, which were safe on both.
Both orientations agree in SIGN at 35 of 35 shapes on both builds, so the
branch ladder's one-orientation coverage in this band hid nothing.

Load: Windows 26 python processes in both rounds, reference-loop drift
0.33 .. 4.30; WSL 18 -> 16, drift 0.26 .. 2.56.  Every reading is a BOUND; the
decision quantity is a ratio taken under the same conditions for all three
routes, and all three routes produced distinct digests at 35 of 35 shapes on
both builds.

---

## The carrier hunk statement, for the `feat/c3-collins-default` merge

`feat/c3-collins-default` rewrites `lumenairy/propagators/carrier.py`'s
readout code.  The two carrier-only commits on this branch are the ones that
have to merge into it cleanly.  Confirmed by `git show --stat`:

* **`4912e229`** -- `lumenairy/propagators/carrier.py` ONLY, **61 insertions,
  13 deletions**, in 23 hunks.
* **`023b717c`** -- `lumenairy/propagators/carrier.py` (**7 insertions, 5
  deletions**, 3 hunks) and `docs/history/carrier.md` (**3 insertions, 2
  deletions**, the AST / token fingerprint re-record CONTRIBUTING.md requires).

**Every hunk is keyword threading or comment wording.  No readout arithmetic,
no default, no control flow other than one new refusal, is touched.**  The
hunks of `4912e229`, in file order:

| hunk | what it does |
|---|---|
| `@@ -1157` | +7: a comment and `mft_method: Optional[str] = None,` appended to `propagate_carrier_referenced`'s positional-or-keyword list -- appended LAST, so no existing positional binding moves |
| `@@ -1354` | +11: the new `ValueError` refusing `mft_method=` on `transport != 'collins'` |
| `@@ -1379` | 1 line: `fn='propagate_carrier_referenced'` -> `..., mft_method=mft_method)` |
| `@@ -2217` | 1 line: `_collins_transport(...)` signature gains `mft_method=None` |
| `@@ -2302` | 1 line: the local import gains `_mft_route_kwargs` |
| `@@ -2501` | +3: `**_mft_route_kwargs(mft_method))` appended to the `_bluestein_centred_2d` call's kwargs |
| `@@ -2535` | 1 line: `_collins_carrier_leg(...)` signature gains `mft_method=None` |
| `@@ -2688` | 1 line: `mft_method=mft_method` forwarded to `_collins_transport` |
| `@@ -2708` | 1 line: `_collins_focus_readout(...)` signature gains `mft_method=None` |
| `@@ -2768` | 1 line: `mft_method=mft_method` forwarded |
| `@@ -4149` | +1: `mft_method` added to `carrier_referenced_focus_readout`'s KEYWORD-ONLY list |
| `@@ -4448` | +1: local `from ._bluestein import _mft_route_kwargs` |
| `@@ -4478` | +3: `**_mft_route_kwargs(mft_method))` on the readout's call |
| `@@ -6632` | +1: `mft_method` added to `carrier_referenced_exact_focus_readout`'s keyword-only list |
| `@@ -7160` | +1: local import |
| `@@ -7197` | +4: `**_mft_route_kwargs(mft_method))` on the exact readout's call |
| `@@ -9416` | +1: `mft_method` added to `propagate_traced_carrier_chain`'s keyword-only list |
| `@@ -10538` | 1 line: `diag=_collins_diag, mft_method=mft_method)` |
| `@@ -10745` | +2: `exact_kw['mft_method'] = mft_method` |
| `@@ -11020` | +4: `_par_kw['mft_method'] = mft_method` |
| `@@ -11115` | 1 line: `fn=_fn, mft_method=mft_method)` |
| `@@ -12122` | +1: `mft_method` added to `propagate_traced_carrier_chain_multi`'s keyword-only list |
| `@@ -12702` | +3: `mft_method=mft_method` in the invariant chain kwargs |

and the three hunks of `023b717c` are pure comment wording ("pre-5.49.0" ->
"before the shape rule") at `@@ -1364`, `@@ -2521` and `@@ -4506`.

**Where the two branches can collide.**  Four hunks land inside code C3
rewrites: `@@ -2501` and `@@ -2521` (the `_bluestein_centred_2d` call inside
`_collins_transport`), `@@ -7197` (the exact readout's primitive call) and
`@@ -10538`/`@@ -11115` (the chain's Collins legs).  All four are ADDITIVE
kwargs on an existing call, so a textual conflict there resolves by keeping
C3's call and re-appending `**_mft_route_kwargs(mft_method)` /
`mft_method=mft_method`.  Nothing in these commits changes what those calls
compute: the whole of `4912e229` is verified by driving both transports
archive-to-archive on this tree, where the readout bytes with `mft_method=None`
are byte-identical to the keyword being absent at 11 of 11 entry points on
both builds.

**Signature-compatibility note for the merge:** `mft_method` is
POSITIONAL-OR-KEYWORD in `propagate_carrier_referenced` (appended last, after
`on_collins_sampling`) and KEYWORD-ONLY in the other four.  If C3 inserts
parameters into `propagate_carrier_referenced`'s positional list it must stay
BEFORE `mft_method`, or a 13th positional argument changes meaning.

---

## Runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCIPY_FFT_WORKERS=1` on the command line and
`pytest -q --capture=sys -p no:randomly`.

| run | build | tail |
|---|---|---|
| the 8 core MFT files + all four C4 test files + `test_verify_c4_round2.py` (11 files) | WIN | **181 passed, 4 warnings in 101.70 s** |
| the same, plus `test_public_api.py` (12 files) | WSL | **1 failed, 189 passed, 4 warnings in 77.64 s** -- the failure is `test_installed_metadata_version_matches_source_version`, ENVIRONMENTAL: re-run on my own `git archive 49ddf4bd` tree it fails identically (`5.11.0` from the WSL venv's editable `.dist-info` against a source `5.48.1`) |
| `tests/unit/test_verify_c4_round2.py` alone, serial, `--durations=0 --durations-min=0 -vv` | WIN | **17 passed in 24.47 s** -- slowest id 22.42 s, every id far inside the 60 s cap |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep (31 files) | WIN | **1402 passed, 14 skipped, 7 warnings in 305.65 s** |
| `test_audit_except_budget.py` + `test_ci_kernel_consistency.py` | WIN | **11 passed in 3.06 s** |
| 12 MFT-touching files (propagator kernels, resample call sites, dispatch, analysis, JAX c64 precision, odd-N grids, tilted leg) | WIN | **969 passed, 5 skipped, 56 warnings in 416.19 s** |
| the 4 carrier readout files (`test_carrier_referenced`, `a25_carrier_focus_readout`, `test_carrier_field`, `a6_verify_carrier`) | WIN | **163 passed, 3 warnings in 153.38 s** |
| the durations-staleness and history-relocation gates | WIN | **756 passed in 144.97 s** |
| the five firing files, with MY rule census plugin | WIN | **276 passed, 4 warnings in 1050.06 s** -- 615 rule calls, 35 answering `'direct'`, 16 ids, 5 files |
| the shipped census test on a scratch tree carrying a route-less exported caller | WIN | **1 failed** -- and it names `('route_less_new_entry_point', ['propagators/mft.py'])` and nothing else |
| `ruff check .` | **WSL** | `All checks passed!` |
| `python -m mypy` (no args) | WIN | `Success: no issues found in 33 source files` |
| `scripts/record_history_fingerprints.py --check` | WIN | `OK: every history document matches its module.` |

`.test_durations`: the 17 new ids measured SERIALLY (one process, no `-n`) and
spliced; **16,674 ids total**, JSON re-parsed after the write, staleness gate
green.

---

## What this round could not measure

1.  **A quiet box.**  Four sibling verification sessions ran heavy Windows and
    WSL pytest work throughout.  Every timing here is a BOUND; the process
    census and the per-shape reference-loop drift are in each probe's JSON,
    and the decision quantity is a ratio taken under the same conditions for
    all three routes.  That is why the SIGN of the 7.99 finding survives two
    builds and two rounds while individual readings move by a factor of two.
2.  **A second Linux build, or a different BLAS wheel on Linux.**  One WSL
    build, as all three rounds have had.  V-R2-1 is a direct consequence: the
    one reading that separates the branch's admissible interval from mine is
    a single WSL round.
3.  **CuPy or JAX.**  The rule is integer arithmetic over four grid sizes and
    is identical on every backend by construction, and the branch measured the
    SELECTION matching NumPy's on both; what is unmeasured is where the thin
    crossover sits on a GPU.  The screen is conservative there either way --
    it refuses, it never captures more.
4.  **The CI matrix.**  Two local builds only.
5.  **A merged C3 x C4 tree.**  The branches are not merged anywhere.  The
    hunk statement above is a static analysis of the two carrier commits plus
    an archive-to-archive drive of both transports on THIS tree; it does not
    say what a merged chain's answer moves by.
6.  **The WSL arm of the rule census.**  Not run.  Redundant for the same
    reason round 2 gives: the rule is four-integer arithmetic, and the same 35
    shapes put to `_auto_selects_direct` on both builds give the same 15
    captured and 20 refused.
7.  **Whether `_EXACT_READOUT_SEPARABLE_BLUESTEIN` is the ONLY reason the
    three `'separable'` entry points read that spelling.**  The correlation is
    exact on both builds and the flag is named at each of those call sites,
    but nothing here drives a tree with the flag flipped.

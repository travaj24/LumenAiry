# The ninth name: the union-grid conical staircase answered differently at two BLAS threads -- 2026-08-06

`FIX_CI_M1_T34_2026_08_06.md` S9/S11 referred a library-level silent-wrong on
the PMM union-grid path:

```text
threads   J00                                    |R+T-1|
1         -0.17117932369944 +0.00906676381430j   6.654e-06
2         -0.27216039162719 -0.09244618541393j   2.135e+01
24        -0.17117932109206 +0.00906676337991j   6.651e-06
```

-- a 1.4e-01 absolute move in a Jones entry at ONE pool size, on both mounts,
at every truncation, with the per-layer path unaffected.  A tenth name
(`test_pmm_m2_window_contract.py::test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too`,
`spread(broken)` reading `3.615e-03` at two threads) was referred alongside it
as "likely the same mechanism".

Both are the same defect.  It is a CLASSIFICATION FLIP, not a null-space draw,
and it is the T3-4 mechanism the campaign already named -- on a new axis, and
with a much larger blast radius than the referral supposed.  It is fixed.

---

## S1.  Reproduction

`tests/unit/test_m1_conditioning_guard.py`'s staircase, verbatim: six lossless
60 nm slices whose walls shift 4 nm per slice, period 1 um, 700 nm, degree 6,
`theta` = 0.15, `phi` = 0.6, `layer_grids="shared"`.  Reproduced on the first
attempt, Windows native:

```text
threads  grid       ffo 7                                 |R+T-1|
1        shared     -0.17117932369944+0.00906676381430j   6.6544e-06
2        shared     -0.27216039162719-0.09244618541393j   2.1348e+01
default  shared     -0.17117932109206+0.00906676337991j   6.6507e-06
1/2/def  per-layer  -0.171147169...  +0.009125110...j     1.0526e-04
```

Identical at `far_field_orders` 7, 11, 21 and 41 -- so the truncation is not
the variable, and neither is the far-field projection.

---

## S2.  WHERE it forks: the modal classification, not the least squares

Two candidate sites, and the referral named both.  The union path's
`_guarded_lstsq(Hsup, rhs)` is genuinely underdetermined -- 2 x `n_glob` = 156
unknowns against 2 x `Nf` = 30 equations at `ffo` = 7 -- so a null-space draw
(the C13 / X-1 class) was the plausible suspect.  It is not the site.

Arming `_MODE_CUT_CENSUS` over the whole solve and comparing the two pools
(rows 0/1 are the superstrate/substrate half-spaces, 2..7 the six patterned
slices):

```text
row  site           1 thread                          2 threads
                    n_prop n_risk margin  n_grow      n_prop n_risk margin  n_grow
0    sup half-sp     6      0      52.5    0           6      0     404.2    0
1    sub half-sp     9      0      78.6    0           9      1       2.21   1
2    patterned       8      1       6.34   0          10      2       1.11   2
3    patterned       9      2       1.28   1          10      2       1.20   2
4    patterned       8      2       1.01   0           8      0      21.8    0
5    patterned      10      0      23.3    0           9      4       1.47   1
6    patterned      10      0      30.8    0           9      4       1.15   1
7    patterned       8      2       5.69   0           9      2       1.58   1
                                          ---                              ---
                            total n_growing  1                                8
```

The T3-4 growth instrument (channel A, `_mode_cut_growth`) separates the two
pools cleanly: 1 growing forward mode at one thread, 8 at two.  The
eigenvalues themselves do not move -- the per-mode `q` agree to ~1e-10 in
every cell.  Only WHICH modes are called propagating, and which way they are
then pointed, moves.

### S2.1  The exact mode, and why its flux is meaningless

Dumping `(flux, q, thr, prop)` at the classification site for row 1 (the
substrate half-space, `n_glob` = 78, `thr` ~ 3.7e-17), sorted by distance from
the cut:

```text
1 thread
  m=147  |flux|/thr = 1.86567    prop=1  q = +1.85e-10 +2.786957e-01j  |Im q|/|q| = 1.00
  m=146  |flux|/thr = 0.09560    prop=0  q = +9.50e-12 +2.786957e-01j  |Im q|/|q| = 1.00

2 threads
  m=152  |flux|/thr = 2.21108    prop=1  q = +2.19e-10 -2.786957e-01j  |Im q|/|q| = 1.00
  m=153  |flux|/thr = 0.06326    prop=0  q = +6.28e-12 -2.786957e-01j  |Im q|/|q| = 1.00
```

Read it as a chain:

1. The union grid carries **exactly double roots** `q^2`.  A uniform half-space
   discretised on a 78-node nodal grid has each transverse mode twice (the two
   tangential polarizations), and `Mbig` is not symmetric, so LAPACK splits
   each double root into a pair whose imaginary parts are ~1e-10 and of
   OPPOSITE sign.
2. `q = sqrt(q2)` then lands one member of every such pair at `Im(q) > 0` and
   the other at `Im(q) < 0`.  Both are the SAME physical mode: `|Im q| / |q|`
   is exactly 1.0, i.e. `q` is purely imaginary -- **evanescent**.
3. An evanescent mode carries **exactly zero z-power**.  So `flux` for both
   members is round-off, and `_mass_flux_threshold` -- correctly -- sits at the
   round-off floor, 3.7e-17.  One member's round-off happens to be 1.9x-2.2x
   the floor and the other's 0.06x-0.10x it.  **Which member is which is
   decided by the BLAS reduction order.**
4. `prop = |flux| > thr` therefore calls one member of an evanescent pair
   PROPAGATING, on round-off.
5. The selector

   ```python
   flip = np.where(prop, flux < 0.0, q.imag < 0.0)
   ```

   hands that member the FLUX-SIGN rule instead of the `Im(q)` rule.  At one
   thread the crossing member has `Im(q) = +0.2787` and the flux sign leaves it
   alone -- harmless.  At two threads the crossing member has
   `Im(q) = -0.2787` and the flux sign leaves it alone too -- and a mode that
   **GROWS along +z** enters the forward set.
6. The cascade is then assembled from a forward set containing a growing mode.
   `R + T` = 22.35.

That is the T3-4 defect exactly as `PMM_FOURNAME_ADJUDICATION_2026_08_05` S4
describes it, with the BLAS thread count as the axis instead of the degree.

### S2.2  It is NOT the null-space draw

Ruled out directly: `_guarded_lstsq` neither refuses nor is reached
differently in the two pools, `Hsup`'s rank and residual are the same, and the
J00 fork survives with `far_field_orders` varied 7 -> 41 (which changes
`Hsup`'s shape from 30x156 to 166x156 and its null space from 126 to 0
dimensions) while the fork does not change by a digit.  A null-space draw
cannot be invariant to removing the null space.  X-1 is untouched by this work
and remains open.

---

## S3.  Why nothing refused a `|R+T-1|` = 21.35 answer

The referral asked, correctly, why conservation was blind when this one is not
even unitary.  Three separate reasons, all now recorded:

1. **The T3-4 guard SAW it and is shipped DISARMED.**  Armed, the verdict on
   the two-thread solve reads *"put 8 GROWING mode(s) in the FORWARD set on 6
   half-space/patterned row(s) ... within a factor 1.11 of the
   propagating/evanescent cut"*.  It is disarmed because its bar was refuted on
   the conical family (`_core.py`'s block comment, M3 S5.5).  So the instrument
   was right and silent by policy.
2. **No energy check runs on this path.**  `_conical_nodal_solve` computes
   `R`/`T` and returns; there is no `_check_energy` equivalent on the PMM
   conical cascade the way `rcwa/_core.py` has one.  Nothing looked at the
   number 22.35.
3. **The test that was looking had had its conservation half screened out.**
   `test_rcond_of_hsup_would_have_been_the_wrong_instrument` was re-anchored
   the day before (S9 of the referring document) so that when the device's own
   baseline misses closure the claim is PRINTED instead of asserted.  That was
   the right call for a test fix and it is exactly what deferred the finding.

Item 2 is a real detection gap and is recorded as open in S8 -- it is a
separate change from this one.

---

## S4.  The fix

`lumenairy/elements/pmm/_core.py`, new `_forward_growth_flip`, which OWNS the
selector policy for every path that used the flux cut:

```python
flip = xp.where(prop, flux < 0.0, q.imag < 0.0)          # the historical rule
if not PMM_FORWARD_GROWTH_REPAIR:
    return flip
qf = xp.where(flip, -q, q)
lim = xp.where(xp.isfinite(thr), _MODE_CUT_MARGIN_WARN * thr, 0.0)
bad = (prop & (xp.abs(flux) < lim)
       & (qf.imag < -_MODE_GROWTH_REL * xp.abs(qf)))
return xp.where(bad, q.imag < 0.0, flip)                 # the repair
```

**The invariant.**  A forward mode of a passive layer may not grow along +z.
Where the flux rule is provably round-off, the direction is taken from `Im(q)`
instead -- the rule the evanescent branch already uses, and the one that
cannot produce a growing forward mode by construction.

**Nothing here is new calibration.**  `bad` is bit-identically
`_mode_cut_growth`'s `grow` mask: same three conjuncts, same two bars
(`_MODE_CUT_MARGIN_WARN` = 10, `_MODE_GROWTH_REL` = 1e-6), both already
calibrated and both already justified in that function's docstring --

* `prop`: the evanescent branch is already safe;
* within the cut's decade: what keeps this safe on a GAIN medium, where a
  forward mode legitimately grows.  An amplifying mode carries REAL power and
  sits ~1e8 x the cut, so it is never touched;
* `Im(q) < -rel |q|`: scale-free.  A genuinely propagating mode reads
  `|Im q| / |q| <= 5e-10` and is never touched.

So the instrument and the repair can never disagree about which modes are
affected -- asserted, not assumed
(`test_the_forward_set_cannot_grow_on_the_union_grid_conical_staircase`).

**The census still reads the RAW state.**  `_record_mode_cut` is deliberately
still called with the pre-repair `prop`/`q`, so `_MODE_CUT_CENSUS` and the
T3-4 verdict keep measuring the DIAGNOSIS -- what the bare selector would have
done -- and not the repaired state.  The guard's behaviour is therefore
unchanged by this fix, which is what lets the M3 suite's calibration table
stand.

**Sites** (all five that used the flux-cut selector, so the NumPy and JAX
twins keep their parity):

```text
_core.py   _sem_modes_tensor           (NumPy, patterned layer)
_core.py   _sem_modes_uniform          (NumPy, uniform half-space)
_core.py   _jpmm_sem_modes_tensor      (JAX twin)
_core.py   _jpmm_sem_modes_uniform     (JAX twin)
_jax_stack.py  the stack's uniform modes (JAX twin)
```

Every operation is elementwise, so it traces -- unlike M1's guard, there is no
data-dependent control flow and no reason to keep the JAX paths on the old
rule.  `bor/_jax_bor.py` carries a structurally similar selector on a
DIFFERENT flux definition (no mass weighting, its own `propagating` test) and
is deliberately NOT touched here.

**Fail-before switch:** `PMM_FORWARD_GROWTH_REPAIR` (default `True`).  `False`
restores the pre-fix selector bit for bit on every path.

---

## S5.  Fail-before / passes-after

### The referred device (library level)

```text
threads  repair  J00 (ffo 7)                           |R+T-1|
1        OFF     -0.17117932369944+0.00906676381430j   6.6544e-06
1        ON      -0.17117932369944+0.00906676381430j   6.6544e-06   (moved 1.6e-15)
2        OFF     -0.27216039162719-0.09244618541393j   2.1348e+01
2        ON      -0.17117932153622+0.00906676357933j   6.6516e-06   (moved 1.4e-01)
default  OFF     -0.17117932109206+0.00906676337991j   6.6507e-06
default  ON      -0.17117932109200+0.00906676337979j   6.6507e-06   (moved 1.3e-13)
```

Thread-count spread of `J00` on the union grid, same code, same geometry:

```text
             1 vs 2 vs default
repair OFF   1.4319e-01
repair ON    2.6434e-09       -- 5.4e7 x tighter
```

The per-layer path moves 2.0e-15 .. 2.9e-15 at every thread count, i.e. it was
and remains unaffected.

### The pin (test level), at two threads and at one

`test_the_forward_set_cannot_grow_on_the_union_grid_conical_staircase`, with
`PMM_FORWARD_GROWTH_REPAIR` forced off:

```text
threads  outcome
1        FAILED -- "the repaired selector STILL leaves 1 growing mode(s) in
         the forward set (raw: 1)"
2        FAILED -- "... STILL leaves 8 growing mode(s) ... (raw: 8)"
```

and green at both with the switch on.  The pin is an INVARIANT rather than a
number precisely so it does not depend on the pool the runner happens to
choose: a test cannot vary `OPENBLAS_NUM_THREADS` in-process, and a bar on the
answer would have reproduced the tenth name's own disease.

---

## S6.  Blast radius: the fix closes T3-4 on the classical path too

This is larger than the referral supposed and is the main result.

### S6.1  The M3 T3-4 family (classical per-layer, `phi` = 0)

`rel` = `|R0 / RCWA_141 - 1|`; `g` = `n_growing` (unchanged by the fix, since
the census reads the raw state):

```text
            1 thread                     2 threads
ns  deg     rel OFF    rel ON      g     rel OFF    rel ON      g
 2   10     0.00500    0.00500     0     0.00500    0.00500     0
 2   12     0.43985    0.00473     8     0.00473    0.00473     0
 2   14     4.66257    0.00461     4     0.00461    0.00461     4
 2   16     4.66249    0.00455     7     4.66249    0.00455     6
 6   10     4.11540    0.00497     2     4.95795    0.00497     1
 6   12     0.00467    0.00467     1     4.11595    0.00467     1
 6   14     0.00454    0.00454     2     0.41842    0.00454     3
 6   16     4.95803    0.00448     4     4.95803    0.00448     5
 2   18     0.43987    0.43987     8     0.00451    0.00451     2
 2   20     4.66242    4.66242     6     4.66242    4.66242    12
 6   18     4.11642    0.00445     7     0.00445    0.00445     2
 6   20     5.13748    0.00443     7     0.00443    0.00443     9
```

Every silent-wrong cell in `_T34_FAMILY` (degrees 10..16) is CURED, at both
thread counts, and the notorious thread-count migrations (`2/12`, `2/14`,
`6/12`, `6/14`) collapse onto one answer.  Two cells in `_T34_WIDEN` --
`2/18` and `2/20` -- are NOT cured, so `test_t34_guard_fires_on_every_
silent_wrong_cell_of_this_build` still finds its wrong cell by widening, still
partitions (min(wrong) 0.43987 vs 20 x max(right) 0.100), and still passes.
That is measured, not hoped: see S7.

The cured (`min_feature` above threshold) ladder is **bit-identical** with the
repair on or off -- 12 of 12 rungs, both `ns`, both thread counts, `dR0`
exactly `0.0`.  That is the null floor.

### S6.2  The M2 `min_feature` contract

```text
                                        1 thread              2 threads
case                                    OFF       ON          OFF       ON
coated ns=2 default    (below thr)      2.7605    0.0036153   0.0036153 0.0036153
coated ns=3 mf 0.5 nm  (below thr)      2.6345    0.0037919   2.8101    0.0037919
coated ns=6 mf 1.5 nm  (below thr)      2.2490    0.0042358   1.9665    0.0042358
uncoated ns=6 mf 1.5nm (below thr)      1.3384    0.0040555   1.3384    0.0040555
uncoated ns=12 default (below thr)      2.0719    0.0046716   2.0719    0.0046716
coated ns=2 mf 0.5 nm  (ABOVE thr)      0.0035592 0.0035592   identical bits
coated ns=3 mf 1.5 nm  (ABOVE thr)      0.0039957 0.0039957   identical bits
coated ns=6 mf 3.0 nm  (ABOVE thr)      0.0058584 0.0058584   identical bits
uncoated ns=6 mf 3.0nm (ABOVE thr)      0.0073248 0.0073248   identical bits
uncoated ns=12 mf 1.5nm(ABOVE thr)      0.0069101 0.0069101   identical bits
```

(spread = degree-ladder peak-to-peak over degrees 6..14, relative to the mean.)

Note the `ns=2 default` row at two threads: `0.0036153` with the repair OFF.
That is `3.615e-03` -- **the tenth name's referred number, exactly**.  The
linkage the referral asked me to confirm or refute is CONFIRMED: it is the
same round-off-flux classification coin flip, and at two threads that
particular cell happened to classify correctly, so the "broken" configuration
read not-broken.  With the repair it reads `0.0036153` at every thread count,
for the right reason.

### S6.3  Stationary is not the same as right

Stationarity alone never proved correctness, so the repaired ladders are
scored against the independent RCWA anchor (141 orders) the M3 suite carries:

```text
                    1 thread                    2 threads
ns  repair  |uncured/cured - 1|  |u/RCWA - 1|   |u/cured - 1|  |u/RCWA - 1|
 2  OFF     4.642                4.663          0.001083       0.008246
 2  ON      0.001083             0.008246       0.001083       0.008246
 6  OFF     4.124                4.115          4.968          4.958
 6  ON      0.007061             0.008803       0.007061       0.008803
```

With the repair the uncured default ladder agrees with the CURED ladder to
0.1-0.7 % and with RCWA to 0.9 %, identically at both thread counts.  Without
it, whichever cell the coin flip lands on is 410-500 % wrong.  The repair
produces the right answer, not merely a stable one.

---

## S7.  Tests changed, and why none of them was weakened

`tests/unit/test_pmm_m2_window_contract.py` only.  No library test outside my
scope was touched; `test_m1_conditioning_guard.py` and
`test_pmm_m3_efficiency.py` are unmodified and pass as landed.

Three tests in the M2 file pinned "below the `min_feature` threshold the
degree ladder COLLAPSES" as a fail-before.  S6.2 shows that collapse is now
cured, so those assertions had to move -- and each of them said so itself
("this test must be re-pinned against the fix, not relaxed").  All three are
re-pinned identically:

* the original collapse assertions are kept **verbatim** -- same cells, same
  bars -- scored with `PMM_FORWARD_GROWTH_REPAIR` off via a fixture.  The
  defect is still reproduced on every build;
* the CURE is then asserted with the switch on (new);
* the above-threshold cells are asserted **bit-identical** with the switch
  either way, tolerance `0.0` (new -- a null control the file did not have);
* and the repaired ladder is scored against RCWA (new -- see S6.3).

The one substantive restructuring is in the tenth name itself: its fail-before
is now scored over the below-threshold FAMILY rather than over the single
`ns=2` cell, because S6.2 measures that cell migrating with the pool.  That is
the same treatment the four-name adjudication applied to the wrong-cell lists,
for the same reason, and the bar (`> 0.5`) is unchanged.

One test was ADDED:
`test_the_forward_set_cannot_grow_on_the_union_grid_conical_staircase`, which
pins the ninth name as an invariant over every modal row of the referred
device -- (a) the raw selector grows something here, (b) the repaired one
grows nothing, (c) the two grid paths agree and the union grid closes.

Two bars in the re-pinned tenth name are worth naming explicitly because they
LOOK like relaxations and are not: the cured-ladder stationarity bar is `5e-3`
for `ns=2` and `2e-2` for `ns=6`.  Those are the file's own pre-existing bars
for those two cells (this test's and its sibling's respectively); the `ns=6`
cured ladder has always spread `5.9e-03`, which is over the `ns=2` bar and
under its own.  They are not interchangeable and neither was moved.

---

## S8.  Result

`-q -p no:randomly`.  Windows = python 3.14.6 / numpy 2.4.4 / scipy 1.17.1,
scipy-openblas 0.3.31 Haswell, MAX_THREADS 24.  WSL = python 3.12.3 / numpy
2.4.6 / scipy 1.17.1, same OpenBLAS release, SkylakeX, MAX_THREADS 64.

### The four named files: `m2` + `m3` + `m1` + `v5_13_0_pmm_tapered`

```text
mount    OPENBLAS_NUM_THREADS   before (m2 only)            after
Windows  1                      3 failed, 11 passed         94 passed
Windows  2                      3 failed, 11 passed         94 passed
Windows  default                3 failed, 11 passed         94 passed   291 s
WSL      1                      3 failed, 11 passed         93 passed, 1 skip
WSL      2                      3 failed, 11 passed         93 passed, 1 skip
WSL      default                3 failed, 11 passed         93 passed, 1 skip  238 s
```

The "before" column is the M2 file alone in its landed state with this fix
present -- i.e. the three fail-befores S7 re-pins.  `m1`, `m3` and the tapered
suite were green before this change and are green after; the count rises from
93 to 94 on Windows because of the added ninth-name pin (WSL's extra skip is
`test_the_guard_is_numpy_only_and_the_traced_path_is_unchanged`,
`pytest.importorskip("jax")` -- that venv has no jax; unrelated and unchanged
by this work).

Intermediate cells measured along the way, all green: `m1` + `m3` at 2 threads
(73 passed), `m3` + `m1` + tapered at 2 threads (79 passed), `m3` + tapered on
WSL at 1 and 2 threads (52 passed each).

### Default-threads blast radius, Windows

```text
suite set                                                     result
22 PMM suites (pure / 2d / conical / stack / per-layer /      378 passed  527 s
  niche / oblique / staggered / tensor)
14 suites incl. every JAX twin this change touches            293 passed,
  (pmm_jax_guards, pmm_autodiff, pmm_jones_autodiff,            2 failed   249 s
  pmm2d_autodiff, pmm_jones_2d_jax, threaded_sweep,
  hybrid_sweep, stabilize, fff_nv, rcwa_upgrades, ...)
```

The two failures are NOT caused by this change, and the check is the switch
rather than an opinion -- with `PMM_FORWARD_GROWTH_REPAIR` forced off they
fail with byte-identical numbers:

```text
test                                                 1 thr  2 thr  default
v5_20_13_pmm_jones_2d_fff_nv::                       pass   FAIL   FAIL
  test_pmm_fff_nv_matches_rcwa_fff_nv
    "RCWA reference unstable", |R+T-2| = 2.9865286872832186e-03
    against a 1e-3 bar -- the SAME 17 digits with the repair on and off
v5_20_0_pmm_rcwa_upgrades::                          pass   FAIL   FAIL
  test_stack_sweep_geom_cache_is_transparent_and_reused
    np.array_equal between a cached and an uncached sweep -- likewise
    identical with the repair on and off
```

Both are the campaign's standing disease on the RCWA / geometry-cache side
(an absolute energy bar and a bit-identity claim, on magnitudes that move with
the BLAS reduction order), not the PMM classification path.  Referred in S9.

### Cost

The M2 file's runtime rises from ~20 s to ~63 s at one thread: each of the
three re-pinned tests now measures its claim with the switch BOTH ways, and
there is a new pin.  A ladder cache keyed on `(ns, min_feature, degrees,
repair_flag)` recovers about a third of that (94 s -> 63 s); the flag is part
of the key deliberately, because a cache that ignored it would fake the
bit-identity null control.

---

## S9.  Open

1. **The PMM conical cascade has no energy check.**  `_conical_nodal_solve`
   returned `|R+T-1|` = 21.35 and nothing looked at it.  The RCWA side has
   `_check_energy`; the PMM conical side has no equivalent.  That is the
   detection gap this incident actually exposed and it is a separate change.
2. **The T3-4 guard still ships DISARMED**, and the conical false positive
   that disarmed it is unchanged -- the census still reads the RAW state by
   design, so the three correct conical cells still carry 1-3 growing forward
   modes and would still be spoken about.  Arming it is still gated on the
   consensus probe.  Note the fix makes this LESS urgent, not more: the
   condition the guard warns about is now repaired rather than merely
   reported.
3. **`2/18` and `2/20` remain silent-wrong** after the repair (S6.1) -- so the
   sliver defect is not fully closed by classification alone at high degree on
   the `ns=2` ladder, and `test_t34_guard_fires_on_every_silent_wrong_cell_
   of_this_build` still has a wrong cell to find.  Whether those are a second
   mechanism or the same one past the `_MODE_CUT_MARGIN_WARN` decade is not
   determined here.
4. **X-1 remains open**, untouched.  S2.2 rules it out as the cause of the
   ninth name; nothing here closes it.
5. **`bor/_jax_bor.py`** carries a structurally similar `where(propagating,
   Pz < 0, Im q < 0)` selector on a different flux definition.  Not examined.
6. **Two new names, found by the blast-radius sweep and NOT caused by this
   change** (S8 has the switch evidence -- both fail with byte-identical
   numbers whether the repair is on or off, and both pass at one thread):
   `test_v5_20_13_pmm_jones_2d_fff_nv::test_pmm_fff_nv_matches_rcwa_fff_nv`
   (an absolute `|R+T-2| < 1e-3` bar reading 2.99e-03) and
   `test_v5_20_0_pmm_rcwa_upgrades::test_stack_sweep_geom_cache_is_
   transparent_and_reused` (an `np.array_equal` bit-identity between a cached
   and an uncached sweep).  Both are the pattern
   `PMM_FOURNAME_ADJUDICATION_2026_08_05` names: a per-build magnitude
   asserted as universal.  Referred, not fixed.

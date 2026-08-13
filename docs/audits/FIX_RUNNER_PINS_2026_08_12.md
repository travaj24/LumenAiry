# Five runner-axis pins on main -- adjudicated and re-stated -- 2026-08-12

Branch `fix/runner-pins` off `origin/main` (21802f9, the PR #32 merge).

Five tests red the GitHub runners while green on both local mounts.  They are
not a regression from the three branches that just merged: the identical names
failed on branches that share no files (`ci_cf.log`, PR #30's carrier-field
tree, and `ci_obl.log`, PR #32's screen-obliquity tree), and every one of them
turns out to be a claim about a NUMBER that a different BLAS / LAPACK build is
entitled to move, asserted as a claim about the physics.

That is the campaign's standing disease, already treated three times in this
directory -- `PMM_FOURNAME_ADJUDICATION_2026_08_05` (the axis is the BLAS
reduction order, not the BLAS build), `FIX_UNION_GRID_2THREAD_2026_08_06` (an
absolute energy bar and a bit-identity claim on magnitudes that move with it),
`FIX_M2_NULL_CONTROL_2026_08_09` (a control that conditioned on the GEOMETRY
while the mechanism conditions on the CENSUS).  This round is the same disease
in five more places, and every fix below is one of the two treatments those
established: score the claim between two arms measured in ONE process, or
condition it on the shipped instrument.

Nothing in `lumenairy/` changes.  All five are test-side.

---

## S1.  What failed, where

| # | test | reading | bar | jobs |
|---|---|---|---|---|
| 1 | `test_niche_audit_w3_oracles.py::test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit` | 8.9003897385e-14 vs frozen 9.0968975e-14 (2.16e-2) | rel 1e-2 | JAX py3.12; unit 3.12 sh1+sh3; 3.13 sh1+sh3 |
| 2 | `test_niche_audit_w6_berreman.py::test_o7_split_fwd_bwd_matches_the_jax_twin_on_physical_tensors` | 1.146 | 1e-11 | JAX py3.12 |
| 3 | `test_niche_audit_w9_eig_vjp.py::test_pmm1d_angle_gradient_at_exactly_zero_stays_bounded[te-0.01]` | 0.026644 | 1e-2 | JAX py3.12 |
| 4 | `test_pmm_m2_window_contract.py::test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper` | 0.00406 | > 0.1 | unit 3.12 sh1 (obl only) |
| 5 | `test_v5_14_1_rcwa_deferred.py::test_tapered_grating_shear` | 1.4305778783807455e-12 | 1e-12 | JAX py3.12; unit 3.13 sh3; 3.12 sh3 |

Local baseline: all six ids (4 carries a parametrization) PASS on Windows
py3.14 / numpy 2.4.4 and WSL py3.12 / numpy 2.4.6, before any change.

Mounts used throughout: **M** = Windows py3.14, `numpy 2.4.4`,
`libscipy_openblas64 0.3.31`, 24 threads; **W** = WSL py3.12, `numpy 2.4.6`,
OpenBLAS, driven at `OPENBLAS_NUM_THREADS` 1 / 2 / default.

---

## S2.  Method

Each pin got the same three steps, in order.

1. **Adjudicate against an oracle before touching the bar.**  A runner reading
   is either a legitimate alternative evaluation of the same quantity or a
   wrong answer, and only the first licenses a re-pin.  Four of the five are
   adjudicated below against something independent of the bar (a converged
   reference, the eigenvalue multiset, the floor's own knob, the shipped mode
   census); the fifth (5) is nine decades away from anything physical.
2. **Re-state the claim where it is portable.**  Either a RATIO between arms
   measured in one process (1, 3, 5, and half of 2), or a partition on the
   shipped instrument (4, and the other half of 2).  No absolute bar on a
   BLAS-dependent magnitude is added anywhere.
3. **Prove the re-statement still has teeth, by emulation.**  Every pin gains
   a fail-before test that drives an injector -- the near-cut mode-cut
   injector already in-tree, a raw-order permutation on the `eig` the JAX twin
   is handed, a live copy of the eig VJP with the degenerate splitting shrunk,
   an out-of-band explicit grid size, one raster pixel of shear -- and asserts
   the restructured claim fires on the real defect and does NOT fire on the
   gauge difference.  Nothing is xfailed and nothing is skipped.

---

## S3.  Pin 1 -- the W4-T1 escape hatch was pinned to a frozen scalar

**The claim** is "`sigma_grid_n=64` reproduces the pre-W4-T1 flat-64 default",
the escape hatch an optimiser loop needs.  It was scored against two merit
values frozen on 74cf31b at rel 1e-2.  The docstring already recorded a 3.1e-3
cross-platform drift and that a tighter version of this same pin broke CI once
at 1664c92 -- the number was never the claim, it was a proxy for one.

**Adjudicated: not a wrong answer.**  What the pin measures is
`|L(2,0)|^2` on a sigma grid that UNDER-RESOLVES the chirp by design.  Against
the converged answer, `n = 64` is itself off by more than the drift, and CI's
reading is the CLOSER of the two to the truth:

```text
R1 = 51.5 mm        merit               vs converged
  n = 64   [M]      9.09689752e-14      +3.89e-2
  n = 64   [CI]     8.90038974e-14      +1.60e-2
  n = 128           7.71720088e-14
  n = 256           8.83407088e-14
  n = 512           8.75633833e-14      (reference)
  n = 768           8.77447797e-14
```

One rung of the cost ladder moves the value by up to 18 %.  A quantity that
swings 18 % on a grid change cannot carry a 1 % cross-build pin; 2.2e-2 is
inside its own aliasing noise.  (M and W agree with each other to 4.0e-10 and
with the frozen anchors to 2.5e-9, so the drift is not reproducible on either
mount -- which is exactly why the fix must not depend on reproducing it.)

**Re-stated on two LIVE arms.**  The adaptive default clamped by
`sigma_grid_n_max=64` resolves to the same `n_grid = 64` and then runs the
identical quadrature, so it is a live stand-in for the pre-W4-T1 flat default.
The two arms must therefore agree BIT FOR BIT in one process:

```text
                 arm delta on L    separation vs adaptive default
R1 = 51.5  [M]   0.0 (array_equal) 2.8892e-02   (n_default 256)
R1 = 51.5  [W]   0.0 (array_equal) 2.8892e-02
R1 = 60.0  [M]   0.0 (array_equal) 2.7848e-01   (n_default 192)
R1 = 60.0  [W]   0.0 (array_equal) 2.7848e-01
```

Three claims now: arm equality (exact, in-process), live separation from the
adaptive default (bar 1e-2, measured 2.9e-2 / 2.8e-1), and a factor-of-two
sanity band on the frozen anchors -- 23x headroom over the observed 2.2e-2,
kept only to catch a gross change in the hatch.

**Fail-before** (`test_w4_t1_escape_hatch_scope_guard_catches_a_non_verbatim_
explicit_n`): the defect the guard exists for is the explicit value ceasing to
be honoured verbatim, and its likeliest form is a tidy-up that rounds it onto
the cost ladder.  64 is already a rung, so the regression only shows BETWEEN
rungs; the fail-before asks for 96 -- what such a regression would deliver --
through the public parameter, nothing monkeypatched, and the arm-equality
claim fires.

### S3.1  Second adjudication -- the SEPARATION arm was still a runner bar

The restructure above was half right.  Arm equality is exact and portable and
has never moved.  The SEPARATION arm it added -- "the hatch is not the adaptive
default", `rel > 1e-2` -- went red on three tip jobs (JAX py3.12, unit 3.12
sh3, 3.13 sh3) at **5.0593e-03**, against 2.9e-2 on both mounts.  A 2.9x
headroom is not headroom when the quantity itself is a residue.

**Adjudicated: the runner is right again, and for the same reason.**  `sep` is
the DIFFERENCE OF TWO GRID-ALIASING ERRORS -- one at `n = 64` and one at
whatever the adaptive default resolved to (256 at R1 = 51.5) -- and at that
design both arms sit inside the same 3-5 % band around the converged answer.
The whole ladder there lives in a 5 % window, and the separation is not even
monotone in `n` [M]:

```text
R1 = 51.5 mm,  merit = |L(2,0)|^2,  hatch n = 64 -> 9.0968975230e-14
  n      merit                sep vs 64
  96     6.6285144129e-14     2.713e-01
  128    7.7172008847e-14     1.517e-01
  192    8.6607999749e-14     4.794e-02
  256    8.8340708759e-14     2.889e-02   <- the adaptive default
  384    8.8999736339e-14     2.165e-02
  512    8.7563383293e-14     3.744e-02
```

A residue of two numbers each entitled to move a couple of percent cannot
carry a 1 % bar, and pinning it is the disease the branch exists to treat.
(R1 = 60 was never at risk -- its `n = 64` is 28 % out, so its separation reads
2.785e-01 -- but the loop scores both designs and the bar has to be right on
the worse one.)

**The axis is the BUILD, not the reduction width**, so nothing local
reproduces the runner's 5.06e-3 and the fix must not need it to.  Bit-identical
at every width [M, `threadpool_limits`]:

```text
threads      R1 = 51.5, n_def 256                    R1 = 60.0, n_def 192
1 / 2 / 4 /  merit64 9.0968975230e-14  sep 2.889190e-02   sep 2.784777e-01
8 / default  (identical to the last digit at every width)
```

**Re-stated on the two things the bar was really guarding**, each where it is
exact:

* the default is still **ADAPTIVE** -- asserted on the RESOLVED `n` the
  shipped result object reports, as an INTEGER: 256 at R1 = 51.5, 192 at
  R1 = 60.  That is `ceil(4*extent*v_max/lam)` rounded onto the cost ladder,
  deterministic integer arithmetic, and the two designs clear their rungs by
  193..256 and 129..192, so no LAPACK build has ever moved one (the sibling
  `test_w4_t1_default_sigma_grid_n_is_adaptive_and_on_the_ladder` pins the
  same integers and has never failed on a runner);
* `sigma_grid_n` is still **LOAD-BEARING** -- two DIFFERENT resolved grids
  must give two different answers, at a ROUND-OFF bar of `1e-6` rather than an
  accuracy one.  This is the arm that stops claim (1) from passing on an
  identity that costs nothing, and it is portable because a default that has
  gone flat makes the two arms the SAME CALL: the reading is then EXACTLY 0.0,
  on every build.  1e-6 sits 3.7 decades below the worst live reading
  (5.06e-3, ubuntu py3.12) and 10 decades above float64 round-off.

**Second fail-before**
(`test_w4_t1_escape_hatch_scope_guard_catches_a_default_that_went_flat`): the
first fail-before drives defect (b), the hatch; this one drives defect (a), the
default, which the separation bar had been carrying silently.  The injection is
the pre-W4-T1 default written the only way the library still offers it --
`sigma_grid_n_max=64` through the public parameter -- and the resolved-`n` arm
raises on it.  The test then measures the load-bearing arm on the same
injection directly and asserts it is `0.0` exactly, so the round-off bar is
never asked to separate two things that are merely close.

---

## S4.  Pin 2 -- an eigenvalue ORDER was being pinned as a partition

**The claim** is S1-13: `berreman._split_fwd_bwd` and
`_berreman_jax._layer_modes_jax` must produce the same partition ELEMENT-WISE,
not merely the same set (an earlier SORTED comparison reported a spurious 0.74
drift purely from ordering ties at `Re(gam) = +-0`).  CI read 1.146 against
1e-11 while both mounts read 1.1e-14.

**Adjudicated: a GAUGE difference, not a physics one.**  Both twins use the
same stable flag-argsort, which keeps each group in the RAW ORDER `eig`
returned it in.  So they agree element-wise only while the two `eig` backends
also agree on that order -- and on both mounts they do, bit-identically:

```text
400 draws       raw-order gap   spectrum gap   flag mismatches   elementwise
[M]             0.0             0.0            0                 1.158e-14
[W]             0.0             0.0            0                 1.130e-14
```

Nothing makes that portable, and an eigenvalue order is not physics: the
cascade contracts over the column order and never sees it.  Injecting the
order a different LAPACK is entitled to return reproduces the runner exactly
while leaving the physics untouched:

```text
injector                          elementwise   invariant
shipped (both backends)            1.158e-14    5.618e-15
raw-order permutation              1.566e+00    5.618e-15   <- CI read 1.146
partition side-flip (real defect)  3.929e+00    1.826e+00
```

The invariant column is not merely small under the permutation, it is the same
reading -- the invariants cannot see a column swap by construction.  (The test
asserts that as a bar rather than an `==`: the QR behind the projector is
entitled to an ulp under a column swap even though neither mount spends one.)

**Re-stated in three layers.**  A per-draw PRECONDITION (both backends solved
the same eigenproblem -- compared by power sums `p1..p4`, not by sorting, since
sorting is what tied at `Re(gam) = +-0` in the first place); the PARTITION, on
every draw, through partition-invariant observables (the two eigenvalue power
sums of each half, which fix the forward multiset without sorting it, plus the
orthonormal projector of the `4x2` modal block, which fixes the subspace);
and the COLUMN ORDER, verbatim, on the draws where both backends returned the
raw spectrum in the same order (400 of 400 on both mounts).  The invariants
are blind to permutation, scaling and in-subspace rotation -- the whole gauge
freedom two `eig` backends have at a fixed partition -- and `cond` of the
`4x2` block is at most 7.07 over the family, so the projector is well posed.

**Fail-before** (`test_o7_partition_claim_survives_a_reorder_and_still_catches_
a_side_flip`): both directions, on the `eig` the twin is HANDED (it takes it
as a parameter, so nothing is monkeypatched).  A raw-order permutation must
break the element-wise claim and leave the invariants at the same number; a
mode nudged 3e-9 of the spectrum scale across the classifier's own 1e-9 cut --
a real side flip, and 33x inside the same-eigenproblem precondition -- must
break the restructured claim.

### S4.1  Second adjudication -- the fail-before's own PRECONDITION

The restructured SHIPPED claim was green everywhere.  Its FAIL-BEFORE was not:
the JAX py3.12 job read

```text
only 78 of 80 draws have the two eig backends in raw-order correspondence
```

against its `assert corr == total`.  That precondition is the branch's own
disease one layer up -- a claim about RAW EIGEN ORDER across two LAPACK builds,
asserted as if it were a fact -- and it is exactly what S4 above says is not
portable, written into the test that proves S4.

**Adjudicated, and it is NOT a degeneracy artefact.**  The obvious excuse
would be that the two runner draws have (near-)degenerate spectra where an
order is genuinely undefined.  They do not: over the 80-draw family the
SMALLEST pairwise eigenvalue separation is `8.556e-03` of the spectrum scale,
and there is no draw below `1e-3`.  Those two draws are two well-separated
spectra that a different LAPACK simply deflated in a different sequence, which
is its right -- `zgeev` promises a set, not an order.

**Not reproducible locally, and that is the finding.**  Every width, both
mounts:

```text
mount / width                       corr    worst raw-order gap   min pair sep
Windows py3.14, threadpool 1/2/4/8  80/80   0.0                   8.556e-03
WSL py3.12, OPENBLAS 1 / 2 / dflt   80/80   0.0                   8.556e-03
Windows py3.14, 400 draws           400/400 0.0                   8.556e-03
ubuntu py3.12 (CI)                  78/80   --                    --
```

So the fix is DERIVED rather than tuned, and the runner's condition is
EMULATED rather than waited for.

**Re-stated in three moves.**

* the fail-before SELECTS the correspondence class and drives injector (a) on
  it, and the verbatim re-run of the original claim is restricted to the same
  class -- without that restriction a runner at 78 of 80 would make the
  `pytest.raises` pass with the injector having done nothing, i.e. the
  fail-before would be vacuous in exactly the condition it exists for;
* the NON-VACUITY floor is the injector's own REACH, not a count.
  `_o7_reorder_reach` measures, on the SAME backend both sides (a pure gauge
  experiment, no LAPACK comparison in it), the draws where `[1,0,3,2]`
  actually changes the twin's modal columns -- a `[f,b,f,b]` raw order is
  invariant under that swap once the stable flag-argsort has run, so the reach
  is structural.  Measured 75 of 80 on both mounts, and the assertion is the
  IDENTITY `disturbed == |class n reach|`, which fires both ways: a shortfall
  means the element-wise comparison has stopped seeing column order, an excess
  means it is seeing something a pure column swap should not have moved;
* the shipped claim gains the floor it never had: `corr > 0`.  With `corr`
  zero the element-wise half would have passed by measuring nothing.

**The runner is emulated exactly.**  `_o7_score` grew a `reorder` parameter
that hands the twin named DRAW INDICES in a permuted raw order (indexed by
draw, not by matrix bytes, so nothing depends on the twin's `Delta` assembly
being bit-identical to `_berreman_delta`'s).  Two of eighty reproduces the CI
reading to the digit, and the SHIPPED restructured claim is asserted green on
it -- which is the adjudication of the 78-of-80 in one assertion instead of a
paragraph:

```text
arm                          corr    invariant     elementwise over the class
shipped                      80/80   2.850e-15     1.158e-14
partial reorder, 2 of 80     78/80   2.850e-15     1.158e-14
full reorder                  0/80   2.850e-15     1.493e+00   (75 disturbed)
side flip (real defect)       0/80   1.999e+00     --
```

Driving the SAME two-draw injector through the BRANCH-TIP helper reproduces
the runner exactly and shows where the fault was: the tip's SHIPPED claim
reads `inv 2.850e-15 / elem 1.158e-14` and is green, and the tip's
`assert corr == total` raises with the runner's own sentence, "only 78 of 80
draws have the two eig backends in raw-order correspondence".  The
restructured claim was never what failed; its precondition was.

The reorder injector's two draws are taken FROM the build's own
correspondence class and the count is scored against that class
(`base_corr - 2`), not against 80 -- writing it as `total - 2` would have been
this fail-before's original mistake in miniature, since on the runner being
emulated the class does not start full.

---

## S5.  Pin 4 -- the threshold rule's NOT-CURED half, on the census

**The claim** is `"1.5 nm should NOT cure ns=6 (off = 1.804 nm)"`, measured as
the spread of the order-0 reflectance over a degree ladder.  CI read 0.00406.

This test was already census-conditioned once, on its CURED pair
(`FIX_M2_NULL_CONTROL_2026_08_09` S5).  The NOT-cured half kept the identical
premise on the SAME cell, and it is the half CI failed.

**Adjudicated: the runner is not wrong; the premise was.**  The spread only
collapses when the near-cut collision FIRES, and whether a mode lands across
the cut is a fact about one build's round-off.  The correlation is exact --
every cell whose census reads zero is stationary, and the one whose census
reads growing modes is the one that scatters:

```text
[M] uncoated taper, repair off, degrees 8/10/12/14/16
  cell                spread     raw census        ladder
  ns=6  mf=1.5nm      1.33841    [0,0,1,2,3]       0.11811 0.11784 0.65302 ...
  ns=6  mf=3.0nm      0.00732    [0,0,0,0,0]
  ns=12 mf=1.5nm      0.00691    [0,0,0,0,0]
  ns=3  mf=1.5nm      0.00372    [0,0,0,0,0]
```

And 0.00406 is not a stranger: this same file already pins it as the value the
BELOW-threshold cell takes with the forward-growth REPAIR ON.  The runner's
pre-repair answer IS the shipped answer, because its round-off never produced
the growing mode the repair exists to redirect.  Nothing was silently wrong.

**Re-stated on the geometry and the instrument** (`_uncured_below_threshold`,
which extends the existing `_ladder_rec` / `_snap_report` machinery and adds no
constants -- 0.1 and 2e-2 are the test's own):

* the GEOMETRY premise, asserted rather than commented: `min_feature` below
  the per-slice offset must leave the window UNSNAPPED, read off the shipped
  snap accounting, which is deterministic and carries no BLAS dependence;
* the MECHANISM, where the instrument says it is armed: a ladder that reads a
  raw growing mode must scatter (measured 1.34 against the bar 0.1);
* where the census reads ZERO on every rung there is no collision to leave
  uncured, so instead of skipping, the in-tree near-cut injector is walked
  down `_INJECTOR_SCALES` until the cell arms -- the mechanism must be
  REACHABLE on every build, and if no scale reaches it the test FAILS with the
  adjudication rather than passing quietly.

**Fail-before** (`test_the_threshold_rules_NOT_CURED_half_conditions_on_the_
census_too`): the injector runs the OTHER WAY from the null controls', and
that is the point -- they need a mode PUSHED across the cut so they scale the
threshold DOWN; this claim needs the runner that has NO mode near the cut, so
it scales UP.  The emulation is exact, not qualitative:

```text
mount / OPENBLAS_NUM_THREADS   cut x1                cut x3
Windows py3.14 default         [0,0,1,2,3]  1.33841  [0,0,0,0,0]  0.00406
WSL py3.12 1                   [0,0,1,1,1]  1.33841  [0,0,0,0,0]  0.00406
WSL py3.12 default             [0,0,1,2,3]  1.33841  [0,0,0,0,0]  0.00406
```

0.00406 is the CI assertion message's number to all four of its figures, on
both mounts.  (x10, x1e2, x1e4 and x1e6 read 0.00406 too: once the near-cut
mode is back under the cut, a higher cut changes nothing further.)  Note also
that the raw census itself moves with the thread count on one mount --
`[0,0,1,1,1]` at one thread against `[0,0,1,2,3]` at default -- which is the
axis, stated without an injector at all.

The three claims are the siblings': the reconditioned half passes under the
injector and reaches its probe branch; the ORIGINAL assertion, verbatim, on the
same reading, fails; and the disarmed reading really is the cured-looking one,
so (a) is not passing on an armed ladder.

Helper teeth, checked directly: driving `_uncured_below_threshold` with an
ABOVE-threshold `min_feature` (3.0 nm) raises on the geometry premise, and the
armed branch returns spread 1.3384.

### S5.1  Second adjudication -- the injector's SCALE was frozen too

The reconditioned half was green everywhere.  Its FAIL-BEFORE was red on four
jobs (unit 3.10 sh1, 3.11 sh2, 3.12 sh2, 3.13 sh2), on the one number it had
frozen: the near-cut injector at **x3** did not DISARM the cell there.  It left
`[0,0,0,1,1]` on py3.11 / 3.12 / 3.13 and `[0,0,0,0,1]` on py3.10.

The test's own message already named the treatment -- "Raise the scale rather
than deleting it: the condition is a CUT POSITION and every build has one" --
so this is that, done properly.

**Reproduced locally, which the campaign had not managed before.**  x3 is not
a property of the mount, it is a property of the mount AT A GIVEN REDUCTION
WIDTH, and the width the earlier table never drove is where it breaks
[WSL py3.12, `OPENBLAS_NUM_THREADS=4`, uncoated ns = 6 / 1.5 nm, degrees
8..16, repair off]:

```text
cut      raw census      spread     n_prop            vs the SHIPPED (repair-on) ladder
x1       [0,0,1,2,1]     1.93102    28,29,30,30,31    4.5471e+00
x3       [0,0,0,1,0]     2.13121    28,28,28,29,28    3.7181e+00   <- the CI condition
x10      [0,0,0,0,0]     0.0040554  28,28,28,28,28    0.0  (BIT-IDENTICAL)
x1e2     [0,0,0,0,0]     0.0040554  28,28,28,28,28    0.0
x1e3     [0,0,0,0,0]     0.0040554  28,28,28,28,28    0.0
x1e4     [0,0,0,0,0]     0.0040554  28,28,28,28,28    0.0
x1e6     [0,0,0,0,0]     0.0040554  28,28,28,28,28    0.0
x1e9     [0,0,0,0,0]     0.0040554   0, 0, 0, 0, 0    7.6961e-09   <- past the ceiling
```

Note the x1 spread there is 1.93102 and not 1.33841: the raw census AND the
scattered value both move with the reduction width, which is the axis, stated
again without an injector.

**Re-stated with the scale READ OFF THE INSTRUMENT** (`_first_disarming_scale`,
`_DISARM_SCALES = (1.0, 3.0) + (W, W^2, W^3, W^4)` with
`W = _core._MODE_CUT_MARGIN_WARN = 1.0e1`).  The disarming scale is not
guessed and not tuned: `_mode_cut_growth` counts a mode as growing only while
it is `prop` AND **within `W` of the cut**, so multiplying the cut by `W`
provably clears every mode the census is currently reporting.  `W` is on the
ladder by construction and the test asserts that it is, so the derivation
moves if the library's constant does.  The walk exists for two reasons and
neither is a search for a number: to pay the CHEAPEST sufficient rung first
(x3 composes back to exactly the shipped cut through the reconditioned
helper's own downward probe, so verifying there costs nothing -- that is where
this fail-before's 2x speed-up over the frozen-x3 version comes from), and to
catch the one case the derivation does not cover, a mode further out than `W`
being pulled INTO the warn window by the raised cut.  The remaining rungs are
one per such generation.

The ceiling is derived twice and the two derivations MEET:

* `W^4 = 1e4 = 1 / min(_INJECTOR_SCALES)`.  `_uncured_below_threshold` probes
  BACK DOWN through `_INJECTOR_SCALES` to show the collision is still
  reachable, and the two injectors compose multiplicatively, so a disarming
  scale above 1e4 would put the shipped cut outside the probe's own reach.
  Asserted in the helper, not commented;
* `1e9 = 1 / 1e-9`, from `_mass_flux_threshold`'s own
  `max(1e-9 max|flux|, round-off floor)`: at that scale the cut sits AT the
  strongest mode's flux and NOTHING is propagating.  The table above confirms
  the derivation exactly -- `n_prop` goes to 0 at x1e9 and nowhere below it.

**And the ceiling has TEETH, because the spread cannot see an over-escalation**
(x1e9 still reads 0.0040554 to six figures).  The oracle is the SHIPPED
answer: the disarmed PRE-repair ladder must equal, BIT FOR BIT, the ladder the
library returns with the forward-growth repair ON.  That is S5's adjudication
turned into an assertion -- the runner's build never made the growing mode the
repair exists to redirect, so its pre-repair answer IS the shipped answer --
and it separates 0.0 from the 7.70e-09 an over-escalation produces.

**And the search itself is falsifiable where it runs green.**  Both mounts are
ARMED at the shipped cut, so the walk must escalate off its first rung, and
the rung it leaves must be the SCATTERING one (`sum(raw) > 0` and
`spread > 0.1`).  A build already in the runner's condition stops at x1 and
that claim is skipped with it, which is correct: there is nothing to escalate
away from there.

```text
mount / width                    walk                          disarmed at
Windows py3.14 default / 1 thr   x1 [0,0,1,2,3] / [0,0,1,1,1]  x3
WSL py3.12 OPENBLAS 1 / default  x1 [0,0,1,1,1] / [0,0,1,2,3]  x3
WSL py3.12 OPENBLAS 2 and 4      x1 [0,0,1,2,1], x3 [0,0,0,1,0]  x10
```

**A latent cache defect, found and fixed on the way** (`near_cut_injector`).
`scaled` wraps whatever `_mass_flux_threshold` already is, so nested injectors
put the cut at the PRODUCT of their scales -- but `_CUT_SCALE`, which is part
of `_LADDER_CACHE`'s key, was being SET to the inner scale rather than
composed with the outer one.  That nesting is not hypothetical: the NOT-CURED
fail-before's upward injector wraps `_uncured_below_threshold`'s downward
probe.  On a build where the sibling `test_threshold_rule_holds_on_a_SINGLE_
REGION_uncoated_taper` also reaches that probe branch un-injected -- i.e. on
exactly the runners this section is about -- the two would have keyed an
effective cut of `S/3` and an effective cut of `1/3` to the same cache entry
and served one for the other.  `_CUT_SCALE` now carries the product.

---

## S6.  Pin 5 -- an absolute 1e-12 on a magnitude of order 0.3

**The claim** is that mirroring the shear mirrors the asymmetry:
`T(-shear)[-1] == T(+shear)[+1]`.  It was scored at an absolute `1e-12` on
numbers of order 0.334, i.e. about 14 ulp out of a 13x13 cascade through an
8-slice staircase.  CI grazed it at 1.4306e-12.

**Adjudicated: nine decades of daylight.**  The residual is round-off; the
smallest shear error the builder can even represent is one `n_x = 256` raster
pixel, and that is nine decades larger:

```text
                            vertical sym    mirror shear    relative
Windows py3.14 OpenBLAS     0.0             4.4498e-13      1.3303e-12
WSL py3.12 OpenBLAS         5.5511e-17      1.5654e-14      4.6799e-14
CI ubuntu py3.12 / py3.13   --              1.4306e-12      4.2777e-12
one raster pixel of shear   --              1.6753e-03      5.0080e-03
```

Both mounts read identically at `OPENBLAS_NUM_THREADS` 1 / 2 / 4 / 8 / default,
so the axis here is the LAPACK build, not the reduction width.  Under
`raster='hard'` a `d(shear)` of 1e-9 through 1e-4 leaves the answer
bit-identical -- it does not move a pixel -- so one pixel really is the floor
of what the check can be asked to see.

**Re-derived comparatively.**  Both symmetry residuals -- the mirror-shear one
that failed and the vertical-symmetry one at the line above it, which carries
the identical exposure and had simply not been unlucky yet -- are scored
against the order they are a symmetry OF, at `_SHEAR_SYM_REL = 1e-9`: 234x
above the worst measured reading and 5.0e6x below the smallest real defect.

**Fail-before** (`test_tapered_grating_shear_relative_envelope_still_sees_one_
raster_pixel`): injects exactly that one pixel and asserts both that it is
visible (rel > 1e-3, measured 5.0e-3) and that the shipped relative bar fires
on it.

---

## S7.  Pin 3 -- the OPEN eig-VJP defect, pinned as a defect

**The claim** was a "characterisation fence": at exactly normal incidence
mirror symmetry forces `d(sum R)/d(theta) = 0`, but the exactly degenerate
`+/-m` pair makes the eig non-differentiable there, so AD returns a finite
WRONG value.  This is the KNOWN LIMIT the module docstring names and the
bridge campaign flagged upstream -- it is OPEN, and the test is not a
correctness pin.  It fenced the wrong value with an ABSOLUTE bar, 1e-2 for TE.

**Adjudicated: the fenced quantity's MAGNITUDE is a per-build fact.**

```text
build                                TE AD(0)
authoring box (2026-07-27)          -2.221e-03
Windows py3.14 / numpy 2.4.4        +7.793e-03      (1.28x headroom on 1e-2)
CI ubuntu py3.12                    -2.664e-02
```

A decade of spread and a sign flip.  Adjudicated with the floor's own knob, in
a live copy of the library VJP: the floored value is INSENSITIVE to the
degenerate pair's numerical splitting -- shrinking it by 1e-8 moves AD(0) from
7.792836e-03 to 7.793047e-03 -- so it is not a `1/D` reading at all.  It is the
eigenvector JUMP, and which direction `V` jumps at an exact degeneracy is
precisely what a LAPACK build chooses.  The same probe shows what the floor IS
for: with the floor removed, that same shrink takes AD(0) to -1.12e+06.

```text
                       AD(0) TE      AD(0)/sweep TE   AD(0)/sweep TM
shipped                +7.793e-03    2.49e-02         1.33e+00
tau = 0, splitting x1  -3.411e-03    1.09e-02         6.11e-01
tau = 0, x 1e-4        -1.120e+02    3.58e+02         1.94e+04
tau = 0, x 1e-8        -1.120e+06    3.58e+06         1.94e+08
tau = 1e-12, x 1e-8    +7.793e-03    2.49e-02         1.33e+00
```

(The clean and sweep arms read 1.7541e-06 / 3.1315e-01 in every row: the
injector touches only pairs that are already degenerate.)

**Re-stated as a fail-when-fixed defect pin**, the shape
`test_x1_defect_is_reproduced_and_flagged_but_NOT_closed` and
`test_t34_bar_is_REFUTED_on_the_conical_family` use -- which is also the
original intent, since the test never claimed correctness.  Two ratio bars,
both between arms measured in one process:

* the DEFECT is still present: `|AD(0)| > 100 * |AD(1e-6)|`, where the clean
  near-normal AD is exactly linear in theta (its sibling pins that), so the
  correct value at 0 is 0 and `AD(1e-6)` is the scale a fixed eig-VJP would
  return something near.  Measured 4.4e3 (TE) / 1.6e5 (TM) -- 44x / 1580x
  headroom.  **A future fix should make this test fail**, and its message says
  to re-pin against the fix rather than loosen the ratio;
* the ENVELOPE: `|AD(0)| < 30 * max|AD|` over a physical angle sweep of the
  same observable.  Measured 2.49e-2 (TE) / 1.33 (TM) -- 1200x / 22x headroom
  -- and, per the table, invariant to a 1e8 change in the splitting that the
  runners disagree about.

The parametrization loses its per-pol bounds; both polarizations now carry the
same two ratios.  The symmetry oracle (`|FD| < 1e-6`, measured 2.8e-11 against
a float64 floor of ~7e-12 for this difference quotient) is unchanged.

**Correction to the pre-existing docstring**, recorded because it was load
bearing for the old bar: it stated the unfloored value was 7.7x WORSE
(-1.709e-02 against -2.221e-03).  On this build it is 2.28x BETTER (floored
+7.793e-03 against unfloored -3.411e-03).  Which of the two is larger at an
exact degeneracy is not a property the floor controls; boundedness under a
shrinking splitting is, and that is what the envelope now pins.

**Fail-before** (`test_the_theta0_defect_pin_fires_when_the_defect_is_absent_
or_unbounded`), both directions:

* DEFECT ABSENT -- the same claims scored against `rcwa1d`, which solves the
  identical grating with ANALYTIC half-space modes and has no degeneracy at
  normal incidence (AD(0) = 1.1e-13, pinned by the control test alongside).
  That is what a fix looks like from outside, and the pin must fail on it;
* VALUE UNBOUNDED -- the live VJP copy at `tau = 0`, splitting x 1e-4, must
  trip the envelope; and the SHIPPED floor must hold the same injection
  bounded.

---

## S8.  Verification

Both mounts, `pytest -q -p no:randomly`, the five touched modules -- 680
collected, all passing, against 675 on `origin/main`:

| module | on main | here | delta |
|---|---|---|---|
| `test_niche_audit_w3_oracles.py` | 179 | 180 | +1 |
| `test_niche_audit_w6_berreman.py` | 428 | 429 | +1 |
| `test_niche_audit_w9_eig_vjp.py` | 29 | 30 | +1 |
| `test_v5_14_1_rcwa_deferred.py` | 20 | 21 | +1 |
| `test_pmm_m2_window_contract.py` | 19 | 20 | +1 |
| **total** | **675** | **680** | **+5** |

The five target ids all stay (pin 3's two parametrizations are renamed,
`[te-0.01]` / `[tm-0.5]` -> `[te]` / `[tm]`, since the per-pol bounds are
gone); the +5 are one fail-before per pin.  Nothing is xfailed, nothing is
skipped, and no test is deleted.

M: 680 / 680 pass.  W: 679 / 680 -- the one failure is
`test_pmm1d_off_normal_angle_gradient_no_regression`, which is NOT one of the
five, is NOT touched by this branch, and fails IDENTICALLY on `origin/main`
with these five files stashed (28 passed / 1 failed there, same assertion,
same value).  See S9.

`ruff check` clean on all five files on both mounts (line-length 100, E/F/I).

### S8.0  Second round -- the three arms the runners refused

The tip of this branch (1657b99) was green on both mounts on its own scope and
red on the runners in three of its five pins, and in every case it was the NEW
arm -- the fail-before or the precondition the restructure had added, i.e. the
part that had never met a real LAPACK.  S3.1, S4.1 and S5.1 above adjudicate
and re-state them.  What is different about this round is that TWO of the
three now reproduce locally, so they are fixed against a measurement rather
than against an argument:

| cluster | runner reading | reproduced locally? | how |
|---|---|---|---|
| pin 1 separation | `sep 5.0593e-03` vs bar 1e-2 | NO | identical at every thread width on M (2.889190e-02 to the last digit at 1 / 2 / 4 / 8 / default); the axis is the build |
| pin 2 precondition | `78 of 80` corresponding | NO, then YES by injection | 80/80 at every width on both mounts; the two-draw `reorder` injector reproduces 78 of 80 and makes the TIP raise the runner's own sentence |
| pin 4 injector scale | census `[0,0,0,1,1]` at cut x3 | YES | WSL py3.12 at `OPENBLAS_NUM_THREADS=4` reads `[0,0,0,1,0]` at x3 -- the tip test fails there with the same assertion, and the fixed one escalates to x10 and passes |

The reduction width is the whole finding for pin 4: the earlier table drove W
at 1 and default and concluded x3 was safe.  Driven at **2 and 4** it is not.
x3 leaves `[0,0,0,1,0]` at both, and the branch-tip test fails there with the
runner's own assertion, verbatim.  2 is the interesting one -- it is the width
a two-core runner behaves like.

```text
W, m2 module, branch, by OPENBLAS_NUM_THREADS
threads   census at x1              x3                   disarms at   result
1         [0,0,1,1,1]  1.3384       [0,0,0,0,0] 0.00406  x3           20 passed
2         [0,0,1,2,1]  1.3384       [0,0,0,1,0] 2.1312   x10          20 passed
4         [0,0,1,2,1]  1.9310       [0,0,0,1,0] 2.1312   x10          20 passed
default   [0,0,1,2,3]  1.3384       [0,0,0,0,0] 0.00406  x3           20 passed
M default [0,0,1,2,3]  1.3384       [0,0,0,0,0] 0.00406  x3           20 passed
M 1 thr   [0,0,1,1,1]  1.3384       [0,0,0,0,0] 0.00406  x3           20 passed

branch TIP at the same widths: FAILS at 2 and at 4, same assertion, same
[0,0,0,1,0]; passes at 1 and default.  That is the CI failure, at home.
```

### S8.0.1  Second round -- verification, and what it cost

Module counts, `pytest -q -p no:randomly`, collected = passed everywhere:

| module | tip [M] | here [M] | tip [W] | here [W] |
|---|---|---|---|---|
| `test_niche_audit_w3_oracles.py` | 180 | 181 | 180 | 181 |
| `test_niche_audit_w6_berreman.py` | 429 | 429 | 429 | 429 |
| `test_pmm_m2_window_contract.py` | 20 | 20 | 20 | 20 |
| **total** | **629** | **630** | **629** | **630** |

The +1 is pin 1's second fail-before.  Nothing is xfailed or skipped and no id
is renamed.

Wall cost, `--durations`, one run at a time:

```text
id                                                    tip [M]   here [M]   tip [W]  here [W]
test_the_threshold_rules_NOT_CURED_half_...  (pin 4)    1.92 s     0.96 s    1.75 s   0.88 s
test_w4_t1_explicit_sigma_grid_n_64_...      (pin 1)    5.06 s     5.09 s      --       --
test_o7_partition_claim_survives_a_reorder_  (pin 2)    2.64 s     4.11 s      --       --
m2 MODULE TOTAL (OPENBLAS_NUM_THREADS=1)              48.11 s    47.71 s   45.12 s  44.65 s
w3 + w6 MODULE TOTAL                                  81.42 s    83.46 s      --     99.05 s
```

Pin 4's fail-before is **half** the tip's cost, and that is the cache-key fix
in `near_cut_injector` paying for itself: with `_CUT_SCALE` carrying the
PRODUCT, the reconditioned helper's downward probe composes to
`3.0 * (1/3) = 1.0` exactly and lands on the un-injected ladder already in
`_LADDER_CACHE`, where the tip re-solved it.  Pin 2's costs +1.5 s for two
extra 80-draw passes (the injector's reach, and the runner emulation); pin 1's
is unchanged.  The m2 module total is flat, and CI's own `.test_durations`
records it at 39.0 s for 19 ids, so it stays where the shard balancer has it.

**OVERSUBSCRIPTION IS AN AXIS OF ITS OWN, and it is worth recording** because
it cost hours of this round.  The same m2 module takes **>2 hours** when four
copies of it run at once on a 24-core box with OpenBLAS uncapped, and **48
seconds** at `OPENBLAS_NUM_THREADS=1` -- three orders of magnitude, on
identical code, and the tip behaves identically (an unmodified copy of 1657b99
was 12 seconds behind the branch after two hours).  These solves are many
small dense eigenproblems and OpenBLAS busy-waits, so oversubscription burns
cores without progress.  Cap the width before timing anything in this module,
and never read a wall clock off a loaded box as evidence about a test.

### S8.1  Mount / thread-count evidence behind the bars

Every table in S3-S7 was measured on both mounts except pin 3's injector
ladder, which is an in-process emulation and is M-only (its point is the
RATIO's invariance under the injected splitting, and the ratio is computed
from arms measured together).  Where the axis is the reduction width rather
than the LAPACK build, it is driven explicitly:
`OPENBLAS_NUM_THREADS` 1 / 2 / default out of process on W, and
`threadpool_limits` 1 / 2 / 4 / 8 in process on M.  The two places it bites
are pin 4's raw census (`[0,0,1,1,1]` at one thread against `[0,0,1,2,3]` at
default, same mount) and nothing else -- pins 1, 2 and 5 read identically at
every width on a given mount and differ only between builds, which is why
their fixes are arm-ratios rather than census partitions.

---

## S9.  Watch items, not fixed here

* `test_tapered_grating_shear`'s ENERGY bar is still absolute, `1e-7`, and its
  own comment records a CI runner closing this staircase at 4.5e-8 -- 2.2x
  headroom.  W reads 2.385e-11 and M 6.675e-13 on the sheared arm.  It has not
  failed and is out of scope here, but it is the same shape as pin 5 and is
  the next one likely to graze.
* `test_o7_split_fwd_bwd_matches_the_jax_twin_in_the_degenerate_fallback`, the
  sibling immediately below pin 2, carries the IDENTICAL exposure: an
  element-wise comparison across the two `eig` backends, at exact bit-identity
  (`worst 0.0`), and on GENERAL complex `Delta` rather than physical tensors --
  i.e. on inputs where the two LAPACKs are MORE likely to differ in raw order,
  not less.  It has not failed on any runner and is out of scope here, but it
  is pin 2 with the luck still holding, and `_o7_partition_invariants` is
  already in the file when it stops.

* **W-only, pre-existing, and worth running down separately:**
  `test_niche_audit_w9_eig_vjp.py::test_pmm1d_off_normal_angle_gradient_no_
  regression` reads `jax.grad(_pmm1)(0.3) = nan` on W (jax 0.10.2 / numpy
  2.4.6).  It reproduces on `origin/main` unchanged, so this branch neither
  causes nor fixes it, and it is green on M and on the CI JAX job.  Diagnosed
  far enough to hand off: it is a KNIFE EDGE at the exact double `0.3` and
  nothing wider -- 0.29, 0.31, `0.3 + 1e-12` and `0.3 - 1e-11` all return the
  correct 0.0363047, both polarizations, and the PRIMAL is finite (0.0182632)
  at 0.3 itself.  It is not the eig-VJP floor: `tau_rel` at 1e-12, 1e-10, 1e-8
  and 1e-6 all still give `nan`.  That signature -- finite primal, NaN
  gradient, measure-zero in the input, floor-independent -- is the JAX
  `jnp.where` trap, where the DEAD branch of a guard evaluates to NaN at the
  exact point its predicate flips and poisons the cotangent.  Something in the
  1-D PMM gradient path hits its guard exactly at this theta on this build.

* Pin 1's CI reading (8.9004e-14) is not reproducible on either mount: M and W
  agree with each other to 4.0e-10 and with the frozen anchors to 2.5e-9.  The
  restructured claim does not depend on reproducing it -- it cancels whatever
  the difference is -- but if it is ever wanted, the place to look is the
  ubuntu numpy wheel's quadrature path, not the library.

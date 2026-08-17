# The 2-D crossed-PMM "FEEC / yee2" accuracy milestone: measured, and refuted as already delivered -- 2026-08-17

Branch `feat/pmm2d-feec` off `origin/main` (`5ecbf7a`, the 5.39.0 release
commit).  `git log origin/main..HEAD` was empty before branching.

**The mandate was to lift the 2-D crossed-PMM hybrid's FMM accuracy floor by
deriving a FEEC / staggered-yee2 discretization of the transverse operator.  S1
records the reading that refutes the premise: that discretization is already in
`main`, has been since v5.11.0, and IS `twod_staggered.py`.  S2 measures the
floor anyway, because a docstring is not a measurement -- the floor is real and
large on the hybrid, and provably absent on the staggered path.  S3 isolates what
actually caps 2-D accuracy once the Fourier floor is gone, and it is not
something any cochain choice can move.  S4 measures the lever that does move it.
S5 is therefore a NO-GO on Phase B as chartered.  S6 records a genuine
silent-wrong defect the floor measurement turned up, which IS fixed here.**

**Mount.**  Windows 11 py3.14.6, numpy 2.4.4, lumenairy 5.39.0, 24 cores /
128 GB (tesla-ryzen).  Every probe ran with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` exported **in the
environment before python started**, and asserted `"lum_fe" in
lumenairy.__file__` (confirmed `C:\tmp\lum_fe\lumenairy\__init__.py` throughout).

**Cells.**  Period 0.9 um, wl 1.55 um, depth 0.3 um, `n_sup` 1.0 / `n_sub` 1.5,
normal incidence, both polarizations, all lossless:

| cell | pattern |
|---|---|
| **A** | eps 4.0 / 1.0 (n=2.0), duty 0.5 |
| **B** | eps 12.25 / 1.0 (Si n=3.5), duty 0.5 -- the high-contrast design-121-like pillar |
| **C** | eps 12.25 / 1.0, duty 0.25 on a 4-segment grid |

---

## S1.  The premise, checked in-repo first

`lumenairy/elements/pmm/twod_staggered.py` is not "a staggered discretization
that exists"; it is precisely the discretization the milestone asks for.  Read
against Arnold-Falk-Winther's requirements for a FEEC subcomplex, every piece is
present and named:

| FEEC requirement | where it is, in `twod_staggered.py` |
|---|---|
| two staggered cochain sets | `Basis1D` builds `Btilde` (continuous C0, hats + Bloch periodic hat, Granet Eq.32-33) and `B` (its discontinuous partner, **equal cardinality**) |
| the de Rham property `d(V_k) subset V_{k+1}` | `span(d Btilde) subset span(B)` -- stated at `:467-477`, measured at **8.925e-15** over 80 cases (M5, S2.3/G1) |
| correct cochain placement (the yee/edge-element staggering) | Eq.34 tensor expansion `E1 = B (x) Btilde`, `E2 = Btilde (x) B`, `E3 = Btilde (x) Btilde` -- each transverse component continuous across the wall it crosses, reduced in the other direction |
| curl into its OWN space, full rank | `Stt = -Curl^dag Gw^-1 Curl` assembled in `Vw = B (x) B`, "renders the de Rham complex EXACT -> spurious-free" (`:473-477`) |
| the dual (gradient) map, full rank | `Ktz` maps `V3 -> [V1; V2]`, "the dual of the curl's exactness" (`:490-496`) |
| material laws as weighted mass operators (the Hodge stars) | `_eps_weighted` / `Meps33`; eps piecewise-constant with **walls on element boundaries**, so the mass integrals are exact per element |
| no stabilization parameter | stated `:13-15` -- continuity is embedded in the basis, so there is no mechanism to inject spurious modes |

The module records the wrong-cochain attempt that FEEC theory predicts will
fail, and why: projecting the curl into `V3 = Btilde (x) Btilde` "loses rank
because `B'` is NOT in `span(Btilde)`, leaving a longitudinal residue = spurious
sea" (`:475-477`).  That is the milestone's core insight, already learned.

**The history explains the stale milestone phrase.**  Both halves are in the
**same** release, v5.11.0 (2026-06-02):

* the `pmm_efficiency_2d` (hybrid) entry says a genuinely no-floor 2-D **nodal**
  method "is blocked by the flux-inconsistent degenerate uniform-region nodal
  eigenproblem ... and is being pursued separately via an **FEEC E-D
  formulation**";
* the `pmm_efficiency_2d_staggered` entry, 40 lines later in the same release,
  delivers the result by the Granet-2023 staggered/mimetic route instead --
  "**spurious-free by construction** (the mimetic `span(d.B~)=span(B)` de Rham
  property, verified to ~1e-14; no stabilization parameter) ... the energy
  balance is **`n_orders`-INDEPENDENT** (no Fourier floor)" -- and already labels
  it "**corner-capped** (algebraic, no-floor -- at-best RCWA parity per DOF on
  vertical pillars, the win being accuracy quality)".

(Both quotes are from the `## [5.11.0] — 2026-06-02` section; they are cited by
entry name rather than line number because the CHANGELOG grows at the top.)

So the aspiration ("an FEEC E-D formulation", for a *nodal* discretization) was
superseded within its own release by a staggered one that met the goal.  The
roadmap phrase "FEEC yee2" survived as a name for an objective that had already
been achieved by a different construction.  It has also since been lifted to a
multilayer cascade, `stack2d_pure.PMM2DStackPure`.

**What did NOT ship** is `N-1`, non-uniform segment boundaries: `Basis1D` still
hard-codes `xb = np.linspace(...)` with a scalar jacobian `J = 0.5*d/N`
(`:172-174`), exactly as `PMM_M5_2D_FEASIBILITY_2026_08_04.md` found and
recommended GO.  S5 is why that, not FEEC, is where the remaining 2-D accuracy
work is.

## S2.  The floor, measured

### S2.1  The hybrid IS floored, and the floor is set by `n_orders`

The cleanest statement is a `(degree x n_orders)` grid.  Cell A, TE; the
no-floor staggered value for this grating is **0.000717496625**:

| `degree` \ `n_orders` | 5 | 9 | 13 | 17 |
|---|---|---|---|---|
| 7 | 0.0011394433 | 0.0008113877 | -- | -- |
| 9 | 0.0016780826 | 0.0008981164 | 0.0007810139 | -- |
| 11 | 0.0017125747 | 0.0010005909 | 0.0008115929 | -- |
| 13 | 0.0017192876 | 0.0012075020 | 0.0008295070 | 0.0007001860 |
| 15 | **0.0017205690** | 0.0012453343 | 0.0009534576 | 0.0007980291 |

Read **down** a column: raising the modal degree at fixed Fourier truncation
**converges to the wrong number** -- the `n_orders=5` column settles on
0.001721, a factor **2.4x** off truth, and settles harder the further down you
go (0.0016781 -> 0.0017126 -> 0.0017193 -> 0.0017206).  That is a floor in the
strict sense: a plateau the refinement knob cannot cross.  Read **across** a
row: only raising `n_orders` moves the plateau, and slowly.  (The two knobs are
also coupled -- `n_orders=17` at `degree<=11` refuses with a representability
`ValueError`, so `n_orders` cannot even be raised independently.)

### S2.2  The staggered path has NO Fourier floor -- measured, not asserted

`n_orders` swept 3/5/7/9/11 at fixed modal degree, i.e. the same solve read
through progressively larger Rayleigh sets:

| cell | pol | `M` | spread of `sum R` over `n_orders` |
|---|---|---|---|
| A | te / tm | 10 | **3.36e-18** / 4.34e-18 |
| B | te / tm | 10 | 2.89e-15 / 4.11e-15 |
| C | te / tm | 6 | 1.01e-16 / 6.94e-17 |

The answer is invariant to the Fourier truncation at round-off.  The floor the
milestone set out to remove is **already absent on this path**.

### S2.3  Energy closure, the two paths side by side

Over the full ladders (3 cells x 2 pols):

| path | `|R+T-1|` range |
|---|---|
| **staggered** (over `M` = 4..11) | **1.55e-15 .. 5.60e-13** |
| **hybrid** (over `n_orders` = 3..15, `degree` 11) | **4.32e-06 .. 1.05e-01** |

### S2.4  Where the methods land, at the largest truncation each reached (TE)

| cell | staggered | hybrid | RCWA-2D |
|---|---|---|---|
| **A** | 0.000717497 (`M`=11, dof/axis 20), closure 5.6e-13 | 0.000794132 (`no`=15), closure 2.5e-04, **dev 7.7e-05** | 0.000990744 (`no`=13), closure 4.9e-14, dev 2.7e-04 |
| **B** | 0.994643230 (`M`=11), closure 2.6e-13 | 0.979127737, closure 2.0e-04, **dev 1.6e-02** | 0.989807213, closure 1.8e-13, dev 4.8e-03 |
| **C** | 0.017092745 (`M`=7, dof/axis 24), closure 2.4e-14 | 0.032991910, closure **1.05e-01**, **dev 1.6e-02** | 0.014395989, closure 2.3e-14, dev 2.7e-03 |

**The independent arbiter confirms the staggered value is the limit.**  RCWA-2D
is monotone in value on all three cells AND its distance from the staggered
answer shrinks strictly monotonically, over `n_orders` 3/5/7/9/11/13:

| cell | RCWA `sum R`, `n_orders` 3 -> 13 | `\|RCWA - staggered\|`, 3 -> 13 |
|---|---|---|
| A | 0.002082 -> 0.000991 | 1.4e-03 -> **2.7e-04** |
| B | 0.950405 -> 0.989807 | 4.4e-02 -> **4.8e-03** |
| C | 0.006645 -> 0.014396 | 1.0e-02 -> **2.7e-03** |

A wholly independent method walking toward the staggered number from the far
side, on every cell and at every rung, is the strongest available evidence that
the no-floor path is right and that the two Fourier-truncated methods are the
ones still converging.  The hybrid is the worst of the three everywhere.

**A second, oracle-free readout of the hybrid's error.**  Lattice translation of
the unit cell is an EXACT symmetry of the efficiencies.  The same physical
grating solved with the pillar at two different cell origins moved the hybrid's
per-order efficiencies by **3.70e-05 (te) / 4.16e-05 (tm)** at `degree=9,
n_orders=7` -- a violation of an exact symmetry, requiring no reference value at
all.  The staggered path's exact-sidewall construction makes it position-
invariant by construction (M5 G6(c) measured a 9.1e-06 spread over four pillar
positions at `M`=6).

## S3.  What actually caps the staggered path -- the isolation experiment

The staggered ladder in `M` on the pillars is slow: successive steps at `M`~10
are still ~1e-6 (cell A: 3.1e-05, 2.7e-05, 6.1e-06, 7.0e-06, 1.7e-06, 2.5e-06,
6.7e-07 over `M` 4->11).  The milestone assumes such a residual is a
discretization defect.  The decisive test holds the discretization FIXED and
removes only the corner:

**Same solver, same basis, homogeneous cell (no corner), against the EXACT
analytic Airy oracle** (`R_exact = 0.1185383100545648`):

| `M` | dof/axis | abs error (te) | abs error (tm) |
|---|---|---|---|
| 4 | 6 | **3.05e-16** | 2.10e-15 |
| 6 | 10 | 4.86e-15 | 2.26e-15 |
| 8 | 14 | 2.53e-14 | 1.69e-14 |
| 10 | 18 | 1.07e-13 | 2.80e-14 |

**The error is at round-off from the smallest degree the basis admits.**  There
is no discretization error to remove: where the solution is smooth, this
discretization is exact, and the slow drift at larger `M` is round-off
accumulation, not truncation.

**A second isolation, varying the corner's STRENGTH.**  The control above
removes the pattern entirely, so it could be objected that it removes transverse
structure rather than the corner.  So: keep the pillar, keep the grid, keep the
ladder, and sweep only `eps_pillar` -- `eps_p -> eps_h` is the limit in which the
corner's singular amplitude vanishes while the cell stays patterned.  Relative
step `|x_M - x_{M-1}| / |x_M|` at `M` 8->9, TE:

| `eps_p` | `tau` | patterned (corner present) | homogeneous (no corner) |
|---|---|---|---|
| 1.01 | 0.0016 | **9.95e-09** | 3.10e-14 |
| 1.10 | 0.0152 | 9.76e-07 | -- |
| 1.50 | 0.0638 | 2.43e-05 | -- |
| 2.50 | 0.1375 | 3.31e-04 | -- |
| 4.00 | 0.1940 | **2.42e-03** | 4.74e-14 |
| 12.25 | 0.2791 | 3.01e-05 (see note) | 1.16e-13 |

With the corner present the residual climbs **monotonically across six decades**
as `tau` rises 0.0016 -> 0.194.  With the corner absent the SAME index contrasts
-- including `eps = 12.25` -- cost nothing at all: round-off, flat.  So the
residual tracks the corner, not the contrast and not the discretization.

*Note on the `eps_p = 12.25` row:* its `M` 7->8 step is 1.8e-09, a sign
crossing, so its local step at this rung is not a usable convergence proxy (the
cell sits near total reflection, `sum R` = 0.9947).  It is reported rather than
dropped; the monotone trend is read off the 1.01 -> 4.00 rows.

So the pillar's slow convergence is a property of the SOLUTION, not of the
operator or its cochain spaces.  Its name is the re-entrant-corner field
singularity, and the repo already ships the classifier for it --
`grating_convergence_class` (Li & Granet, *JOSA A* **28**, 738 (2011)), whose
published finding is that **FMM/RCWA, AMM and PMM alike** converge only
algebraically at such a corner (Type I), and that **no** modal method converges
at a lossless metal-dielectric one (Type II).  On these cells:

| cell | type | `tau` | the classifier's own advice |
|---|---|---|---|
| A (eps 4/1) | I | 0.1940 | "WEAK Type-I singularity ... use `elements_per_region>1, grade=True` to recover the rate" |
| B, C (eps 12.25/1) | I | 0.2791 | same |

**A FEEC/yee2 change cannot lift this.**  Exterior-calculus structure buys
exactness of the discrete complex -- spurious-mode freedom, correct kernels,
stable Hodge operators -- all of which this basis already has and demonstrably
uses.  It does not buy regularity the true field does not possess.  The
remedy the literature and the shipped classifier both name is **mesh grading
toward the corner** (hp-refinement), which is a MESH question.

## S4.  The lever, measured where it already exists (1-D)

2-D grading is exactly what `N-1` (non-uniform segments) would enable and is not
implemented.  In 1-D it ships, so the lever's size can be measured.  Binary
grating, n_ridge 3.5 / n_groove 1.0, duty 0.5, TM (the singular polarization),
corner Type I `tau` = 0.4232.  Reference: RCWA-1D at `n_orders`=251
(self-consistency vs 201: **6.33e-07**).

| dof | ungraded `epr=1` | `epr=4, grade=True` | `epr=8, grade=True` |
|---|---|---|---|
| 12-40 | 3.30e-04 -> 1.45e-05 | -- | -- |
| 48-160 | -- | 1.92e-05 -> 2.45e-06 | -- |
| 96-320 | -- | -- | 5.07e-06 -> 1.81e-06 |

Grading lowers absolute error substantially -- `epr=8` at dof 96 (5.07e-06) beats
ungraded at dof 40 (1.45e-05) by ~2.9x at 2.4x the dof, and reaches 1.81e-06
where ungraded is still at 1.45e-05.  **Rates are NOT claimed from this table:**
the graded arms flatten out at ~2e-06, within ~3x of the reference's own
6.33e-07 self-consistency, so those curves are measuring the ORACLE's floor and
their fitted exponents (1.78, 0.83) are artefacts of that saturation, not
convergence orders.  A rate comparison needs a reference two decades better than
the graded arms, which this experiment did not build.  The robust claim is the
absolute-error reduction, and the direction of the lever.

## S5.  GO / NO-GO

**NO-GO on Phase B as chartered.**  Reasons, in order:

1. **The deliverable exists.**  The FEEC/yee2 transverse discretization is
   `twod_staggered.py` (single layer, shipped v5.11.0) and
   `stack2d_pure.PMM2DStackPure` (multilayer cascade, first tagged v5.21.0 --
   `git tag --contains 07ca820`).  Building it again is rework.
2. **Its central claim is confirmed by independent measurement**, not merely
   documented: `n_orders`-invariance at 3e-18..4e-15, energy closure 1e-15..6e-13,
   and RCWA converging toward it from the far side on all three cells.
3. **The residual cap is not liftable by this class of change.**  The same
   discretization is exact to round-off on a corner-free cell at the smallest
   admissible degree (S3); what remains is the Li-Granet corner singularity,
   which is method-independent and a property of the solution.
4. **The real remaining lever is `N-1`** (non-uniform / graded segments), already
   analysed with an unconditional GO in `PMM_M5_2D_FEASIBILITY_2026_08_04.md`
   (de Rham residual flat at 8.9e-15 up to 1e6 segment ratio; byte-identical
   uniform fast path available; `cond(G)` quadratic in the segment ratio, cap
   recommended at 1e3).  That is a mesh item, not a formulation item, and it is
   where the next accuracy work on this path belongs.

**The one thing that would change this verdict** is a device class whose 2-D
accuracy is limited by something other than corners and other than the Fourier
truncation -- none of the three cells here is.

**Guidance gap -- narrower than it first looked, and half-fixed here.**  On cell
B the hybrid is 1.6e-02 from the no-floor answer and on cell C its energy
closure reaches 1.05e-01, while the staggered path answers the same gratings at
1e-13 closure.  The hybrid remains the right tool for what only it can do --
tapered/`z`-staircase stacks, full anisotropic tensors and out-of-plane coupling,
per-layer walls with no union-grid constraint, and the JAX twin.  Checked, the
routing is already largely handled:

* the **stack** path has a deliberate planned cutover -- `PMM2DStack` is a
  transitional alias that emits a `DeprecationWarning` naming
  `PMM2DStackPure` and is "scheduled to be repointed ... once that reaches
  feature + validation parity" (`stack2d.py:1843-1873`).  Nothing to add.
* the **single-layer function** path was the gap: `pmm_efficiency_2d`'s
  docstring already says it "has a Fourier-truncation floor ... *not* no-floor
  like **the 1-D PMM**" -- comparing itself to the 1-D solver while never naming
  `pmm_efficiency_2d_staggered`, the 2-D no-floor sibling that takes the same
  isotropic rectangular pillar.  A reader is told a floor exists but not where
  to go.  **Fixed here** by a docstring cross-reference carrying the measured
  numbers and the list of cases that genuinely require the hybrid.

## S6.  A silent-wrong defect the floor measurement turned up -- FIXED here

Cell C at `degree=11, n_orders=15` returns **`R+T = 0.895254364`** on a
**provably lossless** structure: a **10.5 % energy deficit**, per-order
efficiencies ~2x wrong (0.032992 against the staggered path's 0.017093), and
**zero warnings**.

The cause is a mismatch between a predicate and its own stated contract.
`_warn_lossless_energy_2d` (`twod.py:134`) first ESTABLISHES losslessness --
returning early if any permittivity has an imaginary part -- and its docstring
and warning text then both say "the structure is provably lossless so `R+T=1` is
exact".  But the test it applied was the **passivity window** inherited from
siblings:

```python
if (not (-_PASSIVE_TOL_2D <= tot <= 1.0 + _PASSIVE_TOL_2D)
        or eff_min < -_PASSIVE_TOL_2D):
```

`0.8953` sits inside `[-0.05, 1.05]`, so nothing fired.  The window catches
MANUFACTURED energy and negative efficiencies; it is blind to LOST energy, which
on a lossless structure is exactly as much a defect.  The siblings
(`PMMStack._warn_stack_energy`, RCWA `_check_energy(..., lossless=)`) legitimately
use a passivity window because they never establish losslessness and a lossy
structure may absorb -- this one does establish it, and then discarded the half
of the information it had just earned.  The docstring's own recorded probes
(`E=1.30`, `E=3.88`) are both on the excess side, which is why the gap survived.

**Fix** (`twod.py`): the predicate becomes the closure test its contract already
described, `abs(tot - 1.0) > _PASSIVE_TOL_2D or eff_min < -_PASSIVE_TOL_2D`, and
the message reports the signed deviation.  This is **strictly more detections** --
the `tot > 1+tol` and negative-efficiency arms are unchanged, so no previously
warning input stops warning and no working solve is altered (the guard only ever
warns; it never raises and never touches the returned arrays).  Both call sites
(`pmm_efficiency_2d`, `pmm_efficiency_2d_cell`) are covered by the one function.

**The bar is the SHIPPED `_PASSIVE_TOL_2D = 5.0e-2`, reused unchanged**, and it
has a measured gap on both sides.  Over a 136-solve lossless matrix (4 cells x 2
pols x `degree` {9,11,13} x `n_orders` {5,7,9,11,13,15}), measured 2026-08-17:

| quantity | value |
|---|---|
| median `\|R+T-1\|` | 3.55e-05 |
| 90th percentile | 5.95e-04 |
| 99th percentile | 3.26e-03 |
| **max over all CLEAN solves** | **4.33e-03** |
| gross violations (`> 5e-2`) | **1 of 136** -- and it is on the **deficit** side |

So the bar sits **11.5x above** the worst clean solve in the matrix and **2.1x
below** the observed defect.  The lower gap is comfortable; the upper gap is
narrow and is stated as measured rather than as decades -- the tolerance was not
chosen here, it is the one already shipped for the other side, and this change
only extends it symmetrically.

**Tests** -- `tests/unit/test_pmm2d_lossless_closure_two_sided.py`, 14 tests,
built to `docs/TESTING_STANDARDS.md`:

| claim | how it is made build-free |
|---|---|
| gross closure violation warns on BOTH sides | the state is **synthesized** through the guard's own interface (an exact `(orders, R, T)` triple summing to a chosen total), so the deficit claim does not depend on the build reproducing the pathological solve |
| the deficit is the arm that regressed | the test evaluates the OLD and NEW predicates side by side on the measured 0.895254364 and asserts old-misses / new-catches -- it carries its own refutation and cannot go vacuous |
| clean solves stay silent | asserted inside the tolerance, both signs |
| lossy inputs still skipped | a complex eps with `R+T = 0.5` and with the measured 0.8953 must NOT warn |
| negative per-order arm unchanged | asserted directly |
| the guard is wired to real solves | an INVARIANT over a `(degree, n_orders)` ladder: `warned == predicate(R+T, eff_min)` on every arm, read from **this build's own** numbers, so LAPACK spread moves both sides together and cannot move the boundary; the ladder asserts `>= 8` surviving arms so it cannot collapse to nothing |

No `pytest.skip` anywhere.  Representability `ValueError`s are `continue`d (they
are a different contract), not skipped over as passes.

## S7.  What is NOT claimed

* **No convergence ORDER is claimed for the staggered path on pillars.**  The
  ladders are short and the successive differences are non-monotone (cell B has a
  1.8e-09 coincidence between `M`=7 and `M`=8 that wrecks any fit).  The claims
  made are the `n_orders`-invariance, the energy closure, and the agreement with
  the RCWA arbiter -- all of which are direct readings, not fits.
* **No rate claim from the 1-D grading table** (S4) -- it is oracle-limited
  below ~2e-06, and that is stated where the table sits.
* **`tau` is not claimed to be a directly-measured exponent here.**  It is the
  published classifier's prediction and is cited as the mechanism's name and the
  reason the cap is method-independent, not as a fitted quantity.
* **Nothing about the hybrid's physics is changed.**  The only library change is
  a warning predicate; no solver path, no returned array, no default moves.
* The corner cap applies to right-angle pillars.  A smooth-field region converges
  spectrally on this basis (S3 measures exactly that), which is why the cap is a
  statement about rectangular pillars and not about the method.

## S8.  Files

| file | change |
|---|---|
| `lumenairy/elements/pmm/twod.py` | `_warn_lossless_energy_2d`: passivity window -> two-sided closure test; signed deviation in the message; the finding recorded in the docstring.  Separately, `pmm_efficiency_2d`'s docstring now cross-references the no-floor sibling with the measured gap (S5) |
| `tests/unit/test_pmm2d_lossless_closure_two_sided.py` | NEW -- 14 tests |
| `docs/audits/EXPERIMENT_PMM2D_FEEC_2026_08_17.md` | this document |
| `CHANGELOG.md` | `[Unreleased]` |

No new module, no new layer type, no public API change -- by design: the
capability the milestone asked for was already present, and the measurement
said so before any of it was written.

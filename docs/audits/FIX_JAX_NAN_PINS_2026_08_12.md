# The W-only JAX NaN, and the two pins S9 left on the watch list -- 2026-08-12

Branch `fix/jax-nan-pins` off `origin/main` (21802f9, the PR #32 merge).

Two items, from `docs/audits/FIX_RUNNER_PINS_2026_08_12.md` S9.

1. A **library defect**.  `test_niche_audit_w9_eig_vjp.py::
   test_pmm1d_off_normal_angle_gradient_no_regression` reads
   `jax.grad(...)(0.3) = nan` on WSL and nowhere else, pre-existing on main.
   It is NOT the eig-VJP floor this file is about; it is the incident-amplitude
   projection, which was the last place the differentiable 1-D PMM still went
   through `jnp.linalg.lstsq`.  Fixed in `lumenairy/`.
2. Two **lucky pins** (S9's first two watch items), re-stated with both-mount
   evidence and an injector each.  Test-side only.

Mounts throughout: **M** = Windows py3.14, `numpy 2.4.4`, `jax 0.11.0`,
OpenBLAS 0.3.31; **W** = WSL py3.12, `numpy 2.4.6`, `jax 0.10.2`, OpenBLAS,
driven at `OPENBLAS_NUM_THREADS` 1 / 2 / 4 / 8.

---

## S1.  What was measured, before anything changed

| | reading |
|---|---|
| `jax.grad(_pmm1)(0.3)` [W] | `nan` (primal 0.018263228330731154, finite) |
| `jax.grad(_pmm1)(0.3)` [M] | 0.036304694697444906 |
| `test_tapered_grating_shear` energy, sheared arm | 3.34e-13 [M] / 1.19e-11 [W] rel, against an ABSOLUTE 1e-7 whose own comment records a CI runner at 4.5e-8 |
| `test_o7_..._in_the_degenerate_fallback` | element-wise 1.05e-13, bar 1e-11, on GENERAL complex `Delta` |

---

## S2.  Item 1 -- localisation

`jax_debug_nans` on the failing call, op by op, names the site exactly:

```text
  File ".../lumenairy/elements/pmm/_core.py", line 3653, in _jpmm_solve
    cinc = jnp.linalg.lstsq(Hsup, delta0, rcond=None)[0]
  File ".../jax/_src/numpy/linalg.py", line 1528, in lstsq
    return _jit_lstsq(a, b, rcond)
jax._src.source_info_util.JaxStackTraceBeforeTransformation:
FloatingPointError: invalid value (nan) encountered in add
```

The handoff's hypothesis was a `jnp.where` whose dead branch evaluates NaN at
the knife edge.  It is the same SHAPE of defect -- a guard that covers one case
and not the neighbouring one -- but it is not in `lumenairy` and it is not a
`where`.  `jnp.linalg.lstsq` differentiates through an SVD, and jax's SVD JVP
(`jax/_src/lax/linalg.py`, `_svd_jvp_rule`) carries

```python
    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = lax._eye(s.dtype, (s.shape[-1], s.shape[-1]))
    #   ^ the "1. where s_diffs is 0., 0. elsewhere" expression this replaced
    #     is commented out one column to the right, in the shipped source
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
```

so `F_ij = 1 / (s_i^2 - s_j^2)` is guarded on the DIAGONAL only.  An
off-diagonal tie divides by zero; a near-tie divides by round-off.

**The projection is structurally degenerate.**  `Hsup = Tp @ Wsup` -- the
uniform superstrate's nodal modes projected onto the Rayleigh orders -- is
21 x 24 on this probe (degree 12, `far_field_orders` 21), and three of its
twenty-one singular values sit on `1 / sqrt(n_glob)`:

```text
theta = 0.3, the cluster (entries 2..7 of the 21 descending singular values)
  [M] 0.20413548193688222 0.20412414524886170 0.20412414523193226
      0.20412414523193210 0.20412414523193190 0.20412414520754504
  [W] 0.20413548193688216 0.20412414524886180 0.20412414523193212
      0.20412414523193210 0.20412414523193198 0.20412414520754500
  1 / sqrt(24) = 0.20412414523193154      cluster spread 3.6e-16 [M] / 1.4e-16 [W]
```

So the splitting of the tied pair is pure round-off -- so pure that the two
LAPACKs in ONE process disagree about whether it exists at all: at theta = 0.3
on W, numpy's `gesdd` returns the pair EXACTLY equal while jaxlib's returns it
one ulp apart.  `F` is therefore decided by the build.  Measured across the
angle from jax's own SVD (the one the VJP uses), in ULP of `s^2`:

```text
theta   [W] splitting  F_max      grad      [M] splitting  F_max      grad
0.20    395 ulp        3.65e+14   finite    388 ulp        3.71e+14   finite
0.29    229 ulp        6.29e+14   finite    247 ulp        5.83e+14   finite
0.30      1 ulp        1.44e+17   nan         9 ulp        1.60e+16   finite
0.31     24 ulp        6.00e+15   finite     28 ulp        5.15e+15   finite
0.40   4069 ulp        3.54e+13   finite   4066 ulp        3.54e+13   finite
0.50     56 ulp        2.57e+15   finite     46 ulp        3.13e+15   finite
```

That is the whole "knife edge at the exact double 0.3": nothing about 0.3 is
special except that on ONE build the two round-off images of the same
structural eigenvalue land one ulp apart.  It is not the eig-VJP floor --
`_EIG_TAU_REL` at 1e-12 / 1e-10 / 1e-8 / 1e-6 all still gave `nan`, as the
handoff recorded -- and the primal never touches `F` at all, which is why it
stayed finite.

The defect is OBLIQUE-only: at a python-literal `kx0 = 0.0` the convection is
skipped, the cluster is absent, and the minimum relative gap is 5.442110e-04 on
both mounts -- six decades away.

---

## S3.  Item 1 -- the fix, and why it is the in-tree one

**Adjudicated: a wrong answer, not an alternative evaluation.**  `nan` is not a
number a different BLAS is entitled to return for a gradient that exists.  The
map being differentiated is smooth; only the DECOMPOSITION jax chose to
differentiate it through (singular vectors) is non-differentiable at a repeated
singular value.  So this licenses a library change.

**It is a copy that was left behind, not a new problem.**  Three places in-tree
already record it, and two of them are the same `Hsup`:

* `_jpmm_jones_solve` (`pmm/_core.py`, v5.18) -- *"``jnp.linalg.lstsq``'s VJP
  NaNs on a rank-deficient / under-determined system ... use the closed-form
  min-norm pseudo-inverse"*;
* `_jax_stack.py` -- the same replacement, same comment, on the stacked Jones
  projection;
* `propagators/asymptotic_jax_twin._differentiable_lstsq` -- *"an SVD-based
  gradient whose formula has a ``1/(s_i^2 - s_j^2)`` term that returns NaN when
  any two singular values are (near-)degenerate"*, and it names the same
  mechanism this document measures.

The SCALAR 1-D twin was never converted.  That is the campaign's standing
multi-copy disease -- one copy behind -- and the fix is the sibling's:

```python
    underdetermined (m <= n):  x = A^H (A A^H)^-1 b     (min-norm)
    overdetermined  (m >  n):  x = (A^H A)^-1 A^H b     (least-squares)
```

now in one named function, `_jpmm_min_norm_projection`, with the mechanism in
its docstring so the next copy has somewhere to point.  `inv` differentiates
through `-A^-1 dA A^-1`: no singular-value differences anywhere, finite for any
invertible Gram however degenerate its spectrum.  The shape branch is on the
CONCRETE shape (static per trace), so it is host control flow and traces.

Cost: one squaring of the condition number.  Measured `cond(Hsup) = 5.196` at
theta 0.3 (5.188 at 0.29, 5.524 at 0.0), so the Gram runs at 27.

`PMM_JAX_MINNORM_PROJECTION = False` restores `jnp.linalg.lstsq` bit for bit --
the campaign's `PMM_FORWARD_GROWTH_REPAIR` idiom -- so the fail-before drives
the shipped code and not a copy of it.

### S3.1  Correctness, against an oracle independent of the fix

The oracle is a RICHARDSON-extrapolated central difference,
`(4 D(h/2) - D(h)) / 3` at `h = 3e-4`, which cancels the `O(h^2)` truncation the
plain central difference in the test carries.  It touches only PRIMALS, which
the fix moves by 3e-15 relative, so it is independent of the change.

```text
        AD (fixed)             AD (pre-fix)         FD oracle              rel [fixed]
[W] 0.20  -0.2963576441934857   -0.29635764419348454  -0.2963576405112202   1.243e-08
[W] 0.29   0.0301588047709468    0.030158804770946908  0.03015880476867265  7.541e-11
[W] 0.30   0.03630469469744451   nan                   0.03630469469617146  3.507e-11
[W] 0.31   0.040497833669218296  0.04049783366922037   0.040497833673507226 1.059e-10
[W] 0.40  -0.02589265112007204  -0.025892651120071455 -0.025892651114790527 2.040e-10
[W] 0.50   0.07246657818191539   0.07246657818191997   0.07246657817762173  5.925e-11

[M] 0.20  -0.2963576441934885   -0.29635764419348815  -0.2963576405112241   1.243e-08
[M] 0.29   0.0301588047709456    0.030158804770945898  0.030158804773061505 7.016e-11
[M] 0.30   0.03630469469744432   0.036304694697444906  0.03630469469664177  2.211e-11
[M] 0.31   0.04049783366921816   0.040497833669220246  0.040497833672747806 8.716e-11
[M] 0.40  -0.02589265112007149   -0.02589265112007133 -0.025892651115070008 1.932e-10
[M] 0.50   0.07246657818191658    0.07246657818192057  0.07246657817664642  7.273e-11
```

Three things at once.  The fixed value at 0.3 is CORRECT (3.5e-11 against the
oracle); it is the same number the healthy mount already returned (5.2e-15
cross-mount); and everywhere the pre-fix route was finite the two agree to
<= 1.6e-14 relative, so the fix is BYTE-NULL in effect off the knife edge --
it removes the NaN and changes nothing else.  (The 1.243e-08 row at theta 0.2
is the oracle's own residual truncation, identical in both AD columns.)

The shipped test keeps its own plain central difference at `h = 3e-4` and its
`1e-4 * |FD|` bar (12x on the 8.2e-06 that difference's truncation produces);
the Richardson arm above is the ADJUDICATION, not a new assertion, and it
sharpens the verdict on the same number by five decades.

The NumPy path is untouched by construction (the JAX branch fires only on JAX
inputs) and keeps `_guarded_lstsq`, which owns the M1 conditioning guard.

### S3.2  Fail-before

`test_the_lstsq_projection_route_is_refuted_on_a_degenerate_projection`, three
claims, none of which needs one build's round-off reproduced:

* **exposure**, read off the SHIPPED projection captured from a live primal
  solve: minimum relative singular-value gap 8.0e-16 [M] / 1.3e-16 [W] at
  theta 0.3 and 2.0e-14 / 1.9e-14 at 0.29 (bar 1e-10), the tied value is
  `1 / sqrt(n_glob)` to 1e-12 and at least three singular values sit on it, and
  the normal-incidence branch does NOT (5.4e-04, bar 1e-6 the other way);
* **mechanism**, deterministic on every build: on `0.7 * eye(21, 24)` all 420
  off-diagonal `s_i^2 - s_j^2` are identically 0, the pre-fix route's gradient
  is not finite on BOTH mounts, and the shipped route's is finite AND right
  (against a central difference of the same loss along a fixed direction);
* **forward nullity**: the two routes return the same minimum-norm solution --
  7.3e-15 on the injector, 3e-15 on the live oblique projection -- so what
  changed is the gradient, not the answer.

### S3.3  Not fixed here

`lumenairy/elements/_lens_jax.py:124` is the one remaining `jnp.linalg.lstsq`
on a differentiated path.  It is a different module with no red test and no
measured degeneracy, so it is out of scope; it is recorded here because the
grep that found it is the one worth re-running after any jax bump.  The three
existing closed-form copies (`_jpmm_jones_solve`, `_jax_stack`, and now
`_jpmm_min_norm_projection`) are candidates for a single shared helper -- left
alone deliberately, since routing the first two through it would move their
bits for no correctness gain.

---

## S4.  Item 2a -- the shear pin's absolute energy bar

**The claim** is that a lossless staircase conserves energy with or without
shear.  It was scored at an ABSOLUTE `1e-7` on a total of 2.0, and its own
comment records a CI runner closing this 8-slice staircase at 4.5e-8 -- 2.2x
headroom, which is the lucky-pin shape S9 flagged.

**Adjudicated, and the adjudication changes how tight the bar may be.**  Energy
closure on a LOSSLESS cell is the campaign's known-weak instrument (the
lossless trap: a lossless cascade auto-balances power even when the per-order
split is wrong).  Measured here it does not even see under-resolution:

```text
n_orders    relative closure [M]    [W]
2           4.992e-12               4.904e-11
4           1.150e-10               9.794e-11
6 (shipped) 3.337e-13               1.192e-11
10          7.729e-12               5.314e-10
```

Four settings spanning a converged and a badly truncated solve, and the closure
tells them apart not at all.  This assertion never was the discriminator
between a resolved and an unresolved staircase -- the per-order +/-1 symmetry
claims below it are.  What it is for is a GROSS flux defect, and a bar 2.2x
above a build-dependent round-off reading cannot deliver that without
eventually crying wolf.

**Re-stated relative**, the pin-5 treatment: the residual is scored against the
total it is a residual OF.

```text
arm            [M] Windows py3.14      [W] WSL py3.12
vertical       2.665e-15               2.998e-15
shear +0.3     3.337e-13               1.192e-11
shear -0.3     4.684e-13               1.663e-13
CI ubuntu      2.25e-8   (the 4.5e-8 absolute its old comment recorded)
```

Every mount reading is IDENTICAL at `OPENBLAS_NUM_THREADS` 1 / 2 / 4 / 8, so
the axis here is the LAPACK build and not the reduction width -- which is why
the fix is a magnitude re-scaling and not a census partition.
`_SHEAR_ENERGY_REL = 1e-6` sits 44x above the worst reading any build has
produced and 13x below the smallest defect proved to fire on it.

(An ARM RATIO was considered and rejected on the measurement: the vertical arm
closes 2-3 decades better than the sheared one, and by different factors on the
two mounts -- 125x [M] against 3976x [W] -- so a ratio bar would be more
build-fragile than the magnitude it replaced, not less.)

**Fail-before** (`test_tapered_grating_shear_energy_envelope_still_sees_a_
lossy_ridge`): a LOSSY ridge, which makes `R + T < 2` by the absorbed fraction
-- a real, controllable flux defect, and one whose readings are
build-independent to seven figures, which is exactly what the round-off they
replace is not:

```text
Im(eps_ridge)   relative closure [M]   [W]
1e-6            1.340145e-06           1.340145e-06
1e-5            1.340149e-05           1.340149e-05
1e-4            1.340187e-04           1.340187e-04
```

Three claims: the shipped lossless arm clears the envelope by at least a decade
(asserted; measured 1.2e-11 [W] / 3.3e-13 [M], i.e. four to seven decades), the
1e-5 injection breaks it (13x over), and the ladder is LINEAR in the injected
loss (bar 1e-3 relative on the ratio, measured 3e-6) -- so the bar's position is
a physical threshold and not a coincidence of one cell.

RESIDUAL, recorded rather than hidden: a mis-assembled cascade that leaks at
~2e-7 relative would now pass this assertion.  It would have passed the old
absolute bar too (1e-7 absolute is 5e-8 relative -- tighter, but only by a
factor of 20 while sitting 2.2x from a false alarm), and the M2 record is
explicit that such a cascade is caught by per-order comparison, not by closure.

---

## S5.  Item 2b -- the o7 degenerate fallback

**The claim** is S1-13's, on the branch the fix was actually written for: when
NOT exactly two modes flag forward, `berreman._split_fwd_bwd` and
`_berreman_jax._layer_modes_jax` must still agree.  It was an ELEMENT-WISE
comparison across the two `eig` backends at exact bit-identity (`worst 0.0`),
on GENERAL complex `Delta` -- i.e. on the inputs where two LAPACKs are MOST
likely to differ in raw order.  It is pin 2's exposure with the luck still
holding.

**Adjudicated, and the treatment is NOT its sibling's.**  Pin 2 re-stated the
physical family on partition-INVARIANT observables, because there a column
re-ordering leaves the partition alone (measured 5.618e-15 either way).  Here
it does not.  When the flag count is not two, the partition is decided by a
stable argsort over the RAW ORDER, so the raw order IS part of the partition --
that is the branch's definition.  Measured, on both mounts:

```text
injector on the eig the twin is handed   corr    invariants   elementwise
shipped                                  57/57   6.464e-15    1.049e-13
raw-order permutation [1,0,3,2]           0/57   1.600694     1.521e+01
```

1.6007 against pin 2's 5.618e-15.  Invariants are the wrong object here, and a
re-statement built on them would have been vacuous.

**Re-stated in three layers instead.**

* the PRECONDITION, per draw -- both backends solved the same eigenproblem,
  compared by POWER SUMS `p1..p4` rather than by sorting (this family puts
  modes at `Re(gam) = +-0`, where a lexicographic sort ties, which is where
  this file's spurious 0.74 drift came from).  Measured 0.0, bar 1e-7;
* the RULE, on EVERY draw -- the JAX twin is handed numpy's OWN raw
  decomposition, so both implementations partition the same spectrum in the
  same order and what remains is purely "do the two rules agree", with no
  LAPACK left in it.  Measured EXACTLY 0.0 on both mounts, bar 1e-11.  This is
  the layer that keeps the claim alive when correspondence is lost;
* the COLUMN ORDER, on the draws where the shipped backends returned the raw
  spectrum in the same order -- the original element-wise comparison, verbatim.
  Measured 1.049e-13 over 57 of 57 draws on both mounts, bar 1e-11.  (The
  residual is the two `Delta` assemblies, not the partition: the rule layer,
  which shares one `Delta`, reads 0.0.)

The correspondence count is asserted non-empty rather than assumed, so the
third layer can never pass by being vacuous.

**Fail-before** (`test_o7_degenerate_claim_survives_a_reorder_and_still_catches_
a_rule_fork`), both directions:

* a raw-order PERMUTATION on the `eig` the twin is HANDED (it takes it as a
  parameter, so nothing is monkeypatched) must empty the correspondence class
  (57 -> 0), knock the ORIGINAL verbatim claim over (1.521e+01, the O(1) a
  divergent LAPACK would produce), and leave the RULE layer exactly where it
  was;
* the PRE-S1-13 DECAY RANKING -- the fork the fix removed, reconstructed in the
  test from the shipped eigendecomposition -- must break the RULE layer:
  measured worst 4.736e+01 over 41 of the 57 degenerate draws, on both mounts.

---

## S6.  Verification

Both mounts, `pytest -q -p no:randomly`, the three touched modules -- 480
collected, all passing, against 477 on `origin/main`:

| module | on main | here | delta |
|---|---|---|---|
| `test_niche_audit_w9_eig_vjp.py` | 29 | 30 | +1 |
| `test_niche_audit_w6_berreman.py` | 428 | 429 | +1 |
| `test_v5_14_1_rcwa_deferred.py` | 20 | 21 | +1 |
| **total** | **477** | **480** | **+3** |

The +3 are one fail-before per change.  Nothing is xfailed, nothing is skipped,
no test is deleted, and the three target ids keep their names.

**M**: 480 / 480 pass.  **W**: 480 / 480 pass -- against `origin/main` on the
same mount, where `test_pmm1d_off_normal_angle_gradient_no_regression` fails
(28 passed / 1 failed in `test_niche_audit_w9_eig_vjp.py`, same assertion, same
`nan`).  That is the fail-before for item 1: the red test on the branch's base,
green on the branch, with the library fix as the only difference.

`ruff check` clean on all four touched files on both mounts (line-length 100,
E/F/I).

Adjacent PMM / JAX suites re-run on M as a blast-radius check on the library
change (`test_v5_12_0_pmm_autodiff.py`, `test_audit_w3_pmm_jax_guards.py`,
`test_niche_audit_w7_pmm.py`, `test_v5_12_0_pmm_jones_autodiff.py`): **304
passed**, with 4 pre-existing `_EnergyWarning`s from `pmm_jones_2d` -- the 2-D
Jones path, which this branch does not touch.

Cost: the two o7 tests together run 7.7 s on an idle W box (the restructured
claim plus its fail-before, sharing one `_o7_degenerate_score` pass per arm --
the fail-before's "how many draws fork" count is read off the same pass rather
than re-walking the family).  `w6 + rcwa` together: 36.2 s [M] / 52.1 s [W].

---

## S7.  Watch items

* `test_tapered_grating_shear`'s two SYMMETRY bars (`< 1e-12` on magnitudes of
  order 0.334) are pin 5's territory on `fix/runner-pins` and are NOT touched
  here; on `origin/main` they are still absolute and still the runner-red pin
  S6 of that document treats.  The two branches edit the same function and will
  conflict; the merge is mechanical (this branch touches only the two energy
  lines and adds one test).
* `_o7_partition_invariants` and `_o7_draws` land on `fix/runner-pins` for the
  PHYSICAL family.  Nothing here duplicates them -- this branch's helpers are
  the general-`Delta` family and a rule arm the physical family does not need
  -- but the two files will conflict textually in the same region.
* `lumenairy/elements/_lens_jax.py:124` -- see S3.3.
* jax's `_svd_jvp_rule` off-diagonal guard is commented out upstream.  If it is
  ever restored, `PMM_JAX_MINNORM_PROJECTION = False` becomes safe again; the
  closed form is still preferable (it costs no SVD).

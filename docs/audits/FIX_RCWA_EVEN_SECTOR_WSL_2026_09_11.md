# The RCWA modal branch-cut defect behind the WSL-only even-sector failure

`tests/unit/test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full`
read `dR = 1.625072e-02` on the WSL build and `2.812111e-10` on the Windows
build from byte-identical code, with LAPACK printing
`** On entry to DLASCL parameter number 4 had an illegal value` alongside.
This document records what was measured, what the mechanism turned out to be,
what changed, and what the change costs.

Branch `fix/rcwa-even-sector-wsl`, cut from `48c8747` (the `wave2/pmm2d` tip).
Probes and their JSON, both builds, both arms:
`validation/probe_fix_rcwa_even_sector_wsl/`.

---

## 1. Terms

**Layer eigenproblem.**  RCWA writes one grating layer's transverse fields as
`d/dz [E; H] = ...`, condenses it to `M = P Q`, and eigendecomposes `M`.  Each
eigenvalue is `lam^2`; the modal decay constant is `lam = sqrt(lam^2)`, and the
layer's forward amplitudes propagate through thickness `L` as
`X = exp(-lam k0 L)`.

**The two roots.**  `lam^2` has two square roots, `+r` and `-r`.  For an
EVANESCENT mode `lam^2` is real POSITIVE and the choice is between a decaying
and a growing propagator -- the principal branch (`Re(lam) >= 0`) settles it.
For a PROPAGATING mode of a LOSSLESS layer `lam^2` is real NEGATIVE, i.e.
exactly ON the principal square root's branch cut, and the two roots are
`+i|kz|` (OUTGOING) and `-i|kz|` (INCOMING).  Nothing about the magnitude
distinguishes them; only the sign of `Im(lam^2)` does, and on the cut that sign
is `+0.0` or `-0.0`.

**Which root the region uses.**  A homogeneous half-space's modes are built by
`_homogeneous_eigenmodes` in EXACT arithmetic: `kz = _sqrt_forward(eps - k^2)`
and `lam = _sqrt_decay(-kz^2)`.  For the fixture's substrate that yields
`lam = +1.5000j` for the (0,0) order and `+0.9000j` for the (+-1,0)/(0,+-1)
orders.  So the library's convention for a forward propagating mode is
`Im(lam) >= 0`, and a LAYER mode must be on the same branch or the two media
cannot be matched.

**Interface mode-match.**  Joining media `a -> b` uses `a = Wb^-1 Wa`,
`b = Vb^-1 Va` and inverts `a + b` EXPLICITLY (`S12 = 2 (a+b)^-1`).  `a + b` is
singular exactly when some combination of `a`'s FORWARD modes reproduces one of
`b`'s BACKWARD modes -- when the interface cannot tell the two apart.

**Even-parity fold.**  At normal incidence with a centro-symmetric cell every
operator commutes with the order flip `G = blockdiag(J, J)`, so the whole
recursion can run in the `(N+1)`-dimensional even sector instead of the full
`2N`.  It is an exact change of basis: it must agree with the full solve to the
arithmetic floor.  The failing test compares the two.

**The fixture.**  `eps` = 2.25 background carrying a centred square of uniaxial
material, `no = 1.5`, `ne = 1.7`, optic axis rotated 0.7 rad;
`n_substrate = 1.5`, `n_superstrate = 1.0`, `n_orders = 5 x 5` (`N = 121`,
`2N = 242`, even sector 122).  Note `no^2 = 1.5^2 = 2.25`: the layer
background, the block's `zz` component and the SUBSTRATE permittivity are all
EXACTLY 2.25.

---

## 2. Reproduction: deterministic per build; the discriminator is the BLAS thread count

The failure reproduces outside pytest, and it is DETERMINISTIC: five repeats of
the lifted test body in one process return IDENTICAL DIGITS on all four
(build, tree) combinations (`r1_repro.py`) --

```
win  prefix(48c8747)  deterministic=True  dR=2.812111e-10
win  branch(fix tip)  deterministic=True  dR=2.775558e-16
wsl  prefix(48c8747)  deterministic=True  dR=1.625072e-02   FAIL
wsl  branch(fix tip)  deterministic=True  dR=4.163336e-17
```

-- so "flaky" was never the right word for it: each build computes one number,
every time.  What moves the number is the BLAS reduction order.

`r4_thread_sweep.py`, pre-fix (= `48c8747`), `max|R_full - R_even|`, bar `1e-8`:

| `OPENBLAS_NUM_THREADS` | Windows py3.14 | WSL py3.12 |
|---|---|---|
| 1        | 2.812111e-10 | **1.625072e-02  FAIL** |
| 2        | 3.053856e-12 | 3.652e-12 |
| 3        | -            | 3.863e-12 |
| 4        | 7.727846e-14 | 1.952e-13 |
| 6        | -            | 1.590e-12 |
| 8        | 1.833488e-11 | 8.367e-12 |
| 16       | 5.513e-14    | 1.749e-12 |
| unpinned | 2.006728e-14 | 7.370e-12 |

Twelve decades of spread on ONE quantity across settings a build is entitled to
choose, with the `1e-8` bar sitting inside it.  Both reported readings are
reproduced exactly at one thread.  The lossless-closure defect of the same
solves moves with it, and CHANGES SIGN: Windows `-3.200e-03`, `-1.003e-03`,
`-1.084e-06`, `+1.264e-03`, `+4.939e-06` at 1 / 2 / 4 / 8 / 16 threads.

**Why release CI was green on `9af9376`.**  `.github/workflows/unit-tests.yml`
pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` for the `slow-tests` and `jax-unit` jobs
and, in its own words, "deliberately does NOT pin BLAS at run time" for the
fast `unit` lane that owns this file.  Unpinned readings are 1e-12 .. 1e-14 --
under the bar.  CI's greenness was a threading accident, not a version
difference: the environments are

| | python | numpy | scipy | BLAS |
|---|---|---|---|---|
| Windows build | 3.14.6 | 2.4.4 | 1.17.1 | scipy-openblas 0.3.31.188.0, DYNAMIC_ARCH, **Haswell**, MAX_THREADS=24 |
| WSL build | 3.12.3 | 2.4.6 | 1.17.1 | scipy-openblas 0.3.31.188.0, DYNAMIC_ARCH, **SkylakeX**, MAX_THREADS=64 |
| CI `unit` lane | 3.10-3.13 | ubuntu wheels | ubuntu wheels | scipy-openblas, unpinned threads |

Same OpenBLAS version, different dispatched kernel and thread count.  The
discriminator is the reduction order, exactly as `docs/TESTING_STANDARDS.md`
says ("the discriminator is the wheel's LAPACK, not the interpreter version").

**Pre-existing, not introduced on wave2.**  `_sqrt_decay` is byte-identical at
`9af9376` (published v5.44.0) and at `48c8747`, and the test fails at
`9af9376` on WSL at one thread and passes there unpinned -- see §7.

**The `DLASCL` line is a bystander.**  Three things say so.  (i) In the
original 55-minute WSL suite log the two `DLASCL` lines sit AFTER pytest's
`1 failed, 1236 passed ... in 3329.96s` summary -- they are buffered Fortran
unit-6 output flushed at interpreter exit, so their position beside this test's
FAILED line is an artifact of buffering, not attribution.  (ii) Running this
test alone on WSL -- pinned and unpinned, at `48c8747` and at `9af9376`, pre-
and post-fix -- emitted no such line in any run.  (iii) `r2_lapack_trace.py`
wraps every `numpy.linalg` entry point the solve reaches and finds ZERO
non-finite inputs or outputs in all 10 calls on either build pre-fix (the
largest output is `2.08e+14` on WSL / `2.77e+14` on Windows -- the explicit
inverse of the singular `a + b`, large but finite).  Nothing in this solve hands
LAPACK a NaN.  Where the `DLASCL` line comes from is NOT established here.

---

## 3. The measurement that localised it

`r3_even_internals.py` (Windows, 1 thread, pre-fix) instruments the operators
the solve actually builds:

| | full `2N` | even `N+1` |
|---|---|---|
| `cond(M)` (layer system matrix) | 1.597e+02 | 1.590e+02 |
| `cond(W)` (eigenvector matrix) | 5.634e+03 | 3.929e+03 |
| `cond(a+b)`, superstrate -> layer | 1.541e+04 | 9.964e+03 |
| `cond(a+b)`, layer -> substrate | **1.973e+15** | **1.074e+15** |
| `cond(a-b)`, layer -> substrate | (not captured) | 4.987e+17 |
| singular values of `a+b` below `1e-10 * s_max`, layer -> substrate | 4 of 242 | 2 of 122 |
| lossless closure defect | -3.1995e-03 | -3.3813e-04 |

(`cond(M)` / `cond(W)` / the even-arm `cond(a-b)` from `r3_even_internals.py`;
the full-arm interface numbers and the singular-value counts from
`r7_nullvector.py`, which patches the `twod` binding as well so the full path's
interfaces are seen too.)

The layer eigenproblem is WELL conditioned.  The singularity is CREATED at one
interface -- the one against the substrate, and only that one.

`r7_nullvector.py` then takes the smallest right singular vector of that
`a + b`.  Its inverse participation ratio is 2.27 (full) and 1.11 (even), i.e.
it is essentially ONE mode, and the modes carrying it have

```
lam = 0.0000 - 1.5000j        lam = 0.0000 - 0.9000j        lam = 0.0000 - 0.9106j
```

against the substrate's own `+1.5000j` and `+0.9000j`.  Same magnitudes,
opposite sign.  The interface is being asked to distinguish a forward layer
mode from a backward substrate mode that are the same wave.

`r8_branch_cut.py` reads the offending eigenvalues out directly:

```
idx 19   lam^2 = -2.249999999999987 -2.911e-15j    lam = +9.704e-16 -1.499999999999996j
idx 14   lam^2 = -2.430648695704599 -2.740e-16j    lam = +8.786e-17 -1.559053782171930j
idx 13   lam^2 = -0.829190880912066 -1.604e-18j    lam = +8.810e-19 -0.910599187849443j
```

`lam^2 = -2.25` is the substrate's own `-kz^2` for the (0,0) order, to 1.4e-14.
`-0.8292` is the (+-1,0)/(0,+-1) orders'.  The layer HAS those modes -- its
background permittivity is the substrate's -- and it was handed the incoming
root for them.

---

## 4. The mechanism

`_sqrt_decay` pinned the outgoing root with

```python
on_cut = r.real == 0
return xp.where(on_cut & (r.imag < 0), -r, r)
```

an EXACT floating comparison.  For `lam^2 = -s + i eta` the principal root is

```
r = eta / (2|r|)  +  i sign(eta) sqrt(s)
```

whose real part is `~1e-16`, not zero.  Confirmed directly:

```
sqrt(-2.25 + 0.0j)        -> +1.5j            r.real == 0 : True   (pin fires)
sqrt(-2.25 - 0.0j)        -> -1.5j            r.real == 0 : True   (pin fires, flips)
sqrt(-2.25 - 2.9e-15j)    -> 9.67e-16 - 1.5j  r.real == 0 : FALSE  (pin blind)
```

So the pin fired for the REGION modes -- built in exact arithmetic, where
`Im(-kz^2)` really is a signed zero -- and never for a structured LAYER's modes,
whose `lam^2` comes out of `eig` carrying a backward error
`|Im(lam^2)| ~ eps_mach * ||M||` (measured 5.7e-20 .. 2.9e-15 here, against
`max|M| = 72.4`).  Each propagating layer mode therefore kept whichever root the
last bit of that backward error chose -- a quantity with no physical content
that changes with the BLAS kernel, the thread count and the platform.

A forward mode carrying the incoming root is the same object as a backward
mode.  Against a region of a DIFFERENT permittivity that is merely wrong by a
little; against a region of the SAME permittivity it is exactly the region's
own backward mode, so `a + b` annihilates it and `S12 = 2 (a+b)^-1` does not
exist.  This is the whole content of the library's long-standing warning text:

> If a LAYER permittivity is EXACTLY EQUAL to a REGION's (a groove, or a
> rotated director's ordinary `no^2`, equal to `n_substrate^2` or
> `n_superstrate^2`) the layer<->region mode match is exactly -- not nearly --
> degenerate at EVERY truncation and no `n_orders` helps: DETUNE one of the
> coincident permittivities by a relative ~1e-6 instead.

**The detune ladder confirms causation both ways** (`r5_detune.py`, Windows,
1 thread, pre-fix): walking `n_substrate` away from the coincidence by a
relative `d` makes the defect fall as `1/d` and return as `d -> 0`.

| relative detune `d` | closure defect, full | closure defect, even | `max\|R_full-R_even\|` | `max\|T_full-T_even\|` |
|---|---|---|---|---|
| 0     | -3.200e-03 | -3.381e-04 | 2.812e-10 | 4.287e-03 |
| 1e-13 | +2.332e-04 | -3.704e-05 | 5.938e-11 | 1.887e-04 |
| 1e-11 | -3.303e-06 | -1.757e-07 | 1.927e-10 | 1.815e-06 |
| 1e-09 | -7.943e-09 | +1.119e-08 | 1.033e-11 | 1.426e-08 |
| 1e-07 | -9.422e-11 | +3.096e-10 | 6.337e-11 | 2.538e-10 |
| 1e-06 | +1.389e-11 | +2.357e-12 | 1.209e-11 | 7.131e-12 |
| 1e-04 | +5.027e-13 | -2.487e-14 | 2.502e-13 | 9.948e-14 |
| 1e-02 | +2.487e-14 | +2.665e-15 | 7.098e-15 | 6.883e-15 |

Across the NINE decades `d` = 1e-13 .. 1e-04 the product `|defect| * d` stays
inside 7.9e-18 .. 5.0e-17 -- i.e. the error is `eps_mach / d` to within a factor
of six, which is what a condition number growing as `1/d` against machine
epsilon looks like, and `d = 0` behaves like `d ~ 1e-14` (the rounding in the
permittivity arithmetic).  The documented
"detune by 1e-6" remedy was therefore correct advice about a real symptom whose
CAUSE was a one-line branch test.

**Why the test caught it and the energy check did not refuse.**  The
disagreement between the two paths lands mostly in TRANSMISSION -- at the
coincidence, `max|T_full - T_even| = 4.287e-03` while `max|R_full - R_even|` is
2.8e-10 on Windows -- because the singular interface is the substrate one.  The
test asserts only on `R` and on the Jones matrix, so on most builds it read a
small number.  On WSL at one thread the noise reached `R` too.

---

## 5. What changed

`lumenairy/elements/rcwa/_core.py`, `_sqrt_decay`, plus a new module constant
`_CUT_BAND_REL`:

```python
scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
on_cut = xp.abs(r.real) <= _CUT_BAND_REL * scale
return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)
```

Three decisions, each with its reason:

1. **The band is relative to the mode spectrum, floored at 1.0** -- the same
   shape and the same `1e-8` factor as the PMM side's
   `pmm/_core._forward_branch_flip`, which solved this identical problem for
   the scalar-vertical generators (audit S1-8; its docstring already says
   "a naive `q.imag < 0` sign test would flip near-real (propagating) modes on
   that noise").  Using one convention for one quantity across the two solvers
   is deliberate.

2. **`conj(r)`, not `-r`.**  On the cut the exact root is pure imaginary, so
   the real part is noise whichever root is taken; `-r` would return
   `Re(lam) = -1e-16` and give up the `|X| <= 1` contraction guarantee that is
   the entire reason this function uses the principal branch.  `conj(r)` keeps
   `Re(lam) >= 0`.  The two differ by `2|Re(r)| ~ 1e-15` in a quantity of size
   `|kz|`.

3. **The predicate only ever ACTS on `Im(r) < 0`**, so the bar's whole job is
   to separate "that sign is rounding" from "that sign is physics".

### The bar, measured on both sides

`r11_band.py`: every `Im(r) < 0` mode of 51 fixtures -- 27 lossless
(twist x n_orders x n_substrate) and 24 on a LOSS LADDER down to
`Im(eps) = 1e-8` -- scored on the ratio the code thresholds,
`|Re(r)| / max(max|r|, 1)`:

| | Windows | WSL |
|---|---|---|
| `Im(r) < 0` modes | 2150 | 2135 |
| below `1e-4`: count / max | 117 / **2.245e-16** | 99 / **1.511e-16** |
| ... all of them lossless PROPAGATING modes | yes | yes |
| ... contributed by any loss ladder rung | none | none |
| above `1e-4`: count / min | 2033 / **8.267e-02** | 2036 / **7.947e-02** |

Fourteen decades of gap with nothing in it.  The bar `1e-8` sits 7.6 decades
above the noise side and 6.9 decades below the signal side.  The loss ladder is
the part that matters: it establishes that even a material with
`Im(eps) = 1e-8` -- far weaker than anything a caller would call lossy -- puts
NO mode on the low side, so the bar cannot silently re-sign a physical decay
rate.

---

## 6. What the change costs

`r10_census.py`, 33 fixtures x both builds, PRE arm (the `48c8747` body copied
into the probe) and POST arm (the shipped code) on the same interpreter.

**Where the sign was physics, nothing moved at all.**  The two lossy fixtures
are bit-identical between arms on both builds (`|post - pre| = 0.00e+00`).

**Where the sign was rounding, the results move -- and that is the fix.**
Bit-identity is neither attainable nor desirable for a lossless cell: the
change's entire purpose is to re-root modes that were mis-rooted.  What the
census shows is that the size of the move tracks the size of the pre-fix error:

| fixture class (rows) | `\|post - pre\|` on `R`, WIN / WSL | pre-fix closure defect, max WIN / WSL | post-fix closure defect, max WIN / WSL | max `cond(a+b)` pre -> post, WIN / WSL |
|---|---|---|---|---|
| non-coincident, lossless (18) | 1.04e-17 .. 5.54e-15 / 1.30e-17 .. 4.83e-15 | 7.55e-15 / 6.22e-15 | 7.99e-15 / 7.77e-15 | 1.54e+04 -> 9.01e+03 / 1.05e+04 -> 8.44e+03 |
| coincident, lossless, normal (11) | 5.41e-16 .. 8.58e-04 / 3.05e-16 .. **1.81e-02** | 4.05e-03 / 6.90e-03 | 4.35e-13 / 6.65e-13 | 4.24e+17 -> 1.50e+05 / 5.85e+16 -> 1.38e+05 |
| coincident, OBLIQUE (2) | 3.47e-16 .. 8.14e-15 / 1.07e-14 .. 1.85e-05 | 7.88e-04 / 7.70e-04 | 4.97e-14 / 1.08e-13 | 4.72e+15 -> 4.76e+02 / 1.74e+15 -> 4.86e+02 |
| LOSSY (2) | **0.00e+00 / 0.00e+00** | 1.28e-01, 9.45e-01 (real absorption) | unchanged | unchanged |

The one row whose CONDITIONING the fix does not improve is `bg1.44_coinc`
(background 1.44 with `n_substrate = 1.2`, so `eps_sub = 1.44` exactly again,
11 x 11 orders): `cond(a+b)` reads 1.50e+05 (WIN) / 1.38e+05 (WSL) both before
and after.  Its accuracy improves anyway -- the even path's closure defect goes
from -4.05e-03 to +4.35e-13 (WIN) and from +1.05e-03 to -6.65e-13 (WSL), and
the full-vs-even disagreement from 1.72e-05 / 1.97e-03 to 1.36e-13 / 5.60e-14.
That 1e+05 is a property of the truncation, not of the branch choice; the fix
does not claim to change it, only to stop the branch choice from adding ten
decades on top of it.

So for a solve that was NOT on the coincidence the change is a rounding-level
regauge; for one that was, it is the difference between a wrong answer and a
right one.  The independent oracle -- a provably lossless cell conserves energy
EXACTLY under the Laurent rule at any truncation -- moves from violated to
satisfied at the arithmetic floor.

`cond(a + b)` at the layer->substrate interface of the failing fixture,
Windows / 1 thread: **1.97e+15 -> 5.63e+03**, eleven decades.

### The same measurement on the WSL build, both trees

`r3_even_internals.py` / `r7_nullvector.py` / `r8_branch_cut.py`, one thread,
run against the read-only `48c8747` archive and against the branch tip:

| | WSL pre-fix | WSL post-fix | WIN pre-fix | WIN post-fix |
|---|---|---|---|---|
| closure defect, full `2N` | -8.5155e-04 | -1.5543e-15 | -3.1995e-03 | -5.9952e-15 |
| closure defect, even `N+1` | +4.9519e-03 | **+0.0000e+00** | -3.3813e-04 | **+0.0000e+00** |
| `cond(a+b)` layer -> substrate, full | 1.761e+15 | 5.286e+03 | 1.973e+15 | 5.633e+03 |
| `cond(a+b)` layer -> substrate, even | 2.25e+15 | 4.521e+03 | 1.074e+15 | 3.928e+03 |
| singular values below `1e-10 s_max`, full | 3 | **0** | 4 | **0** |
| singular values below `1e-10 s_max`, even | 1 | **0** | 2 | **0** |
| null-vector inverse participation, full | 1.01 | 3.96 | 2.27 | 3.31 |

(The pre-fix `cond(a+b)` digits are not meaningful beyond "`>= 1e15`": the
smallest singular value of an exactly singular matrix sits at the SVD's own
floor, which is why the even-arm reading is 2.25e+15 in one probe and 2.94e+15
in another on the same build.  What is meaningful is the ELEVEN-decade drop and
the singular-value COUNT going to zero.  The inverse participation ratio going
from ~1 to ~3-4 says the same thing from the other side: pre-fix the smallest
singular direction was ONE mode; post-fix there is no small direction and the
smallest one is an ordinary spread-out combination.)

`r8_branch_cut.py` states the invariant directly on the real operator.  On the
WSL branch tip, of the 125 (full) and 63 (even) modes with `Im(lam) < 0`, EVERY
one has `|Re(lam)| / |lam| = 1.000e+00` -- i.e. every remaining `Im(lam) < 0`
mode is a purely real (evanescent) root whose imaginary sign is meaningless, and
not one on-cut mode carries the incoming root.  Pre-fix the same census read a
minimum of `9.684e-19` (even) / `1.128e-17` (full).

**The defect was never specific to the even-parity fold.**  The two oblique
census rows skip the fold entirely (it needs normal incidence) and carry the
same defect: closure `+7.877e-04` (WIN) / `-3.110e-05` (WSL) at `theta = 0.3`
and `-1.172e-04` / `-7.696e-04` at `theta = 0.2, phi = 0.7`, going to
`4.97e-14` / `1.08e-13` and `1.33e-15` / `1.47e-14` after.  The full-vs-even
test was the detector, not the location.

**Blast radius, and a limit on it.**  `_sqrt_decay` is shared: `rcwa/oned.py`,
`rcwa/twod.py`, `rcwa/stack.py`, `pmm/twod.py`, `pmm/_jax_twod.py`,
`pmm/_jax_stack2d.py`, `pmm/_jax_twod_jones.py` and `elements/berreman.py` all
call it, and every one now gets the pinned root.  Whether that CHANGES a given
answer depends on a second condition, which `r12_oned_groove.py` measures.

The permittivity coincidence is necessary but NOT sufficient.  What makes
`a + b` singular is a LAYER MODE that numerically equals a REGION MODE -- and
an equal material index only makes that possible.  In the 2-D fixture the
background occupies 3/4 of the cell, so the layer really does carry modes at
`lam^2 = -2.249999999999987` and `-0.81`, i.e. the substrate's own `-kz^2` for
the (0,0) and (+-1,0)/(0,+-1) orders, to 1.3e-14.  In the 1-D binary grating
that the warning text names explicitly -- `n_groove = n_substrate = 1.5`,
ridge 2.1, duty 0.5 -- the ridge hybridises everything: the layer's propagating
eigenvalues are `-3.8330, -2.0847, -1.4490` (TE) and `-3.6153, -1.7444,
-1.6536` (TM) against the substrate's `-2.2500, -0.8100`, the CLOSEST approach
being 0.165.  One of its three on-cut TE modes is mis-rooted pre-fix, exactly as
in 2-D -- but with no region mode to be confused with, the mis-rooting is a
harmless relabelling of a layer-internal direction, and the closure defect reads
8.2e-15 pre-fix and 8.1e-15 post-fix on both builds, at every truncation and
detune tried.

The SCALAR 2-D entry point says the same thing from a third direction: the same
`eps = 2.25` background carrying an isotropic `eps = 4` block, solved through
`rcwa_efficiency_2d`, closes to `-1.3e-14` PRE-fix and `+8.9e-16` post, and
`RCWAStack` on that geometry gives `sum R + T = 2` exactly on both arms.
`pmm_efficiency_2d` on the equivalent pillar-in-host cell (`eps_host = 2.25`,
`n_substrate = 1.5`) is BIT-IDENTICAL between the two arms.  It is
the ANISOTROPIC tensor cell -- whose rotated director leaves the background
modes nearly decoupled -- that actually puts a layer mode on top of a region
mode.

So: the fix removes a build-dependent branch choice everywhere, and REPAIRS an
answer wherever that choice met a matching region mode.  A coincident
permittivity alone -- in 1-D, or in the scalar 2-D path -- is not automatically
one of those places.

Test evidence for the shared paths is in §7.

---

## 7. Per-build evidence

### The failing test, before and after

| build / threads | `48c8747` (pre-fix) | branch tip (post-fix) |
|---|---|---|
| WSL, 1 thread | `dR = 1.625072e-02`  FAIL | `dR = 4.163e-17`, `R+T = 2.000000000000000` |
| WSL, 2 / 4 / 8 threads | 3.65e-12 / 1.95e-13 / 8.37e-12 | 1.53e-16 / 9.02e-17 / 4.16e-17, `R+T = 2` exactly at each |
| Windows, 1 thread | `dR = 2.812111e-10` | `dR = 2.776e-16` |
| Windows, 2 / 4 / 8 / 16 | 3.05e-12 / 7.73e-14 / 1.83e-11 / 5.51e-14 | 2.78e-17 / 1.11e-16 / 1.80e-16 / 1.87e-16 |

### Pre-existing at `9af9376` (published v5.44.0)

`_sqrt_decay` is byte-identical at `9af9376` and `48c8747`, so the defect is
NOT something wave2 introduced.  Run on the detached `9af9376` worktree, WSL,
same interpreter, same numpy 2.4.6 -- only the thread count differs:

```
=== main 9af9376, WSL, OPENBLAS_NUM_THREADS=1 ===
  _EnergyWarning: lossless energy closure violated (sum R+T - 2 = +4.952e-03)
  FAILED tests/unit/test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full
  1 failed, 2 warnings in 1.10s

=== main 9af9376, WSL, unpinned (the CI unit lane's condition) ===
  _EnergyWarning: lossless energy closure violated (sum R+T - 2 = -1.784e-03)
  1 passed, 1 warning in 26.43s
```

One tree, one interpreter, one numpy, two verdicts.  The CI `unit` lane
installs numpy 2.4.6 / scipy 1.17.1 -- the SAME versions the WSL venv carries
(run 34437708929, the last green `unit-tests.yml` on main, head `9af9376`).  So
no version, wheel or platform difference has to be invoked at all: the
discriminator is the BLAS reduction order, and CI's greenness came from leaving
it unpinned.  Note the closure warning fires in BOTH cases -- the library was
telling the truth on every run, including the green ones.

### The new DECISION test, `tests/unit/test_fix_rcwa_even_sector_wsl.py`

Run against a read-only `git archive` of `lumenairy/` at `48c8747`
(`C:/tmp/lum_wslfix_pre`), one thread:

```
=== WINDOWS, pre-fix tree, 1 thread ===
FAILED test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise
FAILED test_coincident_layer_and_region_permittivity_closes_energy[False]   defect -3.200e-03
FAILED test_coincident_layer_and_region_permittivity_closes_energy[True]    defect -3.381e-04
FAILED test_the_defect_is_not_confined_to_normal_incidence[0.3-0.0]         defect +7.877e-04
FAILED test_the_defect_is_not_confined_to_normal_incidence[0.2-0.7]         defect -1.172e-04
FAILED test_even_fold_matches_the_full_solve_at_the_coincidence             dR 2.812e-10 dJ 7.097e-10
FAILED test_no_layer_mode_of_a_lossless_cell_carries_the_incoming_root      10 of 16 on-cut modes incoming
7 failed, 2 passed in 1.56s

=== WSL, pre-fix tree, 1 thread ===
FAILED test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise
FAILED test_coincident_layer_and_region_permittivity_closes_energy[False]
FAILED test_coincident_layer_and_region_permittivity_closes_energy[True]
FAILED test_the_defect_is_not_confined_to_normal_incidence[0.3-0.0]         defect -3.110e-05
FAILED test_the_defect_is_not_confined_to_normal_incidence[0.2-0.7]         defect -7.696e-04
FAILED test_even_fold_matches_the_full_solve_at_the_coincidence             dR 1.625e-02 dJ 4.357e-02
FAILED test_no_layer_mode_of_a_lossless_cell_carries_the_incoming_root      8 of 16 on-cut modes incoming
7 failed, 2 passed in 2.17s
```

The 2 that PASS pre-fix are the two guards the fix must not cost -- the
evanescent branch stays on the decaying root, and a physically-signed root is
returned bit for bit.  They are in the file precisely so that a change which
"fixed" the sign by flipping everything would be caught.

### The original test's own bar was not sound, and now is

`test_jones_2d_even_sector_matches_full` was an S4 shape by
`docs/TESTING_STANDARDS.md`'s taxonomy: a floor bar at `1e-8` on a quantity
whose cross-build, cross-thread spread ran 2.0e-14 .. 1.6e-02, i.e. the bar sat
INSIDE the spread.  It is left at `1e-8` and not touched, because the fix moves
the quantity it reads out of the noise: the post-fix envelope over nine
(build, thread) samples is <= 2.78e-16, so the bar now has more than seven
decades of room below it and the assertion has become a decision instead of a
reading.

Post-fix, one thread, both builds:

```
WINDOWS  tests/unit/test_fix_rcwa_even_sector_wsl.py  ->  9 passed in 1.34s
WSL      tests/unit/test_fix_rcwa_even_sector_wsl.py
         + tests/unit/test_v5_14_2_backlog_batch.py   ->  21 passed, 2 warnings in 11.23s
```

The WSL run above includes the originally failing
`test_jones_2d_even_sector_matches_full` at `OPENBLAS_NUM_THREADS=1`, and the
`_EnergyWarning` that used to accompany it is gone.

### Suites

All at `OPENBLAS_NUM_THREADS=1` -- the setting that exposes the defect -- and
`-p no:randomly`.

**THE GATE, corrected 2026-09-11 (round 2).**  `tests/unit/test_m1_conditioning_guard.py`
belongs in the battery that gates this change and was missing from it.  It is
the ONLY file this fix turns red, and it turns red for the right reason: two of
its tests pinned X-1 as a REPRODUCED, OPEN instability class and a third pinned
a "false positive the equilibration exists for", and all three premises were
this defect.  Round 2 restates them as decisions about the closed state
(`docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md` sections 6 and 7); the file
now reads 28 passed / 0 skipped on both builds.  The gate is therefore

    tests/unit/test_fix_rcwa_even_sector_wsl.py
    tests/unit/test_verify_rcwa_even_sector.py
    tests/unit/test_m1_conditioning_guard.py
    tests/unit/test_v5_14_2_backlog_batch.py
    tests/unit/test_fix_branch_cut_round2.py

-- 86 passed on Windows and on WSL, pinned and unpinned.

| suite | Windows py3.14 | WSL py3.12 |
|---|---|---|
| `test_v5_14_2_backlog_batch.py` + `test_rcwa*.py` + `test_niche*rcwa*.py` (+ the new file on WSL) | 225 passed, 1 skipped, 175.4 s | 235 passed, 241.7 s |
| `test_fix_rcwa_even_sector_wsl.py` | 9 passed, 1.3 s | included above |
| census / walker sweep (`walker\|census\|dispatcher_pin\|public_api\|doc_consistency`, `-x`) | 1282 passed, 12 skipped, 258.9 s | see below |
| `ruff check lumenairy/ tests/ validation/probe_fix_rcwa_even_sector_wsl/` | All checks passed! | All checks passed! |

The Windows figure excludes the new file (run separately); 225 + 9 = 234 plus
the one Windows-only `threadpoolctl` skip matches the WSL 235.

`_sqrt_decay` is differentiable and jit-safe under JAX after the change
(checked directly: `jax.jit(_sqrt_decay)` and `jax.grad` both return, and the
roots come back `+1.5j` / `+0.9j` on the same eigenvalue array that pre-fix
returned `-1.5j` / `-0.9j`).

### Wider blast-radius runs

`_sqrt_decay` is shared, so the change was carried past the files the brief
names.  All at one thread, `-p no:randomly`:

| suite | Windows py3.14 | WSL py3.12 |
|---|---|---|
| every `tests/unit/*pmm*` + `*berreman*` file | **1915 passed**, 3875 s | **1915 passed**, 3829 s |
| jax-guarded selection (`-k jax`) | **699 passed**, 10 skipped, 1286 s | (not run; jax 0.10.2 present) |
| every `tests/unit/test_fix_*` file | 288 passed, 1 skipped, **1 pre-existing failure** (below) | (not run) |
| census / walker sweep | 1282 passed, 12 skipped | 1225 passed, 5 skipped, **1 pre-existing failure** (below) |

Two failures appear, and NEITHER is this change's.  Both were re-run against the
read-only `48c8747` archive / an untouched worktree and fail identically there:

* `test_fix_slant_anchor_v1_v2_o2.py::test_o2_the_mortar_interfaces_take_no_
  explicit_inverse` (Windows) is a SOURCE-INSPECTION assertion -- it greps
  `pmm/_core._interface_smatrix_general_mortar_2d` for the literal string
  `np.linalg.solve`, which that function no longer contains (it now ends in a
  guarded helper).  `git diff 48c8747 HEAD -- lumenairy/elements/pmm/` is EMPTY
  on this branch, and the test fails identically against the pre-fix archive.
  It belongs to the concurrent slant-anchor work.
* `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_
  caught` (WSL) needs real `git` plumbing.  A Windows-created LINKED worktree
  carries a `.git` file pointing at a `D:/...` path, which Linux cannot resolve
  (`fatal: not a git repository: /mnt/c/tmp/lum_wslfix/D:/...`), so the walker
  cannot diff and returns the wrong code.  It fails identically in the
  UNTOUCHED `9af9376` worktree under WSL.  It is a mount artifact of running a
  Windows worktree from WSL, not a repository or CHANGELOG defect -- the same
  test passes on Windows against the same CHANGELOG.

STILL RUNNING at the time of this commit: the FULL Windows `tests/unit` fast
gate (`-m "not integration and not slow"`, single-threaded, ~13 000 tests) --
131 CPU-minutes in and not yet finished.  The four batteries above already
exercise ~4 400 distinct tests across every module that calls `_sqrt_decay`.

---

## 8. What is NOT established

* **The `DLASCL` line is unexplained.**  Section 2 shows it is not this
  solve's (no non-finite value reaches any LAPACK call here, and the lines are
  flushed after pytest's own summary), but WHICH call in the 1237-test WSL run
  emitted it is not established.  It is left open; a targeted hunt would wrap
  XERBLA or bisect the suite.
* **No claim of bit-identity for lossless solves.**  See §6: the change moves
  every lossless solve by 1.04e-17 .. 5.54e-15 even away from the coincidence,
  because the mis-rooted modes exist there too (they simply do not meet a
  same-permittivity region).  Any downstream artifact pinned to a pre-fix
  digit at that level will move.
* **The bar was measured on 2-D anisotropic RCWA fixtures.**  The 1-D core, the
  PMM solvers and Berreman share `_sqrt_decay` and are covered by their suites
  (§7), but the 2150-mode band census itself was taken on the 2-D path.
* **`stabilize=True` was not re-characterised.**  Its retry schedule exists to
  step around exactly this failure class; whether it can now be narrowed is a
  separate question and is not answered here.  The same goes for the
  `_check_energy` warning text quoted in §4, which now names a remedy for a
  cause that has been removed -- correcting it is a documentation change this
  branch deliberately does not make while the fix is unreleased.
* **The `1e-8` band was not swept.**  It was adopted from the PMM side's
  `_forward_branch_flip` and then SHOWN to have decades on both sides of the
  measured populations; no attempt was made to find the widest admissible
  value, because the measured gap (14 decades) makes the exact placement
  inside it immaterial.  **Round 2 corrects the margin numbers** (the
  51-fixture box carried no LAYER-CUTOFF mount, where the noise side reaches
  6.7e-09) and re-decides the band's SHAPE by measurement against a per-mode
  alternative; the shape is kept, the margins are replaced.  See
  `docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md` section 5.
* **The blast-radius paragraph in section 6 is WRONG about the PMM half**, and
  is corrected in the CHANGELOG rather than rewritten here: the five PMM copies
  of `_sqrt_decay` were NOT reached by this change.  Round 2 consolidated them
  into one definition and measured what they cost -- up to `sum R + T = 110` on
  a passive lossless stack.

# The modal branch cut, round 2: one `_sqrt_decay`, and what the five PMM copies cost

Round 1 (`docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`) repaired the
RCWA modal branch selector.  Its independent verification
(`docs/audits/VERIFY_RCWA_EVEN_SECTOR_2026_09_11.md`) confirmed 22 of 25 claims,
refuted one -- the blast-radius paragraph, which asserts that the PMM solvers
"all get the pinned root" when five PMM modules kept their own unrepaired copy
of the function (defect D1) -- and left one question open:

> **Whether the PMM copies can produce a WRONG answer, not merely a
> build-dependent one.**  I built nine PMM 2-D fixtures at and off the
> coincidence and none put a PMM layer mode on top of a PMM region mode the way
> the RCWA anisotropic cell does; the largest motion from fixing the copies was
> 8.660e-15.  The mechanism is present and measured; the consequence is not.

They can.  Section 4 has the measurement.

Branch `fix/branch-cut-round2`, cut from `2898767` (the `wave2/pmm2d` tip
carrying round 1 and its verification).  Probes and JSON, both builds:
`validation/probe_fix_branch_cut_round2/`.

Builds throughout: **WIN** = Windows 11, python 3.14.6, numpy 2.4.4, scipy
1.17.1, scipy-openblas dispatching **Haswell**.  **WSL** = Ubuntu on the same
box, python 3.12.3, numpy 2.4.6, scipy 1.17.1, scipy-openblas dispatching
**SkylakeX**.  Every run pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` on the command line except where a thread count is the
variable under study.

---

## 1. Terms

**Modal decay constant.**  A rigorous layer solve condenses one grating layer's
transverse-field system to `M = P Q` and eigendecomposes it.  Each eigenvalue is
`lam^2`; the modal decay constant is `lam = sqrt(lam^2)`, and the layer's
forward amplitudes propagate through thickness `L` as `X = exp(-lam k0 L)`.
Keeping `Re(lam) >= 0` is what makes `|X| <= 1`, i.e. what keeps the propagator
a contraction rather than a growing exponential.

**On the branch cut.**  For a PROPAGATING mode of a LOSSLESS layer `lam^2` is
exactly real NEGATIVE, which is exactly the principal square root's branch cut.
The two roots there are `+i|kz|` (OUTGOING -- the branch the homogeneous
half-space modes are built on) and `-i|kz|` (INCOMING).  They are separated only
by the sign of `Im(lam^2)`, and for a value that came out of `eig` that sign is
the eigensolver's backward error, not physics.

**The exact-zero pin.**  The branch test this campaign is about:

```python
on_cut = r.real == 0
return xp.where(on_cut & (r.imag < 0), -r, r)
```

For `lam^2 = -s + i eta` the principal root is `r = eta/(2|r|) + i sign(eta) sqrt(s)`,
whose real part is ~1e-16, not zero.  So the pin fires for a REGION mode -- built
in exact arithmetic, where `Im(-kz^2)` really is a signed zero -- and NEVER for a
structured LAYER's, whose root is then decided by the last bit of `eta`.

**The band.**  What round 1 put in its place:

```python
scale  = max(max|r|, 1)
on_cut = |Re(r)| <= _CUT_BAND_REL * scale        _CUT_BAND_REL = 1e-8
r      = conj(r)  where  on_cut and Im(r) < 0
```

Throughout, the **band ratio** of a mode means the quantity this thresholds,
`|Re(r)| / max(max|r|, 1)`, and the **acted-on population** means the modes with
`Im(r) < 0` whose band ratio is at or below `_CUT_BAND_REL`.

**Interface mode-match.**  Joining media `a -> b` forms `a = Wb^-1 Wa`,
`b = Vb^-1 Va` and inverts `a + b` EXPLICITLY, because `S12 = 2 (a+b)^-1`.
`a + b` is singular exactly when a FORWARD mode of `a` reproduces a BACKWARD
mode of `b`.

**Coincidence partner.**  A medium whose modes are built in EXACT arithmetic by
`_homogeneous_eigenmodes` -- a half-space REGION, or a UNIFORM LAYER of the same
stack -- and whose permittivity equals a structured layer's background.  Round 1
named only the region; section 6 of the verification added the uniform layer on
the RCWA side; section 4 below shows the uniform layer is what breaks the PMM.

**Closure defect.**  `sum R + sum T - 1` (one polarization) or `- 2` (a Jones
return).  A provably lossless cell conserves energy EXACTLY under the Laurent
rule at any truncation, so this is an independent oracle whose error floor is the
arithmetic -- it needs no reference solve and no prior reading.

---

## 2. The state round 2 inherited

`_sqrt_decay` had SIX bodies.  One -- `rcwa/_core.py` -- carried round 1's band
and `conj` flip.  Five carried the exact-zero pin and the `-r` flip:

| module | line | handed LAYER eigenvalues? | live? |
|---|---|---|---|
| `pmm/twod.py` | 411 | YES (`_layer_modes_projected`, `_symmetric_solve_2d`) | yes |
| `pmm/twod_staggered.py` | 2286 | never called | **DEAD CODE** |
| `pmm/_jax_twod.py` | 341 | YES (the traced layer solve) | yes |
| `pmm/_jax_stack2d.py` | 175 | YES (the traced stack layer solve) | yes |
| `pmm/_jax_twod_jones.py` | 191 | no (region modes only) | yes |

`-r` and `conj(r)` differ in a way that matters beyond the flip's own purpose:
`-r` returns `Re(lam) = -1e-16`, giving up the `|X| <= 1` contraction guarantee
the principal branch exists to provide.

Two scope facts, measured here and worth stating before anything else, because
they bound what follows:

* **The PURE STAGGERED 2-D engine never called `_sqrt_decay` at all.**  Its
  forward branch comes from `pmm/_core._forward_branch_flip` (already a relative
  band) and, on the OUT-OF-PLANE path, from `rcwa/_core._select_forward_flux` (a
  relative z-flux bar with a `1e-9 * max|Sz|` floor, a `3e-3` projection-noise
  ceiling and a `|Re gam| > 0.5` deep-decay override).  Measured: **0 calls** on
  every staggered surface of the round-2 census -- `pmm_efficiency_2d_staggered`,
  `pmm_jones_2d_staggered` (in-plane tensor, out-of-plane, magnetic) and
  `PMM2DStackPure`.  Neither round of this campaign can have moved it, and its
  closure on the same coincidence that broke the hybrid reads 5.1e-15 .. 1.2e-14.
* **The 1-D PMM likewise.**  `pmm_efficiency_1d` / `pmm_jones_1d` select through
  `_forward_branch_flip`; they are bit-identical between the arms.

So the defect's reach inside the PMM half of the library is the HYBRID 2-D
family (`pmm_efficiency_2d`, `pmm_efficiency_2d_cell`, `PMM2DStackHybrid` /
`PMM2DStack`, `stack2d`, `conical`) and its three JAX twins.

---

## 3. What ships: ONE definition

`rcwa/_core._sqrt_decay` gained two parameters and the five copies were deleted:

```python
def _sqrt_decay(x, xp=None, band: float = _CUT_BAND_REL):
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(_C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    on_cut = xp.abs(r.real) <= band * scale
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)
```

Four decisions, each with its reason.

1. **The shared home is `rcwa/_core.py`, not `pmm/_core.py`.**  The import graph
   runs one way: `pmm/_core.py`, `pmm/twod.py` and `pmm/twod_staggered.py`
   already import `_interface_smatrix`, `_select_forward_flux`,
   `_project_efficiency` and more from `rcwa/_core.py`, and `rcwa/_core.py`
   imports nothing from `pmm/`.  Putting the shared body on the PMM side would
   have created a cycle.  `rcwa._core._sqrt_decay` therefore keeps its name and
   needs no alias -- every existing importer, including the round-1 and
   verification test files, is unchanged.

2. **`xp=None` auto-detects; the JAX twins pass `jnp` explicitly.**  Auto-detect
   via `array_namespace` is what every NumPy and CuPy call site has always done,
   so passing nothing reproduces round 1 exactly.  The three JAX twins call
   `_sqrt_decay(x, jnp)`, following the shape
   `pmm/_core._forward_branch_flip(q, xp=np)` already uses, so the traced body
   is the same object the eager path executes and the two cannot drift apart
   again.  Only `r.size` is read as a Python value and that is static under
   tracing, so the body stays `jit`- and `grad`-safe.

3. **`band` is a real parameter with one derived default.**  Section 5 measures
   the two-sided gap on THREE modal populations -- the RCWA layer `P@Q`
   spectrum, the hybrid PMM's SEM-projected `P@Q` spectrum, and the staggered
   pencil's `gamma^2` spectrum -- on both builds, and derives the same value
   from each.  The parameter exists so a caller whose population differs can
   carry its own; the measurement is why every in-tree caller uses `1e-8`.

4. **The staggered copy is deleted rather than routed.**  It was never called.
   Routing dead code would have created a caller where there was none.

**The refactor is bit-identical.**  Over 4,010 engineered values -- 4,000
pseudo-random over fifteen decades plus signed zeros, a denormal, `lam^2`
exactly on the cut, `nan` and `1e300 + 1e300j` -- and at six array SIZES
(1, 2, 7, 64, 243, 4010), because the band is relative to the array, the shared
body and the transcribed round-1 body return the same bits: `max |diff| = 0`,
`bit_identical = True`, on both builds
(`b4_census.py`, the `refactor` block).

---

## 4. The headline: a uniform spacer makes the hybrid PMM's answer WRONG

### 4.1 The fixture, and why it is shaped this way

The verification could not decide the wrong-answer question because the hybrid
PMM has a FOURIER TRUNCATION FLOOR: at a realistic contrast its own error is
~1e-3, which masks anything smaller.  The instrument has to be a cell whose
truncation error is at the arithmetic floor, so that the closure defect reads
the branch choice and nothing else.  A WEAKLY MODULATED cell does that: a
`6 x 6` grid of `eps = 2.25` carrying a `2 x 2` block at `2.25 * (1 + 1e-6)`.
It is the same shape the verification used to break the RCWA scalar 2-D path
(its fixture `c5`, `cond(a+b)` 2.5e+08 pre-fix).

Around it, a three-layer `PMM2DStackHybrid`:

```
superstrate  n = 1.0
  layer 1    uniform,  eps = 2.25,       thickness 0.10 um      <- the SPACER
  layer 2    patterned, background 2.25,  thickness 0.20 um
  layer 3    uniform,  eps = 2.25,       thickness 0.10 um      <- the SPACER
substrate    n = 1.63   (eps = 2.6569 -- coincides with NOTHING)
```

`symmetry=False`, because at normal incidence with `symmetry='auto'` the whole
hybrid cascade runs in `rcwa/_core._symmetric_cascade_rt`, which uses the FIXED
function; the PMM copy is reached through `_layer_modes_projected` on the full
path.  That is itself a measurement: `weak_both_sym` moves by exactly
`0.000e+00` between the arms and its on-cut incoming count is 0.

### 4.2 The reading

`b7_spacer.py`, both branch bodies installed in ONE interpreter so the arm is
the only thing that changes between two readings of one solve.  WIN, 1 thread:

| fixture | class | PRE closure | POST closure | per-order motion |
|---|---|---|---|---|
| `weak_none_M4` (no spacer, `n_sub` 1.63) | none | +4.885e-15 | -6.439e-15 | 3.331e-15 |
| `weak_region_M4` (no spacer, `n_sub` 1.5) | region | +3.997e-15 | -5.995e-15 | 4.885e-15 |
| **`weak_spacer_M4`** (spacer, `n_sub` 1.63) | **spacer** | **-1.6653e-04** | **-6.661e-15** | **2.394e-03** |
| **`weak_both_M4`** (spacer, `n_sub` 1.5) | **both** | **+2.2274e-04** | **-6.661e-15** | **2.158e-03** |
| `weak_spacer_M3` | spacer | -4.8926e-06 | +3.997e-15 | 2.926e-05 |
| `weak_both_M3` | both | +2.2091e-07 | +4.885e-15 | 2.489e-05 |
| `weak_spacer_M5` | spacer | +4.4193e-05 | +4.441e-16 | 1.604e-04 |
| `lossy_both` | lossy | -2.9763e-03 | -2.9763e-03 | **0.000e+00** |

Three things this settles.

* **The answer is WRONG, not merely build-dependent.**  2.394e-03 in per-order
  efficiency on a cell whose own truncation error is 1e-14.
* **The partner is the LAYER, not the region.**  The worst row has
  `n_substrate = 1.63`: nothing in it coincides with a half-space.  Adding the
  region coincidence on top (`both`) changes nothing material.
* **Where the sign is physics, nothing moves.**  The lossy row is bit-identical
  between the arms.

### 4.3 The detune ladder: it really is the spacer

Walk the SPACER's permittivity off the layer background by a relative `d`,
leaving the substrate, the cell and the truncation exactly where they were
(WIN, 1 thread, `n_orders = 3`; WSL in the JSON):

| `d` | PRE closure | POST closure | motion |
|---|---|---|---|
| 0 | -4.8926e-06 | +3.997e-15 | 2.926e-05 |
| 1e-12 | +7.1742e-05 | +4.441e-15 | 4.084e-04 |
| 1e-09 | -1.2246e-05 | +2.220e-15 | 7.152e-05 |
| **1e-06** | **+8.0560e-05** | +3.997e-15 | 1.079e-04 |
| 1e-03 | -8.7486e-14 | +4.441e-15 | 1.383e-12 |
| 1e-01 | +6.2172e-15 | +5.773e-15 | 8.771e-15 |

The defect follows the SPACER.  It also survives a relative `1e-6` detune --
which is exactly what the library's own remedy text used to recommend -- and is
gone only by `1e-3`.  That is defect D6's correction with a number behind it,
and section 8 records what the message now says instead.

### 4.4 The modulation ladder: why nine fixtures found nothing

| pillar - host, relative | `both` PRE | `none` PRE | motion |
|---|---|---|---|
| 1e-08 | +4.441e-16 | -6.661e-16 | 8.882e-16 |
| **1e-06** | **+2.2091e-07** | +1.021e-14 | **2.489e-05** |
| 1e-04 | +5.5538e-11 | +1.043e-10 | 1.887e-15 |
| 1e-02 | +5.5508e-07 | +1.038e-06 | 5.329e-15 |
| 1e-01 | +5.5346e-05 | +9.871e-05 | 3.664e-15 |
| 1e+00 | +6.1436e-03 | +6.871e-03 | 6.655e-13 |

From `1e-04` up the hybrid's own Fourier floor dominates and the `both` and
`none` columns track each other; the branch choice is invisible under it.  Below
`1e-08` the layer is numerically uniform and routes to the analytic Rayleigh
helper, where the old pin was always correct.  The defect is visible in a window
around `1e-06`, and only there -- which is why nine fixtures at ordinary
contrasts measured 8.7e-15 and concluded the consequence was nil.

### 4.5 The thread ladder, and the second build

`weak_spacer_M4`, PRE arm, `|sum R + T - 2|`:

| `OPENBLAS_NUM_THREADS` | WIN PRE | WIN POST | motion (WIN) |
|---|---|---|---|
| 1 | **1.6653e-04** | 6.661e-15 | 2.394e-03 |
| 2 | 5.5994e-06 | 6.661e-15 | 2.340e-05 |
| 4 | 6.3667e-05 | 5.995e-15 | 4.456e-04 |
| 8 | 3.5970e-05 | 5.995e-15 | 1.214e-03 |
| unpinned | **4.5820e-04** | 5.995e-15 | 1.782e-03 |

Two decades of spread with a sign change, on a quantity that is a conservation
law -- the branch-cut signature exactly as round 1 measured it on the RCWA side.
The POST column is flat to within 7e-16.

WSL tells the same story with a different partition, which is the point:

| fixture | WSL 1 thr PRE | WSL 1 thr POST | motion | WSL 4 thr PRE |
|---|---|---|---|---|
| `weak_spacer_M3` | -2.1611e-05 | +3.109e-15 | 1.264e-04 | -- |
| `weak_both_M3` | +3.5659e-07 | +4.441e-15 | 1.073e-04 | -- |
| `weak_spacer_M4` | -2.220e-15 | -2.443e-15 | 5.551e-16 | -1.776e-15 |
| `weak_spacer_M5` | -7.6899e-06 | +4.441e-16 | 3.849e-05 | -- |
| `spacer_detune_1e-06` | +2.3069e-04 | +3.553e-15 | 6.633e-04 | -- |
| `lossy_both` | -2.9763e-03 | -2.9763e-03 | **0.000e+00** | 0.000e+00 |

`M = 4` is broken on Windows and clean on WSL; `M = 3` and `M = 5` are broken on
both.  Same geometry, same source, two builds, and WHICH truncation is wrong is
decided by the reduction order -- the identical shape the verification's `n4`
row records for RCWA.  Over the seven (build, thread) samples the PRE readings
span 1.8e-15 .. 4.6e-04 with sign changes; every POST reading is `<= 6.7e-15`.

### 4.6 Is the repaired answer RIGHT?

Conservation says the PRE answer was wrong; it cannot say the POST answer is
right, because a solve can conserve energy and still put the power in the wrong
orders.  A DIFFERENT METHOD says it: the Fourier `RCWAStack` on the same device
(the 6 x 6 cell block-replicated 12x, which is exact -- the cell is piecewise
constant on those walls -- so the Fourier path has the sampling it needs).
`b8_reference.py`, WIN, 1 thread:

```
RCWA reference M=4  closure -2.2204e-15   81 orders
RCWA reference M=6  closure -3.3307e-15  169 orders
RCWA reference M=8  closure -2.8866e-15  289 orders
RCWA self-convergence 6 vs 8: max |d| = 4.4409e-16
```

| PMM `n_orders` | PRE closure | PRE vs reference | POST closure | POST vs reference |
|---|---|---|---|---|
| 3 | -4.8926e-06 | **2.9264e-05** | +3.9968e-15 | **5.1070e-15** |
| 4 | -1.6653e-04 | **2.0035e-03** | -6.6613e-15 | **5.9952e-15** |
| 5 | +4.4193e-05 | **2.4583e-04** | +4.4409e-16 | **1.1102e-15** |

The reference has stopped moving to 4.4e-16, so it can calibrate.  The repaired
hybrid PMM agrees with it to **6.0e-15 per order**; the unrepaired one was
**2.0e-03** away.

**Verdict on the verification's open question: the PMM copies could and did
produce a wrong answer.  Round 2 repairs it to the arithmetic floor against an
independent method, on both builds, at every thread count measured.**  Section
4.8 pushes the same fixture family further, to `sum R + T = 110` on a passive
lossless stack.
### 4.7 The sharp instrument: `cond(a + b)` on the PMM's own interfaces

Round 1 localised the RCWA defect not by an accuracy reading but by the
CONDITIONING of the interface mode-match, and that instrument is what settles
the mechanism here too, because it is a property of the operator rather than of
the truncation error.  `b2_pmm_interface.py` patches every PMM binding of
`_interface_smatrix` and records `cond(a + b)` at each mode match (WIN, 1
thread; `inst` is the shipped round-2 code, `pre` the reinstated exact-zero pin):

| fixture | class | `cond(a+b)` PRE | POST |
|---|---|---|---|
| `cell_weak_region_M3` | region | **6.556e+08** | 1.978e+01 |
| `cell_weak_none_M3` | none | 9.918e+01 | 1.978e+01 |
| `cell_weak_region_M5` | region | **9.080e+05** | 6.879e+01 |
| `cell_weak_region_M7` | region | **6.802e+07** | 1.244e+02 |
| `cell_weak_region_M9` | region | **6.486e+08** | 2.031e+04 |
| `cell_weak_superstrate` | superstrate | **1.173e+07** | 5.094e+01 |
| `hyb_weak_spacer_M3` | spacer | **1.959e+09** | 1.453e+01 |
| `hyb_weak_both_M3` | both | **1.959e+09** | 1.453e+01 |
| `hyb_weak_both_oblique` | both | **1.920e+08** | 1.527e+01 |
| `stag_weak_region` | region (staggered) | 1.539e+02 | 1.539e+02 |
| `pure_weak_both` | both (staggered) | 6.590e+01 | 6.590e+01 |

Seven to nine decades on the coincident mounts, nothing on the off-coincidence
control, nothing at all on the staggered engine.  The `cell_weak_superstrate`
row is the SUPERSTRATE-side coincidence (`n_superstrate^2 = 2.25`,
`n_substrate = 1.9`), which behaves exactly as the substrate side does.

**And the conditioning is BUILD-INDEPENDENT where the accuracy reading is not.**
That is exactly why it is the instrument.  WSL, one thread:

| fixture | `cond(a+b)` PRE | POST |
|---|---|---|
| `cell_weak_region_M3` | 4.971e+07 | 3.619e+01 |
| `cell_weak_superstrate` | 4.508e+08 | 1.831e+01 |
| `hyb_weak_spacer_M3` | 2.915e+08 | 2.760e+01 |
| `hyb_weak_both_M3` | 2.915e+08 | 2.760e+01 |

`hyb_weak_spacer_M3` closes at 8.912e-10 on WSL and at 2.141e+00 on Windows,
from the same operator with the same seven-decade conditioning defect: whether a
near-singular explicit inverse produces a visible error is decided by the
reduction order, which is the whole reason a closure reading alone could not
settle the question and nine fixtures found nothing.

### 4.8 The loudest mount: the pre-round-2 hybrid PMM MANUFACTURES energy

The 4.2 fixture samples the cell on a 6-pixel grid.  Sampling the SAME device on
a 32-pixel grid -- three strips per axis instead of three coarse cells, an
identical piecewise-constant permittivity -- makes the pre-round-2 error three
decades larger.  `b10_manufactured_energy.py`, `n_orders` 2..6 with and without
the spacers, plus a spacer-detune ladder, on both builds at 1 / 4 / 8 threads:

| build / threads | worst PRE `sum R + T` | worst PRE per-order motion | worst PRE NO-SPACER control | worst POST (all mounts) |
|---|---|---|---|---|
| WIN 1 | **5.812454299** (2.9x) | 2.788e+00 | 8.8539e-10 | 8.8539e-10 |
| WIN 4 | **1.098858e+02** (55x) | 5.755e+01 | 8.8539e-10 | 8.8539e-10 |
| WIN 8 | **3.567077e+01** (18x) | 4.674e+00 | 8.8539e-10 | 8.8539e-10 |
| WSL 1 | 1.999336151 | 2.076e-02 | 8.8540e-10 | 8.8539e-10 |

A passive lossless stack returning 110 times the incident power is not a
tolerance question.  Five things this adds.

* **The POST envelope and the PRE NO-SPACER control are the SAME number**,
  8.8539e-10, on every one of the eight (build, thread, arm) samples.  It is the
  `n_orders = 4` control mount's own Fourier truncation error -- deterministic,
  arm-independent, and the floor every bar in the round-2 test file is derived
  against.
* **Five decades of spread with the BLAS thread count** on the PRE arm
  (6.6e-04 .. 1.1e+02), and a per-build partition of WHICH truncation breaks:
  Windows breaks `n_orders` 3 and 4, WSL breaks 5 and 6.  That is the branch-cut
  signature, one level up.
* **It is not always loud.**  At `n_orders = 3` Windows emits a `UserWarning`
  (`sum R + T = 4.14` is past the energy tripwire), but the spacer detuned by a
  relative 1e-6 returns `sum R + T = 2.000072107` **SILENTLY**, with per-order
  efficiencies 9.331e-05 away from the repaired answer.  A caller who followed
  the library's own "detune by ~1e-6" advice would have got a quiet wrong
  answer.
* **The no-spacer control never moves**, on either arm, at any truncation or
  thread count.  The spacer is the cause, isolated.
* **Every POST mount reads 2.000000000** to nine decimals.

The gate `test_the_coincident_spacer_stack_does_not_manufacture_energy` asserts
on the LADDER rather than on one mount, precisely because which truncation
breaks is a per-build fact.


---

## 5. The band scale: ARRAY-MAX versus PER-MODE

*(measured in `b6_band_scale.py`; see section 5.3 for the decision)*

### 5.1 The question

The shipped band's scale is the LARGEST root of the array:

```
ARRAY-MAX      |Re r| <= C * max( max|r|, 1 )
```

so its verdict on one mode depends on the other modes of the same layer.  The
verification measured two consequences.  D4: at a mount driven onto a LAYER
CUTOFF a mode whose real part is 0.2 % of its OWN magnitude
(`|Re r| / |r| = 2.0751e-03`) is conjugated, because that mode sits at
`|lam| = 1.7e-07`, i.e. 3.8e-08 of the spectrum's top.  D3: the noise side of
the population reaches 6.7172e-09 at such a mount, only 1.5x under the `1e-8`
bar, so the round-1 docstring's "7.6 decades above the noise side" is
sample-scoped.

The alternative judges each mode in its own terms, with a FLOOR so a mode whose
magnitude has collapsed into the backward error is not divided by noise:

```
PER-MODE       |Re r| <= C * max( |r|, floor )
```

**How `floor` is derived.**  `eig` returns `lam^2` with a backward error
`|d lam^2| ~ eps_mach * ||M||`, and `r = sqrt(lam^2)`, so the smallest magnitude
a root can carry that still means anything is `sqrt(eps_mach * ||M||)`.  `||M||`
is not visible inside the selector, but `max|r|^2` IS the spectrum's own scale
for `lam^2`, so the same statement in the quantities the function has is
`floor = sqrt(eps_mach) * max|r| = 1.4901e-08 * max|r|`.  Note what that means:
the per-mode shape degenerates CONTINUOUSLY into the array-max shape for any
mode more than about eight decades below the spectrum's top, and differs from it
only in the band between.

### 5.2 The measurement

Three populations of the SHARED function's own quantity, both builds.  A mode is
NOISE-side when its `lam^2` is real negative to within the eigensolver's own
backward error -- a lossless propagating mode whose imaginary sign is rounding,
which the band MUST reach -- and SIGNAL-side otherwise, which the band must NOT
reach.  The classification is made on `lam^2`, i.e. on the eigenproblem, never
on the ratio under test.  The band ratio is computed from the RAW `eig` output,
so it is a property of the eigenproblem and identical on both arms of the
change.

**Population 1 -- RCWA, ordinary mounts** (`b6_band_scale.py`: 58 fixtures --
lossless anisotropic 2-D over twist x truncation x substrate, oblique, conical,
scalar 2-D at three contrasts, 1-D TE/TM at three duties, high-contrast gratings
at `n_orders` 21 and 31, a LOSS LADDER `Im(eps)` 1e-2 .. 1e-14 on both paths,
four metals to `eps = -100 + 5j`, three near-Wood mounts; 2497 `Im(r) < 0`
modes):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN | **5.5699e-15** | 1.2612e-02 | **12.35 dec** |
| ARRAY-MAX | WSL | **6.4448e-15** | 1.2612e-02 | **12.29 dec** |
| PER-MODE | WIN | 8.1213e-13 | 9.9215e-01 | 12.09 dec |
| PER-MODE | WSL | 9.3970e-13 | 9.9215e-01 | 12.02 dec |

**Population 2 -- RCWA, LAYER-CUTOFF mounts** (`b6b_cutoff.py`: 72 mounts, each
found by a bounded scalar minimisation of `min |lam^2|` over `n_ridge` after a
90-point bracketing scan, plus a +/- ladder at relative 1e-3 .. 1e-9 around each
so the corner is a family and not one engineered point; deepest mount
`min|lam^2| = 6.5341e-10`):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN and WSL | **9.0475e-13** | 2.1844e-05 | **7.38 dec** |
| PER-MODE | WIN and WSL | **6.8698e-08** | 1.0000e+00 | 7.16 dec |

(The two builds agree to every printed digit here, which is what one expects:
the ratio is read off `eig`'s output before any branch decision, and these
mounts are found by the same deterministic minimisation on both.)

**Population 3 -- the HYBRID PMM's SEM-projected `P@Q` spectrum**
(`b6_band_scale.py`: the `pmm/twod.py` layer eigenproblem over weak and strong
modulation, on and off the coincidence, a metal, three loss rungs, three
truncations, plus the tensor entry; 1463 (WIN) / 1642 (WSL) `Im(r) < 0` modes):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN | **5.5766e-16** | 2.8623e-02 | **13.71 dec** |
| ARRAY-MAX | WSL | **1.7884e-15** | 3.8802e-02 | **13.34 dec** |
| PER-MODE | WIN | 2.2213e-14 | 8.4520e-02 | 12.58 dec |
| PER-MODE | WSL | 1.1973e-13 | 8.4520e-02 | 11.85 dec |

(A fourth population -- the staggered pencil's `gamma^2` -- appears in
`b6_band_scale.py`'s output and is NOT a population of this function: the
staggered engine never calls it, and its selector thresholds a different
quantity on the opposite convention.  Section 5.5 measures it properly, as a
cross-check rather than as a constraint on `band`.)

### 5.3 The decision: KEEP the array-max shape

**ARRAY-MAX has the larger two-sided gap on all three populations and on both
builds** -- 12.35 / 7.38 / 13.71 against 12.09 / 7.16 / 12.58.  The shape is not
changed, and per the brief that leaves only the MARGINS to re-derive (D3).

The cutoff population is where the two shapes were supposed to separate, and it
separates them the other way round.  The reason is physical rather than
arithmetic: **at a layer cutoff the mode's own magnitude collapses**, so judging
its real part against its own magnitude is judging noise against noise.  At the
deepest mount here `|lam| ~ 2.6e-05` while the per-mode floor
`sqrt(eps_mach) * max|r|` is ~4.5e-08, so the per-mode divisor is `|r|` itself
and the ratio blows up: its noise side reaches **6.8698e-08, which is ALREADY
ABOVE the shipped `1e-8`**.  Adopting the per-mode shape at the current constant
would start MISSING modes the array-max shape catches -- the exact failure the
band exists to prevent.  Re-deriving the constant for it would put it near
`2.6e-04` (the geometric mean of 6.87e-08 and 1.0), a four-decade move for a
narrower window.  The spectrum's top is the only stable scale at a cutoff.

### 5.4 D3, re-derived: the margins the constant now states

With the shape kept, the `1e-8` bar sits, on the ARRAY-MAX ratio:

| population | decades above the noise side | decades below the signal side |
|---|---|---|
| RCWA ordinary (both builds) | **6.19** (6.4448e-15) | **6.10** (1.2612e-02) |
| PMM hybrid (both builds) | **6.75** (1.7884e-15) | **6.46** (2.8623e-02) |
| RCWA layer cutoff, this box | **4.04** (9.0475e-13) | **3.34** (2.1844e-05) |
| RCWA layer cutoff, the verification's deeper ladder | **0.17** (6.7172e-09) | -- |

The round-1 docstring's "7.6 decades above the noise side and 6.9 below the
signal side" was measured on 51 fixtures that carried no cutoff mount, and is
replaced by the table above.  The binding number is the last row: the
verification drove `min|lam^2|` to 4.495e-15 and read a noise-side ratio of
6.7172e-09, 1.5x under the bar.  My own ladder reached only 6.5341e-10 and read
9.0475e-13, so I reproduce the SHAPE of that finding rather than its worst
value, and the verification's number stands as the worst known.

**Why the noise side has no floor, and what that means.**  For
`lam^2 = -s + i eta` the principal root's real part is `eta / (2 sqrt(s))`, so
at fixed backward error `eta` the ratio grows without limit as `s -> 0`.  The
noise side is therefore set by `sqrt(|lam^2|_min)` and not by the eigensolver's
backward error alone -- which means no constant can be proved safe at an
arbitrarily deep cutoff, on EITHER shape.  What can be said, and is:

* the mode that reaches the noise side there is no longer cleanly propagating
  (its `lam^2` sits at ~45 degrees), so its imaginary sign is genuinely
  ambiguous rather than wrong;
* it carries no z-directed flux, and `_inv_lam` regularises it downstream, so no
  wrong answer follows from either choice on any fixture measured -- 72 mounts
  here, 28 in the verification, on both builds;
* and the two shapes fail in opposite directions there, so this is a corner of
  the problem rather than a defect of the constant.

**D4 -- the scale-relative corner, quantified again.**  Because the scale is the
array's largest root, a mode whose real part is a fraction `rho` of its OWN
magnitude is conjugated once its magnitude falls below `_CUT_BAND_REL / rho` of
the spectrum's top.  The worst case over the 72 cutoff mounts here is
`|Re r| / |r| = 6.8698e-08`; the verification's deeper mount reached 2.0751e-03.
Both are recorded in the constant's comment.  The consequence is nil for the
reason above, and it is now documented rather than implicit.

**D5 -- unchanged by round 2, deliberately.**  The verification found one
engineered cutoff mount that moves REFUSED -> WARNED across round 1 at a closure
defect of +4.2519e-02.  Round 2's shared body is BIT-IDENTICAL to round 1's
(section 3), so that mount behaves identically before and after this change:
round 2 neither improves nor worsens it, and it stays an open observation about
the round-1 fix.

---

### 5.5 A cross-check that is NOT this function's: `_forward_branch_flip`

Round 1 adopted `1e-8` from `pmm/_core._forward_branch_flip`, calling it "one
convention for one quantity across the two solvers".  The verification already
qualified that ("both are `1e-8 * max(max|.|, 1)`, but they threshold DIFFERENT
quantities and flip differently"), and measuring the staggered population makes
the qualification concrete: **the two selectors use OPPOSITE conventions.**

* `_sqrt_decay` takes `lam^2`, and a PROPAGATING mode has it real NEGATIVE (on
  the principal square root's cut), so its discriminator is the ROOT's REAL
  part;
* `_forward_branch_flip` takes `q = kz/k0 = sqrt(gamma^2)`, and a PROPAGATING
  mode has `q` real, i.e. `gamma^2` real POSITIVE, so its discriminator is
  `|Im(q)|` and its rule is
  `flip <=> Im(q) < -tol or (|Im(q)| <= tol and Re(q) < 0)`.

Scoring the staggered pencil with `_sqrt_decay`'s ratio -- which is what a
single "census every population the same way" pass does -- therefore measures
the wrong thing, and `b6_band_scale.py`'s `pmm_staggered` row (a 1.67-decade
gap) is that mistake.  `b6c_staggered_flip.py` re-measures it on the quantity
the selector actually thresholds, over 116 eigenvalue arrays and 42,310 modes
from the staggered 2-D engine (scalar, tensor, oblique, lossy, the pure stack)
and the 1-D PMM (TE, TM, conical, a loss ladder):

| classification of "the imaginary part is ROUNDING" | noise side max | signal side min | gap |
|---|---|---|---|
| `\|Im g2\| <= 1e-6 \|Re g2\|` (the `_sqrt_decay` census convention) | 1.5509e-08 | 2.4337e-08 | 0.20 dec |
| `\|Im g2\| <= 1e-12 \|Re g2\|` | **1.2160e-15** | 1.5053e-12 | 3.09 dec |

Neither classification is satisfying, and the reason is instructive: on THIS
convention a relative `1e-6` on `gamma^2` admits modes whose `|Im q| / |q|`
reaches 5e-7 -- six decades above the eigensolver's backward error -- while the
`1e-12` one calls "signal" a population of nearly-propagating modes whose
`Im(q)` is a negligible fraction of `Re(q)`, exactly the modes for which the
`Re(q) < 0` rule is the right discriminator.  The natural quantity for that
selector is `|Im q| / |q|`, not `|Im q| / max|q|`.

What CAN be said cleanly, and is the reason this is a cross-check rather than a
finding: **the worst `|Im q| / |q|` that `_forward_branch_flip` actually acts on
over this whole population is 3.5588e-09.**  Every mode the band reaches has an
imaginary part at most 3.6e-09 of its own magnitude, i.e. unambiguously
rounding, so it never re-signs a physical decay rate here.

`_forward_branch_flip` is not changed by round 2 and no claim about it is made
beyond that.  Whether its scale should be per-mode -- which is a different
question from section 5.3's, because its convention is the other one -- is
recorded as an open item in section 11.

---

## 6. X-1 is CLOSED

X-1 is one of the three unguarded solves the M1 campaign hardened
(`docs/audits/PMM_M1_CONDITIONING_2026_08_04.md`): the explicit `inv(a+b)` in
`rcwa/_core._interface_smatrix` and the two Redheffer star denominators.
`tests/unit/test_m1_conditioning_guard.py` pinned it as REPRODUCED AND OPEN on
the `THIN` family -- period 10 um, ridge 1.55, groove 1.5, substrate 1.5,
SUPERSTRATE 1.5, depth 0.5 um, duty 0.5, `n_orders` 6..30.

The geometry is the tell: the groove index equals BOTH half-spaces', so it is a
permittivity coincidence on the substrate and the superstrate side at once.

`b3_x1.py`, the whole ladder on both arms in one process with the library's own
census armed exactly as the M1 tests arm it:

| ladder / arm | raises | flagged cells | worst \|R+T-1\| | worst relative `sum(R)` | min equilibrated `rcond` |
|---|---|---|---|---|---|
| TE WIN 1 thr PRE | **7** of 25 | **14** of 25 | 3.1956e-02 | **152.60x** (M=21) | 3.331e-19 |
| TE WIN 1 thr POST | **0** | **0** | **1.3323e-15** | 5.064e-02 (M=6, truncation) | 6.250e-02 |
| TE WSL 1 thr PRE | **8** | **14** | 3.1956e-02 | **152.60x** (M=21) | 3.614e-19 |
| TE WSL 1 thr POST | **0** | **0** | **1.4433e-15** | 5.064e-02 | 6.250e-02 |
| TM WIN 1 thr PRE | 5 | 9 | 2.6165e-04 | 1.2998x (M=28) | 3.402e-19 |
| TM WIN 1 thr POST | **0** | **0** | **9.9920e-16** | 1.887e-02 (M=6) | 5.603e-02 |
| TM WSL 1 thr PRE | 5 | 9 | 1.5404e-04 | 0.7654x (M=28) | 2.338e-19 |
| TM WSL 1 thr POST | **0** | **0** | **9.9920e-16** | 1.887e-02 | 5.603e-02 |

The four historically pinned cells, `sum(R)`:

| cell | WIN PRE | WIN POST | WSL PRE | WSL POST |
|---|---|---|---|---|
| M = 12 TE | 2.016454e-04 | 2.015824e-04 | 2.014665e-04 | 2.015824e-04 |
| **M = 19 TE** | **1.838764e-02** (closure 1.8182e-02) | **2.053766e-04** | **RAISED `_EnergyError`** | **2.053766e-04** |
| M = 20 TE | 2.088570e-04 | 2.053491e-04 | 2.262156e-04 | 2.053491e-04 |
| **M = 21 TE** | **3.216567e-02** | **2.095174e-04** | **3.216567e-02** | **2.095174e-04** |

M = 19 TE was pinned precisely because it RETURNED `R + T = 1.018` on one build
and RAISED on the other -- a literal build-dependent answer on the DEFAULT 1-D
path.  It now returns 2.053766e-04 on both, to every printed digit.  The POST
`sum(R)` column is IDENTICAL across builds at every rung.

This is a user-visible correctness improvement on the default `rcwa_efficiency_1d`
path, and it is the largest consequence of the round-1 change.  It was recorded
nowhere until now.

**Where X-1 was recorded as open, and what those places say now.**

| place | before | after |
|---|---|---|
| `test_m1_conditioning_guard.py::test_x1_defect_is_reproduced_and_flagged_but_NOT_closed` (4 params) | asserted the defect still returns and is still flagged; SKIPPED post-fix | replaced by `test_x1_is_closed_on_the_cell_it_was_pinned_at`: the cell returns, closes under `1e-8`, and the census flags nothing |
| `test_m1_conditioning_guard.py::test_the_refusal_reproduces_the_prior_answer_with_the_switch` | asserted the wrong answer comes back with the guard off; SKIPPED post-fix | replaced by `test_the_withdrawn_refusal_moves_no_bit_and_the_ladder_carries_no_silent_defect` |
| both | fail-before was a FOUND cell, per-build and per-thread | fail-before is ENGINEERED: the pre-round-1 body reinstalled in-process, which fires at every thread count on both builds |
| the file's X-1 section comment | "X-1 IS REAL AND IS STILL OPEN" | the ladder table above, dated |
| `CHANGELOG.md`, the round-1 entry | silent on X-1 | the round-2 paragraph records the closure |

The M1 file goes from `22 passed, 5 skipped` (the state the verification found on
the merged tree, plus one failure) to **28 passed, 0 skipped**.

---

## 7. M1: the equilibration instrument is KEPT

`rcwa/_core._guarded_inverse` screens every explicit inverse with
`_rcond_1_equilibrated` and, below the screen, scores it with
`_equilibrated_inverse_residual` rather than with the RAW residual.  The
equilibrated instrument was chosen because a population existed where the raw
one would have REFUSED a correct answer -- "the false positive the equilibration
exists for" of `docs/audits/PMM_M1_CONDITIONING_2026_08_04.md`.

A call is MOTIVATING when the raw residual would refuse it
(`raw > _INV_RESID_REFUSE = 1e-8`) and the equilibrated one rescues it
(`eq <= 1e-8`).  `b9_m1_instrument.py` counts that population on both arms over
a 26-fixture sweep -- the M1 uniaxial cascade at five truncations and at normal
incidence, its detuned control, ten rungs of the X-1 `THIN` ladder, the 2-D
anisotropic coincidence and its lossy sibling, the round-2 uniform-spacer stacks,
a loss ladder, a metal and an out-of-plane tensor cell:

| | PRE | POST |
|---|---|---|
| guarded inverses seen, WIN 1 thr | 110 | 110 |
| MOTIVATING calls | **7** | **0** |
| max raw residual | 5.155e-01 | 4.893e-15 |
| max equilibrated residual | 8.398e-04 | 5.659e-15 |
| max raw / equilibrated ratio | **3.241e+14** | **2.165** |

The seven are the six M1-cascade rungs and the 2-D anisotropic coincidence cell
-- every one of them the branch-cut defect.  Post-fix the sweep contains no call
where the two instruments disagree by more than 2.2x.

**Decision: KEEP the instrument, re-date its justification.**  It is
behaviour-preserving; it costs nothing on the screened path; the ARMED `T22`
refusal is a separate population this change does not touch; and a user's own
coincident geometry can still reach it.  Deleting a guard because its found
population is empty would repeat, one level up, the mistake this campaign exists
to prevent.  **No library code was removed in round 2.**  What changed is the
record: `test_the_withdrawn_refusal_moves_no_bit_and_the_ladder_carries_no_silent_defect`
carries a dated paragraph saying the motivating population WAS the branch-cut
defect and is now empty, and
`test_anisotropic_cascade_is_not_falsely_refused` (restated at `6065e1f`, before
this round) already asserts the well-conditioned state.

`tests/unit/test_m1_conditioning_guard.py` is added to the round-1 fix
document's stated gate: it is the one file the round-1 change turned red, and it
turned red for the right reason.

**Not established here:** my sweep reached the ARMED `T22` site on ZERO of its
110 calls (the out-of-plane fixture I built for it routes elsewhere), so the
verification's reading that the armed population's minimum equilibrated `rcond`
is 2.792e-02 on both arms -- eight decades above its own `1e-10` bar -- stands
un-re-measured here.  It is the verification's, not mine.

---

## 8. The remaining verification defects

**D2 -- the `[True]` fail-before is thread-conditional.**  The pre-fix envelope
quoted above `_CLOSURE_BAR` in `tests/unit/test_fix_rcwa_even_sector_wsl.py` is
a FULL-path list, and the test is parametrized over both paths; the EVEN path's
pre-fix closure reaches the arithmetic floor at three of fourteen (build, thread)
samples.  The comment and the docstring now say so explicitly and name three
thread-INDEPENDENT fail-befores for the same defect: BAR 5 of that file (the
on-cut mode census, 5..13 of 16 pre-fix at every setting), gate 2 of
`tests/unit/test_verify_rcwa_even_sector.py` (the uniform-spacer RCWA stack,
REFUSED at `sum R + T = 26` pre-fix), and gate 3 of
`tests/unit/test_fix_branch_cut_round2.py`.  The POST assertion, which is what
gates the release, was never thread-conditional and is unchanged.

**D6 -- the remedy text.**  `_check_energy`'s `_EnergyError` message and its
`_EnergyWarning` sibling both said: if a LAYER permittivity is exactly equal to
a REGION's, "DETUNE one of the coincident permittivities by a relative ~1e-6".
Two things are now wrong with that.  The partner need not be a region -- a
UNIFORM LAYER of the same stack gets its modes from the same analytic Rayleigh
helper, in exact arithmetic, and section 4 shows it is what actually broke the
PMM.  And the detune is not the remedy: measured on the uniform-spacer stack's
PRE arm, a relative `1e-6` detune leaves the closure defect at 8.056e-05 (WIN) /
2.307e-04 (WSL), and only `1e-3` brings it to 8.7e-14 / 1.3e-12.  Both messages
now name a coincident uniform LAYER alongside a region and, instead of
recommending a detune, say that the class was repaired on 2026-09-11 and that
reaching the guard on such a geometry is a report-worthy regression.  The
docstring records the same, with the measurement.

**D3 / D4 / D5** are answered in section 5.

---

## 9. The census

`b4_census.py`, 49 fixtures per build, both branch bodies installed in one
interpreter.  The three classes carry different obligations and are reported
separately, because a single "did anything move?" number would hide the whole
content of the change.

| class | n | bit-identical | worst motion (WIN) | worst fixture |
|---|---|---|---|---|
| LOSSY | 11 | **11 of 11** | **0.0000e+00** | -- |
| ON-coincidence lossless | 21 | 12 | **2.9263e-05** | `stack_spacer_coinc` |
| OFF-coincidence lossless | 17 | 9 | 4.5385e-13 | `stack_strong_off` |

Surfaces covered: `pmm_efficiency_2d_cell` (weak and strong modulation, on and
off the coincidence, folded and full, oblique, conical, three loss rungs and a
metal); `pmm_jones_2d` (on, off, lossy); `PMM2DStackHybrid` (uniform spacer,
detuned spacer, per-layer, SLANTED, lossy, oblique, tensor);
`pmm_efficiency_2d_staggered` and `pmm_jones_2d_staggered` (in-plane tensor,
OUT-OF-PLANE, MAGNETIC, lossy, slanted); `PMM2DStackPure` (spacer, per-layer,
lossy); `pmm_efficiency_1d` / `pmm_jones_1d`; and the RCWA surfaces that share
the function -- `rcwa_jones_2d`, `rcwa_efficiency_2d`, `rcwa_efficiency_1d`,
`RCWAStack`.

Three readings are worth calling out.

* **Every RCWA surface is bit-identical between the arms** (`0.000e+00`, 6 of
  6), which is the refactor half of section 3 seen from the solve side rather
  than from the function's inputs.
* **Every staggered and 1-D PMM surface is bit-identical** -- 0.000e+00 on all
  of them, including the OUT-OF-PLANE and MAGNETIC ones -- because they never
  call the function.  That is the scope claim of section 2, measured.  (Two
  census rows produced no number because the FIXTURE, not the arm, refuses
  them: `stag_slant` raises `NotImplementedError` on both arms -- the staggered
  entry's slant is scoped narrower than I assumed -- and `onedjones_conical`
  raises `TypeError` from a signature I got wrong.  Both raise identically on
  both arms, so neither hides a motion; the slanted PMM path is covered by
  `stack_slant` and `stack_slant_coinc` instead.)
* **`stack_slant_coinc` is bit-identical (0.000e+00) while `stack_spacer_coinc`
  moves 2.9e-05.**  A SLANTED patterned layer is solved through the 4N
  first-order generator and `_generator_modes`, not through `_sqrt_decay`, so
  the slanted arm of the hybrid stack was never exposed.

The refactor's own bit-identity, over 4,010 engineered values at six array
sizes: `True`, `max |diff| = 0.000e+00`, both builds.

---

## 10. Runs

All with `-p no:randomly`.  "pinned" is `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
on the command line, which is the setting that exposed the round-1 defect.

| battery | Windows py3.14 | WSL py3.12 |
|---|---|---|
| the GATE: `test_fix_rcwa_even_sector_wsl.py` + `test_verify_rcwa_even_sector.py` + `test_m1_conditioning_guard.py` + `test_v5_14_2_backlog_batch.py` + `test_fix_branch_cut_round2.py`, pinned | **86 passed**, 107.67 s | **86 passed**, 114.32 s |
| the same, threads UNPINNED | **86 passed**, 2991.10 s | **86 passed**, 3551.84 s |
| `test_pmm*.py` + `test_fix_pmm*.py` + `test_verify_pmm*.py` + `test_rcwa*.py` + `test_niche*rcwa*.py`, pinned | **728 passed, 1 skipped**, 2464.70 s | not run (see section 11) |
| the JAX-guarded PMM / RCWA / Berreman files, pinned | **66 passed**, 180.63 s | not run (see section 11) |
| census / walker sweep (`walker\|census\|dispatcher_pin\|public_api\|doc_consistency`, `-x`) | **1284 passed, 12 skipped**, 254.22 s | not run (see section 11) |
| `ruff check lumenairy/ tests/ validation/probe_fix_branch_cut_round2/` (WSL) | -- | **All checks passed!** |

The one Windows skip in the PMM/RCWA battery is the `threadpoolctl`-dependent
`test_niche_audit_m4_m5_m6_rcwa.py::test_set_blas_threads_numerically_equivalent`.
The 12 skips in the census sweep are the walker's own
"this CHANGELOG block has no such claim" arms.

**The M1 file is the load-bearing row.**  It was the ONE file the round-1 change
turned red, and the verification's ship condition 5 asked for it to be added to
the battery that gates this work.  It is, and it now reads **28 passed, 0
skipped** on both builds -- against `1 failed, 21 passed, 5 skipped` on the
merged tree before this round.

The `DLASCL` lines appear in the WSL logs, exactly twice, from
`test_m1_conditioning_guard.py::test_guarded_lstsq_stands_aside_on_a_non_finite_system`
-- the deliberate NaN matrix the verification attributed them to (its section
10).  They are Fortran unit-6 output from `zgelsd`, not this solve's, and not a
defect.

New DECISION tests, `tests/unit/test_fix_branch_cut_round2.py`, 17 tests,
104.02 s for the whole file pinned; the slowest single test is 42.29 s
(`test_the_pure_staggered_engine_never_reaches_this_function`, whose cost is
the staggered engine's own first solve), well inside the 60 s budget.  Node ids
spliced into `.test_durations` (5 removed for the two renamed M1 tests, 21
added, sorted, json-valid).

---

## 11. What is NOT established

* **The widest admissible `_CUT_BAND_REL`.**  Section 5 chooses the SHAPE by
  measurement and re-derives the shipped constant's margins, but no ladder was
  run over the constant itself.  It is bounded from below by the cutoff
  population (a mode reaches 6.7172e-09 in the verification's deeper ladder, so
  anything at or under ~1e-9 would start missing modes) and from above by the
  signal side (2.1844e-05 at a cutoff mount, 1.2612e-02 in ordinary ones).
* **A deeper cutoff than `min|lam^2| = 6.5341e-10`.**  My bounded minimisation
  reached that; the verification's trisection reached 4.495e-15.  I therefore
  reproduce the SHAPE of D3/D4 rather than their worst values, and quote theirs
  as the binding ones.
* **Whether `_forward_branch_flip`'s scale should be per-mode.**  Section 5.5
  shows its two-sided gap is not cleanly measurable with either classification I
  tried, because its natural discriminator is `|Im q| / |q|` rather than
  `|Im q| / max|q|`.  What is established is that the worst `|Im q| / |q|` it
  actually acts on over 42,310 modes is 3.5588e-09 -- unambiguously rounding --
  on both builds.  Round 2 does not change that function.
* **The ARMED `T22` refusal population.**  My M1 sweep reached it on 0 of 110
  guarded inverses, so the verification's reading (minimum equilibrated `rcond`
  2.792e-02 on both arms, eight decades above its own `1e-10` bar) stands
  un-re-measured here.
* **The WSL side of the wide batteries.**  The PMM/RCWA battery, the JAX files
  and the census sweep were run pinned on Windows only; WSL carried the gate
  battery (pinned and unpinned) and every probe.  The box is shared and the
  Windows PMM/RCWA battery alone took 41 CPU-minutes.
* **CuPy / GPU.**  Every measurement here is NumPy or JAX on CPU.  The shared
  body is `xp`-generic and `array_namespace` routes CuPy the same way it routes
  NumPy, but no GPU arm was run.
* **`stabilize=True`.**  Not re-characterised; its retry schedule exists for
  this failure class and whether it can now be narrowed is still open, as both
  round 1 and the verification say.
* **Whether the hybrid PMM has OTHER coincidence partners.**  Sections 4.2-4.8
  establish the REGION, the SUPERSTRATE and the UNIFORM LAYER.  A slanted
  patterned layer is exempt (it solves through the 4N generator, and
  `stack_slant_coinc` is bit-identical between the arms), but "no other partner
  exists" is not a claim any of this supports.

---

## 12. Reproduction

```
git worktree add -b fix/branch-cut-round2 C:/tmp/lum_bc2 2898767

# every probe, either build -- PYTHONPATH=. is REQUIRED (each probe calls
# b_fixtures.require_local_tree(), which REFUSES to produce a number if
# lumenairy was imported from anywhere but the working directory)
cd C:/tmp/lum_bc2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=. python -u \
  validation/probe_fix_branch_cut_round2/b7_spacer.py out.json
```

Every JSON carries a `_stamp` block naming the `lumenairy.__file__` it measured,
the arm it detected from the LIVE source of the PMM copies, the interpreter,
numpy, the platform and the three thread environment variables.

| probe | what it measures |
|---|---|
| `b0_smoke.py` | which surfaces reach which selector at all |
| `b1_pmm_wrong.py` | 30 PMM surfaces, both arms, with the eig census |
| `b2_pmm_interface.py` | `cond(a+b)` at every PMM interface mode-match |
| `b3_x1.py` | the X-1 `THIN` ladder, both arms, census armed |
| `b4_census.py` | the 49-fixture bit-identity census + the refactor's own |
| `b5_jax.py` | the three JAX twins: forward, NumPy parity, gradients |
| `b6_band_scale.py` | ARRAY-MAX vs PER-MODE over the ordinary populations |
| `b6b_cutoff.py` | the same, on 72 mounts driven onto a LAYER CUTOFF |
| `b6c_staggered_flip.py` | `_forward_branch_flip`'s own band, on its own quantity |
| `b7_spacer.py` | the uniform-spacer family: class, detune, modulation, threads |
| `b8_reference.py` | the repaired answer against an independent RCWA solve |
| `b9_m1_instrument.py` | the M1 equilibration instrument's motivating population |
| `b10_manufactured_energy.py` | the mount that returns `sum R + T` up to 110 |

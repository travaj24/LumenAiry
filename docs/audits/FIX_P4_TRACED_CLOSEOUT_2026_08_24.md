# CLOSE-OUT -- the P4 single-red, the C13 screen, and D14's dead guard

Branch `fix/p4-and-traced-closeout` off `origin/main` @ `bf06b23` (the 5.42.0
release commit).  Three items left open by
`docs/audits/BUILD_DETERMINISTIC_TRACED_FIT_2026_08_23.md`: its S14.6 single
unreproduced failure, its S13 C13-screen question, and its S13 dead `n == 0`
guard.

Everything below was measured on this branch with `PYTHONPATH` pinned and
`lumenairy.__file__` asserted against `LUMENAIRY_ROOT` in every probe, at
explicit OMP/OPENBLAS/MKL widths.  Windows 11, py3.14.6, numpy 2.4.4 /
scipy-openblas 0.3.31.188.0 (DYNAMIC_ARCH, Haswell kernel), Ryzen 9 5950X
(12 physical / 24 logical), 137 GB.  Second build where stated: WSL Ubuntu,
py3.12.3, numpy 2.4.6 / scipy 1.17.1.

---

## 0. VERDICT

| item | verdict |
|---|---|
| **S14.6 P4 single red** | **BOUNDED, NOT REPRODUCED** -- 782 diagnosed/undiagnosed pairs (1564 calls), 0 mismatches, ONE field hash, under 6-to-8-way process load, BLAS widths 1/2/4/8/16 and 4 in-process threads; plus 40 000 solves of the path's only BLAS-adjacent step, one coefficient hash.  The path is also proved STATE-FREE by construction.  **S14.6's own by-construction argument is REFUTED** -- the GBD re-expansion does reach `_solve_lstsq_thread_safe` -- but its conclusion survives for a different, measured reason. |
| **C13 screen at 1e-8** | **RIGHT AS SHIPPED, JUSTIFIED BY MEASUREMENT.**  Derived bars: rcond 6e-14 (residual criterion) and 6e-12 (coefficient criterion), 4-5 decades BELOW the screen, identical on both builds.  The traced fits' rcond is a BASIS CONSTANT (1.61e-9 +- 2% at 28 terms over 15 solves / 5 fixtures; 6.4e-11..1.0e-10 at 120), so the step-down is unconditional there by construction, and conditioning the basis instead costs exactly the QR the re-solve already performs.  Nothing widened, nothing tightened. |
| **`n == 0` in `_det_normal_equations`** | **CONFIRMED DEAD, NOW REACHABLE.**  The guard is hoisted above the reshape that pre-empted it, so the deterministic kernel returns what `(A.T @ A, A.T @ b)` returns on an empty design matrix.  No shipped path moves; `_solve_lstsq_thread_safe` still raises on an empty fit, on both routes, exactly as before. |

Two library changes, both small, both with a measured fail-before; one is a
silent-wrongness surface found while enumerating S14.6's suspects and is
declared as such below (S1.6).

---

## 1. ITEM 1 -- THE P4 SINGLE RED

### 1.1 WHAT THE ASSERTION ACTUALLY ASSERTS

`test_frame_completeness_metric_published` ends:

```python
E_a = _gbd(E, _m5_biconcave(), reexpand='auto', diagnostics=diag)
...
E_b = _gbd(E, _m5_biconcave(), reexpand='auto')
assert np.array_equal(E_a, E_b)
```

Read against `apply_real_lens_gbd`, `diagnostics` is consumed at ONE place --
after `E_out` exists -- and only read from (`p_in_ap`, `p_out_raw`, the two
completeness numbers computed on the re-expansion branch whether or not a dict
was passed).  So the two calls differ in nothing that can reach the field, and
the assertion is a **call-to-call determinism** claim wearing a
"diagnostics is read-only" label.  That is the right thing to adjudicate.

### 1.2 THE BY-CONSTRUCTION ARGUMENT S14.6 MADE IS WRONG

S14.6:

> `_solve_lstsq_thread_safe` **does not exist anywhere in the GBD module** --
> `grep` over `lumenairy/` puts every call site in `_lens_traced.py` and
> `_lens_imap.py` only.  [...] the GBD re-expansion reaches neither.

The grep is right about where the *call sites* are and wrong about what the
GBD path *reaches*, because the reach is one import hop long:

```
lenses_gbd._carrier_referenced_bundle
  -> from ._lens_traced import _compute_carrier      (lenses_gbd.py:200)
     -> _solve_lstsq_thread_safe(A, rhs,             (_lens_traced.py:4700)
            deterministic=bool(DETERMINISTIC_NORMAL_EQUATIONS))
```

MEASURED, not read (`validation/probe_p4_reexpand/census_solves.py`, which
wraps the solver script-side and runs the exact test body):

```
--- iteration 0: 2 solves (1 diagnosed / 1 undiagnosed) ---
  [0] shape=(82092, 5) det=True rcond=9.935e-01 qr_calls=0 refine=- coef=85579519a087
  [1] shape=(82092, 5) det=True rcond=9.935e-01 qr_calls=0 refine=- coef=85579519a087
```

**One least-squares solve per call, and it is a traced-module solve.**  The
conclusion "not D15's" survives, but for a reason that had to be measured
rather than grepped: at `M = 5` the fit is below `_DET_EINSUM_MIN_TERMS` (8),
so it takes D14's ufunc partial, which D15 did not touch; and at
`rcond = 0.9935` it is eight decades clear of the C13 screen, so it takes
neither the QR nor the refinement D15 rewrote.  A grep that had found the call
site would have reached the same answer with the evidence attached.

This is the reusable part: **"function F is not called from module M" is not
the same claim as "module M's path does not reach F", and only the second one
adjudicates anything.**

### 1.3 THE SHARED MUTABLE STATE, ENUMERATED AND THEN MEASURED

Enumerated by reading the path (`apply_real_lens_gbd` ->
`decompose_field_to_beamlets` -> `_prune_zero_beamlets` -> `frame_completeness`
-> `_carrier_referenced_bundle` -> `_compute_carrier` ->
`_solve_lstsq_thread_safe` -> `apply_prescription_persurface_to_beamlets` ->
`reconstruct_field_from_beamlets`):

| candidate | verdict |
|---|---|
| `numexpr` out-retention (the campaign record's class) | **NOT ON THIS PATH.**  `grep --include='*.py' numexpr lumenairy/` puts every use in `elements/lenses.py` (the scaffold) and `elements/lenses_maslov.py`.  The GBD and traced-carrier paths import neither. |
| module flags read at CALL time (`DETERMINISTIC_NORMAL_EQUATIONS`, `DETERMINISTIC_TRACED_FIT`, `LSTSQ_CONDITIONING_STEPDOWN`, `_LSTSQ_GRAM_RCOND_MIN`, `_DET_EINSUM_MIN_TERMS`, `_DET_GRAM_TILE_BYTES`) | live inputs, but nothing on the path writes them (S1.4) |
| `_TRACED_KWARG_DEFAULTS_CACHE` / `_MAIN_GUARD_CACHE` / `_PERSISTENT_POOL` + their locks | `apply_real_lens_traced` machinery; the GBD path calls `_compute_carrier` directly and reaches none of them |
| glass tables (`GLASS_REGISTRY` etc.), mutated by conftest's module guard | module-scoped, so constant across the two calls inside one test |
| `LUMENAIRY_MEM_BUDGET_MB` | **A REAL BYTE-MOVING KNOB** -- see S1.5 |
| warning filters / `__warningregistry__` | `_gbd` enters `warnings.catch_warnings()` per call, which saves and restores them; no numeric effect |
| FFT plan caches | none: the P4 path never enters `_reconstruct_fft` or `_fft_upsample` (S1.6) |
| in-process threading | none in the library on this path; exercised anyway as an arm (S1.7) |

### 1.4 THE PATH IS STATE-FREE, MEASURED

`validation/probe_p4_reexpand/statefree.py` snapshots every module-level
binding of every imported `lumenairy` module (868 bindings over 151 modules:
dict key sets, sequence contents, ndarray shape + L1, scalar reprs), plus the
whole of `os.environ`, plus `warnings.filters`, plus `np.geterr()`, runs the
DIAGNOSED call between two snapshots, and diffs (after a warm call, so lazy
first-call initialisation is not what the diff reports):

```
tracked bindings: 868 over 151 lumenairy modules
DIFF: none -- the diagnosed call left NO module-level state, no environment
change, no warning-filter change and no NumPy error-state change behind.
pair equal=True          h_a=18c25df511ad134f h_b=18c25df511ad134f
reverse-order equal=True h_c=18c25df511ad134f h_d=18c25df511ad134f
input array untouched by either call: True
```

For the diagnosed call to change the undiagnosed one it has to leave something
behind.  It leaves nothing.  That is the by-construction half; the stressor is
the empirical half.

### 1.5 THE ONE ENVIRONMENT VARIABLE THAT DOES MOVE THE FIELD

`LUMENAIRY_MEM_BUDGET_MB` is a hard ceiling on the windowed reconstruct's
per-chunk budget, read at CALL time, and the chunk boundaries fix the
summation order.  Measured (`common.py`, one process per value):

| `LUMENAIRY_MEM_BUDGET_MB` | field hash | `frame_completeness` |
|---|---|---|
| (unset) | `18c25df511ad134f` | 0.999813644838 |
| 1 | `299a670507242cec` | 0.999813644838 |
| 8 | `214b788733e5e36f` | 0.999813644838 |
| 64 | `019a54346ffb675c` | 0.999813644838 |
| 512 (= the default) | `18c25df511ad134f` | 0.999813644838 |

So a leaked value changes every field bit this test reads -- and **cannot break
this assertion**, because both calls in one process read the same value.  It is
now named in the failure message rather than left to be rediscovered.  (Note
also that `frame_completeness` does not move to 12 digits: the chunking moves
the last bits, not the physics.)

### 1.6 A SILENT-WRONGNESS SURFACE FOUND WHILE LOOKING, AND FIXED

The audit's own candidate mechanism was resource pressure.  There is exactly
one place on this path where pressure could have changed the ARITHMETIC:

```python
try:
    return _fft_applicable_impl(beamlets, Nx, Ny, dx, dy, centre)
except Exception:      # jax tracer (jit) / non-inspectable array -> dense
    return False
```

`MemoryError` is a subclass of `Exception`.  The two branches this decision
picks between -- the FFT convolution and the windowed scatter-add -- are the
same sum in a different order and agree to ~7e-16, not bit for bit
(measured 2026-08-24 over three bundles the FFT route applies to: 6.61e-16 /
9.79e-16 / 5.28e-16 relative, byte-identical on none).  So a
swallowed `MemoryError` silently changes a reconstruct's arithmetic route as a
function of how much memory the box had at that instant.  **`MemoryError` now
re-raises.**  Nothing is lost: the inspection allocates a couple of `(n, 3)`
temporaries, orders of magnitude under the reconstruct that follows either way.

Fail-before, measured by stashing the change and re-running:
`DID NOT RAISE <class 'MemoryError'>` (it silently returned a windowed field);
the sibling tracer-fallback arm passes on both sides, so the behaviour that
handler exists for is untouched.

**It is NOT the S14.6 mechanism, and that too is measured rather than assumed.**
On this fixture the decision is not near its boundary -- all three reconstructs
take the windowed route with the FFT route inapplicable by ten decades:

```
reconstruct[0]: n=16384 route=windowed fft_applicable=False dir_ptp=[4.170e-02, 4.170e-02] (boundary 1e-12)
reconstruct[1]: n=16384 route=windowed fft_applicable=False dir_ptp=[1.087e-01, 1.087e-01] (boundary 1e-12)
reconstruct[2]: n=16384 route=windowed fft_applicable=False dir_ptp=[3.374e-02, 3.374e-02] (boundary 1e-12)
```

### 1.7 THE STRESSOR, AND THE BOUND IT ESTABLISHES

No supervisor, no retry, no re-run-to-green: each arm runs its iteration count
and reports.  Every iteration writes a JSONL line carrying the pair verdict,
both field hashes, `max|E_a - E_b|`, and every bar the test asserts -- the
S14.6 instrumentation lesson applied to the instrument itself.

| arm | shape | result |
|---|---|---|
| `run_wave1.sh`: 6 x `stress_pairs` @ `OMP=8` + 2 x `stress_fit` @ `OMP=8`, concurrent (8 processes, 64 BLAS threads on 24 logical cores -- harsher than the 5 pytest x 8 the failure was seen under) | 360 pairs | **0 mismatches, 0 cross-iteration drifts, 1 field hash** |
| the same wave's fit hammer -- the path's ONLY BLAS-adjacent step, `_compute_carrier`'s `(82092, 5)` fit, at high count under that load | 40 000 solves | **1 coefficient hash, 1 Gram hash, 1 rhs hash, 1 rcond value (0.9934560381553126)** |
| `run_wave2.sh`: 5 x `stress_pairs` at BLAS widths 1 / 2 / 4 / 8 / 16, concurrent | 300 pairs | **0 mismatches, 0 drifts, 1 field hash ACROSS ALL FIVE WIDTHS** |
| `stress_threads`: 4 in-process threads, barrier-synchronised, sharing every module-level binding | 80 pairs | **0 mismatches, 1 distinct field hash** |
| `run_interleaved.sh`: A/B/A/B, one arm alone against the same arm inside a 6-way load (the timing control, S1.7.1) | 42 pairs | **0 mismatches** |

**782 pairs = 1564 calls of `apply_real_lens_gbd(reexpand='auto')`, one field
hash `18c25df511ad134f` throughout, zero cross-iteration drifts, zero
exceptions** -- across 22 processes, five BLAS widths, four concurrent threads,
and a deliberately oversubscribed box.

The bound is therefore: **not reproduced in 782 pairs under up to 8-way
concurrent load.**  That is a bound, not an exoneration, and this item closes
as bounded.

#### 1.7.1 The load was real, measured INTERLEAVED

A sequential "alone, then loaded" comparison measures the box's mood as much as
the load, so the control alternates A (one arm alone) with B (the same arm
inside a six-way concurrent load), twice, three pairs each:

| block | per-pair wall time, min / median / max |
|---|---|
| alone, rep 1 | 6.42 / **6.52** / 7.15 s |
| loaded, rep 1 | 11.48 / **12.50** / 13.14 s |
| alone, rep 2 | 6.39 / **6.40** / 7.18 s |
| loaded, rep 2 | 11.58 / **12.17** / 13.37 s |

**1.9x, and both repeats agree.**  The arms were contending; the invariance is
not an artefact of an idle box.

### 1.8 WHAT THE MARGINS SAY ABOUT THE OTHER CANDIDATE ASSERTION

S14.6's other suspect was `diag['frame_completeness'] > 0.99`.  Every decision
the test depends on, measured on this build:

| gate | reading | bar | margin |
|---|---|---|---|
| input angular spread vs the Husimi threshold (gates the re-expansion at all) | 2.465e-02 | 5.000e-04 | **49.3x** |
| naive frame completeness vs `reexpand_threshold` (decides whether it fires) | 0.8679260789 | < 0.98 | **0.112 absolute** |
| `frame_completeness` | 0.9998136448380414 | > 0.99 | **+9.81e-03** |
| `frame_completeness_reexpanded` | 0.9990682552238545 | [0.99, 1.01] | +9.07e-03 / +1.09e-02 |

None of these is a last-bits bar.  For `frame_completeness` to fail, the
re-expansion would have to not fire at all (which drops it to 0.868 and takes
`frame_completeness_reexpanded` out of the dict with it) -- and the two gates
that decide that have 49x and 0.112 of room.  So a recurrence, if there is one,
lands on a DECISION, not on a rounding; the hardened assertions now name which.

### 1.9 THE TEST, HARDENED

`tests/unit/test_niche_p4_gbd_reexpand.py`:

* every assertion carries its reading and its margin -- `frame_completeness`
  with the measured value and the 2026-08-24 margin, the missing-key assertion
  with the gate that decides the key's presence, and the byte-identity
  assertions through a new `_byte_id_message` that reports `max|diff|`, the
  same as a fraction of peak, **how many of the 147 456 cells differ**, and
  both field hashes.  A failure now arrives adjudicable, whatever `grep` the
  batch is piped through;
* `LUMENAIRY_MEM_BUDGET_MB` is read and reported in every failure message
  (S1.5), so an environment leak between parallel pytest processes announces
  itself instead of looking like a library defect;
* the claim is made TWO-SIDED: undiagnosed-then-diagnosed as well as
  diagnosed-then-undiagnosed, plus the cross-pair identity, which is what makes
  it a determinism assertion rather than an accident of ordering;
* the same `_byte_id_message` replaces the one other bare `np.array_equal` in
  the file (`test_reexpand_does_not_fire_on_diverging_positive`).

There is no tmp-file surface to harden: this module writes no files and takes
no `tmp_path`.

---

## 2. ITEM 2 -- THE C13 SCREEN, ADJUDICATED

The open question:

> The 28-term traced Chebyshev fits screen singular at rcond 1.6e-9 on every
> fixture measured.  That is two orders under the C13 screen and it is worth
> asking whether the basis or its weighting should be conditioned rather than
> re-solved.

### 2.1 THE INSTRUMENTS, AND THE ORACLE THAT IS PROVED BEFORE IT IS USED

`validation/probe_c13_screen/` captures every `(A, b)` a traced call makes
(`capture.py`, solver wrapped script-side) and adjudicates them offline
(`adjudicate.py`).  Five fixtures, 29 fits, 39 solves counting multi-RHS
columns: a singlet at `ray_subsample=2` and without, the same singlet at
N = 768 / dx = 20 um, the P4 module's own M5 biconcave with a converging input,
and a design-121-LIKE arm -- a fast (R = 25 mm) lens with a DECENTRED
illuminated patch and a tilted carrier, which is what puts the fit disc off the
basis centre and engages the weighted two-scale rows C13 was written for.

The oracle is `min ||b - A x||` to full working precision, built two
independent ways and required to agree:

* **QR + extra-precise-residual refinement** -- Householder QR in float64, then
  refinement whose residual is computed with D14's exact `_two_product` and
  `math.fsum` (so each row is its exact value, rounded once);
* **the normal equations at 60 decimal digits in mpmath** -- exact mathematics,
  50 digits of margin over a `cond(G)` of 1e9.

Measured agreement on the three `(1457, 28)` fits: **9.31e-17 / 1.72e-17 /
3.87e-17** relative to peak.  The refinement's correction sequence is
`3.8e-14 -> 6.1e-17 -> 6.1e-17 -> 6.1e-17` -- converged in one step and
stagnant after, on every fit including the 120-term ones.  A vectorised
double-double residual (used above 2e5 entries, where the per-row `fsum` loop
stops being affordable) returns a **bit-identical** answer to the `fsum` route
on the small fits, which is how it earns its use on the large ones.

### 2.2 THE CONDITIONING IS A PROPERTY OF THE BASIS, NOT OF THE OPTICS

| fit family | rcond (equilibrated Gram) | cond(A) | solves |
|---|---|---|---|
| carrier, `M = 5` | 1.000e+00 (concentric) / 5.168e-03 (decentred) | 5.94e+02 / 2.32e+03 | 4 |
| traced Chebyshev, `M = 28` | **1.609e-09 .. 1.663e-09** (spread 3.4%) | 3.140e+04 .. 3.192e+04 | 15 |
| inverse-map exit fits, `M = 120` | **6.368e-11 .. 9.963e-11** | 1.317e+05 .. 1.642e+05 | 20 |

Across four prescriptions, apertures from 10 to 24 mm, row counts from 1337 to
270 220, and with the weighted / decentred arm included, the 28-term rcond
moves by 3.4%.  **The decentred+weighted fixture reads 1.617e-09 against the
concentric unweighted 1.612e-09** -- so the weighting is NOT what puts these
fits under the screen, which refutes the natural reading of the open item.

What is: the total-degree tensor-Chebyshev basis on a disc-shaped sample set.
Measured (`basis.py`):

| | `singlet_nosub_03` (1457 x 28) | `fast_decentred_03` (27477 x 28) | `singlet_nosub_04` (1337 x 120) |
|---|---|---|---|
| cond(A), shipped basis | 3.1401e+04 | 3.1848e+04 | 1.6417e+05 |
| rcond(G) raw | 1.0142e-09 | 9.8590e-10 | 3.7102e-11 |
| rcond(G) equilibrated (what the screen reads) | 1.6625e-09 | 1.6170e-09 | 6.3680e-11 |
| cond(A), columns normalised | 2.4526e+04 | 2.4869e+04 | 1.2531e+05 |
| cond(A), orthonormal basis of the SAME space | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |

**Column scaling buys 1.28x.**  The only conditioning that helps is
orthogonalising the basis over the sample set -- which is a QR of `A`, i.e.
exactly the operation the C13 re-solve already performs.  So "condition the
basis rather than re-solve" is not a cheaper alternative; it is the same
alternative, and the re-solve additionally returns the answer.  That closes the
open item's second half.

### 2.3 THE SHIPPED DEFAULT DOES NOT REROUTE TO QR -- IT REFINES

The open item (and `probe_traced_det/p09_stepdown.py`, which pre-dates D15)
describes the screened-in fits as taking `_solve_lstsq_qr`.  On the SHIPPED
5.42.0 defaults they do not.  Branch census (`branch.py`, spying on
`_det_refine`, `_solve_lstsq_qr` and `_lstsq_residual`):

| fit | `DETERMINISTIC_TRACED_FIT=True` (shipped) | `=False` (the 5.41.0 route) |
|---|---|---|
| `M = 5` carrier | not screened; no refine, no QR, no vote | same |
| `M = 28` traced | screened; **refine APPLIED, no QR, 0 residual votes** | QR, **2 residual votes** |
| `M = 120` imap | screened; **refine APPLIED, no QR, 0 residual votes** | QR, **2 residual votes** |

So on the shipped path the screen's job is to decide **which solves get D15's
refinement**, and C13's residual vote never runs on the traced chain at all.

That matters, because the residual vote is degenerate for one of these
families.  The 120-term exit fits are exact to rounding -- `r*/||b||` = 7e-16
to 2.5e-13 -- so `||b - A x||` compares two rounding-level numbers: on the
ladder below, at a perfectly conditioned `cond(A) = 1e2`, the QR answer's
residual is already **20% above** the normal-equations one while its
coefficients are 7e-15 correct.  **The only thing keeping that noise
comparison off the shipped path is D15's refinement arm winning first.**
Named here rather than left implicit; it is an argument against ever widening
the screen, not against the screen.

### 2.4 DOES THE REROUTE CHANGE ACCURACY?  YES, BY 3-7 ORDERS

Errors relative to the oracle, max-norm over coefficients, scaled by peak:

| family | plain normal equations | QR | shipped (screen -> refine) |
|---|---|---|---|
| `M = 28` (15 solves) | 9.80e-11 .. 3.19e-08 | 6.31e-15 .. 8.52e-13 | **1.37e-15 .. 1.15e-14** |
| `M = 120` (20 solves) | 6.92e-09 .. 3.65e-07 | 5.83e-16 .. 3.63e-13 | 2.01e-14 .. 6.68e-13 |
| `M = 5` (4 solves, never screened) | 1.11e-16 .. 5.63e-14 | 2.11e-15 .. 1.15e-14 | identical to plain (correctly) |

On the 28-term fits the shipped answer beats the QR one on all fifteen solves
(**1.14x to 74.8x**, median 8.1x) and the plain normal-equations one by 3 to 7
orders.  The step-down earns its cost.

### 2.5 THE DERIVED BAR -- WHERE THE SCREEN WOULD HAVE TO SIT

The screen's stated requirement is in the source:

> it must not skip a solve whose two candidates could differ by more than
> `_LSTSQ_RESID_MARGIN` [1e-6, relative, on `||b - A x||`]

`ladder.py` engineers the state instead of hoping for it (TESTING_STANDARDS
rule 3): it takes a captured `A`, replaces its singular spectrum with a
geometric one of prescribed `cond(A)`, keeps the fit's own column geometry, row
count and residual fraction, and measures at each rung what the screen reads
and how far the plain normal-equations answer is from the oracle.  On the
28-term fit (`residual fraction 4.174e-08`):

| cond(A) | rcond read | screened? | coefficient error (NE) | residual excess (NE) |
|---|---|---|---|---|
| 1.00e+02 | 1.259e-04 | no | 2.49e-13 | +1.47e-10 |
| 1.00e+04 | 1.512e-08 | **no (last skipped rung)** | 4.73e-10 | +3.80e-11 |
| 3.16e+04 | 1.599e-09 | yes | 2.09e-08 | +1.76e-09 |
| 1.00e+06 | 1.920e-12 | yes | 6.82e-06 | +2.70e-07 |
| 1.00e+07 | 2.187e-14 | yes | 2.47e-04 | **+4.34e-06** |
| 3.16e+07 | 2.347e-15 | yes | 8.74e-03 | +6.00e-04 |

Log-log interpolation of the two rungs that bracket each criterion:

* **residual criterion**: the plain-NE excess reaches the 1e-6 margin at
  **rcond = 6.1e-14** -- **5.2 decades below the shipped 1e-8**;
* **coefficient criterion**: the plain-NE error reaches 1e-6 of peak at
  **rcond = 5.9e-12** -- **3.8 decades below the shipped 1e-8**.

Both ladders reproduce **byte-identically on the second build** (WSL py3.12.3 /
numpy 2.4.6 / scipy 1.17.1: every rcond, every error and every excess agrees to
all printed digits with the Windows py3.14.6 / numpy 2.4.4 run), as does the
whole 28-term adjudication table.  So the derived bar is not one build's
reading.

### 2.6 THE ADJUDICATION

**1e-8 is the right threshold and it does not move.**

* it cannot skip a solve that needed the step-down: the derived bars sit 3.8
  and 5.2 decades below it, on both builds;
* it must not be TIGHTENED: the traced fits sit at 1.61e-9 and 6.4e-11, only
  0.79 and 2.2 decades under it.  A screen at 1e-11 would take the 28-term
  fits off the refinement and restore their 1e-10..3.2e-8 coefficient error --
  a 4-to-7 order regression on the path whose determinism 5.42.0 exists to
  guarantee;
* it must not be WIDENED either: widening past 5.2e-3 would pull the DECENTRED
  carrier fit in, and past 1e-8 in general it starts putting near-exact fits in
  front of a residual comparison that is provably noise for them (S2.3);
* the gap it actually lives in is six decades wide -- 5.168e-03 (the lowest
  carrier reading) to 1.663e-09 (the highest traced reading) -- with nothing
  measured in between over 39 solves.  1e-8 sits in that gap.  The
  cross-fixture spread of the traced population is 3.4% and the cross-build
  spread is zero, against a 0.79-decade (6.2x) gap to the screen, so this is a
  bar with room on both sides by the TESTING_STANDARDS rule-5 test;
* the source's stated justification ("a hundredfold inside the margin") is
  CORRECT IN DIRECTION and understated in size -- the true room is 5.2 decades
  on that criterion.  Its number is now measured rather than argued.

The one thing the item asked that the answer changes: the screen is not "two
orders" of unnecessary caution.  On the shipped default it is the switch that
routes the traced fits to D15's refinement, and every one of those fits is
3-to-7 orders more accurate for it.

---

## 3. ITEM 3 -- THE DEAD `n == 0` GUARD

Confirmed dead by construction, then made reachable.

```
b (0,)   -> ValueError: cannot reshape array of size 0 into shape (0,newaxis)
b (0, 1) -> ValueError: cannot reshape array of size 0 into shape (0,newaxis)
b (0, 3) -> ValueError: cannot reshape array of size 0 into shape (0,newaxis)
```

`B.reshape(B.shape[0], -1)` raises on any zero-row `b` -- `-1` is ambiguous at
size 0 -- so the `if n == 0` branch below it could not be reached from any
input, exactly as S13 recorded.  The consequence S13 did not state: the
deterministic kernel therefore RAISED where the expression it exists to replace
returns zeros (`A.T @ A` -> `(5, 5)`, `A.T @ b` -> `(5,)` / `(5, 1)` / `(5, 3)`).
This function's whole contract is "the same two arrays as the BLAS expression,
in a fixed summation order", so a shape the BLAS expression handles is part of
it.

**Made reachable**: the guard is hoisted above the reshape and returns the zero
Gram and a right-hand side whose shape matches `A.T @ b` for all three `b`
shapes.  Post-change:

```
b (0,)   -> OK G (5, 5) r (5,)      BLAS route: G (5, 5) rhs (5,)
b (0, 1) -> OK G (5, 5) r (5, 1)    BLAS route: G (5, 5) rhs (5, 1)
b (0, 3) -> OK G (5, 5) r (5, 3)    BLAS route: G (5, 5) rhs (5, 3)
solve deterministic=True  -> ValueError: cannot reshape array of size 0 ...
solve deterministic=False -> ValueError: cannot reshape array of size 0 ...
```

**Nothing on a shipped path moves.**  No fit site can produce an empty design
matrix (every one enforces a samples-per-term floor), and the
`_solve_lstsq_thread_safe` entry point still raises on an empty fit -- from
`_solve_lstsq_qr`'s own reshape, on BOTH routes, exactly as before.  The test
(`test_the_empty_fit_returns_what_the_blas_expression_returns`) asserts both
halves, so "the guard is reachable" cannot quietly become "empty fits now
return zeros to callers".

---

## 4. REFUTATIONS

Claims checked against measurement, in both directions.

1. **REFUTED -- S14.6: "the GBD re-expansion reaches neither
   `_solve_lstsq_thread_safe` nor its callers".**  It reaches
   `_solve_lstsq_thread_safe` once per call, through
   `_carrier_referenced_bundle` -> `_compute_carrier` (S1.2).  The conclusion
   survives on measured grounds (`M = 5` < `_DET_EINSUM_MIN_TERMS`;
   rcond 0.99 is eight decades clear of the screen), but the argument as
   written was one import hop short.
2. **REFUTED -- the natural reading of the C13 open item, that the WEIGHTING is
   what puts the traced fits under the screen.**  The weighted, decentred
   fixture reads rcond 1.617e-09 against the concentric unweighted 1.612e-09
   (S2.2).  It is the basis and its degree, and column scaling buys 1.28x.
3. **REFUTED -- "the screened-in traced fits reroute to a threaded QR" on the
   shipped default.**  With `DETERMINISTIC_TRACED_FIT=True` they take
   `_det_refine` and never reach the QR or C13's residual vote (S2.3).  The
   description was accurate for 5.41.0 and is not for 5.42.0.
4. **REFUTED -- the S14.6 resource-pressure hypothesis, for this fixture.**
   The one pressure-sensitive branch on the path (`_fft_reconstruct_applicable`
   swallowing `MemoryError`) is inapplicable here by ten decades of margin
   (S1.6), and no other code on the path reads free memory, `cpu_count`, or a
   thread count.  The hazard is real for other bundles and is fixed anyway.
5. **REFUTED -- "env leakage between parallel pytest processes could have
   broken this pair".**  `LUMENAIRY_MEM_BUDGET_MB` really does move the field
   bits (S1.5), but it is read once per call inside one process, so it moves
   BOTH calls identically and cannot fail `np.array_equal(E_a, E_b)`.  It is
   reported in the failure message anyway, because it changes every other
   number the test reads.
6. **PROVEN OUT -- "the P4 field is thread-count invariant."**  Not previously
   claimed anywhere.  One field hash over BLAS widths 1 / 2 / 4 / 8 / 16, 300
   pairs (S1.7).  Consistent with the census: the path's only threaded-BLAS
   candidate is the `M = 5` carrier fit, which D14 already put on a
   deterministic kernel.
7. **NOT CLAIMED -- that the S14.6 failure did not happen, or that it was a
   flake.**  740 pairs under load is a bound on its rate, not a proof of its
   absence, and one unreproduced red is still evidence that the instrument was
   not watching.  What has changed is that the instrument now watches: a
   recurrence arrives with `max|diff|`, the differing-cell count, both hashes,
   the memory-budget environment and every gate reading attached.

---

## 5. LIBRARY CHANGES

| file | change |
|---|---|
| `lumenairy/elements/_lens_traced.py` | `_det_normal_equations`: the `n == 0` guard hoisted above the reshape that pre-empted it, so the deterministic kernel matches `(A.T @ A, A.T @ b)` on an empty design matrix.  No live caller reaches it; the entry point's behaviour is unchanged (S3). |
| `lumenairy/propagators/gbd.py` | `_fft_reconstruct_applicable`: `MemoryError` re-raises instead of being swallowed into a silent change of reconstruct route (S1.6). |

## 6. TESTS

| file | change |
|---|---|
| `tests/unit/test_niche_p4_gbd_reexpand.py` | `test_frame_completeness_metric_published` instrumented and made two-sided; `_byte_id_message` helper; `test_reexpand_does_not_fire_on_diverging_positive`'s bare `array_equal` given the same message (S1.9). |
| `tests/unit/test_v5_21_gbd_windowed_adaptive.py` | `test_fft_route_decision_does_not_swallow_memory_error` (fail-before measured) and its two-sided partner `test_fft_route_decision_falls_back_on_inspection_failure`. |
| `tests/unit/test_niche_d14_deterministic_carrier_fit.py` | `test_the_empty_fit_returns_what_the_blas_expression_returns` -- the guard is reachable AND the entry point still raises, both asserted. |

## 7. PROBES

| path | what it is |
|---|---|
| `validation/probe_p4_reexpand/common.py` | the P4 fixture, tree-pinned |
| `validation/probe_p4_reexpand/census_solves.py` | which least-squares solves the P4 path makes (S1.2) |
| `validation/probe_p4_reexpand/statefree.py` | 868-binding module-state diff across the diagnosed call (S1.4) |
| `validation/probe_p4_reexpand/route.py` | reconstruct-route decision margins (S1.6) |
| `validation/probe_p4_reexpand/stress_pairs.py`, `stress_fit.py`, `stress_threads.py`, `run_wave1.sh`, `run_wave2.sh` | the stressor (S1.7) |
| `validation/probe_c13_screen/oracle.py` | the two proved least-squares oracles (S2.1) |
| `validation/probe_c13_screen/capture.py`, `adjudicate.py`, `post.py` | the 39-solve conditioning + accuracy table (S2.2, S2.4) |
| `validation/probe_c13_screen/branch.py` | which solver branch each fit takes (S2.3) |
| `validation/probe_c13_screen/basis.py` | what conditioning the basis would buy (S2.2) |
| `validation/probe_c13_screen/ladder.py` | the derived screen bar (S2.5) |

## 8. SUITES

Scope: every test file in the tree that names `_det_normal_equations`,
`_solve_lstsq*`, `_fft_reconstruct_applicable`, `_reconstruct_fft` or
`reconstruct_field_from_beamlets` (16 files, by
`grep -rln --include='*.py' ... tests/`), plus the four GBD suites the changed
reconstruct serves.

| gate | result |
|---|---|
| `test_niche_p4_gbd_reexpand` + `test_v5_21_gbd_windowed_adaptive` + `test_niche_c13_lstsq_conditioning` | **49 passed** |
| `test_niche_d14_deterministic_carrier_fit` + `test_niche_d15_deterministic_traced_fit` | **38 passed** |
| `test_audit_g04_guards_prop` + `test_audit_propagation` + `test_audit_w5_raytrace_bundles` + `test_gbd_feature_complete` + `test_niche_r3_gbd_mem_lstsq` + `test_v5_21_gbd_maslov_perf` | **180 passed** |
| `test_fix_newton_pool_memory` + `test_fix_runner_oom_2026_08_13` + `test_niche_d4_dgrating` + `test_niche_d7_decentred_fit` + `test_niche_newton_pool_both_fits` + `test_niche_p9_decenter_tilt` + `test_niche_p1_gbd_chain` + `test_lens_gbd` + `test_hammer_h7_gbd_diverging` | **218 passed, 1 skipped** |
| re-run of the three edited test files after the final bar edit (`p4_gbd_reexpand` + `v5_21_gbd_windowed_adaptive` + `d14_deterministic_carrier_fit`) | **47 passed** |
| `ruff check lumenairy/ tests/unit/` (the CI gate's own command) | **All checks passed** |
| `ruff check lumenairy/ tests/` | **All checks passed** |

**485 passed, 1 skipped** over the full 16-file scope plus the GBD suites.  The
one skip is pre-existing and environmental (`test_fix_newton_pool_memory`
declines to witness a defect this BLAS does not exhibit at widths 1/2/4).

`validation/` is `extend-exclude`d in `pyproject.toml` and outside the CI gate's
`ruff check lumenairy/ tests/unit/`, as every other probe directory in the tree
is; the probes here are not linted for the same reason.

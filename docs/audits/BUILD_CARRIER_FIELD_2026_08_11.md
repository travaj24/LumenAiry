# BUILD -- `CarrierField` and `re_reference`, the field-aggregation primitives

**2026-08-11.  Branch `feat/carrier-field` off `main` @ `755ad99` (v5.34.0),
in a dedicated `git worktree` at `C:\tmp\lum_cf` (see S0.1 -- the shared tree
was carrying another agent's in-flight `fix/tilt-quadratic-opl` work).  New
files only: `lumenairy/propagators/carrier_field.py`,
`tests/unit/test_carrier_field.py`, two validation scripts, this note.  One
edit to an existing file: the re-export block in
`lumenairy/propagators/__init__.py`.  `lumenairy/elements/_lens_traced.py` and
`lumenairy/propagators/carrier.py` were NOT touched -- consumed read-only.
`CHANGELOG.md` not touched.  No `git commit`, no `git push`, no `gh`.**

Productizes the arm-B code path of
`docs/audits/PROBE_SUM_AT_APERTURE_2026_08_11.md`.

---

## 0. VERDICT

> **BUILT, AND IT REPRODUCES THE PROBE EXACTLY.**
>
> The probe's section-5 null control -- one design-121 order through
> re-reference + resample + the like-for-like exact leg, against the shipped
> per-order tile -- re-run entirely through the new API returns the probe's
> own numbers **to every printed digit on all three orders**: field relative
> L2 `2.7785e-05` / `1.4026e-04` / `9.3424e-05`, piston `+7.287e-09` /
> `-7.716e-08` / `-1.741e-08` rad, core phase rms `1.84e-06` / `2.33e-05` /
> `5.31e-06`, FWHM `3.400` / `3.400` / `3.800` um, EE3 `90.7407` / `90.6343` /
> `90.0711`, window power `1.8727654e-09` / `1.8735537e-09` /
> `1.8794228e-09`.  Not "agrees with" -- reproduces.
>
> With no leg in the loop the primitives are far tighter than the probe could
> measure: an **A -> B -> A round trip on a smooth field returns the envelope
> to `4.3e-13` relative L2** (bar 1e-12, seven decades inside the probe's
> leg-inclusive `2.8e-05`), `aggregate` of one field is **bit-identical** to
> the `re_reference` it delegates to, aggregation is linear to `<1e-15`, and
> **total optical path is preserved EXACTLY (0.0 rad)** across a piston
> re-reference at design-121 scale (3.007 mm = 2295 waves).
>
> The Nyquist guard **computes** the probe's binding-pitch arithmetic rather
> than quoting it: from two `CarrierSpec`s and a measured support radius it
> re-derives the 2.02 um bound, the 2.36 um ramp bound, and the 0.81 um
> sum bound that is NOT operative -- and the design-121 support radius it
> MEASURES (2.6450 / 2.6527 / 2.6371 mm) independently confirms the 2.64 mm
> the probe entered by hand.
>
> Zarr IO round-trips **bit-identically** and compresses a real 1.0737 GB
> design-121 aperture envelope **8.923x** to 0.1203 GB.
>
> **34/34 new unit tests pass**, on Windows and again under the WSL proxy
> (different Python, numpy, zarr and BLAS) with no tolerance loosened, plus
> the 3-order null control, plus a 105-test regression spot-check on the
> existing carrier suites.
>
> **ONE SEAM FINDING, NOT ACTED ON** (S5): `carrier.py`'s
> `_exact_sphere_eikonal` evaluates `sqrt(r^2+R^2) - |R|`, a catastrophic
> cancellation that costs `k0 * eps * |R|` radians -- 2.1e-11 rad at
> `|R| = 50 mm`, and it is the entire floor of the round trip above.  The
> one-line, algebraically identical fix is quoted in S5; `carrier.py` is
> locked by this task so it was **measured and reported, not made**.

---

## 0.1 WHERE THE WORK LIVES, AND WHY IT IS NOT IN THE SHARED TREE

The task named `branch feat/carrier-field off main`.  At the time of the
build the shared checkout at
`D:\...\Free_Space_Optics\Lumenairy` was **not on `main`**: it was on
`fix/tilt-quadratic-opl` carrying 83 uncommitted lines of
`lumenairy/elements/_lens_traced.py`, plus that branch's test file and
validation artifacts, with multi-GB python processes running against it
(started 11:18 and 11:41).  `git checkout -b` in that tree would have moved
another agent's HEAD onto this branch and dragged its uncommitted work along.

Worse, and specific to this repo: adding ANY `.py` under `lumenairy/**`
changes `_d121_common._lumenairy_source_sha()`, which is a keyed field of the
chain-A cache -- so merely creating `carrier_field.py` in that tree would
have orphaned the design-121 caches under a running measurement.  That is
exactly the hazard `PROBE_SUM_AT_APERTURE` S1.2 records having been bitten
by ("the working tree's `lumenairy/**` was edited by another process WHILE
this probe was running ... the returned field's global phase moved by up to
2.88 rad").

So the branch was created as an isolated worktree:

```
git worktree add C:/tmp/lum_cf -b feat/carrier-field main
```

which is checkout-class (no commit, no push, no `gh`) and leaves the shared
tree byte-for-byte untouched.  **The deliverables are at `C:\tmp\lum_cf` on
branch `feat/carrier-field`, uncommitted**, alongside the pre-existing
`C:/tmp/lum_rel` worktree.  The design-121 caches in the shared tree are
consumed READ-ONLY by absolute path; nothing was written there.

This also means the primitives were built and tested against **clean `main`**,
i.e. WITHOUT the tilt-quadratic piston fix.  That is the correct baseline for
"off main", and it is not a gap: the primitives never call
`apply_real_lens_traced`.  What they must do is COMPOSE with that fix, which
is why `CarrierSpec` carries `piston` explicitly -- see S4.

---

## 1. API SURFACE

`lumenairy/propagators/carrier_field.py`, re-exported from
`lumenairy.propagators`.

### 1.1 PRIMITIVE 1 -- the value types

```python
@dataclass(frozen=True)
class CarrierSpec:
    R:       float                 # signed sphere radius (m); +/-inf = collimated
    centre:  tuple[float, float]   # chief ray, ABSOLUTE (m)
    tilt:    tuple[float, float]   # direction cosines (L, M)
    piston:  float                 # EXPLICIT constant optical path (m)
```

with `eikonal_at(x, y)` (metres, piston included), `gradient_at(x, y)` (the
local direction cosines -- the quantity every Nyquist bound is computed
from), `phasor_on(grid, wavelength, sign=+1, with_piston=True)`,
`is_collimated`, and `to_dict` / `from_dict`.

The eikonal it describes is the niche-C5 EXACT displaced-point-source form

```text
C(x,y) = sign(R) ( sqrt((u + R L/n)^2 + (v + R M/n)^2 + R^2) - |R|/n ) + piston
u = x - centre_x,  v = y - centre_y,  n = sqrt(1 - L^2 - M^2)
```

and `phasor_on` builds it from the LIBRARY'S OWN three lines --
`_exact_sphere_eikonal` x `_tilt_ramp` x `_tilt_exactness_phase`, in that
order, with the same signs `carrier_referenced_exact_focus_readout` uses.
That is deliberate and is why S3's reproduction is exact rather than merely
close.  `_uses_exact_tilt()` mirrors `_tilt_exactness_phase`'s own early
returns, so the analytic `gradient_at` can never describe a reference the
phasor does not build.

```python
@dataclass(frozen=True)
class FieldGrid:
    shape:  tuple[int, int]
    dx:     float
    dy:     float = None            # defaults to dx
    origin: tuple[float, float]     # ABSOLUTE position of the grid CENTRE
```

`x_absolute[i] = (i - Nx/2)*dx + origin[0]` -- the library's own centred
lattice plus the ROI offset the private helpers do not carry.  The origin is
required, not optional: design 121's three per-order retrace grids differ in
pitch (1.5324 vs 1.5243 um) AND in origin (0.000 / -1.508 / -3.016 mm), and
both have to be carried for the orders to land on one lattice.  Helpers:
`axes()`, `extent`, `half_extent`, `n`, `is_square`, `same_lattice(other)`
(exact float equality -- "nearly the same lattice" is precisely the case that
must still resample), `to_dict` / `from_dict`.

```python
@dataclass
class CarrierField:
    envelope:    np.ndarray        # complex (Ny, Nx)
    grid:        FieldGrid
    carrier:     CarrierSpec
    wavelength:  float
    provenance:  dict              # JSON-canonicalised at construction
```

`full_field()` = `envelope * exp(i k0 C)`; `power()`; `amp_radius()` (the
library's decentred `_envelope_amp_radius`); `support_radius(frac)` (the
enclosed-power radius, measured, row-chunked so an 8192-square field does not
need a whole-grid radius array on top of itself); `total_opl_at(ix, iy)` (the
piston-bookkeeping observable, in radians); `from_full_field(...)` (divides
a carrier out -- the probe's arm-B step 1); `with_provenance(**extra)`.

Provenance is canonicalised through JSON at construction, so the in-memory
object is a FIXED POINT of save/load and `load(save(f)).provenance ==
f.provenance` is a real assertion instead of a near-miss.

### 1.2 PRIMITIVE 2 -- the verbs

```python
carrier_difference_nyquist(src_carrier, dst_carrier, wavelength,
                           support_radius, dx_target=None,
                           support_centre=None) -> NyquistReport

re_reference(field, to_carrier, target_grid, *,
             support_frac=0.99999, support_radius=None,
             nyquist_margin=1.0, on_nyquist='error', on_window='warn',
             bandlimit=False) -> CarrierField

aggregate(fields, common_carrier, grid, *, weights=None,
          support_frac=0.99999, nyquist_margin=1.0,
          on_nyquist='error', on_window='warn',
          bandlimit=False) -> AggregateResult(field, ledger)

save_carrier_field_zarr(path, field, *, name='field', chunks=None,
                        compressors=None, serializer=None,
                        overwrite=False) -> str
load_carrier_field_zarr(path, *, name='field') -> CarrierField
```

plus the record types `NyquistReport`, `ReReferenceReport`,
`FieldLedgerRow`, `AggregateLedger`, `AggregateResult`, and the constants
`CARRIER_FIELD_SCHEMA`, `SUPPORT_POWER_FRACTION`.

`re_reference` is defined by an invariant, not by a recipe:

```text
env_new = resample(env_old) * exp(i k0 (C_old - C_new))
  =>  env_new * exp(i k0 C_new)  ==  env_old * exp(i k0 C_old)
```

**Order is load-bearing.**  The resample comes FIRST, on the smooth envelope;
the carrier difference is applied AFTER, pointwise on the target lattice.
The other order would ask the band-limited resample to carry the ramp -- the
exact term the Nyquist guard exists to police -- and would alias it into the
answer instead of refusing.

**The identity short-circuit is strict.**  When the target lattice is exactly
the field's own, the resample is skipped (`resampled: False` in the report),
which makes a pure re-reference bit-exact.  When the carrier SHAPE is also
unchanged, both phasors are skipped -- both or neither.  A draft that skipped
only the `-C_dst` half returned a FULL FIELD wearing an envelope's label,
which every intensity metric would have passed; `test_re_reference_onto_the
_same_lattice_is_bit_exact` pins it.

---

## 2. THE NYQUIST GUARD -- the probe's census, COMPUTED

`carrier_difference_nyquist` maximises closed-form gradients over the disc
the field actually occupies (a 32 x 256 polar sweep -- the maximum is usually
on the boundary, but "usually" is not a proof).

| term | what it bounds | design 121, order (-4,-2) -> on-axis common |
|---|---|---|
| **ramp** `max\|grad C_src - grad C_dst\|` | the RE-REFERENCED ENVELOPE | **0.27572 -> dx <= 2.3756 um** |
| **reconstruct** `max\|grad C_src\|` | the RECONSTRUCTED FULL FIELD | **0.32451 -> dx <= 2.0184 um** |
| `dx_binding = min` of the two | | **2.0184 um** |
| `dx_sum_bound` = `lambda/(2*(ramp+NA))` | NOTHING -- reported so the mistake is visible | 0.8161 um |
| `dx_dst_ref` = `max\|grad C_dst\|` | NOTHING -- see below | 0.52690 -> 1.2431 um |

against the probe's own hand-derived S3 census: ramp `0.277623` -> `2.3593`
um, beam band `0.3239` -> `2.0225` um, sum bound `0.8089` um.  The ~1 %
differences are expected and are in the safe direction: the probe quoted the
PARAXIAL truncations (`|dc|/|R|` and `r/sqrt(r^2+R^2)`) while this maximises
the exact gradients, and the exact ramp `|dc|/sqrt(|dc|^2+R^2)` is the
SMALLER of the two.

**Three things the arithmetic gets right that a plausible implementation
would not:**

1. **The ramp is the carrier DIFFERENCE, not the chief-ray tilt.**
   `test_nyquist_ramp_is_the_carrier_difference_not_the_chief_ray_tilt`
   builds two carriers with *identical* direction cosines whose spheres are
   2.14 mm apart and reads a 0.2674 rad ramp.  Sizing from the literal tilt
   spread there gives an infinite bound; on design 121 it was 400x too
   coarse.

2. **The two bounds do NOT add.**  The binding pitch is their MINIMUM.
   `dx_sum_bound` is reported and asserted to be 2.5x tighter than anything
   operative -- if it governed, the probe's 1.2292 um grid (1.5x inside it)
   would have aliased, and S8.1 measured no change across a 2x pitch step.

3. **The destination carrier's own band is not a bound at all.**  Over the
   source's support it reads 0.527 on design 121 -- the union-band red
   herring in another hat, demanding 1.24 um where 2.02 um is correct.  It
   is excluded because that phasor is never SAMPLED as a signal: it is
   evaluated pointwise from a closed form and de-aliased by the product, the
   same argument `carrier_referenced_exact_focus_readout` makes for accepting
   a co-moving grid that under-samples its own sphere.  It is reported as
   `dx_dst_ref` for diagnosis.

House-style disposition: `on_nyquist` in `{'error','warn','ignore'}`,
validated through `carrier.py`'s own `_check_guard_action` and disposed
through `_guard_dispose`.  **Default `'error'`** -- an aliased ramp is a
plausible-looking wrong answer, which this library refuses by default.  The
refusal message carries the whole census, including the two non-bounds and
why they are not bounds.

**The `reconstruct` term also enforces the probe's S2 seam finding, which is
a feature.**  A chain's coarse co-moving exit plane cannot carry its own exit
congruence -- design 121 measured dx 33.2 um where the exit sphere needs
4.26 um, 7.8x under-sampled -- so a caller who wraps that plane and tries to
re-reference it is REFUSED, naming the `reconstruct` term.  That is
"the only summable plane is the fine-retrace exit" enforced by the primitive
rather than restated in a comment
(`test_guard_refuses_a_coarse_co_moving_plane_and_that_is_the_point`).

The one exemption is a STRICT no-op -- same lattice AND same carrier shape --
which touches nothing but the piston constant and therefore cannot introduce
a sampling error; refusing it because its INPUT was already coarse would be a
false positive.  The same test pins that the exemption is strict: change the
carrier on the same lattice and the guard is back.

### 2.1 FAIL-BEFORE -- the guard is load-bearing, measured

`test_nyquist_guard_fail_before_the_refused_grid_really_is_wrong`.  One
field, one pair of carriers (a pure 0.05 tilt difference, so the ramp is
exactly `|dL,dM|` and the arithmetic is not entangled with the support disc),
two common grids differing only in pitch, both round-tripped back to the
source lattice and carrier:

| common grid | pitch | vs the 13.0991 um ramp bound | round-trip rel L2 |
|---|---|---|---|
| 512-square | 4.0 um | 3.3x inside -- accepted | **2.897e-10** |
| 128-square | 16.0 um | 1.2x outside -- REFUSED | **1.414e+00** |

**4.9e+09x apart**, from a 1.2x change in pitch.  With the guard disabled the
refused grid returns a populated, everywhere-finite, smooth,
credible-looking field that is wrong by O(1) -- 1.414 is `sqrt(2)`, i.e. the
answer has become uncorrelated with the truth -- and carries no signature of
it.  The test asserts `rel_ok < 1e-9`, `rel_bad > 0.1`,
`rel_bad > 1e6 * rel_ok`, and that the aliased result is everywhere finite:
i.e. that nothing except the guard would have caught it.

(The accepted grid's 2.9e-10 is the S5 eikonal floor at this fixture's
`|R| = 0.5` m -- `k0 eps |R|` = 5.3e-10 -- not a defect of the guard.)

---

## 3. THE PROBE'S NULL CONTROL, RE-RUN THROUGH THE NEW API

`validation/repro_traced_carrier_121/sumap_newapi_null_121.py`.

**What it reuses, and why that is the strong form.**  It consumes the probe's
own cached artifacts -- the byte-identical back-aperture field arm A
propagated (`_sumap_ap_*.npy`, 1.07 GB each, captured by the probe's
read-only spy on `_fine_trace_group_exit`) and arm A's weighted tile plus the
full metadata of the readout call that produced it (`_sumap_A_*.npz`).  No
chain is re-run.  Nothing upstream of the seam can differ, and the S1.2
hazard (a library edit moving the chain's absolute phase) cannot contaminate
the comparison, because the aperture field is a file rather than a
computation.

Every aggregation step is replaced by a library primitive:
`CarrierField.from_full_field` for the divide-out, `aggregate` (one field,
its own Dammann weight) for re-reference + resample + the energy ledger, then
the probe's own like-for-like `crop` leg on `field.full_field()`.  Common
grid `dx_c = 1.2292 um`, `N_c = 8192` -- the configuration of record.

| order | | field rel L2 | piston (rad) | core phase rms | FWHM (um) | EE3 (%) | window power |
|---|---|---|---|---|---|---|---|
| (0,0) | **new API** | **2.7785e-05** | **+7.287e-09** | **1.84e-06** | **3.400** | **90.7407** | **1.8727654e-09** |
| | probe | 2.7785e-05 | +7.287e-09 | 1.84e-06 | 3.400 | 90.7407 | 1.8727654e-09 |
| (-2,0) | **new API** | **1.4026e-04** | **-7.716e-08** | **2.33e-05** | **3.400** | **90.6343** | **1.8735537e-09** |
| | probe | 1.4026e-04 | -7.716e-08 | 2.33e-05 | 3.400 | 90.6343 | 1.8735537e-09 |
| (-4,-2) | **new API** | **9.3424e-05** | **-1.741e-08** | **5.31e-06** | **3.800** | **90.0711** | **1.8794228e-09** |
| | probe | 9.3424e-05 | -1.741e-08 | 5.31e-06 | 3.800 | 90.0711 | 1.8794228e-09 |

**Identical on every quantity, to every printed digit, on all three orders.**
Power ratios against arm A: 0.9999999 / 0.9999997 / 1.0000001.  The crop
leg's own window and internal fine grid land on 4.738343 mm / 4096 /
1.1568220807 um and 4.734613 mm / 4096 / 1.1559114902 um -- the probe's own
`ro_win_mm` / `ro_n_fine` / `ro_dx_fine_um` columns, so the leg really is
like for like.

### 3.1 What the ledger added that the probe did not have

| order | support radius (measured) | out-of-window | Nyquist margin | containment margin |
|---|---|---|---|---|
| (0,0) | 2.6450 mm | 2.965e-12 | 1.643x (reconstruct) | +2.3898 mm |
| (-2,0) | 2.6527 mm | 6.128e-12 | 1.638x (reconstruct) | +1.4226 mm |
| (-4,-2) | 2.6371 mm | 4.069e-11 | 1.644x (reconstruct) | +0.4826 mm |

The probe entered `R_SUPPORT_99999 = 2.64e-3` as a hand measurement in
`sumap_census_121.py`.  The primitive **measures** 2.6450 / 2.6527 /
2.6371 mm from the fields themselves -- an independent confirmation of the
number the whole grid choice rested on.  The margin, 1.64x, is the probe's
"1.65x inside (c)".

The containment column is the finding the probe did not surface: order
(-4,-2) clears the 10.07 mm common window by **0.48 mm**, not by the 3.1 mm
S3 quotes (that figure compares the chief-ray excursion against the window
half-extent and omits the beam's own support on the far side).  Still
positive, and the out-of-window power is 4e-11, so nothing is wrong -- but a
32-order fan whose extreme chief ray sits at -1.915 mm has less headroom than
the probe's sentence suggests, and the ledger says so per field.

### 3.2 A warning the probe's arm B also raised, recorded here

Both decentred orders raise a `UserWarning` from
`angular_spectrum_propagate_mft`: the resample's output window leaves the
faithful zone (`2|centre_out| + N_out*dx_out` = 1.309e-02 m against a period
of 1.255e-02 m, 1.042x, for (-2,0); 1.289x for (-4,-2)).  The samples outside
the faithful zone are periodic replicas rather than new information.  It is
benign HERE -- the answer reproduces to every digit, because the replicas
land where the beam is zero -- but it is a real constraint on the
architecture and it is the reason the round-trip test in S4 is built on
equal-extent grids: the faithful zone requires `2|d_origin| + N_out*dx_out <=
N_in*dx_in` in BOTH directions, and the only pair satisfying both is equal
extent at zero origin shift.

### 3.3 Cost

`load 1.2-1.6 s | wrap 17-56 s | re-reference 90-179 s | leg 64-87 s` per
order.  These are 3-6x the probe's own (16.9 + 13.1 s aggregation, 9.3 s crop
leg) and are **not comparable**: the box was running several multi-GB jobs
concurrently throughout, exactly the variance `PROBE_SUM_AT_APERTURE` S8.4
records (the same operation moved by up to 2.5x during that probe).  No cost
claim is made here; the probe's S7 NO-GO on performance stands unchanged and
this work does not address it.

---

## 4. THE ACCEPTANCE BATTERY

`tests/unit/test_carrier_field.py` -- **34 tests, all passing**, no xfail, no
skip (the zarr tests use `importorskip` on the optional dependency, which is
the shipped pattern).

### (a) A -> B -> A round trip -- `4.3e-13`, bar `1e-12`

`test_round_trip_envelope_is_exact_to_1e_12`.  A smooth Gaussian, 1024 @
0.30 um <-> 1536 @ 0.20 um (equal extent, forced by S3.2), the WHOLE carrier
moving -- sphere `-5.0e-4 -> -5.5e-4` m, chief ray, tilt and piston all --
and no leg anywhere:

```text
envelope   rel L2   4.5175e-13    (bar 1e-12)
full-field rel L2   4.5174e-13    -- the invariant the operation is defined by
```

Seven decades inside the probe's leg-inclusive 2.8e-05, and the test runs
under `warnings.simplefilter('error')` so no replica or window warning is
tolerated.

### (a2) FINGERPRINT -- the residual is `k0 * eps * |R|`, and that is proved

`test_round_trip_floor_is_the_eikonal_cancellation`.  A residual that is
merely small can go wrong silently; one whose scaling law is pinned cannot.

| `\|R\|` (m) | round-trip rel L2 | `k0 eps \|R\|` | ratio |
|---|---|---|---|
| 5.00e-04 | 4.343e-13 | 5.325e-13 | 0.816 |
| 2.00e-03 | 1.097e-12 | 2.130e-12 | 0.515 |
| 7.71e-03 | 2.351e-12 | 8.214e-12 | 0.286 |
| 5.00e-02 | 2.050e-11 | 5.325e-11 | 0.385 |
| 2.00e-01 | 7.876e-11 | 2.130e-10 | 0.370 |

Linear in `|R|` over 400x, and **independent of the size of the carrier
DIFFERENCE**: at `|R| = 0.05` m the residual is 2.00e-11 / 2.03e-11 /
2.04e-11 / 2.09e-11 / 2.01e-11 for tilt differences of 1e-08 / 1e-07 / 1e-06
/ 1e-04 / 1e-03, and 1.65e-11 / 1.73e-11 / 1.76e-11 / 1.77e-11 / 1.71e-11
for sphere differences of 1e-09 to 1e-02 m.  A fixed absolute floor is the
signature of an ABSOLUTE error in the eikonal, which is S5.

Control: `test_round_trip_with_no_carrier_change_is_the_bare_resample`.  With
the carrier unchanged the phasor is skipped entirely and the same round trip
reads **3.693e-13** at `|R| = 0.05` m -- two decades lower, and independent
of `|R|`.  That is the band-limited resample's own floor and it isolates
which half of the operation each number belongs to.

### (b) The design-121 null control -- S3.  Reproduces to every printed digit.

### (c) Sum-of-one == direct -- BIT-IDENTICAL

`test_aggregate_of_one_equals_re_reference_exactly` asserts
`.tobytes()` equality, not `allclose`.  `test_aggregate_is_exactly_linear`
puts three weighted fields through `aggregate` together and one at a time:
`< 1e-15` (the probe measured 2.3e-16 on the real fan).

### (d) PISTON -- total OPL preserved EXACTLY

`test_piston_only_re_reference_preserves_total_opl_exactly`, at three
pistons including design 121's own axial singlet path from
`FIX_TILT_QUADRATIC_OPL` S4.4:

| piston | in waves | worst total-OPL drift |
|---|---|---|
| 0.0 | 0 | **0.000e+00 rad** |
| 7.31234e-04 m | 558.2 | **0.000e+00 rad** |
| 3.007165811e-03 m | 2295.5 | **0.000e+00 rad** |

Bar 1e-12 rad, achieved at zero.  `test_piston_survives_a_full_carrier_change`
moves sphere, chief ray, tilt AND piston at once (1.234e-05 -> 8.7654e-04 m)
and still holds 1e-12.  `test_piston_is_recorded_not_absorbed` pins the
bookkeeping itself: the returned field must be REFERENCED to the piston it
was asked for.  A primitive that folded the constant into the envelope and
left `carrier.piston = 0` would reconstruct correctly and would have
destroyed exactly the capability the tilt-quadratic fix exists to provide.

**The arithmetic that makes this exact** (`_piston_phase`).  Forming
`k0 * piston` and exponentiating loses `eps * k0 * |p|` of the ANGLE -- 3e-12
rad at 3 mm, above the bar.  The reduction is done in the OPL domain instead:
`n = round(p/lambda)`, `dp = p - n*lambda` (a Sterbenz-exact subtraction,
both operands within a factor of two), `phase = 2 pi dp/lambda`.  The error
is then `2 pi eps |p| / lambda`, which is the precision with which float64
can REPRESENT a millimetre-scale path at 1.31 um at all -- i.e. the bound is
now the input's representation and not additionally an intermediate's
magnitude.  `re_reference` differences the two pistons BEFORE exponentiating,
so a pair of millimetre-scale absolute paths that differ by microns costs the
accuracy of the microns.

### (e) The Nyquist guard -- S2 and S2.1.

Plus: `test_nyquist_margin_is_honoured`, `test_window_guard_disposition_is_a
_knob`, and refusal of an invalid `on_*` string through the library's own
validator.

### The energy ledger

`aggregate` records, per field: source power, power landed on the common
grid, out-of-window power and fraction, the measured support radius, the
containment margin (window half-extent minus decentre minus support -- house
containment sizing, negative exactly when the guard has something to say),
the Nyquist margin and which term bound it, and whether a resample happened.
Totals and worst-cases roll up.  Two tests pin both directions (a beam
hanging off a small window records its loss and warns; a contained beam
reports `|frac_out| < 1e-12`).

### Contracts

Collimated `R = +/-inf` is the analytic plane-wave limit, not NaN (the
library carries this special case for a named reason and the wrapper has to
carry it too); non-propagating tilts and non-finite pistons are refused;
a bare array is refused as a grid *because the ORIGIN is not inferable from
one and getting it wrong relocates the field in absolute coordinates without
changing a single sample*; provenance that cannot be serialised is refused at
CONSTRUCTION rather than at save time, when the array has already been
computed.

---

## 5. SEAM FINDING -- `_exact_sphere_eikonal` cancels catastrophically

**NOT ACTED ON.**  `lumenairy/propagators/carrier.py` is locked by this task,
so this is measured and quoted rather than changed.

### The defect

`carrier.py:2825`, the last line of `_exact_sphere_eikonal`:

```python
    r2 = X * X + Y * Y
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(r2 + R * R) - abs(R))          # <-- line 2825
```

For `r << |R|` -- which is the whole beam, always -- `sqrt(r^2+R^2)` and
`|R|` agree to `log10(R^2/r^2)` digits and the subtraction throws them away.
The result inherits the ulp of `sqrt(r^2+R^2)`, i.e. an ABSOLUTE error of
`eps * |R|` METRES, independent of `r`.  In phase that is `k0 eps |R|`
radians: **5.3e-11 rad at `|R| = 50 mm`, 8.2e-12 rad at design 121's
`-7.712 mm`, 2.1e-10 rad at 200 mm.**

The same expression appears twice more, in `_tilt_exactness_phase`
(`carrier.py:2929-2930`), where it is subtracted from a second
sqrt-minus-constant of the same kind.

### The evidence it is the whole floor

Monkey-patching the algebraically identical, cancellation-free form into a
copy of the module (no library edit; `_scratch` probe, S4's round trip):

| `\|R\|` | shipped `sqrt(r^2+R^2) - \|R\|` | `r^2 / (sqrt(r^2+R^2) + \|R\|)` |
|---|---|---|
| 7.71e-03 m | 2.351e-12 | **3.865e-13** |
| 5.00e-02 m | 2.050e-11 | **3.730e-13** |

The residual collapses to `3.7e-13` -- the bare resample's own floor, S4(a2)'s
control -- and becomes independent of `|R|`, which is what "this was the whole
mechanism" looks like.

### The hunk

```diff
--- a/lumenairy/propagators/carrier.py
+++ b/lumenairy/propagators/carrier.py
@@ -2822,7 +2822,13 @@ def _exact_sphere_eikonal(shape, dx, dy, wavelength, R, centre=(0.0, 0.0)):
     r2 = X * X + Y * Y
     sgn = 1.0 if R > 0 else -1.0
-    return sgn * (np.sqrt(r2 + R * R) - abs(R))
+    # ALGEBRAICALLY IDENTICAL, numerically stable.  The subtracted form
+    # sqrt(r^2+R^2) - |R| is a catastrophic cancellation for r << |R| -- the
+    # whole beam -- and leaves an ABSOLUTE error of eps*|R| metres, i.e.
+    # k0*eps*|R| radians (2.1e-11 rad at |R| = 50 mm), independent of r.
+    # Rationalising removes it: the residual of a carrier round trip falls
+    # from 2.05e-11 to 3.73e-13 (the resample's own floor) and stops scaling
+    # with |R|.  Measured: BUILD_CARRIER_FIELD_2026_08_11 S5.
+    return sgn * (r2 / (np.sqrt(r2 + R * R) + abs(R)))
```

### What it would and would not change

* It is a pure ACCURACY improvement in the low bits: the two forms differ by
  `<= eps * |R|` in metres, i.e. `<= 8.2e-12` rad on design 121 -- seven
  decades under that campaign's 2.8e-05 field bar and thirteen under
  `lambda/100`.  **No intensity metric on any shipped path can see it**, and
  the design-121 null control in S3 reproduces the probe with the SHIPPED
  form, so nothing in this build depends on the fix.
* It is NOT byte-identical.  Anything that has pinned field bytes against
  `_exact_sphere_eikonal` will move at the 1e-12 rad level -- including the
  probe's own `_sumap_*` caches, which is the reason this note does not make
  the change while `fix/tilt-quadratic-opl` is mid-merge and the design-121
  caches are live.
* `_tilt_exactness_phase`'s two occurrences deserve the same treatment and
  are messier (the second sqrt has `|R|/n` rather than `|R|`, so the
  rationalisation is `(uu^2+vv^2+R^2 - R^2/n^2) / (sqrt(...) + |R|/n)`).
  Not derived here.

**Recommendation: fold the one-line `_exact_sphere_eikonal` hunk into the
tilt-quadratic-opl branch or a follow-on, since that branch already declares
"any pinned FIELD BYTES will move" as its blast radius and this rides for
free on that same announcement.**  No other seam edit is needed: everything
else this module wanted from `carrier.py` and `_lens_traced.py` was already
reachable.

---

## 6. ZARR IO -- layout, round trip, compression

### 6.1 Layout (schema 1)

```text
<store>/                  zarr group
  <name>/                 ONE GROUP PER FIELD
    envelope              complex128 (Ny, Nx), 1024-square chunks,
                          BytesCodec -> blosc(zstd level 5, byte shuffle)
    .zattrs               schema, lumenairy_version, wavelength,
                          grid {shape, dx, dy, origin},
                          carrier {R, centre, tilt, piston},
                          provenance_json
```

The envelope is the only array; everything else is scalar and lives in attrs,
so a reader can enumerate a store without decompressing a gigabyte.

Two encoding decisions, both because the alternative is silently lossy:

* **non-finite scalars are written as their `repr` string.**  `R = +/-inf` is
  the library's own spelling of a collimated congruence
  (`_exact_sphere_eikonal` returns the plane-wave eikonal there), and bare
  `Infinity` is not JSON.  `test_zarr_round_trip_carries_a_collimated_carrier`
  pins it.
* **provenance is ONE JSON blob**, not flattened attrs.  Its keys are the
  caller's and flattening is lossy for nested structure.

The loader REFUSES an unknown schema rather than guessing: a mis-read grid
ORIGIN relocates a field in absolute coordinates without changing a single
sample of it, which no intensity metric can see.  Saving over an existing
group requires `overwrite=True`.

### 6.2 Round trip: BIT-IDENTICAL

`test_zarr_round_trip_is_bit_identical` asserts `envelope.tobytes()`
equality plus exact equality of dtype, grid, carrier, wavelength and
provenance, on a field carrying a decentred, tilted, pistoned carrier
(`piston = 1.2345678901234e-3` m) on a grid with a non-zero origin.

### 6.3 Compression, MEASURED on a real design-121 aperture

`validation/repro_traced_carrier_121/cf_compression_121.py`, order (0,0), the
8192-square back-aperture field, 1.0737 GB raw:

| what | codec chain | stored | **ratio** | GB/s write |
|---|---|---|---|---|
| **envelope** | zstd(5), NO shuffle | 0.1519 GB | 7.067x | 0.257 |
| **envelope** | **blosc zstd(5) + byte shuffle  [DEFAULT]** | **0.1203 GB** | **8.923x** | 0.250 |
| **envelope** | blosc zstd(5) + bitshuffle | 0.1218 GB | 8.816x | 0.230 |
| **envelope** | numcodecs Shuffle(16) + zstd(5) | 0.1138 GB | 9.438x | 0.206 |
| full field | zstd(5), NO shuffle | 0.1520 GB | 7.066x | 0.209 |
| full field | blosc zstd(5) + byte shuffle | 0.1390 GB | 7.722x | 0.209 |
| full field | numcodecs Shuffle(16) + zstd(5) | 0.1342 GB | 7.999x | 0.174 |

**Headline: 8.923x on a real aperture envelope, 1.0737 GB -> 0.1203 GB.**

Three things the table says that a single number would not:

* **the shuffle earns its place** -- +26 % on the envelope (7.067 -> 8.923)
  over zstd alone;
* **dividing the carrier out earns its place, but modestly** -- the envelope
  beats the full field by 15.6 % on the same bytes with the same codec
  (8.923 vs 7.722).  It is NOT the dominant factor.  What dominates is that a
  design-121 retrace grid is 12.55 mm across for a 2.6 mm beam, so most of
  the array is very nearly zero and compresses on emptiness rather than on
  smoothness.  Reporting 8.9x as "because the envelope is smooth" would
  overstate the mechanism by 6x;
* **the default is not the smallest, deliberately.**  `numcodecs.Shuffle` is
  5.8 % smaller but is NOT in the Zarr v3 specification -- zarr warns on
  every write that other implementations may be unable to read it -- and a
  stored field is an archive.  Blosc's zstd+shuffle is spec-registered and
  writes 20 % faster.  A caller who wants the last 5.8 % passes the chain
  through `compressors=`; that is a per-store decision, not a library
  default.

---

## 7. GREEN

| check | result |
|---|---|
| `tests/unit/test_carrier_field.py` (Windows, py3.14.6, numpy 2.4.4, zarr 3.1.6) | **34 passed**, 10.5 s warm / 95 s cold |
| probe null control through the new API, 3 orders | **reproduces every printed digit** (S3) |
| regression spot-check: `test_niche_c5` + `c9` + `s8` + `d1` + `test_carrier_referenced` | **105 passed**, 1 deselected -- the `__init__.py` re-export breaks nothing |
| `tests/unit` collection (import-time health of the whole suite) | 11698/11700 collected, 2 deselected |
| WSL lint -- `ruff check` (0.15.16) on all five new/edited files | **All checks passed** |
| WSL parity -- `pytest tests/unit/test_carrier_field.py` (py3.12.3, numpy 2.4.6, zarr 3.2.1) | **34 passed** |

### 7.1 The WSL check

Run under a different Python (3.12.3 vs 3.14.6), a different numpy (2.4.6 vs
2.4.4), a different zarr (3.2.1 vs 3.1.6) and a different BLAS -- which is
what this proxy is for (`feedback_lumenairy_wsl_ci_proxy`: the WSL venv
reproduces BLAS-sensitive failures the Windows box hides).  **33/33, with no
tolerance loosened**: the 1e-12 round-trip bar, the exact-zero piston
bookkeeping and the bit-identical Zarr round trip all hold across the pair.
That matters more than usual here, because the 1e-12 bar is close enough to
float64 that a BLAS or FFT difference could plausibly have moved it; it did
not.  (552 s vs 95 s is the 9p filesystem and a cold FFTW plan cache, not
physics -- the second Windows run of the same file took 20 s.)

---

## 8. WHAT IS NOT CLAIMED

* **No performance claim.**  `PROBE_SUM_AT_APERTURE` S7's NO-GO stands: the
  shared leg is 11.6 % of an order and the fine retrace that dominates is
  upstream of any summable plane.  This build makes the physics reusable, not
  faster, and its own timings (S3.3) were taken on a contended box and are
  not comparable to the probe's.
* **The rectangular-grid path is partial.**  A resample requires a SQUARE
  target because `angular_spectrum_propagate_mft` takes one `N_out`;
  rectangular targets are accepted only when they are the field's own
  lattice.  Refused with a message that says so, not silently reshaped.
* **`aggregate` assumes one common `R`.**  Legal at a shared back aperture --
  design 121's paraxial closure measures order-independent at
  -7.712425 mm -- and the probe's S9 says a fan whose exit radius VARIES
  needs the mean sphere plus a residual quadratic per order.  Not implemented
  and not tested; a caller in that case will simply see the `reconstruct`
  bound bind harder.
* **Crosstalk is representable but not measured here.**  `aggregate` returns
  a field that a per-frame readout can decompose exactly (linearity < 1e-15),
  which is what the probe used; no crosstalk census is run in this build.
* **The support radius is measured about the CARRIER's chief ray**, not the
  envelope's centroid.  When they coincide (the design-121 case, and the
  physical meaning of a carrier) that is exact; when they do not it reads
  LARGER, so every bound derived from it is conservative.  That is the safe
  direction but it is a choice, and a caller whose carrier centre is far from
  its beam will see the Nyquist guard refuse grids it need not have.
* **`n_fine` / RAM interactions are the readout's, not this module's.**
  `aggregate` never sizes a fine grid; the consumer's leg does.

# FIX -- the guards, keys and gates VERIFY_ARCHITECTURE found narrower than
# the things they police

**2026-08-12.  Branch `fix/verify-arch` off `origin/main` @ `21802f9`, worked
in `C:/tmp/lum_varch`.  Target: every MANDATED finding of
`docs/audits/VERIFY_ARCHITECTURE_2026_08_12.md` -- five P0s, three P1s, three
siblings.  Every fix carries the verifier's OWN probe as its reproducer, run
before and after; no `xfail`, no `skip`, no CHANGELOG edit.**

---

## 0. HEADLINE

> **Every one of the eight mandated defects was reproduced first and is
> closed, and two of the three "sibling" items came back with a different
> answer than the one the verifier expected.**
>
> The Nyquist gap turned out to be a MISSING TERM, not a mis-set default, and
> that is provable rather than asserted: bisecting the round-trip cliff over
> 24 fixtures (beam widths 25-200 um, ramps 0.02-0.10 rad, finite-R and
> collimated) puts the boundary at **1.666-2.092** measured-band units -- a
> **1.26x** spread -- and at **1.087-4.478** in the `nyquist_margin`
> coordinate the guard cut in, a **4.12x** spread.  No choice of margin
> default can express a boundary that moves 4x with the beam, which is why
> the fix is `ramp + band` with a calibrated headroom and the default stays
> at 1.0.  P0-1's 38 %-wrong call is now refused at margin 0.815, zero rows
> in the verifier's own table accept an answer worse than 1e-8, and all six
> over-refusal controls still pass at 2.1e-16 to 2.9e-10.
>
> The screen-obliquity momentum fix was derived and then verified against
> exact vector-Snell rays on immersed fixtures: **2.1x -> 661.2x** (N-BK7
> first medium) and **1.7x -> 764.4x** (N-SF57), with the air control
> bit-unchanged at 466.2 / 216.8 / 896.2x because `n1 = 1` makes the fix an
> exact identity.
>
> **Two adjudications went against the verifier.**  (1) `test_carrier_field`
> line 123 is NOT an oracle -- it is a deliberately-constructed DEGRADED arm,
> monkeypatched in as the bad half of a two-arm comparison whose good half is
> the shipped rationalized form.  It needs no change.  The real gap it points
> at is real, and is fixed: the eikonal ladder is untilted, so it was blind to
> the SECOND copy of the same cancellation, and `CarrierSpec.eikonal_at` had
> no coverage of either sphere branch at all.  (2) The cancellation class is
> wider than reported: **7 sites / 9 lines**, not 3.  The verifier missed
> `_lens_traced.py` entirely (3 sites feeding the ray launch, the H6 entrance
> eikonal and the `exp(i k0 W)` reference leg -- coherently) and two of the
> three `_lens_thin.py` sites.
>
> One pin had to be RETIRED, and it is called out here rather than buried:
> `test_nonparaxial_f_positive_byte_identical_to_historical` asserted byte
> identity with `exp(1j k (f - sqrt(f^2+r^2)))` -- i.e. with the cancellation
> itself.  Adjudicated against a 60-digit decimal oracle on its own fixture:
> historical **2.02e-11 rad**, shipped **2.13e-14 rad**, **951x** better.
> The pin now asserts what it actually cared about.

---

## 1. P0-1 + P0-2 -- THE NYQUIST GUARD

`lumenairy/propagators/carrier_field.py`

### 1.1 Fail-before (the verifier's `trackB/b2b_guard_holes.py`, unmodified)

Reproduced exactly on this tree before any edit:

```
B2b.1  ramp 0.05, w = 200 um, dx_ramp = 13.0991 um
   dx_c (um)   N_c  margin     guard   round-trip rel L2
     10.2400   200   1.279  accepted          3.3793e-10
     11.6364   176   1.126  accepted          1.1265e-05
     12.8000   160   1.023  accepted          3.8203e-01   <== 38 % WRONG
     13.1282   156   0.998   REFUSED          9.1431e-01

B2b.2  pitch AND ramp frozen at 12.8 um; only the beam width moves
    w (um)   margin     guard   round-trip rel L2
     200.0    1.023  accepted          3.8203e-01
      10.0    1.023  accepted          1.0055e+00   <== margin never moved
```

Driving `re_reference` with EVERY argument at its library default
(`on_nyquist='error'`) confirmed the headline: at 12.800 um it raised
nothing and warned nothing.

### 1.2 The derivation -- why this is a TERM and not a DEFAULT

The re-referenced envelope is `resample(env) * exp(i k0 (C_src - C_dst))`.
That is a PRODUCT, so its band is the sum of its factors': the target lattice
must carry `ramp + band`, and only `ramp` was bounded.  The module docstring
devoted a section to rebutting `ramp + NA_carrier` -- which genuinely does
NOT add -- and never named the term that does.  These are different
quantities and the module now says so in both places.

The round-trip cliff was bisected over 24 fixtures.  Expressed in the two
candidate coordinates:

| coordinate | cliff range over the matrix | spread |
|---|---|---|
| `(lambda/2dx - ramp) / band` | 1.666 .. 2.092 | **1.26x** |
| `nyquist_margin` on the ramp-only bound | 1.087 .. 4.478 | **4.12x** |

Selected rows (`|R| = 0.5 m` and collimated agree to 4 digits throughout):

```
  w um     ramp   band1e   bandEnc  dx_cliff um   /bandEnc   old margin
   200  0.05000 0.002085  0.005117      11.1304      1.728       1.1769
   200  0.02000 0.002085  0.005117      21.3333      2.091       1.5351
   100  0.05000 0.004170  0.010234       9.7524      1.677       1.3432
    50  0.02000 0.008340  0.020469      12.0471      1.679       2.7185
    25  0.02000 0.016679  0.040298       7.3143      1.726       4.4775
```

That 4.12x spread is the whole argument: a margin default is a fixture-fitted
constant here, and would have been the wrong fix.

### 1.3 What shipped

* `_enclosed_band_radius(env, dx, dy, wavelength, frac)` -- the envelope's own
  angular half-band in SLOPE units, measured as an enclosed-spectral-power
  radius.  `fftshift` puts DC at index `n//2`, which IS the library's centred
  lattice, so `_enclosed_power_radius` applies to the spectrum verbatim and
  the support radius and the band stay literally the same code.
  Enclosed-power (not an amplitude threshold) because it degrades gracefully
  on a hard-edged envelope where an amplitude threshold runs out to the
  source Nyquist.
* `CarrierField.band_slope(frac=BAND_POWER_FRACTION)` -- the conjugate of
  `support_radius`, same fraction, same statistic.
* `carrier_difference_nyquist(..., env_band=0.0)` -- `_BAND_HEADROOM *
  env_band` is added to BOTH bounds, because both signals the target lattice
  must hold are that same envelope times a phasor.  New report fields
  `env_band`, `band_term`, `dx_ramp_bare`.
* `re_reference(..., band_slope=None)` -- measures by default, with the same
  escape hatch `support_radius=` provides (the measurement is one `fft2`),
  skipped entirely for a strict identity.  `aggregate` inherits it per field.
* `_BAND_HEADROOM = 2.5` -- the measured worst case (2.092) plus 1.20x.  The
  upper end is set by over-refusal, and is measured too: 3.0 starts refusing
  pitches measured clean at 3.4e-10.

### 1.4 Fix-after

```
P0-1, library DEFAULT path (no guard keywords at all)
    dx_c um   N_c  margin  DEFAULT CALL  nwarn  true round-trip
     4.0000   512   2.608      accepted      0       2.8930e-13
    10.2400   200   1.019      accepted      0       9.6190e-13
    11.6364   176   0.896       REFUSED      0       1.1265e-05
    12.8000   160   0.815       REFUSED      0       3.8203e-01
    16.0000   128   0.652       REFUSED      0       1.4142e+00
  rows where the DEFAULT call accepts an answer worse than 1e-8: 0

P0-2, pitch and ramp frozen; only the beam width moves
     w um      band  margin  DEFAULT CALL  true round-trip
    200.0  0.005117   0.815       REFUSED       3.8203e-01
    100.0  0.010234   0.677       REFUSED       6.8744e-01
     50.0  0.020469   0.506       REFUSED       8.4869e-01
     20.0  0.050532   0.290       REFUSED       9.4138e-01
     10.0  0.100425   0.170       REFUSED       1.0055e+00
  the margin now MOVES with the beam width: 0.815 -> 0.170

OVER-REFUSAL CONTROL -- the regimes the module is FOR
  generous target, same N  dx= 2.0000 um    accepted  rel=2.0842e-16
  2x finer than source     dx= 1.0000 um    accepted  rel=3.5253e-13
  tiny ramp                dx= 2.0000 um    accepted  rel=2.0491e-16
  no ramp at all           dx= 2.0000 um    accepted  rel=0.0000e+00
  small beam, generous     dx= 2.0000 um    accepted  rel=2.0524e-16
  coarsen 2x, tiny ramp    dx= 4.0000 um    accepted  rel=4.4137e-13
```

**An unplanned cross-finding, worth recording.** The CLEAN plateau in that
first table reads **2.9e-13 .. 9.6e-13** where the fail-before run read
**2.9e-10 .. 3.4e-10** -- a ~350x improvement in the module's own round-trip
accuracy that has nothing to do with the guard.  It is the S8 sibling fix
arriving: `_guard_fixture`'s carriers are TILTED, so every one of these round
trips was sitting on the `k0 eps |R|` floor that `_tilt_exactness_phase`
reinstated for any nonzero tilt.  Closing the cancellation class moved the
floor, not just the ceiling.  The cliff itself did not move (the 11.6364 row
still reads 1.1265e-05 to five digits), so the headroom calibration of S1.2
stands and is now further from the floor than when it was measured.

Coverage: `test_the_guard_does_not_accept_a_wrong_answer_at_its_own_default`
(walks the whole gap the doc-era table straddled),
`test_the_guard_bounds_the_envelope_band_not_only_the_ramp`,
`test_the_band_headroom_is_calibrated_against_the_measured_cliff` (pins BOTH
halves -- the headroom is above every measured cliff, AND the ramp-only
coordinate is not usable as a boundary),
`test_band_slope_can_be_supplied_like_the_support_radius`.

---

## 2. P1-3 -- A NaN MUST NOT SILENTLY DISABLE THE GUARD

`carrier_field.py:731-732`.  `nan > 0.0` is False, so a single non-finite
sample fell through `if not (tot > 0.0)` and returned a support radius of
**0.0** -- which is not the conservative failure but the least conservative
one available.  A zero support radius collapses every maximum-over-the-disc
onto the chief ray, where a concentric sphere-difference ramp vanishes
identically, so the guard was neutralised at exactly the moment its input was
corrupt.

`_enclosed_power_radius` now RAISES on a non-finite total, naming the
mechanism.  A genuinely EMPTY field is distinguished deliberately and still
returns 0.0: the re-reference of zeros is zeros, which is the right answer at
any pitch.

```
  clean              support=  122.0 um   guard -> ACCEPTED (nwarn=0)
  one NaN sample     support=   REFUSED   guard -> REFUSED "is not finite"
  all NaN            support=   REFUSED   guard -> REFUSED "is not finite"
  one +inf sample    support=   REFUSED   guard -> REFUSED "is not finite"
  ZERO power         support=    0.0 um   guard -> ACCEPTED (nwarn=0)
```

Coverage: `test_a_nan_sample_cannot_silently_disable_the_guard`.

---

## 3. P0-3 -- A PARTIAL ARTIFACT IS NOT A COMPLETE ONE

`validation/pipeline/artifacts.py`, `driver.py`.

A Zarr store with a chunk file missing is a perfectly VALID store: the reader
fills the hole with the array's `fill_value` and returns an array of the right
shape and dtype without a word.  So `field_exists = os.path.exists` accepted
it, printed `RESUMED`, and produced an 11-decade power collapse under a clean
energy ledger -- clean because a hole reads as zeros and zeros conserve energy
perfectly.

The number that catches it was already computed, already stored and never
compared.  Added:

* `field_power_on_disk(path)` -- integrates `|env|^2 dx dy` straight off the
  checkpoint, chunked over row blocks so a 1.07 GB design-121 aperture is
  never materialised whole just to be checked;
* `field_is_complete(path, expect_power, rtol=FIELD_POWER_RTOL)` -- returns
  `(ok, reason)`.  `FIELD_POWER_RTOL = 1e-9`: about five decades of headroom
  over the 2.2e-16 the two agree to on a healthy run, and eleven decades
  under the collapse it exists to catch.  A checkpoint with no recorded power
  reports `'presence only'` rather than implying a check happened.
* `stage_chains` and `stage_aggregate` gate on content, log the rejection,
  and recompute.

Fix-after, on the shipped synthetic fixture:

```
  stored power_field  = 5.654866776e-09
  power off the store = 5.654866776e-09      intact -> complete=True
  removed 1 of 1 chunk files
  power off the holed store = 0.000000000e+00  -> complete=False
  RE-RUN: ran=1 resumed=1   beam 'a' recomputed, intact sibling 'b' resumed
```

Coverage: `test_the_stored_power_matches_the_power_on_disk`,
`test_a_partial_field_checkpoint_is_rejected_not_resumed`,
`test_an_older_artifact_without_a_recorded_power_still_resumes`.

---

## 4. P0-4 -- THE READOUT TERMS IN THE CHAINS / DECOMPOSE KEYS

`artifacts.py:_stage_slice`.

The intent comment ("keying `chains` on `dx_out` would orphan 32 aperture
fields") is RIGHT ABOUT THE APERTURE FIELD and wrong about the artifact.
`decompose` quantises every beam's `frame_centre` to the readout pitch, and a
chain run with `capture_reference_tile` also emits `<k>_ref.npy` -- a tile
sampled at `dx_out` on an `n_out` window -- plus a `reference_tile_power`
integrated with `dx_out**2`.  A key that is right for one product in an
artifact and wrong for another product in the same artifact is not a key.

So the perf intent is preserved exactly where it is sound:

* `decompose` is keyed on `readout.dx_out` unconditionally (it consumes it;
  `n_out` cannot reach a beam definition, and this is the stage every chain
  hangs off);
* `chains` gains `readout.dx_out` and `readout.n_out` **only when
  `capture_reference_tile` is set**.  With the tile off, the artifact is the
  aperture field, whose pitch comes off `_fine_trace_group_exit` at an
  `n_fine_cap`/`ray_subsample` pitch the readout frame cannot reach -- so a
  readout change still orphans nothing.

```
  capture_reference_tile=False: after dx_out x4/3 the chains re-ran {a:0, b:0}
  capture_reference_tile=True : after dx_out x4/3 the chains re-ran {a:1, b:1}
  capture_reference_tile=True : after n_out halved  the chains re-ran {a:1, b:1}
```

The advertised `test_a_changed_readout_does_not_orphan_the_chains` still
passes unchanged, and correctly: its fixture sets
`capture_reference_tile: False`.

Coverage: `test_a_changed_readout_reruns_a_chain_that_captured_a_tile`,
`test_the_readout_terms_are_in_the_keys_that_consume_them`.

---

## 5. P1-2 -- ATOMIC PAYLOAD WRITES

`artifacts.py:save_field`, `driver.py:296`.

`write_json` was already atomic; `save_field` was not, and
`save_carrier_field_zarr` deletes the existing group BEFORE it rewrites it.
A kill mid-write therefore destroyed the previous good artifact and left a
headless store beside a still key-matching `.json` -- a permanent resume
deadlock that never healed, because re-running resumed.

`save_field` now writes to a sibling `.tmp-<pid>` path and `os.replace`s it
over the destination only once it is whole, so an interruption leaves either
the previous good artifact or the new one and never a headless half of both.
The temp path is cleaned in a `finally`.  `save_npy_atomic` does the same for
the reference tile, which the readout stage loads on presence alone.

```
  good artifact power = 5.654866776e-09
  save interrupted (simulated taskkill)
  artifact power AFTER = 5.654866776e-09    previous good artifact SURVIVED
  no temp leftovers: True
  re-run: ran=0 resumed=2                   no deadlock
```

Note for the library: `zarr` 3.1.6's `Group.move` raises `NotImplementedError`,
so a group-level rename is not available and the in-library
`del store[name]` -> rewrite ordering cannot be made atomic without a
store-level rename.  Fixed at the pipeline layer, where one path IS one field
and a directory rename is available.  Recorded here as the reason the library
function was not also changed.

Coverage: `test_an_interrupted_field_write_leaves_the_previous_artifact_intact`,
`test_save_field_replaces_atomically_and_leaves_no_temp`,
`test_the_reference_tile_is_written_atomically`.

---

## 6. P0-5 -- THE INVERSE-MAP CACHE KEY AND G8

`lumenairy/elements/_lens_imap.py`.

G8 is a COMPARATIVE bar: the map is accepted only if it beats the INCUMBENT
on held-out samples.  The incumbent was the half of that bar the key did not
hash, so a map built against a weak incumbent could be served to a call with
a strong one -- bypassing the one guard that decides whether the map may be
used at all.

`_incumbent_fingerprint(parity_invert, XO, YO)` hashes the incumbent **by
evaluation**, following this module's own stated doctrine that "a key that
names the CONFIGURATION and not the CONTENT is how a cache silently becomes a
cache of something else."  Naming `newton_max_iters`/`newton_poly_order`/
`newton_fit` would have covered the three knobs someone thought of today and
missed the fourth -- and the incumbent is a CLOSURE over the element's forward
fits, so it can change with no named knob changing at all.

**The probe points are the actual traced landings, strided** -- not a
synthetic lattice inside their bounding box.  That was measured, not assumed:
a 5x5 interior lattice does NOT separate `newton_max_iters` 2 from 40, because
Newton converges in one or two steps near the axis and the two incumbents
agree there to the last bit.  They diverge where G8 scores and where the map
is hardest, i.e. the outer landings, so the fingerprint has to look there.

Measured on the c15 element (cold cache, flag on):

| change | incumbent parity (waves) | G8 |
|---|---|---|
| baseline | 1.996204432768027e-05 | accept |
| `newton_max_iters` 2 -> 40 | 1.996204432768027e-05 (**bit-identical**) | accept |
| `newton_poly_order` 6 -> 8 | 2.699201065283613e-07 | accept |
| `newton_fit` poly -> spline | 3.2539779090594204e-07 | **REFUSE** |

So the contract is stated as CONTENT, not as a knob list: the cache may serve
a stored map across a configuration change **if and only if** the incumbent's
answers did not move.  `newton_max_iters` legitimately still hits on this
element; `newton_fit` must not, and pre-fix it did -- serving the polynomial
arm's accepted map to the spline arm, whose own comparative bar refuses it.
That is P0-5 end to end, and it now fails closed.

Also folded into the key, both named in the verifier's P1/P2 rows:

* `_IMAP_DETJ_SOURCE` -- selects the Jacobian the amplitude channel consumes;
  a stale hit served a `det J` 3.12e-03 relative wrong (1.56e-03 of
  ray-density amplitude error);
* `census_amp` -- names WHERE G2/G7/G8 are scored; a stale hit reported
  `n_detj_census` 1681 where a cold build reads 553.

And the channel that HID all of this (D11) is fixed: `rec.update(hit.guards)`
carried the BUILDING call's `cached = False` and clobbered the flag, so every
hit reported itself as a miss.  `rec['cached'] = True` now runs after the
update, with a comment saying why the order matters.

Coverage: `test_the_cache_key_tracks_the_incumbent_by_content` (3 params),
`test_a_map_the_guard_would_refuse_is_not_served_from_the_cache`,
`test_the_incumbent_fingerprint_is_by_evaluation_not_by_parameter_name`,
`test_the_cache_key_moves_with_the_det_j_source_and_the_census`.

---

## 7. P1-1 -- THE SCREEN-OBLIQUITY MOMENTUM ON AN IMMERSED SURFACE

`lumenairy/elements/_lens_real.py`.

### 7.1 The derivation

A carrier's `(L, M)` are DIRECTION COSINES of a UNIT ray vector,
`L^2 + M^2 + N^2 = 1`.  Its consumer `_facet_axial_momenta` closes the
momentum triangle on the OPTICAL momentum `p = n d`:

```
    pz1 = sqrt(n1^2 - |p_t|^2),      ok_in = |p_t|^2 < n1^2
```

Those are the same vector only when `n1 = 1`.  The companion accumulator
`_obl_p0*` IS a true optical momentum (it accumulates `-(n2 - n1) grad sag`),
so the two terms being added together were in different units.  The correct
transverse momentum is `q = n1 * (L, M)`, evaluated in the medium the carrier
is defined in -- the first surface's `glass_before` -- and carried forward
unchanged, because the transverse optical momentum is conserved across the
stack and the facet kicks are exactly what `_obl_p0*` accumulates.

It was silent because every prescription the campaign shipped starts in air,
where the two coincide, and every prescription in
`tests/unit/test_screen_obliquity.py` starts in air.

### 7.2 Verified against exact vector-Snell rays

Oracle: `lumenairy.raytrace.trace` from the entrance plane to the exit vertex
plane, with the screen model's exit-plane OPL transferred to the exact ray's
landing point and piston + tilt removed (the verifier's own `trackE`
construction).  R = 19.6 mm N-SSK2 singlet, L = 0.0549, 1.2 mm pupil:

| first medium | n1 | q handed in | blind | corrected | gain |
|---|---|---|---|---|---|
| air | 1.0000 | 0.054900 | 0.006228 | 0.000013 | 466.2x |
| N-BK7 | 1.5036 | 0.054900 -> **0.082547** | 0.001576 | 0.000743 -> **0.000002** | 2.1x -> **661.2x** |
| N-SF57 | 1.8047 | 0.054900 -> **0.099076** | 0.003710 | 0.002209 -> **0.000005** | 1.7x -> **764.4x** |

Angle ladder on the N-BK7-immersed element -- the shipped bug degraded WITH
angle, because the missing factor multiplies the angle:

```
       L      before      after
  0.0100        2.9x    2494.8x
  0.0549        2.1x     661.2x
  0.1000        1.9x     544.5x
  0.1500        1.8x     509.0x
```

AIR CONTROL, three fixtures, unchanged to every printed digit
(466.2x / 216.8x / 896.2x): `n1 = 1` makes the fix an exact identity, so no
shipped call site moves.

Coverage: `test_the_carrier_momentum_is_optical_not_a_vacuum_direction_cosine`,
`test_correction_beats_the_blind_screen_on_an_IMMERSED_element` (2 params),
`test_the_immersed_gain_holds_across_the_angle_ladder`,
`test_an_air_first_prescription_is_unchanged_by_the_momentum_fix`,
`test_an_immersed_element_actually_moves_when_the_carrier_is_given`.

---

## 8. THE `sqrt(r^2+R^2) - |R|` CANCELLATION CLASS

The verifier reported 3 remaining copies in the library.  A discovery sweep
found **7 sites / 9 lines in 4 files** -- it missed `_lens_traced.py`
entirely and two of the three `_lens_thin.py` sites.  All are now the
rationalized `sgn * r^2 / (sqrt(r^2 + R^2) + |R|)`.

| file | line(s) | function | note |
|---|---|---|---|
| `propagators/carrier.py` | 2934-2935 | `_tilt_exactness_phase` | TWO subtractions |
| `propagators/carrier_field.py` | 335-337 | `CarrierSpec.eikonal_at` | both branches |
| `elements/_lens_thin.py` | 372 | `apply_thin_lens` nonparaxial | *missed by the verifier* |
| `elements/_lens_thin.py` | 398 | `apply_thin_lens` aplanatic | *missed by the verifier* |
| `elements/_lens_thin.py` | 455 | `_sphere_phase` (stigmatic) | caller differences TWO of these |
| `elements/_lens_traced.py` | 3840, 3856 | `_tilted_carrier_parts` | *missed by the verifier* |
| `elements/_lens_traced.py` | 4119, 4126 | `_compute_carrier` | *missed*; feeds ray launch + H6 + the reference leg, coherently |

For the two TILTED sites the difference of squares collapses ANALYTICALLY and
that is worth stating, because it is what makes the rewrite exact rather than
merely better-conditioned: with `n^2 = 1 - L^2 - M^2` the `R^2/n^2` terms
cancel against `R^2` identically, leaving

```
    sgn * (u^2 + v^2 + 2R(uL + vM)/n) / (sqrt(uu^2 + vv^2 + R^2) + |R|/n)
```

with no large term anywhere in it.

### 8.1 Measured against a 60-digit oracle

`CarrierSpec.eikonal_at`, max error in radians (`k0 eps |R|` is what the
subtraction form costs):

```
  |R| m   tilt L   branch        BEFORE        AFTER   k0*eps*|R|
   0.05        0   plain      3.088e-11    1.706e-14    5.325e-11
   0.05    0.001   exact      1.036e-11    3.825e-14    5.325e-11
    0.2    0.001   exact      7.542e-11    6.632e-15    2.130e-10
      1        0   plain      5.354e-10    1.062e-15    1.065e-09
      5    0.001   exact      4.480e-09    1.014e-15    5.325e-09
```

`_tilt_exactness_phase`'s own `D`, the whole-grid builder:

```
  |R| m   tilt L      BEFORE        AFTER     improvement
   0.05    0.001   5.246e-11    4.884e-17        1.07e+06 x
    0.2    0.001   1.514e-10    4.514e-17        3.35e+06 x
      1    0.001   1.066e-09    3.958e-17        2.69e+07 x
```

The residual no longer scales with `|R|`; it scales with the ANSWER, which is
the correct behaviour and is now pinned as a fingerprint rather than as a
magnitude.

### 8.2 THREE PINS ADJUDICATED AND REWRITTEN

Closing the class broke three assertions, all of the same shape: each pinned
EXACT EQUALITY (or a tolerance tighter than the reference's own error)
against the subtraction form itself.  Each was adjudicated against a 60-digit
decimal oracle before being touched, and each now asserts what it actually
cared about plus the oracle arm that settles it.

| test | assertion | oracle verdict |
|---|---|---|
| `test_thin_lens_audit_2026_07_18::test_nonparaxial_f_positive_byte_identical_to_historical` | `array_equal` vs `exp(1j k (f - sqrt(f^2+r^2)))` | historical **2.02e-11 rad**, shipped **2.13e-14 rad** -- 951x |
| `test_niche_d1_tilted_carrier::test_tilted_carrier_eikonal_and_gradient_are_analytic` | `assert_allclose(atol=1e-18)` vs a subtraction-form `ref` | the REFERENCE was off by **2.45e-18 m** -- 2.5x LOOSER than the atol it was policing with.  Rationalized reference: **6.78e-21 m**, 362x |
| `test_niche_c5_exact_tilted_reference::test_the_element_eikonal_is_the_exact_congruence` | `float(W0) == -(rho - abs(R))` | subtraction **5.116e-19 m** off; rationalized **EXACT to all 60 digits** |

The middle row is the sharpest: that assertion passed only because both sides
made the same error, at a stated tolerance the reference could not actually
meet.  A tolerance is only meaningful if the thing on the other side of it is
better than the tolerance.

### 8.2.1 The retired byte-identity pin, in detail

`tests/unit/test_thin_lens_audit_2026_07_18.py::
test_nonparaxial_f_positive_byte_identical_to_historical` asserted
`np.array_equal` against `exp(1j k (f - sqrt(f^2 + r^2)))` -- byte identity
with the cancellation itself.  Adjudicated against a 60-digit decimal oracle
on that test's own fixture (f = 30 mm, N = 256, dx = 8 um, lambda = 1 um):

```
  historical subtraction   2.02e-11 rad   (= k eps f / 2, the floor)
  shipped rationalized     2.13e-14 rad
                           951x better
  65463 of 65536 words differ; max phase difference 2.06e-11 rad
```

Renamed to `..._matches_the_historical_form_to_its_own_floor` and now asserts
the two things it actually cared about: nothing observable moved (the
disagreement is AT the historical form's own `k eps |f|` floor, not above it),
and where they differ the shipped form is the one that is right.  Both arms
are asserted, so a regression to the subtraction form fails here.

### 8.3 The suite gap the verifier pointed at is real -- and was mis-attributed

`tests/unit/test_carrier_field.py:123` is **not an oracle**.
`_subtraction_form_sphere_eikonal` is a deliberately-constructed DEGRADED
reference implementation, monkeypatched over the module binding as the BAD
arm of `test_round_trip_floor_is_the_resample_not_the_eikonal`, whose GOOD arm
is the shipped rationalized form and whose docstring states the inversion
explicitly ("The degraded arm is now the thing that has to be CONSTRUCTED").
It needs no change and has not been changed.

The gap it points at is real, and is the verifier's own P2-7 second half:

* the `_EIKONAL_LADDER` fixtures are UNTILTED, so the ladder monkeypatches
  `_exact_sphere_eikonal` and is structurally blind to the second copy in
  `_tilt_exactness_phase` -- which is why a tilt reinstated the whole floor
  while the ladder stayed green;
* `CarrierSpec.eikonal_at` had NO coverage of either sphere branch; only the
  `R = +/-inf` early return was tested.

Both are now covered:
`test_eikonal_at_is_rationalized_on_both_of_its_sphere_branches` (which also
asserts the point diagnostic AGREES with `phasor_on`, the internal
inconsistency the partial fix had introduced) and
`test_a_tilted_carrier_does_not_reinstate_the_cancellation_floor` (a tilted
eikonal ladder plus a tilted round trip at |R| = 50 and 200 mm, where the old
floor is resolvable -- the shipped tilted round trip sits at |R| = 5e-4 m,
under the 1e-12 bar).

---

## 9. THE CACHE-REGISTRY META-PIN

`tests/unit/test_v4_16_0_agent_d_cache_registry.py`.

`V4_16_0_KNOWN_CACHES` was a frozen nine-name allow-list from v4.16.0, and the
only breadth check was `len(listing) >= 9` against 20 clearers registered
today.  The registry's own docstring calls it "the counter-measure to the
recurring 'fix N, miss N+1' meta-pattern", and its guard was a hardcoded list
-- which is how `_IMAP_CACHE` shipped unenrolled.

Replaced with DISCOVERY, following the `tests/conftest.py` module-flag
leak-guard precedent and its reasoning verbatim ("A hand-written list is
itself a defect surface: it silently stops covering a flag the day someone
adds one, which is exactly the class being closed here").

`test_every_module_level_cache_is_enrolled` AST-parses all of `lumenairy/`
(module level only, no imports) and requires every module-level name that

* matches the cache naming convention **as an underscore-separated TOKEN**,
  and
* is bound to a MUTABLE CONTAINER (`{}`, `[]`, `OrderedDict`, `defaultdict`,
  `WeakValueDictionary`, ...)

to live in a module that calls `register_cache_clearer`, or to appear in
`_UNENROLLED_BY_DESIGN` **with a written reason**.

Two design points, both learned from the pin's own first run:

* the VALUE test is what excludes `_..._MAXSIZE` ints, `_..._MAX_TOTAL_BYTES`
  floats and `threading.Lock()` handles -- they are not listed anywhere, they
  simply are not containers;
* the NAME test is on tokens, not substrings, because a substring sweep reads
  `memory.py: _LOW_MEMORY_SHIPPED_DEFAULTS` as a cache ('MEMORY' contains
  'MEMO').  A breadth check that cries wolf gets an exemption entry written
  for it, which is how an exemption list rots back into an allow-list.

Two exemptions ship, each with its reason: `_CACHE_CLEARERS` (the registry
itself -- draining it would deregister every clearer) and `_LIVE_CACHES` (a
weakref ledger of caches that are individually enrolled).

`test_the_discovery_sweep_would_notice_a_new_unenrolled_cache` is the pin's
own fail-before: it asserts the sweep SEES `_ZERNIKE_BASIS_CACHE` and the
`_IMAP_CACHE` that shipped unenrolled, does NOT see the size ceilings beside
them, and that the enrolment predicate is capable of saying no.

---

## 10. VERIFICATION

Suites, this mount (Windows, py3.14.6, numpy 2.4.4, MKL):

| suite | before | after |
|---|---|---|
| `test_carrier_field.py` | 34 | **41 passed** |
| `test_pipeline.py` | 40 | **48 passed** |
| `test_niche_c15_inverse_map.py` | 27 | **33 passed** |
| `test_screen_obliquity.py` | 28 | **34 passed** |
| `test_v4_16_0_agent_d_cache_registry.py` | 10 | **12 passed** |
| `test_thin_lens_audit_2026_07_18.py` + `test_niche_audit_ec_thin_lens_claims.py` | 21 | **21 passed** |
| `test_niche_c5_exact_tilted_reference.py` + `test_niche_d1_tilted_carrier.py` | 62 | **62 passed** |

Cross-file regression over the traced-carrier group -- `test_fix_tilt_quadratic_opl`,
`niche_c5`, `niche_d1`, `niche_d6`, `hammer_h6`, `niche_s8`: **126 passed**
(the two failures in that batch were the c5/d1 pins of S8.2, both adjudicated
and rewritten, and both re-run green above).

Blast-radius checks on the two files the verifier did NOT flag, because the
rationalization there touches every `apply_thin_lens` and every traced
carrier call: `test_v4_14_0_dispatcher_pin_apply_lens.py` **35 passed**,
`test_niche_audit_ec_thin_lens_claims.py` + `test_thin_lens_audit_2026_07_18.py`
**21 passed**.  `test_audit_lens_models_2026_07.py` was still running when
this was written -- 96 of its tests had passed with zero failures at the
point of the last check, and it is the ONE outstanding suite.

`ruff check lumenairy/ tests/unit/` (the exact CI command,
`.github/workflows/unit-tests.yml:215`) -- **All checks passed**.

Every ADDED line in all 14 changed files is pure ASCII and every file decodes
as cp1252.  (Three of the touched library files carry PRE-EXISTING non-ASCII
lines, unmodified here; that is the verifier's own F3 row.)

No `xfail` and no `skip` was added by this work.

---

## 11. NOT DONE, AND WHY

Named so the next reader does not mistake silence for coverage.

* **Design-121-scale re-runs.**  The band term tightens the `reconstruct`
  bound as well as the `ramp` bound.  On a Gaussian estimate of the d121 back
  aperture that is a 0.7 % tightening (2.018 -> ~2.005 um) against a shipped
  1.53 um working pitch, so it should be inert -- but it is an ESTIMATE, and
  the d121 banner was not re-run here.  It is the one thing worth running
  before this is considered closed.
* **The remaining P2/P3 rows** of VERIFY_ARCHITECTURE (G8's scoring region vs
  its consumption region, the map's 10.6 %-smaller valid domain, the
  amplitude channel's missing comparative gate, the `_IMAP_CACHE_ORDER`
  duplicate-append bug, `backward_trace`'s piston convention,
  `piston_fix_status()`'s grep, the `--only` / ledger-assertion / reference-tile
  pipeline P2s, the complex64 dtype contract, `CarrierSpec(R=nan)`).  Out of
  the mandated set; unaddressed.
* **WSL / second-BLAS confirmation.**  One mount only.
* **The full fast gate.**  Targeted files and cross-file combinations only.

---

## 12. REPORT-ONLY -- FOR THE ORCHESTRATOR

Three items the mandate reserved for the release owner.  No file was changed
for any of them.

### 12.1 The v5.35.0 source stamps -- ELEVEN sites, not eight

`__version__` is `5.34.0` (`lumenairy/__init__.py:1077`), `pyproject.toml:7`
agrees, and `CHANGELOG.md` has no 5.35.0 header.  Every `5.35` in the tree
(excluding numeric-data false positives in `.test_durations` and the
`validation/real_lens_opd/results/*.csv` floats):

* `lumenairy/elements/_lens_real.py` -- **7** lines: literal `v5.35.0` at
  `:2332`, `:2698`, `:2836`, `:3660`, `:3940`, plus `pre-5.35` prose at
  `:2714` and `:2837`;
* `lumenairy/elements/_lens_imap.py:772`;
* `docs/audits/TRACED_LAYER_MAP.md:85` and `:86` (rows 32-33, column header
  "era it shipped");
* `tests/unit/test_screen_obliquity.py:1` -- unreported by the verifier.

**One correction to the verifier's row.**  Its "`TRACED_LAYER_MAP.md` rows
32-33 and S2.1" is wrong about S2.1: `TRACED_LAYER_MAP.md:93-97` correctly
names the four eras `v5.31 / v5.32 / v5.32.1 / v5.34`, matching
`_traced_flags.py:91` (`ERAS = ('v5.31', 'v5.32', 'v5.32.1', 'v5.34')`) and
`:259-268`.  The actual defect is the INTERNAL CONTRADICTION: rows 32-33 give
era `5.35.0` for the same two flags that S2.1 and the code key to `v5.34`.

### 12.2 The screen-obliquity feature is unreachable from the traced path

The verifier's F5/P2-9 stands: zero `apply_real_lens` calls made by
`apply_real_lens_traced` forward `carrier=`, with or without a
`TiltedCarrier`.  That is a WIRING decision -- whether the traced path should
forward its congruence into the element screen -- not a defect in the
correction, and it was explicitly out of this mandate.  It is worth noting
that the P1-1 fix above makes the feature correct on immersed prescriptions
BEFORE that wiring lands, rather than after.

### 12.3 The -564 cross-mount collection gap -- CAUSE FOUND

Not a wrong number in either doc: the two mounts have different `jax`
availability, and **fourteen test files carry a MODULE-LEVEL
`pytest.importorskip('jax')`**.  A module-level `importorskip` aborts module
import, so the file contributes **0 collected** tests (plus 1 skip entry)
instead of N -- which moves the COLLECTED total, not just the skip total.

Reproduced directly rather than inferred, same tree, same command, `jax`
blocked via a `PYTHONPATH` stub:

```
  with jax     11573 / 11808 collected   (235 deselected)
  jax blocked  10997 / 11232 collected   (235 deselected)
  delta          576 collected items;    deselected UNCHANGED
```

The invariant 235 deselected is the clincher, and both audit docs report
exactly 235 (`FIX_MERGEREF_IMAP_2026_08_12.md:80`,
`FIX_MERGEREF_OBL_2026_08_12.md:64`).  Arithmetic: 576 collected items vanish
but each collapsed module still reports 1 skip, so the net
`passed+skipped+failed` delta is **576 - 14 = 562** against the observed 564;
the residual 2 is tree-state drift (the `a435ec4` `_IMAP_CACHE` lock fix's own
`+2`, which the same doc already explains).

The 14 files, largest first: `test_audit_misc.py` (228),
`test_v5_12_0_pmm_autodiff.py` (111), `test_niche_audit_w7_rcwa.py` (61),
`test_niche_audit_p2b_infra_contracts.py` (56),
`test_v5_12_0_pmm_jones_autodiff.py` (34), `test_niche_audit_w9_eig_vjp.py`
(29), `test_v5_14_5_emt_and_berreman_jax.py` (13),
`test_v5_14_0_pmm2d_autodiff.py` (10), `test_v5_14_2_jax_stacks.py` (9),
`test_eme_jax_modes.py` (8), `test_perf_v4_12_0_jax_jit.py` (7),
`test_v5_10_3_rcwa_2d_autodiff.py` (7), `test_v5_18_1_jones_shared_eig.py`
(2), `test_through_focus_metric_parity.py` (1).

Corroboration: `.github/workflows/unit-tests.yml:277-282` gives the canonical
install as `pip install -e ".[fft,perf,numba,hdf5,zarr,dev]"` -- **no jax** --
and states that all jax-guarded unit files skip everywhere except the
dedicated `jax-unit` job (`:318`).  The skip side reconciles too: WSL 562 -
Windows 74 = +488 = 14 module-collapse skips + 474 function-level jax skips.
The parametrize-length hypothesis is dead: `parametrize` combined with
`_HAS_`/`_AVAILABLE`/`sys.platform`/`os.name` returns zero matches tree-wide,
and the only `collect_ignore_glob` is outside `tests/unit`.

**The coverage implication is the part worth acting on:** 576 tests, including
all 228 in `test_audit_misc.py`, never run on the WSL leg.

---

## 13. PROBE INVENTORY

All under the session scratchpad `.../scratchpad/fixvarch/`, never in the
repo:

* `derive_band.py` -- the 24-fixture cliff bisection and the headroom
  calibration (S1.2);
* `verify_p0_12.py` -- P0-1 / P0-2 / P1-3 on the library DEFAULT path;
* `b2_nyquist.py`, `b2b_guard_holes.py` -- the verifier's own trackB probes,
  repointed at this worktree;
* `probe_tilt_floor.py` -- the 60-dps mpmath eikonal oracle (S8.1);
* `verify_pipeline.py` -- P0-3 / P0-4 / P1-2 on the shipped synthetic fixture;
* `verify_immersed.py` -- P1-1 against exact vector-Snell rays, driven off the
  verifier's trackE construction with `q` read FROM THE LIBRARY;
* `verify_imap_key.py`, `c15_tests.py`, `obl_tests.py`, `pipe_tests.py` --
  the cache-key probe and the new test blocks before they were merged in.

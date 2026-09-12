# WP-A6 -- CHANGELOG text (traced-carrier chain)

Findings C1-C5 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.4
(full derivations in `.../CARRIER.md`).  Files: `lumenairy/propagators/carrier.py`,
`lumenairy/propagators/carrier_field.py`.

---

### Fixed -- carrier readout: the focal peak no longer collapses when the reference carrier is a few percent off the beam (C1)

`carrier_referenced_focus_readout` sized its co-moving stop grid from the
CARRIER and never from the field.  Both the standoff resolver
(`_default_focus_standoff`) and the near-focus gate (`_near_focus_needs_bridge`)
estimate the beam at the stop plane as `w(s) = w0 sqrt(1+(s/zR)^2)` about a waist
at `-R`, which is a statement about the carrier's geometric contraction; when the
envelope carries residual curvature -- i.e. whenever the carrier is not the
beam's own wavefront -- the beam does not contract with the co-moving grid and
nothing between the carrier leg and the Bluestein zoom measured what landed
there.  `propagate_traced_carrier_chain` supplies that carrier from a paraxial
ABCD (`_paraxial_group_r_out`), so the mismatch is structural rather than user
error, and the returned spot looked entirely plausible: core shape intact, power
book-keeping unremarkable, no warning from any guard.

Two independent changes:

* **the resolver now measures the beam.**  The envelope's own residual
  inverse-curvature is fitted (`_fit_carrier_inv`, `estimator='increment'`,
  centred on the amplitude centroid) and composed with the carrier,
  `1/R_eff = 1/R + 1/R_env`; the containment condition is then written against
  the beam's Gaussian ABCD width at the stop plane and solved in closed form
  (`_beam_containment_standoff`, a quadratic in the stop-plane position).  The
  resolved standoff is the LONGER of that and the shipped law, so the leg can
  only ever be lengthened.
* **the result is measured.**  A new `on_focus_containment` guard
  (`'error'` / `'warn'` / `'ignore'`, default `'error'`) reads the beam that
  actually landed on the stop grid and refuses when it does not fit -- below
  `_FOCUS_READOUT_CONTAINMENT_MIN` = 1.0 beam radii of co-moving half-width.  It
  takes two readings and disposes on the worse: the MEASURED amplitude radius,
  and the beam's modelled Gaussian ABCD width there (the measured second moment
  saturates once the beam overfills the grid -- at the worst row of the audit
  fixture it reads 46 um on a 40 um half-grid while the beam's true un-clipped
  radius is 109 um).

Measured on the audit's own fixture (a converging Gaussian, `w = 1 mm`, true
radius `R0 = -20 mm`, NA 0.05, `lambda = 1.31 um`, N = 1024, half-extent 4 mm,
`dx_out = w0/8`, `N_out = 64`; ONLY the reference carrier is varied, the physical
field is identical; `docs/audits/.../repro/CARRIER/p6c_mismatch.py`):

| R/R0 | standoff before | after | containment before | after | peak vs truth before | after |
|---|---|---|---|---|---|---|
| 1.00 |  222.4 um |  222.4 um | 3.21 | 3.21 | 1.000000 | 1.000000 |
| 0.99 |  418.0 um | 1008.6 um | 1.96 | 3.20 | 0.986188 | 0.999514 |
| 0.98 |  613.6 um | 1874.0 um | 1.39 | 3.20 | 0.745432 | 0.998602 |
| 0.95 | 1200.7 um | 4173.3 um | 0.91 | 3.20 | 0.187913 | 0.994304 |
| 0.90 | 2180.1 um | 7144.9 um | 0.87 | 3.20 | 0.026309 | 0.985236 |

-- a 37x recovery on the worst row, with zero warnings on any row.  The
containment now equals `_FOCUS_STANDOFF_MARGIN` = 3.2 exactly (the resolver
solves for equality).  Fed the PRE-FIX leg, the new guard refuses the two rows
whose peak had collapsed (containment 0.909 and 0.863).

**Migration.**  The default standoff is unchanged for every field whose envelope
is flat -- the beam term is exactly `0.0` there, verified over the resolver's own
6 NA x 10 extent calibration matrix (60 cells, 0 with a non-zero beam term, and
the resolved leg bit-identical).  On a field whose envelope carries residual
curvature the leg is LONGER and the answer more accurate; if a caller needs the
old number, pass `standoff=` explicitly.  The new guard refuses configurations
that previously returned a silently-wrong spot: `on_focus_containment='warn'`
returns the field and says so, `'ignore'` restores the old silence.  Three
existing test arms that deliberately exercise a broken readout now pass
`on_focus_containment='ignore'` (see "Changed -- tests" below).

### Fixed -- carrier: `carrier_referenced_fit_radius` fits about the beam, not the grid origin (C2)

`_fit_carrier_inv` built its coordinates about the grid origin with no `centre`
argument, although every sibling helper in the module has one
(`_envelope_amp_radius`, `_envelope_amp_centroid`, `_radial_carrier_phase`,
`_exact_sphere_eikonal`, `_tilt_exactness_phase`, `_sphere_parab_conversion`).
Because the estimator is a moment, a beam decentred by `x0` carrying a perfect
parabola about its OWN centre read `R*(1 + 2 x0^2/w^2)`, and a FLAT wavefront
carrying only a uniform tilt read a finite radius.  Measured
(`repro/CARRIER/p4_fit.py`; `lambda = 1.31 um`, N = 512, dx = 2 um, w = 100 um,
R = 50 mm):

```
decentred parabola            before -> after   (truth 1.0000)
  x0 = 0.5 w   R_fit/R        1.5000 -> 1.0000
  x0 = 1.0 w   R_fit/R        3.0000 -> 1.0000
  x0 = 2.0 w   R_fit/R        9.0000 -> 1.0000
decentred pure TILT           before -> after   (truth inf)
  L = 0.002, x0 =  50 um      0.0750 m -> 4.8e+15 m
  L = 0.020, x0 =  50 um      0.0075 m -> 2.5e+14 m
  L = 0.020, x0 = 200 um      0.0113 m -> 6.8e+13 m
```

`carrier_referenced_fit_radius` gains `centre='auto' | 'origin' | (x0, y0)`,
defaulting to `'auto'` = the intensity centroid; a non-default centre also
projects the residual tilt out of the phase slope before the curvature moment is
taken, so a decentred tilt cannot masquerade as `R` even when the centre is not
exactly the centroid.  `_envelope_amp_centroid` sub-pixel-snaps to exactly
`(0, 0)`, so every effectively-centred field takes the historical origin
arithmetic BIT for bit and no on-axis answer moves; `centre='origin'` pins the
historical behaviour unconditionally.

`carrier_referenced_aperture(refit_carrier=True)` deliberately keeps its internal
fit about the grid origin, because `_rereference` builds its screen on the
centred grid and the two are a matched pair; that is now stated in its docstring
with the decentred alternative (fit with `centre='auto'`, pass the result as
`new_carrier=`).

### Fixed -- carrier: a tilted complex64 chain stays complex64 (C3)

Three chain sites applied the tilt obliquity piston as
`env * np.exp(1j*k0*own*(ob-1))`.  `np.exp(1j*x)` is a numpy complex128 SCALAR,
which is STRONG under NEP 50 and promoted the complex64 envelope; nothing cast
back until the group exit, so `E_full` and every full-grid phase screen in
between ran at complex128 -- defeating the v5.44 complex64 memory campaign for
exactly the tilted per-order DOE-fan configuration it targeted.  Measured
(`repro/CARRIER/p8_c64chain.py`, one-singlet chain, complex64 input):

```
carrier = inf                             -> complex64  (unchanged)
carrier = TiltedCarrier(inf, L=0.02, ...) -> complex128  =>  complex64
```

Fixed by wrapping the scalar in `complex(...)` at all four obliquity sites and in
`_carrier_step_fast`'s NumPy amplitude/piston scale (which additionally removes
one full-grid complex128 temporary per leg for a complex64 field).  Bit-identical
on a complex128 chain; on a complex64 one the product now rounds once instead of
twice.

`carrier_field.py`: `CarrierSpec.phasor_on` gains `dtype=`, routing the sphere
through `_phasor_rows` and forwarding the dtype to `_tilt_ramp` /
`_tilt_exactness_phase`; `full_field()`, `from_full_field()` and `re_reference`
pass the envelope's own dtype, and `aggregate`'s accumulator is sized from
`np.result_type(np.complex64, *[f.envelope.dtype ...])` the way the multi
orchestrator already does.  `CarrierField.full_field()` on a complex64 envelope
returns complex64 (was complex128); the complex64 phasor agrees with the narrowed
complex128 one to one float32 rounding (measured <= 1.2e-07, flat in the
argument).  `phasor_on`'s own default is unchanged (complex128).

**Migration -- `aggregate` now sums at the members' own dtype.**  An
all-complex64 fan is accumulated in complex64 where it used to be widened to
complex128 unconditionally, so `aggregate(...).field.envelope.dtype` follows the
inputs and the coherent sum carries float32 accumulation error: measured
**7.6e-08 relL2** against the complex128 sum of the same 16 fields, inside the
`sqrt(K)*eps32` = 4.8e-07 random-walk bound.  That is the point of the change
(4.29 GB saved per grid at N = 16384, and it is the rule
`propagate_traced_carrier_chain_multi` already used), but a caller who needs the
old precision from complex64 storage must now ask for it -- pass complex128
envelopes, or widen with `dataclasses.replace(f, envelope=f.envelope.astype(
np.complex128))` before aggregating.  Any complex128 member still widens the
whole sum, as `np.result_type` requires.

### Performance -- carrier: separable reference phases, an allocation-free transfer function, and an in-place fine-grid rescale (C4)

All bit-exact or bounded by a measured, derived tolerance.  No test asserts a
timing.

* **`_radial_carrier_phase`, `_tilt_ramp` and `_rereference` are built as outer
  products.**  `exp(i a (x^2+y^2))` factorises exactly as
  `exp(i a x^2) (x) exp(i a y^2)`, so the screens need `2N` exponentials instead
  of `N^2`.  Measured `_radial_carrier_phase` **6.9x** faster at N = 2048
  (197.0 -> 28.6 ms) and **11.5x** at N = 4096 (805.5 -> 70.1 ms), at
  **3.50 -> 1.01** complex128 full grids of `tracemalloc` peak; `_tilt_ramp`
  **4.9x** at N = 2048; `_rereference` 3.50 -> 2.01 grids.  End to end,
  `carrier_referenced_reconstruct` is 4.7x faster at N = 1024 (63.6 -> 13.3 ms)
  and 4.4x at N = 2048 (248.6 -> 55.8 ms) at 3.50 -> 2.00 grids.
  The regrouping is not bit-identical: measured max difference 1.1e-13 (N = 2048)
  and 5.7e-13 (N = 4096), against the ~1e-11 rad float64 representation noise of
  the `k r^2/2R` ~ 1e5-1e6 rad arguments these screens carry, i.e. two decades
  inside the existing noise.  `_SEPARABLE_CARRIER_PHASE = False` restores the
  whole-grid `meshgrid` build bit for bit (the fail-before switch).
* **`_exact_envelope_tf_step` builds `H` with `cos`/`sin` into `H.real`/`H.imag`
  and short-circuits the untilted path in place.**  Profiled, this build was 60 %
  of a carrier leg at N = 2048 (tottime 0.711 s of 1.181 s cumtime, against
  0.223 / 0.208 s for the two FFTs).  Measured **4.00 -> 2.00** complex128 grids
  of peak and 1.54x on the isolated build -- at **exactly 0.0** difference, at
  every shape tested (N = 63/64/65/128/256, tilted and untilted): `np.exp` of a
  pure-imaginary argument IS `cos + i sin` through the same libm, and the
  untilted re-association uses addition and multiplication only, both commutative
  to the bit in IEEE-754.  A carrier leg now peaks at 2.00 grids instead of 4.00.
  The same `cos`/`sin` build goes into `_tf_phase_to_H` (the CuPy / JAX kernels'
  NumPy arm) and `_asm_axis`.
* **`_fourier_upsample_crop` scales in place.**  `out` is the fresh
  `np.fft.fftshift` result its own BUFFER OWNERSHIP note establishes aliases
  nothing, so `out *= scale` saves one full FINE-grid temporary -- 4.29 GB at the
  shipped `n_fine_cap = 16384`, on the function the memory audit already had
  running twice per exact final leg.  Values unchanged (elementwise product;
  measured relL2 5.3e-16 against a raw-numpy oracle).
* **`_envelope_amp_radius` and `_fit_carrier_inv` broadcast instead of
  `meshgrid`.**  Bit-identical (verified over both estimators, all three axes and
  both centre branches); `_envelope_amp_radius` 1.6x faster (98.2 -> 60.8 ms at
  N = 2048).

### Fixed -- carrier: the P3 cluster (C5)

* **`_freq_sq_1d` deleted.**  It had NO call sites and still carried the `- N/2`
  offset that defect D7 fixed everywhere else, while `_freq_sq_1d_bld`'s
  docstring asserted the two agreed.  Measured at N = 5 they did not:
  `[9.87 3.55 0.39 0.39 3.55]` against `[6.32 1.58 0. 1.58 6.32]`.  The false
  claim is gone with it; the surviving builders are `fftfreq` to the ulp at both
  parities.
* **`_asm_axis`'s band-limit mask is back in register at odd N.**  `H` is built on
  the `N//2` frequency axis and the Matsushima mask used `N/2`, so at ODD `N` the
  mask sat half a bin out of register with the transfer function it masks
  (measured 3.85e3 1/m at N = 65, dx = 2 um): one bin too many kept on one side,
  one too many dropped on the other.  Now relL2 0.000e+00 against a hand-built
  1-D angular spectrum at N = 64 / 65 / 127.  Even `N` is unaffected
  (`N/2 == N//2` exactly), so the whole validated surface is bit-identical.
* **The chain's `gap_kernel` reaches the paraxial focus readout.**  It had
  stopped at the gap legs and the bare final leg, so
  `carrier_referenced_focus_readout`'s internal carrier leg always ran its own
  `'auto'` -> `'exact'` default and a chain asked for `'fresnel'` -- whose whole
  documented purpose is to be pinned FP-identical to prior releases -- got a
  MIXED chain.  Both defaults are `'auto'`, so no shipped default path moves.
  The chain's `gap_kernel` is now documented (it had no Parameters entry at all),
  including the two legs it deliberately does NOT reach: the exact final readout
  and the high-NA fine retrace run no carrier transport at all, so there is no
  kernel there to select.
* **`carrier_referenced_focus_readout` gains `tilt=`**, forwarded to its internal
  carrier leg, and `propagate_traced_carrier_chain` passes the congruence's tilt
  on the tilted paraxial landing.  That leg had been transported by an UNTILTED
  kernel while the chain's own `gap_kernel` prose claimed the exact kernel
  "carries the tilt to all orders"; the envelope's diffraction there is
  anisotropic (`z/N^3` along the tilt, `z/N` across it, +0.32 % of effective
  distance at 46 mrad).  The chief-ray advance and the obliquity piston stay with
  the caller, exactly as on a gap leg.
* **`_sphere_parab_conversion` takes `dy`** (it built its y axis on `dx`, so a
  non-square-pixel chain would have converted the y axis against the wrong
  pitch); the band-limit radius `r_safe` now takes the coarser of the two
  pitches.  Every shipped call site is square, so `dy=None`/`dy=dx` is
  bit-identical.
* **`CarrierField` announces attribute assignment as deprecated** and becomes
  hard-`frozen` at v5.48, joining its already-frozen `CarrierSpec` and
  `FieldGrid` members.  Assigning to a built field bypasses every
  `__post_init__` invariant -- envelope shape vs the grid, complexity,
  wavelength, provenance canonicalisation -- and could leave a field whose grid
  no longer described its array.  The assignment still takes effect (this is the
  announcement half of the cycle), so no caller breaks; migrate to
  `with_provenance` / `re_reference` / `dataclasses.replace` / a fresh
  `CarrierField`, and for in-place accumulation to
  `np.add(acc.envelope, other, out=acc.envelope)`, which is the same arithmetic
  bit for bit and needs no rebind.
* **Self-contradicting comments corrected** in `_carrier_step_fast`
  (`'auto'` "resolves by BACKEND" four lines above `'auto'` "resolves to 'exact'
  everywhere") and `propagate_carrier_referenced` (a fast path described as
  byte-identical "on the default `gap_kernel='fresnel'`" when the default is
  `'auto'`), and the in-code claim that the exact kernel is "the physically
  correct transfer function" for a SCALED-frame leg is now stated as what the
  derivation supports (the exact kernel of the paraxially-reduced envelope leg;
  the two kernels are measured to agree to ~1e-4 relative on a real carrier leg).

### Added -- carrier: readout diagnostics published per stage

`carrier_referenced_focus_readout`'s private `_period_out` dict now also
receives `containment` / `containment_model` (beam radii of co-moving half-width
at the stop plane, measured and modelled), `standoff`, and
`window_energy_frac` -- the Bluestein window's power as a fraction of the stop
plane's.  `propagate_traced_carrier_chain` publishes them on the target stage as
`readout_containment` / `readout_containment_model` / `readout_window_energy`,
so the margins are readable without catching a warning, exactly as the `gap_*`
diagnostics are.

A window is a SUB-window of one Bluestein period, so its power cannot exceed the
stop plane's; above `1 + _FOCUS_READOUT_WINDOW_ENERGY_TOL` the transform has
folded periodic REPLICAS in and the surplus is energy it created rather than
measured.  That tripwire warns (silenced by `on_focus_containment='ignore'`) and
is only reachable with `on_replica` downgraded; measured worst 0.999863 across
the resolver's 60-cell calibration matrix, against 169.0 on a deliberately
replica-laden window in `test_niche_tight_focus_readout`.

### Changed -- tests

* `tests/unit/test_audit2609_a6_carrier.py` (new, 79 tests) pins C1-C5.
* `test_niche_c5_exact_tilted_reference.py::test_the_sphere_parabola_conversion_is_untouched`
  asserted `np.array_equal(_radial_carrier_phase(...), np.exp(...))`, i.e. the
  whole-grid build bit for bit.  It now asserts the derived 1e-11 rad agreement
  bound AND the byte-identity behind `_SEPARABLE_CARRIER_PHASE = False`, so the
  pin survives as the fail-before switch rather than as a blocker.
* Five fixtures now waive `on_focus_containment` explicitly:
  `test_niche_r9_highna_final_leg.py` (the paraxial arm the test asserts
  `ee_par < 0.10` on), `test_niche_c1_consolidation.py` (the "no tilt ramp"
  arm it asserts collapses to 3e-6 of the shipped peak, plus the focus-readout
  whitelist fixture), `test_niche_d3_guards.py` (the multi-congruence FAN
  fixture, whose second moment is taken across the beam SEPARATION) and
  `test_niche_d5_dx_flatness_gate.py` (the `paraxial_leg` TEETH arm, a
  deliberately reverted configuration at FWHM 8.58 um / EE2 8.4 %).  In every
  case the arm is a DELIBERATE fail-before that the new guard correctly
  refuses, so the waiver is the same kind, and for the same reason, as the
  `on_replica='ignore'` already sitting next to it; the guard's own default
  refusal is pinned in the new test file instead.  The r9 test additionally
  now asserts that the un-waived call IS refused.
* `test_carrier_field.py::test_aggregate_refuses_mixed_wavelengths` constructed
  its second wavelength by assigning onto a built field; it now builds one, and a
  new test pins the deprecation.

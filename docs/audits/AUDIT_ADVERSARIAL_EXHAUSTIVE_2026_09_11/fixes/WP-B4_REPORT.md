# WP-B4 -- the Collins / ABCD-Fresnel carrier transport with a freely chosen (Bluestein) output pitch

Wave 4, branch `audit-fixes-2026-09`, base 81d5b586 (= release 5.46.0).
Implements §15.9 of the main audit report (second bullet) as designed in
`fixes/WP-A6_REPORT.md` §6.1, deferred there as "Not attempted" for want of a
gate that could score it.  Derivation: `.../CARRIER.md` "Alternative
algorithms" items 1-2.

File owned and changed: `lumenairy/propagators/carrier.py` (+940 / -14).
`carrier_field.py` was not touched.  New test file
`tests/unit/test_audit2609_b4_collins_transport.py` (84 tests).
`docs/history/carrier.md` re-recorded in this change
(`ast f557e0c8... -> ed52d13a...`, `token fb1d7432... -> 7e81bb4b...`).

Everything below was measured on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time, on 2026-09-13.

---

## 1. What shipped

`transport={'sziklas', 'collins'}`, default `'sziklas'`, on
`propagate_traced_carrier_chain` (`carrier.py:8809`),
`propagate_traced_carrier_chain_multi` (`:11497`) and
`propagate_carrier_referenced` (`:981`), plus
`on_collins_sampling={'error','warn','ignore'}` (default `'warn'`) and, on the
single-step entry only, `dx_out` / `carrier_out`.

### 1.1 The transport

Collins (1970) writes the field through any ABCD system as one Fresnel-class
integral.  **Sign convention, stated explicitly as the WP asks.**  The form
printed in Collins' paper and quoted in WP-A6 §6.1,
`E_out = (i/(lambda B)) integral E_in exp(-i k (A u^2 - 2 u x + D x^2)/(2B)) du`,
belongs to the `exp(+i omega t)` / `exp(-i k z)` convention.  CONVENTIONS sec. 7
mandates `exp(-i omega t)` / `exp(+i k z)`, so what is implemented is its
complex conjugate:

    u_out(x) = exp(i k B)/(i lambda B)
               * integral u_in(u) exp(i k (A u^2 - 2 u x + D x^2)/(2 B)) du

(2-D: one `1/(i lambda B)`, the exponent separable in x and y).  Applied to an
ENVELOPE the system is "attach the input carrier, fly `z`, remove the chosen
output carrier" (`_collins_envelope_abcd`, `carrier.py:1541`):

    A = 1 + z/R_in = m,   B = z,   C = 1/R_in - A/R_ref,   D = 1 - z/R_ref

**The sign of `B` on a converging leg.**  `B` is the transfer distance `z` and
is POSITIVE on every forward leg, converging or not; the carrier's sign lives in
`A` and `D`.  A converging leg (`R_in < 0`) drives `A = 1 + z/R_in` down through
0 and into negative values as the leg crosses the geometric focus -- the frame
inverting through the waist -- and the transform carries `A <= 0` natively.
That is the whole of "m -> 0 stops being a singularity".  `z < 0`
(back-propagation) makes `B` negative and is equally well defined.

`det = AD - BC = 1` for every choice of `R_ref`; pinned as an identity over
20000 log-uniform cells spanning three decades in both radii and the leg, worst
`|det - 1| = 1.1e-13` against a cancellation floor of `64 eps x 1e3`.

Three stages (`_collins_transport`, `carrier.py:1901`):

1. a separable pre-chirp `exp(i k A u^2/(2B))` -- the module's own carrier
   screen at `R = B/A`, built per axis (`_collins_axis_chirp`, `:1603`) because
   the two axes carry different `A` and `D` under an astigmatic carrier and the
   output screen lives on a different lattice;
2. a separable centred Bluestein onto the chosen lattice
   (`_bluestein_centred_2d` with `alpha = dx*dx_out/(lambda B)`, `sign = -1`,
   `separable=_EXACT_READOUT_SEPARABLE_BLUESTEIN`) -- the transform the readouts
   already run;
3. a separable post-chirp `exp(i k D x^2/(2B))` and the prefactor.

Substituting `R_ref = R + z` gives `C = 0`, `D = 1/m`, and
`A u^2 - 2ux + D x^2 = A (u - x/m)^2`, which turns the integral into a plain
Fresnel propagation over the reduced distance `z_eff = z/m` read at `x/m`, times
`1/m` and the piston `exp(i k z^2/R_out)` -- term for term `_carrier_step_fast`.
`R_ref = inf` gives `D = 1` and returns the FIELD, which is what the readout
wants.

### 1.2 Where it is wired

* every inter-group gap leg (`carrier.py:9943`) and the bare final leg
  (`:10523`) go through `_collins_carrier_leg` (`:2055`), which returns the same
  `CarrierReferencedField(env, R, dx)` triple so the chain body is unchanged;
* the paraxial focus readout, untilted (`:10445`) and tilted (`:10481`), goes
  through `_collins_focus_readout` (`:2165`): ONE step from the chain-exit plane
  onto the requested `(dx_out, N_out)`, referenced to `R_ref = inf`;
* `_multi` forwards `transport` and `on_collins_sampling` in
  `_common_chain_kwargs` (`:12114`), so K congruences always share one
  transport;
* `final_leg='exact'` is untouched on either setting.  Its fine retrace and
  exact-sphere Bluestein readout run no carrier transport at all (WP-A24 §1
  established the same thing for `gap_kernel`), so there is nothing there for
  `transport` to select, and `test_niche_d6_exact_tilted_leg.py` is unaffected.

### 1.3 The output lattice on a chain leg

The one caller whose output grid is not supplied by the user.  Two decisions,
both from the leg's ABCD and the envelope's MEASURED phase-space box, with no
tuning constant (`_collins_leg_output_axis`, `carrier.py:1782`):

* pitch `d_out = max(|A| d, 2 (|A| r + |B| theta) / N)` -- the co-moving pitch,
  so a leg away from focus lands on the grid the Sziklas transport would have
  produced and the rest of the chain sees no change, floored by the pitch at
  which the output grid still holds the ABCD image of the input box.  The floor
  carries `|B| theta`, the leg's own diffraction, so it cannot follow `A` to
  zero;
* the geometric output carrier `R + z` is kept while the box it implies fits in
  `N` samples (`4 r_out theta_out / lambda <= N`, with `theta_out = theta/|A|`
  since `C = 0` for that reference), and a FLAT reference is used otherwise.
  Near the geometric focus the RAY carrier collapses while the true wavefront
  flattens, so referencing to that collapsing sphere is what would need the
  extra samples.

Measured on a `w = 0.3 mm` Gaussian with `R = -40 mm`, `lambda = 1.31 um`,
N = 1024, `dx = 4 um`:

| z | R_out | resolved pitch | co-moving pitch | flat ref | K1 | K2 |
|---|---|---|---|---|---|---|
| 39.0 mm | -1.000e-03 m | 0.30372 um | 0.100 um | no  | 0.024 | 0.065 |
| 39.9 mm | -1.000e-04 m | 0.27774 um | 0.010 um | no  | 0.022 | 0.597 |
| 40.0 mm | (0)          | 0.27485 um | 0.000 um | YES | 0.021 | 0.009 |
| 41.0 mm | +1.000e-03 m | 0.31746 um | 0.100 um | no  | 0.024 | 0.068 |

-- 28x the co-moving pitch 0.1 mm before the focus, a flat reference exactly on
it, and an ordinary leg 1 mm past it with no split anywhere.

---

## 2. The acceptance gate, in the order the WP asks for it

### (a) An analytic Gaussian-ABCD oracle, NA 0.03-0.45 x grid extents 1.5-10 w

Oracle written in the test file from the `q`-parameter definition, carrying the
absolute piston and the Gouy phase.  Fixture deliberately unlike WP-A6's and
VERIFY-A6's: `lambda = 1.064 um`, N = 512, `w_in = 0.8 mm`, matched carrier
`R0 = -w_in/NA`, target at the geometric focus, `dx_out = w0/8`, `N_out = 64`.
Metric: piston-free relL2 against the oracle, so that the two transports are
compared on the same footing (the shipped readout's absolute phase is not what
this cell is about).  The ABSOLUTE comparison -- piston and Gouy included -- is
made separately and reads 2.17e-14 for `'collins'`:
`TestSameTheorem::test_the_focus_readout_matches_the_analytic_gaussian_absolutely`,
and §3.2's direct-sum table.

```
    NA    ext     dx/um    w0/um |  szik peak  szik relL2 |  coll peak  coll relL2 |  better
  0.03    1.5    4.6875  11.2894 |   0.911331  8.6684e-02 |   0.933357  6.3802e-02 | collins
  0.03    2.5    7.8125  11.2894 |   0.998375  2.3022e-03 |   0.999186  8.4481e-04 | collins
  0.03    4.0   12.5000  11.2894 |   0.999978  3.3625e-04 |   1.000000  3.5104e-08 | collins
  0.03    6.0   18.7500  11.2894 |   0.999993  1.5976e-04 |   1.000000  6.7012e-15 | collins
  0.03   10.0   31.2500  11.2894 |   1.000000  3.0699e-04 |   1.000000  1.2671e-15 | collins
  0.06    1.5    4.6875   5.6447 |   0.931379  7.5226e-02 |   0.933357  6.3802e-02 | collins
  0.06    2.5    7.8125   5.6447 |   0.998020  4.5618e-03 |   0.999186  8.4481e-04 | collins
  0.06    4.0   12.5000   5.6447 |   0.999946  1.3485e-03 |   1.000000  3.5104e-08 | collins
  0.06    6.0   18.7500   5.6447 |   0.999980  6.3783e-04 |   1.000000  4.8766e-15 | collins
  0.06   10.0   31.2500   5.6447 |   0.999987  4.5149e-04 |   1.000000  7.0012e-15 | collins
  0.10    1.5    4.6875   3.3868 |   0.927935  7.5628e-02 |   0.933357  6.3802e-02 | collins
  0.10    2.5    7.8125   3.3868 |   0.996916  8.5737e-03 |   0.999186  8.4481e-04 | collins
  0.10    4.0   12.5000   3.3868 |   0.999821  3.7782e-03 |   1.000000  3.5104e-08 | collins
  0.10    6.0   18.7500   3.3868 |   0.999940  1.7867e-03 |   1.000000  1.5707e-14 | collins
  0.10   10.0   31.2500   3.3868 |   0.999981  1.0016e-03 |   1.000000  9.0597e-15 | collins
  0.18    1.5    4.6875   1.8816 |   0.915115  8.3108e-02 |   0.933357  6.3802e-02 | collins
  0.18    2.5    7.8125   1.8816 |   0.993752  1.9466e-02 |   0.999186  8.4481e-04 | collins
  0.18    4.0   12.5000   1.8816 |   0.998520  1.2633e-02 |   1.000000  3.5104e-08 | collins
  0.18    6.0   18.7500   1.8816 |   0.999638  5.9770e-03 |   1.000000  6.6280e-15 | collins
  0.18   10.0   31.2500   1.8816 |   0.999880  3.2159e-03 |   1.000000  1.3591e-15 | collins
  0.30    1.5    4.6875   1.1289 |   0.883727  1.3703e-01 |   0.933357  6.3802e-02 | collins
  0.30    2.5    7.8125   1.1289 |   0.977832  4.9805e-02 |   0.999186  8.4481e-04 | collins
  0.30    4.0   12.5000   1.1289 |   0.987216  3.8300e-02 |   1.000000  3.5104e-08 | collins
  0.30    6.0   18.7500   1.1289 |   0.996394  1.8236e-02 |   1.000000  4.3297e-14 | collins
  0.30   10.0   31.2500   1.1289 |   0.998831  9.7889e-03 |   1.000000  3.7095e-14 | collins
  0.45    1.5    4.6875   0.7526 |   0.803254  2.7147e-01 |   0.933357  6.3802e-02 | collins
  0.45    2.5    7.8125   0.7526 |   0.924144  1.2495e-01 |   0.999186  8.4481e-04 | collins
  0.45    4.0   12.5000   0.7526 |   0.938239  1.0169e-01 |   1.000000  3.5104e-08 | collins
  0.45    6.0   18.7500   0.7526 |   0.975546  5.1970e-02 |   1.000000  3.9675e-14 | collins
  0.45   10.0   31.2500   0.7526 |   0.987624  2.8766e-02 |   1.000000  4.5350e-14 | collins
```

**0 of 30 cells worse.**  Two structural facts in the Collins column worth
naming, because they are the signature of an exact transport:

* at a given extent the Collins reading is IDENTICAL across NA
  (6.3802e-02 at ext 1.5, 8.4481e-04 at 2.5, 3.5104e-08 at 4.0 on all six NAs).
  It has to be: the error is the input grid's own truncation of the Gaussian and
  nothing else, and that depends on the extent in beam radii, not on the
  focusing.  At ext 4 the Gaussian's amplitude at the grid edge is
  `exp(-16) = 1.1e-7`, which is the 3.5e-08 reading;
* at ext >= 6 it reaches 1.3e-15 to 4.5e-14, i.e. the transform's rounding.

"Materially better in the cells the small-extent branch exists for" -- ext 2.5
sits under the 3.695-beam-radius knee where `_small_extent_focus_standoff_f`
takes over:

| NA | sziklas relL2 | collins relL2 | factor |
|---|---|---|---|
| 0.03 | 2.3022e-03 | 8.4481e-04 | 2.73x |
| 0.06 | 4.5618e-03 | 8.4481e-04 | 5.40x |
| 0.10 | 8.5737e-03 | 8.4481e-04 | 10.15x |
| 0.18 | 1.9466e-02 | 8.4481e-04 | 23.04x |
| 0.30 | 4.9805e-02 | 8.4481e-04 | 58.95x |
| 0.45 | 1.2495e-01 | 8.4481e-04 | 147.90x |

and at ext 1.5 (below where the resolver can reach its margin at any leg length)
1.18x to 4.25x.

### (b) The C1 mismatch matrix

WP-A6's own fixture reproduced (`lambda = 1.31 um`, N = 1024, `w_in = 1 mm`,
NA 0.05 so `R0 = -20 mm`, ext 4, `dx = 7.8125 um`, `z = -R0`,
`w0 = 8.3397 um`, `dx_out = w0/8`, `N_out = 64`; ONE physical field re-enveloped
against each carrier; peak ratio squared against the matched readout).  The
sziklas column reproduces WP-A6's published row to every printed digit, which is
what says the fixture is the same one:

```
  R/R0 |  szik peak  szik vs oracle |  coll peak  coll vs oracle |  coll period
  1.00 |   1.000000      9.3513e-04 |   1.000000      3.5011e-08 |  3353.60 um
  0.99 |   0.999514      4.2401e-03 |   0.999999      2.4674e-07 |  3353.60 um
  0.98 |   0.998602      7.8768e-03 |   0.999998      9.7052e-07 |  3353.60 um
  0.95 |   0.994304      1.7528e-02 |   0.999986      1.2837e-05 |  3353.60 um
  0.90 |   0.985236      2.9969e-02 |   0.999938      1.1624e-04 |  3353.60 um
```

**`'collins'` reads peak ratio 1.0000 at every R/R0**, as §6.1 predicted, and the
mechanism is visible in the last column: the period is `lambda |z| / dx` of the
INPUT grid, so the carrier is not in it and it is the same number on all five
rows (identical to the bit).  The sziklas period moves with the resolved
standoff, which is what C1 made depend on the beam.

### (c) A two-group chain against the brute-force ASM + `apply_real_lens_traced` arm

`p5_chain.py`'s method re-implemented independently (method read, not imported):
`lambda = 1.31 um`, two identical biconvex singlets (+/-51.68 mm, 5 mm centre,
n = 1.5168 fixed, semi-diameter 10 mm) 40 mm apart, a 6 um waist 30 mm in front
of the first vertex, N = 2048 spanning exactly 6 `w_L`, `ray_subsample=2`,
`amplitude_model='ray_density'` + `preserve_input_phase='remap'` +
`remap_sampling='full'`, target at HALF the paraxial image distance.  The brute
arm is plain band-limited ASM on the same pitch with the same element call.

| arm | dx | power | ratio vs brute | r2m | vs brute 1.143892 mm |
|---|---|---|---|---|---|
| brute | 6.1082 um | 6.82760e-06 | 1 | 1.143892 mm | -- |
| `'sziklas'` | 4.479347 um | 6.82806155e-06 | **1.000067** | 1.138817 mm | **0.444 %** |
| `'collins'` | 4.479347 um | 6.82806155e-06 | **1.000067** | 1.138817 mm | **0.444 %** |

-- the audit's own recorded readings (power 1.000067, r2m 1.13882 vs 1.14389 mm,
0.44 %) reproduced on both transports, centroids on axis to 1.1e-10 m.
`'collins'` agrees at least as closely: `np.array_equal` on the two returned
fields.

That equality is the complementarity claim arriving at the chain level, not a
coincidence.  This fixture's only gap leg measures **K1 = 1.9167 > 1**, so the
chirp-Z quadrature is not sampled there and the transfer-function quadrature is;
`'collins'` therefore evaluates the same integral by the same quadrature the
default does.  Reported as `collins_form='tf'` on the stage.  (At N = 1024 both
arms read power 0.999972 and r2m 12.845 % against a brute arm that is itself
under-sampled -- the fixture's N = 2048 is not optional, and both transports
move together.)

The informative variant, the same chain read out AT the image plane with a
`focus_readout` (where the chirp-Z quadrature does run):

| transport | peak | r2m | window power | Bluestein period |
|---|---|---|---|---|
| `'sziklas'` | 9.650122e+01 | 18.0381 um | 6.182473e-06 | 251.59 um |
| `'collins'` | 9.662574e+01 | 18.0436 um | 6.184677e-06 | **12125.55 um** |

-- a 48x wider faithful window, with the spot agreeing to 0.031 % in r2m and
0.13 % in peak.

### (d) `propagate_traced_carrier_chain_multi`: K = 1 and K = 2

| claim | `'sziklas'` | `'collins'` |
|---|---|---|
| K=1 multi vs the single-congruence chain, `max abs diff` | **0.0** | **0.0** |
| stage lists equal | yes | yes |
| K=2 multi vs the hand-summed pair, `max abs diff` | **0.0** | **0.0** |

(two-group relay, N = 512 at 30 um, `w = 4.5 mm`, `r_in = 60 mm`, gaps
20 / 10 mm, `final_distance = 8 mm`, `focus_readout` 256 x 0.5 um; K = 2 as two
untilted congruences differing in carrier radius and amplitude, `readout_tile
=None`.)

### (e) The P2 design battery, with `transport='collins'` injected

`_through_focus`'s own reduction, with the transport injected at
`la.propagate_traced_carrier_chain` and `replica_fill` swept:

```
                        cell  transport   fill |  FWHM/um   theory   ratio |   EE1w    EE2w    EE3w |   dz/mm
    doublet 2.5x (unclipped)   sziklas   zero  |  18.500   17.413  1.0624 | 0.8585  0.9970  0.9980 |  0.1311
    doublet 2.5x (unclipped)   sziklas  repeat |  20.500   17.413  1.1773 | 0.3531  0.4953  0.5032 |  0.3934
    doublet 2.5x (unclipped)   collins   zero  |  18.500   17.413  1.0624 | 0.8585  0.9970  0.9980 |  0.1311
    doublet 2.5x (unclipped)   collins  repeat |  18.500   17.413  1.0624 | 0.8585  0.9970  0.9980 |  0.1311
    doublet 1.2x (truncated)   sziklas   zero  |  23.500   19.172  1.2257 | 0.7535  0.9229  0.9292 |  0.0000
    doublet 1.2x (truncated)   sziklas  repeat |  23.500   19.172  1.2257 | 0.7535  0.9229  0.9292 |  0.0000
    doublet 1.2x (truncated)   collins   zero  |  23.500   19.172  1.2257 | 0.7535  0.9229  0.9291 |  0.0000
    doublet 1.2x (truncated)   collins  repeat |  23.500   19.172  1.2257 | 0.7535  0.9229  0.9291 |  0.0000
```

with, on the 2.5x cell (`R_exit = -65.96537 mm`, grid 1024 x 21.484375 um,
`w_exit = 1.8599 mm`, window 256 um):

| transport | period | window / period | faithful samples |
|---|---|---|---|
| `'sziklas'` | 124.113 um | 2.0626 | (249, 249) of 512 |
| `'collins'` | **4022.208 um** | **0.0636** | **(512, 512)** |

**This is the demonstration the WP asked for.**  The fixture WP-A25 found scoring
a Bluestein replica of its own core needs `replica_fill='zero'` on the shipped
transport -- without it the window's brightest sample is a copy and the cell
reads FWHM 20.500 um / EE2w 0.4953 against an analytic 17.413 um / 0.9970.  On
`'collins'` the requested window is 6.4 % of one period, every sample carries
measurement, and the two fills return the same array: 18.500 um / 1.0624x /
0.8585 / 0.9970 / 0.9980 at `dz = +0.1311 mm` under both.  A freely chosen output
pitch needs no replica handling on this cell.

The truncated 1.2x cell agrees to the last printed digit under both transports
(EE3w 0.9292 vs 0.9291), which is the control: that cell's window is 1.27
periods, under the two-period line, so nothing was being repaired there and
nothing changed.

---

## 3. The Kelly sampling guard

`on_collins_sampling`, default `'warn'`; three conditions
(`_collins_sampling_stats`, `carrier.py:1677`; `_check_collins_sampling`,
`:1736`), each a RATIO against the Nyquist rate itself, bar 1, no margin.

**K1, input.**  The transform samples the product
`g(u) = u_in(u) exp(i k A u^2/(2B))`.  Its local spatial frequency at `u` is
`A u/(lambda B)` from the screen plus at most `theta/lambda` from the envelope,
and the two ADD (nothing cancels -- `g` is what is sampled):

    K1 = 2 dx (|A| r / |B| + theta) / lambda  <=  1.

**K2, output.**  The returned samples must resolve the transported envelope,
whose angular half-width is the ABCD image of the input box's:

    K2 = 2 dx_out (|C| r + |D| theta) / lambda  <=  1.

The naive "chirp-Z frequency plus post-chirp frequency" bound does NOT apply:
stationary phase puts the chirp-Z's own local frequency at `-x/(A lambda B)` and
the post-chirp's at `+D x/(lambda B)`, whose sum is `C x/(A lambda)` -- they
cancel to exactly the ray-transfer term, which is why the phase-space statement
is the right one and the naive sum is wildly pessimistic.

**K3, period.**  The chirp-Z sums over the INPUT lattice, so the output is
periodic with `lambda |B| / dx` -- set by the input pitch and the leg, and by
nothing on the output side.  Disposed of by the EXISTING `on_replica` on this
period, with the existing `[V3]` geometry
`2|centre_out| + N_out dx_out <= period`, so the two guards cannot disagree and
`replica_fill` keeps working unchanged.

**The tolerance, and why it is the only one.**  `r` and `theta` are the field's
own support radii, read per axis from the spatial and angular power marginals on
every call (`_collins_space_support` / `_collins_angle_support`,
`carrier.py:1622` / `:1635`) at the containment level
`_COLLINS_TAIL_FRAC = 1e-6` (`:1518`).  Content outside those radii is the
content the chirp-Z may alias, so the aliased power is bounded by 1e-6 and the
field error by its square root.  Nothing else in the guard is a number.

That this is not a geometric margin is measurable: on a `w = 0.3 mm` Gaussian on
a 2.048 mm half-grid the measured support is 0.7320 mm (against the analytic
1e-6 containment radius of the intensity marginal, `4.892 x w/2 = 0.7338 mm` --
one 4 um cell low, which is the lattice's quantisation), so the geometric form
reads the condition 2.8x high.  On the `z = +8 mm` cell it would refuse a leg
whose measured departure from the direct-summation oracle is 0.000000.

**Stated fail-before, as a ladder.**  Engineered through the API (a coarse grid
on a short leg drives the pre-chirp past Nyquist), at `A = 0.9`, `B = 3 mm`,
`w = 0.9 mm`, measured against the ANALYTIC Gaussian -- the direct-summation
oracle cannot arbitrate here and is not used, because it evaluates the same
DISCRETE sum and aliases identically:

| n | dx | K1 | chirp-Z relL2 | transfer-function relL2 |
|---|---|---|---|---|
| 256 | 40 um | 49.694 | 1.1260e+02 | 6.2838e-11 |
| 512 | 20 um | 24.847 | 5.5158e+01 | 6.2838e-11 |
| 1024 | 10 um | 12.424 | 2.7432e+01 | 6.2838e-11 |
| 2048 | 5 um | 6.212 | 1.3299e+01 | 6.2838e-11 |

The chirp-Z error tracks K1 (ratio 2.27 / 2.22 / 2.21 / 2.14), which is what says
it IS the aliasing rather than something else the cell also has; the guard fires
on every row at `on_collins_sampling='error'`; and the quadrature the transport
actually selects at `K1 > 1` sits on the analytic oracle's own floor
(6.28e-11, identical on all four grids, so it is the oracle and not the grid).

### 3.1 The quadrature selection is complementary, not a threshold

`K1 <= 1` with `r` at the grid half-width is `N dx^2 <= lambda |z_eff|`.  The
transfer-function form (`_carrier_step_fast`) samples the kernel on the
FREQUENCY lattice instead and needs `|z_eff| <= N dx^2/lambda` -- the same
inequality reversed.  So:

* every leg satisfies at least one of the two;
* both are satisfied at the crossover `K1 = 1`;
* where both hold the two evaluations agree to ~1e-11 of peak (measured
  1.1e-11 at `K1 = 0.34`, 3.3e-12 at `K1 = 0.16`), so the selection cannot
  introduce a step and needs no smoothing at its boundary;
* the two regimes are disjoint from the focus machinery: a landing close enough
  to trip `_near_focus_needs_bridge` has `|A| < 0.02` and therefore `K1 << 1`.

A leg whose output lattice is the co-moving one takes whichever form is sampled;
a leg whose lattice is NOT (the pitch floor engaged, a flat reference, a caller
override, the readout) has no transfer-function form to fall back to and stays on
the chirp-Z with the guard speaking.  `collins_form` is published per stage.

On the marginal cells the arbitration went the other way from the naive
expectation, and it is worth recording: at `z = +8 mm` (`K1 = 0.57`) and
`z = -6 mm` (`K1 = 1.08`) the two quadratures differ by 1.6e-02 and 1.0e+00 of
peak, and the direct sum reads the chirp-Z at ratio 1.000000 at every sampled
point while the transfer-function arm departs (1.001321 at the outermost sample
of the -6 mm leg).  The disagreement on those cells is the TF form's own
wrap-around, not the chirp-Z's.

### 3.2 K4 -- and the defect it caught in this WP's own first cut

`gap_kernel` keeps its meaning on this transport.  The Collins stage IS the
ABCD-Fresnel integral, so `'fresnel'` is it unmodified and `'exact'` pre-applies
the diagonal exact/Fresnel kernel ratio on the input grid
(`_collins_exact_kernel_correction`, `carrier.py:1841`).  That is an exact
operator identity -- both kernels are diagonal in the input plane's Fourier
basis, so `P_exact[z_eff] = P_fresnel[z_eff] . D` and applying `D` to the
envelope before the transport gives the exact-kernel answer on the freely chosen
lattice, at one extra FFT pair of `N`.

The refinement lives on the REDUCED frame `z_eff = B/A`, which is unbounded as a
leg approaches the geometric focus -- exactly the regime the chirp-Z quadrature
exists for.  The first cut applied it unconditionally.  It left the core right
and destroyed the halo, and the reduction the P2 battery uses (argmax over the
window, then a ring profile) scored the halo: FWHM 4.500 um against an analytic
17.413 um, EE2w 0.7868.  Caught by the DIRECT SUMMATION oracle -- the same
integral by brute force, no FFT anywhere in it -- on the battery's own exit
field:

```
    x/um      |direct|   collins BEFORE   collins AFTER   sziklas
     0.0  1.303538e+02      1.000000        1.000000      0.999956
     8.0  1.000904e+02      1.057616        1.000000      0.999863
    16.0  4.364616e+01      1.292277        1.000000      0.999925
    24.0  9.239164e+00      1.297749        1.000000      0.999836
    28.0  3.182034e+00      3.909809        1.000000      1.009437
    32.0  1.406111e+00      8.392720        1.000000      0.978094
    40.0  2.551491e-01     37.460765        1.000000      1.160983
    60.0  2.292158e-02    373.947107        1.000000      3.201506
   100.0  1.262922e-02    524.353891        1.000000      0.000000
```

The fix is a fourth condition of the same shape as K1-K3
(`_collins_kernel_wrap_ratio`, `carrier.py:1820`).  The refinement is a pure
phase `phi(q)`, so its impulse response sits at the group delay
`|dphi/dq| = |z_eff| theta (1/sqrt(1-theta^2) - 1)` evaluated at the envelope's
own measured angular half-width; a displacement beyond the grid half-width does
not blur the answer, it WRAPS it:

    K4 = 2 |dphi/dq| / (N dx)  <=  1.

Measured (`lambda = 1.064 um`, N = 1024, `dx = 4 um`, `w = 0.3 mm`,
`R = -40 mm`):

| A | z_eff | K4 | kernel |
|---|---|---|---|
| 0.5 | 40 mm | 2.2784e-07 | exact |
| 0.025 | 1.56 m | 8.8857e-06 | exact |
| 0.0025 | 15.96 m | 9.0907e-05 | exact |
| 0.001 | 39.96 m | 2.2761e-04 | exact |
| 1e-08 | 4.0e+06 m | 2.3e+04 | fresnel |
| 0 | inf | inf | fresnel |

-- so the refinement runs over the whole ordinary range, including well inside
the near-focus zone, and is dropped only where it cannot be represented.  An
explicit `gap_kernel='exact'` there is REFUSED rather than silently downgraded
(the D4 / D11b adjudication); `'auto'` takes the ABCD-Fresnel integral and
records it, because `'auto'` means "the best available for this geometry", which
is the same wording `propagate_carrier_referenced` already uses for the
astigmatic case.  The remaining paraxiality of the Collins integral itself is
`k B theta_full^4/8` at the beam's own angle -- 1.9e-05 rad on the battery cell,
four decades under the 0.30 rad the chain's own `gap_env_phi_tol` tolerates.

The AFTER column above is also the strongest single statement in this report:
against a quadrature that shares no FFT, no Bluestein and no chirp with it, the
Collins readout reads ratio **1.000000** at every sample from the peak down into
the halo, while the shipped readout departs at 1.16x (40 um) and 3.20x (60 um)
and is blanked beyond its 124 um period.

---

## 4. Byte-identity of the default -- proved ARCHIVE TO ARCHIVE

Two statements, and they are not the same statement.

### 4.1 `transport='sziklas'` is the default, in-tree

Naming the default explicitly is `np.array_equal` on the returned field, pitch,
carrier and stage list at every entry point (`TestDefaultIsByteIdentical`):
`propagate_carrier_referenced` over five parametrised cases including a
collimated carrier, a back-propagating leg, an astigmatic `(R_x, R_y)` carrier
and a near-focus landing that takes the focus-crossing split; the chain's field,
`dx`, `R` and whole `stages` list; and the multi orchestrator's field, `dx`,
`centre`.

### 4.2 The default IS release 5.46.0's arithmetic

That second claim cannot be made against the working tree: three other Wave-4
engineers are landing into it, and `lumenairy/elements/_lens_traced.py` -- which
every chain leg calls -- is modified there by someone else.  So it is made
**archive to archive**, with the transport's own file as the only variable:

1. `git archive 81d5b586 lumenairy` extracted twice, to `base/` and `mine/`;
2. `mine/lumenairy/propagators/carrier.py` overwritten with the working tree's
   copy -- `diff -rq` over the two trees reports that one file and no other;
3. one CHILD PROCESS per tree (`sys.path` pointed at its own root, module
   provenance asserted from `carrier.__file__`), each running the same fixture
   set of SHIPPED calls -- no `transport=` argument anywhere;
4. the two `.npz` compared on `shape`, `dtype` and `np.array_equal`.

**41 of 41 entries EQUAL, 0 differ.**  The set:

* the single carrier step, every branch it has -- a short and a long converging
  leg, a back-propagating leg, a collimated carrier, a diverging carrier, a
  near-focus landing, a focus CROSSING, an astigmatic `(R_x, R_y)` pair (`env`,
  `R` and `dx` on each), and an explicit `gap_kernel='fresnel'` leg, which is the
  arm whose entire documented purpose is FP identity with prior releases;
* both public focus readouts (`carrier_referenced_focus_readout` and
  `carrier_referenced_exact_focus_readout`);
* `carrier_referenced_reconstruct` / `_envelope` / `_fit_radius` / `_aperture`;
* `propagate_traced_carrier_chain` with a focus readout AND on its bare final
  leg -- field, `dx`, `R`, and `repr(stages)` (a 959- and a 1211-character
  string, so every published diagnostic on every stage is in the comparison);
* `propagate_traced_carrier_chain_multi` at K = 1 -- field and `dx`.

Structurally the default path gained only vocabulary checks (`_check_transport`,
`_check_guard_action`) and one `if transport == 'collins'` branch ahead of the
existing astigmatic branch; no expression on the default path was re-associated.
The history-fingerprint gate confirms the module's AST and token stream moved
(they must -- the code is new) and the ratchet in
`test_audit2609_a17_history_lint.py` confirms no version narrative was added to
the source.

The verification set is green (§6), which includes the 165 WP-A6 / VERIFY-A6
pins and the 601-test `-k carrier` sweep.

---

## 5. What a later default flip would retire, and what would justify it

**Not removed in this package**, as the WP requires: the default still uses all
of it.  What a flip to `transport='collins'` would retire:

| symbol (line in the changed file) | why it exists | why `'collins'` does not need it |
|---|---|---|
| `_near_focus_needs_bridge` (`carrier.py:2279`) | the co-moving half-width shrinks below the diffraction-limited waist | the output pitch is floored at `2(\|A\| r + \|B\| theta)/N` |
| `_propagate_carrier_focus_crossing` (`:2310`) | `m <= 0` inverts the frame | `A <= 0` is an ordinary value in the exponent |
| `_axis_bridge` (`:2599`) | the astigmatic per-axis form of the same split | ditto, per axis |
| `_default_focus_standoff` (`:3958`), `_beam_containment_standoff` (`:4066`), `_check_focus_containment` (`:4190`), `_small_extent_focus_standoff_f` (`:4353`), `_achievable_focus_margin` (`:402`), `_FOCUS_STANDOFF_*`, `_FOCUS_READOUT_CONTAINMENT_*` | the readout must stop SHORT of the target because its grid collapses at the target | the readout lands on the target in one step; there is no stop plane to size or to guard |
| the replica guard's standoff coupling (not the guard) | `period = N dx_stop` and `dx_stop ~ standoff` | `period = lambda \|z\| / dx` of the input grid |
| C1 itself | the stop grid was sized from the carrier, not the beam | there is no stop grid |

That is roughly 900 lines of `carrier.py` plus their pins.

**The measurements that would justify the flip**, of which this package supplies
the first four:

1. gate (a) extended to the full NA x extent matrix with the SHIPPED chain (not
   the bare readout) -- done here for the readout, 0/30 cells worse;
2. gate (b) -- done, peak ratio 1.0000 at every mismatch;
3. the P2 battery -- done, identical metrics with `replica_fill` inert;
4. `_multi` K = 1 / K = 2 -- done, exactly 0.0.

Still missing, and each is a real gate:

5. **the design-121 acceptance** (FWHM 3.450 / EE3 88.8 % / EE6 99.6 %, and the
   8x4 Dammann fan).  Its assets are in `validation/repro_traced_carrier_122/`
   and are UNTRACKED on this machine -- the same reason WP-A6 §6.1 deferred the
   whole item.  Until that runs, the flip is not defensible: it is the only
   fixture in the library where a whole DOE fan, a tilted congruence, a
   `final_leg='auto'` route flip and a per-order readout tile all interact;
6. **the cost**.  A gap leg is 4.9x-5.2x more expensive on the chirp-Z
   quadrature (§7), so a flip must either keep the complementary selection --
   which it should, since that is what keeps gate (c) bit-identical -- or accept
   that cost on legs that do not need it.  The selection means a flip changes
   nothing on legs the shipped transport already handles, which is the strongest
   argument FOR it and also the thing that must be re-measured on a real design;
7. **the tilted / decentred congruence at the readout**.  `'collins'` carries the
   chief-ray frame exactly as the default does (the ramp, the obliquity piston
   and the chief-ray advance stay outside the transport) and `test_niche_d1` /
   `test_niche_d2` are green with it wired, but no fixture in this package reads
   a strongly tilted congruence THROUGH the Collins readout against an
   independent oracle.  `_check_tilt_fits` also still sizes the decentred beam
   against the co-moving grid, which a flip would want to revisit;
8. **backends**.  `_collins_transport` is NumPy-only (it calls `_fft2`/`_ifft2`
   and passes `xp=np` to the Bluestein).  `test_niche_k2_carrier_backends.py`
   would need a CuPy/JAX arm before a default that every backend reaches.

---

## 6. Verification set

All run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one
process at a time, on HEAD **8dab7de5** (WP-B1, B2, B6 and the B1 follow-up
landed under this package; none of them touches `carrier.py`).

| | result |
|---|---|
| `tests/unit/test_audit2609_b4_collins_transport.py` (new) + `test_audit2609_a6_carrier.py` + `test_audit2609_a6_verify_carrier.py` | **249 passed** (293 s) -- 84 new (76 + 8 slow), 165 WP-A6 / VERIFY-A6 pins |
| `test_audit2609_a24_decentre_calibration.py` + `test_audit2609_a25_carrier_focus_readout.py` + `test_niche_d1_tilted_carrier.py` + `test_niche_p2_design_battery.py` | **83 passed** (186 s) |
| `test_niche_d6_exact_tilted_leg.py` + `test_niche_d2_chain_multi.py` | **76 passed** (682 s) |
| `pytest tests/unit -k carrier` | **601 passed, 4 skipped** (558 s) |
| `python validation/run_all.py` | **ALL 37 files passed** |
| `ruff check lumenairy/propagators/carrier.py tests/unit/test_audit2609_b4_collins_transport.py` | clean |
| `record_history_fingerprints.py lumenairy/propagators/carrier.py --reason ...` then `--check` | `carrier.md` **OK**, `carrier_field.md` **OK** |
| `test_audit2609_a17_history_lint.py` | **5 passed** -- the version-narrative ratchet does not name `carrier.py` |
| `test_audit2609_a17_history_relocation.py` | 740 passed; the 4 failures are `lumenairy.elements._lens_real` and `lumenairy.propagators.system`, both other engineers' in-flight files. `-k carrier` is **12 passed** |

The 4 skips are environmental and pre-existing (`PySide6` absent; `cupy` absent,
which skips the two CuPy arms of `test_niche_k2_carrier_backends.py`).

**What in the tree is not mine.**  Other Wave-4 packages are landing
concurrently, so `record_history_fingerprints.py --check` reports DRIFT on
`lumenairy.elements._lens_traced` (and the relocation test additionally on
`lumenairy.elements._lens_real` and `lumenairy.propagators.system`).
`carrier.md` and `carrier_field.md` are OK on both instruments.

**One transient worth recording for whoever re-runs this.**  A `-k carrier`
sweep taken earlier, while `_lens_traced.py` was being rewritten under it, read
77 failed / 8 errors; every one of those passed on re-run in isolation minutes
later, and two clean sweeps since (649 s and 558 s, before and after the three
packages landed) read 601 passed / 4 skipped identically.  A long sweep taken
across another engineer's write is not a reading.

---

## 7. Cost, measured (no test asserts a timing)

`perf_counter` medians of 7 interleaved runs (5 for the readouts), same machine,
single-threaded.  Per gap leg on the same-`N` co-moving lattice
(`R = -40 mm`, `z = 20 mm`, `lambda = 1.064 um`):

| N | sziklas (TF step) | collins chirp-Z, `gap_kernel='auto'` | ratio | collins, `'fresnel'` | ratio |
|---|---|---|---|---|---|
| 512 | 15.95 ms | 77.26 ms | 4.85x | 65.10 ms | 4.08x |
| 1024 | 67.40 ms | 329.24 ms | 4.88x | 273.65 ms | 4.06x |
| 2048 | 298.59 ms | 1551.23 ms | 5.20x | 1288.70 ms | 4.32x |

The WP predicted ~2-3x; the measured 4-5x is the Bluestein running three
transforms of `next_fast_len(N + N_out - 1) ~ 2N` per axis when `N_out = N`, and
the `'auto'` column carries the exact-kernel refinement's extra FFT pair of `N`
(0.8x of the leg).  The separable route is already on
(`_EXACT_READOUT_SEPARABLE_BLUESTEIN`).

The IMAGE-PLANE READOUT goes the other way, because it replaces a whole carrier
leg, the C1 curvature fit and two containment measurements with one screen and
one Bluestein:

| N | N_out | sziklas readout | collins readout | ratio |
|---|---|---|---|---|
| 1024 | 64 | 305.36 ms | 92.07 ms | **0.30x** |
| 1024 | 256 | 356.96 ms | 105.13 ms | **0.29x** |
| 2048 | 256 | 1299.96 ms | 393.03 ms | **0.30x** |

So on a chain whose legs are in the transfer-function regime (the ordinary case,
and gate (c)'s) `transport='collins'` is *cheaper* end to end, because the legs
cost the same and the readout is 3.3x faster.

---

## 8. Requested changes outside my ownership

**None required.**  The transport consumes
`lumenairy/propagators/_bluestein.py::_bluestein_centred_2d` directly rather than
`propagators/mft.py`, so WP-B3's file is not on the path at all.  Two
observations for the owners, neither blocking:

1. **`_bluestein_centred_2d` needed nothing.**  Verified against a brute
   `N_in x N_out` DFT matrix at this package's own `alpha`, `N_in = 1024`,
   `N_out = 512` with a hard-edged input: max relative difference 1.19e-14
   (separable) and 1.20e-14 (non-separable).  If WP-B3 wants a pin for that, the
   probe is 25 lines and is reproduced in
   `TestSameTheorem::test_the_transform_equals_a_direct_summation_of_the_same_integral`'s
   method.
2. **A finding in a file I do not own, for whoever owns VERIFY-A6's fixtures.**
   `tests/unit/test_audit2609_a6_verify_carrier.py::_abcd_field` builds the
   Gaussian `q` parameter as `1/q = 1/R - i lambda/(pi w^2)` and carries the Gouy
   phase as `angle(q/q2)`.  That is Siegman's convention, which belongs to
   `exp(+i omega t)`; in this library's `exp(-i omega t)` / `exp(+i k z)`
   convention (CONVENTIONS sec. 7) the correct pairing is
   `1/q = 1/R + i lambda/(pi w^2)`, i.e. the Gouy phase has the opposite sign.
   The amplitude and the radius of curvature are unaffected (`Re(1/q)` and
   `|Im(1/q)|` are the same), so every piston-free comparison in that file is
   unaffected -- which is why it has never bitten.  It is exactly `pi` of error
   at a focus, and it is why this package's oracle is written the other way and
   compares ABSOLUTELY: measured, the Collins readout against the corrected
   oracle reads relL2 2.17e-14 including the piston, and against the
   as-shipped-in-a6 form reads 2.000 (a global `-1`).

   It is NOT a one-line fix: flipping the sign at
   `tests/unit/test_audit2609_a6_verify_carrier.py:64` also flips the sign of
   `Im(1/q2)`, and line 68 reads `wz = sqrt(-lam/(pi*Im(inv2)))`, so the width
   would go imaginary.  The whole-function form that is right in this library's
   convention, and the one this WP's own oracle uses, is

   ```python
   q  = 1.0 / (1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2))
   q2 = q + z
   wz = float(np.sqrt(lam / (np.pi * (1.0 / q2).imag)))
   E  = (np.exp(1j * kk * z) / (1.0 + z / q)
         * np.exp(1j * kk * r2 / (2.0 * q2)))
   ```

   -- the `1/(A + B/q)` prefactor carries the amplitude AND the Gouy phase with
   no separate `angle(q/q2)` term and no branch to pick.  I have NOT made this
   change: the file is not mine, and every assertion in it passes as it stands
   (all 83 of them, re-run in §6).

---

## 9. Residual risk

* **`_collins_transport` is NumPy-only.**  It pulls the field to host through
  the module's own `to_numpy` idiom for the measurements and passes `xp=np` to
  the Bluestein.  A CuPy or JAX field handed to `transport='collins'` will be
  transported on the host; the shipped default is unaffected.  The wiring is
  there (`_bluestein_centred_2d` takes `xp`/`fft2`/`ifft2`), so this is an
  hour's work plus a backend arm in `test_niche_k2_carrier_backends.py`.
* **`transport='collins'` with `final_distance == 0` and a `focus_readout`
  raises.**  The Collins integral's `B` is the leg and there is no `B = 0` form:
  at zero distance the readout is a RESAMPLE of the exit plane, not a transport.
  The message says so and names both ways out.  The shipped transport reaches
  that case by backing off to a standoff plane first; nothing in the
  verification set exercises it.
* **The complementary-quadrature selection means `'collins'` is frequently the
  shipped arithmetic.**  On gate (c)'s chain it is bit-identical, because every
  leg there is in the transfer-function regime.  That is the correct answer and
  the reason the transport is safe to opt into, but it does mean a consumer who
  sets `transport='collins'` expecting the chirp-Z everywhere will not get it;
  `collins_form` on each stage is how they find out.
* **K2 is not disposed of separately from K1.**  Both are weighed by
  `on_collins_sampling` and the message names which fired.  A caller who wants
  to accept an under-resolved OUTPUT lattice while refusing an under-sampled
  INPUT has to use `'warn'` and read the stage.
* **The flat-reference switch changes what the chain's `R` means on that leg.**
  When the space-bandwidth test says the geometric carrier is unaffordable the
  leg returns `R = inf`, and a downstream group's `_paraxial_group_r_out` is then
  computed from a collimated input.  That is physically right at the waist (the
  wavefront IS flat there) and it is published as `collins_flat_reference`, but
  no fixture in this package puts a LENS GROUP at the focus of its own input
  carrier, so the composition of that case with a traced element is untested.

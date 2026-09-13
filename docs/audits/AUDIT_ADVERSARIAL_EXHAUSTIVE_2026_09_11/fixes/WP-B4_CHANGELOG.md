# WP-B4 -- CHANGELOG text (Collins / ABCD-Fresnel carrier transport)

Release 5.47.0.  Section 15.9 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`
("Alternative algorithms worth adopting", second bullet), designed in
`fixes/WP-A6_REPORT.md` §6.1 and derived in `.../CARRIER.md` "Alternative
algorithms" items 1-2.  File: `lumenairy/propagators/carrier.py`.

---

### Added -- `transport='collins'`: the carrier chain's free-space legs on the Collins / ABCD-Fresnel integral, with the output pitch chosen freely

`propagate_traced_carrier_chain`, `propagate_traced_carrier_chain_multi` and
`propagate_carrier_referenced` take `transport={'sziklas', 'collins'}`, default
`'sziklas'`.  **The default is unchanged in every bit** (see Migration below).

The shipped transport is the Sziklas-Siegman co-moving step, whose output pitch
is forced to `m*dx` with `m = (R+z)/R`.  Everything the module does around a
focus exists because that pitch collapses with `m`: the auto-split into
`carrier -> through-waist ASM bridge -> carrier`, the standoff resolver that
stops the focus readout short of the target, and the Bluestein period of that
readout, which is `N` times the contracted stop-plane pitch and therefore a
function of a leg length resolved from the beam for unrelated reasons (findings
C1 and the WP-A25 replica regime).

`'collins'` evaluates the same integral in the form Collins (1970, *JOSA* **60**,
1168) gives for an arbitrary ABCD system, factored as chirp x chirp-Z x chirp
(`lumenairy/propagators/carrier.py:1901` `_collins_transport`).  In this
library's `exp(-i omega t)` / `exp(+i k z)` convention (CONVENTIONS sec. 7 -- the
complex conjugate of the form printed in Collins' paper, which uses the opposite
time convention):

    u_out(x) = exp(i k B)/(i lambda B)
               * integral u_in(u) exp(i k (A u^2 - 2 u x + D x^2)/(2 B)) du

with the envelope-to-envelope system "attach the input carrier, fly `z`, remove
the chosen output carrier" (`carrier.py:1541`):

    A = 1 + z/R_in = m,   B = z,   C = 1/R_in - A/R_ref,   D = 1 - z/R_ref

so `det = AD - BC = 1` for every choice of `R_ref` (pinned as an identity over
20000 log-uniform cells).  `B` is the transfer distance and is positive on every
forward leg, converging or not: the carrier's sign lives in `A`, which shrinks to
zero and past it as a leg crosses the geometric focus, and the transform carries
`A <= 0` natively.  The three stages are the module's own separable screen
(`_radial_carrier_phase`'s per-axis factor, `carrier.py:1603`), the separable
centred Bluestein the readouts already run (`_bluestein_centred_2d`), and a
second separable screen.  At `R_ref = R + z` and `dx_out = m*dx` the result is
term for term `_carrier_step_fast` -- measured agreement 7.6e-12 and 3.2e-12 of
peak at two well-sampled legs, on both `gap_kernel` settings.

What the free pitch buys, measured:

* **the image-plane readout is one step.**  `transport='collins'` lands the
  target plane directly on the caller's `(dx_out, N_out)` (`carrier.py:2165`),
  with no standoff plane, no beam-containment resolution and no near-focus
  bridge.  Against an analytic Gaussian-ABCD oracle carrying the absolute piston
  and Gouy phase, over NA 0.03-0.45 x grid extents 1.5-10 beam radii (30 cells),
  `'collins'` is **no worse in any cell** and better in all 30; at extents >= 4
  it sits on the input grid's own truncation floor (1.3e-15 to 4.5e-14 relL2,
  against 3.1e-04 to 2.9e-02 for the shipped readout), and in the small-extent
  cells the `_small_extent_focus_standoff_f` branch exists for it is 1.18x to
  147.90x better.
* **the readout period stops depending on the carrier and on the leg.**  It is
  `lambda |z| / dx` of the chain's own input grid.  On WP-A6's C1 mismatch
  fixture it reads 3353.60 um on every row while the reference carrier is walked
  from `R/R0` 1.00 to 0.90, and the peak ratio reads 1.000000 / 0.999999 /
  0.999998 / 0.999986 / 0.999938 -- 1.0000 at every mismatch, against the
  shipped column's 1.000000 / 0.999514 / 0.998602 / 0.994304 / 0.985236.
* **the WP-A25 replica regime is gone on the fixture it was found on.**  The P2
  design battery's unclipped-doublet cell requests a 256 um window; the shipped
  readout's period there is 124.113 um (2.0626 periods, 249 of 512 samples
  faithful) and `replica_fill` moves its best-focus reading from FWHM 18.500 um
  / EE2w 0.9970 to 20.500 um / 0.4953.  On `'collins'` the period is 4022.208 um
  (0.0636 periods, all 512 samples faithful) and the two fills return **the same
  array**: 18.500 um / 1.0624x theory / EE 0.8585, 0.9970, 0.9980 under both.
* **a near-focus gap leg no longer splits.**  The output pitch is the co-moving
  `|A| dx` floored by `2(|A| r + |B| theta)/N`, the ABCD image of the envelope's
  measured phase-space box, so it carries the leg's own diffraction and cannot
  follow `A` to zero (`carrier.py:1782`); and where referencing to the
  collapsing ray sphere `R + z` would need more samples than the grid has, the
  output is referenced FLAT instead, which is the physical statement that the
  wavefront is flat at the waist.  Measured 0.1 mm before a 40 mm focus: pitch
  0.2777 um against a co-moving 0.0100 um, 28x.  Pinned by poisoning all five
  entry points of the focus machinery (`_propagate_carrier_focus_crossing`,
  `_axis_bridge`, `_default_focus_standoff`, `_small_extent_focus_standoff_f`,
  `_beam_containment_standoff`) and running the transport through them, with the
  falsifier that the same poison fires on the default transport.

**The sampling guard** (`on_collins_sampling={'error','warn','ignore'}`, default
`'warn'`) is written against Kelly, *Appl. Opt.* **53**, 2861 (2014) rather than
against a geometric margin (`carrier.py:1677`, `:1736`).  Three conditions, each
a ratio against the Nyquist rate itself with the bar at 1 and no margin,
evaluated on the field's own measured `1 - 1e-6`-power support in BOTH domains
rather than at the grid edge:

* **K1** `2 dx (|A| r/|B| + theta) / lambda <= 1` -- the sampled product
  `u_in * exp(i k A u^2/2B)`;
* **K2** `2 dx_out (|C| r + |D| theta) / lambda <= 1` -- the returned lattice
  resolves the transported envelope, whose angular half-width is the ABCD image
  of the input box's (the chirp-Z's own local frequency and the post-chirp's
  cancel to exactly that ray-transfer term, so the naive sum of the two is not
  the bound);
* **K3** `2|centre_out| + N_out dx_out <= lambda |B| / dx` -- disposed of by the
  EXISTING `on_replica` on this transport's period, so the two guards cannot
  disagree.

The tolerance is the one number `_COLLINS_TAIL_FRAC = 1e-6` (`carrier.py:1518`),
the power allowed outside the support radii the ratios are formed from, so the
aliased power is bounded by it and the field error by its square root.  Stated
fail-before, as a ladder over four grids at A = 0.9, B = 3 mm: at K1 = 49.694 /
24.847 / 12.424 / 6.212 the chirp-Z quadrature departs from the analytic
Gaussian by relL2 1.13e+02 / 5.52e+01 / 2.74e+01 / 1.33e+01 -- tracking K1, which
is what says it is the aliasing -- while the complementary quadrature sits at
6.28e-11 on every grid.

**Quadrature selection, and why it is not a threshold** (`carrier.py:2055`).  The
chirp-Z form needs `K1 <= 1`, which with `r` at the grid half-width is
`N dx^2 <= lambda |z_eff|`; the transfer-function form (`_carrier_step_fast`)
samples the kernel on the frequency lattice instead and needs the same
inequality REVERSED.  Every leg therefore satisfies at least one of them, both
hold at the crossover, and where both hold the two agree to ~1e-11 of peak.  A
leg whose lattice is the co-moving one takes whichever form is sampled, decided
by the measured ratio against the Nyquist rate; a leg whose lattice is not (the
floor engaged, a flat reference, the readout) has no fallback and the guard
speaks.  The two regimes are disjoint from the focus machinery -- a landing close
enough to trip `_near_focus_needs_bridge` has `|A| < 0.02` and therefore
`K1 << 1` -- so this transport never enters it.  Each leg publishes
`collins_form`, `collins_k1`, `collins_k2`, `collins_flat_reference` and
`collins_dx_floor_hit` on its stage dict.

`gap_kernel` keeps its meaning on this transport: the Collins stage IS the
ABCD-Fresnel integral, and `'exact'` pre-applies the diagonal exact/Fresnel
kernel ratio on the input grid (`carrier.py:1841`), which is an exact operator
identity because both kernels are diagonal in the same basis.  That refinement
lives on the REDUCED frame `z_eff = B/A`, which is unbounded as a leg approaches
the geometric focus, so it is applied only where its own group delay
`|z_eff| theta (1/sqrt(1-theta^2) - 1)` fits inside the grid it is applied on
(`carrier.py:1820`); an explicit `gap_kernel='exact'` there is REFUSED rather
than silently downgraded, and `'auto'` takes the ABCD-Fresnel integral and
records `collins_kernel='fresnel'`.  Applying it anyway leaves the core right and
destroys the halo: measured against a direct summation of the same integral on
the P2 battery's exit field, ratio 1.00 at the peak, 3.91x at 28 um, 8.39x at
32 um, 37.5x at 40 um and 524x at 100 um.

**Cost.**  Per gap leg on the same-`N` co-moving lattice the chirp-Z quadrature
is 4.9x / 4.9x / 5.2x the transfer-function step at N = 512 / 1024 / 2048
(4.1x / 4.1x / 4.3x with `gap_kernel='fresnel'`, which drops the refinement's FFT
pair) -- the Bluestein runs three transforms of `next_fast_len(N + N_out - 1)`
per axis.  The IMAGE-PLANE READOUT goes the other way and is **3.3x faster**
(0.30x / 0.29x / 0.30x of the shipped readout at N = 1024/N_out = 64, 1024/256
and 2048/256), because it replaces a whole carrier leg, the C1 curvature fit and
the containment measurements with one screen and one Bluestein.

**Migration.**  The default did not move, and that is proved ARCHIVE TO ARCHIVE
rather than against a working tree three other work packages are also landing
into: `git archive 81d5b586 lumenairy` extracted twice with only
`propagators/carrier.py` swapped in one copy, one child process per tree, the
same set of SHIPPED calls (no `transport=` argument anywhere) -- **41 of 41
entries `np.array_equal`, 0 differ**.  The set covers every branch of the single
carrier step (short and long converging legs, back-propagation, a collimated
carrier, a diverging carrier, a near-focus landing, a focus CROSSING, an
astigmatic pair, and an explicit `gap_kernel='fresnel'` leg), both public focus
readouts, the reconstruct / envelope / fit-radius / aperture helpers, the chain
with and without a focus readout including `repr(stages)`, and `_multi` at
K = 1.  To opt in, pass `transport='collins'` to
`propagate_traced_carrier_chain` / `propagate_traced_carrier_chain_multi` (it is
forwarded to every congruence, so K congruences always share one transport), or
to `propagate_carrier_referenced`, which additionally exposes `dx_out` and
`carrier_out` for a single step onto a named lattice and reference.  Two
`focus_readout` keys are REFUSED rather than ignored on `'collins'` because they
describe a stop plane it does not have: `standoff` and `on_focus_containment`.
`final_leg='exact'` is unaffected on either setting -- its fine retrace and
exact-sphere Bluestein readout run no carrier transport, so there is nothing
there for `transport` to select.

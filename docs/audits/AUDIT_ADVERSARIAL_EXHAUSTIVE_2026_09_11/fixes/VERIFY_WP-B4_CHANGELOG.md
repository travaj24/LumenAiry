# VERIFY-WP-B4 -- release text

Independent re-verification of WP-B4 (`transport='collins'`, commit 185d64cd).
Two defects found in the opt-in transport and fixed; the default transport is
unchanged and re-proved byte-identical to the pre-change library, archive to
archive, on a 60-entry fixture set.

---

### Fixed -- `transport='collins'`: a chain leg now weighs the chirp-Z's own output period, and takes the transfer-function form when the chirp-Z cannot be represented on the lattice the leg returns

The Collins chirp-Z sums over the INPUT lattice, so its output repeats with
period `lambda |B| / dx` and the window a leg returns has to fit inside one:
`K3 = N d_out / period <= 1`.  That condition was computed on every call and
disposed of only at the focus readout, where `on_replica` owns it.  A chain leg,
the bare final leg and the public single-step entry
`propagate_carrier_referenced(transport='collins')` have no `on_replica`, so K3
was dropped there -- and the quadrature selection was made on `K1` instead,
which is a strictly weaker statement (`K3 >= K1` on every leg lattice, with
equality exactly when the pitch floor set the pitch).  On the co-moving lattice
`K3 = N dx^2 / (lambda |z_eff|)`, so the unguarded band was every leg whose
reduced distance is shorter than `N dx^2/lambda` -- ordinary short legs on
ordinary grids, not a corner.

Two consequences, both now gone:

* **wrapped copies in the returned array, silently.**  A back-propagating leg
  (`w = 0.3 mm` Gaussian, N = 512 at 7.0312 um, `R = -40 mm`, `z = -20 mm`;
  `K1 = 0.7600`, `K3 = 1.7842`) returned a field whose brightest out-of-period
  sample was **59 % of peak**, carrying **24 %** of the returned power and
  **1.3164x** the input power, reading relL2 **0.5625** against the analytic
  Gaussian -- with no warning at `on_collins_sampling='warn'` and no refusal at
  `'error'`.  It now takes the transfer-function form, returns
  `np.array_equal` to `_carrier_step_fast`, conserves power to 1.000000 and
  reads relL2 1.96e-08.  A caller-chosen `dx_out` that takes the window past
  one period is now REFUSED (or warned) naming `K3 (period)` and the period in
  micrometres, where it used to return a field reading relL2 2.24 in silence.
* **the transfer-function fallback was vetoed by a rounding hair.**  It
  required the resolved output pitch to EQUAL the co-moving `|A| dx` exactly,
  and the pitch floor `2(|A| r + |B| theta)/N` exceeds that by `2|B| theta/N`
  whenever the measured support reaches the grid edge -- i.e. for any beam with
  a real aperture.  On a two-group relay (N = 256 at 60 um) that put a **47x
  under-sampled** chirp-Z on an ordinary 20 mm leg: bare final leg
  `'collins'` peak 29.0002 / power 0.0189589 against `'sziklas'` peak 0.960941 /
  power 2.82889e-05, a 670x power inflation.  The two now agree to every
  printed digit on that fixture.

The selection is now stated on the pair that is actually complementary:
`K3 * K_tf = 2 dx theta / lambda`, and `theta` is read from the envelope's own
SAMPLED spectrum so it cannot exceed the grid's Nyquist angle -- the product is
at most 1, so at least one form is always representable and both are at the
crossover.  Measured at the boundary (found by bisection on the running build):
the two evaluations differ there by **2.9e-11 / 1.4e-11** of peak, against
**0.9999** of peak at the old `K1 = 1` boundary.  The new rule is a strict
superset of the old one, so no leg that took the transfer-function form stopped
doing so.

Legs now publish `collins_k3` beside `collins_k1` / `collins_k2`, and
`collins_kernel` -- which gap kernel `'auto'` resolved to -- which was recorded
internally but never reached a stage dict.

The DEFAULT transport is untouched: `git archive 185d64cd^ lumenairy` against
the same archive with only `carrier.py` replaced, one child process per tree
with module provenance asserted, 60 of 60 entries equal on shape, dtype and
`np.array_equal` -- every single-step branch including a focus crossing, two
near-focus bridge landings, a complex64 arm, a tilted exact leg,
`gap_kernel='fresnel'`, both public readouts, `replica_fill='zero'`, the chain's
field / `dx` / `R` / `repr(stages)`, and `_multi` at K = 1 and K = 2.

### Documented -- what bounds the Collins focus readout, and what K4 does not bound

Two measured statements that were absent and are now at their call sites:

* the one-step focus readout has no co-moving frame, so the CHAIN'S exit pitch
  must resolve the exit beam's convergence over the reduced final leg
  (`K1 = 2 dx (|A| r/|B| + theta)/lambda`).  A long final leg on a small beam is
  comfortable (the WP-A6 fixture reads 0.16); an 8 mm final distance on a
  5.4 mm exit beam sampled at 76 um reads **82.36**, and `final_distance = 0` is
  the limit that this transport already refuses.  The guard says it per call;
  now the docstring says it before the call;
* `K4` is a representability condition on the exact-kernel refinement, not an
  accuracy one.  At `K4 = 1` the refinement's own dropped quartic
  `k |z_eff| theta^4/8` reaches `k theta N dx/8` = **8.6 rad** on a
  `w = 0.3 mm` / N = 1024 / `dx = 4 um` fixture.  Measured 1 um from a
  geometric focus (`z_eff = -1600 m`, `K4 = 9.1e-03`, guard silent):
  `gap_kernel='fresnel'` reads **1.7e-14** against the analytic Gaussian and
  `'auto'` reads **2.4e-03**, the `'auto'` column falling exactly as
  `1/|z_eff|` and independent of N.  Near a geometric focus `'fresnel'` is the
  more accurate setting on this transport.  (The shipped `'sziklas'` path
  applies the same refinement over the same `z_eff`; what is particular here is
  that this quadrature operates at small `A` by design.)

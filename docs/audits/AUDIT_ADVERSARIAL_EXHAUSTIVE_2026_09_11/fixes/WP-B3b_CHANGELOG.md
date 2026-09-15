# WP-B3b -- CHANGELOG text (the K6 resample-back call sites)

Release 5.47.0.  Finding **K6** of
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`, second half:
the call-site pass WP-B3 specified in `fixes/WP-B3_REPORT.md` §5.1 (a)
and (b) and §5.2, with the gate spelled in the general form
`fixes/VERIFY_WP-B3.md` §5 derived, plus that document's F6.  Files:
`lumenairy/propagators/system.py`, `lumenairy/elements/_lens_real.py`,
`lumenairy/propagators/mft.py` (docstring only).  Evidence and every
number below: `fixes/WP-B3b_REPORT.md`.

---

### Changed -- `propagate_through_system(method='fresnel')` evaluates the Fresnel integral on the CHAIN grid instead of resampling back onto it (K6)

The `'propagate'` element step's `'fresnel'` leg ran
`fresnel_propagate`, which lands on the single-FFT kernel's own output
grid `lambda*z/(N*dx)`, and then interpolated that back onto the chain's
working pitch so the next lens or aperture saw the right coordinates.
That resample-back carried two errors: it **cropped** -- everything
outside `N*dx` was discarded, which is the diverging-beam case and the
common one -- and it paid the cubic interpolator's MTF, which bites
hardest on exactly this field because the single-FFT Fresnel output's
residual chirp sits at Nyquist at the grid edge by construction.

The leg now calls `fresnel_propagate_mft` with the chain's own pitch and
sample count (`lumenairy/propagators/system.py:927`).  That is the same
Fresnel integral, sampled where the chain wants it, so neither error
exists.  Refereed against the Fresnel integral written out as an explicit
double sum over the input samples -- no FFT, no Bluestein, no library
call, and therefore an absolute reference rather than a comparison
against another propagator:

| chain fixture | relative L2 vs the double sum, before | after |
|---|---|---|
| square 24x24, z = 1 mm | 2.047281e-1 | **9.075287e-16** |
| square 32x32, z = 2 mm | 2.451674e-1 | **5.288263e-16** |
| square 64x64, z = 1 mm | 1.043497e-4 | **3.474020e-15** |
| non-square 24x18, z = 1 mm | 4.261454e-1 | **7.636427e-16** |
| non-square 64x48, z = 1 mm | 2.457938e-1 | **3.609139e-15** |

and the window power now matches that oracle to every printed digit
where before it did not: on a grid-filling top-hat at lambda = 633 nm,
dx = 2 um, `P_out/P_in` 0.986005 -> **0.986945** (N = 256, z = 5 mm,
oracle 0.986945), 0.996685 -> **0.996992** (N = 512, z = 5 mm, oracle
0.996992), 0.995421 -> **0.996072** (N = 256, z = 2 mm, oracle 0.996072).

The two non-square rows are a correctness fix, not a precision one.
`fresnel_propagate` returns distinct `dx_out` and `dy_out`
(`lambda*z/(Nx*dx)` against `lambda*z/(Ny*dy)`) while `resample_field`
reads a single input pitch, so a grid with `Ny != Nx` had its y axis
rescaled by the **x** ratio -- wrong by `Nx/Ny`, with no diagnostic.

Because there is no resample left to crop, the leg no longer calls
`_warn_system_resample_crop` (`system.py:359`); the `'sas'` leg still
does, unchanged.  `fresnel_propagate_mft` carries the same K1
chirp-sampling guard (`lumenairy/propagators/mft.py:972`) plus its own
faithful-zone warning with period `lambda*|z|/dx_in`, so no diagnostic
is lost -- see Migration for the two messages whose wording moves.

Files: `lumenairy/propagators/system.py:53`, `:806-841`, `:359-383`,
`:160-173`, `:452-459`, `:1638-1646`, `:1771-1783`.
Tests: `tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid`
(12 tests).

**Migration.** Every `propagate_through_system(..., method='fresnel')`
step and every per-element `{'method': 'fresnel'}` returns different
numbers.  The new ones are the Fresnel integral on the chain grid,
verified against the double sum to 5.3e-16 .. 5.0e-14; the old ones were
that integral on a different grid, cropped and interpolated.  Size of the
move: **7.1e-15** where the natural grid already equalled the chain grid
(Bluestein against a plain FFT), **~1e-4** on a contained field,
**~1e-1** on a grid-filling one and **2.5e-1** on a non-square sample
count.  A pinned `'fresnel'` chain result must be re-baselined.  Three
diagnostics also move: the K6 crop `RuntimeWarning` ("the fresnel leg
returned its natural output grid ... which CROPS it") no longer fires
from this leg at all; the K1 under-sampled-chirp `RuntimeWarning` and the
`z <= 0` `ValueError` are now named by `fresnel_propagate_mft` rather
than `fresnel_propagate`, with the same bound, the same numbers and a
pointer to `angular_spectrum_propagate_mft` instead of
`angular_spectrum_propagate`.  `method='asm'`, `'sas'` and every tilted
step are unaffected: `'asm'` chains are proved **byte-identical**.

---

### Changed -- the three surviving resample-backs pick the band-limited resampler by window, not by pitch (K6)

`propagate_through_system`'s `'sas'` leg and
`apply_real_lens`'s two in-glass gap legs (`'sas'` and `'fresnel'`) still
have to resample a kernel's natural output grid back onto the working
grid, and they now choose between WP-B3's band-limited chirp-Z
interpolant and the historical cubic spline:

```python
    method=('chirpz' if N_out * dx_out <= N_in * dx_in else 'spline')
```

per axis (`lumenairy/propagators/system.py:972`,
`lumenairy/elements/_lens_real.py:2970` and `:2889`).  The chirp-Z leg
has unit MTF at every frequency the grid represents, but its
reconstruction is **periodic** with period `N_in*dx_in`, so a window
wider than one period returns replicas rather than the zeros the spline
pads with.  Measured on a field that fills its grid, where the replicas
tile exactly: `P_out/P_in` = 1.000000 at one period, **4.000000** at two
(a 2x2 tiling) and **9.000000** at three, against the spline's 1.000000 /
0.974573 / 0.969247.  On the real `'sas'` leg with a grid-filling top-hat
at dx = 2 um, lambda = 633 nm, against the direct Fresnel evaluation on
the same grid:

| fixture | `dx_new/dx` | selected | spline `P/P_in` | chirp-Z `P/P_in` | direct |
|---|---|---|---|---|---|
| N = 256, z = 5 mm | 1.5454 | **chirpz** | 0.986730 | **0.986951** | 0.986945 |
| N = 64, z = 1 mm | 1.2363 | **chirpz** | 0.961488 | **0.962304** | 0.962222 |
| N = 512, z = 5 mm | 0.7727 | spline | 0.950689 | 1.378837 | 0.996992 |
| N = 512, z = 2 mm | 0.3091 | spline | 0.174014 | 1.820005 | -- |
| N = 256, z = 1 mm | 0.3091 | spline | 0.173601 | 1.835050 | -- |

(the last two rows sit below the K1 chirp-sampling bound, so the direct
evaluation is itself aliased there and is not quoted).

The condition is written as a window against a period rather than as
WP-B3's `dx_new >= dx`, which is only equivalent while `N_out == N_in`;
and it is per axis, because `resample_field` applies one input pitch to
both, so on a non-square input the shorter extent sets the period.  Its
`1e-9` slack is `_warn_mft_output_window`'s own tolerance, so the chirp-Z
leg is taken on exactly the windows that resampler would not warn about.

The in-glass legs matter here because they propagate through glass:
`lam_medium = wavelength/n` makes their `dx_new` smaller by `n` than the
same geometry in air, so they land on the spline side far more often.  On
the WP-A15a covering-array doublet (N = 64 over 1.2 aperture diameters,
dx = 112.500 um, lambda = 632.8 nm) **both** gaps sit at `dx_new/dx` =
4.218e-3 and 1.086e-3 and the crossover is a **2.14 m** thickness; the
1 mm N-BK7 plate at dx = 2 um sits at 1.6320 and takes the chirp-Z leg.
Both directions occur in the shipped suite.

Files: `lumenairy/propagators/system.py:953-978`,
`lumenairy/elements/_lens_real.py:2906-2976`, `:2882-2895`.
Tests: `tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate`
(11), `::TestK6ByteIdentityWhereTheGateSelectsTheSpline` (7),
`::TestK6TheImprovementWhereTheGateSelectsChirpZ` (2).

**Migration.** `propagate_through_system(method='sas')` and
`apply_real_lens(wave_propagator='sas'|'fresnel')` move **only** where
the requested window fits inside one chirp-Z reconstruction period -- in
practice, where the pitch coarsens; there the resample-back gains a unit
MTF (measured: the in-glass `'fresnel'` gap on a 1 mm N-BK7 plate moves
`P_out/P_in` 0.898010 -> 0.899250 on a Gaussian input).  Everywhere else
-- every converging case, which includes every realistic lens gap on a
wide grid -- they are **bit-identical** to 5.46.0, proved by running both
libraries from separate `git archive` extractions in child processes
rather than asserted.  No keyword default moved: `resample_field`'s own
default is still `'spline'`, `propagate_through_system`'s `method` is
still `'asm'`, and `apply_real_lens`'s `wave_propagator` is unchanged.

---

### Changed -- `resample_field`'s docstring says what its unit MTF is a property OF

`method='chirpz'`'s MTF table reads 1.000000 at every carrier, and a
reader could take that as a guarantee about the returned power for any
output grid.  It is not: unit gain is a property of the interpolant,
while the power ratio you measure is also a property of the **window**.
`N_out*dx_out == N_in*dx_in` returns the input's power to the last digit;
a shorter window drops the sliver it does not cover and a longer one
reaches into the first replica.  The extent-preserving default
`N_out = round(N_in*dx_in/dx_out)` lands on the period exactly only when
that ratio comes out whole -- x0.5, x1, x2 and x4 at any `N_in`; x1.5
needs an `N_in` divisible by 3, x1.25 by 5, x1.7 by 17.

The docstring now states the condition, which scale factors satisfy it,
and what the departure costs where they do not: measured at `N_in = 128`
with a 0.30 cyc/px carrier, **0.993922** (x1.25), 0.998015 (x1.5),
0.998489 (x1.7) and 1.004048 (x3) for a rim-filling Gaussian envelope
(4.4 % of its power outside `0.49*N*dx`), against 0.999993 -- 1.000000
for a contained one.  With an exact-period `N_out` the reading is 1 to
-1.1e-16 .. +6.7e-16.

Docstring only: `lumenairy/propagators/mft.py:605-630`.  The module's
AST and token fingerprints are unchanged, which is
`scripts/record_history_fingerprints.py --check` confirming it.
Tests: `tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow`
(7).

---

### Changed -- `propagate_through_system_jax`'s refusal message describes the NumPy twin as it now is

The `NotImplementedError` raised for `method='fresnel'` / `'sas'` on the
JAX entry point said both NumPy branches "resample back onto the input
pitch via scipy map_coordinates".  That is now true only of `'sas'`; the
`'fresnel'` branch runs `fresnel_propagate_mft`'s Bluestein pair.  The
message and the function's docstring say so.  Behaviour is unchanged --
the JAX path is still ASM-only and still refuses both, and an
`method='asm'` JAX chain is byte-identical.

Files: `lumenairy/propagators/system.py:1863-1875`, `:1730-1738`.

# WP-B3b — the K6 call sites: `system.py`'s Fresnel and SAS legs, and `_lens_real.py`'s in-glass gaps

Branch `audit-fixes-2026-09`, base HEAD `2680c24e` (VERIFY-B3).  Scope:
WP-B3's report §5.1 (a) and (b), §5.2, and VERIFY-B3's F6.  Date of every
measurement below: **2026-09-13**, this host, every Python run under
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process
at a time.

---

## 1. Summary

| # | item | status | files : lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|---|
| 1 | **K6 §5.1(a)** — the chain's `'fresnel'` leg evaluates the Fresnel integral straight onto the chain grid | **landed; the leg's numbers move — see §5** | `system.py:53`, `:806–841`; `:359–383`, `:160–173`, `:452–459`, `:1638–1646`, `:1771–1783` (the diagnostics and docs that named the old shape) | `TestK6FresnelLegEvaluatesOntoTheChainGrid` (12) | the Fresnel integral written out as an explicit double sum — no FFT, no Bluestein, no library call | relative L2 against that oracle **1.0435e-4 … 4.2615e-1 → 5.29e-16 … 5.04e-14**; window power now equals the oracle's to six digits on every fixture |
| 2 | **K6 §5.1(b)** — the chain's `'sas'` leg keeps `resample_field` and gates the method | **landed** | `system.py:861–886` | `TestK6TheChirpZGate` (11), `TestK6ByteIdentity…` (7), `TestK6TheImprovement…` (2) | the direct Fresnel evaluation for the window power; the exactly-tiling field for the replica count | where the window fits: distance to the direct evaluation's window power **2.15e-4 → 6.0e-6** (N=256, z=5 mm); where it does not: **byte-identical** to the unconditional spline |
| 3 | **K6 §5.2** — `_lens_real._propagate_through_glass`'s two gap legs, same gate | **landed** | `_lens_real.py:2767–2795` (sas), `:2799–2812` (fresnel) | `TestK6TheChirpZGate` (5 of the 11) | as above | the in-glass bias measured: on the WP-A15a covering-array doublet **2 of 2** gaps converge (`dx_new/dx` = 4.218e-3 and 1.086e-3), **16 of 16** over a thickness sweep; crossover at a **2.14 m** gap |
| 4 | **VERIFY-B3 F6** — `resample_field`'s docstring says what the unit MTF is a property OF | **landed (docstring only; the module fingerprint did not move)** | `mft.py:605–623` | `TestF6TheUnitMtfIsAPropertyOfTheWindow` (7) | Parseval on the input itself | exact-period window **1.1e-16 … 6.7e-16** from 1; a rounded window **0.993922** (×1.25) on a rim-filling field against **0.999993** on a contained one |

Everything whose numbers did not move is proved byte-identical
archive-to-archive, not asserted — §4.  **33 of 33** arrays that must not
move are identical; **12** move, every one of them named in §5 and every
one of them a leg this work package was asked to change.

Two things did not survive contact with measurement in the shape the
briefs described them:

* WP-B3 §5.1(b)/§5.2 spelled the gate `dx_new >= dx`.  VERIFY-B3 was
  right that this is only the `N_out == N_in` special case, and the
  shipped gate is the general **window against period** test.  It is
  also the *per-axis* test: `resample_field` reads ONE input pitch for
  both axes, so on a non-square input the shorter extent sets the
  period, which a pitch comparison cannot see at all (§2.2).
* VERIFY-B3's caveat that "a contained field in the converging direction
  is fine under chirp-Z" is fixture-dependent and is not a general
  licence.  On my contained Gaussian it holds at `dx_new/dx = 0.7727`
  (1.000000, matching the direct evaluation) and has already failed by
  `0.3091`, where the chirp-Z leg returns **9.000535** — a 3×3 tiling —
  while the spline returns 0.999999 (§2.2).

---

## 2. Per item

### 2.1 K6 §5.1(a) — the `'fresnel'` leg

**What landed.**  `system.py:838`:

```python
                E = fresnel_propagate_mft(
                    E, z, wavelength, current_dx, current_dx,
                    int(E_in.shape[-1]), dy_in=current_dy,
                    dy_out=current_dx)
```

replacing `fresnel_propagate` onto its natural grid `lambda z/(N dx)`
followed by `resample_field` back onto `N*current_dx`, and deleting that
block's `_warn_system_resample_crop` call.  `fresnel_propagate_mft` is
imported at module scope (`system.py:53`) rather than inside the element
loop.

**The oracle.**  The leg's claim is absolute — *this is the Fresnel
integral sampled on the chain grid* — so it is refereed against the
integral itself, written out as an explicit double sum over the input
samples with no FFT, no Bluestein and no library import (two dense matrix
products, since the quadratic kernel is separable).  That oracle's own
floor is the f64 accumulation of `N_in` terms per output sample,
`eps*sqrt(N_in)` ≈ 1.8e-15 at N = 64.

| chain fixture | relL2 vs the double sum, `2680c24e` | after |
|---|---|---|
| square 24×24, z = 1 mm | 2.047281e-1 | **9.075287e-16** |
| square 32×32, z = 2 mm | 2.451674e-1 | **5.288263e-16** |
| square 64×64, z = 1 mm | 1.043497e-4 | **3.474020e-15** |
| NON-square 24×18, z = 1 mm | 4.261454e-1 | **7.636427e-16** |
| NON-square 64×48, z = 1 mm | 2.457938e-1 | **3.609139e-15** |

The two non-square rows are a correctness result, not a precision one.
`fresnel_propagate` returns distinct `dx_new` and `dy_new`
(`lambda z/(Nx dx)` vs `lambda z/(Ny dy)`); `resample_field` takes a
single input pitch, so the chain resampled the y axis with the **x**
ratio and the y scale came out wrong by `Nx/Ny`.  The direct evaluation
takes `dy_in` and `dy_out` separately and has no such axis to confuse.

**Cost.**  Medians of seven interleaved runs of a one-element chain,
Gaussian input, dx = 2 µm, z = 5 mm (ms): 1.910 → 0.786 (N = 64),
4.085 → 9.733 (128), 22.318 → 19.524 (256), 87.621 → 106.013 (512).  The
within-cell spread is the same size as the differences (the N = 128
baseline itself ranged 3.308–6.860 ms) — three engineers shared this box
throughout, and **this measurement does not separate the two
implementations**.  What it does establish is that the change is not a
decade in either direction.  No test in this work package reads a clock.

### 2.2 K6 §5.1(b) and §5.2 — the gate

**The rule, in the form that ships.**  `system.py:880`:

```python
                    window_out = int(E_in.shape[-1]) * current_dx
                    period_in = min(E.shape[-2], E.shape[-1]) * dx_new
                    E, _ = resample_field(
                        E, dx_new, current_dx, N_out=E_in.shape[-1],
                        method=('chirpz'
                                if window_out <= period_in * (1.0 + 1e-9)
                                else 'spline'))
```

and the same comparison inline at `_lens_real.py:2789` and `:2806`.
Three properties, each deliberate:

1. It is `N_out*dx_out <= N_in*dx_in`, the general form VERIFY-B3 §5
   derived, not `dx_new >= current_dx`.  The two agree today only
   because `N_out == N_in` at all three sites.
2. It is **per axis**: `resample_field` applies one `dx_in` to both
   axes, so on a non-square input the shorter extent is the binding
   period — hence `min(E.shape[-2], E.shape[-1])`.
3. The `1e-9` slack is `_warn_mft_output_window`'s own tolerance, so the
   chirp-Z leg is selected on exactly the windows that resampler would
   NOT warn about.  Wired the other way round the chain would emit a
   faithful-zone warning on a path it had itself chosen.

**Why it must be a gate and not an unconditional switch** — the chirp-Z
reconstruction is periodic with period `N_in*dx_in`, and a field that
fills its grid tiles exactly.  Measured on a band-limited random field,
N = 64, with the window set to m periods:

| window | chirp-Z `P/P_in` | spline `P/P_in` | warns |
|---|---|---|---|
| 1 × period | 1.000000 | 1.000000 | no |
| 2 × period | **4.000000** | 0.974573 | yes |
| 3 × period | **9.000000** | 0.969247 | yes |

and on the real `'sas'` leg (grid-filling top-hat of radius 0.42·N·dx,
λ = 633 nm, dx = 2 µm; `direct` = `fresnel_propagate_mft` onto the chain
grid):

| fixture | `dx_new/dx` | gate | spline `P/P_in` | chirp-Z `P/P_in` | direct |
|---|---|---|---|---|---|
| N = 256, z = 5 mm | 1.5454 | **chirpz** | 0.986730 | **0.986951** | 0.986945 |
| N = 64, z = 1 mm | 1.2363 | **chirpz** | 0.961488 | **0.962304** | 0.962222 |
| N = 512, z = 5 mm | 0.7727 | spline | 0.950689 | **1.378837** | 0.996992 |
| N = 512, z = 2 mm † | 0.3091 | spline | 0.174014 | **1.820005** | — |
| N = 256, z = 1 mm † | 0.3091 | spline | 0.173601 | **1.835050** | — |

† `z` below the K1 chirp-sampling bound `N dx^2/lambda` (z/z_crit = 0.62
on both rows), so the direct evaluation is itself an aliased quadrature
there and is not quoted as a reference.  The chirp-Z replica reading does
not depend on that: it is a property of the window.

**The gate is conservative, and the cost is not zero.**  Two rows on a
contained Gaussian (w0 = 0.06·N·dx), which the period test also sends to
the spline:

| fixture | `dx_new/dx` | window/period | spline | chirp-Z | direct |
|---|---|---|---|---|---|
| N = 512, z = 5 mm | 0.7727 | 1.2942 | 1.000000 | 1.000000 | 1.000000 |
| N = 256, z = 2 mm | 0.6182 | 1.6177 | 1.000000 | 1.000594 | 1.000000 |

At 0.7727 the chirp-Z leg would indeed have been fine and the gate costs
it the unit MTF for nothing — VERIFY-B3's caveat, reproduced.  By 0.6182
it has already begun to accumulate the replica (1.000594 against the
direct 1.000000), and on the grid-filling top-hat at the *same* 0.7727 it
is 1.378837.  An automatic gate cannot tell a contained field from a
grid-filling one without measuring the field, so it takes the window
test; a caller who knows their field is contained can still call
`resample_field(method='chirpz')` directly.

**The in-glass bias (§5.2).**  The gap legs propagate in glass, so
`lam_medium = wavelength/n` makes `dx_new = lam_medium*t/(N*dx)` smaller
by `n` than the same geometry in air.  The WP-A15a covering array does
**not** itself reach these legs — its `propagator` factor is
`[{}, {'wave_propagator': 'rs'}]` (`test_audit2609_a15a_lens_covering_array.py:174`),
so nothing in the shipped array selects `'sas'` or `'fresnel'`.  Driving
that array's own fixture (its `curved_rear_doublet`, N = 64,
dx = 1.2·6 mm/64 = 112.500 µm, λ = 632.8 nm) through the two gap
propagators:

| gap | n | t | `dx_new` | `dx_new/dx` | same gap in air | gate |
|---|---|---|---|---|---|---|
| N-BAF10 | 1.6671 | 9.00 mm | 474.482 nm | **0.004218** | 0.007031 | spline |
| N-SF6HT | 1.7988 | 2.50 mm | 122.148 nm | **0.001086** | 0.001953 | spline |

**2 of 2** converge, and **16 of 16** over `t ∈ {0.5, 1, 2.5, 5, 9, 20,
50, 100} mm` on both glasses: on that grid the crossover is a
`t = N dx^2/lam_medium` = **2137.6 mm** gap.  The direction is not
universal, though — the other gap fixture in the verification set
(`_minimal_prescription`: a 1 mm N-BK7 plate at N = 64, dx = 2 µm, so a
128 µm window) sits at `dx_new/dx = 1.6320` and takes the chirp-Z leg.
Both directions occur in the shipped test suite, which is why the gate is
the right shape here.

### 2.3 VERIFY-B3 F6 — `resample_field`'s docstring

F6 observed that the chirp-Z MTF "reads 0.99996 rather than 1.000000"
whenever the extent-preserving `N_out` rounds off the period, and asked
the docstring to say at which scale factors it is exact.  Re-measured
rather than read: the departure is not an MTF at all, it is the field in
the sliver the rounding adds or drops, so it is a property of the window
AND of the fixture.  N_in = 128, carrier 0.30 cyc/px, default `N_out`:

| scale | `N_out` | window/period | contained (w0 = 0.18 N dx) | rim-filling (w0 = 0.45 N dx) |
|---|---|---|---|---|
| 0.50 | 256 | 1.000000 (exact) | 1.000000 | 1.000000 |
| 0.70 | 183 | 1.000781 | 1.000000 | 1.000238 |
| 1.00 | 128 | 1.000000 (exact) | 1.000000 | 1.000000 |
| 1.25 | 102 | 0.996094 | 0.999993 | **0.993922** |
| 1.50 | 85 | 0.996094 | 1.000000 | 0.998015 |
| 1.70 | 75 | 0.996094 | 1.000000 | 0.998489 |
| 2.00 | 64 | 1.000000 (exact) | 1.000000 | 0.999907 |
| 3.00 | 43 | 1.007812 | 1.000000 | 1.004048 |

(The rim-filling envelope carries 4.370e-2 of its power outside
0.49·N·dx; the contained one 3.079e-7.)  With an explicit exact-period
`N_out` the reading is 1 to **-1.1e-16 … +6.7e-16** at every scale that
does not down-sample.  At ×2 and ×4 an exact window still reads
-3.3e-9 (contained) and -9.3e-5 … -4.6e-4 (rim-filling): there the
0.30 cyc/px carrier is above the new Nyquist — measured out-of-band share
0.9977 … 1.0000 of the input's own spectral power — and what is left is
the interference between folded components, which the docstring already
covers under "neither leg anti-aliases on DOWN-sampling".

The docstring (`mft.py:605–623`) now states the condition
(`N_out*dx_out == N_in*dx_in`), which scale factors the extent-preserving
default satisfies (`N_in*dx_in/dx_out` whole — ×0.5, ×1, ×2, ×4 at any
`N_in`; ×1.5 needs `N_in` divisible by 3, ×1.25 by 5, ×1.7 by 17), and
the measured departures above.  `docs/history/lumenairy.propagators.mft.md`
did **not** need re-recording: `--check` reports it OK, which is the
fingerprint gate confirming a docstring-only edit.

---

## 3. The fixture census — measured, not guessed

Before any edit, the whole verification set was run under a scratch
pytest plugin that records every `resample_field` call made from the four
legs, with the calling frame's locals.  The entire set makes **ten** such
calls:

| leg | calls | N | `dx_new/dx` | gate would pick | tests |
|---|---|---|---|---|---|
| `system` / `fresnel` | 3 | 64 | 2.4727 | (leg retired) | `test_niche_audit_w3_propagators.py::TestNewSystemMethodValidation::test_honoured_methods_still_run[fresnel]`, `::test_per_element_override_still_honoured`, `test_v5_1_0_agent_a.py::TestSetDefaultWavePropagatorSteersPropagateThroughSystem::test_set_default_to_fresnel_changes_system_output` |
| `system` / `sas` | 1 | 64 | 1.2363 | chirpz | `test_niche_audit_w3_propagators.py::TestNewSystemMethodValidation::test_honoured_methods_still_run[sas]` |
| gap / `fresnel` | 4 | 64 | 1.6320 | chirpz | `test_v5_1_0_agent_a.py::TestSetDefaultWavePropagatorSteersApplyRealLens::test_set_default_to_fresnel_changes_apply_real_lens_output` |
| gap / `sas` | 2 | 64 | 0.1813 | spline | `test_audit2609_a2_analytic_lens.py::TestSasRefusesAnAnamorphicPitch::test_square_sas_still_runs` |

Three readings worth recording:

* every `'fresnel'` chain call in the verification set resampled, so §5's
  Migration note applies to all three;
* the covering array reaches neither gap leg (§2.2), which is why its
  direction question had to be answered by driving its fixture
  deliberately;
* the census is small.  That is a statement about test coverage of these
  legs, not about their importance: `set_default_wave_propagator('fresnel')`
  routes every kwargless `apply_real_lens` and `propagate_through_system`
  call in a user's session through them.

---

## 4. Byte-identity proofs

Archive-to-archive, per VERIFY-B3 §4.1.  `git archive 2680c24e lumenairy`
extracted twice into this session's scratch directory; the second copy
then received **only** my three edited files.  Each probe runs in its own
child process with `cwd == PYTHONPATH ==` that tree and prints
`lumenairy.__file__` before any measurement.  Nothing goes through
pytest, and nothing reads the shared working tree — which today carries
uncommitted `raytrace/*`, `rcwa/*`, `analysis/*`, `sources/*`,
`carrier.py` and `_lens_traced.py` edits belonging to other engineers.

**33 arrays byte-identical out of the 33 that must not move; 133 of 144
text records identical.**

| surface | cases | result |
|---|---|---|
| `resample_field`, default (spline) leg | 9 `(N, dx_in, dx_out, N_out, order)` combinations: odd 65, the exact no-op, `order` 0/1/3/5, a non-square 24×18 at the default `N_out`, an odd anamorphic 33×21 → 45 | **identical** |
| `resample_field(method='chirpz')` | 5, including the 2×-period window that warns | **identical** |
| `fresnel_propagate`, `fresnel_propagate_mft`, `angular_spectrum_propagate_mft`, `fraunhofer_propagate_mft`, `scalable_angular_spectrum_propagate` | natural and warning-triggering windows | **identical** |
| `propagate_through_system(method='asm')` | bare, `bandlimit=False`, a three-element chain, a lens+aperture chain, a tilted `'fresnel'` element (which routes to tilted ASM) | **identical** |
| `propagate_through_system(method='sas')` | the two converging fixtures (N = 512 z = 5 mm, N = 256 z = 1 mm) where the gate picks the spline | **identical** |
| `propagate_through_system_jax(method='asm')` | one chain | **identical** |
| `apply_real_lens` | `wave_propagator='asm'`, `'rs'`; the two converging `'sas'` gaps; the converging `'fresnel'` gap (the covering-array doublet) | **identical** |
| guard and error paths | junk `method`, `'rs'`, anamorphic pitch on both legs, SAS's non-square refusal | **identical text** |

The same claim is restated build-free inside the suite
(`TestK6ByteIdentityWhereTheGateSelectsTheSpline`): forcing
`method='spline'` back in process reconstructs the pre-gate library
exactly, the gated chain must equal it bit for bit on a converging
fixture, and — the arm that stops it being vacuous — must **differ** on a
diverging one.

---

## 5. What moved, and the Migration note

Twelve probe arrays differ, all of them a leg this work package was asked
to change:

| probe | relL2 vs `2680c24e` | why |
|---|---|---|
| `sys_fresnel_64` (E = ones, N = 64, z = 1 mm — the census geometry) | 9.2432e-2 | direct evaluation |
| `sys_fresnel_256` (top-hat, z = 5 mm) | 4.3589e-2 | direct evaluation |
| `sys_fresnel_512` (top-hat, z = 5 mm) | 2.8108e-2 | direct evaluation |
| `sys_fresnel_64g` (contained Gaussian) | 1.0435e-4 | direct evaluation |
| `sys_fresnel_c64` (complex64, dtype preserved) | 1.0434e-4 | direct evaluation |
| `sys_fresnel_chain` (propagate–lens–propagate) | 2.5246e-4 | direct evaluation |
| `sys_fresnel_nonsquare` (64×48) | 2.4579e-1 | the y axis is no longer resampled with the x ratio |
| `sys_fresnel_talbot` (z chosen so the natural grid EQUALS the chain grid) | **7.0931e-15** | the old leg did not resample at all here; this is Bluestein vs plain FFT |
| `sys_fresnel_undersampled` (z = 0.247 × the K1 bound) | 9.1977e-1 | both readings are aliased quadratures; both warn |
| `sys_sas_diverging` (N = 64, z = 1 mm) | 1.8603e-2 | the gate selects chirp-Z |
| `lens_fresnel_plate` (E = ones) | 4.6375e-2 | the gate selects chirp-Z |
| `lens_fresnel_plate_g` (Gaussian) | 2.2918e-2 | the gate selects chirp-Z |

**The regression pass on the `'fresnel'` leg** (every fixture in the
verification set that exercises it, plus the WP-B3 §5.1 fixtures),
against the double-sum oracle:

| fixture | relL2 before | relL2 after | `P/P_in` before | after | oracle |
|---|---|---|---|---|---|
| census geometry, E = ones, N = 64, z = 1 mm | 9.243220e-2 | **6.675366e-15** | 0.917862 | **0.911783** | 0.911783 |
| contained Gaussian, N = 64, z = 1 mm | 1.043497e-4 | **3.474020e-15** | 0.999962 | **0.999999** | 0.999999 |
| grid-filling top-hat, N = 256, z = 5 mm | 4.358887e-2 | **1.285827e-14** | 0.986005 | **0.986945** | 0.986945 |
| grid-filling top-hat, N = 512, z = 5 mm | 2.810767e-2 | **5.038972e-14** | 0.996685 | **0.996992** | 0.996992 |
| grid-filling top-hat, N = 256, z = 2 mm | 4.859079e-2 | **4.300800e-14** | 0.995421 | **0.996072** | 0.996072 |

The in-glass `'fresnel'` gap on the census fixture (1 mm N-BK7 plate,
N = 64, dx = 2 µm, gate → chirp-Z) moves `P_out/P_in`
0.116048 → 0.116696 (E = ones) and 0.898010 → 0.899250 (Gaussian); that
difference is the interpolator MTF the band-limited leg does not pay.

**Eleven text records differ**, all of them the `'fresnel'` leg's
diagnostics:

* the `_warn_system_resample_crop` `RuntimeWarning` no longer fires from
  this leg on six probes (there is no resample to crop).  It still fires
  from `'sas'`, unchanged;
* the K1 under-sampled-chirp `RuntimeWarning` is now prefixed
  `fresnel_propagate_mft:` instead of `fresnel_propagate:`.  The bound,
  the numbers and the advice are identical;
* `z <= 0` on a `'fresnel'` element still raises `ValueError`, now with
  `fresnel_propagate_mft`'s message (it points at
  `angular_spectrum_propagate_mft` rather than
  `angular_spectrum_propagate`);
* `propagate_through_system_jax`'s `NotImplementedError` for
  `method='fresnel'`/`'sas'` was reworded (`system.py:1771–1783`) because
  its old text said both NumPy branches resample through
  `map_coordinates`, which is now true only of `'sas'`.

### Migration

**No keyword default moved.**  `resample_field`'s own default is still
`'spline'`, the chain's `method` default is still `'asm'`, and
`apply_real_lens`'s `wave_propagator` default is unchanged.  What moves
is the numerical output of three legs:

1. **Every `propagate_through_system(..., method='fresnel')` step, and
   every per-element `{'method': 'fresnel'}`, returns different numbers.**
   They are the Fresnel integral evaluated on the chain grid — verified
   against an explicit double sum to 5.3e-16 … 5.0e-14 — where before
   they were that integral evaluated on the single-FFT natural grid and
   then cubic-interpolated back, which both cropped the field outside
   `N*dx` and paid the interpolator's MTF.  Size of the move: 7.1e-15
   where the natural grid already equalled the chain grid, ~1e-4 on a
   contained field, ~1e-1 on a grid-filling one, and 2.5e-1 on a
   non-square sample count (where the old path scaled the y axis by the
   x ratio and was simply wrong).  A caller with a pinned `'fresnel'`
   chain result must re-baseline it.
2. **A `'fresnel'` chain step no longer emits the K6 crop warning**
   (`"the fresnel leg returned its natural output grid ... which CROPS
   it"`).  Code filtering on that text for the `'fresnel'` leg will stop
   matching; the `'sas'` leg's copy is unchanged.  The faithful-zone
   diagnostic that replaces it comes from `fresnel_propagate_mft`, with
   period `lambda*|z|/dx_in`.
3. **Two messages from a `'fresnel'` chain step are now named by
   `fresnel_propagate_mft`** rather than `fresnel_propagate`: the K1
   under-sampled-chirp warning and the `z <= 0` refusal.
4. **`propagate_through_system(method='sas')` and
   `apply_real_lens(wave_propagator='sas'|'fresnel')` move only where
   the requested window fits inside one chirp-Z reconstruction period**
   (in practice: where the pitch coarsens).  Everywhere else they are
   bit-identical to 5.46.0 — proved, not asserted (§4).  Where they do
   move, the resample-back gains a unit MTF.

---

## 6. Files touched

| file | what |
|---|---|
| `lumenairy/propagators/system.py` | the `'fresnel'` leg's direct MFT evaluation (`:806–841`) and its module-scope import (`:53`); the `'sas'` leg's gate (`:861–886`); `_require_square_pitch` and `_warn_system_resample_crop` docstrings rewritten to describe the legs as they now are (`:160–173`, `:359–383`); the `method` parameter docs (`:452–459`); the JAX twin's docstring and refusal message (`:1638–1646`, `:1771–1783`) |
| `lumenairy/elements/_lens_real.py` | `_propagate_through_glass`'s two `resample_field` calls gated (`:2767–2795` sas, `:2799–2812` fresnel).  Nothing else in the file was opened for edit |
| `lumenairy/propagators/mft.py` | `resample_field`'s docstring only — the F6 paragraph (`:605–623`).  The history fingerprint did not move |
| `docs/history/lumenairy.propagators.system.md` | re-recorded in this change (`--reason` names both legs) |
| `docs/history/lumenairy.elements._lens_real.md` | re-recorded in this change (`--reason` names the gate and the in-glass bias) |
| `tests/unit/test_audit2609_b3b_resample_call_sites.py` | **new**, 39 tests |
| `docs/audits/.../fixes/WP-B3b_REPORT.md`, `WP-B3b_CHANGELOG.md` | this report and its release text |

No file outside this list was modified.  `mft.py`'s code, `fresnel.py`,
`propagation.py`, `dispatch.py`, `hfpi.py` and everything under
`raytrace/`, `rcwa/`, `pmm/`, `analysis/`, `sources/` were read only.

---

## 7. Tests run

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a5_*.py tests/unit/test_audit2609_b3_propagator_kernels.py tests/unit/test_audit2609_a2_*.py tests/unit/test_audit2609_b2_displaced_remap_inversion.py tests/unit/test_audit2609_a15a_lens_covering_array.py tests/unit/test_audit2609_a17_history_lint.py tests/unit/test_audit2609_b3b_resample_call_sites.py` | **427 passed** | 192.87 s; 206.46 s on the final tree |
| `pytest tests/unit -k "system or propagate_through or fresnel or sas"` | **521 passed, 3 skipped**, 15 389 deselected | 161.41 s |
| `pytest tests/unit -k real_lens` | **160 passed, 3 skipped**, 15 750 deselected | 279.42 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py` | 737 passed, **2 failed** — `lumenairy/raytrace/ray_fan.py`, another package's (§7.2) | 41.84 s |
| `python validation/run_all.py test_propagation test_lenses test_dispatch` | **ALL 3 passed** (27.9 / 2.0 / 8.2 s) | — |
| `python -m ruff check` (whole tree) | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | `lumenairy.propagators.system`, `lumenairy.elements._lens_real`, `lumenairy.propagators.mft` all **OK**; drift elsewhere is `analysis/psf_mtf_otf` and `sources/core`, neither mine | — |
| `pytest tests/unit/test_audit2609_b3b_resample_call_sites.py` alone | **39 passed** (3.33 s pinned order, 3.78 s under the default random order — order-robust) | — |
| `pytest tests/unit/test_audit2609_a15a_durations_staleness.py` | 3 passed, **1 failed** — `.test_durations`, outside my ownership (§7.1) | 28.16 s |
| the pre-change census runs (3 × pytest under the scratch plugin) | 383 / 495+3s / 158+3s passed, plus the 2 `_lens_traced` failures that their owner has since fixed | 190.4 / 153.3 / 262.9 s |
| byte-identity probe pair + double-sum oracle + regression pass + cost ladder, archive-to-archive | as tabulated in §2, §4, §5 | ~30–90 s each |

Every duration is indicative only: three engineers shared this box
throughout, and no test asserts one.

### 7.1 The `.test_durations` lines my tests need (outside my ownership)

`test_audit2609_a15a_durations_staleness` currently reads **327 of 15 924
collected ids (2.05 %)** against a 2 % bar.  My 39 ids are 12 % of that
missing set, and today they are decisive: without them the reading is
288 of 15 885 = **1.81 %**, i.e. green.  The other contributors are
`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` (233),
`test_audit2609_b8_analysis_sources.py` (18),
`test_fix_newton_pool_memory.py` (15),
`test_audit2609_b9_raytrace_perf.py` (14),
`test_audit2609_b6_pmm_basis_and_tensor_cache.py` (7) and
`test_niche_audit_w8_shapes.py` (1) — none of them mine either.  Adding
my 39 lines alone clears the gate as the tree stands.  The exact lines
(this host, one BLAS thread, serial; the 1.36 s on
`test_the_leg_reproduces_the_fresnel_double_sum[square_24]` is the
file's first-call FFT-plan warm-up landing on whichever parametrised
case collects first — the same body at N = 64 reads 0.00 s):

```json
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_a_downsample_at_an_exact_window_is_the_fold_not_the_gain[2.0]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_a_downsample_at_an_exact_window_is_the_fold_not_the_gain[4.0]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_a_rounded_window_departs_by_the_field_in_the_sliver": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_an_exact_period_window_returns_the_power_exactly[0.5]": 0.03,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_an_exact_period_window_returns_the_power_exactly[0.8]": 0.02,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_an_exact_period_window_returns_the_power_exactly[1.0]": 0.02,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestF6TheUnitMtfIsAPropertyOfTheWindow::test_the_docstring_states_the_condition": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_identical_bits_where_the_window_exceeds_the_period[gap_fresnel_doublet]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_identical_bits_where_the_window_exceeds_the_period[gap_sas_bk7_plate]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_identical_bits_where_the_window_exceeds_the_period[sys_sas_256_z1mm]": 0.10,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_identical_bits_where_the_window_exceeds_the_period[sys_sas_512_z5mm]": 0.45,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_the_comparison_is_not_vacuous_where_the_window_fits[gap_fresnel_bk7_plate]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_the_comparison_is_not_vacuous_where_the_window_fits[sys_sas_64_z1mm]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6ByteIdentityWhereTheGateSelectsTheSpline::test_the_default_resampler_leg_is_untouched": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_a_non_positive_z_is_still_refused[-0.001]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_a_non_positive_z_is_still_refused[0.0]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_an_anamorphic_working_pitch_is_still_refused": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_keeps_the_chain_pitch_and_the_input_sample_count": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_performs_no_resample": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_reproduces_the_fresnel_double_sum[nonsquare_24x18]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_reproduces_the_fresnel_double_sum[nonsquare_64x48]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_reproduces_the_fresnel_double_sum[square_24]": 1.36,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_reproduces_the_fresnel_double_sum[square_32_z2mm]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_leg_reproduces_the_fresnel_double_sum[square_64]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_resample_back_is_the_thing_the_bar_separates_from": 0.04,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6FresnelLegEvaluatesOntoTheChainGrid::test_the_undersampled_chirp_guard_still_fires_from_this_leg": 0.05,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_an_ungated_chirpz_returns_the_tiling_power": 0.07,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_covering_array_doublet_lands_in_the_spline_half": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_gate_is_written_in_the_window_period_form": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_in_glass_gap_legs_gate_on_window_against_period[doublet_wide_grid_fresnel]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_in_glass_gap_legs_gate_on_window_against_period[doublet_wide_grid_sas]": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_in_glass_gap_legs_gate_on_window_against_period[plate_fine_grid_fresnel]": 0.58,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_in_glass_gap_legs_gate_on_window_against_period[plate_fine_grid_sas]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_system_sas_leg_gates_on_window_against_period[sas_coarsens_256]": 0.10,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_system_sas_leg_gates_on_window_against_period[sas_coarsens_64]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_system_sas_leg_gates_on_window_against_period[sas_refines_256]": 0.10,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheChirpZGate::test_the_system_sas_leg_gates_on_window_against_period[sas_refines_512]": 0.39,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheImprovementWhereTheGateSelectsChirpZ::test_chirpz_is_the_better_interpolant_on_a_contained_field": 0.07,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestK6TheImprovementWhereTheGateSelectsChirpZ::test_chirpz_tracks_the_direct_evaluation_better_than_the_spline": 0.32,
```

### 7.2 Every failure is another work package's, named

* `test_audit2609_a17_history_relocation` ×2 — AST and token drift on
  `lumenairy/raytrace/ray_fan.py`, a module I never opened.  My three
  (`system`, `_lens_real`, `mft`) are OK, the first two because they were
  re-recorded in this change and `mft` because a docstring edit does not
  move the fingerprint.  `--check` additionally showed
  `lumenairy/analysis/psf_mtf_otf.py` and `lumenairy/sources/core.py`
  drifting an hour earlier; the set moves as their owners re-record.
* `test_audit2609_a15a_durations_staleness` ×1 — §7.1.
* The pre-change census run of `-k real_lens` also showed
  `test_audit2609_a16_lens_config_round_trip` ×2 failing on
  `apply_real_lens_traced` / `prepare_real_lens_traced`
  (`lumenairy/elements/_lens_traced.py`).  Both were green by my
  post-change run of the same selection — their owner fixed them
  in between.

---

## 8. Requested changes outside my ownership

1. **`.test_durations`** — the 39 lines in §7.1.  Adding them alone
   clears `test_audit2609_a15a_durations_staleness` as the tree stands.
2. **Nothing is requested of `mft.py`'s code.**  The F6 edit was a
   docstring; `resample_field`'s chirp-Z leg, `_resample_field_chirpz`
   and `_warn_mft_output_window` all behave exactly as WP-B3 shipped
   them, and all three are in this report's byte-identity set.

Checked rather than assumed, and requested of nobody:

* `fresnel_propagate_mft` already carries the K1 chirp-sampling guard
  (`mft.py:965–967`), so retiring `fresnel_propagate` from this leg costs
  the chain no diagnostic — only the function name in the prefix.
* `dispatch.py` has no allow-list on the way to `propagate_through_system`,
  so nothing there needed touching.
* `propagate_through_system_jax` rejects `'fresnel'` before reaching any
  of this, and is byte-identical on `'asm'`.

---

## 9. Deferred / observed, not fixed

| # | severity | item |
|---|---|---|
| **D1** | P3 | The chain's `'fresnel'` leg does not forward `use_gpu` to `fresnel_propagate_mft`, exactly as it did not forward it to `fresnel_propagate`.  The MFT routine takes the flag and auto-detects CuPy/JAX arrays, so wiring it would be a one-line behaviour change — and untestable here (no CuPy).  Left at parity deliberately. |
| **D2** | P3 | `resample_field`'s `N_out` is a single integer, so the `'sas'` leg cannot preserve a non-square sample count through its resample-back.  It does not need to today: `scalable_angular_spectrum_propagate` refuses a non-square input outright (`"input must be square (got 48x64)"`, byte-identical in both trees).  The `'fresnel'` leg now accepts a non-square input and returns `(N, N)` with `N = E_in.shape[-1]`, which is what the old leg returned whenever it resampled. |
| **D3** | P3 | The gate is conservative on a *contained* field just past one period (§2.2): it pays the spline's MTF where the chirp-Z leg would have been exact.  Closing that needs a containment measurement on the field, not a geometry test, and a measurement that can be fooled is worse than a conservative rule.  Recorded so the next reader does not re-derive it. |
| **D4** | P4 | `_warn_system_resample_crop` now has a single caller.  Inlining it would lose a docstring carrying the K6 decomposition; left as it is. |
| **D5** | P3 | Three of the ten resample-backs in the whole verification set come from one `'fresnel'` chain geometry, and the covering array reaches neither gap leg (§3).  These legs are thinly covered by fixtures relative to how easily `set_default_wave_propagator` routes user code through them.  The new file adds direction coverage for both; a covering-array `propagator` level of `{'wave_propagator': 'fresnel'}` would add it where the rest of the lens matrix already is — that file is WP-A15a's. |

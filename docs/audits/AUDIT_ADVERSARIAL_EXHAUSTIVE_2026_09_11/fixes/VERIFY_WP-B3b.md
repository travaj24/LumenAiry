# VERIFY-WP-B3b — independent adversarial re-verification of WP-B3b

Package under test: commit **908c02d6** on `audit-fixes-2026-09`
(`fix(propagators): WP-B3b -- the chain's Fresnel leg evaluates the Fresnel
integral straight onto the chain grid; the SAS and in-glass resample-back
legs gate chirp-Z on window against period (K6 call sites)`).  Pre-change
reference: **908c02d6^** = `12773c1d`.

Every measurement below was taken on **2026-09-13**, this host, in a child
process under `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
one process at a time.  Every numerical comparison is **archive-to-archive**:
`git archive 908c02d6^ lumenairy` and `git archive 908c02d6 lumenairy`
extracted into
`…/scratchpad/verify_b3b/{before,after}/`, each imported from a child process
whose `cwd` and `PYTHONPATH` are that archive, with `lumenairy.__file__`
asserted to lie under it before any measurement.  **No** numerical measurement
in this document read the shared working tree, which carried other engineers'
uncommitted `_lens_real.py`, `carrier.py`, `lenses_maslov.py`,
`_lens_traced.py`, `lens_config.py`, `rcwa/*` and `analysis/*` edits
throughout this session, and nothing went through pytest.  (§9's suite runs
are the only thing that ran against the working tree, and they are a
green/red gate, not a measurement.)  The two archives differ in exactly
**three** library files (`propagators/system.py`, `elements/_lens_real.py`,
`propagators/mft.py`), which is the whole change surface.  A third tree,
`fixed` = the `after` archive plus my two edited library files
(`propagators/system.py`, `propagators/mft.py`), is what §5's
value-neutrality claim is measured against.

I did not write WP-B3b.  I wrote my own oracles and my own fixtures, and
where the report states a number I re-measured it rather than read it.

---

## 1. Verdict table

| # | claim (WP-B3b report / changelog / source comment) | verdict | my oracle | my numbers |
|---|---|---|---|---|
| 1 | **K6 §5.1(a)** — the chain's `'fresnel'` leg evaluates the Fresnel integral straight onto the chain grid; relL2 against a double sum `5.3e-16 … 5.0e-14` where the resample-back read `1.0e-4 … 4.3e-1` | **VERIFIED** | the Fresnel integral as **four literal nested Python loops**, itself refereeing a dense two-matrix-product form (the two agree to **2.9e-16** at N = 9) | on 16 fixtures the engineer did not use: **8.58e-16 … 4.35e-14** after on the 14 f64 discrete-sum rows, against **1.02e-14 … 8.66e-1** before; window power `P_out/P_oracle` **1.000000** on every one.  The two rows outside that range are the f32 floor (complex64, 1.75e-7) and the continuous-quadrature reference (6.25e-9), both expected (§2.1) |
| 2 | the leg's `dy_in`/`dy_out` handling — "the old leg scaled y by the x ratio" | **VERIFIED, and worse than reported in the tall orientation** | same | a **24-wide × 36-tall** input read relL2 **4.96e-1** with `P_out/P_oracle` = **1.166644** before — the old leg *gained* power; 32 wide × 64 tall read **8.24e-1** / **1.733361**.  The report's two non-square rows are both *wide*; the tall orientation is the one that inflates.  All four non-square fixtures land at **1e-15** after (§2.1) |
| 3 | **K6 §5.1(b)** — the `'sas'` leg keeps `resample_field` and gates `method`; where the window fits, the distance to the direct evaluation's window power drops **2.15e-4 → 6.0e-6** | **VERIFIED-WITH-NOTES** | `fresnel_propagate_mft` onto the chain grid, plus `angular_spectrum_propagate` at the chain pitch as a second, kernel-independent reference | reproduced to the digit (0.986730 / 0.986951 against 0.986945).  **Note:** on that same fixture relL2 against the same reference moves the *other* way, 1.035e-2 → 1.058e-2 — the power reading improves, the field distance does not (§2.2) |
| 4 | the gate is **per axis**, `min(E.shape[-2], E.shape[-1])`, "because `resample_field` reads one input pitch for both" | **VERIFIED — and was UNPINNED** | a constructed non-square in-glass gap | the `min` is reachable at exactly one of the three sites (the in-glass `'fresnel'` leg); on a **32 × 64** grid at `dx_new/dx = 1.5` the per-axis rule picks `spline` and an x-only rule picks `chirpz`, where forcing chirp-Z makes `resample_field` warn **on y**.  Deleting the `min` left all 39 shipped tests green (§4) |
| 5 | the `1e-9` slack "is `_warn_mft_output_window`'s own tolerance, so the chirp-Z leg is chosen on exactly the windows it would not warn about" | **VERIFIED — and was UNPINNED** | `resample_field` driven at a window exactly one period wide, ± the slack | gate ⇔ no-warning on all five boundary rows (1 − 2e-9, 1, 1 + 5e-10 → chirpz/silent; 1 + 2e-9, 1 + 1e-6 → spline/warns).  Removing the slack left all 39 shipped tests green (§2.2, §4) |
| 6 | **K6 §5.2** — the in-glass legs: 2 of 2 gaps converge on the WP-A15a doublet (`dx_new/dx` 4.218e-3 and 1.086e-3); both directions occur | **VERIFIED-WITH-NOTES** | the library's own glass registry + the closed form `t_cross = N dx^2 n / lambda` | `dx_new/dx` = **0.004218** and **0.001086** to the digit; a thickness sweep crosses the gate in both directions for both legs and both glasses (§2.3).  **Note:** the report's crossover **2137.6 mm** recomputes to **2133.9 mm** from its own `n = 1.6671` (0.17 % out) |
| 7 | **byte identity** — "33 of 33 arrays that must not move are identical; 12 move and every one is a leg this package changed" | **VERIFIED** | my own 72-record probe (arrays + guard/warning texts), archive-to-archive | **64 of 72 arrays identical, 8 move** — 3 chain-`'fresnel'`, 1 chain-`'sas'`, 4 in-glass-`'fresnel'`, i.e. only the changed legs; `asm` (bare, un-band-limited, 3-element, lens+aperture, anamorphic, tilted element, tilted chain), `propagate_through_system_jax`, `apply_real_lens('rs'|'asm')`, the five MFT propagators, both `resample_field` legs and every converging gap are bit-for-bit identical.  **66 of 72 text records identical**, the 6 that differ are the four documented ones (§3) |
| 8 | **Migration note 2** — "A `'fresnel'` chain step no longer emits the K6 crop warning … The faithful-zone diagnostic **that replaces it** comes from `fresnel_propagate_mft`" | **REGRESSION — fixed here (V1)** | the two conditions written out on the chain grid, plus six measured fixtures | the two warnings are **disjoint**: on the chain grid the faithful-zone condition reduces to `z < N dx^2/lambda` (the K1 band) and a beam outgrows the chain window only at `z` **above** that bound.  Measured: a top-hat of radius 3 px at N = 64, `z` = 30× the bound returns **3.17 %** of its input power and 908c02d6 says **nothing** (908c02d6^ warned).  Restored as `_warn_system_fresnel_window` (§5.1) |
| 9 | **VERIFY-B3 F6** — `resample_field`'s docstring: "x0.5, x1, x2 and x4 at any `N_in`" | **NOT FIXED — corrected here (V2)** | the arithmetic `N_in/scale ∈ ℤ`, measured on nine `N_in` × eight scales | ×2 needs an **even** `N_in` and ×4 an `N_in` **divisible by 4**: at `N_in = 65` the ×2 default rounds to `N_out = 32`, a 64-`dx_in` window against a 65-`dx_in` period, measured power ratio **0.995181** (rim-filling) against **0.999907** at `N_in = 128`, where the same default *does* land on the period (§5.2) |
| 10 | **D3** — "the gate is conservative on a contained field just past one period … it pays the spline's MTF where the chirp-Z leg would have been exact" | **VERIFIED-WITH-NOTES** | ASM at the chain pitch (no resample) and the direct Fresnel evaluation | true only in the first sliver: at `dx_new/dx = 0.7727` chirp-Z is **5.5×** closer to ASM (1.81e-9 vs 9.93e-9) and both read `P = 1.000000`; by **0.6182** chirp-Z is **four decades worse** in relL2 (2.44e-2 vs 5.63e-6) while its power reads the reported 1.000594.  The cost the report calls "not free" is real over a narrow band and inverts immediately after (§2.2) |
| 11 | **D2** — "the `'fresnel'` leg now accepts a non-square input and returns `(N, N)` with `N = E_in.shape[-1]`, which is what the old leg returned whenever it resampled" | **VERIFIED** | shape probe both trees | 24 × 36 → 24 × 24 in both trees; correct against the oracle on that window after, 4.96e-1 away before |
| 12 | the leg's guard texts: K1 warning and `z <= 0` refusal re-prefixed `fresnel_propagate_mft`, the JAX refusal reworded, the anamorphic refusal unchanged | **VERIFIED** | text diff of 72 records | exactly those four, and nothing else; the `'sas'` crop warning still fires unchanged |
| 13 | `_lens_real._propagate_through_glass`'s `'fresnel'` branch, whose `resample_field` call this package edited | **DEFECT FOUND (V5) — requested, outside my ownership** | my double-sum oracle with `dy` handled separately | that branch accepts an **anamorphic pitch** and a **non-square grid** and resamples the y axis with the **x** ratio — relL2 **0.404**, `P` ratio **1.4999** (= `dy/dx`) — while its `'sas'` sibling refuses both outright.  This is the same defect WP-B3b removed from `system.py`, three functions away (§6) |

---

## 2. Per item — my oracles and my numbers

### 2.1 The `'fresnel'` leg against my own double sum

**The oracle.**  `oracle.fresnel_loops` writes the Fresnel integral as four
literal nested Python loops over `(ky, kx, m, n)` — no vectorisation, no FFT,
no Bluestein, no library import:

```
E_out[ky,kx] = e^{ikz}/(i lam z) * sum_{m,n} E_in[m,n]
               * exp(i k/(2z) [(x_out[kx]-x_in[n])^2 + (y_out[ky]-y_in[m])^2])
               * dx_in * dy_in
```

It is O(N⁴), so it runs at N = 9 (odd) and referees the dense
two-matrix-product form used for the larger fixtures: the two agree to
**2.9e-16**.  For the three-element chain the thin-lens phase is also mine
(`exp(-i k r^2/(2f))`); it is **bit-identical** to `apply_thin_lens`
(relL2 exactly 0.0), so the chain fixture's oracle is end-to-end independent
of the propagator under test.

**Results** (relL2 and `P_out/P_oracle`, both trees, same fixtures):

| fixture | relL2 `908c02d6^` | relL2 `908c02d6` | `P/P_oracle` before | after |
|---|---|---|---|---|
| odd N = 33, z = 1 mm | 2.8855e-2 | **1.7030e-15** | 0.985931 | **1.000000** |
| odd N = 65, z = 1 mm | 1.0485e-4 | **1.0186e-14** | 0.999967 | **1.000000** |
| non-square **24 wide × 36 tall** (longer side in **y**) | 4.9618e-1 | **8.5804e-16** | **1.166644** | **1.000000** |
| non-square **32 wide × 64 tall** (longer side in **y**) | 8.2443e-1 | **1.4240e-15** | **1.733361** | **1.000000** |
| non-square **21 wide × 33 tall**, both odd | 4.8959e-1 | **1.0397e-15** | 1.091688 | **1.000000** |
| non-square 36 wide × 24 tall (longer in x) | 4.4630e-1 | **1.2261e-15** | 0.652925 | **1.000000** |
| decentred hard-edged square, N = 64 (centre +18, −26 µm) | 1.0887e-1 | **4.0499e-15** | 0.990627 | **1.000000** |
| decentred hard-edged square, N = 128, z = 4 mm | 1.0322e-1 | **4.5698e-15** | 0.991368 | **1.000000** |
| decentred top-hat, odd N = 33 | 1.8540e-1 | **2.4036e-15** | 0.957572 | **1.000000** |
| **complex64** input, N = 64 | 1.0435e-4 | **1.7537e-7** | 0.999963 | **1.000000** |
| z **at** the K1 bound `N dx^2/lambda` | 1.0218e-14 | **1.7869e-14** | 1.000000 | **1.000000** |
| z **just below** it (0.999×) | 7.8096e-9 | **1.7091e-14** | 1.000000 | **1.000000** |
| z = 0.5× the bound | 8.6603e-1 | **4.3466e-14** | **0.250000** | **1.000000** |
| z = 1.5× the bound | 4.8910e-6 | **9.2793e-15** | 0.999997 | **1.000000** |
| **lens → propagate → lens** (f = 8 mm, z = 1.5 mm, f = 6 mm) | 6.9354e-4 | **1.8739e-15** | 0.999615 | **1.000000** |
| continuous integral, 8× oversampled quadrature, z = 2× the bound | 1.4563e-4 | **6.2546e-9** | 0.999922 | **1.000000** |

Five readings worth recording.

* **The complex64 row's `1.75e-7` is the f32 floor, not an error**: the leg
  preserves the input dtype (`complex64` in, `complex64` out, both trees) and
  `eps_f32 = 1.19e-7`.
* **The two K1 rows at and just below the bound were already exact before.**
  That is not luck: at `z = N dx^2/lambda` the single-FFT natural grid
  `lambda z/(N dx)` *equals* the chain grid, so the old leg's
  `abs(dx_new - current_dx) > current_dx*1e-6` guard skipped the resample
  entirely.  The engineer found the same thing and called it
  `sys_fresnel_talbot`; I reached it from the K1 side without knowing that.
* **The tall non-square orientation is the worst case and it *gains* power.**
  At 32 wide × 64 tall the old leg returned 1.733× the oracle's window power,
  because it scaled y by `Nx/Ny`.  The report's two non-square rows are both
  *wide* (24 × 18 and 64 × 48, in its `nx × ny` spelling) and read 0.65-class
  power ratios; the tall orientation is the one that inflates.
* **The continuous-integral row** is the one that asks whether the discrete
  sum stands for the physics rather than merely reproducing itself: against a
  Gaussian sampled 8× finer in each axis and integrated onto the coarse
  output grid, the leg reads **6.25e-9** at z = 2× the K1 bound.  The
  agreement with the discrete double sum (1e-15) is an implementation claim;
  this one is the quadrature claim, and it holds.
* The `z <= 0` refusal and the K1 warning are re-prefixed
  `fresnel_propagate_mft` exactly as the Migration note says; the anamorphic
  refusal text is byte-identical between the trees, and a pitch pair inside
  the guard's own `1e-9` slack still runs in both.

### 2.2 The gate

**(a) The boundary, both sides of the `1e-9` slack.**  The gate's boundary is
unreachable *through* the three call sites — all three skip the resample
when `abs(dx_new - dx) <= dx*1e-6`, which keeps `window/period` at least
1e-6 away from 1 — so I drove `resample_field` directly at a window exactly
one period wide, on a field that is an exact sum of five DFT-grid
exponentials (its band-limited interpolant is then known analytically, with
no FFT anywhere):

| window / period | gate says | `resample_field` warns | chirp-Z relL2 vs analytic | spline relL2 vs analytic |
|---|---|---|---|---|
| 0.999999998 | chirpz | no | 8.17e-15 | 1.05e-8 |
| **1.000000000** | **chirpz** | **no** | 8.62e-15 | 2.51e-16 |
| 1.0000000005 (inside the slack) | **chirpz** | **no** | 9.38e-15 | 2.48e-1 |
| 1.000000002 (outside it) | **spline** | **yes** | 8.46e-15 | 2.48e-1 |
| 1.000001 | spline | yes | 6.68e-15 | 2.48e-1 |

The claim "the chirp-Z leg is chosen on exactly the windows it would not warn
about" holds **exactly** at the boundary, in both directions.  (The spline
column's cliff at `+5e-10` is its own edge behaviour, not this package's: a
scale of `1 + 5e-10` pushes the first output coordinate to `-1.6e-8` input
pixels, and `map_coordinates(mode='constant')` zeroes the whole outer row and
column — `P = 0.938594 = (62/64)^2`.  Recorded for the reader; unchanged by
this package, and one more reason the gate is right to take chirp-Z at an
exactly-period window.)

`<` versus `<=` at the boundary is **not** falsifiable at the shipped call
sites and I did not write a test that cannot fail: mutating the gate to a
strict `<` leaves all 60 tests green, because the `1e-6` no-op guard keeps
every square site off the boundary and hitting it on a non-square site needs
a measure-zero thickness.  Recorded in §7 instead.

**(b) The per-axis `min`.**  On a **non-square** input the shorter extent is
the binding period, and this is reachable at exactly one site — the in-glass
`'fresnel'` gap, since `scalable_angular_spectrum_propagate` refuses a
non-square input outright (`"input must be square (got 48x64)"`, identical in
both trees) and the chain's own grid is square.  Driving that leg through a
1 mm-class N-BK7 plate at a thickness chosen to place `dx_new/dx`:

| grid (rows × cols) | `dx_new/dx` | window/period(min) | window/period(x) | per-axis gate | x-only gate | forced chirp-Z |
|---|---|---|---|---|---|---|
| 32 × 64 | 1.500 | 1.3333 | 0.6667 | **spline** | chirpz | warns **on y** |
| 32 × 64 | 1.200 | 1.6667 | 0.8333 | **spline** | chirpz | warns **on y** |
| 16 × 64 | 3.000 | 1.3333 | 0.3333 | **spline** | chirpz | warns **on y** |
| 48 × 64 | 1.200 | 1.1111 | 0.8333 | **spline** | chirpz | warns **on y** |
| 64 × 32 (short side in x) | 1.500 | 0.6667 | 0.6667 | **chirpz** | chirpz | — |

so the `min` is load-bearing, and the last row is the two-sided arm that
stops the first four from passing for the wrong reason.  Confirmed
independently on `resample_field` itself at 48 × 64, 64 × 48 and 33 × 65,
where a window between the short and the long period warns on exactly the
short axis.

**(c) Contained versus grid-filling, either side of one period** (`'sas'`
chain leg, dx = 2 µm, λ = 633 nm; `direct` = `fresnel_propagate_mft` onto the
chain grid, `asm` = `angular_spectrum_propagate` at the chain pitch):

| fixture | `dx_new/dx` | win/per | gate | `P` spline | `P` chirp-Z | `P` direct | relL2 spl / cz vs direct |
|---|---|---|---|---|---|---|---|
| contained Gaussian, N = 512, z = 5 mm | 0.7727 | 1.294 | spline | 1.000000 | 1.000000 | 1.000000 | 8.79e-7 / 8.79e-7 |
| contained Gaussian, N = 256, z = 2 mm | 0.6182 | 1.618 | spline | 1.000000 | **1.000594** | 1.000000 | 5.63e-6 / **2.44e-2** |
| contained Gaussian, N = 256, z = 1 mm † | 0.3091 | 3.235 | spline | 0.999999 | **9.000707** | 1.000239 † | 1.55e-2 / 2.83e0 |
| contained Gaussian, N = 64, z = 1 mm | 1.2363 | 0.809 | **chirpz** | 0.999791 | **0.999994** | 0.999994 | 7.96e-4 / **7.19e-4** |
| filling top-hat, N = 512, z = 5 mm | 0.7727 | 1.294 | spline | 0.950689 | **1.378837** | 0.996992 | 2.18e-1 / 6.04e-1 |
| filling top-hat, N = 256, z = 5 mm | 1.5454 | 0.647 | **chirpz** | 0.986730 | **0.986951** | 0.986945 | **1.035e-2** / 1.058e-2 |
| filling top-hat, N = 64, z = 1 mm | 1.2363 | 0.809 | **chirpz** | 0.961488 | **0.962304** | 0.962222 | 1.28e-2 / **1.22e-2** |
| filling top-hat, N = 256, z = 2 mm | 0.6182 | 1.618 | spline | 0.693897 | **1.818380** | 0.996072 | 5.52e-1 / 8.97e-1 |

† `z/z_crit` = 0.618 at that row (`z_crit = N dx^2/lambda`), so the direct
Fresnel evaluation is itself an aliased quadrature and its 1.000239 is not a
reference; the chirp-Z reading of 9.000707 does not depend on that — it is a
property of the window, and the 3 × 3 tiling is exact.  `P_asm` reads
1.000000 on the same row.

Every number the report quotes on these fixtures reproduces to the digit
(0.950689 / 1.378837, 0.986730 / 0.986951 / 0.986945, 1.000594, 2.15e-4 →
6.0e-6).  Two notes:

* the report's headline for item 2 is a **power** statement.  On its own
  fixture (N = 256, z = 5 mm) the relL2 against the same reference moves the
  other way — 1.035e-2 (spline) against 1.058e-2 (chirp-Z).  WP-B3 §5.1
  reading 2 says why (the spline's roll-off happens to suppress content the
  band-limited leg keeps); WP-B3b's summary row does not repeat it.  The gate
  is still the right choice — unit MTF and the better power — but "better" is
  metric-dependent on a grid-filling field and the report should say so where
  it quotes the improvement;
* **D3's caveat is narrower than it reads.**  Against ASM at the chain pitch
  (a reference that involves no resampling at all) chirp-Z at 0.7727 is
  1.81e-9 against the spline's 9.93e-9 — better, but both already exact, and
  the power readings are identical to six digits.  By 0.6182 chirp-Z is four
  decades *worse* (2.44e-2 vs 5.63e-6).  The band over which the gate "costs
  the unit MTF for nothing" is real but thin.

**(d) The ungated chirp-Z tiles.**  On a band-limited field with the window
set to m periods: `P/P_in` = **1.000000, 4.000000, 9.000000, 16.000000** at
m = 1…4, warning for every m > 1.  The report's "exactly 4.000000 and
9.000000" reproduces, and the m = 4 row extends it.

### 2.3 The in-glass legs

`dx_new = (lambda/n)*t/(N*dx)`, so the crossover is `t = N dx^2 n/lambda`.
On the WP-A15a covering-array doublet (N = 64, `dx = 1.2·6 mm/64` =
112.500 µm, λ = 632.8 nm) the two `'fresnel'` gaps read `dx_new/dx` =
**0.004218** and **0.001086** — the report's 4.218e-3 and 1.086e-3 — and both
take the spline; the `'sas'` gaps read 0.002109 and 0.000543 (half, from
`pad = 2`) and also take the spline.

Both directions are exercised.  A thickness sweep at N = 64, dx = 2 µm over
`t/t_cross ∈ {0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2, 4}` on N-BK7 and
N-SF6HT crosses the gate cleanly on both legs and both glasses: `'fresnel'`
takes the spline for `t/t_cross ≤ 0.99`, skips the resample entirely at
exactly 1.0 (the `1e-6` no-op guard), and takes chirp-Z from 1.01 up;
`'sas'`, whose `dx_new` is half, crosses at `t/t_cross = 2`.

**One correction.**  The report gives the doublet's crossover as a
**2137.6 mm** gap.  From its own inputs (N = 64, dx = 112.500 µm,
n = 1.6671, λ = 632.8 nm) `N dx^2 n/lambda` = **2133.9 mm**; the library's
own `get_glass_index('N-BAF10', 632.8e-9)` gives the same 2133.90 mm to six
figures.  The conclusion (a metres-long gap, so the covering array is deep in
the spline's half) is unaffected; the digit is not reproducible.

**Is the gate's choice the better one in glass?**  Measured on a band-limited
field through the same plate, against `fresnel_propagate_mft` onto the lens
grid:

| `dx_new/dx` | gate | relL2 spl | relL2 cz | `P` spl | `P` cz | `P` direct |
|---|---|---|---|---|---|---|
| 0.25 | spline | 9.68e-1 | **8.5e-14** | 1.218836 | 19.501369 | 19.501369 |
| 0.50 | spline | 8.66e-1 | **4.0e-14** | 1.097278 | 4.389113 | 4.389113 |
| 0.90 | spline | 7.88e-1 | 8.46e-1 | 0.925542 | 1.284106 | 1.306164 |
| 1.10 | chirpz | 5.07e-1 | 5.08e-1 | 0.821772 | **0.833302** | 0.832445 |
| 2.00 | chirpz | 3.08e-1 | **2.29e-1** | 0.450609 | **0.552308** | 0.552759 |
| 4.00 | chirpz | 8.16e-1 | 9.51e-1 | 0.177668 | **0.235432** | 0.230995 |

The top two rows are the sharpest result in this section and they are **not**
a defect: on the spline's side the "direct evaluation" is *itself* outside
its own faithful zone (`window/period` = 4 and 2), and the chirp-Z resample
of the natural grid reproduces it to **1e-14** — the two are the same
periodic reconstruction, replicas and all, `P = 19.5` and `4.39`.  That is
the cleanest available proof that the gate's `else` branch is necessary and
that "the direct evaluation" cannot be used as the reference outside one
period.  The report says as much with its `†` footnote; this is the in-glass
version of it.

---

## 3. Byte identity — my own probe, archive-to-archive

72 records (arrays + guard/warning texts), each measured in a child process
whose `cwd` and `PYTHONPATH` are the archive under test, with
`lumenairy.__file__` asserted first.  Never through pytest.

**64 of 72 arrays identical; 66 of 72 text records identical.**

| surface | cases | result |
|---|---|---|
| `resample_field`, spline leg | 9 — odd N = 65, the exact no-op, `order` 0/1/3/5, non-square 24 × 18 at the default `N_out`, odd anamorphic 33 × 21 → 45 | **identical** |
| `resample_field(method='chirpz')` | 5, including a 2×-period window that warns and a non-square default `N_out` | **identical** |
| `fresnel_propagate` (square + non-square), `fresnel_propagate_mft` (natural / chain-grid / warning window), `angular_spectrum_propagate_mft` (×2), `fraunhofer_propagate_mft`, `scalable_angular_spectrum_propagate` (×2), `angular_spectrum_propagate`, `angular_spectrum_propagate_tilted`, `rayleigh_sommerfeld_propagate` | 13 | **identical** |
| `propagate_through_system(method='asm')` | bare, `bandlimit=False`, 3-element chain, lens+aperture chain, anamorphic pitch, a tilted `'fresnel'` element (routes to tilted ASM), a tilted `'asm'` element, a 3-element tilted chain, a per-element override | 9 | **identical** |
| `propagate_through_system(method='sas')` | the two converging fixtures + a converging contained Gaussian | 3 | **identical** |
| `propagate_through_system_jax` | `'asm'` bare, `'asm'` 3-element chain | 2 | **identical** |
| `apply_real_lens` | `'asm'` plate, `'asm'` doublet, `'asm'` non-square, `'rs'` plate, `'sas'` plate, `'sas'` doublet, `'fresnel'` doublet (converging) | 7 | **identical** |
| guard / error paths | junk `method`, `'rs'` method, junk per-element method, anamorphic on both chain legs, SAS's non-square refusal, `z = 0` and `z < 0` on `'sas'`, junk `resample_field` method, junk `wave_propagator`, the `'sas'` crop warning | **identical text** |

**The 8 arrays that move**, every one a leg this package was asked to change:

| probe | why |
|---|---|
| `sys_fresnel_64` (E = ones, N = 64, z = 1 mm) | direct evaluation |
| `sys_fresnel_talbot` (z = the K1 bound) | the natural grid already equalled the chain grid, so the old leg did not resample: Bluestein against plain FFT.  Both trees sit 1.02e-14 (before) / 1.79e-14 (after) from my double-sum oracle, so the move is at that level |
| `guard_fresnel_undersampled` (z = 0.25× the bound) | direct evaluation, both readings aliased |
| `sys_sas_64_z1mm` (diverging) | the gate selects chirp-Z |
| `lens_fresnel_plate_fine`, `…_fine_g`, `…_anamorphic`, `…_nonsquare` | the gate selects chirp-Z |

**The 6 text records that differ** are exactly the four documented classes:
the K1 warning's prefix, the `z <= 0` refusal's prefix, the added
faithful-zone warning in the under-sampled band, the removed crop warning on
`sys_fresnel_64`, and the two reworded `propagate_through_system_jax`
refusals.  Nothing else moved.

---

## 4. The pins — mutation testing

The shipped file is 39 tests.  I mutated the library in a scratch copy of the
`908c02d6` archive (never the working tree) and ran the shipped file against
each mutant.  **Three mutations left every one of the 39 green.  Two of them
contradict a claim the report makes in so many words; the third (`<` for
`<=`) is genuinely unreachable and stays green with my tests too (§2.2).**

| mutation | 39 shipped pins | 60, with my 21 added |
|---|---|---|
| `min(E.shape[-2], E.shape[-1])` → `E.shape[-1]` (drop the per-axis rule) | **39 passed** | **4 failed** |
| `* (1.0 + 1e-9)` → `* 1.0` (drop the slack) | **39 passed** | **1 failed** |
| `<=` → `<` at the boundary | 39 passed | 60 passed — **genuinely unreachable**, see §2.2 |
| gate → `dx_new >= dx` (the pitch form) | 1 failed (the AST pin) | **6 failed** |
| revert the `'fresnel'` leg to propagate-then-resample | 8 failed | **11 failed** |
| gate → `method='chirpz'` unconditionally | 11 failed | **16 failed** |
| gate → `method='spline'` unconditionally | 6 failed | **8 failed** |

The shipped pins are otherwise sound by `docs/TESTING_STANDARDS.md`: every
bar is derived at runtime or carries its derivation and its measured decades
(the 1e-11 double-sum bar sits ~3.5 decades above the oracle floor and ~7
below the old leg's reading; the tiling test derives `m**2` from the geometry;
the byte-identity class reconstructs the pre-gate library in process and
carries a non-vacuity arm on a diverging fixture).  I found no per-build
reading and no S1–S5 shape.  The two gaps were both *unfalsifiable claims*,
not wrong ones.

Fail-before for my own additions, on the `908c02d6` archive:
`TestV1…::test_the_leg_warns_when_the_window_holds_part_of_the_beam` and
`TestV2…::test_the_docstring_says_the_condition_is_on_Nin` **fail**; the
other 19 pass, which is what makes them regression pins rather than
restatements.

---

## 5. Defects found and fixed (in my owned files)

### 5.1 V1 (P2) — the `'fresnel'` leg lost its K6 window diagnostic

**What the package claims.**  `WP-B3b_REPORT.md` §5, Migration note 2: "A
`'fresnel'` chain step no longer emits the K6 crop warning … The
faithful-zone diagnostic **that replaces it** comes from
`fresnel_propagate_mft`, with period `lambda*|z|/dx_in`."
`WP-B3b_CHANGELOG.md` puts it flatly: "`fresnel_propagate_mft` carries the
same K1 chirp-sampling guard … plus its own faithful-zone warning with period
`lambda*|z|/dx_in`, **so no diagnostic is lost**".  And the source comment
says the same: "which is why this leg needs no
`_warn_system_resample_crop`: there is no resample to crop."

**Why it is wrong.**  The mechanism is gone; the *condition* is not.  On the
chain grid `dx_out = dx_in = dx` and `N_out = N`, so
`fresnel_propagate_mft`'s faithful-zone test
`N_out*dx_out > lambda*|z|/dx_in` reduces to

    N*dx > lambda*z/dx   <=>   z < N*dx^2/lambda

— exactly the K1 under-sampled-chirp band.  A beam outgrows the chain window
in the **other** direction, at `z` *above* that bound.  The two conditions are
disjoint at this call site: no geometry can trip both, so the faithful-zone
warning cannot be the crop warning's replacement.  Meanwhile the old leg *did*
cover it, because for `z > N dx^2/lambda` its natural grid was coarser than
the chain grid and `_warn_system_resample_crop` measured the retained power.

**Measured** (λ = 633 nm, dx = 2 µm; `P_wide` = the same field evaluated on an
8× wider window):

| fixture | `z`/K1 bound | `P_out/P_in` | `P_wide/P_in` | `908c02d6^` | `908c02d6` |
|---|---|---|---|---|---|
| top-hat, radius 3 px, N = 64 | 30× | **0.031674** | 0.786324 | K6 crop warning | **silent** |
| top-hat, radius 4 px, N = 128 | 20× | **0.114889** | 0.879506 | K6 crop warning | **silent** |
| Gaussian, `w0` = 2.6 px, N = 128 | 10× | **0.334870** | **1.000000** | K6 crop warning | **silent** |
| top-hat, radius 5 px, N = 256 | 10× | **0.537900** | 0.973707 | K6 crop warning | **silent** |
| top-hat, radius 4 px, N = 128 | 6× | **0.699974** | 1.094883 | K6 crop warning | **silent** |
| grid-filling top-hat, N = 512 | 2× | 0.995833 | — | K6 crop warning | **silent** |

`P_wide` is the same field evaluated by `fresnel_propagate_mft` on an 8×
wider window.  The rows where it exceeds 1 are its own replica regime at that
pitch, so they bound the loss rather than measure it — which is why the third
row is the decisive one: there `P_wide` reads exactly **1.000000**, so the
step conserves the power and 66.5 % of it is simply not inside the chain's
window.  A chain that returns 3.2 % of its input power without a word is the
silent class K6 was raised about.

**The fix** (`lumenairy/propagators/system.py`).  A new
`_warn_system_fresnel_window(E_before, E_after, dx_target, wavelength, z)`,
called immediately after the `fresnel_propagate_mft` evaluation, warns
(`RuntimeWarning`, the same class and the same `1e-6` retained-power bar as
`_warn_system_resample_crop`) when the chain window holds less of the input
power than that.  It fires only for `|z| > N*dx^2/lambda`, which is *exactly*
the band the retired crop warning covered on this leg — `dx_new =
lambda*z/(N*dx) > dx` is the same inequality — so the partition is enforced
in code, not just described: below the bound a short window is not a crop at
all (the natural grid is finer than the chain's, so the reconstruction
replicates rather than truncates), which is what the faithful-zone and K1
warnings already say, and the leg does not add a third message there.
`_warn_system_resample_crop`'s docstring and the leg's own comment now say
which diagnostic covers which condition, instead of claiming one replaces the
other.

**The bar has decades on both sides.**  Contained Gaussians at 1×, 2× and 3×
the K1 bound read `P_out/P_in = 1.000000000` for N = 64, 65, 128 and 256 —
worst departure **3.1e-8**, four decades below the 1e-6 trigger — against the
0.30…0.97 departures above.

**It moves no values.**  The fixed tree is **72 of 72 arrays byte-identical**
to `908c02d6`, and exactly one text record changes: `sys_fresnel_64` regains a
K6 window warning, which is the probe where `908c02d6^` emitted the crop
warning.  Diagnostic parity with the pre-change library, restored.

Pinned by `TestV1TheFresnelLegStillReportsAWindowLoss` (4 tests: the two
conditions are disjoint across four decades of `z`; the leg warns and the
percentage it quotes matches the measured ratio; it is silent on five
contained fixtures; the field is bit-identical to the un-warned evaluation).

### 5.2 V2 (P3) — the F6 docstring's scale-factor enumeration is wrong

`resample_field`'s F6 paragraph (the one VERIFY-B3 asked for, and the only
part of `mft.py` in this package's ownership) said the extent-preserving
default lands on the period "x0.5, x1, x2 and x4 at **any** `N_in`".  The
condition it states one clause earlier is `N_in*dx_in/dx_out = N_in/scale`
whole, and that is a condition on `N_in` too: ×2 needs an **even** `N_in`,
×4 an `N_in` divisible by 4.  Measured over nine `N_in` × eight scales, the
`N_in`-divisibility rule holds in every cell; the claim fails at
`N_in = 127, 65, 63, 51, 34, 17`.

At `N_in = 65` the ×2 default rounds to `N_out = 32` — a 64-`dx_in` window
against a 65-`dx_in` period — and the power ratio reads **0.995181** on a
rim-filling envelope and **0.999995** on a contained one, against
**0.999907** at `N_in = 128` on the same rim-filling envelope, where the
same default lands on the period exactly.  The measurement fixture the
paragraph quotes is `N_in = 128`, where every one of its four "any `N_in`"
scales happens to be exact, which is why it was not caught.  (The pin
measures both sides at runtime rather than repeating these numbers.)

Corrected in place, with the `N_in = 65` reading added.  The edit is
docstring-only: `scripts/record_history_fingerprints.py --check` reports
`lumenairy.propagators.mft` **OK**.  Pinned by
`TestV2TheExactPeriodScaleFactorsDependOnNin` (3 tests, one of which fails on
`908c02d6`).

### 5.3 V3 / V4 (P3) — two load-bearing properties of the gate were unpinned

Both are claims the report makes explicitly and neither could fail (§4).
Added, with the reachable fixtures §2.2 describes:

* `TestV3ThePerAxisPeriodIsLoadBearing` — 5 tests.  Three parametrised grids
  on the in-glass `'fresnel'` leg (the only non-square-capable site) where
  the per-axis rule picks the spline and an x-only rule would pick chirp-Z,
  a fourth with the short side in **x** where the gate must pick chirp-Z (the
  two-sided arm), and a fifth that forces chirp-Z on one of the first three
  and requires `resample_field`'s own faithful-zone warning to name the **y**
  axis.
* `TestV4TheGateSharesTheResamplersOwnTolerance` — 2 tests.  One measures
  `_warn_mft_output_window`'s tolerance on the resampler itself (silent at
  `1 + 5e-10`, warns at `1 + 2e-9`); the other walks each owner function's
  AST, resolves one level of local naming, and requires the `method=`
  selector's test to carry that same `1e-9` literal.

No library behaviour changes for V3/V4 — they are tests only.

---

## 6. Requested changes outside my ownership

### 6.1 V5 (P2) — `_lens_real._propagate_through_glass`'s `'fresnel'` branch reads `dy` as `dx`

This is the same defect WP-B3b removed from `system.py`'s `'fresnel'` leg,
left standing in the sibling function whose `resample_field` call this package
edited.  `fresnel_propagate` returns two pitches
(`dx_new = lam_medium*t/(Nx*dx)`, `dy_new = lam_medium*t/(Ny*dy)`) and
`resample_field` takes one, so:

* with an **anamorphic pitch** (`dx = 2 µm`, `dy = 3 µm`, N = 64, 1 mm N-BK7)
  `dx_new = 3.264e-6` and `dy_new = 2.176e-6`, the call site resamples both
  axes with `dx_new`, and the result is **relL2 0.404** from my double-sum
  oracle with `P` ratio **1.4999 = dy/dx**.  The field is also returned on a
  `(dx, dx)` grid while the caller believes it is on `(dx, dy)`.
  `apply_real_lens(..., dx=2e-6, dy=3e-6, wave_propagator='fresnel')` runs
  **silently** (both trees);
* with a **non-square grid** a 48 × 64 input comes back 64 × 64 — an element
  that changes its caller's grid shape without a word.  `wave_propagator='asm'`
  returns 48 × 64 on the same call.

`'sas'` refuses both, with dedicated messages
(`"wave_propagator='sas' assumes a square grid pitch"` and
`"input must be square (got 48x64)"`), and `system.py`'s
`_require_square_pitch` now refuses the anamorphic case on the chain's own
`'fresnel'` leg *for exactly this reason* — its rewritten docstring says so.
The `'fresnel'` gap leg is the one path left.

**The exact edit** (in `_propagate_through_glass`, immediately before the
`fresnel_propagate` call at the `elif wave_propagator == 'fresnel':` branch,
mirroring the `'sas'` branch's existing guard):

```python
    elif wave_propagator == 'fresnel':
        from ..propagators.propagation import fresnel_propagate, resample_field
        # ``fresnel_propagate`` returns distinct dx_new/dy_new
        # (lam_medium*t/(Nx*dx) vs lam_medium*t/(Ny*dy)) and the
        # resample-back below takes ONE input pitch and one N_out, so an
        # anamorphic pitch or a non-square grid would be resampled on the
        # x ratio in both axes -- measured relL2 0.404 and a power ratio
        # of exactly dy/dx on a 1 mm N-BK7 gap at dx = 2 um, dy = 3 um.
        # Refuse rather than mislead, exactly as the 'sas' branch above
        # and propagate_through_system's own '_require_square_pitch' do.
        if abs(float(dy) - float(dx)) > abs(float(dx)) * 1e-9:
            raise ValueError(
                f"apply_real_lens: wave_propagator='fresnel' assumes a "
                f"square grid pitch, but this call is anamorphic "
                f"(dx={dx:.6g} m, dy={dy:.6g} m): fresnel_propagate returns "
                f"a different output pitch per axis and the resample back "
                f"onto the lens grid reads only one, so the y axis would be "
                f"scaled by the x ratio.  Use wave_propagator='asm' (or "
                f"'rayleigh_sommerfeld'), which thread the y-pitch "
                f"correctly, or resample to an isotropic grid first.")
        if int(np.shape(E)[-2]) != int(np.shape(E)[-1]):
            raise ValueError(
                f"apply_real_lens: wave_propagator='fresnel' assumes a "
                f"square sample count, but this call is "
                f"{np.shape(E)[-2]}x{np.shape(E)[-1]}: the resample back "
                f"onto the lens grid takes a single N_out, so the y extent "
                f"would be silently replaced by the x one.  Use "
                f"wave_propagator='asm', which keeps the grid.")
```

Checked rather than assumed, so the edit can be applied as written: all
**four** call sites of `_propagate_through_glass` (the two early-`continue`
branches in `apply_real_lens`'s surface loop, the whole-grid branch's
`_split_mode` pair, and `prepare_real_lens`'s cached applier) pass the whole
grid, never a row band — the `sag_chunk_rows` chunking applies to the
sag/phase screen, not to the gap propagation — so neither refusal can fire on
the chunked path.  (Line numbers deliberately omitted: VERIFY-B2 is editing
that file.)

I did **not** apply it: the brief gives me the two `resample_field` calls in
that function and nothing else, and a new refusal is a behaviour change in
someone else's file.  It is a P2 because the anamorphic path is silent, wrong
by 40 % relative, and reachable from `set_default_wave_propagator('fresnel')`
without any keyword at all.  If a hard refusal is judged too strong for this
release, the minimum is a `RuntimeWarning` carrying the same text — but the
`'sas'` sibling already raises on both conditions, so refusing is the
consistent choice.

### 6.2 V6 (P4) — one digit in the report

`WP-B3b_REPORT.md` §2.2 gives the covering-array doublet's gate crossover as
`t = N dx^2/lam_medium` = **2137.6 mm**.  From the report's own inputs it is
**2133.9 mm** (and the library's `get_glass_index` agrees to six figures).
Conclusion unchanged; the digit should be corrected when the report is next
touched.  Requested of the WP-B3b engineer.

### 6.3 `.test_durations`

My 21 new ids are not in `.test_durations`, which is outside my ownership and
already the subject of WP-B3b §7.1's request.  Measured on this host, one
BLAS thread, serial (2026-09-13):

```json
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV1TheFresnelLegStillReportsAWindowLoss::test_it_is_silent_where_the_window_holds_the_beam": 0.04,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV1TheFresnelLegStillReportsAWindowLoss::test_the_diagnostic_moves_no_values": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV1TheFresnelLegStillReportsAWindowLoss::test_the_leg_warns_when_the_window_holds_part_of_the_beam": 0.02,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV1TheFresnelLegStillReportsAWindowLoss::test_the_two_conditions_are_disjoint_on_the_chain_grid": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_an_odd_Nin_does_not_get_an_exact_window_at_x2": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x0.5]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x1.0]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x1.25]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x1.5]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x1.7]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x2.0]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x3.0]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_default_window_is_a_period_iff_Nin_carries_the_divisor[x4.0]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV2TheExactPeriodScaleFactorsDependOnNin::test_the_docstring_says_the_condition_is_on_Nin": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV3ThePerAxisPeriodIsLoadBearing::test_the_leg_the_min_refuses_is_the_one_that_warns": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV3ThePerAxisPeriodIsLoadBearing::test_the_shorter_input_extent_sets_the_period[short_x_binds_and_fits]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV3ThePerAxisPeriodIsLoadBearing::test_the_shorter_input_extent_sets_the_period[short_y_between_the_two_periods]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV3ThePerAxisPeriodIsLoadBearing::test_the_shorter_input_extent_sets_the_period[short_y_between_the_two_periods_b]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV3ThePerAxisPeriodIsLoadBearing::test_the_shorter_input_extent_sets_the_period[short_y_far_between]": 0.00,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV4TheGateSharesTheResamplersOwnTolerance::test_every_gate_carries_that_same_tolerance": 0.01,
    "tests/unit/test_audit2609_b3b_resample_call_sites.py::TestV4TheGateSharesTheResamplersOwnTolerance::test_the_resamplers_tolerance_is_one_part_in_1e9": 0.01,
```

(All 21 sum to 0.09 s; the file as a whole runs in 3.74 s, so they add nothing
material to either CI lane.)

---

## 7. Follow-up (recorded, not fixed)

| # | severity | item |
|---|---|---|
| **F1** | P3 | The `'fresnel'` chain leg is now **noisier in the K1 band**: below `z = N dx^2/lambda` it emits both the K1 under-sampled-chirp `RuntimeWarning` and `fresnel_propagate_mft`'s faithful-zone `UserWarning`, where `908c02d6^` emitted one.  The second's advice ("reduce `N_out*d_out` or `|centre_out|`") names knobs the chain owns, not the caller.  Harmless but redundant: on the chain grid the two fire on the identical condition (§5.1).  Suppressing it would mean a keyword on `fresnel_propagate_mft`, which is not this package's file. |
| **F2** | P3 | `<` versus `<=` in the gate is **not falsifiable** at the three shipped call sites: the `abs(dx_new - dx) > dx*1e-6` no-op guard keeps every square site at least 1e-6 from the boundary, and reaching it on a non-square site needs a measure-zero thickness.  I deliberately did not add a test that cannot fail (`docs/TESTING_STANDARDS.md` restatement 5). |
| **F3** | P3 | On a **grid-filling** field the gate's improvement is metric-dependent: the window power improves (2.15e-4 → 6.0e-6) while relL2 against the same reference worsens slightly (1.035e-2 → 1.058e-2).  WP-B3 §5.1 reading 2 explains it; WP-B3b's summary quotes only the power.  Worth one clause in the report. |
| **F4** | P4 | `resample_field`'s **spline** leg has an asymmetric edge cliff at `dx_out/dx_in = 1`: at exactly 1 it is the identity, at `1 + 5e-10` the first output coordinate lands at `-1.6e-8` input pixels and `mode='constant'` zeroes the whole outer row and column (`P = 0.938594 = (62/64)^2` on a rim-filled field, relL2 0.248), while at `1 - 2e-9` it is 1.05e-8.  Not this package's code and not reachable through the gated call sites, but it is a trap for any future caller that resamples "by about 1". |
| **F5** | P3 | WP-B3b §9 D2 is right that `resample_field`'s single `N_out` is what stops the `'sas'` leg preserving a non-square count.  The **in-glass `'fresnel'` leg has the same limit and no refusal** — see §6.1.  Closing D2 properly means a `(Ny_out, Nx_out)` pair on `resample_field`; until then the refusal is the honest answer. |
| **F6** | P4 | The `'fresnel'` leg reads `int(E_in.shape[-1])` — the *chain's original* input — for `N_out`, not `E.shape[-1]`.  Identical to the pre-change leg and correct for every element type the chain implements today (none change the sample count), but it is a latent coupling: an element that ever did would silently re-shape the field at the next Fresnel step. |
| **F7** | P4 | Both structural pins on the gate — WP-B3b's `test_the_gate_is_written_in_the_window_period_form` and my `test_every_gate_carries_that_same_tolerance` — call `inspect.getsource(_lens_real._propagate_through_glass)`, which resolves the *imported* code object's line numbers against the file **on disk**.  One covering-slice run failed both with `SyntaxError: unmatched ')'` because VERIFY-B2 wrote `_lens_real.py` while the run was in flight; both pass on every run before and after, and the function parses cleanly now.  Not a library defect and not worth a rewrite, but a shared-tree gotcha to recognise rather than re-debug: a structural pin on a module someone else is editing is flaky for the duration of their edit. |

---

## 8. Files I changed

| file | what |
|---|---|
| `lumenairy/propagators/system.py` | new `_warn_system_fresnel_window` (the restored K6 window diagnostic, V1) and its call in the `'fresnel'` leg; `_warn_system_resample_crop`'s docstring and the leg's comment corrected to say which diagnostic covers which condition |
| `lumenairy/propagators/mft.py` | `resample_field`'s F6 paragraph only — the scale-factor enumeration corrected (V2).  Docstring-only; the history fingerprint did not move |
| `docs/history/lumenairy.propagators.system.md` | re-recorded with `scripts/record_history_fingerprints.py`, reason naming V1 |
| `tests/unit/test_audit2609_b3b_resample_call_sites.py` | **+21 tests** (39 → 60); nothing existing weakened or removed |
| `docs/audits/.../fixes/VERIFY_WP-B3b.md`, `VERIFY_WP-B3b_CHANGELOG.md` | this report and its release text |

`lumenairy/elements/_lens_real.py` was **not** modified — its two
`resample_field` calls are byte-for-byte what WP-B3b shipped, and the working
tree's edits to that file belong to VERIFY-B2 (`_apply_displaced_remap`,
`_warn_if_remap_lattice_smooths`, `_normalise_displaced_n_side`,
`apply_real_lens`'s signature).  Nothing else was opened for edit.

---

## 9. Tests run

All under `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one
process at a time, `-p no:randomly` for the pin file.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b3b_resample_call_sites.py` | **60 passed** (39 shipped + 21 mine) on the final tree, both `-p no:randomly` and under the default order — order-robust | 3.63 / 3.69 s |
| `pytest tests/unit/test_audit2609_a5_*.py tests/unit/test_audit2609_b3_propagator_kernels.py tests/unit/test_audit2609_a2_*.py tests/unit/test_audit2609_b2_displaced_remap_inversion.py tests/unit/test_audit2609_a15a_lens_covering_array.py tests/unit/test_audit2609_a17_history_lint.py tests/unit/test_audit2609_b3b_resample_call_sites.py` | **478 passed** on the final tree (264.68 s); **448 passed** earlier (198.93 s, WP-B3b's 427 + my 21 — the +30 between the two runs are VERIFY-B2's new `b2` pins landing in the shared tree).  One intermediate run of this selection failed 2 — see F7 | 264.68 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py -k "system or mft or lens_real"` | **18 passed**, 721 deselected | 2.65 s |
| `pytest tests/unit -k "system or propagate_through or fresnel or sas"` | **525 passed, 3 skipped**, 15 488 deselected (WP-B3b read 521; +4 are mine that match the selector) | 163.44 s |
| `pytest tests/unit -k real_lens` | **160 passed, 3 skipped**, 15 853 deselected (the same 160 WP-B3b read) | 266.01 s |
| `python validation/run_all.py test_propagation test_lenses test_dispatch` | **ALL 3 files passed** | 27.6 / 1.8 / 8.1 s |
| `python -m ruff check` (whole tree) | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | **my two are OK**: `lumenairy.propagators.system` (re-recorded in this change, reason naming V1) and `lumenairy.propagators.mft` (the docstring edit did not move it).  `lumenairy.elements._lens_real` reads OK or DRIFT depending on where VERIFY-B2's uncommitted edits stand at the moment of the check — I neither edited nor re-recorded it.  Drift elsewhere is `carrier.py` and `lenses_maslov.py`, neither mine | 12 s |
| the mutation matrix — 7 mutants × the pin file, each a scratch copy of the `908c02d6` archive, never the working tree | §4 | ~4 s each |
| probes 1–5 and the byte-identity pair, archive-to-archive over `before` (`908c02d6^`), `after` (`908c02d6`) and `fixed` (`908c02d6` + my two edited library files) | §2, §3, §5 | 20–200 s each |

The skips are pre-existing optional-dependency skips, unrelated to this
change: `astropy` in the `-k system…` selection, and `PySide6`, `numexpr` and
the host-specific W5 digest capture in `-k real_lens`.

Every duration is indicative only: the box was shared with other engineers
throughout (their uncommitted `_lens_real.py`, `carrier.py`,
`lenses_maslov.py`, `_lens_traced.py`, `lens_config.py`, `rcwa/*` edits landed
during these runs), and no test asserts one.

---

## Orchestrator note (2026-09-13, at this report's landing)

V5 (section 6.1) was applied as requested: `_propagate_through_glass`'s `'fresnel'` branch now refuses an anamorphic pitch and a
non-square grid with the same shape of message as its `'sas'` sibling.  That removes the one call site V3 (section 5.3) could reach
with a non-square grid, so the per-axis `min` is unreachable by construction at all three sites; `TestV3ThePerAxisPeriodIsLoadBearing`
is restated to pin the property on `resample_field` itself (a window between the short and the long period makes the chirp-Z leg warn
on exactly the short axis; a window inside the short period is silent), to pin the refusal as the reason, and to require every call site
to keep spelling the general form.  Nothing else in this report changes.

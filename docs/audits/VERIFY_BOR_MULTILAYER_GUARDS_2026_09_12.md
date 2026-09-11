# Verifying the BOR multilayer guards — 5.45.1, 2026-09-12

> **STATUS — INDEPENDENT ADVERSARIAL VERIFICATION.** Of the branch
> `fix/bor-multilayer-guards` (nine commits `036dcad` .. `c1189e3`), against the
> tree it forked from (`352173e`, the `wave2/pmm2d` tip). Nothing here reads a
> number out of the build's report and repeats it: every quantity below was
> re-measured on fixtures written for this verification, on both local builds,
> across the OpenBLAS kernel and thread ladders.
>
> Verification tree: `C:\tmp\lum_vbor`, branch `verify/bor-multilayer-guards`.
> Pre-build tree: `C:\tmp\lum_vbor_pre` (detached at `352173e`).
> Probes and JSON: `validation/probe_verify_bor_guards/`.
> New gates: `tests/unit/test_verify_bor_multilayer_guards.py`.
>
> Documents verified against:
> `docs/audits/PLAN_BOR_MULTILAYER_GUARDS_5_45_1_2026_09_12.md`,
> `docs/audits/BUILD_BOR_MULTILAYER_GUARDS_2026_09_12.md`,
> `docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`,
> `docs/TESTING_STANDARDS.md` (binding).

---

## 1. Terms

Defined before they are used, because three of the five decisions below are
made on quantities whose names do not say what they measure.

**BOR — body of revolution.** An axisymmetric stack: layers along `z`, each a
set of concentric rings in `r`, closed at `r = Rbig` by a PEC (Dirichlet) wall.
Solved one azimuthal order `m` at a time; fields go as `exp(i m phi + i q z)`,
with `q` the **axial wavenumber** and `qn = q / k0` its dimensionless form.

**The three radial bases.** `basis='fd'` — a div-conforming Yee staggered
finite-difference basis on ONE uniform radial grid shared by every layer.
`basis='sem'` — a spectral-element basis, one mesh per layer with element
boundaries on that layer's ring walls, layers coupled by a mortar.
`bor_solve.build_layer(basis='nodal')` — the historical nodal FD basis, not
div-conforming, carrying a divergence-violating **spurious mode sea**.

**Channel.** A propagating diffraction channel of the cascade: one entry of the
`R` / `T` arrays. For a UNIFORM PEC-walled half-space of index `n` the number
of channels is a **closed form** — TE modes satisfy `J_m'(gamma Rbig) = 0`, TM
modes `J_m(gamma Rbig) = 0`, and a mode propagates when `gamma < n k0`. That
closed form is used below as an oracle that depends on no solver in this
library.

**Passive medium.** One with `Im eps >= 0` in this library's `exp(-i omega t)`
convention: absorbing or lossless, but not amplifying. For a stack of passive
media, energy conservation reads `R + T + A = 1` with the absorbed fraction
`A >= 0`, so **`R + T <= 1` is a theorem on ANY passive stack, lossy or not**.
On a LOSSLESS passive stack inside a PEC wall there is no absorption and no
other exit, so `R + T = 1` is an **equality**. This distinction decides two of
the findings below.

**The enrichment window.** `BORStack._solve_sem` meshes layer `i` on the union
of the walls of layers `i-1`, `i` and `i+1`. Two neighbouring layers whose
walls differ by `delta` therefore **manufacture** a radial element of width
`delta` in both meshes, which neither layer asked for.

**Move factor.** The measure used here for "is the answer wrong": the change in
the per-channel `R` between a perturbed geometry and its exact `delta -> 0`
reference, divided by the change a LINEAR (physical) response would give,
anchored at a rung the contract calls ordinary. A smooth answer reads ~1; the
library's own 1-D attribution bar for this quantity is **100x**. `R` arrays are
**sorted before differencing** — the solver's channel ORDER follows the
eigensolver and is not stable between two solves, so an elementwise difference
of the raw arrays measures a permutation, not a moved answer. (Two of this
verification's own first-pass numbers were wrong for exactly that reason and
were re-measured.)

**Arms.** A build (Windows py3.14.6 / numpy 2.4.4 / scipy-openblas 0.3.31, or
WSL py3.12.3 / numpy 2.4.6) x a requested `OPENBLAS_CORETYPE` x a thread count.
The **loaded** kernel is read back from `threadpoolctl.threadpool_info()` in
every run and is what is reported; `ZEN` silently aliases Haswell on this host
and `SKYLAKEX` dies on the first LAPACK call. **`SANDYBRIDGE` loads and works
here and is a kernel the build did not run**; it is included below.

---

## 2. Verdict table

| # | Claim | Verdict | The number that decides it |
|---|---|---|---|
| 1 | ONE orientation kernel replaces five copies, band unchanged, bit-identical | **CONFIRMED** | exactly one `def` of each of the four helpers in the package; no residual copy of the old spelling; **58 of 58** of my own BOR fixtures bit-identical pre -> post on Windows AND on WSL |
| 2 | The band becomes relative to the spectrum; the near-cutoff channel count stops moving | **CONFIRMED, and stronger than claimed** | over **20 arms** (5 kernels x 2 thread counts x 2 builds): channel counts `{80, 81}` -> `{81}`, worst lossless closure **3.4310e-04 -> 7.5390e-11 (4.6e6x)**, physically-propagating modes shipped with BACKWARD flux **0 .. 7 -> 0 on every arm** |
| 2a | The four-sided bar's margins | **RESTATED** | noise sides confirmed (6.58 dec ordinary, 2.97 dec deep cutoff); **the SIGNAL side is 0.38 decades, not 0.98** — the build reports a statistic larger than its population's minimum |
| 2b | The `k0` floor is the right floor | **BOUNDED — unverifiable** | the floor binds on **0 of 122** measured layers; closest approach `max|q| / k0 = 2.42`. The choice is correct in principle and has no measurable effect |
| 3 | EME: one branch selector; the mode set stops depending on the build | **CONFIRMED, and stronger than claimed -- but the fix introduces a NEW P2** | over 99 root vectors x 16 arms per build: **PRE 35 of 99 fixtures have a root decided by the BLAS kernel** (worst cross-arm `\|d ky\|` = 4.351e+05); **POST 0 of 99**. Mode count on the gate window: PRE **{18,19,20,21}** across arms against 16 with real `eps`; POST **16 everywhere**. NEW: `cut_band`'s literal 1.0 floor is unit-dependent (D13) |
| 4 | Nodal passivity refusal; the FLAGGED gate | **CONFIRMED as a fix, REFUTED as a two-population story** — see §4 | the flagged 1.02882 answer is **wrong**, not a floor: 8 channels against the closed-form 6, per-channel `R` off by **132 %**. But the bar's gap is **1.18 decades, not 6.81**, and the screen has **two holes** (§7 D1, D2) |
| 5 | SEM manufactured-element contract | **CONFIRMED as a decision, RESTATED as a bar** — see §5 | decision table identical on every arm; no MISS found (worst returned rung **7.4x**, bar 100x); refusals up to a decade conservative (**4.8x .. 7.4x** at the first refused rung). **The Q bar's quoted ordinary margin is vacuous** (§5.3) and the warn edge is entered by a 32-slice shallow taper |
| 6 | Census hooks present and dormant | **CONFIRMED** | 17 `census_inv`/`census_solve` call sites across 4 modules; `_BOR_INV_CENSUS = None` by default; 58/58 fixtures bit-identical with the hooks in place |
| 7 | Bit-identity end to end; runs; ruff | **CONFIRMED** (see §8) | 58 BOR + 15 EME fixtures on two builds; test matrix in §8 |

---

## 3. Claim 1 and 2 — the orientation kernel and the band

### 3.1 Bit-identity on my own battery (task A)

58 `BORStack` fixtures, built for this verification and not copied from the
build's probes: both bases x `m` = 0/1/2/5 x `k0` = 0.8/2.0/3.5, ring-grating
pairs, three-layer stacks with a wall-free spacer, lossy layers at `Im(n)` =
1e-1/1e-3/1e-6, the uniform-equals-region and uniform-equals-neighbour
coincidences, thin rings at `w/Rbig` = 1e-2 and 1e-3, taper staircases at
4/8/16/32/64 slices, an anisotropic (diagonal cylindrical tensor) layer,
hp-refined and graded meshes, and an nm-unit fixture whose `Rbig` and `k0`
differ from the rest by six orders of magnitude. Answers hashed to the SHA-256
of the exact IEEE-754 bytes of `R` and `T`.

| build | arm | BOR identical | BOR moved | EME identical | EME moved |
|---|---|---|---|---|---|
| Windows py3.14 / numpy 2.4.4 | Haswell, 1 thread | **58 / 58** | **0** | 13 / 15 | 2 |
| WSL py3.12 / numpy 2.4.6 | Haswell, 1 thread | **58 / 58** | **0** | 13 / 15 | 2 |

**Every mover, with its cause.** Exactly two, both EME, both the population the
branch fix is entitled to move:

* `striplossy_im1e-30` — a strip solve with `Im(eps) = 1e-30` on the high
  region. 29 of 96 roots are selected differently. Cause: the on-cut roots are
  now CONJUGATED rather than negated. Confirmed to be the fix and not a
  side-effect by the neighbouring arms: `Im(eps) = 1e-12` and `1e-3` are
  bit-identical on both builds, i.e. nothing off the cut moved.
* `layermodes_96_im1e-30` — the mode count of the same layer. Pre-fix **18 on
  Windows and 19 on WSL**; the real-`eps` arm reads **16** on both. Post-fix
  **16 on both builds**. This is the discontinuity at `Im(eps) = 0+` closed.

No BOR fixture moved on either build. No fixture was refused by either new
guard. PML is **not reachable** on the staggered path at all — `zcascade.
layer_modes(staggered=True, R_pml=...)` refuses by name ("the radial PML is a
NODAL-basis feature") — so the PML population was taken on the nodal basis
instead and is reported in §3.4.

### 3.2 The band's decision across 20 arms (task B)

A near-cutoff ladder built here: a uniform PEC-walled layer's transverse
eigenvalues `gamma_j` are `k0`-independent, so measuring one lets `k0` be set
to put a chosen radial order at `qn = n sqrt(delta)` for any `delta`. The
observable is a `BORStack(basis='fd')` solve's channel count and lossless
energy closure at `delta` from 1e-4 to 1e-10.

| requested | loaded | threads | build | channel counts | worst closure | propagating modes shipped BACKWARD |
|---|---|---|---|---|---|---|
| HASWELL | Haswell | 1 | pre | **{81}** | 4.163e-11 | **5** |
| HASWELL | Haswell | 1 | post | {81} | 4.163e-11 | **0** |
| HASWELL | Haswell | 4 | pre | {81} | 7.539e-11 | **6** |
| HASWELL | Haswell | 4 | post | {81} | 7.539e-11 | **0** |
| NEHALEM | Nehalem | 1 | pre | **{80, 81}** | **3.595e-05** | 5 |
| NEHALEM | Nehalem | 1 | post | {81} | 6.476e-11 | 0 |
| NEHALEM | Nehalem | 4 | pre | **{80, 81}** | **3.431e-04** | 4 |
| NEHALEM | Nehalem | 4 | post | {81} | 7.136e-11 | 0 |
| PRESCOTT | **Katmai** | 1 | pre | **{80, 81}** | **3.431e-04** | 1 |
| PRESCOTT | Katmai | 1 | post | {81} | 4.733e-11 | 0 |
| PRESCOTT | Katmai | 4 | pre | {81} | 4.048e-11 | 0 |
| PRESCOTT | Katmai | 4 | post | {81} | 4.048e-11 | 0 |
| SANDYBRIDGE | **Sandybridge** | 1 | pre | **{80, 81}** | **3.431e-04** | 4 |
| SANDYBRIDGE | Sandybridge | 1 | post | {81} | 6.755e-11 | 0 |
| SANDYBRIDGE | Sandybridge | 4 | pre | **{80, 81}** | **3.431e-04** | **7** |
| SANDYBRIDGE | Sandybridge | 4 | post | {81} | 1.632e-11 | 0 |
| *(unset)* | Haswell | 1 | pre | {81} | 4.163e-11 | 5 |
| *(unset)* | Haswell | 1 | post | {81} | 4.163e-11 | 0 |
| *(unset)* | Haswell | 4 | pre | {81} | 7.539e-11 | 6 |
| *(unset)* | Haswell | 4 | post | {81} | 7.539e-11 | 0 |
| *(unset)* | Haswell | 1 | WSL pre | **{79, 80, 81, 82}** | **3.431e-04** | **4** |
| *(unset)* | Haswell | 1 | WSL post | {79, 81, 82} | **1.523e-10** | **0** |

Union over all arms: **pre `{79, 80, 81, 82}` with worst closure 3.4310e-04 and
up to 7 forward modes carrying backward flux; post `{79, 81, 82}` with worst
closure 1.5230e-10 and zero.** (79 / 81 / 82 are the three azimuthal orders
`m` = 2 / 1 / 0 of the ladder; the count that MOVES is `m = 1`'s, 81 -> 80.)

That is the claim, reproduced independently, on a kernel the build did not run
(Sandybridge) and with a larger effect than the build reports (4.6e6x on the
closure against their 619x, because my ladder reaches a rung where a whole
channel is dropped).

**The mechanism, measured directly rather than inferred.** Over a 14,400-mode
census across 60 near-cutoff layers, the OLD band classifies **14** physically
propagating modes (`|Re q| > |Im q|`) as evanescent; the new band classifies
**0**. Neither band ever classifies a physically evanescent mode as
propagating. On the PRE tree **9** of those modes come back with negative
`r dr` z-flux in the returned basis — a forward mode carrying power in `-z`; on
POST, **0**.

### 3.3 The four-sided bar, re-derived (task B)

The statistic that matters on the NOISE side is the worst `sigma = |Im q| /
max(max|q|, k0)` over modes that are PHYSICALLY propagating — not over the
modes the band happens to call propagating, which is bounded by the band itself
and proves nothing. On the SIGNAL side it is the **minimum** over the lossy
population — the closest approach from above — not the maximum.

| side | population | n (modes / layers) | my worst | build's | room at 1e-8 |
|---|---|---|---|---|---|
| NOISE, ordinary lossless | 29 layers, both `m` families, three `k0`, two unit systems 12 decades apart | 7,280 / 29 | **2.623e-15** (WIN) / 2.387e-15 (WSL) | 1.3024e-15 | **6.58 dec** |
| NOISE, deep cutoff | `delta` 1e-4 .. 1e-26, both sides of cutoff, `m` = 0/1/2 | 14,400 / 60 | **1.076e-11** (WIN) / 7.437e-12 (WSL) | 8.7301e-10 | **2.97 dec** |
| SIGNAL, `Im(n)` = 1e-3 | three `m`, uniform lossy | 3 layers | **2.382e-05** | 9.4570e-05 | **3.38 dec** |
| SIGNAL, `Im(n)` = 1e-6 | three `m`, uniform lossy | 3 layers | **2.382e-08** | 9.4570e-08 | **0.38 dec** |

`sigma` is exactly linear in `Im(n)` across my ladder (2.382e-11 at 1e-9 up to
2.382e-03 at 1e-1, to four digits), so the band at 1e-8 absorbs every medium
with `Im(n) < 4.20e-07` into the propagating class.

**RESTATED, not refuted.** The build's SIGNAL rows quote a number larger than
the minimum of their own population, so the thin end carries **0.38 decades**,
not 0.98. The decision is still right, for the reason the build gives and this
verification re-measured: where the two orientation rules would disagree they
do not — over the whole lossy ladder, zero modes classified propagating by the
new band carry backward flux on either build. But the margin as stated is
optimistic by 0.6 decades and the test that re-measures it should read the
minimum, not the maximum.

Also measured: the new band admits modes onto the flux rule whose `Im q` is
negative, i.e. formally on the growing branch. Worst over 14,400 modes:
`Im q = -1.25e-10` (post) against `-3.22e-13` (pre). Amplification over a unit
thickness: `1 + 1.25e-10`. Harmless, and recorded so it is not rediscovered.

### 3.4 Unit-system invariance and the `k0` floor

Scaling the whole unit system by `s` (lengths x `s`, wavenumbers / `s`) over
`s` = 1e-6 .. 1e6 leaves the decision exactly where it was, on both builds:

| `s` | 1e-6 | 1e-3 | 1 | 1e3 | 1e6 |
|---|---|---|---|---|---|
| modes classified propagating | 42 | 42 | 42 | 42 | 42 |
| R/T channels | 42 | 42 | 42 | 42 | 42 |
| largest `R` | 0.336240 | 0.336240 | 0.336240 | 0.336240 | 0.336240 |
| closure | 6.171e-12 | 4.241e-14 | 6.573e-14 | 6.284e-14 | 5.962e-13 |

**But this does not test the floor.** `orient_band_scale` returns
`max(max|q|, k0)`, and `max|q|` is dominated by the largest transverse
eigenvalue, `~ N / Rbig`, which exceeds `k0` on any grid that resolves the
wavelength. Over all 122 layers measured here the floor **binds on zero of
them**, the closest approach being `max|q| / k0 = 2.42`. The `k0`-rather-than-
1.0 choice is right in principle (a literal 1.0 would be unit-dependent) and
**cannot be exercised through `BORStack`**; the invariance above comes from
`max|q|` scaling, not from the floor. Recorded as BOUNDED rather than
CONFIRMED.

### 3.5 The PML and anisotropic paths

`BORStack` never sets `R_pml`, and `zcascade.layer_modes(staggered=True)`
**refuses** `R_pml` outright, so the PML population exists only on the legacy
nodal basis. Measured there (960 modes, 4 layers, `R_pml/Rbig` = 0.7 and 0.85,
with a non-PML nodal control): both bands behave identically (167 vs 169 modes
classified evanescent out of 181 with `|Re q| > |Im q|` — the complex
coordinate stretch makes that split meaningless, as expected), zero backward
modes on either build, and nothing moves pre -> post. The contract makes no
claim there and neither does this verification.

Anisotropic layers (diagonal cylindrical tensor, both bases) are in the
bit-identity battery and did not move.

---

## 4. Claim 4 — the nodal passivity refusal and the FLAGGED gate

### 4.1 The flagged gate: is 1.02882 a floor or a wrong answer?

The stack is the shipped gate's: `m = 1`, `Rbig = 4`, `N = 200`, `k0 = 2`, a
ring-grating layer of thickness 0.5 between two uniform `eps = 2` half-spaces,
every medium LOSSLESS. Reproduced on the pre tree and on this one:
`max(R + T) = 1.028819`.

Scored against three references on the same geometry.

**(a) Against the staggered twin through the same cascade.**

| | nodal | staggered |
|---|---|---|
| `max(R + T)` | **1.028819** | 1.000000000000 |
| channels | **8** | **6** |
| `qn` of the channels | 0.1039, 0.7261, 0.9833, 1.1531, 1.2709, 1.3507, 1.3977, **1.414221** | 0.6189, 0.9282, 1.1095, 1.2474, 1.3306, 1.3954 |

The nodal set contains `qn = 1.414221`, which **exceeds** `n = sqrt(2) =
1.414214`: an axial index larger than the medium's own, i.e. `gamma^2 < 0`.
Of the eight nodal channels only **three** match a staggered channel within 2 %
in `qn`, and on those three the per-channel reflectance differs by **15.0 %,
132.5 % and 3.7 %**.

**(b) Against a converged ladder.** The staggered answer is converged: over
`N` = 100 .. 500 its six channel `R` values move by at most 9e-4 (0.06038 ->
0.06047, 0.11206 -> 0.11180) and its closure stays at 1e-12. The nodal answer
does **not** converge to it — at `N` = 100/200/300/400 it holds its own eight
channels and its own values (0.97487, 0.43942, ...) — and at `N = 150` it
reads `max(R + T) = 5.916`. The "floor" is not even monotone in `N`; the
shipped gate's 1.02882 at `N = 200` was one draw from that.

**(c) Against a closed form.** For the uniform half-space of this stack the
channel count is the number of `J_m` and `J_m'` zeros below `n k0 Rbig`. At
`m = 1`, `Rbig = 2` vacuum wavelengths, `n = sqrt 2` that is **10**; the
staggered basis returns **10**, the nodal basis returns **12**.

**VERDICT on the flagged item: the refusal is a TRUE positive and the build's
change of the gate is right.** The old gate was rationalising a defect. Named
precisely: the nodal basis returns a channel SET that is wrong — two to
twenty-one spurious channels depending on the cell size, at least one of them
at an axial index the medium cannot support — and the 2.9 % energy violation is
the *smallest* of the errors that fixture carries, not the largest. A gate that
called that "within the documented ~1-4 % floor" was reading the one number
that understated the damage by a factor of forty.

### 4.2 The bar's two populations, re-derived

My own population: 144 solves (two bases x `m` = 0/1/2 x `N` = 120/200 x
`Rbig/lambda` = 0.5 .. 14 x {uniform, ring}), guard disarmed so every row is the
number the solver returns.

| population | n | `max(R+T) - 1` |
|---|---|---|
| STAGGERED, every row | 72 | -1.9e-13 .. **6.4586e-12** |
| NODAL, below the 1e-3 bar | 30 | 8.9e-16 .. **6.7158e-04** |
| NODAL, above the bar | 42 | **1.0176e-02** .. 1950.8 |

**The build's "6.81-decade gap with nothing in it" is REFUTED as a population
statement.** My population has rows at **9.3039e-05** and **6.7158e-04** — both
`m = 1`, uniform, `Rbig/lambda = 4`, i.e. ordinary settings. The bar at 1e-3
sits **0.17 decades (1.49x)** above the highest row it does not refuse and
**1.01 decades (10.2x)** below the mildest row it does. The decision is
unchanged and still correct; the *two-sidedness* is a decade tighter than the
build states, and the gap is not empty.

**Both edges checked for correctness, not just for distance.**

* The four mildest REFUSED rows (`max(R+T) - 1` = 1.018e-02 .. 1.689e-02) carry
  nodal channel counts of 64 / 121 / 103 / 59 against staggered 44 / 78 / 81 /
  43 — **16 to 43 spurious channels** — with worst per-channel relative `R`
  errors of 11x to 3.5e4x and medians of 6 % to 16 %. **No false refusal
  found.**
* The two WARNED rows are already wrong: nodal 25 channels against the
  closed-form-confirmed staggered 22, worst channel 16x off, median 2.8 %.

### 4.3 The screen does not track the damage — an independent oracle

Forty rows (`m` = 0/1/2/3 x `Rbig/lambda` = 0.5 .. 8 x `N` = 120/200) scored
against the Bessel-zero channel count:

* the **staggered** basis returns the exact count on **28 of 40** rows, and on
  **13 of 14** rows at `m >= 1` with `Rbig/lambda <= 4` (the 12 misses are
  `m = 0`, where this oracle's mode bookkeeping is itself off by one, and the
  largest cells, where a mode sits within a grid spacing of its cutoff);
* the **nodal** basis returns it on **12 of 40**;
* **20 of the 32 rows the passivity screen calls OK have the wrong nodal
  channel count**, including `m = 1`, `Rbig/lambda = 2`, where the screen reads
  `max(R+T) - 1 = 9.98e-10` and the basis returns **12 channels against the
  exact 10**.

So the passivity excess detects only the part of the nodal defect that happens
to carry power. This does not make the guard wrong — it refuses real damage and
refuses nothing correct — but the build's framing of two clean populations
("the family that is accurate" at 4.4e-09) does not survive a channel-set test.

---

## 5. Claim 5 — the SEM manufactured-element contract

### 5.1 The delta ladder against an exact `delta -> 0` reference (task D)

My own two-layer fixture (`Rbig = 24`, `k0 = 2`, `m = 1`, walls at 9.0 and
`9.0 + delta`), degrees 6 / 8 / 12, scored on channel-SORTED `R` against the
`delta = 0` answer, with the move factor anchored at `delta/Rbig = 1e-3` (a
rung every party agrees is ordinary):

| `delta/Rbig` | cell / `Rbig` | `q` excess | verdict | move factor, deg 6 / 8 / 12 | answer class |
|---|---|---|---|---|---|
| 1e-1 | 5.0e-2 | 15.0 | ok | (anchor region) | correct |
| 1e-2 | 1.0e-2 | 30.0 | ok | (anchor region) | correct |
| 1e-3 | 1.0e-3 | 267 | ok | 1.00 / 1.00 / 1.00 | correct |
| 1e-4 | 1.0e-4 | 2.6e3 | **warn** | 0.99 / 1.19 / 1.08 | correct |
| 1e-5 | 1.0e-5 | 2.6e4 | **warn** | 1.22 / 3.2 / 1.88 | correct |
| 1e-6 | 1.0e-6 | 2.6e5 | **refuse** | 4.8 / **793** / 64.5 | mixed |
| 1e-7 | 1.0e-7 | 2.6e6 | **refuse** | **5.5e4 / 7.7e4 / 3.8e5** | wrong |

Repeated with ONE wall-free spacer between the two ring layers (the spacer
inherits both wall sets, as the contract says it does): identical verdict
table, move factors 0.76 / 1.6 at the warned rungs, 23.6 / 25 / 227 at
`delta/Rbig = 1e-6` and **1.1e6 .. 2.8e7** at 1e-7.

**The attribution control.** The same two walls with TWO wall-free spacers
between them, so the `+-1` window never spans both: **no cross-layer cell at
any rung** (`w_min_union = inf` from `delta/Rbig <= 1e-2` down), `q` excess
pinned at 34.93 on every rung, verdict `ok` everywhere, and the answer scales
LINEARLY with `delta` all the way to 1e-7 (8.13e-04 / 8.12e-05 / 8.11e-06 /
2.04e-06). The geometry is harmless; the window's union of it is not.
CONFIRMED.

**No MISS was found.** Across the adjacent ladder, the one-spacer ladder, the
two-spacer control, a phantom-wall attack and a gentle-taper ladder, **the
worst move factor on any rung the contract RETURNED was 7.4x**, against the
library's own 100x attribution bar.

**The refusals are conservative by up to a decade.** At the first refused rung
the answer is sometimes still nearly right: 4.8x (degree 6, adjacent), 7.4x
(gentle taper at `dr = 5e-4`), 23.6x (one spacer). The build says as much —
`_BOR_MIN_ELEM_FRAC = 1e-6` is "the conservative (widest) end of the measured
onset band 1e-7 .. 3e-6" — and this verification quantifies it: refusing a
solve whose answer is within 5-25x of correct is the price of never returning
one that is 1e5x wrong.

### 5.2 The phantom wall — the sharpest false-positive test available

A segment boundary across which `eps` is the SAME on both sides is physically a
no-op, so the stack it describes is EXACTLY the stack without it, and the right
answer is known independently. Placing one a distance `delta` from a real wall
of the neighbouring layer:

| `delta/Rbig` | verdict | `dR` vs the phantom-free truth (sorted) |
|---|---|---|
| 1e-3 | ok | 1.423e-05 |
| 1e-4 | warn | 9.177e-06 |
| 1e-5 | warn | 6.466e-06 |
| 1e-6 | **refuse** | **4.916e-04** |
| 1e-7 | **refuse** | **8.301e-03** |
| 1e-8 | **refuse** | **1.473** |

The first three rungs sit at the mesh-difference floor (~1e-5, confirmed to
shrink with degree: 1.06e-05 / 1.42e-05 / 6.81e-06 / 2.19e-06 at degrees
6 / 8 / 12 / 16). The refusals begin exactly where the answer leaves that
floor. **The refusal is a true positive even on a geometry whose exact answer
is available.** No false refusal.

### 5.3 What the Q bar's quoted margin actually protects — nothing

`_sem_contract.verdict` refuses when `w_min_union_frac < 1e-6` **and**
`q_excess > 1e4`. Ordinary geometry has **no cross-layer cell at all** —
`w_min_union_frac = inf` on all 61 ordinary families measured here so far —
so it can never be refused **whatever its `q_excess`**. The build's headline
two-sided statement for `_BOR_Q_EXCESS` ("0.95 decades above the worst ordinary
geometry measured, 1.20 decades below the mildest rung it must refuse") is
therefore **half vacuous**: the ordinary side is measured on geometries the
conjunction cannot reach.

The Q bar's only operative effect is in the other direction — it can SUPPRESS a
refusal. And there it has a structural scaling the build did not measure. A
manufactured cell of width `w` gives the operator a `1/w^2` stiffness, so
`|q|max ~ C / w` and

    q_excess  =  |q|max / (n_max k0)  ~  C * 1e6 / (2 pi * n_max * (Rbig/lambda))

at the refusal's own geometric edge `w = 1e-6 Rbig`. Measured, at exactly that
edge, degree 6:

| `Rbig/lambda` | `n_max` | product | `q_excess` |
|---|---|---|---|
| 7.6 | 2 | 15.2 | 2.76e+05 |
| 7.6 | 4 | 30.6 | 1.38e+05 |
| 22.9 | 4 | 91.7 | 4.60e+04 |
| 45.8 | 4 | 183 | 2.30e+04 |

an exact `1/product` law over a 12x range. Extrapolating it, the spectral
conjunct stops firing at the geometric edge once `n_max * (Rbig/lambda)`
exceeds about **420** (degree 6; ~665 at degree 8) — beyond which a cross-layer
cell narrower than `1e-6 Rbig` is only WARNED however damaging it is. **I could
not reach the crossing directly**: the SEM mesh grows with `Rbig/lambda`, and
the run that would have reached it timed out at 1200 s. Recorded as an open
item with the fitted law and its four measured points, not as a demonstrated
miss.

### 5.4 The warn edge is entered by an ordinary shallow taper

The build fixed `_BOR_SLIVER_BAND_FRAC = 1e-4` from a census whose binding
geometry was a taper staircase, and concluded that the edge "still WARNS on a
deep enough taper — a ~2,500-slice one" and "can never REFUSE one, because ... a
taper reaches [1e-6 of `Rbig`] only at ~6 million slices". Both statements are
properties of that census's cone, which changes radius by 6 over its height.
A taper's manufactured cell is `dr / n_slices`, so it is set by the cone's
STEEPNESS as much as by the slice count:

| total `dr` | slices | cell / `Rbig` | verdict | warnings | move factor |
|---|---|---|---|---|---|
| 0.5 | 64 | 3.26e-04 | ok | 0 | (anchor) |
| 0.05 | 64 | 3.26e-05 | **warn** | **64** | 2.85 |
| 0.005 | 64 | 3.26e-06 | **warn** | **64** | 6.6 |
| 5e-4 | 64 | 3.26e-07 | **refuse** | — | 7.4 |
| 5e-5 | 64 | 3.26e-08 | **refuse** | — | 437 |
| 5e-6 | 64 | 3.26e-09 | **refuse** | — | 5e8 |

A 64-slice cone whose radius changes by 0.05 on `Rbig = 24` — 0.2 % of the
domain radius, the kind of geometry a taper-ANGLE sweep walks through on its
way to zero — lands inside the degradation band and emits **64 warnings**, and
a slightly shallower one is REFUSED while its answer is still within 7.4x of
correct. The census's 0.99-decade margin is a statement about cone steepness,
not about taper geometry. New gate:
`test_the_sem_warn_edge_is_reached_by_a_shallow_taper_at_32_slices`.

### 5.5 The within-layer liner, and FD

The liner ladder (`w/Rbig` from 1e-2 to 1e-8, degrees 6/8/12/16) reproduces the
build's table exactly: `w_min_union = inf` at every rung (the cell is correctly
attributed to the caller), verdict `ok` down to `w/Rbig = 1e-5`, `warn_own`
from 1e-6, and **never refused on any of the 28 rungs**. CONFIRMED.

`basis='fd'` produces no mesh report, emits no warning, and its answer is
**bit-identical** (hash `1aed33fb7370c3a0`) at `delta/Rbig` = 0, 1e-4, 1e-6 and
1e-7, with closure 1.363e-12 throughout. CONFIRMED.

### 5.6 The warning's shape

Checked against the task's list. The refusal message names the layer, the
manufactured width absolutely and as a fraction of `Rbig`, both walls and which
layers asked for them, the spectral excess, three remedies and the switch. The
degradation warning names all of those except the switch. **It does not fire
once per solve**: it fires once per AFFECTED LAYER, and one geometric
coincidence affects both neighbours — two layers, two identical messages; a
64-slice taper in the band, 64 messages.

---

## 6. Claim 3 -- EME

### 6.1 The defect, reproduced and enlarged

The discontinuity at `Im(eps) = 0+` is reproduced on the build's own window and
on four of my own, and is worse than the build reports.

**Mode counts, the build's full window** (26055.8, 35530.6), `n_scan = 300`,
rebuilt here rather than copied:

| build | Windows / Haswell / t1 | WSL / Haswell / t1 |
|---|---|---|
| PRE, real `eps` | 62 | 62 |
| PRE, `+ i1e-30` | **69** (bottom mode 26137.9321) | **71** (bottom 26065.9580) |
| POST, either | **62 / 62** (bottom 26116.7063, identical to the real-`eps` set) | **62 / 62** |

Exact reproduction, bottom modes included. On the narrowed gate window across
**16 arms**, though, the build's "69 / 71" is two samples of a wider set:

| window | PRE real | PRE `+ i1e-30`, over 16 arms | POST |
|---|---|---|---|
| the gate's (26055.8, 28500.0), `n_scan` 60 | 16 | **{18, 19, 20, 21}** (Katmai 21, WSL/Nehalem 20) | **16 on all 16 arms** |
| mine, `kx0 = 0.21`, `ky0 = 0.11` | 16 | **{16, 17, 18, 19, 20}** | 16 |
| mine, low band (12000, 16000) | 21 | **{33, 37, 39, 41}** (+95 %) | 21 |
| mine, three strips | 15 | {15, 16, 19} -- and `+ i1e-12` also moves it, {15, 16} | 15 |
| mine, `Nx = 48` | 14 | 14 (below the onset, as the build says) | 14 |

**The strongest single number in this verification's EME leg**: over 99 saved
forward-root vectors x 16 arms per build, **PRE has 35 of 99 fixtures whose root
set is decided by the BLAS kernel** (worst cross-arm `|d ky|` = **4.351e+05**);
**POST has 0 of 99** (worst 1.25e-09, which is 6e-15 relative on the mm-unit
fixture).

**Proved out beyond the build's claim**: the defect is NOT confined to
infinitesimal loss. A three-strip layer at `Im(eps) = +1e-3` -- a genuinely
lossy passive material -- has 4 modes arm-decided on PRE (`|d ky|` = 249.7) and
0 on POST.

Flip census on my own fixtures (Windows / Haswell / t1), separating a NEGATED
root (the mode's backward partner) from a CONJUGATED one (the fix's on-cut
tie-break):

| fixture (`Nx`, `k0`, `Im eps`) | PRE prop negated | POST negated / conjugated | worst `abs(d ky)` |
|---|---|---|---|
| 96, 20pi, 0 | 0 | 0 / 0 | 0 |
| **96, 20pi, 1e-30** | **29** | **0 / 29** | 435.14 |
| 96, 20pi, 1e-20 | 25 | 0 / 25 | 435.14 |
| 96, 20pi, 1e-12 | 0 | 0 / 0 | 0 |
| 128, 40pi, 1e-30 | **51** | 0 / 51 | 869.2 |
| 96 in mm units, 1e-30 | 34 | 0 / 34 | 4.35e+05 |
| 48, 20pi, 1e-30 | 0 | 0 / 0 | 0 (onset above `Nx` 64, as claimed) |

**29 flipped propagating modes at the build's configuration -- exact match**,
and 0 post-fix on every fixture and every arm.

### 6.2 What moved, and nothing else

Same-arm PRE -> POST pairs (the only confound-free comparison -- the raw `lam`
bits move with the kernel on both builds):

| population | moved | cause |
|---|---|---|
| strip `lam` (the eigensolve itself) | **0 / 54** | untouched |
| strip forward-root sets | 19 / 54 | all are `Im(eps)` in {1e-30, 1e-20, 1e-12, -1e-6} -- the branch decision |
| `mode_match` r/t | 6 / 14 | only the `Im(qz2)` = -1e-30 / -1e-20 / -1e-12 arms |
| `diffraction_fd` | 1 / 6 | only `n = 1.5 + 1e-30i`; that its move is arm-dependent is PRE's defect |
| `eme_2d_vector` (`strip_vector_modes`, `layer_vector_modes`) | **0 / 11** | the vector site is genuinely unchanged |
| JAX twins (`ref_2d_modes`, `ref_2d_modes_vector`) | **0**, max abs diff 0.000e+00 | the branch is not on the JAX path |

Nothing moved for any reason other than the branch decision. The build's claim
that `cut_band` "computes the identical quantity this line always did" was
fuzzed with 20,000 random spectra over 28 decades of scale, a third of the
imaginary parts planted exactly at the band and at band +/- 1 ULP: **identical
index-set SHA-256 on both builds**. The one exception is an EMPTY `ky` array,
where PRE raises `ValueError` and POST returns `[]` -- an improvement, but the
word "identical" is false for that input (D16).

### 6.3 Gate counts

| | Windows py3.14 | WSL py3.12 |
|---|---|---|
| PRE, the 7 pre-existing EME files | **136 passed** | **136 passed** |
| POST, those plus `test_fix_eme_branch_cut.py` | **143 passed** | **143 passed** |

Nothing moved -- confirmed. The build's "135 on WSL" is not reproducible: this
WSL venv carries jax 0.10.2, so no JAX gate skips. A report error, not a code
defect.

### 6.4 The band's two sides

The fix does not remove the `2|z|` discontinuity; it moves it from `Im z = 0` to
`|Im z| = band`. Measured on 54 fixtures x 32 runs:

* **the side the band must absorb** (the backward error of a propagating root):
  worst headroom **4.72 decades** (the nm-unit fixture at `Im = 1e-12`), 9.23
  decades at the build's own fixture, range 4.72 .. 24.67. The docstring's
  "seven decades" is a fair typical value, not a floor;
* **the side it must NOT absorb** (real physics): **6 of 21 fixtures are inside
  one decade and 6 are NEGATIVE**, worst **-2.74 decades** -- genuinely complex
  roots 550x INSIDE the band. The docstring's "a mode with real loss large
  enough to matter for the cascade sits far above it" is **refuted as written**;
* **what saves it, measured**: for a PASSIVE medium the anti-Hermitian part of
  `D + diag(eps k0^2)` is positive semi-definite, so `Im(lam) >= 0`, so
  `np.sqrt` already returns `Im >= 0` and **no decision is taken at all** (8,704
  modes over 96 passive configurations; the branch fired only at
  `Im(eps) = 1e-12`, and only on backward error). The negative-margin side is
  reachable only from GAIN, a caller-supplied complex `qz2`, or a non-finite
  spectrum;
* **`|exp(i ky h)| <= 1`** holds on both builds for the two scalar sites (max
  1.0000000000000002, 2 ULP, over 54 fixtures x 32 runs). One pre-existing
  exception on BOTH builds: `eme_2d_vector._strip_split_forward` selects indices
  without moving the root, so an on-cut mode with `Im = -9.74e-15` stays in the
  forward set and `|exp(i ky * 100)| = 1.0000000000009737`. Not a regression.

---

## 7. Defects and gaps

Severity uses the campaign's convention: **P1** — returns a wrong number where
a guard is supposed to prevent exactly that; **P2** — a guard misses a
population it claims; **P3** — message, attribution or documentation.

### D1 (P1) — a negligible loss disarms the nodal passivity screen entirely

`bor_solve._stack_is_provably_passive` requires EVERY layer's
`max|Im eps| / max|Re eps| <= _BOR_LOSSLESS_REL_IM = 1e-12`, and the docstring
justifies disarming otherwise: *"on a lossy stack there is no theorem to
violate."* That is true only of the BELOW-unity direction. `R + T <= 1` holds
on every passive stack. The result is a discontinuity at `Im(eps) = 0+` — the
same shape this very build fixed in the EME module:

| `Im(eps)/Re(eps)` of the ring | armed verdict | `max(R+T)` returned | warnings | staggered twin |
|---|---|---|---|---|
| 0 | **REFUSED** | — | — | 1.000000000000 |
| 1e-14 | **REFUSED** | — | — | 1.000000000000 |
| 1e-13 | **REFUSED** | — | — | 1.000000000000 |
| 1e-12 | **REFUSED** | — | — | 0.999999999998 |
| **3e-12** | **returned** | **1.02882** | **0** | 0.999999999998 |
| 1e-11 | returned | 1.02882 | 0 | 0.999999999991 |
| 1e-8 | returned | 1.02882 | 0 | 0.999999990703 |
| 1e-6 | returned | 1.02882 | 0 | 0.999999070256 |
| 1e-3 | returned | 1.02696 | 0 | 0.999070846337 |

The returned number is the SAME 2.9 % violation on both sides of the threshold;
only the guard changes. A "lossless" glass entered as `n = 1.5 + 1e-8i` — or
any dispersion fit with a residual imaginary part — takes the whole guard out.

**Reproducer** (`validation/probe_verify_bor_guards/v5_nodal_holes.py`, and
`tests/unit/test_verify_bor_multilayer_guards.py::
test_a_negligible_loss_must_not_defeat_the_nodal_passivity_screen`): the
shipped gate's stack with `6.0` replaced by `complex(6.0, 6.0*3e-12)`.

**Remedy.** Disarm on GAIN (`Im eps < 0`), not on loss; the super-unity screen
is valid for every passive medium. The shipped gate
`test_a_lossy_nodal_stack_is_never_judged_by_the_passivity_screen` currently
pins the wrong behaviour and would have to move with it.

### D2 (P2) — the screen is one-sided on a stack it has already proven lossless

`_check_nodal_passivity` tests only `max(R+T) - 1`. On a stack it has already
established is lossless and passive, inside a PEC wall, `R + T = 1` is an
EQUALITY — there is no absorption for a deficit to hide in. Measured: **5 of 48
provably lossless nodal rows return `R + T < 1` with no error and no warning**,
the worst at `R + T = 0.537349` (`m = 2`, `Rbig/lambda = 0.5`, `N = 200`, ring
layer) whose staggered twin returns exactly 1. The deficit is the reldiv screen
dropping genuine channels; it is the same defect, in the direction the guard
does not look.

**Reproducer**: `test_the_nodal_passivity_screen_must_be_two_sided_on_a_
lossless_stack`.

### D3 (P2) — the passivity excess does not track the nodal defect

Twenty of the thirty-two rows the screen calls OK return the wrong number of
channels against the closed form (§4.3), including one at `max(R+T) - 1 =
9.98e-10` returning 12 channels instead of 10. Not a regression — the pre tree
does the same thing and worse — but the build's claim that the population below
the warn edge is "the family that is accurate" is not supported. The bar is a
detector of catastrophe, not of correctness.

**Reproducer**: `test_the_nodal_channel_count_is_wrong_where_the_passivity_
screen_says_ok`.

### D4 (P3) — `warn_own` blames a layer that asked for nothing

`verdict`'s `warn_own` branch reads `w_min_frac`, the narrowest cell of the
POST-WINDOW mesh, and its message tells the caller that *"the LAYER'S OWN
segment list asked for"* that cell. The liner's NEIGHBOUR has the cell only
because the `+-1` window put it there — its own segment list is a single
full-radius entry — and receives the identical message. Measured: the liner
ladder emits **two** `warn_own` messages, blaming layers `[0, 1]`, where layer
1 is `add_layer(0.5, eps=1.21)`. The stated reason for never REFUSING a
`warn_own` cell ("you prescribed the geometry and the library does not overrule
it") does not hold for that layer.

**Reproducer**: `test_warn_own_is_only_reported_for_the_layer_that_prescribed_
the_cell`.

### D5 (P3) — the SEM warning fires once per affected layer, not once per cause

One wall coincidence produces two identical messages (both neighbours); a
64-slice taper inside the degradation band produces 64. See §5.6.

### D6 (P3) — documentation and bar-statement defects

* `bor_solve.solve`'s inline comment says the nodal gap is **8.17 decades**;
  the docstring above it says 6.81, and 4.4336e-09 -> 2.8819e-02 is 6.81. The
  changed gate `test_structured_stack_energy_floor_nodal_is_now_REFUSED`
  repeats the 8.17 and adds "6.71 decades above the healthy ceiling", which is
  neither.
* The build's SIGNAL-side band margins are quoted from a statistic larger than
  their population's minimum (§3.3): 0.98 decades stated, **0.38** measured.
* `_BOR_Q_EXCESS`'s "0.95 decades above the worst ordinary geometry" protects
  nothing, because ordinary geometry has no cross-layer cell and cannot be
  refused at any `q_excess` (§5.3).
* `_orient.forward_orient`'s inline comment justifies `|Im q| == 0` always
  classifying propagating by "the band is strictly positive (its floor is
  1e-300 times 1e-9)". That describes the REMOVED band; the current floor is
  `k0`. The conclusion is still right, the reason is stale.
* `pyproject.toml` still reads `version = "5.44.0"` and the CHANGELOG entry is
  under `[Unreleased]`: nothing in the tree declares 5.45.1.

### D7 (P2) — the release sweep's documentation walkers are inert on this branch

Eight of the twelve skips in the 1288-test census/walker sweep are the
changelog-integrity walkers (V12.2, V12.4, V17.1-17.3, changelog-content,
source-line-citation). They key on the top **versioned** CHANGELOG block, which
is `## [5.45.0]`; this branch's entry is under `## [Unreleased]` and there is no
`## [5.45.1]` block, with `__version__` still `5.44.0`. Nothing therefore
enforces this branch's self-citation, file-count or audit-closure claims, and
the sweep's green result should not be read as covering them. Cutting the
`## [5.45.1]` block (which the release needs anyway) re-arms them; they should
be re-run after that, before the tag.

### D8 (P2) — two shipped bars carry less margin than the cross-build spread they sit above

Re-measured on five kernel/build arms:

| arm | `test_near_cutoff_closure` (bar 1e-8) | ladder `worst` (bar 1e-6) |
|---|---|---|
| Win Haswell t1 | 2.101e-09 — **4.8x** | 1.965e-07 — 5.09x |
| Win Haswell t4 | 3.110e-10 — 32x | 6.198e-08 — 16.1x |
| Win Nehalem t1 | 1.107e-09 — 9.0x | 2.924e-07 — **3.42x** |
| Win Katmai t1 | 7.434e-10 — 13.5x | 3.783e-08 — 26.4x |
| Win Sandybridge t1 | 1.570e-10 — 64x | 3.213e-08 — 31.1x |
| WSL Haswell t1 | 2.584e-11 — 387x | 1.082e-07 — 9.24x |

The measured cross-arm spread is **81x** on the single rung and **9.1x** on the
ladder, against smallest margins of **4.8x (0.68 decades)** and **3.42x (0.53
decades)**. Both bars pass on every arm run here, and the boundary sits above
the whole measured envelope rather than inside it — so neither is per-build in
the strict `docs/TESTING_STANDARDS.md` sense. But they are the two most likely
to refuse a tag on an unseen kernel, and two supporting statements are already
arm-specific: the docstring's "worst rung 1.9655e-07" is a Windows/Haswell/t1
reading (**Nehalem reads 2.924e-07, 1.5x worse than the quoted worst**), and the
closure grows monotonically with rung depth (1.00e-07 at `e = 19`, 1.97e-07 at
`e = 20`), so the bar holds partly because the ladder stops at `e = 20`.
Recommend re-deriving both from the measured envelope, with the date and the
arms in the comment, rather than from one arm's reading.

---

## 8. Runs

Logs in `validation/probe_verify_bor_guards/runs/`. File set:
`tests/unit/test_bor*.py test_audit*bor*.py test_niche*bor*.py test_eme*.py
test_fix_bor*.py test_fix_eme*.py` — 16 files on POST, 14 on PRE (the two
`test_fix_*` files do not exist there). `lumenairy.__file__` was printed and
checked on every arm; no `pip -e` leakage.

| # | build | requested | **loaded** | thr | passed | failed | errors | skipped |
|---|---|---|---|---|---|---|---|---|
| 1 | Win POST | HASWELL | **Haswell** | 1 | **271** | 0 | 0 | 0 |
| 2 | Win POST | NEHALEM | **Nehalem** | 1 | **271** | 0 | 0 | 0 |
| 3 | Win POST | PRESCOTT | **Katmai** | 1 | **271** | 0 | 0 | 0 |
| 4 | Win POST | SANDYBRIDGE | **Sandybridge** | 1 | **271** | 0 | 0 | 0 |
| 5 | Win POST | HASWELL | **Haswell** | 4 | **271** | 0 | 0 | 0 |
| 6 | Win PRE | HASWELL | **Haswell** | 1 | **201** | 0 | 0 | 0 |
| 7 | WSL POST | *(unset)* | **Haswell** | 1 | **271** | 0 | 0 | 0 |
| 8 | WSL POST | *(unset)* | **Haswell** | 4 | **271** | 0 | 0 | 0 |
| 9 | Win POST, the three JAX-guarded files | HASWELL | Haswell | 1 | **22** | 0 | 0 | **0** |
| 10 | Win POST, census/walker/dispatcher/public-API sweep | HASWELL | Haswell | 1 | **1288** | 0 | 0 | 12 |

The accounting reconciles exactly: 201 (PRE) + 68 (the two new gate files,
61 + 7) + 2 (net in `test_bor_solve.py`, one gate replaced by three) = 271.
**Zero failures and zero errors on every arm**, and every tail carried a real
summary line.

Three things the run matrix itself found:

* **WSL has JAX 0.10.2** (Windows 0.11.0), contrary to the build report's "not
  installed": arms 7 and 8 report 271 passed / 0 skipped, identical to Windows,
  and arm 9 confirms the JAX-guarded BOR/EME files RUN rather than skip.
* **`-p no:randomly` is a no-op** — `pytest-randomly` is installed in neither
  environment, so none of these arms supports an order-independence claim.
* **8 of arm 10's 12 skips are the changelog-integrity walkers** (V12.2, V12.4,
  V17.1-17.3, changelog-content, source-line-citation). They key on the top
  VERSIONED block, `## [5.45.0]`; this branch's work is under `## [Unreleased]`
  and there is no `## [5.45.1]` block anywhere, with `__version__` still
  `"5.44.0"`. **The "1288 passed" therefore does not cover this branch's
  self-citation, file-count or audit-closure claims** — see D7.

New gates in `tests/unit/test_verify_bor_multilayer_guards.py`: **5 passed, 3
xfailed, 68.7 s**, slowest gate 32.1 s (under the 60 s shard cap); node ids
spliced into `.test_durations`, sorted and JSON-valid (12,853 entries).

**ruff.** `ruff check lumenairy/ tests/` — **All checks passed** on both trees.
Naming `validation/probe_verify_bor_guards/` explicitly overrides
`pyproject.toml:296`'s own `extend-exclude = ["validation/", ...]`, which is
why that path reports style errors for this verification's probes exactly as it
does for the build's (`validation/` carries 848 further pre-existing errors and
has never been ruff-clean by design); with `--force-exclude` the full command
passes. Not a branch defect either way.

---

## 9. Durability of the new gates (task F)

Every numeric literal inside an `assert` in `test_fix_bor_multilayer_guards.py`
(61 gates), `test_fix_eme_branch_cut.py` (7) and the three changed
`test_bor_solve.py` gates was enumerated by AST and its quantity re-measured on
**16 arms** (four loaded kernels x two thread counts x two builds).

**The headline: all 74 node ids are green on every arm, and no constant quoted
in a comment failed to reproduce.** What the re-measurement did find is a
systematic absence of premise-gating and four bars inside 0.7 decades.

### 9.1 Gates that would FAIL, not skip, on a pathology-absent arm

Exactly **one of 61** gates restates rather than asserting
(`test_a_within_layer_liner_is_warned_and_never_refused`). There are no
`pytest.skip`s (correct per rule 4) and no other restatement. Forced-premise
probes, applied through a pytest `-p` plugin that rebinds module constants
without editing the library:

| perturbation (what an arm without the pathology looks like) | result |
|---|---|
| the legacy nodal cascade does not blow up | **7 gates FAIL**, 0 skip, 0 restate |
| the mildest broken nodal row improves 2.9x | **2 FAIL** |
| the SEM sliver injects no hot spectrum | **2 FAIL**; the liner gate correctly RESTATES |
| the pre-fix BOR band restored | **3 FAIL** (decisive) — and `test_no_forward_mode_of_a_lossless_layer_carries_backward_flux` **passes VACUOUSLY**, its `prop` set having emptied |
| the pre-fix EME pin restored | **4 of 7 FAIL** (decisive); the 3 that pass are the off-cut/structural ones, correctly |

### 9.2 The four thinnest bars, re-measured

| gate | bar | measured envelope over 16 arms | margin |
|---|---|---|---|
| `test_bor_solve::test_the_nodal_refusal_switch_restores_the_previous_number_exactly` | `max\|E-1\| < 0.05` on the DEFECTIVE nodal answer | 0.0382243, spread 1.0x | **1.31x = 0.12 dec** |
| `test_near_cutoff_channel_count_is_stable_over_the_ladder` | `worst < 1e-6` | 2.4651e-08 .. 4.3513e-07 (17.7x) | **2.30x = 0.36 dec** |
| `test_the_nodal_passivity_bar_is_two_sided_on_this_build` | `mild > 10 * bar` | 0.0288187, spread 1.0x | **2.88x = 0.46 dec** |
| `test_band_two_sided_population` (deep-cutoff noise) | `worst_cut < band/2` | 5.3506e-12 .. 1.0333e-09 (**193x**) | **4.84x = 0.68 dec**, eroding to **4.55x** at `N = 360` |

### 9.3 The sharpest finding — a bar the build's own report already crosses

`test_near_cutoff_channel_count_is_stable_over_the_ladder` says in its docstring
*"Over the whole near-cutoff ladder the count must now be ONE number"* and bars
the worst closure at `1e-6`. The build report's own section 4 table records the
same ladder reaching **1.2716e-06 on Windows / PRESCOTT -> Katmai / 1 thread** —
above the bar. Reproduced independently here:

```
m=2, N=120, rungs 8..21, win/Katmai/t1 : counts {3}, worst 1.2716e-06  -> FAILS the bar
m=0, N=120, rungs 8..21, win/Katmai/t1 : counts {3}, worst 3.7831e-08  -> passes
```

The gate escapes only because it hardcodes `m = 0`. Two further refutations of
the "whole ladder" phrasing, on the default arm: at `e = 23` the closure is
already **1.1808e-06** (over the bar), and at `e >= 24` the count drops to 2 —
legitimately, because `channel_core`'s `_BOR_CHANNEL_REAL_FLOOR = 1e-6` in `qn`
bounds the ladder from below, a justification that appears nowhere in the gate.
**The 2-D peer's round-4 sample-vs-family correction was applied to the SEM warn
edge and to `test_ordinary_geometry_census`; it was not applied here.**

### 9.4 Two knife-edge rungs, and four vacuous assertions

The delta ladder's verdict table `ooowwRR` is identical on all 16 arms, but two
of its seven entries are decided by a cancellation residue rather than by a
margin: `w_min_union_frac` reads `9.9999999999988987e-05` against the `1e-4`
bar (813 ULP below) and `9.9999999999174816e-07` against `1e-6`. Both come from
`(6.0 + d) - 6.0` happening to land below the nominal value; had it landed
above, the strict `<` flips. Arm-invariant and numpy-version-stable, so not
per-build — but the docstring's "the geometry is kernel-EXACT (spread 1.0000x)"
conflates kernel-stability with margin. Choosing `delta` off the bar (3e-5,
3e-7) removes it.

Six of the nine `test_ordinary_geometry_census` parametrizations measure
`narrowest = inf`, so `assert narrowest > 3.0 * _BOR_SLIVER_BAND_FRAC` asserts
nothing on them, and `test_no_forward_mode_of_a_lossless_layer_carries_backward_
flux` has no non-empty guard (it passed vacuously under the forced band
perturbation). Both want an `assert n_checked >= ...`, which
`test_band_two_sided_population` already has.

### 9.5 Runtime and `.test_durations`

Two `--durations=0` runs on Windows at 1 thread, both under contention from the
other verification legs: `74 passed in 666.4 s` / `644.9 s`. Best-of-two
slowest: `test_ordinary_geometry_census[mesh-12]` **58.99 s**, i.e. **98 % of
the 60 s shard cap** (it crossed to 74.97 s under load), and
`test_the_taper_staircase_is_ordinary_at_every_slice_count[64-8-2.0]` 53.95 s
(64.83 s under load). Under the cap, with no headroom.

`.test_durations` is sorted and JSON-valid (12,845 entries before this
verification's splice), with **three defects**:

1. **9 of the 74 new node ids are not spliced in** (all sub-second structural
   gates); `pytest-split` will give them the file average.
2. **One STALE key** — `tests/unit/test_bor_solve.py::test_structured_stack_
   energy_floor_nodal`, the renamed gate's old node id, is still present. This
   is the stale-`.test_durations` trap by name.
3. **One entry is 7.6x wrong and is the only entry in either file over 60 s**:
   `test_mode_count_is_build_independent_under_an_infinitesimal_loss` is
   recorded at **167.41 s**, which is the PRE-narrowing cost the file's own
   comment says was avoided ("that pair costs 167 s ... Measured 25 s");
   measured here at 16.9-22.0 s. Several entries are mis-scaled the other way
   (`test_the_kernel_accepts_an_explicit_namespace_and_does_not_sniff` recorded
   17.90 s against 0.61 s measured, 29x), consistent with entries captured
   under unpinned multi-threaded BLAS.

### 9.6 Gates proved out

Bidirectionally, the following could not be broken and are exemplary:
every step-1 grep/identity gate (they pin the running build's own line
numbers); `test_no_exact_zero_branch_pin_survives_in_the_eme_package` (AST, not
grep, so the comments quoting the removed pin are invisible to it);
`test_band_two_sided_population`'s ordinary-noise side (3.55 decades, 4.4x
cross-arm spread); `test_widening_the_band_is_harmless_because_the_two_rules_
agree_there` (387 modes, 0 disagreements, `relmin = 0.11967609435` to thirteen
digits on all 16 arms); the inverse census's two condition bars (min rcond
`2.033567832716004e-05` **bit-identical on 16 arms and both builds**, 4.31
decades of room); and all four bit-identity gates (SHA-256 of the `R` bytes
identical on every arm and both builds). The whole EME file, except its
`real.size >= 10` premise, states its observables as **equalities between two
arms of the same build** — precisely the build-free restatement the standards
ask for, and it is the model the BOR file should have followed.

One cross-arm note that strengthens the EME claim: restoring the exact-zero pin
by monkeypatch gives 16 vs **18** (Haswell, Nehalem, Sandybridge), 16 vs **21**
(Katmai) and 16 vs **19** (WSL Haswell). The build's comment "16 and 18" is one
arm's reading; the conclusion holds on all of them.

---

## 10. Further defects, from the durability audit

### D9 (P2) — `test_near_cutoff_channel_count_is_stable_over_the_ladder` is sample-scoped and its bar is already crossed

See §9.3. The gate claims a ladder property, asserts it on `m = 0` only, and
the build's own report records `m = 2` reaching 1.2716e-06 against the gate's
`1e-6` bar on Katmai. Reproduced. **Remedy**: parametrize over `m` and re-derive
the bar from the measured envelope over `m` and the kernels (worst measured
here: 1.2716e-06, so the bar wants to be ~1e-5), or narrow the docstring to the
`m = 0` claim it actually makes. Either way the `_BOR_CHANNEL_REAL_FLOOR` bound
on the ladder's depth belongs in the gate.

### D10 (P2) — two shipped gates pin a pathology's magnitude with no margin

* `test_the_nodal_refusal_switch_restores_the_previous_number_exactly` asserts
  `max|E - 1| < 0.05` on the **defective** nodal answer, measured 0.0382243 on
  all 16 arms — **1.31x**, the tightest bar in the three files. It fails if the
  nodal defect gets 31 % worse, and its message would say nothing about why.
* `test_structured_stack_energy_floor_nodal_is_now_REFUSED` asserts the string
  `"1.02882"` appears in the refusal text. `"%.6g"` of a quantity whose
  measured envelope is 1.0288186915310455 .. 1.0288186918431719 does give
  `"1.02882"` on every arm — but this is shape S1 (a magnitude-ratio defect
  pin), and the forced-premise probe shows it failing under the smallest
  perturbation that heals one row. **Remedy**: assert the refusal, the named
  switch and the named remedy (which it already does), and drop the magnitude
  literal — or restate it as "the message quotes a number above the bar".

### D11 (P3) — `.test_durations` hygiene

Three defects, §9.5: nine missing node ids, one stale renamed key, and one
entry 7.6x too large (the only entry in either new file over the 60 s cap,
recording a cost the file's own comment says was designed away).

### D12 (P3) — vacuous and knife-edge assertions

Six of nine `test_ordinary_geometry_census` parametrizations assert nothing;
`test_no_forward_mode_of_a_lossless_layer_carries_backward_flux` passes
vacuously when the band shrinks; two delta-ladder rungs are decided 813 ULP
from their bar. §9.4.

---

### D13 (P2, NEW IN THIS BUILD) -- `_branch.cut_band`'s literal 1.0 floor is unit-dependent

`cut_band`'s docstring justifies flooring at a literal 1.0 with "``ky`` here is
DIMENSIONLESS (the EME modules work in ``k0``-normalized units)". That is
false. `strip_x_modes` assembles `d2/dx2 + eps k0^2` on a spacing `Lx / Nx`, so
`lam` carries 1/length^2 and `ky` carries 1/length; `mode_match` forms
`exp(i qz depth)`, which is dimensionless only because `qz` is 1/length. `k0`
is a free argument carrying units, not a normalisation. The floor therefore
engages whenever `max|ky| < 1` in the caller's units -- the NORMAL case for a
sub-micron cell specified in nanometres.

**This is a regression**: the pre-fix exact-zero pin has no scale and is
unit-invariant. Minimal reproducer, one 1 um cell, 96 points, lambda = 1550 nm,
`eps_hi = 12 - 1e-6i` (weak gain), expressed in um and in nm
(`validation/probe_verify_bor_guards/ve_repro.py`, bit-identical on both
builds):

```
POST:  um   max|ky| = 191.85     band = 1.918e-07
       nm   max|ky| = 0.19185    band = 1.000e-09   (the floor makes the band
                                                     5.2x wider in physical terms)
       3 of 96 modes come back on a DIFFERENT ROOT in nm than in um
       worst |d ky| = 26.4382 /um   (mode 95: um -> -13.2190852, nm -> +13.2190852)
PRE:   0 of 96 modes differ, worst |d ky| = 7.96e-13
```

A unit-scaling sweep (`Lx -> s Lx`, `k0 -> k0/s`, `qz2 -> qz2/s^2`,
`depth -> s depth`) flips 15 / 2 / 5 modes at `s = 1e3` for `Im(eps)` =
-1e-7 / -1e-6 / -1e-5 on POST and 0 on PRE; `mode_match` inherits it
(`T00 = 0.875371713142` against `0.875368963078` on the same slab in two unit
systems, against 3e-15 on PRE).

**Remedy**: floor at `k0` -- exactly what `elements/bor/_orient.orient_band_
scale` does, for exactly this reason (audit P2-06) -- or drop the absolute
floor and use `max|z|` alone. The production surface is gain media and
caller-supplied complex `qz2`, which is why it is P2 and not P1.

### D14 (P3, new in this build) -- the band is a GRID quantity, so it absorbs real physics

`max|ky|` for an FD strip is set by the grid (`~ 2 Nx / Lx`), not by the
medium, so `band = 1e-9 * max|ky|` is a grid quantity. A mode whose own `|Im|`
falls under it is CONJUGATED, returning a value that is not a square root of
its own `ky^2` (the error is `2|Re z|`, not `2|Im z|`). Gain scan at `Nx = 96`,
`k0 = 20 pi`, identical on four arms across both platforms:

| `\|Im eps\|` | 1e-12 .. 1e-8 | 1e-7 | 1e-6 | 1e-5 | 1e-4 | >= 1e-3 |
|---|---|---|---|---|---|---|
| propagating modes conjugated, POST | **63 / 96** | 15 | 13 | 3 | 1 | 0 |
| PRE | 0 | 0 | 0 | 0 | 0 | 0 |

Extreme form: a single outlier poisons the whole decision. Adding an unrelated
eigenvalue of 1e300 to the spectrum flips the returned roots of modes carrying
0.5 % loss (`(-100 + 0.5i)` -> `(+100 + 0.5i)`) on POST and not on PRE; `nan`
puts everything off the cut, `inf` puts everything on it. Reproducer:
`ve_poison.py`. Same remedy as D13.

### D15 (P3, PRE-EXISTING on both builds) -- `diffraction_eme`'s answer is still not build-stable

The fix stabilises the mode SET, not the driver. PRE, a 1e-30 no-op replaced all
9 retained modes (worst `|d qz2|` = 697.1); POST returns the identical set to
1.2e-09. But the end-to-end energy still swings **314.976 / 80.716 / 285.386 /
132.889** across `Im(eps)` = 0 / 1e-30 / 1e-20 / 1e-12 **on both builds with
identical `qz2`**. The mechanism is downstream of the branch (`mode_field`'s
near-singular `G` null vector plus the `eig`-vs-`eigh` eigenvector phase),
inside the module's documented non-convergent structured-layer regime. Claim 3
makes no claim here; recorded so it is not mistaken for one.

### D16 (P4) -- "computes the identical quantity" is false for an empty spectrum

`cut_band([])` returns the bare band on POST where the old inline expression
raised `ValueError: zero-size array to reduction`. An improvement; the wording
is not accurate.

## 11. Other census observations

The ordinary-geometry census re-run here (**146 of 160 rows complete at the
time of writing**; the outstanding rows are the three deepest
`elements_per_segment = 32` arms, whose meshes are large and whose shallower
twins already carry the finding below: ring gratings at
four `k0` x four `m`, ring counts 1..8, uniform, coincident walls, a wall-free
spacer, a lossy metal ring, an anisotropic layer, an nm-unit fixture, and hp
refinement at `elements_per_segment` 2..32 with grading on and off, all at
degrees 6/8/12/16) trips nothing — zero warnings, zero refusals — but two rows
bound the build's stated numbers:

* **the ordinary `q_excess` ceiling is not 1134.2.** `elements_per_segment = 16,
  grade = True, degree = 16` on an ordinary two-layer ring stack reads
  **2215**, and `k0 = 0.8, m = 5, degree = 16` on a plain ring grating reads
  667.2 — both above the build's quoted non-taper maximum of 164.4, and 2215
  cuts the quoted 0.95-decade margin to **0.65**. (Per §5.3 that margin
  protects nothing, but the number should be right.)
* **graded hp refinement makes cross-layer cells.** `elements_per_segment = 16,
  grade = True` produces `w_min_union_frac = 8.006e-04` on ordinary geometry —
  within **8x** of the warn edge — because the Chebyshev-Lobatto grading puts a
  narrow sub-cell at each end of every interval, and those ends are walls of
  different layers. The census should sweep that knob; the build's did not.

---

## 12. Ship recommendation

**SHIP 5.45.1, with D1 fixed first.**

Everything the build set out to do, it did, and this verification reproduced
each of it independently and in several places more strongly than the build
claims. The orientation consolidation moves no number on 58 fixtures on two
builds. The band change removes a defect that this verification watched move a
channel count and a lossless energy closure with the BLAS kernel on a kernel
the build never ran. The EME pin removes a mode-count discontinuity that reads
18 on Windows and 19 on WSL. The SEM contract refuses only answers that are
wrong, returns nothing wrong by more than 7.4x, and its attribution control is
exact. The census hooks are inert.

Three things should be taken before the tag.

1. **D1 (P1) — do not ship as it stands.** A guard that a `1e-12` imaginary
   permittivity switches off is not a guard, and the user's order this wave
   serves was *"I don't want any of these errors to show up in any of my
   solvers for multilayers."* The fix is one predicate (`Im eps < 0` rather
   than `|Im eps| > 1e-12`) plus moving one shipped gate. **D2** is the same
   predicate's other half and belongs in the same edit.
2. **D9 (P2) — a shipped gate's bar is already crossed by the build's own
   published number** (1.2716e-06 on Katmai against a `1e-6` bar), and passes
   only because the gate hardcodes `m = 0`. This is precisely the shape that
   refused five tags in the v5.35.x campaign; it should be re-derived from the
   measured envelope before the release matrix runs, not after it fails.
3. **D7 / D11 (P2/P3) — the release gate's own instruments are inert or
   stale.** Eight changelog-integrity walkers skip because there is no
   `## [5.45.1]` block and `__version__` is still `5.44.0`; `.test_durations`
   carries a stale renamed key and one entry 7.6x too large. Both are cheap and
   both are part of cutting the release anyway.

**D10, D3-D6 and D12** are message, magnitude-pin and documentation work; they
should be taken, but they do not block.

Nothing in the library's numerics needs to change for this branch to be
correct — the five steps do what they say. What needs to change is one
predicate and two or three test constants.

---

## 13. What could not be verified

* **The Q bar's miss region (§5.3) at the crossing.** The `1/(n_max *
  Rbig/lambda)` law is measured over a 12x range with an exact exponent, but the
  case where `q_excess` actually falls below 1e4 at the geometric edge needs an
  optically large SEM solve that exceeded the time budget.
* **The `k0` floor (§3.4).** Unreachable through `BORStack` on any grid that
  resolves the wavelength; the choice cannot be exercised.
* **The CI's ZEN kernel.** Same limitation the build records: `ZEN` aliases
  Haswell here. This verification adds **Sandybridge**, which the build did not
  run and on which the pre-fix defect is present (and worst: 7 backward modes).
* **The PML and anisotropic Class-C populations.** PML is unreachable on the
  staggered path by construction; the contract makes no claim there.
* **`eme_diffraction`'s unit-system dependence** (§6) — no fixture here reaches
  the regime where the literal 1.0 floor binds.
* **A full independent accuracy oracle for the SEM answer.** As in the build,
  Class C's wrongness rests on continuity, on the separated control, and here
  additionally on the phantom-wall construction, not on an external solver.

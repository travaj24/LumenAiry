# VERIFY-WP-C3 — independent adversarial verification of `transport='collins'` as the carrier chain's default

Branch under test `feat/c3-collins-default`, HEAD **4d87ff48**, base **49ddf4bd**.
Verification branch `verify/c3-collins-default`, worktree `C:/tmp/lum_vc3`.
Package diff: `lumenairy/propagators/carrier.py` only, +334/-75.

Everything below was RE-MEASURED on this machine with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, on 2026-09-20, on BOTH builds: **WIN-py3.14** (Windows 11, py3.14.6,
numpy 2.4.4, jax 0.11.0, cupy 14.0.1 with a broken cuFFT) and **WSL-py3.12**
(py3.12.3, numpy 2.4.6, jax 0.10.2, no CuPy).  Trees are pinned by setting the
child's **cwd AND `PYTHONPATH`** to the tree root and asserting
`lumenairy.__file__` under it; the PRE tree is this verification's own
`git archive 49ddf4bd`.  No number here is read from the package's report.

---

## 0. SHIP RECOMMENDATION

**DO NOT SHIP AS IS.  Two defects must be closed first; the physics is sound
and the engineering is good.**

The transport work is right.  The complementary selection is real, the way
back is byte-exact, the three fallback fixes are genuine and reproduce to
every printed digit, and the readout's quadrature resolution is the correct
response to VERIFY-WP-B4 F1.  What fails is the *accounting*: the package's
central compatibility claim is measurably false, and a documented contract
was left in the opposite state from the code that implements it.

| # | must close before the tag | why |
|---|---|---|
| **D5** | the CHANGELOG / Migration-Guide claim that **no public call that worked on 5.48.1 raises on 5.49.0** except a JAX one | MEASURED false on both builds: **11 of 12** ordinary chain configurations that return at 49ddf4bd raise `RuntimeError` on the flipped default.  A caller upgrading on the strength of that sentence gets a crash. |
| **D6** | the leg with a **resolved FLAT reference** has no fallback and runs an aliased chirp-Z on the new default, **far from any focus** | a converging grid ladder shows the default DIVERGING (3.0x, 3.2x, 3.2x in r2m at N = 256/512/1024) where `'sziklas'` converges.  This is a correctness defect, not a documentation one. |

Four more should close with them (**D1**, **D2**, **D3**, **D4**) because they
are cheap and each one misdirects a reader of the shipped source.  The
remaining items are recorded for the follow-up.

The recommendation is **not** "revert the flip".  It is: correct the two
claims, close the flat-reference hole, and re-run the blast set.  The
selection mechanism itself survived every attack this package made on it.

---

## 1. VERDICT TABLE

| # | claim | verdict | the measurement that decides it |
|---|---|---|---|
| 1 | the F1 finding: one-step readout 9017 vs a true 1.2359 (x7295), power conserved 2.8289e-05, K1 = 82.36 | **PARTIAL** | The substance is CONFIRMED by a SECOND, independent, CONVERGED quadrature: a sinc-upsampled direct Fresnel sum over the exit field (F = 1 … 256) converges to on-axis **0.9234085468117397** exactly as the fine pitch brings K1 under 1, and agrees with the library's free leg to **\|ratio\| = 1 − 2.2e-08, arg = 2.0e-08 rad** — absolute phase included.  So 9017 IS wrong and the answer IS order 1.  But **1.2359 is not reproducible** (every spelling of the free leg reads 0.9234085), **"on-axis" is actually the window MAXIMUM** (the centre samples are 0.92565 and 1971.39, and the Sziklas max sits at the window CORNER), and the ratio is **9748** (peak/peak) or **2135** (centre/centre), not 7295.  K1 = 82.36047195201569, the power figures and the pitches reproduce to every digit. |
| 2 | the readout resolves on `_collins_readout_k1 <= 1`; the three keys only on `'collins'` | **PARTIAL** | Mechanism CONFIRMED completely: 9/9 archive keys bit-identical with the three keys absent on `'sziklas'`; `z == 0` returns `inf` and routes to Sziklas; the reason string names the branch that ran, verified by bit-identity rather than by reading the key, on 5 fixtures across both routes and all three reasons; four mutations of `_publish_readout_route` are each caught by the branch's own tests.  **But the bound at 1 is a CHOICE, not a measured cliff**: on a dial fixture where only K1 moves, relL2 vs an analytic Gaussian is flat at the fixture floor (5.3–9.3e-06) from K1 = 0.1 all the way to 1.2, the knee is near 1.3, and K1 = 1 buys +25 % over the floor.  The shipped "above 1 … every quadrature … returns the wrong field, and the error is not small" is true at K1 = 82 and false near 1. |
| 3 | the oracle ladder, 22 readings, ratio 1.00000; floor 3.87e-05 | **PARTIAL** | Every printed digit of the ladder reproduces against an independently written and independently VALIDATED analytic Gaussian (paraxial-Helmholtz residual clean dx⁴; 7.5e-16 against an oversampled transfer-function propagation), on both builds and on a second fixture of our own.  **The floor NUMBER is refuted**: a converged Gauss–Legendre quadrature puts it at **6.52e-05**, and the branch's OWN `_truncation_floor()` helper returns **6.3956e-05**; 3.87e-05 is numerically the ladder's own focus-cell reading, which makes "within 13 % of that floor" circular and false for the diverging cell (43 % below).  **The floor's SUBSTANCE is confirmed by an experiment the report did not run**: widening the window at the same pitch drops the floor to 7.11e-10 and 9.09e-17 and, under `gap_kernel='fresnel'`, the Collins column falls with it over five decades. |
| 4 | 42/42 keys identical archive-to-archive; the two public readouts and `final_leg='exact'` do not move | **PARTIAL** | An independent **103-key** harness reads **103/103 identical on both builds**, and the adversarial control (base with `'sziklas'` NAMED vs base with nothing named) is also 103/103 — so the way-back framing is sound.  The two public readouts CONFIRMED unmoved (0 of 18 readout keys, 0 of 5 helper keys).  **`final_leg='exact'` DOES move** (its gap legs still do): 24.7x–557x in peak on ordinary relays.  Every public entry point that moves has the one-keyword way back — see §3. |
| 5 | three fallback fixes (collimated NaN, astigmatic, past focus) | **CONFIRMED** | All three reproduce on the base tree to every claimed digit (0/1048576 finite on `dx=nan`; K1 = 1.0221679019993162 / K3 = 2.7997265892002727; windowed r2m 3333.377597937928 and 6981.939960374399 against an INDEPENDENTLY derived analytic 1105.7044418303942 and 2211.403457250719), and the branch fixes all three bit-identically to `transport='sziklas'`.  The promised triple is exact, not approximate.  **No recursion is possible**, and the keyword is load-bearing rather than decorative: deleting `transport='sziklas'` from the fallback gives `RecursionError` on every fallback geometry. |
| 6 | the three internal call sites pinned to `'sziklas'` | **PARTIAL** | Reproduced to 13 digits (half-width **8.533333333333342 um** against amplitude radius **4.639119050017086 um**).  **But the physics adjudication goes AGAINST the pin.**  Against a converged dense Fresnel oracle (self-consistency 4.47e-05, convergence 5.14e-07) the COLLINS standoff leg reads relL2 **4.73e-05** and the pinned Sziklas one **2.40** (5.14x in peak).  Over 5 geometries x 6 standoffs: pinned **6/30 RAISE** and three more return silently wrong (1.1e-01, 1.3e-01, 4.2e-01); unpinned **0/30 raise** and relL2 <= 3.8e-03 everywhere.  Two of the four sites (the leg's own fallback, and the two `!= 'collins'` arms) are correct and one is mandatory; the readout's standoff leg is the one that should take the default with the containment test restated.  **The AST census fires for 2 of 6 placements** — see D9. |
| 7 | the stop-plane keys SELECT, two-sided | **CONFIRMED** | Base refuses, branch does not.  On a K1 <= 1 fixture (K1 = 0.9672040627727245) the `'collins'` baseline really takes the one-step route and differs from `'sziklas'`; naming `standoff` flips it back to **bit-identical to `'sziklas'` with the same key**, `reason='stop_plane_key'`, `readout_route_k1 is None`, and K1 is genuinely not computed (a call counter reads 1 on every no-key run and **0** on every key run).  The stop-plane key correctly WINS over the K1 test.  Note the branch's own b4 test asserts the field arm at `final_distance=8e-3`, where K1 = 82.36 and both routes coincide, so that one arm is vacuous; its route/reason/`k1 is None` arms are not. |
| 8 | the CuPy arm: one implementation, three censuses, <= 0.71 ULP, the broken-cuFFT decision | **PARTIAL** | Every NUMBER reproduces and survives harder stress than the package applied (positive R, an offset, 2.5e5 rad phase, a 0.31/0.22 tilt; `_exact_dispersion_phase` bitwise 0.0 on all six cells).  ONE implementation CONFIRMED (no `_cupy`/`_jax` twin of any Collins helper).  The substituted-module test CONFIRMED, including on a build with no CuPy, with no leakage.  **Two structural claims REFUTED**: the public CHAIN with a device array dies with **exactly the implicit-conversion `TypeError`** the report says must not happen (`_chain_entry_congruence_stats:6377`), and the host-demotion census **misses 9 of 13** injected demotions, 4 of them real device breaks.  The device run is confirmed unmeasurable here for the stated reason. |
| 9 | the design-121 gate; K1 = 0.99958 sits 4e-4 below the bar | **CONFIRMED on the numbers, FRAMING REFUTED** | All six rows reproduce to every printed digit on Windows and N = 512 / 1024 to every digit on WSL including both K1 values; zero Kelly warnings.  Measured K1 spread over **13 build settings** (2 builds x 6 OpenBLAS coretypes x 3 thread counts) is **3.885780586188048e-15 = 17.5 ULP of 1**, ratio 1.08e11 to the margin, and the discrete containment index `j = 504` never moved — so **the route is stable**.  But the margin is the wrong number: see §2. |
| 10 | 21 Collins calls, 0 Kelly warnings after the resolution, 5 before | see §7 | |
| 11 | C3 x C5: `_GAP_KERNEL_ACCURACY_TAU` "ships OFF and stays off" | see §5 | |
| 12 | the restatements are decisions, not loosenings | see §7 | |
| 13 | the 26 new tests and their mutation matrix | see §7 | |
| 14 | four pre-existing WSL reds | **CONFIRMED** | The census/walker/dispatcher/public-API sweep reads **647 passed, 14 skipped, 0 failed** on Windows and **4 failed, 643 passed, 14 skipped** on WSL, and the four are exactly the four named. |

---

## 2. THE ROUTING CONDITION IS A STAIRCASE — the finding this verification adds

`_collins_readout_k1` is a SUM of two terms, and MEASURED on this build the
split is exact to the bit (`space + angle == K1` in float on every row):

    K1 = space_term + angle_term
    space_term = 2 dx |A| r / (lambda |B|)
    angle_term = 2 dx theta / lambda

Both `r` and `theta` come from `_collins_containment_radius`, which returns
`d[order[i]]` — **a GRID COORDINATE, with no interpolation between samples**.
`theta` is therefore read off `np.fft.fftfreq(N, d=dx)`, whose outermost bin
for even `N` is exactly `1/(2 dx)`.  So

* `angle_term` lies in `{2j/N : j = 0 … N/2}` and is **bounded above by
  EXACTLY 1** (measured: every reading over an 8-rung and a 97-rung scan is an
  integer multiple of `2/N` to 1e-9, and the smallest non-zero increment is
  one bin);
* **the route is a threshold on a STAIRCASE.**  On a w-ladder at N = 512, K1
  crosses 1 by a JUMP from **0.9998355263157893** to **1.004096177944862** —
  it never takes a value in between;
* **`angle_term == 1.0` exactly whenever the envelope's angular support is
  GRID-CLIPPED** (`_collins_containment_radius` "saturates at the outermost
  sample when the grid itself already clipped the tail", its own docstring
  says), and then `K1 = 1 + space_term > 1` at **every leg length** — swept
  over three decades of final distance.  No refinement of the space term
  reaches the bar on such a field.

### What that does to the package's own headline margin

| quantity, design-121 N = 1024 | value |
|---|---|
| reported K1 | 0.9995812250283026 |
| `space_term` | 0.015206225028302638 |
| `angle_term` | 0.984375 = 2 x 504 / 1024 (j = **504 / 512**) |
| margin below the bar | 4.1877497169739986e-04 |
| **one quantisation step, `2/N`** | **1.953125e-03** |
| margin, in steps | **0.214** |
| K1 if the index moved ONE bin up | 1.001534375 → route FLIPS to sziklas |

**The reported margin is 4.66x smaller than one quantum of the quantity the
threshold is taken on, so "4e-4 below the boundary" describes nothing.**  K1
cannot land between 0.99958 and 1.  The stability question is not how close
K1 is to 1; it is whether the containment INDEX can move, and that is a
`searchsorted` of a cumulative power sum against a fixed `1 - 1e-6` bar.
Measured headroom at N = 1024: **1.0240e-07 relative (4.61e8 ULP of the
total) UP** — the direction that flips the route — and 2.2559e-08 DOWN.

**This is a stronger safety argument than the one the package makes, not a
weaker one**, and it should replace it.  At **N = 512 the design-121 exit
field is grid-CLIPPED** (`angle_term` exactly 1.0, j = 256 = N/2, headroom-up
equal to the tail fraction itself): the Sziklas route there is
**structurally forced**, and no value of `space_term` could have made it pass.
That is the sharpest form of the point and it is not in the report.

### The K1 build-spread measurement (claim 9b)

| setting | K1, full precision | route | j / (N/2) |
|---|---|---|---|
| WIN py3.14 np2.4.4, 1 / 4 / 8 threads | 0.9995812250283026 | collins | 504/512 |
| WSL py3.12 np2.4.6, default coretype, 1 / 4 / 8 threads | 0.9995812250283026 | collins | 504/512 |
| WSL `OPENBLAS_CORETYPE=Haswell` / `Zen` / `generic` | 0.9995812250283026 | collins | 504/512 |
| WSL `=Nehalem` / `SandyBridge` / `Prescott` / `Core2` / `Barcelona` | 0.9995812250283065 | collins | 504/512 |
| WSL `=SkylakeX` | **not runnable** (illegal instruction on Zen 3; `numpy.show_config()`'s "SkylakeX" is the BUILD label, not the runtime pick) | — | — |

Spread **3.885780586188048e-15 (17.5 ULP of 1)**, ratio to the margin
**1.08e11**.  **The index matched on all 13 runnable settings** — which is the
only stability evidence that means anything for a discrete decision.

The spread does **not** come from the readout: with a fixed deterministic
field, `_collins_readout_k1` returns `0x1.2036e5ed57de8p-1` **bit-identically**
across 2 builds x 6 coretypes x 3 thread counts (`np.fft` and `np.cumsum` are
not BLAS).  All of it comes from upstream, through the cancellation
`Ax = 1 + z/R` (`z/R ≈ −0.99914`).

**Cost of a flip, if one ever happened** (forced at N = 1024 by substituting
the condition): ΔFWHM −0.0552 %, ΔEE3 +0.135 % rel, Δpeak +0.172 %, and the
forced fallback is **bit-identical to `transport='sziklas'`**.  A
build-dependent flip would be a reproducibility problem, not an accuracy one
— and on the measured evidence it does not happen.

### A reproducibility trap, for whoever re-measures this next

The report's §0 K1 table says it was taken "at the plane the readout actually
runs on (the chain exit, `final_distance = 0`)".  **That reconstruction is not
the array the condition is evaluated on.**  Spying on the chain's own call:

| N | published `readout_route_k1` | `final_distance=0` reconstruction |
|---|---|---|
| 256 | 82.36047195201569 | 82.36047195201569 |
| 512 | 41.44909827159872 | 41.44909827159872 |
| 1024 | **21.4224837152085** | **21.455686840208497** |
| 2048 | **10.94410348692525** | **11.126720674425252** |

The two arrays have the same shape, the same `dx` and the same `R` and differ
at relative L2 **1.4149 on every grid**; at N = 256 and 512 the two K1
readings agree anyway *because both arrays saturate the band*, so the
agreement at the coarse grids is a coincidence of saturation and not a
validation of the method.  At N = 1024 they are apart by **exactly 17 quanta
of 2/N**.  The report's N = 1024 row is the reconstruction's number.

---

## 3. THE ENTRY-POINT WAY-BACK TABLE

Census method: an AST walk of **all of `lumenairy/`** for calls to the three
flipped entry points, both public readouts, `_carrier_step_fast` and every
`_collins_*` helper — **22 hits, every one inside
`lumenairy/propagators/carrier.py`**.  No call site in `analysis/`, `ui/` (the
40-odd docks), `optimize/`, `raytrace/`, `sources/`, `elements/`, `io/`,
`backend/`, `_math/`, `examples/`, `scripts/` or `benchmarks/`.
`propagate_through_system` / `lumenairy.evaluate` have no carrier-chain step
type, so **VERIFY-WP-C1's D4 shape does not recur here**.  Runtime census: of
**731** names in `lumenairy.__all__`, exactly **3** take `transport`.

| # | public entry point | reaches which flipped default | does it MOVE? (measured, both builds) | one-keyword way back? | verdict |
|---|---|---|---|---|---|
| 1 | `propagate_carrier_referenced` (`carrier.py:1147`) | its own | **YES** — 14 of 39 NumPy keys; 13 of 33 on an independent key set | **YES**, `transport='sziklas'`, byte-exact | OK |
| 2 | `propagate_traced_carrier_chain` (`carrier.py:9565`) | its own + the gap leg (`:10743`), the final leg (`:11364`), the readout route (`:11253`) | **YES** — fixture-dependent: 0 of 8 on the WP-B4 relay, 23 of 23 on a converging two-group relay, of which **13 become `RuntimeError`** | **YES**, byte-exact | OK behaviourally; **D5** |
| 3 | `propagate_traced_carrier_chain_multi` (`carrier.py:12347`) | forwards verbatim via `_common_chain_kwargs` (`:12964`) | **YES** — 11 of 11 keys, 10 becoming `RuntimeError` | **YES**, forwarded verbatim (measured, not read) | OK |
| 4 | `carrier_referenced_focus_readout` (`carrier.py:4264`) | `propagate_carrier_referenced` at `:4571`, **PINNED** `'sziklas'` | **NO** — 0 of 13 keys | n/a | OK; pin verified load-bearing (27 keys move without it, 17 of them ok→raise) — but see **D8** on whether the pin is the right call |
| 5 | `carrier_referenced_exact_focus_readout` (`carrier.py:6790`) | nothing | **NO** — 0 of 5 keys | n/a | OK |
| 6 | `carrier_referenced_reconstruct` / `_envelope` / `_fit_radius` / `_aperture` | nothing | **NO** — 0 of 5 keys | n/a | OK |
| 7 | `propagate_carrier_referenced` under `jax.jit` / `jax.grad` | own default → the Tracer refusal at `carrier.py:2622` | **YES** — base ok → branch `ValueError`, both builds | **YES** — `transport='sziklas'` restores the exact bytes (grad sha `234a0f35e088211e` WIN / `88b20dcda32ef848` WSL, identical to base) | OK, and correctly documented |
| 8 | `lumenairy.evaluate`, `propagate_through_system`, `propagate_through_system_jax` | **none** | **NO** (structural) | n/a | OK |
| 9 | `lumenairy/ui/*`, `lumenairy/analysis/*`, `examples/`, `scripts/` | **none** | **NO** | n/a | OK |
| 10 | `validation/pipeline/sources.py:516` (repo harness, not shipped in the package) | the chain, passing no `transport=` | **YES** (it *is* the chain) | the keyword is reachable but no `ChainSpec` field exposes it | recorded, not a package defect |

**The campaign's way-back rule is SATISFIED.**  Every public entry point whose
behaviour moves accepts `transport='sziklas'`, forwards it, and returns the
49ddf4bd arithmetic — **103 of 103 keys on both builds**, with the base arm
spelled to pass no `transport=` at all.  No entry point is missing a keyword;
this is the first work package in the C-series where that is true without a
filed gap.

### The Migration-Guide recipes, run

| recipe | runs? | reproduces the old bytes? |
|---|---|---|
| `propagate_carrier_referenced(..., transport='sziklas')` | yes | **YES** (inside the 103/103) |
| `propagate_traced_carrier_chain(..., transport='sziklas')` | yes | **YES** |
| `jax.grad(... transport='sziklas')` | yes | **YES**, sha identical to base on both builds |
| the near-focus `gap_kernel='fresnel'` recipe | **NO** on an independent fixture — base returned, branch raises the containment `RuntimeError` | n/a — the recipe cannot be applied where it is most needed (**D5**) |
| "the stop-plane keys keep working exactly as they did" | **NO** — the keys are still accepted, but what they return moves and can raise (**D5**) |  |
| "`final_distance=0` with a `focus_readout` still works" | **NO** on the same fixtures (**D5**) |  |

---

## 4. DEFECTS

Nothing below was applied to `lumenairy/`.

### D5 — SHIP BLOCKER, documentation. "No public call that worked on 5.48.1 raises on 5.49.0" is false.

**Reproducer** (both builds):

```
cd /c/tmp/lum_vc3 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VC3_TREE=C:/tmp/lum_vc3 \
  VC3_OUT=C:/tmp/lum_vc3/validation/probe_verify_c3 VC3_TAG=branch_win \
  python validation/probe_verify_c3/probe_vc3_newraise_indep.py
# and the same from /c/tmp/vc3_base with VC3_TREE=C:/tmp/vc3_base
```

Fixture chosen for being ordinary: collimated launch `r_in=inf`, two identical
BK7 biconvex singlets (R = ±120 mm, t = 6 mm) at 20 mm and 15 mm, a plain
`focus_readout=dict(dx_out=0.5 um, N_out=64)`, no stop-plane keys, no tilt, no
`gap_kernel`.  Twelve configurations (N = 256/512/1024, dx = 10/20 um,
w = 2/3 mm, `final_distance` = 5/8/20 mm):

| tree | returns | raises |
|---|---|---|
| base 49ddf4bd (default `'sziklas'`) | **12 / 12** | 0 |
| branch 4d87ff48 (default `'collins'`) | 1 / 12 | **11 / 12** |

Identical pattern on WIN-py3.14 and WSL-py3.12; the one survivor is the same
row on both.  The exception, from `_guard_dispose`:

```
RuntimeError: carrier_referenced_focus_readout: the beam does not fit the
co-moving grid at the stop plane.  The grid half-width there is 4447.8151 um
against a measured amplitude radius of 5130.7213 um -- a containment of
0.867 beam radii ...
```

The mechanism is not the readout: the FLIPPED GAP LEGS change the chain's exit
lattice, the readout then routes to Sziklas on K1 > 1 as designed, and the
guard refuses the lattice it is handed.  The base emits no carrier-module
warning at all on these rows.  A peer archive-to-archive measurement over 103
keys reached the same conclusion independently: **23 of 103 keys ok →
`RuntimeError`** (13 chain, 10 multi), both builds, identical key lists.

`transport='sziklas'` restores the returning behaviour on every row, so the
way back is intact and the defect is in the CLAIM.

**HOW WIDE, measured** — because a ship recommendation needs the width and not
one counterexample.  `validation/probe_verify_c3/probe_vc3_blastwidth.py`
sweeps a deliberately ordinary space with **no `transport=` named on either
side** and no stop-plane key, tilt or `gap_kernel` anywhere: 3 prescriptions
(f = 60 / 120 / 300 mm BK7 biconvex) x 1 and 2 groups x N = 256/512 x
w = 1.5/3.0 mm x collimated and r_in = 60 mm x `final_distance` = 5/15 mm x
with and without a focus readout = **192 cells**.  Base vs branch, **identical
on WIN-py3.14 and WSL-py3.12**:

| classification | cells | share |
|---|---|---|
| IDENTICAL | **118** | 61.5 % |
| MOVED | 52 | 27.1 % |
| **OK → RAISED** | **22** | **11.5 %** |
| RAISED → OK | 0 | 0 % |

All 22 are the same containment `RuntimeError`, and all 22 carry a focus
readout.  So the honest statement is that **the majority of ordinary chains
are bit-identical — which is the complementary selection working — and about
one in nine crashes.**  That proportion is what makes this a blocker rather
than a note.

**The same sweep also refutes the §6 census** (claim 10).  Kelly
"under-sampled" `RuntimeWarning`s over the 192 cells:

| build | base 49ddf4bd | branch 4d87ff48 |
|---|---|---|
| WIN-py3.14 | **0** warnings, 0 of 192 cells | **74** warnings, **51 of 192 cells** (26.6 %) |
| WSL-py3.12 | **0** warnings, 0 of 192 cells | **74** warnings, 51 of 192 cells |

The package reports "21 Collins calls, **0** emitting the Kelly warning, both
builds".  That reading is correct for the fixtures it drives and does not
generalise: on ordinary chains the guard speaks on **one cell in four**, where
it never spoke before, and in every case on a leg with **no caller-named
output lattice** — the case §6 names as the only one where the guard is not
dead.  See also D6, where the warning is the sole trace of a leg that is 3x
wrong.

**Requested edit — `CHANGELOG.md`, the Migration paragraph** (the sentence
beginning "**No public call that worked on 5.48.1 raises on 5.49.0**"), and
the equivalent paragraph in `Migration-Guide.md` §5.49.0.

OLD:

```
**No public call that worked on 5.48.1 raises on 5.49.0**, with ONE exception,
and it is a JAX one: `jax.grad` / `jax.jit` through
```

NEW:

```
**Two kinds of call can now raise where 5.48.1 returned**, and both have the
same one-keyword way back.

FIRST, a chain or multi call with a `focus_readout`.  The flipped gap legs
change the chain's exit lattice; the readout then routes to the Sziklas
readout on K1 > 1 as designed, and THAT readout's containment guard can
refuse the lattice it is handed.  MEASURED 2026-09-20 on both builds, on an
ordinary two-group relay with a collimated launch and no stop-plane keys:
11 of 12 configurations raise `RuntimeError:
carrier_referenced_focus_readout: the beam does not fit the co-moving grid
at the stop plane` where all 12 returned at 5.48.1, and 23 of this package's
103 archive-to-archive keys do the same.  `transport='sziklas'` returns the
5.48.1 behaviour in every bit; `focus_readout={'on_focus_containment':
'ignore'}` suppresses the refusal but does NOT restore the answer.

SECOND, a JAX one: `jax.grad` / `jax.jit` through
```

### D6 — SHIP BLOCKER, correctness. A leg that resolves a FLAT output reference has no fallback and runs an aliased chirp-Z on the new default, far from any focus.

**Reproducer**: `validation/probe_verify_c3/probe_gridladder.py`.  Single
group (N-BK7 biconvex R = ±120 mm, t = 6 mm, ⌀25.4 mm), collimated input
w = 2 mm, λ = 1.31 µm, `gap_before = 20 mm`, bare `final_distance = 10 mm`,
window `N·dx = 10.24 mm` held fixed.  Grid refinement arbitrates — no new
oracle needed.

| N | `'sziklas'` r2m (µm) | default `'collins'` r2m (µm) | ratio | sziklas peak | default peak | Kelly K1 |
|---|---|---|---|---|---|---|
| 256 | 1295.36 | **3823.75** | 2.95 | 3.74544 | **443.571** | **29.1734** |
| 512 | 1278.40 | **3847.81** | 3.01 | 2.01840 | **53.5987** | **14.4718** |
| 1024 | 1274.15 | **4099.07** | 3.22 | 1.54419 | **35.9658** | **7.6621** |

`'sziklas'` CONVERGES (−1.31 %, −0.33 %); the default DIVERGES (+0.63 %,
+6.53 %).  Identical to every printed digit on both builds, and **reproduced
a second time from a script written independently of the probe above** —
r2m 1295.3594 / 3823.7543, 1278.4007 / 3847.8052, 1274.1529 / 4099.0717 and
peaks 3.74544 / 443.57068, 2.01840 / 53.59873, 1.54419 / 35.96576.

Which leg it is, measured from the stage diagnostics and the guard's own
message: the GAP leg resolves to the transfer-function form
(`collins_form='tf'`, K1 = 14.8099, `flat=False`), and it is the **bare final
leg** that runs the chirp-Z — it publishes no `collins_*` stage of its own,
and the only trace it leaves is one `RuntimeWarning` reading **K1 (input)
29.1734** at N = 256 and **7.6621** at N = 1024.  So on this configuration the
Kelly guard fires on a chain leg with **no caller-named output lattice**,
which is the case report §6 says is the only one where the guard is not dead.

Mechanism, measured: the leg resolves `flat=True`
(`sbp = 4 r_out θ/(|A| λ) = 260.945 > N = 256`), so `tf_available` at
`carrier.py:2721-2723` is False and the chirp-Z runs **regardless of K1/K3**.
The leg is nowhere near a focus (`R_out = −108 mm`, `A = 0.915`) and the flat
test clears `N` by only **1.9 %**.

An independent construction reaches the same hole from the other side: a
grid-filling envelope at θ = 0.8 Nyquist on `A = +0.11`, `z = 2 mm` resolves
flat at K1 = K3 = **1.1230** and reads σ_x **151.98 µm** against an analytic
**16.5386 µm** (9.2x), while the untilted control has all three agreeing to
six digits.

This also refutes three shipped statements: (i) `carrier.py:2718-2720` and
report §1.3's "the legs with no fallback are precisely the legs the Sziklas
transport could never evaluate" — `'sziklas'` evaluates this leg and
converges on it; (ii) the checkable rule `N dx² <= λ|z_eff|` (`CHANGELOG.md`,
`Migration-Guide.md`, `carrier.py:1257` and `:10354`) — measured
`N dx² = 4.096e-07` against `λ|z_eff| = 1.4311e-08`, false by **28.6x**, and
the leg moves anyway; (iii) report §6's "the guard still speaks [only] there
[on a caller-NAMED lattice]" — it speaks on a chain leg with no named
lattice.

**Requested edit — `lumenairy/propagators/carrier.py`, lines 2721-2723.**

OLD:

```python
    tf_available = (Ax != 0.0 and Ay != 0.0 and not flat
                    and dx_out is None and dy_out is None
                    and carrier_out is None)
```

NEW — option A (restore the pre-flip answer; the returned reference is then
the geometric `R+z`, which is what `'sziklas'` returns anyway):

```python
    tf_available = (Ax != 0.0 and Ay != 0.0
                    and dx_out is None and dy_out is None
                    and carrier_out is None)
```

NEW — option B (keep the flat reference, but refuse instead of returning an
aliased array); insert immediately after the unchanged assignment:

```python
    if (flat and Ax != 0.0 and Ay != 0.0
            and dx_out is None and dy_out is None and carrier_out is None
            and (max(k1x, k1y) > 1.0 or max(k3x, k3y) > 1.0)):
        raise ValueError(
            f"{fn}: this leg resolves a FLAT output reference (the geometric "
            f"reference's space-bandwidth product exceeds N), which has no "
            f"transfer-function complement, and the chirp-Z is not "
            f"representable on this lattice (K1 = {max(k1x, k1y):.6g}, "
            f"K3 = {max(k3x, k3y):.6g}): the returned window would be "
            f"{max(k3x, k3y):.6g} wrapped periods.  Pass transport='sziklas' "
            f"(the pre-5.49.0 arithmetic, which evaluates this leg), or "
            f"refine the grid until K1 <= 1.")
```

Either way the comment at 2718-2720 must stop claiming the no-fallback set is
"precisely the legs the Sziklas transport could never evaluate", and the
`N dx² <= λ|z_eff|` rule must be qualified with "except where the leg
resolves a flat reference".

### D1 — documentation. The chain's own `transport` docstring still says the stop-plane keys are refused.

`lumenairy/propagators/carrier.py:10343-10345`.  This branch edited the
sentences around it and left the parenthetical describing the PRE-WP-C3 rule,
so the public docstring of the very function whose contract changed tells a
caller the opposite of what the code does.  `Migration-Guide.md` and the
CHANGELOG already carry the new contract.

OLD:

```
          grid -- and the readout's ``standoff`` / ``on_focus_containment``
          keys have no referent (passing one is refused, not ignored; pass
          ``transport='sziklas'`` to use them).
```

NEW:

```
          grid -- and naming the readout's ``standoff`` or
          ``on_focus_containment`` key SELECTS the Sziklas readout instead,
          publishing ``readout_route_reason='stop_plane_key'`` on the readout
          stage with K1 not computed at all.
```

### D2 — documentation. `_collins_readout_k1`'s docstring carries two mutually inconsistent MEASURED readings for one fixture.

`carrier.py:2786` says "exit pitch 76.5 um, exit support **5.76 mm**,
**K1 = 56.0**"; `carrier.py:2795`, three sentences later, says
"**82.4 / 21.4 / 10.9** on N = 256 / 1024 / 2048".  `carrier.py:1242` gives a
third exit beam, "**5.4 mm**".

Measured (hand decomposition agreeing with the helper to **0.0 ULP**, both
builds, on the array the chain itself hands the condition): exit pitch
**76.54441837133348 um**, exit support radius **6.7359088166773455 mm**, and
K1 = **82.36047 / 41.44910 / 21.42248 / 10.94410** at N = 256/512/1024/2048.
So the second triple is right to its printed precision and the first pair is
not.  `K1 = 56.0` is the reading the fixture's INPUT beam radius (4.55 mm)
would give, not its exit support.

Requested edit at `carrier.py:2786-2788`:

OLD: `8 mm, exit pitch 76.5 um, exit support 5.76 mm, K1 = 56.0), the one-step`
NEW: `8 mm, exit pitch 76.5 um, exit support 6.74 mm, K1 = 82.36), the one-step`

and at `carrier.py:1242`, `5.4 mm exit beam` → `6.74 mm exit beam radius`.

### D3 — documentation. "it falls only as `1/dx`, so `N ~ 16000` would be needed" is optimistic by ~2x, and has no floor.

`_collins_readout_k1.__doc__` and report §7 item 1.  What falls as `1/dx` is
the SPACE term only.  Measured on WP-B4's relay (physical window fixed):

| N | K1 | space | angle | j/(N/2) | clipped |
|---|---|---|---|---|---|
| 256 | 82.36047195201569 | 81.3604719520157 | **1.0** | 128/128 | **YES** |
| 512 | 41.44909827159872 | 40.449098271598714 | **1.0** | 256/256 | **YES** |
| 1024 | 21.4224837152085 | 20.455686840208497 | 0.966796875 | 495/512 | no |
| 2048 | 10.94410348692525 | 10.12672067442525 | 0.8173828125 | 837/1024 | no |

Extrapolating the measured law gives K1(16384) ≈ 1.37 and K1(32768) ≈ 0.69,
so the grid needed is **~32768, not ~16000**.  And the sentence is missing the
floor: where the angular support stays clipped there is **no** N at which the
readout becomes representable, because `K1 >= angle_term = 1` identically.

Requested edit: change "`N ~ 16000`" to "`N ~ 32768`", and append: "and that
is only where the envelope's angular support is INSIDE the band — where it is
grid-clipped, `angle_term` is exactly 1 and no refinement of the space term
can reach the bar at all."

Report §7 item 1's "a chain that exits on a far finer grid (N ~ 16000 on this
fixture — a 256x memory cost)" needs the same two corrections; it is one of
the three routes it offers to retiring the standoff machinery, and it does not
exist for a clipped envelope.

### D4 — contract. `focus_readout['bandlimit']` is a Sziklas-only key that is accepted and silently ignored on the Collins readout route.

**Reproducer**: `validation/probe_verify_c3/probe_vc3_bandlimit_drop.py`.
Fixture that actually TAKES the one-step route (measured, not assumed):
N = 512 at 8 µm, w = 0.30 mm, f = 300 mm singlet, `final_distance = 50 mm`,
K1 = **0.7698266441481303**, `readout_route='collins'`,
`readout_route_reason='representable'`.

| transport | route | `bandlimit=False` changes the answer? |
|---|---|---|
| `'sziklas'` | — | **YES** (sha `64af0f30…` → `d3ea87ab…` on WIN; `27e7d304…` → `fe88ada7…` on WSL) |
| `'collins'` (the default) | collins | **NO** — bit-identical, no warning, no refusal |

`bandlimit` is in `_FOCUS_READOUT_KEYS`, is **not** in
`_FOCUS_READOUT_STOP_PLANE_KEYS`, and is excluded from `_par_kw` on the
Collins route at `carrier.py:11258-11262`.  Measured by instrumenting the two
readout entry points: the `'sziklas'` readout receives
`['N_out','_period_out','bandlimit','dx_out','gap_kernel']` and the
`'collins'` one receives `['N_out','_period_out','dx_out','fn','gap_kernel',
'on_collins_sampling']` — `bandlimit` is gone.  And it is not an inert knob:
called directly, `bandlimit=False` vs `True` moves the answer by relL2
**1.066e-01 … 1.007e+00** across six fixtures.

This is exactly the accept-and-ignore shape §1.5 invokes as the reason
`standoff` could not simply be ignored.  Two keys with the same relationship
to the two routes got opposite treatments, and the asymmetry is in no
docstring and in no Migration note.  The nine exact-leg keys are dropped
symmetrically on BOTH transports and are fine.

**Requested edit — `lumenairy/propagators/carrier.py:11586`:**

OLD: `_FOCUS_READOUT_STOP_PLANE_KEYS = ('standoff', 'on_focus_containment')`
NEW: `_FOCUS_READOUT_SZIKLAS_ONLY_KEYS = ('standoff', 'on_focus_containment', 'bandlimit')`

with the use site at `:11248` renamed, the `#:` docstring at `:11582-11586`
restated as "the three `focus_readout` keys that only the SZIKLAS readout
has", and `_publish_readout_route`'s reason string `'stop_plane_key'` renamed
`'sziklas_only_key'` (`carrier.py:5086-5088` and the `readout_route_reason`
bullet at `:5065-5069`).

### D7 — documentation. `final_leg='exact'` does not move — false for the chain's RESULT.

`CHANGELOG.md:221`, `Migration-Guide.md:1802`, report line 124.  The exact
final leg itself runs no carrier transport, but every GAP leg does, so the
returned field moves whenever a gap leg does.  Measured, both builds:
`F4-single-final-leg-exact-bare` peak 3.74544 → **443.571** (118x);
`F4-final-leg-exact-bare` 68.2403 → **38038.0** (557x);
`F4-final-leg-exact-readout` 68.3252 → **0.475848** (0.0070x);
`B5-final-leg-exact-default` 80.4678 → **1987.94** (24.7x).  On the WP-B4
relay it does NOT move, so the claim is fixture-dependent rather than
universally wrong.

Requested edit: "the exact final leg itself runs no carrier transport and is
unchanged; the chain's GAP legs still move, so a `final_leg='exact'` chain's
returned field moves whenever any gap leg does (measured 24.7x–557x in peak
on ordinary two-group relays)."

### D8 — decision. Pinning the OLD transport at `carrier_referenced_focus_readout`'s standoff leg is the wrong one of the four pins.

Three of the four internal sites are right: the leg's own fallback
(`carrier.py:2742`) is **mandatory** — deleting `transport='sziklas'` there
gives `RecursionError` on every fallback geometry — and the two
`transport != 'collins'` arms (`:10753`, `:11371`) are free and correct.

The readout's standoff leg (`:4571`) is the only one where naming a value
CHANGES which quadrature runs, and the measurement goes against the choice.
Against a converged dense separable Fresnel oracle (self-consistency
**4.470e-05**, convergence 64x→256x **5.14e-07**) on the branch's own fixture:

| standoff leg | relL2 vs the oracle | best global scale | peak ratio |
|---|---|---|---|
| **Collins** (unpinned) | **4.7339890671818134e-05** | 0.99999006 | 0.99997617 |
| Sziklas (pinned, guard downgraded to `'warn'`) | **2.4049894966524197** | 2.2738 | **5.1433** |

Over 5 geometries x 6 standoffs: pinned **6/30 RAISE**, and of the 24 that
return three are silently wrong by 11 %, 13 % and 42 % — errors the guard
does NOT catch.  Unpinned: **0/30 raise**, relL2 <= **3.846e-03** on every
row and <= 5.3e-04 on 28/30.

So the pin as shipped turns a correct answer into a refusal on a shipped
configuration, and the guard it preserves is not a safety property on its own
route either.  **Recommended**: take the default at `:4571` and restate
`_check_focus_containment` against the leg's RETURNED grid; failing that,
keep the pin AND widen the guard so the three silent 11–42 % rows are caught.
This is the one place where "an internal site that is pinned never benefits
from an improved default and becomes a silent second code path" is realised
at commit time rather than later.

### D9 — test hygiene. The internal call-site census fires for 2 of 6 placements.

Six un-named fourth call sites added at once to a mutated tree; the whole
shipped file reads **1 failed, 25 passed**, only the census test firing, and
it names only two of them:

| placement | caught? | why |
|---|---|---|
| A — a new module-level helper, plain call | **yes** | |
| B — a call inside an existing function | **yes** | |
| C — `propagate_carrier_referenced(env, R, z, wl, dx, **kw)` | **NO** | excused unconditionally by `not any(kw.arg is None …)`; `**{}` excuses it too |
| D — `_alias = propagate_carrier_referenced; _alias(...)` | **NO** | not counted (the `ast.Name`/`ast.Attribute` name test) |
| E — `globals()['propagate_carrier_referenced'](...)` | **NO** | not counted |
| F — the same call in `lumenairy/propagators/mft.py` | **NO** | the census reads only `pathlib.Path(CA.__file__)` |

**F is the one that matters**: report §1.1 asserts by grep that no other
module names these entry points, and nothing gates it.  (The four existing
splat sites — `carrier.py` 9509, 10506, 12074, 12982 — all DO forward
`transport=transport` today, so nothing is wrong now.)

Requested edits to the census body in `tests/unit/test_c3_collins_default.py`
(~lines 300-322): (1) iterate every shipped module
(`for _p in sorted(pathlib.Path(CA.__file__).parents[1].rglob('*.py')):`) and
record offenders as `f'{_p.name}:{node.lineno}: {nm}(...)'`, keeping
`assert n_sites >= 3`; (2) replace the blanket splat excuse with a curated
allow-list keyed by the ENCLOSING FUNCTION NAME, asserting every splat site is
in it; (3) assert the module source contains no module-level assignment whose
value is a bare `Name` in `_ENTRY_POINTS`, and no `globals()[...]` /
`getattr(...)` lookup of those names.

### D10 — the CuPy censuses are materially weaker than §5.3 describes.

**(a) The public CHAIN host-demotes a device field, with exactly the signature
the report forbids.**  Reproducer:
`cd /c/tmp/vc3_head && PYTHONPATH=/c/tmp/vc3_head python validation/probe_verify_c3/v_cupy_chain.py …`
→ `TypeError: Implicit conversion to a NumPy array is not allowed.` at
`carrier.py:10537` → `:6411 _check_chain_entry_congruence` →
`:6377 _chain_entry_congruence_stats`.  On the default
`on_multi_congruence='warn'`, on BOTH transports, before any transform.
Requested edit at `carrier.py:6377`: `E = np.asarray(env)` →
`from ..backend import to_numpy` / `E = np.asarray(to_numpy(env))` — the idiom
`_collins_power_marginals` already uses at `:1836-1838`.

**(b) The host-demotion census misses 9 of 13 injected demotions, 4 of them
real device breaks.**  Missed and breaking: `_vnp = np; _vnp.asarray(env)`;
`import numpy; numpy.asarray(env)`; `_vtmp = env; np.asarray(_vtmp)`;
`env_a * np.arange(n)` (`TypeError: Unsupported type <class 'numpy.ndarray'>`).
Missed and JAX-relevant only: `float(env[0,0].real)`, `env.ravel()[0].item()`,
`math.sqrt(...)`, `np.exp(env)`, `env_a.astype(np.result_type(...))`.
Requested edits: widen the matcher to any numpy ALIAS and to the full module
name, and invert the local-name test so that inside a Collins helper NO local
may be host-normalised except an explicitly classified one.

**(c) The demotion census walks a hand-written 5-tuple, not the call graph**,
so (a)'s site is invisible; and the xp census only classifies names beginning
`_collins`, so **43 of the 66 reached functions are never checked** and a new
unclassified demoting helper called `_carrier_brand_new` does not fire at all.
Either widen the assertion or narrow the report's sentence to "every helper it
reaches **whose name begins `_collins`**".

**(d)** `_collins_readout_k1` is allow-listed as host-side but takes a FIELD
and reaches the device transform; it should move to `_XP_PARAMETRISED` and be
added to the census roots.

**(e)** The "FIRST branch" assertion on `fft_infra._fft2` is a 2000-character
substring window that includes a ~1700-character docstring; it should be an
AST check that the first non-docstring statement is
`if _is_cupy_array(x):`.

### D11 — the readout's arbitration is one-sided, and the Sziklas answer is not exact either.

Report §0 arbitrates the two readouts by declaring the Sziklas one correct.
Against a converged independent quadrature it is **less wrong, not right**:
complex relL2 **0.09451**, amplitude-only **0.02477**, centre |ratio|
**1.0012151** at arg **1.756 mrad**.  A piston+tilt+defocus fit gives tilt
**6935.307 rad/m in both axes (equal to 11 digits)** and a residual of
**3.211 mrad rms**.  Worth one sentence in §0 so a reader does not take
1.0743 as exact.  No code change.

### D12 — small documentation items

* Report §1.1's line citations are stale: `carrier.py:9518` → **9596**,
  `carrier.py:12275` → **12382** (at the report's own commit the
  `transport: str = 'collins'` lines are 1156, 9596, 12382).
* `carrier.py:2733`'s `K1 = 0.79` is unreproducible on any fixture searched;
  the archive probe's own fixture reads **K1 = 0.9467120853080567** with the
  claimed K3 = 1.2941548183254343.
* Report §2.1's "`gap_kernel='fresnel'` reads within 0.3 % of `'auto'` on
  every Collins cell" is true only because the truncation floor masks both:
  at 9 radii `'auto'` is **138x** (converging) and **6045x** (just past)
  worse than `'fresnel'`.
* §6's zero-warning census is over this package's own fixtures; on chain legs
  that resolve a flat reference the guard fires (K1 = 29.1734 / 14.4718 /
  7.6621).  Scope the sentence.
* `validation/probe_c3_collins_default/probe_d121_acceptance.py` cannot run as
  committed — it dies on `AssertionError … LUMENAIRY_ROOT` unless that
  variable is already exported, so the d121 rows are not reproducible from the
  repository.  Add `os.environ['LUMENAIRY_ROOT'] = _TREE` after
  `sys.path.insert(0, _D121)`.
* The branch did not splice its 26 new ids into `.test_durations` (0 of 26
  present).  Below the staleness gate's 2 % bar, but it degrades the shard
  balance the gate exists to protect.

---

## 5. THE C3 x C5 INTERACTION — the two 5.49.0 branches contradict each other about `_GAP_KERNEL_ACCURACY_TAU`

**They disagree, and C3 is the one that is wrong.**  This was the cross-check
the work package called the most important one, and it fails.

| | branch | file : line | shipped value |
|---|---|---|---|
| WP-C3 | `feat/c3-collins-default` @ 4d87ff48 | `lumenairy/propagators/carrier.py:1729` | `_GAP_KERNEL_ACCURACY_TAU = None` |
| WP-C5 | `feat/c5-three-defaults` @ d8de383c | `lumenairy/propagators/carrier.py:1736` | `_GAP_KERNEL_ACCURACY_TAU = 1e-4` |

C3 does not merely ship it off — it asserts, twice and in the caller-facing
documents, that it **stays** off:

* `CHANGELOG.md:252` (the Migration paragraph): "The accuracy-keyed automatic
  fallback that would do this for you exists in the source as
  `carrier._GAP_KERNEL_ACCURACY_TAU` and **ships OFF and stays off**: it
  changes what `'auto'` means on a leg, which is the maintainer's decision and
  is not taken here (ledger 1.5, resolved by 4.3)."
* `Migration-Guide.md:1899` carries the same sentence.

The maintainer **did** take that decision, on the sibling branch, in commit
`3dcc2667` — "feat(carrier): WP-C5 item 1 — `gap_kernel='auto'` falls back to
the paraxial kernel inside a derived near-focus band (tau = 1e-4)".  Both
branches target 5.49.0 and both edit `lumenairy/propagators/carrier.py` (C3
+334/−75, C5 +114/−59).  **So on the merged 5.49.0 tree the constant is
1e-4 and C3's Migration paragraph is false as shipped.**

### Which legs change route when BOTH defaults land

The C5 switch is **Collins-only**: it lives inside `_collins_transport`
(enclosing `def` at `carrier.py:2230` on the C5 branch) and is reached only
through `if (kernel == 'exact' and _GAP_KERNEL_ACCURACY_TAU is not None)`.
The Sziklas co-moving step never consults it.  That produces a compounding
interaction neither branch's blast radius measured:

| leg | 5.48.1 | C3 alone | C5 alone | BOTH |
|---|---|---|---|---|
| a gap leg near a carrier focus, `gap_kernel='auto'` | co-moving split (carrier → ASM bridge → carrier), exact kernel | **chirp-Z**, exact kernel | unchanged (the leg never reaches the Collins transport) | **chirp-Z AND the kernel swapped to `'fresnel'`** inside the derived band |
| the same leg, `gap_kernel='fresnel'` | co-moving split | chirp-Z, fresnel | unchanged | chirp-Z, fresnel — C5 inert (only `'auto'` falls back) |
| the same leg, an EXPLICIT `gap_kernel='exact'` | co-moving split | chirp-Z, exact | unchanged | chirp-Z, exact — C5 deliberately never overrides an explicit `'exact'` |
| a relay leg away from focus | co-moving | transfer-function (= co-moving, bit-identical) | unchanged | unchanged — the C5 band is not entered |
| the chain's focus readout | Sziklas readout | Sziklas or one-step, by K1 | unchanged | the one-step route is now a Collins stage, so a near-focus readout can pick up the C5 kernel swap too |
| `final_leg='exact'` | unchanged leg, moving gaps | moving gaps | unchanged | moving gaps, now with two independent changes each |

**The headline: C5's near-focus switch is INERT before C3 and LIVE after it**,
on exactly the legs C3 moves onto the chirp-Z.  A near-focus gap leg that at
5.48.1 took the co-moving split will, on the merged tree, take a different
transport AND a different kernel — two independent route changes on one leg
in one release, and neither branch measured the other's.

C5's own comment is stale in the mirror-image way: `carrier.py:2458-2461` on
the C5 branch still reads "ACCURACY-KEYED FALLBACK, **OFF BY DEFAULT** … with
the shipped `None` nothing below is evaluated and the leg is 5.47.0 to the
byte", beside an assignment of `1e-4`.  That is C5's to fix, but it is in the
same file and the same release.

### Requested edits

1. **`CHANGELOG.md:252` and `Migration-Guide.md:1899`** — delete "**and stays
   off**" and the "(ledger 1.5, resolved by 4.3)" attribution, and replace the
   paragraph's advice with the merged behaviour:

   OLD: `exists in the source as ``carrier._GAP_KERNEL_ACCURACY_TAU`` and
   **ships OFF and stays off**: it changes what ``'auto'`` means on a leg,
   which is the maintainer's decision and is not taken here (ledger 1.5,
   resolved by 4.3).`

   NEW: `exists in the source as ``carrier._GAP_KERNEL_ACCURACY_TAU``.  It is
   ``None`` in THIS package and the maintainer turned it on at ``1e-4`` in
   WP-C5 (ledger 0.1), so on the merged 5.49.0 tree ``'auto'`` DOES fall back
   to the paraxial kernel inside the derived near-focus band and the manual
   ``gap_kernel='fresnel'`` above is a belt-and-braces measure rather than the
   only remedy.  Note the two interact: the switch is reached only through
   ``_collins_transport``, so it is inert before this flip and live after it,
   on exactly the legs this flip moves onto the chirp-Z.`

2. **A joint blast measurement before the tag.**  Neither branch's 54-file
   run was taken against the other's `carrier.py`.  The merge should be
   measured once with both changes in place, because the four test files both
   branches edit —
   `test_audit2609_b4_collins_transport.py`,
   `test_fix_v1_v8_readout_guard_and_standoff.py`,
   `test_niche_d2_chain_multi.py` and
   `test_wave5_h2_near_focus_table.py` — are exactly the near-focus and
   Collins fixtures the interaction runs through.

---

## 6. WHAT THIS VERIFICATION COULD NOT MEASURE

* **The end-to-end CuPy device run** — confirmed unmeasurable, concretely.
  Windows has cupy 14.0.1 with one visible device (RTX 4070 Ti, CUDA runtime
  12090) and working elementwise/reduction kernels, but `cupy.fft.fft2` raises
  `ImportError: DLL load failed while importing cufft`; the failure is in
  cupy's lazy `__getattr__` → `import_module`, so every library path that
  transforms on device dies there.  WSL has no CuPy.  The debt stands exactly
  as §5.6 states it.
* **The shipped design-121 acceptance** (3.450 / 88.8 / 99.6 and the 8x4
  Dammann fan) — it hard-codes `sys.path.insert(0, r"D:\…\Lumenairy")`.  A
  through-focus scan shows something the report does not say: **K1 > 1 at
  every Δz except the MSoP plane**, so on the acceptance's own configuration
  (`final_leg='auto'`, reporting at BEST focus) the one-step readout would not
  be taken and the flip would be byte-identical to the old default.  That
  weakens §2.5's "the flip is not defensible until this runs" closure rather
  than strengthening it.
* **Where `1.2359` came from** — it is in no saved probe JSON and no spelling
  of the free leg reproduces it.
* **A fully NON-paraxial truth for claim 1** — the independent reference's own
  paraxiality was BOUNDED (3.92e-05 on axis, 1.13e-03 in L2) rather than
  removed.  Both library transports are paraxial too, so the comparison is
  like-for-like.
* **`OPENBLAS_CORETYPE=SkylakeX`** — illegal instruction on this Zen 3 host.

---

## 7. THE RESTATEMENTS, THE NEW TESTS AND THE CENSUS

### 7.1 The restatement this verification checked most closely — `d3`

The package's most interesting restatement claim is that
`test_niche_d3_guards.py::_mux_chain_field` would have gone VACUOUS on the
flipped default, `||E6 - E4||` reading exactly 0.0.  **Reproduced, exactly**,
by driving the helper's own body with each transport in turn:

| transport | ARM 1 (`launch=False`, the fail-before) | ARM 2 (`launch=True`, the claim) |
|---|---|---|
| `'sziklas'` (what the branch names) | `max|E6-E4| = 0.000000e+00` — inert, as claimed | `||E6|| = 3.782241e-02`, `||E6-E4|| = 8.529958e-01`, **moved = 22.552654** against a bar of `> 1.0` |
| `'collins'` (the flipped default) | `max|E6-E4| = 0.000000e+00` | `||E6|| = 6.347786e+01`, `||E6-E4|| = **0.000000e+00**`, **moved = 0.000000** |

So on the flipped default the test does not silently pass — it FAILS, and it
fails by reporting the residual-eikonal degree as INERT, which is its own
fail-before condition reached for the wrong reason.  That is exactly the shape
the package names, and pinning the transport is the right restatement: the
claim being made is about `apply_real_lens_traced`, and the co-moving pitch is
what makes the subtraction of two chain runs well posed.  **Classified
DECISION, and the sibling `_linearity_error` likewise** — its pre-existing
docstring already required "all five runs land on the same lattice", so the
branch is naming a precondition the test always had, not weakening one.

**A pre-existing stale number, recorded and NOT charged to this package.**
That test's docstring says `||E6-E4||/||E6||` was "measured **39.83**
(Windows) and **43.88** (WSL)".  It reads **22.552654** here — and reads the
same 22.552654 on the base tree 49ddf4bd, so it was already stale before this
branch and the branch did not touch that text.  The bar is `1.0`, 22x inside,
so nothing is fragile; but the restatement was an opportunity to re-measure it
and did not.

### 7.2 Test runs — every tail grepped

| run | build | result |
|---|---|---|
| `tests/unit/test_verify_c3_collins_default.py` (this package's decision tests) | WIN-py3.14 | **9 passed, 4 xfailed** in 14.1 s |
| the same | WSL-py3.12 | **9 passed, 4 xfailed** in 16.1 s |
| `tests/unit/test_c3_collins_default.py` (the branch's own) | WIN-py3.14 | **26 passed** in 11.2 s |
| `tests/unit/test_audit2609_b4_collins_transport.py`, WHOLE file | WIN-py3.14 | **130 passed** in 5:16 |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep, plus `test_audit_except_budget.py` and `test_ci_kernel_consistency.py` | WIN-py3.14 | **647 passed, 14 skipped** in 3:07 |
| the same | WSL-py3.12 | **4 failed, 643 passed, 14 skipped** in 2:39 — the four are exactly §4.4's pre-existing WSL reds |
| `tests/unit/test_audit2609_a15a_durations_staleness.py` | WIN-py3.14 | **4 passed** in 1:40 |
| the 54-file carrier-touching blast set + both C3 files | WIN-py3.14 | *(see below)* |
| the same, `b4` excluded and run per class | WSL-py3.12 | *(see below)* |

### 7.3 Gates

| gate | result |
|---|---|
| `ruff check lumenairy/ tests/ scripts/` (WSL, the CI invocation) | **All checks passed** |
| `ruff check .` (WSL, whole repo under the project config) | **All checks passed** |
| `python -m mypy` (no args) | **Success: no issues found in 33 source files** |
| `scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** |
| `.test_durations` | valid JSON, 16596 → **16609** ids (13 spliced); the staleness gate is 4 passed |

`validation/` is `extend-exclude`d in the ruff config, so the probe files in
`validation/probe_verify_c3/` are outside both gates by design, exactly as the
package's own probes are.

### 7.4 Files this verification wrote

Probes and data under `validation/probe_verify_c3/` (both builds where the
claim is a number): the K1 decomposition, quantisation and tie-margin probes;
the reachability sweep; the `bandlimit` drop; the default-vs-default key set;
the independent new-`RuntimeError` reproducer; the 192-cell blast-width sweep;
and the sub-task harnesses (the 103-key archive-to-archive way-back set, the
validated analytic-Gaussian oracle and its converged upsampled Fresnel
reference, the fallback/recursion/standoff probes, the CuPy censuses and
mutations, and the design-121 driver with its 13-setting build sweep).

Decision tests: `tests/unit/test_verify_c3_collins_default.py` — nine passing
and four strict xfails (D5, D4, D1, D2), the repository's own instrument for a
gap a verification finds and does not fix, so closing one turns its marker red
and forces the marker out with the fix.

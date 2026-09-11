# Building the BOR multilayer guards — 5.45.1, 2026-09-12

> **STATUS — BUILD REPORT.** What the plan
> `docs/audits/PLAN_BOR_MULTILAYER_GUARDS_5_45_1_2026_09_12.md` bound, what was
> measured while building it, and what a caller sees. The scoping this all
> descends from is `docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`.
>
> Tree: `C:\tmp\lum_bor1`, branch `fix/bor-multilayer-guards`, forked from
> `scope/bor-guards` (`657988b`). Probes and JSON:
> `validation/probe_fix_bor_guards/`. Every number below was measured on this
> tree, on both local builds, and every probe asserts the tree it loaded from
> `lumenairy.__file__` before it measures anything.
>
> The user's order this build serves: *"I don't want any of these errors to show
> up in any of my solvers for multilayers."* Three of the five steps end in a
> refusal or a warning; the other two exist so that those three are one line
> each and so that a future population can be measured rather than assumed.

---

## 1. Terms

Defined before they are used, because four of them name a piece of code rather
than a piece of physics and the difference decides which guard applies where.

**BOR — body of revolution.** An axisymmetric stack: layers stacked along `z`,
each layer a set of concentric rings in `r`, closed at `r = Rbig` by a PEC
(Dirichlet) wall. Solved one azimuthal order at a time — fields go as
`exp(i m phi + i q z)`, with `m` the azimuthal order (an integer, fixed per
solve) and `q` the **axial wavenumber**. `qn = q / k0` is the dimensionless
axial index, the cylindrical peer of a Cartesian diffraction order's `kz / k0`.
The public entry point is `BORStack(Rbig, m, ..., basis='fd'|'sem')`; a
lower-level prototype entry point `bor_solve.build_layer` / `bor_solve.solve`
also ships and carries a third, legacy basis.

**The three radial bases.**

* `basis='fd'` — a Yee **div-conforming staggered** finite-difference basis on
  ONE uniform radial grid shared by every layer. The grid does not know where
  the ring walls are.
* `basis='sem'` — **SEM, the spectral-element method**: each layer gets its OWN
  radial mesh whose element boundaries sit ON that layer's ring walls, with a
  polynomial of `degree` inside each element. Layers whose meshes differ are
  coupled by a mortar. This is the accurate basis and the one that carries the
  geometry.
* `bor_solve.build_layer(basis='nodal')` — the historical nodal FD basis. It is
  NOT divergence-conforming, so a large fraction of its modal basis is a
  divergence-violating **spurious sea** carrying zero `z`-flux. Kept as a
  documented escape hatch; `build_layer`'s default was changed away from it in
  an earlier audit precisely because of what step 3 below now refuses.

**The enrichment window.** `BORStack._solve_sem` does not mesh layer `i` on
layer `i`'s own walls. It meshes it on the **union of the walls of layers
`i-1`, `i` and `i+1`**. That union is deliberate — it makes the mortar between
neighbours nearly conforming, hence cheap and accurate — and it is also the
whole of error class C on this engine.

**Mortar.** The weak (Galerkin, `r dr`-weighted) coupling of two adjacent layers
whose radial element meshes do NOT line up: instead of node-by-node continuity
the interface demands that the tangential fields agree against every test
function of the neighbour's space. When the two meshes coincide it degenerates
to the ordinary mode match.

**The three error classes.** **A — the branch cut**: which root of `q^2` is the
forward mode, decided by a test that reads the eigensolver's backward error
rather than physics. **B — interface conditioning**: an explicit cascade
inverse that is numerically singular returns a build-dependent number.
**C — mortar and sliver**: two layers whose walls differ by a small `delta`
cause the discretisation to manufacture an element of width `delta`, whose
`1/w^2` stiffness injects spurious axial wavenumbers.

**Fail-before switch.** A module-level boolean whose `False` value restores the
pre-fix behaviour bit for bit, so a behaviour change on a documented path is
reversible without a downgrade.

**Bit-identity contract.** A step's promise that a named population's numbers do
not move at all — hashed, not eyeballed — established by RUNNING the population
before and after, on both builds.

---

## 2. The two build ladders, and what they actually are

Every build/thread-dependence claim below was taken on BOTH ladders, on BOTH
builds, and the kernel that actually loaded was read back from
`threadpoolctl.threadpool_info()` in every run rather than inferred from the
request.

| requested `OPENBLAS_CORETYPE` | obtained, Windows py3.14 | obtained, WSL py3.12 | usable |
|---|---|---|---|
| `HASWELL` | Haswell | Haswell | yes |
| `NEHALEM` | Nehalem | Nehalem | yes |
| `PRESCOTT` | **Katmai** | **Katmai** | yes |
| `ZEN` | **Haswell** | **Haswell** | **no — silent alias** |
| `SKYLAKEX` | SkylakeX (reported) | SkylakeX (reported) | **no — dies on the first LAPACK call** |

`ZEN` produced results bit-identical to `HASWELL` on every arm run here, which
is the alias confirmed rather than assumed. `SKYLAKEX` was not run: this host is
a Ryzen 9 5950X (Zen 3) with no AVX-512. **The CI's actual ZEN kernel is
therefore still unmeasured**, and section 10 records that.

Builds: **Windows** py3.14.6 / numpy 2.4.4 / scipy-openblas 0.3.31; **WSL**
py3.12.3 / numpy 2.4.6 (the same wheels CI's fast lane uses) / scipy-openblas
0.3.31.

---

## 3. Step 1 — one forward-orientation kernel, band UNCHANGED

Commit `bab82db`.

**What was consolidated.** Five independent bodies of the rule

```
prop = |Im q| < 1e-9 * max(|Re q|, 1e-300)
q    = q if (flux >= 0 if prop else Im q > 0) else -q
```

at `zcascade.py:86` (staggered FD, per-mode loop), `zcascade.py:227`
(`_orient_forward`, legacy nodal), `sem_radial.py:428` (SEM, vectorized),
`_jax_bor.py:98` and `_jax_sem.py:247` (the traced `jnp` twins) become
`lumenairy/elements/bor/_orient.forward_orient(q, flux, k0, *, xp=None,
band=..., scale=None)`. Two companion decisions travelled with it and were
consolidated as far as is honest: the flux normalizer's fallback PREDICATE
(`flux_is_strong`, six copies) and the channel gate's `{imag, real-floor}` core
(`channel_core`, five copies). Each site keeps its own basis-specific leg —
`reldiv` for the nodal basis, the index ceiling for the staggered twins — the
split audit S1-16 already justified, and the normalizer's five fallback SCALES
stay at their call sites because folding them would change numbers rather than
deduplicate a decision.

Two per-mode loops in `zcascade.py` were restructured to evaluate every mode's
un-oriented `z`-flux in a first pass and orient the whole spectrum in one call.
The per-mode arithmetic is unchanged; only its position in the loop nest moved,
which is what makes the bit-identity claim checkable. `_orient_forward` became
`_nodal_zflux`: the basis-specific QUANTITY the shared decision reads.

**The bit-identity contract, established by running.**

| population | Windows | WSL |
|---|---|---|
| 148 BOR gates + `test_bor_sem_jax.py` + `test_v5_20_11_bor_jax.py` | **162 passed**, 992.5 s | **156 passed** (the two JAX files are Windows-only here) |
| 30-fixture R/T battery, SHA-256 of the exact IEEE-754 bytes | **30 of 30 identical, 0 moved** | **30 of 30 identical, 0 moved** |

The battery spans both bases x `m` = 0,1,2,5 x `k0` = 0.8/2.0/3.5 x four
geometry families (ring pair, uniform, three-layer with spacer, lossy ring), and
contains no near-cutoff order by construction — the population step 2
deliberately moves is a separate one.

**New gates (8, 23.7 s).** Exactly one definition of each of the four helpers,
pinned by a GREP over the package rather than by an identity check on an
imported name (an unimported copy is invisible to the latter —
`pmm/twod_staggered.py`'s dead `_sqrt_decay` copy was exactly that); no BOR
module carrying the deleted classifier's spelling; every former copy site
importing the shared kernel with the two JAX sites passing `xp=jnp`; `band=`
live at the comparison; and an explicit `xp` agreeing with a sniffed one.

---

## 4. Step 2a — the band becomes relative to the SPECTRUM

Commit `8169d03`.

**The decision quantity.** `sigma = |Im q| / max(max|q| over the layer's
spectrum, k0)`, replacing `rho = |Im q| / max(|Re q|, 1e-300)`. The floor is
`k0` and not a literal 1.0 because `q` carries units of inverse length; a
literal 1.0 would make the band unit-system-dependent, the failure audit P2-06
fixed for the channel gate.

**What it fixes, on a 39-rung near-cutoff ladder** (Rbig = 24, N = 120,
n = 1.41, `m` = 0/1/2, `qn` from 1.4e-02 to 1.4e-05; the PEC-walled cylindrical
spectrum is discrete, so the ladder solves for one radial order's cutoff
`gamma_j` and approaches it geometrically with `k0 = gamma_j / (n sqrt(1-delta))`,
putting the order at `qn = n sqrt(delta)` exactly):

| quantity | SHIPPED per-mode band | THIS spectrum band |
|---|---|---|
| worst lossless closure | **1.2167e-04** | **1.9655e-07** (619x) |
| distinct R/T channel counts over the ladder | **{2, 3}** | **{3}** |

**The full kernel x thread ladder, Windows** (`s2_band.py`, `--fast`; each
kernel read back from `threadpoolctl`):

| requested | loaded | threads | worst closure | channel counts |
|---|---|---|---|---|
| HASWELL | Haswell | 1 | 1.9655e-07 | {3} |
| HASWELL | Haswell | 2 | 4.1679e-07 | {3} |
| HASWELL | Haswell | 4 | 6.1976e-08 | {3} |
| NEHALEM | Nehalem | 1 | 2.9240e-07 | {3} |
| NEHALEM | Nehalem | 2 | 3.1596e-07 | {3} |
| NEHALEM | Nehalem | 4 | 1.1285e-07 | {3} |
| PRESCOTT | **Katmai** | 1 | 1.2716e-06 | {3} |
| PRESCOTT | **Katmai** | 2 | 3.8318e-07 | {3} |
| PRESCOTT | **Katmai** | 4 | 1.8389e-07 | {3} |
| ZEN | **Haswell** | 1 | 1.9655e-07 | {3} |
| ZEN | **Haswell** | 2 | 4.1679e-07 | {3} |
| ZEN | **Haswell** | 4 | 6.1976e-08 | {3} |

WSL / Haswell / 1 thread: worst closure 1.0818e-07, channel counts {3}. **The
channel count is 3 on every one of the thirteen arms**, where the shipped band
gave 2 on some rungs and 3 on others and moved that verdict with the kernel on
21 of 24 rungs and with the thread count on 35 of 39. The ZEN rows being
bit-identical to the HASWELL rows is the alias measured, not assumed.

**The two-sided bar at `band = 1e-8`, re-measured on this build** (and
re-measured again by `test_band_two_sided_population` on whatever build runs
the suite):

| side | population | n | worst `sigma` | room |
|---|---|---|---|---|
| NOISE (band MUST reach) | lossless propagating, ordinary geometry | 27 | 1.3024e-15 (WIN) / 6.4173e-16 (WSL) | 6.89 decades |
| NOISE at a DEEP CUTOFF (binding) | `delta` 1e-4 .. 1e-26, `qn` to 1.4e-13 | 36 | 8.7301e-10 (WIN) / 2.3089e-10 (WSL) | 1.06 decades |
| SIGNAL (band must NOT reach) | genuinely lossy, `Im(n)` = 1e-3 | 6 | 9.4570e-05 | 3.98 decades |
| SIGNAL, thin end | `Im(n)` = 1e-6 | 6 | 9.4570e-08 | 0.98 decades |

The binding side is the deep cutoff at 1.06 decades. That population IS backward
error and grows with `||K||`, so a much finer radial grid would eat into it —
which is why the gate re-measures rather than pins.

**Why the widening is harmless, measured.** `sigma` is exactly linear in the
imaginary index, so the new band calls media with `Im(n)` between ~4e-07 and
~1e-09 "propagating" and orients them by flux rather than decay. Over **645**
physically propagating modes at `Im(n)` from 1e-06 down to 1e-10 (`m` = 0,1,2)
the flux verdict and the decay verdict agree on EVERY one — 0 disagreements —
with the flux at `|P|/fnrm >= 0.1197`, eleven decades above the normalizer's own
noise fallback. Identical on both builds to twelve digits.

**Bit-identity.** 30 of 30 hashed fixtures unchanged on both builds across the
band change; 175 gates pass on Windows (148 BOR + 14 JAX + 13 new, 605 s).

---

## 5. Step 2b — the EME exact-zero branch pins

Commit `42398be`.

| quantity | with the pin (restored for the check) | with the band |
|---|---|---|
| strip modes on a different root under `Im(eps) = 1e-30`, `Nx = 96` | **29 of 96**, worst `\|d ky\|` = 432.6 | **0 of 96**, worst 4.551e-13 |
| `layer_modes` count, scoping window (26055.8, 35530.6), `n_scan` = 300 | 62 (real `eps`) against 69 WIN / 71 WSL | equal |
| `layer_modes` count, the narrowed gate window, `n_scan` = 60 | 16 against 18 | 16 against 16 |

`lumenairy/elements/eme/_branch.py` is the one definition all three EME sites
read. OFF the cut the forward root is the decaying one, unchanged; ON the cut
(`|Im z| <= 1e-9 * max(1.0, max|z|)`) the root is CONJUGATED rather than
negated, keeping `Re z >= 0` (the vector sibling's own on-cut tie-break) while
also putting `Im z >= 0`. The floor is a literal 1.0 and not `k0`, because `ky`
in these modules is dimensionless.

**Runs.** 136 EME gates pass on Windows (1158 s) and 135 on WSL (827 s), nothing
moved. Seven new gates, 44 s; the expensive one (the `layer_modes` observable)
is pinned at 39 s on the narrowed window, which was CHECKED to still separate
16 from 18 with the pin restored.

---

## 6. Step 3 — the nodal passivity refusal

Commit `442d74a`.

**The population, measured over 132 solves BEFORE the bar was chosen**, with
the guard disarmed so the numbers are the ones the solver RETURNED
(`s3_nodal_passivity.py`: the scoping's five-layer ring stack at Rbig = 1..16
vacuum wavelengths, an N-refinement sweep, every nodal fixture the shipped suite
builds, and a 48-row small-cell census over `m` = 0,1,2 x N = 120/200 x
Rbig/lambda = 0.5..2.0):

| population | `max(R + T) - 1`, Windows | WSL |
|---|---|---|
| STAGGERED, every row, every family | 3.7406e-12 | 1.9959e-11 |
| NODAL, uniform layers on a small cell (the accurate family) | **4.4336e-09** | **4.4336e-09** |
| NODAL, everything else | **2.8819e-02 .. 6899.3** | **2.8819e-02 .. 6899.3** |

A **6.81-decade gap with nothing in it, identical on both builds**.
`_BOR_NODAL_SUPERUNITY_BAR = 1e-3` sits **5.35 decades** above the healthy
ceiling and **1.46 decades** below the mildest broken row, and the ARMED
verdict agrees with that bar on **66 of 66** nodal rows on EACH build. The
warning edge `_BOR_NODAL_SUPERUNITY_WARN = 1e-6` sits 2.65 decades above the
healthy ceiling.

The bar is deliberately NOT the 1-D peer's `_STACK_SUPERUNITY_BAR = 1e-2`: that
value carries only **0.46 decades** here, because the mildest broken row is the
SHIPPED `test_structured_stack_energy_floor_nodal` fixture at 2.8819e-02.

**Five-rung refusal ladder, Windows** (the five-layer ring stack):

| Rbig / lambda | nodal `max(R+T)` | armed verdict | staggered twin | proxy warned? |
|---|---|---|---|---|
| 1 | 3.050 | REFUSED | 1.000000000000 | **no** |
| 2 | 114.4 | REFUSED | 1.000000000001 | **no** |
| 4 | 37.91 | REFUSED | 1.000000000000 | **no** |
| 8 | 16.97 | REFUSED | 1.000000000000 | yes |
| 14 | 789.0 | REFUSED | 1.000000000000 | yes |

The three unwarned rows are the retired `Rbig/lambda > 4` proxy missing the
small end of its own population; a UNIFORM nodal stack at 12 wavelengths, which
the proxy DOES warn about, reads 1.035.

**What moved.** Exactly one shipped gate:
`test_bor_solve::test_structured_stack_energy_floor_nodal`, which asserted the
nodal basis's documented "~1-4% floor" held at `< 0.05`. It reads 1.02882 — a
2.9 % energy violation on a stack of LOSSLESS media, where `R + T <= 1` is a
theorem. That gate now asserts the refusal and names the four things the message
must contain; a sibling asserts the switch hands back the pre-fix number AND the
pre-fix assertion exactly; a third asserts the STAGGERED twin of the same
geometry still returns and closes to 1e-9.

**Runs.** 32 gates on Windows (`test_fix_bor_multilayer_guards` +
`test_bor_solve`, 77 s); 89 on WSL including the three audit files that touch
the nodal classifier (63 s).

---

## 7. Step 4 — the SEM manufactured-element contract

Commit `a072c82`.

**The census was measured FIRST, and it moved the warn edge a decade.** That
order is what the plan bound, and the reason is the 2-D Cartesian peer's round-4
correction: a census margin measured on four geometry families is a SAMPLE
property, not a library one.

**The ordinary-geometry census** (`s4_sem_census.py`, `s4_deep_taper.py`):
**86 families, 0 tripped, 0 errors**. Uniform, uniform-anisotropic,
multi-segment, coincident walls, ring gratings at three periods and duties,
hp-refined and graded meshes, a wall-free spacer between two ring layers,
`m` = 0/1/2/5, `k0` = 0.8/2/3.5/8, an nm-unit fixture whose `Rbig` and
wavelength differ from the rest by six orders of magnitude, and a lossy metal
ring — at degrees 6/8/12/16 — plus the taper staircase:

| slices | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| narrowest manufactured cell / `Rbig` | 3.13e-2 | 3.13e-2 | 1.56e-2 | 7.81e-3 | 3.91e-3 | 1.95e-3 | **9.766e-04** |
| `|q|max / (n_max k0)`, degree 12 | 29.7 | 34.4 | 59.3 | 118.1 | 235.8 | 471.2 | **942.0** |

**The scoping's candidate warn edge of 1e-3 is REFUTED by this census.** A
256-slice taper lands at 9.766e-04 — **0.977x, INSIDE the band it would have
warned on**. The scoping said in as many words that its 3.9x margin at 64
slices was sample-scoped and had to be re-measured before the edge was fixed;
this is that measurement. `_BOR_SLIVER_BAND_FRAC` is **1e-4**, where the
256-slice taper carries **9.77x (0.99 decades)** and all 86 families land
outside.

The Q bar's ordinary margin also shrank under the widened census. The scoping
quoted 1.63 decades against a 64-slice taper's 235.8; the widened census reaches
**1134.2** (256 slices, degree 8, `k0` = 0.8 — a LOWER `k0` RAISES this ratio,
so the low-`k0` arms are the demanding ones and the census sweeps `k0` as well
as the slice count). The honest margins for `_BOR_Q_EXCESS = 1e4` are

* **0.95 decades (8.82x)** above the worst ordinary geometry measured, and
* **1.20 decades (15.9x)** below the mildest rung it must refuse (the union
  ladder at `delta/Rbig` = 1e-6, degree 6, `|q|max`/ceiling = 1.5934e+05).

Both numbers come out **identical on Windows and WSL to sixteen digits**
(`s4_deep_taper_win_HASWELL_t1.json` against `s4_deep_taper_wsl_HASWELL_t1.json`:
`worst_q_excess` = 1134.2240762063623 on both).

**The decision table, on SEVENTEEN arms.** The delta ladder `1e-1 .. 1e-7` at
degrees 6/8/12 (`s4_ladder_matrix.py`) was run under kernels
{HASWELL, NEHALEM, KATMAI-via-PRESCOTT} x threads {1, 2, 4} on Windows, the same
kernels x {1, 4} on WSL, plus an UNPINNED arm on each build. **Every one of the
seventeen returns exactly one verdict table**:

| `delta/Rbig` | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|
| degree 6 | ok | ok | ok | warn | warn | **refuse** | **refuse** |
| degree 8 | ok | ok | ok | warn | warn | **refuse** | **refuse** |
| degree 12 | ok | ok | ok | warn | warn | **refuse** | **refuse** |

and the geometric conjunct reads **1.000000e-07 .. 1.000000e-01 exactly** on
every arm.

**On the SAME seventeen runs the energy closure of a SINGLE rung spreads
1,129x** — degree 8, `delta/Rbig` = 1e-7: 4.6613e-03 to 5.2624e+00 — with the
kernel and thread count alone, against the scoping's 70.90x over three kernels.
That is the measurement that forbids keying any Class-C decision on
super-unity, and the probe records it alongside the verdicts rather than
asserting it. Per-rung spreads above 3x:

| rung | closure across the 17 arms | spread |
|---|---|---|
| degree 8, `delta` 1e-7 | 4.6613e-03 .. 5.2624e+00 | **1129x** |
| degree 6, `delta` 1e-7 | 3.0393e-04 .. 2.6002e-01 | 856x |
| degree 12, `delta` 1e-7 | 1.3732e-02 .. 1.0667e+01 | 777x |
| degree 8, `delta` 1e-6 | 5.2418e-06 .. 2.7685e-04 | 52.8x |
| degree 12, `delta` 1e-5 | 1.3999e-07 .. 1.4283e-06 | 10.2x |

The damage is also **energy-invisible where it starts**: at `delta/Rbig` = 1e-4
— the first warned rung — the closure sits at 1.4e-08 .. 1.4e-07, its healthy
baseline. The warn band exists at a width the closure cannot see.

**The attribution control.** The same two walls, the same `delta`, three layers
apart with TWO wall-free spacers between them, so the `+-1` window never spans
both wall sets:

| `delta/Rbig` | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|
| narrowest cell / `Rbig`, degree 8 | 4.06e-2 | 4.17e-2 | 4.16e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 |
| cross-layer cell? | one | **none** | **none** | **none** | **none** | **none** | **none** |
| `|q|max`/ceiling | 37.0 | 37.0 | 37.0 | 37.0 | 37.0 | 37.0 | 37.0 |
| verdict | ok | ok | ok | ok | ok | ok | ok |

ONE spacer is not enough, and the scoping's first attempt at this control was
invalid for exactly that reason: `win[i] = walls[i-1] | walls[i] | walls[i+1]`,
so a wall-free layer BETWEEN the two ring layers inherits both wall sets. A gate
pins that reach by asserting the spacer DOES carry the sliver, so a future
change to `win[i]` is caught.

**The within-layer liner ladder**, where the narrow cell is the CALLER's own:

| `w/Rbig` | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|
| `|q|max`/ceiling, degree 8 | 28.5 | 255.4 | 2.52e3 | 2.52e4 | 2.52e5 | 2.52e6 |
| verdict, degrees 6 / 8 / 12 / 16 | ok | ok | ok | ok | **warn_own** | **warn_own** |
| cross-layer cell? | none | none | none | none | none | none |

Never refused — the library did not manufacture it — and warned from
`w/Rbig` = 1e-6, which is exactly where the scoping measured the DEGREE ladder
INVERTING (degree 16 becoming 144x worse than degree 6).

**The FD basis.** `basis='fd'` produces no mesh report at all and its per-order
`R` is **bit-identical** at `delta/Rbig` = 1e-4, 1e-6 and 1e-7 to the
`delta = 0` answer — the uniform radial grid cannot resolve the shift. A gate
asserts both.

**Runs.** 56 gates in `test_fix_bor_multilayer_guards.py` on Windows (427 s,
every one under the 60 s shard cap) and 87 on WSL together with `test_bor_sem`
and `test_bor_anisotropic` (414 s).

---

## 8. Step 5 — census hooks on the production inverses

Commit `c02b868`.

Nothing is armed. The population, re-measured THROUGH THE SHIPPED HOOK on this
tree (749 inverses over 14 named sites; both bases x `m` = 0,1,2,5 x
`k0` = 0.8/2/3.5, plus a coincident and a 1e-6-detuned half-space pair):

| site | n | equilibrated rcond | residual (max) |
|---|---|---|---|
| `sem_radial.assemble:inv(Mz)` | 55 | 1.608e-06 .. 8.763e-04 | 7.952e-13 |
| `sem_radial.mortar:inv(I+gamma.alpha)` | 41 | 1.800e-05 .. 1.032e-01 | 6.514e-14 |
| `sem_radial.mortar:gamma` | 41 | 1.945e-05 .. 3.730e-03 | 2.018e-14 |
| `sem_radial.mortar:alpha` | 41 | 2.699e-05 .. 1.154e-02 | 1.127e-14 |
| `coupled_radial.staggered:inv(Mz)` | 54 | 4.510e-05 .. 1.546e-03 | 4.134e-14 |
| `zcascade.interface:solve(Wb,Wa)` | 43 | 7.473e-05 .. 3.742e-03 | 1.051e-14 |
| `zcascade.interface:solve(Vb,Va)` | 43 | 1.015e-04 .. 1.724e-03 | 5.573e-15 |
| `zcascade.star:inv(I-B11.A22)` | 112 | 3.524e-03 .. 1.000e+00 | 8.025e-16 |
| `zcascade.star:inv(I-A22.B11)` | 112 | 6.199e-03 .. 1.000e+00 | 9.598e-16 |
| `zcascade.interface:inv(a+b)` | 43 | 3.838e-03 .. 1.000e+00 | 2.379e-15 |
| `sem_radial.mortar:M0a` / `M0b` | 41 ea | 9.265e-02 .. 1.709e-01 | 2.412e-16 |
| `sem_radial.mortar:M1a` / `M1b` | 41 ea | 1.000e+00 | 2.056e-30 |

**Whole census: rcond 1.6076e-06 .. 1.0000, residual at most 7.9518e-13** — one
population, consistent with the scoping's 2,031-inverse reading (1.986e-07 ..
1.000, residual 3.408e-13) over a wider fixture set. Three to seven decades
clear of every Cartesian bar, so nothing is armed, and the hook is what lets
that be re-measured rather than assumed.

---

## 9. Bit-identity, end to end

The 30-fixture battery hashed to the SHA-256 of the exact IEEE-754 bytes of `R`
and `T`, at 5.45.0 and after ALL FIVE steps:

| build | fixtures | identical | moved |
|---|---|---|---|
| Windows py3.14 / numpy 2.4.4 | 30 | **30** | **0** |
| WSL py3.12 / numpy 2.4.6 | 30 | **30** | **0** |

---

## 10. What could not be measured

* **The CI's ZEN kernel.** `OPENBLAS_CORETYPE=ZEN` silently returns the Haswell
  kernel on this host's OpenBLAS 0.3.31 — confirmed here rather than assumed,
  by every ZEN arm coming out bit-identical to its HASWELL twin — and
  `SKYLAKEX` loads but dies on the first LAPACK factorisation because this CPU
  (Ryzen 9 5950X, Zen 3) has no AVX-512. The ladders therefore cover Haswell,
  Nehalem and Katmai across two builds, seventeen arms in all. That was enough
  to move the SHIPPED rule's decisions and to spread the Class-C closure by
  1,129x; but the specific ZEN row the CI reported is **not reproduced here**,
  and the bars' ZEN behaviour is inferred from their kernel-exactness on the
  three kernels measured, not observed. Re-running `s4_ladder_matrix.py` and
  `s2_band.py --fast` on the EPYC runner is the outstanding item.
* **PML and anisotropic-path re-characterisation — OPEN ITEM, out of scope.**
  `SemRadialMesh(R_pml=...)` is reachable only by direct construction;
  `BORStack._solve_sem` never sets it, so the Class-C ladders were measured on
  the non-PML path only. Whether the manufactured-element contract's
  populations look the same under a PML — whose complex coordinate stretch
  changes `|q|max` by construction — is NOT measured, and the contract makes no
  claim there. Reproducer: build two `SemRadialMesh(bnd, eps, p, R_pml=...)`
  whose walls differ by `delta`, cross-test them with
  `sem_radial.sem_interface_smatrix`, and compare `max|q|` against the non-PML
  twin.
* **The `equalize_meshes` dropped-kwargs latent bug — OPEN ITEM, out of scope.**
  `sem_radial.equalize_meshes` rebuilds every mesh as
  `SemRadialMesh(b, eps, msh.p)`, **dropping `R_pml`, `sigma_max`, `pml_p` and
  `nq_extra`**. A caller who builds PML meshes by hand and equalizes them
  silently loses the PML. Not reachable through `BORStack` today (which never
  sets them), which is why it is an open item and not a step. Reproducer:

      m = SemRadialMesh(bnd, eps, 8, R_pml=0.8 * Rbig)
      out = equalize_meshes([m, SemRadialMesh(bnd2, eps2, 8)])
      assert out[0].R_pml == m.R_pml        # FAILS: it is None

* **Which of `Rbig`, the local wavelength or the layer's own ordinary element
  width governs the Class-C width scale.** The scoping measured all three and
  none holds still to better than 30x, which is why the width fraction
  ATTRIBUTES and the spurious-wavenumber ratio DECIDES. Settling it would need
  a wider `(Rbig, k0, degree)` design than the five cases run.
* **An armed refusal bar for the EME class-B inverses.** The population is
  measured (`a+b` cond 2.09 .. 19.34 healthy, 1.86e+04 approaching a strip band
  edge) but there is no two-sided gap: the EME scalar mode solver has no energy
  oracle, so the only separator available was distance-to-band-edge, which is a
  continuum and not two populations.
* **An external accuracy oracle for the SEM answer.** `basis='fd'` is immune to
  Class C and was used as the structural control, but it is not converged at
  these settings. Class C's wrongness rests on CONTINUITY (the answer moved 586x
  to 5,428x the physical wall shift) and on the separated-arm control, not
  against an oracle.
* **`m` beyond 10, degrees beyond 16, and the JAX twins under Class-C
  fixtures** (their parity was measured on ordinary geometry only).

---

## 11. The full run matrix

Every BOR and EME test file, plus the new gate files, on both builds. The
kernel-and-thread ladders are in sections 4 and 7 — the decision quantities they
cover are the ones that could move with the arithmetic; the test files
themselves are deterministic under them, so the matrix below reports the
pinned-single-thread run of each file group per build and the ladder arms
separately rather than re-running every file twelve times.

| group | files | Windows py3.14, 1 thread | WSL py3.12, 1 thread |
|---|---|---|---|
| BOR gates | `test_bor_sem`, `test_bor_solve`, `test_bor_anisotropic`, `test_audit_bor_grazing_cutoff`, `test_audit_p1_bor_flux`, `test_audit_w6_bor`, `test_niche_audit_w6_bor`, `test_audit_v5_24_2_b2_bor_exports` | in the 368 below | in the 346 below |
| BOR JAX twins | `test_bor_sem_jax`, `test_v5_20_11_bor_jax` | in the 368 below | not installed |
| EME | `test_eme_2d`, `test_eme_2d_vector`, `test_eme_census_determinacy`, `test_eme_diffraction`, `test_eme_jax_modes`, `test_audit_w6_eme`, `test_niche_audit_w6_eme` | in the 368 below | in the 346 below |
| NEW | `test_fix_bor_multilayer_guards` (61), `test_fix_eme_branch_cut` (7) | in the 368 below | in the 346 below |
| **TOTAL, one command** | all of the above | **368 passed**, 1093.9 s | **346 passed**, 1631.0 s |

Plus, on Windows, the census / walker / dispatcher / public-API sweep the
release gate runs:

```
python -m pytest $(ls tests/unit | grep -iE 'walker|census|dispatcher_pin|public_api|doc_consistency' \
                   | sed 's#^#tests/unit/#') -q -x -p no:randomly
-> 1288 passed, 12 skipped, 4 warnings in 125.44s
```

and `ruff` under the WSL venv over everything this build touched:

```
wsl -e bash -lc 'cd /mnt/c/tmp/lum_bor1 && ~/lumvenv/bin/ruff check lumenairy/ tests/ \
                 validation/probe_fix_bor_guards/'
-> All checks passed!
```

**Every new gate is pinned under the 60-second shard cap**, and the node ids are
spliced into `.test_durations` sorted and JSON-validated. The slowest new gates
are `test_ordinary_geometry_census[mesh-12]` at 50.1 s,
`test_the_taper_staircase_is_ordinary_at_every_slice_count[64-8-2.0]` at 42.9 s,
`[256-6-0.8]` at 36.0 s, and
`test_mode_count_is_build_independent_under_an_infinitesimal_loss` at 38.9 s.
Two of the expensive ones were deliberately re-scoped to fit: the EME
`layer_modes` observable runs on a narrowed `qz^2` window (checked to still read
16 against 18 with the pre-fix pin restored, where the scoping's full window
reads 62 against 69/71), and the deep taper arms run at a lower `degree` and
`k0` (which do not touch the geometric conjunct at all and make the spectral one
HARDER, not easier).

---

## 12. What a caller sees, in one table

| path | before 5.45.1 | after |
|---|---|---|
| `BORStack.solve()`, ordinary geometry, either basis | a number | **the same number, bit for bit** |
| `BORStack.solve(basis='fd')` near a radial cutoff | 2 or 3 R/T channels depending on the BLAS kernel and thread count; closure to 1.2e-04 | 3 channels on all 13 arms; closure <= 1.3e-06 |
| `BORStack.solve(basis='sem')` with two layers' walls `< 1e-6` of `Rbig` apart | a per-order answer moved up to 5,428x the physical wall shift, silently | `BORSemMeshError`, naming the layer, both walls and their owners, the spectral excess and three remedies |
| ... `1e-6` to `1e-4` of `Rbig` apart | the same, silently | the answer, under a `UserWarning` |
| `BORStack.solve(basis='sem')` with a caller-prescribed liner below `1e-6` of `Rbig` | silent | the answer, under a `UserWarning` — never refused |
| `bor_solve.solve(basis='nodal')`, lossless stack | `R + T` from 2.9e-02 to 6899, unwarned below 4 wavelengths | `BORNodalPassivityError`, or a `UserWarning` between 1e-6 and 1e-3 |
| `bor_solve.solve(basis='nodal')` with any lossy layer | a number | **the same number** — the screen disarms, there is no theorem to violate |
| `eme_2d.layer_modes` / `eme_diffraction.mode_match` | a mode set that changed with an `Im(eps)` of 1e-30 and differed between Windows and WSL | one mode set |

Both refusals are reversible by a module-level boolean, and both error classes
are importable from `lumenairy.elements.bor` so they can be caught by name.

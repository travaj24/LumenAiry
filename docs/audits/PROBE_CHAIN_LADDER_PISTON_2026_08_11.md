# PROBE -- design-121 chain-grid convergence ladder, and inter-chain piston consistency

**2026-08-11.  Branch `main`, commit `755ad99` (`v5.34.0`), working tree
unmodified under `lumenairy/**`.  No `git` or `gh` command was run;
`CHANGELOG.md` was not touched.  Everything added by this probe is two runner
scripts under `validation/repro_traced_carrier_121/` and this document.**

Two questions, both left open by `CAPSTONE_D121_2026_08_06.md`:

1. **The chain has only ever been run at `N = 1024`, `dx0 = 2.0 um`.**  Nothing
   has ever varied it.  The capstone's own un-silenced warning surface says the
   coarse legs are 2.75x under-sampled on the exit wavefront and that 29-40 %
   of Newton pixels "keep their last iterate".  Does the answer move?
2. **Aggregating beams makes ABSOLUTE OPL bookkeeping load-bearing across
   chains for the first time.**  Is the per-order piston at the aperture
   trustworthy at lambda/100?

---

## 0. VERDICT

> **LADDER.**  The deficiency is **PITCH, not WINDOW** -- and it is not really
> `N` either, it is the RAY LATTICE `N/ray_subsample`.  The 2.048 mm launch
> window is converged at `N = 1024` on every metric (energy **6.4e-08**,
> EE-class **1.1e-13**, and it stays there through a 4x window); the halo
> outside the 1024 footprint measures **1.05e-09** of the power, so there is
> nothing to clip.  Refining the PITCH does move the answer, and the verdict
> splits by metric:
> **aperture ENERGY is NOT converged at 1024** -- `dP/P` = **4.7e-03 / 5.2e-03**
> from 1024 to 2048, **118x / 130x** the campaign's 4e-05 bar, and still
> 1.3e-03 from 2048 to 4096; **envelope rel L2 is marginally not converged**
> (3.8e-03 / 4.4e-03 against 1e-03, reaching 1.2e-03 / 1.5e-03 only by 4096);
> and the **EE-class bar PASSES at 1024** -- 1 - Strehl (Marechal, on the
> piston-removed aperture wavefront) = **1.6e-07 / 2.0e-04** against 1e-03,
> with R90/R99 of the aperture irradiance stable to 1.2e-03 relative.
> **The cheap fix is not a bigger `N`.**  `ray_subsample = 1` at `N = 1024`
> recovers the missing energy (99.979 % against 99.353 %) in **113 s**, where
> `N = 4096` at `rs = 4` reaches only 99.953 % in **400 s**.
>
> **PISTON.**  Reproducibility is perfect and correctness is not.
> (a) a fresh process reproduces the aperture field **BIT-FOR-BIT** (sha256
> equal, `max|dE| = 0`, 8/8 pairs); (c) the inter-order piston DIFFERENCE is
> identical to every printed digit across three independent processes
> (drift 0.0e+00); (d) a 1e-12 relative upstream perturbation moves the piston
> by **1e-07 rad**, a gain of ~**1e+05**, i.e. **seven decades below** the 1e12
> lottery `FIX_C13_BUILD_SPREAD_2026_08_06` measured -- because the C6
> stationary-phase launch is *admissible* on a single congruence here
> (`max|grad a|` = 1.3e-02..3.3e-02 against its physical bound of 1, fit
> residual/data 0.02-0.07).  **But** measured against an INDEPENDENT exact skew
> ray trace of the same chief ray, the inter-order piston is wrong by
> **0.050 to 0.416 waves** -- **5x to 42x outside lambda/100** -- and it does
> **not** converge with grid, with `ray_subsample`, or with any element-fit
> lever.  **Cross-chain piston is NOT trustworthy at lambda/100 today.**
>
> **MECHANISM (named, localised, and reduced to a one-line reproducer).**  The
> tilted-congruence transport accumulates a **tilt-QUADRATIC (obliquity) piston
> that the real relay cancels and the chain does not.**  Measured on a
> continuous tilt axis at the FIRST post-DOE group, the chain reproduces
> **95.2 %** of the tilt-quadratic optical path -- 0.9522989, 0.9522239,
> 0.9521778, 0.9521301, 0.9516800, 0.9504014 at tilts of 23.03, 11.52, 5.76,
> 3.45, 1.15 and 0.35 mrad, a **4.77-4.96 % deficit held to within 0.2 points
> over a 66x span in angle and a 4400x span in the quantity itself**.  A fixed
> FRACTION over four decades of a `theta^2` term is a missing/mis-scaled TERM, not a
> discretisation error, and the per-group leftovers do not cancel: by the sixth
> group the chain carries an effective `theta^2` piston coefficient
> **~2.1e+04 times** the real system's, whose own field-dependent OPL at that
> plane is sub-wave by design (it is an imaging relay).
>
> **CONSEQUENCE FOR SUM-AT-APERTURE.**  Summing chains at the aperture is
> sound in ENERGY and in ENVELOPE SHAPE at `N = 1024` (with `rs = 1`), and the
> per-chain answer is bit-reproducible, so the architecture's determinism is
> not at risk.  **The relative phases between chains are not usable as they
> stand.**  Aggregation needs a piston reference before it can be coherent.

---

## 1. WHAT WAS MEASURED

### 1.1 Box and build

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
127.9 GB physical RAM            103 GB free at launch
python 3.14.6                    numpy 2.4.4   scipy 1.17.1
lumenairy 5.34.0  (main @ 755ad99, tag v5.34.0)
numba present (library gate _lens_traced._NUMBA_AVAILABLE = True)
REMAP_STATIONARY_PHASE_LAUNCH = True,  _REMAP_RESID_DEGREE_CAP = 6
n_workers = 1 on every run (CAPSTONE S5: the shipped 8 drives this box to
0.0 GB free; the library documents and tests the pool as a speed knob only)
```

### 1.2 The stop plane: the last group's BACK APERTURE

`propagate_traced_carrier_chain(..., final_distance=0.0, focus_readout=None)`
lands exactly on the exit vertex of the last post-DOE group -- the plane an
aggregating consumer sums at.  `final_leg` is not read at all on that call
(`carrier.py` consults it only inside `if is_final and focus_readout is not
None`; the capstone proved the two spellings byte-identical in S2.5), so
nothing about the exact high-NA readout is silently in play.

Stopping there is not a convenience.  The final leg is ~90 % of the per-order
wall clock and introduces its OWN grid (`n_fine_cap`, `window_factor`,
`dx_out`) whose convergence is a different question from the COARSE CHAIN GRID
this probe is about.  Measured at the stop plane:

```
groups: 2 before the DOE, 6 after; last-lens -> DOE 51.5393 mm, DOE -> next 7.0000 mm
DOE period 113.7566 um; order (m,n) tilt = (m,n) * lambda / period
R at the aperture      -7.712425 mm     (matches CAPSTONE's -7.712 mm)
dx at the aperture     33.2112 um       aperture window 34.0083 mm  (N = 1024)
amplitude radius       1.185 mm         -> the beam fills 7 % of the window
```

### 1.3 WHICH LADDER -- adjudicated, not assumed

Two deficiencies are possible and they are different experiments:

* **PITCH** -- too coarse a `dx` for the wavefront the chain carries.  Arm:
  `N` = 1024/2048/4096/8192 with `dx0` = 2.0/1.0/0.5/0.25 um, i.e. the
  **launch window held at 2.048 mm**.
* **WINDOW** -- the beam's halo clipped by a launch window that is too small.
  Arm: `dx0` pinned at 2.0 um, `N` = 1024/2048/4096, i.e. the **pitch held**
  and the window grown to 2.048 / 4.096 / 8.192 mm.

Both were run, because the prior evidence points both ways.  The library's own
docstring says the extent-preserving axis is already flat ("the chain is
dx-FLAT ... four significant figures across a 4x refinement") and that "large
`N` is only ever meaningful PITCH-PRESERVING"; the capstone's un-silenced
warnings say the opposite ("2.75x exit-wavefront under-sampling on a coarse
chain leg"), and under-sampling is a pitch statement.  The measurement decides.

The two arms are also the only ones that admit an EXACT comparison.  The
co-moving magnification is a property of the R-chain, not of the grid, so the
PITCH arm lands every rung on the same 34.0083 mm aperture window with `dx_ap`
halved -- fine index `i = nF/2 + s*(ic - nC/2)` hits the coarse rung's own
sample points exactly -- and the WINDOW arm keeps `dx_ap` and makes the coarse
rung the central block of the fine one.  **No score in this document involves
an interpolation.**

### 1.4 The piston observables

Both are read on the envelope with the sphere **and** the tilt ramp divided
out, so neither is contaminated by the carrier's own (huge) phase:

* `piston_c` -- the phase at the **chief ray** (the grid centre pixel).  This
  is the accumulated OPL along the chief ray and it is the quantity an
  aggregating consumer's inter-beam phase is built from.  It is exactly
  comparable to a ray trace: at `u = v = 0` the parabola, the tilt ramp, the
  niche-C5 exactness term, the sphere/parabola conversion and the
  `TiltedCarrier` eikonal itself are ALL identically zero
  (`W(0,0) = sign(R)(|R| sqrt((L^2+M^2)/N^2 + 1) - |R|/N) = 0` for any tilt),
  so `piston_c` is the physical field's own phase at that point.
* `piston_w = arg(sum |E| E)` -- the intensity-weighted circular mean, i.e.
  the beam-weighted global phase.

`piston_c` is not a single-pixel artefact: over the 7x7 block about the chief
ray the envelope phase spans **2.4e-05 rad rms** after one group and
**5.8e-03 rad** after six, against residuals of 0.3-2.6 rad.

### 1.5 Reproducing

```
validation/repro_traced_carrier_121/probe_ladder_run_121.py    one run
validation/repro_traced_carrier_121/probe_ladder_score_121.py  all tables
```

Both are `__main__`-guarded.  Neither installs a blanket warning filter: every
chain call runs inside `catch_warnings(record=True) + simplefilter('always')`,
which REPLACES the filter list and therefore restores the three targeted
`filterwarnings('ignore', message=...)` calls `_d121_common` installs at
import.  That is how the warning census below is un-suppressed without editing
the shared harness.

```
PL_N=1024 PL_DX0=2.0 PL_ORDER=-4,-2 PL_NW=1 python probe_ladder_run_121.py
PL_NGROUPS=1 ...      stop after k post-DOE groups (the localiser)
PL_TSCALE=0.1 ...     scale the order's (L,M) -- the CONTINUOUS tilt axis
PL_X0=1e-12 ...       force the on-axis order down the TILTED branch
PL_PERTURB=field PL_SEED=2 ...   the 1e-12 lottery arm
PL_TKW='{"fit_radius_beam_factor":1.5}'  ...  a traced-element lever
python probe_ladder_score_121.py
```

89 runs; artifacts in `_probe_ladder/` (`.json` per run plus the
carrier- and ramp-free envelope in `_<tag>_env.npz`).

---

## 2. MEASUREMENT 1 -- THE CHAIN-GRID CONVERGENCE LADDER

### 2.1 Per-rung census, with the warnings the harness normally hides

| run | N | dx0 (um) | dx_ap (um) | window (mm) | P_doe/P_in | P_ap/P_in | chain A grp/nwt%/usmp | chain B grp/nwt%/usmp |
|---|---|---|---|---|---|---|---|---|
| (0,0) | 1024 | 2.000 | 33.2112 | 34.0083 | 0.999207 | 0.993526 | 2 / 39.9 % / 5.50x | 5 / 85.4 % / 17.23x |
| (0,0) | 2048 | 1.000 | 16.6056 | 34.0083 | 0.999769 | 0.998214 | 2 / 40.0 % / 2.75x | 5 / 85.4 % / 8.62x |
| (0,0) | 2048 | 2.000 | 33.2112 | 68.0166 | 0.999207 | 0.993526 | 2 / 85.0 % / 5.50x | 6 / 96.4 % / 17.23x |
| (0,0) | 4096 | 0.500 | 8.3028 | 34.0083 | 0.999911 | 0.999533 | 2 / 39.9 % / 1.37x | 5 / 85.4 % / 4.31x |
| (0,0) | 4096 | 2.000 | 33.2112 | 136.0333 | 0.999207 | 0.993526 | 2 / **96.2 %** / 5.50x | 6 / **99.1 %** / 17.23x |
| (-4,-2) | 1024 | 2.000 | 33.2112 | 34.0083 | 0.999207 | 0.993034 | 2 / 39.9 % / 5.50x | 6 / 83.3 % / 20.89x |
| (-4,-2) | 2048 | 1.000 | 16.6056 | 34.0083 | 0.999769 | 0.998200 | 2 / 40.0 % / 2.75x | 6 / 83.4 % / 10.45x |
| (-4,-2) | 2048 | 2.000 | 33.2112 | 68.0166 | 0.999207 | 0.993031 | 2 / 85.0 % / 5.50x | 6 / 95.8 % / 20.89x |
| (-4,-2) | 4096 | 0.500 | 8.3028 | 34.0083 | 0.999911 | 0.999512 | 2 / 39.9 % / 1.37x | 6 / 83.4 % / 5.22x |
| (-4,-2) | 4096 | 2.000 | 33.2112 | 136.0333 | 0.999207 | 0.993029 | 2 / **96.2 %** / 5.50x | 6 / **99.0 %** / 20.89x |

`grp` = groups that warned Newton non-convergence (unique messages -- the
element emits each twice); `nwt%` = the worst per-group non-converged pixel
fraction; `usmp` = the worst exit-wavefront under-sampling ratio
`dx_have / dx_needed` (1.00x = Nyquist on the exit sphere).  Chain A is cached
and order-independent, so its column is read from the COLD build of that rung.

Three things this census settles that the capstone could not see:

1. **The 2.75x under-sampling is the SMALL one.**  At the shipped `N = 1024`
   the coarse legs run at **5.50x** (chain A) and **17.23-20.89x** (chain B) --
   the capstone's 2.75x was the `N = 2048` acceptance run, and it was reading
   chain A.  The ratio halves per octave exactly as it must, and even at
   `N = 4096` the worst leg is still 4.31-5.22x above Nyquist on its own exit
   sphere.  **The chain never resolves the exit wavefront at any affordable
   `N`.**
2. **The "29-40 % Newton non-convergence" headline is largely an artefact of
   the counting window, and the WINDOW arm proves it.**  Holding the physics
   fixed and growing ONLY the launch window (2.048 -> 4.096 -> 8.192 mm) takes
   chain A from **39.9 % to 85.0 % to 96.2 %** and chain B from 85.4 % to
   96.4 % to **99.1 %** -- while the answer moves by 6.4e-08 then 1.8e-08 in
   energy and 5.5e-07 then 1.3e-07 in envelope rel L2.  The fraction is over
   the whole `N/rs` lattice, most of which is far outside the beam, so a 99.1 %
   reading is compatible with a bit-stable, converged field.  It is the wrong
   denominator to report; it is still true that some in-beam pixels keep their
   last iterate, and S2.4 bounds what that costs (nothing: the field is
   byte-identical at 12 and 40 Newton iterations).
3. **Chain A's own throughput is dx-limited too**: 99.9207 / 99.9769 /
   99.9911 % of the launch power at `dx0` = 2.0 / 1.0 / 0.5 um.

### 2.2 The WINDOW arm -- converged at N = 1024

| order | rung pair | amp relL2 | cplx relL2 | piston-free | d piston (rad) | phase rms (rad) | 1 - Strehl | dP/P | halo outside the 1024 window |
|---|---|---|---|---|---|---|---|---|---|
| (0,0) | 1024 -> 2048 | 2.60e-07 | 5.66e-07 | 5.53e-07 | 1.22e-07 | 3.31e-07 | 1.10e-13 | 6.44e-08 | 0.00e+00 |
| (0,0) | 2048 -> 4096 | 5.00e-08 | 1.28e-07 | 1.27e-07 | 1.59e-08 | 6.08e-08 | 3.66e-15 | 1.77e-08 | 0.00e+00 |
| (-4,-2) | 1024 -> 2048 | 5.99e-05 | 2.52e-03 | 1.68e-04 | 2.52e-03 | 1.27e-04 | 1.61e-08 | 3.51e-06 | 1.05e-09 |
| (-4,-2) | 2048 -> 4096 | 2.55e-05 | 1.40e-03 | 3.95e-04 | 1.34e-03 | 3.92e-04 | 1.54e-07 | 1.64e-06 | 1.26e-09 |

**Verdict: the launch window is not the deficiency.**  Doubling it to 4.096 mm
changes the energy by 6e-08 / 4e-06 (bar 4e-05) and the EE-class by 1e-13 /
2e-08 (bar 1e-03); doubling it again to 8.192 mm changes nothing further
(1.8e-08 / 1.6e-06, 4e-15 / 2e-07), and R90/R99 do not move in the sixth
decimal at any rung.  The direct measurement of the thing the arm exists to
detect -- power outside the 1024 footprint at the aperture -- is **1.05e-09**,
rising only to 1.26e-09 at a 4x window.  Note the one column that is NOT
negligible: the extreme order's piston moves **2.5e-03 rad** when only the
window changes.  That is 25x under the lambda/100 bar, but it is a first sign
that the piston is the sensitive observable on this chain (S3).

### 2.3 The PITCH arm -- energy is NOT converged; everything else is

| order | rung pair | amp relL2 | cplx relL2 | piston-free | d piston (rad) | phase rms (rad) | 1 - Strehl |
|---|---|---|---|---|---|---|---|
| (0,0) | 1024 -> 2048 | 3.82e-03 | 3.85e-03 | 3.85e-03 | -2.56e-05 | 4.00e-04 | 1.60e-07 |
| (0,0) | 2048 -> 4096 | 1.16e-03 | 1.20e-03 | 1.20e-03 | 3.59e-05 | 2.88e-04 | 8.27e-08 |
| (-4,-2) | 1024 -> 2048 | 4.44e-03 | 1.96e-02 | 1.19e-02 | 1.55e-02 | 1.40e-02 | 1.97e-04 |
| (-4,-2) | 2048 -> 4096 | 1.35e-03 | 1.48e-02 | 1.49e-03 | 1.47e-02 | 7.88e-04 | 6.21e-07 |

| order | rung | P_ap/P_in | dP/P vs previous | R90 (mm) | R99 (mm) |
|---|---|---|---|---|---|
| (0,0) | 1024 | 0.993525889 | -- | 1.271605 | 1.812373 |
| (0,0) | 2048 | 0.998213711 | **4.718e-03** | 1.270086 | 1.808489 |
| (0,0) | 4096 | 0.999532958 | **1.322e-03** | 1.269054 | 1.807288 |
| (-4,-2) | 1024 | 0.993034135 | -- | 1.271171 | 1.809632 |
| (-4,-2) | 2048 | 0.998199882 | **5.202e-03** | 1.268348 | 1.805819 |
| (-4,-2) | 4096 | 0.999511785 | **1.314e-03** | 1.267288 | 1.804310 |

The aperture power converges cleanly toward 1 as `dx^~1.85` -- it is a
numerical loss, not vignetting (the WINDOW arm holds it fixed at 0.993526 to
six decimals while doubling the field of view).  But at the shipped rung it is
**6.5e-03 short of the limit**, and the step to the next rung is **4.7-5.2e-03
= 118-130x the campaign's 4e-05 energy bar**.  It is still 33x the bar at
`N = 2048 -> 4096`.

Everything else is converged at 1024 on its own bar:

* **EE-class.**  Mapping the campaign's 1e-03 EE bar onto the aperture plane
  through Marechal (`Strehl ~ exp(-sigma^2)` for an intensity-weighted,
  piston-removed rms wavefront difference `sigma`) gives **1.6e-07** on axis
  and **2.0e-04** at the extreme order for 1024 -> 2048, both inside 1e-03.
  The direct EE analogue agrees: R90 and R99 of the aperture irradiance move
  by 1.2e-03 and 2.1e-03 RELATIVE over the same step, and by 8e-04 over the
  next.
* **Envelope shape.**  `|env|` rel L2 3.8e-03 -> 1.2e-03; the complex rel L2 at
  the extreme order is dominated by the piston (1.96e-02 raw against 1.19e-02
  piston-free at the first step, and 1.48e-02 against 1.49e-03 at the second --
  by 4096 the SHAPE has converged 10x and the piston has not moved at all).

### 2.4 The lever is `ray_subsample`, not `N`

The traced entrance->exit map is built on an `N/ray_subsample` lattice and
bilinearly back-filled.  That lattice -- not `dx` -- is what the missing energy
is bounded by, and the PITCH arm only improves it as a side effect.

| order | arm | P_ap/P_in | dP/P vs baseline | Newton non-convergence per group | worst usmp | wall (s) |
|---|---|---|---|---|---|---|
| (0,0) | `N=1024 rs=4` (shipped) | 0.993525889 | -- | 26.8 / 22.8 / 7.7 / 20.0 / 85.4 % | 17.23x | 28.3 |
| (0,0) | `N=1024 rs=1` | **0.999788183** | **+6.30e-03** | 26.5 / 22.9 / 7.7 / 19.6 / 85.4 % | 17.23x | 112.6 |
| (0,0) | `N=2048 rs=4` | 0.998213711 | +4.72e-03 | 26.5 / 23.1 / 7.7 / 19.6 / 85.4 % | 8.62x | 69.6 |
| (0,0) | `N=4096 rs=4` | 0.999532958 | +6.04e-03 | 26.5 / 22.9 / 7.7 / 19.6 / 85.4 % | 4.31x | 400.5 |
| (-1,0) | `N=1024 rs=4` | 0.993271391 | -- | 26.7 / 22.8 / 7.7 / 19.2 / 81.4 % | 18.04x | 24.6 |
| (-1,0) | `N=1024 rs=1` | **0.999756079** | **+6.53e-03** | 26.5 / 22.9 / 7.7 / 19.5 / 81.4 % | 18.04x | 176.9 |

Two readings, both load-bearing:

* **`rs = 1` at `N = 1024` beats `N = 4096` at `rs = 4` on energy (99.979 % vs
  99.953 %) at 3.6x less wall clock (113 s vs 400 s).**  If the aperture energy
  budget matters -- and for a sum-at-aperture consumer it does -- that is the
  configuration to run, not a bigger grid.
* **The Newton non-convergence fractions barely move between `rs = 4` and
  `rs = 1`** -- 26.7/22.8/7.7/19.2/81.4 against 26.5/22.9/7.7/19.5/81.4, i.e.
  <= 0.3 points -- while the energy moves by 6.5e-03.  So the missing power is
  the **lattice back-fill**, not the Newton residual -- confirmed independently
  in S3.6, where raising `newton_max_iters` 12 -> 40 leaves the field
  **byte-identical**.

### 2.5 LADDER VERDICT, per metric

| metric | bar | 1024 -> 2048 | converged at 1024? |
|---|---|---|---|
| aperture energy `dP/P` (PITCH) | 4e-05 | 4.72e-03 / 5.20e-03 | **NO -- 118x / 130x the bar** |
| aperture energy `dP/P` (WINDOW) | 4e-05 | 6.4e-08 / 3.5e-06 | YES |
| EE-class, `1 - Strehl_equiv` | 1e-03 | 1.60e-07 / 1.97e-04 | YES |
| R90 / R99 of the aperture irradiance | 1e-03 | 1.2e-03 / 2.1e-03 (rel) | borderline; 8e-04 by 4096 |
| envelope `|env|` rel L2 | 1e-03 | 3.82e-03 / 4.44e-03 | NO (1.2e-03 / 1.4e-03 by 4096) |
| complex envelope rel L2, piston-free | 1e-03 | 3.85e-03 / 1.19e-02 | NO (1.2e-03 / 1.5e-03 by 4096) |
| carrier-referenced phase rms | -- | 4.0e-04 / 1.4e-02 rad | YES on axis; 1.4e-02 rad = lambda/450 off axis |
| piston (see S3) | lambda/100 = 6.28e-02 rad | -2.6e-05 / +1.55e-02 rad | within the bar, 4x of margin |
| halo outside the 1024 window | -- | 1.05e-09 | **the window is not clipping** |
| under-sampling warnings | -- | 5.50x -> 2.75x (A), 17.2x -> 8.6x (B) | never satisfied at any rung |
| Newton non-convergence | -- | unchanged (39.9 %, 85.4 %) | dx-invariant |

**One-line answer: `N = 1024` is sufficient for the SHAPE and the PHASE
STRUCTURE of the aperture field and insufficient for its ENERGY by two orders
of magnitude, and the remedy is `ray_subsample = 1`, not a larger grid.**

---

## 3. MEASUREMENT 2 -- INTER-CHAIN PISTON CONSISTENCY

### 3.1 (a) Same order, same rung, FRESH PROCESS -- bit-identical

| run | repeat | d piston_c | d piston_w | max abs(dE) | sha256 equal |
|---|---|---|---|---|---|
| (0,0) | rep1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (0,0) | rep2 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (-1,0) | rep1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (-2,0) | rep1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (-3,0) | rep1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (-4,0) | rep1 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| (-4,-2) | rep1, rep2 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |

**8 of 8 pairs bit-identical**, including under different concurrent system
load (the repeats were run while the 4096 rungs were in flight).  A ninth pair
is stronger still: the `(0,0)` baseline was re-run at the end of the session
with `PL_NOCACHE=1`, i.e. with chain A REBUILT FROM SCRATCH rather than
reloaded from its cache, and reproduced `piston_c = -1.5418771246688567` and
the same `field_sha256` -- so the bit-identity survives the cache round trip
as well as the process boundary.  The piston is bit-stable in the sense (a)
asks for.

### 3.2 (b) Piston shift with the grid rung

| order | axis | rung pair | d piston_c (rad) | vs lambda/100 |
|---|---|---|---|---|
| (0,0) | PITCH | 1024 -> 2048 | -2.56e-05 | 2455x inside |
| (0,0) | PITCH | 2048 -> 4096 | +3.59e-05 | 1750x inside |
| (0,0) | WINDOW | 1024 -> 2048 | +1.22e-07 | -- |
| (-4,-2) | PITCH | 1024 -> 2048 | +1.55e-02 | **4.0x inside** |
| (-4,-2) | PITCH | 2048 -> 4096 | +1.47e-02 | **4.3x inside** |
| (-4,-2) | WINDOW | 1024 -> 2048 | +2.52e-03 | 25x inside |

The on-axis piston is grid-converged.  **The extreme order's piston is not**:
it moves 1.5e-02 rad per octave and is NOT shrinking (1.55e-02 then 1.47e-02),
so it is not a discretisation residual with a convergence order -- it is a
term whose value happens to depend weakly on the grid.  It stays inside
lambda/100 with only 4x of margin, and it has no demonstrated limit.

### 3.3 (c) Inter-order piston DIFFERENCE, across independent processes

| process | piston_c (0,0) | piston_c (-4,-2) | difference | drift |
|---|---|---|---|---|
| run 1 | -1.541877125 | +0.649377994 | -2.191255119 | 0.0e+00 |
| run 2 (fresh) | -1.541877125 | +0.649377994 | -2.191255119 | 0.0e+00 |
| run 3 (fresh) | -1.541877125 | +0.649377994 | -2.191255119 | 0.0e+00 |

**Zero drift, to every printed digit.**  Whatever the chain computes for the
inter-order phase, it computes the same thing every time.

### 3.4 (d) The 1e-12 sensitivity -- gain ~1e+05, not 1e+12

Perturbation applied strictly DOWNSTREAM of the chain-A cache, so the clean
and perturbed arms share a bit-identical chain A.  `field` = complex Gaussian
relative jitter on the DOE-plane envelope (the size two backward-stable LAPACK
paths differ by); `rdoe` = the DOE-plane carrier radius scaled by `1 + 1e-12`.

| order | perturbation | d piston_c (rad) | d piston_w (rad) | rel L2 of dE | gain per unit eps |
|---|---|---|---|---|---|
| (0,0) | field, seed 1 | 2.42e-07 | 9.19e-08 | 1.39e-07 | 2.4e+05 |
| (0,0) | field, seed 2 | -4.20e-09 | -5.51e-09 | 5.39e-08 | 4.2e+03 |
| (0,0) | field, seed 3 | 5.65e-07 | 2.12e-07 | 2.90e-07 | 5.6e+05 |
| (0,0) | rdoe | 1.04e-07 | 4.54e-08 | 1.56e-07 | 1.0e+05 |
| (-4,-2) | field, seed 1 | 2.23e-07 | 1.60e-07 | 9.86e-07 | 2.2e+05 |
| (-4,-2) | field, seed 2 | 4.24e-07 | 3.79e-07 | 7.60e-07 | 4.2e+05 |
| (-4,-2) | field, seed 3 | 9.70e-08 | 1.01e-07 | 1.89e-06 | 9.7e+04 |
| (-4,-2) | rdoe | 2.88e-07 | 2.72e-07 | 5.99e-07 | 2.9e+05 |

**The c13 lottery is not running here.**  A 1e-12 perturbation buys ~1e-07 rad
of piston -- lambda/6e+07 -- against `FIX_C13_BUILD_SPREAD_2026_08_06`'s 1e12
gain and "the piston to any value on the circle".  The reason is measurable and
is the same one that document gives: the C6 stationary-phase launch is
degenerate only on a MULTI-VALUED (multiplexed) input.  Here every chain
carries ONE congruence:

| run | C6 calls | engaged | degree | max abs(grad a) | fit residual / data rms | admissible (abs(grad a) <= 1)? |
|---|---|---|---|---|---|---|
| (0,0), N=1024 | 8 | 8 | 6 | **1.263e-02** | 0.0645 | YES |
| (-1,0), N=1024 | 6 | 6 | 6 | 1.509e-02 | 0.0724 | YES |
| (-4,-2), N=1024 | 6 | 6 | 6 | **3.257e-02** | 0.0633 | YES |
| (0,0), N=2048 | 8 | 8 | 6 | 1.264e-02 | 0.0196 | YES |
| (-4,-2), N=2048 | 6 | 6 | 6 | 3.324e-02 | 0.0268 | YES |

Against the multiplexed fixture's `max|grad a| = 974` (a transverse direction
cosine two to three orders past its physical maximum of 1) and residual/data
rms of 1.034 (a fit explaining NONE of its own data), these are textbook fits:
`|grad a| <= 3.3e-02`, and the model explains 93-98 % of the measured slope.
**The C6 machinery is in the path and it is benign in this configuration.**
That is a per-configuration finding, not a general one -- it says nothing about
a chain that multiplexes.

### 3.5 Is the OPL accounting DOCUMENTED as absolute?

**Yes, explicitly, in three places, and the code matches the documentation.**

* `carrier.py`, the multi-orchestrator note: *"D1's tilted focus readout takes
  `centre_out` in ABSOLUTE (optical-axis) coordinates and returns the window
  carrying the absolute tilt ramp and path piston, so every congruence lands
  already expressed on the same absolute lattice and the recombination is a
  plain add."*
* `_fresnel_tf_axis` / the astigmatic driver: *"Carries the on-axis piston
  `exp(i k z)` ... so composed legs stay phase-faithful"*, and the astigmatic
  path explicitly divides the doubled `exp(i k z)` back out once "a global
  phase -- it does not touch any intensity, but keeps composed legs
  phase-faithful".
* `_lens_traced.py`: *"Also add the bulk glass piston (constant `k*n*t_i` in
  each glass) ... The piston is a rigid offset but keeping it preserves
  absolute-phase consistency"*.
* The exact-kernel note is explicit that the piston is the CALLER's:
  *"The caller already owns the piston (`exp(i k z (1/N - 1))` on top of the
  kernel's own `k z`) and the chief-ray advance ... Re-supplying either here
  would double-count it."*

So the contract is absolute, the consumer is entitled to rely on it, and
S3.6 measures whether it is met.

### 3.6 THE KILL -- the piston is reproducible and WRONG

Stability is not correctness.  The inter-order piston difference was scored
against an INDEPENDENT oracle: an exact skew ray trace of the same chief ray
from the DOE plane to the same exit vertex, whose accumulated `opd` is the
geometric OPL.  Both congruences enter at (0,0) on the DOE plane with the same
field value; both are referenced to their own chief ray, where every reference
term is identically zero (S1.4); and the chain's own chief-ray prediction
agrees with this trace to **0.0 nm** at every order.  The only thing between
`k0 * dOPL` and the measured `d piston_c` is the chain's optical-path
bookkeeping.

Full chain, `N = 1024`, `dx0 = 2.0 um`, reference order (0,0):

| order | tilt (mrad) | oracle `k0*dOPL` (rad) | measured `d piston_c` (rad) | residual (rad) | residual (waves) | vs lambda/100 |
|---|---|---|---|---|---|---|
| (-1,0) | 11.5158 | -0.000875290 | -0.674758902 | **-0.673883612** | **-0.1073** | **10.7x OUTSIDE** |
| (-2,0) | 23.0316 | +0.054791812 | -2.557771562 | **-2.612563374** | **-0.4158** | **41.6x OUTSIDE** |
| (-3,0) | 34.5474 | +0.344574100 | +0.861113601 | **+0.516539501** | **+0.0822** | **8.2x OUTSIDE** |
| (-4,0) | 46.0633 | +1.173573314 | -2.507425451 | **+2.602186542** | **+0.4142** | **41.4x OUTSIDE** |
| (-4,-2) | 51.5003 | +1.874492340 | +2.191255119 | **+0.316762779** | **+0.0504** | **5.0x OUTSIDE** |

and it does not converge with the grid:

| order | rung | residual (rad) | residual (waves) |
|---|---|---|---|
| (-4,-2) | N=1024 dx0=2.0 | +0.316762779 | +0.050414 |
| (-4,-2) | N=2048 dx0=1.0 | +0.330517770 | +0.052604 |
| (-4,-2) | N=2048 dx0=2.0 | +0.319346304 | +0.050826 |
| (-4,-2) | N=4096 dx0=0.5 | +0.344887 | +0.054891 |

nor with any lever the chain's own docstring nominates as the dx-invisible one:

| order | arm | d piston_c vs baseline (rad) | residual vs oracle (waves) |
|---|---|---|---|
| (-1,0) | baseline | -- | -0.10725 |
| (-1,0) | `fit_radius_beam_factor = 1.5` | +2.62e-03 | -0.10561 |
| (-1,0) | `fit_radius_beam_factor = 3.0` | +5.23e-03 | -0.11599 |
| (-1,0) | `newton_poly_order = 10` | -1.97e-03 | -0.10636 |
| (-1,0) | `newton_max_iters = 40` | **0.00e+00** (field byte-identical) | -0.10725 |
| (-1,0) | `ray_subsample = 1` | -3.28e-03 | -0.10785 |
| (-4,-2) | `fit_radius_beam_factor = 1.5` | +4.73e-02 | +0.05916 |
| (-4,-2) | `fit_radius_beam_factor = 3.0` | +1.67e-01 | +0.06741 |
| (-4,-2) | `newton_poly_order = 10` | +5.87e-02 | +0.06096 |
| (-4,-2) | `newton_max_iters = 40` | **0.00e+00** | +0.05041 |
| (-4,-2) | `ray_subsample = 1` | +1.50e-02 | +0.05273 |

Not the traced OPL fit, not the fit disc, not the polynomial order, not the
Newton cap (which is **provably inert**: the field is byte-identical at 12 and
40 iterations), not the ray lattice, not `dx`, not `N`.

### 3.7 The mechanism, on a continuous tilt axis

`PL_TSCALE` scales the order's `(L, M)` continuously, which converts the
residual from a function of ORDER INDEX (where a large wrapped `theta^2` term
looks like noise) into a function of ANGLE.  Measured after the FIRST
post-DOE group only (`PL_NGROUPS=1`), reference = the same run at zero tilt:

| tilt (mrad) | oracle `k0*dOPL` (rad, unwrapped) | measured (unwrapped) | **measured / oracle** | deficit |
|---|---|---|---|---|
| 0.1152 | 0.000701394 | 0.000663955 | 0.9466222 | 5.338 % |
| 0.3455 | 0.006312546 | 0.005999453 | 0.9504014 | 4.960 % |
| 1.1516 | 0.070139437 | 0.066750301 | **0.9516800** | **4.832 %** |
| 3.4547 | 0.631257729 | 0.601039495 | **0.9521301** | **4.787 %** |
| 5.7579 | 1.753509241 | 1.669652615 | **0.9521778** | **4.782 %** |
| 11.5158 | 7.014328516 | 6.679211473 | **0.9522239** | **4.778 %** |
| 23.0316 | 28.061980285 | 26.723392214 | **0.9522989** | **4.770 %** |

**A constant 4.8 % deficit across a 66x span in angle (0.345 -> 23.03 mrad)
and a 4400x span in the quantity itself**; the row at 0.115 mrad reads 0.9466
because its residual is only 3.7e-05 rad, where other numerical noise enters.
The oracle here is essentially a pure `theta^2` term (it scales as `theta^2`
to 4 digits over the same span), so this is the chain reproducing
**95.22 % of the tilt-quadratic optical path** at one group.  A fixed FRACTION
is a missing or mis-scaled TERM: a discretisation error would not hold a
constant ratio while the quantity itself changes by four decades.  The control
arms say the same thing directly -- the same 1.1516 mrad measurement, one
group, at three different discretisations:

| arm | oracle `k0*dOPL` (rad) | measured (rad) | measured / oracle | deficit |
|---|---|---|---|---|
| `N=1024 dx0=2.0 um rs=4` (shipped) | 0.07013944 | 0.06675030 | 0.951680 | 4.83 % |
| `N=1024 dx0=2.0 um rs=1` | 0.07013944 | 0.06678858 | 0.952226 | 4.78 % |
| `N=2048 dx0=1.0 um rs=4` | 0.07013944 | 0.06676432 | 0.951880 | 4.81 % |

Halving `dx` moves the deficit by 0.02 points; taking the ray lattice from
256 to 1024 moves it by 0.05 points.  **The deficit is a property of the
model, not of the grid.**

Two further facts fix the class:

* **It is not a scalar-vs-tilted BRANCH offset.**  `_parse_chain_carrier` sets
  `tilted = bool(L or M or x0 or y0)`, so forcing the on-axis order down the
  tilted branch with `x0 = 1 pm` (physically inert against a 33 um pitch)
  isolates the branch from the tilt.  Measured: the two (0,0) references agree
  to **3.4e-07 rad** at the full chain and to 5e-07 rad at every truncation,
  and every residual above is unchanged to 9 digits whichever reference is
  used.  The error is genuinely TILT-dependent.
* **The per-group leftovers do not cancel, and that is why the full chain is
  so much worse than 4.78 %.**  Design 121 is an imaging relay, so its true
  field-dependent OPL from the DOE plane to the last exit vertex is nearly
  zero by construction -- the oracle reads `k0*dOPL` = -0.000875 rad for the
  (-1,0) order at 11.5 mrad, i.e. an integer number of waves to within 0.2 nm.
  The chain, carrying a ~5 % leftover from each of six groups with no
  cancellation, produces instead a `theta^2` piston whose coefficient
  corresponds to an equivalent air path of **0.374 m** against the system's own
  **1.8e-05 m** -- a factor of **2.1e+04**.  Measured on the same continuous
  axis at the FULL chain: residual **+0.011895 rad at 0.1152 mrad** and
  **+1.186825 rad at 1.1516 mrad**, a ratio of **99.8 for a 10x tilt**, i.e.
  `theta^2` to three digits.

**Statement of the defect, for the owner:** *the tilted-congruence transport
(`propagate_traced_carrier_chain` with `r_in=TiltedCarrier`) under-counts the
tilt-quadratic (obliquity) component of the chief-ray optical path by a fixed
4.78 % per group at design 121's first post-DOE group; because the surrounding
system is an imaging relay whose own tilt-quadratic OPL cancels to sub-wave
over the whole fan, the uncancelled leftovers dominate the inter-order piston
by four orders of magnitude.*  The one-line reproducer is
`PL_NGROUPS=1 PL_TSCALE=<s>` at two values of `s`.

**The FREE LEGS are exonerated by measurement, not by code reading.**  The
chain's free-leg piston is `exp(i k z)` from the kernel times
`exp(i k z (ob - 1))` with `ob = 1/cos(theta)`, i.e. `k z / cos(theta)` -- the
geometric path along the tilted chief ray.  Driving
`propagate_carrier_referenced(env, R_doe, 7.0 mm, lam, dx, tilt=(L, 0))`
directly on the design-121 DOE-plane grid and reading the chief-ray phase
against the analytic value:

| L (mrad) | analytic `k0 z (ob-1)` (rad, wrapped) | measured d piston (rad) | residual (rad) |
|---|---|---|---|
| 1.0000 | 0.016787149 | 0.016787149 | -6.9e-11 |
| 5.0000 | 0.419686277 | 0.419686275 | -1.8e-09 |
| 11.5158 | 2.226424857 | 2.226424847 | -9.8e-09 |
| 23.0316 | 2.625172615 | 2.625172576 | -3.9e-08 |
| 46.0633 | -2.022918467 | -2.022918623 | -1.6e-07 |
| 51.5003 | 0.630668098 | 0.630667902 | -2.0e-07 |

**Exact to 2.0e-07 rad at the extreme order's tilt** -- seven decades below the
0.3-2.6 rad defect.  So the 4.78 % lives at the ELEMENT hand-off (the
band-limit shift onto the axis-centred grid, the decentred sphere + ramp +
niche-C5 exactness reconstruction, `apply_real_lens_traced` with the matching
`TiltedCarrier`, and the inverse re-reference at the transferred chief ray),
not in the transport between elements.

### 3.8 PISTON VERDICT

| sub-question | result | verdict |
|---|---|---|
| (a) fresh-process re-run, same order | bit-identical, 8/8 | **PASS** |
| (b) piston shift with the rung | 2.6e-05 rad on axis; **1.5e-02 rad/octave, non-shrinking** at (-4,-2) | inside lambda/100 with 4x of margin, no demonstrated limit |
| (c) inter-order difference, fresh processes | drift 0.0e+00 | **PASS (reproducible)** |
| (d) 1e-12 upstream perturbation | 1e-07 rad, gain ~1e+05 | **PASS -- the c13 amplifier is not in this path** |
| documentation | absolute, in four places | **as documented** |
| **correctness vs an exact chief-ray OPL oracle** | **0.050 to 0.416 waves** | **FAIL -- 5x to 42x outside lambda/100** |

> **Cross-chain piston is NOT trustworthy at lambda/100 today.**  It is
> perfectly reproducible and it is wrong by up to half a wave.  Aggregation
> needs a piston-reference fix first, and the mechanism is named in S3.7:
> an uncancelled per-group tilt-quadratic path deficit in the tilted-congruence
> transport, 4.78 % per group, `theta^2` in the tilt, invariant under grid,
> lattice and every element-fit lever.

---

## 4. WHAT THIS IMPLIES FOR THE SUM-AT-APERTURE ARCHITECTURE

1. **Determinism is safe.**  Every chain reproduces bit-for-bit in a fresh
   process, and a BLAS-scale 1e-12 upstream difference moves the aperture
   piston by 1e-07 rad.  A distributed sum-at-aperture that farms one order per
   process will get the same answer every time, and the `FIX_C13` lottery does
   not apply to single-congruence chains (S3.4, with the C6 fit diagnostics
   that prove it rather than assume it).

2. **Sum the INTENSITIES today; do not sum the FIELDS across orders.**  The
   per-order envelope and its energy are sound; the relative phase is not.  A
   coherent add would impose an essentially arbitrary phase per order.  For the
   fan acceptance this is invisible, because the orders land on separate
   480 um frames and the acceptance integrates `|F|^2` per cell -- which is
   exactly why the defect has survived: **no existing acceptance metric on this
   path reads the inter-chain piston.**  Any consumer that makes beams OVERLAP
   at a shared aperture reads it directly.

3. **Run the aperture stage at `N = 1024, ray_subsample = 1`, not at a bigger
   `N`.**  113 s per order and 99.979 % of the launch power, against 400 s and
   99.953 % for `N = 4096`.  The default `rs = 4` throws away 6.5e-03 of the
   power at the aperture -- 160x the campaign's energy bar -- and no grid
   refinement recovers it as cheaply.

4. **Do not report the Newton non-convergence percentage as it stands.**  It
   is a fraction of the whole `N/rs` lattice, most of which is outside the
   beam: holding the physics fixed and doubling only the launch window takes it
   from 39.9 % to 85.0 % while the answer moves by 6e-08.  It is also
   `newton_max_iters`-inert (the field is byte-identical at 12 and 40
   iterations), so it is not measuring what its wording implies.

5. **The under-sampling warning is real, is 6x worse than the capstone
   recorded, and is not the thing that breaks the piston.**  At the shipped
   rung the coarse legs run 17-21x above Nyquist on their own exit sphere
   (against the capstone's 2.75x, which was a different leg at `N = 2048`), and
   the chain never reaches Nyquist at any affordable `N`.  What that costs is
   bounded by the ladder: 4e-04 to 1.4e-02 rad of carrier-referenced phase rms
   per octave, i.e. inside the EE-class bar.  It is a caveat, not the defect.

---

## 5. WHAT WAS NOT RUN, AND WHAT IS STILL OPEN

1. **The `N = 8192` PITCH rung did not complete inside this session.**  Its
   chain A DID (561 s cold, and it is the one rung where the exit-wavefront
   under-sampling finally clears -- **zero** under-sampling warnings at
   `dx0 = 0.25 um`, against 5.50x / 2.75x / 1.37x at 2.0 / 1.0 / 0.5 um --
   with `P_doe/P_in` = **99.994616 %** continuing the 99.9207 / 99.9769 /
   99.9911 trend).  Chain B was still running at 26.5 GB RSS after 58 min,
   against a shared box (another campaign's `sumap_probe_121.py` held 16.4 GB
   concurrently).  The 1024/2048/4096 trend -- `dx^~1.85` on energy, 10x per
   octave on the piston-free rel L2, and a piston that does NOT shrink --
   settles every verdict above without it, and the WINDOW arm was carried to
   4096 (an 8.192 mm window) instead, where it is flat to 1.8e-08.  The rung is
   worth finishing on an idle box: `PL_N=8192 PL_DX0=0.25`.
2. **The final leg.**  This probe stops at the back aperture on purpose
   (S1.2).  The exact high-NA readout's own grid (`n_fine_cap`,
   `window_factor`, `dx_out`) was NOT laddered here, and the capstone's
   own numbers say it is separately converged (`NFC` 12288 vs 16384: EE3
   65.26 vs 65.26).
3. **Which line the 4.78 % lives on.**  `lumenairy/**` is out of scope for
   this probe, so the defect is characterised (fixed fraction, `theta^2`,
   per group, grid- and lattice-invariant, not the free legs, not the element
   fit) but not attributed to a specific term.  The reproducer is two runs.
4. **Whether the deficit is design-121-specific.**  Every measurement here is
   on one prescription.  The `theta^2` signature is generic enough that it
   should be checked on a synthetic relay before the fraction is treated as a
   constant of the library.
5. **The `(0,0)` order is the only fan order that takes the SCALAR branch.**
   Proven harmless for the piston (3.4e-07 rad against the forced-tilted
   reference), but worth knowing when reading a fan table.
6. **Group 2 is not a scaled version of group 1.**  At 1.15 mrad the residual
   after one group is -3.4e-03 rad and after two it is +1.62e-01 rad -- so the
   per-group deficit is not a single constant fraction of each group's own
   `theta^2` OPL.  The 4.78 % figure is measured at group 1; the full-chain
   behaviour is the `theta^2` law in S3.7, which is what a consumer feels.

---

## 6. RUN TABLE

| stage | runs | wall |
|---|---|---|
| PITCH ladder, `N` = 1024 / 2048 / 4096, 2 orders | 6 | 28 s / 70-78 s / 400-410 s per run |
| WINDOW ladder, `N` = 2048 / 4096 at `dx0` = 2.0 um, 2 orders | 4 | 84 s / ~400 s per run |
| piston repeats + 1e-12 perturbations, `N` = 1024 | 12 | ~20 s each |
| tilt scan, 6 orders + repeats | 8 | ~20 s each |
| per-group localiser, `PL_NGROUPS` = 1..5 | 21 | 8-25 s each |
| forced-tilted-branch reference, `PL_X0` = 1 pm | 6 | ~15 s each |
| element-fit levers (`frbf`, `poly10`, `nit40`, `rs1`) | 14 | 17-177 s each |
| continuous tilt-scale scan | 14 | ~15 s each |
| `N = 8192` PITCH rung | chain A only | 561 s cold; chain B did not complete (S5.1) |

89 artifacts in `_probe_ladder/`.  `n_workers = 1` throughout; peak RSS well
under 10 GB except at `N = 4096` and above.  `PL_NGROUPS=0` is refused by the
library (*"groups contains only DOE entries and no lens group"*), so the
free-leg-only rung was not run through the chain -- its piston is measured
directly against the analytic `k z / cos(theta)` in S3.7 instead, to 2.0e-07
rad.

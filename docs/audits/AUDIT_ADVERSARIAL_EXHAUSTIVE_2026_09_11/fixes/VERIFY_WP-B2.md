# VERIFY-WP-B2 — independent adversarial re-verification of WP-B2 (finding L9)

Subject: commit `47bc7a79` on `audit-fixes-2026-09` — the 2-D displaced remap's
symmetric input window, structured inversion, lattice 181 → 257 and
`displaced_n_side`.  I did not write it.

Environment: CPython 3.14, numpy 2.4.6, scipy 1.17.1, Windows 11,
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1` on every invocation, one process at a time.  No git
write command of any kind was run; no process was killed.  Before/after
comparisons are **archive-to-archive**: `git archive 47bc7a79^ lumenairy` and
`git archive 47bc7a79 lumenairy` extracted into the session scratchpad and
imported from a child process whose cwd and `PYTHONPATH` are that archive, with
`lumenairy.__file__` asserted in the script — never through pytest, never
against the shared working tree (which carries other work packages'
uncommitted edits).  Working tree at the time of the fixes: HEAD `908c02d6`,
with `lumenairy/elements/_lens_real.py` and `lumenairy/elements/lens_config.py`
clean against it.

**My own fixtures throughout**, chosen to share nothing with WP-B2's:
λ = 1.064 µm (not 1.31), a FREEFORM (`sag_callable`) front face plus a
spherical rear face (not a two-conic singlet), N = 640 at dx = 6.5 µm,
w0 = 2.4 mm, decenter ±0.35 mm, 8 mm aperture; a doubly-tilted plate
(tilt `(3e-3, −5e-3)`, 2.4 mm thick, n = 1.5093, 5 mm aperture); a
rotationally symmetric `R = 42.5 / −63 mm` singlet for the 1-D remap; and an
aperture-binding fixture whose grid is 1.66× the aperture.  Lattices swept:
11, 21, 33, 51, 65, 81, 97, 111, 128, 129, 145, 161, 199, 256, 341, 400, 640,
768 — the report used 65, 91, 181, 257, 361, 512, 513, 1025, 2049.

---

## 1. Verdict table

| # | Claim (WP-B2 report / changelog) | Verdict | My oracle | Measured |
|---|---|---|---|---|
| 1 | **L9-a** the reflection instability is a one-pixel input-window asymmetry selected by the launch pitch, **not** the Delaunay backend | **VERIFIED** | the mirror symmetry of the physics, on a fixture and lattices the engineer never used; the band predicate evaluated independently from the fan geometry | Parent (`47bc7a79^`): broken at **exactly** the two swept lattices whose launch axis enters `(x[-1], x[-1]+dx]` — 400 → 5.81e-02 exit / 5.85e-03 image, 768 → 3.94e-02 / 2.56e-03 — and at none of the other six (97/128/199/256/341/640: 1.22e-13 … 7.73e-13). After: 5.41e-13 / 8.47e-13 at 400 / 768, ≤ 8.5e-13 at every lattice. The predicate is perfectly discriminating on a fixture it was not fitted to. |
| 2 | the same, **below** the shipped lattice (`displaced_n_side` has a floor of 3) | **VERIFIED-WITH-NOTES** | paired-grid mirror residual, both backends, n_side 11 … 768 | structured 3.9e-16 … 8.5e-13, delaunay 9.8e-17 … 8.5e-13 — equal, at every lattice. **Note:** scored on the WHOLE grid (the convention `_mirror_x` uses) the structured backend reads up to 4.9e-03 at n_side ≤ 97. That is the *unpaired* outermost column (`x = −N/2·dx`, whose mirror `+N/2·dx` is off the grid), whose support is decided by the one-launch-cell halo of the bilinear interpolant and therefore by the lattice. It is not a symmetry violation — index 0 is its own partner — but the shipped mirror metric cannot tell the two apart. My new pins score the paired part. §3.1. |
| 3 | **L9-b** the structured inversion is exact where the exit map is affine | **VERIFIED-WITH-NOTES** | a lumenairy-free closed form written from Snell's law for a **doubly**-tilted plate (different tilt, thickness, index, wavelength, aperture and grid from the b2 fixture) | traced map vs the closed form 4.3e-19 m (map) and 8.7e-19 m (OPL). Exit field: **6.2e-15 … 2.1e-14** of peak in amplitude and **8.1e-12 … 9.4e-12** rad at n_side 91 / 257 / 513, lattice-independent. **Note:** the *scattered* backend is exact to the same digits on the same fixture, so the affine result does not discriminate between the backends — the changelog's "the remap NOW has no discretisation error" reads as if the inversion bought it, and it did not. |
| 4 | the structured backend is **5–6 % closer in amplitude** than the scattered one at every lattice | **NOT REPRODUCED — fixture-specific** | a refined-lattice reference (n_side 1025) built with **each backend in turn**, so the referee cannot favour either | On my freeform fixture the two are indistinguishable: \|E\| rms/peak at 181 = 1.2433e-04 (delaunay) vs 1.2486e-04 (structured); at 257 = 5.942e-05 vs 5.985e-05; at 513 = 1.459e-05 vs 1.484e-05 — the structured arm is 0.2–1.7 % *worse* in amplitude rms and ~0.4 % better in phase rms, with the same verdict under either reference. Both are clean second order. The 5–6 % is a property of that fixture, not of the backend. |
| 5 | no triangulation-hull holes inside the illuminated pupil (61–360 → 0) | **VERIFIED** | direct count on an aperture-binding fixture (grid half-width 6.656 mm against a 4.0 mm aperture radius) | delaunay **212 / 164 / 66** exactly-zero pixels in the lit core at n_side 181 / 257 / 513; structured **0** at all three. |
| 6 | transmitted power 0.97720 → 0.98658 | **VERIFIED, and strengthened to an absolute oracle** | the exact geometric answer: the remap is lossless inside the aperture, so the exit power **is** the input power inside (aperture ∩ carried window) | Over four grid/beam combinations the structured backend lands within **7.9e-05 … 2.1e-04** of that answer; the scattered backend is short by **2.7e-04 … 1.0e-03**, always one-sided (power lost at the hull). I specifically tried to break the structured arm the other way — it cuts the aperture on the *inverted* launch coordinate whose bilinear stencil can straddle the pupil edge, so it could leak beyond-aperture power — and it does not: the excess never exceeds the oracle's own boundary resolution. §3.3. |
| 7 | **L9-c** 257 is *derived*: the largest lattice whose trace costs less than the interpolation it feeds, on every grid measured | **VERIFIED-WITH-NOTES** | interleaved medians of 3 on this box, same fixture shape | trace 0.0790 / 0.1546 / 0.4465 / 1.1199 s at 181 / 257 / 361 / 513; structured interpolation at n_side 257 is 0.1796 / 0.7280 / 1.2682 s at N = 512 / 1024 / 2048. Criterion "trace < interp on **every** grid": 181 True, 257 True, 361 **False**, 513 **False** — the derivation selects 257 on this box too. **Note:** the binding comparison is 0.1546 against 0.1796 s, a **14 % margin on a wall clock**; a box whose ray trace is 15 % slower relative to its interpolation would derive 181. It is a defensible engineering choice, not an invariant, and nothing asserts it (correctly — S1). |
| 8 | the lattice governs input STRUCTURE; contrast transfer collapses in launch-samples-per-period | **VERIFIED** | ripple contrast through the remap on a 1024² grid | at a 24-field-pixel period: 0.62 (181) / 0.77 (257) / 0.96 (835) / 0.98 (2049), monotone in samples/period, consistent with the report's collapse curve. |
| 9 | **L9-d** the restated warning "names a `displaced_n_side` that actually clears the bar" | **NOT FIXED** → **fixed here** | the ray-map builder's own `dstep` | The warning scored the pitch as `2r/(n−1)` while the fan is thrown 3 % wider, so the trace uses `2·1.03·r/(n−1)` — ratio measured **1.03000** at n_side 181 / 257 / 513. The lattice it named therefore did **not** clear the bar it claimed: at a 10 mm aperture it named **626** for dx = 8 µm (real pitch **16.48 µm** against the 16.00 µm bar) and **1251** for dx = 4 µm (**8.24** against 8.00). It silenced itself only because it re-used the same wrong formula. This is the "names a keyword value that cannot fix it" class WP-B2's own §2.6 says it removed a prototype warning to avoid. Now 645 and 1289, real pitch 15.99 and 8.00 µm. §4.2. |
| 10 | byte-identity: the 1-D remap, the non-displaced surface models and the traced family did not move | **VERIFIED** | SHA-256 of the raw output bytes, archive `47bc7a79^` against archive `47bc7a79`, eleven paths | identical on `thin` (symmetric and asymmetric), `tangent_facet`, `tangent_facet_remap`, displaced-symmetric default, the **1-D symmetric remap**, `displaced_mode='split'`, `displaced_obliquity='pointwise'` (freeform and tilted), `apply_real_lens_traced` (symmetric and asymmetric). Exactly one fingerprint moved: the 2-D asymmetric remap, `6e8b283c11cfb2a9` → `9f2d80c93918072a`. |
| 11 | `displaced_n_side` validation (bool, string, tuple, float with fraction, below the floor, a call that would discard it) | **VERIFIED** | 25 hostile values and 8 routing cases of my own | every refusal begins `apply_real_lens:` and names the keyword — including `bytes`, `list`, `dict`, `Decimal`, `Fraction`, a 0-d and a 1-d `ndarray`, `complex`, `np.float32(2.5)`, `±inf`, `nan`, `-0.0`, which the shipped tests do not cover. Every discarding route is refused by name, including four the shipped test does not exercise (`displaced_obliquity='meridional'`, `displaced_mode='split'`, `tangent_facet`, and `displaced+symmetric+remap`). |
| 12 | `LensNumerics` round trip (`from_kwargs` / `to_kwargs` / `narrowed_to`), the covering array | **VERIFIED-WITH-NOTES** | direct, with WP-B10's `fit_basis` present (19 numerics fields) | `from_kwargs(displaced_n_side=513, fit_basis='zernike').to_kwargs()` round trips to an equal config; `narrowed_to('apply_real_lens')` → `{'displaced_n_side': 513}`, `narrowed_to('apply_real_lens_traced')` → `{'fit_basis': 'zernike'}`, `narrowed_to('apply_real_lens_maslov')` → `{}`. Covering array 45/45. **Note:** `LensNumerics(displaced_n_side=181.0)` raises while `apply_real_lens(displaced_n_side=181.0)` accepts — the config layer is strictly stricter than the call layer on the same keyword (§5, Follow-up 3). |
| 13 | **deferred §6.1** — the 1-D symmetric remap carries the same input-window asymmetry, unexposed because "no fixture in the suite exercises a mirror pair through it" | **CONFIRMED, LARGER THAN STATED, AND FIXED HERE** | the mirror symmetry of a rotationally symmetric element on a **centred** input | It needs no mirror pair and no decentred input: a centred, rotationally symmetric Gaussian through a rotationally symmetric singlet came back with a crescent of exactly-zero pixels on +x only — paired mirror relL2 **3.301e-02** (N = 640) and **4.586e-02** (N = 512), **1236** and **988** pixels off by more than 1e-9 of peak, worst pixel **0.489** and **0.650** of peak against an exact **0** on its mirror. Identical readings on the parent archive, so pre-existing, not a WP-B2 regression. After the fix: **1.18e-16** and **1.01e-16**, zero pixels off. §4.1. |
| 14 | the 50 new pins: derived envelopes, real fail-before | **VERIFIED-WITH-NOTES** | a seven-mutant campaign against the library source | four mutants go red as they should; **three deliverables were unpinned** — the lattice raise 181 → 257 (all 80 tests stayed green with 181 restored), the affine Newton seed, and the Newton sweep cap. §6. |

---

## 2. What I could not break

* **The band diagnosis.** I computed the predicate "does a launch coordinate
  fall in `(x[-1], x[-1]+dx]`?" from the fan geometry alone, on a fixture with
  a different wavelength, grid, beam, decenter and surface type, and swept
  eight lattices none of which the report used.  It picked 400 and 768 and
  nothing else, and the parent library broke at 400 and 768 and nowhere else,
  by 5.8e-02 and 3.9e-02 against a 1e-13 floor.  That is as clean a
  confirmation of a mechanism as this kind of measurement gets.
* **The aperture cut on the inverted launch coordinate.** The structured
  backend reads `amp_src` at the inverted `(u, v)` with a bilinear stencil that
  can include launch points *outside* `r_ap` (they are deliberately kept as
  interpolation neighbours), so a pre-apertured pupil edge could leak.  On an
  aperture-binding fixture it does not: the transmitted power never exceeds the
  exact geometric answer by more than the oracle's own boundary resolution, at
  four grid/beam combinations and three lattices.
* **The search box's exactness claim.**  `_remap2d_interp_structured`'s
  docstring says the bounding box is exact because "the scattered path's hull
  is empty out there too".  I instrumented the inversion and compared a
  mirrored pair point by point: the Newton itself is mirror-symmetric to
  3.4e-16 m in `u` and `v`, every point converges in the same number of sweeps
  bar the unpaired column, and the box differs between the two runs by exactly
  the unpaired column — which is legitimate (see §3.1), not a box defect.
* **Validation and the config round trip.**  25 hostile inputs, 8 routing
  cases, and the round trip with WP-B10's new field in place: no gap found
  beyond the strictness divergence in §5.

---

## 3. Where the report is right but says more than it measured

### 3.1 The mirror metric and the unpaired column

`x = (arange(N) − N/2)·dx` puts index 0 at `−N/2·dx`; its mirror `+N/2·dx` is
not on the grid, so under the `np.roll(A[:, ::-1], 1)` convention index 0 is
its own partner and **carries no symmetry constraint**.  On the p10 fixture the
outermost column happens to be zero in both arms, so this never shows.  On mine
it does: at n_side 11 / 21 / 33 / 51 / 65 / 81 / 97 the structured backend
carries 0.0 … 0.36 of peak in that column for one decenter sign and a different
value (sometimes exactly 0) for the other, and the whole-grid metric reads
9.6e-04 … 4.9e-03 as a result.  On the paired grid the same fields are
mirror-exact to 3.9e-16 … 8.5e-13.

The mechanism is the launch pitch, not a defect: a bilinear interpolant's
support extends one launch cell beyond its data, so at a coarse lattice a field
point just outside the carried window is still reachable and at a fine lattice
it is not.  It is the same statement as "the transverse resolution is the
launch pitch".  The Delaunay backend never carries it (its hull stops at the
outermost retained exit point), so the two backends disagree about the output's
outermost rim — worth knowing, since the report leans on their agreement.

I spent a substantial part of this pass treating the whole-grid reading as a
regression before localising it.  Recording that: **the first number I measured
was real and the conclusion I drew from it was wrong**, and the thing that
separated them was localising the residual rather than re-running it.

### 3.2 The affine plate does not discriminate between the backends

Both backends reproduce my hand-written Snell closed form to 1e-14 of peak and
9e-12 rad at every lattice.  The property being demonstrated is "the exit map
is degree 1, so *any* second-order interpolant is exact", which is true of
barycentric interpolation over the exit triangulation as well.  The shipped
test only runs the default backend, so it claims nothing false; the changelog's
framing does.

### 3.3 The power claim is better than the report makes it

The report scores transmitted power backend-against-backend (0.97720 vs
0.98658).  There is an absolute oracle available — the transfer is lossless
inside the aperture — and on it the structured backend is correct to 1e-04
while the scattered one is systematically short.  That is the stronger
statement, and it is now what the test file asserts, scored against the
oracle's own boundary resolution rather than a chosen tolerance.

---

## 4. Defects found and fixed

### 4.1 The 1-D symmetric remap's input window (deferred §6.1) — FIXED

`_apply_displaced_remap` reads the input at the ENTRANCE height `X · scale`,
`scale = h_in / r_out`.  For a converging element the ray walks inward, so
`scale > 1` and the read runs off the +x end of the axis while its mirror — one
whole sample further out on −x — is still on the grid and returns the full
envelope.  `map_coordinates(mode='constant')` returns exactly `cval` outside
`[0, N−1]` rather than interpolating, so the +x rim is *dead*, not attenuated.

**Fail-before (parent archive `47bc7a79^`, and identically on `908c02d6`):**

| fixture | paired mirror relL2 of \|E\| | pixels off by > 1e-9 of peak | worst pixel |
|---|---|---|---|
| N = 640, dx = 6.5 µm, w0 = 2.4 mm | 3.301e-02 | 1236 | 0.489 of peak vs an exact 0 |
| N = 512, dx = 8 µm, w0 = 3.0 mm | 4.586e-02 | 988 | 0.650 of peak vs an exact 0 |
| the same, input decentred ±0.45 mm | 3.560e-02 / 4.726e-02 | 1236 / 988 | 0.631 / 0.754 of peak |

**After:** 1.183e-16 / 1.011e-16 (centred) and 1.162e-16 / 1.007e-16
(decentred), zero pixels off.

`lumenairy/elements/_lens_real.py:1948` — the same rule the 2-D remap uses,
`|X·scale| ≤ (Nx/2−1)dx` and `|Y·scale| ≤ (Ny/2−1)dy`.  The price is the
outermost ring of the input on **both** sides instead of one.

**The deferral's stated blocker does not hold.**  WP-B2 §6.1 says the fix
"moves the byte-identity pin"
`test_niche_p10_…::test_symmetric_remap_is_the_p2_1d_remap_byte_identical`.
That pin compares `apply_real_lens(displaced_mode='remap')` against a *direct
call to `_apply_displaced_remap`*, so both sides move together — it is a
routing pin, not a value pin, and it passes unchanged (verified: the whole p10
file, 13 tests, green).

Blast radius, measured archive-style over eleven paths: **exactly one
fingerprint moves**, the 1-D symmetric remap.  The 2-D remap's output is
bit-identical to `47bc7a79`'s.

### 4.2 The warning's pitch and the lattice it names — FIXED

Measured ratio of quoted pitch to traced pitch: **1.03000** at n_side 181, 257
and 513 (55.556 / 39.062 / 19.531 µm quoted against 57.222 / 40.234 / 20.117 µm
traced).  The named lattice therefore missed its own bar:

| field pitch | named before | its real pitch | bar (2·dx) | cleared? | named after | its real pitch |
|---|---|---|---|---|---|---|
| 8 µm | 626 | 16.48 µm | 16.00 µm | **no** | 645 | 15.99 µm |
| 4 µm | 1251 | 8.24 µm | 8.00 µm | **no** | 1289 | 8.00 µm |

Fix: `_DISP_REMAP_2D_FAN_FACTOR = 1.03` is now one module constant, read both
by `_build_displaced_ray_map_2d` (which throws the fan) and by
`_warn_if_remap_lattice_smooths` (which scores it), so the pitch the caller is
told about is the pitch the trace uses and `n_clear = ceil(span/(2h)) + 1`
clears the real bar by construction.  The message also now names the fan and
the aperture separately instead of quoting the pitch "across the aperture".

The same 3 % error was in the prose: the `displaced_n_side` docstring, the
`_DISP_REMAP_2D_N_SIDE` comment's pitch column (55.6 / 39.1 / 19.5 / 9.8 µm),
`_normalise_displaced_n_side`'s refusal message and `LensNumerics`'s field
docstring all stated `2·r_aperture/(n−1)`.  All corrected to the traced pitch
(57.2 / 40.2 / 20.1 / 10.1 µm).  Note the WP-B2 **report itself uses both
conventions**: §2.2's `dstep` column is the true pitch, §2.4's "55.6 µm to
39.1 µm" is the bare formula.

### 4.3 A stale docstring — FIXED

`_build_displaced_ray_map_2d`'s docstring still read "see that constant for why
it is a fixed 181", describing a default that is now 257 and no longer fixed.
`_lens_real.py:2204`.

---

## 5. Requested changes outside my ownership

None are required for the tree to be green.  One recorded for the record:

**1. `docs/lens_configuration.md`** — its `displaced_n_side` row description (if
it repeats the pitch formula) should read
`2 · 1.03 · r_aperture / (displaced_n_side − 1)`.  I did not open it for edit;
the row WP-B2 requested carries no formula, so this is conditional on whatever
text the orchestrator landed.

**2. `docs/subsystems/real_lens.md`** — WP-B2's requested §3.1 addition quotes
the launch pitch as `2·r_aperture/(displaced_n_side − 1)`; if it landed in that
form it should gain the `1.03` factor, and "Default 257 rays a side" is
correct.

**3. `lumenairy/elements/lens_config.py::_require_positive_int`** (shared by
every integer field, so not mine to change): it refuses any `float`, while
`_lens_real._normalise_displaced_n_side` accepts a float with no fractional
part.  `LensNumerics(displaced_n_side=181.0)` therefore raises where
`apply_real_lens(displaced_n_side=181.0)` succeeds.  Stricter-at-the-config is
the safe direction, so this is a consistency report rather than a bug; the
resolution is a decision about which contract the family wants, not a patch I
should make inside an L9 verification.

---

## 6. The new pins, attacked by mutation

Each mutant was applied to `lumenairy/elements/_lens_real.py` in place, the
targeted subset run, and the file restored from a byte copy (verified identical
at the end of the campaign).

| mutant | what it restores / breaks | b2 tests red |
|---|---|---|
| `amp_in = np.where(_win, …)` removed | the 2-D asymmetric input window | **8 of 21** |
| `amp = np.where(_win, …)` removed | the 1-D asymmetric input window (my fix) | **5 of 7** |
| `_DISP_REMAP_2D_INV_MAX_ITERS = 1` | the Newton stops after one sweep | **0 of 9** |
| affine seed → identity seed | the affine seed dropped | **0 of 7** |
| `_DISP_REMAP_2D_N_SIDE = 181` | the lattice raise undone | **0 of 80** → **2 of 3** after §6.1 below |
| `span = 2.0 * r` in the warning | the fan factor dropped | **6 of 8** |
| `_DISP_REMAP_2D_INV_MISS_FRAC = 1e-9` | the coverage bar tightened | **5 of 9** |

Three findings from that:

**6.1 The lattice raise was not pinned at all.**  Putting `181` back left all 80
tests green — the headline deliverable of L9-c was indistinguishable from its
own pre-state.  Added `TestTheDefaultLatticeIsTheOneTheContractNames`: the
public docstring's literal default must equal the constant (build-free — it
compares two things inside one build), and the shipped default must be strictly
more accurate than 181 against a refined-lattice reference.  Both go red on the
mutant.  I deliberately did **not** pin the literal 257, which would refuse a
future re-derivation.

**6.2 The Newton loop is not load-bearing on any fixture measured.**  Capping it
at a single sweep changes nothing: the affine seed lands within
2.4e-06 m, one Newton squares that to ~1e-11 m, and the coverage bar is
1e-3 · cell.  The report's "95 % of a 512² grid was still moving at sweep 32"
is about the *bitwise* fixed point, which is a different question and is
correctly answered; but the cap of 32 and the claim "two full sweeps plus a 3 %
tail" are a safety margin, not an enforced property.  Recorded rather than
pinned — a two-sided operation-count bar would be exactly the per-build shape
TESTING_STANDARDS S1 forbids.

**6.3 The affine seed's *operation* benefit does not reproduce.**  The report
says it "removes a whole Newton sweep".  Swapping the seed for the identity in
process, the whole `_apply_displaced_remap_2d` call costs **20
`map_coordinates` reads either way** on the p10 singlet at n_side 181 — two
sweeps plus the final check in both cases.  What does reproduce is the seed
*residual* advantage (23× on this build, against a documented 23×), and that is
what I pinned.  The claim as a cost claim is unsupported.

Additionally: `_DISP_REMAP_2D_INV_MISS_FRAC` was previously unpinned and is now
held by the power/hole oracle (mutant 7).

---

## 7. Tests added

30 tests appended to `tests/unit/test_audit2609_b2_displaced_remap_inversion.py`
(50 → 80).  Nothing was weakened or removed.

* `TestTheOneDimensionalRemapCarriesTheSameWindow` (7) — the symmetric-element
  claim, the decentred-input mirror pair, an in-process fail-before on a frozen
  copy of the assembly that is checked bit-for-bit against the shipped function
  on every run, and a pin that the window's cost stays confined to the
  outermost ring.
* `TestTheWarningQuotesThePitchTheTraceUses` (8) — the quoted pitch parsed out
  of the message and compared with the builder's own `dstep`; the named lattice
  measured **on the builder**, not on the formula; a fail-before showing the
  bare formula's value misses the bar; and a pin that the builder and the
  warning read one constant.
* `TestTheMirrorSymmetryHoldsBelowTheDefaultLattice` (6) — n_side 11 / 33 / 97,
  both backends, paired grid, envelope derived from this build's own reading at
  the shipped default.
* `TestTheTransmittedPowerAgainstTheGeometricOracle` (3) — the absolute power
  oracle and the hull holes, on a fixture whose aperture binds, scored against
  the oracle's own boundary resolution.
* `TestTheDefaultLatticeIsTheOneTheContractNames` (3),
  `TestTheAffineSeedIsWhyTheLoopIsShort` (2), `test_interp_method_is_validated`
  (1).

No wall-clock assertion anywhere (S1).  Every bar is derived at runtime or
carries its derivation and its measured values beside it.

---

## 8. Commands, counts, durations

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; pytest
runs with `-q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b2_displaced_remap_inversion.py` | **80 passed** | 112.7 s |
| `pytest tests/unit/test_niche_p10_transverse_walk_remap.py …b2…` | **63 passed** | 183.6 s |
| `pytest …a2_displaced_models …a2_analytic_lens …a2_verify_lens_analytic …p2_displaced_extreme …p3_pointwise_obliquity …a15a_lens_covering_array` | **176 passed** | 274.7 s |
| `pytest …a16_lens_config_round_trip …a16_lens_arch …a16_lens_config_bit_identity …a16_verify_config_and_arch` | **166 passed** | 22.8 s |
| `pytest tests/unit -k real_lens` | **160 passed, 3 skipped** (PySide6 / host-specific digests / numexpr) | 284.5 s |
| `pytest …a17_history_lint …a17_history_relocation` | **740 passed, 4 failed** — `carrier` and `lumenairy.propagators.system`, both other work packages' uncommitted edits; `lumenairy.elements._lens_real` green | 47.5 s |
| final combined: p10 + b2 + three a2 + a15a + a16 round trip | **331 passed** | 275.0 s |
| `python validation/run_all.py test_lenses` | **PASS** (`ALL 1 files passed`) | 25.2 s |
| `ruff check` on my three source/test files | **All checks passed** | — |
| `python scripts/record_history_fingerprints.py --check` | `lumenairy.elements._lens_real.md` **OK**; the remaining DRIFTs are `carrier.py` and `lumenairy/propagators/system.py` | — |
| mutation campaign, 7 mutants | table §6; source restored byte-identical | ~5 min |

Measurement scripts (session scratchpad, not committed): the mirror sweep over
eight lattices on the freeform fixture against the parent archive; the
coarse-lattice paired-grid sweep over both backends; the instrumented inversion
that localised the unpaired column; the 1-D remap probe; the archive-to-archive
byte fingerprints over eleven paths; the lumenairy-free Snell closed form; the
power/hole oracle at four grid/beam combinations; the interleaved trace /
interpolation timings; the contrast-transfer sweep; the validation and
round-trip probes; the cross-backend convergence referee.

---

## 9. Follow-up (ruled open, not fixed here)

1. **The warning's bar is a sampling statement, not a fidelity one.**  "Launch
   pitch ≤ 2 × field pitch" puts ~3 launch samples on a 6-field-pixel period,
   which the remap transfers at 0.63 of input contrast (measured; 2049 gives
   0.85).  The message says the named lattice "resolves the field pitch", which
   overstates what it delivers.  Restating the bar in samples-per-period needs
   its own fixture ladder — the same measurement WP-B2 §6.2 already defers for
   the Nyquist diagnostic, and the two should be done together.

2. **The two backends disagree about the output's outermost rim** (§3.1).  The
   structured backend carries up to 0.36 of peak in the unpaired outermost
   column at a coarse lattice; the scattered one never does.  Neither is
   "right" — the input sample there was discarded by the symmetric window — but
   the model currently has a rim whose support depends on the launch pitch.
   Options: extend the window cut to the output (zero any field point whose
   inverted launch coordinate is outside the carried window), or document the
   rim.  Wants a measurement of which is closer to a Kirchhoff sum.  ~0.5 d.

3. **`LensNumerics` is stricter than the call it configures** (§5 item 3).  A
   decision, not a patch.

4. **The Newton cap and the affine seed are cost machinery with no measured
   cost benefit at the shipped bar** (§6.2, §6.3).  Either find the regime
   where they bind (a strongly folded exit map, a caller-supplied congruence
   with large curvature) and pin it there, or simplify the loop.  ~0.5 d.

5. **WP-B2's own deferred items 2–5** (the Nyquist diagnostic, the trace-break
   tolerance, the inverse-on-a-coarse-grid optimisation, the `interp_method`
   spelling) remain open and I found nothing that changes their priority.
   `interp_method` stays private; I added a validation pin for it.

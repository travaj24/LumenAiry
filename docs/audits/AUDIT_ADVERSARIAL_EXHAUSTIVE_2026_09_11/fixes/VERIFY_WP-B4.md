# VERIFY-WP-B4 -- independent adversarial re-verification of WP-B4 (`transport='collins'`)

Branch `audit-fixes-2026-09`.  Subject: commit **185d64cd**
(`feat(carrier): WP-B4 -- transport='collins' ...`); parent `185d64cd^` is the
pre-change library and is identical to 81d5b586 for
`lumenairy/propagators/carrier.py`.  I did not write WP-B4.

Everything below was measured on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time, on 2026-09-13.

**Method.**  Nothing here was run against the shared working tree, which
carries four other engineers' in-flight files (`system.py`, `_lens_real.py`,
`_lens_traced.py`, `analysis/*`, `sources/*`, `raytrace/*`, `lenses_maslov.py`).
Three trees were extracted READ-ONLY into
`.../scratchpad/verify_b4/` with `git archive` and every measurement ran in a
CHILD PROCESS whose `cwd` and `PYTHONPATH` were that tree, with
`carrier.__file__` asserted at the top of each script:

| tree | contents |
|---|---|
| `base/` | `git archive 185d64cd^ lumenairy` -- the pre-change library |
| `b4orig/` | `git archive 185d64cd lumenairy` -- WP-B4 exactly as shipped (the fail-before arm) |
| `mine/` | `b4orig` with `carrier.py` replaced by this pass's; `diff -rq` reports that ONE file and no other |

Oracles are written from scratch in each probe.  The absolute-phase Gaussian
oracle is written in THIS library's `exp(-i omega t)` / `exp(+i k z)`
convention (`1/q = 1/R + i lambda/(pi w^2)`, prefactor `q/q2`); WP-B4's note in
its sec.8 about `test_audit2609_a6_verify_carrier.py::_abcd_field` carrying
Siegman's opposite pairing is correct and that function was NOT used here.

**One caution for whoever writes the next such oracle.**  My first version took
the 2-D prefactor as `1/sqrt((1 + z/q)^2)`.  That is the principal branch of a
square, and past the waist `1 + z/q` leaves the right half-plane, so the oracle
picked up exactly `pi` -- reading 2.0 against a correct transport on every leg
with `A < 0` and 0 on every leg with `A > 0`.  The continuous form is
`sqrt(q/q2)` per axis (`arg(q/q2)` is strictly inside `(-pi, pi)` because
`Im(q) = Im(q2) < 0`), i.e. `q/q2` in 2-D.  WP-B4's own oracle uses
`1/(1 + z/q)` and is right; mine was wrong for twenty minutes and blamed the
library.

---

## 1. Verdict table

| # | WP-B4 claim | verdict | my oracle, my numbers |
|---|---|---|---|
| 1 | **Sign convention**: the implemented Collins form is the complex conjugate of the printed one; `B = z` is POSITIVE on every forward leg, converging or not; `A = 1 + z/R_in` crosses zero through the geometric focus as an ordinary value | **VERIFIED** | Absolute-phase Gaussian oracle (piston + Gouy), N = 512 at 6 w extent, `R = -40 mm`, `carrier_out=inf`.  On the shipped `gap_kernel='auto'`: relL2 **1.95e-08** (diverging), **5.87e-08** (converging, `A = +0.5`), **1.41e-15** at the focus (`A = 0`), **2.39e-06 / 1.76e-07** past it (`A = -0.025 / -0.5`), **3.03e-15** astigmatic; `arg` of the best global phase `<= 1e-6` rad on every cell, so nothing hides in a piston.  Under `gap_kernel='fresnel'` -- the ABCD-Fresnel integral the oracle IS -- the same cells read **4.87e-12 / 2.42e-14 / 1.41e-15 / 1.41e-15 / 4.58e-15**: the residual above was the exact-kernel refinement (row 10), and the QUADRATURE itself is exact to the transform's rounding on every leg I tried.  Setting `R_in = inf, R_ref = inf` reduces the implemented exponent to `exp(ik(x-u)^2/2z)` with prefactor `exp(ikz)/(i lambda z)` -- the textbook Fresnel integral in `exp(+ikz)`, which is the convention statement, checked as arithmetic rather than as a citation.  `det - 1 = 0.0e+00` exactly on the four sign cases I tried |
| 2 | **`m -> 0` stops being a singularity** | **VERIFIED** | Sweep `z = 39.500 .. 40.500 mm` across `R = -40 mm` on the shipped `gap_kernel='auto'`: absolute relL2 `4.6e-06, 2.3e-05, 2.3e-04, 1.4e-15, 2.3e-04, 2.3e-05, 4.7e-06` -- symmetric about `A = 0`, matching the oracle's `arg` at the centre sample to every printed digit on all seven, and with no discontinuity anywhere.  The residual either side is NOT the transport: it is the exact-kernel refinement's own `k |z_eff| theta^4/8` (row 10 / F3) -- under `gap_kernel='fresnel'` the same sweep reads **8.9e-15 to 1.7e-14** at every cell, 1 um from the focus included.  The shipped `'sziklas'` transport cannot reach that cell at all (`ValueError: carrier_referenced_envelope: R_carrier == 0`) |
| 3 | **Gate (a)**: 0 of 30 cells worse than `'sziklas'`; the Collins reading is identical across NA at a given extent; rounding floor at ext >= 6 | **VERIFIED** on a fixture the engineer did not use | `lambda = 0.633 um`, **N = 768** (not a power of two), `w_in = 0.45 mm`, NA 0.02/0.045/0.08/0.24/0.60 x ext 2.0/3.2/5.0/8.0/12.0, `N_out = 96`, target at the geometric focus: **0 of 25 cells worse**; the Collins column is identical across all five NAs at each extent (9.9584e-03 at ext 2.0, 1.4526e-05 at 3.2, **4.1209e-12** at 5.0) exactly as the report's structural argument predicts; 2.8e-15 to 9.1e-14 piston-free at ext >= 8 (to 1.8e-13 absolute).  Absolute and piston-free readings agree to the last digit, so the Gouy phase is right on all 25.  Power ratio 1.0000000 on every cell.  `'sziklas'` on the same cells runs 1.6e-02 to 7.0e-01 |
| 4 | **Gate (b)**, C1 mismatch: peak ratio 1.0000 at every `R/R0`, period carrier-free | **VERIFIED** (re-derived, not re-run) | The period is `lambda|z|/dx` of the input grid by construction -- it contains no carrier term -- and I confirmed it numerically at `_period_out['period']` for two beam widths on one grid: the Collins period does not move, the Sziklas one does.  The mismatch matrix itself I did not re-run; the mechanism is arithmetic |
| 5 | **Gate (c)**, two-group chain, `np.array_equal` to `'sziklas'` because that chain's legs are transfer-function sampled | **VERIFIED-WITH-NOTES** | True on that fixture, and the note is defect **D1** below: on the *gate (d)* fixture (`_chain_fixture`, N = 256 at 60 um) the same statement fails badly on 185d64cd -- bare final leg, `'sziklas'` peak 0.960941 / power 2.82889e-05 against `'collins'` peak **29.0002** / power **0.0189589** (a 670x power inflation).  After the fix the two are identical to every printed digit on that fixture too |
| 6 | **Gate (d)**, `_multi` K = 1 / K = 2 exactly 0.0 | **VERIFIED-WITH-NOTES** | Reproduced, but the gate is weaker than it reads: it compares `_multi` against the single chain *within one transport*, so it cannot see that the transport itself was 670x off on that very fixture (row 5) |
| 7 | **K1 derivation** and the fail-before ladder | **VERIFIED** | Re-derived independently and reproduced: `A = 0.9`, `B = 3 mm`, `w = 0.9 mm`, n = 256/512/1024/2048 gives K1 = **49.694 / 24.847 / 12.424 / 6.212** and chirp-Z relL2 **1.1260e+02 / 5.5158e+01 / 2.7432e+01 / 1.3299e+01** (ratios 2.27 / 2.22 / 2.21 / 2.14 -- the error tracks K1), while the transfer-function arm sits on my oracle's floor at 4.05e-12 on all four grids.  `on_collins_sampling='error'` raises on every row, naming K1 |
| 8 | **K2 derivation** | **VERIFIED-WITH-NOTES** | The stationary-phase cancellation argument is right (the chirp-Z's `-x/(A lambda B)` and the post-chirp's `+D x/(lambda B)` sum to `C x/(A lambda)`).  But K2 is a REPRESENTABILITY statement, not an accuracy one, and it has no fail-before: sweeping `dx_out` from `w0/8` to `4 w0` at the focus takes K2 from 0.203 to **6.488** while the pointwise relL2 against the analytic Gaussian stays at 4.5e-12 -- the transform evaluates the integral exactly AT the requested points whether or not they resolve the field.  K2 bites only for a consumer that resamples or transforms the result.  Worth saying in the message; it currently reads like an accuracy claim |
| 9 | **K3, "disposed of by the EXISTING `on_replica` ... so the two guards cannot disagree"** | **NOT FIXED -> FIXED HERE** (defect **D1**) | True at the readout only.  A chain leg and the public single-step entry have no `on_replica`, and `_check_collins_sampling` disposed of K1 and K2 only, so K3 was computed and dropped.  See sec. 2 |
| 10 | **K4** -- the exact-kernel wrap condition | **VERIFIED-WITH-NOTES** | `|dphi/dq|` re-derived by hand and by central differences: `-z_eff theta (1/sqrt(1-theta^2) - 1)` agrees with the numerical derivative to 1.2e-06 at `theta = 0.05` and 9.6e-09 at 0.5.  `_collins_kernel_wrap_ratio` equals my own formula to the last digit on seven cells; the report's published table reproduces exactly (`A = 0.5 -> 2.2784e-07`, `A = 0.0025 -> 9.0907e-05`).  NOTE: K4 bounds the WRAP, not the accuracy.  On the WP's own fixture, 1 um from the geometric focus (`z_eff = -1600 m`, **K4 = 9.1e-03**, guard silent), `gap_kernel='fresnel'` reads **1.71e-14** against the analytic Gaussian and `'auto'` reads **2.35e-03**, falling exactly as `1/|z_eff|` (2.35e-04 / 2.36e-05 / 2.41e-06 at 10 um / 100 um / 1 mm) -- and the whole column is INDEPENDENT of N (identical at 512, 1024, 2048, 4096 on the same extent), which is what says it is the refinement and not the grid.  See Follow-up F3 |
| 11 | **`gap_kernel='exact'` is REFUSED near focus rather than silently downgraded** | **VERIFIED** | The refusal fires and names `z_eff`, `theta` and K4.  On a `w = 0.3 mm` Gaussian it needs `|z_eff| > 1.8e+05 m`, i.e. `A < 2e-07`; `'auto'` resolves to `'exact'` through `A = 2.5e-06` and drops to `'fresnel'` at `A = 1e-08` and `A = 0`, matching the report's table row for row |
| 12 | **`'auto'` "takes the ABCD-Fresnel integral and RECORDS it"** | **NOT FIXED -> FIXED HERE** (minor) | It was recorded only in `stats_out`, which no caller of the chain sees: the leg's published diagnostics carried `collins_form / k1 / k2 / flat_reference / dx_floor_hit` and no kernel key.  `collins_kernel` is now published beside them |
| 13 | **Byte-identity of the default, archive to archive** | **VERIFIED** on my own fixture set | My 60-entry set (not the engineer's 41): every single-step branch including a focus CROSSING, two near-focus bridge landings, a **complex64** arm, a tilted exact leg, `gap_kernel='fresnel'`, both public readouts, `replica_fill='zero'`, reconstruct/envelope/fit_radius/aperture, the chain with `repr(stages)`, and `_multi` at K = 1 and K = 2.  `base` vs `b4orig`: **60 of 60 EQUAL** (shape + dtype + `np.array_equal`).  `base` vs `mine` after my fix: **60 of 60 EQUAL** |
| 14 | **Cost**: a chirp-Z gap leg is 4-5x a transfer-function step; `next_fast_len(N + N_out - 1)`, 3 FFTs per axis-pass | **VERIFIED** by FFT-call counting | Counting `_bluestein._fft_1d` and `fft_infra._fft2/_ifft2` in memory: a Collins gap leg runs **exactly 6** batched 1-D transforms (3 per axis) of length **`next_fast_len(N + N_out - 1)`** -- 1024, 2048, 4096 at N = 512, 1024, 2048 and 1089 / 1280 for `N_out = 64 / 256` -- plus one `fft2(N)` for the angular-support measurement and, under `'auto'`, one `ifft2(N)` for the kernel refinement.  A Sziklas step under `'auto'` runs `fft2 + ifft2` of N and nothing else.  Wall-clock medians of interleaved runs are in sec. 5 |
| 15 | **Cost**: the image-plane readout is 0.30x | **VERIFIED-WITH-NOTES** | The timing reproduces, but not the explanation.  By FFT work the Collins readout is *more* expensive (1 x `fft2(1024^2)` + 6 batched 1-D of 1089, against the Sziklas readout's 2 x `fft2(1024^2)`); the speed-up is in what it does NOT do -- the standoff resolution, the containment measurements and the C1 curvature fit -- which FFT counting cannot see and the report's sentence attributes to the transform |
| 16 | **`final_distance == 0` + `focus_readout` raises as documented** | **VERIFIED** | `propagate_traced_carrier_chain(..., final_distance=0.0, focus_readout=..., transport='collins')` raises `ValueError` whose message names both ways out; `'sziklas'` returns (dx 5e-07, peak 1.02263).  `_collins_transport` refuses `B = 0` separately, and `_collins_carrier_leg` short-circuits `z == 0` to a bit-identical identity |
| 17 | **§9 residual risk: the flat reference composed with a traced GROUP at the focus of its own input carrier is untested** | **VERIFIED** -- and it works | Built it: `R_in = -20 mm` with `gap_before = 20 mm`, so the leg lands exactly on `R + z = 0` (`flat_reference=True`, `R_out = inf` -- and one gap either side gives `R_out = -/+1e-4 m`, so the switch is exactly on the focus).  Against my own band-limited ASM of the FULL field plus the same `apply_real_lens_traced` call: power ratio **0.999458**, r2m 324.80 um vs 328.19 um (1.03 %), centroid on axis to 0.02 nm.  The `'sziklas'` arm cannot run this composition at all -- it raises `R_carrier == 0` |
| 18 | **§5 item 7: no fixture reads a strongly tilted congruence THROUGH the Collins readout against an independent oracle** | **VERIFIED** -- and it is exact | Wrote that oracle (Fresnel's shift theorem, absolute).  `L = 46 mrad`: relL2 **3.10e-14**.  `(L, M) = (20, -12) mrad`: **1.83e-14**.  With a `(500, -300) um` decentre on top: **4.55e-13**.  `(46, 30) mrad` with `(800, 800) um`: 8.24e-09.  The `centre_out` post-chirp offset and the Bluestein output-centre index are therefore both right, including their signs |
| 19 | **"the selection cannot introduce a step and needs no smoothing at its boundary"** | **NOT FIXED -> FIXED HERE** (defect **D1**) | Measured AT the boundary, which the package never did (its two cells are K1 = 0.34 and 0.16).  On 185d64cd the boundary sits at K1 = 1 (z = 8.0108 mm on the WP's own fixture) and the two forms differ there by **0.9999 of peak**.  See sec. 2 |
| 20 | **"every leg satisfies at least one of the two [conditions]"** | **NOT FIXED -> FIXED HERE** | K1 is not the complement of the transfer-function condition.  The complement is K3: `K3 * K_tf = 2 dx theta / lambda <= 1` identically, because `theta` is read from the envelope's own SAMPLED spectrum.  `K1 * K_tf` carries an extra `4 |z_eff| theta^2/(N lambda)` that nothing bounds.  Now pinned as an identity |

---

## 2. Defects found, and the fixes

### D1 -- the chirp-Z's output PERIOD was never disposed of outside the readout, and the quadrature selected on the wrong half of the complementary pair

**What it was.**  `_collins_sampling_stats` computes K3 -- the condition that the
returned window fit inside one chirp-Z period `lambda|B|/dx` -- and
`_check_collins_sampling` deliberately ignored it ("K3 is the replica guard's").
That is true at `_collins_focus_readout`, which calls `_check_readout_replica`.
It is NOT true at `_collins_carrier_leg`, which is every inter-group gap, the
bare final leg, and the public `propagate_carrier_referenced(transport=
'collins')` entry.  Those callers have no `on_replica` argument, so K3 was
computed and dropped on the floor.

Two things follow, and both are measurable:

**(a) the selection.**  On a leg's lattice `d_out = max(|A| dx, 2 r_out/N)`, so

    K3 = N d_out / (lambda |B| / dx)  >=  2 r_out dx/(lambda|B|)  =  K1,

with equality exactly when the pitch floor set the pitch.  Selecting the
transfer-function form on `K1 > 1` therefore left the whole band
`K1 <= 1 < K3` running a chirp-Z whose window is several periods wide.  That
band is not exotic: on the co-moving lattice `K3 = N dx^2/(lambda |z_eff|)`, so
it is EVERY leg whose reduced distance is shorter than `N dx^2/lambda`.  The
measured-support refinement the report is proud of in its sec. 3 is precisely
the size of the hole -- `K3/K1 = (N dx/2)/r_measured`, the same 2.8x it quotes
as an improvement.

**(b) the fallback's availability.**  `co_moving` required `d_out` to EQUAL
`|A| dx` exactly.  The floor is `2(|A| r + |B| theta)/N`, and when the measured
support `r` saturates at the grid half-width -- which it does for any beam whose
1e-6 tail reaches the grid edge, i.e. anything realistic after an aperture --
the floor exceeds `|A| dx` by `2|B| theta/N` and the fallback is vetoed by that
hair.  On the gate-(d) chain fixture leg 0 reads floor 80.1599 um against a
co-moving 80.0000 um (0.2 %), so a 47x under-sampled chirp-Z ran on an ordinary
relay leg 20 mm long.

**Fail-before, on `b4orig` (185d64cd as shipped), Gaussian `w = 0.3 mm`,
N = 512, `dx = 7.0312 um`, `R = -40 mm`:**

| probe | 185d64cd | this pass |
|---|---|---|
| leg `z = -20 mm` (K1 = 0.7600, K3 = 1.7842) | `chirp-z`, `P_out/P_in` = **1.316400**, relL2 vs the analytic Gaussian **5.6249e-01**, brightest out-of-period sample **59.15 % of peak**, 24.0 % of the returned power outside one period, **0 warnings** | `tf`, `P_out/P_in` = **1.000000**, relL2 **1.9578e-08**, `np.array_equal` to `_carrier_step_fast` |
| leg `z = -30 mm` and `z = +12 mm` (K3 = 1.3877) | `chirp-z`, relL2 **5.0515e-04**, 0 warnings | `tf`, relL2 **2.5187e-08 / 2.5172e-08** |
| `propagate_carrier_referenced(transport='collins', dx_out=12 um)`, K3 = 2.0301, `on_collins_sampling='error'` | **raised nothing**; relL2 vs the oracle **2.2374** | **raises**, naming `K3 (period)` and the period in um |
| the step at the selection boundary | boundary at K1 = 1 (z = 8.0108 mm); the two forms differ there by **9.9987e-01** of peak | boundary at K3 = 1 (z = 14.9177 mm); the two forms differ there by **2.9382e-11 / 1.3899e-11** of peak |
| the gate-(d) `_chain_fixture`, bare final leg | `'sziklas'` peak 0.960941 / power 2.82889e-05 vs `'collins'` peak **29.0002** / power **0.0189589** | identical to `'sziklas'` on peak, power, `R` and `dx` |
| a 768-sample leg at K3 = 3.8183 (`_collins_transport` directly) | `P_out/P_in` = **12.108701890** | unchanged -- the primitive IS periodic; it is the LEG that now refuses or falls back |

**The fix**, all in `lumenairy/propagators/carrier.py`:

1. `_check_collins_sampling` grows a `check_period` arm that disposes of K3
   with the same bar and the same message shape as K1 / K2.  Exactly one guard
   owns K3 per call -- `on_replica` at the readout (which also has
   `replica_fill`), `on_collins_sampling` on a leg -- so the "the two guards
   cannot disagree" property the report wanted is kept, and is now stated at
   both ends.
2. `_collins_transport` takes `check_period` and passes it down.
   `_collins_focus_readout` leaves it false; `_collins_carrier_leg` passes true.
3. `_collins_carrier_leg` computes K3 on the lattice it resolved, and takes the
   transfer-function form when the chirp-Z is not representable there
   (`K1 > 1 or K3 > 1`) and that form exists: a scalar carrier, `A > 0`, the
   geometric output reference, and no caller override.  The pitch-equality
   clause is gone -- it was the thing that vetoed the fallback over 0.2 %.
4. `collins_k3` and `collins_kernel` join the published stage diagnostics, so
   both quadratures publish the same key set.

The new rule is a strict SUPERSET of the old one (`co_moving` implies the new
availability test, and `K1 > 1` implies `K1 > 1 or K3 > 1`), so no leg that used
to take the transfer-function form stopped doing so; gate (c)'s
`np.array_equal` and the whole byte-identity set are untouched, and re-measured
below.

### D2 -- `'auto'`'s kernel resolution was not published on a leg

`_collins_transport` records `st['kernel']`, but the leg copied only
`collins_form / k1 / k2 / flat_reference / dx_floor_hit` into the stage, so a
chain consumer could not see that `'auto'` had dropped to `'fresnel'`.  Fixed by
publishing `collins_kernel` (and `None` on the transfer-function branch, which
resolves its own kernel inside `_carrier_step_fast`).

### Not defects, but the docstrings now say them

* `_collins_focus_readout` carries what actually bounds it: one step from the
  exit plane has no co-moving frame, so the CHAIN'S exit pitch has to resolve
  the exit beam's convergence over the reduced final leg.  That is easy for a
  long final leg on a small beam (the WP-A6 fixture reads K1 = 0.16) and
  impossible for a short one on a wide beam -- the gate-(d) fixture's 8 mm final
  distance on a 5.4 mm exit beam at 76 um reads **K1 = 82.36**, and its readout
  is correspondingly garbage (peak 94.96 against `'sziklas'` 1.036).  The guard
  says so, once per call; nothing in WP-B4 reports a readout K reading on that
  fixture.  See Follow-up F1.
* the chain's `transport` and `on_collins_sampling` entries now state which
  guard owns K3 and what the readout's own sampling condition is.

---

## 3. What I could not break

* the transport's arithmetic.  Every absolute-phase comparison against an
  independently written oracle -- on axis, off axis, tilted to 46 mrad,
  decentred, astigmatic, through the geometric focus, back-propagating --
  landed between 1e-15 and 1e-12 under `gap_kernel='fresnel'` (the integral the
  oracle is) wherever the output window fitted in one chirp-Z period.  The
  factorisation (pre-chirp, centred Bluestein, post-chirp, prefactor) is right,
  including the `1/(i lambda B)`, the `exp(i k B)` and the `centre_out` offset's
  sign;
* the byte-identity of the default (60/60, twice);
* K1 and K4 as derivations, and the K1 fail-before ladder;
* the flat-reference switch, on its own and composed with a traced group;
* `final_distance == 0`, `z == 0`, `R_carrier == 0`, the vocabulary refusals.

---

## 4. Follow-up

**F1 -- the readout's own sampling is the binding constraint on a default flip,
and it is not in the report's list.**  WP-B4 sec. 5 lists four missing
measurements (design-121, cost on a real design, the tilted congruence, the
backends).  Item 3 of those is now done (row 18 above).  But the readout's K1 is
a fifth, and it is the one that decides whether a flip is usable: the one-step
readout must sample the exit beam's convergence over `z_eff = z/A` on the
CHAIN'S exit pitch, `K1 = 2 dx (|A| r/|B| + theta)/lambda`.  Measured on the
package's own gate-(d) fixture at three grids: 82.36 (N = 256), 21.42
(N = 1024), 10.94 (N = 2048) -- it improves as `1/dx` and would need N ~ 16000
to be sampled.  The Sziklas readout pays for the same thing with a standoff
plane, where only the ENVELOPE has to be sampled.  So a flip would have to
either keep the Sziklas readout for short final legs or state the applicability
window; the cost table's 0.30x is measured in the regime where the Collins
readout is legal, and says nothing about the regime where it is not.

**F2 -- what a flip would now retire, revised.**  The list in WP-B4 sec. 5 is
right about the near-focus bridge, the focus-crossing split, the standoff
resolvers and C1 (measured here: the shipped transport RAISES on the exact-focus
composition that `'collins'` handles to 0.05 % of power).  I would add one entry
and remove none: with the selection now on K3, a flip also retires the question
"which legs change" -- the answer is exactly the legs with
`N dx^2 <= lambda |z_eff|`, which is checkable per design without running
anything, and everything else stays bit-identical.

**F3 -- K4 bounds the wrap, not the refinement; near a focus
`gap_kernel='fresnel'` is the more accurate setting and nothing says so.**  The
exact-kernel refinement is applied over the REDUCED frame `z_eff = B/A`, which
diverges at the focus.  Its own dropped quartic is `k |z_eff| theta^4/8` at the
envelope's measured angle, and `K4 = 1` sits at `|z_eff| ~ span/theta^3`, so the
bound permits `k theta span/8` radians of it -- **8.639 rad** on the
`w = 0.3 mm` / N = 1024 / `dx = 4 um` / `lambda = 1.064 um` fixture, against the
**9.842e-07 rad** of paraxiality the leg it refines carries.  Measured on that
fixture's focus-readout lattice, relL2 against the analytic Gaussian:

| dz from the geometric focus | `z_eff` | K4 | `'fresnel'` | `'auto'` | `k|z_eff| theta^4/8` |
|---|---|---|---|---|---|
| 1 um | -1600 m | 9.114e-03 | **1.71e-14** | **2.350e-03** | 0.0787 rad |
| 10 um | -160 m | 9.116e-04 | 1.33e-14 | 2.350e-04 | 0.00787 |
| 100 um | -16.0 m | 9.136e-05 | 8.87e-15 | 2.356e-05 | 0.000789 |
| 1 mm | -1.64 m | 9.341e-06 | 1.15e-14 | 2.408e-06 | 8.07e-05 |
| 5 mm | -0.36 m | 2.051e-06 | 1.37e-14 | 5.287e-07 | 1.77e-05 |

-- `'auto'` is linear in `|z_eff|`, K4 is three to six decades below its bar the
whole way, and the entire `'auto'` column is INDEPENDENT of N (identical at 512,
1024, 2048 and 4096 on the same physical extent), which is what rules out the
grid.  This is NOT a WP-B4 regression: the shipped `'sziklas'` path applies the
same refinement over the same `z_eff`.  What is particular to `'collins'` is
that this quadrature operates at small `A` by design, so it is the arm where the
term is met.  A second condition of the same shape -- `k |z_eff| theta^4/8`
against `gap_env_phi_tol`, the tolerance the chain already carries for exactly
this quantity -- would be about eight lines and would make `'auto'` drop to
`'fresnel'` where the refinement stops helping.  I did not write it: it changes
what `gap_kernel='auto'` means on a leg, and that is the default owner's
decision, not a verifier's.  What I did write is the measurement, in
`_collins_kernel_wrap_ratio`'s docstring and as a two-sided pin
(`TestKernelRefinementNearTheFocus`: `'fresnel'` on the oracle at 1e-14 with
`K4 < 1` on every cell), so the next person cannot assume K4 covers it.

**F4 -- K2 has no fail-before and its message reads like one.**  See row 8.
Suggested wording change only: say "does not RESOLVE" (it does) rather than
implying the returned samples are wrong.

**F5 -- for whoever owns VERIFY-A6's fixtures.**  WP-B4 sec. 8's finding about
`tests/unit/test_audit2609_a6_verify_carrier.py::_abcd_field` is confirmed
independently: `1/q = 1/R - i lambda/(pi w^2)` is Siegman's `exp(+i omega t)`
pairing and conjugates the Gouy phase.  I reproduced the `pi` by making the
mirror-image mistake in my own oracle (see the caution at the top).  The
whole-function replacement WP-B4 suggests is the right one.  That file is not
mine either.

---

## 5. Cost, measured (no test asserts a timing)

FFT-call counts, by patching `_bluestein._fft_1d` and `fft_infra._fft2/_ifft2`
in memory (the separable Bluestein route does NOT go through the caller-supplied
`fft2`, which is why counting only the latter reports 1-2 and misses the work):

| call | 2-D transforms | 1-D batched passes | `next_fast_len(N + N_out - 1)` |
|---|---|---|---|
| Sziklas gap leg, `'auto'` | `fft2(N^2)` + `ifft2(N^2)` | -- | -- |
| Collins gap leg, `'auto'`, `N_out = N` | `fft2(N^2)` + `ifft2(N^2)` | **6** of length `2N` | 1024 / 2048 / 4096 at N = 512 / 1024 / 2048 |
| Collins gap leg, `'fresnel'` | `fft2(N^2)` | **6** of length `2N` | as above |
| Sziklas readout, N = 1024 | `fft2` + `ifft2` of `N^2` | -- | -- |
| Collins readout, N = 1024, `N_out = 64 / 256` | `fft2(N^2)` | **6** of length **1089 / 1280** | 1089 / 1280 |

-- "3 FFTs per axis-pass of `next_fast_len(N + N_out - 1)`" is exact, on every
row.

Wall-clock medians of interleaved runs (one rep of each arm in turn, 7 reps,
5 at N = 2048 and for the readouts), `perf_counter`, single-threaded, on a box
that also had two other engineers' pytest runs on it -- which is why these are
reported as a measurement and not asserted anywhere:

| per gap leg (`R = -40 mm`, `z = 20 mm`) | sziklas TF | collins `'auto'` | ratio | collins `'fresnel'` | ratio |
|---|---|---|---|---|---|
| N = 512 | 15.92 ms | 84.88 ms | **5.33x** | 71.92 ms | 4.52x |
| N = 1024 | 69.09 ms | 349.47 ms | **5.06x** | 285.73 ms | 4.14x |
| N = 2048 | 315.47 ms | 1528.59 ms | **4.85x** | 1231.93 ms | 3.91x |

| image-plane readout | sziklas | collins | ratio |
|---|---|---|---|
| N = 1024, `N_out` = 64 | 292.06 ms | 78.64 ms | **0.27x** |
| N = 1024, `N_out` = 256 | 350.87 ms | 96.91 ms | **0.28x** |
| N = 2048, `N_out` = 256 | 1274.91 ms | 364.93 ms | **0.29x** |

-- the report's 4.85-5.20x and 0.29-0.30x reproduce.

---

## 6. Test runs -- exact commands, counts, durations

Every one prefixed with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time.  The box was shared with two other engineers' pytest runs throughout, so
the durations are upper bounds.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b4_collins_transport.py -q -p no:randomly --durations=8` | **126 passed** (84 the engineer's, 42 added here) | 268 s |
| `pytest tests/unit/test_audit2609_a6_carrier.py tests/unit/test_audit2609_a6_verify_carrier.py -q -p no:randomly` | **165 passed** -- the WP-A6 / VERIFY-A6 pins | 53 s |
| `pytest tests/unit/test_audit2609_a24_decentre_calibration.py tests/unit/test_audit2609_a25_carrier_focus_readout.py tests/unit/test_niche_d1_tilted_carrier.py tests/unit/test_niche_p2_design_battery.py -q -p no:randomly` | **83 passed** | 219 s |
| `pytest tests/unit -k carrier -q -p no:randomly` | **599 passed, 4 skipped, 2 failed** -- the 2 were the carrier history fingerprints, before the re-record; 601 + 4 total, matching WP-B4's own sweep.  The 4 skips are environmental and pre-existing (`PySide6` / `cupy` absent) | 595 s |
| `pytest tests/unit/test_niche_d2_chain_multi.py tests/unit/test_niche_d6_exact_tilted_leg.py -q -p no:randomly` | **76 passed** | 654 s |
| `python scripts/record_history_fingerprints.py lumenairy/propagators/carrier.py --reason "..."` | DRIFT reported on both fingerprints, re-recorded with the reason | 3 s |
| `pytest tests/unit/test_audit2609_a17_history_relocation.py -q -p no:randomly -k carrier` (after the re-record) | **12 passed**, 727 deselected | 5 s |
| `pytest tests/unit/test_audit2609_a17_history_lint.py -q -p no:randomly` | **5 passed** -- the version-narrative ratchet does not name `carrier.py` | 8 s |
| `ruff check lumenairy/propagators/carrier.py tests/unit/test_audit2609_b4_collins_transport.py` | clean | -- |
| `python validation/run_all.py` | **ALL 37 files passed** | 272 s |
| byte-identity, archive to archive (`byte_fixture.py` in a child process per tree, `byte_compare.py`) | `base` vs `b4orig` **60/60 EQUAL**; `base` vs `mine` **60/60 EQUAL** (re-run after every edit, last time after the docstrings) | 2 x 65 s |
| fail-before / fail-after (`failbefore.py` against `b4orig/` then `mine/`) | the six rows in sec. 2 | 2 x ~90 s |

Probe scripts (each in a child process with `carrier.__file__` asserted, under
`.../scratchpad/verify_b4/`): `p1_absolute_oracle.py`, `p1b_backprop.py`,
`p2_k3_hole.py`, `p3_crossover.py`, `p3b_crossover.py`, `p4_gate_a.py`,
`p5_k4.py`, `p6_exact_kernel_arbiter.py`, `p7_misc.py`, `p8_chain.py`,
`p9_legdiag.py`, `p10_why_no_tf.py`, `p11b_tilt_and_cost.py`,
`p12_flat_ref_group.py`, `p13_new_crossover.py`, `p14_chain_after.py`,
`p15_walltime.py`, `p16_nearfocus_residual.py`, `p17_kernel_cost.py`,
`p18_fresnel_sweep.py`, plus `byte_fixture.py` / `byte_compare.py` (the
archive-to-archive byte-identity pair) and `failbefore.py` (run against
`b4orig/` and `mine/`).

---

## 7. Files changed

* `lumenairy/propagators/carrier.py` -- D1 and D2 above, plus the docstring
  statements they require.  `carrier_field.py` was not touched.
* `tests/unit/test_audit2609_b4_collins_transport.py` -- **+10 test functions**
  (43 -> 53), added; none weakened, none removed.
* `docs/history/carrier.md` -- re-recorded (`--reason` given), as
  `CONTRIBUTING.md` requires for a module with a history document.
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B4.md`
  (this file) and `VERIFY_WP-B4_CHANGELOG.md`.

**What in the tree is not mine.**  The working tree also carries other Wave-4
engineers' in-flight edits -- `lumenairy/elements/_lens_real.py`,
`lumenairy/elements/rcwa/_core.py`,
`lumenairy/propagators/asymptotic_canonical_fit.py`, their history and
subsystem documents, and the B3b / B5 test files -- and
`record_history_fingerprints.py --check` reports drift on those modules
accordingly.  Only `carrier.md` is mine to re-record; `carrier_field.md` is
untouched and OK.  `.test_durations` shows as modified: it is a shared,
test-run-written artifact and several of us were running pytest on this box at
once, so I make no claim on its contents and it is not in my file list.

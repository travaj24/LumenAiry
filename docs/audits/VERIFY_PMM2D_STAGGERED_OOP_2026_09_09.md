# VERIFY -- adversarial verification of Stage B (OUT-OF-PLANE anisotropic pure staggered 2-D PMM)

Date: 2026-09-09.  Worktree `C:/tmp/lum_aniso`, branch
`feat/pmm2d-staggered-anisotropic`, HEAD at the start of this verification
`6e49bed` (the merge of `feat/pmm2d-staggered-oop`: build commits `63893d1`,
`514381e`, `7621256`, `aa88fdc`).  The pre-integration head -- the point Stage A
was verified at -- is `0c3e871`.

Verifier: an agent that did not build this.  Every number below was RE-MEASURED
with scripts under `validation/probe_verify_staggered_oop/` (see its
`README.md`).  Nothing was accepted from a comment, a table or a docstring, and
the fixtures are deliberately not the build's.

Machine for every measurement: tesla-ryzen, Windows 11 Pro 26200, CPython
3.14.6, NumPy 2.4.4, SciPy 1.17.1 (scipy-openblas),
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` unless a row
says otherwise.  Every probe asserts `lumenairy.__file__` is under the root it
was told to use.

---

## 0. Verdict summary

| # | claim (build doc) | verdict | key numbers |
|---|---|---|---|
| 1 | the scalar AND in-plane-tensor answers did not move | **CONFIRMED** | 15 fixtures / 101 hashed fields bit-identical HEAD vs `0c3e871`; 8 / 51 vs the main clone `f70628d` |
| 2a | `_OOP_ROT_SIGN = -1` is right | **CONFIRMED on independent CHIRAL fixtures** | chiral out-of-plane stripe at 25 deg: shipped `dR 8.28e-06` INSIDE the two 1-D engines' own `9.23e-06` spread; `rot = +1` gives `1.17e-03 / 1.34e-02 / 1.41e-02` |
| 2b | the rot flip is "provably invisible at NORMAL incidence" | **REFUTED AS STATED, CONFIRMED IN SCOPE** | invisible on a rho-SYMMETRIC cell (uniform slab, arm gap `6.06e-15`); VISIBLE at normal on a chiral cell (`dR 4.59e-04`, `dJones 9.88e-04`; 2-D chiral `dR 2.35e-03`) |
| 2c | `_OOP_H_GAUGE = -1j` is right | **CONFIRMED** | pure single out-of-plane slab vs Berreman: `-1j` `dJones 1.5e-14`, `+1j` `3.9e-02`, `+/-1` O(1) with `R+T` 3.69 / 11.94 |
| 2d | "`_OOP_H_GAUGE` cancels in a pure out-of-plane stack, so the T4 reduction is its ONLY gate" | **REFUTED** | it does not cancel: the Berreman uniform-slab gate (G3) separates `-1j` from `+1j` by 12 decades on the Jones, and from `+/-1` by 14 |
| 3 | the reduction (T4) | **CONFIRMED** | 6 fresh cells x 2 mounts, worst `3.04e-14`; wrong-gauge arms `2.2e-02 .. 3.0e+02` |
| 3 | dispersion + sum-of-roots (T3) | **CONFIRMED** | own quartic: `d(+k_t) 1.63e-14` vs `d(-k_t) 3.11e-02`; sum of roots `-0.156407` (conical) / `0` (normal, in-plane); transpose gap `2.3e-15` |
| 3 | Berreman ladder, two-sided (T2) | **CONFIRMED** | oblique 25, fresh tensor: `1.78e-03 (M3) -> 8.37e-14 (M8)`, 1.3e10; at NORMAL the residual RISES (roundoff plateau) -- the ladder claim is oblique-only, correctly |
| 3 | 1-D stripe per order (T5) | **CONFIRMED** | fresh stripe M=8: `dR 7.5e-07 / dT 1.2e-05` (normal) vs an oracle spread of `1.4e-06 / 2.9e-06`; y-leak `1e-26` |
| 3 | the 2-D corner cell at the oracles' spread (T6) | **CONFIRMED, BOUNDED** | fresh chiral 2-D cell at 25 deg: staggered `dR 1.49e-04`, i.e. 2.0x the largest oracle-vs-`rcwa(9)` number (`7.6e-05`); no Fourier arm converged (the hybrid's two `E_z` rules differ by `1.28e-03`) |
| 3 | no Fourier floor, two-sided (T6) | **CONFIRMED** | `n_orders` 3 -> 8 moves the staggered arm `5.6e-16 / 1.9e-07 / 4.5e-08`; the hybrid's two rules differ by `1.1e-03 / 1.3e-03 / 1.0e-03` |
| 3 | closure at 3 depths, growth factor, split (T7) | **CONFIRMED** | Hermitian closure flat in depth (`1.4e-14 -> 8.8e-14`), split exactly `2q^2/2q^2` in every fixture, max forward growth `1.0000e+00` |
| 3 | stacks: `1x0.4 == 2x0.2`, multilayer vs Berreman (T8) | **CONFIRMED** | `1.04e-16 / 5.98e-16`; fresh multilayer `dR 1.17e-13` at M=8 (its own ladder `2.6e-07 -> 1.2e-13`) |
| 3 | `layer_absorption` on the generalized cascade (T8c/d) | **CONFIRMED** | budget residual `3.9e-08 -> 3.7e-13` (M 5 -> 7) lossless, `3.8e-10` lossy at M=8; lossless layers carry `<= 1.1e-14` |
| 3 | the dispatch floor and its `1e-16` stray bit-identity (T4) | **CONFIRMED and EXTENDED** | strays at `1e-16`, `1e-14`, `1e-12` (relative) all leave `offplane` False and R/T/Jones BYTE-EQUAL; `1e-11` flips it, and crossing the boundary moves R by only `1.5e-13` |
| 3 | the T9 fail-before controls | **CONFIRMED, with a sharper transposition statement** | at NORMAL incidence the transpose is INVISIBLE to R/T (`3.7e-14`) and visible only in the Jones (`7.8e-03`) -- 11 decades apart |
| 4 | the forward/backward split survives stress | **CONFIRMED** | lossy metal, on-cutoff, high contrast at M=8, oblique 60: split exactly `2q^2/2q^2` in all 16 fixtures AND already `2q^2` before the defensive rebalance, growth `<= 1`, no warnings but the library's own cutoff warning |
| 5 | test durability (S1-S5) | **8 findings: 6 fixed as test-only changes, the rest kept with measured envelopes** | tightest bar `1.93x` (G5's Jones, envelope `< 4e-11`); tightest bar/envelope PAIR is G7's rotation gauge, `8.4x` against a `5.7e-03` envelope (~3.2 decades) |
| 6 | cost 1.33-2.03x time, ~3.0x peak | **CONFIRMED to 3 s.f.** | `(3,3)` M=8 ratio 1.328 (claim 1.33), M=6 1.600 (claim 1.60); peak 142.7 -> 433.4 MB and 37.3 -> 112.9 MB, ratios 3.036 / 3.026 -- the build's own MB figures reproduce exactly |

**Library defects against a stated contract: none.**  Three library
*documentation* findings (section 5): the `_OOP_H_GAUGE` cancellation claim, an
unreachable guard whose docstring promises a raise, and a duplicated stale
comment.  Two BUILD-DOC claims REFUTED (sections 2.2 and 2.3), neither of which
makes a shipped number wrong.  Eight durability findings (section 6): six fixed
as test-only changes (three restatements, three corrected measurements), and
the remaining tight bars kept with their measured cross-kernel envelopes and
fixture sensitivities written into the docstrings.

---

## 1. Task 1 -- the SCALAR and IN-PLANE paths are untouched, across CODE VERSIONS

`v1_paths_untouched.py`, run once per `PYTHONPATH`:

* this worktree `C:\tmp\lum_aniso\lumenairy\__init__.py` (5.42.1, HEAD `6e49bed`)
* the pre-integration head `C:\tmp\lum_aniso_pre\lumenairy\__init__.py`
  (5.42.1, `0c3e871`) -- a temporary `git worktree`, removed at the end
* the read-only main clone
  `D:\Metacept\...\Lumenairy\lumenairy\__init__.py` (5.42.1, `f70628d`), which
  predates Stage A and therefore has no tensor entry (scalar fixtures only)

Nine public solves (four scalar efficiency fixtures, one scalar pure stack,
four tensor Jones fixtures, one mixed tensor stack) plus five assembled-operator
records (`Lmat`, `Rmat`, `Stt`, `Schur`, `Et_blocks`, `Et_offdiag`, `W`, `V`,
sorted `lam`).

**HEAD vs `0c3e871`: 15 fixtures, 101 hashed fields, 0 mismatches.**
**HEAD vs `f70628d`: 8 fixtures, 51 hashed fields, 0 mismatches.**

| fixture | R sha256[:16] | T | Jones | sum R |
|---|---|---|---|---|
| `(2,2)` pillar TE normal | `524b5a6424c1642b` | `93b84eaf08a4e4cf` | -- | 0.2303468080766423 |
| `(3,3)` TM oblique 0.27 | `77c7552a9552def2` | `925e6d7ca2981000` | -- | 0.08798744890481952 |
| `(2,2)` LOSSY conical 0.19/0.71 | `a86633ec739a8042` | `1b9b434d15aaa572` | -- | 0.11275392405061355 |
| `(2,2)` TE conical 0.41/1.13 | `65d4d254d072e9b8` | `166297f3b09690ad` | -- | 0.16546250665158485 |
| scalar 2-layer stack, Jones | `a2cc5727ecdd5290` | `7df9c505f2eb74d8` | `773f66f504cd733c` | R+T 0.9997329480619113 |
| LC-host + iso pillar, normal | `48477f324f99472f` | `7b5ac1f361d7c804` | `89b88cf2b37a4a77` | 0.07377941624297225 |
| gyrotropic `(3,3)` oblique | `545442f13ac27506` | `2074e734194ba4d8` | `7434ba9e744d1cf1` | 0.08566140868028244 |
| uniform in-plane LC conical | `ea380898deee2a00` | `c17650f548dc77eb` | `be0fb3ff9021e015` | 0.06562097787282697 |
| mixed tensor+scalar stack | `d05c0afa63a9c429` | `43081445891c90ae` | `1d878ef492b2c3d0` | R+T 1.000000029052142 |

| assembly | `Lmat` | `Rmat` | `Stt` | `Schur` | sorted `lam` | `W` | `V` |
|---|---|---|---|---|---|---|---|
| scalar `(2,2)` M=6 | `823a4ab5a6faafa6` | `843e0bbc1012fa2c` | `3e6405708a9c6c07` | `109f559b864bcfd1` | `8a1ab1b9deee1400` | `7d6d6d6227d4df88` | `434cc830982d8bdd` |
| scalar `(3,3)` M=5 | `0b094f884641e6b6` | `427923531679b8e8` | `8fba1cf1e5e4b210` | `13fc4f41f9ccdb82` | `e6a54ebb9dd084c9` | `ac2545c8535fccfa` | `186da772d6545e47` |
| scalar lossy `(2,2)` M=6 | `49e3a9dd5085545c` | `f444fef455380649` | `246304b911f7d2a9` | `05ef57c4a45dd812` | `e4c34e95baf9bb70` | `7e8099b09cdba916` | `e77969563f295477` |
| in-plane LC `(2,2)` M=6 | `3c37a78efeacf3c5` | `843e0bbc1012fa2c` | `3e6405708a9c6c07` | `79057541084bf358` | `28b2c17f0795e88b` | `8ffc00afaa3c2ec1` | `aa9ea1871c2373f5` |
| gyrotropic `(3,3)` M=5 | `61e139caa492f593` | `427923531679b8e8` | `8fba1cf1e5e4b210` | `2c9916f8062a97b4` | `50fa632a98279dd3` | `2c1bb912cb4782d1` | `d553887f2cc07128` |

(The eps-free `Rmat` / `Stt` hashes repeat across cells of the same grid and
`M`, as they must -- an internal consistency check on the hashing.)

**Verdict: CONFIRMED.**  Neither the scalar nor the in-plane tensor path moved a
byte when `_require_block_form` became `_tile_needs_oop`, when `_region_modes`
grew its guard, or when `PMM2DStackPure.solve` grew the `any_oop` branch.

---

## 2. Task 2 -- the two gauge constants

This is the crux, and the build's own evidence for both constants is measured on
cells that are their OWN 180-degree image (a uniform slab, and pillars/stripes
that are 180-degree symmetric up to a lattice shift).  Such a cell can only see
`rho`'s action on the TENSOR, never on the PATTERN.  So the fixtures here are
CHIRAL.

### 2.1 `_OOP_ROT_SIGN = -1` -- CONFIRMED, on four independent arms

#### (a) a CHIRAL 3-segment out-of-plane stripe vs the two 1-D engines

`v2a_gauge_chiral_1d.py`.  Segments `[uniaxial(1.48, 1.73, 0.62, 0.37),
2.10 I, uniaxial(1.55, 1.80, 1.02, 2.20)]` -- three DISTINCT tensors, so the
reversed sequence is not a lattice translate of itself, and two of them carry
out-of-plane coupling that flips sign under `rho`.  `px = 1.15 um`,
`wl = 0.70 um`, `depth = 0.31 um`, `n_sub = 1.45`.  Oracles:
`pmm_jones_1d_segments(degree=18, far_field_orders=31, stabilize=False)` and
`rcwa_jones_1d_segments(n_orders=41)` -- two engines that both accept full
`(3,3)` out-of-plane tensors at planar incidence and share no code with the
staggered path.  Compared PER ORDER (31 orders), both polarizations, on R, T
and the Jones.

**OBLIQUE 25 deg.  The oracles' own mutual spread is the bar:
`dR 9.229e-06`, `dT 9.268e-06`, `dJones 2.356e-05`.**

| arm | vs `pmm_jones_1d_segments` dR / dT / dJones | vs `rcwa_jones_1d_segments` | `sum R+T` |
|---|---|---|---|
| shipped `rot = -1`, M=5 | 3.813e-05 / 7.370e-05 / 9.450e-05 | 4.735e-05 / 8.114e-05 / 1.178e-04 | 0.999972348 |
| M=6 | 1.323e-05 / 1.464e-05 / 3.328e-05 | 2.246e-05 / 2.391e-05 / 5.684e-05 | 0.999997441 |
| M=7 | 8.912e-06 / 8.931e-06 / 2.268e-05 | 1.814e-05 / 1.770e-05 / 4.623e-05 | 0.999999957 |
| **M=8** | **8.282e-06 / 8.020e-06 / 2.100e-05** | 1.751e-05 / 1.729e-05 / 4.456e-05 | 1.000000001 |
| **`rot = +1`, M=7** | **1.174e-03 / 1.340e-02 / 1.412e-02** | 1.184e-03 / 1.340e-02 / 1.411e-02 | 0.999999963 |

The shipped arm converges monotonically to INSIDE the two oracles' own mutual
spread; the flipped arm is 127x / 1446x / 599x outside it.  Note `sum R+T` is
1.000000 on BOTH arms -- no energy check can see this.

**NORMAL incidence, SAME chiral cell.**  Spread `1.064e-05 / 2.758e-05 /
2.251e-05`; shipped M=8 `9.089e-06 / 2.182e-05 / 1.917e-05` (inside);
`rot = +1` M=7 **`4.497e-04 / 6.509e-03 / 9.679e-04`**.

#### (b) a genuinely 2-D CHIRAL out-of-plane cell vs the two Fourier oracles

`v2b_gauge_chiral_2d.py`.  `(3,3)` cell, air background, two DIFFERENT
out-of-plane tensors at `(0,0)` and `(1,0)` and an isotropic pixel at `(1,2)`
(not 180-degree symmetric in pattern or tensor).  `px = py = 1.10 um`,
`wl = 0.68 um`, `depth = 0.36 um`, `n_sub = 1.50`.  Oracles: `rcwa_jones_2d`
pixel-upsampled at `n_orders` 5 / 7 / 9 (its own ladder bounds its drift) and
`pmm_jones_2d(degree=9, n_orders=13)` in BOTH `E_z` rules.

| mount | rcwa 5 vs 9 | rcwa 7 vs 9 | hyb-laurent vs rcwa(9) | hyb-li vs rcwa(9) | **staggered M=7** | self-move M6->M7 | `\|R+T-1\|` | **`rot=+1` M=6** |
|---|---|---|---|---|---|---|---|---|
| normal | 2.072e-04 | 8.921e-05 | 2.144e-04 | 2.769e-04 | **3.421e-04** | 2.28e-05 | 4.23e-07 | **2.353e-03** |
| oblique 25 | 1.251e-04 | 6.312e-05 | 7.615e-05 | 5.941e-05 | **1.489e-04** | 1.51e-04 | 1.90e-08 | **2.499e-03** |
| conical 25/40 | 1.065e-04 | 6.772e-05 | 9.107e-05 | 6.104e-05 | **1.341e-04** | 4.27e-05 | 7.17e-08 | **3.851e-03** |

(dR per order.)  The shipped arm sits at 1.2 - 2.0x the LARGEST of the four
oracle-vs-`rcwa(9)` numbers in the same row -- i.e. at the oracles' own mutual
spread on a cell where no Fourier arm is converged (the hybrid's two rigorous
`E_z` rules differ by `1.10e-03 / 1.28e-03 / 1.02e-03`); the flipped arm is
6.9x / 16.8x / 28.7x further out again.  **BOUNDED, not machine-precision** -- exactly the
build's own open item 1, reproduced on a different shape.

#### (c) the DISPERSION discriminator, with my own quartic

`v3_dispersion_and_berreman.py` section A.  The quartic
`det[eps + k k^T - |k|^2 I] = 0` in `q = kz/k0` is built here entry by entry
with `np.polymul`/`polyadd`/`polysub` -- no library dispersion routine is
consulted -- and its four roots compared to the assembled generator's
Cholesky-whitened eigenvalue set, `(2,2)` cell, M=8.

| tensor | mount | sum of the four roots | rot | d to nearest eigenvalue at `+k_t` | at `-k_t` |
|---|---|---|---|---|---|
| in-plane control (forced dispatch) | normal | `-0` | -1 / +1 | 3.068e-14 / 3.068e-14 | 3.068e-14 / 3.068e-14 |
| in-plane control | conical 25/40 | `-2.47e-17` | -1 / +1 | 1.449e-14 / 1.449e-14 | 1.449e-14 / 1.449e-14 |
| symmetric out-of-plane | normal | `-0` | -1 / +1 | 2.403e-14 / 2.173e-14 | same |
| **symmetric out-of-plane** | **conical 25/40** | **`-0.156407`** | **-1** | **1.633e-14** | **3.107e-02** |
| symmetric out-of-plane | conical 25/40 | | +1 | 3.107e-02 | 1.633e-14 |
| NON-RECIPROCAL | conical 25/40 | `-0.156407` | -1 | 1.340e-14 | 3.017e-02 |

Three separate claims land here, all CONFIRMED:

* the sum of the fundamental's four roots is EXACTLY zero for an in-plane
  tensor and for any tensor at `k_t = 0`, and non-zero only through the
  out-of-plane coupling -- so it is a genuine discriminator;
* `rot = -1` puts the generator on the `+k_t` (physical) roots and `rot = +1`
  on the `-k_t` ones, 12 decades apart, in BOTH directions;
* **transpose-blindness** (the build's S5-T3 theorem): the root set for `eps`
  and for `eps^T` differ by `2.30e-15` / `9.05e-16`, and the sum is identical
  for the symmetric and the non-reciprocal tensor.  No eigenvalue or energy
  measurement can see an `e13`/`e31` swap.  (Proof, for the record:
  `M(eps^T) = M(eps)^T` because `k k^T` and `I` are symmetric, and
  `det M^T = det M`.)

#### (d) the uniform slab vs `berreman_jones_1d`, fresh tensor

`uniaxial(1.52, 1.78, 0.55, phi 0.93)`, `px = 1.15 um`, M=8:

| mount | shipped `rot = -1` dR / dT / dJones | `rot = +1` | arm gap (dJones) |
|---|---|---|---|
| normal | 5.442e-14 / 2.787e-14 / 1.262e-13 | 5.322e-14 / 3.109e-14 / 1.260e-13 | **6.064e-15 (invisible)** |
| oblique 25 | 1.768e-13 / 9.026e-13 / 4.730e-13 | 9.408e-04 / 9.408e-04 / 2.825e-02 | 2.825e-02 |
| conical 25/40 | 6.559e-14 / 4.996e-14 / 1.535e-13 | 1.850e-04 / 1.850e-04 / 7.513e-03 | 7.513e-03 |

Order leak `1e-26 .. 8e-19` throughout.

### 2.2 The "invisible at normal incidence" claim -- REFUTED AS STATED

The build doc and `test_g7_rotation_gauge_is_two_sided` say the rot flip is
"unobservable at NORMAL incidence on a uniform cell ... the rotation maps that
configuration onto itself".  The parenthetical is the whole content: it is a
statement about `rho`-SYMMETRIC cells, not about normal incidence.

| cell | mount | arm gap under the flip |
|---|---|---|
| UNIFORM slab (`rho`-symmetric) | normal | dJones **6.06e-15** -- invisible |
| CHIRAL 3-segment stripe | normal | dR **4.594e-04**, dJones **9.878e-04** |
| CHIRAL 2-D cell | normal | dR **2.353e-03** (per order, vs `rcwa_jones_2d(9)`; shipped arm 3.42e-04) |

Mechanism, MEASURED rather than argued (`v8_rot_equals_negate.py`):
**`solve(cell, rot = +1)` is BIT-IDENTICAL to `solve(negate_oop(cell),
rot = -1)`** on every fixture and mount tried -- uniform slab, chiral stripe
and chiral 2-D cell at normal / oblique 25 / conical 25/40, `max|dR| =
max|dJones| = 0.0` in all nine.  So `_OOP_ROT_SIGN` IS the "negate the
out-of-plane block" control, and everything known about one transfers to the
other, including exactly when it is invisible: negating the out-of-plane block
is `rho` acting on the tensor alone, which on a `rho`-symmetric cell at normal
incidence is a symmetry of the whole configuration and on a chiral cell is not.

(Corollary worth recording for the build doc: T2's `rot = +1` column and T9's
`negate` row are therefore the SAME measurement, not two independent
fail-befores -- which is why they print the same `7.80e-06`.  The independent
controls are `drop` and, on a non-reciprocal tensor, `transpose`.)

The gate as written is still sound (it uses a uniform cell); what needed
correcting is the STATEMENT, which is now in the test docstring.

### 2.3 `_OOP_H_GAUGE = -1j` -- the VALUE is CONFIRMED, the stated MECHANISM is REFUTED

`v2c_hgauge.py` walks the constant over `-1j` (shipped), `+1j`, `+1`, `-1` and
`-2j` (a pure scale) against `berreman_jones_1d`.

**The build doc's claim** (section 2 T4, the module constant's docstring, and
`test_g1_oop_generator_reduces_to_the_inplane_path`'s docstring): *"It cancels
in a pure out-of-plane stack and does NOT cancel in a MIXED one"*, therefore
the in-plane reduction *"is its only gate"*.

**It does not cancel.**  A PURE single uniform out-of-plane layer between
isotropic half-spaces, vs the exact Berreman oracle (`uniaxial(1.47, 1.71,
0.68, phi 1.10)`, `depth 0.29 um`, `px 0.87 um`, `wl 0.66 um`, `n_sub 1.55`,
M=6):

| gauge | normal dR / dJones | oblique 25 | conical 25/40 | `sum R+T` (normal) |
|---|---|---|---|---|
| **`-1j` (shipped)** | **1.55e-15 / 1.52e-14** | **3.34e-10 / 8.53e-10** | **1.10e-11 / 6.99e-11** | 1.000000000 |
| `+1j` | 1.58e-15 / **3.905e-02** | 9.130e-04 / **4.301e-02** | 2.811e-04 / **4.378e-02** | **1.000000000** |
| `+1` | 2.395e+00 / 1.344e+00 | 1.430e+00 / 9.735e-01 | 1.565e+00 / 1.033e+00 | **3.692409265** |
| `-1` | 1.178e+01 / 3.506e+00 | 3.097e+00 / 1.959e+00 | 3.050e+00 / 1.884e+00 | **11.937235084** |
| `-2j` (pure scale) | 4.131e-01 / 4.750e-01 | 3.740e-01 / 4.355e-01 | 3.706e-01 / 4.460e-01 | 1.000000000 |

So `test_g3_uniform_oop_slab_matches_berreman` gates this constant as hard as
the reduction does -- 12 decades on the Jones against `+1j` and 14 against
`+/-1`.  Two things are worth keeping from the correct version of the claim:

* at NORMAL incidence on a SINGLE layer, `+1j` leaves R and T matching Berreman
  to **1.575e-15** while the Jones is **3.905e-02** out.  A pure sign flip on
  the layer's H is invisible to the efficiencies there.  Add a second
  out-of-plane layer and even R breaks (`dR 2.202e-02`);
* `sum R + T` is **exactly 1.000000000 on every arm of the `+/- i` sweep**.  The
  lossless trap in its purest form: this constant is 12 decades wrong and no
  energy check moves.

The other two arms, for completeness:

| stack | mount | `-1j` dJones | `+1j` | `+1` | `-1` | `-2j` |
|---|---|---|---|---|---|---|
| two pure out-of-plane layers | conical 25/40 | 6.44e-11 | 1.334e-01 | 1.269e+00 | 4.256e+00 | 4.470e-01 |
| MIXED (out-of-plane + in-plane + iso) | conical 25/40 | 9.45e-11 | 1.140e-01 | 6.694e-01 | 5.313e-01 | 3.189e-01 |

**The gauge flip on the CHIRAL fixtures too** (`v2d_hgauge_on_chiral.py`), so
that both constants have a fail-before on both shapes:

| fixture | mount | `-1j` (shipped) dR / dT / dJones | `+1j` | `sum R+T` on the `+1j` arm |
|---|---|---|---|---|
| chiral out-of-plane stripe, M=7 (vs `pmm_jones_1d_segments`) | oblique 25 | 8.912e-06 / 8.931e-06 / 2.268e-05 | **4.254e-03 / 3.495e-02 / 2.651e-02** | 1.000021923 |
| same | normal | 9.674e-06 / 2.293e-05 / 2.040e-05 | **6.110e-03 / 4.115e-02 / 2.408e-02** | 1.000016857 |
| chiral 2-D cell, M=6 (vs `rcwa_jones_2d(9)`) | oblique 25 | 1.681e-04 / 2.104e-03 / 1.511e-03 | **2.791e-01 / 9.761e-02 / 7.368e-02** | **1.139051128** |

On a PATTERNED cell the wrong gauge does eventually break energy as well
(`R+T` 1.00002 on the stripe, 1.139 on the 2-D cell) -- but only on the 2-D
cell is the violation large enough for the closure tripwire's `5e-02` window to
see it, and on the uniform slab it stays exactly 1.

**The forced in-plane REDUCTION (the build's T4), on six FRESH cells x two
mounts** (in-plane LC / gyrotropic / lossy-diagonal, each as a uniform cell and
as a patterned pillar; cross terms hard-zeroed; dispatch forced by patching
`_tile_is_offplane`):

| gauge | worst `max(dR, dT, dJones)` over the 12 comparisons | smallest break |
|---|---|---|
| **`-1j` (shipped)** | **3.042e-14** | -- |
| `+1j` | 4.504e-01 | 2.207e-02 |
| `+1` | 9.835e+01 | -- |
| `-1` | 3.018e+02 | -- |
| `-2j` | 4.344e-01 | -- |

**Verdict: the constant is right; the reason given for why it needs gating is
wrong, and so is the claim that T4 is its only gate.**  Reported as a library
documentation defect (section 5) and corrected in the test docstring.

**`layer_absorption` under the gauge sweep** (lossy out-of-plane + in-plane +
iso, conical 20/35, M=6): the budget identity `sum A == 1 - R - T` holds on
EVERY arm (`2.8e-11 .. 3.0e-10`), while the absorbed fraction goes from
`+0.23998689` (shipped) to `-0.30212362` (`+1j`, unphysical, with `R+T = 1.302`
for a passive stack).  So the budget check is not a gauge discriminator either;
only the comparison against an oracle is.

### 2.4 Task 2(d) -- the non-reciprocal transposition trap

`v3_dispersion_and_berreman.py` section C.  Tensor: `uniaxial(1.44, 1.76, 0.72,
phi 0.44)` with `e13 -> e13 + 0.23i` and `e31 = conj(e13)` -- Hermitian
(lossless) but `eps != eps^T`.  Uniform slab, M=8, vs `berreman_jones_1d`.

| control | normal dR / dJones | oblique 25 | conical 25/40 |
|---|---|---|---|
| *reference (shipped)* | *3.079e-14 / 7.712e-14* | *2.204e-14 / 8.324e-14* | *1.815e-14 / 7.535e-14* |
| drop out-of-plane | 8.254e-03 / 1.874e-02 | 8.899e-03 / 3.312e-02 | 9.001e-03 / 2.806e-02 |
| negate out-of-plane | 2.748e-14 (no-op) / 6.924e-14 | 4.315e-03 / 4.691e-02 | 2.530e-03 / 3.537e-02 |
| **TRANSPOSE out-of-plane** | **3.664e-14 / 7.834e-03** | 3.085e-03 / 4.691e-02 | 2.484e-03 / 3.595e-02 |

and the symmetric-tensor control, where `transpose` is a literal no-op
(`1.59e-14 .. 2.73e-14`, i.e. bit-comparable to the reference) and `negate` is
a no-op at NORMAL incidence only (`3.176e-14` vs a reference of `3.163e-14`).

**The sharpest new statement:** at NORMAL incidence on a non-reciprocal tensor
the `e13 <-> e31` swap is INVISIBLE to R and T (`3.66e-14`, i.e. at the
reference) and visible ONLY in the Jones (`7.83e-03`).  Eleven decades apart.
The build's T9 reports the conical row only, where both are visible; the
normal-incidence row is the one that shows why a Jones oracle is structurally
required.  The shipped path is on the correct side of all of it.

---

## 3. Task 3 -- every build gate re-measured on fresh fixtures

### 3.1 Reduction, dispatch floor, dispersion

Covered by sections 2.3 (reduction, `3.042e-14` worst) and 2.1c (dispersion).
The **dispatch floor** (`v3` section D; in-plane LC cell with an iso pillar,
conical 20/35, M=6, a stray injected into one `xz` slot as a multiple of the
cell's own scale):

| stray / scale | `offplane` | R / T / Jones byte-identical to the clean cell | dR |
|---|---|---|---|
| 1e-16 | False | **yes** | 0.0 |
| 1e-14 | False | **yes** | 0.0 |
| 1e-12 | False | **yes** | 0.0 |
| 1e-11 | True | no | 1.507e-13 |
| 1e-3 | True | no | 1.526e-05 |

Two things beyond the build's claim: the floor really is at `1e-12` relative
(not merely "somewhere between 1e-16 and 1e-3"), and **crossing it costs
1.5e-13** -- the two branches agree ACROSS the dispatch boundary, so nothing
hinges on which side a marginal cell lands.

### 3.2 The Berreman ladder, two-sided

`v3` section B, uniform slab, `px 0.91 um`, `wl 0.67 um`, `depth 0.33 um`,
`n_sub 1.55`; `dJones` vs `berreman_jones_1d`:

| tensor | mount | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|---|---|---|
| symmetric | normal | 1.35e-15 | 2.11e-15 | 6.88e-15 | 9.87e-15 | 2.74e-14 | 7.61e-14 |
| symmetric | oblique 25 | 1.78e-03 | 1.65e-05 | 1.44e-07 | 1.23e-09 | 7.10e-12 | **8.37e-14** |
| symmetric | conical 25/40 | 4.94e-04 | 2.84e-06 | 2.15e-08 | 1.05e-10 | 3.67e-13 | **7.66e-14** |
| NON-RECIPROCAL | oblique 25 | 1.94e-03 | 1.52e-05 | 1.42e-07 | 1.22e-09 | 7.04e-12 | **8.32e-14** |

Spectral, monotone, 1.3e10 over five degrees at oblique -- and at NORMAL
incidence the residual RISES with M, because it is already at the roundoff
plateau at M=3.  The build's decision to make the ladder claim at oblique
incidence only is therefore correct, and a normal-incidence ladder would be a
false gate.

### 3.3 The y-uniform stripe, per order

`v4_stripe_cascade_stacks.py` T5.  FRESH fixture: ridge `uniaxial(1.49, 1.75,
0.64, phi 0.28)`, groove `2.00 I` (dielectric corners, unlike the build's air
groove), `px = py = 1.18 um`, `wl 0.64 um`, `depth 0.37 um`, duty 0.5.
Oracles: `rcwa_jones_1d_segments(41)` and `pmm_jones_1d_segments(degree=18,
far_field_orders=31, stabilize=False)`.

| mount | oracles' own spread dR / dT / dJones | M | dR | dT | dJones | y-leak R / T | `\|R+T-1\|` |
|---|---|---|---|---|---|---|---|
| normal | 1.437e-06 / 2.915e-06 / 4.202e-06 | 5 | 1.085e-04 | 4.305e-04 | 3.145e-04 | 3.0e-29 / 9.9e-29 | 1.24e-04 |
| | | 6 | 2.882e-05 | 1.664e-04 | 7.772e-05 | 1.6e-27 / 7.0e-28 | 1.40e-05 |
| | | 7 | 9.166e-07 | 3.806e-05 | 1.143e-05 | 1.9e-27 / 2.0e-27 | 8.63e-07 |
| | | **8** | **7.533e-07** | **1.204e-05** | **2.802e-06** | 3.1e-26 / 1.4e-26 | 2.71e-08 |
| oblique 25 | 2.329e-06 / 2.157e-06 / 9.439e-06 | 5 | 1.720e-04 | 1.120e-03 | 1.197e-03 | 1.1e-28 / 4.9e-29 | 2.13e-04 |
| | | **8** | **8.619e-06** | **9.547e-06** | **5.177e-05** | 1.6e-26 / 6.2e-27 | 6.75e-07 |

Monotone; at normal the M=8 residual is INSIDE the oracles' own spread on R and
Jones.  **Y-momentum is conserved to 1e-26**: a y-invariant cell puts nothing in
`(m, n != 0)`, which a mis-placed `A23`/`A32` would break.

### 3.4 Cascade stability vs DEPTH, and the split

`v4` T7, M=7, `px = py = 1.22 um`, depths 0.25 / 1 / 3 wavelengths.

| cell | mount | 0.25 lam | 1 lam | 3 lam | split | raw fwd count BEFORE the rebalance | max fwd growth | max bwd growth |
|---|---|---|---|---|---|---|---|---|
| uniform Hermitian | normal | 1.42e-14 | 2.95e-14 | 8.82e-14 | 288/288 | 288 | 1.0000e+00 | 1.0000e+00 |
| uniform Hermitian | conical 25/40 | 3.44e-11 | 1.70e-11 | 2.73e-11 | 288/288 | 288 | 1.0000e+00 | 1.0000e+00 |
| pillar Hermitian | normal | 2.63e-08 | 4.47e-06 | 7.65e-06 | 288/288 | 288 | 1.0000e+00 | 1.0000e+00 |
| pillar Hermitian | conical 25/40 | 3.98e-06 | 2.15e-05 | 2.45e-05 | 288/288 | 288 | 1.0000e+00 | 1.0000e+00 |
| pillar LOSSY | normal | 4.19e-02 | 1.43e-01 | 3.55e-01 | 288/288 | 288 | 9.0381e-01 | 9.0381e-01 |
| pillar LOSSY | conical 25/40 | 4.39e-02 | 1.52e-01 | 3.67e-01 | 288/288 | 288 | 8.9027e-01 | 8.9348e-01 |

Closure does not grow with depth on the Hermitian arms; absorption rises
monotonically on the lossy one; the split is exact.  The "raw fwd count" column
is measured with an INDEPENDENT re-implementation of `_select_forward_flux`'s
classification rule taken BEFORE its defensive rebalance -- see section 5,
observation 2.

### 3.5 Stacks and `layer_absorption`

`v4` T8.

| claim | mount | measurement |
|---|---|---|
| one 0.4-lam out-of-plane layer == two 0.2-lam layers, per order / Jones | normal | 1.041e-16 / 4.677e-16 |
| same | conical 20/35 | 5.551e-17 / 5.978e-16 |
| uniform out-of-plane MULTILAYER (lossless + non-reciprocal + lossy) vs `berreman_jones_1d`, dR / dT / dJones | oblique 25, M=5 | 2.627e-07 / 1.554e-06 / 7.688e-07 |
| | oblique 25, M=6 | 3.073e-09 / 1.665e-08 / 7.890e-09 |
| | oblique 25, M=7 | 2.562e-11 / 1.337e-10 / 6.494e-11 |
| | **oblique 25, M=8** | **1.170e-13 / 7.200e-13 / 3.252e-13** |
| | **conical 25/40, M=8** | **5.565e-14 / 4.752e-14 / 1.125e-13** |

`retain_internal` / `layer_absorption` on the generalized cascade
(out-of-plane over in-plane over uniform, conical 20/35):

| stack | M | `\|R+T-1\|` | `\|sum A - (1-R-T)\|` | max per-layer `\|A_i\|` on the lossless layers |
|---|---|---|---|---|
| LOSSLESS mixed | 5 | 3.91e-08 | 3.913e-08 | 2.99e-15 |
| LOSSLESS mixed | 6 | 3.63e-12 | 3.646e-12 | 8.41e-15 |
| LOSSLESS mixed | 7 | 3.66e-13 | 3.821e-13 | 9.34e-15 |
| LOSSY out-of-plane + in-plane | 7 | -- | 9.216e-09 | 8.97e-15 |
| **LOSSY** | **8** | -- | **3.813e-10** | 1.08e-14 |

The residual falls with degree, so it is discretization-limited, not a formula
error -- the build's conclusion, on a different fixture.

### 3.6 No Fourier floor

Measured on the chiral 2-D out-of-plane cell (`v2b`) rather than the build's L:

| mount | staggered, `n_orders` 3 -> 8 (per order / Jones) | the hybrid's two `E_z` rules at `n_orders = 13` |
|---|---|---|
| normal | 5.551e-16 / 8.356e-17 | 1.098e-03 |
| oblique 25 | 1.873e-07 / 8.058e-08 | 1.275e-03 |
| conical 25/40 | 4.467e-08 / 1.972e-08 | 1.017e-03 |

Twelve and four decades of contrast.  **CONFIRMED.**

---

## 4. Task 4 -- break attempts

`v5_break_attempts.py`, sixteen fixtures.  For each: the forward/backward
split, the forward count BEFORE the rebalance, `min Re(lam_f)` (negative would
mean a growing mode was classified forward), the max forward and backward
growth factors at the solve's own depth, and -- on the Hermitian arms -- the
closure.  `px = 0.98 um`, `wl = 0.62 um` unless stated.

| stressor | split | raw fwd | `min Re(lam_f)` | max fwd growth | `sum R+T` | closure |
|---|---|---|---|---|---|---|
| `eps = -20 + 2i` METAL in an out-of-plane LC host, normal | 288/288 | 288 | **+1.926e-03** | 9.9416e-01 | 0.918605 | (lossy) |
| same, oblique 60 | 288/288 | 288 | +2.285e-03 | 9.9308e-01 | 0.837980 | (lossy) |
| same, conical 60/40 | 288/288 | 288 | +3.178e-03 | 9.9038e-01 | 0.825883 | (lossy) |
| same, oblique 60, depth 3 lam | 288/288 | 288 | +2.285e-03 | 9.5783e-01 | 0.630867 | (lossy) |
| Rayleigh cutoff, `n_sub - kt = 3e-3` (kt^2 gap 9.0e-03) | 200/200 | 200 | -3.85e-14 | 1.0000e+00 | 0.999999 | 1.31e-06 |
| `1e-3` (kt^2 gap 3.0e-03) | 200/200 | 200 | -2.84e-14 | 1.0000e+00 | 0.999996 | 4.08e-06 |
| `3e-4` (kt^2 gap 9.0e-04) | 200/200 | 200 | -1.77e-14 | 1.0000e+00 | 0.999991 | 9.14e-06 |
| `1e-4` (kt^2 gap 3.0e-04, just OUTSIDE the warning band) | 200/200 | 200 | -2.81e-14 | 1.0000e+00 | 0.999984 | 1.63e-05 |
| `3e-5` (kt^2 gap 9.0e-05, library WARNS) | 200/200 | 200 | -2.13e-14 | 1.0000e+00 | 0.999977 | 2.27e-05 |
| exactly ON the cutoff (the wavelength nudge moves it to 4.5e-07; library WARNS) | 200/200 | 200 | -2.11e-14 | 1.0000e+00 | 1.000000 | **9.11e-12** |
| high contrast (`n_e = 3.6` host + `eps = 16` pillar) M=8 normal | 392/392 | 392 | -1.08e-14 | 1.0000e+00 | 0.999917 | 1.09e-04 |
| same, oblique 60 | 392/392 | 392 | -1.36e-14 | 1.0000e+00 | 0.996018 | 3.98e-03 |
| same, `(3,3)` M=8 conical 60/40 | 882/882 | 882 | -9.36e-15 | 1.0000e+00 | 1.000000 | 1.87e-07 |
| oblique 60, depth 0.25 / 1 / 3 lam | 288/288 | 288 | -8.22e-15 | 1.0000e+00 | 1.000006 | 6.1e-06 / 1.3e-05 / 6.1e-06 |

**Nothing broke.**  Every split came out exactly `2q^2 / 2q^2` and -- more
usefully -- the raw classification came out `2q^2` BEFORE the defensive
rebalance in all sixteen, so the rebalance never had to mask a
misclassification.  `min Re(lam_f)` is `>= -1.5e-14` everywhere and strictly
POSITIVE (`+1.9e-03 .. +3.2e-03`) on the lossy-metal cells, as a passive medium
requires.  No forward growth above 1 at any depth.  Every result finite, every
efficiency in `[0, 1]`.

Two observations, neither a defect:

* **the near-cutoff degradation is monotone and bounded.**  Approaching the
  substrate's `(1,0)` cutoff from OUTSIDE the library's own `1e-4` warning band,
  the closure goes `1.31e-06` (kt^2 gap 9.0e-03) -> `1.63e-05` (3.0e-04, the
  last unwarned point) -> `2.27e-05` (9.0e-05, warned).  The documented `~1/sqrt(distance)`
  behaviour, no blow-up.  Landing EXACTLY on the cutoff is the best-behaved of
  all (`9.11e-12`), because `_grazing_safe_wavelength` nudges away from it --
  the worst case is just outside the nudge and just inside the anomaly.
* **oblique 60 on a high-contrast pillar at M=8 closes to only 3.98e-03**,
  which is under-resolution (the closure tripwire's window is `5e-02`, so this
  is silent).  It is the documented degree limit of the basis, not a cascade
  failure: the same cell at normal incidence closes to `1.09e-04` and the
  `(3,3)` conical arm to `1.87e-07`.

---

## 5. Library defects and observations

**Defects against a stated contract: none.**  Three documentation-level
findings:

1. **`_OOP_H_GAUGE`'s docstring is wrong about the mechanism** (and the build
   doc's section 2 / T4 repeat it).  `lumenairy/elements/pmm/twod_staggered.py`
   line 181 states *"It cancels in a pure out-of-plane stack and does NOT
   cancel in a MIXED one, so it is applied in `_region_modes_oop`"*, and
   `_region_modes_oop`'s own docstring (line 1201) repeats it.  Reproducer:
   `validation/probe_verify_staggered_oop/v2c_hgauge.py <root> <out.json> 1`
   -- a PURE single uniform out-of-plane layer between isotropic half-spaces
   moves by `dJones 3.905e-02` when the constant is flipped to `+1j`, and by
   O(1) (`sum R+T` = 3.69 / 11.94) for `+/-1`.  The constant's VALUE is right;
   only the reason for needing it is misstated.  No library change is required
   for correctness -- the effect is to under-sell the gating that
   `test_g3_uniform_oop_slab_matches_berreman` already provides.

2. **`_region_modes_oop`'s `2 q^2 / 2 q^2` guard is unreachable, and its
   docstring promises a raise that cannot happen.**
   `lumenairy/elements/pmm/twod_staggered.py` line 1230 raises `RuntimeError`
   when `fidx.size != 2 * qq`, and the docstring (line 1196) says *"A split that does not
   come out exactly `2 q^2 / 2 q^2` raises rather than cascading a
   rank-deficient set."*  But `rcwa._core._select_forward_flux` ends with a
   defensive rebalance (`fwd_fixed[order[:2 * N]] = True; return
   np.where(fwd_fixed)[0]`) that returns EXACTLY `2N` indices unconditionally,
   so `fidx.size` is always `2 * qq` and the guard can never fire.  A genuine
   misclassification is therefore SILENTLY rebalanced rather than raised.
   Reproducer / measurement: `v5_break_attempts.py` and `v4` T7 re-implement the
   pre-rebalance classification rule independently and report it alongside the
   shipped split; on all sixteen stress fixtures the raw count already equals
   `2 q^2`, so nothing is currently being masked -- but the guard is not the
   safety net the docstring describes.  A real guard would compare the
   pre-rebalance count, or assert `Re(lam_f) >= 0`.  `test_g6_*` now asserts the
   latter (section 6).

3. **`stack2d_pure.solve` has a duplicated stale comment block.**  Lines 411-419
   carry the old three-line "Per-layer modes: uniform -> shared geom ..."
   comment immediately followed by its replacement.  Cosmetic; no behaviour.

Two Stage-A findings still stand and were re-confirmed in passing: the closure
tripwire is blind to a modal-admittance error, and the Wood-anomaly `eps` list
differs between the scalar and tensor paths on a layer cutoff.

---

## 6. Task 5 -- durability audit of the new tests

`tests/unit/test_pmm2d_staggered_oop.py` (20 test functions, 35 parametrized
cases) and the two rewritten out-of-plane guard functions of
`tests/unit/test_pmm2d_staggered_anisotropic.py`, against S1-S5 of
`docs/TESTING_STANDARDS.md`.  Every margin below is MEASURED on the test file's
OWN fixtures (`v6_durability_margins.py`), and the "envelope" column is the
relative move of that same quantity between `OPENBLAS_NUM_THREADS` 1 and 4 --
the partial substitute for a second LAPACK that Stage A used.

| # | bar | test | measured on the test's own fixture | margin | envelope (1 vs 4 threads) | shape | action |
|---|---|---|---|---|---|---|---|
| 1 | `5e-12` (dR/dT/dJones) | `g1_oop_generator_reduces_to_the_inplane_path` | worst 2.70e-14 | **185x** | 3.0e-01 (plateau) | none | keep |
| 2 | `1.5e-10` (spectral) | same | worst 1.587e-12 | **94x** | 1.5e-02 -- the LARGEST move of any above-1e-12 quantity | S4-adjacent | keep, margin recorded (94x against a 1.5% envelope is ~3.8 decades of true margin) |
| 3 | exact byte equality | `g1_dispatch_floor_is_relative_and_bit_identical` | exact | exact | -- | a decision | keep |
| 4 | `1e-11` (quartic distance) | `g2_uniform_slab_dispersion_and_sum_of_roots` | 1.63e-14 / 1.34e-14 (fresh conical, symmetric / non-reciprocal), 2.4e-14 (normal) | ~600x | -- | none | keep |
| 5 | `1e-2` (`\|sum roots\|`) | same | **-0.0916247382** re-measured on the test's own fixture with an independently written quartic (0.156407 on a fresh tensor) | 9.2x / 15.6x | deterministic algebra | S1 in form only | keep -- the quantity is an algebraic number of the fixture, not a solver output.  The companion bars: `\|sum roots\| < 1e-12` at normal measures 5.0e-17 (symmetric) and 1.33e-15 (non-reciprocal), 750x; the transpose-blindness bar `1e-12` measures 0.0 and 8.9e-16 |
| 6 | `1e-3` (wrong-`k_t` distance) | same | 3.11e-02 | 31x | -- | none | keep |
| 7 | `1e-11` (Berreman R/T/Jones) | `g3_uniform_oop_slab_matches_berreman` (x9) | worst dT 1.102e-13 (1 thread), 2.215e-13 (4) | **91x / 45x** | 5.0e-01 (plateau) | S4 | keep, margin recorded -- ~1.7 decades |
| 8 | `1e-20` (order leak) | same | 4.60e-26 | 6 decades | 5.2e-01 (plateau) | none for THIS fixture, but strongly fixture-calibrated | keep, sensitivity recorded: a uniform slab at `px = 1.64 lam` with `n_orders = 4` leaks **7.9e-19** at conical (probe `v2a`), 7 decades more, because the far-field projection retains more orders.  The bar is a statement about `px = 0.9 lam`, `n_orders = 3` |
| 9 | `0.5` ladder ratio | `g3_berreman_convergence_is_two_sided_in_m` | 5.41e-09 -> 1.92e-11 -> 4.77e-14 (1 thread); 5.42e-09 -> 1.92e-11 -> 6.67e-14 (4) | 141x / 144x | 1.2e-04 / 5.9e-04 / 2.9e-01 | none -- the claim is a DROP, not a value | keep; **the docstring's M=5 value (2.0e-08) does not reproduce -- corrected to 5.41e-09, and the M=8 plateau's 29% cross-kernel move recorded** |
| 10 | `5e-4` (stripe per order) | `g4_oop_stripe_matches_the_1d_engines_per_order` (x2) | dT 1.352e-04 (normal), 1.414e-04 (oblique) | **3.7x / 3.5x** | **1.8e-09** | S1/S4 | keep, envelope + fixture sensitivity written into the docstring |
| 11 | `1e-20` (y-momentum) | same | 1.46e-26 | 6 decades | 2.0e-01 (plateau) | none | keep |
| 12 | `4e-4` / `4e-3` / `4e-3` | `g5_2d_reentrant_corner_cell_matches_the_fourier_oracles` | 8.17e-05 / 1.58e-03 / **2.07e-03** | 4.9x / 2.5x / **1.93x** | **< 4e-11** | S4 -- the tightest bar in the file | keep, envelope + re-derivation instruction written into the docstring |
| 13 | `1e-10` / `1e-5` / `1e4` | `g5_no_fourier_floor_two_sided` | 4.44e-16 / 6.24e-03 / 1.4e13 | 5.4 decades / 624x / 1.4e9 | 8.2e-01 (plateau) / 1.4e-11 | none | keep |
| 14 | `4e-3` / `4e-2` (vs hybrid) | same | 1.203e-04 / 1.140e-03 | 33x / 35x | 7e-13 | none | keep |
| 15 | `1e-10` (closure x6) | `g6_hermitian_oop_closes_at_three_depths...` | worst 3.67e-13 | **272x** | 1.4e-01 (plateau) | S4-adjacent | keep |
| 16 | `1 + 1e-12` (growth) | same | exactly 1.0 | exact | -- | a decision | keep |
| 17 | `lam_f.size == 2q` | same | 288/288 | -- | -- | **VACUOUS** (section 5, observation 2) | kept, and a NON-vacuous physical claim added: `min Re(lam_f) > -1e-10` (measured -1.51e-14 / -3.96e-15), `max Re(lam_b) < 1e-10` (measured 2.11e-14 / 8.85e-15) |
| 18 | `1e-3` (absorbed), monotone | `g6_lossy_oop_closes_below_one_and_absorbs...` | 2.4e-02 at 0.25 lam | 24x | deterministic | none | keep |
| 19 | `1e-4` (x3 controls) | `g7_oop_block_controls_move_the_berreman_residual` | drop 1.039e-03, negate 1.677e-03, transpose 1.671e-03 | **10.4x** | **1.2e-12** | S1 in form; the quantity is deterministic physics | keep, margin recorded |
| 20 | `1e-11` (`good`) | `g7_rotation_gauge_is_two_sided` | **1.195e-12** (the docstring quotes 2.38e-13, which is the dR alone; the assertion reads `max(dR, dJones)`) | **8.4x** | 5.7e-03 | S4 -- the tightest bar/envelope PAIR in the file | keep, corrected margin written into the docstring; 8.4x against a 0.6% envelope is ~3.2 decades of true margin |
| 21 | `1e-4` (`bad`), `1e7` ratio | same | 4.4895e-03, ratio 3.76e9 | 45x, 376x | 2.8e-13 | none | keep |
| 22 | `1e-11` (normal-incidence agreement) | same | 1.28e-15 | 4 decades | 5.9e-01 (plateau) | none | keep; the STATEMENT was corrected (section 2.2) |
| 23 | `1e-11` (split == single) | `g8a_one_oop_layer_equals_two_half_thickness_layers` | 1.11e-15 | 4.9 decades | 8.0e-01 (plateau) | none | keep |
| 24 | `1e-11` at M=8 | `g8b_uniform_oop_multilayer_matches_berreman_multilayer` | 5.45e-14 (oblique), 5.67e-14 (conical) | **180x** | 1.6e-01 / 3.3e-01 (plateau) | S4-adjacent | keep |
| 25 | -- | same | the docstring calls the claim "two-sided in M" and the test ran **only M=8** | -- | -- | claim stated, not asserted | **LADDER ADDED**: M=4 residual > 1e-9 (measured 3.91e-07 / 4.47e-08, envelope < 3e-08) and > 1e4 x the M=8 residual (measured 1.7e7 / 3.9e6) |
| 26 | `0.05` ratio, `1e-4`, `1e-2` | `g8c_layer_absorption_closes_on_a_mixed_oop_stack` | 4.53e-03, 4.18e-06, deficit 2.75e-02 | 11x / 24x / 2.75x | 1.1e-07 | none | keep |
| 27 | `1e-11` / `1e-3` | `g8c_lossless_mixed_stack_absorbs_nothing` | 1.67e-14 / 7.29e-05 | 600x / 14x | 1.9e-01 (plateau) / 2.8e-07 | none | keep |
| 28 | exception types / `match=` | `g9_*` (3) | decisions | -- | -- | none | keep |
| 29 | `0.0 < off < 1e-12 * max` | `g10_offplane_test_is_relative_not_strict` (anisotropic file) | the LOWER side is an exact-non-zero assertion on `cos(pi/2)` roundoff | -- | -- | **S5-adjacent** (flagged by the Stage-A verification, item 24, and reserved for this stage) | **RESTATED**: upper side kept; the "a strict `> 0` would misroute" half is now ENGINEERED -- a stray at `1e-3` of THIS build's measured floor must stay in-plane and byte-identical |

**S2 (pre-fix-referencing arm): none.**  Every fail-before constructs its
broken state in-process, by monkeypatching `TS._OOP_ROT_SIGN` or by building a
mis-assembled tensor.

**S3 (env-dependent precondition): none.**  The file's only environment
interaction is `os.environ.setdefault` of the three thread caps at import, which
FORCES a state rather than reading one, and there is **no `pytest.skip` and no
`xfail` anywhere in either file**.

**S5: one, item 17, now backed by a non-vacuous physical claim; plus item 29,
restated.**

**Overall envelope finding** (`results/v6_envelope_1_vs_4.txt`, 60 quantities
measured at both thread counts):

| quantity magnitude | count | largest relative move between 1 and 4 threads | where |
|---|---|---|---|
| `> 1e-6` | 22 | **2.78e-07** | G8c's lossless closure |
| `> 1e-9` | 26 | 1.20e-04 | G3's M=5 ladder point |
| `> 1e-11` | 28 | 5.94e-04 | G3's M=6 ladder point |
| `> 1e-12` | 31 | 1.52e-02 | G1's spectral distance (1.587e-12 vs 1.563e-12) |
| everything (incl. the roundoff plateau) | 60 | 8.2e-01 | G5's `n_orders` 3 -> 8 movement (4.4e-16 vs 2.4e-15) |

So the three tight bars that read a quantity ABOVE `1e-6` -- items 10 (`3.7x`),
12 (`1.93x`) and 19 (`10.4x`) -- sit 5 to 7 decades above the cross-kernel
spread of what they read, and are safe against a BLAS or a build.  Item 20
(`8.4x`) is the exception worth naming: it reads a `1.195e-12` quantity whose
cross-kernel move is `5.7e-03`, so its true margin is ~3 decades rather than
~10.  None of the four is safe against an intentional change to the basis or to
an oracle -- which the standards call the gate working, provided the
re-derivation is done rather than the bar loosened.  Item 12's docstring now
says so explicitly.

---

## 7. Task 6 -- cost

`v7_cost.py`, run with nothing else in flight.  The two arms ALTERNATE inside
one loop (so a machine drift cannot be read as a formulation difference) and
only ratios are reported; the in-plane arm is the SAME uniform tilted-uniaxial
cell with its four out-of-plane slots zeroed, so the arms differ only in the
formulation.  Peak allocation is a separate `tracemalloc` pass (its bookkeeping
perturbs the timings).  Region assembly + mode solve, single threaded.

| grid | M | dim in-plane -> out-of-plane | time ratio (min of reps) | time ratio (median) | peak in-plane | peak out-of-plane | peak ratio |
|---|---|---|---|---|---|---|---|
| (2,2) | 8 | 392 -> 784 | **1.744** | 1.744 | 28.3 MB | 85.7 MB | **3.022** |
| (3,3) | 8 | 882 -> 1764 | **1.328** | 1.322 | 142.7 MB | 433.4 MB | **3.036** |
| (3,3) | 6 | 450 -> 900 | **1.600** | 1.604 | 37.3 MB | 112.9 MB | **3.026** |

**CONFIRMED, and the build's own numbers reproduce exactly.**  T10's `(3,3)`
M=8 row claims 1.33x and 142.7 MB -> 433.4 MB (ratio 3.04); measured 1.328x and
142.7 MB -> 433.4 MB (3.036).  Its M=6 row claims 1.60x and 37.3 -> 112.9 MB
(3.03); measured 1.600x and 37.3 -> 112.9 MB (3.026).  The headline range
"1.33 - 2.03x" is a `(3,3)` statement: at `(2,2)` M=8 the ratio is 1.744, and
the trend (ratio falling with size) is as described -- the in-plane path pays a
QZ while the out-of-plane path pays a Cholesky-whitened standard eig on a matrix
twice the size.

---

## 8. Task 7 -- suites and lint

Command shape for every run (from the worktree, one process each):

```
PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest -q -p no:randomly [-W always] <files>
```

| run | result |
|---|---|
| `tests/unit/test_pmm2d_staggered_oop.py` + `tests/unit/test_pmm2d_staggered_anisotropic.py` (after this verification's changes) | **76 passed, 152.58 s** |
| slowest | 25.51 s `g5_no_fourier_floor_two_sided`, 10.30 s `g5_three_engines_agree`, 9.76 s `g8b_uniform_oop_multilayer`, 6.83 s `g6_hermitian_oop_closes`, 6.62 s `g3_multisegment`, 5.04 s `g5_2d_reentrant_corner` |
| the 12-file staggered / pure regression slice, `-W always` (the two above plus `test_v5_12_0_pmm2d_staggered`, `test_v5_21_pmm2d_staggered_oblique`, `test_staggered`, `test_audit_p1_staggered_guard`, `test_p2c_pmm2d_stack_cascade`, `test_p2t_pmm2d_tree_cascade`, `test_pmm2d_lossless_closure_two_sided`, `test_audit_s1_3_pmm2d_lossless_tripwire`, `test_v5_14_0_pmm2d_stack`, `test_niche_audit_w7_pmm`) | **361 passed, 35 warnings, 483.45 s (8:03)** |
| `PMM2DStackPure.solve` closure-tripwire firings in that run | **0** (27 distinct warning texts are listed -- 13 `DeprecationWarning`, 9 `UserWarning`, 5 `_EnergyWarning`; every `_EnergyWarning` is `pmm_jones_2d`'s, raised deliberately inside `test_niche_audit_w7_pmm`, and NONE is a `PMM2DStackPure.solve` closure violation) |
| `tests/unit/test_audit_dynameta_consumer_api_2.py` -- the heavy DynaMeta consumer gate on `PMM2DStackPure` `retain_internal` / per-order amplitudes / `layer_absorption`.  Its pure cells are SCALAR, so what it gates is the SYMMETRIC branch of the rewrite: `_modes_as_general` now hands every layer a 6-tuple and `_layer_flux` / `layer_amplitudes` read `Wb`, `Vb`, `lam_b` where they used to read one `lam` -- the "reduces term for term for a symmetric region" claim | **12 passed, 770.09 s (12:50)** -- slowest `b2_pure_amplitudes_vs_rcwa_conical` 289.6 s, `c3_pure_lossless_zero_and_contract` 211.6 s, `c3_pure_absorption_budget_and_hybrid_crossgate` 140.7 s, `b2_evanescent_kz_public_branch_and_contract` 106.0 s |
| `ruff check lumenairy/ tests/ validation/probe_verify_staggered_oop/` | **All checks passed** |

The new file stays inside the plan's budget after this verification's additions
(20 test functions / 35 cases, 90.6 s -> the same file now inside a 152.6 s
two-file run; no test above 26 s; grids `(2,2)` and `(3,3)`, `M <= 8`).

---

## 9. Changes committed on this branch by the verification

| commit | what |
|---|---|
| `5c0906c` | `validation/probe_verify_staggered_oop/` -- the first seven probes, their README and their recorded readings and logs |
| `4d1b97b` | three test RESTATEMENTS and three corrected measurements (section 6).  Tests only; no library file touched |
| `81e5be2` | `v2d` -- the H-gauge flip on the two CHIRAL fixtures as well, so both constants have a fail-before on both shapes |
| `a3b920a` | `v8` -- the measured identity `rot = +1` == negate the out-of-plane block (bit-identical on nine fixture x mount combinations) |
| `8798383` | a correction to this verification's OWN comment: the G3 ladder's M=8 four-thread reading is 6.67e-14, not the 4.72e-14 first written.  Caught by re-deriving the envelope table rather than re-reading the comment |
| `32db73c` | `results/v6_envelope_1_vs_4.txt` -- the 60-quantity cross-kernel envelope the durability table's envelope column is read from |
| (this file) | the verification report |

**No library file was touched.  No merge, push, tag or version bump.**  The
temporary worktree `C:/tmp/lum_aniso_pre` created for task 1 was removed with
`git worktree remove` at the end.

---

## 10. What could not be verified

* **A CHIRAL patterned out-of-plane cell at CONICAL incidence against an
  INDEPENDENT 1-D oracle.**  `pmm_jones_1d_conical_tensor` deliberately
  REFUSES a patterned out-of-plane cell at conical incidence (it raises,
  citing `AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12`), and
  `rcwa_jones_1d_segments` / `pmm_jones_1d_segments` are planar-incidence only.
  So the conical arm of the chiral fixtures is graded against `rcwa_jones_2d`
  and `pmm_jones_2d`, which both route their tensor layer through
  `rcwa._core._layer_eigenmodes_tensor` -- the build's own open item 1.  The
  conical claim is therefore BOUNDED (`1.34e-04` against an oracle spread of
  `6.1e-05 .. 9.1e-05`), not machine-precision.  What would settle it is a
  conical 1-D out-of-plane engine, or a staircase-refined staggered cell.
* **Cross-platform / cross-LAPACK.**  Everything here is one machine.  The
  substitute is the `OPENBLAS_NUM_THREADS` 1 vs 4 sweep of section 6, which
  exercises different OpenBLAS kernels and reduction orders on 60 quantities.
  It is not a second wheel.
* **The JAX twin and the full library suite.**  The staggered path is
  NumPy-only and nothing in Stage B changes that; the suite list is the plan's.
* **`M > 8` and grids above `(3,3)`.**  The operational budget caps both, so
  GATE 0's headline M=9 self-convergence number (`4.80e-07`) could not be
  reproduced at M=9.  The M-ladder self-convergence was re-measured to M=8 on
  the 1-D chiral fixture (`8.91e-06 -> 8.28e-06`, i.e. `6.3e-07` of self-move)
  and to M=7 on the 2-D chiral cell (`2.28e-05` / `1.51e-04` / `4.27e-05`
  self-move at normal / oblique / conical), which is the same statement one
  degree lower.
* **The normal-incidence involution accelerator** (build open item 3) is not
  integrated and was not exercised.

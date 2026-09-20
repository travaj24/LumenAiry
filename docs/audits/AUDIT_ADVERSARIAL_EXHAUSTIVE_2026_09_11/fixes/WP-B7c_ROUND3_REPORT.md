# WP-B7c round 3 -- the bar's margin is a reading of the z ladder, the fold ring's two fidelity populations stop being disjoint at sixteen optics, and the fallback route's accuracy IS ordered by a reading already in the diagnostics

Wave 5, round 3 of item C, answering the follow-ups
`fixes/VERIFY_WP-B7c_ROUND2.md` left: defects **E1**, **E4**, **E5**, **E6**,
**E7** and claims **1b / 1c / 5 / 6**.  Branch `fix/wp-b7c-round3` off
`wave5/audit-leftovers` (`dcaa21f0`, which already carries round 2 `fcb9ed12`
/ `e8ddec9d` and the verification `35db475c`).  Worktree `C:/tmp/lum_mb3`,
with three more used read-only or as scratch: `C:/tmp/lum_mb3_pre` and
`C:/tmp/lum_mb3_post` (`git archive` exports, for bit identity) and
`C:/tmp/lum_mb3_mut` (the mutation matrix).

| build | interpreter | numpy | scipy | tree |
|---|---|---|---|---|
| **win** | Windows py3.14.6 | 2.4.4 | 1.17.1 | `C:/tmp/lum_mb3` |
| **wsl** | WSL py3.12.3 | 2.4.6 | 1.17.1 | `/mnt/c/tmp/lum_mb3` |

Probes and JSON: `validation/probe_wp_b7c_round3/`.  New decision tests:
`tests/unit/test_wp_b7c_round3.py` (8 ids).

---

## 0. What moved, per item

| item | what the verification asked | what round 3 did | verdict |
|---|---|---|---|
| **R3-1** (E1, claims 1c / 5) | re-derive the bar's margins two-sided on >= 12 optics and >= 300 planes; state what 1.06 costs and what 1.04 / 1.08 / a spread-derived bar cost | 1304 oracle-scored planes, 16 prescriptions, 21 (prescription, grid) pairs, 394 on the completion route | **the margin is not a property of the quantity** -- it falls 1.040x -> 1.012x -> 1.00047x as the SAME optics' ladders are refined 30x then 10x.  The bar is KEPT at 1.06, which is the derived centre (1.0600253) of its own gap to three figures, and its derivation is now a COST table |
| **R3-2** (E4) | the Pearcey route mislabels the branch sum's reading -- fix the label and pin it | label and a new machine-readable scope; pinned by a test that reads the number back and requires it to BE the branch sum's, to the bit | **fixed and pinned** |
| **R3-3** (E5) | the two renders' pixel centres do not span the same window -- align them, state the convention, re-run bit identity before/after | the convention is stated and given ONE definition; the hull-aligned alternative was BUILT and MEASURED and is strictly worse | **convention stated, single definition shipped, alternative measured and rejected** |
| **R3-4** (E7) | characterise the fallback route's failure mode; add a third reading that orders it if one exists, else say the field is unarbitrated | both: the LOSS sides already reported DO order it (254 flagged, 0 false alarms, 9 misses of 263), and the diagnostics now carry the route's scope | **a third reading exists and is named; 13 of 476 planes remain unordered** |
| **R3-5** (claims 1b / 6) | dated CORRECTED paragraphs for the round-2 numbers that did not reproduce | six in `WP-B7c_ROUND2_REPORT.md`, one in `caustic_fold_truth.py`, and one that corrects the VERIFICATION instead | **done** (section 11) |
| **R3-6** | every new pin must FAIL (not skip) under the verifier's 13 mutations plus mine, on both builds | 18 mutations against the whole five-file gate | **18 of 18 caught** (section 9) |

**Behaviour: nothing moves.**  Archive to archive over a 157-case matrix on
seventeen optics -- **116 identical, 0 MOVED, 0 newly refused, 0 newly
returned, 41 refused on both** (section 6).

---

## 1. The population

Sixteen distinct prescriptions, twenty-one (prescription, grid) pairs, **1304
oracle-scored planes**.  It CONTAINS both earlier populations: the eight
optics the bar was derived on, the five the verification added, and `P` --
the f/1.2 optic round 2 EXCLUDED because the shared oracle's Debye `J0` form
does not hold at `y_max/z = 0.45`, and which is scored here because the
angular spectrum has no such ceiling.  Three are new to the campaign.

| optic | provenance | NA | grid | planes | fold ring | fallback |
|---|---|---|---|---|---|---|
| `A` | WP-B7c round 2 | 0.143 | 512 x 3.00 um | 59 | 23 | 36 |
| `AS` | **ROUND 3 (new)** | 0.198 | 512 x 2.60 um | 59 | 27 | 32 |
| `AS_alt` | **ROUND 3 (new)** | 0.198 | 640 x 2.15 um | 59 | 27 | 32 |
| `C` | VERIFY round 2 | 0.033 | 512 x 4.00 um | 25 | 0 | 25 |
| `F_alt` | VERIFY-WP-B7c | 0.211 | 640 x 1.40 um | 59 | 26 | 33 |
| `G` | WP-B7c round 2 | 0.301 | 768 x 1.20 um | 77 | 26 | 51 |
| `HN` | **ROUND 3 (new)** | **0.412** | 640 x 2.00 um | 77 | 26 | 51 |
| `K` | WP-B7c round 2 | 0.202 | 512 x 2.00 um | 59 | 23 | 36 |
| `M` | VERIFY-WP-B7c | 0.184 | 512 x 3.60 um | 68 | 5 | 63 |
| `MC` | **ROUND 3 (new)** | 0.164 | 512 x 1.90 um | 59 | 26 | 33 |
| `P` | WP-B7c round 2 (**EXCLUDED there**) | 0.365 | 512 x 1.80 um | 59 | 5 | 54 |
| `Q` | VERIFY-WP-B7c | 0.108 | 512 x 2.60 um | 59 | 25 | 34 |
| `S` | VERIFY-WP-B7c | 0.129 | 512 x 1.50 um | 59 | 5 | 54 |
| `V` | VERIFY-B7b (the fixture the bar is argued on) | 0.216 | 512 x 2.20 um | 59 | 24 | 35 |
| `V_alt` | the same, second grid | 0.216 | 400 x 2.80 um | 59 | 2 | 57 |
| `W` | VERIFY round 2 | 0.107 | 512 x 2.40 um | 59 | 22 | 37 |
| `W_alt` | the same, second grid | 0.107 | 640 x 1.90 um | 59 | 23 | 36 |
| `X` | VERIFY round 2 | 0.387 | 640 x 1.40 um | 77 | 26 | 51 |
| `X_alt` | the same, second grid | 0.387 | 512 x 1.75 um | 77 | 26 | 51 |
| `Y` | VERIFY round 2 | 0.197 | 512 x 1.80 um | 59 | 27 | 32 |
| `Z` | VERIFY round 2 | 0.119 | 512 x 1.60 um | 77 | 0 | 77 |
| **total** | | | | **1304** | **394** | **910** |

The three new ones, and why each:

* **`AS`** -- a genuinely ASPHERIC singlet: an even-aspheric departure
  (`aspheric_coeffs={4: -1.00e7, 6: +2.00e13}`) on the convex surface of an
  N-BK7 plano-convex, +3.36 um at the clear semi-aperture (4.3 waves at
  780 nm).  Neither earlier round used `aspheric_coeffs` at all -- round 2's
  `K` and the verification's `Y` are CONIC, a two-parameter deformation of a
  sphere that leaves the ray map analytic in `h^2`.  The two terms have
  opposite signs, which puts an interior STATIONARY POINT in the focal locus
  (this round's geometry probe reads one turn in `f(h)` where every spherical
  and conic optic in the campaign reads zero), so the landing map has a
  second interior caustic and the fold ring is not one monotone branch in
  launch height.  That is the geometry a completion built on ONE interior
  turning point is least likely to be right on -- and it is where NINE of the
  thirteen planes nothing orders (section 7) live.
* **`MC`** -- a CONCAVE-FIRST positive meniscus (N-SK16, R = -4.5 / -1.5 mm).
  Round 2's `M` is a positive meniscus run CONVEX-first; bending it the other
  way puts the steep surface last.
* **`HN`** -- a plano-convex N-LASF9 at **NA 0.412**, above the
  verification's `X` (0.387) and round 2's `G` (0.301).  It supplies the
  worst RETURNED fold-ring field in the study.

**Ladders are derived, not copied.**  `r3geom.py` measures each optic's
paraxial and marginal foci and the z window in which the meridional landing
map has an interior turning point; `r3ladder.py` is a function of that and of
nothing else.  Then two REFINEMENT passes (`r3fine.py`) bracket the bar per
optic from the coarser scan and re-scan between the straddling planes at
20-30x the step, twice.  That is what section 3 is about.

**The oracle.**  The three propagators and the scoring helpers are imported
VERBATIM from the round-2 verifier's own oracle
(`validation/probe_verify_b7c_round2/vroracle.py`) -- independently written,
convergence-controlled there, and deliberately not re-derived a third time.
Round 3 adds an EVEN-ASPHERIC term to the ray trace, with the control that it
is bit-inert on a spherical or conic prescription (`0.0` on every column, on
both `V` and the conic `K`), and the index control against
`lumenairy.glass.get_glass_index` reads **2.2e-16** at worst over ten glasses
and seven wavelengths.  Every fidelity is taken against the band-limited
ANGULAR SPECTRUM, at full radius; section 8 bounds it against an exact
azimuthal quadrature.

---

## 2. The fold ring at the shipped bar -- and the claim that does not survive

| population | n | continuity | oracle fidelity |
|---|---|---|---|
| **RETURNED** | 250 | 0.87237 .. 1.05951 | **0.9421** .. 0.9985 |
| **REFUSED** | 144 | 1.06431 .. 3.99879 | 0.0018 .. **0.9520** |

**The two FIDELITY populations OVERLAP.**  0.9421 < 0.9520.  That is round 2's
strongest statement -- "the two fidelity populations do not overlap, and that
statement chooses no accept criterion: the split is the guard's own" -- and it
is the one claim that had survived the round-2 verification, which narrowed
the separation from (0.9593, 0.9302) to (0.9747, 0.9578) on 124 planes and
warned it was narrowing.  At 394 fold-ring planes over sixteen optics it
closes.

| | optic | z [um] | C | fidelity | power / oracle |
|---|---|---|---|---|---|
| worst RETURNED | `HN` (NA 0.41) | 2069.329 | 1.01329 | **0.9421** | 1.2662 |
| | `HN` | 1908.878 | 1.00164 | 0.9499 | 1.2395 |
| | `HN` | 1857.119 | 1.00066 | 0.9505 | 1.2463 |
| best REFUSED | `W_alt` | 4928.969 | 1.08123 | **0.9520** | 1.1350 |
| | `G` | 967.785 | 1.07291 | 0.9497 | 1.1062 |
| | `F_alt` | 1074.277 | 1.06431 | 0.9466 | 1.0945 |

The overlap is not a knife edge between two neighbouring planes: `HN`'s WHOLE
fold ring is returned at fidelity 0.9421-0.9564 with the reading between
0.8724 and 1.0133 and the returned power **1.24x-1.27x** the oracle's.  On the
fastest optic in the campaign the arbiter sees nothing wrong and the field
carries a quarter more energy than it should.  That is a miss class, not a
boundary case, and it is invisible to every optic slower than NA 0.4.

Per optic:

| optic | n | RETURNED C / fidelity | REFUSED C / fidelity |
|---|---|---|---|
| `A` | 23 | 0.9939..1.0129 / 0.9890..0.9936 | 3.4045..3.9959 / 0.0771..0.3093 |
| `AS` | 27 | 0.9414..1.0164 / 0.9789..0.9900 | 2.0990..3.9829 / 0.0549..0.5292 |
| `AS_alt` | 27 | 0.9874..1.0012 / 0.9820..0.9925 | 2.3839..3.9844 / 0.0499..0.4601 |
| `F_alt` | 26 | 0.9944..1.0508 / 0.9537..0.9890 | 1.0643..3.9832 / 0.0721..0.9466 |
| `G` | 26 | 0.9995..1.0595 / 0.9517..0.9985 | 1.0729..3.9557 / 0.1121..0.9497 |
| **`HN`** | 26 | 0.8724..1.0133 / **0.9421..0.9564** | 2.9820..3.9981 / 0.0024..0.2939 |
| `K` | 23 | 0.9786..1.0064 / 0.9855..0.9929 | 3.7714..3.9985 / 0.0096..0.1412 |
| `M` | 5 | 0.9897..0.9988 / 0.9756..0.9845 | -- |
| `MC` | 26 | 0.9978..1.0089 / 0.9928..0.9970 | 1.6178..3.9745 / 0.0979..0.7328 |
| `P` | 5 | 0.9996..1.0005 / 0.9954..0.9974 | -- |
| `Q` | 25 | 0.9933..1.0098 / 0.9703..0.9931 | 3.8164..3.9986 / 0.1872..0.2100 |
| `S` | 5 | 0.9961..1.0065 / 0.9773..0.9860 | -- |
| `V` | 24 | 0.9840..1.0069 / 0.9862..0.9924 | 2.0399..3.9980 / 0.1693..0.6635 |
| `V_alt` | 2 | 0.9997..1.0063 / 0.9847..0.9870 | -- |
| `W` | 22 | 0.9750..1.0092 / 0.9843..0.9918 | 3.7000..3.9960 / 0.0095..0.1227 |
| `W_alt` | 23 | 0.9909..1.0089 / 0.9847..0.9922 | 1.0812..3.9988 / 0.0071..0.9520 |
| `X` | 26 | 0.9942..1.0193 / 0.9836..0.9962 | 1.1638..3.9752 / 0.0592..0.8968 |
| `X_alt` | 26 | 0.9949..1.0011 / 0.9919..0.9956 | 1.0807..3.9701 / 0.0685..0.9437 |
| `Y` | 27 | 0.9917..1.0206 / 0.9789..0.9944 | 1.3991..3.9885 / 0.0018..0.7832 |

`Z` and `C` produce no fold-ring plane at all -- the well-corrected doublet
and the slow control never reach the completion route at a sane grid, which
round 2's verification also found and which is worth keeping in view: the
route the bar is derived on is not the route most optics take.

---

## 3. The margin is a reading of the z ladder

The round-2 verification's section 2.3 found that near a fold onset the
reading is **not smooth in z**: 1.0002 -> 1.1810 -> 0.9989 over 20 nm of
defocus, with the FIELD moving with it.  A ladder cannot see an excursion
narrower than its own step, so refining the ladder keeps finding readings
closer to the bar.  Measured by scanning the SAME sixteen optics three times,
each ladder 10-30x finer than the last:

| ladder | planes | fold-ring planes | fold-ring gap | fold margin above | fold margin below | all-planes gap | all margin above | all margin below |
|---|---|---|---|---|---|---|---|---|
| band + tail (~20 um steps) | 468 | 147 | 1.37061 | 1.03996 | 1.31795 | 1.01612 | 1.01306 | 1.00302 |
| + fine pass 1 (~1 um) | 882 | 259 | 1.03172 | 1.01193 | 1.01956 | 1.00378 | 1.00076 | 1.00302 |
| + fine pass 2 (~0.1 um) | **1304** | **394** | **1.00453** | **1.00047** | **1.00406** | **1.00098** | **1.00047** | **1.00051** |

Round 2's "**1.0683x**, a 7 % gap, measured on eight optics and two builds"
and the verification's "**1.0376x** on 124 fold-ring planes" are both readings
of their ladders, and so is 1.0045x.  There is no sign of a floor: each 10-30x
refinement roughly halves the logarithm of the gap.  **No margin statement can
be made about this bar at all**, which is why the derivation in the constant
is now a cost and not a gap.

The two-sided margin at 1.06 is therefore **1.00047x** -- 0.047 %, far below
the 1 % that would call for re-centring.  The geometric centre of the gap the
bar sits in, over all 1304 planes, is **1.0600253**: 1.06 to three figures IS
the derived centre, so the constant does not move.  (The fold-ring-only centre
is 1.0619044; moving there changes no fold-ring decision at all and returns
one more wrong field over all planes.)

---

## 4. What a CONVERGED render actually reads

The argument that makes this quantity usable is that it has a FIXED
reference.  Round 2 stated it as "**1 exactly**, on any optic, at any plane,
at any grid"; the round-2 verification measured 0.999529 .. 1.000207 on eight
planes of one slow optic and attributed the residue to the two renders not
spanning the same window (E5).  Round 3 measures the distribution, on planes
chosen by a GEOMETRIC criterion -- at least a quarter of the paraxial focal
distance short of the interior fold, so no reading selects the planes its own
spread is then measured on (`r3control.py`, `control_win.json`):

| population | n | reading | rms deviation from 1 |
|---|---|---|---|
| 102 planes, all 17 optics, far from every caustic | 102 | **0.9994103 .. 1.0004414** | **1.48e-4** |
| the same, worst single plane | | 5.90e-4 from 1 | |
| HEALTHY FOLD planes (fidelity >= 0.99, power within 2 % of the oracle) | 82 | 0.97865 .. 1.00892 | 4.09e-3 |

Two ladders identify where the residue comes from:

| GRID ladder (window held, `N` and `dx` both scaled) | N = 128 | 192 | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|
| deviation from 1 | 1.22e-4 | 3.41e-4 | 2.41e-4 | 1.36e-4 | 1.18e-4 | 7.91e-5 | **5.62e-5** |

| WINDOW ladder (`dx` held at 4.00 um, `N` grows so the window grows) | N = 256 | 384 | 512 | 640 | 896 |
|---|---|---|---|---|---|
| deviation from 1 | **1.44e-3** | 1.46e-4 | 1.18e-4 | 1.18e-4 | 1.18e-4 |

The window ladder is the measurement of **E5's mechanism**.  On a window too
small to contain the field the reading is 1.4e-3 from 1; as soon as the window
holds the field it drops by 12x and then does not move at all (1.18e-4, bit
for bit, for N = 512, 640 and 896).  That is exactly what a Voronoi-hull
mismatch at the window EDGE predicts, and it is 60x below the bar's distance
to 1.

**So the reference is real but it is not exact, and it does not hold where the
guard works.**  On healthy FOLD planes the same reading spreads by up to
2.1e-2 -- 36x the converged spread -- because a collapsing ring is precisely
where a point-sampled estimator has the most variance.  A bar derived from the
converged spread is therefore unusable, and section 5 measures what it costs.

---

## 5. The bar-cost table

Reported by FIDELITY BAND rather than at one accept criterion, because the
accept criterion is a maintainer decision (section 10).  "False refusal" =
refused with fidelity above the criterion; "miss" = returned with fidelity
below it.  `s_conv` = 5.897e-4, the converged population's worst deviation
from 1 (section 4).

### 5.1 FOLD RING (n = 394)

| bar | refused | returned | margin above | margin below | false refusals @ 0.95 | misses @ 0.95 | false refusals @ 0.883 | misses @ 0.883 |
|---|---|---|---|---|---|---|---|---|
| 1.0018 (`1 + 3 s_conv`) | 207 | 187 | 1.00000 | 1.00001 | **63** | 1 | 85 | 0 |
| 1.0059 (`1 + 10 s_conv`) | 173 | 221 | 1.00019 | 1.00001 | **29** | 1 | 51 | 0 |
| 1.0177 (`1 + 30 s_conv`) | 152 | 242 | 1.00122 | 1.00155 | **9** | 2 | 30 | 0 |
| 1.04 | 148 | 246 | 1.00787 | 1.00256 | 5 | 2 | 26 | 0 |
| **1.06 (shipped)** | 144 | 250 | 1.00047 | 1.00406 | **1** | **2** | 22 | 0 |
| 1.0600253 (derived centre, all planes) | 144 | 250 | 1.00049 | 1.00404 | 1 | 2 | 22 | 0 |
| 1.0619 (derived centre, fold ring) | 144 | 250 | 1.00226 | 1.00226 | 1 | 2 | 22 | 0 |
| 1.08 | 140 | 254 | 1.00323 | 1.00068 | 1 | 6 | 18 | 0 |

### 5.2 ALL PLANES (n = 1304)

| bar | refused | returned | margin above | margin below | false refusals @ 0.95 | misses @ 0.95 | false refusals @ 0.883 | misses @ 0.883 |
|---|---|---|---|---|---|---|---|---|
| 1.0018 (`1 + 3 s_conv`) | 692 | 612 | 1.00000 | 1.00001 | 76 | 226 | 141 | 133 |
| 1.0059 (`1 + 10 s_conv`) | 651 | 653 | 1.00005 | 1.00001 | 36 | 227 | 100 | 133 |
| 1.0177 (`1 + 30 s_conv`) | 622 | 682 | 1.00122 | 1.00089 | 14 | 234 | 73 | 135 |
| 1.04 | 603 | 701 | 1.00296 | 1.00044 | 7 | 246 | 58 | 139 |
| **1.06 (shipped)** | 578 | 726 | 1.00047 | 1.00051 | **1** | 265 | 46 | 152 |
| 1.0600253 (derived centre, all planes) | 578 | 726 | 1.00049 | 1.00049 | 1 | 265 | 46 | 152 |
| 1.0619 (derived centre, fold ring) | 577 | 727 | 1.00128 | 1.00023 | 1 | 266 | 46 | 153 |
| 1.08 | 560 | 744 | 1.00323 | 1.00068 | 1 | 283 | 40 | 164 |

Reading the table:

* **1.06 is at the knee.**  On the fold ring it costs ONE false refusal and
  TWO misses at the 0.95 criterion.  Nothing measured does better on both
  axes: 1.04 costs five times the false refusals for the same misses, 1.08
  costs three times the misses for the same false refusals.
* **the spread-derived bars are unusable**, and the reason is section 4: they
  are tolerances on a reference that is 36x tighter than the reading's own
  spread where the guard operates.  `1 + 10 s_conv` refuses 29 fold-ring
  planes whose fields score above 0.95 and buys one miss back.
* **the all-planes miss column is not this arm's failure.**  265 of the 1304
  planes are returned below fidelity 0.95, and 263 of them are on the FALLBACK
  route, where the module has already warned that the completion does not
  apply and where the dominant error is the dark tail this arm cannot see.
  Section 7 is about those.

---

## 6. Bit identity, archive to archive

`r3bitid.py`: **157 cases** -- for each of the seventeen optics a healthy fold
plane, a blown-up plane, a fallback, a gap-edge plane, the branch sum alone,
the exit vertex (`output_plane_distance = 0`), `caustic_band='plain'`,
`ray_subsample=4` and a complex64 input, plus four second-grid cases.  Both
trees are `git archive` exports (no `.git`, no working-tree state, no stale
`__pycache__`) differing in exactly two files; a CHILD process per tree with
`cwd` and `PYTHONPATH` set to it, `lumenairy.__file__` ASSERTED under it,
`LUMENAIRY_MEM_BUDGET_MB=2048` PINNED (VERIFY-WP-B12 D-4: unpinned digests are
chunking-dependent), SHA-256 over `ndarray.tobytes()`.

`dcaa21f0` against the round-3 library: **116 identical, 0 MOVED, 0 newly
refused, 0 newly returned, 41 refused on both** (`bitid_win.json`).  Both
trees report `bar = 1.06`.

That is the point of shipping the NESTED convention rather than the
hull-aligned one (section 3 of the commit message, and
`_HALF_PITCH_CENTRE_OFFSET`): `_render_centre_origin` at the shipped offset is
the same arithmetic the straight-line code did, and `half_pitch_centres(N,
dx)` is the expression the completion inlined, so the refactor to ONE
definition is free.

### 6.1 The alternative E5 asks for, measured

E5 asks for the fine lattice to be offset by half a fine pixel so the two
Voronoi HULLS coincide exactly.  It was built and run.  It is strictly worse,
and the module's own shipped gate refuses it:

| | nested (shipped) | hull-aligned |
|---|---|---|
| `V` z = 1768 um, the blow-up plane | **3.998** | **7343.9** |
| `V` z = 1761 um, branch sum | 1.0636 | 1.0180 |
| `V` z = 1761 um, completion | 1.0021 | 0.9797 |
| that plane's oracle fidelity | 0.9879 | 0.9879 (the FIELD does not move) |
| `test_b7c2_a_converged_render_reads_one_and_a_blown_up_one_reads_four` | passes | **fails** |
| `test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum` | passes | **skips** (the build stops straddling the bar) |

The mechanism is not subtle.  With NESTED lattices a mapped triangle that
catches a coarse centre catches the coincident fine centre too, so
`p_out(dx/2) >= p_out(dx)/4` always and the reading is BOUNDED by 4 in the
collapse limit -- which is the `~4 per halving` identity VERIFY-WP-B7c
measured by hand on four optics and which the whole diagnosis rests on.  With
no shared sample points a collapsed triangle can catch a coarse centre and no
fine centre at all, and the bound is gone.  Nesting also makes the two
estimators positively correlated, which is why the ratio's noise (1.5e-4 rms,
section 4) is far below either estimator's own.

What nesting COSTS is the hull mismatch, and section 4's window ladder
measures it: **1.4e-3** on a window too small to hold the field, **1.2e-4** on
every larger one.  That is 60x below the bar's distance to 1, and it is why
"a converged render reads 1 **exactly**" is not what the code computes.  Both
facts are now written where the constant is.

What round 3 DOES ship for E5 is the seam: the branch sum's fine render and
the completion's half-pitch fill were placed by two COPIES of the same
expression, and they now both call `half_pitch_centres`.  Nothing compared
them, so a divergence there would have been silent -- the reading would have
been a ratio of two integrals taken on different geometry, which is exactly
the defect E5 names.

---

## 7. The fallback route (R3-4 / E7)

E7: "on the fallback route a wrong field is returned silently with both arms
at their nominal values", with one plane at oracle fidelity 0.4639.

The population that question is about is the fallback planes the SHIPPED bars
RETURN -- **476** of the 910 scored (the other 434 are refused, and crediting a
candidate with ordering those would be crediting it with cases no caller
sees).  Of the 476, **263 score below fidelity 0.95** and 152 below 0.883; the
range is **0.5358 .. 0.9999**.  Round 2 quoted "returned down to 0.764" from a
population a fifth the size.

### 7.1 Does anything order it?  Yes -- two readings already reported

`r3fallback.py` scores ten scalar candidates by Spearman rank correlation
against the oracle fidelity and by the best separation any threshold on them
achieves, plus four derived boolean ones.

| reading the SHIPPED library already reports | flagged | of which wrong | of which RIGHT | not flagged | of which wrong |
|---|---|---|---|---|---|
| continuity LOSS arm (`pixel_continuity < 1/1.06`) | 192 | **192** | **0** | 284 | 71 |
| launched-power LOSS arm (`bracket < 0.5`) | 47 | **47** | **0** | 429 | 216 |
| either loss arm | 192 | 192 | 0 | 284 | 71 |
| either loss arm OR `reason != 'no_fold'` | 267 | 250 | 17 | 209 | **13** |
| a TIGHTER power loss bar, `bracket < 0.889` | 254 | **254** | **0** | 222 | **9** |

and the scalar candidates, for completeness:

| candidate | Spearman vs fidelity | best threshold at 0.95 (balanced accuracy, false refusals, misses) |
|---|---|---|
| `power_ratio` | **0.846** | accept above 0.889 -- **0.983**, 0 false, 9 misses |
| `multibranch_power_ratio_bracketed` | **0.837** | accept above 0.889 -- **0.983**, 0 false, 9 misses |
| `power_ratio_triangles` | 0.821 | accept above 0.891 -- 0.983, 0 false, 9 misses |
| `go_blur_um` (the geometrical spot radius) | 0.807 | accept above 135.1 um -- 0.862, 57 false, 2 misses |
| `go_over_airy` (that over `lambda z / D`) | 0.723 | accept above 11.24 -- 0.796, 5 false, 101 misses |
| `pixel_continuity` | 0.591 | accept above 0.9875 -- 0.911, 3 false, 43 misses |
| `edge_fraction` | 0.370 | -- 0.592, 174 false |
| `n_branch_max` | -0.260 | -- 0.573 |
| `airy_um` | -0.175 | -- 0.691 |

**So the answer to "add a third reading that orders it if one exists" is that
one exists and is already there.**  It is the LOSS side of the launched-power
bracket, and the reason nobody noticed is that its bar (0.5) is set for the
FOLD-RING route, where the completion replaces the dark side and a dark-side
deficit does not reach the caller.  On the FALLBACK route that justification
is false by construction: the completion declined, so the branch sum's dark
deficit reaches the caller verbatim.  The same is true of the continuity loss
arm, which flags 192 planes on this population with **zero false alarms**.

By fallback reason, which orders it too:

| reason | n | fidelity |
|---|---|---|
| `no_fold` | 271 | 0.5358 .. 0.9999, median 0.9937 |
| `fold Airy scale under-resolved by the grid` | 69 | 0.7682 .. 0.9768, median 0.8917 |
| `zeta_nonlinear` | 65 | 0.7369 .. 0.9411, median 0.8783 |
| `bad_kappa` | 50 | 0.7397 .. 0.9409, median 0.8648 |
| `too few bright-band pixels for the fit` | 21 | 0.8369 .. 0.9259, median 0.8643 |

Every reason except `no_fold` has its whole returned population below 0.95.

### 7.2 What is left

After both loss arms and the reason, **13 of 476 returned planes** are wrong
with nothing to see it:

| optic | z [um] | fidelity | `pixel_continuity` | bracket |
|---|---|---|---|---|
| `C` (the slow control) | 22739.95 | **0.5358** | 0.98722 | 0.9883 |
| `AS_alt` | 4552.2 | 0.8118 | 1.03661 | 0.6940 |
| `AS` | 4555.1 | 0.8190 | 1.04933 | 0.7362 |
| `AS_alt` | 4552.9 | 0.8192 | 0.98326 | 0.6904 |
| `AS` | 4555.9 | 0.8239 | 1.01581 | 0.7240 |
| `AS_alt` | 4553.7 | 0.8287 | 0.96378 | 0.6859 |
| `AS` | 4556.6 | 0.8359 | 0.98220 | 0.7284 |
| `AS_alt` | 4576.1 | 0.9326 | 1.04315 | 1.0144 |
| `AS_alt` | 4583.5 | 0.9447 | 1.04553 | 1.0527 |
| `AS` | 4583.5 | 0.9467 | 1.03107 | 1.0134 |
| `MC` | 3577.6 | 0.9485 | 1.04633 | 1.0473 |
| `HN` | 1554.1 | 0.9499 | 1.00047 | 0.9993 |
| `HN` | 2510.3 | 0.9499 | 0.99995 | 0.9986 |

`C` z = 22.74 mm is E7's case, reproduced (the verification found 0.4639 at
z = 22.80 mm): the slow control a hundred microns from its own focus, where
geometrical optics has no answer at all and every reading is nominal because
the quadrature IS converged -- on a field that is simply the wrong model.
**Nine of the remaining twelve are on the new ASPHERE**, on both of its grids,
at its non-monotone focal locus -- the geometry that optic was added to probe,
and the only place in the study where the completion declines at a plane whose
readings are all nominal and whose field is 0.81-0.95.  The last three are the
NA 0.41 plano-convex (twice) and the concave-first meniscus, all at 0.9485 or
above, i.e. at the criterion rather than past it.

### 7.3 What shipped, and what is a maintainer decision

**Shipped:** the diagnostics now say what the reading buys on THIS route.
`pixel_continuity_of` names the field the number was measured on;
`pixel_continuity_scope` is a new enum -- `'returned_field'`,
`'returned_field_quadrature_only'`, `'underlying_branch_sum'` -- and
`pixel_continuity_scope_note` carries the same statement in words including
the confusion numbers above.  An unknown scope raises inside `_arbitrate`
rather than reaching a caller with the key absent.

**Not shipped, recommended:** making either loss arm REFUSE on the fallback
route.  On this population the continuity loss arm would refuse 192 planes and
every one of them is wrong; a launched-power loss bar at 0.889 would refuse
254 with none false and only 9 wrong ones left.  That is a behaviour change
with a Migration cost -- every caller whose fallback plane reads below the bar
starts getting an exception -- and it is the maintainer's to take.  The
measurement is here; section 10 states the recommendation.

---

## 8. The oracle floor at FULL radius (R3-1's scoring requirement, and E6)

E6: round 2's "Debye vs exact" column substituted the exact azimuth ONLY
inside the 99.95 %-energy core, outside which the two compared fields are
identical by construction -- while the dropped quadratic term grows with
radius.  The fitted rule `rel L2 ~ 0.37 eps` "holding over three decades of
`eps`" is fitted to that core-confined number.

Round 3 re-measures it with the substitution everywhere, and takes every
comparison POINTWISE at a radius-stratified, importance-weighted sample of the
output grid's own pixels (480 per plane, 24 equal-radius bins, weight
`N_bin / n_drawn`), so neither arm is interpolated and the far dark tail is
not read as if it were most of the field.

| optic | plane [um] | `y_max/z` | `eps` | `J0` vs exact, FULL radius | ... as FIDELITY | `J0` vs exact, CORE ONLY | **ASM vs exact** | ... as FIDELITY |
|---|---|---|---|---|---|---|---|---|
| `Q` (round 2's own row) | 5680.00 | 0.108 | 8.5e-04 | **0.0563** | 0.99842 | 0.00537 | **0.00188** | 0.999998 |
| `W` | 4900.00 | 0.109 | 1.5e-03 | **0.0952** | 0.99551 | 0.01123 | **0.00470** | 0.999989 |
| `S` | 3214.78 | 0.133 | 1.4e-03 | **0.0833** | 0.99656 | 0.00565 | **0.00337** | 0.999995 |
| `M` | 2201.74 | 0.200 | 1.1e-02 | **0.0787** | 0.99701 | 0.02684 | **0.00738** | 0.999973 |
| `Y` (the verification's row) | 1980.00 | 0.209 | 2.6e-02 | **0.0341** | 0.99950 | 0.01649 | **0.00161** | 0.999999 |
| `G` | 950.00 | 0.354 | 2.4e-01 | **0.1535** | 0.99226 | 0.15187 | **0.00068** | 1.000000 |
| `P` (f/1.2, round 2's EXCLUDED optic) | 888.00 | 0.452 | 1.4e+00 | **0.9037** | **0.72818** | 0.90318 | **0.00263** | 0.999997 |
| `X` | 960.00 | **0.461** | 7.2e-01 | **0.3276** | **0.96345** | 0.32677 | **0.00068** | 1.000000 |

Convergence controls on every row: doubling the exact arm's azimuthal safety
factor moves it by **4.7e-14 .. 1.0e-13**, doubling its ring count by
**1.1e-4 .. 7.0e-4**; doubling the angular spectrum's refinement moves it by
**4.5e-4 .. 2.3e-3**.  Both arms are converged at least a decade below the
quantity they measure.

Three findings:

1. **the published column is under-reported by a further order of magnitude
   at low NA.**  At round 2's own `Q` row the published value is **0.0003**;
   the round-2 verification's full-radius exact quadrature gave 0.01225 and
   this round's pointwise measurement gives **0.0563**.  The core-confined
   number on the same sample is 0.00537 -- ten times smaller than the
   full-radius one, which is E6's mechanism showing directly.  The fitted rule
   `rel L2 ~ 0.37 eps` predicts 3.2e-4 at that `eps`; the measurement is 178x
   that.  Nor does the rule track the data: over the seven rows `eps` spans
   three decades (8.5e-4 to 0.72) while the full-radius relative L2 spans a
   factor of 10 (0.034 to 0.328), and the two are not even monotone together
   (`Y` at `eps = 0.026` reads 0.0341, BELOW `Q` at `eps = 8.5e-4`).  **The
   rule does not hold at full radius and should not be quoted.**
2. **the fidelity conclusion is unchanged and the NA ceiling is real.**  In
   FIDELITY the `J0` form costs 5.0e-4 .. 4.5e-3 over `y_max/z` 0.108-0.209
   and **3.65e-2 at 0.461** -- at the top of that range an order of magnitude
   more than the fidelity differences either study reads (0.9421 against
   0.9985), so an NA-0.4 optic cannot be scored against the `J0` form.  That is why round 3 scores everything
   against the angular spectrum.
3. **the scoring arm is bracketed, and it has no NA ceiling here.**  ASM
   against the exact azimuthal quadrature over all eight rows: **0.00068 ..
   0.00738 in relative L2 and 0.999973 .. 1.000000 in fidelity**, including
   at `P`, the f/1.2 optic round 2 EXCLUDED, where the `J0` form is wrong by
   0.90 in relative L2 and 0.728 in fidelity and the ASM still agrees with
   the exact quadrature to 0.0026 and 0.999997.  That is what makes scoring
   `P` at all legitimate, three decades below every fidelity difference this report
   reads, and the agreement is best exactly where the `J0` form is worst.  So
   "an oracle that holds above NA 0.35" is met by construction rather than by
   an envelope argument.

Round 2's `J0` ENERGY CLOSURE column is also re-measured, from a
well-resolved radial profile: 0.99992 at `y_max/z = 0.108`, 0.99913 at 0.200,
0.99916 at 0.209, 0.99616 at 0.354, 0.98989 at 0.452 and **0.99222** at
0.461.  The round-2 verification reported
0.9972 and 0.9741 at the first two of those and could not reconcile them with
the published 0.99999 / 0.99921; the discrepancy was its own radial
reconstruction, not the propagator -- a coarse profile loses energy on both
arms equally (measured 0.894 on both at 256 radii).  **The published closure
column is closer to right than the verification concluded**, and that
correction is recorded in section 11.

---

## 9. The mutation matrix (R3-6)

Eighteen plausible regressions -- the round-2 verification's thirteen, and
five of round 3's own -- each applied to a SEPARATE detached worktree
(`C:/tmp/lum_mb3_mut`, detached at the round-3 library commit) and run against
the WHOLE five-file gate: the three files round 2 shipped, the verification's
file and this round's.  `lumenairy/` on the round-3 branch is never touched.
`PYTHONPATH` pins the mutant tree; `-p no:randomly`, `--capture=sys`, `-rs`.

**18 of 18 are caught, on BOTH builds.**  "round-3 ids" names which of this
round's own eight pins fired; where the column is empty the mutation is caught
by the round-2 gate alone, which is the correct outcome for a round-2
regression.

| # | regression | win | wsl | round-3 ids that fire |
|---|---|---|---|---|
| **M1** | an ALIAS that bypasses `_multibranch_render` | 24 failed, 43 passed | 24 failed, 43 passed | lattice provenance, lattice-vs-field, both bar arms |
| **M2** | the arbiter reads the BRANCH SUM instead of the returned field | 1 failed, 41 passed | 1 failed, 41 passed | -- (the verification's own pin) |
| **M3** | the bar is loosened to 1.20 | 1 failed, 38 passed, 3 skipped | 1 failed, 38 passed, 3 skipped | -- (the verification's own pin) |
| **M4** | the second render is taken at `(N, dx)` | 19 failed, 22 passed, 1 skipped | 19 failed, 22 passed, 1 skipped | bar above/below |
| **M5** | the LOSS arm is made to refuse too | 4 failed, 38 passed | 4 failed, 38 passed | -- |
| **M6** | the band stops being symmetric in the log | 4 failed, 38 passed | 4 failed, 38 passed | converged spread |
| **M7** | the entry cap drops to 1e6 entries | 14 failed, 26 passed, 2 skipped | 14 failed, 26 passed, 2 skipped | lattice provenance, lattice-vs-field, both bar arms |
| **M8** | the completion stops asking for the reading | 18 failed, 24 passed | 18 failed, 24 passed | lattice provenance, lattice-vs-field, both bar arms |
| **M9** | the ratio is inverted | 10 failed, 29 passed, 3 skipped | 10 failed, 29 passed, 3 skipped | bar above/below |
| **M10** | the FALLBACK path stops arbitrating | 5 failed, 37 passed | 5 failed, 37 passed | both bar arms |
| **M11** | an unmeasurable reading silently becomes `'ok'` | 2 failed, 40 passed | 2 failed, 40 passed | -- |
| **M12** | the launched-power tripwire is disabled | 1 failed, 34 passed, 7 skipped | 1 failed, 34 passed, 7 skipped | -- |
| **M13** | the entry cap drops to 2e6 entries | 2 failed, 39 passed, 1 skipped | 2 failed, 39 passed, 1 skipped | -- |
| **M14** (R3-3) | the half-pitch lattice stops NESTING (the hull-aligned alternative) | 3 failed, 37 passed, 2 skipped | 3 failed, 37 passed, 2 skipped | **the nesting identity**, lattice-vs-field |
| **M15** (R3-2) | the Pearcey route labels the branch sum's reading as the cusp field's | 1 failed, 41 passed | 1 failed, 41 passed | **the cusp label** |
| **M16** (R3-4) | every route claims the reading arbitrates the field it returns | 2 failed, 40 passed | 2 failed, 40 passed | **the cusp scope**, **the fallback scope** |
| **M17** (R3-1) | the bar is put inside the converged reading's own spread (1.004) | 6 failed, 36 passed | 6 failed, 36 passed | **the converged spread** |
| **M18** (R3-3) | the completion rebuilds the half-pitch lattice inline | 1 failed, 41 passed | 1 failed, 41 passed | **the lattice provenance** |

Two of these are worth naming.

**M17 caught the pin that was not doing its job.**  On the first run of the
matrix the bar-at-1.004 mutation was caught by three PRE-EXISTING tests and
NOT by `test_b7c3_the_bar_clears_the_converged_readings_own_spread`, which is
the pin that exists for exactly it: at three times the measured spread its own
threshold was 1.00055, which 1.004 clears.  The factor is now derived from the
measurement -- the four planes of that ladder read 4.08e-05 .. 1.82e-04 from
1, identical to the printed digit on both builds, so the shipped bar clears
the worst of them by **330x** -- and the pin asserts 50x, which fires at 1.004
and still leaves 6.6x of headroom to the shipped value.  The matrix is green
either way; the difference is whether the gate would survive the round-2 tests
being deleted.

**M14 is also caught by the round-2 gate**, and that is the strongest evidence
for the convention shipped in section 6.1: `test_b7c2_a_converged_render_reads_one_and_a_blown_up_one_reads_four`
fails under the hull-aligned lattice because the blow-up plane reads 7343.9
instead of ~4, and `test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum`
stops straddling the bar and SKIPS.  The alternative E5 asks for does not just
change a number; it breaks the identity the whole diagnosis rests on, and the
existing gate says so without being asked.


---

## 10. Recommendation

**Ship the change.  Keep the bar at 1.06.  Do not re-state the fold ring's
fidelity separation, and take two decisions.**

What verifies and should ship as it stands:

* **the mechanism**, again and on three more optics.  Every blow-up plane in
  the study reads 3.40-4.00 and healthy ones 0.87-1.06;
* **the reading's placement.**  Claim 2 reproduces to the bit on this tree:
  the branch sum reads 1.0636048339 where the completion reads 1.0021337041
  at oracle fidelity 0.9879;
* **the refactor to one lattice definition**, which is free (0 moved of 157)
  and closes a seam nothing compared;
* **the two labelling fixes** (E4, E7), which cost nothing and remove two
  places where the diagnostics said more than the measurement supports.

Two decisions are the maintainer's, and this report deliberately does not take
them:

1. **the ACCEPT CRITERION.**  The fold ring's two fidelity populations now
   overlap (0.9421 returned against 0.9520 refused), so "right" and "wrong"
   can no longer be read off the guard's own split.  The cost table of section
   5 is given at 0.95 and 0.883 precisely so the choice can be made on
   numbers.  **Recommendation: 0.95, and keep the bar at 1.06**, which costs
   one false refusal and two misses over 394 fold-ring planes -- the best
   trade measured, by a factor of five on each axis against its neighbours.
2. **whether a loss arm should REFUSE on the fallback route.**  On 476
   returned fallback planes the continuity loss arm flags 192 with zero false
   alarms, and a launched-power loss bar at 0.889 flags 254 with zero false
   alarms and nine wrong ones left.  **Recommendation: yes, on the FALLBACK
   route only**, because the physical argument for not refusing -- the
   completion replaces the dark side -- is false exactly there.  It is a
   behaviour change with a Migration note and is not taken here.

And one thing should not be claimed again in any form:

* **that the bar has a margin.**  It does not have one that can be measured:
  1.371x -> 1.032x -> 1.0045x over three ladder refinements of the same
  sixteen optics, with no floor.  What the bar has is a COST, and the cost is
  low.

---

## 11. Corrections to the earlier rounds (R3-5)

Dated corrected paragraphs are appended in place in
`fixes/WP-B7c_ROUND2_REPORT.md` (sections 0, 3.1, 3.2, 3.3, 3.4, 7 and 10) and
in `validation/oracles/caustic_fold_truth.py`.  In summary:

| round-2 claim | round-3 measurement |
|---|---|
| "a converged quadrature reads **1 exactly**, on any optic, at any plane, at any grid" | 0.9994103 .. 1.0004414 over 102 planes on 17 optics, rms 1.48e-4; the residue is a window-EDGE term that reads 1.4e-3 on an undersized window and 1.2e-4 on every larger one |
| the gap is **1.0683x**, margins 1.037x / 1.030x, "a 7 % gap ... and a ninth optic could narrow it" | the gap is a reading of the z ladder: 1.371x / 1.032x / **1.0045x** at 468 / 882 / 1304 planes on the SAME optics.  The two-sided margin at 1.06 is 1.00047x |
| "**0** false refusals at fidelity >= 0.95" on the fold ring | **1** of 394 (`W_alt` z = 4929.0 um, reading 1.0812, fidelity 0.9520).  The round-2 verification found 2 of 124 |
| "the two FIDELITY populations do not overlap ... the split is the guard's own" | they OVERLAP at 394 fold-ring planes: worst returned 0.9421, best refused 0.9520 |
| the oracle-floor rule **`rel L2 ~ 0.37 eps`**, "holding over three decades of `eps`" | does not hold at full radius.  At round 2's own `Q` row the rule predicts 3.2e-4 and the pointwise full-radius measurement is **0.0563** -- 176x.  The core-confined value on the same sample is 0.0054 |
| "returned down to 0.764" on the fallback route | **0.5358**, over 476 returned fallback planes (round 2 measured 104 planes in total) |

One correction runs the OTHER way, against the round-2 VERIFICATION rather
than the report: its E6 also concluded that the `J0` arm's energy closure is
0.9972 / 0.9741 where the published table reads 0.99999 / 0.99921.  Measured
here from a well-resolved radial profile the closure is **0.99992 / 0.99913**,
i.e. the published column is right and the verification's number was its own
radial reconstruction losing energy (a 256-radius profile closes at 0.894 on
BOTH arms).  That is recorded in `VERIFY_WP-B7c_ROUND2.md` as well.

---

## 12. Runs

All from `cd /c/tmp/lum_mb3`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `--capture=sys` and `-p no:randomly`, `PYTHONPATH` pinning the tree.

| run | build | result | duration | log |
|---|---|---|---|---|
| `test_wp_b7c_round3.py` alone | win | **8 passed** | 7.7 s | -- |
| the same | **wsl** | **8 passed** | 12.7 s | -- |
| the five-file b7c gate + `a16` + `niche_r2` + `niche_r5` + `v5_21` | win | **404 passed**, 2 deselected, 51 warnings | 3384.6 s | `pytest_core_win.txt` |
| the same | **wsl** | **403 passed**, 1 skipped (optional `astropy` absent), 2 deselected, 52 warnings | 1711.4 s | `pytest_core_wsl.txt` |
| `-k census` (whole suite) | win | **45 passed**, 16344 deselected | 1055.2 s | `pytest_census_win.txt` |
| `-k walker` (whole suite) | win | **116 passed, 8 skipped**, 16265 deselected | 279.1 s | `pytest_walker_win.txt` |
| `-k dispatcher_pin` (whole suite) | win | **511 passed, 6 skipped**, 15872 deselected, 7 warnings | 132.9 s | `pytest_dispatch_win.txt` |
| `-k changelog` (whole suite, after the CHANGELOG edit) | win | **50 passed, 7 skipped**, 16332 deselected | 134.7 s | -- |
| `test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py` + `test_audit_except_budget.py` | win | 18 passed, **1 failed** -- see below | 14.6 s | `pytest_publicapi_win.txt` |
| the 18-mutation matrix against the whole five-file gate | win | **18 of 18 caught** | ~21 min | `mutation_win.json`, `mut_*_win.txt` |
| the same | **wsl** | **18 of 18 caught** | ~22 min | `mutation_wsl.json`, `mut_*_wsl.txt` |
| `ruff check lumenairy/ tests/ validation/probe_wp_b7c_round3/ validation/oracles/` | **wsl** | **All checks passed** | -- | -- |
| `scripts/record_history_fingerprints.py --check` | win | **OK: every history document matches its module** | -- | -- |

**The one red is not this round's and predates it.**
`test_installed_metadata_version_matches_source_version` fails because the
editable install's distribution metadata reads 5.47.0 while
`lumenairy.__version__` reads 5.47.1 -- the merge of main that this branch
starts from bumped the source and the `.pth` shim was not re-installed.  Run
against the PRE archive (`C:/tmp/lum_mb3_pre`, the parent commit with none of
this round's changes) it fails identically, which is what the test is for; the
fix is `pip install -e .` on the dev box and it is not a library change.

**The history fingerprint was not re-recorded, and should not have been.**
The gate pins modules that have a document under `docs/history/`; the only one
in this family is `lumenairy.elements._lens_traced.md`, and
`lumenairy/elements/_lens_traced.py` is UNTOUCHED by this round
(`git diff dcaa21f0..HEAD -- lumenairy/elements/_lens_traced.py` is empty --
every change is in `_lens_traced_multibranch.py` and `_lens_traced_uniform.py`,
neither of which has a history document).  `--check` is green without a
re-record, and appending a `re_recorded:` line for a module that did not
change would assert a change that did not happen.  That also keeps this branch
out of the POOL region of `_lens_traced.py`, which another agent is editing.

```
python -m pytest tests/unit/test_wp_b7c_round3.py \
  tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py \
  tests/unit/test_audit2609_b7c_multibranch_envelope.py \
  tests/unit/test_verify_b7c_multibranch.py \
  tests/unit/test_verify_b7c_round2.py \
  tests/unit/test_audit2609_a16*.py tests/unit/test_niche_r2*.py \
  tests/unit/test_niche_r5*.py tests/unit/test_v5_21*.py -q --capture=sys
python -m pytest tests/ -q --capture=sys -k census
python -m pytest tests/ -q --capture=sys -k walker
python -m pytest tests/ -q --capture=sys -k dispatcher_pin
python -m pytest tests/unit/test_public_api.py \
  tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
  tests/unit/test_audit_except_budget.py -q --capture=sys
MUT_TREE=C:/tmp/lum_mb3_mut python validation/probe_wp_b7c_round3/r3mutate.py mutation_win.json
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mb3 && ~/lumvenv/bin/ruff check lumenairy/ tests/ \
  validation/probe_wp_b7c_round3/ validation/oracles/'
python scripts/record_history_fingerprints.py --check
```

`.test_durations` carries the eight new ids with their measured values and
reloads as valid JSON (16 310 entries), so a sharded CI run schedules none of
them as unknown-duration.


---

## 13. What I could not measure

* **the 1304-plane population on the wsl build.**  As in both earlier rounds
  the oracle-scored population is a win measurement.  What is measured on both
  builds is the DECISION gate (every test file, section 12) and the mutation
  matrix, both of which re-derive their premises on the running build.  The
  oracle never imports lumenairy, so the fidelity column cannot move across
  builds; the READING could, and the round-2 verification measured 23 of them
  identical to 4.1e-16 across the two builds on this same library;
* **the oracle floor on all twelve planes.**  Eight of the twelve completed
  (`Q`, `W`, `S`, `M`, `Y`, `G`, `P`, `X`), spanning `y_max/z` 0.108 to
  0.461 -- the whole NA axis, including both planes the earlier rounds
  published a number for and the f/1.2 optic round 2 excluded.  The remaining
  four (`F_alt`, `HN`, `AS`, `MC`) add coverage, not a new regime, and the
  exact full-radius quadrature costs 10-25 minutes per plane;
* **whether the fold-ring gap has a floor below 1.0045x.**  Each refinement
  halved the logarithm of the gap and the third pass was still finding
  crossings; a fourth would take it lower.  What is established is the trend
  and its mechanism (the reading's own excursion in z), not a limit;
* **an accept criterion of my own that is independent of the oracle's
  fidelity.**  Energy error tracks fidelity closely on this population (the
  worst returned fold-ring fields carry 1.24-1.27x the oracle's power), but
  the two are not independent axes and no third was measured;
* **whether the twelve unordered `AS` planes are a property of the
  non-monotone focal locus or of that optic's grid.**  Both `AS` and `AS_alt`
  show them at the same z, which argues for the geometry, but only two grids
  were run;
* **the wall-clock cost of the change**, which is zero by construction (0
  moved of 157, and `_render_centre_origin` at the shipped offset is the same
  arithmetic), so it was not timed.

# VERIFY-WP-C5 -- the three default flips, re-measured against independent fixtures and an independent oracle

Independent adversarial verification of `feat/c5-three-defaults` (commits
`3dcc2667`, `6149af4f`, `f6ba0852`, `750f2d02`, `38b218c1`, `d8de383c` on
`49ddf4bd`), against
`fixes/WP-C5_THREE_DEFAULTS_REPORT.md`, `tests/unit/test_c5_three_defaults.py`
and `validation/probe_c5_three_defaults/`.

Nothing below is read out of that report.  Every number here was taken on a
fresh worktree (`verify/c5-three-defaults` at `C:/tmp/lum_vc5`) with probes
written for this verification
(`validation/probe_verify_c5/`), on fixtures this verification chose, and --
for the one question the WP-C5 report files under "could not be measured" --
against an oracle that is not paraxial and shares no machinery with the
library.

## The two builds, and how every run was pinned

| | Windows | WSL |
|---|---|---|
| python | 3.14.6 (MSC v.1944) | 3.12.3 (GCC 13.3.0) |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 | 1.17.1 |
| interpreter | `python` | `~/lumvenv/bin/python` |
| tree pin | `PYTHONPATH=C:/tmp/lum_vc5` | `PYTHONPATH=/mnt/c/tmp/lum_vc5` |
| PRE pin | `PYTHONPATH=C:/tmp/vc5_arch/base` | `PYTHONPATH=/mnt/c/tmp/vc5_arch/base` |

Every command carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line; every GBD run additionally carried
`LUMENAIRY_MEM_BUDGET_MB=2048`; every pytest run carried `--capture=sys` and
`-p no:randomly`; every probe writes the resolved `lumenairy.__file__` into
its JSON and prints it.  The PRE tree is this verification's own `git archive
49ddf4bd` extracted to `C:/tmp/vc5_arch/base` and diffed against the clean
worktree before use -- it differs in exactly two files,
`lumenairy/propagators/carrier.py` and `lumenairy/propagators/gbd.py`, which
are the two the branch edits.

## Verdict table

Every row is a claim from the WP-C5 report, restated in that report's own
words, with THIS verification's number beside it.  "CONFIRMED" means
re-measured on both builds and within the claim; a number in bold is where
this verification reads differently.

### Item 1 -- `_GAP_KERNEL_ACCURACY_TAU = 1e-4`

| claim | WP-C5 | here (Windows) | here (WSL) | verdict |
|---|---|---|---|---|
| hygiene-2 ladder inert, worst departure | 4.72e-06, all 9 rungs `'exact'` | 4.716053e-06, 9/9 `'exact'` | 4.716053e-06, 9/9 | CONFIRMED |
| hygiene-2 ladder byte-identical to PRE | yes | 9/9 digests identical | 9/9 | CONFIRMED |
| F3 falls back at 1 um: 2.35e-03 -> 1.46e-14 | 2.3496e-03 -> 1.4568e-14 | 2.3496e-03 -> 1.4568e-14 | -> 1.4564e-14 | CONFIRMED |
| F3 falls back at 10 um: 2.35e-04 -> 1.18e-14 | 2.3490e-04 -> 1.1822e-14 | same | -> 1.1817e-14 | CONFIRMED |
| F3 does NOT fall back at 100 um | `'exact'`, 2.3438e-05 | `'exact'`, 2.3437e-05 dep | same | CONFIRMED |
| band: F3 fires inside 23.48 um of focus | 23.4823 um, threshold 68.10 m | 23.48231 um, 68.096394 m | identical | CONFIRMED |
| band: hygiene-2 fires inside 1.54 um of its `A = 0` plane | 1.5427 um, threshold 260.09 m | 1.5427023 um, 260.0868 m | identical | CONFIRMED |
| archive-to-archive, F3's two rungs the only movers | 46/48 identical | 46/52 identical, 6 moved, and ALL SIX are legs built here to trip the rule | identical set | CONFIRMED |
| `tau=None` restores byte identity | 14/14 | 10/10 keys, `kernel_departure` absent | 10/10 | CONFIRMED |
| the one suite leg outside the tau fixtures | b4 `mismatch_matrix`, z_eff -0.180 m, theta_env 5.5712e-03, departure 1.2733e-04 | reproduced exactly: \|z_eff\| 0.180000 m, theta_env 5.571183e-03, departure 1.273287e-04 | identical | CONFIRMED |
| that leg is "a LARGE-ANGLE leg" rather than near-focus | asserted | **measured, and stronger than stated**: the rule fires on a leg 9.00 mm from its `A = 0` plane and 48.7 Rayleigh ranges from its beam's focus | identical | CONFIRMED, scope wider than the ledger's prose |
| which kernel is right on that leg | "could not be measured" | **MEASURED**: against an exact scalar oracle the plain kernel reads 8.212599e-02 and the refined one 8.222558e-02, so the rule moves that leg TOWARD the truth | identical to 7 digits | ANSWERED -- not a defect |

### Item 2 -- `DENSE_MEM_BUDGET_ACCOUNTING = 'measured'`

| claim | WP-C5 | here (Windows) | here (WSL) | verdict |
|---|---|---|---|---|
| explicit `'legacy'` identical to PRE | 24/24 | 12/12 explicit-mode keys identical (6 `'legacy'`, 6 `'measured'`) + 2 windowed | identical | CONFIRMED |
| default path moves, the rest explained | 14/22 moved | 3 of my 18 keys moved and all three are `default/*`; the fourth default key (`default/512/8MB`) does not move because both accountings already floor the chunk at one column at that budget -- the report's own explanation, reproduced | identical set | CONFIRMED |
| overrun 6.0x -> 0.76x at 512 MB / 1024 beamlets | 6.00x / 6.02x -> 0.756x / 0.762x | 6.0003x / 6.0214x -> 0.7587x / 0.7621x | 5.5009x / 5.5215x -> 0.6961x / 0.7005x | CONFIRMED |
| fitted constants: `fixed` ~48.5, `c` 96 Windows / 88 WSL | 48.551 / 96 / 88 | 48.556 (N=512), 49.375 (N=320), c = 96.000, worst model deviation 4.0e-07 | 48.183 / 48.420, c = 88.000, 4.3e-07 | CONFIRMED |
| the floor is an upper bound on the one-column peak by 1.2x-1.3x | 1.205-1.294 | 1.2106 (N=320) / 1.2175 (N=512) | 1.2901 / 1.2924 | CONFIRMED |
| above the floor the peak never reaches the budget, worst 0.917 at 1.5x | 0.917 | worst 0.9143 at 1.5x (N=320), 0.9112 at 1.5x (N=512); no notice anywhere on the sweep | worst 0.8501 | CONFIRMED |
| below the floor 1.66x with the notice naming the floor and `window=5.0` | 1.661 / 1.643 | 1.6524 / 1.6428, notice names the floor, `window=5.0`, `raise mem_budget_mb` and the helper | 1.5505 / 1.5476 | CONFIRMED |
| `'legacy'` is silent below the floor | silent, 6.008x / 6.006x | silent, 6.0157x / 6.0064x | silent, 5.5502x / 5.5475x | CONFIRMED |
| refusal would break default calls past N = 1706 | "any square grid past N = 1706" | **off by one**: the floor is 511.64 MB at N = 1705 and 512.24 MB at N = 1706, so it binds AT 1706 | identical | CONFIRMED with a correction (D4) |
| a default call past that still completes | not run | **run**: N = 2048, default budget -> completes, finite, exactly ONE notice, 601 MB RSS; the same call under `'legacy'` takes 3013 MB and says nothing | 570 MB / 2783 MB | CONFIRMED, and the decision is vindicated |
| the unknown-mode refusal | refused by name | refused for `'Measured'`, `'MEASURED'`, `'legacy '`, `''`, `16`, `'honest'`, and through the public entry point; on the PRE tree the same value is silently ACCEPTED as `'legacy'` | identical | CONFIRMED |
| `'legacy'` vs `'measured'` fields differ by ~2e-17 relative | 2.117e-17 / 1.824e-18 | **1.799e-16 (N=320), 7.260e-17 (N=512)** -- same last-bit character, an order of magnitude larger on these grids | identical to every digit | CONFIRMED in kind; the quoted figure is grid-specific |
| RSS | "could not be measured" | **MEASURED** on both builds with a sampled high-water mark: at a 300 MB budget on 512^2, `'measured'` peaks at 0.7146x the budget in resident set and `'legacy'` at 5.9924x | 0.7789x / 5.4679x | ANSWERED |

### Item 3 -- `replica_fill='zero'`

| claim | WP-C5 | here (Windows) | here (WSL) | verdict |
|---|---|---|---|---|
| a faithful window is returned by identity on either fill | same object | `_fill_readout_replicas` returns the input object on both fills; all 12 faithful-window digests (repeat / zero / default) identical to PRE | identical | CONFIRMED |
| oversized: inside byte-identical, outside exactly 0.0 | yes | 16/16 windows past one period: inside byte-identical, `max abs` outside = exactly 0.0 | 16/16 | CONFIRMED |
| faithful counts `2*floor(p/2/dx_out)+1` at 1.10/1.60/2.20 | 233 / 161 / 117 | 233 / 161 / 117 on a DIFFERENT geometry (1.55 um, R -25 mm, w 0.60 mm) | identical | CONFIRMED |
| off axis the surviving band is off-centre | asserted on one case | **16 windows past one period** (3 on-axis, 6 off-axis, 3 anamorphic, 2 finely sampled, 2 through the Sziklas readout), **11 of them with the field- and window-centred bands differing**: the zeroed set equals the complement of the FIELD-centred band in every one of the 16, and never the complement of the window-centred band in any of the 11 | identical | CONFIRMED |
| exact readout at 1.71 periods | faithful (2401, 2401) | 1.71 periods, faithful (2395, 2395) = the formula on this geometry, inside byte-identical, outside exactly 0, default == `'zero'` | identical | CONFIRMED |
| `_period_out['replica_fill']` and `readout_replica_fill` | published | published on both fills and on faithful as well as oversized windows; the chain publishes `readout_replica_fill` per stage and it follows the keyword | identical | CONFIRMED |
| refusal census identical cell for cell | 1.00 served, 1.02 refused | identical cell for cell across both fills AND identical to the PRE tree, on a 12-cell census; 1.00 served, 1.02 refused | identical | CONFIRMED |
| blast radius: only default-fill calls past one period move | 16/20, 4 moved | 46/63 identical, 17 moved, every one a `default` key on a window past one period; every explicit `'repeat'`, every explicit `'zero'` and every faithful default identical | identical set | CONFIRMED |
| the periodicity the fill is justified by | "`E(u + period) == E(u)` identically in ABSOLUTE output coordinates" | **true for the Sziklas readout (1.5e-14) and the exact readout (1.8e-13); on `_collins_focus_readout` only the MODULUS is periodic (2.0e-14)** -- the complex field obeys `E(u+p) = exp(i[2 pi u/dx_in + pi lam z/dx_in^2]) E(u)`, verified to 7.2e-09 | identical | DOCUMENTATION DEFECT (D5); behaviour unaffected |

---

## Item 1, in full -- and the question the report could not answer

### What was re-measured

The two published ladders were rebuilt from their own definitions (not copied
from the WP-C5 probe) and reproduce every digit of the report's tables on both
builds, including the two rungs that fall back and the one that does not.  A
THIRD near-focus ladder of this verification's own (HeNe 0.633 um, `w` 0.20 mm,
`R` -30 mm, `dx` 3 um, `dx_out` 3.1 um) behaves the same way and places the
band where the closed form says:

| fixture | `theta_env` (rad) | closed-form `\|z_eff\|` threshold | bisected switch-over | agreement |
|---|---|---|---|---|
| VERIFY-B4 F3 | 1.1289390e-03 | 68.096394 m | 23.48231 um from focus | 1e-9 |
| hygiene-2, walked to its own `A = 0` | 7.9512089e-04 | 260.0868 m | 1.5427023 um | 1e-9 |
| **this verification's** (0.633 um) | 1.0074508e-03 | 63.88117 m | 14.08205 um | 1e-9 |

All three are identical to every printed digit on both builds.

### The suite leg the rule moves, and which kernel is right there

The WP-C5 report names one leg outside the tau fixtures that changes kernel --
the `mismatch_matrix` fixture of `test_audit2609_b4_collins_transport.py` --
and files "which kernel is the more physical" under "could not be measured",
because its oracle is the paraxial whole-function `q` form.

**That question is answerable, and it is answered here.**  The fixture's input
is `exp(-r^2/w^2) exp(i k r^2 / 2 R0)`, i.e. `exp(-a r^2)` with COMPLEX `a`,
whose 2-D Fourier transform is `(pi/a) exp(-pi^2 f^2 / a)` in closed form -- so
nothing has to be sampled on the input lattice, which the carrier aliases by a
factor of 2400.  Propagating that analytic spectrum with the EXACT transfer
function `exp(i k z sqrt(1 - lambda^2 f^2))` and inverting it by a 1-D Hankel
quadrature on the readout's own abscissae gives the exact scalar (Helmholtz)
field, with no shared machinery with the library.

The oracle is validated two ways before it is used:

* **quadrature self-check** -- halving the sample count moves it by 3.670e-09,
  five decades under the 1.5e-04 gap it is asked to resolve;
* **the physics it must reproduce** -- the non-paraxial departure of a
  converging Gaussian is `sqrt(3/2) k \|z\| NA^4 / 8` to leading order, so a
  10x sweep in NA (at `z = w/NA`) must move it by 1000x at a FIXED ratio to
  that law.  Measured: 8.4961e-05 / 6.7273e-04 / 5.3695e-03 / 8.2126e-02 at
  NA 0.005 / 0.01 / 0.02 / 0.05, i.e. ratios to the law of 0.9256 / 0.9162 /
  0.9141 / 0.8948 -- a monotone drift of 3 % over three decades, which is the
  next term in the expansion and not a quadrature error.  A quadrature error
  would not scale as NA^3; a silently-paraxial oracle would read zero.

On the leg in question (fr = 0.90, `\|z_eff\|` 0.180 m, theta_env 5.5712e-03,
departure 1.2733e-04 = 1.27 tau):

| | relative L2 against the EXACT scalar field | against the paraxial analytic |
|---|---|---|
| `gap_kernel='fresnel'` (what `'auto'` now returns) | **8.212599e-02** | 3.5011e-08 |
| `gap_kernel='exact'` (what `'auto'` returned before) | **8.222558e-02** | 1.1624e-04 |

and on this verification's own wide-envelope leg (1.55 um, NA 0.04, fr = 0.85,
departure 1.7729e-04):

| | against the EXACT scalar field |
|---|---|
| `'fresnel'` | **2.899331e-02** |
| `'exact'` | **2.914391e-02** |

Both readings are identical on both builds to seven digits.  Two conclusions,
and the second matters more than the first:

1. **The rule does not make that leg worse.**  On six of the seven mismatched
   rows measured across the two fixtures the exact-kernel refinement moves the
   answer AWAY from the exact scalar field, and the rule removes it on both
   rows where it fires.  The seventh (`fr = 0.99`, departure 2.68e-07, two
   decades under tau and therefore untouched by the rule) is 4e-08 better with
   the refinement -- at that size the comparison is the oracle's own last
   digits.  The scope of the rule is not a defect.
2. **The decision is not a physics choice at this fixture's NA.**  Both
   kernels sit 8.21e-02 from the true field, because the refinement lives in
   the REDUCED frame on the ENVELOPE's angle while the leg's own
   non-paraxiality is set by the beam's NA (0.05).  The rule changes the
   returned field by 1.2733e-04 relative and changes its error against the
   true field by 9.96e-05, against a modelling error of 8.2126e-02 --
   factors of 645 and 825.  Neither kernel is "right" there; the plain one is
   merely not wrong in a second way.

### The scope: the rule is a band in `k |z_eff| theta^4`, not a distance

The ledger's plain-language section 0.1 says the switch matters "only for a leg
that lands within about 100 micrometres of a geometric focus".  That is true of
the two fixtures it was built on and is not true of the rule.  Measured here on
one fixture family (one physical beam re-enveloped against three mismatched
carriers, so only the envelope's angle differs):

| `fr` | distance to the leg's own `A = 0` plane | `theta_env` | departure | resolves |
|---|---|---|---|---|
| 0.85 | **3.00 mm** | 7.0857e-03 | 1.7729e-04 | **fresnel** |
| 0.90 | 2.00 mm | 4.4870e-03 | 4.5281e-05 | exact |
| 0.95 | 1.00 mm | 2.1937e-03 | 5.4617e-06 | exact |

The leg FURTHEST from its own `A = 0` plane is the one that falls back: the
ordering is the opposite of a distance rule.  And with the readout moved to a
QUARTER of the way to the focus -- where the beam is 0.600 mm wide, 48.7 focal
waists, i.e. 48.7 Rayleigh ranges short of its focus, and 9.00 mm from the
carrier's `A = 0` plane -- `fr = 0.70` still falls back (departure 4.1794e-04,
`k4` 8.95e-05) while `fr = 0.80`, which is FURTHER from that plane, does not
(4.5478e-05).  There is no focus within two decades of that plane by either
reading.

This is not a defect in the branch -- the constant's own note in `carrier.py`
states the law correctly and calls the band a band -- but the ledger's
plain-language entry and the CHANGELOG/Migration prose that a caller will read
describe a rule that is narrower than the one that ships.  See D6.

---

## Item 2, in full

### The constants, re-fitted on different grids

`peak/(Ny Nx) = fixed + c * chunk` over a 1/2/4/8/16/32 ladder at a budget far
too large to bind, one warm-up run discarded (the report's note about the first
`tracemalloc` window in a process is correct and reproduces):

| grid | build | `fixed` (B/cell) | `c` (B/cell-col) | worst deviation |
|---|---|---|---|---|
| 512 x 512 | Windows | 48.5562 | 96.0000 | 4.0e-07 |
| 320 x 320 | Windows | 49.3747 | 96.0000 | 9.5e-07 |
| 512 x 512 | WSL | 48.1828 | 88.0000 | 4.3e-07 |
| 320 x 320 | WSL | 48.4199 | 88.0000 | 1.0e-06 |

The grid-INDEPENDENT-offset reading holds on this verification's grids too:
Windows' two `fixed` readings differ by 0.819 B/cell, which at N = 320 is
83.8 KB, the same order as the 102 KB the report measured at N = 256.  `c` is
build-dependent (96 against 88) exactly as reported, and 128 sits above both.

### The floor, and both instruments

| grid | build | published floor | `tracemalloc` one-column peak | ratio | sampled RSS one-column peak | ratio |
|---|---|---|---|---|---|---|
| 320 | Windows | 18.0224 MB | 14.8874 MB | 1.2106 | 14.8070 MB | 1.2172 |
| 512 | Windows | 46.1373 MB | 37.8956 MB | 1.2175 | 37.9576 MB | 1.2155 |
| 320 | WSL | 18.0224 MB | 13.9694 MB | 1.2901 | 0.0041 MB | n/a |
| 512 | WSL | 46.1373 MB | 35.6995 MB | 1.2924 | 25.1699 MB | 1.8330 |

**The RSS column is reported, not gated, and the WSL rows say why.**  An
in-process resident-set reading only sees a transient that has to map new
pages; a small transient following a large one is served out of pages glibc
already holds and reads essentially zero (the 0.0041 MB cell above).  Run to
run at N = 1024 the same reading spread 134.09-159.26 MB on WSL, 19 %, against
a floor that sits 16 % above it -- an S4 bar in the sense of
`docs/TESTING_STANDARDS.md`, so it is stated here and asserted nowhere.

What IS decades wide, and what the new decision test asserts instead, is the
flip itself.  At a 300 MB budget on a 512-square grid, 256 beamlets, with the
honest arm run FIRST so its pages are fresh:

| build | mode | `tracemalloc` / budget | sampled RSS peak / budget |
|---|---|---|---|
| Windows | `'measured'` | 0.7135 | **0.7146** |
| Windows | `'legacy'` | 5.9983 | **5.9924** |
| WSL | `'measured'` | 0.6573 | **0.7789** |
| WSL | `'legacy'` | 5.5017 | **5.4679** |

So the flip makes `mem_budget_mb` a bound on the RESIDENT SET and not only on
the Python allocator's own accounting: separation 8.39x (Windows) and 7.02x
(WSL), with the two instruments agreeing to within 19 % on every arm.

### Warn, not refuse -- the premise, checked where it binds

The report's arithmetic is right and its prose is off by one.  At the shipped
`mem_budget_mb = 512.0` and the floor's 176 B/cell, `N* = 1705.6057`:

| N | floor | against the 512 MB default |
|---|---|---|
| 1705 | 511.6364 MB | does NOT bind |
| **1706** | **512.2367 MB** | **binds** |
| 1707 | 512.8374 MB | binds |

so the floor binds AT N = 1706, not "past" it (D4).

The decision itself is vindicated by a measurement the report does not make --
a DEFAULT-path call on a grid where the default budget binds:

| build | mode | completes | finite | floor notices | peak RSS |
|---|---|---|---|---|---|
| Windows | default (`'measured'`) | yes | yes | **exactly 1** | 601.1 MB |
| Windows | `'legacy'` | yes | yes | 0 | **3012.5 MB** |
| WSL | default | yes | yes | **exactly 1** | 570.4 MB |
| WSL | `'legacy'` | yes | yes | 0 | **2782.7 MB** |

(N = 2048, 48 beamlets, `dx` 5.0e-07 so the bundle's own profile still decays
across the grid.)  A refusal would have turned a call that completes in
601 MB into a hard error; the notice names the floor (738.198 MB), both
remedies (`window=5.0` and `raise mem_budget_mb`) and the helper that
publishes the number.  The PRE tree runs the identical call at 3019 MB and
says nothing, and its returned bytes are identical to the branch's `'legacy'`
arm -- which is the archive-to-archive statement that `'legacy'` is 5.48.x.

### The element family's exposure, re-grepped

The report's narrowing claim checks out.  `mem_budget_mb` reaches this loop
from `lumenairy/elements/lenses_gbd.py` at four call sites (lines 540, 548,
610, 616) and every one of them also passes `window=window`, whose parameter
default is `5.0` (line 272) -- so all four take the WINDOWED path, whose
accounting this item does not touch, unless a caller passes `window=None`.
`carrier.py`'s and `fga.py`'s `mem_budget_mb` are different budgets and never
reach the dense loop.  Measured beside that: the two `windowed/*` digests in
this verification's set are byte-identical to the parent archive on both
builds.

### The unknown-mode refusal, two-sided against the parent

| spelling | PRE (`49ddf4bd`) | branch |
|---|---|---|
| `'Measured'`, `'MEASURED'`, `'legacy '`, `''`, `16`, `'honest'` | silently used as `'legacy'` | `ValueError` naming the knob and its whole vocabulary |
| through `reconstruct_field_from_beamlets` | **ACCEPTED** (ran, silently under-counting) | refused |

---

## Item 3, in full -- the misapplication case

The WP-C5 probe computes the faithful region with the SAME expression the
library's mask uses (`|u + centre_out| <= period/2`), so it cannot refute a
mask keyed on the wrong centre.  Here the band is derived three ways that do
not share that expression:

1. **empirically**, from the returned `'repeat'` array -- the smallest integer
   sample shift that reproduces its modulus, with the window ratios chosen so
   one period IS a whole number of samples;
2. **from the closed form** of the transport (`lambda |z| / dx` per axis),
   which `_period_out` is then compared against, never used to build the band;
3. **physically**, against the analytic focused Gaussian the fixture is.

Sixteen windows past one period, on a geometry of this verification's own
(1.55 um, `R` -25 mm, `w` 0.60 mm, input 512 x 5 um).  `yes`/`no` in the two
band columns is `np.array_equal` on the boolean mask, not a tolerance:

| case | centre (periods) | window/period | `dy/dx` | faithful | zeroed == complement(FIELD band) | zeroed == complement(WINDOW band) | inside byte-identical | max abs outside, `'zero'` | max abs outside, `'repeat'` |
|---|---|---|---|---|---|---|---|---|---|
| on-axis 1.10 | 0, 0 | 1.10 | 1 | (233, 233) | yes | (same band) | yes | **0.0** | 1.390e-03 |
| on-axis 1.60 | 0, 0 | 1.60 | 1 | (161, 161) | yes | (same band) | yes | **0.0** | 2.318e-03 |
| on-axis 2.20 | 0, 0 | 2.20 | 1 | (117, 117) | yes | (same band) | yes | **0.0** | 7.219e+00 |
| off x | 0.30, 0 | 1.60 | 1 | (161, 161) | yes | **no** | yes | **0.0** | 2.904e+01 |
| off y | 0, 0.30 | 1.60 | 1 | (161, 161) | yes | **no** | yes | **0.0** | 2.904e+01 |
| off xy | 0.30, -0.45 | 1.60 | 1 | (161, 136) | yes | **no** | yes | **0.0** | 2.904e+01 |
| off edge | 0.62, 0.62 | 1.60 | 1 | (109, 109) | yes | **no** | yes | **0.0** | 1.879e+01 |
| off small | 0.07, -0.21 | 1.15 | 1 | (222, 192) | yes | **no** | yes | **0.0** | 1.278e-03 |
| off wide | 0.55, 0.15 | 2.40 | 1 | (107, 107) | yes | **no** | yes | **0.0** | 2.904e+01 |
| anamorphic | 0.25, 0.40 | 1.60 / 2.72 | 1.7 | (161, 94) | yes | **no** | yes | **0.0** | 1.458e+01 |
| anamorphic | -0.33, 0.18 | 1.60 / 0.96 | 0.6 | (155, 214) | yes | **no** | yes | **0.0** | 2.177e+01 |
| anamorphic | 0.40, -0.10 | 1.30 / 3.25 | 2.5 | (148, 79) | yes | **no** | yes | **0.0** | 2.563e+01 |
| fine on-axis (N = 2048) | 0, 0 | 1.60 | 1 | (1281, 1281) | yes | (same band) | yes | **0.0** | 2.331e-03 |
| fine off-axis (N = 2048) | 0.30, 0 | 1.60 | 1 | (1281, 1281) | yes | **no** | yes | **0.0** | 2.904e+01 |
| Sziklas on-axis | 0, 0 | 1.60 | 1 | (161, 161) | yes | (same band) | yes | **0.0** | 4.455e-02 |
| Sziklas off-axis | 0.30, 0 | 1.60 | 1 | (161, 161) | yes | **no** | yes | **0.0** | 2.896e+01 |

Identical on both builds on every structural key.  Two readings are worth
pulling out:

* **off axis SPENDS the period, and that is what makes the blanking matter
  more there, not less.**  At 1.60 periods ON axis the part outside the band
  holds 2.318e-03 against an in-band peak of 29.04 -- 8e-05 of it, a wing.
  Push the same window 0.30 periods off axis and the part outside holds
  **29.04**, a full-amplitude copy of the core, because the window now reaches
  the replica at exactly one period.  The maintainer's stated concern was
  misapplication; on this axis the flip is doing more work off axis than on it,
  and it is doing it in the right place.
* **the physical check separates by 170x**: on a finely-sampled pair (2048
  samples, `dx_out` = 0.29 focal waists) the normalised deviation of the
  `'repeat'` readout from the analytic focused Gaussian is 5.83e-03 INSIDE the
  field-centred band and **1.000** outside it, with the field-centred and
  window-centred bands disagreeing.

The refusal census is identical cell for cell under both fills AND identical to
the PRE tree, and this verification adds the axis the branch's census does not
walk -- the OFFSET:

| window/period | `centre_out` | `'repeat'` | `'zero'` |
|---|---|---|---|
| 0.50 | 0 | served | served |
| 0.50 | 0.20 p | served | served |
| 0.90 | 0 | served | served |
| 0.90 | **0.20 p** | **refused** | **refused** |
| 1.00 | 0 | served | served |
| 1.02 | 0 | refused (ALIASES) | refused (ALIASES) |

-- i.e. the ratio at which a window is refused depends on the offset exactly as
`2|centre_out| + N_out dx_out <= period` says, and the fill does not enter that
decision on either axis.

---

## Cross-item: what WP-C3 would add to item 1's blast radius

The C5 report notes that WP-C3 (the `transport='collins'` default, on a
concurrent branch) will ENLARGE item 1's blast radius, "because every leg that
moves from `'sziklas'` to `'collins'` becomes a leg the near-focus rule can
see", and does not quantify it.  Quantified here.

**Method.**  The same census plugin
(`validation/probe_verify_c5/v_kernel_census.py`) with
`C5V_FORCE_COLLINS=1`, which rewrites the `transport` keyword default of
`propagate_traced_carrier_chain` and `propagate_traced_carrier_chain_multi`
from `'sziklas'` to `'collins'` before collection -- i.e. it simulates C3's
flip for every call site that does not name the transport itself.  Run over the
same 27 carrier / traced-chain files plus `test_c5_three_defaults.py` and
`test_wave5_h2_collins_jax.py`.

| | shipped (`'sziklas'` default) | forced `'collins'` |
|---|---|---|
| legs the `k4` gate resolved to `'exact'` (i.e. legs the rule can see) | 167 | **284** |
| legs the rule MOVED | 24 | **96** |
| test ids that reach the rule | 48 | **73** |
| test ids with at least one fallback | 6 | **15** |
| pytest outcome | 1036 passed, 1 skipped | 950 passed, 64 failed, 22 errors, 1 skipped |

**The leg list.**  Nine ids gain a fallback; no id loses one.  `|z_eff|` and
`theta_env` are this verification's measurements off the running build.

| test id (all `tests/unit/`) | fallbacks, shipped -> collins | departure, min..max | `theta_env` (rad) | `\|z_eff\|` (m) |
|---|---|---|---|---|
| `test_niche_d3_guards.py::test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it` | 0 -> **25** | 1.448e-04 .. **2.396e+00** | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.326 |
| `test_niche_d3_guards.py::test_c13_makes_the_d3_separation_build_independent` | 0 -> **15** | 1.448e-04 .. 2.396e+00 | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.326 |
| `test_niche_d3_guards.py::test_the_guarded_input_really_is_the_wrong_answer` | 0 -> **15** | 1.448e-04 .. 2.396e+00 | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.326 |
| `test_niche_d3_guards.py::test_the_residual_degree_moves_the_multiplexed_route_only_through_c6` | 0 -> **8** | 1.013e+00 .. 2.396e+00 | 4.54e-02 .. 9.32e-02 | 0.0433 .. 0.326 |
| `test_niche_d3_guards.py::test_the_verdict_is_identical_through_a_slow_and_a_fast_chain` | 0 -> **4** | 1.637e-04 .. 2.364e-03 | 1.11e-02 .. 2.17e-02 | 0.0145 .. 0.0263 |
| `test_niche_gap_frame_observable.py::test_arm_c_has_its_own_knob_and_does_not_silence_arms_a_b` | 0 -> **2** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_gap_frame_observable.py::test_arm_c_catches_what_the_carrier_na_proxy_misses` | 0 -> **1** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_gap_frame_observable.py::test_gap_env_phi_tol_zero_disables_the_trip_but_keeps_the_number` | 0 -> **1** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_d2_chain_multi.py::test_memory_budget_is_honoured` | 0 -> **1** | 1.367e-04 | 6.92e-03 | 0.0811 |

and the six ids that already fall back are unchanged
(`test_audit2609_b4_collins_transport.py` 1, `test_c5_three_defaults.py` 1 + 19,
`test_wave5_h2_near_focus_table.py` 2 + 1 + 1).

**What the C3 merge should be checked against.**

1. **The blast radius is 4x, not 1.3x.**  24 moved legs become 96.  Every one
   of the nine new ids is a CHAIN call (`fn = propagate_traced_carrier_chain`),
   so a C3 merge changes them twice: once by the transport and once by the
   rule, and only the transport change will be visible in C3's own
   archive-to-archive digests unless the rule is disarmed for that comparison.
   The clean way to separate them is to run C3's blast radius with
   `carrier._GAP_KERNEL_ACCURACY_TAU = None` as well as with the shipped
   `1e-4`, and to report both.
2. **None of the new legs is near a focus, and most are far outside the band's
   linear regime.**  `|z_eff|` on the new legs runs 0.014 to 0.33 m -- four
   decades under VERIFY-B4 F3's 1600 m -- while `theta_env` runs to
   1.26e-01 rad, 112x F3's.  The departures reach **9.33**, i.e. 93 000 tau: at
   that reading the "exact kernel" is not a refinement of anything and falling
   back is almost certainly right, but nothing has SCORED those legs against an
   oracle and this verification did not either.
3. **The census is a LOWER bound.**  The forced-Collins run had 64 failures and
   22 collection/fixture errors (tests that assert the `'sziklas'` route, which
   is exactly what C3 will be re-pinning).  A leg inside a test that errors
   before reaching the transport is not counted, so the real figure after C3
   re-pins its fixtures can only be larger.
4. **The `d3` guard ids are the ones to watch.**  Five of the nine are
   `test_niche_d3_guards.py` ids whose subject is a numerical SEPARATION
   between two routes (`test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it`
   and its siblings).  A kernel change on both sides of such a comparison can
   move the separation without moving either side's correctness, so those are
   the ids whose bars have to be re-derived rather than re-recorded.

---

## Defects

None of these blocks the ship.  Numbering is D<n>; each carries a reproducer
and the exact edit requested.  D6 is the only one with a consequence for a
caller; D9 is a test-coverage scope finding; the rest are documentation
precision.

### D1 (source comment, item 1) -- the in-function comment still says the rule is OFF

`lumenairy/propagators/carrier.py:2458-2460`, inside `_collins_transport`,
still reads:

```
        # ACCURACY-KEYED FALLBACK, OFF BY DEFAULT.  See
        # :data:`_GAP_KERNEL_ACCURACY_TAU`: with the shipped ``None`` nothing
        # below is evaluated and the leg is 5.47.0 to the byte.  Only 'auto'
```

The shipped constant is `1e-4` (line 1736).  This is the comment a reader of
the transport meets first, and it states the opposite of what ships.

Reproducer:

```
$ grep -n "OFF BY DEFAULT" lumenairy/propagators/carrier.py
2458:        # ACCURACY-KEYED FALLBACK, OFF BY DEFAULT.  See
$ grep -n "^_GAP_KERNEL_ACCURACY_TAU" lumenairy/propagators/carrier.py
1736:_GAP_KERNEL_ACCURACY_TAU = 1e-4
```

**Exact edit** -- replace lines 2458-2460 with:

```
        # ACCURACY-KEYED FALLBACK, ARMED BY DEFAULT.  See
        # :data:`_GAP_KERNEL_ACCURACY_TAU`: setting it back to ``None``
        # leaves nothing below evaluated and the leg is 5.48.x to the byte.
        # Only 'auto'
```

### D2 (docstring, item 1) -- the departure law's docstring says the rule is off

`lumenairy/propagators/carrier.py:1782-1783`, in
`_collins_exact_kernel_departure`:

```
    :data:`_GAP_KERNEL_ACCURACY_TAU` for the readings and for why the rule it
    feeds is off by default.
```

**Exact edit** -- replace those two lines with:

```
    :data:`_GAP_KERNEL_ACCURACY_TAU` for the readings, for the band the rule
    it feeds fires in, and for the caveat that goes with it.
```

### D3 (docstring rendering, item 3) -- a paragraph at column 0 inside an indented docstring

`lumenairy/propagators/carrier.py:5300`.  The new paragraph of
`_fill_readout_replicas`'s docstring begins in column 0 while the rest of the
docstring is indented four:

```
$ sed -n '5299,5301p' lumenairy/propagators/carrier.py
    replicas in them.

``'zero'`` IS THE DEFAULT, AND WHAT THAT DOES NOT CHANGE.  The
```

`inspect.cleandoc` and Sphinx both key the common indent off the non-blank
lines, so a column-0 line makes the whole docstring's common prefix zero and
every other paragraph renders with four leading spaces (a literal block).

**Exact edit** -- indent line 5300 by four spaces:

```
    ``'zero'`` IS THE DEFAULT, AND WHAT THAT DOES NOT CHANGE.  The
```

### D4 (documented number, item 2) -- the floor binds AT N = 1706, not past it

Four places say "any square grid past N = 1706".  The floor is `Ny*Nx*176`
bytes, so at the shipped `mem_budget_mb = 512.0`:

| N | floor | binds? |
|---|---|---|
| 1705 | 511.636400 MB | no |
| **1706** | **512.236736 MB** | **yes** |

Reproducer (both builds, identical):

```
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=C:/tmp/lum_vc5 python -c \
  "from lumenairy.propagators.gbd import _dense_budget_floor_bytes as f; \
   print(f(1705,1705)/1e6, f(1706,1706)/1e6)"
511.6364 512.236736
```

**Exact edits** -- in all four, replace `any square grid past N = 1706` with
`any square grid from N = 1706 up`:

* `lumenairy/propagators/gbd.py:1626`
* `CHANGELOG.md:141`
* `tests/unit/test_c5_three_defaults.py:421` (docstring of
  `test_a_budget_below_the_floor_is_loud_and_names_the_floor`)
* `Migration-Guide.md:1860-1861` -- `the floor binds on any square grid past
  N = 1706` -> `the floor binds on any square grid from N = 1706 up`

The new id
`test_verify_c5_three_defaults.py::test_the_default_budget_binds_at_N_1706_and_the_call_still_completes`
derives the crossing at runtime from the shipped default and the two
constants, and asserts it is 1706, so the number cannot rot silently again.

### D5 (docstring, item 3) -- `E(u + period) == E(u)` is not literally true on the Collins readout

`_fill_readout_replicas`'s docstring justifies the fill's geometry with

> Both public readouts finish on
> `angular_spectrum_propagate_mft`, whose reconstruction obeys
> `E(u + period) == E(u)` identically in ABSOLUTE output coordinates.

but the function also serves `_collins_focus_readout`, which does not finish
on that transform.  Measured on this verification's fixture (1.55 um,
R -25 mm, `dx_in` 5 um, 256-sample window at 1.60 periods, shift 160 samples =
exactly one period):

| readout | complex residual / peak | modulus residual / peak |
|---|---|---|
| `carrier_referenced_focus_readout`, on axis / 0.30 p off | 1.4512e-14 / 1.1317e-13 | 1.4508e-14 / 1.9514e-14 |
| `carrier_referenced_exact_focus_readout` | 1.7664e-13 | 1.7186e-13 |
| **`_collins_focus_readout`**, on axis / 0.30 p off | **1.3878e-04 / 6.9577e-03** | **1.2360e-14 / 2.0128e-14** |

The Collins readout obeys instead

    E(u + period) = exp(i [2 pi u / dx_in + pi lambda z / dx_in^2]) E(u)

-- its post-chirp is quadratic in the ABSOLUTE output coordinate, so it
contributes a phase that is unity only where `u / dx_in` is an integer.
Verified to **7.1698e-09** (Windows) and 7.1737e-09 (WSL) over 377 sample
pairs -- `validation/probe_verify_c5/v_item3_periodicity.py`, whose JSON on
both builds carries every cell of the table above.  On this fixture
`pi lambda z / dx_in^2` is `1550 pi` and drops out, and the core and its own
replica happen to land on `u/dx_in` integers -- which is why they agree to
3.0e-11 and why an on-axis reading of the peak never sees it, while a wing
pair at 0.4 % of the peak differs by 1.66 rad of phase.

**Behaviour is unaffected.**  The fill's mask is field-independent and built
from the period alone, and "a replica is a FULL-AMPLITUDE image of the core"
-- the statement every reduction the fill protects depends on -- is the
MODULUS statement, which holds to 2.0e-14 on all three readouts.

**Exact edit** -- in `_fill_readout_replicas`'s docstring replace

```
    Both public readouts finish on
    :func:`~lumenairy.propagators.mft.angular_spectrum_propagate_mft`, whose
    reconstruction obeys ``E(u + period) == E(u)`` identically in ABSOLUTE
    output coordinates.
```

with

```
    Both public readouts finish on
    :func:`~lumenairy.propagators.mft.angular_spectrum_propagate_mft`, whose
    reconstruction obeys ``E(u + period) == E(u)`` identically in ABSOLUTE
    output coordinates (measured 1.5e-14 and 1.8e-13).
    :func:`_collins_focus_readout` shares the GEOMETRY and not that last
    statement: its post-chirp is quadratic in the absolute output coordinate,
    so there ``E(u + period) = exp(i[2 pi u/dx_in + pi lambda z/dx_in^2])
    E(u)`` -- the MODULUS is periodic to 2.0e-14 and the complex field only
    where ``u/dx_in`` is an integer (measured 2026-09-20, VERIFY-WP-C5).  What
    this function is about is unaffected: the mask is field-independent, and
    "a replica is a full-amplitude image of the core" is the modulus
    statement.
```

### D6 (Migration note and CHANGELOG, item 1) -- "who is affected" is stated as a distance, and the rule is a band

This is the only defect with a consequence for a caller.

`Migration-Guide.md`, the `gap_kernel='auto'` subsection of the 5.49.0
section:

> **Who is affected.**  Only legs whose reduced frame `z_eff = B/A` is large,
> i.e. legs close to the carrier's own `A = 0` plane. ... the rule fires
> within **23.5 um** of the focus and nowhere else

and `CHANGELOG.md`, the item-1 entry:

> Because the departure is linear in `|z_eff|`, the rule is a NEAR-FOCUS rule
> with a closed-form band

Both are false as scope statements, and the library's own suite shows it.  The
criterion is `sqrt(3/2) k |z_eff| theta_env^4 / 8 > tau`, in which a WIDE
envelope substitutes for a large `z_eff` at the FOURTH power.  Measured here
on both builds:

| leg | \|z_eff\| | `theta_env` (rad) | distance to its `A = 0` plane | fires? |
|---|---|---|---|---|
| VERIFY-B4 F3, 1 um short of focus | 1600 m | 1.1289e-03 | 1 um | yes |
| **`test_audit2609_b4_collins_transport.py` `mismatch_matrix`, fr 0.90** | **0.180 m** | **5.5712e-03** | **2.00 mm** | **yes** |
| this verification's relay, fr 0.85, readout at 0.25 of the focal distance | **0.00778 m** | 1.10e-02 | **9.00 mm**, and 48.7 Rayleigh ranges from the beam's focus | **yes** |
| the same relay at fr 0.80, FURTHER from `A = 0` | 0.00727 m | 8.79e-03 | 11.0 mm | no |

The third row is not near any focus by any reading; the fourth is further from
`A = 0` than the third and does not fire, so the ordering is the opposite of a
distance rule.  A caller with a mismatched-carrier relay -- which is exactly
what `mismatch_matrix` is -- would read the current Migration note, conclude
that only near-focus legs move, and be wrong.

The CHANGELOG's blast-radius statement ("2 keys move, 46 are byte-identical,
and the two are F3's 1 um and 10 um rungs") is true of that digest set and,
read as scope, understates it: the suite census (reproduced independently
here: 167 legs over 48 ids, 24 fallbacks) finds one more, in `mismatch_matrix`.

**Exact edits.**

1. `Migration-Guide.md`, the **Who is affected** paragraph -- replace

```
**Who is affected.**  Only legs whose reduced frame `z_eff = B/A` is large,
i.e. legs close to the carrier's own `A = 0` plane.  The threshold is closed
form -- `|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)` -- so a design can be
checked without running anything.  Measured: on the VERIFY-B4 F3 fixture
(`w = 0.3 mm`, `lambda = 1.064 um`) the rule fires within **23.5 um** of the
focus and nowhere else; on the Wave-5 hygiene-2 fixture it fires within
**1.54 um** of that carrier's `A = 0` plane, which its published ladder never
reaches.
```

with

```
**Who is affected.**  Legs for which `k |z_eff| theta_env^4` is large -- a
NEAR-FOCUS condition at a fixed envelope angle (`z_eff = B/A` grows without
bound as a leg approaches the carrier's `A = 0` plane) and a WIDE-ENVELOPE
condition at a fixed distance, because `theta_env` enters at the FOURTH power.
Both happen in practice: a carrier mismatched to its beam leaves a residual
lens on the envelope and makes `theta_env` large on a leg that is nowhere near
a focus.  The threshold is closed form --
`|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)` -- so a design can be checked
without running anything, and `theta_env` is `2 sqrt(<theta^2>)` of the
ENVELOPE's own sampled spectrum, not the beam's angle.  Measured: on the
VERIFY-B4 F3 fixture (`w = 0.3 mm`, `lambda = 1.064 um`, `theta_env`
1.13e-03 rad) the rule fires within **23.5 um** of the focus and nowhere else;
on the Wave-5 hygiene-2 fixture within **1.54 um** of that carrier's `A = 0`
plane, which its published ladder never reaches; and on a 0.90-mismatched
NA-0.05 relay (`theta_env` 5.57e-03 rad) it fires **2.00 mm** from the `A = 0`
plane -- the one leg in the library's own test suite that this flip moves.
```

2. `CHANGELOG.md`, the item-1 entry -- replace

```
Because the departure is linear in `|z_eff|`, the rule is a NEAR-FOCUS rule
with a closed-form band: it fires only where
`|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)`.
```

with

```
Because the departure is linear in `|z_eff|` and QUARTIC in `theta_env`, the
rule is a band in `k |z_eff| theta_env^4` -- near-focus at a fixed envelope
angle, wide-envelope at a fixed distance.  It fires exactly where
`|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)`.
```

3. `CHANGELOG.md`, after the item-1 blast-radius paragraph -- append

```
Over the library's own suite the reach is one leg wider than that digest set.
A pytest census of every leg the `k4` gate resolved to `'exact'` across the 27
carrier / traced-chain files (167 legs over 48 ids) finds 24 fallbacks: 23 in
the four ids that exist to exercise the rule, and one in
`test_audit2609_b4_collins_transport.py`'s `mismatch_matrix` fixture
(`|z_eff|` 0.180 m, `theta_env` 5.5712e-03 rad, departure 1.2733e-04 = 1.27
tau) -- a WIDE-ANGLE leg 2.00 mm from its own `A = 0` plane, not a near-focus
one.  That file stays green: its gate is `|peak - 1| < 1e-4` on a column
reading 1.000000 to 0.999938.
```

### D7 (report arithmetic, item 1) -- the census attribution

`fixes/WP-C5_THREE_DEFAULTS_REPORT.md`, "The item-1 census, over the suite":

> Twenty-three are in the three ids that exist to exercise the rule -- twenty
> of those are the bisection inside
> `test_c5_three_defaults.py::test_the_rule_fires_only_inside_the_band_the_law_predicts`

The branch's own `item1_census_win.json` says FOUR ids and NINETEEN, and this
verification's independent census plugin reproduces both exactly:

```
  1  test_audit2609_b4_collins_transport.py::TestGateBMismatchMatrix::test_the_sziklas_column_reproduces_the_published_matrix
  1  test_c5_three_defaults.py::test_the_fallback_replaces_the_departure_with_the_oracle_floor
 19  test_c5_three_defaults.py::test_the_rule_fires_only_inside_the_band_the_law_predicts
  2  test_wave5_h2_near_focus_table.py::test_tau_1e_4_leaves_this_ladder_inert_and_catches_f3
  1  test_wave5_h2_near_focus_table.py::test_the_departure_law_predicts_what_the_refinement_actually_changes   (kernel_asked = None, the direct call, correctly counted out)
  1  test_wave5_h2_near_focus_table.py::test_the_shipped_default_arms_the_rule_at_tau_1e_4
```

**Exact edit** -- "three ids" -> "four ids", "twenty of those" -> "nineteen of
those".

### D8 (report scope, item 2) -- the 2e-17 figure is grid-specific

The report states the `'legacy'`-vs-`'measured'` difference as "2e-17
relative".  On this verification's grids it reads **1.799e-16** (N = 320) and
7.260e-17 (N = 512), identical on both builds -- the same last-bit character,
an order of magnitude above the quoted number.  The branch's test bar is 1e-12
and is unaffected either way.

**Exact edit** -- in the report's "What the flip costs, and byte identity"
section, add: `the figure is grid-specific: 1.8e-16 at N = 320 and 7.3e-17 at
N = 512 (VERIFY-WP-C5, both builds), so the claim is "the last bits", not
"2e-17"`.

### D9 (test coverage, item 3) -- the "refusal before the fill" demonstration covers one readout of three

`WP-C5_THREE_DEFAULTS_REPORT.md`'s mutation matrix says of its three
mutations:

> All three aim at one helper, `_assert_confined`, which is also what the
> non-mutated ids assert, so a mutation that slipped past the helper would
> slip past the real ids too and the matrix would be measuring nothing.

For the two mutations that key the fill that is exactly right.  The third --
"the fill applied on the REFUSAL path", demonstrated by
`test_the_refusal_is_taken_before_the_fill_is_reached` -- does not aim at that
helper, it aims at a CALL SITE, and there are three of them
(`lumenairy/propagators/carrier.py` lines 2772, 4493 and 7235).  The id
exercises the one at 4493.

Measured by mutating each call site in turn (waiving its `on_replica`) and
running BOTH C5 test files, 43 ids:

| call site | readout | ids that caught it, of 43 |
|---|---|---|
| 4493 | `carrier_referenced_focus_readout` | 2 (`test_the_refusal_is_taken_before_the_fill_is_reached`, `test_the_replica_refusal_is_unchanged_by_the_fill`) |
| 2772 | `_collins_focus_readout` | **1**, and it is this verification's `test_an_off_axis_window_is_refused_where_the_same_ratio_is_served` |
| 7235 | `carrier_referenced_exact_focus_readout` | **0** |

The library itself is covered on all three -- re-run against wider sets, the
exact readout's call site is caught by
`test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ExactReadout::test_one_period_off_the_chief_ray_is_refused`
and
`test_niche_tight_focus_readout.py::test_the_exact_readout_guards_the_same_way_on_its_own_period`
(control 121 passed, mutant 2 failed / 119 passed), and the Collins one by
`test_audit2609_b4_collins_transport.py::TestKellyGuard::test_the_period_is_the_input_grid_s_and_the_replica_guard_sees_it`
(control 285 passed, mutant 2 failed / 283 passed).  So this is a claim about
the branch's matrix and about what the C5 file set alone would catch, not a
hole in the guards.  It matters because `_collins_focus_readout` is the readout WP-C3 is about to
make the default route: its refusal now rests on ONE pre-existing id in the b4
file (plus this verification's), while the paraxial readout's has a dedicated
fail-before demonstration in the C5 file itself.

**Exact edits.**

1. In `WP-C5_THREE_DEFAULTS_REPORT.md`, the "(f) The mutation matrix"
   paragraph -- replace

```
All three aim at one helper, `_assert_confined`, which is also what the
non-mutated ids assert, so a mutation that slipped past the helper would slip
past the real ids too and the matrix would be measuring nothing.
```

with

```
The first two aim at one helper, `_assert_confined`, which is also what the
non-mutated ids assert, so a mutation that slipped past the helper would slip
past the real ids too and the matrix would be measuring nothing.  The third
aims at a CALL SITE rather than at the helper, and there are three of them
(carrier.py 2772 / 4493 / 7235); this id exercises the paraxial readout's
(4493).  The Collins readout's (2772) is covered by
`tests/unit/test_verify_c5_three_defaults.py::test_an_off_axis_window_is_refused_where_the_same_ratio_is_served`
and the exact readout's (7235) by
`test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ExactReadout::test_one_period_off_the_chief_ray_is_refused`
and
`test_niche_tight_focus_readout.py::test_the_exact_readout_guards_the_same_way_on_its_own_period`
(VERIFY-WP-C5, measured per call site).
```

2. Optional but cheap: give `test_the_refusal_is_taken_before_the_fill_is_reached`
   a second arm on `CA._collins_focus_readout`, with the same raising
   sentinel and the same counter-pin, so the C5 file itself covers the
   transport WP-C3 is about to default to.
---

## Durability

### The decision tests

`tests/unit/test_verify_c5_three_defaults.py`, 18 ids, every bar derived at
runtime, every claim two-sided, each id under 60 s (slowest 14.8 s including a
module-scoped oracle build).  They are in `.test_durations` (18 ids spliced,
52.54 s total; the staleness gate's four ids pass).

| id | gap it closes |
|---|---|
| `test_the_rule_is_a_band_in_k_z_theta4_and_not_a_distance_to_a_focus` | the rule's SCOPE: on one fixture family the leg furthest from its `A = 0` plane falls back and the two closer ones do not |
| `test_the_rule_fires_on_a_leg_that_is_near_no_focus_at_all` | the same, with the readout moved off the beam's focus as well (48.7 Rayleigh ranges, 9.00 mm from `A = 0`) |
| `test_an_explicit_exact_survives_the_band_on_the_wide_angle_leg` | the opt-out on the leg the suite actually moves |
| `test_on_the_wide_angle_leg_the_fallback_is_not_a_step_away_from_physics` | the WP-C5 report's "could not be measured": a NON-PARAXIAL oracle, with its own two-sided validation |
| `test_the_closed_form_band_holds_on_a_third_fixture` | the band on a fixture neither WP-C5 nor VERIFY-B4 used |
| `test_the_flip_bounds_the_resident_set_and_not_only_tracemalloc` | the second instrument |
| `test_the_default_budget_binds_at_N_1706_and_the_call_still_completes` | "warn, not refuse" checked where it binds, on a DEFAULT call |
| `test_a_floor_that_forgot_its_fixed_term_would_be_caught` | the floor mutation, applied to the module |
| `test_the_zeroed_set_is_one_period_about_the_field_origin[x030,y030,xy,edge,wide,ana2,ana4]` | the misapplication case on 7 windows, with the period measured EMPIRICALLY |
| `test_keying_the_fill_on_the_window_centre_would_be_caught` | the mutation the maintainer's concern names |
| `test_blanking_the_exact_readouts_faithful_band_would_be_caught` | the same class of mistake on the OTHER public readout |
| `test_an_off_axis_window_is_refused_where_the_same_ratio_is_served` | the refusal on the OFFSET axis, under both fills |

**They fail, not skip, on a silent regression.**  Run against the PRE tree
(`git archive 49ddf4bd`, i.e. all three defaults reverted at once):
**16 of 18 FAILED, 2 passed** in 63.6 s.  The two that pass are the two whose
subject is a mechanism the flip does not move
(`test_keying_the_fill_on_the_window_centre_would_be_caught`, which drives
`_fill_readout_replicas` directly with explicit fills, and
`test_an_off_axis_window_is_refused_where_the_same_ratio_is_served`, whose
claim IS that the refusal is fill-independent).  No id skipped on any arm.

### The mutation matrix

Twelve source mutations applied one at a time to a scratch copy of the branch
tip (`C:/tmp/vc5_mut` and `/mnt/c/tmp/vc5_mut_wsl`, never the worktree; the
driver restores the file in a `finally` and the tree is diffed against the
worktree afterwards), each scored against BOTH C5 test files
(`test_c5_three_defaults.py` + `test_verify_c5_three_defaults.py`, 43 ids).
Control: **43 passed** on both builds.  **The two builds agree cell for cell
on every row, and on the SET of ids that caught each mutation, not only on the
counts** (`v_mutations_win.json` / `v_mutations_wsl.json`).

| mutation | ids that caught it (Windows) | ids that caught it (WSL) |
|---|---|---|
| item 1: `_GAP_KERNEL_ACCURACY_TAU` back to `None` | **9** | **9** |
| item 1: the rule keyed on the CONTAINMENT half-angle instead of the analytic one | **4** | **4** |
| item 2: `_dense_budget_floor_bytes` loses its fixed term | **4** | **4** |
| item 2: the floor notice suppressed | **2** | **2** |
| item 2: `_dense_cell_bytes` falls through to `'legacy'` on an unknown mode | **1** | **1** |
| item 2: `DENSE_MEM_BUDGET_ACCOUNTING` back to `'legacy'` | **2** | **2** |
| item 3: the fill keyed on the WINDOW's centre (`centre_out` dropped) | **9** | **9** |
| item 3: the fill keyed on HALF the period | **16** | **16** |
| item 3: `replica_fill` back to `'repeat'` on all three readouts | **6** | **6** |
| item 3: the refusal waived on `_collins_focus_readout` | **1** | **1** |
| item 3: the refusal waived on `carrier_referenced_focus_readout` | **2** | **2** |
| item 3: the refusal waived on `carrier_referenced_exact_focus_readout` | **0** | **0** |

**11 of 12 caught by the two C5 files; the twelfth is a SCOPE finding, not a
library defect.**  The branch's matrix demonstrates "the refusal is taken
before the fill is reached" on ONE of the three readouts
(`carrier_referenced_focus_readout`, through
`test_the_refusal_is_taken_before_the_fill_is_reached`), and the two C5 files
between them cover two of the three:

* `_collins_focus_readout` -- the readout the chain reaches on
  `transport='collins'`, i.e. the one WP-C3 is about to make the default --
  is caught by exactly ONE id in the 43, and it is this verification's
  (`test_an_off_axis_window_is_refused_where_the_same_ratio_is_served`).
  Before this file there was nothing in the C5 set that would have noticed a
  waived Collins refusal.  Re-run against a wider set (8 files, 285 ids) it IS
  caught, by one pre-existing id --
  `test_audit2609_b4_collins_transport.py::TestKellyGuard::test_the_period_is_the_input_grid_s_and_the_replica_guard_sees_it`
  -- alongside this verification's (control 285 passed, mutant 2 failed /
  283 passed).
* `carrier_referenced_exact_focus_readout` is caught by NOTHING in the 43.
  Re-run against a wider set it IS caught, by two pre-existing ids --
  `test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ExactReadout::test_one_period_off_the_chief_ray_is_refused`
  and
  `test_niche_tight_focus_readout.py::test_the_exact_readout_guards_the_same_way_on_its_own_period`
  (control 121 passed, mutant 2 failed / 119 passed) -- so the LIBRARY is
  covered and it is the branch's own matrix whose reach is narrower than its
  prose ("aims at one helper, which is also what the non-mutated ids assert").
  See D9.

The three mutations the WP-C5 report names are all in the table
(floor-without-fixed-term, the notice, the silent `'legacy'`), plus this
verification's own three (the window-centred fill, the wrong angle, the
per-readout refusal split) and the three default reversions.

### The three reds that were this campaign's own

All green on this tree, run explicitly:

```
tests/unit/test_public_api.py::test_installed_metadata_version_matches_source_version PASSED
tests/unit/test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached PASSED
tests/unit/test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines PASSED
```

The first of those is the one the WP-C5 report records as a pre-existing
environmental failure ("the editable install's metadata reads 5.47.0 against a
source `__version__` of 5.48.1").  It does NOT reproduce here: on this box
`importlib.metadata.version('lumenairy')` and `lumenairy.__version__` both read
5.48.1.  So the report's "one pre-existing red" is a property of that agent's
environment, not of the tree, and there are now no known reds on this branch.

---

## Runs

All Windows runs carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 LUMENAIRY_MEM_BUDGET_MB=2048 PYTHONPATH=C:/tmp/lum_vc5`,
`--capture=sys`, `-p no:randomly`.

### Windows py3.14.6 / numpy 2.4.4

| what | result |
|---|---|
| the 27 carrier / traced-chain files + `test_c5_three_defaults.py` + `test_wave5_h2_collins_jax.py`, with the census plugin | **1036 passed, 1 skipped** in 2670 s |
| `test_gbd_feature_complete.py` + `test_wave5_gbd_dense_mem_budget.py` + `test_verify_b14_known_reds.py` + `test_niche_tight_focus_readout.py` + `test_niche_d2_chain_multi.py` + `test_fix_v1_v8_readout_guard_and_standoff.py` + both C5 test files | **169 passed** in 919 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep, 33 files incl. `test_audit_except_budget.py` and `test_niche_audit_w4_input_kind.py` | **1752 passed, 14 skipped** in 242 s |
| `test_verify_c5_three_defaults.py` alone | **18 passed** in 53-88 s |
| `test_audit2609_a15a_durations_staleness.py` after the splice | **4 passed** in 44 s |
| the mutation matrix, 12 mutations (13 pytest runs on a scratch copy) | control **43 passed**; 11/12 caught by the two C5 files, the twelfth caught by two pre-existing ids in a wider set (D9) |
| the three periodicity probes (D5), both builds | Windows and WSL agree to 4 digits on every cell |
| the forced-Collins census | 950 passed, 64 failed, 22 errors (expected: the run re-routes ids that assert `'sziklas'`) |
| `test_verify_c5_three_defaults.py` against the PRE tree | **16 failed, 2 passed** -- the regression gate |

The one skip in the 1036 run is
`test_niche_exact_gap_kernel.py:637: m == 1 identically for a collimated
carrier` -- a mathematical premise, not a resource check.  The 14 skips in the
sweep are pre-existing and none is a resource skip on a C5 path.

### WSL py3.12.3 / numpy 2.4.6

`PYTHONPATH=/mnt/c/tmp/lum_vc5`, `~/lumvenv/bin/python`.

| what | result |
|---|---|
| `test_verify_c5_three_defaults.py` + `test_c5_three_defaults.py` + `test_wave5_h2_near_focus_table.py` + `test_verify_hyg2_round2.py` + `test_wave5_gbd_dense_mem_budget.py` + `test_verify_b14_known_reds.py` + `test_audit2609_a25_carrier_focus_readout.py` + `test_niche_tight_focus_readout.py` + `test_fix_v1_v8_readout_guard_and_standoff.py` | **148 passed** in 327 s |
| `test_verify_c5_three_defaults.py` alone | **18 passed** in 79 s |
| `test_audit2609_b4_collins_transport.py`, PER CLASS (16 classes) | **124 passed across 15 classes**; `TestGateCTwoGroupChain` alone hits the 1500 s guard |

`test_audit2609_b4_collins_transport.py` was run per class because the file
stalls whole on WSL on an idle `multiprocessing` spawn pool -- the pre-existing
condition VERIFY-WAVE5-HYGIENE2 round 2 section 11 records and the WP-C5 report
could not work around.  Per class, **15 of the 16 classes complete (124 ids,
all passing) and the stall localises to exactly one: `TestGateCTwoGroupChain`**,
which is the class whose fixture builds the two-group chain through a worker
pool.  That is a sharper statement than "the file cannot be run whole on a WSL
CI lane": the other fifteen classes can be, and a WSL lane that deselects one
class gets 124 of the file's ids back.

It also closes the WP-C5 report's "the other build could not be asked" on the
one suite leg this flip moves: **`TestGateBMismatchMatrix` passes on WSL**
(3 ids, 4.08 s) and this verification's probe reproduces its leg there to every
digit (`|z_eff|` 0.180000 m, `theta_env` 5.571183e-03, departure
1.273287e-04).

### Walkers and gates

```
python scripts/record_history_fingerprints.py --check    OK: every history document matches its module.
python scripts/check_doc_identifiers.py                  630 API-claiming tokens, 0 unresolved
python scripts/check_source_line_citations.py            declines: the v5.48.1 block carries no source:line citation (exit 2, as on the branch)
python -m ruff check lumenairy/ tests/   (WSL)           All checks passed!
python -m mypy                                           Success: no issues found in 33 source files
```

`validation/` is in `pyproject.toml`'s `extend-exclude`, so neither this
verification's probes nor the WP-C5 probes are in the ruff gate's scope; named
explicitly, mine report 3 import-ordering notes and the WP-C5 probes 5, which
is the directory's existing convention.

---

## Ship recommendation

**SHIP all three items.**  Every load-bearing claim in
`WP-C5_THREE_DEFAULTS_REPORT.md` reproduces on both builds, on fixtures this
verification chose, and the two things the report could not measure both come
out in the branch's favour when measured:

* the one suite leg the near-focus rule moves is moved TOWARD the exact scalar
  field, not away from it (8.212599e-02 against 8.222558e-02), so the rule's
  scope is not a defect;
* the honest dense accounting bounds the process RESIDENT SET and not only
  `tracemalloc` (0.7146x the budget against `'legacy'`'s 5.9924x), and the
  "warn, not refuse" decision is vindicated by a default-path call at N = 2048
  that completes in 601 MB and would have been a hard error under a refusal --
  the same call takes 3013 MB and says nothing on the parent commit.

The nine defects are documentation and test scope.  Only two are worth
gating on: **D6**, a Migration note whose "who is affected" is stated as a
distance to a focus and would mislead a caller with a mismatched-carrier relay
-- that one should be fixed before the release note goes out -- and **D4**, an
off-by-one in a number documented in shipped source (`N = 1706`), which is
cheap and is now pinned by a runtime-derived test.  D9 is a scope finding
about the branch's own mutation matrix, not a hole in the library's guards.
None of the nine changes a line of behaviour and none needs a re-measurement.

One thing for the release manager rather than for this branch: the forced-
Collins census says WP-C3 multiplies item 1's moved-leg count by **4** (24 ->
96) and takes it onto nine chain ids whose departures reach 9.33.  That is a
C3 acceptance item, and the clean way to separate the two flips is to run C3's
archive-to-archive comparison twice, with `tau = 1e-4` and with `tau = None`.

---

## What could not be measured

* **Which kernel is "right" in general.**  The oracle built here is the exact
  scalar (Helmholtz) field of a Gaussian with a parabolic carrier, which is
  closed form.  It settles the b4 leg and the wide-envelope legs; it does not
  generalise to an arbitrary envelope, and it says nothing about a vector
  field.  What it DOES establish beyond those fixtures is the shape of the
  answer: the refinement lives in the reduced frame on the envelope's angle
  while the leg's non-paraxiality is set by the beam's NA, so at moderate NA
  the two are unrelated quantities and the kernel choice is far below the
  modelling error either way.

* **The nine new Collins legs, physically.**  The forced-Collins census
  measures WHICH legs enter the band and by how much; it does not score any of
  them against an oracle.  At departures of 1e+00 to 9.33 the exact-kernel
  refinement is outside any regime the departure law was fitted in, so even
  the departure figure is an extrapolation there.

* **The multi entry point on an OVERSIZED window.**  `replica_fill` is in
  `_OUTPUT_GRID_PASSTHROUGH` and all three spellings are accepted through
  `propagate_traced_carrier_chain_multi`, with the default returning the
  `'zero'` bytes -- but on the relay fixture used here every window that fits
  in memory was FAITHFUL, so the multi path's fill was verified by passthrough
  plus the readout-level proof rather than end to end.

* **The floor as a bar on the RSS peak.**  Measured (1.2172x / 1.2155x on
  Windows), reported, and deliberately NOT asserted: an in-process RSS
  high-water mark spread 19 % run to run at N = 1024 on WSL and read 4 KB for a
  24.6 MB transient that followed a larger one, so a bar at 1.22x sits inside
  its own noise.  The decision test asserts the flip's 8.4x / 7.0x separation
  instead.

* **A CuPy device.**  `test_niche_k2_carrier_backends.py`'s CuPy ids skip on
  this box for the same reason the WP-C5 report gives.  Neither the kernel rule
  nor the fill was exercised on a GPU array by this verification either.

* **The C3 merge itself.**  The census forces the `transport` DEFAULT; it does
  not merge C3's diff, which also adds a CuPy arm to `_collins_transport`.  The
  leg list above is what the merge should be checked against, not a
  measurement of the merge.

* **The b4 file whole on WSL.**  Run per class (16 classes) because the file
  stalls whole there; that is a pre-existing condition and nothing in this
  branch touches it.  Per-class execution loses nothing here -- the file has no
  module-level tests and the one module-scoped fixture that matters
  (`mismatch_matrix`) is built inside `TestGateBMismatchMatrix`'s own class run.

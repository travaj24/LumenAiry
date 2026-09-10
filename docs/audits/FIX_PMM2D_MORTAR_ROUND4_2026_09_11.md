# FIX -- round 4 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_mortar4`, branch
`fix/pmm2d-mortar-round4`, created from `15af675` (the wave2/pmm2d tip, which
carries round 3 and its verification)

**Under repair** the two P3 defects and the two durability flags of
`docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md` -- S8 DEFECT 1
("either side promoted" over-states the mechanism), S8 DEFECT 2 (the band
warning names an axis that carries no mortar), and the two SAMPLE-scoped bars
S14 calls out -- plus the release-note line S9 asks for.

**Method** every number below was measured in this worktree by
`validation/probe_fix_mortar_round4/`, on BOTH builds except where S6 says
otherwise.  The DEFECT-2 probe runs round 3's rule and round 4's rule in the
SAME interpreter, so the two arms differ in the change and in nothing else.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. What changed

| # | item | verdict |
|---|---|---|
| 1 | DEFECT 1 -- the mechanism needs an ASYMMETRIC interface, EXACTLY ONE promoted side | wording corrected in the constant, the site comment, the refusal message a user reads, the round-3 fix doc and the CHANGELOG.  **No behaviour change** |
| 2 | DEFECT 2 -- the band warning now asks "does THIS AXIS carry a mortar" | `_stag_mortared_axes` returns an `(x, y)` pair; `_stag_band_narrowest` searches only live axes.  **801 leaves compared across the change on WIN and 801 on WSL: 0 answer-hash differences, 0 other differences, 23 warning differences on each** -- and every one of the 23 is the false positive stopping or the correct axis being named |
| 3 | S14 -- `max(ctrl.on) < 0.95` | restated as a SPREAD comparison re-measured on both sides each run; 4.3 decades / 1.5e+04x / 8.4x of margin against 0.015 before |
| 4 | S14 -- `e_band > 1.15 * e_ord` at `M` = 6 | the gate now runs `M` = 5 AND 6, VERIFIES from the ladder that the ordinary arm is still falling, and requires the ratio at BOTH rungs; worst measured ratio 1.551, i.e. 1.35x |
| 5 | S9 -- "no warning" is not "no degradation" | one sentence in the CHANGELOG, in `_warn_stag_sliver_band`'s docstring and in `_STAG_SLIVER_BAND_FRAC`'s comment, with the census margin restated at its real 1.67x |

---

## S1. Terms, defined before they are used

**Mortar.**  The L2 projection that couples two adjacent per-layer element
grids.  It exists on an interface only where the two layers' wall arrays
DIFFER; where they coincide the interface is the plain square modal match and
no projection happens (`stack2d_pure.py`: `same = ga.key() == gb.key()`).

**A MORTARED AXIS.**  An axis (x or y) on which some adjacent pair of layers
carries different wall arrays.  `StagGridOps.key()` is the fingerprint PAIR
`(fingerprint(bx), fingerprint(by))`, so the x axis is mortared iff the stack
shows more than one x fingerprint -- and if it does, some ADJACENT pair differs
on x, because a sequence whose entries are not all equal has two neighbours
that are not equal.  The set reading and the adjacency reading are therefore
the same statement, which is what makes the one-line test sound.

**The band.**  `[_STAG_MIN_SEG_FRAC, _STAG_SLIVER_BAND_FRAC)` = `[1e-3, 3e-2)`
of the period, both edges carrying 1e-9 relative slack.  Below it a stack is
REFUSED by the width contract; inside it the solve proceeds and WARNS; above it
it is silent.

**Promotion, and an ASYMMETRIC interface.**  `_modes_as_general` rewrites a
symmetric in-plane region's modes as `(W, V, lam, W, -V, -lam)` so they can
enter the generalized cascade that an out-of-plane tensor or a slanted layer
forces on the whole stack.  An interface is ASYMMETRIC when EXACTLY ONE of its
two sides is promoted, and SYMMETRIC when both are.

**The two arms.**  ROUND 3 is the shipped round-3 rule -- ask whether the STACK
builds a cross-grid interface anywhere, then scan BOTH axes of every grid.
ROUND 4 is the per-axis rule.  Both are run in one interpreter: the probe and
the gate replace `_stag_mortared_axes` by the collapsed form
`m = force or len({g.key()}) > 1; return (m, m)`, which reproduces round 3
exactly, because `len({key}) > 1` is precisely `any(per-axis pair)`.

---

## S2. The two builds

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, set in the SHELL | same |
| `lumenairy` | `C:\tmp\lum_mortar4\lumenairy\__init__.py` 5.44.0 | `/mnt/c/tmp/lum_mortar4/...` 5.44.0 |

Both link scipy-openblas, so every cross-build spread here is a LOWER bound --
the same caveat rounds 2 and 3 and both verifications carry.  Every probe
imports `_path.py` first, which puts this worktree at the front of `sys.path`
and REFUSES to run if `lumenairy.__file__` resolves anywhere else; the resolved
path is recorded in every JSON.  (The round-3 verification lost a probe run to
exactly that trap.)

---

## S3. DEFECT 1 -- the mechanism is the ASYMMETRY

**What was wrong.**  Round 3 derived `_MORTAR_RESID_REFUSE` from "the operand
is RANK-DEFICIENT BY CONSTRUCTION whenever ONE side of the interface is an
IN-PLANE region promoted", and repeated it as "whenever either side is" in the
site comment, in the CHANGELOG and in the refusal MESSAGE the user reads.  A
reader deriving a bar from that would mis-predict by five decades.

**The measurement, re-made here on 33 operands and both builds**
(`p2_durability.py --parts p`, over every class the round-3 and verification
fixture families build: an out-of-plane layer beside a uniform spacer, beside
an in-plane patterned layer, beside an in-plane MAGNETIC layer, a slanted layer
beside an in-plane one, a strongly out-of-plane layer, a 1e-8-rad slant, a
three-layer stack whose middle interface has both sides promoted, and the
both-out-of-plane / both-slanted / nearly-in-plane controls):

| class | operands | `s_min/s_max` | near-null `max(on)` | SPREAD `min(on)/max(on)` |
|---|---|---|---|---|
| **exactly ONE promoted** | 21 | 4.3089e-14 .. 1.4856e-10 | 0.9999999999999649 .. 1.0000000000000009 | **1.3735e-08 .. 2.6587e-07** |
| **BOTH promoted** | 3 | 2.4161e-08 .. 2.6310e-05 | 0.8492 .. 0.9139 | **0.4442 .. 0.6218** |
| NEITHER promoted (controls) | 9 | 1.4955e-05 .. 8.1053e-04 | 0.7106 .. 0.9355 | 0.3777 .. 0.9902 |

WIN and WSL agree on every entry of that table to at least **ten significant
figures**.  A both-promoted interface sits five decades above a one-promoted
one in `s_min/s_max` and its near-null direction is not localised at all: it is
on the CONTROLS' side of the divide, not the defect's.

**What ships.**  One word in four places, and it is a word a user sees:

* `_MORTAR_RESID_REFUSE`'s derivation, plus a dated **CORRECTION, ROUND 4**
  paragraph carrying the measurement above rather than a silent rewrite;
* `_guarded_mortar_solve`'s `screen='residual'` bullet;
* the site comment inside `_interface_smatrix_general_mortar_2d`;
* the refusal MESSAGE: "RANK-DEFICIENT BY CONSTRUCTION whenever **EXACTLY ONE**
  side of the interface is an in-plane region promoted...".

`docs/audits/FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md` carries a dated CORRECTION
note at the two places that state it; its measured tables are untouched.  The
CHANGELOG's round-3 paragraph is corrected in place.

**Nothing shipped behaves differently**, and that is measured, not argued: the
residual screen accepts all three classes, and round 3's own healthy population
already contained a both-promoted middle interface.

---

## S4. DEFECT 2 -- the band warning, per AXIS

### 4.1 What was wrong

`_warn_stag_sliver_band(gof, force_mortar or len({g.key() for g in gof}) > 1)`
asked the question of the STACK; `_stag_band_narrowest(grids)` then scanned
BOTH axes of every grid.  A stack whose layers differ on x and share the y wall
array EXACTLY therefore warned about a narrow y segment -- while the y
cross-mass is that axis's own mass matrix, so nothing on y is projected across
grids and nothing on y is degraded.

### 4.2 The change

```py
def _stag_mortared_axes(grids, force=False):     # NEW
    if force:
        return (True, True)
    keys = [g.key() for g in grids]
    return (len({k[0] for k in keys}) > 1, len({k[1] for k in keys}) > 1)

def _stag_band_narrowest(grids, axes=(True, True)):   # ``axes`` is new
    ...
        for ax, b, live in (("x", g.bx, axes[0]), ("y", g.by, axes[1])):
            if not live:
                continue

def _warn_stag_sliver_band(grids, mortared_axes, fn=...):
    ...                                    # a bool is still accepted
    frac, where = _stag_band_narrowest(grids, mortared_axes)
    if where is None:
        return False
```

and the one caller becomes
`_warn_stag_sliver_band(gof, _stag_mortared_axes(gof, force_mortar))`.
`force_mortar` -- the test instrument that drives the mortar algebra where the
grids coincide -- makes both axes live, which is what round 3 did.

### 4.3 (a) The no-mortar axis stops warning, and the answer does not move

The VERIFICATION's own DEFECT-2 fixture, reproduced knob for knob (period 1.07,
wavelength 0.79, `theta` 0.31, uniform 2.5 / 3.5 permittivity, x walls
0.25/0.60 against 0.31/0.66, the y pair swept about 0.585).  `M` = 5:

| y width | `R00` | band/ordinary RATIO | ROUND 3 | ROUND 4 |
|---|---|---|---|---|
| 3.0e-01 | 0.247088457739 | 1.000000000000 | -- | -- |
| 1.0e-01 | 0.247088457739 | 1.000000000000 (1.7e-14) | -- | -- |
| 5.0e-02 | 0.247088457739 | 0.999999999999 (8.5e-13) | -- | -- |
| 3.0e-02 | 0.247088457739 | 1.000000000000 (2.8e-13) | -- | -- |
| **1.0e-02** | 0.247088457739 | 1.000000000000 (4.2e-13) | **warns, y, 1.000e-02** | **silent** |
| **3.0e-03** | 0.247088457739 | 1.000000000000 (3.1e-13) | **warns, y, 3.000e-03** | **silent** |
| **1.2e-03** | 0.247088457739 | 1.000000000000 (3.9e-13) | **warns, y, 1.200e-03** | **silent** |

WSL reproduces `R00` to all twelve printed figures and reads the moves at
**6.4e-15 .. 1.3e-13**, against WIN's 1.7e-14 .. 8.5e-13; both give a
band/ordinary ratio of 1.000000000000 at every rung.  The same ladder at
`M` = 4 reads 1.0e-10 .. 3.6e-10 -- the residue is the discretisation, not the
wall separation.  **The answer's sha256 is IDENTICAL between the two arms at
every rung, on both builds.**

The message claimed "about 4-5x the error of the same device on an ordinary
partition".  Stated as the ratio the message is about, it is
**1.000000000000** -- with one WIN rung (5e-2) reading 0.999999999999, an
8.5e-13 move.  The claimed factor is not merely small here; it is absent to
round-off.

A control on the same fixture with x made conforming TOO -- so the stack builds
no mortar at all -- is silent in both arms at every rung, which is the round-3
behaviour this change must not disturb.

### 4.4 (b) Every warning that fired for a real reason still fires

Both surfaces the census identifies as reaching the band carry their narrow
segment on the axis that IS mortared, and neither moves:

The narrowest segment named below is the one on the MORTARED axis; both
fixtures carry an ordinary, SHARED y wall array (narrowest 2.7000e-01), which
is exactly the shape round 3 could have warned about and did not need to.

| fixture | narrowest on x (mortared) | ROUND 3 | ROUND 4 |
|---|---|---|---|
| closing taper, 8 slices | 3.1250e-02 (1.04x OUTSIDE the edge) | silent | silent |
| **closing taper, 9 slices** | 2.7778e-02 | **1, x, 2.778e-02** | **1, x, 2.778e-02** |
| **closing taper, 16 slices** | 1.5625e-02 | **1, x, 1.562e-02** | **1, x, 1.562e-02** |
| **closing taper, 32 slices** | 7.8125e-03 | **1, x, 7.812e-03** | **1, x, 7.812e-03** |
| round-2 `delta` sweep, 0.30 | 2.1000e-01 | silent | silent |
| **round-2 `delta` sweep, 1e-2** | 1.0000e-02 | **1, x, 1.000e-02** | **1, x, 1.000e-02** |

Identical on WSL, and every answer hash identical across the arms.

### 4.5 (c) Mixed -- the fix SELECTS an axis, it does not suppress

| fixture | ROUND 3 | ROUND 4 |
|---|---|---|
| x mortared at 9e-3, y CONFORMING at 3e-3 | 1 warning, **y**, 3.000e-03 | 1 warning, **x**, 9.000e-03 |
| the same widths, y mortared too (CONTROL) | 1 warning, y, 3.000e-03 | 1 warning, y, 3.000e-03 |
| x ORDINARY at 2e-1, y CONFORMING at 3e-3 | 1 warning, y, 3.000e-03 | **silent** |

The control is what separates "names x" from "always names x": when y really
does carry a mortar and really is the narrowest, both rules name it.  One
warning per SOLVE either way.

### 4.6 (d) The ordinary census stays at 0 warnings

47 geometries built through the public builders -- the ROUND-2 battery (15) and
the round-3 VERIFICATION's own census (32: uniform lattices `N` = 1..16, duty
0.1..0.9 pillars, a nested refinement, two pillars, straight and CLOSING tapers
at 4..128 slices, `add_tapered_pillars` at 4..32).  38 are ORDINARY (the
closing tapers are the surface the band exists for and are scored separately):

* **0 land in the band on the round-3 reading and 0 on the round-4 reading**,
  on both builds;
* the per-axis narrowest is `>=` the all-axes narrowest for **every one of the
  47**, which is why the change can never ADD a warning -- it searches a
  SUBSET of the axes.  Asserted geometry by geometry, not argued;
* the narrowest ORDINARY geometry is **5.0000e-02, a duty-0.9 pillar**, =
  **1.67x** the 3e-2 edge (S5.3);
* the eight ordinary geometries closest to the edge were SOLVED in both arms:
  **0 band warnings in 16 solves**, every answer hash identical.

Four geometries were skipped for cost and the skip is recorded in the JSON: a
16-cell uniform lattice is a 4608-dimension dense region eigenproblem at
`M` = 4 (the 8-cell one, which WAS solved, is 1152 and costs 200 s a solve),
and `add_tapered_pillars` at 16/32 slices is 16/32 layers.  All four are read
geometrically and all four sit above the edge.

### 4.7 (e) Bit-identity

`p1_axis_band.py` flattens both arms' whole result tree and compares every
leaf:

| | WIN | WSL |
|---|---|---|
| leaves compared | 801 | 801 |
| **answer-hash differences** | **0** | **0** |
| warning differences | 23 | 23 |
| other differences (`R00`, closure, census geometry, ...) | **0** | **0** |

The 23 are: **18** in part (a) -- the three banded rungs at `M` = 4 and again
at `M` = 5, each contributing `n_warn`, the axis and the width, 3 x 2 x 3;
**3** on part (c)'s x-ordinary stack (the same three fields); and **2** on part
(c)'s mixed stack, where the axis moves `y -> x` and the width `3.000e-03 ->
9.000e-03` while the COUNT stays 1.  Every one of the 23 is a field of a
WARNING.  No field of an ANSWER moves, on either build.

---

## S5. The durability items (S14 of the verification)

### 5.1 `max(ctrl.on) < 0.95` -- restated as a SPREAD, family-scoped

**What was wrong.**  The bar has 0.12 of slack on the gate's own control but
only **0.015** against a legitimate neighbouring fixture: an out-of-plane
tensor that is in-plane to 1e-9 reads `max(on)` = 0.9355.  It is a fixed number
on one side of a comparison whose other side is re-measured.

**What ships.**  The DECISION quantity becomes the SPREAD
`min(on_a, on_b) / max(on_a, on_b)` -- scale-free, and re-measured on BOTH
sides of the comparison every run:

```py
assert s_mixed < 1e-3
assert s_ctrl > 1e3 * s_mixed
assert s_ctrl > 0.1
```

| bar | population it is measured against | margin |
|---|---|---|
| `s_mixed < 1e-3` | 21 one-promoted operands: 1.3735e-08 .. 2.6587e-07 | **3.6 decades** (worst of both builds) |
| `s_ctrl > 1e3 * s_mixed` | the same two operands, both re-measured | **1.5e+04x** on the gate's own fixtures (0.8392 against 5.4975e-05) |
| `s_ctrl > 0.1` | 9 neither-promoted + 3 both-promoted operands: 0.3777 .. 0.9902 | **3.8x** on the worst legitimate reading, 8.4x on the gate's own control |

The same restatement is applied to the verification's own G1, whose
`max(f_both.on) < 0.95` had 0.066 of slack; it now reads `f_both["spread"] >
1e3 * f_one["spread"]` and `> 0.1`, with 0.4442 measured.

### 5.2 `e_band > 1.15 * e_ord` at `M` = 6 -- two rungs, and the precondition measured

**What was wrong.**  A ratio at ONE modal rung is contaminated whenever the
ORDINARY arm has not converged, and the verification's independent fixture
reads 1.30 at 3e-3 and **0.18** at 1.5e-1 for exactly that reason.  The gate
asserted 1.15 against a measured 1.55 on a rung it ASSUMED was clean.

**The ladder, re-measured on the round-3 gate's own fixture**
(`p2_durability.py --parts l`; the oracle is the exact 1-D `PMMStack` at
degree 12 and 14, whose self-gap is **1.491151e-05**, three decades under the
errors it is used to compare):

| `M` | err(3e-01) ORDINARY | err(3e-03) BAND | ratio | ordinary arm's fall |
|---|---|---|---|---|
| 5 | 8.333039e-02 | 1.342166e-01 | **1.6107** | -- |
| 6 | 2.262084e-02 | 3.507489e-02 | **1.5506** | 3.683x |
| 7 | 5.225545e-03 | 8.558415e-03 | **1.6378** | 4.329x |
| 8 | 4.678401e-04 | 2.252256e-03 | **4.8142** | 11.17x (WIN only) |

**WSL reproduces the first three rungs to SEVEN significant figures** -- it is
a pure discretisation quantity, which is what makes a ratio bar on it a
decision rather than noise.  The `M` = 8 rung reproduces the round-3 fix's own
4.81 reading and is the rung where the ordinary arm has fallen below the
floor, so the cost is fully visible; it costs 327 s and stays in the probe.
The ORDINARY arm falls monotonically at every rung of this fixture, which is
the property the new precondition tests -- and the property the verification's
independent fixture did not have at `M` = 6.

**What ships.**  The gate runs `M` = 5 AND 6, and:

1. the oracle precondition stays: `self_gap < 0.05 * |e_band - e_ord|`
   (1.4912e-05 against 1.2454e-02 = **840x**, asserted at 20x);
2. a NEW precondition taken from the ladder rather than assumed -- the ordinary
   arm must still be FALLING with the modal count, or the baseline is a floor:
   `e_ord(6) < 0.5 * e_ord(5)`, measured 0.271 (a 3.683x fall), **1.84x** of
   margin;
3. the decision at BOTH rungs: `e_band > 1.15 * e_ord`, worst measured
   **1.5506**, i.e. **1.35x**.

`M` = 7 is 121 s a rung and stays in the probe.

### 5.3 The census margin is 1.67x, and it is SAMPLE-scoped

The round-3 constant says "3.6x below the narrowest ordinary geometry".  That
is a property of the ROUND-2 BATTERY, which contains no high-duty pillar and no
fine uniform lattice.  Over the 47-geometry census of S4.6:

| geometry | narrowest / period | x the 3e-2 edge |
|---|---|---|
| **duty-0.9 pillar** | **5.0000e-02** | **1.67** |
| uniform lattice `N` = 16 | 6.2500e-02 | 2.08 |
| uniform lattice `N` = 12 | 8.3333e-02 | 2.78 |
| two pillars / duty-0.1 pillar | 1.0000e-01 | 3.33 |
| nested refinement | 1.2500e-01 | 4.17 |

A 5 %-wide trench either side of a pillar is an entirely ordinary photonic
feature, and a 94 %-duty pillar would sit ON the edge.  Zero ordinary
geometries land in the band on either build, so the EDGE stands; what does not
stand is reading 3.6x as a property of the library.  Recorded in
`_STAG_SLIVER_BAND_FRAC`'s own comment and gated at 1.25x (1.33x of slack over
the measurement) by
`test_no_ordinary_geometry_lands_in_the_band_on_a_mortared_axis`, which also
asserts that the wider census reaches INSIDE the round-3 gate's 3x -- otherwise
it would not be testing the scope.

### 5.4 "No warning" is not "no degradation"

The band's upper edge is set by the FALSE-POSITIVE census, not by where the
accuracy loss begins.  On the verification's independent fixture the error has
already grown **13.5x** (`M` = 7) / **22.9x** (`M` = 8) by the time a segment
reaches the edge (3e-2), and grows only a further **1.24x** from there to the
width contract -- about 92 % of the damage, on a log scale between 3e-1 and
1e-3, happens SILENTLY above the edge.  That is a consequence of the census
constraint, not a defect; but the documentation invited the wrong reading.  The
sentence now appears in the CHANGELOG's round-3 paragraph, in
`_warn_stag_sliver_band`'s docstring and in `_STAG_SLIVER_BAND_FRAC`'s comment,
and is gated by
`test_the_band_documents_that_no_warning_is_not_no_degradation`.

---

## S6. What could NOT be measured

1. **A THIRD BLAS FAMILY.**  Both builds link scipy-openblas.  Every
   cross-build spread here is a LOWER bound -- the caveat rounds 2 and 3 and
   both verifications carry.
2. **The `M` = 8 ladder rung on WSL.**  WIN was run at `M` = 5, 6, 7 and 8;
   WSL at 5, 6 and 7, which reproduce WIN to seven significant figures.  `M` =
   8 is ~10 minutes a rung on a shared box and the three rungs that agree are
   the evidence for the fourth.
3. **A SOLVE of the two fine uniform lattices and the two 16/32-slice
   `add_tapered_pillars` geometries** (S4.6).  All four are read geometrically
   and sit above the edge; the 16-cell lattice is a 4608-dimension dense region
   eigenproblem, which is the same arithmetic the verification uses to BOUND
   that class rather than run it.
4. **The degradation ladder was NOT re-measured on the verification's own
   fixture.**  The 13.5x / 22.9x / 1.24x figures in S5.4 are the
   verification's, quoted as theirs; what is re-measured here is the round-3
   gate's fixture, because that is the one a bar is being restated on.
5. **The whole `pmm2d` surface was not re-run per ARM.**  The two-arm
   comparison covers the probe's 47 geometries and 20 solved fixtures; the
   suites in S7 were run on the round-4 tree only.  What bounds the rest is
   the bit-identity result (S4.7) plus the fact that the change touches only
   which axes a WARNING scans -- there is no path from it to an answer.
6. **The GATE's cost.**  Adding the `M` = 5 rung takes
   `test_the_band_the_warning_names_carries_a_measurable_cost` from 30 s to
   54 s.  That is paid deliberately: it was the thinnest bar in the round-3
   file.  A cheaper form -- asserting the ratio at one rung and the ordinary
   arm's convergence at another -- was not adopted, because the ratio at the
   unverified rung is exactly what the verification refuted.

---

## S7. Runs

See S8 for the commands; every log is kept beside the probes in
`validation/probe_fix_mortar_round4/`.

| run | WIN | WSL |
|---|---|---|
| the six round-2/3/4 mortar + non-uniform + slant suites, plus the new round-4 file, on the FINAL tree (`_run_7suites_{win,wsl}_final.txt`) | **91 passed**, 3 warnings, 834 s | **91 passed**, 3 warnings, 845 s |
| the whole `pmm2d` surface -- `test_pmm2d*.py test_fix_pmm2d*.py test_verify_pmm2d*.py` (`_run_surface_win.txt`) | **378 passed**, 24 warnings, 2395 s | -- |
| census / walker / dispatcher-pin / public-API / doc-consistency sweep (`_run_census_win.txt`) | **1284 passed, 12 skipped**, 265 s | -- |
| `ruff check lumenairy/ tests/ validation/probe_fix_mortar_round4/` | -- | **All checks passed!** |

The warnings on both seven-suite runs are the expected ones, and ONE OF THEM IS
the evidence for S4.4: the round-2 `delta` sweep still raises `layer 1's
element grid has a segment 1.000e-02 of the period wide on the **x axis**` --
on both builds, on the axis that carries the mortar.  Across the whole 378-test `pmm2d`
surface `grep -c "degradation band"` reads **1**, on that gate -- the round-3
verification's reading (1 of 367) unchanged, and now naming the x axis.

---

## S8. Commands

```sh
# every shell command starts with the cd -- a pytest launched elsewhere prints
# "no tests ran in 0.02s" and exits 0.  ONE BLAS thread, on the COMMAND LINE,
# because a module's own os.environ.setdefault is a no-op in a multi-file run.

# probes, both builds
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p1_axis_band.py \
  --tag win --parts abcd
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p2_durability.py \
  --tag win --parts p
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_mortar_round4/p2_durability.py \
  --tag win_ladder --parts l --ladder-M 5,6,7,8
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/lumvenv/bin/python \
  validation/probe_fix_mortar_round4/p1_axis_band.py --tag wsl --parts abcd'

# the six suites plus the new file, BOTH builds
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/test_fix_pmm2d_mortar_round4.py \
  tests/unit/test_fix_pmm2d_mortar_round3.py \
  tests/unit/test_verify_pmm2d_mortar_round3.py \
  tests/unit/test_fix_pmm2d_mortar_round2.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_pmm2d_staggered_nonuniform.py \
  tests/unit/test_verify_pmm2d_perlayer_slant.py -q -p no:randomly

# the whole pmm2d surface, Windows
cd /c/tmp/lum_mortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest tests/unit/test_pmm2d*.py \
  tests/unit/test_fix_pmm2d*.py tests/unit/test_verify_pmm2d*.py \
  -q -p no:randomly

# census / walker / dispatcher sweep
cd /c/tmp/lum_mortar4 && python -m pytest $(ls tests/unit | grep -iE \
  'walker|census|dispatcher_pin|public_api|doc_consistency' | \
  sed 's#^#tests/unit/#') -q -x -p no:randomly

# ruff
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mortar4 && ~/lumvenv/bin/ruff check \
  lumenairy/ tests/ validation/probe_fix_mortar_round4/'
```

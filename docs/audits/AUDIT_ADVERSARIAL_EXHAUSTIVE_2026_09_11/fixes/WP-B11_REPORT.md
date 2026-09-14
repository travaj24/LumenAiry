# WP-B11 (Wave 4, last) -- the hygiene pass, **part a**

Branch `audit-fixes-2026-09`, base HEAD `21110326`.  Scope restriction from the
orchestrator: items **1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
19, 20**, in that order.  Items **5** (the `LensPhysics` dataclass) and **8**
(the `doe.py` sentinel and the warning `stacklevel` sweep) belong to part b and
were not opened.  WP-B7 ran concurrently and owns `lenses_maslov.py`,
`lenses_gbd.py`, `_lens_jax.py`, `propagators/asymptotic*.py`, `fga.py` and
`gbd.py`; none of those was opened for writing here.

## How every refactor was gated

`git archive 21110326 lumenairy` extracted read-only into a scratch tree.  Each
probe runs TWICE in a child process, once with cwd and `PYTHONPATH` set to that
archive and once with them set to the working tree, asserting
`lumenairy.__file__` is under the expected root before it computes anything, and
the two JSON outputs are compared key by key.  Never through pytest (pytest puts
the repo root ahead of `PYTHONPATH`) and never against the shared working tree's
in-place modules (it carries WP-B7's uncommitted edits).  Every hash is SHA-256
over the exact IEEE-754 bytes plus dtype and shape.  Every python run carried
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

Harness: `scratchpad/b11/bi.py` (driver), `probelib.py` (bind + hash),
`probe_item*.py` (per item).

---

## 1. Summary

| # | item | status | files:lines | tests | oracle | measured, before -> after |
|---|---|---|---|---|---|---|
| 1 | one branch band + one selector | **done** | `lumenairy/_branchcut.py` (new, 118); `rcwa/_core.py:1607,1617`; `eme/_branch.py:164`; `pmm/_core.py:795,828,830`; `bor/_orient.py:251,253` | `b11::TestTheBranchCutLeaf` (5); `test_ci_kernel_consistency.py` | archive-vs-tree, 4 kernels x 7 populations + 4 end-to-end engines | **185/185 bit-identical**; kernel census unmoved (7 passed) |
| 2 | `rcwa/_core.py` organisation | **partial** | `rcwa/_geometry.py` (new, 235); `rcwa/_core.py:1-42,130,2192,2204`; `bor/coupled_radial_eigensolver.py:57-76`; `docs/audits/MEASURE_BOR_STAGGERED_WALL_ANCHOR_2026_09_13.md` (new) | `test_rcwa.py`, `test_niche_audit_w8_shapes.py`, `test_niche_audit_w9_overlap_exact.py` (231 passed) | archive-vs-tree, 8 predicates + every raised message + a 2-D shape solve | **13/13 bit-identical**; `_core.py` 5119 -> 4967 lines |
| 3 | three surface bodies -> one band generator | **partial** | `_lens_real.py:2910,2925,6747,6768,7262,7409` | `b11::TestTheRowBandSchedule` (12) | rebuilt `repro/RL-CORE/p10_banded_identity.py`: 5 prescriptions x 7 chunk widths + 3 kw combos x 3 + a carrier arm | **56/56 bit-identical**; banded == whole grid **47/47 in BOTH trees** |
| 4 | `elements/_lens_kernels.py` leaf | **partial** | `_lens_kernels.py` (new, 419); `lenses.py:56`; `_lens_traced.py:597`; `_lens_real.py:150`; `docs/lens_configuration.md` | `b11::TestTheLensKernelsLeaf` (4); the lens slice | archive-vs-tree on all four helpers, both spellings, warning text + filename | **7/7 bit-identical**; family module-level 2-cycles **3 -> 2** |
| 5 | `LensPhysics` dataclass | **not mine** (part b) | -- | -- | -- | -- |
| 6 | `to_kwargs(strict=True)` | **done** | `lens_config.py:1099,1159,1180` | `b11::TestToKwargsStrict` (5) | additive; default path asserted unchanged | silent drop -> refusal, **default unchanged** |
| 7 | 1-D symmetric remap input window | **done** (code was already at HEAD) | test-only: `b11::TestTheSymmetricRemapInputWindow` | 7 tests | a +d / -d pair of DECENTRED INPUT FIELDS on a symmetric element | mirror residual **2.4e-16 .. 2.6e-16** windowed against **6.8e-03 .. 3.1e-02** unwindowed (worst pixel 0.31 of peak) |
| 8 | `doe.py` sentinel / warning stacklevel | **not mine** (part b) | -- | -- | -- | -- |
| 9 | `apply_aperture(edge='gray')` as default? | **measured, recommended, not shipped** | -- | -- | RS spatial kernel + HF OPL quadrature against the closed form | hard order **1.31 / 3.29 / -0.62** (no reliable order) -> grey **2.04 / 2.16 / 1.69**; **21.1x** (RS) and **8.3x** (HF) at N = 1024 |
| 10 | `maslov` family + gap-leg coverage in the a15a array | **done** | `tests/unit/test_audit2609_a15a_lens_covering_array.py` (+199) | 9 new tests (7 array rows + default-identity + the gap-leg gate) | pairwise over 7 maslov factors; 18-kwarg default identity; both in-glass legs | maslov arms finite, no energy gain, **18/18 defaults byte-identical**; the two gap legs gain **1.0397e4** of P_in and only one of them says so |
| 11 | `pmm_jones_2d` double assembly | **done** | `pmm/twod_jones.py:1052,1058,1074` | `b11::TestJones2DAssemblesOnce` (4) | archive-vs-tree over in-plane / off-plane / slanted x normal / oblique x 2 formulations x 2 symmetry | **24/24 bit-identical**; builds per call **2 -> 1** |
| 12 | `sampling=` on the free-space HFPI pair | **done** | `hfpi.py:989-992,1020-1049,1062-1120` | `b11::TestHfpiFreeSpaceSampling` (5) | default byte-identity + both samplers routing + the refusal | default **byte-identical**, `'stratified'` / `'sobol'` route, totals within 1.6 % of the uniform draw |
| 13 | odd-N grid centring, swept | **measured, recommended** | -- | -- | 149-site AST census + `compute_psf(method='fft')` | coordinate-coupled sites **27 of 149**; odd-N offset **exactly -0.5 px**, N-independent |
| 14 | direct-matrix MFT branch | **not reached** | -- | -- | -- | -- |
| 15 | `get_glass_index` memo | **done** | `glass.py:1615,1618,1646,1811-1836,1954-1972,2163` | `b11` + glass slice | archive-vs-tree over 96 (name, wavelength) pairs, three warning passes, the callable arm, the complex sibling, a 40-array trace | **8/8 bit-identical**; `'N-BK7'` **13.283 -> 0.559 us** (23.8x) |
| 16 | knife-edge asymptotic pins | **skipped -- no ruling exists** | -- | -- | -- | no edit, as instructed |
| 17 | VERIFY-A6 Gaussian oracle's convention | **done** | `tests/unit/test_audit2609_a6_verify_carrier.py:58-96,1410` | 86 passed (85 existing + 1 new) | one ABSOLUTE piston-included assertion | Gouy **+arctan(z/zR) -> -arctan(z/zR)**; all existing numbers unchanged |
| 18 | `_collins_transport` is NumPy-only | **not reached** | -- | -- | -- | JAX IS importable here, so the brief's skip clause does not apply |
| 19 | `PMM2DStackHybrid` attribute guards | **done** | `pmm/stack2d.py:88,91,103,458,467,476,364` | `b11::TestStack2DAttributeGuards` (6) | vocabulary parity between constructor and setter | `st.formulation = 'fff_nv'` accepted -> refused |
| 20 | exact-kernel refinement near a focus | **partial** | `carrier.py:1068` (the K1 window on the public docstring) | -- | -- | the near-focus accuracy table was NOT obtained (see 2.20) |

Three items were not reached (**14**, **18**, and the measurement half of
**20**); **16** was skipped as instructed.  Items **2**, **3** and **4** are
marked *partial* because each shipped a bounded, provable slice of a larger
refactor; what remains is written down below and in
`docs/lens_configuration.md`.

---

## 2. Per item

### 2.1 One branch-band mask, and TWO selectors -- item 1

`lumenairy/_branchcut.py` carries `band_mask(r, *, scale, band, xp=None)`, which
is `|r| <= band * scale` and nothing else.  Four sites call it:

| engine | component | scale | floor |
|---|---|---|---|
| `rcwa/_core._sqrt_decay` | `Re(r)` | `max(max|r|, 1.0)` | dimensionless 1.0 |
| `eme/_branch.forward_decaying_root` | `Im(z)` | `cut_band()`'s product | `\|k0\|`, or none |
| `pmm/_core._forward_branch_flip` | `Im(q)` | `max(max\|q\|, 1.0)` | dimensionless 1.0 |
| `bor/_orient.forward_orient` | `Im(q)` | `max(max\|q\|, \|k0\|)` | `\|k0\|` |

The SCALE stays at each call site on purpose: two of the four engines carry a
DIMENSIONED wavenumber and floor at `|k0|` because a dimensionless literal makes
the branch decision depend on the caller's unit system (audit P2-06, and the EME
peer's own measurement of 3 roots moving between a micrometre and a nanometre
statement of one cell).  Folding the scales would move numbers, not deduplicate
a decision.  `eme._branch.cut_band` returns the PRODUCT, because its two
one-sided readers compare against it directly, so it passes the product as the
scale with `band=1.0`; `x * 1.0` is exact, and
`b11::test_a_unit_band_against_the_product_is_the_same_comparison` asserts the
two spellings agree over 4096 magnitudes spanning 35 decades.

**The finding: there is no ONE selector.**  The brief asked for one; measured,
the two shapes in the tree are different functions on complex input.

| `z` | `where(flip, -z, z)` | `z * where(flip, -1.0, 1.0)` | what differs |
|---|---|---|---|
| `1+0j`, flip | `-1-0j` | `-1+0j` | sign of the zero imaginary part |
| `-0j`, flip | `-0+0j` | `0j` | sign of the zero real part |
| `inf+0j`, no flip | `inf+0j` | `inf+nanj` | the `inf * 0` cross term |
| `nan+0j`, no flip | `nan+0j` | `nan+nanj` | same cross term |

A complex multiply forms `(a*1 - b*0, a*0 + b*1)`; negation touches only sign
bits.  The multiply poisons an infinite part even where it does NOT flip.  So
`_branchcut.py` ships BOTH, named and documented against each other, and each
engine keeps the one it was measured with -- the RCWA layer root takes
`signed_forward` because multiplying by a real constant is holomorphic and its
JAX twins differentiate through the flip (the gradient measurement that rejected
`conj` is in `_sqrt_decay`'s own docstring); PMM and BOR take `negate_forward`.
`b11::test_the_two_selectors_disagree_on_complex_input` pins the four rows and
asserts the two AGREE bit for bit on real input, so a future "obvious"
merge fails loudly instead of moving a sign bit in four engines.

**Gate.** `probe_item1_branchcut.py`: 7 deliberately-edged populations (ordinary
lossless; on-cut rounding noise; genuinely lossy; a deep cutoff spanning 20
decades of magnitude; exact ties and signed zeros; a sub-unit spectrum where the
floor decides; empty) through all four kernels at three bands and five `k0`
values, plus `rcwa_efficiency_1d` (TE and TM), `eme_2d.cell_smatrix` +
`strip_vector_modes`, a `BORStack` S-matrix and channel set, and a 1-D `PMMStack`
solve.  **185/185 bit-identical.**
`tests/unit/test_ci_kernel_consistency.py` 7 passed -- no census decision moved.

### 2.2 `rcwa/_core.py` organisation -- item 2 (partial)

Two halves.

**The measurement prose.**  `STAGGERED_WALL_ANCHOR` carried ~45 `#:` lines of
convergence tables, a rejected-alternative derivation and a flip-cost census on
one constant.  Those are now
`docs/audits/MEASURE_BOR_STAGGERED_WALL_ANCHOR_2026_09_13.md`; the constant
states the live contract (what each value means, which ships, that `'ghost'` is
known-defective rather than an alternative discretisation) and points there.

**The file.**  `elements/rcwa/_geometry.py` is a new leaf holding
`_OVERLAP_SLACK_FRAC`, `_shape_support`, `_ELLIPSE_BISECT_STEPS`,
`_point_ellipse_distance`, `_ellipse_hits_unit_disk`, `_shapes_overlap`,
`_box_hits_unit_disk` and `_shapes_y_varying` -- `numpy` and nothing else from
the library.  `_core` imports and re-exports them, `__all__` is untouched, and
`_core.py` drops 5119 -> 4967 lines.  Its module docstring now names the
thirteen banner-delimited sections and says what `__all__` is: not a module
boundary but a list of import paths that are contracts even for private names,
which is why it cannot simply be shrunk.

**A hazard found by the gate, and avoided.**  The first cut moved
`_validate_shapes` into the leaf too.  That turned
`test_niche_audit_w9_overlap_exact.py::test_w9d_the_predicate_counter_sees_a_pair_that_needs_it`
red: it substitutes a counting wrapper at `_core._shapes_overlap` to prove the
exact predicate is reached, and a `_validate_shapes` living in another module
resolves that name in ITS globals.  `_validate_shapes` therefore stays in
`_core` -- which is the better boundary anyway (policy, not geometry) -- and the
reason is written at both ends.  This is the class of breakage a "test-only-diff
split" has to avoid, and the only way to find it is to run the slice.

**Gate.** `probe_item2_geometry.py`: `__all__` as a set and as an ordered list;
the support function on 257 unit directions for all three kinds; 400 exact
point-ellipse distances; 600 each of the two unit-disk hit predicates; the
overlap verdict over 2000 random shape pairs across all 9 kind combinations;
`_shapes_y_varying` over five lists; **every message `_validate_shapes` raises**
on 13 malformed or illegal inputs; and a `rcwa_efficiency_2d_shapes` solve.
**13/13 bit-identical.**  `test_rcwa.py` + the two W8/W9 files: **231 passed.**

**Not done, and why.**  The remaining ~4900 lines are not a mechanical move.
The blocks that could follow (`_blas.py`; the Wood-anomaly / grazing guards; the
Redheffer algebra; the tensor convolutions) each carry module-level MUTABLE
state (`_BLAS_STATE`, `_BLAS_CONTROLLER`, `_HOMOG_CACHE`, `_HOMOG_LOCK`,
`_JAX_EIG_STABLE`) or are monkeypatched by name from four test files, so each
needs the same per-block hazard analysis the `_validate_shapes` cut needed, plus
a PEP 562 forward for the live globals.  Deferred to B11b with that list.

### 2.3 One row-band schedule -- item 3 (partial)

`_lens_real.py`'s chunked surface paths derived the same integer schedule in
four places -- `r0`/`r1`, the halo `[h0:h1)` clipped at the true grid edges, and
the band's own slice `[lo:hi)` inside that halo -- in two different spellings
(`sag_h[_lo:_lo + (r1 - r0)]` in one, a separately-derived `_hi` in another).
It is now `_row_bands(n_rows, chunk_rows, halo)` and `_band_in_halo(r0, r1, h0)`,
read by `_band_any_sag`, `_obl_band_delta`, the plain chunked screen and the
slant/fresnel chunked screen.

**Gate.** `probe_item3_bands.py` rebuilds `repro/RL-CORE/p10_banded_identity.py`'s
matrix from its method: 5 prescriptions (plain conic+aspheric 3-surface;
decentred mid surface; clear-aperture mid; stop at mid; mixed band/whole-grid)
x 7 `sag_chunk_rows` settings, plus 3 slant/fresnel keyword combinations x 3
chunk widths, plus a `carrier=` arm x 3 (the widened-halo obliquity path) --
**56/56 hashes bit-identical archive-to-tree**, and inside EACH tree the banded
result equals the whole-grid result **47/47**.  `b11::TestTheRowBandSchedule`
asserts the three properties the byte-identity rests on (the bands tile the rows
exactly once; the halo never leaves the grid; `[lo:hi)` selects the band out of
the halo) over nine (n_rows, chunk, halo) triples including `n_rows=1`,
`chunk > n_rows` and odd row counts.

**Not done, and why.**  The brief asks for the three surface bodies -- the two
banded ones and the WHOLE-GRID one -- collapsed into one
`for band in bands(...)`.  The whole-grid path is not a band loop: it is a
single full-grid pass whose numexpr decision is taken on `E.size`, whose
`_ensure_full_grids` allocation the banded paths deliberately never reach, and
whose fresnel dtype promotion happens at a different pipeline step.  Expressing
it as a one-band iteration would move all three of those, i.e. move numbers, so
it is not a bit-identical refactor and does not belong in a hygiene pass.  What
IS shared -- the schedule -- is shared.  Recorded for B11b.

### 2.4 `elements/_lens_kernels.py` -- item 4 (partial)

Re-measured with a module-level-only AST walk over the 11-module lens family
(`scratchpad/b11/cycles.py`), the family had **3** module-level 2-cycles at
`21110326`, not the audit's 4: `_lens_thin <-> lenses` had already been closed
by `backend/_optional.py`.

The `_lens_traced <-> lenses` back-edge carried exactly ONE name,
`_warn_if_aperture_exceeds_grid`.  The grid-versus-aperture bookkeeping it
belongs to -- `_collect_semi_diameters`, `check_grid_vs_apertures`,
`recommend_grid_for_prescription` and the warning itself -- is now
`elements/_lens_kernels.py`, a leaf that imports `numpy` and `warnings` and
nothing from `lumenairy`.  `lenses` re-exports all four (the SAME objects, by
identity, asserted), `_lens_traced` reads the leaf, and the cycle is gone:
**3 -> 2**.

**Gate.** `probe_item4_kernels.py`: the semi-diameter census on both
prescription shapes; `check_grid_vs_apertures` over 18 (N, dx, prescription)
combinations with every warning it emits; `recommend_grid_for_prescription` over
12 combinations including its refusals; the warning's category, message AND
reported filename (i.e. its `stacklevel`) at three grids; and an
`apply_real_lens` call with its warnings.  **7/7 bit-identical**, and the leaf's
objects are `is`-identical to the facade's.

**What remains.**  `_lens_real <-> lenses` needs `surface_sag_general` and
`surface_sag_biconic` to move, and with them the optional-backend plumbing they
read from module scope.  Two of those are live state, which is the whole
difficulty: `_NUMBA_AVAILABLE` is monkeypatched to `False` by the suite and read
at call time, and `cp` / `_ne` are populated on first use, so a plain re-export
would bind a stale `None`.  The mechanical answer is a PEP 562 `__getattr__` on
`lenses.py` forwarding to the leaf.  `lenses <-> lenses_maslov` is a one-line
edit in a file WP-B7 owns.  Both are written up, with the exact edits, in
`docs/lens_configuration.md` section "Module layout".

### 2.6 `LensConfig.to_kwargs(strict=True)` -- item 6

Raises on a field the entry point does not accept instead of dropping it.  The
definition of "a field" is the one that matters: only a REQUEST -- a field whose
value differs from its dataclass default -- can be dropped meaningfully, so
`strict` is quiet about default-valued fields even under
`include_defaults=True`, where dropping one changes no argument.  `strict=True`
with no `entry_point` is refused, because nothing is narrowed there and
accepting it silently is the same quiet no-op the switch exists to remove.  The
refusal names `narrowed_to`, and
`b11::test_every_entry_point_accepts_a_strictly_narrowed_config` asserts
`cfg.narrowed_to(ep).to_kwargs(entry_point=ep, strict=True)` never raises for
any of the seven entry points.

### 2.7 The 1-D symmetric remap's input window -- item 7

**The code fix was already at HEAD.**  VERIFY-B2 (`b496e442`) gave
`_apply_displaced_remap` the same centred window `_apply_displaced_remap_2d`
carries.  What WP-B2's deferred item also asked for -- the mirror fixture that
would have caught it -- did not exist, because the 1-D remap is rotationally
symmetric and every p10 mirror fixture decenters the ELEMENT, which routes to
the 2-D remap instead.

The fixture that reaches this path is a **decentred INPUT FIELD on a symmetric
element**.  MEASURED 2026-09-13, +d against the x-mirror of -d, relative L2 of
intensity:

| N | dx | d | windowed (shipped) | unwindowed | worst pixel / peak, unwindowed |
|---|---|---|---|---|---|
| 512 | 8 um | 0.8 mm | 2.44e-16 | 1.04e-02 | 0.118 |
| 384 | 10 um | 1.0 mm | 2.39e-16 | 3.12e-02 | 0.315 |
| 640 | 6 um | 0.6 mm | 2.25e-16 | 6.81e-03 | 0.089 |

Fourteen decades apart, so the bar is 1e-12 with the unwindowed arm measured
beside it as the falsification (S5 V1).  Through the full
`_apply_displaced_remap` the same pairs read 2.63e-16 / 2.61e-16 / 2.43e-16.

**The byte-identity pin does NOT move.**
`test_niche_p10_transverse_walk_remap.py::test_symmetric_remap_is_the_p2_1d_remap_byte_identical`
compares two calls that BOTH run `_apply_displaced_remap`, so the window is on
both sides of it; restated as a property in
`b11::test_the_byte_identity_pin_is_not_moved_by_the_window` rather than left as
an assumption.  That file passes unchanged.

### 2.9 `apply_aperture(edge='gray')` as the default? -- item 9

MEASURED through the library's own builder (WP-B3 section 3.2 built its grey
mask analytically), lambda = 633 nm, a = 100 um, window 512 um, on-axis relative
error against `U = exp(ikz) - (z/r_a) exp(i k r_a)`.

**RS spatial kernel, z = 16 mm**

| N | dx | `edge='hard'` | `edge='gray'` |
|---|---|---|---|
| 128 | 4.000 um | 8.3008e-03 | 1.4847e-03 |
| 256 | 2.000 um | 3.3548e-03 | 3.6098e-04 |
| 512 | 1.000 um | 3.4207e-04 | 8.1013e-05 |
| 1024 | 0.500 um | 5.2718e-04 | 2.5030e-05 |
| order | | **1.307 / 3.294 / -0.624** | **2.040 / 2.156 / 1.694** |

**HF OPL quadrature, z = 5 mm, one on-axis output point**

| N | `edge='hard'` | `edge='gray'` |
|---|---|---|
| 128 | 2.7708e-02 | 1.4438e-02 |
| 256 | 1.1120e-02 | 3.4928e-03 |
| 512 | 1.1509e-03 | 8.5680e-04 |
| 1024 | 1.7601e-03 | 2.1122e-04 |
| order | **1.317 / 3.272 / -0.613** | **2.047 / 2.027 / 2.020** |

The hard-edge arms reproduce WP-B3 section 3.2's numbers to the last digit, so
this is the same measurement on an independent build of the mask.  The gain at
N = 1024 is **21.1x** (RS) and **8.3x** (HF), and the qualitative statement is
stronger than the ratio: the hard-edge arm has **no reliable order at all**
(a circle's staircase area error does not shrink monotonically, hence the
negative last row), while the grey arm is second order.

**What it costs.**  `edge_samples` ladder at N = 512 (RS): 1 -> 3.4207e-04,
2 -> 1.6563e-04, **4 -> 8.1013e-05**, 8 -> 8.3954e-05, 16 -> 8.3959e-05.  The
shipped default of 4 is the knee; raising it buys nothing, because the residual
is then the propagator's, not the mask's.  The build cost is confined to the
boundary pixels: at N = 256 there are 312 of them (0.476 % of the grid) and at
N = 1024 there are 1196 (0.114 %), so the extra indicator evaluations are
**0.076x** and **0.018x** of one full-grid pass.

**What moves.**  Every hard-aperture fixture's numbers:

| N | mask transmitted-area error, hard - gray | propagated field, relative L2 | worst pixel |
|---|---|---|---|
| 256 | +0.4428 % | 7.678e-03 | 3.660e-03 |
| 512 | +0.3108 % | 2.930e-03 | 1.251e-03 |
| 1024 | +0.1270 % | 1.237e-03 | 5.434e-04 |

**RECOMMENDATION (maintainer's ruling).**  Flip `edge` to `'gray'`.  It buys a
convergence RATE, not a constant -- the single largest accuracy lever available
to a caller of either spatial kernel -- for ~2-8 % of one mask-building pass,
and `edge_samples=4` is already at the knee.  The price is that every existing
hard-aperture fixture moves at the 1e-3 level, which is a re-pinning exercise
with a known, bounded size.  **Not shipped**: a default move is the maintainer's
call, and `edge='gray'` is already reachable by keyword.

### 2.10 A `maslov` family in the covering array, and the in-glass gap legs -- item 10

**The family.**  `MASLOV_FACTORS` is a pairwise array over seven
`apply_real_lens_maslov` physics factors (`poly_order`; the ray sampling;
`extract_linear_phase`; `normalize_output`; `output_plane_distance`;
`fold_split`; `input_wavevector_saddle`) on the SAME diverging, clipped
fixture the other two arrays use, plus an 18-kwarg default-identity test.
MEASURED: every level of every factor returns finite and correctly shaped, no
combination raises (so `MASLOV_EXCLUSIONS` is empty *because it was measured*),
and all 18 keywords passed at their signature default are **byte-identical** to
omitting them.

`integration_method` is deliberately NOT a factor.  Measured on this fixture,
`'auto'` returns in 1.5 s and the explicit `'quadrature'` in **72.6 s**; a
pairwise array over its five levels would put minutes of one entry point's
quadrature into the fast lane.  It keeps its coverage in the default-identity
test, which is the arm a caller splatting a `LensConfig` actually hits.

The energy bar means something different here and the test says so:
`normalize_output='power'` rescales the output to carry exactly the input power,
so on every arm that leaves it alone `P_out/P_in` is 1.000000000 by
construction.  The `normalize` factor's second level turns it off, and that is
the arm with teeth -- **0.405890451** there.

**The gap legs, and a finding.**  The array's `propagator` factor is `[{},
{'wave_propagator': 'rs'}]`, so nothing in the lens matrix reached the in-glass
`'sas'` / `'fresnel'` gap legs.  MEASURED on the covering-array doublet (N = 64,
dx = 112.5 um, lambda = 632.8 nm, gaps 9.0 and 2.5 mm of N-BAF10 / N-SF6HT):

| `wave_propagator` | `P_out/P_in` | diagnostics |
|---|---|---|
| default (ASM) | 0.996170598 | none |
| `'rs'` | 0.996170187 | none |
| `'fresnel'` | **10396.714211** | 2 x `RuntimeWarning` |
| `'sas'` | **10396.710108** | **none** |

Both gap legs gain **four decades** of power.  The geometry is why: the
single-FFT Fresnel kernel's own validity bound is
`max(N dx^2)/lambda_medium = 2.13 m` against a 9 mm gap, i.e. the chirp is
aliased by 240x.  It is not a fixture artefact a finer grid removes -- the bound
FALLS with dx, so at N = 512 the same fixture reads 0.142 (`'fresnel'`) and
0.0405 (`'sas'`) of P_in instead, and a validly-sampled in-glass Fresnel leg on
a 7.2 mm window would need N ~ 15 000.

The ASYMMETRY is the finding, and it is exactly the "both directions of the
window-against-period gate" the brief asked to cover: `'fresnel'` warns twice
and names the bound; `'sas'` returns the same four-decade gain **in silence**,
because its only validity gate is the far-field direction (`z > z_limit`,
`propagators/sas.py:199`) and this failure is the near one.  The two legs
therefore could NOT be added as covering-array levels without conceding the
array's one-sided energy bar; they are a dedicated test
(`test_the_in_glass_gap_legs_are_reached_and_only_one_of_them_is_gated`) that
pins the facts, including the silence, so the day a gate lands the test goes red
and the author records the change deliberately.  Requested change below.

### 2.11 `pmm_jones_2d` assembles the tensor operators once -- item 11

The double build was reachable on exactly the cells the even-parity fold cannot
take: at normal incidence with `symmetry` on, the fold probe ran the whole
projected-operator assembly, answered `None` because the cell is out-of-plane or
slanted, and the cascade then assembled it again.  `_tensor_projected_ops` is
hoisted once per layer and passed to both calls; `_tensor_layer_modes` only READS
that dict (it rebinds its own locals for the `keep` restriction), so one build
serves both.

**Gate.** `probe_item11_jones2d.py`: in-plane / out-of-plane / slanted cells x
normal and oblique incidence x `laurent` and `li` x `symmetry` `'auto'` and
`False` -- **24/24 bit-identical**.  The count test
(`b11::TestJones2DAssemblesOnce`) asserts exactly one build per call on all
three cell kinds, with a counter-pin that two solves read exactly two builds so
a patch that never fired could not pass.

### 2.12 `sampling=` on the free-space HFPI entry points -- item 12

`propagate_hfpi_freespace_aperture` gains `sampling`, `sampler`, `n_strata_xy`
and `n_strata_dir`, routing to `init_paths_stratified` exactly as the
prescription walk does; `propagate_hfpi` reaches them through `**kwargs`.  The
default is `'uniform'`, NOT the walk's `'stratified'`: this pair has returned
the uniform draw since it was written, and the output is a Monte-Carlo estimate
whose realisation depends on the placement rule, so flipping the default would
move every existing caller's numbers.  A `sampler` or a stratum count passed
without `sampling='stratified'` is refused rather than dropped, the same
contract the walk carries.

MEASURED on a fixture where paths actually land (32x32 source, 200 um aperture,
16 384 paths, `cone_half_angle=0.35`): default and explicit `'uniform'` are
**byte-identical**; `'stratified'` and `'sobol'` both differ from it and from
each other, and the three totals sit within 1.6 % of the uniform draw
(stratified -1.59 %, sobol +1.45 %) -- the spread of one estimator at this path
count, which is what makes the switch a variance choice rather than a physics
one.  `b11::TestHfpiFreeSpaceSampling` asserts the byte-identity, that
the fixture lands more than 100 non-zero pixels (so the comparison is not two
empty grids), the routing, the agreement and both refusals.

### 2.13 The odd-N grid-centring disagreement, swept -- item 13

**The disagreement.**  `ifftshift` centres an axis on index `N // 2`; the package
coordinate convention `(arange(N) - N/2) * dx` centres it on `N/2`.  For even N
these are the same index and a sample sits exactly at 0.  For odd N they are half
a pixel apart and **no sample sits at 0** at all:

| N | coordinate zero at index | `ifftshift` centre index | offset | `x[centre]` | sample at 0? |
|---|---|---|---|---|---|
| 4 | 2.0 | 2 | +0.0 | +0.0 dx | yes |
| 5 | 2.5 | 2 | **-0.5** | -0.5 dx | **no** |
| 8 | 4.0 | 4 | +0.0 | +0.0 dx | yes |
| 9 | 4.5 | 4 | **-0.5** | -0.5 dx | **no** |
| 63 | 31.5 | 31 | **-0.5** | -0.5 dx | **no** |
| 64 | 32.0 | 32 | +0.0 | +0.0 dx | yes |
| 65 | 32.5 | 32 | **-0.5** | -0.5 dx | **no** |

**The census.**  An AST-aware walk over `lumenairy/` (docstrings and comments
excluded from the count, string mentions bucketed separately) finds **149**
`fftshift` / `ifftshift` sites:

| bucket | count | odd-N reachable? |
|---|---|---|
| frequency array (`fftfreq` and friends) | 29 | reachable, but self-consistent: the convention is `fftfreq`'s, not the package coordinate array's |
| round-trip around an FFT (the shift is its own inverse) | 61 | reachable and HARMLESS: no coordinate is implied |
| **coordinate-coupled** (a spatial field or a package coordinate array) | **27** | **reachable and coupled** |
| import / re-export / docstring-only | 32 | not a site |

The coordinate-coupled 27, by file: `propagators/asm.py` 7,
`ui/phase_retrieval_dock.py` 4, `backend/fft.py` 4,
`analysis/phase_retrieval.py` 3, `analysis/detector.py` 2, `propagators/sas.py`
2, and one each in `analysis/through_focus.py`, `elements/doe.py`,
`propagators/carrier.py`, `ui/coronagraph_dock.py`, `ui/psf_mtf_dock.py`.

**The half pixel, quantified.**  `compute_psf(method='fft')` on a circular pupil:

| N | PSF shape | peak index | centre index | peak offset | centroid |
|---|---|---|---|---|---|
| 64 | (64, 64) | (32, 32) | 32.0 | (+0.0, +0.0) px | (-0.0000, -0.0000) px |
| **65** | (65, 65) | (32, 32) | 32.5 | **(-0.5, -0.5) px** | **(-0.5000, -0.5000) px** |
| 128 | (128, 128) | (64, 64) | 64.0 | (+0.0, +0.0) px | (-0.0000, -0.0000) px |
| **129** | (129, 129) | (64, 64) | 64.5 | **(-0.5, -0.5) px** | **(-0.5000, -0.5000) px** |
| **255** | (255, 255) | (127, 127) | 127.5 | **(-0.5, -0.5) px** | **(-0.5000, -0.5000) px** |
| 256 | (256, 256) | (128, 128) | 128.0 | (+0.0, +0.0) px | (-0.0000, -0.0000) px |

Exactly half a pixel, on both axes, INDEPENDENT of N -- it is a convention
mismatch, not a discretisation error, so it does not converge away.  The peak
and the centroid agree, so it is a rigid shift of the whole answer.

**RECOMMENDED FIX (one helper, no default moved here).**  Add
`lumenairy/_math/centring.py` with two functions that state the convention once:
`coordinate_axis(N, d)` returning `(arange(N) - N/2) * d` (what the package
means by "the grid") and `dc_index(N)` returning the index `ifftshift` treats as
the origin, plus `centre_shift(N)` = `dc_index(N) - N/2`, which is `0` for even
N and `-0.5` for odd.  Then the 27 coordinate-coupled sites each either (a)
assert even N, or (b) apply a half-pixel `exp(i pi f d)` correction derived from
`centre_shift`.  Adopting (b) MOVES the answer on odd grids, which is a numerical
default and the maintainer's call; the helper and the classification can land
without it, and that is what makes the choice reviewable one site at a time.

### 2.15 `get_glass_index` memo -- item 15

WP-B9's request asked for an `lru_cache` on `(glass_name, wavelength)`.  One
existed -- `_cached_glass_value`, keyed at picometre resolution, LRU-bounded,
invalidated by `_invalidate_glass_name`, drained by `_clear_glass_caches` through
`_cache_registry`.  The finding is that it was read in the WRONG PLACE and
covered the wrong arm.

MEASURED before (20 000 calls each, threads pinned): a warm
`get_glass_index('N-BK7', 1.55 um)` cost **13.283 us** against **0.140 us** for
`'air'`, and clearing the value cache before every call cost **13.990 us** --
i.e. the evaluation the memo short-circuited was **0.7 us of 14**, and the other
13.3 us was the walk down to it, dominated by `_maybe_warn_outside_validity`'s
array comparison on a scalar.  And `'N-BK7'` is a CATALOGUE name on a box with
`refractiveindex` installed, so it never reached the memo at all -- only the
bundled-Sellmeier and polynomial arms did.

Two changes.  The memo is read once, immediately after the user-callable branch,
so the whole immutable-catalogue resolution is covered; and the live
refractiveindex.info arm is memoised like the closed forms, with the non-finite
refusal inside the memoised body so a page with no data at that wavelength
raises on every call rather than caching a refusal.

| glass | before | after | ratio |
|---|---|---|---|
| `'air'` | 0.140 us | 0.166 us | 0.84x |
| `'N-BK7'` (catalogue) | 13.283 us | **0.559 us** | **23.8x** |
| `'N-SF11'` (bundled Sellmeier) | 4.976 us | 0.585 us | 8.5x |
| `'N-BAF10'` | 12.809 us | 0.618 us | 20.7x |
| `'N-SF6HT'` | 15.437 us | 0.552 us | 28.0x |

The `'air'` row is the honest cost: the short-circuit returns before the memo, so
it pays nothing, and the 26 ns is measurement noise on a 0.14 us call.

**Warning-neutral, and proved so.**  A memo hit can only follow a miss for the
same `(name, wavelength)` that already ran `_maybe_warn_outside_validity`, whose
warn-once set is keyed more COARSELY (0.1 nm against the memo's 1 pm), so every
warning the hit skips was already suppressed.  `_clear_glass_caches` empties both
together.  `probe_item15_glass.py` asserts this directly by hashing the warning
stream on a FIRST pass, a SECOND pass and a pass AFTER a full drain: the first
and third hashes are equal and the second differs, identically in both trees.

**The generation counter.**  `glass.glass_registry_generation()` returns a
monotone counter bumped inside the cache lock by `_invalidate_glass_name` and
`_clear_glass_caches`.  It is for downstream caches of DERIVED glass values --
`raytrace.jax_trace._build_jax_prescription`'s compiled prescription above all,
which keys on the prescription and not on the registry, so a re-pointed glass
keeps serving the old index with nothing said.  Keying on
`(prescription_key, glass_registry_generation())` closes that for one integer
compare.  A direct `GLASS_REGISTRY[name] = ...` cannot bump it; that is the
contract the value cache has always had and this neither widens nor narrows it.

**Gate.** 96 `(name, wavelength)` pairs across `'air'` spellings, bundled
Sellmeier, polynomial, catalogue and unknown names x 8 wavelengths including
out-of-validity ones; the three warning passes above; an array wavelength (which
must bypass the memo); a registered COUNTING CALLABLE, which must NOT be
memoised because it is user code (five calls, five different values, identical
in both trees); `get_glass_index_complex` on three glasses; and a real ray trace
through a 3-surface N-BK7 / N-SF11 doublet, 40 arrays harvested and hashed.
**8/8 bit-identical.**

### 2.16 The knife-edge asymptotic pins -- item 16

**No edit.**  The brief makes this conditional on an orchestrator ruling that
`sphere_normal='analytic'` becomes the ray tracer's default.  No such ruling
exists (the memory's "decisions still owed" list carries it as open), so the two
pins are untouched, exactly as instructed.

### 2.17 The VERIFY-A6 Gaussian oracle's phase convention -- item 17

`_abcd_field` built `1/q = 1/R - i lam/(pi w^2)` -- Siegman's `exp(+i omega t)`
pairing -- and then carried the Gouy phase separately as `angle(q/q2)`.  In this
library's `exp(-i omega t)` convention that is the Gouy phase with the WRONG
SIGN: an error of `2 arctan(z/zR)`, exactly pi across a focus.  Every assertion
in the file is piston-free, so nothing failed and nothing could have.

Restated in the whole-function form
`E = exp(i k z)/(1 + z/q) * exp(i k r^2/(2 q(z)))` with
`1/q = 1/R + i lam/(pi w^2)` and `w(z) = sqrt(lam/(pi Im(1/q(z))))`.  The point
of that form is that it cannot carry the two halves in different conventions:
the amplitude `|q/q2| = w_in/w(z)`, the wavefront curvature and the Gouy phase
are all the argument of ONE complex number.

MEASURED (lambda = 1 um, w0 = 200 um, zR = 125.664 mm), on-axis Gouy after removing
the `k z` piston:

| z / zR | corrected oracle | `arctan(z/zR)` | old spelling |
|---|---|---|---|
| 0.0 | +0.000000 | 0.000000 | +0.000000 |
| 0.5 | **-0.463648** | +0.463648 | +0.463648 |
| 1.0 | **-0.785398** | +0.785398 | +0.785398 |
| 3.0 | **-1.249046** | +1.249046 | +1.249046 |

i.e. exactly `-arctan(z/zR)`, a RETARDATION, which is what this convention
requires.  `w(z)` is identical to all printed digits and the piston-free
normalised profiles differ by at most 1.8e-10 (the round-off between two
algebraic spellings of the same curvature), which is why every existing
assertion still passes with the same numbers: **86 passed** (85 existing + the
new absolute pin).

The new pin,
`test_the_gaussian_oracle_carries_this_librarys_phase_convention`, is the one
ABSOLUTE piston-included assertion the brief asked for.  Its bar is DERIVED: the
piston `k z` reaches 3.95e+05 rad at z = 3 zR, so `exp(1j k z)` carries
`eps * k z ~ 8.8e-11` rad of representation error before any physics; the bar is
1e-9, ~11x that and nine decades below the 0.93 rad the wrong convention
produces at z = 0.5 zR.  It also asserts the total Gouy swing across a focus is
pi, and carries a falsification arm asserting the oracle does NOT agree with
both sign conventions.

### 2.19 `PMM2DStackHybrid`'s validated attributes -- item 19

`formulation`, `cascade` and `symmetry` are properties now, sharing ONE
vocabulary with the constructor: `_check_formulation` and `_check_cascade` are
module-level helpers called from both `__init__` and the setters, and the
`symmetry` setter runs `_symmetry_on`, so `st.symmetry = 'auto'` reads back
`True` exactly as the constructor argument does.  `__init__` keeps its up-front
checks (so the refusal ORDER relative to the other constructor arguments is
unchanged) and now calls the same helpers, so the two cannot drift.

`b11::test_the_constructor_and_the_setter_share_one_vocabulary` sweeps every
legal value of both vocabularies plus two illegal ones through BOTH routes and
asserts the two verdicts agree with the vocabulary, which is the property that
makes "one definition" checkable rather than asserted.

**Same class, not fixed:** `truncation` is validated in `__init__` and is a
plain attribute afterwards, exactly as these three were.  It is outside the
brief's list; recorded for B11b as a one-line addition to the same pattern.

### 2.20 The exact-kernel refinement near a focus -- item 20 (partial)

**Delivered.**  The Collins one-step readout's applicability window (VERIFY-B4
F1) was derived at length in `_collins_readout`'s own docstring but nowhere a
caller choosing `transport=` would read it.  The public `transport` parameter
now carries it: the K1 condition `2 dx (|A| r/|B| + theta)/lambda` on the
CHAIN'S OWN exit pitch, the reduced final leg `z_eff = z/A` it is really about,
the two measured readings that bracket it (K1 = 0.16 on the WP-A6 fixture;
K1 = 82 for an 8 mm final distance on a 5.4 mm exit beam at 76 um), why
`final_distance = 0` is refused, and what the Sziklas readout pays instead.

**NOT delivered: the near-focus accuracy table.**  The fixture is built
(`scratchpad/b11/m_item20_gapkernel.py`: a converging Gaussian carrier, f = 20
mm, w0 = 15.915 um, theta = 20.0 mrad, evaluated 1 um .. 5 mm short of the geometric
focus against the analytic Gaussian) and the dropped quartic
`k |z_eff| theta^4 / 8` is computed and does span the decision (5.03e+01 at 1 um
from focus down to 7.54e-03 at 5 mm, against
`_GAP_ENV_PHI_TOL_DEFAULT = 0.3`), but the field comparison itself is not yet
valid: `propagate_carrier_referenced` takes and returns an ENVELOPE referenced
to a carrier, and the first two spellings of the reference bookkeeping gave O(1)
residuals and then a blanket `ValueError`.  Publishing a table from a fixture I
have not validated would be worse than publishing none, so the recommendation is
NOT made here.  The remaining work is the envelope/field bookkeeping only; the
quartic, the tolerance and the sziklas/collins sweep are already in place.

---

## 3. Files touched

Modified (library):

* `lumenairy/elements/rcwa/_core.py` -- item 1 (branch band + selector), item 2
  (section map, geometry re-export)
* `lumenairy/elements/eme/_branch.py` -- item 1
* `lumenairy/elements/pmm/_core.py` -- item 1 (`_PMM_CUT_BAND_REL`, band, selector)
* `lumenairy/elements/bor/_orient.py` -- item 1
* `lumenairy/elements/bor/coupled_radial_eigensolver.py` -- item 2 (prose out)
* `lumenairy/elements/_lens_real.py` -- item 3 (`_row_bands`, `_band_in_halo`,
  four call sites), item 4 (leaf import)
* `lumenairy/elements/lenses.py` -- item 4 (leaf re-export, block removed)
* `lumenairy/elements/_lens_traced.py` -- item 4 (reads the leaf)
* `lumenairy/elements/lens_config.py` -- item 6
* `lumenairy/elements/pmm/twod_jones.py` -- item 11
* `lumenairy/elements/pmm/stack2d.py` -- item 19
* `lumenairy/propagators/hfpi.py` -- item 12
* `lumenairy/propagators/carrier.py` -- item 20 (docstring)
* `lumenairy/glass.py` -- item 15

Added (library):

* `lumenairy/_branchcut.py` -- item 1
* `lumenairy/elements/rcwa/_geometry.py` -- item 2
* `lumenairy/elements/_lens_kernels.py` -- item 4

Modified (tests):

* `tests/unit/test_audit2609_a15a_lens_covering_array.py` -- item 10
* `tests/unit/test_audit2609_a6_verify_carrier.py` -- item 17

Added (tests):

* `tests/unit/test_audit2609_b11_hygiene.py` -- items 1, 3, 4, 6, 7, 11, 12, 19

Modified (docs):

* `docs/lens_configuration.md` -- item 4 ("Module layout" re-measured and
  re-planned)
* `docs/history/` re-recorded in this change (nine documents):
  `lumenairy.elements.pmm._core.md`, `lumenairy.elements.bor._orient.md`,
  `lumenairy.elements.eme._branch.md`, `lumenairy.elements._lens_real.md`,
  `lumenairy.elements.lenses.md`, `lumenairy.elements._lens_traced.md`,
  `lumenairy.glass.md`, `lumenairy.elements.pmm.stack2d.md`,
  `lumenairy.propagators.hfpi.md`.

  Two modules with history documents were changed WITHOUT a re-record, and the
  recorder itself says why: `bor/coupled_radial_eigensolver.py` (item 2) and
  `propagators/carrier.py` (item 20) were touched only in comments and
  docstrings, which the AST and token fingerprints exclude by construction --
  that exclusion is the whole point of the gate, since it is what lets it prove
  a documentation-only edit really was documentation-only.  Both read `OK`
  against their recorded hashes, and `record_history_fingerprints.py --check`
  reports every document matching its module.

Added (docs):

* `docs/audits/MEASURE_BOR_STAGGERED_WALL_ANCHOR_2026_09_13.md` -- item 2
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B11_REPORT.md`
  (this file) and `WP-B11_CHANGELOG.md`

---

## 3b. Tests run

Every run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`-p no:randomly`.

| what | result |
|---|---|
| `ruff check lumenairy/` | **All checks passed** |
| `ruff check` on the three changed / added test files | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** |
| `tests/unit/test_audit2609_a17_history_lint.py` | **5 passed** |
| `python validation/run_all.py` | **ALL 37 files passed** |
| `tests/unit/test_audit2609_b11_hygiene.py` | **48 passed** |
| `tests/unit/test_audit2609_a6_verify_carrier.py` | **86 passed** (85 existing + the new absolute pin) |
| `tests/unit/test_ci_kernel_consistency.py` | **7 passed** -- no census decision moved |
| the a15a maslov / gap-leg selection | **10 passed** in 10.7 s |
| the 19 affected files in one run (b11, a15a, a6, both a17 gates, the kernel census, rcwa, W8, W9, W6-BOR, grazing cutoff, a14 + a14-verify, the BOR guard grep, eme vector, a16 round-trip + a16 verify, p10, b2) | **1620 passed, 2 failed** in 31:46 -- both failures `lumenairy.elements.lenses_maslov` in the A17 relocation gate, i.e. WP-B7's uncommitted edits |
| `-k "real_lens or glass or lens_config or hfpi or stack2d or jones_2d or twod_jones"` over `tests/unit` | **869 passed, 6 skipped, 0 failed** in 11:20 |
| `-k "rcwa or eme or bor or pmm"` over `tests/unit` | **3580 passed, 5 failed** -- see below |

The six bit-identity probes were re-run after the LAST edit of the work package,
not only after their own item: 185/185, 13/13, 56/56 (plus 47/47 banded ==
whole grid in both trees), 7/7, 24/24, 8/8.

**On the failures.**  In the 19-file run the only two reds are WP-B7's
uncommitted `lenses_maslov.py` edits against the A17 relocation fingerprints;
no module this work package touched drifts (`--check` is clean).

**On the five failures in the earlier wide `rcwa or eme or bor or pmm` slice.**  That run was started while this
work package was still editing the tree, so two of them
(`test_niche_audit_w3_elements.py::...test_el2_lock_exists_at_module_scope...`
and `test_niche_perf_round2_2026_08_10.py::test_the_element_builds_its_coords_without_np_indices`)
read SOURCE FILES that changed under them; both were re-run on the settled tree
and **pass**.  Two are WP-B7's uncommitted `lenses_maslov.py` edits failing the
A17 relocation gate, not this work package's.  The fifth,
`test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`,
is the known-red T3-1 arm that fails at the audit base on this box too and is
documented as such in the 5.46.0 release intro.

## 4. Requested changes (for the orchestrator / maintainer)

1. **`apply_aperture(edge='gray')` as the default** -- section 2.9 has the case
   and the price.  A numerical-default move; not shipped.
2. **The in-glass `'sas'` gap leg has no near-field gate** -- section 2.10.
   `wave_propagator='sas'` returns a 1.04e+04 energy gain on a standard doublet
   with NO diagnostic, where `'fresnel'` warns twice about the same aliasing.
   `sas.py:199`'s only validity test is `z > z_limit` (too far); the failure
   here is the opposite direction.  The two legs cannot re-enter the lens
   covering array until this is gated.  Owner: whoever owns `propagators/sas.py`
   and `_lens_real._propagate_through_glass`.
3. **The odd-N centring convention** -- section 2.13.  A shared helper can land
   without moving anything; adopting the half-pixel correction at any of the 27
   coordinate-coupled sites moves that site's answer on odd grids and is a
   numerical default.
4. **`PMM2DStackHybrid.truncation`** has the same unguarded-after-`__init__`
   shape the other three had (section 2.19).  One line, same pattern, outside
   this brief's list.
5. **`lenses_maslov.py:282`** -- the one-line import change that closes the
   `lenses <-> lenses_maslov` cycle, written out in
   `docs/lens_configuration.md`.  WP-B7 owns that file.

## 5. Items not reached, and what is deferred to B11b

* **Item 14 (direct-matrix MFT branch)** -- not started.  Needs the crossover
  measured in BOTH memory and time, the tolerance between the two reductions
  derived and stated, and a ruling on opt-in versus threshold-automatic.
* **Item 18 (`_collins_transport` is NumPy-only)** -- not started.  The brief's
  skip clause does NOT apply: `jax` IS importable on this box (`cupy` is not).
  The work is larger than the entry point: `_collins_transport` calls
  `np.asarray` / `np.ascontiguousarray(..., dtype=np.complex128)` and then
  `_fft2`, `_collins_angle_support`, `_collins_exact_kernel_correction`,
  `_collins_space_support`, `_collins_sampling_stats` and
  `_bluestein_centred_2d`, none of which takes `xp` today.  Threading it is a
  chain, not a signature.
* **Item 20's near-focus table** -- section 2.20; the fixture exists, the
  envelope/field bookkeeping does not.
* **Item 2's remaining `_core.py` split** -- section 2.2, with the per-block
  hazard list (module-level mutable state and four monkeypatching test files).
* **Item 3's whole-grid body** -- section 2.3; folding it into the band
  generator would move the numexpr gate, the `_ensure_full_grids` path and the
  fresnel dtype-promotion point, so it is not a bit-identical refactor.
* **Item 4's remaining two cycles** -- section 2.4 and
  `docs/lens_configuration.md`, including the PEP 562 forward the
  `_NUMBA_AVAILABLE` monkeypatch and the lazy `cp` / `_ne` slots need.

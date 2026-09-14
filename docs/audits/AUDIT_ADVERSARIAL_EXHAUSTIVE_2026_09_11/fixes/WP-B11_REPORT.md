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

---
---

# WP-B11 (Wave 4, last) -- the hygiene pass, **part b**

Branch `audit-fixes-2026-09`, base HEAD `c62c2f14` (part a landed).  Scope from
the orchestrator, in priority order: the **P1 near-field gate for the in-glass
`'sas'` gap leg** (part a sec. 2.10's finding), **item 5** (`LensPhysics`),
**item 8** (the `doe.py` fill and the warning `stacklevel` sweep), the
**`PMM2DStackHybrid.truncation`** guard, then the deferred list.

WP-B7's files were uncommitted in the tree at launch and are owned by VERIFY-B7
concurrently (`lenses_maslov.py`, `lenses_gbd.py`, `_lens_jax.py`,
`propagators/asymptotic*.py`, `fga.py`, `gbd.py`).  **None was opened for
writing by this work package.**  The edits item 8 needs in two of them are
written out verbatim in section 5b for the orchestrator to apply after
VERIFY-B7 lands.

WP-B7 landed mid-package as `f64444ec`.  `git diff --name-only c62c2f14
f64444ec` touches **none** of this package's modules, so the base above is
still the correct pre-change library for every gate reported here, and no gate
was re-run against a moving target.

## How every change in part b was gated

The same harness as part a, re-based: `git archive c62c2f14 lumenairy` extracted
read-only into `scratchpad/b11b/archive`.  Each probe runs TWICE in a child
process -- once with cwd and `PYTHONPATH` set to that archive, once with them set
to the working tree -- asserting `lumenairy.__file__` is under the expected root
before it computes anything, and the two JSON outputs are compared key by key.
Never through pytest, and never against the shared working tree's in-place
modules.  Every hash is SHA-256 over the exact IEEE-754 bytes plus dtype and
shape.  Every python run carried
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one at a time.
The working tree carries WP-B7's uncommitted edits, so only the modules this
package owns are reported.

Harness: `scratchpad/b11b/bi.py` (driver), `probelib.py` (bind + hash),
`probe_item3_sas.py`, `probe_item5_physics.py`, `probe_item8_stack.py`,
`probe_item4_stack2d.py`.  Measurement scripts: `m_item3_sasnear.py`,
`m_item3_sasnear2.py`, `m_item3_window.py`, `m_item5_sigs.py`,
`m_item8_stack.py`.

---

## 1b. Summary

| # | item | status | files:lines | tests | oracle | measured, before -> after |
|---|---|---|---|---|---|---|
| 3b (P1) | the in-glass `'sas'` gap leg has no near-field gate | **done** | `propagators/sas.py:38-110` (new), `:143-158`, `:303-316` | `b11::TestTheSasNearFieldGate` (6); `a15a::test_the_in_glass_gap_legs_are_reached_and_both_of_them_are_gated` (restated) | archive-vs-tree over 9 (N, dx, lambda, z/z_near, pad) cases + dtype / skip-phase arms + the doublet end to end; accuracy oracle = the SAME kernel at 8x finer input pitch | **24/24 bit-identical**; the doublet's `'sas'` leg **1.0397e4 x P_in in silence -> the same 1.0397e4 with 2 RuntimeWarnings**, one per under-sampled gap |
| 5b | `LensPhysics` | **done** | `lens_config.py:705-826` (new class), `:1-12,49-60,206-216,936-960,975,1150,1153,1232,1260,1441-1520`; `_lens_real.py:150-159,5224,6108-6156`; `lumenairy/__init__.py`; `elements/__init__.py`; `docs/lens_configuration.md` | `b11::TestLensPhysics` (9); the a16 census +17 | archive-vs-tree: 9 keyword spellings, the three shipped groups' resolved calls, both field tables and both default lists | **20/20 bit-identical**; `physics=` == the keyword call on **9/9** cases; config fields **38 -> 47** |
| 8b | `doe.py`'s zero fill, and warning attribution | **done** | `doe.py:532-556`; `_lens_kernels.py:33-92` (new helper), `:412-416,434-441,476-479`; `_lens_real.py` (9 sites); `_lens_traced.py` (31 sites); `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py:118-135` | `b11::TestTheZonePlateZeroFill` (8), `b11::TestWarningAttribution` (4) | archive-vs-tree: 7 zone-plate arms with dtypes + 7 lens fields | **20/20 bit-identical**; a configured `apply_real_lens` notice **`_lens_real.py:6130` -> the caller**; `prepare_real_lens_traced` **5 of 5 notices in `_lens_traced.py` -> 5 of 5 at the caller** |
| 4b | `PMM2DStackHybrid.truncation` | **done** | `pmm/stack2d.py:90-91,115-125,325,477-486,487-501` | `b11::TestStack2DTruncationGuard` (4) | archive-vs-tree: 2 truncations x 2 formulations, the whole solve output + the four attribute reads | **8/8 bit-identical**; `st.truncation = 'circle'` **accepted -> refused** |
| 5b-NEW | the `lenses <-> lenses_maslov` cycle | **deferred, edit written out** | -- | -- | -- | section 5b.1; `lenses_maslov.py` is WP-B7's |
| deferred | items 2, 3 (whole grid), 4 (two cycles), 14, 18, 20's table | **not reached** | -- | -- | -- | section 6b |

Three behaviour changes ship -- one diagnostic and two refusals, each where a
silent wrong answer was returned before -- and each carries a Migration note in
the changelog: the SAS near-field warning (2b.1), the `truncation` refusal
(2b.4), and the moved warning ATTRIBUTION (2b.3), which
`warnings.filterwarnings(module=...)` keys on.  **No numerical default moves.**

---

## 2b. Per item

### 2b.1 The in-glass `'sas'` gap leg's near-field gate -- P1, from part a sec. 2.10

**The finding, restated.**  On the WP-A15a covering-array doublet (N = 64,
dx = 112.5 um, lambda = 632.8 nm, 9.0 mm of N-BAF10 and 2.5 mm of N-SF6HT)
`apply_real_lens(wave_propagator='sas')` returned `P_out/P_in = 1.0397e4` with
NO diagnostic while `'fresnel'` warned twice about the same aliasing.
`propagators/sas.py`'s only validity test was the paper's FAR bound
`z > z_limit`; this failure is the near one.

**The derivation.**  SAS's third step is the same single-FFT Fresnel sum
`fresnel_propagate` evaluates,

```
E_out(q) ~ sum_m psi(x_m) exp(i k x_m^2 / (2z)) exp(-2 pi i x_m q / (lambda z))
```

over the INPUT grid `x_m` at pitch `dx`.  The DFT represents the linear,
output-dependent factor exactly -- it *is* the DFT kernel -- so the sampling
requirement falls entirely on the quadratic chirp, whose local spatial frequency
at `x` is `x / (lambda z)`.  The grid resolves at most `1 / (2 dx)`, so the sum
is a valid quadrature only while `x_max / (lambda z) <= 1 / (2 dx)`.  A
propagator cannot know the field's own support, so the bound is taken at the
worst case it can know -- a field filling its input window, `x_max = N dx / 2`:

```
z  >=  z_near  =  N * dx^2 / lambda
```

which is exactly `fresnel_propagate`'s K1 bound, and exactly the complement of
the `Q = lambda |z| / (N dx^2) >= 1` band the dispatcher trips ASM over to SAS
on.

**`pad` does not enter it, and that is measured rather than assumed.**  Padding
enlarges the array the chirp is evaluated on, so `pad * N * dx^2 / lambda` is
the obvious alternative.  It is wrong: the precompensation `delta_H` is a
band-limited phase filter whose impulse response stays concentrated on the input
window, so the chirp's outer, unresolved turns multiply the zero padding.
MEASURED on a window-filling super-Gaussian (N = 128, dx = 2 um,
lambda = 633 nm, so `z_near` = 0.809 mm), against an oracle that is the SAME SAS
kernel on an 8x finer input pitch over the same physical window -- its own bound
is 8x smaller, and its output pitch `lambda z / (pad N dx)` is IDENTICAL, so the
comparison is sample against sample with no interpolation:

| z / z_near | rel. field error, pad 2 | pad 4 | pad 1 | output power / oracle, pad 2 |
|---|---|---|---|---|
| 0.05 | 9.19 | 9.21 | 9.14 | 85.2x |
| 0.10 | 4.79 | 4.86 | 4.81 | 23.6x |
| 0.20 | 2.24 | 2.30 | 2.26 | 6.00x |
| 0.35 | 0.840 | 0.773 | 1.059 | 1.67x |
| 0.50 | 0.167 | 0.067 | 0.516 | 1.028x |
| 0.75 | 2.0e-3 | 1.8e-3 | 9.6e-2 | 1.000x |
| 1.00 | 1.1e-3 | 1.3e-3 | 4.6e-3 | 1.000x |
| 2.00 | 8.6e-4 | 6.7e-4 | 6.9e-4 | 1.000x |

The three `pad` columns break down at the same ABSOLUTE `z`, which is what says
the bound is set by `N dx` and not by `pad N dx`.  (`pad = 1` is a little worse
through the transition because it also suffers circular wraparound, which is
what `pad` actually buys.)  The residual ~1e-3 above the bound is the oracle's
own discretisation, not the coarse run's error.

**The guard.**  `propagators/sas.py:38::_warn_sas_chirp_sampling` -- a
`RuntimeWarning` in exactly `fresnel._warn_fresnel_chirp_sampling`'s shape, with
the `CONVENTIONS.md` sec. 2 `f"{fn_name}: ..."` prefix, naming the bound, the
grid, the measured symptom and the way out (`angular_spectrum_propagate`, exact
in this regime).  It warns and does not raise, because that is what
`fresnel_propagate` does for K1 and the two legs have to stay comparable.  The
public docstring now states the validity window at BOTH ends, and
`verbose=True` prints the near bound beside `z_limit`.

**The window is never empty.**  `z_limit` bounds `z` from above and the new
bound from below.  Over eight grids the ratio `z_limit / z_near` runs from
**45.9** (N = 1024, dx = 0.5 um, lambda = 633 nm -- the tightest measured) to
**5.6e6** (the doublet's in-glass gap), so the pair is a window and not a
contradiction.

**Measured on the doublet, after:**

| `wave_propagator` | `P_out/P_in` | diagnostics |
|---|---|---|
| default (ASM) | 0.996171 | none |
| `'rs'` | 0.996170 | none |
| `'fresnel'` | 10396.714211 | 2 x RuntimeWarning |
| `'sas'` | 10396.710108 | **2 x RuntimeWarning** (was none) |

Every value is unchanged to the digit part a recorded; only the diagnostic
column moved.  The two gaps sit at 0.00422x and 0.00109x of the bound.

**Gate.**  `probe_item3_sas.py`: 9 (N, dx, lambda, z/z_near, pad) cases spanning
both sides of the bound, plus `skip_final_phase`, a complex64 input (the kernel
cache's dtype arm) and the doublet end to end through `apply_real_lens`.
**24/24 bit-identical**, with the warning census reported separately: silent in
the archive on all 9 cases, and in the tree firing on exactly the 6 that are
below the bound.

**The pin part a left behind.**
`test_the_in_glass_gap_legs_are_reached_and_only_one_of_them_is_gated` pinned
the SILENCE on purpose, so that closing the gap would go red.  It did.  It is
restated as `..._and_both_of_them_are_gated`, now pinning the SYMMETRY of the
two legs and the COUNT (2, one per gap), so a guard that fires once, or on the
wrong leg, still fails.  Its docstring carries what moved and why.

**Not fixed, and reported instead (section 4b request 1):** both gap legs'
warnings are attributed to `_lens_real.py`'s own propagator call, because the
`stacklevel` is a literal inside the propagator and `_propagate_through_glass`
is its caller.  Making them name the user needs the LENS to re-emit, which is a
different change in a file the propagators do not own.

### 2b.2 `LensPhysics` -- item 5

**The fourth role.**  WHAT problem (geometry), HOW it is discretised
(numerics), WHICH MACHINE runs it (resources), **WHICH TERMS the model carries**
(physics).  The line against `LensNumerics` is the one that needed stating,
because both change the number that comes back: a `LensNumerics` field moves the
answer by its own TRUNCATION error -- refine it far enough and the answer stops
moving -- while a `LensPhysics` field moves it by a TERM, and no amount of
refinement anywhere else produces that term.
`b11::test_the_line_against_lensnumerics_is_measurable_not_asserted` puts that
distinction in falsifiable form (a discretisation witness that is
byte-identical, a term witness that is not, and the term orthogonal to the
discretisation).

**The nine fields**, all of them `apply_real_lens` parameters that were
keyword-only: `fresnel`, `slant_correction`, `absorption`, `seidel_correction`,
`seidel_poly_order`, `surface_frame`, `displaced_mode`, `displaced_obliquity`,
`screen_obliquity`.  MEASURED against the live signatures (`m_item5_sigs.py`):
**every one of them exists on `apply_real_lens` and on no other entry point** --
the traced / Maslov / GBD / FGA models build their screens from a ray trace
rather than from the thin-element OPD, so none of these terms has a switch
there.  That is why `physics=` is the one configuration parameter that is not on
all seven entry points, and why `_PHYSICS_FOR`'s six empty dicts are
load-bearing rather than placeholders: they are what turns a physics request
handed to a sibling through `config=` into the "not a setting X accepts" refusal
that names the owner.
`a16::test_the_physics_parameter_is_declared_exactly_where_it_applies` gates
those two facts against each other in BOTH directions -- missing where it
applies is a `TypeError` instead of a configuration, present where it does not
is a parameter that can only raise.

**Validation.**  `__post_init__` checks only what a field can be judged on
alone: three enums against `_lens_real`'s own live vocabulary tuples (borrowed
through the existing `_vocab` cache, now eight keys), five strict `bool`s and a
positive `int`.  `screen_obliquity` uses identity for the booleans and equality
for the string, exactly as `_check_screen_obliquity_support` does, so `1` cannot
be accepted here and then refused by the call, and a caller-built (non-interned)
`'auto'` is accepted.  The CROSS-field rules stay where they are and are NOT
restated -- `slant_correction` + `seidel_correction` double-counting the facet
obliquity, every one of these terms being refused under
`surface_model='displaced'`, `screen_obliquity=True` needing `carrier=` --
because two of those need a sibling config object and the third needs the
prescription, and a copy here would drift.
`b11::test_the_cross_object_rules_still_fire_through_the_config` drives all
three through the CONFIG spelling, so a config cannot be a way around a guard,
and asserts that building the same objects alone does NOT raise.

**`input_wavevector_saddle`: the decision the brief asked for, with the
reason.**  It stays KEYWORD-ONLY.  It names which stationary point the two
asymptotic evaluators expand about (audit S6), and WHICH saddle is the right one
is a property of the INPUT FIELD's spectrum, not of the optic.  Every object in
this module is designed to be built once and reused across fields; a
field-dependent setting inside one would be silently wrong the first time the
config outlived the field it was chosen for -- which is the failure the config
objects exist to prevent, not to introduce.  The `KWARG_ONLY` entry now says it
was RE-EXAMINED when `LensPhysics` landed, so a later reader can tell a decision
from an oversight, and
`b11::test_input_wavevector_saddle_is_still_keyword_only_with_the_reason` pins
both the exclusion and the presence of that sentence.

**Three settings that fit the role and did NOT move**, recorded rather than
fixed: `surface_model` (`LensGeometry`), `caustic` and `fit_basis`
(`LensNumerics`).  All three have been shipped config fields since WP-A16;
moving one is a migration for a caller who wrote
`LensGeometry(surface_model='displaced')`, and it buys nothing a caller can do
that they could not do before.  `fit_basis` is on the numerics side of the line
by its own docstring anyway (Chebyshev and Zernike span the same polynomial
space at the same total degree, so it changes the conditioning of the least
squares and not the model).  They are the partition's known ragged edge;
`docs/lens_configuration.md` says so in a table and
`b11::test_the_three_settings_that_did_not_move_are_where_they_were` pins it.

**Wiring.**  `_GROUPS` gains a fourth entry, so the resolver, `from_kwargs`,
`to_kwargs` (including part a's `strict=`), `narrowed_to`, `field_names` and
every structural walker picked it up without further edits -- which is exactly
what the WP-A16 deferral predicted.  `resolve_entry_point_kwargs` and
`_wants_config` take `physics` LAST with a `None` default, so the six entry
points that have no physics parameter -- four of them in WP-B7's files -- call
them completely unchanged.

**Gate.**  `probe_item5_physics.py`: 9 keyword spellings through
`apply_real_lens` (including `surface_model='displaced'` and a carrier-borne
`screen_obliquity`), the three shipped config groups' resolved calls, the
`LensConfig` triple, and the serialised field tables and default lists of all
three shipped groups.  **20/20 bit-identical.**  The tree run additionally
asserts in-process that `physics=LensPhysics(...)` lands on the same bytes as
the equivalent keywords -- **9/9 cases, no mismatches** -- which is the half the
archive cannot express, because the parameter does not exist there.

**Census.**  `test_audit2609_a16_lens_config_round_trip.py` gains the
`LensPhysics` column everywhere it walks, 13 refusal rows, the legal-value
counter-pin, a `_GROUPS` counter-pin (a group added there but not here would be
walked by every loop and tabled by none, and the file would still pass), and the
four behaviour tests above: **111 -> 128 tests**, all green.

### 2b.3 `doe.py`'s zero fill, and warning attribution -- item 8

**The fill.**  `elements/doe.py:555` -- `T = np.where(inside, T, 0.0 + 0j)`
becomes `T = np.where(inside, T, np.zeros((), T.dtype))`, WP-A22 sec. F5's exact
one-line request, and its `('elements/doe.py', 539)` entry is deleted from
`_P3_ALLOWLIST` in `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`, so the
structural walk now CONFIRMS the site instead of exempting it.

**A correction to WP-A22's rationale, because it is measured.**  That report
rated the site P3 today and "P1 the moment the phase is built at a narrower
dtype".  The second half does not survive re-measurement.  MEASURED on
NumPy 2.4.6:

| `T.dtype` | literal `0.0 + 0j` | `np.zeros((), T.dtype)` |
|---|---|---|
| complex64 | complex64 | complex64 |
| complex128 | complex128 | complex128 |
| float32 | complex64 | float32 |
| float64 | complex128 | float64 |

Under NEP 50 weak promotion a Python complex scalar does not widen a complex
array at all, so a complex64 phase would have kept complex64 with the literal
too; the arm where the literal really changes the dtype is a REAL `T`, which
this entry point's `exp(1j * phase)` can never produce.  The migration is still
worth making -- it is explicit and version-independent, where NumPy 1.x decided
this by value-based casting and 2.x by weak promotion -- but not because a
promotion was about to happen here.  The table is a test
(`b11::test_what_the_literal_fill_actually_promotes`), so if it moves the
reasoning moves with it, and the corrected statement is in the allowlist comment
and in `doe.py`'s own comment.

**Warning attribution: what was measured, and what was wrong.**  `stacklevel`
counts frames, so a literal is right for exactly one call path -- and this
family has several to the same source line.  MEASURED (`m_item8_stack.py`, every
call made from the measurement file so that "the caller" is unambiguous):

| call | notice | attributed to, BEFORE | AFTER |
|---|---|---|---|
| `apply_real_lens(...)` | aperture > grid | the caller | the caller |
| `apply_real_lens(..., numerics=...)` | aperture > grid | **`_lens_real.py:6130`** | the caller |
| `apply_real_lens(..., physics=...)` | aperture > grid | **`_lens_real.py:6130`** | the caller |
| `apply_real_lens(..., freeform)` | freeform dropped | the caller | the caller |
| `apply_real_lens_traced(...)` | aperture > grid (its own) | the caller | the caller |
| `apply_real_lens_traced(...)` | aperture > grid x2, through its internal `apply_real_lens` | **`_lens_traced.py:10388,10413`** | `concurrent/futures/thread.py:73` (see below) |
| `apply_real_lens_traced(...)` | Newton-inversion notice | **`_lens_traced.py:13340`** | the caller |
| `prepare_real_lens_traced(...)` | all five notices | **`_lens_traced.py:15238,10433,10462`** | the caller, 5 of 5 |

The `numerics=` / `physics=` rows are the interesting ones: WP-A16's
configuration objects made every entry point re-enter ITSELF once when a config
is passed (`return apply_real_lens(E_in, **resolve(...))`), so on a configured
call every hard-coded level in the body was one frame short and named the
library's own re-entry line.  The v5.40 wrapper/impl split had done the same
thing one release earlier, which is precisely what `_WARN_STACKLEVEL = 3` was
introduced for; the config objects then broke it again.  A literal cannot be
right for both.

**The fix.**  `_lens_kernels.caller_stacklevel()` -- walk out from the calling
frame to the first frame whose file is not under the `lumenairy` package
directory, and return that depth (counted the way `warnings.warn` counts, so no
offset is needed when the helper is called from the frame that warns).  It is
the same rule Python 3.12's `warnings.warn(..., skip_file_prefixes=)` applies,
written out so it also holds on the 3.10 and 3.11 this package supports
(`pyproject.toml`: `requires-python = ">=3.10"`), and it runs only on a warning
path, which is never hot.  Applied at **9 sites in `_lens_real.py`** (the 7 that
read `_WARN_STACKLEVEL`, which is gone, and the 2 accumulator-store cleanup
warnings that read `stacklevel=2`) and **31 sites in `_lens_traced.py`**, plus
`_warn_if_aperture_exceeds_grid`, whose `stacklevel=` now defaults to `None`
meaning "compute it" and still honours an explicit integer.
`b11::test_no_literal_stacklevel_is_left_in_the_two_lens_bodies` is the ratchet:
an AST walk that fails on a `warnings.warn` carrying a literal integer level in
either file.

**The one row that still names non-user code is correct.**  On the parallel-amp
path `apply_real_lens` runs inside a `ThreadPoolExecutor` worker, so the stack
bottoms out in `concurrent/futures/thread.py` -- there IS no user frame on that
thread, and the outermost frame is the honest answer.  The helper's "whole stack
inside the package" arm returns the outermost in-package depth for the same
reason; both arms are tested.

**Gate.**  `probe_item8_stack.py`: 7 `create_fresnel_zone_plate` arms with their
dtypes (both branches x `n_zones` in {None, 3, 12}, plus an off-centre odd
grid), and 7 lens fields (analytic plain / warned / chunked / complex64, traced
plain / warned, and a `PreparedTracedLens` call).  **20/20 bit-identical** -- a
`stacklevel` is metadata, and no field moved.

**Scope, stated.**  `lenses_maslov.py` (11 sites) and `lenses_gbd.py` (1) are
WP-B7's; the exact edits are section 5b.  `_lens_jax.py` has no `warnings.warn`
at all (measured, so item 8's "JAX warning sites" is an empty set).
`_lens_traced_multibranch.py` (4), `_lens_thin.py` (2), `_lens_imap.py` (1) and
`_lens_traced_uniform.py` (1) are lens-family files this brief does not name;
they are section 4b request 2.  `propagators/carrier.py` was restricted by the
brief to item 20's table and was not swept -- request 3.

### 2b.4 `PMM2DStackHybrid.truncation` -- item 4

Part a guarded `formulation`, `cascade` and `symmetry`, and recorded that
`truncation` had the same shape: validated in `__init__`, a plain attribute
afterwards, and read through `!= "circular"` tests that a typo silently fails --
so `st.truncation = 'circle'` was ACCEPTED and the stack quietly solved the
larger, slower, DIFFERENT rectangular full box.

It is now a property sharing ONE vocabulary with the constructor:
`_TRUNCATIONS` and `_check_truncation` (which keeps `__init__`'s original
message wording, so a caller who has been reading that sentence sees the same
one from the setter), called from both.
`b11::test_the_constructor_and_the_setter_share_one_vocabulary` sweeps both
legal values and five illegal ones through BOTH routes and asserts the two
verdicts agree with the vocabulary, which is what makes "one definition"
checkable rather than asserted, and
`test_every_validated_model_choice_is_now_a_property` is the census that catches
a fifth one added as a plain attribute.

The caches already key on it correctly (`_geom_key` carries it), so the only
behaviour change is the refusal.  **Gate:** `probe_item4_stack2d.py` -- a
patterned Si cell solved at both truncations x both formulations, hashing the
whole solve output and the four attribute reads: **8/8 bit-identical**.

---

## 3b. Files touched

**Library**

* `lumenairy/propagators/sas.py` -- `_warn_sas_chirp_sampling` (new, with its
  derivation and measurement table), the call site, the `verbose` line, the
  public docstring's validity-window paragraph.
* `lumenairy/elements/lens_config.py` -- `LensPhysics` (new), `_PHYSICS_FOR`,
  `_GROUPS`, `LensConfig.physics`, `from_kwargs`, `CONTRACT_PARAMETERS`,
  `_CONFIG_PARAMETERS`, `resolve_entry_point_kwargs(physics=)`,
  `_wants_config(physics=)`, `_vocab`'s three new keys, `__all__`, the module
  docstring; nine `KWARG_ONLY` entries removed (they are fields now) and
  `input_wavevector_saddle`'s reason extended.
* `lumenairy/elements/_lens_real.py` -- `physics=` on `apply_real_lens` and its
  docstring, the resolver call, `_WARN_STACKLEVEL` replaced by
  `_caller_stacklevel()` at 9 sites, the `_lens_kernels` import.
* `lumenairy/elements/_lens_traced.py` -- 31 literal `stacklevel=` replaced by
  `_caller_stacklevel()`, the `_lens_kernels` import, one docstring restated.
* `lumenairy/elements/_lens_kernels.py` -- `caller_stacklevel` (new),
  `_PACKAGE_ROOT`, and `_warn_if_aperture_exceeds_grid`'s `stacklevel=None`
  default.
* `lumenairy/elements/doe.py` -- the zone-plate fill, and a comment on each of
  the two `np.where` branches saying why only one of them needed it.
* `lumenairy/elements/pmm/stack2d.py` -- `_TRUNCATIONS`, `_check_truncation`,
  the `truncation` property, `__init__` calling the shared check.
* `lumenairy/__init__.py`, `lumenairy/elements/__init__.py` -- `LensPhysics`
  imported and exported.

**Tests**

* `tests/unit/test_audit2609_b11_hygiene.py` -- `TestTheSasNearFieldGate` (6),
  `TestLensPhysics` (9), `TestTheZonePlateZeroFill` (8),
  `TestWarningAttribution` (4), `TestStack2DTruncationGuard` (4), and the module
  docstring's section list.  **54 -> 82 tests**; nothing weakened or removed.
* `tests/unit/test_audit2609_a15a_lens_covering_array.py` -- the gap-leg pin
  restated and renamed `..._and_both_of_them_are_gated`.
* `tests/unit/test_audit2609_a16_lens_config_round_trip.py` -- the census
  extended to the fourth group; **111 -> 128 tests**.
* `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` -- the
  `elements/doe.py` allowlist entry deleted, with the reason and the
  re-measurement.

**Docs / history**

* `docs/lens_configuration.md` -- four objects rather than three, the
  `LensPhysics` (9 fields) table and its validation contract, the keyword-only
  table pruned by nine rows and given the `input_wavevector_saddle` row, and
  "Deferred: `LensPhysics`" replaced by "`LensPhysics`, and the three settings
  that did not move".
* `docs/history/lumenairy.propagators.sas.md`,
  `lumenairy.elements._lens_real.md`, `lumenairy.elements._lens_traced.md`,
  `lumenairy.elements.doe.md`, `lumenairy.elements.pmm.stack2d.md` -- all
  re-recorded in this change with `scripts/record_history_fingerprints.py`.
  `_lens_kernels.py` has no history document (it is new in part a).
* This section of `WP-B11_REPORT.md`, and `WP-B11_CHANGELOG.md`.

---

## 3b-2. Tests run

Every run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one
process at a time.

| slice | result |
|---|---|
| `-k "lens_config or a16 or real_lens or lens"` -- the whole lens family | **1100 passed, 19 skipped** |
| `..._b11_hygiene.py`, `..._a15a_lens_covering_array.py`, `..._a16_lens_config_round_trip.py`, `..._a16_lens_config_bit_identity.py`, `..._a16_verify_config_and_arch.py`, `..._a17_history_lint.py`, `..._a17_history_relocation.py`, `test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` | **1285 passed** |
| `-k "sas or lens_config or doe or stack2d"` | **605 passed, 4 skipped** |
| `-k "real_lens"` | **160 passed, 3 skipped** |
| `-k "sas or propagat or doe or fresnel or pmm or stack2d or warn"` (the wide sweep) | **3868 passed, 17 skipped, 4 failed -- none of them this package's; see below** |
| `ruff check` | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **every history document matches its module** |
| `python validation/run_all.py` | **ALL 37 files passed** |
| `validation/elements/test_lenses.py` | **46/46 passed** |
| `validation/propagators/test_propagation.py` | **46/46 passed** |

**The four failures in the wide sweep, each traced to its owner.**  The tree
carries other packages' uncommitted work, so every one was re-run against the
pristine `c62c2f14` checkout (`git archive c62c2f14 lumenairy tests`, run with
cwd and `PYTHONPATH` set to it, `lumenairy.__file__` asserted):

| failure | at pristine `c62c2f14` | verdict |
|---|---|---|
| `test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` | **also fails**, identical message (`every cell of both devices was screened out ... screened: [('uncoated ns=3', 10, 0, 2, 0), ...]`) | pre-existing at HEAD |
| `test_v4_16_0_agent_d_validity_ranges.py::test_validity_warning_is_one_shot_per_pair` | **also fails**, identically, when run after `test_audit_w4_glass_registry_meshgrid.py` (which warms the same `('N-BK7', 200e-9)` pair into `glass._validity_warned`); passes alone in both trees | pre-existing order-dependent state leak, NOT the warning-attribution change -- `glass.py` was not touched here and the one-shot set is its own, not the filter registry |
| `test_audit_propagation.py::...SolveEnvelopeStationaryBatchContract::test_singular_hessian_pixels_flagged_not_converged` | **passes** | working-tree only: `AttributeError: '_WellPosedFit' object has no attribute 'basis_index_columns'` raised from `asymptotic_maslov.py:705` -- VERIFY-B7's files |
| `...::test_genuinely_converged_pixel_still_marked_true` | **passes** | same, same traceback |

---

## 4b. Requested changes (for the orchestrator / maintainer)

1. **The in-glass gap legs' warnings name `_lens_real.py`, not the user.**  Both
   `'sas'` and `'fresnel'` now warn on the covering-array doublet, but the
   warning's `stacklevel` is a literal inside the propagator and its caller is
   `_lens_real._propagate_through_glass`, so a user who wrote
   `apply_real_lens(wave_propagator='sas')` is pointed at library source.
   Making them name the user needs the LENS to catch and re-emit at its own
   entry point, which is a design decision about whose diagnostic it is.
   Owner: `propagators/` plus `_lens_real._propagate_through_glass`.
2. **The rest of the lens family's `stacklevel` literals.**
   `_lens_traced_multibranch.py` (4 sites), `_lens_thin.py` (2),
   `_lens_imap.py` (1) and `_lens_traced_uniform.py` (1) are lens-family modules
   this brief does not name.  The change is mechanical and identical to
   section 5b's: import `caller_stacklevel` from the leaf and replace the
   literal.  The b11 ratchet's file tuple should grow to match.
3. **`propagators/carrier.py`'s warning chain** was excluded by the brief
   (carrier.py was scoped to item 20's table only).  Item 8's original text
   names it; it is unswept.
4. **`surface_model` / `caustic` / `fit_basis`** fit the `LensPhysics` role and
   were deliberately left on `LensGeometry` / `LensNumerics` (section 2b.2).
   Moving one is a migration for a shipped API; it is a maintainer's call.
5. **WP-A22's forward-looking P3 -> P1 rating for the `doe.py` fill is wrong**
   under NEP 50 (section 2b.3).  The migration landed anyway; that report's
   rationale should be read with the corrected table.

---

## 5b. Deferred to the orchestrator -- the exact edits in WP-B7's files

Apply AFTER VERIFY-B7 lands.  Line numbers are RE-DERIVED against the working
tree at `f64444ec` + VERIFY-B7's in-flight edits (they moved by ~28 lines when
WP-B7 landed, which is exactly why the ANCHORS below are the text and not the
numbers -- re-grep before applying).

**None of WP-B7's files was opened for writing by this work package.**
`git status` shows `lumenairy/elements/lenses_maslov.py` modified; that edit is
**not mine** -- its added lines are VERIFY-B7's re-measurement prose (dated
2026-09-14, "RE-MEASURED (VERIFY-B7, 2026-09-14) on a THIRD chart ...") and
contain no occurrence of `_lens_kernels`, `caller_stacklevel` or `stacklevel`.
`git diff HEAD -- lumenairy/elements/lenses_maslov.py | grep '^+' | grep -E
'_lens_kernels|caller_stacklevel|stacklevel'` is empty.

Also re-confirmed: `git diff --name-only c62c2f14 f64444ec` touches none of this
package's modules, so the `git archive c62c2f14 lumenairy` baseline every gate
above ran against is still the right pre-change library for them.

### 5b.1 `lumenairy/elements/lenses_maslov.py` -- the import

This is item 5-NEW's one-line change and item 8's prerequisite in the same edit.
Replace (at `lenses_maslov.py:318-325`, under the comment
`# Other shared helpers still live in lenses.py.` at line 317):

```python
from ._lens_real import _normalise_stop_index
from .lenses import (
    NUMEXPR_AVAILABLE,
    _ensure_numexpr_loaded,
    _fit_normaliser,
    _multi_indices_total_degree,
    _warn_if_aperture_exceeds_grid,
)
```

with:

```python
from ._lens_kernels import (
    _warn_if_aperture_exceeds_grid,
    caller_stacklevel as _caller_stacklevel,
)
from ._lens_real import _normalise_stop_index
from .lenses import (
    NUMEXPR_AVAILABLE,
    _ensure_numexpr_loaded,
    _fit_normaliser,
    _multi_indices_total_degree,
)
```

`_lens_kernels` is a LEAF (it imports `numpy`, `warnings`, `os` and `sys` and
nothing from `lumenairy`), so this adds no edge.  The back-edge to `lenses`
carries **5 -> 4** names; it closes ENTIRELY only when the remaining four
(`NUMEXPR_AVAILABLE`, `_ensure_numexpr_loaded`, `_fit_normaliser`,
`_multi_indices_total_degree`) follow into the leaf, which
`docs/lens_configuration.md` "Module layout" already records.  Gate: the
a4 / b1 / b7 Maslov fixtures byte-identical, plus
`tests/unit/test_v5_2_walker_shell_vs_canonical.py`.

### 5b.2 `lumenairy/elements/lenses_maslov.py` -- the 11 warning sites

Each is a `warnings.warn(...)` argument; replace the literal with the computed
level.  Current lines and spellings:

| line | from | to |
|---|---|---|
| 559 | `RuntimeWarning, stacklevel=3)` | `RuntimeWarning, _caller_stacklevel())` |
| 1602 | `RuntimeWarning, stacklevel=3)` | `RuntimeWarning, _caller_stacklevel())` |
| 2377 | `RuntimeWarning, stacklevel=2,` | `RuntimeWarning, _caller_stacklevel(),` |
| 2394 | `RuntimeWarning, stacklevel=2,` | `RuntimeWarning, _caller_stacklevel(),` |
| 2542 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |
| 2572 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |
| 2824 | `UserWarning, stacklevel=2)` | `UserWarning, _caller_stacklevel())` |
| 2874 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |
| 3135 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |
| 3240 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |
| 4300 | `RuntimeWarning, stacklevel=2)` | `RuntimeWarning, _caller_stacklevel())` |

Equivalently, once 5b.1 is in place, `re.subn(r'stacklevel=[23]\b',
'_caller_stacklevel()', src)` -- then CHECK that no hit landed inside a
docstring.  One did in `_lens_traced.py` (a closure's own
"``stacklevel=3`` on all three" note, restated by hand); `lenses_maslov.py` has
no such prose, but the check costs one grep.

### 5b.3 `lumenairy/elements/lenses_gbd.py` -- one site

Add after the existing `from .lens_config import (...)` block (lines 50-57):

```python
from ._lens_kernels import caller_stacklevel as _caller_stacklevel
```

and at line 483, in `apply_real_lens_gbd` (unmoved by WP-B7):

```
-            "it act.", RuntimeWarning, stacklevel=2)
+            "it act.", RuntimeWarning, _caller_stacklevel())
```

### 5b.4 After applying 5b.1 - 5b.3

* re-record both modules in the same change:
  `python scripts/record_history_fingerprints.py lumenairy/elements/lenses_maslov.py lumenairy/elements/lenses_gbd.py --reason "..."`;
* extend `b11::test_no_literal_stacklevel_is_left_in_the_two_lens_bodies`'s
  file tuple to cover them, so they cannot regress either;
* `_lens_jax.py` needs NO edit: it contains no `warnings.warn` (measured).

---

## 6b. Items not reached

Taken in the brief's priority order, the budget reached items 3 (the P1), 5, 8
and 4.  Nothing on the deferred list was started:

* **Item 5-NEW (the `lenses <-> lenses_maslov` cycle)** -- the edit is written
  out (section 5b.1) but not applied, because the file is WP-B7's.  It reduces
  the back-edge by one name; it does not close the cycle.
* **Item 2's remaining `rcwa/_core.py` split** -- not started (part a sec. 2.2
  carries the per-block hazard list: module-level mutable state, four
  monkeypatching test files).
* **Item 3's whole-grid body** -- not started; part a sec. 2.3 records why it is
  NOT a bit-identical refactor (folding it would move the numexpr gate, the
  `_ensure_full_grids` path and the fresnel dtype-promotion point).
* **Item 4's remaining two cycles** -- not started; the PEP 562 forward that the
  `_NUMBA_AVAILABLE` monkeypatch and the lazy `cp` / `_ne` slots need is written
  up in `docs/lens_configuration.md`.
* **Item 14 (direct-matrix MFT)** -- not started.
* **Item 18 (`_collins_transport` on JAX)** -- not started; part a measured that
  the brief's skip clause does not apply (jax IS importable here) and that the
  work is a chain of `xp` plumbing through six helpers, not a signature.
* **Item 20's near-focus table** -- not started; the fixture is
  `scratchpad/b11/m_item20_gapkernel.py` and what is missing is the
  envelope/field bookkeeping, per part a sec. 2.20.

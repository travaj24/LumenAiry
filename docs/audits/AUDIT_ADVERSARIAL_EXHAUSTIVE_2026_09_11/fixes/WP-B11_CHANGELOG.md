# WP-B11a -- release text for 5.47.0

Part **a** of the hygiene pass.  Every entry below is either a refactor that
ships with a bit-identity proof, an additive keyword whose default is unchanged,
or a measurement written down.  **No default moves in this work package**, so
there are no migration notes.

---

### Added -- `lumenairy/_branchcut.py`, the one on-cut band the modal engines share

The RCWA, EME, PMM and BOR engines each resolve a square root's sign with a band
relative to their own spectrum, and the comparison itself was written out four
times.  It is now one function, `band_mask(r, *, scale, band, xp=None)`
(`lumenairy/_branchcut.py:63`), called from `rcwa/_core.py:1607`,
`eme/_branch.py:164`, `pmm/_core.py:828` and `bor/_orient.py:251`.  Each engine
keeps its own derived SCALE, because the four are different quantities: the
Cartesian engines floor the spectrum's top at a dimensionless 1.0 while the EME
and BOR engines carry a wavenumber and floor at `|k0|`, where a dimensionless
literal would make the branch decision depend on the caller's unit system.

The PMM band gains a name, `pmm/_core.py:795::_PMM_CUT_BAND_REL`, so the 1e-8 it
shares with `rcwa/_core._CUT_BAND_REL` is visible rather than inlined.

### Added -- two forward selectors, and the measurement that says they are not one

`negate_forward` (`_branchcut.py:92`) selects with `where(flip, -z, z)`;
`signed_forward` (`_branchcut.py:105`) multiplies by a real `+/-1`.  On real
arrays they agree bit for bit.  On COMPLEX arrays they do not, and the module
docstring carries the census: a complex multiply forms `(a*1 - b*0, a*0 + b*1)`,
so it rewrites the sign of a zero imaginary part (`1+0j` -> `-1+0j` against
`-1-0j`) and turns an infinite real part's cross term into a NaN
(`inf+0j` -> `-inf+nanj`), where negation touches only sign bits.  The RCWA layer
root keeps the real-multiply form its gradient measurement chose; the PMM and BOR
selectors keep negation.  Merging them would move every branch decision in the
library by a sign bit, which is why they are two named functions rather than one
with a flag.

### Changed -- `rcwa/_core.py` is read in sections, and its geometry predicates are a leaf

`elements/rcwa/_geometry.py` (new, 235 lines) holds the analytic-shape support
function, the exact pair-overlap test, the unit-disk hit tests and the
y-invariance test -- numpy and nothing else from the library, so it cannot be
half of an import cycle.  `_core` re-exports all of them
(`rcwa/_core.py:130`), so `_core._shapes_overlap` and
`rcwa._shape_support` resolve exactly as before and `__all__` is unchanged.
`_core.py` 5119 -> 4967 lines, and its module docstring now carries the section
map and says why `__all__` lists 95 names of which 90 are private.

`_validate_shapes` deliberately STAYS in `_core` (`rcwa/_core.py:2204`): it is
policy rather than geometry, and it must resolve `_shapes_overlap` through
`_core`'s own globals, which is where
`test_niche_audit_w9_overlap_exact.py::test_w9d_the_predicate_counter_sees_a_pair_that_needs_it`
substitutes a counting wrapper.

### Changed -- the BOR staggered-wall anchor's measurement prose moved to `docs/audits/`

`STAGGERED_WALL_ANCHOR` (`elements/bor/coupled_radial_eigensolver.py:76`) carried
~45 lines of convergence tables on one constant.  The constant now states the
live contract -- what each value means, which ships, and that `'ghost'` is a
known-defective escape hatch -- and points at
`docs/audits/MEASURE_BOR_STAGGERED_WALL_ANCHOR_2026_09_13.md`, which carries the
p = 0.99 against p = 1.99 tables, the rejected antisymmetric-ghost stencil and
the grazing-cutoff reproducer's numbers.

### Changed -- one row-band schedule for the chunked lens surface paths

`_lens_real.py` built the band arithmetic -- `r0`, `r1`, the clipped halo
`[h0:h1)` and the band's own slice `[lo:hi)` inside it -- in four places, in two
different spellings.  It is now `_row_bands(n_rows, chunk_rows, halo)`
(`elements/_lens_real.py:2929`) and `_band_in_halo(r0, r1, h0)` (`:2910`), read
by `_band_any_sag` (`:6747`), the obliquity band (`:6768`), the plain chunked
screen (`:7262`) and the slant/fresnel chunked screen (`:7409`).  The halo is
clipped at the true grid edges by the generator, which is what keeps the first
and last band's one-sided gradient stencils identical to the whole grid's -- the
property the banded paths' byte-identity rests on.

### Added -- `elements/_lens_kernels.py`, and one fewer import cycle in the lens family

The grid-versus-aperture bookkeeping (`_collect_semi_diameters`,
`check_grid_vs_apertures`, `recommend_grid_for_prescription`,
`_warn_if_aperture_exceeds_grid`) moved out of the `lenses` facade into a leaf
that imports `numpy` and `warnings` and nothing from `lumenairy`
(`elements/_lens_kernels.py`).  `lenses` re-exports all four
(`elements/lenses.py:56`) -- the same objects, by identity -- and
`_lens_traced.py:597` now reads the leaf instead of reaching back into the
facade, which closes the `_lens_traced <-> lenses` module-level 2-cycle
outright.  The family's module-level 2-cycle count is 3 -> 2; the remaining two
and the exact edit each needs are in `docs/lens_configuration.md` section
"Module layout".

### Added -- `LensConfig.to_kwargs(strict=True)`

`elements/lens_config.py:1250`.  Raises instead of dropping when a field the
config actually REQUESTS -- one whose value differs from its dataclass default --
is not a keyword of the named entry point.  A field left at its default is not a
request and is still dropped quietly, including under `include_defaults=True`,
because dropping it changes no argument.  `strict=True` without an
`entry_point` is refused (`:1159`), since nothing is narrowed there and nothing
could be dropped.  `narrowed_to(ep).to_kwargs(entry_point=ep, strict=True)`
therefore never raises, and remains the explicit "yes, drop them" route.
Default behaviour is unchanged.

### Added -- `sampling=` on the free-space HFPI entry points

`propagate_hfpi_freespace_aperture` (`propagators/hfpi.py:989`) and, through its
`**kwargs`, `propagate_hfpi` take `sampling={'uniform', 'stratified'}` with
`sampler`, `n_strata_xy` and `n_strata_dir`, routing to `init_paths_stratified`
(`hfpi.py:1087`) exactly as the prescription walk does.  **The default stays
`'uniform'`**, which is the draw this pair has always made: HFPI is a
Monte-Carlo estimator whose realisation depends on the placement rule, so
flipping it would move every existing caller's numbers.  A `sampler` or a
stratum count passed WITHOUT `sampling='stratified'` is refused rather than
silently dropped (`hfpi.py:1082`).

### Changed -- `pmm_jones_2d` assembles a layer's tensor operators once

On an out-of-plane or slanted cell at normal incidence the even-parity fold
probe ran the whole projected-operator assembly, answered `None` because the
fold does not apply there, and the cascade then assembled it again.  The build
is hoisted (`elements/pmm/twod_jones.py:1052`) and passed to both calls
(`:1058`, `:1074`).  `_tensor_layer_modes` only reads that dict, so one build
serves both and the answer is bit-identical by construction.

### Changed -- `PMM2DStackHybrid` keeps validating `formulation` / `cascade` / `symmetry`

`__init__` refused an out-of-vocabulary value and then stored it as a plain
attribute, so `st.formulation = 'fff_nv'` was ACCEPTED and read through an
`== 'li'` test that a typo silently fails -- the stack behaved as `'laurent'`
and said nothing.  The three are properties now (`elements/pmm/stack2d.py:470`,
`:467`, `:476`) sharing one vocabulary with the constructor
(`_check_formulation` `:91`, `_check_cascade` `:103`, `_symmetry_on`), and
`st.symmetry = 'auto'` resolves to `True` on assignment exactly as the
constructor argument does.  The caches already keyed on these attributes
correctly, so the only behaviour change is the new refusal.

### Changed -- `get_glass_index` memoises the whole immutable-catalogue resolution

The memo was read four branches down, so it short-circuited only the closed-form
evaluation and not the walk to it.  It is read once, immediately after the
user-callable branch (`lumenairy/glass.py:1825`), and the live
refractiveindex.info arm is memoised too (`:1972`) -- the arm that dominates,
since a bundled-Sellmeier name reached a memo and a catalogue name never did.
MEASURED on this box (20 000 calls each, threads pinned):

| glass | before | after |
| --- | --- | --- |
| `'air'` | 0.140 us | 0.166 us |
| `'N-BK7'` (catalogue) | 13.283 us | 0.559 us |
| `'N-SF11'` (bundled Sellmeier) | 4.976 us | 0.585 us |
| `'N-BAF10'` | 12.809 us | 0.618 us |
| `'N-SF6HT'` | 15.437 us | 0.552 us |

The hoist is warning-neutral: a memo hit can only follow a miss for the same
`(name, wavelength)` that already ran `_maybe_warn_outside_validity`, whose
warn-once set is keyed more coarsely (0.1 nm against the memo's 1 pm), and
`_clear_glass_caches` empties both together.  A registered CALLABLE is still not
memoised -- it is user code.

### Added -- a glass-registry generation counter

`glass.glass_registry_generation()` (`lumenairy/glass.py:1618`) returns a
monotone counter bumped inside the cache lock by `_invalidate_glass_name` and
`_clear_glass_caches`.  A downstream cache that holds DERIVED glass values --
`raytrace.jax_trace._build_jax_prescription`'s compiled prescription most of all
-- can key on it and invalidate automatically for one integer compare.  A direct
`GLASS_REGISTRY[name] = ...` cannot bump it, which is the contract the value
cache has always had.

### Changed -- the VERIFY-A6 Gaussian oracle is stated in this library's convention

`tests/unit/test_audit2609_a6_verify_carrier.py::_abcd_field` built
`1/q = 1/R - i lam/(pi w^2)` -- Siegman's `exp(+i omega t)` pairing -- and then
added the Gouy phase separately as `angle(q/q2)`, which in this library's
`exp(-i omega t)` convention (`CONVENTIONS.md` Section 7) is the wrong sign: a
`2 arctan(z/zR)` error, exactly pi across a focus.  Every assertion in the file
is piston-free, so nothing failed and nothing could have.  The oracle is now the
whole-function form `exp(i k z)/(1 + z/q) * exp(i k r^2/(2 q(z)))` with
`1/q = 1/R + i lam/(pi w^2)`, in which the amplitude, the curvature and the Gouy
phase are the argument of ONE complex number and cannot be in two conventions at
once, and one ABSOLUTE piston-included assertion pins it.  MEASURED: the
corrected oracle's on-axis Gouy reads 0.000000 / -0.463648 / -0.785398 /
-1.249046 rad at z/zR = 0 / 0.5 / 1 / 3, i.e. exactly `-arctan(z/zR)`.  All 85
existing assertions in the file pass unchanged.

### Added -- a `maslov` family in the lens covering array, and the in-glass gap legs

`tests/unit/test_audit2609_a15a_lens_covering_array.py` gains a pairwise array
over seven `apply_real_lens_maslov` physics factors on the same diverging
fixture, an 18-kwarg default-identity test, and a dedicated test for the
in-glass `'sas'` / `'fresnel'` gap legs, which the array's `propagator` factor
(`{}` and `'rs'`) never reached.  The exclusion table is empty because the full
pairwise cross-product was swept and nothing refuses, not because nobody
looked.  `integration_method` is deliberately not a
factor: measured, `'auto'` returns in 1.5 s and `'quadrature'` in 72.6 s on this
fixture, so its levels belong in a slow lane.

### Changed -- the Collins one-step readout's applicability window is on the public docstring

`propagators/carrier.py:1068`.  The K1 condition
`2 dx (|A| r/|B| + theta)/lambda` on the CHAIN'S OWN exit pitch was derived at
length in `_collins_readout`'s docstring but nowhere a caller choosing
`transport=` would read it.  The `transport` parameter now carries the window
and the two measured readings that bracket it (K1 = 0.16 on the WP-A6 fixture;
K1 = 82 for an 8 mm final distance on a 5.4 mm exit beam at 76 um).

---

# WP-B11b -- release text for 5.47.0, part b

Part **b** of the hygiene pass.  Three entries below are behaviour changes -- a
diagnostic and two refusals, each where a silent wrong answer was returned
before -- and each carries a Migration note.  **No numerical default moves.**

### Added -- the SAS propagator's NEAR-field validity bound

`scalable_angular_spectrum_propagate` bounded `z` from ABOVE only (the paper's
`z_limit`, past which the band-limit filter eats the components the
precompensation exists to correct).  Its third step is the same single-FFT
Fresnel sum `fresnel_propagate` evaluates, on the same input grid, so its
quadratic chirp `exp(i k x^2 / 2z)` has to be resolved at pitch `dx` in exactly
the same way -- and below `z_near = N*dx^2/lambda` it is not.
`propagators/sas.py:38::_warn_sas_chirp_sampling` is that bound, emitted as a
`RuntimeWarning` in `fresnel._warn_fresnel_chirp_sampling`'s shape.  Values are
unchanged: 24/24 bit-identical against `c62c2f14` over both sides of the bound,
three padding factors, `skip_final_phase`, a complex64 input and an end-to-end
in-glass lens gap.

The derivation is in the function's docstring, with the measurement that fixes
the constant: the chirp's local spatial frequency at the window edge is
`(N dx/2)/(lambda z)` against a Nyquist of `1/(2 dx)`.  `pad` does NOT enter it
-- the precompensation is a band-limited phase filter whose impulse response
stays on the input window, so the chirp's unresolved outer turns multiply the
zero padding.  MEASURED on a window-filling super-Gaussian against the same
kernel at 8x finer input pitch, the relative field error at `z = 0.2 z_near` is
2.24 at pad 2, 2.30 at pad 4 and 2.26 at pad 1 -- the same ABSOLUTE `z` breaks
all three -- and the output carries 6.0x the oracle's power there, 23.6x at
`0.1 z_near` and 85x at `0.05 z_near`.  The two bounds bracket a window that is
never empty: over eight grids `z_limit / z_near` runs from 45.9 to 5.6e6.

**Migration.**  `wave_propagator='sas'` on an in-glass lens gap now emits
`RuntimeWarning`s it did not emit before.  On the WP-A15a covering-array doublet
(N = 64, dx = 112.5 um, 9.0 mm N-BAF10 + 2.5 mm N-SF6HT) that leg returned
`P_out/P_in = 1.0397e4` in silence; it returns the same `1.0397e4` -- every
digit unchanged -- with two warnings, one per gap, at 0.0042x and 0.0011x of
the bound.  The sibling `'fresnel'` leg has warned about the same aliasing since
the K1 guard landed.  A caller relying on the silence should move to
`wave_propagator='asm'`, which is exact in this regime, or filter
`RuntimeWarning` from `lumenairy.propagators.sas`.
`test_audit2609_a15a_lens_covering_array.py::test_the_in_glass_gap_legs_are_reached_and_only_one_of_them_is_gated`
pinned that silence deliberately and is restated as
`..._and_both_of_them_are_gated`, now pinning the symmetry of the two legs and
the warning COUNT (one per gap).

### Added -- `LensPhysics`, the fourth lens configuration object

`lumenairy.LensPhysics` joins `LensGeometry` / `LensNumerics` / `LensResources`
and carries the nine `apply_real_lens` model-term switches that were
keyword-only: `fresnel`, `slant_correction`, `absorption`, `seidel_correction`,
`seidel_poly_order`, `surface_frame`, `displaced_mode`, `displaced_obliquity`,
`screen_obliquity`.  `apply_real_lens` gains `physics=`, `LensConfig` gains a
fourth component, and `from_kwargs` / `to_kwargs` (including part a's
`strict=`) / `narrowed_to` / `field_names` reach it for free because they all
walk `_GROUPS`.  Purely additive: every keyword still works with the same
default, a config whose fields are all at their defaults is indistinguishable
from passing none, and `physics=LensPhysics(fresnel=True)` is byte-identical to
`fresnel=True` (9/9 cases; 20/20 bit-identical against `c62c2f14`).

The line against `LensNumerics` is stated and testable: a numerics field moves
the answer by its own TRUNCATION error, a physics field moves it by a TERM.
`physics=` is the one configuration parameter that is not on all seven entry
points, because MEASURED against the live signatures every one of these nine is
a parameter of `apply_real_lens` and of no sibling -- the traced / Maslov / GBD
/ FGA models build their screens from a ray trace, so none of these terms has a
switch there.  A physics request handed to a sibling through `config=` raises
and names `apply_real_lens` as the owner.

`input_wavevector_saddle` was re-examined and stays keyword-only: which
stationary point the asymptotic evaluators expand about is a property of the
INPUT FIELD's spectrum, not of the optic, and every object here is built to
outlive the field it was configured against.  `surface_model`, `caustic` and
`fit_basis` fit the physics role but were already shipped config fields; moving
one is a migration for a caller who wrote `LensGeometry(surface_model=...)`, so
they stay, and `docs/lens_configuration.md` records the three as the partition's
known ragged edge.

### Changed -- every warning in the lens family names the caller's frame

`stacklevel` counts frames, so a literal is right for exactly one call path, and
this family has several to the same source line.  MEASURED: WP-A16's
configuration objects made each entry point re-enter ITSELF once when a config
is passed (`return apply_real_lens(E_in, **resolve(...))`), so on a configured
call every hard-coded level in `_lens_real.py` was one frame short and the
aperture notice named `_lens_real.py` line 6130 (as it then was) -- the library's own re-entry line.
`prepare_real_lens_traced` attributed all five of its pre-flight notices to
`_lens_traced.py`, and `apply_real_lens_traced`'s Newton-inversion notice named
`_lens_traced.py` line 13340 (as it then was).

`elements/_lens_kernels.py::caller_stacklevel` walks out from the calling frame
to the first frame outside the `lumenairy` package and returns that depth -- the
same rule Python 3.12's `warnings.warn(..., skip_file_prefixes=)` applies,
written out so it also holds on the 3.10 this package supports.  It is now the
level at 9 sites in `_lens_real.py` (`_WARN_STACKLEVEL` is gone) and 31 in
`_lens_traced.py`, and the default for `_warn_if_aperture_exceeds_grid`.  Fields
are unchanged: 20/20 bit-identical.  Once VERIFY-B7 had landed the same sweep
reached the rest of the family: `lenses_maslov.py` (11 sites; it also takes
`_warn_if_aperture_exceeds_grid` from the `_lens_kernels` leaf now, so its
back-edge to `lenses` carries four names instead of five), `lenses_gbd.py` (1),
`_lens_traced_multibranch.py` (4), `_lens_thin.py` (2) and `_lens_imap.py` (1).
An AST ratchet (`b11::test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies`)
fails on a literal that creeps back into any of the eight -- `_lens_traced_uniform.py`
(2 sites, once WP-B7b's verifier had landed) completes the lens family.
`propagators/carrier.py`'s chain is still literal.

The one case that still names non-user code is correct -- on the parallel-amp
path the call runs in a `ThreadPoolExecutor` worker, so there is no user frame
on that thread and the outermost one is the honest answer.

**Migration.**  `warnings.filterwarnings(..., module=...)` keys on the
ATTRIBUTED frame's module, and so does the default filter's per-location dedup
registry.  A filter written against `lumenairy.elements._lens_real` or
`lumenairy.elements._lens_traced` to silence one of these notices will no longer
match; filter on the category and message, or on the calling module, instead.
Warnings that already named the caller are unaffected.

### Changed -- `doe.create_fresnel_zone_plate`'s outside-the-aperture fill

`np.where(inside, T, 0.0 + 0j)` becomes
`np.where(inside, T, np.zeros((), T.dtype))` (WP-A22 sec. F5's one-line
request), and the `('elements/doe.py', 539)` entry is deleted from
`_P3_ALLOWLIST` in `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`,
so the structural walk confirms the site instead of exempting it.
Bit-identical on all seven measured arms, dtypes included.

Re-measured while applying it (NumPy 2.4.6), and it CORRECTS WP-A22's
forward-looking rating: under NEP 50 weak promotion a Python complex scalar does
not widen a complex array, so a complex64 phase would have kept complex64 with
the literal too.  The literal only changes the dtype for a REAL `T`, which
`exp(1j * phase)` cannot produce.  The migration is worth making because it is
explicit and version-independent -- NumPy 1.x decided this by value-based
casting, 2.x by weak promotion -- not because a promotion was about to happen.
The table is pinned in `b11::test_what_the_literal_fill_actually_promotes`.

### Changed -- `PMM2DStackHybrid.truncation` is guarded on assignment

The fourth validated model choice joins `formulation` / `cascade` / `symmetry`
(part a).  It was validated in `__init__` and a plain attribute afterwards, and
the solve reads it through `!= "circular"` tests that a typo silently fails, so
`st.truncation = 'circle'` was accepted and the stack quietly solved the larger
rectangular full box.  It is now a property sharing one vocabulary with the
constructor (`_TRUNCATIONS` / `_check_truncation`, called from both, with
`__init__`'s original message wording preserved).  The caches already key on it
correctly (`_geom_key` carries it), so the only behaviour change is the refusal:
8/8 bit-identical over both truncations x both formulations.

**Migration.**  `st.truncation = <anything but 'rectangular' or 'circular'>` now
raises `ValueError` instead of being stored.  Code that relied on the silent
acceptance was solving the rectangular box whatever it wrote.

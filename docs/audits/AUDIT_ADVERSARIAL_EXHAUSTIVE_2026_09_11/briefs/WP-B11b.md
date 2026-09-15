# WP-B11b -- the second half of the hygiene pass: the two items that touch the Maslov / asymptotic files (launch after WP-B7 and VERIFY-B7 land)

Read `WP-B11_HYGIENE_PASS.md` (same directory) for the standards, the deliverable shape and the rules; this brief narrows it to items 5 and 8 plus
anything WP-B11a's report lists under "deferred to B11b".  HEAD is the newest commit; the pre-change library is `git archive <HEAD-at-launch>
lumenairy` (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest).

* **Item 5 -- `LensPhysics`** (WP-A16 deferred): the physics-model fields (`surface_model`, `displaced_mode`, `caustic`, `fit_basis`,
  `input_wavevector_saddle`, ...) as a fourth frozen dataclass beside `LensGeometry` / `LensNumerics` / `LensResources`, same `from_kwargs` /
  `to_kwargs` / precedence-or-raise contract, the A16 census updated (`tests/unit/test_audit2609_a16_lens_config_round_trip.py`), `docs/lens_configuration.md`
  gaining the table.  Note `input_wavevector_saddle` is classified KWARG_ONLY because it is a property of the INPUT FIELD (WP-B1 follow-up): decide,
  with the reason written, whether a physics-model object may carry it at all (the answer is probably no -- keep it keyword-only and say why).
* **Item 8 -- `doe.py`'s sentinel and warning `stacklevel` attribution** (WP-A22 deferred): the sentinel becomes a named module constant; every
  `warnings.warn` in the lens family (`_lens_real`, `_lens_traced`, `lenses_maslov`, `lenses_gbd`, `_lens_jax`, `lens_config`) and the carrier chain
  points at the caller's frame -- measure with `pytest.warns(...).list[0].filename` before and after, per site, and pin one site per module.
* **Item 3 (NEW, P1) -- the in-glass `'sas'` gap leg has no near-field gate** (WP-B11a sec. 2.10): on the covering-array doublet (N = 64,
  dx = 112.5 um, 9.0 mm N-BAF10 and 2.5 mm N-SF6HT gaps) `apply_real_lens(wave_propagator='sas')` returns `P_out/P_in = 1.04e4` with NO diagnostic,
  while `'fresnel'` warns twice about the same aliasing: the single-FFT chirp `exp(i k x^2 / 2z)` is under-sampled whenever
  `z < N dx^2 / lambda_medium` (2.13 m here against 9 mm), and `propagators/sas.py:199`'s only validity test is the far-field direction
  `z > z_limit`.  Derive the near-field bound for the SAS kernel yourself (its chirp is sampled on the input grid exactly like the single-FFT
  Fresnel's; state the condition and its derivation), add the guard in `scalable_angular_spectrum_propagate` (warn or refuse -- match what
  `fresnel_propagate` does for K1, with the sec. 2 prefix), and pin both directions on the doublet fixture and on a validly-sampled one; re-run
  `test_audit2609_a15a_lens_covering_array.py::test_the_in_glass_gap_legs_are_reached_and_only_one_of_them_is_gated` and restate it (it pins the
  SILENCE today so that this change is recorded deliberately).  You own `lumenairy/propagators/sas.py` for this item.
* **Item 4 (NEW) -- `PMM2DStackHybrid.truncation`** is validated in `__init__` and a plain attribute afterwards, the shape items 19 fixed for
  `formulation` / `cascade` / `symmetry`; one more property with the shared vocabulary helper, one test.  You own `pmm/stack2d.py` for this.
* **Item 5 (NEW) -- the `lenses <-> lenses_maslov` module-level import cycle**: the one-line import change at `lenses_maslov.py:282` written out in
  `docs/lens_configuration.md` "Module layout" (WP-B11a sec. 2.4), gated on bit-identity of the a4 / b1 / b7 Maslov files' fixtures.
* **Deferred-to-B11b items from WP-B11a's report** (sec. 5): item 2's remaining `_core.py` split (per-block hazard list: module-level mutable state,
  four monkeypatching test files), item 3's whole-grid body (NOT a bit-identical refactor -- record why, do not force it), item 4's remaining two
  cycles (the PEP 562 forward), item 14 (direct-matrix MFT), item 18 (`_collins_transport` on JAX -- a chain of `xp` plumbing), item 20's near-focus
  table (fixture exists in `scratchpad/b11/m_item20_gapkernel.py`; the envelope/field bookkeeping is what is missing).  Take these only after the
  items above and only as far as the budget allows; state what was not reached.

Deliverable: `WP-B11_REPORT.md` gains a "part b" section (append; do not rewrite part a), `WP-B11_CHANGELOG.md` gains the entries, tests go into
`tests/unit/test_audit2609_b11_hygiene.py` (add, do not weaken).  Every module you change with a history document is re-recorded in the same change.
Rules as in the main brief: thread env vars, one process at a time, no git write commands, do not kill processes, comments say what the code does now.

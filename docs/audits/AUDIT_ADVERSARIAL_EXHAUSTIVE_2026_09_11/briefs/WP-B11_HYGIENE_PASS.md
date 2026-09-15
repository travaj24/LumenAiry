# WP-B11 (Wave 4, last) -- the hygiene pass: consolidations and the small deferred items, each gated on bit-identity

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09` (HEAD is the newest commit; every
other Wave-4 package and verifier has landed -- read their reports' "deferred" and "requested changes" sections, listed below, because several of
these items are theirs).  Launch condition: no other engineer is editing the files below.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document" -- almost every module below has one; re-record each you change with `python scripts/record_history_fingerprints.py <module path>
--reason "..."` in the same change).  Source comments describe what the code does NOW and why; no version narrative.

## The items (each is a measured, bit-identical refactor or a small addition; take them in this order and stop when the budget runs out, stating
## which were not reached)

1. **One branch-band mask and one forward/backward mode selector** (WP-A14 D4 / D5; `rcwa._core._CUT_BAND_REL`, `eme._branch._EME_CUT_BAND_REL`,
   `pmm._core._forward_branch_flip`, `bor._orient.orient_band_scale`; `_select_forward_flux`, `_strip_split_forward`, `forward_decaying_root`,
   `bor._orient.forward_orient`): a `lumenairy/_branchcut.py` leaf with `band_mask(r, *, scale, band)` and one selector, each caller keeping its
   own derived scale/population.  Gate: bit-identity per site against all four populations (the kernel census `tests/unit/test_ci_kernel_consistency.py`
   must not move a decision).
2. **`rcwa/_core.py` organisation** (WP-A14 D6): 4 400+ lines, 88 `__all__` names of which 83 are private; the `STAGGERED_WALL_ANCHOR`
   measurement prose belongs in `docs/audits/`.  A test-only-diff split (no behaviour change; every existing import path kept through re-exports),
   gated on the whole `-k "rcwa or eme or bor"` slice byte-identical.
3. **The three surface bodies in `_lens_real.py`** (WP-A2 sec. 6 item 4: `_narrow_chunk`, `_slant_narrow_chunk`, whole grid) collapsed into one
   `for band in bands(...)` generator, gated on the 44-configuration byte-identity matrix `repro/RL-CORE/p10_banded_identity.py` (read its method;
   rebuild it in your test).
4. **`elements/_lens_kernels.py`** (WP-A16 addendum 10): extract the shared leaf the `_lens_*` / `lenses` module-level import 2-cycles need, ONLY
   with bit-identity on the lens test files (`-k real_lens` plus the A2/A3/A4/A16/B2/B10 files); otherwise document the plan in
   `docs/lens_configuration.md` "Module layout".
5. **A `LensPhysics` configuration object** (WP-A16 deferred): the physics-model fields (`surface_model`, `displaced_mode`, `caustic`,
   `fit_basis`, `input_wavevector_saddle`, ...) that today live as keyword-only per entry point, as a fourth frozen dataclass alongside
   `LensGeometry` / `LensNumerics` / `LensResources`, with the same `from_kwargs` / `to_kwargs` / precedence-or-raise contract and the A16 census
   updated (`tests/unit/test_audit2609_a16_lens_config_round_trip.py`).
6. **`LensConfig.to_kwargs(strict=True)`** (WP-A16 deferred): raise on a field the entry point does not accept instead of dropping it.
7. **The 1-D symmetric remap's input-window asymmetry** (WP-B2 deferred 1): the same centred window `_apply_displaced_remap` needs, with a mirror
   fixture on a DECENTRED INPUT FIELD; this moves `test_niche_p10_...::test_symmetric_remap_is_the_p2_1d_remap_byte_identical`, so restate that
   pin with the measurement.
8. **`doe.py:539`'s sentinel** (WP-A22 deferred) and **warning `stacklevel` attribution** (WP-A22 deferred): the sentinel becomes a module
   constant with a name; every `warnings.warn` in the lens family and the carrier chain points at the caller's frame (measure with
   `pytest.warns(...).list[0].filename`).
9. **`apply_aperture(edge='gray')` as the default?** (WP-B3 deferred 4): MEASURE on the RS and HF fixtures what the switch buys (second-order
   convergence on hard-aperture inputs, 25x at N = 1024 per WP-B3 sec. 3.2) and costs (every hard-aperture fixture's numbers move); recommend,
   with the table, and ship it ONLY behind the existing keyword unless the orchestrator has ruled otherwise -- default moves are the maintainer's.
10. **A `maslov` family in the lens covering array** (WP-B1 request 4): `tests/unit/test_audit2609_a15a_lens_covering_array.py` gains
    `apply_real_lens_maslov` cells on the existing diverging fixture (finite, shaped, no energy gain, default-passed == omitted).  Also
    (WP-B3b D5): the array's `propagator` factor is `[{}, {'wave_propagator': 'rs'}]`, so nothing in it reaches the in-glass `'sas'` /
    `'fresnel'` gap legs; add a `{'wave_propagator': 'fresnel'}` level (and `'sas'` if the fixture is square) so both directions of the
    window-against-period gate are covered where the rest of the lens matrix already is.
11. **`pmm_jones_2d` assembles the tensor operators twice on an out-of-plane or slanted normal-incidence cell** (WP-B6 deferred 3): build once with
    `_tensor_projected_ops`, pass to both calls; bit-identical by construction, with a count test.
12. **`sampler=` on the free-space HFPI entry points** (WP-B3 deferred 6): a `sampling='uniform' | 'stratified'` kwarg on `propagate_hfpi` /
    `propagate_hfpi_freespace_aperture` routing to `init_paths_stratified`, default unchanged.

13. **The odd-N grid-centring disagreement, swept** (WP-B8 sec. 6.1): `ifftshift` centres an odd axis on index `N // 2` while the package
    coordinate convention `(arange(N) - N/2) * dx` centres it on `N/2` -- half a pixel apart; `compute_psf(method='fft')` inherits it and so does
    anything that pairs `fftshift` / `ifftshift` with the package coordinate arrays (the A11 Z4 shape).  MEASURE: grep every `fftshift` /
    `ifftshift` site, classify each as "odd N reachable" or not, and quantify the half-pixel on a reachable odd fixture; report the table and a
    recommended single fix (a shared centring helper) WITHOUT moving any default in this package.
14. **A direct-matrix MFT branch** (WP-B8 sec. 6.2; `lumenairy/propagators/mft.py`, ONLY after VERIFY-B3 and WP-B3b have released it): Soummer's
    matrix triple product is O(N^2 M) with no padding, while the chirp-Z pads to `next_fast_len(N_in + N_out - 1)`; measured 3.1-7.5x the padded
    FFT's memory on the natural grid.  Add the triple-product branch below a DERIVED size threshold (crossover measured, both memory and time), with
    the tolerance between the two reductions derived and stated; opt-in or threshold-automatic is a ruling to request, default byte-identical.
15. **`get_glass_index(name, wavelength)` memo** (WP-B9 request 1; `lumenairy/glass.py`): 16.8 us per 'N-BK7' resolution against 0.17 us for
    'air' -- the Sellmeier evaluation is re-done on every call, and `trace()` pays two of them per call plus `_build_jax_prescription`'s residual
    53 us is 64 % this.  An `lru_cache` keyed on `(glass_name, wavelength)` invalidated by `_invalidate_glass_name` (and a registry generation
    counter the JAX prescription cache could key on).  Gate: bit-identity of every trace / prescription probe (WP-B9's `scratchpad/b9/byte_identity.py`
    method), registry-mutation re-key test, `clear_asm_caches()` empties it via `_cache_registry`.
16. **One remaining knife-edge asymptotic pin** (WP-B9 request 3; the `ModalAsymptoticStillBitEqual` arms were ALREADY restated to 3e-8 at the
    VERIFY-B9 landing because they had 4 % margin at the shipped defaults; ONLY if the orchestrator rules that `sphere_normal='analytic'`
    becomes the default):
    `test_audit_propagation.py::...ModalAsymptoticStillBitEqual` 1e-8 -> 3e-8 (one saddle-basin flip = 1.04e-8 relative, measured 2026-09-13) and
    `test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star...` 1e-15 -> 5e-15 (measured 1.15e-15).  Without that ruling: no edit.
17. **The VERIFY-A6 Gaussian oracle's phase convention** (WP-B4 sec. 8 item 2; `tests/unit/test_audit2609_a6_verify_carrier.py::_abcd_field`):
    it builds `1/q = 1/R - i lambda/(pi w^2)` and carries the Gouy phase as `angle(q/q2)` -- Siegman's `exp(+i omega t)` pairing, which in
    this library's CONVENTIONS sec. 7 convention has the Gouy phase's sign wrong (exactly pi at a focus).  Every assertion there is
    piston-free, so nothing fails today.  Restate the oracle in the whole-function form `E = exp(i k z)/(1 + z/q) * exp(i k r^2/(2 q2))`
    with `1/q = 1/R + i lambda/(pi w^2)` and `wz = sqrt(lambda/(pi Im(1/q2)))`, and add ONE absolute (piston-included) comparison so the
    convention is pinned; measure that every existing assertion still passes with the same numbers.
18. **`_collins_transport` is NumPy-only** (WP-B4 sec. 9): a CuPy / JAX field passed to `transport='collins'` is transported on the host.
    The Bluestein already takes `xp` / `fft2` / `ifft2`; thread them through and add a backend arm to `test_niche_k2_carrier_backends.py`,
    gated on bit-identity of the NumPy path.  Skip if no accelerator backend is importable on the box (say so).
19. **`PMM2DStackHybrid`'s `formulation` / `symmetry` / `cascade` are validated in `__init__` but plain attributes afterwards** (VERIFY-B6
    follow-up 3): `st.formulation = 'fff_nv'` is accepted and silently behaves as `'laurent'`.  A property guard mirroring `__init__`'s
    validation (raise with the sec. 2 prefix on an out-of-vocabulary assignment); the caches already key on the attributes correctly, so the
    only behaviour change is the new refusal.  Gate: the b6 mutation tests stay green; one derived test per attribute.
20. **The exact-kernel refinement near a focus, MEASURE and RECOMMEND only** (VERIFY-B4 F3; `lumenairy/propagators/carrier.py`, both transports):
    `gap_kernel='auto'` applies the exact/Fresnel kernel ratio over the REDUCED frame `z_eff = z/A`, which diverges at the geometric focus; K4
    bounds the WRAP of that refinement, not its accuracy, and its own dropped quartic `k |z_eff| theta^4 / 8` reaches 0.08 rad 1 um from the
    focus where `gap_kernel='fresnel'` reads 1.7e-14 against the analytic Gaussian and `'auto'` reads 2.4e-3 (linear in |z_eff|, independent of N).
    Reproduce that table on your own fixture, measure the same on the SHIPPED `'sziklas'` transport (VERIFY-B4 says it applies the same refinement),
    and recommend -- with the table -- whether `'auto'` should drop to `'fresnel'` where `k |z_eff| theta^4 / 8 > gap_env_phi_tol` (the chain's
    own tolerance for exactly this quantity).  Do NOT ship the switch: it changes what `gap_kernel='auto'` means on a leg (a default), so it is the
    maintainer's ruling.  Also state the Collins one-step readout's applicability window (its K1 on the chain's exit pitch, VERIFY-B4 F1) in the
    `transport` docstring if WP-B4's text does not already carry it.

## Deliverable

Per item: bit-identity proof against `git archive <HEAD>^ lumenairy` (child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__`
asserted; never through pytest), derived tests in `tests/unit/test_audit2609_b11_hygiene.py` (S5; no wall-clock assertions), history documents
re-recorded, report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B11_REPORT.md` (summary table: item / status / files:lines /
tests / oracle / measured before -> after; per-item sections; files touched; tests run; requested changes; items not reached) and
`WP-B11_CHANGELOG.md` (5.47.0 release text, `### Changed -- ...` / `### Added -- ...` style, citations on non-trivial lines, Migration notes only
for defaults that move -- none should without a ruling).  Every item's whole relevant test slice green; `ruff check`; `python
scripts/record_history_fingerprints.py --check`; `tests/unit/test_audit2609_a17_history_lint.py`; `python validation/run_all.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* NO git write commands of any kind; read-only git is fine.  Do not kill processes.  The orchestrator commits with an explicit file list, one
  commit per item group if the report separates them.
* You own every file an item above names, plus the new `lumenairy/_branchcut.py`, the new `elements/_lens_kernels.py` and the new b11 test file;
  nothing else.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.

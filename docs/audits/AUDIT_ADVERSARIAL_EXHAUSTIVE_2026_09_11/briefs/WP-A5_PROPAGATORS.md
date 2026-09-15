# WP-A5 — Propagator kernels: ASM / Fresnel / RS / SAS / HF / HFPI / vectorial HFPI / MHS / Richards–Wolf / FFT infra

Read first: `COMMON.md`, then the partition reports `PROP-CORE.md`, `PROP-CORE-SUB1.md`, `PROP-HF.md`,
`ORCHESTRATOR.md` (rows on `fresnel_propagate`, `backend.scipy.jv`, RS aliasing, Richards–Wolf) and report
sections §3 (rows K1–K24 in §3, §3.1, §3.2) and §15 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.
Repro: `repro/PROP-CORE/`, `repro/PROP-CORE-SUB1/`, `repro/PROP-HF/`, `repro/orch/verify_rs_alias.py`,
`repro/orch/verify_rw_ez.py`.

## Files you own
`lumenairy/propagators/`: `asm.py`, `fresnel.py`, `rs.py`, `sas.py`, `hf.py`, `hfpi.py`, `vectorial_hfpi.py`, `mhs.py`,
`vector_diffraction.py`, `fft_infra.py` (kernel/plan/lock code only — the global `set_*` knobs' context-manager
work belongs to a later WP), `dispatch.py`, `_bluestein.py`, `mft*.py`, `propagation.py`, `result.py`, `system.py`
(only if a finding needs it), and `lumenairy/backend/scipy.py` (K2). NOT `gbd.py`, `carrier*.py`, `asymptotic*.py`
(other WPs). Tests: propagation/propagator test files, plus new `tests/unit/test_audit2609_a5_*.py`.

## Findings to implement
- **K16 (P0 ✔)** `richards_wolf_focus` E_z sign: `Pz = P*(px*cp*s + py*sp*s)` at `vector_diffraction.py:347`; add the
  build-free pin `Im(E_z/E_x) < 0` for x_f > 0 on an x-polarised uniform pupil (derive the bar from the oracle in
  `repro/orch/verify_rw_ez.py`, which is independent of the auditor's), check E_z is odd in x and zero on the y-axis;
  **K10** state in the docstring that `pupil` is indexed by the physical exit-pupil (aperture) coordinate and add a
  decentred-sub-aperture test pinning the sign of the focal phase ramp. Check `debye_wolf_psf` and the other RW
  consumers for anything that depended on the old sign.
- **K9 (P0 ✔)** the point-sampled Rayleigh–Sommerfeld kernel aliases for z < 2·N·dx²/λ and `bandlimit=True` is all-pass
  there. Implement the exact RS-I transfer function built in the FREQUENCY domain on the padded grid (with the
  evanescent cut) — the sub-auditor measured it at relL2 6e-8 and power 1.000000 on every failing grid — and route the
  near-field regime to it (or raise with the guard when a caller forces the spatial kernel). Preserve the far-field
  behaviour the audit found correct (RS exact to 4e-8 for z ≥ 200 µm on the probe grid, beating ASM at long z):
  measure both regimes before/after with `repro/orch/verify_rs_alias.py` and the SUB1 scripts, and fix the docstrings
  ("near-field remedy", "machine precision") to what is measured. Also consider the Shen–Wang (2006) pixel-integrated
  kernel for the spatial path — PROP-HF measured the point-sampled kernel's first-order-in-dx convergence; implement it
  if it is cheap and measurably better, otherwise document it as deferred.
- **K1 (P1 ✔)** `fresnel_propagate` chirp aliasing (33 % at z = 0.25·N·dx²/λ): add the sampling guard (raise or
  warn + document), matching how ASM guards.
- **K2 (P1 ✔)** `backend.scipy.jv(v, x)` calls `scipy.special.jv(x, v)` — fix, add a value test; grep for every
  importer.
- **K3 (P1)** SAS cancels in float32 (0.9 rad at z = 1 m): compute the sensitive phase in float64 (reduce mod 2π before
  any float32 cast, the way the ASM does) or refuse float32 for SAS with a clear message.
- **K4 (P2)** one lock per pyFFTW plan slot instead of the global lock (measured concurrency 1 → 2–4).
- **K5 (P2, byte-identical)** cap `_get_asm_H_natural` chunking like the streamed path (4.06 → 1.26 grids); also
  evaluate folding the fftshift/ifftshift pair into the kernel (checkerboard sign / natural-order H) — RL-CORE measured
  two extra complex grids per in-glass gap; keep bit-identity or document the ULP-level difference.
- **K6, K7, K8** — read the rows; implement as stated.
- **K11 (P1)** `hf.py:179–183` / `mhs.py:633–637`: the resample + Parseval renormalisation fabricates energy when the
  output window is smaller than the source's — rescale by the power inside the target window, or warn on the crop;
  de-duplicate the block.
- **K12 (P1)** `apply_aperture_diffraction` / `apply_vector_aperture_diffraction`: `wavelength` keyword-required, raise
  on ≤ 0 (no silent 1/(iλ) drop).
- **K13 (P1) + K18 (P1) + K19 (P1)** HFPI: the estimator's amplitude depends on output pixel area and source pixel count —
  weight each path by `r·N_src_px/(dx_out²·cosθ_out)` at bin time (the exact bias law the auditor derived and
  confirmed) and fix the `Ny·Nx` source-area normalisation at `hfpi.py:222–225` and `:814–817`; `rng=None` must draw
  fresh entropy (five `RandomState(rng if rng is not None else 0)` sites) — or, if determinism-by-default is wanted,
  document it and make `_spawn_rng` honest; correct the 38-line normalisation warning to what is now true.
- **K17 (P1)** `vectorial_hfpi`: the default path is bit-identical to two scalar runs and there is no E_z. Implement
  the honest minimum: carry `Ez` on the bundle, project the Jones vector onto the transverse plane AT EMISSION and at
  each re-emission, drop the doubled obliquity, reconstruct `E_z` from ∇·E = 0 at accumulation; verify energy is
  conserved by the projection and that a 45° input depolarises off-axis. If the full dyadic re-emission is out of reach
  in this pass, rewrite the module docstring to describe what it does, delete the "m-theory dipole obliquity tensor"
  claim, and route high-NA vector focusing to `richards_wolf_focus` in the docs.
- **K14 / K23 (P2)** `cone_half_angle` accepted by both free-space entry points and threaded to the two call sites; the
  vector accumulator shares the v5.31 under-sampling guard (`_check_landed` helper) and accepts `on_undersampled`;
  `init_paths_stratified` honours `n_paths` as a cap (implement the documented sub-sampling, or raise naming the count);
  Richards–Wolf `precision='single'` either genuinely single or documented; `MhsPipeline._validate` compares `centre`;
  HFPI CuPy sites go through `to_numpy` (desk-check, CuPy absent).
- **K22 (P2)** the HF OPL quadrature: chunk the output loop by broadcasting `s2x/s2y` to `(n_chunk,1,1)` (17 `opl_fn`
  evaluations per output pixel today; 119 min for 256² → minutes), build the kernel in the caller's dtype, and
  list-accumulate on JAX; measure before/after.
- **K20, K21 (P2)** `hf` free-space return-type flip (unpack in the dispatcher or make the tuple unconditional; keep the
  `propagate(...)` contract identical across methods) and the `np.isclose` `atol=1e-8 m` no-op short-circuit
  (`abs(dx − target_dx) <= 1e-12·dx`).
- **K15, K24 (P3)** the docstring items (`hf.py` module docstring / recommended entry point, `rs.py` claims,
  `finite_diff_step`'s documented accuracy, `_spawn_rng` Generator mutation → `SeedSequence(entropy=[parent, i])`,
  replay run identity, `from_prescription` vs `prescription_subdomain` default method, `__all__` integrity,
  `_resolve_output_shape` / cone sampler de-duplication, `mhs.py` docstring honesty).
- **New feature (optional, if time allows after the above):** a `sampler='jittered'|'sobol'` option for HFPI path
  generation using `scipy.stats.qmc.Sobol(d=4)` — O(N⁻¹) for smooth integrands and it makes `n_paths` an exact cap;
  keep `jittered` the default; test that both converge to the same field.

## Verification specifics
- The audit verified these as correct — keep them so and re-check after your changes: ASM transfer function, medium
  handling, evanescent treatment, Matsushima band limit, odd-N centering, H-cache keys, tilted ASM 2e-14, Fresnel
  prefactor/pitch, Bluestein/MFT vs DFT 1e-14, SAS physics, the Van Vleck density identity and the HF kernel's exact
  reduction to RS-I, HFPI binning, MHS composition 4e-16, RW E_x/E_y functions and apodisation.
- Re-run every repro script for your rows before/after; quote numbers.

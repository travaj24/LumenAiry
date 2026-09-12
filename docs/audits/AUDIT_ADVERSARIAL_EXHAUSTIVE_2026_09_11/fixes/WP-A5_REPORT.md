# WP-A5 — Propagator kernels (K1–K24): remediation report

Audit: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §3, §3.1,
§3.2, §15. Partition reports read in full: `PROP-CORE.md`,
`PROP-CORE-SUB1.md`, `PROP-HF.md`, `ORCHESTRATOR.md`.
Branch `audit-fixes-2026-09`, base `8bb4b03c`. No git write commands were
run; no file outside the WP's ownership list was modified.

All 24 findings were reproduced on the current HEAD before any change and
re-measured after. Both P0s are fixed and verified against oracles
independent of the code under test.

---

## 1. Summary

| ID | Sev | Status | Files:lines | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|---|
| **K16** | P0 ✔ | **fixed** | `vector_diffraction.py:347` (now `:388`) | `test_audit2609_a5_propagators.py::TestK16RichardsWolfEzSign` (4) | radial Bessel Debye–Wolf form `E_z/E_x = −2i I01/(I00+I02)` via `scipy.quad`; independent of the FFT path | code/oracle for `E_z` **−1.0006 / −0.9954 / −0.9994 / −0.9670 → +1.0006 / +0.9954 / +0.9994 / +0.9670** at `x_f` = 0.60/1.21/1.81/3.01 µm; `Im(E_z/E_x)` at `x_f>0` **+0.6966 → −0.6966**; `E_x` unchanged at +1.00 |
| **K10** | P3 | **fixed** (docs + test) | `vector_diffraction.py:37–60, 100–110` | `…::test_k10_pupil_is_indexed_by_the_aperture_coordinate` | ray-geometry prediction `x_f = +u·λ·f` | convention now stated; decentred-pupil focal centroid **+2.3600 µm** vs predicted **+2.3737 µm**; the ray-direction convention would give −2.37 µm |
| **K9** | P0 ✔ | **fixed** | `rs.py:36–71, 297–420` | `…::TestK9RayleighSommerfeldNearField` (17) | exact Hankel angular-spectrum quadrature (Gauss–Legendre, no FFT/grid), floor ~1e-12 | `P_out/P_in` **21.44 / 5.31 / 25.70 → 1.000000** and relL2 **4.50 / 2.08 / 4.95 → 5.3e-8 / 6.1e-8 / 5.3e-8**; far field bit-identical |
| **K1** | P1 ✔ | **fixed** | `fresnel.py:36–92, 262–265`; `mft.py:789–796` | `…::TestK1FresnelChirpGuard` (5) | 8×-oversampled direct Fresnel quadrature (audit repro `p11`) | warnings at `z = 0.25 / 0.50 z_crit` **0 → 1 each**; 0 at `z ≥ z_crit`; errors unchanged (3.34e-1 / 1.29e-4 / 3.99e-6) |
| **K2** | P1 ✔ | **fixed** | `backend/scipy.py:74–104, 130–160` | `…::TestK2BackendBesselArgumentOrder` (5) | `scipy.special.jv` | `jv(0,2.0)` **0.000000000 → 0.223890779**; `jv(2,1.5)` **0.491293779 → 0.232087672**; array case fixed |
| **K3** | P1 | **fixed** | `sas.py:218–224, 256–303` | `…::TestK3SasSinglePrecisionCancellation` (2) | the complex128 twin on identical geometry + the closed-form cancellation-free difference | complex64 phase error **2.9e-3 rad @ 3.24 mm / 0.902 rad @ 1 m → 1.50e-5 / 6.76e-6 rad** (flat in `z`, ratio 309 → 0.45) |
| **K4** | P2 | **fixed** | `fft_infra.py:1150–1170, 1322, 1343` | `…::TestK4PerSlotPlanLocks` | instrumented lock-concurrency counter (audit repro `p8_perf2`) | max simultaneous threads in the pyFFTW critical section **1 → 2** (the `n_bufs` ceiling) |
| **K5** | P2 | **fixed** (byte-identical) | `asm.py:71–80, 440–470` | `…::TestK5AsmKernelWorkspaceCap` (2) | tracemalloc + byte comparison | H-build transient **4.06 → 1.26 grids** at N=2048 (2.03 at N=1024), byte-identical, time flat. *Full cold-call peak unchanged (6.55 grids) — see §2.* |
| **K6** | P2 | **fixed** (docs + warning) | `mft.py:530–566`; `system.py:735–781` | measured inline (§2) | measured MTF table; power-in-window decomposition | `resample_field` MTF documented (0.9–6.8 % at "4 px/feature", not "<0.1 %"); the `system.py` crop now warns (measured 0.996685 retained → warned) |
| **K7** | P2 | **fixed** | `fft_infra.py:168–195, 2000–2030` | `…::TestK7PyfftwBlacklistKey` | direct blacklist inspection | one complex128 failure at (512,512) blacklisted **4 of 4 (shape, dtype, direction) combinations → 1 of 4** |
| **K8** | P3 | **fixed** (4 items) | `asm.py:240–290`; `fft_infra.py:1760–1785, 1815`; `result.py:183–212` | `…::TestK8BitExactnessAndCacheHygiene` (8) | byte comparison; numpy-2 `__array__` protocol | odd-N `max|dH|` **9.096e-13 → 0**; cached `H` **writeable → read-only** (public return still writeable); `np.array(result, copy=False)` **ValueError → works** |
| **K11** | P1 | **fixed** | `hf.py:100–170, 287–296`; `mhs.py:636–660` | `…::TestK11ResampleDoesNotFabricateEnergy` (2) | power of the source field inside the requested window (bracketed) | returned power over a ±16 µm window **100.00 % of the source (window holds 67.27 %) → the window's own fraction**, and the crop is reported |
| **K12** | P1 | **fixed** | `hfpi.py:313–354`; `vectorial_hfpi.py:296–340` | `…::TestK12WavelengthIsRequired` (2) | the `1/(iλ)` prefactor's own value | omitting `wavelength` **silently dropped 1/λ = 1.58e6 and −90° → TypeError**; `wavelength ≤ 0` → ValueError |
| **K13** | P1 | **fixed** (free space); **deferred** (prescription walk) | `hfpi.py:159–186, 560–605, 620–700`; `vectorial_hfpi.py:560–700` | `…::TestK13BinningJacobian` (2) | band-limited ASM (itself 6.1e-8 vs the Hankel oracle), read as an unbiased least-squares complex scale | HFPI/ASM scale **7.8e-12 → 0.976–1.012** across seeds and 0.5–8 M paths |
| **K14** | P2 | **fixed** | `hfpi.py:700–760`; `vectorial_hfpi.py:640–700`; `mhs.py:156–200, 545–560` | `…::TestK14K23GuardsAndCaps` (5), `…::TestK14MhsSurfaceCentre` | direct API probes | `cone_half_angle` **TypeError → accepted** on both free-space entries; MHS centre jump **accepted → ValueError** |
| **K15** | P3 | **fixed** | `hf.py:1–45, 215–230`; `rs.py` docstring; `vectorial_hfpi.py:1–60`; `mhs.py:1–40` | — (prose) | — | four false module/function claims replaced with what was measured |
| **K17** | P1 | **fixed** (honest minimum) | `vectorial_hfpi.py:41–140, 160–360, 560–760` | `…::TestK17VectorialHfpi` (5) | orthogonality of the rotation; reduction to the Richards–Wolf aplanatic matrix; the scalar propagator as the `vector_projection=False` reference | `|Ez|²` fraction **0 → 1.40e-2**; cross-pol `|Ey|²` **0 → 9.3e-5**; 45° `max|Ey/Ex−1|` **1.3e-16 → 0.143**; projection energy **0.841 → 1−1.1e-15** |
| **K18** | P1 | **fixed** | `hfpi.py:255–275, 905–915`; `vectorial_hfpi.py:215–230` | `…::TestK18SourceAreaNormalisation` (3) | closed-form source term `(dx²/(iλ))·π·sin²θ_max` | `Σw/exact` **1.000004 / 0.062140 / 0.003945 / 0.000260 → 0.999985 / 1.0106 / 0.9515 / 1.2089** at 1×1 / 4×4 / 16×16 / 64×64 |
| **K19** | P1 | **fixed** | `hfpi.py` (3 sites); `vectorial_hfpi.py` (2 sites) | `…::TestK19RngDefaultDrawsEntropy` (3) | byte comparison of two default runs | two `rng=None` runs **byte-identical → differ**; `rng=None` **≡ rng=0 → differs** |
| **K20** | P2 | **fixed** | `dispatch.py:1310–1330` | `…::TestK20UniformReturnType` (4) | the other three methods' return type | `propagate(method='hf', output_grid=…)` **tuple → ndarray**, matching asm/gbd/hfpi |
| **K21** | P2 | **fixed** | `hf.py:268–286` | `…::TestK21PitchGate` (3) | the returned field vs the un-resampled native field | a 0.5 % pitch change at 1 µm and a 10 % change at 100 nm **silent no-ops → resampled**; an exact no-op still bit-identical |
| **K22** | P2 | **partially fixed** (mechanism landed; the audit's projected gain is **not reproducible**) | `hf.py:56–62, 300–360, 470–560` | `…::TestK22HfQuadratureChunking` (4) | interleaved medians of 5 + byte comparison | bit-identical for every `chunk_output`; best measured gain **1.24× at N_in = 64**, and 3–5× SLOWER for a batch that leaves L2 — see §2 |
| **K23** | P2 | **fixed** | `hfpi.py:560–660, 760–800, 1370–1385`; `vectorial_hfpi.py:560–700` | `…::TestK14K23GuardsAndCaps` (5) | direct API probes | vector accumulator **silent → warns**; `init_paths_stratified(n_paths=100, (32,32)/(32,32))` **1 048 576 paths → 100** |
| **K24** | P3 | **fixed** (6 items) | `hfpi.py:110–160`; `mhs.py:156–200, 300–340, 545–560`; `hf.py:245–300`; `hf.py`/`hfpi.py` `__all__` | `…::TestK24SpawnRngIsAPureFunction` (2), `…::TestK14MhsSurfaceCentre` | purity check across draw orders | `_spawn_rng` stream 1 after stream 0 **≠ stream 1 alone → identical**; `finite_diff_step` doc corrected; `__all__` gaps closed |

**Totals.** 24 findings: 22 fixed, 1 partially fixed (K22 — the code
landed and is bit-identical, the projected speed-up is not reproducible
and the docstring now says so), 1 fixed-with-a-deferred-remainder (K13 on
the prescription walk). None not-reproducible: every finding reproduced
on HEAD.

---

## 2. Per finding

### K16 (P0) — Richards–Wolf `E_z` sign

**What was wrong.** `phi_p = arctan2(Yp, Xp)` (`vector_diffraction.py:249`)
is the APERTURE-point azimuth — pinned by PROP-CORE's `fft2` measurement,
and re-confirmed here by the decentred-pupil test under K10 — while the
textbook aplanatic strength vector is written in the RAY-DIRECTION
azimuth `phi_ray = phi_p + π`. `e_x` and `e_y` are even under
`phi → phi + π` and were unaffected; `e_z` is the only component that is
odd, so the `-` in `Pz = P*(-px*cp*s - py*sp*s)` produced the negative of
the correct value, on every call, since the function was written.

**What I changed.** `Pz = P*(px*cp*s + py*sp*s)`, with a why-comment
giving both the azimuth argument and the first-principles one (the ray
entering at aperture point (+a, 0) converges along (−sin t, 0, cos t) with
its polarisation rigidly rotated to (cos t, 0, +sin t), i.e. `E_z > 0`
there). Added the `pupil` coordinate-convention note (K10) and a
`Returns` note stating `E_z`'s symmetry and sign.

**How I verified.** Three independent readings, all agreeing:

1. `repro/orch/verify_rw_ez.py` (the orchestrator's own oracle, derived
   from the rigid rotation and reduced to the radial Bessel form, via
   `scipy.integrate.quad`): code/oracle for `E_z` went from
   **−1.0006 / −0.9954 / −0.9994 / −0.9670** to
   **+1.0006 / +0.9954 / +0.9994 / +0.9670** at
   `x_f` = 0.60 / 1.21 / 1.81 / 3.01 µm.
2. `repro/PROP-HF/p1_rw_ez_sign.py` (the sub-auditor's Gauss-Legendre ×
   trapezoid quadrature in the ray-direction parameterisation, no FFT and
   no aperture coordinate anywhere): per-component code/oracle for `E_z`
   now **+1.00179 / +1.00601 / +0.99782**, with `E_x` unchanged at
   +1.00788 / +1.01523 — the 0.2–1.5 % shared residual is the FFT
   discretisation and appears identically on all three components.
3. The regression test's own `scipy.quad` oracle, rebuilt from the
   derivation rather than copied.

Symmetries unchanged: `E_z` odd in `x` to 4.5e-16, `E_x` even to
2.4e-16, `E_z` on the `y` axis 3.1e-19 of its own maximum.

Consumers checked: `debye_wolf_psf` (intensity — bit-identical),
`ui/richards_wolf_dock.py` (intensity), `analysis/detector.py` and
`elements/polarization.py` (no `E_z` use). Nothing depended on the old
sign.

**Residual risk.** None identified. The FFT-vs-quadrature residual
(0.2–1.5 % on this fixture) is pre-existing discretisation, not a
consequence of the fix.

### K9 (P0) — the Rayleigh–Sommerfeld near-field kernel

**What was wrong.** The Green's function was POINT-SAMPLED in space on
the padded 2N grid. Its local spatial frequency at radius ρ is
`sin(θ)/λ`, so the sampled phase step is `k·sin(θ)·dx`, which exceeds the
π/pixel Nyquist limit at the padded rim whenever `z < 2·N·dx²/λ` —
and nothing checked it. With all-default arguments the convolution
CREATED energy. `bandlimit=True` was all-pass in exactly that regime
(same algebraic condition) and five decades harmful outside it.

**What I changed.** A new `kernel` kwarg, `{'auto','transfer','spatial'}`,
default `'auto'`:

* `'transfer'` evaluates the exact RS-I transfer function
  `exp(i k z √(1−(λf)²))` analytically in the frequency domain on the
  padded grid, with the evanescent set zeroed. This is the closed-form
  Fourier transform of the Goodman 3-43 impulse response — the same
  operator, without discretising a chirp the grid cannot carry. It is
  built by `_get_asm_H_natural` on the padded geometry, so it inherits
  the evanescent zeroing, the complex64 mod-2π mitigation, chunked
  construction, the Matsushima band limit and the H cache (the cache
  entry is legitimately shared with an ASM call at that geometry — it is
  the same array).
* `'spatial'` is the historical build and RAISES below the threshold.
* `'auto'` routes: `'transfer'` below `2·N·dx²/λ`, `'spatial'` at and
  above it.

**Why routed rather than always-transfer.** I first made `'transfer'` the
unconditional default and MEASURED a regression the audit did not
anticipate: multiplying by `H` is a CIRCULAR convolution on the padded
grid, so light that leaves the padded window wraps back in, while the
spatial build truncates `h` at the rim and discards it. On the
N = 128 / dx = 1 µm / w0 = 6 µm probe at z = 3 mm (63 % of the power still
inside the window) the always-transfer form measured **1.9e-2** relative
L2 against the Hankel oracle where the spatial build measures **4.4e-8**.
Routing preserves exactly the far-field behaviour the audit verified as
correct — at and above the threshold the default IS the historical
kernel, **bit-identical** (verified at five (N, dx, z) points).

**How I verified.**

* `repro/orch/verify_rs_alias.py`: `RS P/P0` **21.440 / 5.313 → 1.000 /
  1.000**, ASM 1.0000 on the same grids throughout.
* `repro/PROP-CORE-SUB1/t14_rs_fix2.py` (the sub-auditor's own verified
  fix script, unmodified): confirms relL2 6.08e-8 / 5.26e-8 with power
  1.0000 for the frequency-domain build on the failing grids.
* My own Hankel-oracle sweep across near and far field (scratch script):

  | N | dx | z | relL2 before | after | `P/P0` before | after |
  |---|---|---|---|---|---|---|
  | 64 | 2 µm | 50 µm | 4.50 | 5.26e-8 | 21.44 | 1.0000 |
  | 128 | 1 µm | 50 µm | 2.08 | 6.08e-8 | 5.31 | 1.0000 |
  | 128 | 2 µm | 50 µm | 4.95 | 5.26e-8 | 25.70 | 1.0000 |
  | 128 | 1 µm | 200 µm | 6.13e-8 | 6.13e-8 | 1.0000 | 1.0000 |
  | 128 | 1 µm | 1 mm | 6.81e-8 | 6.81e-8 | 0.9996 | 0.9996 |
  | 128 | 1 µm | 3 mm | 4.42e-8 | 4.42e-8 | 0.6323 | 0.6323 |

* Crossover continuity, measured at `z = z_crit`: the step between the
  two branches is 6.3e-5 / 1.0e-9 / 3.8e-14 at (N, dx) = (64, 0.50 µm) /
  (128, 0.40 µm) / (256, 0.25 µm), against each arm's own error of
  2.9e-4 / 6.59e-8 / 6.07e-8 — always far below, and converging.
* `bandlimit` honesty: at z = 3 mm, relL2 **4.4e-8 (False) vs 1.8e-2
  (True)**.

**Shen–Wang (2006) pixel-integrated kernel.** Considered as the WP asked
and **deferred** — see §6. It would restore `O(dx²)` convergence to the
`'spatial'` branch (PROP-HF measured the point-sampled kernel converging
first-order: 2.86e-2 → 3.79e-3 → 2.88e-3 → 1.08e-3 at N = 256…2048), but
the default path is now exact where that matters and the far-field
branch already measures at the grid's own floor, so the cost/benefit does
not justify it in this pass.

**Residual risk.** A caller whose beam overfills the padded window *and*
whose `z` is below the alias threshold gets the transfer kernel's
wrap-around. That combination requires `z < 2N dx²/λ` with the beam
already spread past `N·dx` — physically a very fast divergence over a
very short distance — and is strictly better than the pre-fix behaviour
(which created energy). Documented in the `kernel` docstring.

### K1 (P1) — Fresnel chirp guard

Derivation, the measured table and the MFT sibling's inclusion are in the
changelog. Verified by re-running `repro/PROP-CORE/p11_fresnel_alias.py`:
**0 warnings → 1 warning each at `z = 0.25 z_crit` and `z = 0.50 z_crit`,
0 at and above `z_crit`**; the relative errors themselves are unchanged
(3.34e-1 / 1.29e-4 / 3.99e-6 / 1.24e-6 / 4.61e-7), which is the point —
this is a diagnostic, not a numerical change.

I extended the guard to `fresnel_propagate_mft`, which the audit measured
as evaluating the identical aliased sum (agreeing with the single-FFT
form to 3e-14 at every `z`, which is why it is not a valid oracle for
this question). The audit's row named only `fresnel_propagate`.

### K2 (P1) — `backend.scipy.jv`

`_dispatch_special` hard-coded "the array is the first argument", which
is wrong for a two-argument special function. It now takes an `arg_pos`
keyword; `jv` passes `arg_pos=1`. Verified against `scipy.special.jv`
directly (table in the changelog) including the array case whose first
element was right by coincidence. I grepped every importer: the only
in-library consumers (`elements/bor/farfield.py`,
`bor/fiber_oracle.py`, `bor/stepindex_oracle.py`) import `jv` from
`scipy.special` directly, so nothing in the library changed behaviour —
confirming the audit's "latent" classification.

### K3 (P1) — SAS float32 cancellation

Verified at two levels. **Kernel level** (the quantity the audit
measured): `max|Δ(h_AS − h_Fr)|` float32 vs float64 = **9.091e-08** on the
N_new = 1024 / dx = 1 µm / λ = 633 nm grid, i.e. **0.902 rad** at z = 1 m
— reproduced exactly. The closed form's own residual against the float64
subtraction is 1.7e-16, i.e. 1.7e-9 rad at z = 1 m. **End to end**, the
complex64 field against its complex128 twin, max phase error over the
bright region: **1.50e-5 / 1.05e-5 / 5.50e-6 / 6.76e-6 rad at
z = 3.24 mm / 1 cm / 10 cm / 1 m** — flat in `z`, which is the library's
own complex64 contract, versus a pre-fix ratio of 309 between the first
and last.

The regression test pins the RATIO across a 300× range of `z` rather than
an absolute noise floor, so it cannot become a per-build bar.

### K4, K5, K7, K8 — FFT infrastructure and ASM

All four are covered in the changelog with their measurements. Two notes
where my findings differ from the audit's:

* **K5.** The audit's proposed one-liner —
  `chunk = min(Ny, max_chunk, _ASM_STREAM_BAND_ELEMS // Nx)` — is a NO-OP
  at the grid the audit measured: `_ASM_STREAM_BAND_ELEMS = 2²²` gives
  2048 rows at N = 2048, i.e. the whole grid. I measured a band ladder
  (byte-identical at every width, time flat to slightly better) and
  introduced a separate `_ASM_H_BUILD_BAND_ELEMS = 2¹⁸` for the plain
  builder, which reproduces the audit's headline 4.06 → 1.26 grids at
  N = 2048. The two builders legitimately want different bands: the
  streamed path's band is live alongside the whole spectrum.
* **K5, not reproducible.** The audit's implied improvement to the FULL
  cold ASM call is not observed: measured in a cold process at N = 2048
  the peak is **6.55 grids either way** (matching the audit's own 6.54).
  With warm pyFFTW plans the H build is not the peak; in a cold process
  the aligned-buffer allocation dominates. The H-build transient is
  genuinely 3.2× smaller, which is what matters at very large N
  (~8.6 GB → ~134 MB at N = 32768) and on the batch /
  `return_transfer_function` paths.

### K6 (P2) — `resample_field` MTF and the `system.py` crop

Both halves addressed: the docstring now carries the measured MTF table
(0.9–6.8 % at the "4 px per feature" case it rated at "<0.1 %"), the
reason it bites hardest on the single-FFT Fresnel output, the pointer to
the band-limited alternative, and the missing anti-alias note; and
`propagate_through_system` warns when the resample-back crops. Verified:
on the audit's own fixture the `fresnel` leg warns (retained window holds
0.996980 of the power) and `asm` does not. The `sas` leg does NOT warn on
that fixture and correctly so — its `dx_new < dx` there, so the resample
pads rather than crops; its 0.9507 power ratio is SAS's own band-limit
mask, not a crop, which the audit did not decompose.

The band-limited chirp-Z resampler the audit recommends as the deeper fix
is **deferred** (§6).

### K11, K21, K20 — `hf` free-space wrapper and the dispatcher

Covered in the changelog. On K11 I de-duplicated the block into one
shared helper used by both `hf.py` and `mhs.py`, as the WP asked, and
fixed the three stale cross-references the sub-auditor listed.

On K20 the three tests in `tests/unit/test_v5_3_hf_freespace_output_grid.py`
PINNED the inconsistency (`assert isinstance(out, tuple)`); I updated
them to the uniform contract with a comment saying what they used to pin
and why it changed.

### K22 (P2) — the HF OPL quadrature

**The mechanism landed and is bit-identical**; the audit's projected gain
is **not reproducible**, and I have documented the measurement rather
than the projection.

The audit's reasoning was that `opl_fn` is already vectorised over the
input grid, so broadcasting the output coordinates amortises the 17
evaluations "for free … same flops, ~n_chunk× fewer Python-level
dispatches and temporaries". The flops claim is right; the conclusion is
not, because the computation is memory-bandwidth-bound and Python
dispatch is a small share of it. Measured ladder (medians of 5
interleaved runs, ms per output pixel, Van Vleck on, complex128):

| N_in | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| 64 | 0.401 | 0.360 | **0.323** | 0.328 | 0.449 | 1.634 |
| 128 | **1.450** | 1.465 | 2.099 | 7.077 | 6.422 | 6.419 |
| 256 | **9.563** | 30.45 | 28.02 | 29.50 | 30.34 | 28.82 |

Best available gain **1.24× at N_in = 64**; a batch whose working array
leaves the L2 cache is 3–5× SLOWER (tracemalloc confirms the mechanism:
110 MB peak = 210 grids at n_chunk = 16 / N_in = 256, against 9.4 MB =
18 grids per-pixel). I therefore size the auto batch by CACHE (~128 KB
per working array), which selects the measured optimum at all three
sizes, and the docstring carries the ladder plus the honest advice: for a
full-plane output use a Fourier route, since this quadrature earns its
keep only for a non-shift-invariant `Φ`.

I also note that this box measures **9.56 ms/px at N_in = 256** where the
audit measured 109 ms/px — the audit's timings were taken under ~20
concurrent auditors, so its "119 min for 256²" is ~10 min here.

The two sub-items landed unconditionally: the JAX branch list-accumulates
instead of allocating a full output array per output pixel, and the
complex64 path builds its kernel in single precision after a float64
mod-one-cycle fold of `Φ` (which is in waves and of order `z/λ`, so a
naive float32 cast would be meaningless).

### K12, K13, K18, K19, K14, K23 — HFPI

All covered in the changelog with measurements. Three things worth
calling out:

**K13's reach.** The per-path Jacobian needs the GEOMETRIC distance since
the last emission. `paths.opl` cannot serve — it is `n_medium·|t|` on the
free-space legs and the ABSOLUTE accumulated `opd` from the ray tracer on
the prescription walk — so `PathBundle` / `VectorPathBundle` gained an
optional trailing `leg` field, maintained by `propagate_to_plane`,
`_hfpi_segment_trace` and reset by `apply_aperture_diffraction`. The
field is optional with a `None` default, so positional construction of
the pre-v5.46 bundle still works.

**K13 on the prescription walk — deferred remainder.** That walk bins the
bundle at the LAST SURFACE; there is no final hop to a separate output
plane. When that surface is a diffractor the paths have just been
re-emitted, their current leg has length zero, and `1/r` is undefined. I
made the walk default to `normalisation='legacy'` and WARN (it carried no
amplitude warning at all before — one of the audit's complaints), rather
than silently returning zeros or a wrong number. Concrete design in §6.

**K13's verification method.** The naive read-out `mean(|E_hfpi|/|E_asm|)`
is biased HIGH because `|Σw|² = |signal|² + noise power`; at 6 M paths it
reads 1.38 where the truth is 1.00. I therefore use the unbiased
least-squares complex scale `Σ(E_hfpi·conj(E_asm))/Σ|E_asm|²`, which
reads 0.976–1.012 across seeds and 0.5–8 M paths at z = 2 mm, and
confirms the legacy path still carries exactly the documented bias law.
(This is the kind of readout error the audit's §15.2 is about, so it is
worth recording.)

**K23's stratified cap changes the DEFAULT path too**, by design: with
`n_paths = 20000` the 4th-root rule gives `n_total = 12⁴ = 20736`, and
`n_paths` is now an exact cap, so 20000 paths are allocated rather than
20736. The counter-pin test bounds the default at `[0.9·n_paths,
n_paths]`.

### K17 (P1) — vectorial HFPI

I implemented the honest minimum the WP describes, choosing the RIGID
ROTATION (Rodrigues, from the incoming to the outgoing direction) over
the orthogonal projection the audit's text suggests. Reason: the rigid
rotation is orthogonal, so it conserves `|E|²` exactly and introduces no
amplitude factor — which is precisely what lets the scalar Kirchhoff
obliquity be applied exactly ONCE without the double count the audit
flagged. It also reduces term-by-term to the Richards–Wolf aplanatic
matrix for `s_from = +z`, so the module is consistent with the library's
own vector-focusing conventions (and with the K16 sign, since here the
azimuth is genuinely the ray-direction one).

Verified: energy preserved to 1.1e-15 (against 0.841 for the v5.4.6
2-component projection on the same distribution), transversality
`max|E'·ŝ|` = 4.6e-16, reduction to the RW matrix to 1e-13, and the
physics the module existed for now present (table in the changelog).
`vector_projection=False` still reproduces two scalar HFPI runs to 1e-15
absolute on an `|Ex|max` of 3.3.

I did NOT implement full Stratton–Chu dyadic re-emission (no material
response at the aperture edge, still an isotropic scalar re-emission
cone), and the module docstring now says so explicitly and routes
rigorous high-NA vector focusing to `richards_wolf_focus`.

### K15, K24 — documentation and integrity

Six items, all in the changelog. The `finite_diff_step` correction is
worth restating: the documented "−2.53e-8 at the 1e-6 default" was
measured on an exact-quadratic OPL where the 4th-order truncation term
vanishes identically; on a spherical OPL the default gives +2.4e-7 and
the optimum moves a decade. Both remain ~4 decades below the
quadrature's own floor, so it is documentation only — but it is exactly
the "right conclusion, wrong numbers" shape `TESTING_STANDARDS.md` warns
about.

---

## 3. Files touched

**Library (15):**

| file | findings |
|---|---|
| `lumenairy/propagators/vector_diffraction.py` | K16, K10 |
| `lumenairy/propagators/rs.py` | K9, K8, K15 |
| `lumenairy/propagators/fresnel.py` | K1 |
| `lumenairy/propagators/mft.py` | K1 (MFT sibling), K6 |
| `lumenairy/propagators/sas.py` | K3 |
| `lumenairy/propagators/asm.py` | K5, K8 |
| `lumenairy/propagators/fft_infra.py` | K4, K7, K8 |
| `lumenairy/propagators/result.py` | K8 |
| `lumenairy/propagators/hf.py` | K11, K15, K20, K21, K22, K24 |
| `lumenairy/propagators/hfpi.py` | K12, K13, K14, K18, K19, K23, K24 |
| `lumenairy/propagators/vectorial_hfpi.py` | K12, K13, K17, K18, K19, K23 |
| `lumenairy/propagators/mhs.py` | K11, K14, K15, K24 |
| `lumenairy/propagators/dispatch.py` | K20 |
| `lumenairy/propagators/system.py` | K6 |
| `lumenairy/backend/scipy.py` | K2 |

**Tests updated (6 unit + 2 validation)** — each pinned behaviour a
finding says was wrong, or a synthetic fixture the new defaults refuse;
every change carries a comment saying what it used to pin:

| file | why |
|---|---|
| `tests/unit/test_v5_3_hf_freespace_output_grid.py` | pinned the K20 return-type flip (`assert isinstance(out, tuple)`), 3 tests |
| `tests/unit/test_audit_w5_propagators.py` | pinned the wording of the K18-wrong normalisation warning, 1 test |
| `tests/unit/test_audit_w6_propagators.py` | pinned `chunk_output`'s deprecation and the per-pixel call count (K22), 2 tests |
| `tests/unit/test_audit_propagation.py` | synthetic bundles with `opl = 0` pinning index sharing, not photometry — now name `normalisation='legacy'`, 4 sites |
| `tests/unit/test_v4_14_0_dispatcher_pin_hfpi.py` | same, for the alive-mask pins, 4 sites |
| `tests/unit/test_niche_audit_w9_hfpi_doe.py` | same, for the sampling-guard pins, 3 sites |
| `validation/propagators/test_hfpi.py` | omitted `wavelength` at the aperture (K12 migration) |
| `validation/propagators/test_vectorial_hfpi.py` | same |

**New file (1):** `tests/unit/test_audit2609_a5_propagators.py` — 80
tests across 16 classes.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`.

| command | result | duration |
|---|---|---|
| `python -m pytest tests/unit/test_audit2609_a5_propagators.py -q --no-header -p no:cacheprovider` | **80 passed** | 32.8 s |
| `python -m pytest tests/unit/test_audit2609_a5_propagators.py test_propagation.py test_audit_propagation.py test_v5_3_hf_freespace_output_grid.py test_v4_14_0_dispatcher_pin_hfpi.py test_niche_s9_vector_diffraction_registration.py test_audit_w5_propagators.py test_audit_w6_propagators.py test_niche_audit_w3_propagators.py test_niche_audit_w9_dispatch.py test_niche_audit_w9_dispatch2.py test_niche_audit_w9_hfpi_doe.py test_audit_g04_guards_prop.py test_audit_s2_10_asm_H_consolidation.py test_audit_v5_24_2_g09_seams_prop2.py test_niche_audit_p2_fresnel_tf_buffer.py test_v5_17_1_tilted_asm_dtype.py test_v5_21_gbd_asm_interop.py test_audit_w2_fft_state.py test_perf_v4_12_0_fft_infra.py test_v5_4_6_wave5_delegated.py test_v5_2_physics_fixes.py -q` (22 files, the full propagator surface) | **602 passed, 1 skipped** | 214.8 s |
| `python -m pytest tests/unit/test_audit_misc.py test_v4_14_1_dispatcher_pin_cache_clears.py test_v4_14_2_dispatcher_pin_cache_locks.py test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py test_audit_w4_rcwa_homog_fftlock.py test_audit_jax_c64_propagator_precision.py test_v4_16_0_walker_dy_threading.py test_v4_16_0_walker_xp_of_dispatch.py test_v4_16_0_walker_all_symmetry.py test_v4_16_2_dispatcher_pin_doc_consistency.py (+4 more)` | 411 passed, 12 skipped, **3 failed (all other agents' — see below)** | 121.5 s |
| `python -m pytest tests/unit/test_v5_2_physics_fixes.py test_v5_21_2_subsystem_audits.py test_niche_audit_w4_p5_return_contract.py test_v5_1_0_agent_e_split.py test_v4_15_1_agent_f.py test_audit_except_budget.py test_v4_15_dispatcher_pin_validate_grid_params.py test_v4_16_0_walker_sentinel_propagation.py test_v4_15_3_dispatcher_pin_2d_scalar_field.py test_v5_4_6_wave5_delegated.py test_niche_audit_p2b_infra_contracts.py` | 290 passed, 1 skipped, **2 failed (pre-existing — see below)** | 22.8 s |
| `python validation/run_all.py test_propagation test_hf test_hfpi test_vectorial_hfpi test_mhs test_dispatch test_advanced_diffraction test_new_propagators_smoke test_gbd test_subaperture --quiet` | **10/10 files pass** (re-run at the end: 8/8 of the propagator set, 76.2 s) | ~45 s |

Skips are genuine capability gaps, not resource preconditions: `numexpr`
and `cupy` are not installed, and the JAX x64 arm is off process-wide.

### Pre-existing / other-agent failures found

1. **`test_audit_except_budget.py`** (2 tests) — **pre-existing on this
   branch**. The non-`ui/` `except Exception:` budget is 48; base commit
   `8bb4b03c` already measures **51** (I counted it from
   `git show HEAD:<file>` across every non-UI module). Another agent's
   `elements/_lens_real.py` adds 2 more; my work REMOVES one (the
   `_spawn_rng` JAX swallow, narrowed as part of K24) and its one new
   catch — the `opl_fn` broadcast probe — is narrowed to an enumerated
   tuple so it does not count. Current total 52. Not mine to fix:
   `_lens_real.py`, `raytrace/exit_vertex.py`, `elements/_berreman_jax.py`
   etc. are other WPs' files.
2. **`test_v4_16_0_walker_all_symmetry.py`** and
   **`test_v4_14_1_dispatcher_pin_cache_clears.py`** — other agents'
   `__all__` additions without the top-level re-export. The violating
   names are all in `lumenairy.analysis`, `lumenairy.analysis.core`,
   `lumenairy.analysis.opd`, `lumenairy.elements.pmm.twod`,
   `lumenairy.optimize`, `lumenairy.raytrace*` — **none in
   `propagators` or `backend`**. My two `__all__` additions
   (`propagate_huygens_fresnel`, `propagate_hfpi`) are already in
   `lumenairy.__all__`, which I verified explicitly.
3. **`test_audit_misc.py::…StopIndexWarn::test_traced_emits_warning_for_stop_index_2`**
   — `apply_real_lens` / `_lens_real.py` (finding L14), another WP.
4. **`test_audit_w6_propagators.py::TestP352X64Policy::test_twins_raise_without_x64_and_do_not_mutate_global`**
   — intermittent, **environmental**: it spawns a subprocess and this
   sandboxed session cannot duplicate stdio handles
   (`OSError: [WinError 50] The request is not supported`, raised inside
   `subprocess.Popen._make_inheritable` before any library code runs).
   Reproduced identically under both the Bash and PowerShell tools when
   run alone, and PASSES when the file is run as part of a larger
   session. Unrelated to this WP.

### Fail-before verification

Every substantive pin was demonstrated to REJECT the pre-fix state, by
re-introducing each defect in process (scratch harness
`scratchpad/wp_a5/fail_before.py`) and running the new assertions against
it. **20/20 pins fail on the pre-fix state**, and the failure messages
reproduce the audit's own numbers:

| pin | reported on the pre-fix state |
|---|---|
| K16 `E_z/E_x` vs oracle | ratio −1.0006 |
| K16 `Im(E_z/E_x) < 0` | +0.696627 |
| K10 decentred pupil | centroid −2.3698 µm (wrong side) |
| K9 energy (3 grids) | `P_out/P_in` 21.44 / 5.31 / 25.70 |
| K9 field (3 grids) | relL2 4.50 / 2.08 / 4.95 |
| K2 `jv(0, 2.0)` | +0.000000000 |
| K8 cached `H` read-only | writeable |
| K19 `rng=None` | two default runs byte-identical |
| K18 source area (2 grids) | 0.0632 / 0.0037 |
| K13 estimator | HFPI/ASM scale 0.0000 (7.8e-12) |
| K17 `E_z` / depolarisation | 0 and 1.298e-16 |
| K1 Fresnel guard | no warning at `z = 0.25 z_crit` |

---

## 5. Requested changes outside my ownership

1. **`lumenairy/elements/_lens_real.py:1886–1888` — SAS is the only glass
   propagator that never receives `dy` (PROP-CORE P2, the K7 row's second
   half).** `_propagate_through_glass` forwards `dy` to the `fresnel`,
   `rs` and `asm` branches but calls
   `scalable_angular_spectrum_propagate(E, thickness, lam_medium, dx)` —
   a function with no `dy` parameter at all. SAS rejects non-square
   ARRAYS but nothing rejects a square array on an anamorphic PITCH, so
   `dy != dx` is silently propagated as if `dy == dx`. `system.py` guards
   the same branch with `_require_square_pitch` (`:768`);
   `_lens_real.py` does not. **Requested change:** add the same
   `_require_square_pitch(current_dx, current_dy, 'sas')` guard to the
   `sas` branch of `_propagate_through_glass`. (Owner: the
   `_lens_real.py` WP. I cannot add `dy` to SAS itself usefully without
   that call site.)
2. **`lumenairy/__init__.py` — promote `_rs_alias_free_distance` to a
   public `rs_alias_free_distance`.** It computes the RS spatial-kernel
   sampling threshold `2·N·dx²/λ` and is genuinely useful to a caller
   sizing a grid. I kept it private because a public name in
   `rs.__all__` requires a matching top-level re-export, and
   `lumenairy/__init__.py` is outside my ownership. **Requested change:**
   `from .propagators.rs import rs_alias_free_distance` plus the
   `__all__` entry; I will rename on request.
3. **`CONVENTIONS.md` §7 — add the Richards–Wolf pupil-coordinate row.**
   The K10 note now lives in `richards_wolf_focus`'s docstring; the
   audit's fix text also asks for it in the conventions table. Suggested
   row: *"Richards–Wolf `pupil` indexing | indexed by the physical
   exit-pupil (aperture) coordinate, NOT the projected ray direction; the
   two differ by a point inversion | `vector_diffraction.py::richards_wolf_focus`"*.
   (Owner: whoever holds `CONVENTIONS.md` this round.)

---

## 6. Deferred items

1. **K13 on `propagate_hfpi_through_prescription`.** *Design:* give the
   walk an explicit output plane. Add `z_output: float | None = None`
   (default: the last surface's `z`, i.e. today's behaviour) and, when it
   differs from the last surface, a closing
   `propagate_to_plane(paths, z_target=z_output, wavelength=…)`. That
   gives every path a non-zero final leg, at which point
   `normalisation='physical'` becomes well-defined and can become the
   default here too. *Effort:* ~1 h plus an oracle test against ASM
   through a thin-lens prescription. *Interim state:* the walk defaults
   to `normalisation='legacy'` and now WARNS that its amplitudes are not
   photometric (it carried no such warning at all before), so nothing is
   silently wrong.
2. **Shen–Wang (2006) pixel-integrated RS kernel (K9's second half).**
   *Design:* replace the point evaluation of `h` in the `'spatial'`
   branch with the analytic integral of the RS-I Green's function over
   each pixel (Shen & Wang, Appl. Opt. 45, 1102, already cited in
   `rs.py`'s references). *Why deferred:* the default path is now exact
   where the point-sampling error mattered, and on the far-field branch
   the measured error is already at the grid's own floor (4.4e-8). The
   gain is the `O(dx) → O(dx²)` convergence rate the audit measured for
   hard-aperture inputs (2.86e-2 → 1.08e-3 over N = 256…2048). *Effort:*
   ~3–4 h including a convergence-rate test; the same kernel would fix
   `hf.py`'s quadrature floor.
3. **Band-limited (chirp-Z) resampler for `resample_field` (K6's second
   half).** *Design:* route pitch changes of a sampled band-limited field
   through `_bluestein_centred_2d`, which the package already contains
   and which performs exactly that operation exactly;
   `angular_spectrum_propagate_mft(z=0, …)` is a drop-in alias-free
   resampler for the square case today. *Why deferred:* it changes the
   numbers of every `fresnel`/`sas` leg in `system.py` and
   `_lens_real.py` (a file I do not own), so it wants to be one
   coordinated change. *Interim state:* the true MTF is documented and
   the crop warns. *Effort:* ~2 h in `mft.py`, plus a coordinated pass
   over the two call sites.
4. **A `sampler='jittered'|'sobol'` option for HFPI (the WP's optional
   feature).** Not implemented — the P0/P1 work filled the pass. *Design:*
   `scipy.stats.qmc.Sobol(d=4).random(n)` over the same
   `(pixel_x, pixel_y, cosθ, φ)` cube that `init_paths_stratified`
   already strata-samples, keeping `jittered` the default; `n_paths`
   becomes an exact cap for free (K23 already delivers that
   independently). *Effort:* ~1.5 h including the "both converge to the
   same field" test. Note the QMC advantage (`O(N⁻¹)` vs `O(N⁻¹ᐟ²)`)
   applies to a smooth integrand; the HFPI integrand has a hard aperture
   edge, so the measured gain should be checked before advertising it.
5. **CuPy paths.** CuPy is not installed here. The K23 `to_numpy`
   routing, the `accumulate_*` fallbacks and the `_get_asm_H_natural`
   CuPy branch used by `kernel='transfer'` were **desk-checked only**.
   The K7 blacklist key, the K4 per-slot locks and the K8 read-only cache
   are NumPy-path only by construction (CuPy/JAX arrays are kept out of
   the host caches).
6. **JAX parity.** `jax` is importable here but the session has x64 OFF,
   and the suite's JAX-precision arm skips for that reason. The JAX
   branches I touched — `_get_asm_H_natural`'s host-build path (reached
   by RS `kernel='transfer'`), the HF quadrature's list accumulation, and
   `accumulate_vector_to_grid`'s scatter — are exercised structurally by
   the existing tests but not measured at x64. Worth one x64-enabled run
   in CI.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A5_CHANGELOG.md`


---
---

# 8. Follow-up (v5.46.1) — VERIFY-A5's open items and two cross-WP requests

Input: `fixes/VERIFY_WP-A5.md` (read in full, §3.2 and §3.3 in
particular), plus two coordinator requests from `fixes/WP-A6_REPORT.md`
§5.3 and `fixes/WP-A2_REPORT.md` §5 item 4. Base for this round:
`4996daeb` (the verifier's commit). Nothing outside WP-A5's ownership
list was modified.

## 8.1 Summary

| item | source | status | evidence |
|---|---|---|---|
| **V1** | VERIFY §3.3, P1 open | **fixed** | residual `n_paths x z1` removed; transparency property 1.0 within MC scatter; independent RS-I double quadrature 0.908–1.046 |
| **V6** | VERIFY §3.1, P2 | **fixed** | guard fires at ring 7.2–36.1 %, silent at 5.5e-22–8.1e-14, O(1) cost |
| **V8** | VERIFY §2, P3 | **fixed** (changelog) | complex128 movement measured 1.76e-12 … 5.43e-10 |
| **V5** | VERIFY §2, P3 | **stated** | K1/K2 label collision; both items fixed |
| **V2, V3, V4** | fixed by the verifier in my files | **reviewed, kept** | re-ran their pins; 20/20 pass, and the `fft_infra` latch survives my ASM change |
| **A6 §5.3** | coordinator | **not reproducible** + **pinned** | the MFT family already preserves complex64 in dtype, working precision and memory; the promotion is `carrier.py:9460` |
| **A2 §5 item 4** | coordinator | **landed** | byte-identical at every configuration; 1.50× / 1.30× at N = 1024 / 2048 |

## 8.2 V1 — the cascaded HFPI measure (P1)

**Reproduced first.** The verifier's script, unmodified, on the
as-committed code returned `two/one x n_paths x z1` = 1.098 / 1.046 /
0.860 / 0.920 / 1.339 / 0.801 — its own six numbers to four digits — and
I extended it to the vector twin, which is affected identically (0.861 /
0.920 on the same rows).

**Diagnosis, confirmed by derivation.** I re-derived the estimator's
measure from scratch rather than taking the verifier's reading. Writing
an intermediate surface integral in the direction variables the
estimator samples,

    dOmega_m = dS_{m+1} cos(theta_m) / r_m^2
    =>  dS_{m+1} (cos(theta_m)/r_m) e^{ik r_m} = dOmega_m * r_m * e^{ik r_m}

so an INTERMEDIATE leg's Kirchhoff kernel is
`(1/(i lambda)) e^{ikr} * r * dOmega` — a factor `r`, not `1/r`, and no
obliquity. Requiring the estimator to carry, after the *m*-th leg,

    W_m = E(Q_1) (1/(i lambda))^m (A_src Omega_1/n) Omega_2..Omega_m
          r_1..r_{m-1} e^{ik sum r} * cos(theta_m)

— the trailing `cos(theta_m)` being exactly what `_binning_jacobian`'s
`r/(dx_out^2 cos(theta_out))` cancels when that leg turns out to be the
last — and propagating it from leg *m* to *m+1* gives

    F = (1/(i lambda)) * Omega_out * r_in * cos(theta_out) / cos(theta_in)

which is what the shared `_reemission_measure` now applies. That
reproduces both halves of the verifier's diagnosis (the spurious `/n` and
the missing `r`) and additionally shows the symmetric Kirchhoff obliquity
`0.5(cos_in + cos_out)` is not what an exact RS-I composition wants: the
outgoing cosine alone is, with a `1/cos_in` that is pure bookkeeping
(undoing the source's `cos(theta)`, applied when it was not yet known
that its leg would not be the last).

The formula composes, so it is correct for any number of apertures — the
prescription walk's aperture site threads the same value.

**Verified three ways.**

1. *Closed form, exact.* `_reemission_measure` is a pure function;
   against the formula written out independently it agrees to < 1e-16
   (bar 1e-13), and the `'legacy'` branch reproduces the pre-fix
   expression to the same precision. Their ratio equals
   `n * r_in * cos_out/cos_in / (0.5(cos_in+cos_out))` exactly — the
   residual law, as an identity rather than a measurement.
2. *Oracle-free property.* An unobstructed aperture plane is
   transparent. At 8 M paths the two-leg/one-leg least-squares complex
   scale reads 1.009 / 1.010 / 1.057 / 0.963 / 1.245 / 0.977 across
   z1 = 0.25 / 0.5 / 1.0 mm and two seeds; at 2 M, 1.096 / 0.847 /
   1.044 / 0.940 / 0.586 / 1.052. Pre-fix the same quantity was
   `1/(n_paths z1)` — 9.84e-4 / 1.01e-3 / 3.19e-4 in the fail-before run,
   against `1/(n z1)` = 1.0e-3 / 1.0e-3 / 2.5e-4.
3. *Independent oracle.* A direct RS-I **double** quadrature through a
   real 25 µm clipping aperture — super-sampled midpoint over the source
   plane (3× per pixel) and a 120×320 polar midpoint over the aperture
   disc, no FFT and no library propagator anywhere on the oracle side —
   compared on a 9×9 block of output points:

   | propagator | n_paths | seed | LS scale | arg (rad) |
   |---|---|---|---|---|
   | scalar | 8 M | 3 | 0.9715 | −0.173 |
   | scalar | 8 M | 9 | 0.8800 | +0.239 |
   | scalar | 32 M | 3 | 1.0457 | −0.037 |
   | scalar | 32 M | 9 | 0.9080 | +0.019 |
   | vector Ex | 8 M | 3 | 0.9715 | −0.173 |
   | vector Ex | 8 M | 9 | 0.8799 | +0.239 |

   i.e. absolute amplitude correct to the Monte-Carlo scatter, which
   halves from 8 M to 32 M as `1/sqrt(N)`. Pre-fix the same scale would
   be `1/(n z1)` = 2.5e-7 at 8 M. The `'legacy'` path against the same
   oracle reads 2.52e-11 — non-photometric by contract, as documented.

**Residual risk.** The per-pixel relative L2 after scaling is 0.70–1.82
on that small-aperture fixture at 8–32 M paths and halves with path
count: the AMPLITUDE is right, the per-pixel shot noise is the
estimator's own, and it is exactly what the v5.31 under-sampling guard
exists to flag. Nothing here changes that HFPI is a Monte-Carlo method.

**Not defaulted to `'legacy'`.** The coordinator offered that as the
fallback if `'physical'` could not be made correct on the chain. It can,
and is, so the default stands.

## 8.3 V6 — the RS transfer-branch wrap-around guard (P2)

Implemented as the verifier suggested: the power that has reached the
outer 1/8 of the padded window after the multiply, against the input
power, warned above 2 %.

Two corrections to the obvious implementation, both found by
measurement:

* **The detector was mis-centred at first.** `fftshift`-ing the padded
  convolution moves the (already centred) field to the corners, so the
  "ring" read 1.0 for a perfectly contained Gaussian. The padded array
  holds the input at `[N/2 : 3N/2]` and a natural-order `H` convolves in
  place, so the outer band of the raw array IS the outer ring — no shift.
* **A full reduction over the ring is too expensive for a diagnostic**:
  measured **+91 %** of the call at N = 256 and **+15 %** at N = 1024.
  The ring fraction is a smooth spatial statistic, so the shipped
  detector is a fixed-budget strided estimate (4096 points per band),
  O(1) in grid size: **0.035–0.084 ms**, flat from N = 128 to 1024,
  against a 5.5–187 ms call, and matching the full reduction to **0.5 %**
  (7.2 / 30.3 / 30.2 / 36.1 / 28.0 % sampled against 7.22 / 30.35 /
  30.12 / 36.09 / 27.86 % exact).

Calibration and the 12-decade counter-fixture margin are in the
changelog. The guard fires only on the `'transfer'` branch — the
`'spatial'` branch truncates rather than wraps — and the returned field
is pinned bit-identical with the guard disabled.

**Not done: exposing the pad factor.** The verifier offered that as an
alternative. It is a larger change (the 2N pad is assumed throughout the
function and in the `'RS'` cache key) and the guard is what was asked
for; recorded as deferred item 7 below.

## 8.4 WP-A6 §5.3 — the MFT complex64 request: NOT REPRODUCIBLE

The request was to make `angular_spectrum_propagate_mft` dtype-preserving
for complex64. **It already is**, and so are its two siblings. Measured,
not read:

| entry point | c8 in | c16 in | relL2(c8, narrowed c16) |
|---|---|---|---|
| `angular_spectrum_propagate_mft` | complex64 | complex128 | 2.734e-07 |
| `fresnel_propagate_mft` | complex64 | complex128 | 2.027e-07 |
| `fraunhofer_propagate_mft` | complex64 | complex128 | 2.172e-07 |

and the preservation is not a final cast over double-precision work: an
instrumented `_bluestein_2d` is entered and left at the caller's dtype
with `target_cdtype` threaded through, and the tracemalloc peak at
N = 512 is **86.75 MB at complex64 against 168.49 MB at complex128** —
the same 41.4 / 40.2 input-grid units, i.e. half the bytes for the same
structure. `_bluestein_centred_2d` and `_bluestein_axis_1d` already build
every chirp in float64 and narrow via `astype(target_cdtype)` before use,
and `const_c` is already a `target_cdtype` scalar; the pattern the
coordinator asked for is the pattern that is there.

**Where the chain's complex128 actually comes from.**
`carrier.py:9460–9461`:

```python
field = field * np.exp(
    1j * k0 * (tilt_L * _u[None, :] + tilt_M * _v[:, None]))
```

`_u` / `_v` are float64, so this is a genuine complex128 ARRAY and
`complex64 × complex128 array` promotes under NEP 50 — **downstream of**
the MFT call, in `carrier.py`, which WP-A6 owns. Demonstrated in
isolation: `complex64 * ramp` → complex128, `complex64 *
ramp.astype(complex64)` → complex64, the two differing by 0.0 on a
unit-modulus ramp. WP-A6's own comment at `:9463–9465` says exactly
this; only its inference about `angular_spectrum_propagate_mft` was
wrong.

**What I did instead.** Pinned the contract (12 tests), so the one-line
`carrier.py` fix can depend on it. Requested that fix in §8.7.

## 8.5 WP-A2 §5 item 4 — folding the ASM spatial shift pair: LANDED

v5.5.3 and S5-8g had already removed the two SPECTRUM-domain shifts. The
remaining pair is around the FIELD, and for **even N it is the
identity** — the two `(-1)^k` phases the shift theorem produces cancel
against each other, so no checkerboard array is needed at all. For odd N
the shift is not by `N/2` and the phase is not `±1`, so the fold is gated
on both axes being even (measured `max|diff|` 2.1e-16 / 2.3e-16 /
9.5e-16 / 1.1e-15 at N = 63 / 65 / 127 / 255 — small, but not zero, and a
pinned-bits contract is a pinned-bits contract).

**Byte-identical** to the pre-change module at N = 64 / 128 / 256 / 512 /
1024 and 63 / 65 / 127 / 255, with `dy = 2 dx`, at complex64, with
`bandlimit` on and off, and on the streamed path: `max|diff|` exactly
0.0 in all eleven configurations. End to end with a warm H cache:
**54.74 → 36.48 ms (1.50×)** at N = 1024 and **242.55 → 186.28 ms
(1.30×)** at N = 2048; tracemalloc peak unchanged, because the `.copy()`
the folded path needs (the dropped `fftshift` used to detach the pyFFTW
ping-pong buffer — the v5.4.6 audit F-3 hazard) replaces the transient
the rolls allocated. The audit's own ASM repro numbers are unchanged.

Pinned three ways: the propagation still equals the textbook shifted
idiom written out in raw numpy (bar 1e-14, even and odd N); the shift
identity itself holds at even N and fails at odd N (a statement about the
maths, so the gate is not untestable); and the returned array owns its
memory after three later same-shape calls.

## 8.6 Files touched in the follow-up

| file | change |
|---|---|
| `lumenairy/propagators/hfpi.py` | V1: shared `_reemission_measure` + `_paths_weight_dtype`; `normalisation` on `apply_aperture_diffraction`, threaded from both entry points and the prescription walk |
| `lumenairy/propagators/vectorial_hfpi.py` | V1: the twin uses the shared helper; `normalisation` on `apply_vector_aperture_diffraction` and the entry point |
| `lumenairy/propagators/rs.py` | V6: `_warn_rs_transfer_wraparound` + two module constants, called on the `'transfer'` branch |
| `lumenairy/propagators/asm.py` | A2 §5 item 4: the even-N spatial shift fold (main and streamed paths) |
| `tests/unit/test_audit2609_a5_followup.py` | **new** — 38 tests (V1 ×7, V6 ×10, MFT dtype ×12, ASM fold ×8, + 1 shared) |
| `tests/unit/test_audit2609_a5_propagators.py` | K12 prefactor test restated on the `'legacy'` measure (it read the re-emission factor, which V1 changed) |
| `tests/unit/test_v4_14_0_dispatcher_pin_hfpi.py` | two RNG-stream pins apply an aperture on the emission plane; named `normalisation='legacy'` |
| `validation/propagators/test_hfpi.py`, `test_vectorial_hfpi.py` | the aperture checks now propagate a leg first (a stronger form of the same check) |

`mft.py` and `_bluestein.py` were NOT changed in this round — verified
byte-identical to `4996daeb`, so the complex128 MFT path is trivially
unmoved.

## 8.7 Tests run (follow-up)

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a5_followup.py -q` | **38 passed**, 14.6 s |
| `pytest` over the 25-file propagator surface incl. both verifier files | **662 passed, 1 skipped**, 106.5 s |
| `pytest` over the 8 ASM-adjacent files + the audit's `p1_asm_core.py` repro | **201 passed, 6 skipped**; repro numbers unchanged |
| `python validation/run_all.py test_propagation test_hf test_hfpi test_vectorial_hfpi test_mhs test_dispatch test_advanced_diffraction test_new_propagators_smoke test_gbd test_subaperture` | **10/10 files pass** |
| `ruff check lumenairy/propagators/ lumenairy/backend/scipy.py` + both new test files | clean for my files (3 remaining errors are in `asymptotic.py`, `asymptotic_maslov.py` and `carrier_field.py` — other WPs; `carrier_field.py:469` is an **F821 undefined name `S`**, a live bug worth flagging) |

**Fail-before.** All 8 substantive follow-up pins reject the pre-fix
state, re-introducing each defect in process: V1's closed form (factor
38.2 off), V1's transparency at three z1 (scale 9.84e-4 / 1.01e-3 /
3.19e-4 against the predicted `1/(n z1)`), and V6's four firing cases
(no warning at ring fractions 7.2–36.1 %). The earlier round's 20/20
still hold.

**Verifier changes reviewed and kept.** V2's `fft_infra` buffer-privatisation
latch, V3's `rs.py` docstring correction and V4's two `vectorial_hfpi.py`
docstring rewrites are in my files; I re-read all three, agree with them,
and their 20 tests pass alongside my ASM fold (which touches the same
`_ifft2` return path — the fold's own `.copy()` is independent of the
latch and the "returned array owns its memory" pin covers their
interaction).

## 8.8 Requested changes outside my ownership (follow-up)

4. **`lumenairy/propagators/carrier.py:9460–9461` — narrow the chief-ray
   ramp to the field's dtype (WP-A6's C3 residual risk).** This is the
   only reason the traced-carrier chain's tilted paraxial landing returns
   complex128; `angular_spectrum_propagate_mft` is not involved (§8.4).
   Exact change:

   ```python
   _ramp = np.exp(1j * k0 * (tilt_L * _u[None, :] + tilt_M * _v[:, None]))
   field = field * (_ramp.astype(field.dtype)
                    if np.iscomplexobj(field) else _ramp)
   ```

   The narrowing is exact for a unit-modulus ramp up to float32 rounding
   (measured 0.0 on the isolated probe). The comment at `:9463–9465`
   should lose its "which is not my file" clause. Owner: WP-A6.

5. **`lumenairy/propagators/carrier_field.py:469` — `F821 Undefined name
   `S``.** `ruff` fails on the committed tree:
   `ph = _phasor_rows(lambda r0, r1: (sign * 1j * k) * S[r0:r1], ...)`
   references an `S` that is not bound in scope. Not my file; flagged
   because it is a live `NameError` on whatever path reaches it.

## 8.9 Deferred (follow-up)

7. **Expose the RS pad factor.** The verifier's alternative to the V6
   guard. `rayleigh_sommerfeld_propagate` assumes a 2N pad throughout —
   the `'RS'` cache key, the `y0/x0` placement and the crop all encode it
   — so a `pad` parameter is a signature plus four internal changes plus
   a cache-key bump, and it interacts with the `'transfer'` branch's
   shared ASM cache entry (a 3N pad would no longer share with an ASM
   call). *Effort:* ~2 h plus a convergence test showing the guard goes
   quiet as `pad` rises. *Interim state:* the guard names the remedy
   (enlarge N, coarsen dx, or step in z).

# Independent verification probes -- Stage B (OUT-OF-PLANE staggered 2-D PMM)

Adversarial verification of the Stage-B integration (branch
`feat/pmm2d-staggered-oop`, commits `63893d1`, `514381e`, `7621256`, `aa88fdc`,
merged into `feat/pmm2d-staggered-anisotropic` as `6e49bed`).  The report is
`docs/audits/VERIFY_PMM2D_STAGGERED_OOP_2026_09_09.md`; the build's own claims
are in `docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md`.

Every probe here RE-MEASURES.  Nothing is accepted from a comment, a table or a
docstring, and the fixtures are deliberately NOT the build's: fresh tensors,
fresh periods / wavelengths / depths / angles, and two shapes the build never
used (a CHIRAL out-of-plane stripe and a chiral 2-D out-of-plane cell), because
the two gauge constants under test are invisible on any cell that is its own
180-degree image.

Every script asserts `lumenairy.__file__` is under the root passed as its first
argument, and sets `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS`
to 1 before importing numpy unless the caller overrides them.

Run pattern (from the worktree root):

```
PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_staggered_oop/<probe>.py \
  /c/tmp/lum_aniso validation/probe_verify_staggered_oop/results/<out>.json
```

| probe | what it measures |
|---|---|
| `v1_paths_untouched.py` | the SCALAR and IN-PLANE-TENSOR answers across three CODE VERSIONS (this HEAD, the pre-integration `0c3e871`, the main clone `f70628d`): sha256 of R / T / Jones and of `Lmat`/`Rmat`/`Stt`/`Schur`/`Et_blocks`/`Et_offdiag`/`W`/`V`/sorted `lam`.  Run once per `PYTHONPATH` |
| `v2a_gauge_chiral_1d.py` | `_OOP_ROT_SIGN` on a CHIRAL 3-segment out-of-plane stripe vs `pmm_jones_1d_segments` and `rcwa_jones_1d_segments` (both accept full `(3,3)` tensors at planar incidence), plus the uniform-slab Berreman arms, plus both flip arms.  Separates "invisible at normal incidence" (true for a rho-SYMMETRIC cell) from "invisible at normal incidence on a chiral cell" (false) |
| `v2b_gauge_chiral_2d.py` | the same on a genuinely 2-D CHIRAL out-of-plane cell vs `rcwa_jones_2d` (n_orders ladder 5/7/9, pixel-upsampled) and `pmm_jones_2d` in BOTH `E_z` rules; plus the no-floor property on that cell |
| `v2c_hgauge.py` | `_OOP_H_GAUGE` walked over `-1j, +1j, +1, -1, -2j` against `berreman_jones_1d` on a PURE single out-of-plane layer, a pure two-layer stack and a MIXED multilayer; the forced in-plane reduction on fresh cells; and the `layer_absorption` sweep.  Takes an optional 3rd argument selecting parts (`"1234"`, default) |
| `v3_dispersion_and_berreman.py` | the exact quartic `det[eps + kk^T - \|k\|^2 I] = 0` built from scratch in polynomial arithmetic, its four roots vs the assembled generator's spectrum at `+k_t` and `-k_t`, the sum-of-roots discriminator and the transpose-blindness identity; the Berreman M ladder; the drop / negate / TRANSPOSE fail-before controls on a symmetric AND a non-reciprocal tensor; the dispatch floor's byte-identity |
| `v4_stripe_cascade_stacks.py` | the y-uniform stripe per order (fresh fixture) with the y-momentum leak, the depth ladder with the forward/backward split and the growth factors, the `1x0.4 == 2x0.2` identity, the uniform multilayer vs Berreman, and the `layer_absorption` budget |
| `v5_break_attempts.py` | break attempts: a lossy metal (`eps = -20 + 2i`) region inside an out-of-plane host, a wavelength walked onto a Rayleigh cutoff from outside the warning band, a high-contrast pillar at `M = 8`, and oblique 60 degrees.  Includes an INDEPENDENT re-implementation of the pre-rebalance forward classification, because `_select_forward_flux` rebalances to exactly `2N` unconditionally and the library's `2 q^2 / 2 q^2` guard therefore cannot fire |
| `v6_durability_margins.py` | every bar of `tests/unit/test_pmm2d_staggered_oop.py` re-measured on the TEST FILE's own fixtures, so the durability table's margins are measurements.  Run twice, `OPENBLAS_NUM_THREADS` 1 and 4, for a cross-kernel envelope |
| `v7_cost.py` | the T10 cost claims: INTERLEAVED in-plane vs out-of-plane region-solve timings (ratios only) and `tracemalloc` peak allocation.  Run alone |

`results/` holds the JSON and the logs each run printed.

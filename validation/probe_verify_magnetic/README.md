# probe_verify_magnetic -- INDEPENDENT verification of the magnetic staggered 2-D PMM

Measurement probes for
`docs/audits/VERIFY_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`.  Nothing here is
read from the build's own probes (`validation/probe_pmm2d_staggered_magnetic/`)
-- every oracle, fixture and transform rule is written from scratch in these
files, and every derivation is stated in the module docstring so the algebra can
be checked without running anything.

Run each from THIS directory (some import their siblings) with

```
cd /c/tmp/lum_vmag && PYTHONPATH=/c/tmp/lum_vmag OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_verify_magnetic/<probe>.py
```

Every probe asserts `lumenairy.__file__` for its arm.

| probe | verifies | report table |
|---|---|---|
| `v1_bitidentity.py` | bit-identity of the three NONMAGNETIC paths (scalar / in-plane tensor / out-of-plane) between the MAIN clone `fb3fd93` and this worktree, over 12 fixtures / 75 hashed quantities.  Three arms: `main`, `wt`, `diff a.json b.json`. | V1 |
| `v1b_localise_wood.py` | the LOCALISATION control for V1 -- a scalar layer placed exactly on its own `(2,0)` cut-off, where the Wood-list change in the same range IS visible.  Two arms: `main`, `wt`. | V1 |
| `v2_derivation.py` | the magnetic operator algebra: `R`, `S_tt`, `K_tz` (through `Schur`) and `L` re-assembled here from the 1-D primitives for a uniform cell, plus placement knockouts and the R-vs-Gram trap. | V2 |
| `v3_airy.py` | G2 -- a uniform magnetic slab against a 2x2 transfer-matrix oracle derived here in the `exp(-i w t)` convention.  `python v3_airy.py convention` measures the time-convention branch. | V3 |
| `v4_duality.py` | G3 -- electromagnetic duality with the tangential-basis transform derived here (`T'_m = A(k^_m) T_m A(k^_inc)^-1`). | V4 |
| `v5_1d_bridge.py` | G4 -- the y-uniform MAGNETIC stripe against `pmm_jones_1d` and `rcwa_jones_1d` through duality, plus the engine census (`python v5_1d_bridge.py census`). | V5 |
| `v6_closure_symmetry.py` | G5 (lossless / lossy closure + the tripwire predicate), G6 (the x<->y transpose with the mu blocks swapped), G7 (every guard, both sides), and the composition claims (`layer_absorption`, the generalized cascade). | V6 |
| `v7_wood_followup.py` | follow-up 5 -- the `eps*mu` Wood cut-off products: the bit-identity of every NONMAGNETIC call site before/after, and the two-sided magnetic behaviour. | V7 |
| `v8_durability.py` | the durability audit of `tests/unit/test_pmm2d_staggered_magnetic.py`: every numeric bar re-measured on both sides. | V8 |

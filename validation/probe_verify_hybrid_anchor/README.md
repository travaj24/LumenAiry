# `probe_verify_hybrid_anchor` -- the INDEPENDENT verification probes

Companion to `docs/audits/VERIFY_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md`
(the adversarial verification of the hybrid 2-D PMM frame-anchor fix and of the
JAX slant refusal).  These scripts are **not** the fix's own probes
(`validation/probe_fix_hybrid_slant_anchor/`); every fixture here is built from
the geometry in `_lib.py` with a different period, wavelength, cell, slant
magnitude, walk fraction and staircase ladder, so a shared modelling mistake
cannot survive in both sets.

| | this set | the fix's own set |
|---|---|---|
| period | `0.90 um` (and `1.05 / 1.20 um` in the O2 probes) | `1.00 um` (`1.20 um` in `p1`) |
| wavelength | `0.62 um` | `0.68 um` |
| cell | 6 x 4 x-asymmetric, y-varying, `eps` 1.30 .. 3.05 | 6 x 6, `eps` 1.15 .. 3.24 on 1.44 |
| walk | a **QUARTER** period (`slant = 0.5`, `d = 0.45 um`) | a HALF period |
| staircase | K = 5 / 15 / 25 of 600-column cells, MIDPOINT rule | K = 3 / 5 / 15 of 60-column cells |
| mounts | normal, oblique 25, conical 25-40, oblique 40 | oblique 25, conical 25-40 (+ normal in `p1`) |

## Arms

Every script calls `_lib.arm()`, which decides `fix` / `v5440` from
`lumenairy.__file__` alone and stamps the resolved path, interpreter, numpy,
scipy and the thread caps into its JSON.

| arm | tree | how it is run |
|---|---|---|
| `fix` | `C:/tmp/lum_vhyb` (branch `verify/hybrid-slant-anchor`) | `PYTHONPATH=/c/tmp/lum_vhyb:<this dir>` |
| `v5440` | `C:/tmp/lum_v5440` (tag `v5.44.0`, READ-ONLY) | `PYTHONPATH=/c/tmp/lum_v5440:<this dir>` |

`v5.44.0` is `50824e9`; its `lumenairy/elements/pmm/` tree is byte-identical to
`9b36ded`, the commit the fix branched from (`git diff v5.44.0 9b36ded --
lumenairy/` touches only `_lens_traced.py` and `propagators/carrier.py`), so it
is a legitimate "without" arm for this fix.  The only other pmm change between
the two arms that the hybrid can even reach is `_warn_stack_energy`'s new
`stack=None` keyword, whose 2-D caller passes nothing and whose
`_sliver_refusal` returns immediately on `stack is None`.

Run everything with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

## The scripts

| script | task | what it decides |
|---|---|---|
| `_lib.py` | -- | fixtures, the exact-integer staircase builder, the residual metrics, arm detection, JSON dump |
| `q1_fixtures.py` | 1 | 29 fixtures x 7 keys = 200 sha256 hashes, run on both arms |
| `q1_compare.py` | 1 | diffs the two hash tables and classifies every move |
| `q2_anchor.py` | 2 | the anchor against the engine's OWN K = 5/15/25 staircase, three mounts, four arms; the unimodularity census |
| `q2b_pure.py` | 2(b) | the cross-engine arm (`PMM2DStackPure`, `n_modes` 4 and 5), bounded by the hybrid's own `n_orders` ladder |
| `q2c_oned.py` | 2(c) | the 1-D engine: which factorization exposes a transmitted field on a sheared grating, whether that field is lab-referenced, and the 2-D hybrid against the 1-D staircase |
| `q3_composition.py` | 3 | A slanted-over-film / B film-above / D two DIFFERENT slants / E+F the two null shapes below / G the reflection round trip / H the shear-continued SCOPE |
| `q3b_split.py` | 3 | the layer-split identity and its BLINDNESS to the walk sum |
| `q4_jax.py` | 4 | seven traced routes to the jnp twin, two controls, an AD-vs-FD gradient, and the `pmm_jones_2d` route the fix did not close |
| `q5_o2_blowup.py` | 5 | the O2 grid on two of this set's own fixtures, plus the generalized-cascade instrumentation (`Probe`: `cond(Mb)`, `cond(T22)`, the forward/backward eigenvalue split, the largest propagation factor) |
| `q5b_o2_exact.py` | 5 | the O2 grid on the EXACT fixture the fix's probes rejected, 4 stack shapes x 4 mounts x 6 `n_orders` |
| `q5c_mechanism.py` | 5 | the MODE-level instrumentation (`ModeProbe` on `_layer_modes_projected`): how many modes are on the wrong side of the forward/backward split, and the growing exponential that puts into the cascade -- plus the VERTICAL control |
| `q5d_conditioning.py` | 5 | the INTERFACE-level half: `cond(T22)` at the pairs `q5c` found, the full cut-off table (superstrate / substrate / every cell value), and the PURE control |
| `q5e_discriminators.py` | 5 | the four one-axis scans that decide what the blow-up depends on: DETUNE, `theta`, the slant magnitude, the spectral `degree` |
| `q6_durability.py` | 6 | re-measures every numeric bar in `tests/unit/test_fix_hybrid_slant_transmission_anchor.py` through the test module's own fixtures and prints bar / measured / margin |

`results/` holds every run's JSON (suffixed by arm) and the console logs.

## Traps met here, recorded so the next run does not meet them again

1. **The staircase roll must be an EXACT integer number of pixels.**  A walk of
   `period / D` staircased at `K` rungs by the MIDPOINT rule needs
   `cols * U / (2 D K)` integral for every rung;
   `_lib.staircase_upsample_factor` computes the smallest `U` and
   `staircase_cells` refuses a non-integer roll.  A quarter walk at
   K = 5/15/25 needs `U = 100` on a 6-column base -- 600-column cells.
2. **A slanted layer's frame CONTINUES into the next slanted layer.**  The
   lower half of a split layer takes the cell AS WRITTEN, not the cell rolled
   by half the walk: the cascade already places it at the accumulated walk.
   Rolling it measures a different solid (`3.68e-01` against the staircase
   instead of `1.97e-02`).  This is the same statement as the SCOPE finding.
3. **`PMM2DStackPure`'s cost is `(segments x n_modes)` per axis per
   component.**  `n_modes = 7` on a 6 x 6 cell is a ~7000-dof dense complex
   eig and runs for tens of minutes; `n_modes = 4/5` is the usable rung and is
   already converged for this comparison (shipped moves < 1% between them).
4. **`pmm_jones_2d` needs an ODD `degree`** (per-axis node count must be odd).
5. Run scripts with `python -u`: a backgrounded buffered run shows nothing
   until it exits.

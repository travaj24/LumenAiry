# `probe_verify_c2_round2` -- VERIFY-WP-C2 ROUND 2

Independent re-verification of the twelve defect closures on
`feat/c2-analytic-normal-round2` (`8ac607ee`).  Report:
[`VERIFY_WP-C2_ROUND2.md`](../../docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C2_ROUND2.md).

Every probe runs in its own process with `LUMENAIRY_ROOT` (or `--root`) on
`sys.path` and `lumenairy.__file__` asserted inside it, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, on **Windows py3.14.6 / numpy 2.4.4 / jax 0.11.0** and **WSL py3.12.3 /
numpy 2.4.6 / jax 0.10.2**.  PRE trees are this verification's own
`git archive` extractions: `49ddf4bd` -> `C:/tmp/lum_vc2b_pre49`, `61ffe596`
-> `C:/tmp/lum_vc2b_pre61`, `eadc67ba` -> `C:/tmp/lum_vc2b_preead`.

| probe | defect / claim | what it answers |
|---|---|---|
| `vr2_census.py` | D4, claim 1 | a MODULE-QUALIFIED census of every exported function whose body names a tracer, taken from every importable `lumenairy` module rather than seven.  20 exported + tracing, 16 with both keywords (keyword-only, default `None`), 4 = the jax twins, both builds |
| `vr2_wayback.py`, `vr2_wayback_compare.py` | D4, claim 1 | archive-to-archive byte identity over all sixteen entry points on THIS verification's fixtures (a spherical doublet, an ASPHERIC singlet, a two-SPHERICAL-MIRROR system): 37 cases, **1506 arrays**, four arms (`pre`, `post_oldkw`, `post_default`, `post_none`).  `--spy` records the `(sphere_normal, renormalize)` every internal tracer call received, which is how "the keyword reaches EVERY trace call" is checked rather than assumed |
| `vr2_alias_entrypoint.py` | **VR2-D1** | the SEVENTEENTH entry point.  `elements/_lens_real.py` imports the tracer as `trace as _rt_trace`, so `apply_real_lens` is invisible to both censuses; this probe spies the call, digests the field archive to archive, and shows the answer moves at the shipped defaults with no keyword to get it back |
| `vr2_transitive.py` | claim 2 | a module-scoped transitive walk (same module, then the module a name was imported FROM, then a globally unique definition; ambiguous names refused), and a check of the round-2 population's 11 unresolved and 35 claimed parents |
| `vr2_sphere_oracle.py` | D1, claim 3 | an exact-input sphere-normal oracle in TWO algebraically independent formulations (sag-derivative and geometric), agreeing to 3.3e-28 before either is used; 4284 points, 18 radii of both signs, refracting and MIRROR; plus what the retired `repr` conversion costs on this set |
| `vr2_kappa.py` | D2, claim 4 | the induced `inf<-2` norm of the field's Jacobian recomputed from scratch three ways (closed-form Gram eigenvalue, full per-row SVD, power iteration as a lower bound) at three FD steps; the other float inputs' sensitivities; and the 10x bar bracketed at 0.5x / 0.9x / 1.1x / 2x / 100x |
| `vr2_ghost.py` | D5, claim 5 | the ghost leg's route: spied, followed (the helper is made to answer differently), byte-compared at the refraction step against `trace`'s default AND against the generic route, with `renormalize`'s effective value read from `_refract`'s signature |
| `vr2_reanchor_abuse.py` | D7, claim 6 | twelve doctored `EDITED_IN_PLACE` cases through the real `_edited_in_place`, including the three round-1 abuses, a version bump, a same-content-different-owner line and a re-recorded digest |
| `vr2_d3_floors.py` | D8, claim 7, claim 13 | the two d3 arms' one-ULP floors over all four directions, whether `max(up, down)` brackets them, the margins at each, and the wall-clock COST of the second direction |
| `vr2_d3_d9.py` | D3, D9, claim 9 | the history-drift ladder against `n_surfaces * eps` (**VR2-D2**: exceeded 1.33x at three surfaces), the two routes' domain gates bisected at eight radii on the meridian, and the ball-lens / hemisphere `RAY_NAN` census under all four settings |
| `vr2_w6a2_newton.py` | claim 10 | the `w6_a2` second-Newton-step ratio at the shipped `v*` and at five deliberately unconverged points, so the gap above the 1e-4 bar is a measurement (**VR2-D5**: the dropped-prior mode reads 2.5e-05) |
| `vr2_mutants.sh`, `vr2_mutants2.sh` | claim 8, claim 11, **VR2-D3**, **VR2-D4** | twelve mutants, each a fresh `git archive` of the tip with one edit, run against both C2 test files.  M4 and M8-M11 SURVIVE |

`vr2_neutrality_594_*.json` are diffs of the round-1 verifier's own 594-key
`vc2_wayback.py` dump: `61ffe596` against the tip at the DEFAULT (594/594
identical -- round 2 moved no default), `49ddf4bd` against the tip with the
old keywords (594/594), and `49ddf4bd` against the tip at the default
(172/594, 422 moved).

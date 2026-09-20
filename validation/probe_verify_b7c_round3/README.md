# VERIFY-WP-B7c round 3 -- the independent verifier's probes

Evidence for `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7c_ROUND3.md`,
the adversarial verification of WP-B7c round 3 (`fix/wp-b7c-round3`,
`fc66a8f8 .. bf0b358b` off `wave5/audit-leftovers` at `dcaa21f0`).

Nothing here imports a builder probe.  The oracle, the optics, the geometry,
the ladders and the mutation set are written for this verification, and in
every case where WP-B7c round 3 re-used the round-2 verifier's module
VERBATIM (`r3oracle.py` *is* `probe_verify_b7c_round2/vroracle.py`) this one
differs in METHOD, so a shared mistake shows up as a disagreement rather than
cancelling.  `v3oracle.py`'s docstring has the method-by-method table.

## The modules

| file | what it is |
|---|---|
| `v3oracle.py` | the verifier's own oracle: Sellmeier typed here, a vectorised damped-Newton / bisection conic + EVEN-ASPHERIC meridional trace, angle-form Snell, a 4th-order ray-map Jacobian, a Simpson radial quadrature, a GAUSS-LEGENDRE exact azimuthal quadrature, a band-limited angular spectrum whose source is laid down by cubic interpolation of AMPLITUDE and OPTICAL PATH (not of the real and imaginary parts), and the scoring helpers |
| `v3fixtures.py` | the optics.  FOUR are new to the campaign (`VA` an even asphere on the EXIT surface with a non-monotone focal locus, `VX` at NA 0.437, `VC` a HYPERBOLIC conic, `VD` an air-spaced pair run negative-element-first); four more (`V`, `HN`, `W`, `Q`) are re-typed from the published prescriptions the claims are stated ON |
| `v3geom.py` | index control, NA, paraxial / marginal foci, the number of interior stationary points in the focal locus, and the interior-turning-point window every ladder is laid out from |
| `v3scan.py` | one library call per plane with both refusal bars lifted at run time, the shipped decision re-derived from constants captured before the patch, and a full-radius oracle score |
| `v3ladder.py` | the three-level z ladder (band + tail, then 20x, then 10x between every straddling pair) and the gap it reads |
| `v3control.py` | the converged reference, the GRID ladder, the WINDOW ladder, the healthy-fold spread, and the NESTED vs HULL-ALIGNED lattice A/B |
| `v3fbsweep.py` | the rest of the propagation axis, so the fallback population is not selected by its distance from the fold ring |
| `v3floor.py` | the oracle floor at FULL radius: `J0` vs exact, core-confined vs full, ASM vs exact, and the energy closure at three radial resolutions |
| `v3bitid.py` | bit identity ARCHIVE to ARCHIVE, a child process per tree, `lumenairy.__file__` asserted under it, `LUMENAIRY_MEM_BUDGET_MB` pinned |
| `v3mutate.py` | the 18-mutation matrix re-run, plus five of this verification's own |
| `v3tables.py` | the joined tables: the fold ring's two fidelity populations, the bar-cost table, the derived centre, and the fallback confusion |

## Reproducing

```sh
cd validation/probe_verify_b7c_round3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=<worktree> python v3geom.py geom_win.json
sh run_ladders.sh   win VA VA_alt VD          # one shard per group
sh score_nearbar.sh win V_alt W VX_alt        # the near-bar planes, scored
sh run_fbsweep.sh   win VD V V_alt HN W W_alt
sh run_controls.sh  win
python v3floor.py oraclefloor_win.json Q:5680 W:4900 VX:900
python v3bitid.py C:/tmp/lum_vmb3_pre C:/tmp/lum_vmb3_post bitid_win.json
MUT_TREE=C:/tmp/lum_vmb3_mut python v3mutate.py mutation_round3gate_win.json
python v3tables.py joined_win.json 'ladder_*_win.json' 'nearbar_*_win.json' \
                   'fbsweep_*_win.json'
```

The two archives are `git archive dcaa21f0` and `git archive bf0b358b`
extracted into `C:/tmp/lum_vmb3_pre` and `C:/tmp/lum_vmb3_post` with this
directory copied in; the mutation tree is a third export.  They are EXPORTS
and not worktrees on purpose: a worktree's `.git` is a Windows gitdir pointer
a WSL git cannot resolve, and the mutation driver restores by COPY from a
snapshot for the same reason.

`_win` / `_wsl` name the build (Windows py3.14.6 / numpy 2.4.4; WSL py3.12.3 /
numpy 2.4.6).  Every probe prints `lumenairy.__file__` as its first line.

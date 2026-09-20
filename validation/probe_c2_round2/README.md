# WP-C2 round 2 probes

Evidence for the defects VERIFY-WP-C2 raised, closed on
`feat/c2-analytic-normal-round2`.  Report: the "Round 2 (VERIFY-WP-C2)"
addendum in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-C2_ANALYTIC_NORMAL_REPORT.md`;
the defects themselves are in `VERIFY_WP-C2.md` beside it.

Every probe takes `--root` and asserts `lumenairy.__file__` is inside it before
anything else is imported, prints the interpreter, numpy (and jax) versions,
and writes one JSON per build.  All runs used
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, on **Windows py3.14.6 / numpy 2.4.4 / jax 0.11.0** and **WSL py3.12.3 /
numpy 2.4.6 / jax 0.10.2**.  `*_win.json` / `*_wsl.json` are the recorded
outputs.

| probe | defect | what it settles | headline |
|---|---|---|---|
| `r2_wayback_entrypoints.py` + `r2_wayback_compare.py` | D4 | the way back through all SIXTEEN internally-tracing entry points, archive to archive in separate processes | **742 / 742 arrays byte-identical** on both builds with the old keywords forced; **16 of 16** entry points move at the default; `None` byte-identical to omitted on all 742 |
| `r2_ghost_normal.py` | D5 | what the ghost leg's normal route costs, and that it now matches `trace` | RMS spot radius moves 9.663e-13 mm (Win) / 5.400e-13 mm (WSL); a spy sees `analytic_sphere=True` from the ghost retrace AND from `trace` |
| `r2_conditioning_kappa.py` | D2 | the TRUE condition number of the ModalAsymptotic field, against the sampled one | `kappa = 2.9059e+07` on both builds and both fixtures, **4.65x** the shipped random draw; the reading is 1.57-1.62 floors, not 7-9 |
| `r2_d3_floor_directions.py` | D8 | the d3 arms' one-ULP floor over four perturbation directions | spread **3.22x / 4.79x** (arm 2) and **2.07x / 26.99x** (arm 1); the same degree twice reads exactly **0.0** |
| `r2_rim_clamp.py` | D9 | whether the rim band exists on the meridian, and what a ball lens loses to the clamp | both gates bisect to the SAME float `0.9999499987499374` at all eight radii; **3024 of 60 000** rim-packed rays (5.04 %) die `RAY_NAN` on BOTH routes |
| `r2_jax_clamp.py` | D6 | how far apart the two backends' sphere-domain gates are | 1962 of 40 000 rays past the clamp: CPU keeps **0** under all four settings, JAX keeps **1962** |

Two probes already in the tree were RE-RUN rather than replaced, because the
defect was in them rather than in what they measured:

* `../probe_c2_analytic_normal/sphere_oracle.py` -- D1, the exact input
  conversion (`Decimal(float(x))`, prec 80).  Re-run on both builds, and again
  at `C2_PREC=120` to demonstrate convergence rather than assume it.
* `../probe_c2_analytic_normal/pins_restated.py` -- D2, the four
  `(renormalize, sphere_normal)` combinations against the corrected bar.

and three of the verification's own probes were re-run on this tree to confirm
the numbers the release text now quotes: `../probe_verify_c2/vc2_opcount.py`
(the element-op count), `vc2_wayback.py` (594 / 594) and `vc2_trace_truth.py`
(the 60-digit end-to-end oracle and the drift sensitivity).

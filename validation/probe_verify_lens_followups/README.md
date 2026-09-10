# probe_verify_lens_followups -- INDEPENDENT verification of the seven 5.44.0 lens follow-ups

Measurement for the adversarial verification of D1-D7 of
`docs/audits/FIX_LENS_5440_FOLLOWUPS_2026_09_11.md` (which are the defects
D1-D7 of `docs/audits/VERIFY_LENS_BANDED_COMPLEX64_2026_09_10.md`).  Findings
and every number: `docs/audits/VERIFY_LENS_5440_FOLLOWUPS_2026_09_11.md`.

Everything here is re-derived rather than re-used: `_vf.py` carries its own
prescriptions (an N-BAF10 positive meniscus, a fast N-LAK22 biconvex, a
cemented N-LAK22 / N-SF6 doublet), its own wavelength (1.55 um, against the
builder's 1.064 um and the shipped tests' 1.31 um), its own grids and its own
beams (including a speckled one, so no hash can be satisfied by a smooth-field
coincidence).  The two exceptions are stated where they occur: `q10` measures
the SHIPPED TESTS' constants, so it must use the shipped tests' fixtures, and
`run_builder_probe.py` re-runs the builder's own probes unchanged when the
question is whether their RECORD reproduces.

Every script prints `lumenairy.__file__` / `__version__` / python / numpy
first and REFUSES to run when the import did not come from the tree named by
`--tree`; the JSON carries the same block, so an arm cannot be mistaken for
another.

## Arms

| arm | tree | build |
|---|---|---|
| NEW | `C:/tmp/lum_vlens2` @ `verify/lens-followups` (= `wave2/pmm2d` merged) | Windows 11, py 3.14.6, numpy 2.4.4, scipy 1.17.1 |
| v5.44.0 | `C:/tmp/lum_v5440c` @ `v5.44.0` (`9af9376`) | as NEW |
| v5.43.0 | `C:/tmp/lum_v5430v` @ `v5.43.0` (`78e4091`) -- D6 attribution only | as NEW |
| SECOND BUILD | WSL Ubuntu `~/lumvenv` on `/mnt/c/tmp/lum_vlens2` | py 3.12.3, numpy 2.4.6 |

Windows worktrees for v5.44.0 / v5.43.0 are read-only and removed at the end.

## Run

```
cd /c/tmp/lum_vlens2
PYTHONPATH=/c/tmp/lum_vlens2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -P validation/probe_verify_lens_followups/<q>.py \
  validation/probe_verify_lens_followups/results/<out>.json \
  --tree /c/tmp/lum_vlens2 --tag new
```

`python -P` matters: without it `sys.path[0]` is the working directory and the
`PYTHONPATH` arm is shadowed by the tree you are standing in.  Swap
`PYTHONPATH` and `--tree` for `C:/tmp/lum_v5440c` to get the other arm; the
banner refuses the mismatch.

Second build:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vlens2 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vlens2 \
  ~/lumvenv/bin/python -P validation/probe_verify_lens_followups/<q>.py \
  <out.json> --tree /mnt/c/tmp/lum_vlens2 --tag wsl"
```

## Scripts

| script | follow-up | what it measures |
|---|---|---|
| `_vf.py` | -- | the fixtures, the arm banner (with its refusal), the field / record hashers, the traced-call driver |
| `qdiff.py` | -- | structural leaf diff of two probe JSONs, ignoring the wall clock and the arm block; `--allow` names the leaves a claim says may move |
| `q1_c128_chain_identity.py` | task 1 | every complex128 carrier chain (2 and 3 groups, sphere / parabola reference, paraxial / exact final leg), the two public carrier helpers at scalar / astigmatic / single-axis carriers, two exact focus readouts, both crop branches at both dtypes |
| `q2_traced_identity.py` | task 1 | 11 traced fixtures x band heights {None, 0, 7, 32, 128}: field hash, `sum abs(E)^2`, `max abs(E)`, the whole diagnostic record, and the warning list as MESSAGES and as attribution |
| `q3_d1_warn_attr.py` | D1 | `w.filename` of five ray-density notices driven over threshold, at band heights 0 / 32 / 7, plus a two-caller-module probe of the default filter's per-location dedup registry |
| `q4_d7_probe_rc.py` | D7 | the niche-C15 probe on 5 routes x band heights {0, 7, 32, 128}, with probe pixels straddling every band boundary and both edges, against the whole-grid arm, and the field with and without the probe |
| `q5_d2_memory.py` | D2 | whole-call `tracemalloc` peak of one public carrier-helper call at N = 1024 / 2048 / 4096 x scalar / astigmatic / single-axis x complex64 / complex128, and the full-grid complex128 census of a two-group chain by helper, caller and signature |
| `q5b_band_crossover.py` | D2 | the same peak straight through the band/grid crossover (N = 512 ... 2048), which is where "below N ~ 1414 there is no transient saving" is tested |
| `q5c_phasor_rows_penalty.py` | D2 | the four helpers that ALREADY took `dtype=` at 5.44.0, so the sub-crossover penalty can be attributed to `_phasor_rows` rather than to this change |
| `q7_d6_route.py` | D6 | six routes x {wall clock best-of-K, instrumented stage split}: `eval_into` calls and channels/pixel, `domain_mask` calls and PIXELS TESTED in whole grids, `build_inverse_map`, `map_coordinates`, and the field hash of every arm |
| `q9_d3_fft.py` | D3 | a two-group chain with an exact readout in three arms (shipped complex64 pair / pair forced to complex128 and narrowed once / complex128), the crop-call count of a paraxial chain, and the crop's own ladder at n_fine 256 .. 2048 |
| `q10_durability.py` | D4/D5, task 6 | every constant the three touched test files assert, with the bar and the margin on both sides -- run on Windows and on WSL |
| `run_builder_probe.py` | -- | runs one of the BUILDER's probes unchanged against an arbitrary arm (their `banner()` pins their own worktree name in a default argument; only that assertion is relaxed) |

`results/` holds the JSON.  `*_new.json` is the follow-up branch, `*_v5440.json`
/ `*_old.json` the released 5.44.0, `*_5430.json` v5.43.0, `*_wsl.json` the
second build; `p1_rerun_*` / `v6_rerun_*` are the builder's own probes re-run
on a named arm.

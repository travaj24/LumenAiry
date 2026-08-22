# PROBE -- can the design-121 analytic 8x8 run at N=32768?

Focused footprint probe, 2026-08-17, lumenairy 5.38.1 / 5.39.1.  Companion to
`AUDIT_DESIGN121_MODEL_CONVERGENCE_2026_08_17.md` (S7), which carries the
campaign context; this document is only the memory question and its answer.

**Answer: no, not on 136.6 GB, and every lever is now measured rather than
argued.**  The run needs **135.6 GB free on a 136.6 GB box**.  `exp31` at
N=16384 is the analytic result of record.

Binding law: `docs/TESTING_STANDARDS.md`.

## 0. The question

`surface_model='tangent_facet'` + `carrier='auto'`, 8x8 emitters, DOE on,
`complex64` field, aperture-containing 29.58 mm extent.  At N=16384 it runs in
64 min (exp31).  At N=32768 the preflight refuses.  Is that refusal reducible
without giving up accuracy?

Accuracy is the binding constraint on every candidate below: dropping
`carrier=` or falling back to `surface_model='displaced'` would fit, and both
are excluded because they change the model and break the like-for-like against
exp31.

## 1. The budget

| quantity | value |
|---|---|
| `need` (peak 100.5 GB x 1.15 safety) | 115.6 GB |
| `FREE_RAM_FLOOR_BYTES` (5.39.0) | 20.0 GB |
| free required (`need` + floor) | **135.6 GB** |
| box B free | 121.1 GB of 136.6 |
| shortfall | **-14.5 GB** |

The 5.39.0 remediation already removed the one term that was genuinely
mis-priced (the `screen_obliquity` phantom under the tangent-facet family,
dropped behind a `>= 5.37.0` gate).  What remains is real.

## 2. Lever 1 -- `sag_dtype=np.float32`: accuracy-safe, worth nothing

The geometry (coordinate / sag / OPD) stack is float64 regardless of
`FIELD_DTYPE`.  Halving it is the obvious move.

**Accuracy: safe.**  `lens_sag_float32_opd_error` per group, design 121, at
`field_check_n=512`, `field_check_dx=0.90 um`:

| group | max OPD error | field rel error | ok |
|---|---|---|---|
| S3-4 | 1.907e-04 waves (0.250 nm) | 1.338e-06 | True |
| S5-7 | 1.585e-04 (0.208 nm) | 7.626e-07 | True |
| S14-15, S16-17 | 0 (flat faces) | 0 | True |
| S18-20 | 1.886e-04 (0.247 nm) | 3.762e-07 | True |
| S21-22 | 2.796e-04 (0.366 nm) | 4.188e-07 | True |
| S23-24 | 2.130e-04 (0.279 nm) | 6.064e-07 | True |
| **S25-27** | **7.738e-04 (1.014 nm)** | **1.226e-06** | True |

Worst field relative error **1.338e-06 against the 1e-3 bar**, and the worst
OPD error is BELOW `tangent_facet`'s own 0.0032-wave residual, so float32 sag
would not be the limiting error anywhere in this design.  Two qualifications:
the probe runs at `field_check_n=512`, i.e. **~2 % of a 32768 pupil**, and it
is per-prescription.

**Saving: 0.001 grids.  Dead.**  The route reads 14.002 grids at N=4096 and
14.001 at N=8192 in BOTH dtypes.  float32 is worth ~3.5 grids only BELOW the
`N >= 4096` auto-band threshold: once the route bands, the full-grid geometry
is never materialised, so halving its dtype halves nothing.  Confirmed
independently -- preflight output is byte-identical in both dtypes.

## 3. Lever 2 -- `sag_chunk_rows`: flat

The preflight prices this route off a BINARY `_banded = N >= 4096` flag
(7.7 grids with a carrier), whereas `_slant_bytes` beside it scales with
`_band_rows / N`.  That asymmetry looks like an unclaimed credit.

Warmed `tracemalloc`, S25-27, extras over the paraxial no-carrier call at the
same N, in float64 grids of `8*N*N`:

```
N = 4096  (AUTO = 256 rows)      N = 8192  (AUTO = 512 rows)
  0 (whole-grid)   19.74           0 (whole-grid)   19.74
  256 (AUTO)       11.23           512 (AUTO)       11.23
  512              11.23           256              11.23
  128              11.23           128              11.23
```

**Flat from 512 rows down to 128 -- a 4x band reduction changes nothing.**
Banding is binary exactly as the preflight models it: the ~8.5-grid credit is
for banding at all, and there is none for banding harder.  That follows from
the halo being a FIXED 3 rows of sag + 2 of accumulator per band rather than a
proportional cost.  Extras are also identical at N=4096 and N=8192, confirming
the ANCHOR's "flat in N at and above 4096".

## 4. Lever 3 -- per-surface streaming: already done

If a group accumulated full-grid temporaries across its surfaces, freeing at
last use would cut the peak (the 5.34 traced-path fix recovered 16.25
grid-equivalents that way).  Same methodology, every design-121 group:

| group | surfaces | peak (grids) | EXTRA |
|---|---|---|---|
| S3-S4 | 2 | 14.62 | 11.23 |
| S5-S7 | 3 | 14.62 | 11.23 |
| S14-S15 | 2 | 14.62 | 11.23 |
| S16-S17 | 2 | 14.62 | 11.23 |
| S18-S20 | 3 | 14.62 | 11.23 |
| S21-S22 | 2 | 14.62 | 11.23 |
| S23-S24 | 2 | 14.62 | 11.23 |
| S25-S27 | 3 | 14.62 | 11.23 |

**delta per extra surface: +0.00 grids.**  The library already frees between
surfaces; there is no accumulation to reclaim.

## 5. Why there is no fourth lever

Banding already IS the serialisation, and it is where the 8.5-grid credit
comes from (19.74 -> 11.23).  The residual 11.23 grids is what must be
simultaneously live:

* the **momentum accumulator**, persistent by construction -- it is "advanced
  by the gradient of every screen imprinted so far", so it cannot be streamed
  per band or per surface;
* the **sag gradients** feeding it, which is what sets the 3-row halo;
* the **ASM work arrays** -- a global FFT, which cannot be tiled at all.

Streaming further would require an out-of-core FFT, which the library does not
have and which would trade a memory bound for an I/O bound on a 4-hour job.

## 6. Side finding -- the ANCHOR under-prices by ~3.5 grids

The measurement reads **11.23 grids against the preflight's 7.7** on this
prescription.  That is the wrong direction for a preflight.  The ANCHOR
(2026-08-16) was taken on a biconvex singlet; design-121 groups are 2- and
3-surface assemblies of Schott glasses.

Consequence: `FREE_RAM_FLOOR_BYTES = 20.0e9` is load-bearing rather than
belt-and-braces, and it is the only thing that made the refusal correct.  Had
the floor not landed in 5.39.0, an honest-looking `need` of 115.6 GB against
120.5 GB free would have been accepted and the run would have died at ~129 GB
system usage -- which is exactly what happened on 2026-08-16 before the floor
existed.

**Recommend re-anchoring the tangent-facet term on a multi-surface group.**

## 7. What would actually make it fit

| route | cost | verdict |
|---|---|---|
| a box with >= 136 GB free | -- | nothing on the mesh; box B totals 136.6 GB |
| drop `carrier=` | 7.7 -> 4.1 grids, `need` ~80 GB | FITS, but changes the model |
| `surface_model='displaced'` | unmeasured | different, less accurate model |
| out-of-core FFT | not implemented | would trade memory for I/O |
| accept N=16384 | -- | **exp31, already delivered** |

## 8. Method

Every figure above: warmed `tracemalloc` (call once to warm, `reset_peak`,
call again), peak reported as EXTRAS over the paraxial no-carrier call at the
SAME N, in float64 grids of `8*N*N` bytes -- the same shape as the library's
own ANCHORs so the numbers compare directly to the 7.7 coefficient.  Probe
group S25-27 unless stated.  Design-121 glasses registered from the design
study runner's `_NEW_GLASSES` Sellmeier table before any prescription naming
them is evaluated, or every group raises `Glass 'N-SK2' not in registry`.

Scripts: `band_scan.py`, `surf_scan.py`, `f32_check.py` (campaign scratch).

Author: footprint probe, 2026-08-17.

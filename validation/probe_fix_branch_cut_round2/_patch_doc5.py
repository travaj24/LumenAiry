"""One-shot editor: fill section 5 (the band-scale decision) of the round-2
report."""
import io

NEW = r"""### 5.2 The measurement

Three populations of the SHARED function's own quantity, both builds.  A mode is
NOISE-side when its `lam^2` is real negative to within the eigensolver's own
backward error -- a lossless propagating mode whose imaginary sign is rounding,
which the band MUST reach -- and SIGNAL-side otherwise, which the band must NOT
reach.  The classification is made on `lam^2`, i.e. on the eigenproblem, never
on the ratio under test.  The band ratio is computed from the RAW `eig` output,
so it is a property of the eigenproblem and identical on both arms of the
change.

**Population 1 -- RCWA, ordinary mounts** (`b6_band_scale.py`: 58 fixtures --
lossless anisotropic 2-D over twist x truncation x substrate, oblique, conical,
scalar 2-D at three contrasts, 1-D TE/TM at three duties, high-contrast gratings
at `n_orders` 21 and 31, a LOSS LADDER `Im(eps)` 1e-2 .. 1e-14 on both paths,
four metals to `eps = -100 + 5j`, three near-Wood mounts; 2497 `Im(r) < 0`
modes):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN | **5.5699e-15** | 1.2612e-02 | **12.35 dec** |
| ARRAY-MAX | WSL | **6.4448e-15** | 1.2612e-02 | **12.29 dec** |
| PER-MODE | WIN | 8.1213e-13 | 9.9215e-01 | 12.09 dec |
| PER-MODE | WSL | 9.3970e-13 | 9.9215e-01 | 12.02 dec |

**Population 2 -- RCWA, LAYER-CUTOFF mounts** (`b6b_cutoff.py`: 72 mounts, each
found by a bounded scalar minimisation of `min |lam^2|` over `n_ridge` after a
90-point bracketing scan, plus a +/- ladder at relative 1e-3 .. 1e-9 around each
so the corner is a family and not one engineered point; deepest mount
`min|lam^2| = 6.5341e-10`):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN and WSL | **9.0475e-13** | 2.1844e-05 | **7.38 dec** |
| PER-MODE | WIN and WSL | **6.8698e-08** | 1.0000e+00 | 7.16 dec |

(The two builds agree to every printed digit here, which is what one expects:
the ratio is read off `eig`'s output before any branch decision, and these
mounts are found by the same deterministic minimisation on both.)

**Population 3 -- the HYBRID PMM's SEM-projected `P@Q` spectrum**
(`b6_band_scale.py`: the `pmm/twod.py` layer eigenproblem over weak and strong
modulation, on and off the coincidence, a metal, three loss rungs, three
truncations, plus the tensor entry; 1463 (WIN) / 1642 (WSL) `Im(r) < 0` modes):

| shape | build | noise-side max | signal-side min | gap |
|---|---|---|---|---|
| ARRAY-MAX | WIN | **5.5766e-16** | 2.8623e-02 | **13.71 dec** |
| ARRAY-MAX | WSL | **1.7884e-15** | 3.8802e-02 | **13.34 dec** |
| PER-MODE | WIN | 2.2213e-14 | 8.4520e-02 | 12.58 dec |
| PER-MODE | WSL | 1.1973e-13 | 8.4520e-02 | 11.85 dec |

### 5.3 The decision: KEEP the array-max shape

**ARRAY-MAX has the larger two-sided gap on all three populations and on both
builds** -- 12.35 / 7.38 / 13.71 against 12.09 / 7.16 / 12.58.  The shape is not
changed, and per the brief that leaves only the MARGINS to re-derive (D3).

The cutoff population is where the two shapes were supposed to separate, and it
separates them the other way round.  The reason is physical rather than
arithmetic: **at a layer cutoff the mode's own magnitude collapses**, so judging
its real part against its own magnitude is judging noise against noise.  At the
deepest mount here `|lam| ~ 2.6e-05` while the per-mode floor
`sqrt(eps_mach) * max|r|` is ~4.5e-08, so the per-mode divisor is `|r|` itself
and the ratio blows up: its noise side reaches **6.8698e-08, which is ALREADY
ABOVE the shipped `1e-8`**.  Adopting the per-mode shape at the current constant
would start MISSING modes the array-max shape catches -- the exact failure the
band exists to prevent.  Re-deriving the constant for it would put it near
`2.6e-04` (the geometric mean of 6.87e-08 and 1.0), a four-decade move for a
narrower window.  The spectrum's top is the only stable scale at a cutoff.

### 5.4 D3, re-derived: the margins the constant now states

With the shape kept, the `1e-8` bar sits, on the ARRAY-MAX ratio:

| population | decades above the noise side | decades below the signal side |
|---|---|---|
| RCWA ordinary (both builds) | **6.19** (6.4448e-15) | **6.10** (1.2612e-02) |
| PMM hybrid (both builds) | **6.75** (1.7884e-15) | **6.46** (2.8623e-02) |
| RCWA layer cutoff, this box | **4.04** (9.0475e-13) | **3.34** (2.1844e-05) |
| RCWA layer cutoff, the verification's deeper ladder | **0.17** (6.7172e-09) | -- |

The round-1 docstring's "7.6 decades above the noise side and 6.9 below the
signal side" was measured on 51 fixtures that carried no cutoff mount, and is
replaced by the table above.  The binding number is the last row: the
verification drove `min|lam^2|` to 4.495e-15 and read a noise-side ratio of
6.7172e-09, 1.5x under the bar.  My own ladder reached only 6.5341e-10 and read
9.0475e-13, so I reproduce the SHAPE of that finding rather than its worst
value, and the verification's number stands as the worst known.

**Why the noise side has no floor, and what that means.**  For
`lam^2 = -s + i eta` the principal root's real part is `eta / (2 sqrt(s))`, so
at fixed backward error `eta` the ratio grows without limit as `s -> 0`.  The
noise side is therefore set by `sqrt(|lam^2|_min)` and not by the eigensolver's
backward error alone -- which means no constant can be proved safe at an
arbitrarily deep cutoff, on EITHER shape.  What can be said, and is:

* the mode that reaches the noise side there is no longer cleanly propagating
  (its `lam^2` sits at ~45 degrees), so its imaginary sign is genuinely
  ambiguous rather than wrong;
* it carries no z-directed flux, and `_inv_lam` regularises it downstream, so no
  wrong answer follows from either choice on any fixture measured -- 72 mounts
  here, 28 in the verification, on both builds;
* and the two shapes fail in opposite directions there, so this is a corner of
  the problem rather than a defect of the constant.

**D4 -- the scale-relative corner, quantified again.**  Because the scale is the
array's largest root, a mode whose real part is a fraction `rho` of its OWN
magnitude is conjugated once its magnitude falls below `_CUT_BAND_REL / rho` of
the spectrum's top.  The worst case over the 72 cutoff mounts here is
`|Re r| / |r| = 6.8698e-08`; the verification's deeper mount reached 2.0751e-03.
Both are recorded in the constant's comment.  The consequence is nil for the
reason above, and it is now documented rather than implicit.

**D5 -- unchanged by round 2, deliberately.**  The verification found one
engineered cutoff mount that moves REFUSED -> WARNED across round 1 at a closure
defect of +4.2519e-02.  Round 2's shared body is BIT-IDENTICAL to round 1's
(section 3), so that mount behaves identically before and after this change:
round 2 neither improves nor worsens it, and it stays an open observation about
the round-1 fix.

---

"""

p = "docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md"
s = io.open(p, encoding="utf-8").read()
start = s.index("### 5.2 The measurement")
end = s.index("## 6. X-1 is CLOSED")
io.open(p, "w", encoding="utf-8", newline="").write(s[:start] + NEW + s[end:])
print("section 5 filled")

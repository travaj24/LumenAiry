"""Append the infra CI items (6.4-6.7) to the report."""
import io
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

ANCHOR = "## 7. Runs\n"

NEW = '''### 6.4 `--maxfail` 10 -> 50, and what the lower cap actually cost

The 2026-09-12 change to 10 argued that "50 per shard across 5 shards x 5
pythons tolerated up to 1 250 failures before any job aborted".  That sums the
MATRIX, but nothing waits on the matrix sum: a job's wall clock is bounded by
its own budget, and a catastrophically broken job reaches 50 in its first
minutes exactly as it reaches 10, so the early-abort property survives the
revert.

What it cost, measured on run 34914295323: all five py3.11 shards stopped at
exactly `10 failed`, having executed 188-233 of ~2 933 selected ids -- **6.4 to
7.9 %**, so **more than 92 % of the 3.11 lane never ran**.  49 of the 50
failures are ONE parametrised gate with 123 cases, and the same case ids pass on
the other interpreters in the same run.  Independent measurement (6.2) puts the
true size of that class at **110 of 123**, so the run reported 45 % of it.  At
~22 cases per shard, 50 enumerates the class whole in one run and 10 truncates
every shard at the same 45 %.

The cap costs a healthy lane nothing: the 3.12 shards reported 3 and 2 failures,
3.13 reported 4 / 2 / 4, 3.14 reported 8 and 1 -- all below 10, so the cap never
fired there.  It fires precisely on the lane whose census is worth having.  The
comment also records that this is a FAILURE budget, not an error budget: a
collection error aborts the shard whatever `--maxfail` says, which is the 3.10
mechanism of 6.1.

### 6.5 The slow lane -- the timeout was predictable before the run started

All five slow shards hit the 30-minute step cap (1 847-1 863 s; shard 2 stopped
at 65 %, shard 5 at 98 %, neither with a summary line).  The brief's hypothesis
was a balance problem from files moved into the lane without durations.  **That
is not what it was**: measured against the committed `.test_durations`, the slow
selection is **935 ids with 935 entries -- a 100 % coverage, zero gap**, so
nothing needed regenerating and the file was not touched.

The real cause is size.  The lane now totals **9 630.0 s** against the
**5 914.9 s** the 2026-09-12 note recorded -- **+62.8 %**, because later waves
kept moving files over the two-minute bar without re-running the sum.  At
`--splits 5`, `least_duration` balances that to **1 926.0 s per shard, +/-0.0 %**,
against an 1 800 s step cap: the lane was predicted to time out on every shard
before the run started.

Both changes are needed, and the arithmetic is in the workflow comment: at 8
splits the per-shard budget is 1 203.8 s; the runners scale the recorded seconds
by 0.98x (shard 5) to 1.48x (shard 2), so sizing on the worst factor gives
~1 782 s -- against the old 1 800 s cap that is a 1 % margin, not a margin, and
raising the cap alone leaves 1 926.0 x 1.48 = ~2 850 s against 2 700 s, still
over.  Together: ~1 782 s against 2 700 s, 34 % headroom.  Shipped as `--splits`
5 -> 8 with the shard list in step, the slow step cap 30 -> 45 min and the job
cap 35 -> 50, plus an explicit "`--splits` must equal the length of the `shard:`
list" warning at both sites.

Open, and flagged in the comment for an owner: the 0.98-1.48x spread itself.
100 % duration coverage rules out the missing-entry explanation; the leading
candidate is that `.test_durations` was captured with BLAS unpinned while the
slow lane pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` in its `env` block, so eig-bound
entries are on a different scale there than where they were timed.

### 6.6 mypy strict -- one error, fixed at the expression

`decompose_lg` normalised its `only` argument with
`tuple(tuple(k) for k in only)`.  `tuple(iterable)` types as
`tuple[_T_co, ...]`, so `tuple(k)` of a `(p, ell)` pair **widens**
`tuple[int, int]` to `tuple[int, ...]`: the length information was destroyed by
the constructor, at the call site.  The callee's annotation is the truth, not
the lie -- it consumes `only` as `for (p, ell) in only`, twice.  Unpacking by
name instead (`tuple((int(p), int(ell)) for (p, ell) in only)`) types exactly as
`tuple[tuple[int, int], ...]`, keeps the normalisation the re-wrap existed for
(a caller handing `[[0, 1]]` or numpy scalars still arrives hashable for the
frozenset and the cache key) and keeps the loud failure, one frame earlier.
Neither the whitelist nor an `ignore` was touched.

Before: `Found 1 error in 1 file (checked 33 source files)`.  After:
`Success: no issues found in 33 source files`, on Windows py3.14 and WSL py3.12.
Behaviour proved bit-identical against the full decomposition for tuple, list
and numpy-scalar inputs, with the same exception class and message shape for
wrong-length input.  Reported and NOT edited: the identical redundant re-wrap in
`asymptotic_aberration_tensor.py`, which is outside the mypy whitelist today and
will raise the same error the moment that module joins the ratchet.

### 6.7 The platform / kernel bit pins

Each was root-caused before any bar moved, and in three of the six the cause was
NOT rounding.

* **`test_d3_offplane_fff_nv_keeps_the_cells_own_mirror`** (b5) -- the fixture's
  precondition, an exact `np.array_equal` mirror test, failed on Linux and
  passed on Windows.  Cause: `uniaxial_tensor` builds the x and y legs from
  different expressions in `cos(phi)` and `sin(phi)`, so at `phi = 45 deg` the
  fixture's symmetry is only as exact as the platform libm's
  `sin(pi/4) == cos(pi/4)`.  MSVC returns both as `0x3fe6a09e667f3bcd`; glibc
  returns `sin` one ULP low, and the tensor's `exz`/`eyz` and `ezx`/`ezy` pairs
  then differ by 3.33e-16.  **Fixed in the construction, not the bar**: the
  fixture is averaged with its own mirror, which is exact on every IEEE-754
  platform (addition is commutative and multiplication by 0.5 is exact), so
  `array_equal` is kept.  A derived 8-ULP guard bounds how far the raw tensor
  may sit from its mirror before the averaging would be manufacturing a symmetry
  rather than repairing round-off.

* **`test_h4_bor_pencil_eigh_accuracy`** (a14) -- 1.326e-12 against a 1e-12 bar.
  The bar was one arm's reading.  It is now derived from the pencil: for a
  symmetric-definite pencil reduced by Cholesky, LAPACK's backward-error result
  bounds the relative error on `sqrt(lam_i)` by `p(n) eps lam_max / (2 lam_i)`,
  worst at the smallest eigenvalue, with `lam_max` read off the solver's own
  spectrum and pinned to its mesh-determined range so a broken solve cannot
  inflate its own bar.  The ladder shows an **11x swing on one machine from the
  BLAS kernel alone** (HASWELL 2.758e-13, NEHALEM 8.882e-14, KATMAI 9.924e-13,
  SANDYBRIDGE 5.732e-13), with KATMAI landing 0.8 % under the old bar; WSL reads
  2.758e-13, bit-identical to the Windows HASWELL arm, which is the control that
  says the swing is the kernel and not the OS.  Fail-before by mutating the
  discretisation through the public API: degree 8 -> 5 puts the residual 2-4
  decades outside on every fixture, and dropping the `m^2` axis term puts it 11
  decades outside.

* **`test_the_tf_step_is_bit_identical`** and
  **`test_the_on_axis_answer_moves_by_at_most_two_ulp`** (a6) -- the CI reading
  of exactly `0.0` is not rounding.  `_fit_carrier_inv` evaluates the moment as
  `sum(xm * (wgt*slope))` when it projects and as `sum((wgt*xm) * slope)` when
  it does not: a different ASSOCIATION of the same three factors, which
  `carrier.py`'s own comment there already said "moves the answer by a few ulp".
  Multiplication is commutative but not associative, so byte equality was never
  an invariant of that pair.  Measured, the projection correction is 3e-21 to
  2e-19 ULP of the moment it corrects -- it cannot move any bit -- and the whole
  difference is the re-association, 0-4 ULP across the eight-arm ladder and
  0-2 ULP on Linux.  Bar 16 ULP, unconditional: 4x over the worst arm measured
  and 14 decades under the defect it guards (a genuine centring error is
  `1 + 2 x0^2/w^2`, ~2e15 ULP).

* **`test_z3_stokes_and_dop_peak_arrays`** (a11) and
  **`test_b8_apply_jones_matrix_peak_full_grid_arrays`** (b8) -- both are
  allocation counts, not timings, and both moved because NumPy's `temp_elide.c`
  rewrite of `a*X + b*Y` / `Ex * conj(Ey)` into an unreferenced temporary's own
  buffer is a **BUILD** property (it needs `backtrace()` and a stack walk that
  can confirm the temporary came from the interpreter).  It does not follow the
  operating system: on this run the add form was elided on the py3.14 Linux
  runner and not on py3.12 or py3.13, while the conj form was elided on py3.11.
  So each file MEASURES its own elision premise on the running arm, with a
  companion arm that binds the temporary to a name (lifting its refcount out of
  elision's reach) asserted at the un-elided count, so a reading of "no elision
  here" can never come from an instrument that measured nothing.  The slack
  above a whole number of grids is derived two-sided: tracemalloc's own
  bookkeeping measured 448-7 712 B over 9 repeats on two arms, at most 4.6e-04
  grids, and the slack is 0.05 -- two decades above that spread and 1.3 decades
  below the 1.0 that separates one allocation count from the next.

* **`test_z3_estimate_lens_memory_real_bounds_apply_real_lens`** (a11, found on
  this box's own ladder rather than in the CI logs) -- read 0.88 against a
  fail-safe bar of 1.0, i.e. the pre-flight estimate under-reserving.  It was
  neither load nor a real under-reservation: the test's warm-up call ran on a
  64x64 block, BELOW the grid size at which `apply_real_lens` takes a
  deferred-import branch, so ~11.2 MB of one-time module imports landed inside
  the measured region (16.78 MB from `fft_infra`, 9.55 MB from
  `importlib._bootstrap_external`, 4.19 + 4.19 MB from `asm.py`).  The
  Windows-minus-Linux excess is 13.7 MB and is EXACTLY constant across N and
  dtype -- bytecode, not lens arithmetic.  Warming at 256x256 makes all four
  (N, dtype) cells byte-identical on both arms (est/peak 1.064 / 1.239 / 1.072 /
  1.323), with both bars unchanged, plus a new unconditional guard that the
  measured call must RETAIN 5.5-7.0 complex grids (clean 6.01-6.09; contaminated
  8.38 under pytest and 9.49 standalone -- 15 % above the worst clean and 16 %
  below the lowest contaminated).

* **`test_w3_t3b_*`** (w3 oracles) -- `val_a` swings 2.7x across BLAS kernels on
  one box while `val_b` holds to 1.6e-3 and the response to 3.2e-05, because the
  merit's aberration-free reference is COLLAPSED on that chart, so `val_a`'s
  scale is not a measurement at all.  The physics pins keep their bars; `val_a`
  gets a derived order-of-magnitude band (4.8x under the lowest reading, 18x
  over the highest, with the smallest rescale it must still catch nine decades
  away); and the collapse itself becomes a POSITIVE pin -- the warning must fire
  and name a coupling far outside [0, 1] -- so the band can never quietly become
  a band on a Strehl.  The library-side repair (a saddle-local reference) is
  VERIFY-A4 O-3b and is not taken here.

'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if src.count(ANCHOR) != 1:
        print('anchor not unique:', src.count(ANCHOR))
        return 1
    src = src.replace(ANCHOR, NEW + ANCHOR)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('sections 6.4-6.7 written')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Insert the ``## [Unreleased]`` block above ``## [5.47.0]``.

Never a second Unreleased header: if one already exists the script refuses.
No ``path.py:N`` citations are written -- the V18 walker treats those as live
source-line citations on the topmost block.
"""
import io
import sys

P = 'CHANGELOG.md'
ANCHOR = '## [5.47.0] — 2026-09-14\n'

BLOCK = '''## [Unreleased]

Wave 5 item D of the 2026-09-11 adversarial audit's remediation
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PLAN_WAVE5_LEFTOVERS_2026_09_14.md`):
the four reds the 5.47.0 gate carried, the CI matrix that release turned red, and the
mechanical half of the warning-attribution work.  The report is
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B14_KNOWN_REDS_REPORT.md`;
every probe and its per-arm JSON is under `validation/probe_known_reds/`.

### Fixed

- **The c7 / c8 halo pins were a real regression, not box state.**  The 5.47.0 handoff
  recorded four ids in `tests/unit/test_niche_c7_ray_density_halo_check.py` and
  `tests/unit/test_niche_c8_inverse_support_bound.py` as failing "on this box" with the
  cause untraced.  They fail on the Linux CI runners too, agreeing with this box to twelve
  significant figures (1.5217194665917948e-04 against 1.5217194665917584e-04), and they
  bisect to one commit: WP-A26, which re-derived the decentred ray fit's order from 10 to
  16 for niche D7 and did not restate the two fixtures that depended on the old default.
  The fixtures' manufactured lobe is reachable at order 10 and at order 10 only on this
  geometry -- measured beyond three beam radii, with the C8 support bound off: 1.463e-04
  at order 6, 8 and 12, **4.595e-02 at order 10**, 1.533e-04 at 14, 1.522e-04 at 16.  The
  stimulus is now STATED by the fixtures (`decentred_fit_poly_order=10`) instead of
  inherited from a default that moved, the same way the M2 window contract already states
  its `min_feature`; at 10 the fixtures reproduce their own documented readings exactly,
  including the 51.5x the support bound's docstring quotes.  No bar was loosened and the
  fail-before assertion stays a hard assertion.  WP-A26's order-16 default is not
  challenged.  (4 failed / 24 passed -> 28 passed.)

- **The glass-validity one-shot pin was a partial reset of coupled state.**
  `test_validity_warning_is_one_shot_per_pair` was red only when it ran after
  `test_audit_w4_glass_registry_meshgrid.py`.  The state that leaks is not the warn-once
  set the test's fixture was clearing: `get_glass_index` memoises the whole
  (name, wavelength) evaluation and returns on a hit BEFORE it reaches the validity
  warning, which the memo's own rationale says is warning-neutral only because
  `clear_asm_caches()` empties both together.  The fixture cleared one of the pair,
  producing exactly the state the library's design excludes.  It now drains through the
  library's registered drain.  Green in both orders and alone.

- **`test_pmm_m2_window_contract`'s T3-1 no longer classifies an outcome by the BLAS
  build.**  Reproduced under the kernel ladder: green on seven of eight arms, red on
  SANDYBRIDGE at four threads, where the flux cut's growth census screens the degree-10
  halfwidth-2 cell.  Which cell round-off classifies that way is, in the module's own
  words, a per-build, per-thread-count fact.  The window contract is now asserted
  unconditionally on every cell that passes the screen; the spectral-decay claim is
  asserted over whatever rungs were measured rather than only over a complete ladder (so
  the failing arm now carries a decay claim it previously dropped); measuring NOTHING is
  still a hard failure; and only the existence of a complete three-rung ladder -- the one
  reading that moves with the build -- is premise-gated, skipping with the measured cells
  and the full screened census in the message.

### Changed

- `lumenairy.propagators.carrier`, `lumenairy.propagators.system` and
  `lumenairy.propagators.carrier_field` join the warning-attribution sweep: 21 literal
  `warnings.warn` stacklevels and 23 threaded literals below them are retargeted to
  `lumenairy.elements._lens_kernels.caller_stacklevel()`, which walks out to the first
  frame outside the package and is therefore correct at every call depth.  MEASURED
  before the sweep with a two-caller instrument: the tilt-inert notice named the caller
  when `propagate_carrier_referenced` was called directly and named library source when
  the identical warn site was reached one frame deeper through
  `carrier_referenced_focus_readout` -- 2 of 4 emissions misattributed, 0 of 4 after.  The
  three warning helpers take `stacklevel=None` meaning "compute it"; an explicit integer
  keeps exactly its old meaning, so external callers are unaffected.  The b11 ratchet
  gains a sibling over the three chain modules and the two-caller fixture as a test.  The
  remaining sixteen propagator modules that a static screen flags are recorded with their
  census as open work, not swept blind: each needs its own measurement first.

### Added

- `lumenairy.propagators.gbd.DENSE_MEM_BUDGET_ACCOUNTING` (`'legacy'`, the default and
  byte-identical, or `'measured'`).  The dense beamlet reconstruction sized its chunk from
  16 bytes per output cell per beamlet-column, a figure whose comment claimed to cover
  three float64 buffers and one complex128.  Measured with `tracemalloc` over a
  64/128/192/256 grid ladder at 512 and 64 MB budgets: the live peak is **72.0 to 96.8
  bytes** per cell-column, so the loop overruns its own `mem_budget_mb` by 1.2x to 6.0x,
  saturating at 6.0x once the chunk is the binding constraint -- a 512 MB budget peaked at
  **3 073 MB**.  With `'measured'` the same cell peaks at 387 MB, under the budget, and the
  two arms differ by 2.1e-17 relative, which is summation-order round-off and nothing else.
  It is opt-in because correcting the constant moves the chunk boundary and therefore the
  output bytes on a default path; **whether it becomes the default is a decision reserved
  for the maintainer**.  Without flipping it, `window=5.0` (whose accounting is correct) or
  dividing `mem_budget_mb` by six are the mitigations.  This BOUNDS, and does not close,
  the access-violation crash the 5.47.0 handoff recorded in this path: no fault was
  reproduced, but the transient is up to six times the size the caller asked for.

'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if '## [Unreleased]' in src:
        print('refusing: an Unreleased header already exists')
        return 1
    if src.count(ANCHOR) != 1:
        print('anchor not unique:', src.count(ANCHOR))
        return 1
    src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('inserted [Unreleased] above [5.47.0]')
    return 0


if __name__ == '__main__':
    sys.exit(main())

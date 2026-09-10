"""One-shot editor: restate the two X-1 tests of
``tests/unit/test_m1_conditioning_guard.py`` as decisions about the CLOSED
state.  Kept in the probe directory so the edit is reproducible."""
import io

NEW = r'''# X-1 on the library's own documented instability class -- CLOSED 2026-09-11
# ---------------------------------------------------------------------------
#
# X-1 WAS the RCWA modal branch-cut defect, and the round-1 fix
# (`docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`) closes it.  The
# geometry says why: `THIN`'s groove index equals BOTH half-spaces' (1.5), so
# it is a permittivity coincidence on the substrate AND the superstrate side at
# once, and a propagating layer mode handed the INCOMING root is then exactly a
# half-space BACKWARD mode -- which makes the interface mode-match `a + b`,
# whose explicit inverse IS `S12`, singular.  That is the near-cancelling
# denominator this file's census was built to record.
#
# Re-measured independently on the whole ladder, both arms in one process, with
# the census armed exactly as the tests below arm it
# (`validation/probe_fix_branch_cut_round2/b3_x1.py`, 2026-09-11):
#
#   ladder / arm            raises  flagged cells  worst |R+T-1|  worst sum(R)
#   TE  WIN 1 thr  PRE         7         14          3.1956e-02     152.60x
#   TE  WIN 1 thr  POST        0          0          1.3323e-15    5.064e-02*
#   TE  WSL 1 thr  PRE         8         14          3.1956e-02     152.60x
#   TE  WSL 1 thr  POST        0          0          1.4433e-15    5.064e-02*
#   TM  WIN 1 thr  PRE         5          9          2.6165e-04       1.2998x
#   TM  WIN 1 thr  POST        0          0          9.9920e-16    1.887e-02*
#   TM  WSL 1 thr  PRE         5          9          1.5404e-04       0.7654x
#   TM  WSL 1 thr  POST        0          0          9.9920e-16    1.887e-02*
#
#   (*) the POST "worst" is the COARSEST rung, M = 6 -- ordinary truncation
#   convergence, not a defect; every finer rung is better.
#
# The four historically pinned cells, Windows one thread, sum(R):
#
#   M = 12 TE  2.016454e-04 -> 2.015824e-04
#   M = 19 TE  1.838764e-02 -> 2.053766e-04
#   M = 20 TE  2.088570e-04 -> 2.053491e-04
#   M = 21 TE  3.216567e-02 -> 2.095174e-04
#
# and M = 19 TE, the cell whose whole point was that it RETURNED 1.018 on
# Windows and RAISED on WSL, now returns 2.053766e-04 on BOTH builds.
#
# The two tests below are therefore restated as DECISIONS ABOUT THE CLOSED
# STATE.  Their fail-before is ENGINEERED rather than found -- the pre-round-1
# branch body is reinstalled in-process -- which is strictly stronger than the
# old form: it fires at every thread count on both builds instead of depending
# on which truncation a given BLAS reduction order happens to break.
# ---------------------------------------------------------------------------


def _pre_branch_cut_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-round-1 body: the EXACT ``Re(r) == 0`` pin and the ``-r`` flip.

    An ``eig`` output never satisfies ``Re(sqrt(lam^2)) == 0`` -- its real part
    is the eigensolver's backward error, ~1e-16 -- so this pin fires only for
    the REGION modes, built in exact arithmetic, and never for a structured
    LAYER's.  Reinstalling it is how the tests below keep a fail-before now
    that the defect no longer occurs in the shipped tree.
    """
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


class _pre_branch_cut:
    """Install :func:`_pre_branch_cut_sqrt_decay` at every module binding of
    the shared selector, for the duration of a ``with`` block."""

    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.stack",
                "lumenairy.elements.pmm.twod")

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import importlib
        for name in self._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _pre_branch_cut_sqrt_decay
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def _thin_scan_uncached(pol):
    """The ladder, solved fresh -- never through ``_THIN_SCAN``, whose entries
    would otherwise carry an engineered arm's readings into every later test in
    the file."""
    return [_thin_solve(M, pol) for M in _THIN_LADDER]


#: The POST-fix closure envelope on the whole THIN ladder: <= 1.5e-15 over
#: 25 truncations x 2 polarizations x 2 builds.  The bar sits 6.8 decades above
#: it and 2.4 decades below the smallest PRE reading that manifests (3.5e-06,
#: 20 TE Windows), inside a gap the measurement leaves empty.
_X1_CLOSED_CLOSURE = 1e-8


@pytest.mark.parametrize("M,pol", [(19, "te"), (21, "te"), (12, "te"),
                                   (20, "te")])
def test_x1_is_closed_on_the_cell_it_was_pinned_at(M, pol):
    """X-1 IS CLOSED.  This pins the FIX, not the defect.

    Until 2026-09-11 this test asserted the opposite -- that the four cells
    still returned a wrong answer and that the census still flagged them -- and
    its own docstring asked for this restatement ("a future fix should make
    this test fail").  What closed it is the modal branch-cut fix; see the
    block comment above for the whole ladder on both builds and both arms.

    Three claims, each a decision rather than a reading:

      (a) the cell RETURNS -- nothing is refused, which is the regression guard
          the old form also carried;
      (b) it CLOSES energy to ``_X1_CLOSED_CLOSURE``, on a lossless cell where
          conservation is exact at any truncation under the Laurent rule.  The
          old form could not make this claim on any cell: 19 TE read
          ``R+T = 1.018`` on one build and RAISED on the other;
      (c) the census FLAGS NOTHING there -- the near-cancelling denominator the
          instrument exists to see is gone, not merely below a screen.

    The fail-before is the sibling test below, which reinstates the pre-fix
    branch body and re-reads the same ladder.
    """
    rows = _thin_scan(pol)
    hit = next(r for r in rows if r["M"] == M)
    assert hit["raised"] is None, (
        f"{M} {pol.upper()} raised {hit['raised']} on a cell that reads "
        f"|R+T-1| <= 1.5e-15 on both builds since the branch-cut fix.\n    "
        + _thin_table())
    assert not hit["refused"]
    assert hit["close"] < _X1_CLOSED_CLOSURE, (
        f"{M} {pol.upper()} closes at {hit['close']:.3e}: X-1 has REOPENED, "
        f"which on this cell means a propagating layer mode is carrying the "
        f"incoming root again (the groove index equals BOTH half-spaces').\n"
        f"    " + _thin_table())
    assert not hit["flagged"], (
        f"{M} {pol.upper()}: the census flags "
        f"{len(hit['flagged'])} near-singular inverse(s) (min equilibrated "
        f"rcond {min(c[2] for c in hit['flagged']):.3e}) on a cell that is "
        f"now well conditioned.\n    " + _thin_table())


def test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix():
    """The two-sided form of the claim above, on the LADDER rather than four
    cells -- because which truncation manifests was always a per-build fact
    (see the partition comment at the top of this file), and the thing that is
    now build-independent is that NONE of them does.

    POST: 0 raising cells, 0 flagged cells, worst closure <= 1.5e-15 over
    25 truncations x 2 polarizations on both builds, and every rung's
    ``sum(R)`` within 5.1e-02 of the converged 2.05e-04 (that worst case is
    the COARSEST rung, M = 6).

    PRE (engineered, this process): 7-8 raising cells and 14 flagged of 25 in
    TE, worst closure 3.196e-02 and a ``sum(R)`` 152.60x the converged value at
    M = 21 -- on both builds, at every thread count measured.
    """
    for pol in ("te", "tm"):
        rows = _thin_scan(pol)
        assert all(r["raised"] is None for r in rows), (
            f"{sum(1 for r in rows if r['raised'])} of {len(rows)} "
            f"{pol.upper()} truncations raise.\n    " + _thin_table((pol,)))
        worst = max(r["close"] for r in rows)
        assert worst < _X1_CLOSED_CLOSURE, (
            f"worst {pol.upper()} closure {worst:.3e} over the ladder.\n    "
            + _thin_table((pol,)))
        assert not any(r["flagged"] for r in rows), (
            f"{sum(1 for r in rows if r['flagged'])} {pol.upper()} cells are "
            f"flagged.\n    " + _thin_table((pol,)))

    # ---- the fail-before, ENGINEERED: put the pre-round-1 branch back.
    with _pre_branch_cut():
        pre = _thin_scan_uncached("te")
    n_bad = sum(1 for r in pre
                if r["raised"] is not None or r["close"] > 1e-6)
    n_flag = sum(1 for r in pre if r["flagged"])
    assert n_bad >= 5 and n_flag >= 5, (
        f"the pre-round-1 branch body does not reopen X-1 on this build "
        f"({n_bad} bad cells, {n_flag} flagged of {len(pre)}): the "
        f"fail-before has stopped demonstrating anything, so the POST claims "
        f"above are no longer two-sided")
    # ... and the answer really is WRONG there, not merely non-conserving.
    ref = _thin_converged_sumR("te")
    worst_ratio = max((abs(r["sumR"] / ref - 1.0) for r in pre
                       if np.isfinite(r["sumR"])), default=0.0)
    assert worst_ratio > 10.0, (
        f"the worst pre-fix sum(R) is only {worst_ratio:.2f}x from the "
        f"converged {ref:.6e}: the engineered arm is not reproducing the "
        f"152x error the defect is documented at")


'''

p = "tests/unit/test_m1_conditioning_guard.py"
s = io.open(p, encoding="utf-8").read()
start = s.index("# X-1 on the library's own documented instability class")
end = s.index("def test_thin_grating_clean_truncations_are_untouched():")
old = s[start:end]
s = s[:start] + NEW + s[end:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("replaced %d chars with %d" % (len(old), len(NEW)))

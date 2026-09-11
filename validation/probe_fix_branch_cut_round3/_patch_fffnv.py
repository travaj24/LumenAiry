"""ROUND 3: re-derive the fff_nv file's mode-match-degeneracy gate, as its own
docstring instructs once the solver becomes degeneracy-robust."""
import io

p = "tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py"
s = io.open(p, encoding="cp1252").read()

old = '''    MEASURED 2026-09-10, worst |sum R + sum T - 2| over n_orders 11..41, on
    Windows-1-thread / Windows-4-threads / WSL: clean 2.083e-13 / 1.821e-13 /
    1.861e-13 (16 of 16 truncations sound on every one) and coincident
    2.761e-02 / 2.309e-02 / 5.001e-02 (0 / 1 / 0 of 16 sound) -- a ratio of
    ~1.3e11 against the 1e5 asserted, i.e. six decades of margin, and the
    coincident worst sits 5 decades above the 1e-13 floor the ratio is taken
    from.  If the solver is ever made degeneracy-robust this test fails, and
    that is the gate working: it must then be re-derived (durability rule),
    not widened.
    """
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    ladder = range(11, 42, 2)

    def worst(eps_groove):
        eg = np.diag([eps_groove] * 3).astype(complex)
        out = 0.0
        for n in ladder:
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
            out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
        return out

    clean = worst(_STRIPE_EPS_GROOVE)
    assert clean < _ONED_SOUND_CLOSURE, (
        f"the reference fixture violates its own exact lossless closure by "
        f"{clean:.3e} somewhere in {ladder.start}..{ladder.stop - 1}")
    # the coincidence IS the thing being avoided, so prove it still bites.
    # Floor the ratio on 1e-13 (the float64 level a clean closure lives at) so
    # a lucky clean run cannot inflate the requirement.
    with pytest.warns(UserWarning, match="lossless energy closure violated"):
        degenerate = worst(_DEGENERATE_EPS_GROOVE)
    assert degenerate > 1e5 * max(clean, 1e-13), (
        f"the index-coincident cell closed to {degenerate:.3e} against the "
        f"clean {clean:.3e}: the mode-match degeneracy this fixture was moved "
        f"off no longer bites, so the move (and this test) must be re-derived")
'''

new = '''    RE-DERIVED 2026-09-11 (branch-cut ROUND 3).  The version of this test that
    ran until this date asserted the OPPOSITE of what follows: that the
    index-coincident groove still violated the 1-D lossless theorem by at least
    1e5 x the clean fixture's closure, and that a closure warning fired while it
    did.  Its own closing sentence anticipated this exactly -- "If the solver is
    ever made degeneracy-robust this test fails, and that is the gate working:
    it must then be re-derived (durability rule), not widened."  The modal
    branch-cut fix made it degeneracy-robust, so the gate fired and this is the
    re-derivation.

    What is asserted now, on the same two arms measured in the same run:

      (a) the CLEAN fixture still holds its own exact closure (unchanged);
      (b) the COINCIDENT groove now holds it TOO, and is indistinguishable from
          the clean one -- their ratio is O(1), not 1e11;
      (c) no closure warning fires on either;
      (d) the ENGINEERED pre-round-1 arm still reproduces the old behaviour, so
          the claim is two-sided and the reason the fixture was moved off 2.25
          in the first place is still on the record rather than merely asserted.

    Nothing is widened: (b) and (c) are TIGHTER than what they replace, and (d)
    keeps the original claim alive on the arm where it is still true.  The
    fixture itself stays at 2.10 -- the move is now belt-and-braces rather than
    load-bearing, and reverting it would be a separate decision with its own
    evidence.

    MEASURED 2026-09-11 over ``n_orders`` 11..41, on SIXTEEN configurations --
    (Windows py3.14.6/numpy 2.4.4, WSL py3.12.3/numpy 2.4.6) x
    (HASWELL, NEHALEM, KATMAI, SANDYBRIDGE) x (1, 4) BLAS threads, every
    ``OPENBLAS_CORETYPE`` confirmed through ``threadpoolctl``:

        arm                      worst |sum R + T - 2|   degen/clean   warned
        POST clean  (2.10)       1.52e-13 .. 5.36e-13          --       0/16
        POST degen  (2.25)       1.39e-13 .. 5.19e-13    0.72 .. 2.49   0/16
        PRE  degen  (2.25)       7.29e-04 .. 7.19e-02    3.1e+09 ..     11..16
                                                         2.7e+11        of 16

    The POST ratio bar of 10 sits 4x above the worst POST reading; the PRE bar
    of 1e5 sits 4.5 decades below the smallest PRE reading.  Neither population
    comes within four decades of the other on any of the sixteen.
    """
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    ladder = range(11, 42, 2)

    def worst(eps_groove):
        """(worst |sum R + sum T - 2| over the ladder, closure warnings)."""
        eg = np.diag([eps_groove] * 3).astype(complex)
        out, warned = 0.0, 0
        for n in ladder:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                _o, R1, T1, _J = rcwa_jones_1d_segments(
                    PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL,
                    theta=0.0, n_orders=n)
            warned += sum(1 for w in rec if "lossless energy closure violated"
                          in str(w.message))
            out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
        return out, warned

    clean, clean_warned = worst(_STRIPE_EPS_GROOVE)
    assert clean < _ONED_SOUND_CLOSURE, (
        f"the reference fixture violates its own exact lossless closure by "
        f"{clean:.3e} somewhere in {ladder.start}..{ladder.stop - 1}")
    assert clean_warned == 0
    # (b) + (c): the coincidence is CURED -- it closes like the clean fixture.
    degenerate, degen_warned = worst(_DEGENERATE_EPS_GROOVE)
    assert degenerate < _ONED_SOUND_CLOSURE, (
        f"the index-coincident cell closed to {degenerate:.3e}: the modal "
        f"branch cut has REOPENED and a propagating layer mode is carrying "
        f"the incoming root again")
    assert degenerate < 10.0 * max(clean, 1e-13), (
        f"the index-coincident cell closes at {degenerate:.3e} against the "
        f"clean {clean:.3e} -- a ratio of "
        f"{degenerate / max(clean, 1e-13):.3e}.  It is still sound, but it is "
        f"no longer INDISTINGUISHABLE from the clean fixture, which is what "
        f"round 3 established")
    assert degen_warned == 0, (
        f"{degen_warned} closure warnings fired on a cell that closes at "
        f"{degenerate:.3e}: the guard is warning about a right answer")
    # (d) the fail-before, ENGINEERED: the pre-round-1 branch body still bites.
    with _fffnv_pre_branch_cut():
        pre_clean, _ = worst(_STRIPE_EPS_GROOVE)
        with pytest.warns(UserWarning,
                          match="lossless energy closure violated"):
            pre_degen, pre_warned = worst(_DEGENERATE_EPS_GROOVE)
    assert pre_degen > 1e5 * max(pre_clean, 1e-13), (
        f"the pre-round-1 arm closes the index-coincident cell to "
        f"{pre_degen:.3e} against its clean {pre_clean:.3e}: the engineered "
        f"arm no longer reproduces the degeneracy this fixture was moved off, "
        f"so (b) is no longer two-sided")
    assert pre_warned >= 1
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

# ---- the engineered arm helper, placed just above the test
anchor = "def test_stripe_fixture_is_free_of_the_mode_match_degeneracy("
helper = '''class _fffnv_pre_branch_cut:
    """Reinstate the PRE-ROUND-1 modal branch body -- the EXACT ``Re(r) == 0``
    pin, which an ``eig`` output never satisfies, so it fires only for the
    REGION modes (built in exact arithmetic) and never for a structured
    LAYER's.  With it installed a propagating layer mode of an index-coincident
    groove keeps whichever root the last bit of ``Im(lam^2)`` chose, which is
    the defect the 2.10 fixture was chosen to avoid.  See
    ``docs/audits/FIX_BRANCH_CUT_ROUND3_2026_09_11.md``."""

    _MODULES = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
                "lumenairy.elements.berreman")

    @staticmethod
    def _body(x, xp=None, band=1e-8):
        from lumenairy.backend.array import array_namespace
        if xp is None:
            xp = array_namespace(x)
        x = xp.asarray(x).astype(complex)
        r = xp.sqrt(x)
        return xp.where((r.real == 0) & (r.imag < 0), -r, r)

    def __enter__(self):
        import importlib
        self._saved = []
        for name in self._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = self._body
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def test_stripe_fixture_is_free_of_the_mode_match_degeneracy('''
assert s.count(anchor) == 1, ("anchor", s.count(anchor))
s = s.replace(anchor, helper)

if "\nimport warnings" not in s and "\nimport numpy as np" in s:
    s = s.replace("\nimport numpy as np", "\nimport warnings\n\nimport numpy as np", 1)

io.open(p, "w", encoding="cp1252", newline="").write(s)
print("patched", p)

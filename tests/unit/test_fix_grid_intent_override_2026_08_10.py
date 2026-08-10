# Pins for the ALLOW-WITH-WARNINGS override on the record-run grid intent --
# docs/audits/FIX_CLAMP_RECAL_OVERRIDE_2026_08_10.md sec 3.
#
# WHAT CHANGED.  ``validation/repro_traced_carrier_121/_grid_intent.preflight``
# hard-refused (``SystemExit(2)``) whenever the modelled peak exceeded free
# memory.  The model is a deliberate UPPER-BOUND ENVELOPE over measured peaks
# (worst ratio ~1.3x), so "modelled > free" means "the envelope says it might
# not fit", not "it will not fit" -- and refusing an EXPLICIT intent on that
# basis is an envelope over-ruling an operator who has already read what the
# run costs.  A binding box check now emits a prominent ``RuntimeWarning``
# naming modelled need, free memory and the MEASURED reference peak, and
# PROCEEDS at the requested grid.
#
# WHAT DID NOT CHANGE, AND IS PINNED HERE BECAUSE IT IS THE WHOLE RISK:
#
#   * the CLAMP refusal.  If the clamp would bind, the leg runs at a SMALLER
#     grid under the label the runner configured -- the silent-8192 failure of
#     ADJUDICATION_NFC_8192_2026_08_10.md sec 2.1.  Warn-and-proceed must not
#     reopen that hole, so that arm is still a hard exit 2 whatever
#     ``on_box_budget`` says.  The run gets the grid it named or it refuses;
#     it never silently shrinks.
#   * the refusal for a run the box PHYSICALLY cannot allocate (modelled need
#     at or above TOTAL physical memory, not merely above what is free).
#   * the post-run degradation assertion, which is the shadow-``n_fine``
#     detector: it must still catch a clamp that fired, and the override's own
#     warning must not be mistaken for one (nor mask one).
#
# Every test below FAILS on the pre-change module: arms A/B/C of the
# fail-before probe read exit 2 / exit 2 / exit 2 with ZERO warnings, and this
# file requires exit 0 + one warning on arm A.
#
# Scale: everything is driven at ``n_fine_cap`` 512-16384 against SYNTHETIC box
# memory, so the whole module is arithmetic -- no chain, no fine grid, no
# allocation.  The box read is injected through the module's own
# ``_box_memory`` seam, which exists because free and TOTAL memory are two
# different thresholds here and a test that cannot set both cannot pin the
# difference between them.
#
# WHY THE FIXTURES COMPUTE THEIR OWN "FREE" FIGURE.  The clamp constants are
# re-measured every time the fine leg's array traffic moves (they moved twice
# in three days).  A fixture with a hard-coded free figure would silently stop
# BINDING when they next fall, and a test that no longer reaches the branch it
# names is worse than a failing one, so every binding fixture is derived from
# the model itself.
import importlib.util
import inspect
import io
import os
import sys
import warnings

import pytest

from lumenairy.propagators import carrier as C

_HERE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'validation',
    'repro_traced_carrier_121'))
_INTENT = os.path.join(_HERE, '_grid_intent.py')

if not os.path.exists(_INTENT):        # pragma: no cover - trimmed checkout
    pytest.skip('validation/ is not present in this checkout',
                allow_module_level=True)


def _load(env=None):
    """Import ``_grid_intent`` from the validation tree, freshly.

    Fresh, not cached: the module reads ``RAMRES`` / ``ONBOX`` at import time,
    so the env-default tests need their own instance.  Loaded by path rather
    than by name so this file does not depend on ``sys.path`` carrying the
    validation directory.
    """
    old = {k: os.environ.get(k) for k in ('RAMRES', 'ONBOX')}
    if env is not None:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    try:
        spec = importlib.util.spec_from_file_location(
            '_grid_intent_under_test', _INTENT)
        mod = importlib.util.module_from_spec(spec)
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


GI = _load({'ONBOX': None, 'RAMRES': None})
_NPX = 1024 * 1024


def _need_gb(n_fine_cap, workers=1, n_px=_NPX, n_out=0):
    """The pre-flight's own modelled total, in GB -- transcribed so a fixture
    can place the synthetic box relative to it instead of guessing."""
    per = C._fine_grid_peak_bytes(n_fine_cap, n_px)
    parent = ((16.0 * float(n_out) ** 2 + C._FINE_GRID_BASE_BYTES)
              if workers > 1 else 0.0)
    return (workers * per + parent) / 1e9


def _fly(gi, monkeypatch, avail_gb, total_gb, n_fine_cap=1024, ramb='inf',
         workers=1, n_px=_NPX, n_out=0, reserve_gb=8.0, capture=True, **kw):
    """Run ``preflight`` against a synthetic box.  Returns
    ``(exit_code, returned, transcript, warnings)``; ``exit_code`` is 0 when
    the call returned, else whatever the ``SystemExit`` carried.

    ``capture=False`` leaves ``showwarning`` ALONE (only the filters are
    forced), which is what the ``record_warnings`` tests need:
    ``catch_warnings(record=True)`` installs its own ``showwarning`` and would
    silently bypass the module's tee -- i.e. it would make the recorder look
    empty and the test pass for the wrong reason.
    """
    monkeypatch.setattr(gi, '_box_memory',
                        lambda: (float(avail_gb) * 1e9, float(total_gb) * 1e9))
    monkeypatch.setattr(gi, 'RESERVE_GB', float(reserve_gb))
    buf = io.StringIO()
    with warnings.catch_warnings(record=capture) as rec:
        warnings.simplefilter('always')
        try:
            got = gi.preflight(n_fine_cap, ramb, workers=workers, n_px=n_px,
                               n_out=n_out, label='pin', stream=buf, **kw)
            code = 0
        except SystemExit as exc:
            got, code = None, exc.code
    return code, got, buf.getvalue(), list(rec or ())


# ===========================================================================
# 1. The change itself: a binding box check on an explicit intent.
# ===========================================================================
def test_a_binding_box_check_warns_and_proceeds(monkeypatch):
    """FAIL-BEFORE arm A.  A modelled need above free memory once the reserve
    is taken, on a box with room to spare in total: the pre-change module
    exited 2 with no warning.  It must now return the ram_budget override,
    exit nothing, and warn exactly once."""
    need = _need_gb(1024)
    assert need > 4.0 - 8.0, 'the fixture must actually bind'
    assert need < 64.0, 'and must NOT be the physically-impossible arm'
    code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=4.0, total_gb=64.0)
    assert code == 0, txt
    assert got == float('inf'), (
        'the caller must still receive the ram_budget it asked for')
    assert len(rec) == 1 and issubclass(rec[0].category, RuntimeWarning), (
        [(_r.category, str(_r.message)[:80]) for _r in rec])


def test_the_warning_names_modelled_need_free_and_the_measured_reference(
        monkeypatch):
    """The three numbers, because a warning that does not carry them is a
    warning an operator cannot act on: what the model says the run needs,
    what the box says is free, and what a run of this shape has actually been
    MEASURED to peak at."""
    need = _need_gb(8192)
    free = need - 1.0
    code, _got, txt, rec = _fly(GI, monkeypatch, avail_gb=free, total_gb=137.4,
                                n_fine_cap=8192)
    assert code == 0, txt
    msg = ' '.join(str(rec[0].message).split())
    assert '%.1f GB' % need in msg, (need, msg)
    assert '%.1f GB free' % free in msg, (free, msg)
    assert '%.1f GB' % GI.MEASURED_PEAK_GB[8192] in msg, msg
    for frag in ('PROCEEDING', 'EXPLICIT', 'MEASURED', 'ONBOX=error'):
        assert frag in msg, (frag, msg)


def test_the_warning_quotes_the_grid_that_was_measured(monkeypatch):
    """The reference is the MEASURED peak at the grid asked for when there is
    one, and is honest about being a LOWER bound when there is not -- it never
    extrapolates a number it did not measure."""
    assert set(GI.MEASURED_PEAK_GB) == {4096, 8192, 16384}
    assert GI._measured_reference(8192) == (8192, GI.MEASURED_PEAK_GB[8192])
    assert GI._measured_reference(12288) == (8192, GI.MEASURED_PEAK_GB[8192])
    assert GI._measured_reference(1024) is None
    _c, _g, _t, rec = _fly(GI, monkeypatch, avail_gb=_need_gb(12288) - 1.0,
                           total_gb=137.4, n_fine_cap=12288)
    msg = ' '.join(str(rec[0].message).split())
    assert 'no peak has been MEASURED at n_fine=12288' in msg, msg
    assert 'LOWER bound' in msg, msg


def test_the_transcript_says_the_grid_is_not_degraded(monkeypatch):
    """The printed block is the half an operator reads in a log tail.  It has
    to say, in as many words, that proceeding did NOT buy the fit by shrinking
    the grid -- otherwise this looks exactly like the failure the module was
    written for."""
    code, _got, txt, _rec = _fly(GI, monkeypatch, avail_gb=4.0, total_gb=64.0)
    assert code == 0
    assert 'PROCEEDING UNDER WARNING -- pin' in txt, txt
    assert '~' * 74 in txt, txt
    assert 'UNCHANGED at 1024' in txt, txt
    assert 'ONBOX=error' in txt, txt


# ===========================================================================
# 2. What the override does NOT dispose of.
# ===========================================================================
def test_the_clamp_refusal_is_still_hard_under_warn(monkeypatch):
    """FAIL-BEFORE arm C, and the reason this change is not a regression.

    A binding CLAMP is not a capacity question: the leg would return a
    128x128 answer wearing a 1024 label.  ``on_box_budget`` must not reach
    it -- asserted at BOTH actions and at the default, since 'warn' becoming
    the default is exactly how such a hole would be opened."""
    for act in ('warn', 'error', None):
        code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=900.0,
                                   total_gb=1000.0, ramb='0.02',
                                   on_box_budget=act)
        assert code == 2, (act, txt)
        assert got is None
        assert not rec, (act, [str(r.message)[:60] for r in rec])
        assert 'would DEGRADE the fine grid from 1024 to 128' in txt, txt


def test_the_refusal_survives_when_the_model_exceeds_total_physical(
        monkeypatch):
    """FAIL-BEFORE arm B.  Freeing every byte on the machine cannot rescue a
    run modelled above TOTAL physical memory, so there is nothing for an
    operator to decide and no disposition applies."""
    assert _need_gb(1024) >= 2.0, 'fixture: the model must exceed a 2 GB box'
    for act in ('warn', 'error', None):
        code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=1.0, total_gb=2.0,
                                   on_box_budget=act)
        assert code == 2, (act, txt)
        assert got is None
        assert not rec
        assert 'PHYSICALLY cannot allocate' in txt, txt
        assert 'TOTAL physical memory' in txt, txt


def test_the_total_ram_refusal_is_what_keeps_k2_at_16384_refused(monkeypatch):
    """The concrete case the branch's documents turn on: two congruence
    workers at ``NFC=16384`` on this box (137.4 GB physical).  Before the
    override a binding FREE check refused it; now that the free check warns,
    the TOTAL check is the only thing left holding that line."""
    two = _need_gb(16384, workers=2, n_out=8192)
    one = _need_gb(16384)
    assert two > 137.4, (two, 'fixture assumes 2 workers exceed this box')
    code, _got, txt, rec = _fly(GI, monkeypatch, avail_gb=one - 1.0,
                                total_gb=137.4, n_fine_cap=16384, workers=2,
                                n_out=8192)
    assert code == 2, txt
    assert not rec
    assert 'PHYSICALLY cannot allocate' in txt, txt
    # ... and ONE worker at the same grid is the case that now proceeds
    code1, got1, _t1, rec1 = _fly(GI, monkeypatch, avail_gb=one - 1.0,
                                  total_gb=137.4, n_fine_cap=16384, workers=1)
    assert code1 == 0 and got1 == float('inf')
    assert len(rec1) == 1


def test_on_box_budget_error_restores_the_pre_change_refusal(monkeypatch):
    """The unattended-batch arm: same fixture as arm A, ``'error'``, exit 2."""
    code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=4.0, total_gb=64.0,
                               on_box_budget='error')
    assert code == 2, txt
    assert got is None
    assert not rec
    assert "on_box_budget='error'" in txt, txt
    assert 'ONBOX=warn' in txt, txt


# ===========================================================================
# 3. The on_* action pattern, spelled the way the library spells it.
# ===========================================================================
def test_the_default_action_is_warn_and_the_env_moves_it():
    """``'warn'`` is the shipped default for an explicit intent; ``ONBOX``
    moves it for a whole run without editing the runner."""
    assert GI.ON_BOX_BUDGET == 'warn'
    assert _load({'ONBOX': 'error'}).ON_BOX_BUDGET == 'error'
    assert _load({'ONBOX': ''}).ON_BOX_BUDGET == 'warn'
    assert _load({'ONBOX': None}).ON_BOX_BUDGET == 'warn'


def test_ignore_is_refused_by_vocabulary(monkeypatch):
    """The library's third action is deliberately NOT offered.  A silent
    over-commit is the failure this pre-flight exists to prevent, so the
    vocabulary is two words and the refusal says why -- before anything is
    printed, resolved or allocated."""
    assert GI._BOX_ACTIONS == ('warn', 'error')
    code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=4.0, total_gb=64.0,
                               on_box_budget='ignore')
    assert isinstance(code, str), code
    assert "'ignore' is deliberately not accepted" in code
    assert got is None and txt == '' and not rec
    code, _g, txt, _r = _fly(GI, monkeypatch, avail_gb=900.0, total_gb=1000.0,
                             on_box_budget='nonsense')
    assert isinstance(code, str) and 'nonsense' in code
    assert txt == ''


def test_the_action_routes_through_the_librarys_guard_dispose():
    """House style, pinned on the source: the warn arm is
    ``carrier._guard_dispose``, the same dispatcher every ``on_*`` knob in the
    library uses, not a bare ``warnings.warn`` that would drift from it."""
    body = inspect.getsource(GI.preflight)
    assert "_C._guard_dispose('warn'" in body, (
        'the override must dispose through the shared on_* dispatcher')
    assert 'on_box_budget' in inspect.signature(GI.preflight).parameters


# ===========================================================================
# 4. The shadow-``n_fine`` detector still catches a real degradation.
# ===========================================================================
def test_the_override_warning_is_not_a_degradation_mark(monkeypatch):
    """The override warning is teed into the same recorder the post-run check
    reads.  If its text carried the clamp's vocabulary, a run that proceeded
    correctly would fail its own post-run check with exit 3 ("the model and
    the clamp disagree") -- a false defect report on every warn-and-proceed
    run."""
    with GI.record_warnings() as rec:
        code, _got, _txt, _w = _fly(GI, monkeypatch, avail_gb=4.0,
                                    total_gb=64.0, capture=False)
    assert code == 0
    assert len(rec.records) == 1, rec.records
    assert rec.degradations() == [], rec.degradations()
    for mark in GI._DEGRADE_MARKS:
        assert mark not in rec.records[0]['msg'], mark
    buf = io.StringIO()
    GI.assert_no_grid_degradation(rec, 1024, label='pin', stream=buf)
    assert 'the leg ran at the 1024 it was asked for' in buf.getvalue()


def test_a_real_clamp_bind_still_trips_the_post_run_check(monkeypatch):
    """The shadow assertion itself, driven by the LIBRARY's own clamp rather
    than by a hand-typed message: a warn-and-proceed pre-flight followed by a
    leg that was in fact degraded must still exit 3.  This is the hole
    warn-and-proceed must not reopen."""
    buf = io.StringIO()
    with GI.record_warnings() as rec:
        code, _got, _txt, _w = _fly(GI, monkeypatch, avail_gb=4.0,
                                    total_gb=64.0, capture=False)
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            n = C._memory_bounded_n_fine(8192, 'pin',
                                         ram_budget=64 * 1024 ** 2)
    assert code == 0 and n == 256, (code, n)
    assert rec.degradations(), rec.records
    with pytest.raises(SystemExit) as exc:
        GI.assert_no_grid_degradation(rec, 8192, label='pin', stream=buf)
    assert int(exc.value.code) == 3
    assert 'the fine grid WAS degraded during the run' in buf.getvalue()


def test_proceeding_never_returns_a_smaller_grid(monkeypatch):
    """The contract in one line: ``preflight`` hands back a RAM BUDGET, never
    a grid, so there is no path by which warn-and-proceed can quietly hand the
    runner a smaller ``n_fine_cap``.  Pinned at three grids and at both
    budget forms an explicit intent can take."""
    for n in (512, 1024, 8192):
        code, got, txt, _rec = _fly(GI, monkeypatch, avail_gb=4.0,
                                    total_gb=137.4, n_fine_cap=n)
        assert code == 0 and got == float('inf'), (n, txt)
        pin = _need_gb(n) * 4.0
        code, got, txt, _rec = _fly(GI, monkeypatch, avail_gb=4.0,
                                    total_gb=137.4, n_fine_cap=n,
                                    ramb='%.6f' % pin)
        assert code == 0, (n, txt)
        # (equal to the precision the env string carries, which is the
        # contract: what goes in as GB comes out as bytes, unreduced)
        assert got == pytest.approx(pin * 1e9, rel=1e-6), n


# ===========================================================================
# 5. The loud shapes are unchanged where they survive.
# ===========================================================================
def test_the_refusal_banner_is_intact_on_a_scaled_config(monkeypatch):
    """Both surviving refusals still print the full 74-column banner, the
    label, the reasons and the remedy list -- the shape ``run_focus.py``'s
    transcripts are read by."""
    for kw, mark in (({'avail_gb': 1.0, 'total_gb': 2.0}, 'PHYSICALLY'),
                     ({'avail_gb': 4.0, 'total_gb': 64.0,
                       'on_box_budget': 'error'}, "on_box_budget='error'"),
                     ({'avail_gb': 900.0, 'total_gb': 1000.0,
                       'ramb': '0.02'}, 'would DEGRADE')):
        code, _got, txt, _rec = _fly(GI, monkeypatch, **kw)
        assert code == 2, (kw, txt)
        assert txt.count('!' * 74) == 3, txt
        assert 'REFUSED -- pin' in txt, txt
        assert '  Remedies:' in txt, txt
        assert mark in txt, (mark, txt)


def test_an_approved_run_still_says_so_and_warns_about_nothing(monkeypatch):
    """The fits-outright path is untouched: same verdict line, no warning,
    no banner of either kind."""
    code, got, txt, rec = _fly(GI, monkeypatch,
                               avail_gb=_need_gb(1024) + 20.0, total_gb=137.4,
                               n_fine_cap=1024)
    assert code == 0 and got == float('inf')
    assert not rec
    assert 'VERDICT: the clamp cannot bind and the box can hold it.' in txt
    assert 'PROCEEDING UNDER WARNING' not in txt
    assert '!' * 74 not in txt


def test_the_paraxial_intent_is_untouched(monkeypatch):
    """``final_leg='paraxial'`` builds no fine grid, so neither check runs and
    neither disposition applies -- before or after."""
    code, got, txt, rec = _fly(GI, monkeypatch, avail_gb=0.5, total_gb=1.0,
                               paraxial=True)
    assert code == 0 and got == float('inf')
    assert not rec
    assert 'no fine grid is built' in txt, txt

"""VERIFY-WP-C4 ROUND 2 -- the gaps this re-verification closed.

Round 2 of WP-C4 (`feat/c4-mft-direct-round2`) added a SECOND, build-free
condition to ``_auto_selects_direct`` and a ``mft_method=`` way back at every
public entry point the shape rule moves.  Re-measuring it on both builds
confirmed the safety claim and found five smaller things nothing gated.  Each
id below is one of them, and each was FAILED BEFORE it was written -- the
measurement that showed the gap is in the docstring.

Report: ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
VERIFY_WP-C4_ROUND2.md``.  Probes and JSON:
``validation/probe_verify_c4_round2/``.

WHAT IS HERE

1.  THE WORK SCREEN'S COMPARISON IS INCLUSIVE.  ``_auto_selects_direct``
    compares ``flops >= w * entries`` and the constant is set to EXACTLY the
    work ratio of the shape the derivation names as the smallest measured safe
    (``256x64 -> 4x1`` reads exactly 16.00).  A ``>`` would refuse that shape.
    MEASURED: of nine mutations put to the shipped claims on both builds, this
    is the one NOTHING refused
    (``validation/probe_verify_c4_round2/vc4b_mutations_{win,wsl}.json``).
2.  THE WORK SCREEN COUNTS THE ARITHMETIC THE ROUTE ACTUALLY DOES.  The
    screen's ``entries`` and ``flops`` are asserted against a LIVE call --
    the elements ``np.exp`` is handed and the operand shapes ``xp.matmul``
    receives -- and not against a second copy of the same formula.  Nothing
    else ties the screen to ``_direct_matrix_2d``'s association rule, so a
    change to that rule would leave the screen counting a cost the route no
    longer pays.
3.  THE CROSS-BUILD MEMORY GUARD CAN FAIL.  ``test_c4_round2_memory_claim.py``
    refuses the string ``'readings are identical to the byte across builds'``.
    MEASURED: that string is absent from the report and the CHANGELOG at
    ``57f71923`` (before the correction), at ``bedcabe8`` and at the branch
    tip, and the sentence it was written to refuse reads "the two builds'
    readings are IDENTICAL TO THE BYTE at every shape" -- so the guard passes
    unchanged on the exact text it names.  The predicate here is exercised
    against that sentence as a literal, which is the fail-before the guard
    has not got.
4.  THE REFUSAL IS TWO-SIDED, AND ITS SILENT DOORS ARE A CLOSED SET.
    ``mft_method=`` is a ``ValueError`` at three entry points where no
    transform is reached and is ACCEPTED where one is; at two further entry
    points it is accepted and ignored.  Both halves are pinned, so a fourth
    silent door cannot appear unnoticed.
5.  THE COLLINS LEG CANNOT FIRE THE RULE, MEASURED.  The report argues from
    the signature that ``propagate_carrier_referenced(transport='collins')``
    keeps the input's ``N`` and therefore never reaches a captured shape.
    Here the rule is SPIED on during that call and every answer must be
    ``False`` -- and the keyword must still arrive, or the entry point would
    be carrying a way back to nowhere.
6.  THE GUIDE'S WAY-BACK TABLE COVERS EVERY KEYWORD-BEARING ENTRY POINT, and
    its SPELLING column matches the archive-to-archive evidence.  Six of the
    nine signatures carrying ``mft_method=`` do not document it in their own
    docstrings (``propagate``, ``propagate_carrier_referenced``, both traced
    chains and both carrier readouts), so the Guide is the only place a
    caller can learn the spelling; that makes the table load-bearing.

Author:  Andrew Traverso
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import warnings

import numpy as np
import pytest

import lumenairy
from lumenairy.propagators import _bluestein as B

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_PROBE = os.path.join(_REPO, 'validation', 'probe_verify_c4_round2')
_GUIDE = os.path.join(_REPO, 'Migration-Guide.md')

WL = 1.31e-6


def _gauss(n, dx, w):
    g = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)).astype(
        np.complex128)


def _work(ny, nx, my, mx):
    """Multiply-adds per transcendental kernel entry, as the rule counts."""
    return (min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
            / (my * ny + mx * nx))


# ===========================================================================
# 1.  The work screen's comparison is inclusive at the constant
# ===========================================================================

def _a_thin_shape_the_ratio_admits():
    """A THIN shape the FIRST condition admits, found by scan.

    Thin, because the work screen only ever decides a thin shape: at the
    boundary ratio a SQUARE shape reads ``(N + M)/2``, which clears any
    sensible constant.  Nothing here types a number -- the scan is driven by
    the boundary constant, so a retune keeps the id.
    """
    r = float(B._MFT_DIRECT_MAX_RATIO)
    if not (0.0 < r < 1.0):
        return None
    n_at = int(round(1.0 / r))
    for p in range(0, 9):
        for q in range(0, 9):
            ny, nx = n_at * (1 << p), n_at * (1 << q)
            for my in (1, 2, 4, 8):
                for mx in (1, 2, 4, 8):
                    if my > ny or mx > nx:
                        continue
                    if max(my / ny, mx / nx) > r:
                        continue
                    if _work(ny, nx, my, mx) > 1.0:
                        return (ny, nx, my, mx)
    return None


def test_the_work_screens_comparison_is_inclusive_at_the_constant():
    """``flops >= w * entries``, not ``>``.

    WHY IT MATTERS.  The constant is not a round number chosen for its looks:
    round 2 set it to the work ratio of the SMALLEST shape it measured safe on
    both builds (``256x64 -> 4x1``, which reads exactly 16.00), so that shape
    sits exactly ON the boundary and a strict comparison would refuse the one
    shape the derivation rests on.  ``_auto_selects_direct``'s Notes state the
    inclusive comparison; nothing tested it.  MEASURED 2026-09-20 on both
    builds: a ``>`` mutant passed all eight shipped claims of
    ``test_c4_mft_direct_default.py`` and ``test_c4_round2_mft_method.py``
    (``validation/probe_verify_c4_round2/vc4b_mutations_{win,wsl}.json``).

    The claim is about the CONSTANT and not about one shape: set the constant
    to a shape's own work ratio and that shape must be INSIDE; move the
    constant up by one ULP and it must be OUTSIDE.  Two-sided, and independent
    of what the shipped constant happens to be.
    """
    shape = _a_thin_shape_the_ratio_admits()
    assert shape is not None, (
        "no thin shape at the boundary ratio survives the scan; the first "
        "condition no longer admits an anisotropic shape and this id needs "
        "re-deriving")
    ny, nx, my, mx = shape
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    w0 = flops / entries
    saved = B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY
    try:
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = w0
        assert B._auto_selects_direct(ny, nx, my, mx), (
            f"with the work constant set to {shape}'s own work ratio "
            f"({w0!r}) the rule refuses it.  The comparison has become "
            f"strict -- and the shipped constant is set to exactly the work "
            f"ratio of the smallest shape round 2 measured safe, so a strict "
            f"comparison drops the shape the whole derivation names")
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = float(
            np.nextafter(w0, np.inf))
        assert not B._auto_selects_direct(ny, nx, my, mx), (
            f"{shape} is still captured with the work constant one ULP above "
            f"its own work ratio; the rule is not reading the constant, so "
            f"the assertion above proves nothing")
    finally:
        B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY = saved

    # ... and the SHIPPED constant is itself a real shape's work ratio, which
    # is what makes the inclusive comparison load-bearing rather than
    # cosmetic.  Skipped silently if a retune has moved it off every shape
    # here -- the claim above does not depend on it.
    w = float(saved)
    for s2 in ((256, 64, 4, 1), (64, 256, 1, 4),
               (256, 128, 8, 1), (128, 256, 1, 8)):
        if _work(*s2) == w:
            assert B._auto_selects_direct(*s2), (
                f"{s2} reads exactly the shipped work constant {w!r} and is "
                f"refused; round 2 measured that shape SAFE on both builds "
                f"and set the constant to its reading")


# ===========================================================================
# 2.  The screen counts the arithmetic the route actually does
# ===========================================================================

_COUNTED = [(256, 256, 8, 8), (2048, 64, 64, 2), (64, 2048, 2, 64),
            (256, 64, 4, 1), (64, 256, 1, 4), (1024, 128, 32, 4)]


@pytest.mark.parametrize('shape', _COUNTED)
def test_the_work_screen_counts_the_arithmetic_the_dense_route_performs(shape):
    """``entries`` and ``flops`` measured on a LIVE call, not re-derived.

    ``_auto_selects_direct`` scores a shape with
    ``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds over
    ``My*Ny + Mx*Nx`` kernel entries, and the comment says those are the two
    costs ``_direct_matrix_2d`` itself compares.  Nothing checked that against
    the function.  Here ``np.exp`` and ``np.matmul`` are wrapped for one call:
    the elements handed to ``exp`` must equal ``entries`` and the multiply-adds
    implied by the operand shapes ``matmul`` receives must equal ``flops`` --
    including WHICH association order is taken, which differs between the two
    orientations and is what the ``min`` is about.

    MEASURED 2026-09-20 on both builds at 18 shapes: exact at all of them
    (``validation/probe_verify_c4_round2/vc4b_formula_{win,wsl}.json``).
    """
    ny, nx, my, mx = shape
    E = np.zeros((ny, nx), dtype=np.complex128)
    alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2

    seen_exp = [0]
    seen_matmul = []
    real_exp, real_matmul = np.exp, np.matmul

    def counted_exp(x, *a, **k):
        seen_exp[0] += int(np.asarray(x).size)
        return real_exp(x, *a, **k)

    def counted_matmul(a1, a2, *a, **k):
        seen_matmul.append((tuple(np.shape(a1)), tuple(np.shape(a2))))
        return real_matmul(a1, a2, *a, **k)

    np.exp, np.matmul = counted_exp, counted_matmul
    try:
        B._direct_matrix_2d(E, alpha, alpha, my, mx, sign=-1, xp=np)
    finally:
        np.exp, np.matmul = real_exp, real_matmul

    entries = my * ny + mx * nx
    cy = my * ny * nx + my * nx * mx
    cx = ny * nx * mx + my * ny * mx
    observed = sum(a[0] * a[1] * b[1] for a, b in seen_matmul)
    assert seen_exp[0] == entries, (
        f"{shape}: the dense route built {seen_exp[0]} transcendental kernel "
        f"entries and the work screen counts {entries}.  The screen's "
        f"denominator is no longer the number of exponentials the route pays "
        f"for, so its constant is calibrated against something the route does "
        f"not do")
    assert observed == min(cy, cx), (
        f"{shape}: the dense route issued {observed} multiply-adds "
        f"(operands {seen_matmul}) and the work screen counts "
        f"min({cy}, {cx}) = {min(cy, cx)}.  Either the association rule in "
        f"_direct_matrix_2d has changed or the screen's numerator has; the "
        f"two have to be the same arithmetic or the constant means nothing")
    taken = 'y_first' if seen_matmul and seen_matmul[0][0] == (my, ny) \
        else 'x_first'
    predicted = 'y_first' if cy <= cx else 'x_first'
    assert taken == predicted, (
        f"{shape}: the route associated {taken} where the two costs predict "
        f"{predicted}; the screen's ``min`` is not the order the route takes")


# ===========================================================================
# 3.  The cross-build memory guard can fail
# ===========================================================================

#: The sentence the round-2 correction was written to remove, as it stood at
#: ``57f71923`` -- the newline is where the report wrapped it.
_REFUTED_SENTENCE = (
    "builds, by 6.4x to 334.9x, and the two builds' readings are IDENTICAL "
    "TO THE\nBYTE at every shape** (the `identical WIN/WSL` column is `yes` "
    "at 42 of 42)."
)


#: Words that mark an occurrence as a QUOTATION of a corrected wording rather
#: than an assertion.  Both documents rightly quote the old sentence where they
#: record the correction, and a guard that could not tell the two apart would
#: have to be either vacuous or permanently red.
_CORRECTION_MARKERS = ('earlier wording', 'contradicted', 'corrected',
                       'an earlier', 'refuted', 'said "', 'said “')


def _claims_the_readings_are_identical(text):
    """Does this prose ASSERT that the two builds' memory readings agree?

    Whitespace-insensitive, because the sentence it is looking for wrapped
    across a line break in the document it came from -- which is why a plain
    substring search for a one-line paraphrase could never have matched it.
    An occurrence within 220 characters after a correction marker is a
    quotation and does not count.
    """
    flat = ' '.join(text.lower().split())
    needles = ('readings are identical to the byte',
               'readings are identical across builds')
    for needle in needles:
        at = flat.find(needle)
        while at >= 0:
            back = flat[max(0, at - 220):at]
            if not any(m in back for m in _CORRECTION_MARKERS):
                return True
            at = flat.find(needle, at + 1)
    return False


@pytest.mark.parametrize('doc', ('report', 'changelog'))
def test_the_cross_build_memory_guard_would_have_refused_the_old_sentence(doc):
    """The fail-before ``test_c4_round2_memory_claim.py``'s prose id has not
    got.

    THE GAP.  That id asserts ``'readings are identical to the byte across
    builds' not in text.lower()``.  MEASURED 2026-09-20 by reading the
    committed documents out of git: that string is absent at ``57f71923``
    (before the correction), at ``bedcabe8`` and at the tip, in BOTH
    documents -- so the id passed, unchanged, on the exact prose it was
    written to refuse.  The sentence actually printed was "the two builds'
    readings are IDENTICAL TO THE BYTE at every shape", wrapped across a line
    break.

    This id carries the predicate AND the sentence, so the predicate is shown
    to reject the real text before it is applied to the current one.
    """
    assert _claims_the_readings_are_identical(_REFUTED_SENTENCE), (
        "the predicate does not recognise the sentence it was written for; "
        "it would pass on the defect the same way the id it replaces did")
    path = (os.path.join(
        _REPO, 'docs', 'audits', 'AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11',
        'fixes', 'WP-C4_MFT_DIRECT_DEFAULT_REPORT.md')
        if doc == 'report' else os.path.join(_REPO, 'CHANGELOG.md'))
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    # The round-2 addendum QUOTES the refuted sentence where it records the
    # correction, which is the right thing for it to do, so the claim is made
    # about the report's BODY -- everything above that addendum's heading.
    cut = text.find('# Round 2 (VERIFY-WP-C4)')
    body = text[:cut] if cut > 0 else text
    assert not _claims_the_readings_are_identical(body), (
        f"{os.path.basename(path)} still asserts that the two builds' "
        f"tracemalloc readings agree byte for byte.  Re-counted from the "
        f"committed JSON 2026-09-20: 0 of 42 on the branch ladder and 0 of 51 "
        f"on round 2's.  What IS build-free is the ORDERING (42 of 42 and "
        f"51 of 51)")


# ===========================================================================
# 4.  The refusal is two-sided, and its silent doors are a closed set
# ===========================================================================

def _refuses(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn()
        return None
    except ValueError as exc:
        return str(exc)


def test_the_mft_method_refusal_is_two_sided_at_the_three_doors_that_have_one():
    """Accepted where a transform IS reached, refused where it is not.

    A one-sided reading of either half is worthless: a signature that refused
    everything would pass "it refuses", and one that accepted everything would
    pass "it accepts".  Both are asserted at each of the three doors, and the
    refusal has to NAME the keyword -- an unrelated ``ValueError`` from
    somewhere further in would otherwise read as the contract working.
    """
    from lumenairy.propagators import carrier as C
    dx = 2e-6
    E = _gauss(256, dx, 256 * dx / 6.0)

    # compute_psf: the MFT sampler reaches one, the FFT sampler does not.
    assert _refuses(lambda: lumenairy.compute_psf(
        E, WL, 50e-3, dx, N_psf=8, method='mft', dx_psf=1e-6,
        mft_method='bluestein')) is None, (
        "compute_psf(method='mft') refuses mft_method=, where the transform "
        "IS reached")
    msg = _refuses(lambda: lumenairy.compute_psf(
        E, WL, 50e-3, dx, N_psf=256, method='fft', mft_method='bluestein'))
    assert msg and 'mft_method' in msg, (
        f"compute_psf(method='fft') did not refuse mft_method= by name; it "
        f"reaches no matrix Fourier transform, so the keyword would be "
        f"dropped on a caller asking for the pre-shape-rule bytes (got "
        f"{msg!r})")

    # propagate: only the output-grid MFT legs reach one.
    assert _refuses(lambda: lumenairy.propagate(
        E, z=2e-3, wavelength=WL, dx=dx, method='asm',
        output_grid=(8, 0.4e-6), mft_method='bluestein')) is None
    msg = _refuses(lambda: lumenairy.propagate(
        E, z=2e-3, wavelength=WL, dx=dx, method='asm',
        mft_method='bluestein'))
    assert msg and 'mft_method' in msg, (
        f"propagate(method='asm') with no output grid did not refuse "
        f"mft_method= by name (got {msg!r})")

    # propagate_carrier_referenced: only the Collins transport reaches one.
    env = _gauss(256, 0.4e-6, 25e-6)
    assert _refuses(lambda: C.propagate_carrier_referenced(
        env, -6.0e-4, 3.0e-4, WL, 0.4e-6, transport='collins',
        on_collins_sampling='ignore', mft_method='bluestein')) is None
    msg = _refuses(lambda: C.propagate_carrier_referenced(
        env, -6.0e-4, 3.0e-4, WL, 0.4e-6, transport='sziklas',
        mft_method='bluestein'))
    assert msg and 'mft_method' in msg, (
        f"propagate_carrier_referenced(transport='sziklas') did not refuse "
        f"mft_method= by name (got {msg!r})")


def test_the_entry_points_that_accept_and_ignore_the_keyword_are_the_known_one():
    """The other half of the contract, pinned so a second silent door is loud.

    RESTATED in WP-C4 round 3: ``resample_field(method='spline')`` now REFUSES
    the keyword like the other three doors (V-R2-3 closed), so the known
    accept-and-ignore set is the traced chain without a ``focus_readout`` --
    documented in its docstring, and inert there because on the Sziklas
    transport no transform is reached.

    MEASURED 2026-09-20 on both builds
    (``validation/probe_verify_c4_round2/vc4b_refusal_{win,wsl}.json``): two
    entry points accept ``mft_method=`` in a state where no transform is
    reached and ignore it -- ``resample_field(method='spline')``, which
    DOCUMENTS the ignore, and ``propagate_traced_carrier_chain`` with no
    ``focus_readout``, which does not.  That is a contract with a hole in it
    rather than two contracts, and this id fixes the hole's size: if one of
    them starts refusing, or a third appears, the set changes and the id says
    which way.
    """
    dx = 2e-6
    E = _gauss(256, dx, 256 * dx / 6.0)
    msg = _refuses(lambda: lumenairy.resample_field(
        E, dx, dx * 256 / 8.0, N_out=8, method='spline',
        mft_method='bluestein'))
    assert msg is not None and 'mft_method' in msg and 'spline' in msg, (
        "resample_field(method='spline') accepted mft_method= and dropped it "
        "(V-R2-3): the spline leg reaches no matrix Fourier transform, so the "
        "keyword must be refused there with a message naming it, as the other "
        "three doors do", msg)
    # Two-sided: the same call on the chirp-Z leg ACCEPTS the keyword.
    assert _refuses(lambda: lumenairy.resample_field(
        E, dx, dx * 256 / 8.0, N_out=8, method='chirpz',
        mft_method='bluestein')) is None
    doc = lumenairy.resample_field.__doc__ or ''
    assert 'ValueError' in doc and 'spline' in doc, (
        "resample_field's docstring no longer says the spline leg refuses "
        "mft_method=")


# ===========================================================================
# 5.  The Collins leg cannot fire the rule -- measured, not argued
# ===========================================================================

def test_the_collins_carrier_leg_cannot_fire_the_shape_rule(monkeypatch):
    """``propagate_carrier_referenced(transport='collins')`` carries
    ``mft_method=`` although the report argues, from the signature, that the
    rule can never fire there.

    An argument from a signature is not a measurement, and the keyword was
    added on the strength of it.  Here the rule is wrapped for the duration of
    one Collins call: it must be ASKED at least once (or the entry point
    reaches no transform at all and the keyword is a way back to nowhere), it
    must answer ``False`` every time, and naming a route must still change the
    bytes -- which is what makes the keyword worth having when the rule's
    default changes.
    """
    from lumenairy.propagators import carrier as C
    original = B._auto_selects_direct
    seen = []

    def spy(ny, nx, my, mx):
        ans = original(ny, nx, my, mx)
        seen.append(((int(ny), int(nx), int(my), int(mx)), bool(ans)))
        return ans

    import sys
    for mod in list(sys.modules.values()):
        if not getattr(mod, '__name__', '').startswith('lumenairy'):
            continue
        for attr in list(vars(mod)) if hasattr(mod, '__dict__') else ():
            try:
                if getattr(mod, attr) is original:
                    monkeypatch.setattr(mod, attr, spy)
            except Exception:                                # noqa: BLE001
                continue

    dx = 0.4e-6
    env = _gauss(256, dx, 25e-6)

    def run(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = C.propagate_carrier_referenced(
                env, -6.0e-4, 3.0e-4, WL, dx, transport='collins',
                on_collins_sampling='ignore', **kw)
        for attr in ('env', 'envelope', 'field', 'E'):
            if hasattr(out, attr):
                out = getattr(out, attr)
                break
        return np.ascontiguousarray(np.asarray(out)).tobytes()

    plain = run()
    assert seen, (
        "the Collins transport never asked the shape rule, so it reaches no "
        "MFT primitive at all on this path and mft_method= has nothing to "
        "name; either the transport changed or the keyword is in the wrong "
        "place")
    fired = [s for s, ans in seen if ans]
    assert fired == [], (
        f"the Collins transport reached a CAPTURED shape at {fired}.  The "
        f"report's blast radius says it cannot -- its output lattice keeps "
        f"the input's N, so both ratios are 1 -- and the Migration Guide "
        f"tells callers no way back is needed there.  Re-measure both")
    named = run(mft_method='bluestein')
    assert named != plain, (
        "mft_method='bluestein' did not change the Collins leg's bytes, so "
        "the keyword is accepted and dropped between this signature and the "
        "primitive; the way back this entry point advertises is not one")


# ===========================================================================
# 6.  The Guide's way-back table is load-bearing, and it is right
# ===========================================================================

def _guide_section():
    with open(_GUIDE, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    # The 5.49.0 release consolidates the eight default flips under ONE
    # `## 5.49.0` header; this work package is the `###` subsection below it.
    start = text.find("### `method='auto'` selects the direct-matrix")
    assert start >= 0, (
        "the direct-matrix MFT subsection is gone from Migration-Guide.md's "
        "5.49.0 section")
    ends = [e for e in (text.find('\n### ', start + 10),
                        text.find('\n## ', start + 10)) if e > 0]
    return text[start:min(ends) if ends else len(text)]


def test_the_guide_names_every_entry_point_that_carries_mft_method():
    """Six of the nine signatures carrying ``mft_method=`` do not document it
    in their own docstrings, so the Guide is where a caller learns of it.

    MEASURED 2026-09-20: ``propagate``, ``propagate_carrier_referenced``,
    ``propagate_traced_carrier_chain``, ``propagate_traced_carrier_chain_multi``,
    ``carrier_referenced_focus_readout`` and
    ``carrier_referenced_exact_focus_readout`` accept the keyword and say
    nothing about it in ``help()``.  That is a documentation defect in its own
    right; while it stands, this id keeps the one place that does describe the
    keyword complete.
    """
    section = _guide_section()
    from lumenairy.propagators import carrier as C
    from lumenairy.propagators.carrier_field import re_reference
    carriers = [lumenairy.compute_psf, lumenairy.resample_field,
                lumenairy.propagate, re_reference,
                C.propagate_carrier_referenced,
                C.propagate_traced_carrier_chain,
                C.propagate_traced_carrier_chain_multi,
                C.carrier_referenced_focus_readout,
                C.carrier_referenced_exact_focus_readout]
    def named(fn):
        # The Guide's table abbreviates the second traced-chain row as
        # ``propagate_traced_carrier_chain`` / ``..._multi``, which is a row
        # for both functions rather than a missing one.
        return (fn.__name__ in section
                or (fn.__name__.endswith('_multi') and '..._multi' in section))

    missing = [fn.__name__ for fn in carriers
               if 'mft_method' in inspect.signature(fn).parameters
               and not named(fn)]
    assert missing == [], (
        f"{missing} accept mft_method= and are not named in the Migration "
        f"Guide's 5.49.0 way-back section.  Most of them do not document the "
        f"keyword in their own docstring either, so a caller has nowhere to "
        f"learn which spelling reproduces their previous bytes")


@pytest.mark.parametrize('build', ('win', 'wsl'))
def test_the_guides_spelling_column_matches_the_measured_evidence(build):
    """The spelling is a MEASUREMENT, and the Guide has to carry the one that
    was measured.

    ``vc4b_entry_compare_{win,wsl}.json`` records, archive-to-archive against
    ``49ddf4bd`` at ``512 -> 16`` captured and ``512 -> 32`` refused, which
    ``mft_method`` value reproduces the base bytes at each entry point.  Three
    read ``'separable'`` -- the three callers that pass the separable flag into
    the primitive -- and the rest read ``'bluestein'``.  This id refuses a
    Guide that names the other one.
    """
    path = os.path.join(_PROBE, f"vc4b_entry_compare_{build}.json")
    assert os.path.exists(path), f"{path} is not committed"
    with open(path, encoding='cp1252') as fh:
        rows = json.load(fh)['rows']
    section = _guide_section()
    separable = sorted(r['entry_point'] for r in rows
                       if r['spelling_reproducing_base'] == ['separable'])
    assert separable, "no entry point measured 'separable'; the evidence is " \
                      "not discriminating and this id proves nothing"
    for line in section.splitlines():
        if not line.startswith('|'):
            continue
        for r in rows:
            name = r['entry_point']
            base = name.replace('_collins', '').replace('_sziklas', '')
            if f"`{base}" not in line:
                continue
            if not r['spelling_reproducing_base']:
                continue
            want = r['spelling_reproducing_base'][0]
            other = 'bluestein' if want == 'separable' else 'separable'
            if f"'{want}'" in line:
                continue
            assert f"'{other}'" not in line or f"'{want}'" in line, (
                f"{build}: the Guide's row for {base} names '{other}' and "
                f"the archive-to-archive measurement says {want!r} "
                f"reproduces the pre-5.49.0 bytes there.  Line: {line!r}")


def test_the_measured_spelling_split_is_the_one_the_report_states():
    """Exactly three entry points read ``'separable'``, and they are the three
    that pass the separable flag into the primitive.

    Asserted from the committed evidence on BOTH builds rather than from one,
    because "which arm the caller was on" is a property of the source and must
    not depend on the box.
    """
    per_build = {}
    for build in ('win', 'wsl'):
        with open(os.path.join(_PROBE, f"vc4b_entry_compare_{build}.json"),
                  encoding='cp1252') as fh:
            rows = json.load(fh)['rows']
        per_build[build] = {r['entry_point']:
                            tuple(r['spelling_reproducing_base'])
                            for r in rows}
    assert per_build['win'] == per_build['wsl'], (
        f"the two builds disagree about which spelling reproduces the "
        f"pre-5.49.0 bytes: "
        f"{ {k: (per_build['win'][k], per_build['wsl'][k]) for k in per_build['win'] if per_build['win'][k] != per_build['wsl'][k]} }"
        f".  The arm a caller was on is a property of the source")
    sep = sorted(k for k, v in per_build['win'].items() if v == ('separable',))
    assert sep == ['carrier_referenced_exact_focus_readout',
                   'propagate_carrier_referenced_collins',
                   'propagate_traced_carrier_chain_collins',
                   're_reference'], (
        f"the 'separable' set is {sep}; it is supposed to be exactly the "
        f"callers that pass the separable flag into the primitive.  If a "
        f"caller has changed arm, the Migration Guide's spelling column and "
        f"the report's way-back table both need re-measuring")
    # ... and the source says why: those are the callers wired to the
    # separable flag, which is a fact about the tree and not about the run.
    src = inspect.getsource(
        __import__('lumenairy.propagators.carrier', fromlist=['x']))
    assert '_EXACT_READOUT_SEPARABLE_BLUESTEIN' in src, (
        "carrier.py no longer names the separable-arm flag the 'separable' "
        "spelling was measured from")


# ===========================================================================
# 7.  The AST census agrees with a walk taken the other way round
# ===========================================================================

def test_the_census_count_survives_walking_the_call_graph_forwards():
    """``test_c4_round2_mft_method.py`` walks BACKWARDS from the two
    primitives and stops at the first public function.  A walk FORWARDS from
    each exported function has to reach the same set, or one of the two has a
    resolver bug -- and both are module-qualified for the same reason (a bare
    name joins unrelated subpackages).

    MEASURED 2026-09-20: 13 exported entry points, the same thirteen, from
    both directions (``validation/probe_verify_c4_round2/
    vc4b_census_win.json``).  Kept under 60 s by parsing the package once.
    """
    from tests.unit.test_c4_round2_mft_method import (
        _parse_package, _reaching_functions)
    mods = _parse_package()
    backwards = {name for _mod, name in _reaching_functions(mods)}
    root = os.path.dirname(os.path.abspath(lumenairy.__file__))

    # forwards: a public function reaches a primitive if some call chain
    # through PRIVATE functions (and the three MFT propagators) gets there.
    defined = {}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.setdefault(node.name, set()).add(rel)
    imported = {rel: {} for rel in mods}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported[rel].setdefault(
                        alias.asname or alias.name,
                        set()).update(defined.get(alias.name, set()))

    def resolve(rel, name):
        out = set()
        if rel in defined.get(name, set()):
            out.add((rel, name))
        for src in imported.get(rel, {}).get(name, set()):
            out.add((src, name))
        return out

    callees = {}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            here = callees.setdefault((rel, node.name), set())
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    nm = (getattr(sub.func, 'id', None)
                          or getattr(sub.func, 'attr', None))
                    if nm:
                        here |= resolve(rel, nm)

    prim = ('_bluestein_2d', '_bluestein_centred_2d')
    public = ('angular_spectrum_propagate_mft', 'fresnel_propagate_mft',
              'fraunhofer_propagate_mft')

    def reaches(start):
        seen, stack = set(), [(start, True)]
        while stack:
            node, is_start = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            mod, name = node
            if mod.startswith('ui/'):
                continue
            if not is_start:
                if name in prim:
                    return True
                if not (name.startswith('_') or name in public):
                    continue
            for nxt in callees.get(node, ()):
                stack.append((nxt, False))
        return False

    forwards = set()
    for exported in sorted(getattr(lumenairy, '__all__', ())):
        obj = getattr(lumenairy, exported, None)
        if not inspect.isfunction(obj):
            continue
        for mod in defined.get(obj.__name__, ()):
            if reaches((mod, obj.__name__)):
                forwards.add(exported)
                break
    exported_backwards = {
        e for e in sorted(getattr(lumenairy, '__all__', ()))
        if inspect.isfunction(getattr(lumenairy, e, None))
        and getattr(lumenairy, e).__name__ in backwards}
    assert forwards == exported_backwards, (
        f"the two walks disagree: forwards-only {sorted(forwards - exported_backwards)}, "
        f"backwards-only {sorted(exported_backwards - forwards)}.  One of the "
        f"two resolvers is wrong, and the census that decides which entry "
        f"points need a way back is the backwards one")
    assert len(forwards) == 13, (
        f"the census now finds {len(forwards)} exported entry points that can "
        f"reach an MFT primitive, against the 13 round 2 measured: "
        f"{sorted(forwards)}.  A new one needs a route keyword and a row in "
        f"the Migration Guide's way-back table; one fewer means a door was "
        f"closed and the table has a stale row")
    assert os.path.isdir(root)

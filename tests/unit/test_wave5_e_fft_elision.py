"""WAVE5-E item E1 -- what the pyFFTW ping-pong actually changes, and what it
does not.

VERIFY-WP-B14 section 3 took apart WP-B14's "``fft_infra`` DETERMINISM DEFECT"
and established the mechanism.  Restated:

  * ``_fft2`` / ``_ifft2`` **are** functions of their inputs, in every mode;
  * what the ping-pong changes is the KIND OF OBJECT handed back -- a live,
    non-owning view of the plan workspace with it on, a private ``buf.copy()``
    with it off;
  * NumPy's temporary elision can tell those apart, because ``temp_elide``
    claims an operand only when it is an unreferenced, NumPy-OWNED temporary.
    On the manylinux numpy 2.4.6 wheel a RIGHT-elided complex128 multiply does
    not give the same last bits as the named form (measured rel 1.0e-16 to
    1.8e-16 over 16-17 % of the doubles at n >= 128); on the Windows wheel of
    either 2.4.4 or 2.4.6 all four spellings agree.  That is an upstream NumPy
    non-invariance -- ``validation/probe_fft_elision/NUMPY_ISSUE_DRAFT.md`` is
    the report, and ``numpy_elision_reproducer.py`` beside it is the five-line
    reproducer.

THE DECISION (measured, 2026-09-15, probes in ``validation/probe_fft_elision/``).
The two candidate remedies were (a) privatising the dispatchers' return at the
shapes where it matters and (b) scoping the contract sentence.  (b) was chosen
on the measurement:

  * COST of (a): ``angular_spectrum_propagate`` runs +22.1 % / +19.0 % /
    +18.6 % slower at 512^2 / 1024^2 / 2048^2 on Windows and +13.0 % at 2048^2
    on WSL, with the raw copy 0.30-0.42 of one forward transform on both
    builds (``e1_copy_cost_*.json``).
  * BENEFIT of (a): ZERO for any lumenairy output.  Every in-library site that
    multiplies a dispatcher result NAMES the other operand, so nothing is
    elided with the ping-pong on and only the left operand is with it off --
    and left-elided equals named on every build measured.  Four entry points x
    three shapes x both builds: byte-identical across the switch, 12 of 12
    cells (``e1_decision_*.json``).

So this file pins, unconditionally, the two halves of the scoped sentence that
hold on every build, and gates the third on the running build showing the
elision asymmetry at all.
"""
from __future__ import annotations

import hashlib
import inspect
import sys

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as _carrier
from lumenairy.propagators import fft_infra as _fi

#: The scoped sentence VERIFY-WP-B14 D4 asked for, verbatim.  It is the
#: contract, so it is pinned as text, not paraphrased.
_D4_SENTENCE = (
    "The transform's values are byte-identical either way; the object handed "
    "back is a live workspace view in one mode and a private copy in the "
    "other, which NumPy's temporary elision can distinguish.")

_WL, _DX, _Z = 1.31e-6, 1.0e-6, 2.0e-3


def _md5(a):
    if isinstance(a, tuple):
        a = a[0]
    return hashlib.md5(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def _field(n, seed=11):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n))).astype(np.complex128)


@pytest.fixture
def restore_fft_state():
    """Restore the ping-pong switch and drain the caches around each test."""
    old = _fi.get_fft_double_buffer()
    try:
        yield
    finally:
        _fi.set_fft_double_buffer(old)
        la.clear_asm_caches()


def _elision_asymmetry(n=512):
    """Does THIS build's complex128 multiply depend on which operand is an
    elidable temporary?  The premise of the gated test below, measured on the
    running interpreter with no lumenairy in it."""
    rng = np.random.default_rng(5)
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    P = rng.standard_normal((n, n)) * 1e6
    a, h = A * 1.0, np.exp(1j * P)
    named = a * h
    right = a * np.exp(1j * P)
    return named, right, bool(not np.array_equal(right, named))


# ===========================================================================
# 1.  Unconditional -- the transforms are functions of their inputs
# ===========================================================================
@pytest.mark.parametrize('n', [256, 512])
@pytest.mark.parametrize('double_buffer', [True, False])
def test_the_transforms_are_functions_of_their_inputs(
        n, double_buffer, restore_fft_state):
    """WP-B14 raised this as "``_ifft2(_fft2(E)*H)`` is not a function of its
    input values alone".  VERIFY-WP-B14 refuted that as to mechanism; this is
    the unconditional statement of what IS true, kept as a live pin so a future
    change to the ping-pong cannot quietly make the transforms non-deterministic
    and hide behind the elision story.

    Six evaluations of ``_fft2`` on one fixed operand, and six of ``_ifft2`` on
    another, must give ONE byte image each -- in both modes, at both shapes,
    with the plan cache warm and the ping-pong slot alternating between calls
    (the two slots are distinct buffers, so alternation is what the ping-pong
    actually does).  Measured: 1 distinct image out of 6 in every cell, on both
    builds (VERIFY-WP-B14 section 3.1 and
    ``validation/probe_fft_elision/e1_decision_*.json``).
    """
    _fi.set_fft_double_buffer(double_buffer)
    la.clear_asm_caches()
    E = _field(n, seed=3)
    S = _fi._fft2(E).copy()          # a fixed operand for the inverse leg
    fwd = {_md5(_fi._fft2(E)) for _ in range(6)}
    inv = {_md5(_fi._ifft2(S)) for _ in range(6)}
    assert len(fwd) == 1, (
        f'_fft2 returned {len(fwd)} distinct byte images for one input at '
        f'n={n}, double_buffer={double_buffer}: {sorted(fwd)}')
    assert len(inv) == 1, (
        f'_ifft2 returned {len(inv)} distinct byte images for one input at '
        f'n={n}, double_buffer={double_buffer}: {sorted(inv)}')


# ===========================================================================
# 2.  Unconditional -- no lumenairy entry point moves across the switch
# ===========================================================================
_ENTRY_POINTS = ('angular_spectrum_propagate', 'fresnel_propagate',
                 'carrier_exact_envelope_tf_step')


@pytest.mark.parametrize('n', [256, 512])
@pytest.mark.parametrize('entry', _ENTRY_POINTS)
def test_no_library_entry_point_moves_across_the_ping_pong_switch(
        n, entry, restore_fft_state):
    """THE REASON REMEDY (a) WAS NOT TAKEN, asserted rather than asserted-about.

    Every in-library consumer of a dispatcher result spells the product with a
    NAMED right operand (``asm.py:919/922/1391``, ``carrier.py:1401/7126``,
    ``fresnel.py:216`` are all ``_fft2(...) * H``), so with the ping-pong ON
    neither operand is an unreferenced temporary and with it OFF only the LEFT
    one is -- and left-elided equals named on every build measured.  The
    library is therefore not exposed to the upstream non-invariance, which is
    why paying 13-22 % of the ASM hot path to privatise every transform buys
    nothing.

    If a future edit rewrites one of those sites as ``_fft2(E) * np.exp(...)``
    -- right operand a fresh temporary -- this test goes red on the Linux
    build, which is exactly the event that would make remedy (a) worth its
    cost.  Measured byte-identical in 12 of 12 cells on both builds, 2026-09-15
    (``validation/probe_fft_elision/e1_decision_*.json``).
    """
    E = _field(n)
    if entry == 'angular_spectrum_propagate':
        def call():
            return la.angular_spectrum_propagate(E, _Z, _WL, _DX)
    elif entry == 'fresnel_propagate':
        def call():
            return la.fresnel_propagate(E, _Z, _WL, _DX)
    else:
        def call():
            return _carrier._exact_envelope_tf_step(
                E, 5e-3, 1.55e-6, 2e-6, 2e-6, tilt=(0.03, -0.02))

    digests = {}
    for mode in (True, False):
        _fi.set_fft_double_buffer(mode)
        la.clear_asm_caches()
        digests[mode] = _md5(call())
    assert digests[True] == digests[False], (
        f'{entry} at n={n} moved across set_fft_double_buffer: '
        f'ping-pong {digests[True]} vs copy {digests[False]}.  Either a call '
        f'site now multiplies a dispatcher result by an unnamed temporary (see '
        f'this file\'s docstring -- that is when privatising the dispatchers '
        f'starts to be worth its 13-22 % on the ASM hot path), or the '
        f'dispatchers no longer return the same values in the two modes.')


# ===========================================================================
# 3.  Unconditional -- the contract sentence is where D4 asked for it
# ===========================================================================
def test_the_double_buffer_contract_carries_the_scoped_sentence():
    """VERIFY-WP-B14 D4: the knob's registered doc said "values are
    byte-identical either way", which is true of the TRANSFORM's values and
    false of any downstream NumPy expression on the returned array, on the
    Linux build.  The accurate sentence is pinned here so the scope cannot be
    dropped back to the shorter claim by a later edit.
    """
    from lumenairy._knobs import knob_doc                # noqa: PLC0415

    doc = knob_doc('fft_double_buffer')
    norm = ' '.join(doc.split())
    want = ' '.join(_D4_SENTENCE.split())
    assert want.lower() in norm.lower(), (
        f"set_fft_double_buffer's registered doc must carry the scoped "
        f'sentence VERIFY-WP-B14 D4 derived.\nwant: {want}\ngot:  {norm}')
    setter_doc = ' '.join((inspect.getdoc(_fi.set_fft_double_buffer)
                           or '').split())
    assert 'temporary elision' in setter_doc.lower(), (
        'set_fft_double_buffer\'s own docstring must name the elision scope; '
        f'got: {setter_doc}')


# ===========================================================================
# 4.  PREMISE-GATED -- only where the build shows the elision asymmetry
# ===========================================================================
def test_the_chosen_remedy_holds_where_the_elision_asymmetry_exists():
    """The gated half of the scoped sentence: "which NumPy's temporary elision
    can distinguish".

    PREMISE, measured here and reported in the skip message: does this build's
    complex128 multiply depend on which operand is an elidable temporary?  On
    the manylinux numpy 2.4.6 wheel it does; on the Windows wheel of 2.4.4 or
    2.4.6 it does not, so there is nothing for this test to observe and it
    says so with the reading rather than with a bare skip.

    WHERE THE PREMISE HOLDS the remedy's two claims are asserted together:

      1. the OBJECT differs -- ``_fft2`` hands back a non-owning workspace view
         with the ping-pong on and an owning array with it off, at
         n >= ``FFTW_MIN_SIZE``.  That is the whole of what the knob changes.
      2. a CALLER that multiplies the returned array by a fresh unnamed
         temporary DOES see the last bits move across the switch, while (test 2
         above) no library entry point does.  Measured rel 3.5e-16 / 3.4e-16 /
         4.0e-16 at n = 256 / 512 / 1024 on WSL py3.12.3 / numpy 2.4.6,
         2026-09-15, and 0.0 at every n on Windows.

    Together those say the scoped sentence is exactly right and the shorter one
    was not -- which is the remedy.
    """
    named, right, asymmetric = _elision_asymmetry(512)
    if not asymmetric:
        sc = float(np.max(np.abs(named))) or 1.0
        rel = float(np.max(np.abs(right - named))) / sc
        pytest.skip(
            f'premise absent on this build: numpy {np.__version__} on '
            f'{sys.platform} gives '
            f'right-elided == named exactly (max rel {rel:.3e} over '
            f'{named.size * 2} doubles), so temporary elision cannot '
            f'distinguish a view from a copy here.  The elision asymmetry is '
            f'a property of the wheel, not of lumenairy: measured present on '
            f'manylinux numpy 2.4.6 and absent on win_amd64 numpy 2.4.4 AND '
            f'2.4.6 (validation/probe_fft_elision/e1_spellings_*.json).')

    if not (_fi.PYFFTW_AVAILABLE and _fi.USE_PYFFTW):
        pytest.skip(
            f'premise absent: the pyFFTW path is not live '
            f'(PYFFTW_AVAILABLE={_fi.PYFFTW_AVAILABLE}, '
            f'USE_PYFFTW={_fi.USE_PYFFTW}), so the ping-pong that makes the '
            f'returned object a view never runs.')

    n = max(256, int(_fi.FFTW_MIN_SIZE))
    old = _fi.get_fft_double_buffer()
    try:
        E = _field(n, seed=7)
        rng = np.random.default_rng(7)
        rng.standard_normal((n, n))
        rng.standard_normal((n, n))
        P = rng.standard_normal((n, n)) * 1e3

        owns = {}
        outs = {}
        for mode in (True, False):
            _fi.set_fft_double_buffer(mode)
            la.clear_asm_caches()
            _fi._fft2(E)                      # build the plan at this key
            owns[mode] = bool(np.asarray(_fi._fft2(E)).flags.owndata)
            outs[mode] = np.array(
                _fi._ifft2(_fi._fft2(E) * np.exp(1j * P)), copy=True)

        # 1 -- the object
        assert owns[True] is False and owns[False] is True, (
            f'the ping-pong must hand back a non-owning workspace view and the '
            f'single-buffer path an owning copy at n={n}; measured owndata '
            f'ping-pong={owns[True]} copy={owns[False]}')

        # 2 -- the caller spelling that the elision reaches
        sc = float(np.max(np.abs(outs[False]))) or 1.0
        rel = float(np.max(np.abs(outs[True] - outs[False]))) / sc
        assert not np.array_equal(outs[True], outs[False]), (
            f'this build DOES distinguish the elided spellings (premise '
            f'measured above), so _ifft2(_fft2(E) * np.exp(1j*P)) -- right '
            f'operand a fresh temporary -- must differ across the switch; '
            f'measured rel {rel:.3e}.  If it no longer does, the dispatchers '
            f'have started privatising their return and the docstring scope '
            f'(remedy (b)) should be revisited against the cost measurement '
            f'in this file\'s docstring.')
        assert rel < 1e-14, (
            f'the caller-visible divergence must stay at the elision floor '
            f'(~1 ULP; measured 3.4e-16 to 4.0e-16 on WSL numpy 2.4.6, '
            f'2026-09-15).  {rel:.3e} is far above that and is not an elision '
            f'effect -- the transforms themselves would have to differ.')
    finally:
        _fi.set_fft_double_buffer(old)
        la.clear_asm_caches()

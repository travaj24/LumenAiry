"""Independent re-verification of WP-A16's config objects and lens architecture.

WHY THIS FILE EXISTS, beside the three ``test_audit2609_a16_*`` files.  Those
were written by the work package that made the change and they all stand on ONE
fixture (WP-A15a's curved-rear cemented doublet, N = 64, square grid,
complex128, on-axis, 632.8 nm) and on the library's own helpers.  This file is
the adversarial half: a different fixture built here, the inputs the covering
array does not carry, and gates on the contracts the re-dispatch design rests
on rather than on the values it happens to produce.

WHAT IS NEW HERE (each item is a property the shipped A16 files do not assert):

1. **A second fixture.**  A curved-rear *singlet* (not a doublet) in N-SF11 at
   1064 nm on a **non-square N = 40 grid pitch** (``dy = 1.15 dx``), fed a
   **complex64** off-centre diverging wave, plus a **decentred + tilted** rear
   surface.  Those are five axes the covering array holds fixed.
2. **Identical REFUSAL.**  Half of "the configured path is the keyword path"
   is what happens when the library says no: an illegal combination must raise
   the same exception, with the same message, through all three spellings.  A
   resolver that dropped or reordered a setting would refuse differently (or
   not at all) and the existing bit-identity tests -- which only compare
   successful calls -- would not see it.
3. **The ``locals()`` contract.**  ``resolve_entry_point_kwargs`` reads the
   entry point's ``locals()`` and trusts that no parameter has been rebound
   yet.  Nothing pinned that.  An AST gate now does.
4. **Value equality with an array-valued field.**  ``carrier`` is documented
   to accept a wavefront ndarray and ``_same`` carries an array arm for it,
   but the generated ``__eq__`` compared field TUPLES and raised
   ``ValueError`` on exactly those values -- so the module docstring's own
   round trip (``from_kwargs(**to_kwargs()) == cfg``) crashed.  Fixed in
   ``lens_config`` by VERIFY-A16; pinned here, with the hash contract as its
   counter-pin.
5. **One rule for ``sag_dtype``, three spellings.**  ``LensResources`` and
   ``set_lens_sag_dtype`` refuse anything but float32/float64; the per-call
   keyword resolved an unrecognised dtype to float64 in silence -- the
   discarded-setting class the config objects exist to close.  Fixed in
   ``_lens_real._resolve_sag_real`` by VERIFY-A16 and pinned here in all three
   places, with the byte-identity of the LEGAL values as its counter-pin.
6. **Falsifiability of the shipped structural gate**, driven from outside it.

No wall-clock is asserted anywhere (TESTING_STANDARDS S1).  The two numeric
bars in this file (the Ludwig ones live next door) are derived in their
docstrings; everything else is exact -- ``np.array_equal``, set equality, or an
exception's own text.

RUNTIME.  MEASURED 2026-09-12 on this box: ~9 s, dominated by the FGA numba
compile and the two multibranch arms.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import pickle
import types
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.backend import _optional
from lumenairy.elements import (
    _lens_kernels,
    _lens_real,
    _lens_traced,
    lenses,
)
from lumenairy.elements import lens_config as lc
from lumenairy.elements._lens_real import apply_real_lens, prepare_real_lens
from lumenairy.elements._lens_traced import (
    apply_real_lens_traced,
    prepare_real_lens_traced,
)
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.elements.lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensResources,
)
from lumenairy.elements.lenses_gbd import apply_real_lens_gbd
from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
from lumenairy.propagators.fga import apply_real_lens_fga

ENTRY_POINTS = {
    'apply_real_lens': apply_real_lens,
    'apply_real_lens_traced': apply_real_lens_traced,
    'prepare_real_lens_traced': prepare_real_lens_traced,
    'apply_real_lens_maslov': apply_real_lens_maslov,
    'apply_real_lens_gbd': apply_real_lens_gbd,
    'apply_real_lens_fga': apply_real_lens_fga,
    'apply_real_lens_traced_multibranch': apply_real_lens_traced_multibranch,
}

# ---------------------------------------------------------------------------
# The independent fixture
# ---------------------------------------------------------------------------
LAM = 1064.0e-9                # not the 632.8 nm every other lens file uses
APERTURE = 8.0e-3
N_GRID = 40                    # not 64, not a power of two after 8


def _singlet(decentred: bool = False) -> dict:
    """A curved-rear N-SF11 singlet.  ``decentred`` puts a decenter AND a tilt
    on the REAR surface only, so the element is genuinely asymmetric rather
    than globally shifted (which a coordinate change would undo)."""
    rear = dict(radius=-95.0e-3, glass_before='N-SF11', glass_after='AIR')
    if decentred:
        rear = dict(rear, decenter=(25.0e-6, -18.0e-6), tilt=(8.0e-4, 3.0e-4))
    return dict(
        surfaces=[dict(radius=42.0e-3, glass_before='AIR',
                       glass_after='N-SF11'), rear],
        thicknesses=[4.0e-3], aperture_diameter=APERTURE)


def _fixture(dtype=np.complex128, decentred: bool = False):
    """``(E_in, dx, dy, prescription)`` -- an OFF-CENTRE diverging wave on an
    anamorphic grid, so the decentred arm has something to be asymmetric
    about and ``dy`` is never silently equal to ``dx``."""
    dx = 1.30 * APERTURE / N_GRID
    dy = 1.15 * dx
    x = (np.arange(N_GRID) - (N_GRID - 1) / 2.0) * dx
    y = (np.arange(N_GRID) - (N_GRID - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y)
    amp = np.exp(-((xx - 0.4e-3) ** 2 + (yy + 0.25e-3) ** 2)
                 / (0.28 * APERTURE) ** 2)
    phase = (2.0 * np.pi / LAM) * (xx ** 2 + yy ** 2) / (2.0 * 80.0e-3)
    return ((amp * np.exp(1j * phase)).astype(dtype), dx, dy,
            _singlet(decentred))


def _field(out):
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _call(fn, e_in, base, **extra):
    with warnings.catch_warnings():
        # the fixture is deliberately non-collimated, off-centre and clipped,
        # so several arms emit the library's (correct) advisories.
        warnings.simplefilter('ignore')
        return _field(fn(e_in, **base, **extra))


#: ``(id, entry point, dtype, decentred, settings)``.  ``dy`` is anamorphic on
#: every arm that takes it; ``apply_real_lens_traced`` is absent because it
#: refuses ``dy != dx`` outright -- that refusal is itself a case below.
_CASES = [
    ('analytic-c64-anamorphic', 'apply_real_lens', np.complex64, False,
     dict(surface_model='tangent_facet_remap', remap_order=5,
          wave_propagator='rayleigh_sommerfeld', sag_dtype=np.float32,
          sag_chunk_rows=4)),
    ('analytic-decentred', 'apply_real_lens', np.complex128, True,
     dict(surface_model='displaced', conjugate=-80.0e-3,
          wave_propagator='asm', bandlimit=False,
          accumulator_store='ram')),
    ('maslov-c64-anamorphic', 'apply_real_lens_maslov', np.complex64, False,
     dict(output_plane_distance=1.7e-3, output_plane_n=1.0003,
          output_subsample=2)),
    ('gbd-anamorphic', 'apply_real_lens_gbd', np.complex128, False,
     dict(output_plane_distance=1.7e-3, clip_aperture=False,
          output_subsample=2)),
    ('fga-anamorphic', 'apply_real_lens_fga', np.complex128, False,
     dict(output_plane_distance=1.7e-3)),
    ('multibranch-decentred', 'apply_real_lens_traced_multibranch',
     np.complex128, True,
     dict(output_plane_distance=1.5e-3, output_plane_n=1.0003,
          ray_subsample=1, min_area_ratio=1e-5, caustic_band='plain')),
]


@pytest.mark.parametrize('name,ep,dtype,decentred,settings', _CASES,
                         ids=[c[0] for c in _CASES])
def test_configured_call_equals_the_keyword_call_on_an_independent_fixture(
        name, ep, dtype, decentred, settings):
    """P1 on a fixture WP-A16 did not use.

    ``np.array_equal``, not ``allclose``: the re-dispatch merges keywords and
    re-enters the same function, so it has no numerical content of its own and
    any difference at all is a resolver defect.  The arm also asserts that the
    settings MOVE the field, or an all-dropping resolver would pass.
    """
    fn = ENTRY_POINTS[ep]
    e_in, dx, dy, rx = _fixture(dtype, decentred)
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    if 'dy' in inspect.signature(fn).parameters:
        settings = dict(settings, dy=dy)
    cfg = LensConfig.from_kwargs(entry_point=ep, **settings)

    by_kwargs = _call(fn, e_in, base, **settings)
    by_config = _call(fn, e_in, base, config=cfg)
    by_triple = _call(fn, e_in, base, geometry=cfg.geometry,
                      numerics=cfg.numerics, resources=cfg.resources)
    bare = _call(fn, e_in, base)
    by_empty = _call(fn, e_in, base, config=LensConfig())

    for label, got in (('config=', by_config), ('triple', by_triple)):
        assert got.shape == by_kwargs.shape and got.dtype == by_kwargs.dtype, (
            f'{ep} [{label}]: {got.shape}/{got.dtype} != keyword call '
            f'{by_kwargs.shape}/{by_kwargs.dtype}.')
        assert np.array_equal(got, by_kwargs), (
            f'{ep} [{label}]: the configured call differs from the identical '
            f'keyword call by max |diff| = '
            f'{float(np.max(np.abs(got - by_kwargs))):.6e}.')
    assert np.array_equal(bare, by_empty), (
        f'{ep}: an all-default LensConfig changed the result by max |diff| = '
        f'{float(np.max(np.abs(bare - by_empty))):.6e}.')
    assert not np.array_equal(by_kwargs, bare), (
        f'{ep}: {settings} produce the byte-identical field to the all-default '
        f'call on this fixture, so the identity above cannot tell an honoured '
        f'config from a discarded one.')


@pytest.mark.parametrize('ep,settings,needle', [
    # dx != dy: the traced engine refuses square-pixel-only geometry
    ('apply_real_lens_traced',
     dict(newton_poly_order=5, min_coarse_samples_per_aperture=0),
     'square pixels'),
    # surface_model='displaced' requires the ASM in-glass propagator
    ('apply_real_lens', dict(surface_model='displaced',
                             wave_propagator='sas'), 'ASM'),
])
def test_an_illegal_combination_refuses_identically_through_every_spelling(
        ep, settings, needle):
    """The other half of "the configured path IS the keyword path".

    The existing bit-identity tests compare successful calls only.  A resolver
    that silently dropped a setting would make the configured call SUCCEED
    where the keyword call refuses -- which is the audit's failure class, and
    is invisible to a comparison of two returned fields.
    """
    fn = ENTRY_POINTS[ep]
    e_in, dx, dy, rx = _fixture()
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    if 'dy' in inspect.signature(fn).parameters:
        settings = dict(settings, dy=dy)
    cfg = LensConfig.from_kwargs(entry_point=ep, **settings)

    def _refusal(**extra):
        with pytest.raises(Exception) as exc:      # noqa: PT011 -- see below
            _call(fn, e_in, base, **extra)
        return type(exc.value).__name__, str(exc.value)

    # deliberately broad: the POINT is that the three spellings raise the same
    # thing, whatever that is, so the expected type must not be baked in here.
    kw = _refusal(**settings)
    assert needle in kw[1], (
        f'{ep}: the keyword call did not refuse for the reason this case is '
        f'about ({needle!r}); it said {kw[1][:200]!r}.')
    assert _refusal(config=cfg) == kw, (
        f'{ep}: config= refused differently from the keyword call.\n'
        f'  keyword: {kw}\n  config : {_refusal(config=cfg)}')
    assert _refusal(geometry=cfg.geometry, numerics=cfg.numerics,
                    resources=cfg.resources) == kw, (
        f'{ep}: the three-component call refused differently.')


def test_the_config_block_runs_before_any_parameter_is_rebound():
    """The contract ``resolve_entry_point_kwargs`` rests on, pinned.

    The resolver reads ``caller_locals[name]`` for EVERY parameter in the
    signature and compares each against that parameter's signature default to
    decide whether the caller asked for it.  If a statement above the block
    normalised a parameter in place -- ``dy = dx if dy is None else dy`` is
    the obvious one -- the resolver would read the normalised value, conclude
    the caller had requested it, and then refuse a config that set the same
    field.  Nothing else in the suite would notice.
    """
    offenders = []
    for ep, fn in ENTRY_POINTS.items():
        src = inspect.getsource(inspect.unwrap(fn))
        tree = ast.parse(''.join(src.splitlines(keepends=True)).lstrip())
        body = tree.body[0]
        params = {p for p in inspect.signature(fn).parameters}
        guard_line = None
        for i, node in enumerate(body.body):
            names = {n.id for n in ast.walk(node)
                     if isinstance(n, ast.Name) and n.id == '_wants_config'}
            if names:
                guard_line = i
                break
        assert guard_line is not None, f'{ep}: no _wants_config block found'
        for node in body.body[:guard_line]:
            for sub in ast.walk(node):
                targets = []
                if isinstance(sub, ast.Assign):
                    targets = sub.targets
                elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                    targets = [sub.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id in params:
                        offenders.append(f'{ep}: {t.id} is rebound at body '
                                         f'statement {body.body.index(node)}, '
                                         f'before the config block')
    assert not offenders, (
        'the config re-dispatch reads locals() and needs every parameter still '
        'holding what the caller passed:\n  ' + '\n  '.join(offenders))


# ---------------------------------------------------------------------------
# Value semantics
# ---------------------------------------------------------------------------

def test_value_equality_survives_an_array_valued_field():
    """``carrier`` is documented to accept a wavefront ndarray.

    FAIL-BEFORE: with the dataclass-generated ``__eq__`` (which compares the
    two field TUPLES and therefore calls ``bool()`` on ``arr == arr``) every
    assertion below raised ``ValueError: The truth value of an array with more
    than one element is ambiguous`` -- including the round trip the class
    docstring advertises.  Re-check by reverting ``lens_config``'s
    ``__eq__ = _dataclass_eq`` to the generated one.
    """
    w = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    same, other = w.copy(), w + 1.0
    assert LensGeometry(carrier=w) == LensGeometry(carrier=same)
    assert LensGeometry(carrier=w) != LensGeometry(carrier=other)
    assert LensGeometry(carrier=w) != LensGeometry(carrier=w, dy=1e-5)
    cfg = LensConfig(geometry=LensGeometry(carrier=w))
    assert cfg == LensConfig(geometry=LensGeometry(carrier=same))
    assert LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg
    assert cfg.narrowed_to('apply_real_lens') == cfg
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    assert dataclasses.replace(cfg.geometry, dy=1e-5) == LensGeometry(
        carrier=same, dy=1e-5)
    # ...and it is still a bool, not an array, for a mismatched type
    assert (LensGeometry(carrier=w) == 'not a LensGeometry') is False


def test_hash_is_still_by_value_and_agrees_with_equality():
    """Counter-pin to the test above.

    The three classes declare ``eq=False`` so their ``__eq__`` survives, which
    means ``__hash__`` has to be supplied by hand.  If that were forgotten,
    ``hash`` would fall back to identity and every dict/set of configs would
    silently stop de-duplicating -- with no other test noticing.
    """
    for cls, kwargs in ((LensGeometry, dict(dy=1e-5)),
                        (LensNumerics, dict(newton_poly_order=8)),
                        (LensResources, dict(n_workers=3))):
        a, b = cls(**kwargs), cls(**kwargs)
        assert a is not b and a == b
        assert hash(a) == hash(b), (
            f'{cls.__name__}: two equal instances hash differently, so a set '
            f'of configs would hold duplicates.')
        assert len({a, b}) == 1
        assert hash(cls()) != hash(a) or cls() == a
    cfg = LensConfig(numerics=LensNumerics(newton_poly_order=8))
    assert hash(cfg) == hash(LensConfig(
        numerics=LensNumerics(newton_poly_order=8)))
    # an array-valued field is unhashable -- as an ndarray always is.  Stated
    # here so the limit is documented rather than discovered.
    with pytest.raises(TypeError):
        hash(LensGeometry(carrier=np.zeros(4)))


def test_an_unpickled_config_resolves_identically_at_the_call():
    """Configs are meant to be stored in tables and shipped between processes;
    an unpickled one bypasses ``__init__`` (and therefore ``__post_init__``),
    so the resolver has to work on it unchanged."""
    e_in, dx, dy, rx = _fixture()
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    cfg = LensConfig(numerics=LensNumerics(wave_propagator='rs'),
                     geometry=LensGeometry(dy=dy))
    back = pickle.loads(pickle.dumps(cfg))
    assert back == cfg
    assert np.array_equal(_call(apply_real_lens, e_in, base, config=back),
                          _call(apply_real_lens, e_in, base,
                                wave_propagator='rs', dy=dy))


# ---------------------------------------------------------------------------
# One rule for sag_dtype
# ---------------------------------------------------------------------------

_SAG_ILLEGAL = (np.float16, np.complex128, np.int32, 'int32')


@pytest.mark.parametrize('bad', _SAG_ILLEGAL,
                         ids=[str(getattr(b, '__name__', b))
                              for b in _SAG_ILLEGAL])
def test_all_three_spellings_of_sag_dtype_refuse_the_same_set(bad):
    """FAIL-BEFORE: the per-call keyword ACCEPTED every one of these and
    returned the float64 field with no warning (measured 2026-09-12:
    ``apply_real_lens(..., sag_dtype=np.float16)`` was byte-identical to the
    default call), while ``LensResources`` and ``set_lens_sag_dtype`` refused
    them.  One setting with three spellings and two different ideas of what is
    legal is the discarded-setting class the config objects exist to close.
    """
    e_in, dx, _dy, rx = _fixture()
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    with pytest.raises(ValueError) as by_field:
        LensResources(sag_dtype=bad)
    assert 'sag_dtype' in str(by_field.value)

    keep = _lens_real.get_lens_sag_dtype()
    try:
        with pytest.raises(ValueError) as by_knob:
            _lens_real.set_lens_sag_dtype(bad)
        assert 'sag_dtype' in str(by_knob.value)
    finally:
        _lens_real.set_lens_sag_dtype(keep)

    with pytest.raises(ValueError) as by_kwarg:
        _call(apply_real_lens, e_in, base, sag_dtype=bad)
    msg = str(by_kwarg.value)
    assert msg.startswith('apply_real_lens:'), msg[:160]
    assert 'sag_dtype' in msg and 'float32 or float64' in msg, msg[:200]

    with pytest.raises(ValueError) as by_prepare:
        prepare_real_lens(prescription=rx, wavelength=LAM, dx=dx,
                          N=e_in.shape[0], sag_dtype=bad)
    assert str(by_prepare.value).startswith('prepare_real_lens:')


def test_the_legal_sag_dtypes_are_untouched_by_that_refusal():
    """Counter-pin: a validator that refused everything would make the test
    above pass.  ``None`` and ``np.float64`` must stay byte-identical to the
    call that omits the keyword, and ``np.float32`` must remain the documented
    opt-in that actually changes the geometry lineage."""
    e_in, dx, _dy, rx = _fixture()
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    bare = _call(apply_real_lens, e_in, base)
    assert np.array_equal(_call(apply_real_lens, e_in, base, sag_dtype=None),
                          bare)
    assert np.array_equal(
        _call(apply_real_lens, e_in, base, sag_dtype=np.float64), bare)
    assert np.array_equal(
        _call(apply_real_lens, e_in, base, sag_dtype='float64'), bare)
    f32 = _call(apply_real_lens, e_in, base, sag_dtype=np.float32)
    assert not np.array_equal(f32, bare), (
        'sag_dtype=np.float32 is byte-identical to float64 on this fixture, so '
        'this test cannot tell an honoured dtype from a discarded one.')
    assert _lens_real._resolve_sag_real(np.float32) is np.float32
    assert _lens_real._resolve_sag_real('f4') is np.float32
    assert _lens_real._resolve_sag_real(None) is np.float64


# ---------------------------------------------------------------------------
# Falsifiability of the shipped structural gate
# ---------------------------------------------------------------------------

class _FakeParam:
    kind = inspect.Parameter.KEYWORD_ONLY

    def __init__(self, default):
        self.default = default


def test_the_shipped_default_agreement_gate_would_catch_a_drifted_default():
    """Drive WP-A16's own walker with a corrupted signature view.

    The gate that makes the precedence rule well defined is "every field's
    dataclass default equals that entry point's signature default".  It is
    asserted in ``test_audit2609_a16_lens_config_round_trip.py``; this drives
    that test function with ``_kwonly`` patched so ``apply_real_lens`` appears
    to default ``remap_order`` to 5 instead of 3, and requires it to fail.
    Without this, a gate that had silently become vacuous would still be green.
    """
    rt = pytest.importorskip(
        'tests.unit.test_audit2609_a16_lens_config_round_trip')
    real = rt._kwonly

    def drifted(fn):
        out = dict(real(fn))
        if fn.__name__ == 'apply_real_lens':
            out['remap_order'] = _FakeParam(5)
        return out

    rt._kwonly = drifted
    try:
        with pytest.raises(AssertionError) as exc:
            rt.test_every_table_entry_names_a_real_keyword_with_the_same_default(
                'apply_real_lens')
        assert 'remap_order' in str(exc.value)
    finally:
        rt._kwonly = real
    # ...and the un-corrupted call still passes, so the patch was the cause
    rt.test_every_table_entry_names_a_real_keyword_with_the_same_default(
        'apply_real_lens')


def test_the_shipped_classification_gate_would_catch_a_new_keyword():
    """Same treatment for the "every parameter is classified" walker."""
    rt = pytest.importorskip(
        'tests.unit.test_audit2609_a16_lens_config_round_trip')
    real = rt._kwonly

    def extra(fn):
        out = dict(real(fn))
        if fn.__name__ == 'apply_real_lens_gbd':
            out['a_knob_nobody_classified'] = _FakeParam(None)
        return out

    rt._kwonly = extra
    try:
        with pytest.raises(AssertionError) as exc:
            rt.test_every_keyword_is_classified_as_field_contract_or_documented_exclusion(
                'apply_real_lens_gbd')
        assert 'a_knob_nobody_classified' in str(exc.value)
    finally:
        rt._kwonly = real


def test_the_forwarded_parameter_set_is_every_keyword_only_one():
    """``_signature_info`` forwards only KEYWORD_ONLY parameters.

    A parameter added as POSITIONAL_OR_KEYWORD would be silently dropped by
    the re-dispatch -- the configured call would run with that parameter at
    its default while the keyword call honoured it.  Today ``E_in`` is the
    only one and it is passed positionally; this asserts both halves so the
    assumption cannot quietly stop holding.
    """
    for ep, fn in ENTRY_POINTS.items():
        sig = inspect.signature(fn).parameters
        pok = [n for n, p in sig.items()
               if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        assert pok in ([], ['E_in']), (
            f'{ep} has non-keyword-only parameters {pok}; the config '
            f're-dispatch forwards keyword-only ones and would drop them.')
        forwarded = set(lc._signature_info(fn)['params'])
        expected = {n for n, p in sig.items()
                    if p.kind is inspect.Parameter.KEYWORD_ONLY
                    and n not in lc._CONFIG_PARAMETERS}
        assert forwarded == expected, (
            f'{ep}: the re-dispatch forwards {sorted(forwarded)} but the '
            f'signature has {sorted(expected)}.')


def test_the_re_dispatch_terminates_and_costs_exactly_one_frame():
    """Two structural properties of the design.

    (a) The merged mapping has the four config parameters removed, so
    re-entering the entry point cannot recurse -- asserted for an all-default
    config, a one-field config and all three components at once, each of which
    takes the re-dispatch branch.
    (b) It costs ONE extra frame of the entry point and no more, measured by
    counting frames from inside the call through the ``progress`` sink.
    """
    e_in, dx, _dy, rx = _fixture()
    base = dict(prescription=rx, wavelength=LAM, dx=dx)
    for extra in (dict(config=LensConfig()),
                  dict(numerics=LensNumerics(wave_propagator='rs')),
                  dict(geometry=LensGeometry(), numerics=LensNumerics(),
                       resources=LensResources())):
        _call(apply_real_lens, e_in, base, **extra)   # RecursionError if not

    seen = {}

    class _Probe:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, *a, **k):
            import traceback
            seen[self.tag] = len([f for f in traceback.extract_stack()
                                  if f.name == 'apply_real_lens'])

    _call(apply_real_lens, e_in, base, progress=_Probe('plain'))
    _call(apply_real_lens, e_in, base,
          resources=LensResources(progress=_Probe('configured')))
    assert seen.get('plain') == 1 and seen.get('configured') == 2, (
        f'the configured path should add exactly one apply_real_lens frame; '
        f'measured {seen}.')


def test_the_configured_path_emits_the_same_warnings():
    """A re-dispatched call must produce the same diagnostics.

    Only the CATEGORY and the MESSAGE are asserted, deliberately: the
    ``stacklevel`` attribution does shift by one frame on the configured path
    (measured and recorded in VERIFY_WP-A16.md as an open item), and pinning
    that here would turn red the day someone fixes it.
    """
    e_in, dx, _dy, _rx = _fixture()
    rx = dict(_singlet(), aperture_diameter=40.0e-3)   # far wider than N*dx
    base = dict(prescription=rx, wavelength=LAM, dx=dx)

    def _emit(**extra):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            apply_real_lens(e_in, **base, **extra)
        return [(w.category.__name__, str(w.message)) for w in rec]

    plain = _emit()
    assert plain, ('the fixture stopped warning, so this test compares two '
                   'empty lists; widen the aperture further.')
    assert _emit(numerics=LensNumerics(bandlimit=True)) == plain
    assert _emit(config=LensConfig()) == plain


# ---------------------------------------------------------------------------
# Architecture: the shared optional-dependency helper
# ---------------------------------------------------------------------------

# ``lenses`` is not in this tuple since WP-B11c: its copy of the plumbing moved
# into the ``_lens_kernels`` leaf, which is where ``_is_cupy_array`` now reads
# ``_optional_is_cupy_array`` and ``_ensure_cupy`` from.  Substituting them on
# the facade instead would bind attributes the function never reads -- the
# arm would go green while measuring the UNPATCHED code, which is the exact
# shape this test exists to catch.  The facade's own obligation (the live
# two-way forward of ``cp``) is pinned in ``test_audit2609_a16_lens_arch.py``
# and in ``test_audit2609_b11c_structure.py``.
@pytest.mark.parametrize('mod', (_lens_real, _lens_traced, _lens_kernels),
                         ids=lambda m: m.__name__.rsplit('.', 1)[-1])
def test_a_true_cupy_answer_really_binds_the_module_cp(mod, monkeypatch):
    """The dedupe's load-bearing coupling, exercised rather than inspected.

    The GPU branches read the module-level NAME (``xp = cp if
    _is_cupy_array(E) else np``), so ``_is_cupy_array(x) is True`` must imply
    ``cp`` is bound.  The shipped A16 test asserts that by looking for
    ``_ensure_cupy_loaded`` in the source; this one FAKES a CuPy answer and
    checks the name afterwards, which is what a CUDA box would check.  CuPy is
    not installed here, so this is the only way to reach the True branch.
    """
    stub = types.ModuleType('cupy_stub')
    monkeypatch.setattr(mod, 'CUPY_AVAILABLE', True)
    monkeypatch.setattr(mod, 'cp', None)
    monkeypatch.setattr(mod, '_optional_is_cupy_array', lambda x: True)
    monkeypatch.setattr(mod, '_ensure_cupy', lambda: stub)
    try:
        assert mod._is_cupy_array(object()) is True
        assert mod.cp is stub, (
            f'{mod.__name__}._is_cupy_array answered True without binding the '
            f'module-level cp; the GPU branches would raise NameError.')
    finally:
        mod.cp = None
    # and the NumPy answer stays False and cheap once restored
    monkeypatch.undo()
    assert mod._is_cupy_array(np.zeros(3)) is False
    assert mod.CUPY_AVAILABLE is _optional.CUPY_AVAILABLE


def test_no_lens_family_module_imports_scipy_at_module_scope():
    """The import-time item as a STRUCTURAL gate over the whole family.

    The shipped A16 test asserts ``scipy.linalg`` is absent from a fresh
    interpreter's ``sys.modules``; this asserts the cause, so a new
    module-level ``from scipy.x import y`` in any lens module is caught at the
    line rather than after it has been paid for.
    """
    root = pathlib.Path(la.__file__).resolve().parent
    offenders = []
    for py in sorted((root / 'elements').glob('*lens*.py')) + [
            root / 'elements' / 'lenses.py',
            root / 'elements' / 'lenses_gbd.py',
            root / 'elements' / 'lenses_maslov.py',
            root / 'propagators' / 'fga.py',
            root / 'backend' / '__init__.py']:
        if not py.exists():
            continue
        for node in ast.parse(py.read_text(encoding='utf-8')).body:
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split('.')[0] == 'scipy':
                    offenders.append(f'{py.name}:{node.lineno} {m}')
    assert not offenders, (
        'these modules pay for scipy at ``import lumenairy``:\n  '
        + '\n  '.join(offenders))


def test_lens_config_stays_a_leaf():
    """``lens_config`` is imported at module scope by all seven entry points.

    If it ever imported one of them back at module scope the family would gain
    a fifth import cycle and ``import lumenairy`` would depend on the order
    the elements package happens to walk.  The vocabularies it borrows from
    ``_lens_real`` come through an in-function import for exactly this reason.
    """
    src = pathlib.Path(lc.__file__).read_text(encoding='utf-8')
    bad = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            if node.module.split('.')[0] not in ('_cache_registry',):
                bad.append(f'line {node.lineno}: from {"." * node.level}'
                           f'{node.module}')
        elif isinstance(node, ast.ImportFrom) and node.level and not node.module:
            bad.append(f'line {node.lineno}: from {"." * node.level} import '
                       + ', '.join(a.name for a in node.names))
    assert not bad, (
        'lens_config must stay a leaf (only the stdlib-only _cache_registry is '
        'allowed at module scope):\n  ' + '\n  '.join(bad))

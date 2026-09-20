"""WP-C4 round 2, V-C4-D2 -- every public entry point that can reach the MFT
primitives exposes a ONE-KEYWORD way back, and ``None`` stamps nothing.

WHAT THE DEFECT WAS.  v5.49.0's shape rule moves answers at every call whose
output grid is at least 32x coarser than its input.  The campaign rule is that
a default flip leaves a ONE-KEYWORD way back on every public entry point it
moves.  ``method=`` was already spent on something else at seven of them --
the PSF sampler, the resampler, the propagator family, the transport -- so
they moved with no way back at all, and the Migration Guide offered a private,
process-wide module constant instead.  WP-C4 round 2 adds ``mft_method=``,
which is free on every one of those signatures, and threads it to the
``_bluestein_2d`` / ``_bluestein_centred_2d`` call.

THE THREE CLAIMS HERE.

1.  THE CENSUS.  An AST sweep decides, from the SOURCE and not from a list,
    which exported entry points can reach a primitive; every one of them must
    expose a route keyword.  It is a census and not a fixture: a new caller
    added without the keyword fails it the day it lands.  Its own falsifier is
    the id below it, which drops the keyword from one entry point and requires
    the census to name that entry point.
2.  ``None`` STAMPS NOTHING.  The default is ``None``, not ``'auto'``, and it
    means the keyword is left OFF the call the entry point makes -- so
    whatever the callee's own default is at the time governs, and a future
    change to that default reaches these callers without eight edits.
    Asserted by spying on the MFT layer (the two primitives AND the three
    public propagators, because most of these entry points forward through
    one of those and its own ``method`` default would otherwise hide the
    sentinel) and reading the FIRST recorded call.
3.  NAMING A ROUTE REACHES THE MFT LAYER.  The two-sided arm: the same spy
    must see ``method='bluestein'`` when the caller passes
    ``mft_method='bluestein'``, or claim 2 would pass on an entry point that
    silently drops the keyword.

The bytes themselves are proved ARCHIVE TO ARCHIVE against ``49ddf4bd`` --
which no in-process test can do -- by ``validation/probe_c4_round2/
r2_entry.py`` and ``r2_entry_compare.py``; the last id here asserts that the
committed evidence covers every entry point the census finds and that it
reports a way back at each.

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

#: The two primitives the rule lives in.  ``_direct_matrix_2d`` is NOT a seed:
#: it is the route, not the decision, and a caller that reaches only it has
#: already named the route.
_PRIMITIVES = ('_bluestein_2d', '_bluestein_centred_2d')

#: Public functions the census may pass THROUGH rather than stop at.  Their own
#: ``method=`` is the route keyword, so a caller that reaches a primitive only
#: through one of them is still a caller for this purpose -- which is the whole
#: point of ``compute_psf``, whose ``method=`` names the sampler and whose
#: transform is reached through ``fraunhofer_propagate_mft``.
_MFT_PUBLIC = ('angular_spectrum_propagate_mft', 'fresnel_propagate_mft',
               'fraunhofer_propagate_mft')

#: The keyword names that count as a way back.  ``method`` is the spelling the
#: three MFT propagators already had; ``mft_method`` is the one added where
#: ``method`` was taken.
_ROUTE_KEYWORDS = ('method', 'mft_method')

_ROOT = os.path.dirname(os.path.abspath(lumenairy.__file__))


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def _parse_package():
    """``{'relative/path.py': ast.Module}`` for every module in the package."""
    mods = {}
    for dirpath, _dirnames, filenames in os.walk(_ROOT):
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, _ROOT).replace(os.sep, '/')
            with open(path, encoding='cp1252', errors='replace') as fh:
                src = fh.read()
            try:
                mods[rel] = ast.parse(src)
            except SyntaxError:                       # pragma: no cover
                pytest.fail(f"{rel} does not parse")
    return mods


def _call_graph(mods):
    """``(callers, defined)`` -- a MODULE-QUALIFIED backwards call graph.

    A called NAME is resolved to a definition only when that name is defined
    in the calling module or is imported into it from another module of this
    package.  Resolving by bare name alone over-approximates catastrophically
    (a generic helper name shared by two subpackages joins their graphs, and
    the reachable set grows from 22 functions to 535); qualifying by module
    is what makes this a census rather than a guess.
    """
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
                        alias.asname or alias.name, set()).update(
                            defined.get(alias.name, set()))

    def resolve(rel, name):
        out = set()
        if rel in defined.get(name, set()):
            out.add((rel, name))
        for src in imported.get(rel, {}).get(name, set()):
            out.add((src, name))
        return out

    callers = {}
    for rel, tree in mods.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                name = (getattr(sub.func, 'id', None)
                        or getattr(sub.func, 'attr', None))
                if not name:
                    continue
                for target in resolve(rel, name):
                    callers.setdefault(target, set()).add((rel, node.name))
    return callers, defined


def _reaching_functions(mods):
    """``{(module, function)}`` that can reach a primitive.

    The walk goes BACKWARDS from the primitives and passes through a node only
    while that node is PRIVATE (or one of the three MFT propagators, whose own
    ``method=`` is the keyword).  It records a public function and stops
    there: a caller of ``compute_psf`` is a caller of ``compute_psf``, not of
    the transform, and its way back is ``compute_psf``'s keyword.  ``ui/`` is
    excluded -- it is an application, not a library entry point.
    """
    callers, defined = _call_graph(mods)
    seed = {(mod, name) for name in _PRIMITIVES
            for mod in defined.get(name, ())}
    assert seed, "the AST sweep found neither MFT primitive; the census is " \
                 "looking in the wrong place"

    def transparent(node):
        mod, name = node
        return ((name.startswith('_') or name in _MFT_PUBLIC)
                and not mod.startswith('ui/'))

    reach, frontier = set(seed), set(seed)
    while frontier:
        nxt = set()
        for node in frontier:
            if not transparent(node):
                continue
            for caller in callers.get(node, ()):
                if caller[0].startswith('ui/') or caller in reach:
                    continue
                reach.add(caller)
                nxt.add(caller)
        frontier = nxt
    return reach


def _forwards_star_kwargs_into_an_mft_call(mods, module, func_name):
    """Does ``func_name`` splat ``**kwargs`` into an MFT propagator call?

    That is the ONE legitimate way back that is not a named parameter:
    ``asm_propagate`` has no ``method`` of its own and forwards
    ``**method_kwargs`` straight into the three MFT propagators, so a caller
    writes ``asm_propagate(..., method='separable')`` and it arrives.  Checked
    structurally rather than assumed.
    """
    tree = mods.get(module)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name):
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                name = (getattr(sub.func, 'id', None)
                        or getattr(sub.func, 'attr', None))
                if name not in _MFT_PUBLIC:
                    continue
                if any(kw.arg is None for kw in sub.keywords):
                    return True
    return False


def _exported_entry_points_without_a_way_back(signature_of=None):
    """``[(name, module)]`` for every exported entry point that can reach a
    primitive and exposes no route keyword.

    ``signature_of`` is injected so the id below can MUTATE one entry point's
    parameter set without touching the library -- the census has to be a
    function of what it reads, or it cannot be falsified.
    """
    if signature_of is None:
        def signature_of(fn):
            return set(inspect.signature(fn).parameters)
    mods = _parse_package()
    reach = _reaching_functions(mods)
    by_name = {}
    for mod, name in reach:
        by_name.setdefault(name, set()).add(mod)
    offenders = []
    for exported in sorted(getattr(lumenairy, '__all__', ())):
        if exported.startswith('_'):
            continue
        obj = getattr(lumenairy, exported, None)
        if not inspect.isfunction(obj):
            continue
        mods_for = by_name.get(obj.__name__)
        if not mods_for:
            continue
        params = signature_of(obj)
        if params & set(_ROUTE_KEYWORDS):
            continue
        if any(_forwards_star_kwargs_into_an_mft_call(mods, m, obj.__name__)
               for m in mods_for):
            continue
        offenders.append((exported, sorted(mods_for)))
    return offenders


def test_every_exported_entry_point_that_reaches_the_mft_has_a_route_keyword():
    """THE CENSUS.  Decided from the source, so a new caller is covered the
    day it lands rather than the day somebody remembers this file.

    MEASURED 2026-09-20: the sweep finds 22 functions that can reach a
    primitive, of which 13 are exported at the top level of ``lumenairy``.
    Twelve of the thirteen name a route with ``method=`` or ``mft_method=``;
    the thirteenth, ``asm_propagate``, has no ``method`` of its own and
    forwards ``**method_kwargs`` into the MFT propagators, which is checked
    structurally above rather than excused here.
    """
    offenders = _exported_entry_points_without_a_way_back()
    assert offenders == [], (
        f"these exported entry points can reach _bluestein_2d / "
        f"_bluestein_centred_2d and expose no route keyword: {offenders}.  "
        f"v5.49.0's shape rule MOVES their answers wherever the output grid "
        f"is at least 32x coarser than the input, and the campaign rule is "
        f"that every moved entry point has a ONE-KEYWORD way back.  Add "
        f"``mft_method=None`` to the signature and forward it with "
        f"``**_mft_route_kwargs(mft_method)``")


def test_the_census_names_the_entry_point_whose_keyword_was_dropped():
    """The census's own falsifier.

    Without this, a census that had silently stopped finding anything would
    read as a perfect result.  Here one entry point's parameter set is
    rewritten to remove the route keyword -- the exact edit a future refactor
    could make -- and the census must name THAT entry point and no other.
    """
    victim = 'resample_field'
    assert hasattr(lumenairy, victim)

    def crippled(fn):
        params = set(inspect.signature(fn).parameters)
        if fn.__name__ == victim:
            params -= set(_ROUTE_KEYWORDS)
        return params

    offenders = _exported_entry_points_without_a_way_back(signature_of=crippled)
    assert [name for name, _mods in offenders] == [victim], (
        f"dropping the route keyword from {victim!r} was reported as "
        f"{offenders}; the census is not reading the signatures it claims to")


# ---------------------------------------------------------------------------
# None stamps nothing, and naming a route reaches the primitive
# ---------------------------------------------------------------------------

WL = 1.31e-6
N_IN = 256
N_CAP = 8                                    # N_IN/32 -- captured by the rule


def _gauss(n, dx, w):
    g = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)).astype(
        np.complex128)


def _rebind_everywhere(monkeypatch, original, replacement):
    """Replace ``original`` with ``replacement`` at EVERY module attribute of
    this package that holds it.

    A module-level ``from .mft import angular_spectrum_propagate_mft`` binds
    the function OBJECT into the importing module's namespace, so patching
    only the defining module would miss ``carrier_field`` and
    ``propagation``.  Rebinding by IDENTITY finds every holder without a list
    that could go stale.
    """
    import sys
    for mod in list(sys.modules.values()):
        if not getattr(mod, '__name__', '').startswith('lumenairy'):
            continue
        if not hasattr(mod, '__dict__'):
            continue
        for attr in list(vars(mod)):
            try:
                if getattr(mod, attr) is original:
                    monkeypatch.setattr(mod, attr, replacement)
            except Exception:                            # noqa: BLE001
                continue


@pytest.fixture
def spy(monkeypatch):
    """Record ``(callee, method=)`` for every call an entry point makes into
    the MFT layer, in order, passing each through untouched.

    WHY THE PUBLIC PROPAGATORS ARE SPIED TOO, and not only the primitives.
    Most of these entry points forward through
    ``angular_spectrum_propagate_mft`` / ``fraunhofer_propagate_mft``, whose
    OWN ``method`` parameter defaults to ``'auto'`` and is passed on
    unconditionally.  By the time a primitive is reached the sentinel has
    already been resolved, and a spy placed there cannot tell "the caller
    named nothing" from "the caller named 'auto'" -- which is precisely the
    distinction the ``None`` default exists to keep.  The FIRST recorded
    event is the call the entry point itself makes, and that is where the
    claim lives.
    """
    from lumenairy.propagators import mft as M
    seen = []

    def wrap(label, original):
        def spied(*args, **kwargs):
            seen.append((label, kwargs.get('method', '<absent>')))
            return original(*args, **kwargs)
        return spied

    for name in _PRIMITIVES:
        monkeypatch.setattr(B, name, wrap(name, getattr(B, name)))
    for name in _MFT_PUBLIC:
        original = getattr(M, name)
        _rebind_everywhere(monkeypatch, original, wrap(name, original))
    return seen


def _drive(name, kw):
    """Call one entry point at a CAPTURED shape, with ``kw`` merged in."""
    if name == 'compute_psf':
        dx = 4e-6
        lumenairy.compute_psf(_gauss(N_IN, dx, N_IN * dx / 6.0), WL, 50e-3,
                              dx, N_psf=N_CAP, method='mft', dx_psf=2e-6,
                              **kw)
    elif name == 'resample_field':
        dx = 1e-6
        lumenairy.resample_field(_gauss(N_IN, dx, N_IN * dx / 6.0), dx,
                                 dx * N_IN / N_CAP, N_out=N_CAP,
                                 method='chirpz', **kw)
    elif name == 'propagate':
        dx = 2e-6
        lumenairy.propagate(_gauss(N_IN, dx, N_IN * dx / 6.0), z=2e-3,
                            wavelength=WL, dx=dx, method='asm',
                            output_grid=(N_CAP, 0.5e-6), **kw)
    elif name == 'carrier_referenced_focus_readout':
        rmag, na = 20.0e-3, 0.05
        w = na * rmag
        dx = 8.0 * w / N_IN
        lumenairy.carrier_referenced_focus_readout(
            _gauss(N_IN, dx, w), -rmag, rmag, WL, dx, dx_out=0.25e-6,
            N_out=N_CAP, on_replica='ignore',
            on_focus_containment='ignore', **kw)
    elif name == 're_reference':
        from lumenairy.propagators.carrier_field import (
            CarrierField, CarrierSpec, FieldGrid, re_reference)
        dx, R = 0.3e-6, -5.0e-4
        fa = CarrierField(_gauss(N_IN, dx, 20e-6),
                          FieldGrid((N_IN, N_IN), dx), CarrierSpec(R=R), WL)
        re_reference(fa, CarrierSpec(R=R * 1.1),
                     FieldGrid((N_CAP, N_CAP), N_IN * dx / N_CAP),
                     on_nyquist='ignore', on_window='ignore', **kw)
    else:                                                # pragma: no cover
        raise AssertionError(f"no driver for {name!r}")


#: The entry points driven in process here.  The two carrier readouts' exact
#: arm and the traced chain are covered archive-to-archive by the probe
#: instead -- they cost seconds each, and what this id adds over the probe is
#: the KEYWORD's arrival, which these five already witness on both doors
#: (``_bluestein_2d`` through ``propagate``/``compute_psf``,
#: ``_bluestein_centred_2d`` through the rest).
_DRIVEN = ('compute_psf', 'resample_field', 'propagate',
           'carrier_referenced_focus_readout', 're_reference')


@pytest.mark.parametrize('name', _DRIVEN)
def test_mft_method_none_stamps_nothing_on_the_call_it_makes(name, spy):
    """``None`` is not ``'auto'``.  It means the keyword is left OFF.

    The difference is invisible in the bytes today -- ``'auto'`` IS the
    callee's default -- and it is the whole reason the default is a sentinel:
    a caller who named nothing must keep whatever that default becomes, while
    a caller who wrote ``mft_method='auto'`` pinned the name.  Asserted on the
    recorded CALL, because the bytes cannot tell the two apart.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _drive(name, {})
        silent = list(spy)
        spy.clear()
        _drive(name, {'mft_method': None})
        with_none = list(spy)
    assert silent, (
        f"{name} made no call into the MFT layer at a captured shape; this "
        f"id is measuring nothing")
    assert silent[0][1] == '<absent>', (
        f"{name} stamped method={silent[0][1]!r} on its {silent[0][0]} call "
        f"although the caller named no route")
    assert with_none == silent, (
        f"{name} with mft_method=None recorded {with_none} against {silent} "
        f"with the keyword absent entirely.  ``None`` must be inert: the "
        f"keyword is left OFF the call, so the callee's own default governs")


@pytest.mark.parametrize('name', _DRIVEN)
def test_naming_a_route_reaches_the_mft_layer(name, spy):
    """The two-sided arm.  Without it, an entry point that accepted
    ``mft_method=`` and threw it away would pass the id above."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _drive(name, {'mft_method': 'bluestein'})
    assert spy, f"{name} made no call into the MFT layer at a captured shape"
    assert spy[0][1] == 'bluestein', (
        f"{name} was asked for mft_method='bluestein' and its {spy[0][0]} "
        f"call carried method={spy[0][1]!r} -- the keyword is accepted and "
        f"dropped somewhere between the signature and the call")


def test_the_route_keyword_helper_is_the_only_spelling_of_the_sentinel():
    """``_mft_route_kwargs`` is where ``None`` becomes "no keyword", in ONE
    place, so eight call sites cannot drift apart about what the sentinel
    means."""
    assert B._mft_route_kwargs(None) == {}
    for spelling in ('auto', 'bluestein', 'separable', 'direct'):
        assert B._mft_route_kwargs(spelling) == {'method': spelling}


# ---------------------------------------------------------------------------
# The archive-to-archive evidence
# ---------------------------------------------------------------------------

_PROBE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))),
    'validation', 'probe_c4_round2')


@pytest.mark.parametrize('build', ('win', 'wsl'))
def test_the_committed_evidence_gives_every_moved_entry_point_a_way_back(build):
    """The committed archive-to-archive readings, asserted rather than quoted.

    ``r2_entry.py`` drives each entry point on a ``git archive 49ddf4bd`` tree
    and on this branch, at a captured shape and at a refused one;
    ``r2_entry_compare.py`` reduces that to three facts per entry point.  This
    id refuses a JSON in which any entry point moved without a way back, or in
    which an entry point silently stopped moving (which would mean the driver
    no longer reaches the rule and the row proves nothing).
    """
    path = os.path.join(_PROBE, f"r2_entry_compare_{build}.json")
    assert os.path.exists(path), (
        f"{path} is missing; the archive-to-archive evidence for "
        f"{build} is not committed")
    with open(path, encoding='cp1252') as fh:
        data = json.load(fh)
    assert data['rows'], "the comparison recorded no entry points"
    no_way_back = [r['entry_point'] for r in data['rows']
                   if not r['way_back']]
    assert no_way_back == [], (
        f"{build}: {no_way_back} moved between 49ddf4bd and this branch and "
        f"no mft_method spelling reproduced the base bytes")
    not_moving = [r['entry_point'] for r in data['rows']
                  if not r['moves_at_captured']]
    assert not_moving == [], (
        f"{build}: {not_moving} did not move at the captured shape, so those "
        f"rows do not witness the rule and their way-back reading is vacuous")
    wider = [r['entry_point'] for r in data['rows']
             if not r['identical_at_refused']]
    assert wider == [], (
        f"{build}: {wider} also moved at a REFUSED shape, so the flip is "
        f"wider than the rule says")

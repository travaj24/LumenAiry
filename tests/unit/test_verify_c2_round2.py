"""VERIFY-WP-C2 ROUND 2 -- decision tests for the gaps the ROUND-2
verification found.

Round 2 closed twelve defects.  Re-measuring them found five things the
round-2 work leaves open, and each one gets an arm here.  As in round 1,
every bar is an identity, a quantity this build measures for itself, or a
census pinned so it can only move deliberately; recorded readings appear in
comments with their date and build.

Evidence: ``validation/probe_verify_c2_round2/`` (both builds) and
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C2_ROUND2.md``.

| arm | the gap it closes |
|-----|-------------------|
| ``..._an_aliased_tracer_import_hides_an_exported_entry_point`` | both censuses decide "this body traces" by looking for the NAME ``trace``; ``_lens_real.py`` imports it as ``_rt_trace`` and ``apply_real_lens`` is invisible to them (VR2-D1) |
| ``..._the_history_drift_envelope_needs_two_n_eps`` | ``n_surfaces * eps`` is called THE BOUND and is exceeded 1.33x on an ordinary 3-surface stack (VR2-D2) |
| ``..._library_trace_default_agrees_with_trace_for_every_keyword`` | the helper is pinned for ``sphere_normal`` only; a mutant that returns the wrong ``renormalize`` survives the whole C2 suite (VR2-D3) |
| ``..._the_forward_reaches_the_tracer_in_the_seven_unparametrized_entry_points`` | the in-process forwarding pin covers 9 of the 16; dropping the forward in any of the other 7 survives (VR2-D4) |
| ``..._the_w6a2_second_step_bar_misses_the_dropped_prior_term`` | the 1e-4 bar's message names three failure modes and one of them reads 2.5e-05 (VR2-D5) |
| ``..._the_edited_in_place_override_refuses_the_three_round_one_abuses`` | D7's closure, re-derived with this verification's own doctoring |
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import pathlib
import sys

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace.surface import RayBundle, Surface
from lumenairy.raytrace.trace import (
    _library_trace_default, _WAY_BACK_KEYWORDS, trace)

REPO = pathlib.Path(__file__).resolve().parents[2]
WL = 632.8e-9
EPS = float(np.finfo(np.float64).eps)

#: The names a census has to recognise as "this body traces".
_TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
            'trace_jax', 'trace_jax_world'}

#: Exported functions that reach a tracer through an ALIASED import and
#: carry no way back.  Measured 2026-09-20 (VR2-D1); the requested fix is
#: either the keyword pair on ``apply_real_lens`` or the alias removed.
#: Shrinking this set is the fix; GROWING it is the regression.
_ALIAS_HIDDEN_WITHOUT_A_WAY_BACK = {'apply_real_lens'}


# ---------------------------------------------------------------- helpers

def _s(R, th, gb, ga, sd=np.inf, conic=0.0):
    return Surface(radius=R, conic=conic, thickness=th, glass_before=gb,
                   glass_after=ga, semi_diameter=sd)


def _bundle(n, hmax, tilt_deg, seed):
    rng = np.random.default_rng(seed)
    r = hmax * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    L = np.full(n, np.sin(np.radians(tilt_deg)))
    return RayBundle(x=r * np.cos(th), y=r * np.sin(th), z=np.zeros(n),
                     L=L, M=np.zeros(n),
                     N=np.sqrt(np.maximum(1.0 - L ** 2, 0.0)),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _prescription():
    return {
        'name': 'vr2 spherical doublet',
        'aperture_diameter': 0.0280,
        'surfaces': [
            {'radius': 0.0731, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.0437, 'conic': 0.0,
             'glass_before': 'N-BK7', 'glass_after': 'N-SF5'},
            {'radius': -0.1013, 'conic': 0.0,
             'glass_before': 'N-SF5', 'glass_after': 'air'},
        ],
        'thicknesses': [0.0082, 0.0031],
    }


def _sibling_module(name):
    """Import a test module that sits next to this one, without relying on
    ``tests/unit`` being on ``sys.path`` (it is under ``-p no:randomly``
    but not under every invocation)."""
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ('cp1252', 'utf-8'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')


# ===========================================================================
# VR2-D1 -- an ALIASED tracer import hides an exported entry point
# ===========================================================================

def _alias_map(tree):
    """``{local name: tracer}`` for every ``from ... import trace as X`` in
    this module, at module level OR inside a function body.

    This is the shape both censuses miss.  ``lumenairy/elements/_lens_real.py``
    does the import INSIDE ``_apply_real_lens_impl``::

        from ..raytrace import (..., trace as _rt_trace)

    so a walk that looks for the NAME ``trace`` in a function body sees
    ``_rt_trace`` and reports "this function does not trace".
    """
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name in _TRACERS and a.asname:
                    out[a.asname] = a.name
    return out


def _alias_aware_census():
    """``{module:function: [tracer names reached]}`` for every function whose
    body reaches a tracer under ANY name, aliases resolved."""
    root = pathlib.Path(la.__file__).parent
    found = {}
    for f in sorted(root.rglob('*.py')):
        rel = f.relative_to(root).as_posix()
        mod = 'lumenairy.' + rel[:-3].replace('/', '.')
        if mod.endswith('.__init__'):
            mod = mod[:-len('.__init__')]
        try:
            tree = ast.parse(_read(f))
        except SyntaxError:                       # pragma: no cover
            continue
        aliases = _alias_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hits = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    if sub.id in _TRACERS:
                        hits.add(sub.id)
                    elif sub.id in aliases:
                        hits.add(aliases[sub.id] + ' (as %s)' % sub.id)
                elif isinstance(sub, ast.Attribute) and sub.attr in _TRACERS:
                    hits.add(sub.attr)
            if hits:
                found['%s:%s' % (mod, node.name)] = sorted(hits)
    return found


def test_vr2_an_aliased_tracer_import_hides_an_exported_entry_point():
    """VR2-D1.  Both entry-point censuses -- the shipped
    ``_c2_entry_point_census`` and VERIFY-WP-C2's own -- decide "this body
    traces" by looking for the literal NAME ``trace``.
    ``lumenairy/elements/_lens_real.py`` imports the tracer as ``_rt_trace``
    inside ``_apply_real_lens_impl``, so the exported ``apply_real_lens``
    -- which calls that helper one hop away whenever
    ``seidel_correction=True`` -- is invisible to both, and to the
    transitive walk that produced the 46-function population in
    ``validation/probe_c2_round2/r2_transitive_parents_*.json``.

    MEASURED 2026-09-20 (``validation/probe_verify_c2_round2/vr2_alias_*.json``,
    both builds): ``apply_real_lens(seidel_correction=True)`` reaches
    ``trace`` once with BOTH keywords omitted; its answer MOVES archive to
    archive at the shipped defaults (max |delta| 2.8389e-13 on Windows,
    2.8387e-13 on WSL, over a 256 x 256 complex field), and the only way
    back is to monkeypatch the tracer, because the signature has neither
    keyword.

    The arm is a CENSUS with a named exemption, not a pin on the defect:
    shrinking ``_ALIAS_HIDDEN_WITHOUT_A_WAY_BACK`` (by adding the keywords,
    or by removing the alias) keeps it green; a NEW alias-hidden tracing
    entry point turns it red.
    """
    # PREMISE 1: the alias really is in the source, at the coordinate the
    # defect names -- so this arm is not passing because the shape is gone.
    src = _read(pathlib.Path(la.__file__).parent / 'elements' / '_lens_real.py')
    assert 'trace as _rt_trace' in src, (
        'the aliased tracer import this arm is about has gone from '
        '_lens_real.py.  If the alias was removed, delete this arm and '
        'the VR2-D1 row of the round-2 verification report; if it moved, '
        're-derive the census below from the new spelling.')

    census = _alias_aware_census()
    aliased = {k: v for k, v in census.items()
               if any('(as ' in t for t in v)}
    assert aliased, (
        'the alias-aware census found no aliased tracer call at all, so it '
        'has stopped resolving aliases and cannot see the defect it '
        'exists for.  Measured 2026-09-20: at least '
        '_lens_real:_apply_real_lens_impl.')

    # PREMISE 2: the NAME-only census -- the one both shipped censuses use
    # -- really does miss it.  This is what makes the arm two-sided.
    name_only = {k for k, v in census.items()
                 if any('(as ' not in t for t in v)}
    assert set(aliased) - name_only, (
        'every aliased tracer call also names a tracer directly, so the '
        'blind spot this arm measures does not exist on this tree.')

    # the exported callers that reach one of those aliased helpers, and
    # their way back
    import lumenairy.elements._lens_real as _lr
    offenders = set()
    for name in dir(la):
        if name.startswith('_'):
            continue
        fn = getattr(la, name, None)
        if not inspect.isfunction(fn):
            continue
        if getattr(fn, '__module__', '') != _lr.__name__:
            continue
        try:
            body = inspect.getsource(fn)
        except (OSError, TypeError):              # pragma: no cover
            continue
        if '_apply_real_lens_impl' not in body:
            continue
        params = inspect.signature(fn).parameters
        if not all(k in params for k in _WAY_BACK_KEYWORDS):
            offenders.add(name)

    assert offenders == _ALIAS_HIDDEN_WITHOUT_A_WAY_BACK, (
        f'exported functions that reach the tracer through the '
        f'_lens_real alias and carry no way back:\n'
        f'  newly without one: '
        f'{sorted(offenders - _ALIAS_HIDDEN_WITHOUT_A_WAY_BACK)}\n'
        f'  no longer in the exempt set: '
        f'{sorted(_ALIAS_HIDDEN_WITHOUT_A_WAY_BACK - offenders)}\n'
        f'VR2-D1: the campaign rule is one keyword per flip on every '
        f'entry point that traces.  This set is the measured exception '
        f'and it is allowed to SHRINK, never to grow.')


def test_vr2_the_alias_hidden_entry_point_really_traces_and_really_moves():
    """The other half of VR2-D1: the hidden entry point is not a
    theoretical one.

    A spy on ``lumenairy.raytrace.trace`` counts the calls
    ``apply_real_lens(seidel_correction=True)`` makes and records the
    keywords each received; the same call with the correction OFF is the
    control and must not trace at all.  Then the tracer is wrapped with the
    pre-WP-C2 keywords forced and the answers compared, which is the way
    back a keyword would have given.

    MEASURED 2026-09-20, both builds: 1 trace call, both keywords
    ``<omitted>``; 0 calls with the correction off; forcing the old
    arithmetic moves the returned field by 2.8389e-13 (Windows) /
    2.8387e-13 (WSL) in max |delta|, against a field whose own scale is
    O(1).
    """
    import lumenairy.raytrace as rt
    from lumenairy.io.prescriptions_builders import make_singlet

    n, dx = 128, 100.0e-6
    a = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(a, a, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / (4.0e-3) ** 2).astype(np.complex128)
    pres = make_singlet(0.0300, -0.0300, 0.0060, 'N-BK7', 0.0200)
    pres['aperture_diameter'] = 0.0180
    kw = dict(prescription=pres, wavelength=587.6e-9, dx=dx,
              seidel_correction=True)

    real = rt.trace
    seen = []

    def spy(*args, **kwargs):
        seen.append((kwargs.get('sphere_normal', '<omitted>'),
                     kwargs.get('renormalize', '<omitted>')))
        return real(*args, **kwargs)

    def forced(*args, **kwargs):
        kwargs.setdefault('sphere_normal', 'generic')
        kwargs.setdefault('renormalize', 'surface')
        return real(*args, **kwargs)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rt.trace = spy
        try:
            base = np.asarray(la.apply_real_lens(E, **kw))
        finally:
            rt.trace = real
        n_on = len(seen)
        seen.clear()
        rt.trace = spy
        try:
            la.apply_real_lens(E, prescription=pres, wavelength=587.6e-9,
                               dx=dx, seidel_correction=False)
        finally:
            rt.trace = real
        n_off = len(seen)
        rt.trace = forced
        try:
            old = np.asarray(la.apply_real_lens(E, **kw))
        finally:
            rt.trace = real

    assert n_on >= 1, (
        'apply_real_lens(seidel_correction=True) no longer reaches the '
        'tracer, so VR2-D1 has been fixed by removing the trace rather '
        'than by adding a keyword.  Measured 1 call on both builds.')
    assert n_off == 0, (
        f'the control traced {n_off} times: apply_real_lens reaches the '
        f'tracer even with seidel_correction=False, so the attribution '
        f'above is not to the Seidel block.  Measured 0 on both builds.')

    delta = float(np.max(np.abs(old - base)))
    scale = float(np.max(np.abs(base)))
    assert scale > 0.0, 'the fixture produced an all-zero field'
    assert delta > 0.0, (
        'forcing the pre-WP-C2 keywords produced a byte-identical field, '
        'so this entry point is not affected by the flip after all and '
        'VR2-D1 is not a defect.  Measured 2.8389e-13 (Windows) / '
        '2.8387e-13 (WSL) on 2026-09-20.')
    # two-sided: the move is a LAST-BIT move, not a broken call
    assert delta < 1e-6 * scale, (
        f'the two normal routes now differ by {delta:.3e} against a field '
        f'scale of {scale:.3e}; that is not a last-bit difference and the '
        f'Seidel block has changed meaning.')


# ===========================================================================
# VR2-D2 -- ``n_surfaces * eps`` is not a bound
# ===========================================================================

def _worst_history_drift(surfs, tilt, seed=20260920, n=1500):
    res = trace(_bundle(n, 0.0090, tilt, seed), surfs, WL,
                output_filter='all')
    worst = 0.0
    for i in range(len(surfs) - 1):
        r = res.rays_at(i)
        m = np.asarray(r.alive, dtype=bool)
        if not m.any():
            continue
        d = np.sqrt(np.asarray(r.L)[m] ** 2 + np.asarray(r.M)[m] ** 2
                    + np.asarray(r.N)[m] ** 2)
        worst = max(worst, float(np.max(np.abs(d - 1.0))))
    return worst


def test_vr2_the_history_drift_envelope_needs_two_n_eps_not_one():
    """VR2-D2.  ``trace``'s ``renormalize`` docstring says
    ``n_surfaces * eps`` "is the BOUND" and tells a consumer to size a
    tolerance from it, and
    ``test_c2_history_bundles_are_not_unit_under_the_new_default`` asserts
    ``worst <= len(S) * eps`` as "the derived envelope, not a reading".

    It is a reading.  It holds on that test's ladder -- the same doublet
    repeated -- and fails on an ordinary 3-surface stack with different
    radii and a different glass.

    MEASURED 2026-09-20 (``validation/probe_verify_c2_round2/vr2_d3_d9_*.json``
    and the sweep in the report), identical on Windows py3.14 / numpy 2.4.4
    and WSL py3.12 / numpy 2.4.6: over 90 (surface count, glass, radius
    pair, field angle) combinations the ratio to ``n_surfaces * eps``
    reaches **1.3333 at three surfaces** and **1.10 at five**, and 6 of the
    90 exceed 1.0.  ``2 * n_surfaces * eps`` holds on every one of them
    with 1.5x to spare.

    This arm is a COMPARISON between two fixtures measured in the same
    process, so nothing here is a recorded number: the shipped ladder must
    satisfy ``n * eps`` (the premise -- otherwise the sibling test would
    already be red and this one is redundant) and the counterexample ladder
    must exceed it while staying inside ``2 n eps``.
    """
    shipped = []
    for _ in range(1):
        shipped += [_s(0.0515, 0.004, 'air', 'N-BK7', 0.0127),
                    _s(-0.0515, 0.006, 'N-BK7', 'air', 0.0127)]
    shipped.append(_s(np.inf, 0.0, 'air', 'air', 0.030))

    counter = [_s(0.0623, 0.0048, 'air', 'N-SF5', 0.0105),
               _s(-0.0771, 0.0083, 'N-SF5', 'air', 0.0105),
               _s(0.3100, 0.0250, 'air', 'N-BK7', 0.0105)]

    assert len(shipped) == len(counter) == 3
    env = 3 * EPS

    d_shipped = _worst_history_drift(shipped, 0.0)
    # the counterexample's worst over the field angles measured
    d_counter = max(_worst_history_drift(counter, t) for t in (0.0, 2.0, 4.0))

    assert 0.0 < d_shipped <= env, (
        f'PREMISE: the shipped ladder must satisfy n * eps, or the '
        f'sibling arm is already red and this comparison says nothing. '
        f'Measured {d_shipped:.4e} against {env:.4e}.')
    assert d_counter > env, (
        f'VR2-D2 has stopped reproducing: the counterexample ladder now '
        f'drifts {d_counter:.4e}, inside the {env:.4e} envelope the '
        f'docstring calls a BOUND.  Measured 8.8818e-16 (ratio 1.3333) on '
        f'both builds on 2026-09-20.  If the tracer changed, re-derive '
        f'the coefficient the docstring should carry.')
    assert d_counter <= 2.0 * env, (
        f'the drift now exceeds even 2 * n_surfaces * eps '
        f'({d_counter:.4e} against {2.0 * env:.4e}); the requested '
        f'docstring bound of 2 n eps is no longer safe either and has to '
        f'be re-derived.  Measured worst ratio 1.3333 over 90 stack, '
        f'glass, radius and field-angle combinations on 2026-09-20.')


# ===========================================================================
# VR2-D3 -- ``_library_trace_default`` is pinned for one keyword only
# ===========================================================================

def test_vr2_library_trace_default_agrees_with_trace_for_every_keyword():
    """VR2-D3.  ``_library_trace_default`` exists so a direct caller of
    ``_refract`` / ``_reflect`` can ASK the library rather than write a
    route down, and its docstring offers it for both switches.  Only
    ``sphere_normal`` is pinned anywhere.

    MEASURED 2026-09-20: a mutant in which ``_library_trace_default``
    returns ``'surface'`` for ``renormalize`` while ``trace`` defaults to
    ``'exit'`` passes the WHOLE C2 suite -- 58 passed, 0 failed, on
    Windows py3.14 (``validation/probe_verify_c2_round2/vr2_mutants_win.txt``,
    mutant M4).  This arm is the pin that mutant needed, and it reads the
    keyword list from the library rather than naming it.
    """
    params = inspect.signature(trace).parameters
    assert set(_WAY_BACK_KEYWORDS), 'the library names no way-back keywords'
    for key in _WAY_BACK_KEYWORDS:
        assert key in params, (
            f'trace no longer has a {key!r} parameter, so the helper '
            f'cannot answer for it.')
        want = params[key].default
        got = _library_trace_default(key)
        assert got == want, (
            f'_library_trace_default({key!r}) answers {got!r} while '
            f'trace defaults to {want!r}.  Every direct caller that asks '
            f'the helper -- analysis.ghost today, anything else tomorrow '
            f'-- would then refract off a route the public tracer does '
            f'not use, which is exactly the defect D5 closed for '
            f'sphere_normal.')
    # and the cache cannot serve a stale answer after the signature moves.
    # (``lumenairy.raytrace.trace`` the ATTRIBUTE is the function, so the
    # module has to come from sys.modules.)
    mod = sys.modules['lumenairy.raytrace.trace']
    cache = dict(mod._LIBRARY_TRACE_DEFAULTS)
    try:
        mod._LIBRARY_TRACE_DEFAULTS.clear()
        for key in _WAY_BACK_KEYWORDS:
            assert _library_trace_default(key) == params[key].default, key
    finally:
        mod._LIBRARY_TRACE_DEFAULTS.clear()
        mod._LIBRARY_TRACE_DEFAULTS.update(cache)


# ===========================================================================
# VR2-D4 -- the forward is pinned in process for 9 of the 16
# ===========================================================================

_UNPARAMETRIZED_SEVEN = (
    'caustic_diagnostic', 'eval_image_plane_wfe', 'plot_lens_layout',
    'fit_canonical_polynomials', 'fit_hf_polynomials',
    'apply_real_lens_traced', 'apply_real_lens_maslov')


@pytest.mark.parametrize('name', _UNPARAMETRIZED_SEVEN)
def test_vr2_the_forward_reaches_the_tracer_in_the_seven_unparametrized_entry_points(
        name):
    """VR2-D4.  ``test_c2_none_stamps_nothing_on_the_entry_points`` is the
    only IN-PROCESS pin that the forwarded keyword actually REACHES the
    internal trace, and it parametrizes nine of the sixteen entry points.
    The other seven are covered only by a committed probe JSON.

    MEASURED 2026-09-20: dropping the forward (``_way_back_kwargs()`` with
    no arguments) in ``elements/_lens_traced.py``,
    ``propagators/asymptotic_canonical_fit.py``,
    ``analysis/image_plane_wfe.py`` or ``analysis/aberration.py`` leaves the
    whole C2 suite green -- **58 passed, 0 failed** on every one of the four
    (``validation/probe_verify_c2_round2/vr2_mutants2_win.txt``).  The
    keyword stays in the signature, so the census passes; it simply goes
    nowhere.

    This arm spies on the tracer instead of comparing answers, so it costs
    one call per entry point: measured 4.4 s for all seven on Windows
    py3.14.  It is two-sided -- the forced keywords must ARRIVE at every
    internal trace call, and the default call must arrive with NEITHER, or
    the sentinel is stamping something.
    """
    import warnings

    P = _prescription()
    seen = []

    def _install(spy_of):
        """Replace every lumenairy module attribute bound to the real
        tracer.  The entry points bind ``trace`` at import time, so
        patching only the defining module reaches the late-binding callers
        alone."""
        patched = []
        for mn, m in list(sys.modules.items()):
            if not mn.startswith('lumenairy') or m is None:
                continue
            if not hasattr(m, '__dict__'):
                continue
            for an in list(vars(m)):
                try:
                    v = getattr(m, an)
                except Exception:                 # pragma: no cover
                    continue
                if id(v) in spy_of:
                    setattr(m, an, spy_of[id(v)])
                    patched.append((m, an, v))
        return patched

    tmod = sys.modules['lumenairy.raytrace.trace']
    wmod = sys.modules['lumenairy.raytrace.world_trace']
    reals = [tmod.trace, wmod.trace_world]

    def _mk(real):
        def spied(*a, **k):
            seen.append((k.get('sphere_normal', '<omitted>'),
                         k.get('renormalize', '<omitted>')))
            return real(*a, **k)
        return spied

    def _call(**kw):
        if name == 'caustic_diagnostic':
            from lumenairy.analysis.aberration import caustic_diagnostic
            return caustic_diagnostic(P, WL, fan_radius=2.5e-3,
                                      n_z_per_gap=6,
                                      z_after_last_surface=0.0800, **kw)
        if name == 'eval_image_plane_wfe':
            from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
            return eval_image_plane_wfe(
                dict(P, object_distance=float('inf')), WL, field=(0.0, 0.4),
                n_pupil=9, field_max_rad=0.022, **kw)
        if name == 'plot_lens_layout':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            from lumenairy.analysis.plotting import plot_lens_layout
            fig, _ax = plot_lens_layout(P, wavelength=WL, show_rays=True,
                                        n_field_angles=2, max_field_deg=1.5,
                                        rays_per_fan=5, **kw)
            plt.close(fig)
            return None
        if name == 'fit_canonical_polynomials':
            from lumenairy.propagators.asymptotic_canonical_fit import (
                fit_canonical_polynomials)
            return fit_canonical_polynomials(P, WL, n_field=3, n_pupil=5,
                                             poly_order=4, **kw)
        if name == 'fit_hf_polynomials':
            from lumenairy.propagators.asymptotic_canonical_fit import (
                fit_hf_polynomials)
            return fit_hf_polynomials(P, WL, n_field=3, n_pupil=5,
                                      poly_order=4, **kw)
        n, dx = 48, 25e-6
        a = (np.arange(n) - n // 2) * dx
        X, Y = np.meshgrid(a, a, indexing='xy')
        E = np.exp(-(X ** 2 + Y ** 2) / (0.35e-3) ** 2).astype(np.complex128)
        if name == 'apply_real_lens_traced':
            from lumenairy.elements import apply_real_lens_traced
            return apply_real_lens_traced(
                E, prescription=P, wavelength=WL, dx=dx, ray_subsample=8,
                bandlimit=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                on_fit_domain_basis='silent', on_pool_memory='silent',
                n_workers=1, **kw)
        from lumenairy.elements import apply_real_lens_maslov
        return apply_real_lens_maslov(
            E, prescription=P, wavelength=WL, dx=dx, ray_field_samples=5,
            ray_pupil_samples=5, poly_order=4, output_subsample=2, **kw)

    # warm the imports BEFORE the patch, so the modules the entry point
    # binds from are already in sys.modules when the spies go in
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _call()

    spy_of = {id(r): _mk(r) for r in reals}
    patched = _install(spy_of)
    assert patched, 'no module attribute was bound to the tracer to patch'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seen.clear()
            _call(sphere_normal='generic', renormalize='surface')
            forced = list(seen)
            seen.clear()
            _call()
            default = list(seen)
    finally:
        for m, an, v in patched:
            setattr(m, an, v)

    assert forced, (
        f'{name}: the tracer was never called, so this arm measured '
        f'nothing.  Measured at least one call per entry point on '
        f'2026-09-20.')
    assert set(forced) == {('generic', 'surface')}, (
        f'{name}: the forwarded keywords did NOT reach every internal '
        f'trace call -- the tracer saw {sorted(set(forced))} over '
        f'{len(forced)} call(s).  A keyword that is accepted and then '
        f'dropped before the trace is the VR2-D4 shape: it leaves the '
        f'signature census green and the way back broken.')
    assert set(default) == {('<omitted>', '<omitted>')}, (
        f'{name}: the DEFAULT call stamped {sorted(set(default))} on the '
        f'tracer.  ``None`` must name nothing, or this release\'s default '
        f'is frozen into the call site and the next flip reaches none of '
        f'the sixteen.')


# ===========================================================================
# VR2-D5 -- the w6_a2 second-step bar and the mode it does not catch
# ===========================================================================

def test_vr2_the_w6a2_second_step_bar_misses_the_dropped_prior_term():
    """VR2-D5.  The ``w6_a2`` tautology was correctly retired for a claim
    that CAN fail: a second Newton step from ``v*`` must be < 1e-4 of the
    first.  Its failure message names three modes -- "a wrong Hessian, a
    missing prior term, a sign error" -- and says all three give "a ratio
    of order 1, four decades above it".

    Measured, two of the three do and one does not.

    MEASURED 2026-09-20 (``validation/probe_verify_c2_round2/vr2_w6a2_*.json``):

        construction                       Windows      WSL
        shipped v*                        1.9355e-07  8.7659e-08
        no step taken at all              1.0000e+00  1.0000e+00
        a sign error in the step          2.0000e+00  2.0000e+00
        half of the first step            5.0000e-01  5.0000e-01
        the PRIOR TERM dropped from H     2.5329e-05  2.5698e-05   <- passes

    The prior term is ``I / w_p**2`` against ``J^T J / w_s**2``, and with
    ``w_s = 20e-6`` against ``w_p = 0.02`` it is six orders smaller, so
    dropping it moves the step by 2.5e-05 of itself -- four times UNDER the
    1e-4 bar.  A bar of 1e-5 catches it and still leaves the shipped
    reading 51x (Windows) / 114x (WSL) of headroom.

    This arm asserts the two halves that stay true whatever the bar becomes:
    the gross modes really are O(1), and the dropped-prior mode really does
    sit between 1e-5 and 1e-4.
    """
    from lumenairy.propagators.asymptotic import (
        _solve_envelope_stationary_batch)

    w6 = _sibling_module('test_niche_audit_w6_asymptotic')
    fit = w6._fit()
    w_s, w_p = 20e-6, 0.02
    v_c = np.array([float(fit.v2x_centre), float(fit.v2y_centre)])
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=v_c[0], v_cy=v_c[1])
    v_star = np.array([float(vx[0]), float(vy[0])])

    def _rH(v, prior=True):
        s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(
            np.asarray(float(fit.s2x_centre)).reshape(()),
            np.asarray(float(fit.s2y_centre)).reshape(()),
            np.asarray(v[0]).reshape(()), np.asarray(v[1]).reshape(()))
        J = np.array([[float(jxx), float(jxy)], [float(jyx), float(jyy)]])
        ds1 = np.array([float(s1x), float(s1y)])
        r = (J.T @ ds1) / w_s ** 2
        H = (J.T @ J) / w_s ** 2
        if prior:
            r = r + (v - v_c) / w_p ** 2
            H = H + np.eye(2) / w_p ** 2
        return r, H

    r_c, H_c = _rH(v_c)
    predicted = -np.linalg.solve(H_c, r_c)
    n1 = float(np.linalg.norm(predicted))
    assert n1 > 0.0, 'the first Newton step is identically zero'

    def ratio_at(v):
        r2, H2 = _rH(np.asarray(v, dtype=float))
        return float(np.linalg.norm(np.linalg.solve(H2, r2))) / n1

    shipped = ratio_at(v_star)
    gross = {
        'no step taken': ratio_at(v_c),
        'half the step': ratio_at(v_c + 0.5 * predicted),
        'sign error': ratio_at(v_c - predicted),
    }
    no_prior_step = -np.linalg.solve(H_c - np.eye(2) / w_p ** 2, r_c)
    dropped_prior = ratio_at(v_c + no_prior_step)

    # PREMISE: the shipped point is converged well inside the bar
    assert shipped < 1e-5, (
        f'the shipped v* now reads {shipped:.4e} of the first step, above '
        f'the 1e-5 this arm needs as a premise (the shipped bar is 1e-4). '
        f'Measured 1.9355e-07 (Windows) / 8.7659e-08 (WSL).')
    # DECISION 1: the gross modes really are O(1)
    assert min(gross.values()) > 0.1, (
        f'a deliberately unconverged expansion point no longer reads O(1): '
        f'{gross}.  The w6_a2 replacement bar is justified by exactly that '
        f'claim; if the gross modes have moved below 0.1 the 1e-4 bar has '
        f'lost its gap and has to be re-derived.  Measured 0.5 to 2.0 on '
        f'both builds.')
    # DECISION 2: and the mode the message names that the bar does NOT catch
    assert 1e-5 < dropped_prior < 1e-4, (
        f'the dropped-prior-term construction reads {dropped_prior:.4e}. '
        f'VR2-D5 measured 2.5329e-05 (Windows) / 2.5698e-05 (WSL): BELOW '
        f'the 1e-4 bar whose message names "a missing prior term" as one '
        f'of the modes it catches, and above 1e-5.  If it has risen past '
        f'1e-4 the message is now true and this arm should go; if it has '
        f'fallen below 1e-5 the suggested 1e-5 bar no longer catches it '
        f'either and the fix has to be re-derived.')


# ===========================================================================
# D7 re-derived -- the three round-one abuses, doctored independently
# ===========================================================================

def test_vr2_the_edited_in_place_override_refuses_the_three_round_one_abuses():
    """D7's closure, re-derived with this verification's own doctoring of
    the tool's line cache rather than with the shipped test's fixtures.

    The three abuses VERIFY-WP-C2 demonstrated -- the default silently
    REVERTED, set to NONSENSE, and left as a STALE COPY at the mapped
    coordinate while the declaration moved -- must each be REFUSED with a
    record that carries both lines; the shipped content and a package
    exactly AT the recorded release must still fire; and a package already
    PAST the recorded release must refuse.

    MEASURED 2026-09-20, Windows py3.14
    (``validation/probe_verify_c2_round2/vr2_reanchor_abuse_win.json``).
    This arm is Windows-only in practice: run from WSL against a Windows
    worktree, ``git show`` cannot resolve the repository at all (the same
    condition that makes the two walker-citation ids red there), so the
    override returns "no opinion" for every case including the shipped one
    and the comparison is vacuous.  That is detected and skipped rather
    than asserted.
    """
    spec = importlib.util.spec_from_file_location(
        'vr2_reanchor_tool', REPO / 'scripts' / 'reanchor_citations.py')
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    path, base_num, base = 'lumenairy/raytrace/trace.py', 61, 'f4f18851'
    assert (path, base_num) in T.EDITED_IN_PLACE, (
        'the map entry this arm abuses has gone; re-derive it from the '
        'current EDITED_IN_PLACE.')
    real_lines = list(T.lines(path))
    shipped = real_lines[base_num - 1]

    def run(mutate, version=None):
        T._cache.clear()
        T.EDITED_IN_PLACE_REFUSALS.clear()
        base_src = list(T.lines(path, base))
        T._cache[(path, None)] = mutate(list(real_lines))
        T._cache[(path, base)] = base_src
        real_ver = T._source_version
        if version is not None:
            T._source_version = lambda: version
        try:
            num, _how = T._edited_in_place(path, base_num, base)
        finally:
            T._source_version = real_ver
        return num, list(T.EDITED_IN_PLACE_REFUSALS)

    fired, _ = run(lambda h: h)
    if fired is None:
        pytest.skip(
            'the re-anchor tool cannot read the base commit from this '
            'mount (git cannot resolve the repository), so every case '
            'returns "no opinion" and the abuses below would pass '
            'vacuously.  This is the WSL-against-a-Windows-worktree '
            'condition documented on the two walker-citation ids.')

    def _sub(h, text):
        h = list(h)
        h[base_num - 1] = text
        return h

    # --- the three abuses, each REFUSED with both lines printed
    for label, mutate in (
            ('the default silently reverted',
             lambda h: _sub(h, "    sphere_normal: str = 'generic',")),
            ('the default set to nonsense',
             lambda h: _sub(h, "    sphere_normal: str = 'not-a-route',")),
            ('a stale copy left at the mapped line',
             lambda h: _sub(list(h[:base_num + 2]) + [shipped]
                            + list(h[base_num + 2:]),
                            "    sphere_normal: str = 'generic',"))):
        num, refusals = run(mutate)
        assert num is None, (
            f'the override still fires for {label}; D7 pinned the '
            f'content digest precisely so it would not.')
        assert len(refusals) == 1, (
            f'{label} was refused SILENTLY ({len(refusals)} records); the '
            f'refusal is the most important thing the tool can say and it '
            f'has to be reported, not swallowed.')
        r = refusals[0]
        assert r['base_line'] and r['found_line'], (
            f'{label}: the refusal record does not carry both lines.')
        assert r['expected_digest'] != r['found_digest'], (
            f'{label}: the refusal was not decided on the content digest.')

    # --- and the two-sided half: the shipped content fires, at the
    #     recorded release and below it, and refuses past it
    num, refusals = run(lambda h: h,
                        version=T.EDITED_IN_PLACE[(path, base_num)][3])
    assert num == base_num and not refusals, (
        'the override refuses the SHIPPED content at exactly the release '
        'it was recorded for, so it can never fire and the guard is a '
        'permanent red rather than a check.')
    num, refusals = run(lambda h: h, version='9.99.0')
    assert num is None and len(refusals) == 1, (
        'the override still fires when the package is releases past the '
        'one the entry was recorded for; D7 asked for exactly that '
        'version pin.')

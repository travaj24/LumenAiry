"""WP-B12b ROUND 2 -- the two LIBRARY defects VERIFY-WP-B12b filed, closed.

Added 2026-09-19, closing
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B12b.md``
(verdict SHIP, three edits requested).

**D-5 -- an IMMERSED exit was reachable from the public API and served
silently.**  ``propagators.gbd.apply_prescription_persurface_to_beamlets``'s
local branch adds its image-side leg as ``t = z_image / Nz2``, which carries
no exit index, while the projection that produced ``dt.opd`` resolves one.
VERIFY-WP-B12b measured **1846.26 waves** of missing optical path at a 2 mm
leg with ``n_exit = 1.72``, reachable through ``apply_real_lens_gbd``, with
the returned phase matching the VACUUM prediction to 7.3e-05 waves.  The four
``propagators.fga`` sites already refuse that class (WAVE5-E); GBD was the
last unguarded consumer.  It now reaches the SAME guard --
``fga._require_non_immersed_exit``, whose tolerance has exactly ONE
definition in ``fga._immersed_exit_tolerance`` -- so there is one boundary
and not two.  The ids below bisect the GBD CALL SITE (not the helper) and
prove the boundary it decides on IS the helper's own return, two-sided at
1.01x and 0.99x.

**D-4 -- a MIRROR-terminated prescription through the LOCAL branch returned a
wrong field silently.**  ``Nz2 = 1/sqrt(1+u^2)`` is positive whatever the true
``N``, and it feeds the returned direction, the leg length AND the Moebius
step, so after a mirror all three run along ``+z`` while the light travels
toward ``-z``: measured spot RMS **7756x** the traced one at ``z_image = +f``,
and a **0.48-wave** piston at ``z_image = -f`` where the positions are right.
WP-B12b's own open item recommended refusing with a message naming
``world_output_plane``; VERIFY-WP-B12b measured that that branch refuses a
CURVED terminating fold itself, so the recommended remedy is a dead end.  What
ships is a REFUSAL that says so, and distinguishes the FLAT case (which the
world branch does serve, through an explicit plane -- measured here).

**D-2 -- one of WP-B12b's own assertions could not fail.**  ``"'radius'" not
in fn_flat`` was searched on a token stream that drops STRING tokens.  The
repaired check searches the source text through
``test_audit2609_b12b_gbd_projection.last_surface_radius_reads``; this file
runs BOTH forms against the committed PRE source
(``validation/probe_wp_b12b_round2/pre_b12_beamlet_fn.py.txt``, commit
1218b24f, which literally reads the last surface's radius) and asserts the old
form passes and the new one fails.

**D-6 -- GBD field digests depend on ``LUMENAIRY_MEM_BUDGET_MB``** through the
public entry.  Asserted here as a property (the bytes move, the field does
not), and pinned in every byte-identity id in this file and in both B12b
files.

Every bar is derived at run time from a quantity the running build measures,
every contrast is premise-gated, and the two guard ids have a companion that
DELETES the guard from the shipped function's own source and asserts the named
check goes red.  No wall-clock assertion.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import fga as F
from lumenairy.propagators import gbd as G

_REPO = Path(__file__).resolve().parents[2]
_PROBE = _REPO / 'validation' / 'probe_wp_b12b_round2'
_PRE_SRC = _PROBE / 'pre_b12_beamlet_fn.py.txt'
_MUTATE = _REPO / 'validation' / 'probe_verify_b12b' / 'vb12b_mutate.py'
_FN = 'apply_prescription_persurface_to_beamlets'

#: VERIFY-WP-B12b D-6: a GBD field's SHA-256 depends on the memory budget
#: through the public entry, and the environment variable is a CEILING on the
#: keyword, so a byte-identity id must pin BOTH.  Measured in
#: ``validation/probe_wp_b12b_round2/probe_r3_budget.py`` on both builds.
_MEM_BUDGET_MB = 2048.0

#: The optic every field id here uses -- a biconvex singlet in a
#: dispersionless MODEL glass, CONIC last surface (nothing in this file
#: depends on the sag repair), 633 nm.  96 x 4.0 um spans 0.384 mm against a
#: 0.400 mm clear aperture.  The size is a COST choice: every id in this file
#: is well inside the 60 s budget with the box loaded (measured 2026-09-19).
_LAM = 633e-9
_SEMI = 0.20e-3
_W0 = 0.12e-3
_N, _DX = 96, 4.0e-6
_GLASS, _GLASS_N = 'R2B-M158', 1.58
#: Two real immersion media and the verifier's own n = 1.72, registered
#: probe-locally so the registry resolves them exactly and the premise ids
#: read a number rather than a catalogue fit.
_MEDIA = {'R2B-WATER133': 1.333, 'R2B-OIL152': 1.5180, 'R2B-T172': 1.72}
#: The beamlet frame, named so the three public entry points are comparable
#: byte for byte (they have different sampling DEFAULTS).
_FRAME = dict(sample_step=4, waist_factor=4.0)


@pytest.fixture(autouse=True)
def _pin_mem_budget(monkeypatch):
    """Every id in this file runs at ONE memory budget (D-6) unless it is the
    id that varies it, so a digest here is a property of the library and not
    of the shell that invoked it.

    ``propagate_gbd_through_prescription`` has NO ``mem_budget_mb`` keyword at
    all, so for that entry point the environment variable is the ONLY pin --
    which is why this is a fixture and not a keyword passed at each call site.
    """
    monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', str(int(_MEM_BUDGET_MB)))


def _register():
    from lumenairy import glass as _g
    _g.GLASS_REGISTRY[_GLASS] = lambda wl: _GLASS_N
    for name, n in _MEDIA.items():
        _g.GLASS_REGISTRY[name] = (lambda v: (lambda wl: v))(n)


def _presc(exit_glass='air', *, mirror=False, R2=-4.0e-3, semi=_SEMI):
    """A biconvex singlet, or a bare terminating MIRROR."""
    _register()
    if mirror:
        s = {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
             'glass_before': 'air', 'glass_after': 'MIRROR',
             'semi_diameter': semi}
        return {'name': 'r2b_mirror', 'aperture_diameter': 2 * semi,
                'surfaces': [s], 'thicknesses': [0.0], 'stop_index': 0}
    s0 = {'radius': 4.0e-3, 'conic': 0.0, 'thickness': 0.8e-3,
          'glass_before': 'air', 'glass_after': _GLASS,
          'semi_diameter': semi}
    s1 = {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
          'glass_before': _GLASS, 'glass_after': exit_glass,
          'semi_diameter': semi}
    return {'name': 'r2b_singlet', 'aperture_diameter': 2 * semi,
            'surfaces': [s0, s1], 'thicknesses': [0.8e-3], 'stop_index': 0}


def _flat_last_presc(exit_glass='air'):
    """Plano-convex with the CURVED side FIRST: the projection short-circuits
    structurally, so this is the byte-identity control."""
    return _presc(exit_glass, R2=float('inf'))


def _fold_then_flat_presc():
    """A FLAT fold mirror in the MIDDLE, terminating in a transmissive flat.

    The class the mirror guard deliberately does NOT cover: it is what
    ``world_output_plane`` exists for, and it is loud rather than silent
    through the local branch.  Kept as the guard's scope control.
    """
    _register()
    return {'surfaces': [
        {'radius': 4.0e-3, 'glass_before': 'air', 'glass_after': _GLASS,
         'semi_diameter': _SEMI, 'surf_num': 1},
        {'radius': float('inf'), 'glass_before': _GLASS, 'glass_after': 'air',
         'semi_diameter': _SEMI, 'surf_num': 2},
        {'radius': float('inf'), 'glass_before': 'air',
         'glass_after': 'MIRROR', 'semi_diameter': 2 * _SEMI, 'surf_num': 15},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
         'semi_diameter': 2 * _SEMI, 'surf_num': 25}],
        'thicknesses': [0.8e-3, 1.0e-3, 0.0, 0.0],
        'aperture_diameter': 2 * _SEMI,
        'coord_breaks': [
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': -1.0e-3}]}


def _E_in(n=_N, dx=_DX, w0=_W0):
    xs = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a))).hexdigest()


def _surfs(presc):
    from lumenairy.raytrace import surfaces_from_prescription
    return list(surfaces_from_prescription(presc))


def _n_exit(presc):
    return float(la.raytrace.exit_vertex.resolve_exit_index(
        _surfs(presc), _LAM, fn_name='wp_b12b_round2'))


def _bundle():
    return G.decompose_field_to_beamlets(_E_in(), _DX, wavelength=_LAM,
                                         **_FRAME)


def _gbd_field(presc, z, **extra):
    kw = dict(_FRAME)
    kw.setdefault('mem_budget_mb', _MEM_BUDGET_MB)
    kw.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            _E_in(), prescription=presc, wavelength=_LAM, dx=_DX,
            output_plane_distance=float(z), **kw))


# ===========================================================================
# The entry points onto the LOCAL branch, as callables, so each id can be
# stated once and run at every one of them.
# ===========================================================================
def _enter(entry, presc, z):
    """Run ``entry`` on ``presc``; return whatever it returns."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if entry == 'beamlet_function':
            return G.apply_prescription_persurface_to_beamlets(
                _bundle(), presc, _LAM, z_image=z).positions
        if entry == 'apply_real_lens_gbd':
            return _gbd_field(presc, z)
        if entry == 'apply_real_lens_universal_gbd':
            return np.asarray(la.apply_real_lens_universal(
                _E_in(), prescription=presc, wavelength=_LAM, dx=_DX,
                output_plane_distance=z, method='gbd',
                method_kwargs={'gbd': dict(_FRAME,
                                           mem_budget_mb=_MEM_BUDGET_MB)}))
        if entry == 'propagate_gbd_through_prescription':
            return np.asarray(G.propagate_gbd_through_prescription(
                _E_in(), _DX, presc, wavelength=_LAM, per_surface=True,
                z_image=z, output_shape=(_N, _N), output_dx=_DX,
                **_FRAME))
    raise AssertionError(f'unknown entry {entry!r}')


_ENTRIES = ('beamlet_function', 'apply_real_lens_gbd',
            'apply_real_lens_universal_gbd',
            'propagate_gbd_through_prescription')


# ===========================================================================
# The two named CHECKS.  Each is called both by its own id and by the id that
# DELETES the guard from the shipped source, so "deleting the guard reddens a
# named test" is proven with the very assertion that names it.
# ===========================================================================
def check_an_immersed_exit_is_refused(entry, glass, z):
    """The D-5 decision at one entry point, on one immersed medium."""
    presc = _presc(glass)
    n = _n_exit(presc)
    assert abs(n - _MEDIA[glass]) < 1e-12, (
        f'premise: {glass} must resolve to {_MEDIA[glass]!r}; got {n!r}')
    with pytest.raises(NotImplementedError) as ei:
        _enter(entry, presc, z)
    msg = str(ei.value)
    assert msg.startswith(_FN + ':'), (
        f'the refusal must name the GBD site that refused, not an upstream '
        f'one; it reads {msg[:120]!r}')
    assert 'IMMERSED' in msg, (
        f'the refusal must name the class it refuses; it reads {msg[:200]!r}')
    return msg


def check_a_mirror_terminated_prescription_is_refused(presc, z):
    """The D-4 decision on one mirror-terminated prescription."""
    surfs = _surfs(presc)
    assert bool(getattr(surfs[-1], 'is_mirror', False)), (
        'premise: this fixture must terminate in a mirror')
    with pytest.raises(NotImplementedError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.apply_prescription_persurface_to_beamlets(
                _bundle(), presc, _LAM, z_image=z)
    msg = str(ei.value)
    assert msg.startswith(_FN + ':'), (
        f'the refusal must name the GBD site that refused; it reads '
        f'{msg[:120]!r}')
    assert 'MIRROR' in msg, (
        f'the refusal must name the class it refuses; it reads {msg[:200]!r}')
    return msg


# ===========================================================================
# The mutation vehicle: delete a guard from the SHIPPED function's own source
# and rebind it, so the fail-before arm runs the library's code minus one
# statement rather than a wrapper around it.  VERIFY-WP-B12b's own vehicle is
# reused (``validation/probe_verify_b12b/vb12b_mutate.py``) so there is one
# recompile-and-rebind implementation and not two.
# ===========================================================================
def _mutate_module():
    spec = importlib.util.spec_from_file_location('_vb12b_mutate_r2', _MUTATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _delete_statements(src, names):
    """Remove every statement of the function whose source mentions ``names``.

    Formatting-independent: the statements are located with ``ast`` and their
    line ranges blanked, so a reflow of the guard call cannot silently make
    this mutation a no-op (it raises instead).
    """
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == _FN)
    lines = src.splitlines(keepends=True)
    kill = set()
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Expr, ast.ImportFrom, ast.Assign)):
            continue
        seg = ''.join(lines[node.lineno - 1:node.end_lineno])
        if any(nm in seg for nm in names):
            kill.update(range(node.lineno - 1, node.end_lineno))
    if not kill:
        raise AssertionError(
            f'the mutation matched no statement for {names!r}: the guard call '
            f'has been reshaped and this fail-before arm is vacuous')
    return ''.join('' if i in kill else ln for i, ln in enumerate(lines))


class _GuardDeleted:
    """Context manager: the shipped function minus the named guard call(s)."""

    def __init__(self, *names):
        self.names = names

    def __enter__(self):
        import lumenairy.elements.lenses_gbd as L
        self._mods = [m for m in (G, L, la) if hasattr(m, _FN)]
        self._orig = getattr(G, _FN)
        mut = _mutate_module()
        mut._recompile(lambda s: _delete_statements(s, self.names))
        return self

    def __exit__(self, *exc):
        for m in self._mods:
            setattr(m, _FN, self._orig)
        return False


_IMMERSED_GUARD = ('_require_non_immersed_exit',)
_MIRROR_GUARD = ('_require_forward_going_local_exit',)

#: What "the named check goes RED" looks like from inside another test.
#: ``pytest.raises(...)`` failing to see its exception raises
#: ``pytest.fail.Exception``, which derives from ``BaseException`` and NOT
#: from ``Exception``, so a fail-before arm that catches ``Exception`` catches
#: nothing and reports the red it was trying to demonstrate as its own
#: failure (measured 2026-09-19 while writing this file).
_RED = (AssertionError, pytest.fail.Exception)


# ===========================================================================
# 1.  D-5 -- the immersed exit
# ===========================================================================
@pytest.mark.parametrize('glass', sorted(g for g in _MEDIA))
def test_an_immersed_exit_is_reachable_and_the_leg_omits_waves(glass):
    """PREMISE for every D-5 arm, measured rather than assumed.

    The guard is only worth having if (a) the registry resolves this exit
    medium to something that is not 1, (b) an AIR-terminated prescription
    resolves to EXACTLY 1 so the control sits at the tolerance's origin, and
    (c) the optical path the index-free leg omits is large.  All three are
    read from the library here.

    The omission is ``(n_exit - 1) * z_image * sec`` of optical path, i.e. at
    least ``|n - 1| * z / lambda`` waves; the bar is ONE wave, which is three
    decades above the guard's own budget (1e-3 waves) and two below the
    smallest reading this file's fixtures produce.
    """
    presc = _presc(glass)
    n = _n_exit(presc)
    assert abs(n - _MEDIA[glass]) < 1e-12, (
        f'premise: {glass} must resolve to {_MEDIA[glass]!r}; got {n!r}')
    assert _n_exit(_presc('air')) == 1.0, (
        'premise: the AIR control must resolve to EXACTLY 1.0 on this '
        'registry, else the guard could refuse it on rounding')
    z = 2.0e-3
    waves = abs(n - 1.0) * z / _LAM
    assert waves > 1.0, (
        f'premise: the index-free leg must omit more than a wave on this '
        f'fixture; it omits {waves:.3f}')
    budget = F._FGA_IMAGE_LEG_WAVE_BUDGET
    assert waves > 1e3 * budget, (
        f'premise: the omission ({waves:.3g} waves) must sit decades above '
        f'the guard\'s own wave budget ({budget:g}), else this fixture does '
        f'not separate a refusal from a rounding decision')


@pytest.mark.parametrize('entry', _ENTRIES)
def test_an_immersed_exit_is_refused_at_every_local_gbd_entry_point(entry):
    """DECISION (D-5): every public route onto the local branch refuses an
    immersed exit, and the refusal names the GBD site rather than an upstream
    one.

    Four routes, two immersion media (water-like 1.333 and the verifier's own
    1.72), at a 2 mm image leg.  ``apply_real_lens_universal(method='gbd')``
    is a dispatcher and ``propagate_gbd_through_prescription`` a
    propagators-level entry; both must reach the same guard, or the public API
    has a hole the beamlet function does not.
    """
    for glass in ('R2B-WATER133', 'R2B-T172'):
        check_an_immersed_exit_is_refused(entry, glass, 2.0e-3)


@pytest.mark.parametrize('entry', _ENTRIES)
def test_an_air_terminated_prescription_is_not_refused_at_any_entry_point(
        entry):
    """THE OTHER SIDE (D-5).  The guard must not fire on what the library
    actually serves: every GBD fixture in this suite exits into air.

    Only the guard is under test, so the call's numerical outcome is not
    asserted; what is asserted is that no ``NotImplementedError`` naming an
    immersed exit escapes, and that the call returns something finite -- a
    guard that refused everything would also pass a pure no-exception check.
    """
    for presc in (_presc('air'), _flat_last_presc('air')):
        try:
            out = np.asarray(_enter(entry, presc, 2.0e-3))
        except NotImplementedError as exc:                  # pragma: no cover
            raise AssertionError(
                f'the {entry} entry refused an AIR-terminated prescription: '
                f'{exc}') from exc
        assert out.size > 0 and np.isfinite(out).all(), (
            f'{entry} returned a non-finite or empty result on an '
            f'air-terminated prescription')


def test_the_gbd_site_refuses_at_the_fga_helper_s_own_tolerance(monkeypatch):
    """DECISION (D-5): the boundary the GBD SITE decides on IS
    ``fga._immersed_exit_tolerance``'s own return -- ONE definition, reached,
    not resembled.

    Bisected THROUGH THE SHIPPED CALL, not through the helper: the bundle
    handed to the function is a tripwire whose first attribute access raises,
    so a guard decision costs the guard and the 60-step bisection runs the
    real call site.  Three claims:

      1. the site's boundary equals the helper's return to 1e-6 relative at
         every image leg tried (the bisection's own resolution is 2^-60 of
         the bracket, ~9e-19, so the bar is nine decades of slack);
      2. two-sided about that return: 1.01x refused, 0.99x served;
      3. the site READS the helper.  Monkeypatching
         ``fga._immersed_exit_tolerance`` to a different number moves the
         site's boundary to the new number -- which a private copy of the
         arithmetic beside the GBD call could not do.  This is the check that
         VERIFY-WAVE5-E's D3 showed a pin needs: rewriting the formula must be
         visible to the site that uses it.
    """
    import lumenairy.raytrace.exit_vertex as _ev

    class _Served(Exception):
        pass

    class _Tripwire:
        @property
        def positions(self):
            raise _Served()

    presc = _presc('air')

    def refuses(n, z):
        orig = _ev.resolve_exit_index
        _ev.resolve_exit_index = lambda *a, **k: float(n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                G.apply_prescription_persurface_to_beamlets(
                    _Tripwire(), presc, _LAM, z_image=z)
        except _Served:
            return False
        except NotImplementedError:
            return True
        finally:
            _ev.resolve_exit_index = orig
        return False

    def boundary(z):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if refuses(1.0 + mid, z):
                hi = mid
            else:
                lo = mid
        return hi

    tol = F._immersed_exit_tolerance
    for z in (0.0, _LAM, 1.0e-5, 3.5e-4, 2.0e-3, 1.0e-2):
        assert refuses(2.0, z), (
            f'premise: the GBD site must refuse n_exit = 2 at z_image={z!r}')
        assert not refuses(1.0, z), (
            f'premise: the GBD site must serve n_exit = 1 exactly at '
            f'z_image={z!r}')
        want = float(tol(_LAM, z))
        got = boundary(z)
        assert abs(got - want) <= 1e-6 * want, (
            f'at z_image={z!r} the GBD site refuses from |n-1| > {got!r}, but '
            f'fga._immersed_exit_tolerance -- the ONE definition the four '
            f'fga sites reach -- returns {want!r}.  The site must BE the '
            f'helper, not merely resemble it.')
        assert refuses(1.0 + 1.01 * want, z), (
            f'1.01x the helper\'s tolerance ({want!r}) must be refused at '
            f'z_image={z!r}')
        assert not refuses(1.0 + 0.99 * want, z), (
            f'0.99x the helper\'s tolerance ({want!r}) must be served at '
            f'z_image={z!r}')

    # 3.  the site READS the helper: move the helper, the site moves with it.
    z = 2.0e-3
    base = float(tol(_LAM, z))
    shifted = 7.0 * base
    monkeypatch.setattr(F, '_immersed_exit_tolerance',
                        lambda *a, **k: shifted, raising=True)
    moved = boundary(z)
    assert abs(moved - shifted) <= 1e-6 * shifted, (
        f'the GBD site\'s boundary stayed at {moved!r} when '
        f'fga._immersed_exit_tolerance was moved to {shifted!r}: the site is '
        f'using a COPY of the tolerance, not the one definition')
    assert abs(moved - base) > 0.5 * base, (
        'premise: the shifted tolerance must be far enough from the shipped '
        'one that the two cannot be confused')


def test_deleting_the_immersed_guard_from_the_gbd_site_reddens_a_named_check():
    """FAIL-BEFORE (D-5): with ``_require_non_immersed_exit`` deleted from the
    shipped function's own source,
    :func:`check_an_immersed_exit_is_refused` -- the assertion
    ``test_an_immersed_exit_is_refused_at_every_local_gbd_entry_point`` makes
    -- FAILS, and the immersed prescription is served again.

    The mutation is a recompile of the function from its own source with the
    guard statement removed by ``ast`` (so a reflow cannot make it a silent
    no-op) and a rebind everywhere the library re-exports it, through
    VERIFY-WP-B12b's own vehicle.  Two-sided: the MIRROR guard must still
    fire under this mutation, so the two guards are shown to be independently
    reachable and not one check wearing two names.
    """
    check_an_immersed_exit_is_refused('beamlet_function', 'R2B-T172', 2.0e-3)
    with _GuardDeleted(*_IMMERSED_GUARD):
        with pytest.raises(_RED) as ei:
            check_an_immersed_exit_is_refused(
                'beamlet_function', 'R2B-T172', 2.0e-3)
        assert 'DID NOT RAISE' in str(ei.value) or isinstance(
                ei.value, AssertionError), (
            f'with the immersed guard deleted the named check went red for '
            f'the wrong reason: {ei.value}')
        # the OTHER guard is untouched by this deletion
        check_a_mirror_terminated_prescription_is_refused(
            _presc(mirror=True, R2=-8.0e-3), 4.0e-3)
    # restored
    check_an_immersed_exit_is_refused('beamlet_function', 'R2B-T172', 2.0e-3)


# ===========================================================================
# 2.  D-4 -- the mirror-terminated local branch
# ===========================================================================
@pytest.mark.parametrize('label,R2', [('verifier_R15', -15.0e-3),
                                      ('mine_curved_R8', -8.0e-3),
                                      ('mine_flat', float('inf'))])
def test_a_mirror_terminated_prescription_is_refused_on_the_local_branch(
        label, R2):
    """DECISION (D-4): the local branch refuses a mirror-terminated
    prescription instead of returning a field that is wrong by 7.8e3 in spot
    RMS.

    Three fixtures: VERIFY-WP-B12b's own concave mirror (R = -15 mm), a
    different curved one (R = -8 mm) and a FLAT one.  The flat arm matters --
    the defect is the unsigned ``Nz2``, which has nothing to do with the sag,
    so a flat terminating mirror is just as wrong and must be refused too.

    Premise: the traced propagation direction really is reversed, read from
    the library's own ``_exit_direction_sign`` on the surface list, so the arm
    is not asserting a refusal of something that was fine.
    """
    from lumenairy.raytrace import differential as D
    presc = _presc(mirror=True, R2=R2, semi=0.30e-3)
    assert D._exit_direction_sign(_surfs(presc)) == -1.0, (
        'premise: one mirror must reverse the propagation direction')
    msg = check_a_mirror_terminated_prescription_is_refused(presc, 4.0e-3)
    assert 'Nz2' in msg, (
        f'the refusal must name the quantity that is wrong; it reads '
        f'{msg[:200]!r}')


def test_the_mirror_refusal_does_not_send_the_caller_to_a_dead_end():
    """DECISION (D-4, the part WP-B12b got wrong): the message must not name
    ``world_output_plane`` as the remedy for a CURVED terminating mirror,
    because that branch refuses the same class.

    Both premises are MEASURED here on this build rather than quoted:

      * a CURVED terminating mirror through ``world_output_plane`` raises
        ``NotImplementedError`` naming powered fold mirrors -- so there is
        nowhere to send the caller;
      * a FLAT terminating mirror through an EXPLICIT ``(p0, R_out)`` plane is
        SERVED -- so the message is entitled to name that case, and must
        distinguish it rather than collapse the two.

    The assertion on the message is therefore conditional on what the world
    branch actually does, not on what the message says about itself.
    """
    curved = _presc(mirror=True, R2=-15.0e-3, semi=0.30e-3)
    flat = _presc(mirror=True, R2=float('inf'), semi=0.30e-3)
    plane = (np.array([0.0, 0.0, -4.0e-3]), np.eye(3))

    with pytest.raises(NotImplementedError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.apply_prescription_persurface_to_beamlets(
                _bundle(), curved, _LAM, world_output_plane=plane)
    world_msg = str(ei.value).lower()
    assert 'fold' in world_msg and 'mirror' in world_msg, (
        f'premise: the world branch was expected to refuse a CURVED '
        f'terminating fold; it raised {ei.value}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        served = G.apply_prescription_persurface_to_beamlets(
            _bundle(), flat, _LAM, world_output_plane=plane)
    assert np.asarray(served.positions).shape[0] > 0, (
        'premise: the world branch was expected to SERVE a FLAT terminating '
        'mirror through an explicit plane; it returned nothing')

    msg = check_a_mirror_terminated_prescription_is_refused(curved, 4.0e-3)
    low = msg.lower()
    assert 'world_output_plane' in low, (
        'the refusal should say what the world branch does with this class, '
        'one way or the other')
    assert 'curved' in low and 'flat' in low, (
        f'the refusal names world_output_plane without distinguishing the '
        f'CURVED case (which that branch refuses, measured above) from the '
        f'FLAT one (which it serves): {msg!r}')
    i_curved, i_flat = low.index('curved'), low.index('flat')
    assert i_curved < i_flat, (
        'the refusal should state the refused (curved) case before the '
        'served (flat) one, so a caller reading the first sentence is not '
        'sent to the dead end')


def test_a_transmissive_prescription_and_a_mid_prescription_fold_are_served():
    """THE OTHER SIDE (D-4), and the guard's SCOPE, both asserted.

    The guard refuses a mirror-TERMINATED prescription -- the class
    VERIFY-WP-B12b measured.  It must NOT refuse:

      * an ordinary transmissive prescription (curved or flat last surface);
      * a prescription with a fold mirror in the MIDDLE and a transmissive
        last surface.  That class is what ``world_output_plane`` exists for,
        it is loud rather than silent through the local branch
        (``test_gbd_feature_complete.py``'s periscope id pins the local arm's
        non-finite / smeared result), and widening the guard to cover it would
        change behaviour nothing here has measured.  Asserted as scope, with
        the premise that the fixture really does contain a mirror.
    """
    for presc in (_presc('air'), _flat_last_presc('air')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = G.apply_prescription_persurface_to_beamlets(
                _bundle(), presc, _LAM, z_image=2.0e-3)
        assert np.asarray(out.positions).shape[0] > 0

    folded = _fold_then_flat_presc()
    surfs = _surfs(folded)
    assert any(bool(getattr(s, 'is_mirror', False)) for s in surfs), (
        'premise: the scope fixture must contain a fold mirror')
    assert not bool(getattr(surfs[-1], 'is_mirror', False)), (
        'premise: the scope fixture must NOT terminate in a mirror')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = G.apply_prescription_persurface_to_beamlets(
            _bundle(), folded, _LAM, z_image=1.0e-3)
    assert np.asarray(out.positions).shape[0] >= 0, (
        'a mid-prescription fold must still reach the local branch; the '
        'guard is scoped to a mirror-TERMINATED prescription')


def test_the_mirror_guard_looks_past_a_trailing_coordinate_break():
    """DECISION, two-sided: "the last surface" means the last surface that
    carries optics, so a trailing COORDINATE BREAK does not hide a mirror
    from the guard -- and does not invent one after a transmissive surface.

    A coordinate break carries no power and no medium of its own; it is a
    frame change.  ``_last_optical_surface`` therefore skips trailing ones,
    and this id pins both directions of that skip, plus the empty-list case,
    because a helper whose branch nothing reaches is a branch that can rot.

    The coordinate break is built from the library's own ``Surface`` with
    ``is_coordbrk=True``, which is the state ``surfaces_from_prescription``
    produces for one, rather than a stub that merely resembles it.
    """
    import copy as _copy

    from lumenairy.propagators.gbd import (
        _last_optical_surface,
        _require_forward_going_local_exit,
    )
    mirror_surfs = _surfs(_presc(mirror=True, R2=-15.0e-3, semi=0.30e-3))
    trans_surfs = _surfs(_presc('air'))
    assert bool(getattr(mirror_surfs[-1], 'is_mirror', False))
    assert not bool(getattr(trans_surfs[-1], 'is_mirror', False))

    def _cb(from_surf):
        cb = _copy.copy(from_surf)
        cb.is_coordbrk = True
        cb.is_mirror = False
        return cb

    with_cb = list(mirror_surfs) + [_cb(trans_surfs[-1])]
    assert _last_optical_surface(with_cb) is mirror_surfs[-1], (
        'the guard must look past a trailing coordinate break to the last '
        'surface that carries optics')
    with pytest.raises(NotImplementedError, match='MIRROR'):
        _require_forward_going_local_exit(with_cb, _FN)

    # the other side: a coordinate break after a TRANSMISSIVE last surface
    # must not be refused, and must not be mistaken for the optical one
    trans_with_cb = list(trans_surfs) + [_cb(trans_surfs[-1])]
    assert _last_optical_surface(trans_with_cb) is trans_surfs[-1]
    assert _require_forward_going_local_exit(trans_with_cb, _FN) is False, (
        'the guard must return the DECISION False (nothing refused) rather '
        'than merely not raising, so both sides are observable')
    assert _require_forward_going_local_exit(trans_surfs, _FN) is False
    assert _last_optical_surface([]) is None, (
        'an empty surface list must not raise inside the guard'
    )
    assert _require_forward_going_local_exit([], _FN) is False


def test_deleting_the_mirror_guard_from_the_gbd_site_reddens_a_named_check():
    """FAIL-BEFORE (D-4): with ``_require_forward_going_local_exit`` deleted
    from the shipped function's own source,
    :func:`check_a_mirror_terminated_prescription_is_refused` FAILS -- and
    what comes back instead is the wrong field the guard exists to prevent.

    Measured on the served arm rather than described: the returned direction's
    ``z`` sign is ``+1`` where the library's own ``_exit_direction_sign``
    reads ``-1``.  Two-sided: the IMMERSED guard still fires under this
    mutation.
    """
    from lumenairy.raytrace import differential as D
    presc = _presc(mirror=True, R2=-15.0e-3, semi=0.30e-3)
    check_a_mirror_terminated_prescription_is_refused(presc, 4.0e-3)
    with _GuardDeleted(*_MIRROR_GUARD):
        with pytest.raises(_RED) as ei:
            check_a_mirror_terminated_prescription_is_refused(presc, 4.0e-3)
        assert 'DID NOT RAISE' in str(ei.value) or isinstance(
                ei.value, AssertionError), (
            f'with the mirror guard deleted the named check went red for the '
            f'wrong reason: {ei.value}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = G.apply_prescription_persurface_to_beamlets(
                _bundle(), presc, _LAM, z_image=4.0e-3)
        got = float(np.sign(np.median(np.asarray(r.directions)[:, 2])))
        true = D._exit_direction_sign(_surfs(presc))
        assert true == -1.0
        assert got == +1.0 and got != true, (
            f'premise for the refusal: the unguarded branch must return a '
            f'FORWARD-going direction ({got:+.0f}) where the prescription '
            f'sends the light toward {true:+.0f}; it returned {got:+.0f}')
        check_an_immersed_exit_is_refused(
            'beamlet_function', 'R2B-T172', 2.0e-3)
    check_a_mirror_terminated_prescription_is_refused(presc, 4.0e-3)


# ===========================================================================
# 3.  Byte identity where no guard fires
# ===========================================================================
def test_neither_guard_moves_one_byte_of_an_air_terminated_field():
    """INVARIANT: on every prescription this library serves, the two guards
    change nothing -- the returned field is BIT-IDENTICAL to the one the same
    function produces with both guard statements deleted from its source.

    This is the in-process half of the claim; the archive-to-archive half (a
    ``git archive`` of the integration tip 76019ede against this branch, 15
    fixtures x 2 planes x 2 builds) is in
    ``validation/probe_wp_b12b_round2/probe_r2_identity.py`` and in the
    round-2 addendum.  Run at a PINNED memory budget, both the environment
    ceiling and the keyword (D-6), because a GBD digest depends on it.

    Two-sided, and the second half is the one that matters: the mutation must
    reach the SAME code path ``_gbd_field`` uses.  ``apply_real_lens_gbd``
    calls the beamlet function through ``elements.lenses_gbd``'s own
    module-level import, so a mutation that rebound only
    ``propagators.gbd``'s attribute would leave the public entry running the
    ORIGINAL function -- and the byte identity below would be a tautology.
    The id therefore asserts, inside the mutated block, that the PUBLIC entry
    SERVES a fixture the guarded build refuses.
    """
    fixtures = {'conic_last': _presc('air'),
                'flat_last': _flat_last_presc('air')}
    shipped = {k: _gbd_field(p, 2.0e-3) for k, p in fixtures.items()}
    immersed = _presc('R2B-T172')
    with pytest.raises(NotImplementedError):
        _gbd_field(immersed, 2.0e-3)          # premise: guarded today
    orig_fn = getattr(G, _FN)
    with _GuardDeleted(*(_IMMERSED_GUARD + _MIRROR_GUARD)):
        assert getattr(G, _FN) is not orig_fn, (
            'premise: the mutation must actually replace the function')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.apply_prescription_persurface_to_beamlets(
                _bundle(), immersed, _LAM, z_image=2.0e-3)
        # the PUBLIC entry, through elements.lenses_gbd, must reach the
        # mutated function too -- else the identity below proves nothing
        served = _gbd_field(immersed, 2.0e-3)
        assert np.asarray(served).size > 0, (
            'the mutation did not reach apply_real_lens_gbd, so the '
            'byte-identity claim below would be vacuous')
        unguarded = {k: _gbd_field(p, 2.0e-3) for k, p in fixtures.items()}
    for k in fixtures:
        a, b = shipped[k], unguarded[k]
        assert _sha(a) == _sha(b), (
            f'{k}: the guards moved the shipped field; relative L2 '
            f'{np.linalg.norm(a - b) / np.linalg.norm(a):.3e}')


# ===========================================================================
# 4.  D-2 -- the assertion that could not fail
# ===========================================================================
def _pre_function_source():
    """The committed PRE source of the beamlet function (commit 1218b24f).

    The file carries an explanatory header comment; it is cut at the first
    ``def`` so the header -- which necessarily names the thing it is about --
    can neither satisfy nor defeat either form of the check.
    """
    assert _PRE_SRC.exists(), (
        f'premise: the PRE source fixture {_PRE_SRC} is missing, so this id '
        f'cannot prove anything')
    text = _PRE_SRC.read_text(encoding='cp1252')
    i = text.index('\ndef ')
    return text[i + 1:]


def test_the_second_sag_kernel_check_now_fails_against_the_pre_source():
    """FAIL-BEFORE (D-2), both halves, against the source the check was
    written to reject.

    The PRE function literally reads the last surface's radius through
    ``getattr`` with a quoted attribute name.  Asserted here:

      1. the OLD form -- ``"'radius'" not in fn_flat`` on a token stream that
         drops COMMENT and STRING tokens -- PASSES against that text.  That is
         the defect: a quoted attribute name is a STRING token, so the
         searched text can never contain it and the assertion could not fail;
      2. the REPAIRED check
         (``test_audit2609_b12b_gbd_projection.last_surface_radius_reads``,
         the ONE definition both files call) FAILS against it, naming the read
         it found;
      3. and PASSES against the shipped function, so the repair did not simply
         invert the assertion into one that always fires.

    The premise for (1) is stated as its own assertion: the PRE text must in
    fact contain the read, or the whole id is vacuous.
    """
    import io as _io
    import tokenize as _tok

    spec = importlib.util.spec_from_file_location(
        '_b12b_suite_under_test',
        Path(__file__).resolve().parent
        / 'test_audit2609_b12b_gbd_projection.py')
    b12b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b12b)

    pre = _pre_function_source()
    assert "getattr(surfs[-1], 'radius'" in pre, (
        'premise: the PRE source must contain the last-surface radius read; '
        'this fixture is stale')

    # (1) the OLD form, run verbatim on the PRE text
    code = []
    for t in _tok.generate_tokens(_io.StringIO(pre).readline):
        if t.type in (_tok.COMMENT, _tok.STRING):
            continue
        code.append(t.string)
    flat = ' '.join(code).replace(' ', '')
    assert "'radius'" not in flat and '"radius"' not in flat, (
        'the OLD assertion FAILED on the PRE source -- which would mean D-2 '
        'was not a defect.  Re-derive this id.')

    # (2) the repaired check FAILS there
    hits = b12b.last_surface_radius_reads(pre)
    assert hits, (
        'the repaired check found no last-surface radius read in the PRE '
        'source, so it is no better than the one it replaced')

    # (3) and PASSES on the shipped function
    shipped = inspect.getsource(getattr(G, _FN))
    assert not b12b.last_surface_radius_reads(shipped), (
        f'the repaired check fires on the SHIPPED function '
        f'({b12b.last_surface_radius_reads(shipped)}), which would make it an '
        f'assertion that always fails rather than one that can')


# ===========================================================================
# 5.  D-6 -- the digest depends on the memory budget, the field does not
# ===========================================================================
def test_a_gbd_field_digest_depends_on_the_memory_budget_and_the_field_does_not(
        monkeypatch):
    """EVIDENCE METHODOLOGY (D-6): ``LUMENAIRY_MEM_BUDGET_MB`` changes the
    BYTES of a GBD field returned by the public entry, and not the field.

    ``_reconstruct_windowed`` chunks each bucket of the coherent beamlet sum
    to stay under the budget, so the boundaries change the grouping of a
    scatter-add.  The variable is a CEILING on the ``mem_budget_mb`` keyword,
    which is why every byte-identity id in this package pins both.

    Stated as a LADDER, per ``docs/TESTING_STANDARDS.md`` rule 3: the budget
    is lowered until the digest moves, and the id hard-fails only when the
    ladder is exhausted -- so it does not depend on one grid size happening to
    straddle a chunk boundary on one build.

    The agreement bar is derived: a regrouped sum of the same terms differs by
    floating-point reassociation only, so the bound is machine epsilon of the
    field's own scale times 1e4 of slack for the accumulated operations.
    """
    presc = _presc('air')
    monkeypatch.delenv('LUMENAIRY_MEM_BUDGET_MB', raising=False)
    ref = _gbd_field(presc, 2.0e-3, mem_budget_mb=512.0)
    ref_sha = _sha(ref)
    scale = float(np.abs(ref).max())
    assert scale > 0 and np.isfinite(ref).all(), (
        'premise: the reference field must be finite and non-trivial')
    eps = float(np.finfo(np.float64).eps)
    bar = 1e4 * eps * scale

    rows = [('unset', ref_sha, 0.0)]
    moved = None
    for mb in (4096, 2048, 512, 128, 64, 16, 8, 4, 2, 1):
        monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', str(mb))
        F_mb = _gbd_field(presc, 2.0e-3, mem_budget_mb=512.0)
        gap = float(np.abs(F_mb - ref).max())
        rows.append((str(mb), _sha(F_mb), gap))
        assert gap <= bar, (
            f'budget {mb} MB changed the FIELD by {gap:.3e} against a derived '
            f'reassociation bar of {bar:.3e} (scale {scale:.3e}): this is no '
            f'longer a chunking difference')
        if moved is None and _sha(F_mb) != ref_sha:
            moved = mb
    assert moved is not None, (
        'the ladder 4096 .. 1 MB never moved the digest on this build, so '
        'this fixture cannot demonstrate D-6.  Re-derive it on a grid whose '
        'reconstruction chunks: the readings were '
        + ', '.join(f'{m}:{s[:8]}' for m, s, _g in rows))
    assert any(s == ref_sha for m, s, _g in rows if m in ('4096', '2048')), (
        'premise: a budget at or above the default must be a no-op (the '
        'variable is a CEILING), so those arms must reproduce the unset '
        'digest exactly')

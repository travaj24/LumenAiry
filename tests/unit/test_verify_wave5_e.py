"""VERIFY-WAVE5-E -- the decision tests the independent verification of Wave-5
item E needed and did not find already built.

Each id below closes a gap measured in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WAVE5_E.md``;
probes and per-arm JSON in ``validation/probe_verify_wave5_e/``.  Nothing here
re-states a number the item-E pins already assert -- these are the arms whose
absence a mutation or a measurement showed.
"""
from __future__ import annotations

import ast
import copy
import os
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import fga as _fga
from lumenairy.raytrace import exit_vertex as _ev
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)
from lumenairy import raytrace as rt


# ===========================================================================
# 1.  E1 -- the elision exposure, asserted STRUCTURALLY rather than per build
# ===========================================================================
#: The FFT dispatchers whose pyFFTW return is a NON-OWNING workspace view while
#: the ping-pong is on.  NumPy's ``temp_elide`` cannot claim such a view, so in
#: ``dispatcher(x) * other`` the OTHER operand is the one that gets elided into
#: -- and on the manylinux numpy 2.4.6 wheel a right-elided complex128 multiply
#: does not give the named form's last bits (measured rel 9.7e-17 .. 1.8e-16
#: over 16.5 % of the doubles at n >= 128, 2026-09-19,
#: ``validation/probe_verify_wave5_e/e1_spellings_linux_312_np246.json``).
_FFT_DISPATCHERS = frozenset({
    '_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
    '_rfft2', '_irfft2', '_fftn', '_ifftn'})

_BINOPS = (ast.Mult, ast.Add, ast.Sub, ast.Div)


def _callee(node):
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _basic_slice(sl):
    parts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    return all(isinstance(p, (ast.Slice, ast.Constant)) for p in parts)


def _operand_kind(node):
    """Is this operand something NumPy's elision could claim?

    ``Name`` / ``Attribute``  -- a live reference, refcount > 1, never elided.
    basic ``Subscript``       -- a VIEW (``owndata`` False), never elided.
    advanced ``Subscript``, ``Call``, ``BinOp``, ``UnaryOp``
                              -- a fresh NumPy-OWNED temporary: ELIDABLE.
    """
    if isinstance(node, (ast.Name, ast.Attribute)):
        return 'named'
    if isinstance(node, ast.Constant):
        return 'scalar'
    if isinstance(node, ast.Subscript):
        return 'view' if _basic_slice(node.slice) else 'elidable'
    return 'elidable'


def _scope_own_nodes(scope):
    stack, out = list(ast.iter_child_nodes(scope)), []
    while stack:
        nd = stack.pop()
        out.append(nd)
        if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(nd))
    return out


def _dispatcher_product_sites(pkg_dir):
    """Every in-library binary op with an FFT-dispatcher result as an operand.

    Both spellings are walked: the dispatcher called INLINE
    (``_fft2(E) * H``) and the dispatcher's result held under a NAME
    (``spec = _fft2(E); spec * H``) -- the second is the shape a grep for
    ``_fft2(`` misses, and it is the one three ``rs.py`` sites use.
    """
    sites = []
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, 'rb') as fh:
                src = fh.read().decode('utf-8')
            tree = ast.parse(src, filename=p)
            for sc in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.Module, ast.FunctionDef,
                                         ast.AsyncFunctionDef))]:
                own = _scope_own_nodes(sc)
                bound = {}
                for n in own:
                    if isinstance(n, ast.Assign) \
                            and _callee(n.value) in _FFT_DISPATCHERS:
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                bound[t.id] = _callee(n.value)
                for n in own:
                    if not (isinstance(n, ast.BinOp)
                            and isinstance(n.op, _BINOPS)):
                        continue
                    for me, other in ((n.left, n.right), (n.right, n.left)):
                        c = _callee(me)
                        via = (c if c in _FFT_DISPATCHERS
                               else (bound.get(me.id)
                                     if isinstance(me, ast.Name) else None))
                        if via is None:
                            continue
                        sites.append(dict(
                            file=os.path.basename(p), line=n.lineno,
                            dispatcher=via,
                            kind=_operand_kind(other),
                            src=ast.unparse(n)[:120]))
    uniq, seen = [], set()
    for s in sites:
        k = (s['file'], s['line'], s['dispatcher'], s['src'])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def test_no_in_library_fft_product_spells_an_elidable_operand():
    """THE BUILD-FREE FORM OF ITEM E1'S DECISION.

    ``test_wave5_e_fft_elision.py`` asserts the same property by MEASUREMENT --
    three entry points x two shapes must be byte-identical across
    ``set_fft_double_buffer``.  That arm can only fire on a build whose NumPy
    shows the elision asymmetry: measured 2026-09-19, the Windows wheels of
    numpy 2.4.4 AND 2.4.6 give right-elided == named at every n, so on Windows
    a call site rewritten to ``_fft2(E) * np.exp(1j*P)`` stays GREEN there
    (verified: the mutation ``e1_unnamed_right_operand`` is caught only on the
    Linux build).  The property the decision actually rests on -- that no
    in-library site hands the elision anything to claim -- is a property of the
    SOURCE, so it is asserted here as one, and it fires on every build.

    Measured on this tree: 10 product sites, 0 elidable
    (``validation/probe_verify_wave5_e/e1_ast_sites_v2.json``).  The published
    note names six of them (``asm.py:919/922/1391``, ``carrier.py:1401/7126``,
    ``fresnel.py:216``); the four it does not name --
    ``asm.py:1147`` (a basic-slice VIEW, so also not elidable) and
    ``rs.py:936/939/942`` (the dispatcher's result under a name) -- are why
    this walk is structural instead of a list.
    """
    pkg = os.path.dirname(os.path.abspath(la.__file__))
    sites = _dispatcher_product_sites(pkg)
    assert len(sites) >= 6, (
        f'premise: the walk must find the known dispatcher-product sites; it '
        f'found {len(sites)}.  If the dispatchers were renamed, update '
        f'_FFT_DISPATCHERS -- do not let this test pass vacuously.')
    bad = [s for s in sites if s['kind'] == 'elidable']
    assert not bad, (
        'an in-library site multiplies an FFT dispatcher result by an UNNAMED '
        'temporary, which NumPy\'s temporary elision may claim.  With the '
        'ping-pong ON the dispatcher\'s return is a non-owning workspace view '
        'that cannot be elided, so the OTHER operand is, and on the manylinux '
        'numpy 2.4.6 wheel that changes the last bits.  Name the operand (or '
        'revisit remedy (a), priced at +23.7 % to +32.7 % of '
        'angular_spectrum_propagate at 512^2..2048^2 on Windows, 2026-09-19).'
        '\n  ' + '\n  '.join(
            '%s:%d  %s' % (s['file'], s['line'], s['src']) for s in bad))


# ===========================================================================
# 2.  E5 / O-3 -- the guard's tolerance, read out of the GUARD
# ===========================================================================
_LAM = 1.55e-6
_SEMI = 0.18e-3


def _o3_surfs():
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': 0.70e-3,
          'glass_before': 'air', 'glass_after': 'N-LAK22',
          'semi_diameter': _SEMI}
    s2 = {'radius': -1.22e-3, 'conic': -0.25, 'thickness': 0.0,
          'glass_before': 'N-LAK22', 'glass_after': 'N-LAK22',
          'semi_diameter': _SEMI}
    presc = {'name': 'verify_o3', 'aperture_diameter': 2 * _SEMI,
             'surfaces': [s1, s2], 'thicknesses': [0.70e-3], 'stop_index': 0}
    s = [copy.copy(x) for x in surfaces_from_prescription(presc)]
    s[-1].thickness = 0.0
    return s


def _refuses(n_exit, z_image, surfs, monkeypatch):
    monkeypatch.setattr(_ev, 'resolve_exit_index',
                        lambda *a, **k: n_exit, raising=True)
    try:
        _fga._require_non_immersed_exit(surfs, _LAM, z_image, 'verify_o3')
        return False
    except NotImplementedError:
        return True


def test_the_immersed_exit_guards_boundary_is_the_guards_own_derivation(
        monkeypatch):
    """THE TOLERANCE, MEASURED THROUGH THE GUARD INSTEAD OF BESIDE IT.

    ``test_wave5_e_exit_vertex_dead_rays.py::
    test_the_guard_tolerance_is_the_wavefront_it_protects`` re-implements the
    formula inside the test and then asserts properties of its OWN copy, so it
    cannot see the library's copy change.  Verified by mutation on 2026-09-19:
    rewriting ``_require_non_immersed_exit``'s tolerance from
    ``waves*lam/max(|z|,lam)`` to ``waves*lam/lam`` -- dropping the image
    leg's length entirely -- leaves all 29 ids of that file GREEN.

    This id bisects the REAL guard for the smallest ``|n_exit - 1|`` it
    refuses, at several image distances, and requires that boundary to be the
    derived one to 1e-9 relative.  Both sides are asserted: the boundary
    tightens in proportion to the leg (a longer leg spends the same index
    error over more waves) and floors at the wave budget for a zero-length leg.
    """
    surfs = _o3_surfs()
    waves = _fga._FGA_IMAGE_LEG_WAVE_BUDGET
    floor = _fga._FGA_EXIT_INDEX_NOISE_FLOOR
    rows = []
    for z in (0.0, _LAM, 1.0e-5, 1.0e-4, 1.0e-3):
        lo, hi = 0.0, 1.0
        assert _refuses(1.0 + hi, z, surfs, monkeypatch), (
            f'premise: the guard must refuse n = 2.0 at z_image={z!r}')
        assert not _refuses(1.0, z, surfs, monkeypatch), (
            f'premise: the guard must NOT refuse an exactly-unity exit index '
            f'at z_image={z!r}')
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _refuses(1.0 + mid, z, surfs, monkeypatch):
                hi = mid
            else:
                lo = mid
        want = max(floor, waves * _LAM / max(abs(z), _LAM))
        rows.append((z, hi, want))
        assert abs(hi - want) <= 1e-9 * want, (
            f'the guard refuses at |n-1| > {hi!r} for z_image={z!r}, but its '
            f'documented derivation -- the index error that costs '
            f'{waves:g} waves over that leg -- is {want!r}.  The tolerance '
            f'must BE the derivation, not merely resemble it.')
    # two-sided: the floor at zero leg, and strict tightening with the leg
    assert rows[0][1] == pytest.approx(waves), (
        f'a zero-length leg must floor the tolerance at the wave budget; got '
        f'{rows[0][1]!r} against {waves!r}')
    tight = [rows[i + 1][1] < rows[i][1] for i in range(2, len(rows) - 1)]
    assert all(tight), (
        f'the tolerance must tighten strictly as the image leg grows; '
        f'boundaries were {[r[1] for r in rows]}')


# ===========================================================================
# 3.  E5 / O-4 -- the ray the round-2 fix exists for must EXIST
# ===========================================================================
_RT_LAM = 1.03e-6
_RT_SEMI = 0.15e-3
_RT_N = 121


def _rt_surfs(radius, conic, glass_after='air'):
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': 0.55e-3,
          'glass_before': 'air', 'glass_after': 'N-SSK8',
          'semi_diameter': _RT_SEMI}
    s2 = {'radius': radius, 'conic': conic, 'thickness': 0.0,
          'glass_before': 'N-SSK8', 'glass_after': glass_after,
          'semi_diameter': _RT_SEMI}
    presc = {'name': 'verify_o4', 'aperture_diameter': 2 * _RT_SEMI,
             'surfaces': [s1, s2], 'thicknesses': [0.55e-3], 'stop_index': 0}
    s = [copy.copy(x) for x in surfaces_from_prescription(presc)]
    s[-1].thickness = 0.0
    return s


def _rt_fan(over):
    h = np.linspace(-over * _RT_SEMI, over * _RT_SEMI, _RT_N)
    z = np.zeros(_RT_N)
    return h, z.copy(), z.copy(), z.copy()


def test_a_companion_dead_but_base_alive_ray_exists_and_is_still_projected():
    """THE POSITIVE FORM OF THE ROUND-2 JUSTIFICATION.

    WAVE5-E round 2 changed the freeze from ``transfer.alive`` to
    ``reached_surface`` because the finite-difference backend's ``alive`` is
    ``base_alive & companion_alive`` (VERIFY-WP-B12 O-4): a ray whose 9-ray FD
    companion bundle vignettes while the BASE ray landed did reach the vertex
    plane, and ``at_exit_vertex()`` projects it.  The item-E pin asserts that
    only inside ``if companion_only.any():`` -- if no such ray ever appeared,
    the whole reason for round 2 would be unpinned and the file would still be
    green.

    Here the existence is a LADDER over the aperture over-fill, and it is
    ASSERTED, not skipped: at least one rung of at least one surface class
    must produce such a ray, and on it the vertex-referenced state must agree
    with ``at_exit_vertex`` (i.e. the ray was PROJECTED, not frozen).
    Measured 2026-09-19 on both builds: 17 such rays over 12 of 44 probe
    cells, agreeing with ``at_exit_vertex`` to 0.0 m
    (``validation/probe_verify_wave5_e/e5_freeze_post_*.json``).
    """
    found = []
    for radius, conic in ((-1.05e-3, -0.35), (-0.90e-3, -0.60)):
        for over in (2.6, 3.0, 4.0):
            surfs = _rt_surfs(radius, conic)
            h, y, ux, uy = _rt_fan(over)
            nz = np.ones(_RT_N)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                b = rt.RayBundle(x=h.copy(), y=y.copy(),
                                 z=np.zeros(_RT_N), L=ux * nz, M=uy * nz,
                                 N=nz, wavelength=_RT_LAM,
                                 alive=np.ones(_RT_N, bool),
                                 opd=np.zeros(_RT_N))
                res = rt.trace(b, surfs, _RT_LAM)
                ev = res.at_exit_vertex()
                srf = ray_transfer_jacobian(
                    h.copy(), y.copy(), ux.copy(), uy.copy(), surfs,
                    _RT_LAM, reference='surface')
                vtx = ray_transfer_jacobian(
                    h.copy(), y.copy(), ux.copy(), uy.copy(), surfs,
                    _RT_LAM, reference='exit_vertex')
            reached = np.asarray(res.image_rays.alive, bool)
            usable = np.asarray(srf.alive, bool)
            only = reached & ~usable
            if not only.any():
                continue
            d = float(np.nanmax(np.abs(
                np.asarray(vtx.opd, float)[only]
                - np.asarray(ev.opd, float)[only])))
            moved = float(np.nanmax(np.abs(
                np.asarray(vtx.opd, float)[only]
                - np.asarray(srf.opd, float)[only])))
            found.append((radius, over, int(only.sum()), d, moved))
            assert d < 1e-12, (
                f'r={radius!r} over={over!r}: a companion-dead-but-base-alive '
                f'ray must still be PROJECTED and land where at_exit_vertex '
                f'puts it; it is {d:.3e} m away.  Freezing it is what round 1 '
                f'did, and it reddened 7 of the 18 WP-B12 reference-plane '
                f'pins.')
            assert moved > 0.0, (
                f'r={radius!r} over={over!r}: the same ray must have MOVED '
                f'from the surface-referenced state -- if it did not, the '
                f'freeze has swallowed it and the agreement above is vacuous')
    assert found, (
        'premise ASSERTED, not skipped: no ray on any rung of this ladder is '
        'companion-dead-but-base-alive, so the whole reason WAVE5-E round 2 '
        'exists (VERIFY-WP-B12 O-4) is unobservable here.  Widen the ladder '
        'or re-derive the FD backend\'s alive -- do not let the round-2 '
        'justification go unpinned.')


# ===========================================================================
# 4.  E5-new -- the JAX analytic backend reports no vignetting (KNOWN DEFECT)
# ===========================================================================
def test_the_jax_analytic_backend_reports_no_vignetting_KNOWN_DEFECT():
    """A KNOWN-RED PIN, in the shape ``test_verify_b14_known_reds.py`` uses.

    ``differential._adrt_jax`` ends ``alive=jnp.ones((n,), dtype=bool)``: the
    JAX path of ``ray_transfer_jacobian_analytic`` reports EVERY ray alive
    whatever the aperture, while its NumPy twin -- the same algorithm, the
    same ``_adrt_step`` -- kills the vignetted ones.  Measured 2026-09-19 on
    both builds over a six-rung aperture ladder: 0 dead of 201 on the JAX path
    at every rung, against 0 / 68 / 104 / 134 / 156 / 174 on the NumPy path
    (``validation/probe_verify_wave5_e/e5_jax_alive_*.json``).  The VALUES
    agree to 1.1e-19 m on the rays NumPy kills, so it is the MASK alone.

    This id exists so the defect cannot be lost: it asserts the current,
    WRONG behaviour together with the correct behaviour of the NumPy path.
    WHEN IT FAILS BECAUSE THE JAX PATH STARTS VIGNETTING, the defect is
    fixed -- delete this id and assert the parity instead.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp                                 # noqa: PLC0415

    n_rays = 201
    half = 0.45e-3
    h = np.linspace(-half, half, n_rays)
    z = np.zeros(n_rays)
    surfs = _rt_surfs(-1.05e-3, -0.35)
    surfs[0].semi_diameter = 0.15e-3
    surfs[-1].semi_diameter = 0.15e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        npa = ray_transfer_jacobian_analytic(
            h.copy(), z.copy(), z.copy(), z.copy(), surfs, _RT_LAM,
            reference='surface')
        jxa = ray_transfer_jacobian_analytic(
            jnp.asarray(h), jnp.asarray(z), jnp.asarray(z), jnp.asarray(z),
            surfs, _RT_LAM, reference='surface')
    np_dead = int((~np.asarray(npa.alive, bool)).sum())
    jx_dead = int((~np.asarray(jxa.alive, bool)).sum())
    assert np_dead >= n_rays // 4, (
        f'premise: the NumPy analytic backend must vignette most of a fan '
        f'{half / 0.15e-3:.1f}x the clear aperture; it killed {np_dead} of '
        f'{n_rays}.  Without that there is nothing for the JAX path to be '
        f'missing and this pin says nothing.')
    assert jx_dead == 0, (
        f'the JAX analytic backend now reports {jx_dead} dead rays where it '
        f'used to report 0 -- THE DEFECT THIS PIN RECORDS IS FIXED.  Delete '
        f'this id and replace it with the parity assertion '
        f'`np.array_equal(np.asarray(jxa.alive), np.asarray(npa.alive))`, '
        f'and drop the defect from VERIFY_WAVE5_E.md.')
    # the values are right; it is the mask alone -- so the fix is local
    dead = ~np.asarray(npa.alive, bool)
    d = float(np.nanmax(np.abs(np.asarray(jxa.opd, float)[dead]
                               - np.asarray(npa.opd, float)[dead])))
    assert d < 1e-15, (
        f'the two analytic paths must still agree on the VALUES they compute '
        f'for the rays NumPy kills (measured 1.1e-19 m); got {d:.3e} m.  If '
        f'they no longer do, the JAX path has diverged in more than its mask.')


# ===========================================================================
# 5.  E4 / D3 -- the budget is a bound IFF it clears the one-column floor
# ===========================================================================
def test_the_measured_budget_is_a_bound_exactly_above_the_one_column_floor():
    """THE SCOPE SENTENCE AS A FUNCTION OF THE FLOOR, NOT AT ONE N.

    WAVE5-E's ``gbd.py`` note and ``test_verify_b14_known_reds.py`` state the
    scope at N = 256: 512 MB and 16 MB are bounds (0.84x, 0.60x) and 4 MB and
    1 MB are not (2.39x, 9.58x).  Re-measured here 2026-09-19 -- all four
    Windows figures reproduce to the digit on an independently written bundle,
    and WSL reads 0.76 / 0.56 / 2.23 / 8.92.  Those are READINGS; the
    build-free statement is the IFF below, and this id asserts it at TWO grids
    so that "bounded" tracks the floor
    ``Ny*Nx*(48 + _DENSE_CELL_BYTES_MEASURED)`` rather than one N's arithmetic
    (11.53 MB at N = 256, 6.49 MB at N = 192; measured bounded at 16 MB and
    not at 4 MB for the first, bounded at 8 MB and not at 4 MB for the
    second).

    Cheap by construction: every budget here puts the chunk at 1 or 2, so no
    cell allocates the 400 MB the 512 MB arm does.
    """
    tracemalloc = pytest.importorskip('tracemalloc')
    from lumenairy.propagators import gbd as G               # noqa: PLC0415

    def bundle(n, seed):
        rng = np.random.default_rng(seed)
        kw = dict(positions=rng.normal(0.0, 1.7e-4, size=(n, 3)),
                  Q=np.full(n, 1.0 / (0.8e-3 - 0.03j), dtype=np.complex128),
                  amplitude=(rng.normal(size=n)
                             + 1j * rng.normal(size=n)).astype(np.complex128),
                  waist0=np.full(n, 1.2e-3))
        try:
            return G.BeamletBundle(directions=np.zeros((n, 3)), **kw)
        except TypeError:
            return G.BeamletBundle(**kw)

    def peak(b, N, budget):
        old = G.DENSE_MEM_BUDGET_ACCOUNTING
        G.DENSE_MEM_BUDGET_ACCOUNTING = 'measured'
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                G.reconstruct_field_from_beamlets(
                    b, Ny=N, Nx=N, dx=2.0e-6, wavelength=1.0e-6,
                    chunk_beamlets=4096, mem_budget_mb=budget)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
            G.DENSE_MEM_BUDGET_ACCOUNTING = old

    rows = []
    for N, nb, above, below in ((256, 512, 16.0, 4.0), (192, 384, 8.0, 4.0)):
        floor = N * N * (48.0 + G._DENSE_CELL_BYTES_MEASURED) / 1e6
        b = bundle(nb, 4242 + N)
        assert below < floor < above, (
            f'premise: at N={N} the one-column floor is {floor:.2f} MB, which '
            f'must sit strictly between the two budgets chosen ({below} and '
            f'{above} MB) for the IFF below to have a side each.  Re-derive '
            f'the budgets from _DENSE_CELL_BYTES_MEASURED '
            f'({G._DENSE_CELL_BYTES_MEASURED!r}).')
        for budget in (above, below):
            ratio = peak(b, N, budget) / (budget * 1e6)
            rows.append((N, floor, budget, ratio))
            bounded = ratio < 1.0
            assert bounded is (budget > floor), (
                f"N={N}: 'measured' accounting must bound the budget exactly "
                f"when the budget clears the one-column floor "
                f"({floor:.2f} MB).  Budget {budget} MB read {ratio:.3f}x, "
                f"so bounded={bounded} where above-floor={budget > floor}.  "
                f"The chunk cannot go below 1 beamlet column and the fixed "
                f"~48 B/cell term is outside the chunk arithmetic, so no "
                f"accounting constant can put the loop under a budget below "
                f"that floor -- if this fires, either the floor moved or the "
                f"loop's allocation did.\n  " + '\n  '.join(
                    'N=%d floor=%.2f MB budget=%.1f MB -> %.3fx' % r
                    for r in rows))

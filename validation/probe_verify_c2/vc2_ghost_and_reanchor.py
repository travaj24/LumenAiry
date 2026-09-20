"""VERIFY-WP-C2 items 9 and 10.

**Item 9 -- ``analysis.ghost`` keeps the generic normal.**  It calls
``_refract`` / ``_reflect`` directly (``ghost.py:934,955``), and those
private helpers kept ``sphere_normal='generic'`` / ``renormalize=True``.
So after WP-C2 a ghost path and the main trace refract off DIFFERENT
arithmetic on the same spherical surface.  Measured here on a real
spherical prescription: the ghost path re-traced by ``retrace_ghost_path``
under the private default, against the same path walked with the
``analytic`` normal the public trace now uses.

**Item 10 -- the ``EDITED_IN_PLACE`` map.**  Four abuse attempts against
``scripts/reanchor_citations.py::_edited_in_place``:

* A. the cited line's CONTENT changed beyond the default (same leading
  token, different meaning) -- does the override still fire?
* B. the cited line moved AND was edited -- does the fixed new coordinate
  silently re-anchor onto whatever now sits there?
* C. the leading token guard: an unrelated line that happens to start with
  the same token.
* D. is the map VERSION-PINNED?  It names "5.49.0"; nothing checks that
  the tree is at that release, so it keeps firing for every release after.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_ghost_and_reanchor.py OUT.json``
"""
import importlib.util
import json
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace import intersection as _isect         # noqa: E402
from lumenairy.analysis import ghost as _ghost                # noqa: E402

WL = 587.5618e-9


def _doublet():
    return [Surface(radius=0.0517, thickness=0.0090, glass_before='air',
                    glass_after='N-BK7', semi_diameter=0.0125),
            Surface(radius=-0.0345, thickness=0.0025, glass_before='N-BK7',
                    glass_after='N-SF5', semi_diameter=0.0125),
            Surface(radius=-0.1200, thickness=0.0400, glass_before='N-SF5',
                    glass_after='air', semi_diameter=0.0125)]


def _bundle(n=2000, hmax=0.0080):
    rng = np.random.default_rng(4242)
    r = hmax * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return RayBundle(x=r * np.cos(th), y=r * np.sin(th), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _ghost_item():
    """Re-trace every 2-bounce ghost path of a spherical doublet with the
    PRIVATE default (generic normal, per-surface rescale) and with the
    arithmetic the PUBLIC trace now uses, and difference them.

    ``ghost.retrace_ghost_path`` imports ``_refract`` / ``_reflect`` from
    ``raytrace.intersection`` INSIDE its own body at call time, so patching
    the ``intersection`` module is what the call actually sees.
    """
    from lumenairy.io.prescriptions_builders import make_doublet
    pres = make_doublet(0.0517, -0.0345, -0.120, 0.009, 0.0025,
                        'N-BK7', 'N-SF5', aperture=0.025)
    surfs = _doublet()
    n = len(surfs)
    pairs = _ghost.enumerate_ghost_paths(n)
    out = {'pairs': [list(p) for p in pairs], 'per_path': {}}
    real_refract, real_reflect = _isect._refract, _isect._reflect

    def _mk(which):
        # ``renormalize=False`` is NOT a candidate edit for ghost: ghost
        # owns its own loop and has no exit pass, so turning it off would
        # leave the directions unnormalised for good.  It is measured only
        # to bound the whole gap; the RECOMMENDED edit is the normal alone.
        def _pr(rays, surface, n1, n2, **kw):
            kw['sphere_normal'] = 'analytic'
            if which == 'both':
                kw['renormalize'] = False
            return real_refract(rays, surface, n1, n2, **kw)

        def _pl(rays, surface, **kw):
            kw['sphere_normal'] = 'analytic'
            if which == 'both':
                kw['renormalize'] = False
            return real_reflect(rays, surface, **kw)
        return _pr, _pl

    def _run(pth):
        return _ghost.retrace_ghost_path(pres, pth, WL,
                                         semi_aperture=0.0080, n_rays=256,
                                         image_plane_z=0.040)

    for (i, j) in pairs:
        pth = _ghost._path_from_pair(n, int(i), int(j))
        base = _run(pth)
        rec = {}
        for which in ('normal_only', 'both'):
            _isect._refract, _isect._reflect = _mk(which)
            try:
                alt = _run(pth)
            finally:
                _isect._refract = real_refract
                _isect._reflect = real_reflect
            for k in sorted(set(base) | set(alt)):
                a, b = base.get(k), alt.get(k)
                if isinstance(a, (int, float)) and isinstance(b,
                                                              (int, float)):
                    rec['d_%s_%s' % (which, k)] = float(abs(a - b))
                    rec['v_' + k] = float(a)
        rec['bit_identical'] = all(v == 0.0 for kk, v in rec.items()
                                   if kk.startswith('d_'))
        out['per_path'][f'{int(i)},{int(j)}'] = rec
    keys = set()
    for v in out['per_path'].values():
        keys |= {k for k in v if k.startswith('d_')}
    out['worst'] = {k: max(v.get(k, 0.0) for v in out['per_path'].values())
                    for k in sorted(keys)}
    out['any_path_differs'] = any(not v['bit_identical']
                                  for v in out['per_path'].values())
    import inspect
    out['private_defaults'] = {
        nm: {k: (str(v.default) if v.default is not inspect._empty
                 else '<required>')
             for k, v in inspect.signature(fn).parameters.items()
             if k in ('renormalize', 'sphere_normal')}
        for nm, fn in (('_refract', real_refract), ('_reflect', real_reflect))}
    return out


def _load_reanchor():
    path = os.path.join(_ROOT, 'scripts', 'reanchor_citations.py')
    spec = importlib.util.spec_from_file_location('_ra', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reanchor_item():
    ra = _load_reanchor()
    res = {'map_size': len(ra.EDITED_IN_PLACE),
           'map': {f'{k[0]}:{k[1]}': list(v)
                   for k, v in ra.EDITED_IN_PLACE.items()}}
    base = ra.DEFAULT_BASE
    res['default_base'] = base

    # --- what the guard reads, per mapped coordinate ---------------
    probes = {}
    for (path, num), (new_num, reason) in ra.EDITED_IN_PLACE.items():
        want, _ctx = ra.base_line(path, num, base)
        hay = ra.lines(path)
        got = hay[new_num - 1] if 1 <= new_num <= len(hay) else None
        lead = want.strip().split('=')[0].strip() if want else None
        fires, how = ra._edited_in_place(path, num, base)
        probes[f'{path}:{num}'] = dict(
            base_line=(want.rstrip() if want else None),
            current_line=(got.rstrip() if got else None),
            leading_token_the_guard_checks=lead,
            fires=(fires is not None), how=how, reason=reason)
    res['mapped_coordinates'] = probes

    # --- the abuse harness: swap in a DOCTORED copy of one file -----
    TGT = 'lumenairy/raytrace/trace.py'
    real_lines = ra.lines

    def _doctored(new_line_61=None, move_away=None):
        hay = list(real_lines(TGT))
        if new_line_61 is not None:
            hay[60] = new_line_61
        if move_away is not None:
            decl = hay[60]
            hay[60] = move_away
            hay.append(decl)
        return hay

    def _try(hay):
        def _patched(path, rev=None):
            if path == TGT and rev is None:
                return hay
            return real_lines(path, rev)
        ra.lines = _patched
        try:
            n, how = ra._edited_in_place(TGT, 61, base)
        finally:
            ra.lines = real_lines
        return dict(fires=(n is not None), new_num=n, how=how)

    REVERTED = "    sphere_normal: str = 'generic',"
    NONSENSE = "    sphere_normal: str = 'not-a-route',"
    STALE = "    sphere_normal: str = 'generic',  # a STALE COPY"

    res['abuse_A_default_silently_reverted'] = dict(
        doctored_line=REVERTED, result=_try(_doctored(REVERTED)),
        verdict=('FIRES -- the override re-anchors a citation whose claim '
                 '(the default moved to analytic) is now false'))
    res['abuse_A2_default_set_to_nonsense'] = dict(
        doctored_line=NONSENSE, result=_try(_doctored(NONSENSE)),
        verdict='FIRES -- everything after the "=" is unchecked')
    res['abuse_B_declaration_moved_away'] = dict(
        doctored_line=STALE, result=_try(_doctored(move_away=STALE)),
        verdict=('FIRES -- the map hard-codes the new NUMBER, so a stale '
                 'copy at that line satisfies it while the real '
                 'declaration has moved elsewhere'))
    res['abuse_C_unrelated_line_control'] = dict(
        doctored_line='    renormalize: str = "exit",',
        result=_try(_doctored('    renormalize: str = "exit",')),
        verdict='REFUSED -- the leading-token guard does work, one-sided')
    res['abuse_C2_file_truncated_control'] = dict(
        doctored_line='(file truncated to 30 lines)',
        result=_try(list(real_lines(TGT))[:30]),
        verdict='REFUSED -- out-of-range new coordinate')

    # --- ABUSE D: version pinning ----------------------------------
    src = open(os.path.join(_ROOT, 'scripts', 'reanchor_citations.py'),
               encoding='utf-8').read()
    res['abuse_D_version_pinned'] = dict(
        mentions_version_in_reason_text=True,
        checks_package_version=('__version__' in src),
        checks_base_commit=('DEFAULT_BASE' in src),
        note=('the release number lives only in the human-readable reason '
              'string; nothing compares it to the package version or to '
              'the CHANGELOG block being re-anchored, so the map is '
              'permanent rather than scoped to the release that needed '
              'it'))

    # --- does the tool report clean on this tree? -------------------
    try:
        _txt, changed, notes = ra.reanchor(base=base)
        res['reanchor_run'] = dict(changed=len(changed), notes=len(notes),
                                   notes_sample=[str(n) for n in notes[:8]],
                                   changed_sample=[str(c)
                                                   for c in changed[:8]])
    except SystemExit as e:
        res['reanchor_run'] = dict(error=str(e))
    return res


def main(out_path):
    res = dict(meta=dict(python=sys.version.split()[0],
                         numpy=np.__version__, lumenairy=la.__version__,
                         file=la.__file__),
               ghost=_ghost_item(), reanchor=_reanchor_item())
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    g = res['ghost']
    print('GHOST private defaults:', json.dumps(g['private_defaults']))
    print('GHOST any path differs under the public default arithmetic:',
          g['any_path_differs'])
    print('GHOST worst:', json.dumps(g['worst'], sort_keys=True))
    for k, v in g['per_path'].items():
        print('  path', k, 'identical', v['bit_identical'],
              {kk: vv for kk, vv in sorted(v.items())
               if kk.startswith('d_') and vv})
    r = res['reanchor']
    print()
    print('REANCHOR map size', r['map_size'], 'base', r['default_base'])
    for k, v in r['mapped_coordinates'].items():
        print(' ', k, 'fires', v['fires'], '|', v['how'])
        print('     base   :', v['base_line'])
        print('     current:', v['current_line'])
        print('     token  :', v['leading_token_the_guard_checks'])
    print(' run:', json.dumps(r['reanchor_run'])[:900])


if __name__ == '__main__':
    main(sys.argv[1])

"""VERIFY-WP-B12b -- a pytest plugin that MUTATES the WP-B12b repair in
memory.

Used as ``python -m pytest -p vb12b_mutate ...`` with ``VB12B_MUTATION`` set
to one of the keys below (``PYTHONPATH`` must reach this directory).  Each
mutation is a plausible WRONG version of the repair; a durable pin must go RED
under the mutation that breaks the property it claims to protect.

The three gbd.py mutations are applied by recompiling
``apply_prescription_persurface_to_beamlets`` from its own source with one
textual edit, in the module's own globals, and rebinding the recompiled
function everywhere the library re-exports it -- so the mutation is the code
the library runs, not a wrapper around it.

Mutations
---------
``fold_restored``    the deleted ``-sag`` fold is put back while the
                     ``reference='exit_vertex'`` projection stays: the sag is
                     then subtracted TWICE on a conic last surface.
``fold_restored_renamed``  the same double correction with the five local
                     names changed, so a check that bans ``_Rl`` / ``_kl`` /
                     ``_cl`` by SPELLING cannot see it.
``local_surface``    the local-frame branch asks for ``reference='surface'``
                     (the repair undone -- pre-v5.22 on a curved base).
``world_exit_vertex``  the ``world_output_plane`` branch asks for
                     ``reference='exit_vertex'`` (the mirrored defect: its
                     own leg starts at the last-surface INTERSECTION, so the
                     sag is now counted twice there).
``state_only``       the projection moves the state and leaves the 4x4
                     Jacobian on the last surface.
``sign_plus``        ``_exit_direction_sign`` always returns +1, so a
                     mirror-terminated prescription gets the vertex transfer
                     with the wrong sign.
``identity``         the projection is a no-op (the pre-WP-B12 tree), kept as
                     the coarse control.

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import inspect
import os
import textwrap

MUTATIONS = ('fold_restored', 'fold_restored_renamed', 'local_surface',
             'world_exit_vertex', 'state_only', 'sign_plus', 'identity')

_FOLD = """    _Rl = float(getattr(surfs[-1], 'radius', np.inf))
    _kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
    if np.isfinite(_Rl) and _Rl != 0.0:
        _cl = 1.0 / _Rl
        _r2 = dt.x ** 2 + dt.y ** 2
        _sag = _cl * _r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
    else:
        _sag = np.zeros_like(dt.x)
    t = (z_image - _sag) / Nz2
"""


def _recompile(edit):
    """Rebuild ``apply_prescription_persurface_to_beamlets`` from source with
    one textual edit and rebind it everywhere the library exports it."""
    import lumenairy
    import lumenairy.elements.lenses_gbd as L
    from lumenairy.propagators import gbd as G
    name = 'apply_prescription_persurface_to_beamlets'
    src = textwrap.dedent(inspect.getsource(getattr(G, name)))
    new = edit(src)
    if new == src:
        raise SystemExit(f'vb12b_mutate: the edit for {name} matched nothing')
    ns = dict(G.__dict__)
    fname = f'<vb12b_mutate:{name}>'
    # Register the mutated source with ``linecache`` so ``inspect.getsource``
    # on the recompiled function returns the MUTATED text.  Without this the
    # builder's token check raises OSError and is scored as a catch when it
    # is only an artefact of the mutation vehicle.
    import linecache
    linecache.cache[fname] = (len(new), None, new.splitlines(keepends=True),
                              fname)
    exec(compile(new, fname, 'exec'), ns)
    fn = ns[name]
    fn.__module__ = G.__name__
    for mod in (G, L, lumenairy):
        if hasattr(mod, name):
            setattr(mod, name, fn)
    return fn


def pytest_configure(config):
    apply_named(os.environ.get('VB12B_MUTATION', ''))


def apply_named(name):
    if not name:
        return None
    if name not in MUTATIONS:
        raise SystemExit(f'unknown VB12B_MUTATION {name!r}')
    import lumenairy.raytrace.differential as D

    if name == 'fold_restored':
        _recompile(lambda s: s.replace('    t = z_image / Nz2\n', _FOLD))
    elif name == 'fold_restored_renamed':
        renamed = (_FOLD.replace('_Rl', '_r_last').replace('_kl', '_k_last')
                   .replace('_cl', '_c_last').replace('_r2', '_rho2')
                   .replace('_sag', '_vsag'))
        _recompile(lambda s: s.replace('    t = z_image / Nz2\n', renamed))
    elif name == 'local_surface':
        _recompile(lambda s: s.replace(
            "    _reference = 'surface' if world_output_plane is not None "
            "else 'exit_vertex'\n",
            "    _reference = 'surface'\n"))
    elif name == 'world_exit_vertex':
        _recompile(lambda s: s.replace(
            "    _reference = 'surface' if world_output_plane is not None "
            "else 'exit_vertex'\n",
            "    _reference = 'exit_vertex'\n"))
    elif name == 'state_only':
        orig = D._project_to_exit_vertex_plane

        def state_only(transfer, *a, **kw):
            out = orig(transfer, *a, **kw)
            if out is transfer:
                return out
            return D.DifferentialTransfer(
                jacobian=transfer.jacobian, x=out.x, y=out.y, ux=out.ux,
                uy=out.uy, opd=out.opd, alive=out.alive)
        D._project_to_exit_vertex_plane = state_only
    elif name == 'sign_plus':
        D._exit_direction_sign = lambda surfaces: 1.0
    elif name == 'identity':
        D._project_to_exit_vertex_plane = (lambda transfer, *a, **kw: transfer)
    print(f'VB12B MUTATION ACTIVE: {name}')
    return name

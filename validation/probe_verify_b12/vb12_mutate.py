"""VERIFY-WP-B12 -- a pytest plugin that MUTATES the WP-B12 repair in memory.

Used as ``python -m pytest -p vb12_mutate ...`` with ``VB12_MUTATION`` set to
one of the keys below.  Each mutation is a plausible wrong version of the
repair; a durable pin must go RED under the mutation that breaks the property
it claims to protect, and must stay GREEN under the ones that do not touch it.

Mutations
---------
``identity``        the projection is a no-op -- the pre-WP-B12 tree.
``state_only``      the state is projected, the JACOBIAN is left on the
                    last surface (the design WP-B12 measured and rejected).
``sign_plus``       ``_exit_direction_sign`` always returns +1 (a mirror's
                    reflected leg then accrues the optical path of the wrong
                    sign).
``conic_only``      the projection sees a last surface stripped of its
                    aspheric coefficients and its biconic y-branch -- i.e.
                    ``gbd.py``'s in-line copy, promoted into the helper.
``biconic_drop``    ONLY the biconic y-branch is stripped (the narrow half of
                    ``conic_only``), so a suite that pins the asphere but not
                    the biconic stays green.
``field_frame_drop``  ONLY the field-frame decenter / tilt / sag callable is
                    stripped.
``always_flat``     ``_last_surface_sag_vanishes`` always answers True, so
                    the short-circuit swallows every projection.
``opd_sign``        the transverse projection is right and the OPTICAL PATH
                    correction is added instead of subtracted.
``no_sec``          the optical path correction drops ``sec(theta)`` (the
                    obliquity factor) -- right on axis, wrong at the rim.
``n_exit_one``      the exit-medium index is hard-coded to 1.0 instead of
                    being resolved from the prescription (the brief's own
                    suggested inline edit).
"""
from __future__ import annotations

import copy
import os

import numpy as np

MUTATIONS = ('identity', 'state_only', 'sign_plus', 'conic_only',
             'biconic_drop', 'field_frame_drop', 'always_flat', 'opd_sign',
             'no_sec', 'n_exit_one')


def pytest_configure(config):
    name = os.environ.get('VB12_MUTATION', '')
    if not name:
        return
    if name not in MUTATIONS:
        raise SystemExit(f'unknown VB12_MUTATION {name!r}')
    import lumenairy.raytrace.differential as D
    orig = D._project_to_exit_vertex_plane

    if name == 'identity':
        D._project_to_exit_vertex_plane = (
            lambda transfer, *a, **kw: transfer)
    elif name == 'state_only':
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
    elif name == 'conic_only':
        def conic_only(transfer, surfaces, *a, **kw):
            surfs = list(surfaces)
            last = copy.copy(surfs[-1])
            try:
                last.aspheric_coeffs = None
                last.aspheric_coeffs_y = None
                last.radius_y = None
                last.conic_y = None
                last.freeform = None
            except Exception:                              # noqa: BLE001
                pass
            surfs[-1] = last
            return orig(transfer, surfs, *a, **kw)
        D._project_to_exit_vertex_plane = conic_only
    elif name in ('biconic_drop', 'field_frame_drop'):
        strip = (('radius_y', 'conic_y', 'aspheric_coeffs_y')
                 if name == 'biconic_drop'
                 else ('field_decenter', 'field_tilt', 'field_sag_callable'))

        def narrow(transfer, surfaces, *a, **kw):
            surfs = list(surfaces)
            last = copy.copy(surfs[-1])
            for attr in strip:
                try:
                    setattr(last, attr, None)
                except Exception:                          # noqa: BLE001
                    pass
            surfs[-1] = last
            return orig(transfer, surfs, *a, **kw)
        D._project_to_exit_vertex_plane = narrow
    elif name == 'always_flat':
        D._last_surface_sag_vanishes = lambda surface: True
    elif name == 'opd_sign':
        def opd_sign(transfer, *a, **kw):
            out = orig(transfer, *a, **kw)
            if out is transfer:
                return out
            return D.DifferentialTransfer(
                jacobian=out.jacobian, x=out.x, y=out.y, ux=out.ux,
                uy=out.uy, opd=2.0 * transfer.opd - out.opd, alive=out.alive)
        D._project_to_exit_vertex_plane = opd_sign
    elif name == 'n_exit_one':
        import lumenairy.raytrace.exit_vertex as EV
        orig_idx = EV.resolve_exit_index
        EV.resolve_exit_index = (
            lambda surfaces, wavelength, **kw: 1.0)
        D._vb12_orig_idx = orig_idx
    elif name == 'no_sec':
        def no_sec(transfer, *a, **kw):
            out = orig(transfer, *a, **kw)
            if out is transfer:
                return out
            sec = np.sqrt(1.0 + np.asarray(out.ux) ** 2
                          + np.asarray(out.uy) ** 2)
            d = np.asarray(transfer.opd) - np.asarray(out.opd)   # n*sgn*s*sec
            return D.DifferentialTransfer(
                jacobian=out.jacobian, x=out.x, y=out.y, ux=out.ux,
                uy=out.uy, opd=np.asarray(transfer.opd) - d / sec,
                alive=out.alive)
        D._project_to_exit_vertex_plane = no_sec
    print(f'\n[vb12_mutate] ACTIVE MUTATION: {name}\n')

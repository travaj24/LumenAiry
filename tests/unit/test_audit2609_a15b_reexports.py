"""WP-A15b -- the re-export reconciliation requested by the other work packages.

``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`` section 14 V6 asks for
one public surface rather than a submodule surface plus a partially-overlapping
root.  Seven work packages left the concrete requests in their reports'
"Requested changes outside my ownership" sections; this file pins that each
landed AND that the top-level name is the SAME OBJECT as the submodule's, not
a copy or a look-alike:

===============================  ==========================================
requested by                     name(s)
===============================  ==========================================
WP-A7 section 5.1 / 5.2          ``unwrap_phase_2d``, ``clear_meshgrid_cache``,
                                 ``meshgrid_cache_bytes``,
                                 ``zernike_basis_cache_bytes``
WP-A7 section 5.3                ``seed_entrance_eikonal``
WP-A10 section 5.1               ``MinEdgeThicknessMerit``, ``edge_thickness``
WP-A13 section 5.1 / VERIFY-A13  ``pmm_2d_order_drift``
audit section 15.1 / WP-A1       ``exit_vertex_transfer`` + its pieces and
                                 its JAX twin
WP-A15b (this one)               ``override``
===============================  ==========================================

The two meta-walkers (``test_v4_16_0_walker_all_symmetry.py``,
``test_v4_14_1_dispatcher_pin_cache_clears.py``) are the general gate and were
RED on 19 names before this change; this file is the specific one, so a
regression names the API rather than the walker.
"""
from __future__ import annotations

import importlib

import pytest

import lumenairy as la

#: ``top-level name -> the module that DEFINES it``.  Identity against this
#: module is the assertion: a re-export that rebuilt or wrapped the object
#: would pass a ``hasattr`` pin and fail here.
_REEXPORTS = {
    # WP-A7 -- analysis (masked 2-D unwrap + the cache accessors)
    'unwrap_phase_2d': 'lumenairy.analysis.opd',
    'clear_meshgrid_cache': 'lumenairy.analysis.beam_stats',
    'meshgrid_cache_bytes': 'lumenairy.analysis.beam_stats',
    'zernike_basis_cache_bytes': 'lumenairy.analysis.zernike',
    # WP-A10 -- optimize (edge-thickness merit + its free-function oracle)
    'MinEdgeThicknessMerit': 'lumenairy.optimize.core',
    'edge_thickness': 'lumenairy.optimize.core',
    # WP-A13 / VERIFY-A13 -- the 2-D PMM convergence signal
    'pmm_2d_order_drift': 'lumenairy.elements.pmm.twod',
    # audit section 15.1 / WP-A1 -- the ONE shared exit-vertex transfer
    'exit_vertex_transfer': 'lumenairy.raytrace.exit_vertex',
    'EXIT_VERTEX_GRAZING_TOL': 'lumenairy.raytrace.exit_vertex',
    'resolve_exit_index': 'lumenairy.raytrace.exit_vertex',
    'vertex_plane_transfer_t': 'lumenairy.raytrace.exit_vertex',
    'exit_vertex_transfer_jax': 'lumenairy.raytrace.jax_trace',
    # WP-A7 section 5.3 -- the R2 entrance-eikonal seeder
    'seed_entrance_eikonal': 'lumenairy.raytrace.trace',
    # WP-A15b -- the generic knob context manager
    'override': 'lumenairy._knobs',
}


@pytest.mark.parametrize('name,where', sorted(_REEXPORTS.items()))
def test_name_is_reexported_at_top_level_and_is_the_same_object(name, where):
    assert hasattr(la, name), (
        f'lumenairy.{name} is missing; it was requested by the work package '
        f'named in this module\'s docstring and is in {where}.__all__ or its '
        f'package facade.')
    assert name in la.__all__, (
        f'lumenairy.{name} resolves but is not in lumenairy.__all__, so '
        f'``from lumenairy import *`` does not carry it.')
    mod = importlib.import_module(where)
    assert getattr(la, name) is getattr(mod, name), (
        f'lumenairy.{name} is not the object {where}.{name} -- a re-export '
        f'must alias, never copy.')


@pytest.mark.parametrize('name', [
    'seed_entrance_eikonal', 'exit_vertex_transfer',
    'EXIT_VERTEX_GRAZING_TOL', 'resolve_exit_index',
    'vertex_plane_transfer_t', 'exit_vertex_transfer_jax',
])
def test_raytrace_package_also_carries_the_exit_vertex_family(name):
    """WP-A7 section 5.3: ``analysis/image_plane_wfe.py`` had to reach into
    ``..raytrace.trace`` because the package did not re-export the seeder,
    while its sibling ``exit_vertex_transfer`` was already there.  Both ends
    of that asymmetry are closed."""
    import lumenairy.raytrace as rt
    assert hasattr(rt, name)
    assert name in rt.__all__
    assert getattr(rt, name) is getattr(la, name)


def test_pmm_facade_exports_the_order_drift_helper():
    """VERIFY-A12 added it to ``elements/pmm/__init__.py::__all__``; the
    top-level half (WP-A13 section 5.1) is what this work package added.
    Both halves pinned here so neither can be reverted alone."""
    from lumenairy.elements import pmm
    assert 'pmm_2d_order_drift' in pmm.__all__
    assert hasattr(pmm, 'pmm_2d_order_drift')
    assert pmm.pmm_2d_order_drift is la.pmm_2d_order_drift


def test_from_prescription_stays_a_classmethod_only():
    """The ONE request deliberately not taken (WP-A11 section 5.1, which
    offered both readings).

    ``lumenairy.algebra.from_prescription`` is a MODULE whose ``__all__``
    exports a function of the same name; the top-level public spelling is
    :meth:`lumenairy.Operator.from_prescription`.  Adding a free function
    would put two names with identical semantics on a 722-name surface, which
    is what the ``__all__``-symmetry walker's own exemption registry argues
    against -- so the exemption stands and this test records the decision so
    it is not silently re-litigated.
    """
    assert 'from_prescription' not in la.__all__
    assert hasattr(la.Operator, 'from_prescription')
    import lumenairy.algebra as alg
    assert callable(alg.from_prescription)


def test_no_duplicate_entries_in_the_top_level_all():
    assert len(la.__all__) == len(set(la.__all__))


def test_every_new_name_has_a_docstring():
    """CONVENTIONS: new public API needs a docstring in the surrounding
    style.  Constants are exempt (they carry a ``#:`` comment instead)."""
    undocumented = []
    for name in _REEXPORTS:
        obj = getattr(la, name)
        if isinstance(obj, float):
            continue                              # EXIT_VERTEX_GRAZING_TOL
        if not (getattr(obj, '__doc__', None) or '').strip():
            undocumented.append(name)
    assert not undocumented, (
        f'newly exported public names with no docstring: {undocumented}')

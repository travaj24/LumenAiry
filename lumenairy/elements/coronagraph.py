"""
lumenairy.elements.coronagraph -- coronagraph element builders.

Pure namespace module (4.3.0+) that re-exports the six coronagraph
element factories for discoverability.  All functions remain defined
in :mod:`lumenairy.elements.elements`; this module exists so users
who explore the package layout can find the coronagraph builders
together as a family rather than scattered across the general
elements file.

Nothing inside the library imports this module -- that is by design
(it is a user-facing discoverability alias, documented in the README),
NOT dead code.  ``tests/unit/test_niche_audit_w3_elements.py`` pins the
re-export set against ``lumenairy.elements`` so the two cannot drift
(v5.30, audit E-L18: the module had zero coverage and its docstring
said "four" while it re-exported six).

Element factories
-----------------
* :func:`apply_lyot_focal_plane_mask` -- on-axis amplitude block in
  the focal plane (classical Lyot coronagraph).
* :func:`apply_vortex_phase_mask` -- topological vortex phase in
  the focal plane (charge-2 / charge-4 vortex coronagraph).
* :func:`apply_lyot_stop` -- pupil-plane Lyot stop (annular or
  apodized).
* :func:`apply_apodized_pupil` -- pupil apodization (cos2 / cos_power
  / gaussian / sonine) for shaped-pupil designs.
* :func:`create_four_quadrant_phase_mask` -- canonical four-quadrant
  phase-mask coronagraph (Rouan 2000) as a complex (N, N) ndarray.
* :func:`create_eight_octant_phase_mask` -- canonical eight-octant
  phase-mask coronagraph (Murakami 2008) as a complex (N, N) ndarray.

For the analysis side -- post-coronagraph contrast curves --
see :mod:`lumenairy.analysis.coronagraph`.
"""

from .elements import (
    apply_apodized_pupil,
    apply_lyot_focal_plane_mask,
    apply_lyot_stop,
    apply_vortex_phase_mask,
    create_eight_octant_phase_mask,
    create_four_quadrant_phase_mask,
)

__all__ = [
    'apply_lyot_focal_plane_mask',
    'apply_vortex_phase_mask',
    'apply_lyot_stop',
    'apply_apodized_pupil',
    # v5.4 Phase 5: phase-mask builders (FQPM + 8OPM).
    'create_four_quadrant_phase_mask',
    'create_eight_octant_phase_mask',
]

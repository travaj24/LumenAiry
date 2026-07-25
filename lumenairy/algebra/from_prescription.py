"""lumenairy.algebra.from_prescription -- prescription bridge.

Provides :func:`from_prescription`, a factory that walks a LumenAiry
prescription dict (via :func:`lumenairy.raytrace.surfaces_from_prescription`)
and produces an equivalent :class:`CompositeOperator`.

The resulting CompositeOperator's ABCD matches
``system_abcd(surfaces, wavelength)`` to within numerical roundoff
(``< 1e-12`` absolute on representative singlet / doublet /
telephoto prescriptions; pinned in
``tests/unit/test_v4_15_1_agent_g_matches_system_abcd.py``).  The
field-application path is paraxial-only (each refractive surface
becomes a :class:`ThinLens` with ``f = 1/phi`` where
``phi = (n2 - n1) / R``) -- for high-NA / freeform / aspheric
prescriptions, users should call :func:`propagate_through_system`
on the original prescription rather than reduce it to operators.

Mirror surfaces are handled per the Welford convention used by
:func:`system_abcd` -- a single mirror flips the propagation sign
for every subsequent thickness, matching ``mirror_parity`` in the
ground-truth implementation.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import CompositeOperator, Operator
from .primitives import FreeSpace, ThinLens


def from_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    method: str = 'auto',
) -> CompositeOperator:
    """Build a :class:`CompositeOperator` from a LumenAiry prescription.

    Walks the prescription's surface list and emits one
    :class:`ThinLens` (or identity, for flat surfaces) per refractive
    surface plus one :class:`FreeSpace` per thickness.  Mirrors flip
    the propagation sign so subsequent reduced-thickness terms
    advance in the correct direction, matching the Welford convention
    used by :func:`lumenairy.raytrace.system_abcd`.

    Parameters
    ----------
    prescription : dict
        LumenAiry prescription dict.  Must satisfy
        :func:`lumenairy.raytrace.validate_prescription`.
    wavelength : float
        Vacuum wavelength [m] -- used to resolve glass indices.
    method : str, default ``'auto'``
        Propagator method passed to every emitted :class:`FreeSpace`.

    Returns
    -------
    CompositeOperator
        Operator chain whose ``.abcd`` matches ``system_abcd(...)``
        to within ``1e-12`` absolute.

    Notes
    -----
    **Paraxial only.**  The emitted thin-lens phase masks are the
    closed-form paraxial reduction of each refractive surface; they
    do NOT capture spherical aberration, coma, distortion, or any
    higher-order departure from the paraxial limit.  For
    production simulation through a real prescription use
    :func:`lumenairy.propagate_through_system`, which honours sag,
    aspheric coefficients, biconic / freeform departures, glass
    chromatic dispersion at the full spectrum, etc.  The
    operator-algebra path is for **algebraic analysis**
    (composition with FreeSpace segments to read system ABCDs,
    overlay 4f systems on prescription-driven sims, etc.) -- not
    for high-fidelity diffraction.

    See also
    --------
    lumenairy.raytrace.system_abcd : Ground-truth ABCD for tests.
    lumenairy.propagate_through_system : Full prescription-driven
        wave propagation.

    Examples
    --------
    >>> import lumenairy as la
    >>> rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3,
    ...                       glass='N-BK7', aperture=25.4e-3)
    >>> sys = la.Operator.from_prescription(rx, wavelength=633e-9)
    >>> abs(sys.efl - 50e-3) < 5e-3  # close to a 50mm-radius singlet
    True
    """
    # Defer heavy raytrace imports to call-time so the algebra
    # subpackage stays cheap to ``import``.
    from ..glass import get_glass_index
    from ..raytrace import surfaces_from_prescription

    try:
        wl = float(wavelength)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"from_prescription: wavelength must be a real number, got "
            f"{wavelength!r}."
        ) from e
    if not np.isfinite(wl) or wl <= 0.0:
        raise ValueError(
            f"from_prescription: wavelength must be positive and "
            f"finite, got {wl}."
        )

    surfaces = surfaces_from_prescription(prescription)
    if not surfaces:
        raise ValueError(
            "from_prescription: prescription produced an empty surface "
            "list.  Check the input dict has at least one surface."
        )

    chain: List[Operator] = []
    mirror_parity = 0  # 0 = unflipped, 1 = post-odd-mirror (n -> -n)

    for i, surf in enumerate(surfaces):
        sign = 1.0 if mirror_parity == 0 else -1.0

        # 3.7.0 (system_abcd): coord-breaks are frame transforms with
        # no optical power; they only contribute the thickness gap.
        if getattr(surf, 'is_coordbrk', False):
            if i < len(surfaces) - 1:
                t = surf.thickness
                n_after = sign * get_glass_index(surf.glass_after, wl)
                # Reduced thickness for FreeSpace ABCD.  The geometric
                # propagation through air of physical thickness t in a
                # medium of index n_after has paraxial ABCD
                # [[1, t/n_after], [0, 1]] -- this is the
                # "reduced thickness" used throughout the system_abcd
                # branch.
                d_eff = t / n_after
                if d_eff != 0.0:
                    chain.append(FreeSpace(d_eff, method=method))
            continue

        n1 = sign * get_glass_index(surf.glass_before, wl)
        n2 = sign * get_glass_index(surf.glass_after, wl)
        R = surf.radius

        if surf.is_mirror:
            # Welford mirror: n2 = -n1, phi = -2 n1 / R.
            #
            # S11-1 (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): the
            # mirror branch is R-INDEPENDENT.  ``n2 = -n1`` and the
            # parity toggle encode the REFLECTION, not the power: a flat
            # fold has phi = 0 but still reverses the propagation
            # direction, so the post-fold reduced thickness ``t / n2``
            # must carry the flipped sign.
            #
            # v4.15.2 (audit P1-NEW-B) had made this layer skip the
            # parity toggle for FLAT mirrors specifically to match
            # ``raytrace.system_abcd``, whose own ``elif surf.is_mirror
            # and np.isfinite(R)`` gating dropped it.  That made the two
            # layers agree on the WRONG answer -- ``system_abcd`` is now
            # fixed (see the S11-1 note in raytrace/seidel.py, with the
            # exact-3-D-trace oracle numbers), so this twin follows it
            # back to the R-independent form.  Bit-identical for curved
            # mirrors and for every mirror-free prescription.
            n2 = -n1
            if np.isfinite(R):
                phi = (n2 - n1) / R
                if phi != 0.0:
                    f_eff = 1.0 / phi
                    chain.append(ThinLens(f_eff))
            # Toggle mirror parity for the post-mirror legs.
            mirror_parity ^= 1
        elif np.isfinite(R):
            # Refractive curved surface: power phi = (n2 - n1)/R.
            phi = (n2 - n1) / R
            if phi != 0.0:
                f_eff = 1.0 / phi
                chain.append(ThinLens(f_eff))
            # phi == 0 (rare: same glass on both sides with curved
            # radius, or n2 == n1 from a chromatic crossover) is the
            # identity, intentionally skipped.
        else:
            # Flat refractive surface: no power contribution.
            pass

        # Thickness gap to next surface.  ``n2`` here already
        # reflects any mirror sign flip on this surface.
        if i < len(surfaces) - 1:
            t = surf.thickness
            d_eff = t / n2
            if d_eff != 0.0:
                chain.append(FreeSpace(d_eff, method=method))

    if not chain:
        # Degenerate prescription (e.g. single flat surface with no
        # thickness).  Emit a single identity ThinLens(inf) so the
        # returned object is still a CompositeOperator with the
        # expected interface.
        chain.append(ThinLens(np.inf))

    return CompositeOperator(chain)


__all__ = ['from_prescription']

"""
lumenairy.propagators.mhs -- Multiple Huygens Surface (MHS)
ray tracing.

Hybrid wave / ray propagator following the framework introduced in
the IEEE 2023 paper on Multiple Huygens Surface ray tracing.  Splits
the propagation volume into subdomains separated by Huygens surfaces;
within each subdomain, rays propagate geometrically (so the system's
local refraction is captured exactly).  At each Huygens surface, the
ray bundle is converted to a complex field via a Huygens-surface
integral, and that field is the new source for the next subdomain.

This complements the existing propagators in three regimes:

* **HFPI / GBD** propagate from a single source plane to a single
  output plane.  MHS partitions a long propagation chain into
  multiple sub-problems, each amenable to its own propagator.
* **Wave-only methods** (ASM, Fresnel) use a fixed grid throughout
  and break down at refractive interfaces.  MHS lets each subdomain
  use its own grid + propagator.
* **Pure raytrace** loses diffraction.  MHS interleaves rays with
  field reconstruction at every Huygens surface.

The MHS framework is most useful when:

* You have a system with multiple natural "planes" where the field
  must be sampled (intermediate image planes, pupil planes,
  apertures with diffraction).
* Different parts of the system are best handled by different
  propagators (raytrace through a thick lens, ASM through free
  space, GBD through a smooth pupil, HFPI through an aperture).

This module provides the **structural framework**: ``HuygensSurface``
data type, ``MhsPipeline`` class, and helpers to compose subdomain
propagators.  The actual per-subdomain propagation calls into the
existing :mod:`lumenairy.propagators.propagation`,
:mod:`lumenairy.propagators.gbd`, etc.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..backend import array_namespace


@dataclass
class HuygensSurface:
    """A flat 2-D Huygens surface in the propagation chain.

    Attributes
    ----------
    z : float
        Axial position [m].
    Ny, Nx : int
        Grid sampling on this surface.
    dx : float
        Grid pitch [m] (assumed isotropic).
    centre : (float, float)
        Transverse centre of the grid (default origin).
    label : str, optional
        Human-readable label for diagnostics ("aperture", "pupil",
        "image plane", ...).
    """
    z: float
    Ny: int
    Nx: int
    dx: float
    centre: Tuple[float, float] = (0.0, 0.0)
    label: str = ''

    def grid(self):
        """Return ``(X, Y)`` meshgrid arrays for this surface."""
        cx, cy = self.centre
        x = (np.arange(self.Nx) - self.Nx / 2 + 0.5) * self.dx + cx
        y = (np.arange(self.Ny) - self.Ny / 2 + 0.5) * self.dx + cy
        return np.meshgrid(x, y, indexing='xy')


@dataclass
class MhsSubdomain:
    """One subdomain of an MHS pipeline.

    A subdomain spans from one Huygens surface to the next, and
    has an associated propagator that maps a complex field from
    the start surface to the end surface.

    Attributes
    ----------
    propagator : callable
        ``f(E_in, in_surface, out_surface, **kwargs) -> E_out``
        where ``E_in`` is the complex field on ``in_surface``,
        and ``E_out`` is on ``out_surface``.
    in_surface : HuygensSurface
    out_surface : HuygensSurface
    kwargs : dict
        Extra keyword arguments forwarded to ``propagator``.
    label : str, optional
    """
    propagator: Callable
    in_surface: HuygensSurface
    out_surface: HuygensSurface
    kwargs: dict = field(default_factory=dict)
    label: str = ''


class MhsPipeline:
    """Compose multiple subdomain propagators into a single chain.

    Each subdomain has its own propagator (ASM, Fresnel, GBD,
    HFPI, prescription-based, etc.); the pipeline applies them in
    sequence and returns the field at every Huygens surface for
    inspection / diagnostics.

    Example::

        from lumenairy.propagators.mhs import (
            HuygensSurface, MhsSubdomain, MhsPipeline,
        )

        # Three Huygens surfaces along the chain.
        s0 = HuygensSurface(z=0.0, Ny=64, Nx=64, dx=5e-6, label='source')
        s1 = HuygensSurface(z=5e-3, Ny=64, Nx=64, dx=5e-6, label='aperture')
        s2 = HuygensSurface(z=10e-3, Ny=64, Nx=64, dx=5e-6, label='image')

        # Two subdomains with different propagators.
        def asm_subdomain(E, in_s, out_s, **kw):
            from lumenairy.propagators.propagation import angular_spectrum_propagate
            return angular_spectrum_propagate(
                E, z=out_s.z - in_s.z, wavelength=kw['wavelength'],
                dx=in_s.dx,
            )

        sub1 = MhsSubdomain(asm_subdomain, s0, s1, kwargs=dict(wavelength=633e-9))
        sub2 = MhsSubdomain(asm_subdomain, s1, s2, kwargs=dict(wavelength=633e-9))

        pipe = MhsPipeline([sub1, sub2])
        fields = pipe.run(E_in)        # list of (surface, E_at_surface)
    """

    def __init__(self, subdomains: List[MhsSubdomain]):
        self.subdomains = list(subdomains)
        self._validate()

    def _validate(self):
        for i in range(len(self.subdomains) - 1):
            cur = self.subdomains[i]
            nxt = self.subdomains[i + 1]
            if cur.out_surface is not nxt.in_surface:
                # Allow distinct objects if their grids match.
                if (cur.out_surface.z != nxt.in_surface.z
                        or cur.out_surface.Ny != nxt.in_surface.Ny
                        or cur.out_surface.Nx != nxt.in_surface.Nx
                        or cur.out_surface.dx != nxt.in_surface.dx):
                    raise ValueError(
                        f"MHS subdomain mismatch: subdomain {i} ends at "
                        f"surface {cur.out_surface.label or cur.out_surface.z} "
                        f"but subdomain {i+1} starts at "
                        f"{nxt.in_surface.label or nxt.in_surface.z}.")

    @property
    def n_subdomains(self):
        return len(self.subdomains)

    def surfaces(self) -> List[HuygensSurface]:
        """Return the ordered list of surfaces along the chain."""
        out = [self.subdomains[0].in_surface]
        for sub in self.subdomains:
            out.append(sub.out_surface)
        return out

    def run(self, E_in,
            return_intermediate: bool = True
            ) -> Union[List[Tuple[HuygensSurface, object]], object]:
        """Run the pipeline.

        Parameters
        ----------
        E_in : array (Ny, Nx) complex
            Input field at the first surface.
        return_intermediate : bool
            If True (default), return the field at every surface
            along the chain, paired with that surface.  If False,
            return only the final-surface field.

        Returns
        -------
        list of (HuygensSurface, array) | array
        """
        E_current = E_in
        if return_intermediate:
            history = [(self.subdomains[0].in_surface, E_in)]

        for sub in self.subdomains:
            E_next = sub.propagator(E_current, sub.in_surface,
                                     sub.out_surface, **sub.kwargs)
            if return_intermediate:
                history.append((sub.out_surface, E_next))
            E_current = E_next

        if return_intermediate:
            return history
        return E_current


# ---------------------------------------------------------------------------
# Convenience builders for common subdomain patterns
# ---------------------------------------------------------------------------

def asm_subdomain(in_surface: HuygensSurface,
                   out_surface: HuygensSurface,
                   *,
                   wavelength: float,
                   bandlimit: bool = True) -> MhsSubdomain:
    """Build an MHS subdomain that uses Angular Spectrum free-space
    propagation between two Huygens surfaces."""
    from .propagation import angular_spectrum_propagate

    def _prop(E, in_s, out_s, **kw):
        return angular_spectrum_propagate(
            E, z=out_s.z - in_s.z,
            wavelength=kw['wavelength'],
            dx=in_s.dx,
            bandlimit=kw['bandlimit'],
        )

    return MhsSubdomain(
        propagator=_prop,
        in_surface=in_surface,
        out_surface=out_surface,
        kwargs={'wavelength': wavelength, 'bandlimit': bandlimit},
        label='asm',
    )


def aperture_subdomain(in_surface: HuygensSurface,
                        aperture_radius: float,
                        *,
                        shape: str = 'circular',
                        centre: Tuple[float, float] = (0.0, 0.0)
                        ) -> MhsSubdomain:
    """Build an MHS subdomain that applies a hard aperture mask in
    place (in_surface == out_surface; same z).  Useful as a thin
    "operator" subdomain between two propagation legs."""
    if not (shape in ('circular', 'square')):
        raise ValueError(
            f"aperture_subdomain shape must be 'circular' or 'square', got {shape!r}.")

    out_surface = in_surface  # zero-thickness operator

    def _prop(E, in_s, out_s, **kw):
        xp = array_namespace(E)
        cx, cy = centre
        x = (xp.arange(in_s.Nx) - in_s.Nx / 2 + 0.5) * in_s.dx + in_s.centre[0]
        y = (xp.arange(in_s.Ny) - in_s.Ny / 2 + 0.5) * in_s.dx + in_s.centre[1]
        X, Y = xp.meshgrid(x, y, indexing='xy')
        if shape == 'circular':
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= aperture_radius ** 2
        else:
            mask = (xp.abs(X - cx) <= aperture_radius) & (xp.abs(Y - cy) <= aperture_radius)
        return E * mask.astype(E.dtype)

    return MhsSubdomain(
        propagator=_prop,
        in_surface=in_surface,
        out_surface=out_surface,
        kwargs={},
        label=f'aperture[{shape}, r={aperture_radius:.3e}]',
    )


def gbd_freespace_subdomain(in_surface: HuygensSurface,
                              out_surface: HuygensSurface,
                              *,
                              wavelength: float,
                              waist_factor: float = 1.0,
                              sample_step: int = 1,
                              chunk_beamlets: int = 4096
                              ) -> MhsSubdomain:
    """MHS subdomain that uses GBD free-space propagation."""
    from .gbd import propagate_gbd_freespace

    def _prop(E, in_s, out_s, **kw):
        return propagate_gbd_freespace(
            E, in_s.dx,
            z=out_s.z - in_s.z,
            wavelength=kw['wavelength'],
            output_grid=(out_s.Ny, out_s.Nx),
            output_dx=out_s.dx,
            output_centre=out_s.centre,
            waist_factor=kw['waist_factor'],
            sample_step=kw['sample_step'],
            chunk_beamlets=kw['chunk_beamlets'],
        )

    return MhsSubdomain(
        propagator=_prop,
        in_surface=in_surface,
        out_surface=out_surface,
        kwargs={
            'wavelength': wavelength,
            'waist_factor': waist_factor,
            'sample_step': sample_step,
            'chunk_beamlets': chunk_beamlets,
        },
        label='gbd_freespace',
    )


def prescription_subdomain(in_surface: HuygensSurface,
                            out_surface: HuygensSurface,
                            prescription: dict,
                            *,
                            wavelength: float,
                            method: str = 'maslov',
                            **method_kwargs) -> MhsSubdomain:
    """MHS subdomain that uses the dispatcher to propagate through a
    full prescription between two Huygens surfaces.

    ``method`` is forwarded to
    :func:`lumenairy.propagators.dispatch.propagate`.
    """
    from .dispatch import propagate

    def _prop(E, in_s, out_s, **kw):
        return propagate(
            E,
            wavelength=kw['wavelength'],
            dx=in_s.dx,
            prescription=kw['prescription'],
            method=kw['method'],
            output_grid=(out_s.Ny, out_s.Nx),
            output_dx=out_s.dx,
            **{k: v for k, v in kw.items()
               if k not in ('wavelength', 'prescription', 'method')},
        )

    kwargs = {
        'wavelength': wavelength,
        'prescription': prescription,
        'method': method,
    }
    kwargs.update(method_kwargs)

    return MhsSubdomain(
        propagator=_prop,
        in_surface=in_surface,
        out_surface=out_surface,
        kwargs=kwargs,
        label=f'prescription[{method}]',
    )


__all__ = [
    'HuygensSurface',
    'MhsSubdomain',
    'MhsPipeline',
    'asm_subdomain',
    'aperture_subdomain',
    'gbd_freespace_subdomain',
    'prescription_subdomain',
]

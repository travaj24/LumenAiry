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
            return_intermediate: bool = True,
            checkpoint=None,
            store=None,
            label_prefix: str = 'mhs',
            return_result: bool = False,
            wavelength: float = 0.0,
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
        checkpoint : callable, optional
            ``checkpoint(idx, surface, field)`` is called after every
            subdomain finishes, including the input plane (idx=0,
            surface=first in_surface, field=E_in).  Useful for
            streaming progress bars or custom persistence.  Returning
            from the callback is fine; raising aborts the run.
        store : str, pathlib.Path, or storage handle, optional
            If given, each plane is appended to a unified Zarr / HDF5
            store via :func:`lumenairy.io.storage.append_plane`.
            ``label_prefix`` controls the per-plane key, which is
            ``f"{label_prefix}_{idx:03d}_{surface.label or 'plane'}"``.
            The store's parent directory must exist; the file will be
            created on first write.
        label_prefix : str
            Prefix for plane labels when ``store`` is provided.
        return_result : bool, default False
            If True, return a
            :class:`lumenairy.propagators.PropagationResult` carrying
            the final field plus a ``history`` list of
            ``(label, field, dx)`` tuples for every Huygens surface
            (including the input plane).  ``return_intermediate`` is
            superseded -- the result always holds the full history.
        wavelength : float, default 0.0
            Forwarded onto the ``PropagationResult.wavelength`` field
            when ``return_result=True``.  Default 0 keeps the legacy
            (non-result) call signature unchanged.

        Returns
        -------
        list of (HuygensSurface, array) | array | PropagationResult
        """
        if store is not None:
            from ..io.storage import append_plane

        def _persist(idx, surface, field):
            if checkpoint is not None:
                checkpoint(idx, surface, field)
            if store is not None:
                label = (f"{label_prefix}_{idx:03d}_"
                         f"{surface.label or 'plane'}")
                append_plane(store, field, surface.dx,
                              z=surface.z, label=label)

        first_surf = self.subdomains[0].in_surface
        _persist(0, first_surf, E_in)

        E_current = E_in
        # Always retain (surface, field) tuples internally when either
        # return_intermediate or return_result is set; for legacy
        # call sites that pass return_intermediate=False (default
        # before this change), we skip the list entirely.
        track_history = return_intermediate or return_result
        if track_history:
            history = [(first_surf, E_in)]

        for i, sub in enumerate(self.subdomains, start=1):
            E_next = sub.propagator(E_current, sub.in_surface,
                                     sub.out_surface, **sub.kwargs)
            _persist(i, sub.out_surface, E_next)
            if track_history:
                history.append((sub.out_surface, E_next))
            E_current = E_next

        if return_result:
            from .result import PropagationResult
            last_surf = self.subdomains[-1].out_surface
            return PropagationResult(
                field=E_current,
                dx=last_surf.dx,
                wavelength=float(wavelength),
                z=last_surf.z,
                method='mhs',
                history=[
                    (s.label or f'plane_{j}', f, s.dx)
                    for j, (s, f) in enumerate(history)
                ],
            )
        if return_intermediate:
            return history
        return E_current

    @classmethod
    def from_prescription(cls,
                           prescription: dict,
                           *,
                           wavelength: float,
                           dx: float,
                           Ny: int,
                           Nx: int,
                           pre_distance: float = 0.0,
                           post_distance: float = 0.0,
                           method: str = 'gbd',
                           **method_kwargs) -> 'MhsPipeline':
        """Build a 3-subdomain ASM -> prescription -> ASM pipeline.

        Convenience for the common case "free-space lead-in + lens
        prescription + free-space lead-out".  The lens block uses the
        chosen ``method`` (any prescription-capable propagator:
        ``'gbd'``, ``'hf'``, ``'hfpi'``, ``'maslov'``).  Free-space
        legs use Angular Spectrum.

        Parameters
        ----------
        prescription : dict
            Lens prescription (``make_singlet`` / Zemax-loaded /
            user-built).
        wavelength : float
            Wavelength [m].
        dx : float
            Sample spacing [m] for the input plane.  All four planes
            share this spacing.
        Ny, Nx : int
            Grid shape, used for all surfaces.
        pre_distance : float, default 0
            Distance from input plane to the first lens surface.  Set
            to 0 for the input plane to coincide with the lens vertex.
        post_distance : float, default 0
            Distance from the last lens surface to the output plane.
            Set to 0 for the output plane to coincide with the back
            vertex.  Set to e.g. ``bfl`` to land on the image plane.
        method : str, default 'gbd'
            Propagator routed via :func:`la.propagate` for the lens
            block.
        **method_kwargs
            Forwarded to the chosen lens propagator.
        """
        # Reference z=0 at the front lens vertex; pre-input plane sits
        # at z=-pre_distance; post-output at z = thickness_total +
        # post_distance.
        thicknesses = list(prescription.get('thicknesses', []))
        thickness_total = float(sum(thicknesses)) if thicknesses else 0.0

        s_input = HuygensSurface(z=-pre_distance, Ny=Ny, Nx=Nx, dx=dx,
                                  label='mhs:input')
        s_pre_lens = HuygensSurface(z=0.0, Ny=Ny, Nx=Nx, dx=dx,
                                     label='mhs:pre-lens')
        s_post_lens = HuygensSurface(z=thickness_total, Ny=Ny, Nx=Nx,
                                      dx=dx, label='mhs:post-lens')
        s_output = HuygensSurface(z=thickness_total + post_distance,
                                   Ny=Ny, Nx=Nx, dx=dx,
                                   label='mhs:output')

        subs = []
        if pre_distance > 0:
            subs.append(asm_subdomain(s_input, s_pre_lens,
                                       wavelength=wavelength))
        subs.append(prescription_subdomain(
            s_pre_lens if pre_distance > 0 else s_input,
            s_post_lens,
            prescription, wavelength=wavelength, method=method,
            **method_kwargs))
        if post_distance > 0:
            subs.append(asm_subdomain(s_post_lens, s_output,
                                       wavelength=wavelength))
        return cls(subs)


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

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` meshgrid arrays for this surface."""
        cx, cy = self.centre
        # v4.12.1 (B1-10): pixel-centred `(arange(N) - N/2)*dx`, matching
        # the library-wide convention (ASM, Fresnel, RS, sources,
        # ``apply_fresnel_curvature``).  A Huygens surface's grid is the
        # physical grid that the propagated field is sampled on, so it
        # must match the upstream / downstream propagator grids exactly
        # to keep coherent overlays phase-aligned.
        x = (np.arange(self.Nx) - self.Nx / 2) * self.dx + cx
        y = (np.arange(self.Ny) - self.Ny / 2) * self.dx + cy
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

    def __init__(self, subdomains: List[MhsSubdomain]) -> None:
        self.subdomains = list(subdomains)
        self._validate()

    def _validate(self) -> None:
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
    def n_subdomains(self) -> int:
        return len(self.subdomains)

    def surfaces(self) -> List[HuygensSurface]:
        """Return the ordered list of surfaces along the chain."""
        out = [self.subdomains[0].in_surface]
        for sub in self.subdomains:
            out.append(sub.out_surface)
        return out

    def run(
        self,
        E_in: np.ndarray,
        return_intermediate: bool = True,
        checkpoint: Optional[Callable] = None,
        store: Optional[Any] = None,
        label_prefix: str = 'mhs',
        return_result: bool = False,
        wavelength: float = 0.0,
    ) -> Union[List[Tuple[HuygensSurface, np.ndarray]], np.ndarray, Any]:
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
    def from_prescription(
        cls,
        prescription: Dict[str, Any],
        *,
        wavelength: float,
        dx: float,
        Ny: int,
        Nx: int,
        pre_distance: float = 0.0,
        post_distance: float = 0.0,
        method: str = 'gbd',
        **method_kwargs: Any,
    ) -> 'MhsPipeline':
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

def asm_subdomain(
    in_surface: HuygensSurface,
    out_surface: HuygensSurface,
    *,
    wavelength: float,
    bandlimit: bool = True,
) -> MhsSubdomain:
    """Build an MHS subdomain that uses Angular Spectrum free-space
    propagation between two Huygens surfaces."""
    from .propagation import angular_spectrum_propagate

    # 4.10: ASM preserves grid pitch, so in_surface.dx must equal
    # out_surface.dx.  Pre-4.10 silently ignored a mismatch and
    # labelled the output field with out_surface.dx while the data
    # was actually at in_surface.dx, corrupting any downstream
    # subdomain that consumed the labelled coordinates.
    if abs(in_surface.dx - out_surface.dx) > 1e-12 * in_surface.dx:
        raise ValueError(
            f"asm_subdomain: in_surface.dx ({in_surface.dx:.6e} m) "
            f"differs from out_surface.dx ({out_surface.dx:.6e} m); "
            f"ASM preserves grid pitch.  Use "
            f"`angular_spectrum_propagate_mft` for a grid-changing "
            f"variant, or insert an intermediate resample step.")

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


def aperture_subdomain(
    in_surface: HuygensSurface,
    aperture_radius: float,
    *,
    shape: str = 'circular',
    centre: Tuple[float, float] = (0.0, 0.0),
) -> MhsSubdomain:
    """Build an MHS subdomain that applies a hard aperture mask in
    place (in_surface == out_surface; same z).  Useful as a thin
    "operator" subdomain between two propagation legs."""
    if shape not in ('circular', 'square'):
        raise ValueError(
            f"aperture_subdomain shape must be 'circular' or 'square', got {shape!r}.")

    out_surface = in_surface  # zero-thickness operator

    # ``out_s`` / ``**_kw`` are unused here but are NOT removable (audit
    # P13): :meth:`MhsPipeline.run` calls every subdomain uniformly as
    # ``sub.propagator(E, sub.in_surface, sub.out_surface, **sub.kwargs)``,
    # so the third positional parameter is part of the propagator protocol
    # and dropping it raises ``TypeError``.  This is a zero-thickness
    # operator (``out_surface is in_surface``), which is exactly why it has
    # nothing to read from ``out_s``.  Underscore-prefixed to mark the
    # non-use as deliberate.
    def _prop(E, in_s, _out_s, **_kw):
        xp = array_namespace(E)
        cx, cy = centre
        # v4.12.1 (B1-10): pixel-centred grid (drop the `+0.5`), matches
        # ``HuygensSurface.grid`` and the library-wide convention so the
        # aperture mask aligns 1:1 with the field sample positions.
        x = (xp.arange(in_s.Nx) - in_s.Nx / 2) * in_s.dx + in_s.centre[0]
        y = (xp.arange(in_s.Ny) - in_s.Ny / 2) * in_s.dx + in_s.centre[1]
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


def gbd_freespace_subdomain(
    in_surface: HuygensSurface,
    out_surface: HuygensSurface,
    *,
    wavelength: float,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
) -> MhsSubdomain:
    """MHS subdomain that uses GBD free-space propagation."""
    from .gbd import propagate_gbd_freespace

    def _prop(E, in_s, out_s, **kw):
        return propagate_gbd_freespace(
            E, in_s.dx,
            z=out_s.z - in_s.z,
            wavelength=kw['wavelength'],
            # MHS-1: v5.2 renamed the shape-only kwarg to ``output_shape``;
            # ``output_grid`` is the deprecated shim (fires a warning every
            # subdomain call and breaks when the shim is removed).
            output_shape=(out_s.Ny, out_s.Nx),
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


def prescription_subdomain(
    in_surface: HuygensSurface,
    out_surface: HuygensSurface,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    method: str = 'maslov',
    **method_kwargs: Any,
) -> MhsSubdomain:
    """MHS subdomain that uses the dispatcher to propagate through a
    full prescription between two Huygens surfaces.

    ``method`` is forwarded to
    :func:`lumenairy.propagators.dispatch.propagate`.

    .. versionchanged:: 5.2
        v5.2 (AUDIT_V4_13_1 Part 2 P1-C, option a -- raise) raised
        ``ValueError`` at subdomain construction time when
        ``method='maslov'`` was paired with an ``out_surface`` declaring
        a grid that differed from ``in_surface``.  Pre-v5.2 the request
        was silently dropped (the maslov dispatcher branch does not
        forward ``output_grid`` / ``output_dx`` to
        :func:`apply_real_lens_maslov`) and downstream MHS pipeline
        stitching saw a mis-shaped intermediate field.

    .. versionchanged:: 5.2.3
        v5.2.3 (AUDIT_V4_13_1 P1-C substantive closure): the maslov
        branch now **actually resamples** onto ``out_surface`` instead
        of raising.  The propagation runs natively on the input grid
        (which is all the maslov kernel supports), then a one-step
        :func:`resample_field` lands the output on ``out_surface``.
        Total power is re-normalised across the resample so the
        resampling step preserves L2 energy to within the bicubic
        interpolator's numerical precision.

        For a non-resampling alternative -- when the resample step's
        bicubic interpolation error at the new grid's edges is not
        acceptable -- pass ``method='gbd'`` / ``'hfpi'`` / ``'hf'``
        (their underlying propagators sample the output grid natively
        via Bluestein chirp-Z or HFPI integration).

        The narrow retained-raise corner case: the maslov kernel itself
        requires a SQUARE input field (``E_in.shape[0] == E_in.shape[1]``)
        and SQUARE pixels (``dx == dy``).  If the caller hands the
        subdomain a non-square input grid or a non-square output grid,
        we raise at construction time (rather than letting the kernel
        raise mid-pipeline).

    .. versionchanged:: 5.30
        Both dispatcher calls now pass ``return_result=False`` explicitly
        (audit P5 / roadmap Part F1): v5.30 flipped
        :func:`~lumenairy.propagators.dispatch.propagate`'s default return to
        a :class:`~lumenairy.propagators.PropagationResult`, and this
        subdomain must hand a bare **field** to the MHS pipeline.  Outputs are
        bit-identical to pre-v5.30.  A ``return_result`` passed through
        ``**method_kwargs`` is therefore ignored rather than forwarded -- the
        subdomain's own return contract is the pipeline's, not the caller's.
    """
    from .dispatch import propagate
    from .mft import resample_field

    # v5.2.3 (AUDIT_V4_13_1 P1-C substantive closure): the maslov kernel
    # requires square inputs (``apply_real_lens_maslov`` enforces
    # ``E_in.shape[0] == E_in.shape[1]``) and isotropic pitch.  We only
    # need the post-resample path for maslov, so guard those upstream
    # invariants here so a non-square out_surface fails at construction
    # time with a clear message rather than mid-run inside the kernel.
    if method == 'maslov':
        if int(in_surface.Ny) != int(in_surface.Nx):
            raise ValueError(
                "prescription_subdomain(method='maslov'): the maslov "
                "kernel requires a square input grid (Ny == Nx); got "
                f"in_surface shape ({in_surface.Ny}, {in_surface.Nx}).  "
                "Pre-resample the input to a square grid, or pick a "
                "method that supports anamorphic inputs "
                "(``method='gbd'`` / ``'hfpi'`` / ``'hf'``)."
            )
        if int(out_surface.Ny) != int(out_surface.Nx):
            raise ValueError(
                "prescription_subdomain(method='maslov'): the maslov "
                "kernel's downstream resampler requires a square output "
                "grid (Ny == Nx); got out_surface shape "
                f"({out_surface.Ny}, {out_surface.Nx}).  Pick a method "
                "that supports anamorphic outputs (``method='gbd'`` / "
                "``'hfpi'`` / ``'hf'``)."
            )

    def _prop(E, in_s, out_s, **kw):
        # v5.2.3 (AUDIT_V4_13_1 P1-C substantive closure): for
        # ``method='maslov'`` the dispatcher's maslov branch produces
        # output on the INPUT grid (regardless of ``output_grid`` /
        # ``output_dx``).  When the caller declares a different
        # ``out_surface``, run the maslov kernel on the input grid and
        # then resample to the requested output grid via
        # :func:`resample_field`.  Power is re-normalised so the
        # resample is L2-energy preserving (matching the maslov kernel's
        # built-in ``normalize_output='power'`` contract).
        if kw['method'] == 'maslov':
            # v5.30 (audit P5 / roadmap F1, flip-day migration): pass
            # ``return_result=False`` explicitly.  This function's contract is
            # to hand a bare FIELD back to the MHS pipeline (it is measured
            # with ``np.abs`` / ``.dtype`` just below and stitched into the
            # next subdomain), and since v5.30 the dispatcher's DEFAULT return
            # is a ``PropagationResult``.  The roadmap's F1 inventory listed
            # this site as NOT flip-safe for exactly that reason; naming the
            # legacy contract keeps the output bit-identical.
            E_native = propagate(
                E,
                wavelength=kw['wavelength'],
                dx=in_s.dx,
                prescription=kw['prescription'],
                method=kw['method'],
                return_result=False,
                **{k: v for k, v in kw.items()
                   if k not in ('wavelength', 'prescription', 'method',
                                'return_result')},
            )
            same_shape = (int(in_s.Ny) == int(out_s.Ny)
                          and int(in_s.Nx) == int(out_s.Nx))
            same_dx = abs(float(in_s.dx) - float(out_s.dx)) < 1e-15
            if same_shape and same_dx:
                return E_native
            # Resample onto the declared output grid.  ``resample_field``
            # returns ``(E_out, dx_out)``; we only need the field.
            E_resampled, _ = resample_field(
                E_native,
                dx_in=float(in_s.dx),
                dx_out=float(out_s.dx),
                N_out=int(out_s.Nx),
                order=3,
            )
            # Re-normalise power across the resample so total energy is
            # preserved (bicubic interpolation introduces a small power
            # drift, mainly at the new grid's edges where it crops or
            # extends the support).  This matches the maslov kernel's
            # default ``normalize_output='power'`` contract.
            p_in = float(np.sum(np.abs(E_native) ** 2)) * (float(in_s.dx) ** 2)
            p_out = float(np.sum(np.abs(E_resampled) ** 2)) * (float(out_s.dx) ** 2)
            if p_out > 0.0 and p_in > 0.0:
                scale = float(np.sqrt(p_in / p_out))
                E_resampled = E_resampled * scale
            # Preserve dtype (resample_field promotes complex64 -> complex128
            # via map_coordinates' float64 output).
            if E_resampled.dtype != E_native.dtype:
                E_resampled = E_resampled.astype(E_native.dtype)
            return E_resampled

        # Non-maslov methods (gbd / hfpi / hf) honour ``output_grid`` /
        # ``output_dx`` natively in their underlying propagators.
        #
        # MHS-2: the dispatcher parses a *tuple* ``output_grid`` as
        # ``(N_out, dx_out)`` -- passing ``(out_s.Ny, out_s.Nx)`` mis-reads
        # ``out_s.Nx`` as a pitch (masked only because ``output_dx`` overrides
        # it).  Use the explicit ``{'N': ..., 'dx': ...}`` dict form.  Note the
        # dispatcher's ``_resolve_dispatcher_output_grid`` always resolves to a
        # SQUARE ``(N_out, N_out)`` grid, so a genuinely non-square out-surface
        # cannot be delivered through this path -- warn rather than silently
        # square it (the maslov-guard advertises gbd/hfpi/hf as anamorphic).
        if int(out_s.Ny) != int(out_s.Nx):
            import warnings as _w
            _w.warn(
                f"prescription_subdomain: non-square out_surface "
                f"({out_s.Ny}x{out_s.Nx}) requested for method "
                f"{kw['method']!r}; the dispatcher output-grid path resolves "
                f"to a square ({out_s.Ny}x{out_s.Ny}) grid.  Build the "
                f"sub-propagator directly if a true anamorphic output is "
                f"required.",
                RuntimeWarning, stacklevel=2)
        # v5.30 (audit P5 / roadmap F1, flip-day migration): ``return_result``
        # named explicitly for the same reason as the maslov branch above --
        # this return goes straight into the MHS pipeline as a field, so it
        # must stay the kernel's bare output now that the dispatcher default
        # is a ``PropagationResult``.
        return propagate(
            E,
            wavelength=kw['wavelength'],
            dx=in_s.dx,
            prescription=kw['prescription'],
            method=kw['method'],
            output_grid={'N': out_s.Ny, 'dx': out_s.dx},
            output_dx=out_s.dx,
            return_result=False,
            **{k: v for k, v in kw.items()
               if k not in ('wavelength', 'prescription', 'method',
                            'return_result')},
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

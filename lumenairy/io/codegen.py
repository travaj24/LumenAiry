"""
Zemax Prescription → Simulation Script Generator
==================================================

Generates a standalone Python script from a parsed Zemax prescription
that models the full optical system using the ``lumenairy``
library.  Each lens group is represented with ``apply_real_lens`` (multi-
surface split-step ASM), free-space gaps use ``angular_spectrum_propagate``,
mirrors use ``apply_mirror``, and aperture stops use ``apply_aperture``.

The generated script is self-contained: it imports the library, defines
the prescription data inline, builds an element list, and calls
``propagate_through_system`` (or an unrolled loop for more control).

This module is designed to be integrated into the ``lumenairy``
package (in ``lumenairy/codegen.py``) but does NOT modify any
existing modules.

Usage
-----
From a .zmx file::

    import lumenairy as la
    from lumenairy.codegen import generate_simulation_script

    rx = la.load_zemax_zmx('my_design.zmx')
    code = generate_simulation_script(rx, wavelength=1.31e-6)

    with open('sim_my_design.py', 'w') as f:
        f.write(code)

From a prescription text export::

    rx = la.load_zemax_prescription_data_txt('design-prescription.txt')
    code = generate_simulation_script(rx)

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

# CG-1 (AUDIT_IO_STORAGE_CODEGEN): the Forbes Q-type freeform keys the Zemax
# loader attaches to QBFS/QCON surfaces -- forwarded through codegen so a
# Q-type surface is not silently degraded to its base conic in the generated
# ``apply_real_lens`` script (which understands them via surface_sag_freeform).
_CODEGEN_FREEFORM_KEYS = ('freeform_type', 'q_bfs_coeffs', 'q_con_coeffs',
                          'r_max')

from ..glass import GLASS_REGISTRY
from .prescriptions import load_zemax_prescription_data_txt, load_zemax_zmx

# ============================================================================
# Public API
# ============================================================================

def generate_simulation_script(
    prescription: Dict[str, Any],
    wavelength: Optional[float] = None,
    N: int = 2048,
    dx: Optional[float] = None,
    source_sigma: Optional[float] = None,
    output_path: Optional[str] = None,
    style: str = 'unrolled',
    include_plotting: bool = True,
    include_analysis: bool = True,
    header_comment: Optional[str] = None,
) -> str:
    """
    Generate a Python simulation script from a parsed Zemax prescription.

    Takes the output of :func:`load_zemax_zmx` or
    :func:`load_zemax_prescription_data_txt` and produces a complete, runnable
    Python script that models the optical system using the
    ``lumenairy`` library.

    Each contiguous group of refracting surfaces (a "lens group") is
    represented as a single ``apply_real_lens`` call with the appropriate
    multi-surface prescription.  Free-space gaps between groups use
    ``angular_spectrum_propagate``.  Mirrors and aperture stops are
    included as separate elements.

    Parameters
    ----------
    prescription : dict
        Parsed Zemax prescription from :func:`load_zemax_zmx` or
        :func:`load_zemax_prescription_data_txt`.  Must contain keys:
        ``'elements'``, ``'all_thicknesses'``, ``'aperture_diameter'``,
        and ``'name'``.

    wavelength : float or None
        Operating wavelength [m].  If None, uses the wavelength stored
        in the prescription dict (if present).  If neither is supplied,
        a :class:`ValueError` is raised so that visible-band Zemax
        files do not silently convert to a 1310 nm NIR script
        (v4.13.0 onward; pre-v4.13.0 this defaulted to 1.31e-6).

    N : int, default 2048
        Grid size for the simulation (N x N).

    dx : float or None
        Grid spacing [m].  If None, auto-computed from the aperture
        diameter so the beam fits comfortably on the grid.

    source_sigma : float or None
        Gaussian source 1/e field radius [m].  If None, defaults to
        1/4 of the aperture diameter.

    output_path : str or None
        If given, write the generated script to this file path.
        Otherwise, the script is returned as a string only.

    style : str, default ``'unrolled'``
        Code generation style:

        ``'unrolled'``
            Each propagation step and element is written out as its own
            code block with comments.  Best for understanding and
            modifying the script.  Intermediate fields are stored for
            later analysis and plotting.

        ``'system'``
            Builds an element list and calls
            :func:`propagate_through_system` in a single invocation.
            More compact, less flexible.

    include_plotting : bool, default True
        Include ``matplotlib`` plotting code at the end of the script.

    include_analysis : bool, default True
        Include beam analysis calls (centroid, D4σ, power) at key planes.

    header_comment : str or None
        Custom comment block to insert at the top of the script.
        If None, an auto-generated header is used.

    Returns
    -------
    script : str
        The complete Python script as a string.

    Examples
    --------
    >>> import lumenairy as la
    >>> from lumenairy.codegen import generate_simulation_script
    >>>
    >>> rx = la.load_zemax_zmx('AC254-200-C.zmx')
    >>> code = generate_simulation_script(rx, wavelength=1.31e-6,
    ...                                   output_path='sim_ac254.py')
    """
    # ------------------------------------------------------------------
    # Resolve parameters
    # ------------------------------------------------------------------
    # S2b (v4.13.0): Pre-fix this silently defaulted to 1310 nm whenever
    # neither the user nor the prescription supplied a wavelength.  A
    # visible-band Zemax file converted to a NIR script with no warning.
    # The less-surprising behaviour is to raise: the simulation
    # downstream is wavelength-sensitive at every refraction, so a
    # bogus default produces a wrong-but-plausible output rather than
    # an outright error.
    if wavelength is None:
        wavelength = prescription.get('wavelength')
    if wavelength is None:
        raise ValueError(
            "generate_simulation_script: no wavelength supplied and the "
            "prescription dict has no 'wavelength' key.  Pass "
            "``wavelength=<value_in_meters>`` explicitly (e.g. "
            "``wavelength=587.6e-9`` for the d-line, ``wavelength=1.31e-6`` "
            "for telecom O-band).  Pre-v4.13.0 this defaulted to 1.31e-6 "
            "(1310 nm) silently, which silently mis-modelled visible-band "
            "designs."
        )

    aperture = prescription.get('aperture_diameter', 25.4e-3)
    sys_name = prescription.get('name', 'Zemax System')

    if dx is None:
        # Auto-size: at least 20 samples across the aperture radius
        dx = aperture / (N / 4)

    if source_sigma is None:
        source_sigma = aperture / 4

    # ------------------------------------------------------------------
    # Decompose the prescription into simulation steps
    # ------------------------------------------------------------------
    steps = _decompose_prescription(prescription)

    # ------------------------------------------------------------------
    # Collect all glass names that need to be in the registry
    # ------------------------------------------------------------------
    glasses_used = set()
    for step in steps:
        if step['type'] == 'real_lens':
            for surf in step['prescription']['surfaces']:
                for g in (surf['glass_before'], surf['glass_after']):
                    if g.lower() != 'air':
                        glasses_used.add(g)

    # ------------------------------------------------------------------
    # Generate the script
    # ------------------------------------------------------------------
    if style == 'unrolled':
        script = _generate_unrolled(
            steps, wavelength, N, dx, source_sigma,
            aperture, sys_name, glasses_used,
            include_plotting, include_analysis, header_comment,
        )
    elif style == 'system':
        script = _generate_system_style(
            steps, wavelength, N, dx, source_sigma,
            aperture, sys_name, glasses_used,
            include_plotting, include_analysis, header_comment,
        )
    else:
        raise ValueError(f"Unknown style '{style}'. Use 'unrolled' or 'system'.")

    # ------------------------------------------------------------------
    # Optionally write to file
    # ------------------------------------------------------------------
    if output_path is not None:
        # Pin UTF-8 so non-latin glass / system names in the generated script
        # do not raise a cp1252 UnicodeEncodeError under the Windows locale
        # (and so output is byte-identical across platforms).
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)

    return script


# ============================================================================
# Convenience wrappers: file -> script in one call
# ============================================================================

def generate_script_from_zmx(filepath: str,
                             wavelength: Optional[float] = None,
                             **kwargs: Any) -> str:
    """
    Load a ``.zmx`` file and generate a simulation script in one step.

    All keyword arguments are forwarded to :func:`generate_simulation_script`.

    Parameters
    ----------
    filepath : str
        Path to the ``.zmx`` file.
    wavelength : float or None
        Operating wavelength [m].

    Returns
    -------
    script : str
    """
    rx = load_zemax_zmx(filepath)
    return generate_simulation_script(rx, wavelength=wavelength, **kwargs)


def generate_script_from_txt(filepath: str,
                             wavelength: Optional[float] = None,
                             **kwargs: Any) -> str:
    """
    Load a Zemax prescription text export and generate a simulation script.

    Parameters
    ----------
    filepath : str
        Path to the prescription text file.
    wavelength : float or None
        Operating wavelength [m].

    Returns
    -------
    script : str
    """
    rx = load_zemax_prescription_data_txt(filepath)
    return generate_simulation_script(rx, wavelength=wavelength, **kwargs)


# ============================================================================
# Internal: prescription decomposition
# ============================================================================

def _decompose_prescription(prescription):
    """
    Break a parsed Zemax prescription into an ordered list of simulation steps.

    Groups contiguous refracting surfaces into lens groups (each becomes one
    ``apply_real_lens`` call), identifies free-space gaps, mirrors, and stops.

    Returns
    -------
    steps : list of dict
        Each dict has ``'type'`` (str) plus type-specific keys:

        - ``'type': 'propagate'`` → ``'z'``: propagation distance [m]
        - ``'type': 'real_lens'`` → ``'prescription'``: lens prescription dict
        - ``'type': 'mirror'``    → ``'radius'``, ``'conic'``, etc.
        - ``'type': 'aperture'``  → ``'diameter'``: aperture diameter [m]

    Stop emission (S2a, v4.13.0)
    ----------------------------

    Any element flagged ``is_stop=True`` in the input ``elements`` list
    causes a ``{'type': 'aperture', 'params': {'diameter': D}}`` step
    to be inserted *immediately before* the element it belongs to.  The
    stop diameter ``D`` comes from the element's ``semi_diameter * 2``
    if available, otherwise from the prescription-level
    ``aperture_diameter`` field.  Pre-v4.13.0 the stop flag was dropped
    silently and Zemax designs with a STOP surface produced scripts
    whose generated simulation never clipped the beam at the design's
    real aperture.
    """
    elements = prescription['elements']
    all_thicknesses = prescription['all_thicknesses']
    aperture = prescription.get('aperture_diameter', 25.4e-3)

    # Loader compatibility: if the prescription was produced before
    # v4.13.0 (loader didn't propagate ``is_stop`` into elements but
    # did populate a top-level ``stop_index`` -- e.g. .qos / .seq
    # loaders), translate the stop_index into a synthetic is_stop on
    # the matching refracting element.  ``stop_index`` is documented as
    # "zero-based index of the aperture stop **among refracting
    # surfaces**".
    stop_index = prescription.get('stop_index')

    def _is_stop_elem(idx, elem):
        if elem.get('is_stop'):
            return True
        if stop_index is None:
            return False
        # Count refracting surfaces up to (and including) idx.
        if elem.get('element_type') != 'surface':
            return False
        refr_count = sum(
            1 for k, e in enumerate(elements)
            if k <= idx and e.get('element_type') == 'surface'
        )
        return refr_count - 1 == stop_index

    def _stop_diameter(elem):
        sd = elem.get('semi_diameter', 0) or 0
        if sd > 0:
            return float(sd) * 2.0
        return float(aperture)

    steps = []

    # Identify contiguous groups of refracting surfaces
    # (surfaces separated by glass-to-glass transitions with no air gap)
    i = 0
    n_elem = len(elements)

    while i < n_elem:
        elem = elements[i]

        if elem['element_type'] == 'mirror':
            # S2a: emit aperture step before the mirror if this element
            # is the stop surface.
            if _is_stop_elem(i, elem):
                steps.append({
                    'type': 'aperture',
                    'diameter': _stop_diameter(elem),
                    'surf_num': elem.get('surf_num', -1),
                    'comment': 'Aperture stop (from STOP marker)',
                })
            # CG-1: the generated ``apply_mirror`` call carries only
            # radius/conic/aperture -- a mirror aspheric (or Q-type) surface
            # cannot be represented and would silently flatten to base conic.
            # Warn rather than drop it silently (matching the .zmx exporter's
            # P2-20 aspherized-mirror warning).
            if elem.get('aspheric_coeffs') or any(
                    elem.get(_fk) is not None for _fk in _CODEGEN_FREEFORM_KEYS):
                warnings.warn(
                    f"codegen: mirror surface "
                    f"{elem.get('surf_num', '?')} carries aspheric/freeform "
                    f"terms that ``apply_mirror`` cannot represent; the "
                    f"generated script flattens it to a conic mirror "
                    f"(radius/conic only).",
                    UserWarning, stacklevel=2)
            steps.append({
                'type': 'mirror',
                'radius': elem['radius'],
                'conic': elem.get('conic', 0.0),
                'aperture_diameter': elem.get('semi_diameter', 0) * 2,
                'comment': elem.get('comment', ''),
                'surf_num': elem.get('surf_num', -1),
            })
            # Propagate the thickness after the mirror (use abs for folded paths)
            if i < len(all_thicknesses):
                t = all_thicknesses[i]
                t_abs = abs(t)
                if t_abs > 0 and not np.isinf(t):
                    steps.append({'type': 'propagate', 'z': t_abs})
            i += 1
            continue

        if elem['element_type'] == 'surface':
            # Check if this is an air-to-air surface (DOE, dummy, reference plane)
            gb = elem.get('glass_before', 'air').lower()
            ga = elem.get('glass_after', 'air').lower()
            is_air_to_air = (gb == 'air' and ga == 'air')

            if is_air_to_air:
                # Air-to-air surface: no refraction (Δn = 0).
                # Could be a DOE, a dummy plane, or a reference surface.
                # Emit as a comment/placeholder, not a real_lens call.
                comment = elem.get('comment', '')
                asph = elem.get('aspheric_coeffs')
                surf_num = elem.get('surf_num', -1)

                # S2a: stop on an air-to-air surface (the common case
                # for a thin dummy STOP plane).  Emit aperture step.
                if _is_stop_elem(i, elem):
                    steps.append({
                        'type': 'aperture',
                        'diameter': _stop_diameter(elem),
                        'surf_num': surf_num,
                        'comment': 'Aperture stop (from STOP marker)',
                    })

                if asph:
                    # Has aspheric/diffractive phase — emit as a DOE placeholder
                    steps.append({
                        'type': 'doe_placeholder',
                        'surf_num': surf_num,
                        'aspheric_coeffs': asph,
                        'comment': comment or f'Diffractive/DOE surface {surf_num}',
                    })
                else:
                    # Pure dummy surface — skip it, just handle the thickness
                    pass

                # Propagate the thickness after this surface
                if i < len(all_thicknesses):
                    t = all_thicknesses[i]
                    t_abs = abs(t)
                    if t_abs > 0 and not np.isinf(t):
                        steps.append({'type': 'propagate', 'z': t_abs})

                i += 1
                continue

            # S2a: emit aperture step before the lens group if the
            # group's first surface is the STOP.  Stops on later
            # surfaces of a group are unusual in Zemax practice (the
            # stop is generally placed on a dedicated dummy or on the
            # group's front face); for robustness we still scan all
            # surfaces of the upcoming group and emit an aperture step
            # immediately before this lens-group step if any matches.
            # (One aperture step per group at most; the diameter is
            # taken from the stop-flagged surface.)
            group_stop_elem = None
            j_scan = i
            while j_scan < n_elem:
                cand = elements[j_scan]
                if cand.get('element_type') != 'surface':
                    break
                if _is_stop_elem(j_scan, cand):
                    group_stop_elem = cand
                    break
                if cand.get('glass_after', 'air').lower() == 'air':
                    break
                j_scan += 1
            if group_stop_elem is not None:
                steps.append({
                    'type': 'aperture',
                    'diameter': _stop_diameter(group_stop_elem),
                    'surf_num': group_stop_elem.get('surf_num', -1),
                    'comment': 'Aperture stop (from STOP marker)',
                })

            # Start of a real lens group: collect contiguous refracting surfaces
            group_start = i
            group_surfaces = [elem]

            # Walk forward: include next surface if the current surface's
            # glass_after is not air (meaning we're still inside glass)
            j = i
            while j < n_elem - 1:
                current = elements[j]
                if current['element_type'] != 'surface':
                    break
                # If glass_after is not air, the next surface is part
                # of the same lens group
                if current.get('glass_after', 'air').lower() != 'air':
                    j += 1
                    if j < n_elem and elements[j]['element_type'] == 'surface':
                        group_surfaces.append(elements[j])
                    else:
                        break
                else:
                    break

            group_end = group_start + len(group_surfaces) - 1

            # Build the lens prescription for this group
            surfaces_list = []
            thicknesses_list = []
            max_semi_dia = 0.0

            for k, surf in enumerate(group_surfaces):
                _sd = {
                    'radius': surf['radius'],
                    'conic': surf.get('conic', 0.0),
                    'aspheric_coeffs': surf.get('aspheric_coeffs'),
                    'glass_before': surf.get('glass_before', 'air'),
                    'glass_after': surf.get('glass_after', 'air'),
                }
                # CG-1: forward the Q-type freeform keys (was dropped -> Q-type
                # surface silently degraded to base conic in the script).
                for _fk in _CODEGEN_FREEFORM_KEYS:
                    if surf.get(_fk) is not None:
                        _sd[_fk] = surf[_fk]
                surfaces_list.append(_sd)
                sd = surf.get('semi_diameter', 0)
                if sd > max_semi_dia:
                    max_semi_dia = sd

                # Thickness between this surface and the next in the group
                # Use abs() for folded (reflected) paths where Zemax uses
                # negative thicknesses after mirrors.
                if k < len(group_surfaces) - 1:
                    elem_idx = group_start + k
                    if elem_idx < len(all_thicknesses):
                        thicknesses_list.append(abs(all_thicknesses[elem_idx]))
                    else:
                        thicknesses_list.append(0.0)

            lens_rx = {
                'name': _lens_group_name(group_surfaces),
                'aperture_diameter': max_semi_dia * 2 if max_semi_dia > 0 else aperture,
                'surfaces': surfaces_list,
                'thicknesses': thicknesses_list,
            }

            steps.append({
                'type': 'real_lens',
                'prescription': lens_rx,
                'surf_nums': [s.get('surf_num', -1) for s in group_surfaces],
                'comment': group_surfaces[0].get('comment', ''),
            })

            # Free-space propagation after the lens group (abs for folded paths)
            if group_end < len(all_thicknesses):
                t = all_thicknesses[group_end]
                t_abs = abs(t)
                if t_abs > 0 and not np.isinf(t):
                    steps.append({'type': 'propagate', 'z': t_abs})

            i = group_end + 1
            continue

        # Fallback: skip unknown element types
        i += 1

    # ------------------------------------------------------------------
    # Merge consecutive propagation steps into single steps
    # ------------------------------------------------------------------
    merged = []
    for step in steps:
        if (step['type'] == 'propagate'
                and merged
                and merged[-1]['type'] == 'propagate'):
            merged[-1]['z'] += step['z']
        else:
            merged.append(step)

    return merged


def _lens_group_name(surfaces):
    """Generate a descriptive name for a lens group."""
    glasses = []
    for s in surfaces:
        g = s.get('glass_after', 'air')
        if g.lower() != 'air' and g not in glasses:
            glasses.append(g)
    if not glasses:
        return 'Lens group'
    nums = [str(s.get('surf_num', '?')) for s in surfaces]
    return f"Lens ({'/'.join(glasses)}) [surfaces {nums[0]}-{nums[-1]}]"


# ============================================================================
# Internal: code generation — unrolled style
# ============================================================================

def _generate_unrolled(steps, wavelength, N, dx, source_sigma,
                       aperture, sys_name, glasses_used,
                       include_plotting, include_analysis, header_comment):
    """Generate an unrolled simulation script."""
    lines = []

    # --- Header ---
    # v4.15 (P1-CG): stamp the generating library version into the
    # header comment AND assert at runtime that the executing
    # ``lumenairy`` is recent enough to honour the script.  Pre-4.15
    # generated scripts carried no version pin, so a 4.14.2-produced
    # file run against a 4.13.x install silently mis-dispatched on
    # any signature that had drifted (e.g. the v4.14.0 dispatcher
    # adds for ``apply_lens`` / ``apply_real_lens_traced``) -- the
    # failure mode was a confusing TypeError or, worse, a silently
    # wrong result.  We import ``lumenairy`` and refuse to proceed
    # when the major.minor.patch tuple is below the generating
    # version.
    from .. import __version__ as _LA_VERSION
    if header_comment:
        lines.append(f'"""\n{header_comment}\n\n'
                      f'Generated by lumenairy {_LA_VERSION}\n"""')
    else:
        lines.append('"""')
        lines.append(f'ASM Simulation — {sys_name}')
        lines.append(f'{"=" * 50}')
        lines.append('')
        lines.append('Auto-generated from Zemax prescription by')
        lines.append('lumenairy.codegen.generate_simulation_script()')
        lines.append('')
        lines.append(f'lumenairy_version: {_LA_VERSION}')
        lines.append(f'Wavelength: {wavelength * 1e9:.1f} nm')
        lines.append(f'Grid: {N} x {N}, dx = {dx * 1e6:.2f} um')
        lines.append('"""')

    lines.append('')
    lines.append('import numpy as np')
    lines.append('import time')
    lines.append('import lumenairy as la')
    lines.append('import lumenairy')
    lines.append('import warnings as _la_codegen_warnings')
    lines.append('')
    # Runtime version pin (P1-CG): refuse to run on an older
    # lumenairy than the one that generated this script.  Doubled
    # f-string braces (``{{`` / ``}}``) escape the f-string itself
    # so the EMITTED source carries a normal Python f-string.
    lines.append('# v4.15 (P1-CG): runtime version pin.')
    _maj_min_pat = tuple(int(x) for x in _LA_VERSION.split('.')[:3])
    lines.append(
        f'if tuple(map(int, lumenairy.__version__.split(".")[:3])) '
        f'< {_maj_min_pat}:')
    lines.append(
        f'    raise RuntimeError(\n'
        f'        f"This script was generated by lumenairy '
        f'{_LA_VERSION}; current version is "\n'
        f'        f"{{lumenairy.__version__}}.  Upgrade lumenairy.")')
    # v4.15.1 (P3-2 / Agent E): upper-bound major-version-bump warning.
    # Pre-v4.15.1 the runtime pin was lower-bound only -- a user on
    # lumenairy 5.x running a v4.15-generated script got no warning
    # that the script was written against the v4.x API (which may have
    # breaking changes by v5.0).  Emit UserWarning (NOT RuntimeError)
    # so the script still runs but the user knows to re-generate.
    _gen_major = int(_LA_VERSION.split('.')[0])
    _next_major = (_gen_major + 1, 0, 0)
    lines.append(
        f'if tuple(map(int, lumenairy.__version__.split(".")[:3])) '
        f'>= {_next_major}:')
    lines.append(
        f'    _la_codegen_warnings.warn(\n'
        f'        f"This script was generated against lumenairy '
        f'{_LA_VERSION}; "\n'
        f'        f"current version is {{lumenairy.__version__}}.  Major "\n'
        f'        f"version {_next_major[0]}+ may have breaking API '
        f'changes; "\n'
        f'        f"consider regenerating the script via "\n'
        f'        f"lumenairy.io.codegen.generate_simulation_script.",\n'
        f'        UserWarning, stacklevel=2)')
    lines.append('')

    # --- Parameters ---
    lines.append('# ' + '=' * 70)
    lines.append('# PARAMETERS')
    lines.append('# ' + '=' * 70)
    lines.append(f'WAVELENGTH = {wavelength:.17e}  # [m]')
    lines.append(f'N = {N}')
    lines.append(f'dx = {dx:.17e}  # [m]')
    # v5.25 (audit S3-16): emit the canonical ``w0`` waist (1/e^2 intensity
    # radius = source_sigma * sqrt(2)) so generated scripts use the
    # non-deprecated create_gaussian_beam signature.
    lines.append(f'SOURCE_W0 = {source_sigma * 2 ** 0.5:.17e}  '
                 f'# beam waist (1/e^2 intensity radius) [m]')
    lines.append('')

    # --- Glass registry additions ---
    if glasses_used:
        unknown = [g for g in sorted(glasses_used) if g not in GLASS_REGISTRY]
        if unknown:
            lines.append('# ' + '-' * 70)
            lines.append('# GLASS REGISTRY — add missing glasses here')
            lines.append('# Browse refractiveindex.info to find the correct')
            lines.append('# (shelf, book, page) tuple for each glass.')
            lines.append('# ' + '-' * 70)
            for g in unknown:
                lines.append(
                    f"la.GLASS_REGISTRY['{g}'] = "
                    f"('specs', 'CATALOG', 'PAGE')  # TODO: fill in correct path"
                )
            lines.append('')

    # --- Lens prescriptions ---
    lines.append('# ' + '=' * 70)
    lines.append('# LENS PRESCRIPTIONS')
    lines.append('# ' + '=' * 70)

    lens_var_names = {}
    lens_counter = 0
    for step in steps:
        if step['type'] == 'real_lens':
            lens_counter += 1
            var_name = f'LENS_{lens_counter}_RX'
            lens_var_names[id(step)] = var_name
            rx = step['prescription']
            lines.append('')
            comment = step.get('comment', '')
            if comment:
                lines.append(f'# {comment}')
            lines.append(f'{var_name} = {{')
            lines.append(f'    "name": {rx["name"]!r},')
            lines.append(f'    "aperture_diameter": {rx["aperture_diameter"]:.17e},')
            lines.append('    "surfaces": [')
            for surf in rx['surfaces']:
                r_str = "float('inf')" if np.isinf(surf['radius']) else f'{surf["radius"]:.17e}'
                asph = surf.get('aspheric_coeffs')
                asph_str = repr(asph) if asph else 'None'
                lines.append(f'        {{"radius": {r_str}, '
                             f'"conic": {surf["conic"]},')
                lines.append(f'         "aspheric_coeffs": {asph_str},')
                # CG-1: emit the forwarded Q-type freeform keys so the
                # generated apply_real_lens reproduces the Forbes surface
                # (surface_sag_freeform consumes them) instead of base conic.
                for _fk in _CODEGEN_FREEFORM_KEYS:
                    if surf.get(_fk) is not None:
                        lines.append(f'         "{_fk}": {surf[_fk]!r},')
                lines.append(f'         "glass_before": {surf["glass_before"]!r}, '
                             f'"glass_after": {surf["glass_after"]!r}}},')
            lines.append('    ],')
            thk_str = ', '.join(f'{t:.17e}' for t in rx['thicknesses'])
            lines.append(f'    "thicknesses": [{thk_str}],')
            lines.append('}')

    lines.append('')

    # --- Simulation ---
    lines.append('# ' + '=' * 70)
    lines.append('# SIMULATION')
    lines.append('# ' + '=' * 70)
    lines.append('')
    lines.append('def run_simulation(verbose=True):')
    lines.append('    """Run the full optical system simulation."""')
    lines.append('    t_start = time.time()')
    lines.append('    planes = []')
    lines.append('')

    # Source
    lines.append('    # --- Source ---')
    lines.append('    E, x, y = la.create_gaussian_beam(N, dx, WAVELENGTH, w0=SOURCE_W0)')
    lines.append("    planes.append({'field': E.copy(), 'dx': dx, 'z': 0.0, "
                 "'label': 'Source'})")
    if include_analysis:
        lines.append("    P0 = la.beam_power(E, dx)")
        lines.append("    if verbose: print(f'Source power: {P0:.4e}')")
    lines.append('')

    # Walk through steps
    z_total = 0.0
    step_num = 0
    for step in steps:
        step_num += 1

        if step['type'] == 'propagate':
            z = step['z']
            z_total += z
            lines.append(f'    # --- Step {step_num}: Free-space propagation '
                         f'({z * 1e3:.3f} mm) ---')
            lines.append(f'    if verbose: print("Propagating {z * 1e3:.3f} mm ...")')
            lines.append(f'    E = la.angular_spectrum_propagate('
                         f'E, {z:.17e}, WAVELENGTH, dx)')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': 'After {z * 1e3:.2f} mm prop'}})")
            lines.append('')

        elif step['type'] == 'real_lens':
            var_name = lens_var_names[id(step)]
            rx = step['prescription']
            label = rx['name']
            lines.append(f'    # --- Step {step_num}: {label} ---')
            lines.append(f'    if verbose: print("Applying {label} ...")')
            lines.append(f'    E = la.apply_real_lens(E, prescription={var_name}, '
                         f'wavelength=WAVELENGTH, dx=dx)')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': {label!r}}})")
            if include_analysis:
                lines.append("    if verbose:")
                lines.append("        P = la.beam_power(E, dx)")
                lines.append("        print(f'  Power after lens: {P:.4e}')")
            lines.append('')

        elif step['type'] == 'mirror':
            r = step['radius']
            # v5.4.6 (audit F-14): the prescription stores the mirror radius
            # in the raytrace (Zemax/Welford) convention, where a CONCAVE
            # focusing mirror has R < 0, but la.apply_mirror uses the
            # wave-side convention where R > 0 is concave/focusing (verified
            # empirically).  Negate so the generated script reproduces the
            # prescription's focusing sign.  Flat mirrors (R = inf) are
            # sign-agnostic and stay None.
            r_str = 'None' if np.isinf(r) else f'{-r:.17e}'
            conic = step.get('conic', 0.0)
            ap = step.get('aperture_diameter')
            ap_str = f'{ap:.17e}' if ap and ap > 0 else 'None'
            comment = step.get('comment', 'Mirror')
            lines.append(f'    # --- Step {step_num}: {comment} ---')
            lines.append('    if verbose: print("Applying mirror ...")')
            lines.append(f'    E = la.apply_mirror(E, WAVELENGTH, dx, '
                         f'radius={r_str}, conic={conic}, '
                         f'aperture_diameter={ap_str})')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': {comment!r}}})")
            lines.append('')

        elif step['type'] == 'aperture':
            d = step['diameter']
            lines.append(f'    # --- Step {step_num}: Aperture stop ---')
            lines.append(f'    E = la.apply_aperture(E, dx, shape="circular", '
                         f'params={{"diameter": {d:.17e}}})')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': 'Aperture stop'}})")
            lines.append('')

        elif step['type'] == 'doe_placeholder':
            comment = step.get('comment', 'DOE surface')
            surf_num = step.get('surf_num', '?')
            asph = step.get('aspheric_coeffs', {})
            lines.append(f'    # --- Step {step_num}: {comment} (surface {surf_num}) ---')
            lines.append('    # TODO: This is a diffractive/DOE surface from the Zemax model.')
            lines.append(f'    # The Zemax aspheric/diffractive coefficients are: {asph}')
            lines.append('    # Replace this with your DOE phase mask, e.g.:')
            lines.append('    #   doe_mask = create_periodic_phase_mask(N, dx, doe_phase, doe_pixel_size)')
            lines.append('    #   E = E * doe_mask')
            lines.append('    if verbose: print("  [DOE placeholder — not applied]")')
            lines.append('')

    # Final summary
    lines.append('    # --- Done ---')
    lines.append("    t_elapsed = time.time() - t_start")
    if include_analysis:
        lines.append("    P_final = la.beam_power(E, dx)")
        lines.append("    if verbose:")
        lines.append("        print(f'\\nSimulation complete in {t_elapsed:.1f}s')")
        lines.append("        print(f'Final power: {P_final:.4e}')")
        lines.append("        print(f'Throughput: {P_final/P0:.4f}')")
    else:
        lines.append("    if verbose:")
        lines.append("        print(f'\\nSimulation complete in {t_elapsed:.1f}s')")
    lines.append('')
    lines.append('    return E, planes')
    lines.append('')

    # --- Plotting ---
    if include_plotting:
        lines.append('')
        lines.append('# ' + '=' * 70)
        lines.append('# PLOTTING')
        lines.append('# ' + '=' * 70)
        lines.append('')
        lines.append('def plot_results(planes):')
        lines.append('    """Plot all intermediate planes."""')
        lines.append('    try:')
        lines.append('        fig, axes = la.plot_planes_grid(')
        lines.append(f'            planes, suptitle={sys_name!r})')
        lines.append('        return fig')
        lines.append('    except ImportError:')
        lines.append('        print("matplotlib not installed — skipping plots")')
        lines.append('        return None')
        lines.append('')

    # --- Main ---
    lines.append('')
    lines.append("if __name__ == '__main__':")
    lines.append(f"    print('Running: {sys_name}')")
    lines.append('    print()')
    lines.append('    E_out, planes = run_simulation(verbose=True)')
    if include_plotting:
        lines.append('    fig = plot_results(planes)')
        lines.append('    try:')
        lines.append('        import matplotlib.pyplot as plt')
        lines.append('        plt.show()')
        lines.append('    except ImportError:')
        lines.append('        pass')
    lines.append('')

    return '\n'.join(lines)


# ============================================================================
# Internal: code generation — system list style
# ============================================================================

def _generate_system_style(steps, wavelength, N, dx, source_sigma,
                           aperture, sys_name, glasses_used,
                           include_plotting, include_analysis, header_comment):
    """Generate a compact script using propagate_through_system()."""
    lines = []

    # v4.15 (P1-CG): same version-pin rationale as _generate_unrolled.
    from .. import __version__ as _LA_VERSION

    # Header
    if header_comment:
        lines.append(f'"""\n{header_comment}\n\n'
                      f'Generated by lumenairy {_LA_VERSION}\n"""')
    else:
        lines.append('"""')
        lines.append(f'ASM Simulation — {sys_name} (system-list style)')
        lines.append('Auto-generated by lumenairy.codegen')
        lines.append(f'lumenairy_version: {_LA_VERSION}')
        lines.append('"""')

    lines.append('')
    lines.append('import numpy as np')
    lines.append('import lumenairy as la')
    lines.append('import lumenairy')
    lines.append('import warnings as _la_codegen_warnings')
    lines.append('')
    # Runtime version pin (P1-CG):
    lines.append('# v4.15 (P1-CG): runtime version pin.')
    _maj_min_pat = tuple(int(x) for x in _LA_VERSION.split('.')[:3])
    lines.append(
        f'if tuple(map(int, lumenairy.__version__.split(".")[:3])) '
        f'< {_maj_min_pat}:')
    lines.append(
        f'    raise RuntimeError(\n'
        f'        f"This script was generated by lumenairy '
        f'{_LA_VERSION}; current version is "\n'
        f'        f"{{lumenairy.__version__}}.  Upgrade lumenairy.")')
    # v4.15.1 (P3-2 / Agent E): upper-bound major-version-bump warning.
    # See _generate_unrolled for the rationale.
    _gen_major = int(_LA_VERSION.split('.')[0])
    _next_major = (_gen_major + 1, 0, 0)
    lines.append(
        f'if tuple(map(int, lumenairy.__version__.split(".")[:3])) '
        f'>= {_next_major}:')
    lines.append(
        f'    _la_codegen_warnings.warn(\n'
        f'        f"This script was generated against lumenairy '
        f'{_LA_VERSION}; "\n'
        f'        f"current version is {{lumenairy.__version__}}.  Major "\n'
        f'        f"version {_next_major[0]}+ may have breaking API '
        f'changes; "\n'
        f'        f"consider regenerating the script via "\n'
        f'        f"lumenairy.io.codegen.generate_simulation_script.",\n'
        f'        UserWarning, stacklevel=2)')
    lines.append('')

    # Parameters
    lines.append(f'WAVELENGTH = {wavelength:.17e}')
    lines.append(f'N = {N}')
    lines.append(f'dx = {dx:.17e}')
    lines.append('')

    # Glass registry
    unknown = [g for g in sorted(glasses_used) if g not in GLASS_REGISTRY]
    if unknown:
        for g in unknown:
            lines.append(
                f"la.GLASS_REGISTRY['{g}'] = "
                f"('specs', 'CATALOG', 'PAGE')  # TODO"
            )
        lines.append('')

    # Lens prescriptions (inline)
    lens_var_map = {}
    lens_counter = 0
    for step in steps:
        if step['type'] == 'real_lens':
            lens_counter += 1
            var_name = f'lens_{lens_counter}_rx'
            lens_var_map[id(step)] = var_name
            rx = step['prescription']
            lines.append(f'{var_name} = {{')
            lines.append(f'    "name": {rx["name"]!r},')
            lines.append(f'    "aperture_diameter": {rx["aperture_diameter"]:.17e},')
            lines.append('    "surfaces": [')
            for surf in rx['surfaces']:
                r_str = "float('inf')" if np.isinf(surf['radius']) else f'{surf["radius"]:.17e}'
                asph = surf.get('aspheric_coeffs')
                asph_str = repr(asph) if asph else 'None'
                lines.append(f'        {{"radius": {r_str}, "conic": {surf["conic"]}, '
                             f'"aspheric_coeffs": {asph_str}, '
                             f'"glass_before": {surf["glass_before"]!r}, '
                             f'"glass_after": {surf["glass_after"]!r}}},')
            lines.append('    ],')
            thk_str = ', '.join(f'{t:.17e}' for t in rx['thicknesses'])
            lines.append(f'    "thicknesses": [{thk_str}],')
            lines.append('}')
            lines.append('')

    # Element list
    lines.append('elements = [')
    for step in steps:
        if step['type'] == 'propagate':
            lines.append(f"    {{'type': 'propagate', 'z': {step['z']:.17e}}},")
        elif step['type'] == 'real_lens':
            var_name = lens_var_map[id(step)]
            lines.append(f"    {{'type': 'real_lens', 'prescription': {var_name}}},")
        elif step['type'] == 'mirror':
            r = step['radius']
            # v5.4.7 (audit AUDIT_V5_4_6 gap #1): negate the mirror radius
            # here too.  The v5.4.6 F-14 fix only patched the unrolled style
            # (above); this system style emitted the raytrace-convention
            # radius un-negated into the step dict, which
            # propagate_through_system forwards straight to apply_mirror
            # (wave-side convention) -> a curved fold mirror got the
            # OPPOSITE focusing sign (a concave R=-0.5 prescription emitted
            # radius=-0.5, diverging instead of focusing -- a >5000x
            # focus-vs-defocus inversion).  Refractive surf['radius'] sites
            # (feeding apply_real_lens) must NOT be negated; only mirrors.
            r_str = 'None' if np.isinf(r) else f'{-r:.17e}'
            ap = step.get('aperture_diameter')
            ap_str = (f'{ap:.17e}'
                      if (ap is not None and float(ap) > 0)
                      else 'None')
            lines.append(f"    {{'type': 'mirror', 'radius': {r_str}, "
                         f"'conic': {step.get('conic', 0.0)}, "
                         f"'aperture_diameter': {ap_str}}},")
        elif step['type'] == 'aperture':
            d = step['diameter']
            lines.append(f"    {{'type': 'aperture', 'shape': 'circular', "
                         f"'params': {{'diameter': {d:.17e}}}}},")
        elif step['type'] == 'doe_placeholder':
            comment = step.get('comment', 'DOE')
            lines.append(f"    # TODO: DOE placeholder — {comment}")
            lines.append("    # Add your DOE mask element here")
    lines.append(']')
    lines.append('')

    # Source and run.  v5.25 (audit S3-16): emit the canonical ``w0`` waist
    # (1/e^2 intensity radius = source_sigma * sqrt(2)).
    lines.append('E, x, y = la.create_gaussian_beam(N, dx, WAVELENGTH, w0='
                 f'{source_sigma * 2 ** 0.5:.17e})')
    lines.append('E_out, intermediates = la.propagate_through_system(')
    lines.append('    E, elements, WAVELENGTH, dx, verbose=True)')
    lines.append('')

    if include_analysis:
        lines.append("P_in = la.beam_power(E, dx)")
        lines.append("P_out = la.beam_power(E_out, dx)")
        lines.append("print(f'Throughput: {P_out/P_in:.4f}')")

    return '\n'.join(lines)


# A-13ish (AUDIT_ADVERSARIAL_CODEBASE 2026-07-25): declare the public
# surface explicitly, matching the convention every ``analysis/`` module
# already follows.  Names re-exported through ``lumenairy.io``
#  and the top-level facade.
__all__ = [
    'generate_simulation_script',
    'generate_script_from_zmx',
    'generate_script_from_txt',
]

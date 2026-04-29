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

    import lumenairy as op
    from lumenairy.codegen import generate_simulation_script

    rx = op.load_zmx_prescription('my_design.zmx')
    code = generate_simulation_script(rx, wavelength=1.31e-6)

    with open('sim_my_design.py', 'w') as f:
        f.write(code)

From a prescription text export::

    rx = op.load_zemax_prescription_txt('design-prescription.txt')
    code = generate_simulation_script(rx)

Author: Andrew Traverso
"""

import numpy as np

from .prescriptions import load_zmx_prescription, load_zemax_prescription_txt
from .glass import GLASS_REGISTRY


# ============================================================================
# Public API
# ============================================================================

def generate_simulation_script(
    prescription,
    wavelength=None,
    N=2048,
    dx=None,
    source_sigma=None,
    output_path=None,
    style='unrolled',
    include_plotting=True,
    include_analysis=True,
    header_comment=None,
):
    """
    Generate a Python simulation script from a parsed Zemax prescription.

    Takes the output of :func:`load_zmx_prescription` or
    :func:`load_zemax_prescription_txt` and produces a complete, runnable
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
        Parsed Zemax prescription from :func:`load_zmx_prescription` or
        :func:`load_zemax_prescription_txt`.  Must contain keys:
        ``'elements'``, ``'all_thicknesses'``, ``'aperture_diameter'``,
        and ``'name'``.

    wavelength : float or None
        Operating wavelength [m].  If None, uses the wavelength stored
        in the prescription dict (if present), otherwise defaults to
        1.31e-6 m (1310 nm).

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
    >>> import lumenairy as op
    >>> from lumenairy.codegen import generate_simulation_script
    >>>
    >>> rx = op.load_zmx_prescription('AC254-200-C.zmx')
    >>> code = generate_simulation_script(rx, wavelength=1.31e-6,
    ...                                   output_path='sim_ac254.py')
    """
    # ------------------------------------------------------------------
    # Resolve parameters
    # ------------------------------------------------------------------
    if wavelength is None:
        wavelength = prescription.get('wavelength', 1.31e-6)
    if wavelength is None:
        wavelength = 1.31e-6

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
        with open(output_path, 'w') as f:
            f.write(script)

    return script


# ============================================================================
# Convenience wrappers: file -> script in one call
# ============================================================================

def generate_script_from_zmx(filepath, wavelength=None, **kwargs):
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
    rx = load_zmx_prescription(filepath)
    return generate_simulation_script(rx, wavelength=wavelength, **kwargs)


def generate_script_from_txt(filepath, wavelength=None, **kwargs):
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
    rx = load_zemax_prescription_txt(filepath)
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
    """
    elements = prescription['elements']
    all_thicknesses = prescription['all_thicknesses']
    aperture = prescription.get('aperture_diameter', 25.4e-3)

    steps = []

    # Identify contiguous groups of refracting surfaces
    # (surfaces separated by glass-to-glass transitions with no air gap)
    i = 0
    n_elem = len(elements)

    while i < n_elem:
        elem = elements[i]

        if elem['element_type'] == 'mirror':
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
                surfaces_list.append({
                    'radius': surf['radius'],
                    'conic': surf.get('conic', 0.0),
                    'aspheric_coeffs': surf.get('aspheric_coeffs'),
                    'glass_before': surf.get('glass_before', 'air'),
                    'glass_after': surf.get('glass_after', 'air'),
                })
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
    if header_comment:
        lines.append(f'"""\n{header_comment}\n"""')
    else:
        lines.append(f'"""')
        lines.append(f'ASM Simulation — {sys_name}')
        lines.append(f'{"=" * 50}')
        lines.append(f'')
        lines.append(f'Auto-generated from Zemax prescription by')
        lines.append(f'lumenairy.codegen.generate_simulation_script()')
        lines.append(f'')
        lines.append(f'Wavelength: {wavelength * 1e9:.1f} nm')
        lines.append(f'Grid: {N} x {N}, dx = {dx * 1e6:.2f} um')
        lines.append(f'"""')

    lines.append('')
    lines.append('import numpy as np')
    lines.append('import time')
    lines.append('import lumenairy as op')
    lines.append('')

    # --- Parameters ---
    lines.append('# ' + '=' * 70)
    lines.append('# PARAMETERS')
    lines.append('# ' + '=' * 70)
    lines.append(f'WAVELENGTH = {wavelength:.17e}  # [m]')
    lines.append(f'N = {N}')
    lines.append(f'dx = {dx:.17e}  # [m]')
    lines.append(f'SOURCE_SIGMA = {source_sigma:.17e}  # 1/e field radius [m]')
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
                    f"op.GLASS_REGISTRY['{g}'] = "
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
            lines.append(f'')
            comment = step.get('comment', '')
            if comment:
                lines.append(f'# {comment}')
            lines.append(f'{var_name} = {{')
            lines.append(f'    "name": {rx["name"]!r},')
            lines.append(f'    "aperture_diameter": {rx["aperture_diameter"]:.17e},')
            lines.append(f'    "surfaces": [')
            for surf in rx['surfaces']:
                r_str = "float('inf')" if np.isinf(surf['radius']) else f'{surf["radius"]:.17e}'
                asph = surf.get('aspheric_coeffs')
                asph_str = repr(asph) if asph else 'None'
                lines.append(f'        {{"radius": {r_str}, '
                             f'"conic": {surf["conic"]},')
                lines.append(f'         "aspheric_coeffs": {asph_str},')
                lines.append(f'         "glass_before": {surf["glass_before"]!r}, '
                             f'"glass_after": {surf["glass_after"]!r}}},')
            lines.append(f'    ],')
            thk_str = ', '.join(f'{t:.17e}' for t in rx['thicknesses'])
            lines.append(f'    "thicknesses": [{thk_str}],')
            lines.append(f'}}')

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
    lines.append('    E, x, y = op.create_gaussian_beam(N, dx, SOURCE_SIGMA)')
    lines.append("    planes.append({'field': E.copy(), 'dx': dx, 'z': 0.0, "
                 "'label': 'Source'})")
    if include_analysis:
        lines.append("    P0 = op.beam_power(E, dx)")
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
            lines.append(f'    E = op.angular_spectrum_propagate('
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
            lines.append(f'    E = op.apply_real_lens(E, {var_name}, '
                         f'WAVELENGTH, dx)')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': {label!r}}})")
            if include_analysis:
                lines.append(f"    if verbose:")
                lines.append(f"        P = op.beam_power(E, dx)")
                lines.append(f"        print(f'  Power after lens: {{P:.4e}}')")
            lines.append('')

        elif step['type'] == 'mirror':
            r = step['radius']
            r_str = 'None' if np.isinf(r) else f'{r:.17e}'
            conic = step.get('conic', 0.0)
            ap = step.get('aperture_diameter')
            ap_str = f'{ap:.17e}' if ap and ap > 0 else 'None'
            comment = step.get('comment', 'Mirror')
            lines.append(f'    # --- Step {step_num}: {comment} ---')
            lines.append(f'    if verbose: print("Applying mirror ...")')
            lines.append(f'    E = op.apply_mirror(E, WAVELENGTH, dx, '
                         f'radius={r_str}, conic={conic}, '
                         f'aperture_diameter={ap_str})')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': {comment!r}}})")
            lines.append('')

        elif step['type'] == 'aperture':
            d = step['diameter']
            lines.append(f'    # --- Step {step_num}: Aperture stop ---')
            lines.append(f'    E = op.apply_aperture(E, dx, shape="circular", '
                         f'params={{"diameter": {d:.17e}}})')
            lines.append(f"    planes.append({{'field': E.copy(), 'dx': dx, "
                         f"'z': {z_total:.17e}, 'label': 'Aperture stop'}})")
            lines.append('')

        elif step['type'] == 'doe_placeholder':
            comment = step.get('comment', 'DOE surface')
            surf_num = step.get('surf_num', '?')
            asph = step.get('aspheric_coeffs', {})
            lines.append(f'    # --- Step {step_num}: {comment} (surface {surf_num}) ---')
            lines.append(f'    # TODO: This is a diffractive/DOE surface from the Zemax model.')
            lines.append(f'    # The Zemax aspheric/diffractive coefficients are: {asph}')
            lines.append(f'    # Replace this with your DOE phase mask, e.g.:')
            lines.append(f'    #   doe_mask = create_periodic_phase_mask(N, dx, doe_phase, doe_pixel_size)')
            lines.append(f'    #   E = E * doe_mask')
            lines.append(f'    if verbose: print("  [DOE placeholder — not applied]")')
            lines.append('')

    # Final summary
    lines.append('    # --- Done ---')
    lines.append("    t_elapsed = time.time() - t_start")
    if include_analysis:
        lines.append("    P_final = op.beam_power(E, dx)")
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
        lines.append('        fig, axes = op.plot_planes_grid(')
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

    # Header
    if header_comment:
        lines.append(f'"""\n{header_comment}\n"""')
    else:
        lines.append(f'"""')
        lines.append(f'ASM Simulation — {sys_name} (system-list style)')
        lines.append(f'Auto-generated by lumenairy.codegen')
        lines.append(f'"""')

    lines.append('')
    lines.append('import numpy as np')
    lines.append('import lumenairy as op')
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
                f"op.GLASS_REGISTRY['{g}'] = "
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
            lines.append(f'    "surfaces": [')
            for surf in rx['surfaces']:
                r_str = "float('inf')" if np.isinf(surf['radius']) else f'{surf["radius"]:.17e}'
                asph = surf.get('aspheric_coeffs')
                asph_str = repr(asph) if asph else 'None'
                lines.append(f'        {{"radius": {r_str}, "conic": {surf["conic"]}, '
                             f'"aspheric_coeffs": {asph_str}, '
                             f'"glass_before": {surf["glass_before"]!r}, '
                             f'"glass_after": {surf["glass_after"]!r}}},')
            lines.append(f'    ],')
            thk_str = ', '.join(f'{t:.17e}' for t in rx['thicknesses'])
            lines.append(f'    "thicknesses": [{thk_str}],')
            lines.append(f'}}')
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
            r_str = 'None' if np.isinf(r) else f'{step["radius"]:.17e}'
            lines.append(f"    {{'type': 'mirror', 'radius': {r_str}, "
                         f"'conic': {step.get('conic', 0.0)}}},")
        elif step['type'] == 'aperture':
            d = step['diameter']
            lines.append(f"    {{'type': 'aperture', 'shape': 'circular', "
                         f"'params': {{'diameter': {d:.17e}}}}},")
        elif step['type'] == 'doe_placeholder':
            comment = step.get('comment', 'DOE')
            lines.append(f"    # TODO: DOE placeholder — {comment}")
            lines.append(f"    # Add your DOE mask element here")
    lines.append(']')
    lines.append('')

    # Source and run
    lines.append('E, x, y = op.create_gaussian_beam(N, dx, '
                 f'{source_sigma:.17e})')
    lines.append('E_out, intermediates = op.propagate_through_system(')
    lines.append('    E, elements, WAVELENGTH, dx, verbose=True)')
    lines.append('')

    if include_analysis:
        lines.append("P_in = op.beam_power(E, dx)")
        lines.append("P_out = op.beam_power(E_out, dx)")
        lines.append("print(f'Throughput: {P_out/P_in:.4f}')")

    return '\n'.join(lines)

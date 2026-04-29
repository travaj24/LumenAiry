"""
User library — persistent storage for custom materials, lenses, and phase masks.

Provides save/load/list/delete functions for three categories of user-defined
optical components:

* **Materials** — custom glasses with fixed or dispersive refractive indices.
  When loaded, they register in :data:`glass.GLASS_REGISTRY` so
  ``get_glass_index('MyGlass', wavelength)`` works everywhere.

* **Lenses** — prescription dicts (from ``make_singlet``, ``make_doublet``,
  ``thorlabs_lens``, or hand-built).  Loaded prescriptions work directly
  with ``apply_real_lens``.

* **Phase masks** — mathematical expressions (evaluated on a grid),
  pre-computed 2-D arrays, or glass-block definitions.  Loaded masks
  are complex transmission arrays ready for ``apply_mask(E, mask)``.

Storage is JSON files in ``~/.lumenairy/library/`` with optional
``.npy`` sidecar files for large arrays.

Usage from Python (no GUI needed)::

    from lumenairy.user_library import (
        save_material, load_material, list_materials,
        save_lens, load_lens, list_lenses,
        save_phase_mask, load_phase_mask, list_phase_masks,
    )

Author: Andrew Traverso
"""

import json
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Library path
# ════════════════════════════════════════════════════════════════════════

_library_path = None


def get_library_path():
    """Return the user library directory, creating it if needed."""
    global _library_path
    if _library_path is not None:
        return Path(_library_path)

    home = Path.home()
    lib_dir = home / '.lumenairy' / 'library'
    lib_dir.mkdir(parents=True, exist_ok=True)
    (lib_dir / 'materials').mkdir(exist_ok=True)
    (lib_dir / 'lenses').mkdir(exist_ok=True)
    (lib_dir / 'phase_masks').mkdir(exist_ok=True)
    return lib_dir


def set_library_path(path):
    """Override the library directory."""
    global _library_path
    _library_path = str(path)
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / 'materials').mkdir(exist_ok=True)
    (p / 'lenses').mkdir(exist_ok=True)
    (p / 'phase_masks').mkdir(exist_ok=True)


def _safe_name(name):
    """Sanitise a name for use as a filename."""
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_')


# ════════════════════════════════════════════════════════════════════════
# Materials
# ════════════════════════════════════════════════════════════════════════

def save_material(name, shelf=None, book=None, page=None,
                  n=None, dispersion=None, description=''):
    """Save a material to the user library.

    Parameters
    ----------
    name : str
        Material name (used as the glass name in prescriptions).
    shelf, book, page : str or None
        refractiveindex.info coordinates.  If all three are given, the
        material is a catalog glass.
    n : float or None
        Fixed refractive index (constant, no dispersion).
    dispersion : dict or None
        Dispersion coefficients (Cauchy: {'A': ..., 'B': ...}, etc.).
    description : str
        Human-readable description.
    """
    lib = get_library_path() / 'materials'
    data = {'name': name, 'description': description}

    if shelf and book and page:
        data['type'] = 'catalog'
        data['shelf'] = shelf
        data['book'] = book
        data['page'] = page
    elif n is not None:
        data['type'] = 'fixed'
        data['n'] = float(n)
        if dispersion:
            data['dispersion'] = dispersion
    else:
        raise ValueError("Provide either (shelf, book, page) or n.")

    filepath = lib / f'{_safe_name(name)}.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_material(name):
    """Load a material and register it in GLASS_REGISTRY.

    After loading, ``get_glass_index(name, wavelength)`` will work.

    Parameters
    ----------
    name : str
        Material name.

    Returns
    -------
    data : dict
        The saved material data.
    """
    lib = get_library_path() / 'materials'
    filepath = lib / f'{_safe_name(name)}.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Material '{name}' not found in library.")

    with open(filepath) as f:
        data = json.load(f)

    mat_name = data['name']

    if data['type'] == 'catalog':
        from .glass import GLASS_REGISTRY
        GLASS_REGISTRY[mat_name] = (data['shelf'], data['book'], data['page'])

    elif data['type'] == 'fixed':
        register_fixed_glass(mat_name, data['n'])

    return data


def list_materials():
    """List all saved material names."""
    lib = get_library_path() / 'materials'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_material(name):
    """Delete a saved material."""
    lib = get_library_path() / 'materials'
    filepath = lib / f'{_safe_name(name)}.json'
    if filepath.exists():
        filepath.unlink()


def register_fixed_glass(name, n):
    """Register a fixed-index material so get_glass_index works with it.

    Parameters
    ----------
    name : str
        Glass name.
    n : float
        Refractive index (constant for all wavelengths).
    """
    from .glass import GLASS_REGISTRY, _glass_cache

    class _FixedIndex:
        def __init__(self, n_val):
            self._n = n_val
        def get_refractive_index(self, wv_nm, unit='nm'):
            return self._n

    GLASS_REGISTRY[name] = ('__user__', '__fixed__', '__fixed__')
    _glass_cache[name] = _FixedIndex(n)


# ════════════════════════════════════════════════════════════════════════
# Lenses
# ════════════════════════════════════════════════════════════════════════

def _serialize_prescription(rx):
    """Convert a prescription dict to JSON-safe form.

    Handles (recursively, anywhere in the nested dict/list tree):

    * ``float('inf')``  -> ``'Infinity'``
    * ``float('-inf')`` -> ``'-Infinity'``
    * ``np.integer``    -> ``int``
    * ``np.floating``   -> ``float``
    * ``np.ndarray``    -> ``list``

    The matching :func:`_deserialize_prescription` reverses the
    string sentinels back to ``float`` values.
    """
    def _conv(obj):
        if isinstance(obj, float) and np.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isinf(v):
                return 'Infinity' if v > 0 else '-Infinity'
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return json.loads(json.dumps(rx, default=_conv))


def _deserialize_prescription(data):
    """Convert JSON-loaded prescription back to proper types.

    Walks the full nested tree and replaces ``'Infinity'`` /
    ``'-Infinity'`` string sentinels with ``float('inf')`` /
    ``float('-inf')``.  Previously this only handled the
    ``surfaces[i]['radius']`` slot, so any other field containing
    infinity (thickness, conic constant, aperture) came back as a
    string and caused downstream ``TypeError`` surprises.
    """
    def _fix(obj):
        if isinstance(obj, str):
            if obj == 'Infinity':
                return float('inf')
            if obj == '-Infinity':
                return float('-inf')
            return obj
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(v) for v in obj]
        return obj
    return _fix(data)


def save_lens(name, prescription, description=''):
    """Save a lens prescription to the user library.

    Parameters
    ----------
    name : str
        Lens name.
    prescription : dict
        Prescription dict (from ``make_singlet``, ``thorlabs_lens``, etc.).
    description : str
        Human-readable description.
    """
    lib = get_library_path() / 'lenses'
    data = {
        'name': name,
        'description': description,
        'prescription': _serialize_prescription(prescription),
    }

    filepath = lib / f'{_safe_name(name)}.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_lens(name):
    """Load a lens prescription from the user library.

    Parameters
    ----------
    name : str

    Returns
    -------
    prescription : dict
        Ready to pass to ``apply_real_lens``.
    """
    lib = get_library_path() / 'lenses'
    filepath = lib / f'{_safe_name(name)}.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Lens '{name}' not found in library.")

    with open(filepath) as f:
        data = json.load(f)

    return _deserialize_prescription(data['prescription'])


def list_lenses():
    """List all saved lens names."""
    lib = get_library_path() / 'lenses'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_lens(name):
    """Delete a saved lens."""
    lib = get_library_path() / 'lenses'
    filepath = lib / f'{_safe_name(name)}.json'
    if filepath.exists():
        filepath.unlink()


# ════════════════════════════════════════════════════════════════════════
# Phase masks
# ════════════════════════════════════════════════════════════════════════

def save_phase_mask(name, expression=None, array=None, dx=None,
                    wavelength=None, mask_type=None, n=None,
                    thickness=None, description=''):
    """Save a phase mask / DOE / glass block to the user library.

    Three modes:

    1. **Expression** — a mathematical formula evaluated on (X, Y) grids.
       Example: ``expression='atan2(Y, X) * 3'`` for a spiral phase plate.
       Available variables: X, Y (metres), R (radius), THETA (angle),
       k (wavenumber), pi. All numpy functions available.

    2. **Array** — a pre-computed 2-D phase array (radians).  Saved as
       a ``.npy`` sidecar file alongside the JSON.

    3. **Glass block** — a flat slab with fixed index and thickness.
       Applies a uniform phase ``k * (n - 1) * thickness``.

    Parameters
    ----------
    name : str
    expression : str or None
    array : ndarray or None
    dx : float or None
        Grid spacing [m] (required for array mode).
    wavelength : float or None
        Wavelength [m] (stored as metadata).
    mask_type : str or None
        'expression', 'array', or 'glass_block' (auto-detected if None).
    n : float or None
        Refractive index (for glass_block mode).
    thickness : float or None
        Thickness [m] (for glass_block mode).
    description : str
    """
    lib = get_library_path() / 'phase_masks'

    data = {'name': name, 'description': description}

    if expression is not None:
        data['type'] = 'expression'
        data['expression'] = expression
    elif array is not None:
        data['type'] = 'array'
        data['dx'] = dx
        data['wavelength'] = wavelength
        data['shape'] = list(array.shape)
        # Save array as .npy sidecar
        npy_path = lib / f'{_safe_name(name)}.npy'
        np.save(str(npy_path), array)
    elif n is not None and thickness is not None:
        data['type'] = 'glass_block'
        data['n'] = float(n)
        data['thickness'] = float(thickness)
    else:
        raise ValueError(
            "Provide expression, array, or (n + thickness) for glass block.")

    if mask_type:
        data['type'] = mask_type
    if wavelength is not None:
        data['wavelength'] = wavelength
    if dx is not None:
        data['dx'] = dx

    filepath = lib / f'{_safe_name(name)}.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_phase_mask(name, N=None, dx=None, wavelength=None):
    """Load a phase mask and return a complex transmission array.

    Parameters
    ----------
    name : str
    N : int or None
        Grid size (required for expression and glass_block modes).
    dx : float or None
        Grid spacing [m] (required for expression mode).
    wavelength : float or None
        Wavelength [m] (required for glass_block mode).

    Returns
    -------
    mask : ndarray (complex, N x N)
        Complex transmission: ``exp(1j * phase)``.
    """
    lib = get_library_path() / 'phase_masks'
    filepath = lib / f'{_safe_name(name)}.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Phase mask '{name}' not found in library.")

    with open(filepath) as f:
        data = json.load(f)

    mask_type = data['type']

    if mask_type == 'expression':
        if N is None or dx is None:
            raise ValueError("N and dx required for expression masks.")
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X ** 2 + Y ** 2)
        THETA = np.arctan2(Y, X)
        k = 2 * np.pi / wavelength if wavelength else 1.0
        pi = np.pi

        # Evaluate with numpy namespace
        ns = {
            'X': X, 'Y': Y, 'R': R, 'THETA': THETA,
            'k': k, 'pi': pi, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'abs': np.abs,
            'exp': np.exp, 'log': np.log,
            'atan2': np.arctan2, 'arctan2': np.arctan2,
            'mod': np.mod, 'floor': np.floor, 'ceil': np.ceil,
        }
        phase = eval(data['expression'], {"__builtins__": {}}, ns)
        return np.exp(1j * phase)

    elif mask_type == 'array':
        npy_path = lib / f'{_safe_name(name)}.npy'
        phase = np.load(str(npy_path))
        return np.exp(1j * phase)

    elif mask_type == 'glass_block':
        if N is None:
            raise ValueError("N required for glass block masks.")
        n_glass = data['n']
        t = data['thickness']
        wv = wavelength or data.get('wavelength', 1.0)
        k = 2 * np.pi / wv
        phase = k * (n_glass - 1) * t
        return np.full((N, N), np.exp(1j * phase), dtype=complex)

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def load_phase_mask_info(name):
    """Load phase mask metadata without generating the array."""
    lib = get_library_path() / 'phase_masks'
    filepath = lib / f'{_safe_name(name)}.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Phase mask '{name}' not found.")
    with open(filepath) as f:
        return json.load(f)


def list_phase_masks():
    """List all saved phase mask names."""
    lib = get_library_path() / 'phase_masks'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_phase_mask(name):
    """Delete a saved phase mask (JSON + any .npy sidecar)."""
    lib = get_library_path() / 'phase_masks'
    for ext in ('.json', '.npy'):
        filepath = lib / f'{_safe_name(name)}{ext}'
        if filepath.exists():
            filepath.unlink()


# ════════════════════════════════════════════════════════════════════════
# Load all materials on import (auto-register saved glasses)
# ════════════════════════════════════════════════════════════════════════

def load_all_materials():
    """Load all saved materials into GLASS_REGISTRY.

    Called automatically on import so that saved materials are
    immediately available in any script.
    """
    for name in list_materials():
        try:
            load_material(name)
        except Exception:
            pass


# Auto-load on import
try:
    load_all_materials()
except Exception:
    pass

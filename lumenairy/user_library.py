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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ════════════════════════════════════════════════════════════════════════
# Library path
# ════════════════════════════════════════════════════════════════════════

_library_path = None


def get_library_path() -> Path:
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


def set_library_path(path: str) -> None:
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

def save_material(name: str,
                  shelf: Optional[str] = None,
                  book: Optional[str] = None,
                  page: Optional[str] = None,
                  n: Optional[float] = None,
                  dispersion: Optional[Dict[str, Any]] = None,
                  description: str = '') -> str:
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


def load_material(name: str) -> Dict[str, Any]:
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
        from .glass import GLASS_REGISTRY, _invalidate_glass_name
        # v5.4.6 (audit F-35): warn before overriding a built-in registry
        # entry of the same name (the prior code clobbered it silently,
        # and load_material can auto-run at import).  The override still
        # proceeds -- the user explicitly saved this material -- but is
        # now surfaced, matching register_fixed_glass's collision guard.
        if mat_name in GLASS_REGISTRY:
            import warnings
            warnings.warn(
                f"load_material: user material '{mat_name}' overrides the "
                f"built-in GLASS_REGISTRY entry of the same name.",
                UserWarning, stacklevel=2)
        GLASS_REGISTRY[mat_name] = (data['shelf'], data['book'], data['page'])
        # v5.17.1 (audit P2-41): re-pointing the registry must also drop
        # any stale cached resolution for this name.  Pre-v5.17.1 a name
        # that previously resolved through ``_glass_cache`` (a user-fixed
        # ``_FixedIndex`` or a ``RefractiveIndexMaterial`` for a
        # different catalogue page) kept serving the OLD index forever:
        # ``get_glass_index``'s tuple branch trusts ``_glass_cache``
        # unconditionally.  Mirrors ``register_fixed_glass``'s hygiene
        # (which overwrites the cache entry and clears the value cache)
        # with the surgical per-name pattern from
        # ``raytrace.trace._register_fixed_index``.
        _invalidate_glass_name(mat_name)

    elif data['type'] == 'fixed':
        # 4.11.2: ``save_material`` accepts a ``dispersion`` dict and
        # writes it to JSON, but ``register_fixed_glass`` only honours
        # the scalar ``n``.  Surface the silent drop so users do not
        # assume their dispersion data round-trips.  Round-3 audit
        # (AUDIT_ROUND3_2026_05_16.md, IO section).
        if data.get('dispersion'):
            import warnings as _w
            _w.warn(
                f"load_material({mat_name!r}): saved file includes a "
                f"'dispersion' field but register_fixed_glass only "
                f"honours the scalar ``n``; dispersion data is "
                f"dropped on load.  Either re-save with explicit "
                f"shelf/book/page (catalog dispatch via "
                f"refractiveindex.info) or evaluate the dispersion "
                f"externally and call register_fixed_glass per "
                f"wavelength.", RuntimeWarning, stacklevel=2)
        register_fixed_glass(mat_name, data['n'])

    return data


def list_materials() -> List[str]:
    """List all saved material names."""
    lib = get_library_path() / 'materials'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_material(name: str) -> None:
    """Delete a saved material."""
    lib = get_library_path() / 'materials'
    filepath = lib / f'{_safe_name(name)}.json'
    if filepath.exists():
        filepath.unlink()


def register_fixed_glass(name: str, n: float) -> None:
    """Register a fixed-index material so get_glass_index works with it.

    Parameters
    ----------
    name : str
        Glass name.  Must be a non-empty string after stripping
        whitespace; cannot be ``''`` or ``'   '``.
    n : float
        Refractive index (constant for all wavelengths).  Must be in
        ``[1.0, 4.0]``: 1.0 is vacuum / "air", and the upper bound
        comfortably covers high-index semiconductors used in mid-IR
        / THz design (Si ~3.4, Ge ~4.0).  Values outside this range
        are almost certainly a typo or unit mistake.

    Raises
    ------
    ValueError
        If ``name`` is not a string or strips to empty, or if ``n``
        is outside the physical ``[1.0, 4.0]`` range.

    Warns
    -----
    UserWarning
        If ``name`` is already in :data:`glass.GLASS_REGISTRY`; the
        existing entry is overwritten anyway (sometimes that is the
        caller's intent -- e.g. re-registering a tweaked test glass
        -- but the audit P1-GL-2 noted that pre-v4.14.3 the overwrite
        was silent and could clobber catalog glasses like ``'N-BK7'``
        without warning).
    """
    # v4.14.3 (P1-GL-2): input validation.  Pre-v4.14.3 accepted any
    # name string (including ``''``) and any ``n`` (including
    # ``n < 1.0``, which is unphysical for ordinary materials), and
    # silently clobbered existing registry entries.
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"register_fixed_glass: name must be a non-empty string; "
            f"got name={name!r}.")
    try:
        n_f = float(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"register_fixed_glass: n must be a real number; got "
            f"n={n!r}.") from exc
    if not (np.isfinite(n_f) and 1.0 <= n_f <= 4.0):
        raise ValueError(
            f"register_fixed_glass: n={n_f} unphysical; expected "
            f"1.0 <= n <= 4.0.  (n < 1 is left-medium / metamaterial "
            f"territory and not supported by the scalar transmission "
            f"path; the upper bound covers Si ~3.4, Ge ~4.0, and any "
            f"common semiconductor used in mid-IR / THz design.)")

    from .glass import (
        _GLASS_CACHE_LOCK,
        GLASS_REGISTRY,
        _glass_cache,
        _glass_value_cache,
    )

    if name in GLASS_REGISTRY:
        import warnings as _w
        _w.warn(
            f"register_fixed_glass: overwriting existing entry "
            f"{name!r} in GLASS_REGISTRY.  If you meant to add a new "
            f"glass, choose a unique name; if the overwrite is "
            f"intentional, silence this warning with "
            f"warnings.simplefilter.",
            UserWarning, stacklevel=2)

    class _FixedIndex:
        def __init__(self, n_val):
            self._n = n_val
        def get_refractive_index(self, wv_nm, unit='nm'):
            return self._n

    GLASS_REGISTRY[name] = ('__user__', '__fixed__', '__fixed__')
    # v5.6: a re-registered name must not serve a stale value from the
    # immutable-branch value cache (e.g. a catalogue glass overwritten by a
    # fixed index).  Clearing the whole value cache is cheap (registration is
    # rare) and fully safe.  v5.17.1 (audit P3-40): mutations go under the
    # glass cache lock now that the value cache is a shared LRU OrderedDict.
    with _GLASS_CACHE_LOCK:
        _glass_cache[name] = _FixedIndex(n_f)
        _glass_value_cache.clear()


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


def save_lens(name: str, prescription: Dict[str, Any],
              description: str = '') -> str:
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


def load_lens(name: str) -> Dict[str, Any]:
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


def list_lenses() -> List[str]:
    """List all saved lens names."""
    lib = get_library_path() / 'lenses'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_lens(name: str) -> None:
    """Delete a saved lens."""
    lib = get_library_path() / 'lenses'
    filepath = lib / f'{_safe_name(name)}.json'
    if filepath.exists():
        filepath.unlink()


# ════════════════════════════════════════════════════════════════════════
# Phase masks
# ════════════════════════════════════════════════════════════════════════

def save_phase_mask(name: str,
                    expression: Optional[str] = None,
                    array: Optional[np.ndarray] = None,
                    dx: Optional[float] = None,
                    wavelength: Optional[float] = None,
                    mask_type: Optional[str] = None,
                    n: Optional[float] = None,
                    thickness: Optional[float] = None,
                    description: str = '') -> str:
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


def load_phase_mask(name: str,
                    N: Optional[int] = None,
                    dx: Optional[float] = None,
                    wavelength: Optional[float] = None) -> np.ndarray:
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
        # 4.10: require an explicit caller-supplied wavelength when one
        # is needed by the expression.  Pre-4.10 silently defaulted to
        # `wavelength = 1.0` (k = 2*pi rad/m), producing meaningless
        # phase masks that "worked" without any failure.
        if wavelength is None:
            wavelength = data.get('wavelength')
        if wavelength is None:
            raise ValueError(
                f"load_phase_mask({name!r}): expression-type mask requires a "
                f"wavelength (pass `wavelength=...` or store it in the "
                f"mask metadata).")
        k = 2 * np.pi / wavelength
        pi = np.pi

        # 4.10: eval() is a code-execution risk if anyone can write to
        # ~/.lumenairy/library/phase_masks/*.json.  Restrict the
        # globals to "no builtins" and use a whitelisted namespace of
        # numpy primitives; reject any expression that contains
        # underscore-prefixed names (typical attack surface) or
        # uses dunder access.  This is a defence-in-depth measure;
        # the real fix is to convert the expression DSL to a small
        # symbolic-AST evaluator, tracked separately.
        expr = str(data['expression'])
        if ('__' in expr) or expr.startswith('_') or ('._' in expr):
            raise ValueError(
                "load_phase_mask: expression contains forbidden tokens "
                "('__' / leading underscore / '._').  Phase-mask "
                "expressions must use only the documented X, Y, R, THETA, "
                "k, pi, np, sin, cos, ... whitelisted names.")
        ns = {
            'X': X, 'Y': Y, 'R': R, 'THETA': THETA,
            'k': k, 'pi': pi, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'abs': np.abs,
            'exp': np.exp, 'log': np.log,
            'atan2': np.arctan2, 'arctan2': np.arctan2,
            'mod': np.mod, 'floor': np.floor, 'ceil': np.ceil,
        }
        phase = eval(expr, {"__builtins__": {}}, ns)
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


def load_phase_mask_info(name: str) -> Dict[str, Any]:
    """Load phase mask metadata without generating the array."""
    lib = get_library_path() / 'phase_masks'
    filepath = lib / f'{_safe_name(name)}.json'
    if not filepath.exists():
        raise FileNotFoundError(f"Phase mask '{name}' not found.")
    with open(filepath) as f:
        return json.load(f)


def list_phase_masks() -> List[str]:
    """List all saved phase mask names."""
    lib = get_library_path() / 'phase_masks'
    return sorted(p.stem for p in lib.glob('*.json'))


def delete_phase_mask(name: str) -> None:
    """Delete a saved phase mask (JSON + any .npy sidecar)."""
    lib = get_library_path() / 'phase_masks'
    for ext in ('.json', '.npy'):
        filepath = lib / f'{_safe_name(name)}{ext}'
        if filepath.exists():
            filepath.unlink()


# ════════════════════════════════════════════════════════════════════════
# Load all materials on import (auto-register saved glasses)
# ════════════════════════════════════════════════════════════════════════

def load_all_materials() -> None:
    """Load all saved materials into GLASS_REGISTRY.

    Called automatically on import so that saved materials are
    immediately available in any script.
    """
    for name in list_materials():
        try:
            load_material(name)
        except (OSError, ValueError, KeyError, RuntimeError, ImportError):
            # Corrupted / partially-written user material file; skip
            # it and continue loading the others.
            pass


# Auto-load on import
try:
    load_all_materials()
except (OSError, ValueError, KeyError, RuntimeError, ImportError):
    # Library import must never fail because of a broken user-library
    # store -- the directory may be missing, permission-locked, or
    # contain corrupted entries.  Library functions still work; the
    # user just won't see their saved materials until they fix the
    # underlying issue.
    pass
